import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, List, Tuple

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

def l2norm(t):
    return F.normalize(t, dim = -1)

# bias-less layernorm

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# geglu feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult * 2 / 3)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim_head,
        causal = True,
        cosine_sim = False,
        qk_rmsnorm = False,
        cosine_sim_scale = 8
    ):
        super().__init__()
        self.causal = causal

        assert not (cosine_sim and qk_rmsnorm)
        self.cosine_sim = cosine_sim
        self.qk_rmsnorm = qk_rmsnorm
        self.cosine_sim_scale = cosine_sim_scale

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

    def forward(self, q, k, v, mask = None):

        scale = q.shape[-1] ** -0.5

        if self.cosine_sim or self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            scale = self.cosine_sim_scale

        if self.qk_rmsnorm:
            q = q * self.q_scale
            k = k * self.k_scale

        kv_einsum = 'b j d' if k.ndim == 3 else 'b h j d'

        sim = einsum(f'b h i d, {kv_einsum} -> b h i j', q, k) * scale

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = q.device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum(f'b h i j, {kv_einsum} -> b h i d', attn, v)

        return out

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        causal = True,
        dim_head = 64,
        heads = 8,
        cosine_sim = False,
        qk_rmsnorm = False,
        cosine_sim_scale = 8,
        num_state_vectors = 0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.norm = LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.attn = Attention(dim_head, causal = causal, cosine_sim = cosine_sim, qk_rmsnorm = qk_rmsnorm, cosine_sim_scale = cosine_sim_scale)

        self.is_recurrent_layer = num_state_vectors > 0

        self.to_out = nn.Linear(inner_dim * (2 if self.is_recurrent_layer else 1), dim, bias = False)

        if self.is_recurrent_layer:
            self.state_norm = LayerNorm(dim)

            self.q_to_state = nn.Linear(dim, inner_dim, bias = False)
            self.q_from_state = nn.Linear(dim, inner_dim, bias = False)

            self.state_to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
            self.init_state = nn.Parameter(torch.randn(num_state_vectors, dim))
            self.state_pos_ids = nn.Parameter(torch.randn(num_state_vectors, dim))

            self.to_state_out = nn.Linear(inner_dim * 2, dim, bias = False)

            self.to_state_cross_attn = Attention(dim_head, causal = False, cosine_sim = cosine_sim, qk_rmsnorm = qk_rmsnorm, cosine_sim_scale = cosine_sim_scale)

            self.state_self_attn = Attention(dim_head, causal = False, cosine_sim = cosine_sim, qk_rmsnorm = qk_rmsnorm, cosine_sim_scale = cosine_sim_scale)
            self.from_state_cross_attn = Attention(dim_head, causal = False, cosine_sim = cosine_sim, qk_rmsnorm = qk_rmsnorm, cosine_sim_scale = cosine_sim_scale)

            # gating related parameters - using the fixed simple config

            self.state_out_to_gate = nn.Linear(dim, dim)
            self.learned_ema_beta = nn.Parameter(torch.randn(dim))

    def forward(
        self,
        x,
        xl_memories: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None
    ):
        batch = x.shape[0]

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        memories = torch.stack((k, v))

        if exists(xl_memories):
            past_k, past_v = xl_memories
            k = torch.cat((past_k, k), dim = -2)
            v = torch.cat((past_v, v), dim = -2)

        out = self.attn(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        new_states = None

        if self.is_recurrent_layer:
            if not exists(states):
                states = self.init_state

            orig_states = states

            # pre norm state for attention

            states = self.state_norm(states)

            # add the positional ids, as stated in the paper critical for it to work

            states = states + self.state_pos_ids

            # get queries for cross attention, which they do not share, although they share key / values. another intriguing detail

            q_to_state = self.q_to_state(x)
            q_from_state = self.q_from_state(states)

            q_to_state, q_from_state = map(lambda t: rearrange(t, '... n (h d) -> ... h n d', h = self.heads), (q_to_state, q_from_state))

            # self attention qkv for states

            states = self.state_to_qkv(self.init_state).chunk(3, dim = -1)
            state_q, state_k, state_v = map(lambda t: repeat(t, 'n (h d) -> b h n d', h = self.heads, b = batch), states)

            # cross attend to the past states key values

            to_state_out = self.to_state_cross_attn(q_to_state, state_k, state_v)

            to_state_out = rearrange(to_state_out, 'b h n d -> b n (h d)')

            # concat the output of cross attending to the state vectors

            out = torch.cat((out, to_state_out), dim = -1)

            # states must also undergo self attention

            if q_from_state.ndim == 3:
                q_from_state = repeat(q_from_state, '... -> b ...', b = batch)

            state_out = self.state_self_attn(state_q, state_k, state_v)

            from_state_out = self.from_state_cross_attn(q_from_state, memories[0], memories[1])

            state_out = torch.cat((state_out, from_state_out), dim = -1)
            state_out = rearrange(state_out, 'b h n d -> b n (h d)')

            state_out = self.to_state_out(state_out)

            # use the best performing configuration
            # fixed simple gate - nothing more than a learned EMA with some resemblance to highway networks

            z = self.state_out_to_gate(state_out)
            learned_ema_decay = self.learned_ema_beta.sigmoid()

            new_states = learned_ema_decay * z + (1 - learned_ema_decay) * orig_states

        return self.to_out(out), memories, new_states

# classes

@beartype
class BlockRecurrentTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        cosine_sim = False,
        qk_rmsnorm = True,
        cosine_sim_scale = 8,
        ff_mult = 4,
        ignore_index = -100,
        max_seq_len = 2048,
        xl_memories_layers: Optional[Tuple[int, ...]] = None,
        recurrent_layers: Optional[Tuple[int, ...]] = None,
        num_state_vectors = 512
    ):
        super().__init__()
        xl_memories_layers = default(xl_memories_layers, tuple(range(1, depth + 1)))
        self.xl_memories_layers = set(xl_memories_layers)

        assert all([0 < layer <= depth for layer in xl_memories_layers])

        recurrent_layers = default(recurrent_layers, (depth // 2,)) # default to one recurent layer at middle of the network
        self.recurrent_layers = set(recurrent_layers)

        assert all([0 < layer <= depth for layer in recurrent_layers])

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([])

        for layer in range(1, depth + 1):
            is_recurrent_layer = layer in self.recurrent_layers
            layer_num_state_vectors = num_state_vectors if is_recurrent_layer else 0

            self.layers.append(nn.ModuleList([
                AttentionBlock(
                    dim,
                    causal = True,
                    dim_head = dim_head,
                    heads = heads,
                    cosine_sim = cosine_sim,
                    qk_rmsnorm = qk_rmsnorm,
                    cosine_sim_scale = cosine_sim_scale,
                    num_state_vectors = layer_num_state_vectors
                ),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.max_seq_len = max_seq_len
        self.ignore_index = ignore_index

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        length,
        prime,
        temperature = 1.,
        filter_thres = 0.9
    ):
        orig_len = prime.shape[-1]
        output = prime

        for _ in range(length):
            logits, *_ = self.forward(output[:, -self.max_seq_len:])
            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            output = torch.cat((output, sampled), dim = -1)

        return output[:, orig_len:]

    def forward(
        self,
        x,
        return_loss = False,
        xl_memories: List[torch.Tensor] = tuple(),
        states: List[torch.Tensor] = tuple(),

    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        xl_memories = iter(xl_memories)
        states = iter(states)

        next_xl_memories = []
        next_states = []

        for ind, (attn, ff) in enumerate(self.layers):

            # determine if the layer requires transformer xl memories

            layer = ind + 1

            is_xl_layer     = layer in self.xl_memories_layers
            is_state_layer  = attn.is_recurrent_layer

            # whether to pass in xl memories

            attn_kwargs = dict()

            if is_xl_layer:
                attn_kwargs.update(xl_memories = next(xl_memories, None))

            if is_state_layer:
                attn_kwargs.update(states = next(states, None))

            # attention layer

            residual = x
            attn_branch_out, layer_xl_memories, layer_next_states = attn(x, **attn_kwargs)

            # save states if needed

            if exists(layer_next_states):
                next_states.append(layer_next_states.detach())

            # save current xl memories if needed

            if is_xl_layer:
                next_xl_memories.append(layer_xl_memories.detach())

            x = attn_branch_out + residual

            # feedforward layer

            x = ff(x) + x

        logits = self.to_logits(x)

        if not return_loss:
            return logits, next_xl_memories, next_states

        logits = rearrange(logits, 'b n c -> b c n')
        loss = F.cross_entropy(logits, labels, ignore_index = self.ignore_index)

        return loss, next_xl_memories, next_states
