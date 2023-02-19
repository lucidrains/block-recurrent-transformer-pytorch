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

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm(t):
    return F.normalize(t, dim = -1)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

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

    def forward(self, q, k, v, mask = None, attn_bias = None):

        scale = q.shape[-1] ** -0.5

        if self.cosine_sim or self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            scale = self.cosine_sim_scale

        if self.qk_rmsnorm:
            q = q * self.q_scale
            k = k * self.k_scale

        kv_einsum = '... j d' if k.ndim == 3 else '... h j d'

        sim = einsum(f'... h i d, {kv_einsum} -> ... h i j', q, k) * scale

        if exists(attn_bias):
            sim = sim + attn_bias

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = q.device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum(f'... h i j, {kv_einsum} -> ... h i d', attn, v)

        return out

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        block_width,
        causal = True,
        dim_head = 64,
        heads = 8,
        cosine_sim = False,
        qk_rmsnorm = False,
        cosine_sim_scale = 8,
        num_state_vectors = 0,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.norm = LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.attn = Attention(dim_head, causal = causal, cosine_sim = cosine_sim, qk_rmsnorm = qk_rmsnorm, cosine_sim_scale = cosine_sim_scale)

        self.block_width = block_width
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
        rel_pos_bias = None,
        attn_mask = None,
        xl_memories: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None
    ):
        batch, seq_len, _, width = *x.shape, self.block_width

        # first make sure to pad the sequence length to multiple of the block widths
        # for local attention

        if not divisible_by(seq_len, width):
            padding_to_width_multiple = math.ceil(seq_len / width) * width - seq_len
            x = pad_at_dim(x, (0, padding_to_width_multiple), dim = -2, value = 0)

        # pre normalization

        x = self.norm(x)

        # queries, keys, values and split out heads

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # bucket the queries, keys, values by block width

        bq, bk, bv = map(lambda t: rearrange(t, 'b h (w n) d -> b w h n d', n = width), (q, k, v))

        # save the last key / values as memories for recurrence

        memories = torch.stack((bk[:, -1], bv[:, -1]))

        if exists(xl_memories):
            # if past memories are passed in, concat as the first bucket
            past_k, past_v = xl_memories
            past_k, past_v = map(lambda t: rearrange(t, 'b h n d -> b 1 h n d'), (past_k, past_v))
            bk = torch.cat((past_k, bk), dim = 1)
            bv = torch.cat((past_v, bv), dim = 1)
        else:
            # otherwise add padding
            bk = pad_at_dim(bk, (1, 0), value = 0., dim = 1)
            bv = pad_at_dim(bv, (1, 0), value = 0., dim = 1)

            # and make sure not to attend to this padding
            if exists(attn_mask):
                attn_mask = repeat(attn_mask, 'i j -> w 1 i j', w = bq.shape[1])
                attn_mask[0, 0, :, :width] = False

        # local attention with look back of one bucket - in paper they did total receptive field of 2 * block_width, with 1 block_width worth of memories, seems like a more efficient transformer-xl design?

        bk = torch.cat((bk[:, :-1], bk[:, 1:]), dim = -2)
        bv = torch.cat((bv[:, :-1], bv[:, 1:]), dim = -2)

        # attention, but of course

        out = self.attn(bq, bk, bv, attn_bias = rel_pos_bias, mask = attn_mask)

        # merge the heads as well as the buckets

        out = rearrange(out, 'b w h n d -> b (w n) (h d)')

        # in case there is padding during sampling, splice it out

        out = out[:, :seq_len]

        new_states = None

        # if designated a recurrent layer, do all the state logic
        # it was hard moving this to a separate module, as the attention is closely intertwined between the current tokens and state tokens

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
        max_seq_len = 1024,
        block_width = 512,
        xl_memories_layers: Optional[Tuple[int, ...]] = None,
        recurrent_layers: Optional[Tuple[int, ...]] = None,
        num_state_vectors = 512,
        dynamic_pos_bias_dim = None,
        enhanced_recurrence = False
    ):
        super().__init__()
        xl_memories_layers = default(xl_memories_layers, tuple(range(1, depth + 1)))
        self.xl_memories_layers = set(xl_memories_layers)

        assert all([0 < layer <= depth for layer in xl_memories_layers])

        recurrent_layers = default(recurrent_layers, (depth // 2,)) # default to one recurent layer at middle of the network
        self.recurrent_layers = set(recurrent_layers)

        assert all([0 < layer <= depth for layer in recurrent_layers])

        self.token_emb = nn.Embedding(num_tokens, dim)

        pos_mlp_dim = default(dynamic_pos_bias_dim, dim // 4)
        self.dynamic_pos_bias_mlp = nn.Sequential(
            nn.Linear(1, pos_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_mlp_dim, pos_mlp_dim),
            nn.SiLU(),
            nn.Linear(pos_mlp_dim, heads)
        )

        self.layers = nn.ModuleList([])

        for layer in range(1, depth + 1):
            is_recurrent_layer = layer in self.recurrent_layers
            layer_num_state_vectors = num_state_vectors if is_recurrent_layer else 0

            self.layers.append(nn.ModuleList([
                AttentionBlock(
                    dim,
                    causal = True,
                    block_width = block_width,
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
        self.block_width = block_width

        assert divisible_by(max_seq_len, block_width)

        self.ignore_index = ignore_index

        self.enhanced_recurrence = enhanced_recurrence

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
        xl_memories: List[torch.Tensor] = [],
        states: List[torch.Tensor] = [],

    ):
        device = x.device

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        # get sequence length i and j for dynamic pos bias

        assert x.shape[-1] <= self.max_seq_len

        w = self.block_width

        # token embedding

        x = self.token_emb(x)

        # dynamic pos bias

        rel_dist = torch.arange(w, dtype = x.dtype, device = device)
        rel_dist = rearrange(rel_dist, '... -> ... 1')
        pos_bias = self.dynamic_pos_bias_mlp(rel_dist)

        i_arange = torch.arange(w, device = device)
        j_arange = torch.arange(w * 2, device = device)
        rel_pos = ((rearrange(i_arange, 'i -> i 1') + w) - rearrange(j_arange, 'j -> 1 j')).abs()

        attn_mask = rel_pos < w  # make sure each token only looks back a block width
        rel_pos = rel_pos.masked_fill(~attn_mask, 0)

        pos_bias = pos_bias[rel_pos]
        pos_bias = rearrange(pos_bias, 'i j h -> h i j')

        # enhanced recurrence

        if self.enhanced_recurrence and len(xl_memories) > 1:
            xl_memories = [*xl_memories[1:], xl_memories[0]]

        # ready xl memories and states

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

            attn_kwargs = dict(
                rel_pos_bias = pos_bias,
                attn_mask = attn_mask
            )

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
