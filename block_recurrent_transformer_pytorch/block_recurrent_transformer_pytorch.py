import math
from random import random
from functools import wraps, partial
from collections import namedtuple
from packaging import version

from einops import rearrange

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack

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

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

def compact(arr):
    return [*filter(exists, arr)]

def and_reduce(arr: List[torch.Tensor]):
    if len(arr) == 0:
        return None
    head, *rest = arr
    for t in rest:
        head = head & t
    return head

print_once = once(print)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def l2norm(t):
    return F.normalize(t, dim = -1)

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

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

# rotary positional embedding w/ xpos
# https://arxiv.org/abs/2104.09864
# https://arxiv.org/abs/2212.10554v1

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        scale_base = 512,
        use_xpos = True,
        theta = 10000
    ):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self.use_xpos = use_xpos
        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self, seq_len):
        device = self.device
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not self.use_xpos:
            return freqs, torch.ones(1, device = device)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, pos, scale = 1.):
    scale = default(scale, 1.)

    seq_len = t.shape[-2]

    assert pos.shape[-2] >= seq_len

    pos = pos[-seq_len:]

    if isinstance(scale, torch.Tensor):
        assert scale.shape[-2] >= seq_len
        scale = scale[-seq_len:]

    return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)

# maybe flash attention, if using pytorch 2.0

# constants

Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# main class

class Attend(nn.Module):
    def __init__(
        self,
        causal = False,
        use_flash_attn = False
    ):
        super().__init__()
        self.causal = causal
        self.register_buffer("mask", None, persistent=False)

        self.use_flash_attn = use_flash_attn
        assert not (use_flash_attn and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = Config(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not use_flash_attn:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major == 8 and device_properties.minor == 0:
            print_once('A100 GPU detected, using flash attention if input tensor is on cuda')
            self.cuda_config = Config(True, False, False)
        else:
            print_once('Non-A100 GPU detected, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = Config(False, True, True)

    def get_mask(self, n, device):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len, is_cuda = *q.shape, k.shape[-2], q.is_cuda

        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if k.ndim == 3:
            k = repeat(k, 'b ... -> b h ...', h = q.shape[1])

        if v.ndim == 3:
            v = repeat(v, 'b ... -> b h ...', h = q.shape[1])

        # Check if mask exists and expand to compatible shape
        # The mask is B L, so it would have to be expanded to B H N L

        masks = []

        if self.causal:
            i, j = q_len, k_len
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = q.device).triu(j - i + 1)
            masks.append(~causal_mask)

        if exists(mask):
            if mask.ndim != 2:
                mask = repeat(mask, 'w ... -> (b w) ...', b = q.shape[0] // mask.shape[0])

            masks.append(mask)

        attn_mask = and_reduce(masks)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, causal, softmax_scale

        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = attn_mask
            )

        return out

    def forward(self, q, k, v, mask = None, use_flash_attn = None):
        use_flash_attn = default(use_flash_attn, self.use_flash_attn)

        b, n, device = q.shape[0], q.shape[-2], q.device

        q, ps = pack_one(q, '* h n d')
        k, _ = pack_one(k, '* n d')
        v, _ = pack_one(v, '* n d')

        if use_flash_attn:
            out = self.flash_attn(q, k, v, mask = mask)
            return unpack_one(out, ps, '* h n d')

        scale = q.shape[-1] ** -0.5

        k_einsum = 'b j d' if k.ndim == 3 else 'b h j d'
        v_einsum = 'b j d' if v.ndim == 3 else 'b h j d'

        # similarity

        sim = einsum(f"b h i d, {k_einsum} -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            if mask.ndim != 2:
                mask = repeat(mask, 'w ... -> (b w) ...', b = b)

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = q.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum(f"b h i j, {v_einsum} -> b h i d", attn, v)

        return unpack_one(out, ps, '* h n d')

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
        qk_rmsnorm = False,
        qk_rmsnorm_scale = 8,
        use_flash_attn = False
    ):
        super().__init__()
        self.causal = causal

        self.qk_rmsnorm = qk_rmsnorm
        self.qk_rmsnorm_scale = qk_rmsnorm_scale

        self.attend = Attend(causal = causal, use_flash_attn = use_flash_attn)

        if qk_rmsnorm:
            self.q_scale = nn.Parameter(torch.ones(dim_head))
            self.k_scale = nn.Parameter(torch.ones(dim_head))

    def forward(
        self,
        q, k, v,
        mask = None,
        rotary_pos_emb = None,
        xpos_scale = None
    ):

        scale = q.shape[-1] ** -0.5

        if self.qk_rmsnorm:
            q, k = map(l2norm, (q, k))
            scale = self.qk_rmsnorm_scale

        if self.qk_rmsnorm:
            q = q * self.q_scale
            k = k * self.k_scale

        # rotary positional embedding with xpos for length extrapolation

        if exists(rotary_pos_emb):
            q = apply_rotary_pos_emb(q, rotary_pos_emb, xpos_scale)
            k = apply_rotary_pos_emb(k, rotary_pos_emb, xpos_scale ** -1)

        # attention

        out = self.attend(q, k, v, mask = mask)

        return out

class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        block_width,
        causal = True,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        qk_rmsnorm_scale = 8,
        num_state_vectors = 0,
        use_flash_attn = False
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.attn = Attention(dim_head, causal = causal, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

        self.block_width = block_width
        self.is_recurrent_layer = num_state_vectors > 0

        self.to_out = nn.Linear(inner_dim * (2 if self.is_recurrent_layer else 1), dim, bias = False)

        if self.is_recurrent_layer:
            self.state_norm = LayerNorm(dim)

            self.q_to_state = nn.Linear(dim, inner_dim, bias = False)
            self.q_from_state = nn.Linear(dim, inner_dim, bias = False)

            self.state_to_q = nn.Linear(dim, inner_dim, bias = False)
            self.state_to_kv = nn.Linear(dim, dim_head * 2, bias = False)

            self.init_state = nn.Parameter(torch.randn(num_state_vectors, dim))
            self.state_pos_ids = nn.Parameter(torch.randn(num_state_vectors, dim))

            self.to_state_out = nn.Linear(inner_dim * 2, dim, bias = False)

            self.to_state_cross_attn = Attention(dim_head, causal = False, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

            self.state_self_attn = Attention(dim_head, causal = False, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)
            self.from_state_cross_attn = Attention(dim_head, causal = False, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

            # gating related parameters - using the fixed simple config

            self.state_out_to_gate = nn.Linear(dim, dim)
            self.learned_ema_beta = nn.Parameter(torch.randn(dim))

    def forward(
        self,
        x,
        rotary_pos_emb = None,
        xpos_scale = None,
        attn_mask = None,
        return_memories_and_states = None,
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

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        split_head = partial(rearrange, pattern = 'b n (h d) -> b h n d', h = self.heads)
        q = split_head(q)

        # bucket the queries, keys, values by block width

        bq, bk, bv = map(lambda t: rearrange(t, 'b ... (w n) d -> b w ... n d', n = width), (q, k, v))

        # save the last key / values as memories for recurrence

        memories = None

        if return_memories_and_states:
            memories = torch.stack((bk[:, -1], bv[:, -1]))

        if exists(xl_memories):
            # if past memories are passed in, concat as the first bucket
            past_k, past_v = xl_memories
            past_k, past_v = map(lambda t: rearrange(t, 'b ... n d -> b 1 ... n d'), (past_k, past_v))
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

        out = self.attn(
            bq, bk, bv,
            rotary_pos_emb = rotary_pos_emb,
            xpos_scale = xpos_scale,
            mask = attn_mask
        )

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

            q_to_state = self.q_to_state(x[:, :seq_len])
            q_from_state = self.q_from_state(states)

            q_to_state, q_from_state = map(lambda t: rearrange(t, '... n (h d) -> ... h n d', h = self.heads), (q_to_state, q_from_state))

            # self attention qkv for states

            state_q, state_k, state_v = (self.state_to_q(self.init_state), *self.state_to_kv(self.init_state).chunk(2, dim = -1))
            state_q = repeat(state_q, 'n (h d) -> b h n d', h = self.heads, b = batch)

            # cross attend to the past states key values

            to_state_out = self.to_state_cross_attn(q_to_state, state_k, state_v)

            to_state_out = rearrange(to_state_out, 'b h n d -> b n (h d)')

            # concat the output of cross attending to the state vectors

            out = torch.cat((out, to_state_out), dim = -1)

            if return_memories_and_states:
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
        all_layers_qk_rmsnorm = False,
        ff_mult = 4,
        max_seq_len = 1024,
        block_width = 512,
        xl_memories_layers: Optional[Tuple[int, ...]] = None,
        recurrent_layers: Optional[Tuple[int, ...]] = None,
        num_state_vectors = None,
        enhanced_recurrence = False,
        ignore_index = -100,
        rotary_use_xpos = True,
        use_flash_attn = False
    ):
        super().__init__()
        num_state_vectors = default(num_state_vectors, block_width)
        xl_memories_layers = default(xl_memories_layers, tuple(range(1, depth + 1)))
        self.xl_memories_layers = set(xl_memories_layers)

        assert all([0 < layer <= depth for layer in xl_memories_layers])

        recurrent_layers = default(recurrent_layers, (depth // 2,)) # default to one recurent layer at middle of the network
        self.recurrent_layers = set(recurrent_layers)

        assert all([0 < layer <= depth for layer in recurrent_layers])

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.rotary_pos_emb = RotaryEmbedding(dim = dim_head, use_xpos = rotary_use_xpos)

        self.layers = nn.ModuleList([])

        for layer in range(1, depth + 1):
            is_recurrent_layer = layer in self.recurrent_layers
            is_xl_layer = layer in self.xl_memories_layers

            layer_num_state_vectors = num_state_vectors if is_recurrent_layer else 0

            # only layers with xl memories
            # or has recurrence in horizontal direction
            # use qk rmsnorm (in paper, they use cosine sim attention, but i think qk rmsnorm is more proven given Vit 22B paper)
            # one can also override to use all qk rmsnorm by setting all_layers_qk_rmsnorm = True

            qk_rmsnorm = all_layers_qk_rmsnorm or is_recurrent_layer or is_xl_layer

            self.layers.append(nn.ModuleList([
                AttentionBlock(
                    dim,
                    causal = True,
                    block_width = block_width,
                    dim_head = dim_head,
                    heads = heads,
                    qk_rmsnorm = qk_rmsnorm,
                    num_state_vectors = layer_num_state_vectors,
                    use_flash_attn = use_flash_attn
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
        prime,
        length = None,
        xl_memories: List[torch.Tensor] = [],
        states: List[torch.Tensor] = [],
        temperature = 1.,
        filter_thres = 0.9,
        return_memories_and_states = False
    ):
        length = default(length, self.max_seq_len + 1)
        start_len = prime.shape[-1]

        assert start_len < self.max_seq_len
        assert length <= (self.max_seq_len + 1)
        assert start_len < length

        output = prime

        memories = []
        states = []

        for ind in range(length - start_len):

            logits, next_memories, next_states = self.forward(
                output,
                xl_memories = xl_memories,
                states = states
            )

            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            output = torch.cat((output, sampled), dim = -1)

            if divisible_by(output.shape[-1] - 1, self.max_seq_len): # on the sampling of the last token in the current window, set new memories and states
                memories = next_memories
                states = next_states

        output = output[:, start_len:]

        if return_memories_and_states:
            return output, memories, states

        return output

    def forward(
        self,
        x,
        return_loss = False,
        xl_memories: List[torch.Tensor] = [],
        states: List[torch.Tensor] = [],
        return_memories_and_states = None  # can force to either return memory + state or not. by default will only return when number of tokens == max_seq_len
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

        i_arange = torch.arange(w, device = device)
        j_arange = torch.arange(w * 2, device = device)
        rel_pos = ((rearrange(i_arange, 'i -> i 1') + w) - rearrange(j_arange, 'j -> 1 j')).abs()
        attn_mask = rel_pos < w  # make sure each token only looks back a block width

        rotary_pos_emb, xpos_scale = self.rotary_pos_emb(2 * w)

        # enhanced recurrence

        if self.enhanced_recurrence and len(xl_memories) > 1:
            xl_memories = [*xl_memories[1:], xl_memories[0]]

        # ready xl memories and states

        xl_memories = iter(xl_memories)
        states = iter(states)

        next_xl_memories = []
        next_states = []

        return_memories_and_states = default(return_memories_and_states, self.max_seq_len == x.shape[-2])

        # go through layers

        for ind, (attn, ff) in enumerate(self.layers):

            # determine if the layer requires transformer xl memories

            layer = ind + 1

            is_xl_layer     = layer in self.xl_memories_layers
            is_state_layer  = attn.is_recurrent_layer

            # whether to pass in xl memories

            attn_kwargs = dict(
                rotary_pos_emb = rotary_pos_emb,
                xpos_scale = xpos_scale,
                attn_mask = attn_mask,
                return_memories_and_states = return_memories_and_states
            )

            if is_xl_layer:
                attn_kwargs.update(xl_memories = next(xl_memories, None))

            if is_state_layer:
                attn_kwargs.update(states = next(states, None))

            # attention layer

            residual = x
            attn_branch_out, layer_xl_memories, layer_next_states = attn(x, **attn_kwargs)

            if return_memories_and_states:
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

# recurrent trainer wrapper

@beartype
class RecurrentTrainerWrapper(nn.Module):
    def __init__(
        self,
        transformer: BlockRecurrentTransformer,
        xl_memories_dropout = 0.,
        state_dropout = 0.
    ):
        super().__init__()
        self.transformer = transformer
        self.seq_len = transformer.max_seq_len

        self.xl_memories_dropout = xl_memories_dropout
        self.state_dropout = state_dropout

    @eval_decorator
    @torch.no_grad()
    def generate(
        self,
        prime,
        length,
        **kwargs
    ):
        seq_len = self.seq_len
        start_len = prime.shape[-1]
        assert start_len < length

        output = prime
        current_len = start_len

        memories = []
        states = []

        # determine lengths

        has_remainder = not divisible_by(length, seq_len)
        remainder_amount = length % seq_len
        total_segments = math.ceil(length / seq_len)

        if not has_remainder:
            lengths = (*((seq_len + 1,) * (total_segments - 1)), seq_len)
        elif remainder_amount == 1:
            lengths = (seq_len + 1,) * (total_segments - 1)
        else:
            lengths = (*((seq_len + 1,) * (total_segments - 1)), remainder_amount)

        # loop through lengths

        for next_length in lengths:

            segment_output, memories, states = self.transformer.generate(
                output[:, -current_len:],
                length = next_length,
                xl_memories = memories,
                states = states,
                return_memories_and_states = True,
                **kwargs
            )

            output = torch.cat((output, segment_output), dim = -1)
            current_len = 1

        return output[:, start_len:]

    def forward(self, x, return_memories_and_states = False):
        total_seq_len, seq_len = x.shape[1], self.seq_len

        assert divisible_by(total_seq_len - 1, seq_len), f'length of sequence ({total_seq_len}) must be equal to a multiple of {seq_len} + 1 (one extra token) during training'
        segments = total_seq_len // seq_len

        total_loss = 0.

        memories = []
        states = []

        for ind in range(segments):
            start = ind * seq_len
            end = start + seq_len + 1

            if self.training and random() < self.xl_memories_dropout:
                memories.clear()

            if self.training and random() < self.state_dropout:
                states.clear()

            loss, memories, states = self.transformer(
                x[:, start:end],
                xl_memories = memories,
                states = states,
                return_loss = True
            )

            total_loss = total_loss + (loss / segments)

        if return_memories_and_states:
            return total_loss, memories, states

        return total_loss
