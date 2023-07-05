import math
from random import random
from functools import wraps, partial
from itertools import zip_longest
from collections import namedtuple, defaultdict
from packaging import version


import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, List, Tuple

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_empty(t: torch.Tensor):
    return t.numel() == 0

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def all_unique(arr):
    return len(arr) == len(set(arr))

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

print_once = once(print)

def compact(arr):
    return [*filter(exists, arr)]

def and_reduce(arr: List[torch.Tensor]):
    if len(arr) == 0:
        return None
    head, *rest = arr
    for t in rest:
        head = head & t
    return head

def safe_cat(*args, dim = 1):
    args = compact(args)

    if len(args) == 0:
        return None

    return torch.cat(args, dim = dim)

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
        width,
        scale_base = 512,
        theta = 10000
    ):
        super().__init__()
        self.width = width

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent = False)

        self.scale_base = scale_base
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale, persistent = False)

        self.register_buffer('cached_freqs', None, persistent = False)
        self.register_buffer('cached_scales', None, persistent = False)

    @property
    def device(self):
        return next(self.buffers()).device

    def forward(self):
        device, seq_len = self.device, self.width

        if exists(self.cached_freqs):
            cached_seq_len = self.cached_freqs.shape[-2]
            if cached_seq_len >= seq_len:
                return self.cached_freqs[:seq_len], self.cached_scales[:seq_len]

        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        self.register_buffer('cached_freqs', freqs, persistent = False)
        self.register_buffer('cached_scales', scale, persistent = False)
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

# memory management

class MemoryManager(nn.Module):
    def __init__(
        self,
        dim,
        *,
        layers = 1,
        mem_lengths = 512,
        compress_factors = 1
    ):
        super().__init__()
        mem_lengths = cast_tuple(mem_lengths)
        compress_factors = cast_tuple(compress_factors)

        assert all([mem_length > 0 for mem_length in mem_lengths])
        assert len(mem_lengths) == len(compress_factors)
        assert layers >= 1

        self.mem_lengths = mem_lengths
        self.compress_factors = compress_factors

        self.layers = nn.ModuleList([])

        for _ in range(layers):
            compress_fns = nn.ModuleList([])

            for compress_factor in compress_factors:
                compress_fn = nn.Identity()
                if compress_factor > 1:
                    compress_fn = nn.Sequential(
                        Rearrange('b n d -> b d n'),
                        nn.Conv1d(
                            dim * 2,
                            dim * 2,
                            compress_factor,
                            stride = compress_factor,
                            groups = 2
                        ),
                        Rearrange('b d n -> b n d'),
                    )

                compress_fns.append(compress_fn)

            self.layers.append(compress_fns)

    def forward(
        self,
        past_memories: List[torch.Tensor],
        new_memories: List[torch.Tensor]
    ):
        next_memories = []

        for past_memory, new_memory, compress_fns in zip_longest(past_memories, new_memories, self.layers):

            # edge case if neither memories exist

            if not (exists(past_memory) or exists(new_memory)):
                next_memories.append(None)
                continue

            next_memory = None

            for mem_length, compress_factor, compress_fn in zip(self.mem_lengths, self.compress_factors, compress_fns):

                # first get the memories for the given compression factor "current_memory"

                current_memory = None
                if exists(past_memory):
                    past_memory, current_memory = past_memory[..., :-mem_length, :], past_memory[..., -mem_length:, :]

                # compress the new memories coming in, based on the compression factors set at init

                if (not is_empty(new_memory)) and compress_factor > 1:
                    # make sure memory length is divisible by compression factor

                    new_mem_length = new_memory.shape[-2]

                    curtailed_length = (new_mem_length // compress_factor) * compress_factor

                    curtailed_slice = slice(-curtailed_length, None) if curtailed_length > 0 else slice(0, 0)
                    new_memory = new_memory[..., curtailed_slice, :]

                    # compress the memory pushed to the next stage

                    if new_memory.shape[-2] > 0:
                        new_memory = rearrange(new_memory, 'm b n d -> b n (m d)')
                        new_memory = compress_fn(new_memory)
                        new_memory = rearrange(new_memory, 'b n (m d) -> m b n d', m = 2)

                # fifo memory queue
                # add the new memory on the right

                current_memory = safe_cat(current_memory, new_memory, dim = -2)
                # "new" memory is new with respect to the next compressed segment

                new_memory, current_memory = current_memory[..., :-mem_length, :], current_memory[..., -mem_length:, :]
                # concat the new memory to the left into the past

                next_memory = safe_cat(current_memory, next_memory, dim = -2)

            next_memories.append(next_memory)

        return next_memories

# maybe flash attention, if using pytorch 2.0

# constants

Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

# state container

class StateContainer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        num_state_vectors,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        qk_rmsnorm_scale = 8,
        use_flash_attn = False
    ):
        super().__init__()
        assert num_state_vectors > 0
        self.heads = heads
        inner_dim = dim_head * heads

        self.state_norm = LayerNorm(dim)

        self.q_to_state = nn.Linear(dim, inner_dim, bias = False)
        self.q_from_state = nn.Linear(dim, inner_dim, bias = False)

        self.state_to_q = nn.Linear(dim, inner_dim, bias = False)
        self.state_to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.init_state = nn.Parameter(torch.randn(num_state_vectors, dim))
        self.state_pos_ids = nn.Parameter(torch.randn(num_state_vectors, dim))

        self.to_state_out = nn.Linear(inner_dim * 2, dim, bias = False)

        self.to_state_cross_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

        self.state_self_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)
        self.from_state_cross_attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

        # gating related parameters - using the fixed simple config

        self.state_out_to_gate = nn.Linear(dim, dim)
        self.learned_ema_beta = nn.Parameter(torch.randn(dim))

        # since each read should be followed by a write, just store cache in the container

        self.cache = None
        self.next_read_state = None

    def set_next_read_state(
        self,
        states
    ):
        if not exists(states):
            states = self.init_state

        self.next_read_state = (states,)

    def read(self, x):
        assert exists(self.next_read_state), 'states to be read must be set with .set_next_read_state'

        states, = self.next_read_state
        self.next_read_state = None

        # pre norm state for attention

        normed_states = self.state_norm(states)

        # add the positional ids, as stated in the paper critical for it to work

        normed_states = normed_states + self.state_pos_ids

        # get queries for cross attention, which they do not share, although they share key / values. another intriguing detail

        q_to_state = self.q_to_state(x)
        q_to_state = rearrange(q_to_state, '... n (h d) -> ... h n d', h = self.heads)

        # self attention qkv for states

        state_k, state_v = self.state_to_kv(normed_states).chunk(2, dim = -1)

        # cross attend to the past states key values

        to_state_out = self.to_state_cross_attn(q_to_state, state_k, state_v)

        to_state_out = rearrange(to_state_out, 'b h n d -> b n (h d)')

        # cache for next write

        self.cache = (states, normed_states, state_k, state_v)

        return to_state_out

    def write(
        self,
        *,
        memories
    ):
        assert exists(self.cache)

        k, v = memories
        batch = k.shape[0]

        # get cached values from the previous read

        states, normed_states, state_k, state_v = self.cache

        self.cache = None

        # derive queries

        q_from_state = self.q_from_state(normed_states)
        q_from_state = rearrange(q_from_state, '... n (h d) -> ... h n d', h = self.heads)

        state_q = self.state_to_q(normed_states)
        state_q_einsum = 'n (h d)' if state_q.ndim == 2 else 'b n (h d)'
        state_q = repeat(state_q, f'{state_q_einsum} -> b h n d', h = self.heads, b = batch)

        # states must also undergo self attention

        if q_from_state.ndim == 3:
            q_from_state = repeat(q_from_state, '... -> b ...', b = batch)

        state_out = self.state_self_attn(state_q, state_k, state_v)

        from_state_out = self.from_state_cross_attn(q_from_state, k, v)

        state_out = torch.cat((state_out, from_state_out), dim = -1)
        state_out = rearrange(state_out, 'b h n d -> b n (h d)')

        state_out = self.to_state_out(state_out)

        # use the best performing configuration
        # fixed simple gate - nothing more than a learned EMA with some resemblance to highway networks

        z = self.state_out_to_gate(state_out)
        learned_ema_decay = self.learned_ema_beta.sigmoid()

        # set new state with the learned EMA gating

        return learned_ema_decay * z + (1 - learned_ema_decay) * states

    def forward(self, x):
        raise NotImplementedError

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
        causal = False,
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
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        qk_rmsnorm_scale = 8,
        use_flash_attn = False,
        num_state_vectors = 0,
        num_external_state_reads = 0,
        state_read_before_write = True  # this will be defaulted to on as in the paper, but will be turned off in the case the researcher wants to test out reading the state at a lower layer
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)

        self.attn = Attention(dim_head, qk_rmsnorm = qk_rmsnorm, qk_rmsnorm_scale = qk_rmsnorm_scale, use_flash_attn = use_flash_attn)

        self.block_width = block_width
        self.is_recurrent_layer = num_state_vectors > 0

        # decide how many states this attention layer is going to read from

        num_state_reads = int(self.is_recurrent_layer and state_read_before_write) + num_external_state_reads

        self.to_out = nn.Linear(inner_dim * (1 + num_state_reads), dim, bias = False)

        if not self.is_recurrent_layer:
            return

        self.state_read_before_write = state_read_before_write

        self.state_container = StateContainer(
            dim,
            dim_head = dim_head,
            heads = heads,
            num_state_vectors = num_state_vectors,
            qk_rmsnorm = qk_rmsnorm,
            qk_rmsnorm_scale = qk_rmsnorm_scale,
            use_flash_attn = use_flash_attn
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        rotary_pos_emb = None,
        xpos_scale = None,
        attn_mask = None,
        xl_memories: Optional[torch.Tensor] = None,
        read_from_state_containers: List[StateContainer] = []
    ):
        batch, seq_len, _, width, device = *x.shape, self.block_width, self.device

        # pre normalization

        x = self.norm(x)

        # queries, keys, values and split out heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = -1))

        split_head = partial(rearrange, pattern = 'b n (h d) -> b h n d', h = self.heads)
        q = split_head(q)

        # save the last key / values as memories for recurrence

        memories = torch.stack((k, v))

        mem_len = 0

        if exists(xl_memories):
            # if past memories are passed in, concat as the first bucket
            mem_len = xl_memories.shape[-2]
            past_k, past_v = xl_memories
            k = torch.cat((past_k, k), dim = 1)
            v = torch.cat((past_v, v), dim = 1)

        # handle cropping of attention mask and positional embeddings

        if exists(attn_mask):
            attn_mask = attn_mask[:seq_len, :seq_len]
            attn_mask = F.pad(attn_mask, (mem_len, 0), value = True)

        # attention, but of course

        out = self.attn(
            q, k, v,
            rotary_pos_emb = rotary_pos_emb,
            xpos_scale = xpos_scale,
            mask = attn_mask
        )

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')

        # early return if not a recurrent layer

        if not self.is_recurrent_layer and len(read_from_state_containers) == 0:
            return self.to_out(out), memories, None

        # whether to read from own state container, default to on, but may pass in more

        if self.is_recurrent_layer and self.state_read_before_write:
            read_from_state_containers = [self.state_container, *read_from_state_containers]

        for read_state_container in read_from_state_containers:
            # read from the states ...

            to_state_out = read_state_container.read(x)

            # and concat it to the output of self-attention

            out = torch.cat((out, to_state_out), dim = -1)

        new_states = None

        if self.is_recurrent_layer:
            # then write to the states as well if need be

            new_states = self.state_container.write(memories = memories)

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
        recurrent_layers: Optional[Tuple[int, ...]] = None,
        read_recurrent_layers: Optional[Tuple[int, ...]] = None,
        num_state_vectors = None,
        ignore_index = -100,
        use_flash_attn = False,
        use_compressed_mem = False,
        compressed_mem_factor = 4
    ):
        super().__init__()
        num_state_vectors = default(num_state_vectors, block_width)

        # set recurrent layers

        recurrent_layers = default(recurrent_layers, (depth // 2,)) # default to one recurent layer at middle of the network

        assert all([0 < layer <= depth for layer in recurrent_layers]), f'recurrent layers must range from 1 to the depth {depth}'
        assert all_unique(recurrent_layers), 'recurrent layers must be all unique. no duplicate layers'

        self.recurrent_layers = recurrent_layers

        # set read recurrent layers

        read_recurrent_layers = default(read_recurrent_layers, recurrent_layers)

        assert all([read_layer <= write_layer for read_layer, write_layer in zip(read_recurrent_layers, recurrent_layers)]), 'the recurrent read layer must be always less than or equal to the write layer'
        assert all([0 < layer <= depth for layer in read_recurrent_layers])
        assert len(read_recurrent_layers) == len(recurrent_layers)

        self.read_recurrent_layers = read_recurrent_layers

        # token embedding

        self.token_emb = nn.Embedding(num_tokens, dim)

        self.rotary_pos_emb = RotaryEmbedding(dim = dim_head, width = (2 if not use_compressed_mem else 3) * block_width)

        self.layers = nn.ModuleList([])

        self.write_to_read_map = {write_layer: read_layer for write_layer, read_layer in zip(recurrent_layers, read_recurrent_layers)}

        self.read_state_router = defaultdict(list)

        for layer in range(1, depth + 1):
            is_recurrent_layer = layer in self.recurrent_layers

            layer_num_state_vectors = num_state_vectors if is_recurrent_layer else 0

            num_external_state_reads = sum([int(layer == read_layer) for read_layer in read_recurrent_layers])

            # only layers with xl memories
            # or has recurrence in horizontal direction
            # use qk rmsnorm (in paper, they use cosine sim attention, but i think qk rmsnorm is more proven given Vit 22B paper)
            # one can also override to use all qk rmsnorm by setting all_layers_qk_rmsnorm = True

            qk_rmsnorm = all_layers_qk_rmsnorm or is_recurrent_layer

            attn_block = AttentionBlock(
                dim,
                block_width = block_width,
                dim_head = dim_head,
                heads = heads,
                qk_rmsnorm = qk_rmsnorm,
                num_state_vectors = layer_num_state_vectors,
                use_flash_attn = use_flash_attn,
                num_external_state_reads = num_external_state_reads,
                state_read_before_write = False,
            )

            ff_block = FeedForward(dim, mult = ff_mult)

            if is_recurrent_layer:
                read_layer = self.write_to_read_map[layer]
                self.read_state_router[read_layer].append(attn_block.state_container)

            self.layers.append(nn.ModuleList([
                attn_block,
                ff_block
            ]))

        # (compressed) memory management

        self.mem_manager = MemoryManager(
            dim = dim_head,
            layers = depth,
            mem_lengths = block_width if not use_compressed_mem else (block_width, block_width // 2),
            compress_factors = 1 if not use_compressed_mem else (1, compressed_mem_factor)
        )

        # to logits

        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        self.max_seq_len = max_seq_len
        self.block_width = block_width

        assert divisible_by(max_seq_len, block_width)

        self.ignore_index = ignore_index

        self.register_buffer('cached_causal_attn_mask', None, persistent = False)

    @property
    def device(self):
        return next(self.parameters()).device

    def get_causal_attn_mask(self, width):
        if exists(self.cached_causal_attn_mask):
            cached_mask = self.cached_causal_attn_mask
            cached_width = cached_mask.shape[-2]
            padding = (width - cached_width) // 2
            j_slice = Ellipsis if padding == 0 else slice(padding, -padding)
            return cached_mask[:cached_width, j_slice]

        device = self.device
        causal_mask = torch.ones((width, width), device = device, dtype = torch.bool).triu(1)
        return ~causal_mask

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

        attn_mask = self.get_causal_attn_mask(w)
        rotary_pos_emb, xpos_scale = self.rotary_pos_emb()

        # only return memories and state if at the full block width, but can be overridden

        return_memories_and_states = default(return_memories_and_states, self.max_seq_len == x.shape[-2])

        # ready output tensor, to be concatted to block by block

        batch, _, dim = x.shape

        out = torch.empty(batch, 0, dim, dtype = x.dtype, device = self.device)

        # split input into blocks of width w

        input_blocks = x.split(w, dim = -2)

        # process each block at a time

        for input_block in input_blocks:
            input_block_length = input_block.shape[-2]

            # ready xl memories and states

            iter_xl_memories = iter(xl_memories)
            iter_states = iter(states)

            next_xl_memories = []
            next_states = []

            # set the states on the appropriate state containers

            for attn, _ in self.layers:
                if not attn.is_recurrent_layer:
                    continue

                attn.state_container.set_next_read_state(next(iter_states, None))

            # go through layers

            for ind, (attn, ff) in enumerate(self.layers):

                # determine if the layer requires transformer xl memories

                layer = ind + 1

                # whether to pass in xl memories

                attn_kwargs = dict(
                    rotary_pos_emb = rotary_pos_emb,
                    xpos_scale = xpos_scale,
                    attn_mask = attn_mask,
                    xl_memories = next(iter_xl_memories, None),
                    read_from_state_containers = self.read_state_router[layer]
                )

                # attention layer

                residual = input_block
                attn_branch_out, layer_xl_memories, layer_next_states = attn(input_block, **attn_kwargs)

                if exists(layer_xl_memories):
                    next_xl_memories.append(layer_xl_memories)

                if exists(layer_next_states):
                    next_states.append(layer_next_states)

                input_block = attn_branch_out + residual

                # feedforward layer

                input_block = ff(input_block) + input_block

            # concat to output

            out = torch.cat((out, input_block), dim = -2)

            # set new xl memories and states

            states = next_states

            if input_block_length == w:
                xl_memories = self.mem_manager(xl_memories, next_xl_memories)


        # project to logits

        logits = self.to_logits(out)

        # detach the states and memories

        returned_next_states = list(map(torch.detach, states)) if return_memories_and_states else None
        returned_next_xl_memories = list(map(torch.detach, xl_memories)) if return_memories_and_states else None

        # whether to return logits

        if not return_loss:
            return logits, returned_next_xl_memories, returned_next_states

        # cross entropy loss

        logits = rearrange(logits, 'b n c -> b c n')
        loss = F.cross_entropy(logits, labels, ignore_index = self.ignore_index)

        return loss, returned_next_xl_memories, returned_next_states

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

    def forward(
        self,
        x,
        return_memories_and_states = False
    ):
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
