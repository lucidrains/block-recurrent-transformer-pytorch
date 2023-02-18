import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

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
        cosine_sim_scale = 8
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        self.norm = LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.attn = Attention(dim_head, causal = causal, cosine_sim = cosine_sim, qk_rmsnorm = qk_rmsnorm, cosine_sim_scale = cosine_sim_scale)

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        out = self.attn(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# classes

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
        qk_rmsnorm = False,
        cosine_sim_scale = 8,
        ff_mult = 4,
        ignore_index = -100,
        max_seq_len = 2048
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                AttentionBlock(dim, causal = True, dim_head = dim_head, heads = heads, cosine_sim = cosine_sim, qk_rmsnorm = qk_rmsnorm, cosine_sim_scale = cosine_sim_scale),
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
            logits = self.forward(output[:, -self.max_seq_len:])
            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            output = torch.cat((output, sampled), dim = -1)

        return output[:, orig_len:]

    def forward(self, x, return_loss = False):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels, ignore_index = self.ignore_index)
