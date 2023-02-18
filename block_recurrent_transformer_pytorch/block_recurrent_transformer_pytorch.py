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

# sampling helpers

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

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

        sim = einsum(f'b h i d, {kv_einsum} -> b h i j') * scale

        max_neg_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, max_neg_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), device = q.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum(f'b h i j, {kv_einsum} -> b h i d', attn, v)

        return out

# classes

class BlockRecurrentTransformer(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

    @torch.no_grad()
    @eval_decorator
    def generate(self, length):
        pass

    def forward(self, x):
        x = self.token_emb(x)

        return x
