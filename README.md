<img src="./block-recurrent-transformer.png" width="450px"></img>

## Block Recurrent Transformer - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2203.07852">Block Recurrent Transformer</a> - Pytorch

## Install

```bash
$ pip install block-recurrent-transformer
```

## Usage

```python
import torch
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer

model = BlockRecurrentTransformer(
    num_tokens = 2000,
    dim = 512,
    depth = 4,
    dim_head = 64,
    heads = 8,
    xl_memories_layers = (3, 4),
    recurrent_layers = (2, 3)
)

seq = torch.randint(0, 2000, (1, 1024))

out, mems1, states1 = model(seq)
out, mems2, states2 = model(seq, xl_memories = mems1, states = states1)
out, mems3, states3 = model(seq, xl_memories = mems2, states = states2)
```

## Citations

```bibtex
@article{Hutchins2022BlockRecurrentT,
    title   = {Block-Recurrent Transformers},
    author  = {DeLesley S. Hutchins and Imanol Schlag and Yuhuai Wu and Ethan Dyer and Behnam Neyshabur},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2203.07852}
}
```

*Memory is Attention through Time* - Alex Graves
