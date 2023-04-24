<img src="./block-recurrent-transformer.png" width="450px"></img>

## Block Recurrent Transformer - Pytorch

Implementation of <a href="https://arxiv.org/abs/2203.07852">Block Recurrent Transformer</a> - Pytorch. The highlight of the paper is its reported ability to remember something up to 60k tokens ago.

This design is SOTA for recurrent transformers line of research, afaict.

It will also include <a href="https://arxiv.org/abs/2205.14135">flash attention</a> as well as routed memories of up to 250k tokens using ideas from <a href="https://github.com/lucidrains/CoLT5-attention">this paper</a>

## Appreciation

- <a href="https://stability.ai/">Stability.ai</a> for the generous sponsorship to work and open source cutting edge artificial intelligence research

## Install

```bash
$ pip install block-recurrent-transformer-pytorch
```

## Usage

```python
import torch
from block_recurrent_transformer_pytorch import BlockRecurrentTransformer

model = BlockRecurrentTransformer(
    num_tokens = 20000,             # vocab size
    dim = 512,                      # model dimensions
    depth = 6,                      # depth
    dim_head = 64,                  # attention head dimensions
    heads = 8,                      # number of attention heads
    max_seq_len = 1024,             # the total receptive field of the transformer, in the paper this was 2 * block size
    block_width = 512,              # block size - total receptive field is max_seq_len, 2 * block size in paper. the block furthest forwards becomes the new cached xl memories, which is a block size of 1 (please open an issue if i am wrong)
    num_state_vectors = 512,        # number of state vectors, i believe this was a single block size in the paper, but can be any amount
    recurrent_layers = (4,),        # where to place the recurrent layer(s) for states with fixed simple gating
    use_compressed_mem = False,     # whether to use compressed memories of a single block width, from https://arxiv.org/abs/1911.05507
    compressed_mem_factor = 4,      # compression factor of compressed memories
    use_flash_attn = True           # use flash attention, if on pytorch 2.0
)

seq = torch.randint(0, 2000, (1, 1024))

out, mems1, states1 = model(seq)
out, mems2, states2 = model(seq, xl_memories = mems1, states = states1)
out, mems3, states3 = model(seq, xl_memories = mems2, states = states2)
```

## Test on Enwik8

First `pip install -r requirements.txt`, then

```bash
$ python train.py
```

## Todo

- [x] use dynamic positional bias
- [x] add enhanced recurrence
- [x] setup local attention blocks, as in the paper
- [x] wrapper transformer class for training
- [x] take care of generation with recurrence in `RecurrentTrainWrapper`
- [x] add ability to dropout to entire memories and states during each segment step during trainng
- [x] test full system on enwik8 locally and ablate states and memories and see effects first  hand
- [x] make sure attention allow for single head key / values too
- [x] run a few experiments of fixed gating in regular transformers - does not work
- [x] integrate <a href="https://github.com/hazyresearch/flash-attention">flash attention</a>
- [x] cache attention mask + rotary embeddings
- [x] add <a href="https://github.com/lucidrains/compressive-transformer-pytorch">compressed memories</a>

- [ ] revisit <a href="https://github.com/lucidrains/memformer">memformer</a>
- [ ] try routing long distance memories of up to 250k using coordinate descent (Wright et al.)

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

```bibtex
@article{Shazeer2019FastTD,
    title   = {Fast Transformer Decoding: One Write-Head is All You Need},
    author  = {Noam M. Shazeer},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1911.02150}
}
```

```bibtex
@inproceedings{Sun2022ALT,
    title     = {A Length-Extrapolatable Transformer},
    author    = {Yutao Sun and Li Dong and Barun Patra and Shuming Ma and Shaohan Huang and Alon Benhaim and Vishrav Chaudhary and Xia Song and Furu Wei},
    year      = {2022}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@inproceedings{Ainslie2023CoLT5FL,
    title   = {CoLT5: Faster Long-Range Transformers with Conditional Computation},
    author  = {Joshua Ainslie and Tao Lei and Michiel de Jong and Santiago Ontan'on and Siddhartha Brahma and Yury Zemlyanskiy and David Uthus and Mandy Guo and James Lee-Thorp and Yi Tay and Yun-Hsuan Sung and Sumit Sanghai},
    year    = {2023}
}
```

*Memory is Attention through Time* - Alex Graves
