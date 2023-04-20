import torch
from packaging import version

if version.parse(torch.__version__) >= version.parse('2.0.0'):
    from einops._torch_specific import allow_ops_in_compiled_graph
    allow_ops_in_compiled_graph()

from block_recurrent_transformer_pytorch.block_recurrent_transformer_pytorch import BlockRecurrentTransformer, RecurrentTrainerWrapper
