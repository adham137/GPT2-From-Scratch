from torch import nn
from .block import Block

class GPT2Transformer:
    def __init__(self, config):
        super.__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_size),
            wpe = nn.Embedding(config.block_size, config.embedding_size),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.embedding_size)
        ))
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)
