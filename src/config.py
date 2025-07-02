from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int
    embedding_size: int
    block_size: int             # The context window
    n_layer: int                # number of layers in the transformers