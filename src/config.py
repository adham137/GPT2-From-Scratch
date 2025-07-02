from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int = 50257     # Number of tokens in vocab 
    embedding_size: int = 768   # Number of embedding dimensions 
    block_size: int = 1024      # The context window
    n_layer: int = 12           # Number of layers in the transformers
    n_head: int = 12            # Number of attention heads
    bias: bool