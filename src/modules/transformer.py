from torch import nn
import torch

from ..config import GPTConfig
from .block import Block

class GPT2Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embedding_size),
            wpe = nn.Embedding(config.block_size, config.embedding_size),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.embedding_size)
        ))
        self.lm_head = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()   # idx (B, T)
        # get the position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos)
        # get the token embeddings
        tok_embd = self.transformer.wte(idx)
        # add them
        x = pos_embd + tok_embd
        # forward through each block of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward through the final layer norm
        x = self.transformer.ln_f(x)
        # get the logits
        logits = self.lm_head(x)

        return logits


    @classmethod
    def load_from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict

        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and embedding_size are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, embedding_size=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, embedding_size=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, embedding_size=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, embedding_size=1600), # 1558M params
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints

        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT2Transformer(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]        # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

