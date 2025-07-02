from src.modules.transformer import GPT2Transformer

model = GPT2Transformer.load_from_pretrained('gpt2')

for k, v in model.state_dict().items():
    print(k, v.shape)