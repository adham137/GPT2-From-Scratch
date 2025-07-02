import torch
from src.modules.transformer import GPT2Transformer
from torch.nn import functional as F

num_of_sequences = 5
max_length = 30

model = GPT2Transformer.load_from_pretrained('gpt2')
model.eval()
model.to('cuda')
# for k, v in model.state_dict().items():
#     print(k, v.shape)

import tiktoken
encoder = tiktoken.get_encoding('gpt2')
input = "Hello, I'm a language model, "
tokens = encoder.encode(input)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_of_sequences, 1) # (5, number of tokens)
x = tokens.to('cuda')

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # get the last predicted token only
        logits = logits[:, -1, :] # (B, vocab_size)
        # get the probs of these tokens
        logits = F.softmax(logits, 1)
        # select the to 50
        top_k_probs, top_k_indicies = torch.topk(logits, 50, -1)
        # sample from them
        idx = torch.multinomial(top_k_probs, 1) # (B, 1)
        xcol = torch.gather(top_k_indicies, -1, idx)
        # append to x 
        x = torch.cat((x, xcol), dim=1)


# Print the generated text
for i in range(num_of_sequences):
    tokens = x[i, :].tolist()
    decoded = encoder.decode(tokens)
    print(f"> {decoded}")
