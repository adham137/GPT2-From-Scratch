from torch import nn

class MLP(nn.Module):
    def __init__(self, config):
        super.__init__()
        self.config = config

        self.c_fc = nn.Linear(config.embeeding_size, 4*config.embeeding_size) # Project into higher dimensions to learn something
        self.activation = nn.GELU( approximate='tanh') 
        self.c_proj = nn.Linear(4*config.embeeding_size, config.embeeding_size) # Project back to lower dimensions

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        return x