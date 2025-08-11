import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, mlp_layers=2):
        super().__init__()
        self.layers = self._build_mlp(input_size, hidden_size, mlp_layers)
        # for layer in self.layers:
        #     if isinstance(layer, nn.Linear):
        #         import pdb;pdb.set_trace()
        #         nn.init.xavier_uniform_(layer.weight)
        #         if layer.bias is not None:
        #             nn.init.zeros_(layer.bias)
    def _build_mlp(self, input_size, hidden_size, mlp_layers):
        layers = []
        for i in range(mlp_layers):
            in_dim = input_size if i == 0 else hidden_size
            out_dim = input_size if i == mlp_layers - 1 else hidden_size
            layers.append(nn.Linear(in_dim, out_dim))
            if i < mlp_layers - 1:
                layers.append(nn.GELU())
        layers.append(nn.LayerNorm(input_size))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)
def add_mlp(input_size, hidden_size, mlp_layers=2):
    """返回一个继承 nn.Module 的 MLP 实例"""
    return MLP(input_size, hidden_size, mlp_layers)