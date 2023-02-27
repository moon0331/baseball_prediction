import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, df, decay=0.5, n_hidden_layers=2, labels_dim=(1,)):
        """
        input_dim = 
        """
        super(MLP, self).__init__()
        df = self.df

        input_dim = len(df.columns)
        hidden_dim = int(decay * input_dim)

        self.input_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList()

        for i in range(n_hidden_layers):
            n_hidden_dim = int(decay * n_hidden_dim)
            self.hidden_layers.append(nn.Linear(hidden_dim, n_hidden_dim))
