import torch 
import torch.nn as nn

class ODENET(nn.Module):
    def __init__(self, input_dims, hidden_dims, condition_dims, non_linearity):
        super().__init__()
        layers = []
        self.input_dims = input_dims
        self.condition_dims = condition_dims#[X,Z]
        self.hidden_dims = hidden_dims
        if non_linearity == 'relu':
            self.non_linearity = nn.ReLU()
        elif non_linearity == 'sigmoid':
            self.non_linearity = nn.Sigmoid()
        elif non_linearity == 'tanh':
            self.non_linearity = nn.Tanh()
        elif non_linearity == 'elu':
            self.non_linearity = nn.ELU()
        else:
            assert 0
        in_dim = input_dims+ condition_dims + 1#[y,z_prev, x, t]
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(self.non_linearity)
            in_dim = h
        layers.append(nn.Linear(in_dim, self.input_dims))
        self.net = nn.Sequential(*layers)
        self.conditioned = None
    
    def forward(self, t, y):
        #print('yes')
        tmp =  self.net(torch.cat([y, t*torch.ones(y.shape[0],1)], dim=1))
        return torch.cat([tmp, torch.zeros(tmp.shape[0],self.condition_dims)], dim=1)
        
        return self.net(torch.cat([y, self.conditioned, t*torch.ones(y.shape[0],1)], dim=1))

if __name__ == "__main__":
    print(ODENET(4, [32,32,32], 8, 'relu'))