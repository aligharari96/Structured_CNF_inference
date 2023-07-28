import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint


class CNF(nn.Module):
    def __init__(self, odefunc, solver='dopri5', atol=1e-5, rtol=1e-5):
        super().__init__()
        self.odefunc = odefunc
        self.solver = solver
        self.rtol = rtol
        self.atol = atol
    
    def forward(self, z, logpz=None, integration_times=None, reverse=False):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz
        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)
        
        z_t, logpz_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=self.atol,
                rtol=self.rtol,
                method=self.solver,
            )

        return z_t, logpz_t


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

if __name__ == "__main__":
    print(_flip(torch.tensor([0.0, 1.0]), 0))
        