import torch.nn as nn
import torch


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


def sample_gaussian_like(y):
    return torch.randn_like(y)

def divergence_bf(dx, y, **unused_kwargs):
    sum_diag = 0.
    for i in range(y.shape[1]):
        sum_diag += torch.autograd.grad(dx[:, i].sum(), y, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()

def divergence_approx(f, y, e=None):
    #print(f.shape)
    
    #print(e)
    #e_dzdx = torch.autograd.grad(f, y, torch.ones_like(e), create_graph=True)[0]
    
    e[:,3:] = 0.0
    e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
    #print(e_dzdx, e)
    #print(e_dzdx.shape)
    #assert 0
    #e[:, 3:] = 0
    #assert 0
    e_dzdx_e = e_dzdx * e
    #print(e_dzdx_e)
    #assert 0
    
    approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
    #print(approx_tr_dzdx)
    #assert 0
    return approx_tr_dzdx

class ODEFUNC(nn.Module):
    def __init__(self, diffeq, divergence_fn='approximate', rademacher=False):
        super().__init__()
        self.diffeq = diffeq
        self._e = None
        self.divergence_fn = divergence_approx

    def forward(self, t, states):
        y = states[0]
        batch_size = y.shape[0]
        self._e = torch.randn_like(y)
        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            dy = self.diffeq(t,y)
            divergence = self.divergence_fn(dy, y, e=self._e).view(batch_size, 1)
            #print(divergence.shape)
            #divergence = divergence_bf(dy, y)
            #print(divergence.shape)
            #assert 0
        return [dy, -divergence]
    
            

            