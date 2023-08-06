import torch
import torch.nn as nn
from torch.distributions import Normal
from torchdiffeq import odeint_adjoint as odeint
from asic import optimize_all_masks
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n, t, mu, sigma):
    z0 = Normal(torch.tensor([mu]), torch.tensor([sigma])).sample((n,))
    Z = torch.zeros((n,t))
    for i in range(t):
        Z[:, i] = z0.reshape(-1,)
        z0 = Normal((z0+0.1).reshape(-1,), torch.ones_like(z0.reshape(-1,))).sample((1,))
    X = Normal(2*Z, torch.ones_like(Z)).sample((1,)).squeeze()
    print(Normal(2*Z, torch.ones_like(Z)).log_prob(X).sum(dim=1).mean())
    #assert 0
    means = torch.zeros_like(Z)
    means[:,0] = torch.tensor([mu])
    means[:,1:] = Z[:,0:-1]+0.1

    return Z, X, means

def create_mask(dim_in, dim_out, max_seq_len):
    mask = torch.ones(((dim_in+dim_out)*max_seq_len, dim_out*max_seq_len))
    for i in range(dim_out * max_seq_len):
        mask[(i+1)*dim_out: (dim_in+dim_out)*max_seq_len-(max_seq_len-i)*dim_in,i] = 0.0
    return mask.transpose(0, 1)

class ConcatSquashLinearSparse(nn.Module):
    def __init__(self, dim_in, dim_out, mask):
        super(ConcatSquashLinearSparse, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self._weight_mask = mask
        lin = nn.Linear(dim_in, dim_out)
        self._weights = lin.weight
        self._bias = lin.bias
        

    def forward(self, x):
        w = torch.mul(self._weight_mask, self._weights)
        res = torch.addmm(self._bias, x, w.transpose(0,1))
        return res 

class ODENET(nn.Module):
    def __init__(self, dim_in, dim_out, hiddens, masks):
        super().__init__()
        layers = []
        in_dim = dim_in
        for i in range(len(hiddens)):
            layers.append(ConcatSquashLinearSparse(in_dim, hiddens[i], masks[i].transpose(0,1)))
            in_dim = hiddens[i]
            layers.append(nn.ReLU())
        layers.append(ConcatSquashLinearSparse(in_dim, dim_out, masks[-1].transpose(0,1)))
        self.layers = nn.ModuleList(layers)
        self.conditioned = None
    
    def forward(self, t, y):
        dy = torch.cat([y, self.conditioned], dim=1)
        for l in self.layers:
            dy = l(dy)
        return dy

class ODEFUNC(nn.Module):
    def __init__(self, diffeq):
        super().__init__()
        self.diffeq = diffeq
        self._e = None

    def forward(self, t, states):
        y = states[0]
        batch_size = y.shape[0]
        self._e = torch.randn_like(y)
        with torch.set_grad_enabled(True):
            y.requires_grad_(True)
            t.requires_grad_(True)
            dy = self.diffeq(t,y)
            divergence = self.divergence_fn(dy, y, e=self._e).view(batch_size, 1)
        return [dy, -divergence]
    
    def divergence_fn(self, f, y, e=None):
        e_dzdx = torch.autograd.grad(f, y, e, create_graph=True)[0]
        e_dzdx_e = e_dzdx * e    
        approx_tr_dzdx = e_dzdx_e.view(y.shape[0], -1).sum(dim=1)
        return approx_tr_dzdx
    
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

#def Emitter(z):
#    return 2*z, torch.ones_like(z)*1

class Emitter(nn.Module):
    def __init__(self, dim_in, dim_out, hiddens) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        in_dim = dim_in
        layers = []
        for h in hiddens:
            layers.append(nn.Linear(in_dim, h))
            in_dim = h
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(in_dim, dim_out)
        self.log_std = nn.Linear(in_dim, dim_out)
    def forward(self, x):
        rep = self.net(x)
        return self.mu(rep), torch.exp(self.log_std(rep))


#def transition(z):
#   return z + 0.1, torch.ones_like(z)
class Transition(nn.Module):
    def __init__(self, dim_in, dim_out, hiddens) -> None:
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        in_dim = dim_in
        layers = []
        for h in hiddens:
            layers.append(nn.Linear(in_dim, h))
            in_dim = h
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.mu = nn.Linear(in_dim, dim_out)
        self.log_std = nn.Linear(in_dim, dim_out)
    def forward(self, x):
        rep = self.net(x)
        return self.mu(rep), torch.exp(self.log_std(rep))

class Dataset(torch.utils.data.Dataset):
  def __init__(self, x, z, means):
        'Initialization'
        self.x = x
        self.z = z
        self.lenght = torch.ones((self.x.shape[0],1))*self.x.shape[1]
        self.means = means

  def __len__(self):
        return len(self.z)

  def __getitem__(self, index):
        x = self.x[index,:]
        z = self.z[index,:]
        length = self.lenght[index, :]
        means = self.means[index, :]
        return z, x, length, means

def KL(logq_z, logp_z):
    return (logq_z - logp_z).sum(dim=1)


    

        
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    data_z, data_x, data_means = generate_data(1000, 25, 0.0, 0.01)
    #val_z = data_z[-100:]
    #val_x = data_x[-100:]
    #data_z = data_z[-100:]
    #data_x = data_x[-100:]
    
    
    dataloader = torch.utils.data.DataLoader(Dataset(data_z, data_x, data_means), batch_size=64)
    mask = create_mask(1,1,25)
    
    mask_numpy = optimize_all_masks('greedy',[32,32], mask)
    masks = [torch.from_numpy(m).type(torch.float32) for m in mask_numpy]
    odenet = ODENET(50, 25, [32,32], masks)
    odefunc = ODEFUNC(odenet)
    cnf = CNF(odefunc)
    #cnf.load_state_dict(torch.load('cnf.pt'))
    #assert 0
    emitter = Emitter(1, 1, [4])
    transition = Transition(1,1,[4])
    #emitter.load_state_dict(torch.load('emitter.pt'))
    #transition.load_state_dict(torch.load('transition.pt'))
    dist_z0 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    min_val_loss = 1000
    optimizer = torch.optim.Adam(list(cnf.parameters())+list(emitter.parameters()) + list(transition.parameters()), lr=1e-3)
    for itr in range(100):
        if itr % 10 == 0:
                z__ = torch.randn((1,1))
                generated__ = torch.zeros(1,25)
                generated__[0,0] = z__
                for i in range(24):
                    z_mu__, z_std__ = transition(z__)
                    generated__[0, i+1] = z_mu__ + z_std__ * torch.randn((1,))
                    z__ = generated__[0,i+1].reshape(-1,1)
                print(z__)
                x_mu__, x_std__ = emitter(generated__.reshape(-1,1))
                plt.plot(x_mu__.reshape(-1,).detach().numpy())
                plt.savefig("results/"+str(itr)+".png")
                plt.close()
        
        for z, x, l, _ in dataloader:      
            optimizer.zero_grad()
            cnf.odefunc.diffeq.conditioned = x
            z0 = Normal(torch.tensor([0.0]), torch.tensor([1.0])).sample(z.shape).squeeze()
            logp_z0 = Normal(torch.tensor([0.0]), torch.tensor([1.0])).log_prob(z0).sum(dim=1).reshape(-1,1)
            zt, logp_zt = cnf(z0, logp_z0)
            
            p_mu, p_std = transition(zt[-1].reshape(-1,1))
            
            
            prior_means = p_mu.reshape(x.shape)[:,:-1]
            prior_std = p_std.reshape(x.shape)[:,:-1]
            prior_means = torch.concat([torch.zeros((prior_means.shape[0],1)), prior_means], dim=1)
            prior_std = torch.concat([torch.ones((prior_means.shape[0],1)), prior_std], dim=1)

            #print(prior_means.shape)
            #assert 0
            
            logprob_prior = Normal(prior_means, prior_std).log_prob(zt[-1]).sum(dim=1)
            kl = (logp_zt[-1].reshape(-1,) - logprob_prior).mean()
            #recon_mu, recon_std = Emitter(zt[-1])
            r_mu, r_std = emitter(zt[-1].reshape(-1,1))
            
            recon_mu = r_mu.reshape(x.shape)
            recon_std = r_std.reshape(x.shape)
            recon_loss = torch.mean(Normal(recon_mu, recon_std).log_prob(x).sum(dim=1))
            loss = kl - recon_loss
            loss.backward()
            optimizer.step()
            print(loss, kl)
            



        



    