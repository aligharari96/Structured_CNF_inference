import torch
import torch.nn as nn 
from modules import GatedTransition, PostNet, Encoder 
from deepHMM.helper import reverse_sequence, sequence_mask, gVar
from CNF.ODENET import ODENET
from CNF.odefunc import ODEFUNC
from CNF.cnf import CNF
from torch.optim import Adam
from deepHMM.data_loader import PolyphonicDataset, SyntheticDataset
from torch.distributions import Normal
import sys
#sys.setrecursionlimit(10000)

class Q_Z_0(nn.Module):
    def __init__(self, config):
        super(Q_Z_0, self).__init__()
        self.input_dim = config['rnn_dim'] + config['z_dim']
        self.z_dim = config['z_dim']
        self.net = nn.Sequential(nn.Linear(self.input_dim, self.input_dim),
                                 nn.ReLU(),
                                 nn.Linear(self.input_dim, 16), nn.ReLU(),
                                 nn.Linear(16, 8), nn.ReLU())
        self.mu = nn.Linear(8, self.z_dim)
        self.log_var = nn.Linear(8, self.z_dim)
    
    def forward(self, x):
        tmp = self.net(x)
        return self.mu(tmp), self.log_var(tmp)

class Emitter(nn.Module):
    def __init__(self, config):
        super(Emitter, self).__init__()
        self.output_dim = config['input_dim']
        self.input_dim = config['z_dim']
        self.emission_dim = config['emission_dim']
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.emission_dim),
            nn.ReLU(),
            nn.Linear(self.emission_dim, self.emission_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(self.emission_dim, self.output_dim)
        self.logvar = nn.Linear(self.emission_dim, self.output_dim)
    
    def forward(self, x):
        tmp = self.net(x)
        return self.mu(tmp), self.logvar(tmp)

class DHMM_CNF(nn.Module):
    def __init__(self, config):
        super(DHMM_CNF, self).__init__()
        self.input_dim = config['input_dim']
        self.z_dim = config['z_dim']
        self.emission_dim = config['emission_dim']
        self.trans_dim = config['trans_dim']
        self.rnn_dim = config['rnn_dim']
        self.clip_norm = config['clip_norm']
        self.q_z_0 = Q_Z_0(config)
        self.emitter = Emitter(config)
        #self.emitter = nn.Sequential( #Parameterizes the bernoulli observation likelihood `p(x_t|z_t)`
        #    nn.Linear(self.z_dim, self.emission_dim),
        #    nn.ReLU(),
        #    nn.Linear(self.emission_dim, self.emission_dim),
        #    nn.ReLU(),
        #    nn.Linear(self.emission_dim, self.input_dim),
        #    nn.Sigmoid()#sure?
        #)
        self.trans = GatedTransition(self.z_dim, self.trans_dim)
        self.rnn = Encoder(None, self.input_dim, self.rnn_dim, False, 1)
        self.odenet = ODENET(self.z_dim,[32,32], self.z_dim + self.rnn_dim, non_linearity='relu')
        self.odefunc = ODEFUNC(self.odenet)
        self.cnf = CNF(self.odefunc)
        self.z_0_mu = nn.Parameter(torch.zeros(self.z_dim))
        self.z_0_log_var = nn.Parameter(torch.randn(self.z_dim))
        self.z_q_0 = nn.Parameter(torch.zeros(self.z_dim))
        self.h_0 = nn.Parameter(torch.zeros(1, 1, self.rnn_dim))    
        self.rep = nn.Sequential(nn.Linear(19, 32), nn.ReLU(), nn.Linear(32,19))
        
        self.optimizer = Adam(self.parameters(), lr=config['lr'], betas= (config['beta1'], config['beta2']))
        self.continous_outcome = True

    def sample(self, mu, logvar):
        return mu + torch.randn(mu.shape)*0.5*torch.exp(logvar)
    
    def KL_DIV(self, log_prob_p, log_prob_q):
        return log_prob_p - log_prob_q
    
    def infer(self, x, x_rev, x_lens):
        batch_size, _, x_dim = x.size()
        T_max = x_lens.max()
        #run input through RNN
        h_0 = self.h_0.expand(1, batch_size, self.rnn.hidden_size).contiguous()
        _, rnn_out = self.rnn(x_rev, x_lens, h_0)
        rnn_out = reverse_sequence(rnn_out, x_lens)
        #define loss placeholders
        rec_losses = torch.zeros((batch_size, T_max), device=x.device) 
        kl_states = torch.zeros((batch_size, T_max), device=x.device)  
        #Z_(-1)
        z_prev = self.z_q_0.expand(batch_size, self.z_q_0.size(0))
        #sample z0_prior
        z_prior_mu = self.z_0_mu.expand(batch_size, self.z_0_mu.size(0))
        z_prior_logvar = self.z_0_log_var.expand(batch_size, self.z_0_log_var.size(0))
        #z_prior = self.sample(z_0_mu, z_0_log_var)
        #log_p_z_qz = torch.mean(Normal(z_0_mu,torch.exp(0.5*z_0_log_var)).log_prob(z_prev), dim=1, keepdim=True)
        #########log_prob_z_prior = torch.mean(Normal(z_0_mu,torch.exp(0.5*z_0_log_var)).log_prob(z_prior), dim=1, keepdim=True)

        
        for t in range(T_max):
            #self.odenet.conditioned = torch.cat([rnn_out[:, t, :], z_prev], dim=1)
            #self.odenet.conditioned = self.rep(self.odenet.conditioned)
            q_z_0_mu, q_z_0_log_var = self.q_z_0(torch.cat([rnn_out[:, t, :], z_prev], dim=1))
            #q_z_0_mu, q_z_0_log_var = q_z_0_mu[:,:3], q_z_0_log_var[:,:3]
            q_z_0 = self.sample(q_z_0_mu, q_z_0_log_var)
            #q_z_0 = z_prev
            
            q_z, log_pz_t = self.cnf(torch.cat([q_z_0, rnn_out[:, t, :], z_prev], dim=1))

            q_z = q_z[-1][:,:3]#[batch, z_dim]
            
            log_pz_t = log_pz_t[-1]#[batch, 1]
            
            log_prob_q_z_0 = Normal(q_z_0_mu,torch.exp(0.5*q_z_0_log_var)).log_prob(q_z_0)#[batch, z_dim]
            log_prob_q_z_0 = torch.mean(log_prob_q_z_0, dim=1, keepdim=True)#[batch, 1]
            
            log_prob_q_z = log_prob_q_z_0 + log_pz_t
            log_p_z_qz = torch.mean(Normal(z_prior_mu,torch.exp(0.5*z_prior_logvar)).log_prob(q_z), dim=1, keepdim=True)
            kl_states[:, t] = self.KL_DIV(log_prob_q_z, log_p_z_qz).reshape(-1,)
            
            recon_mu, recon_logvar = self.emitter(q_z)
            #print(self.NLL_continous(recon_mu, recon_logvar, x[:,t,:]).shape)
            
            rec_losses[:,t] = self.NLL_continous(recon_mu, recon_logvar, x[:,t,:])
            
            z_prev = q_z
            z_prior, z_prior_mu, z_prior_logvar = self.trans(z_prev)
            #log_p_z_qz = torch.mean(Normal(z_prior_mu,torch.exp(0.5*z_prior_logvar)).log_prob(z_prev), dim=1, keepdim=True)

        x_mask = sequence_mask(x_lens)
        x_mask = x_mask.gt(0).view(-1)
        rec_loss = rec_losses.view(-1).masked_select(x_mask).mean()
        kl_loss = kl_states.view(-1).masked_select(x_mask).mean()
        return rec_loss, kl_loss
    
    def NLL_continous(self, x_hat_mu, x_hat_log_var, x):
        std = torch.exp(0.5*x_hat_log_var)
        result =  0.5 * torch.log(torch.tensor([2*torch.pi])) + 0.5 * x_hat_log_var + 0.5 * ((x - x_hat_mu)/std)**2
        return torch.mean(result, dim=1)



    def train_AE(self, x, x_rev, x_lens, kl_anneal):
        self.rnn.train() # put the RNN back into training mode (i.e. turn on drop-out if applicable)
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        
        loss = rec_loss + kl_anneal*kl_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)
        self.optimizer.step()
        
        return {'train_loss_AE':loss.item(), 'train_loss_KL':kl_loss.item()}
    
    def valid(self, x, x_rev, x_lens):
        self.eval()
        rec_loss, kl_loss = self.infer(x, x_rev, x_lens)
        loss = rec_loss + kl_loss
        return loss
            


if __name__ == "__main__":
    config = {'input_dim': 3,
    'z_dim':3,
    'emission_dim':8,
    'trans_dim':8,
    'rnn_dim':16,
    'clip_norm':20.0,
    'lr':3e-4, # autoencoder learning rate
    'beta1':0.96, # beta1 for adam
    'beta2':0.999
    }
    train_set = SyntheticDataset('/Users/alihosseinghararifoomani/deep-structured_hmm/synth.pickle')
    train_loader=torch.utils.data.DataLoader(dataset=train_set, batch_size=128, shuffle=True)
    
    model = DHMM_CNF(config)
    kl_anneal = 1.0
    #print(model)
    for _ in range(10):
        for x, x_rev, x_lens, z in train_loader:
            loss = model.train_AE(x, x_rev, x_lens, kl_anneal)
            print(loss)
            #loss = model.train_AE(x, x_rev, x_lens, kl_anneal)
            #print(loss)
            
    
    