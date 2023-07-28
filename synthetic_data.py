import torch
import numpy as np 
from torch.distributions import Normal
import pickle 

if __name__ == "__main__":
    mu = torch.rand([1,3])
    scale = torch.ones([1,3]) * 0.01
    z_dist = Normal(mu, scale)
    z_train = z_dist.sample((1000,))
    time_step = 5
    train_data = torch.zeros([1000, 5, 3])
    for i in range(5):
        train_data[:, i:i+1, :] = z_train
        z_train = z_train + 0.5
    seqlens = torch.ones((1000,))*5
    seqlens = seqlens.numpy()
    train_data = train_data.numpy()
    train_obs = 2 * train_data
    train_dict = {'z': train_data, 'sequences': train_obs, 'seq_lens': seqlens}

    
    
    with open('synth.pickle', 'wb') as handle:
        pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)