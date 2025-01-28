# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:23:40 2025

@author: nicolas reinaldet eichenauer
Adapted from:https://github.com/yuanhangzhang98/transformer_quantum_state/blob/main/model.py

"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import sample, sample_without_weight
from torch.nn import TransformerEncoderLayer

from s4 import S4Block

pi=np.pi

class Quantum_S4(nn.Module):
    
    def __init__(self, system_sizes, param_dim, hidden_size,
                 param_encoding=1, phys_dim=2, dropout=0.0,
                 minibatch=None):
        super(Quantum_S4,self).__init__()
        
        self.system_sizes=torch.tensor(system_sizes, dtype=torch.int64)
        assert len(self.system_sizes.shape) == 2
        self.n = self.system_sizes.prod(dim=1)  # (n_size, )
        self.n_size, self.n_dim = self.system_sizes.shape
        max_system_size, _ = self.system_sizes.max(dim=0)  # (n_dim, )
        
        self.size_idx = None
        self.system_size = None
        self.param = None
        self.salt = None
        
        self.param_dim = param_dim
        self.param_encoding=param_encoding
        self.phys_dim = phys_dim
        
        self.hidden_size=hidden_size
        self.param_range=None
        self.dropout=dropout
        self.minibatch=minibatch
        
        
        self.salt_len=(self.n_dim+self.param_dim)*self.param_encoding+2*self.n_dim
        
        self.input_dim=self.salt_len+self.phys_dim
        
        self.S4=S4Block(d_model=hidden_size,dropout=self.dropout,transposed=False)
        
        self.Pre_S4_Lin=nn.Linear(in_features=self.input_dim,
                                  out_features=self.hidden_size)
        
        self.Amp_S4_Lin=nn.Linear(in_features=self.hidden_size,
                                  out_features=self.phys_dim)
        
        self.Phase_S4_Lin=nn.Linear(in_features=self.hidden_size,
                                  out_features=self.phys_dim)
        
        
    def set_param(self, system_size=None, param=None):
        self.size_idx = torch.randint(self.n_size, [])
        if system_size is None:
            self.system_size = self.system_sizes[self.size_idx]
        else:
            self.system_size = system_size
            self.size_idx = None
        if param is None:
            self.param = self.param_range[0] + torch.rand(self.param_dim) * (self.param_range[1] - self.param_range[0])
        else:
            self.param = param
        self.salt = self.salt_seq()

    def salt_seq(self):
        if self.n_dim==1:
            parity=(torch.tensor([1,0])-self.system_size)%2
        else:
            parity=torch.cat((
            (torch.tensor([1,0])-self.system_size[0])%2,
            (torch.tensor([1,0])-self.system_size[1])%2))
            
        salt=torch.zeros(self.salt_len)
        
        parity_len=2*self.n_dim
        size_encoding_end=parity_len+self.n_dim*self.param_encoding
        
        salt[:parity_len]=parity
        salt[parity_len:size_encoding_end]=torch.tensor([i**j for i in 
                                                          torch.log(
                                                              torch.tensor(self.system_size))
                                                          for j in range(1,self.param_encoding+1)])
        salt[size_encoding_end:]=torch.tensor([i**j for i in 
                                                          self.param
                                                          for j in range(1,self.param_encoding+1)])

        return salt
    
    def wrap_spins(self,spins):
        n,batch=spins.shape
        src=torch.zeros(n+1,batch,self.input_dim)
        src[1:,:,:self.phys_dim]=F.one_hot(spins.to(torch.int64),
                                          num_classes=self.phys_dim)
        src[:,:,self.phys_dim:]=self.salt
        
        return src
        
    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.Pre_S4_Lin, -initrange, initrange)
        nn.init.zeros_(self.Pre_S4_Lin.bias)
        nn.init.uniform_(self.Amp_S4_Lin.weight, -initrange, initrange)
        nn.init.zeros_(self.Amp_S4_Lin.bias)
        nn.init.uniform_(self.Phase_S4_Lin.weight, -initrange, initrange)
        nn.init.zeros_(self.Phase_S4_Lin.bias)
        
        
    @staticmethod
    def softsign(x):
        """
            Defined in Hibat-Allah, Mohamed, et al. 
                        "Recurrent neural network wave functions." 
                        Physical Review Research 2.2 (2020): 023358.
            Used as the activation function on the phase output
            range: (-2pi, 2pi)
            NOTE: this function outputs 2\phi, where \phi is the phase
                  an additional factor of 2 is included, to ensure \phi\in(-\pi, \pi)
        """
        return 2 * pi * (1 + x / (1 + x.abs()))        
        
        
    def forward(self, spins, compute_phase=True):
        # src: (seq, batch, input_dim)
        # use_symmetry: has no effect in this function
        # only included to be consistent with the symmetric version
        
        src=self.wrap_spins(spins) ### (seq+1,batch,input)
        
        result=[]
        
        if self.minibatch is None:
            src=torch.swapaxes(self.Pre_S4_Lin(src),0,1) ### (batch,seq+1,input)
            src=torch.swapaxes(self.S4(src)[0],0,1) ### (seq+1,batch,input)
            amp=F.log_softmax(self.Amp_S4_Lin(src),dim=-1) ### (seq+1,batch,phys_dim)
            result.append(amp)
            if compute_phase:
                phase=self.softsign(self.Phase_S4_Lin(src)) ### (seq+1,batch,phys_dim)
                result.append(phase)
        else:
            batch = src.shape[1]
            minibatch = self.minibatch
            repeat = int(np.ceil(batch / minibatch))
            amp = []
            phase = []
            for i in range(repeat):
                src_i = src[:, i * minibatch:(i + 1) * minibatch] ### (seq+1,minibatch,input)
                src_i=torch.swapaxes(self.Pre_S4_Lin(src_i),0,1) ### (minibatch,seq+1,input)
                src_i=torch.swapaxes(self.S4(src_i)[0],0,1) ### (seq+1,minibatch,input)
                amp_i=F.log_softmax(self.Amp_S4_Lin(src_i),dim=-1) ### (seq+1,minibatch,phys_dim)
                amp.append(amp_i)
                if compute_phase:
                    phase_i=self.softsign(self.Phase_S4_Lin(src_i)) ### (seq+1,minibatch,phys_dim)
                    phase.append(phase_i)
            amp = torch.cat(amp, dim=1)
            result.append(amp)
            if compute_phase:
                phase = torch.cat(phase, dim=1)
                result.append(phase)
        return result
    
        
        
        
        
        
        
        
        