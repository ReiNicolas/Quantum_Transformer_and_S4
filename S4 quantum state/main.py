
# -*- coding: utf-8 -*-
"""
Created on Sun May 15 23:30:45 2022

@author: Yuanhang Zhang
"""

from s4_model import Quantum_S4
from Hamiltonian import Ising, XYZ
from optimizer import Optimizer

import os
import numpy as np
import torch

import time

torch.set_default_tensor_type(torch.cuda.FloatTensor
                                    if torch.cuda.is_available()
                                    else torch.FloatTensor)
# torch.set_default_tensor_type(torch.FloatTensor)
try:
    os.mkdir('results/')
except FileExistsError:
    pass

system_sizes = np.array([[4,4]])

Hamiltonians = [Ising(system_size_i, periodic=False) for system_size_i in system_sizes]

param_dim = Hamiltonians[0].param_dim
embedding_size = 106
hidden_size = embedding_size
dropout = 0
minibatch = 10000

model = Quantum_S4(system_sizes, param_dim, hidden_size,
             param_encoding=2, dropout=dropout,minibatch=minibatch)
num_params = sum([param.numel() for param in model.parameters()])
print('Number of parameters: ', num_params)
folder = 'results/'
name = type(Hamiltonians[0]).__name__
save_str = f'S4_{name}_{embedding_size}'
# missing_keys, unexpected_keys = model.load_state_dict(torch.load(f'{folder}ckpt_100000_{save_str}_0.ckpt'),
#                                                       strict=False)
# print(f'Missing keys: {missing_keys}')
# print(f'Unexpected keys: {unexpected_keys}')

param_range = None  # use default param range
# param = torch.tensor([1.0])
# param_range = torch.tensor([[param], [param]])
point_of_interest = None
use_SR = False

time_start=time.time()

optim = Optimizer(model, Hamiltonians, point_of_interest=point_of_interest,ckpt_freq = 2000)
optim.train(1000000, batch=1000, max_unique=100, param_range=param_range,
            fine_tuning=False, use_SR=use_SR, ensemble_id=int(use_SR))

time_end=time.time()

print(time_end-time_start)