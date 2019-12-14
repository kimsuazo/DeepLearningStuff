import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer

torch.manual_seed(1)
device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'
  torch.cuda.manual_seed_all(1)


class MyUnidirectionalRNN(nn.Module):

  def __init__(self, num_inputs, rnn_size=128):
    super().__init__()

    # Linear layers
    # Define the input activation matrix W
    self.W = nn.Linear(num_inputs, rnn_size, bias=False)
    # TODO: Define the hidden activation matrix U
    self.U = nn.Linear(rnn_size, rnn_size)
    
    self.rnn_size = rnn_size
    # Define the bias
    self.b = nn.Parameter(torch.zeros(1, rnn_size))

  def forward(self, x, state=None):
    # Assuming x is of shape [batch_size, seq_len, num_feats]
    xs = torch.chunk(x, x.shape[1], dim=1)
    hts = []
    if state is None:
      state = self.init_state(x.shape[0])
    for xt in xs:
      # turn x[t] into shape [batch_size, num_feats] to be projected
      xt = xt.squeeze(1)
      ct = self.W(xt)
      ct = ct + self.U(state)
      state = ct + self.b
      # give the temporal dimension back to h[t] to be cated
      hts.append(state.unsqueeze(1))
    hts = torch.cat(hts, dim=1)
    return hts

  def init_state(self, batch_size):
    return torch.zeros(batch_size, self.rnn_size)

# To correctly assess the answer, we build an example RNN with 10 inputs and 32 neurons
rnn = MyUnidirectionalRNN(10, 32)
# Then we will forward some random sequences, each of length 15
xt = torch.randn(5, 15, 10)
# The returned tensor will be h[t]
ht = rnn(xt)
assert ht.shape[0] == 5 and ht.shape[1] == 15 and ht.shape[2] == 32, \
'Something went wrong within the RNN :('
print('Success! Output shape: {} sequences, each of length {}, each '\
      'token with {} dims'.format(ht.shape[0], ht.shape[1], ht.shape[2]))

# we will work with 10 input features
NUM_INPUTS = 10
# and sequences of length 25
SEQ_LEN = 25
# and 5 samples per batch
BATCH_SIZE = 5 
# and 128 neurons
HIDDEN_SIZE = 128

# The first RNN contains a single layer
rnn1 = nn.RNN(NUM_INPUTS, HIDDEN_SIZE)
print(rnn1)

# Now let's build a random input tensor to forward through it
xt = torch.randn(SEQ_LEN, BATCH_SIZE, NUM_INPUTS)
ht, state = rnn1(xt)
print('Output h[t] tensor shape: ', ht.shape)
print('Output state tensor shape: ', state.shape)


# 2 Layer RNN
rnn2 = nn.RNN(NUM_INPUTS, HIDDEN_SIZE, num_layers=2)
ht, state = rnn2(xt)
print('RNN 2 layers >> ht shape: ', ht.shape)
print('RNN 2 layers >> state shape: ', state.shape)

# Batch Size first RNN
xt_bf = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_INPUTS)
rnn3 = nn.RNN(NUM_INPUTS, HIDDEN_SIZE, num_layers=2, batch_first=True)
ht, state = rnn3(xt_bf)
print('RNN 2 layers, batch_first >> ht.shape: ', ht.shape)
print('RNN 2 layers, batch_first >> state.shape: ', state.shape)

# TODO: build the bidirectional RNN layer
bi_rnn = nn.RNN(NUM_INPUTS, HIDDEN_SIZE, bidirectional = True) 

# forward xt_bf
bi_ht, bi_state = bi_rnn(xt_bf)
print('Bidirectional RNN layer >> bi_ht.shape: ', bi_ht.shape)
print('Bidirectional RNN layer >> bi_state.shape: ', bi_state.shape)