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


def prepare_sequence(seq, char2idx, onehot=True):
    # convert sequence of words to indices
    idxs = [char2idx[c] for c in seq]
    idxs = torch.tensor(idxs, dtype=torch.long)
    if onehot:
      # conver to onehot (if input to network)
      ohs = F.one_hot(idxs, len(char2idx)).float()
      return ohs
    else:
      return idxs

with open('friends.txt', 'r') as txt_f:
  training_data = [l.rstrip() for l in txt_f if l.rstrip() != '']

# merge the training data into one big text line
training_data = '$'.join(training_data)

# Assign a unique ID to each different character found in the training set
char2idx = {}
for c in training_data:
    if c not in char2idx:
            char2idx[c] = len(char2idx)
idx2char = dict((v, k) for k, v in char2idx.items())
VOCAB_SIZE = len(char2idx)
RNN_SIZE = 1024
MLP_SIZE = 2048
SEQ_LEN = 50
print('Number of found vocabulary tokens: ', VOCAB_SIZE)

class CharLSTM(nn.Module):

    def __init__(self, vocab_size, rnn_size, mlp_size):
        super().__init__()
        self.rnn_size = rnn_size 

        # TODO:
        self.lstm=nn.LSTM(vocab_size, rnn_size, batch_first = True)

        self.dout = nn.Dropout(0.4)

        # TODOs:
        # An MLP with a hidden layer of mlp_size neurons that maps from the RNN 
        # hidden state space to the output space of vocab_size
        self.mlp = nn.Sequential(
          nn.Linear(rnn_size, mlp_size,), # Linear layer
          nn.ReLU(), # Activation function
          nn.Dropout(0.4),
          nn.Linear(mlp_size, vocab_size), # Linear layer
          nn.LogSoftmax() # Output layer
        )

    def forward(self, sentence, state=None):
        bsz, slen, vocab = sentence.shape
        ht, state = self.lstm(sentence, state)
        ht = self.dout(ht)
        h = ht.contiguous().view(-1, self.rnn_size)
        logprob = self.mlp(h)
        return logprob, state

# Let's build an example model and see what the scores are before training
model = CharLSTM(VOCAB_SIZE, RNN_SIZE, MLP_SIZE)
# This should output crap as it is not trained, so a fixed random tag for everything

def gen_text(model, seed, char2idx, num_chars=150):
  model.eval()
  # Here we don't need to train, so the code is wrapped in torch.no_grad()
  with torch.no_grad():
      inputs = prepare_sequence(seed, char2idx)
      # fill the RNN memory with the seed sentence
      seed_pred, state = model(inputs.unsqueeze(0))
      # now begin looping with feedback char by char from the last prediction
      preds = seed
      curr_pred = torch.topk(seed_pred[-1, :], k=1, dim=0)[1]
      curr_pred = idx2char[curr_pred.item()]
      preds += curr_pred
      for t in range(num_chars):
        curr_pred, state = model(prepare_sequence(curr_pred, char2idx).unsqueeze(0), state)
        curr_pred = torch.topk(curr_pred[-1, :], k=1, dim=0)[1]
        curr_pred = idx2char[curr_pred.item()]
        if curr_pred == '$':
          # special token to add newline char
          preds += '\n'
        else:
          preds += curr_pred
      return preds

      
print(gen_text(model, 'Monica was ', char2idx))

BATCH_SIZE = 64
T = len(training_data)
CHUNK_SIZE = T // BATCH_SIZE
# let's first chunk the huge train sequence into BATCH_SIZE sub-sequences
trainset = [training_data[beg_i:end_i] \
            for beg_i, end_i in zip(range(0, T - CHUNK_SIZE, CHUNK_SIZE),
                                    range(CHUNK_SIZE, T, CHUNK_SIZE))]
print('Original training string len: ', T)
print('Sub-sequences len: ', CHUNK_SIZE)
print(len(trainset[0]))

# Let's now build a model to train with its optimizer and loss
model = CharLSTM(VOCAB_SIZE, RNN_SIZE, MLP_SIZE)
model.to(device)
loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
NUM_EPOCHS = 5000
tr_loss = []
state = None
timer_beg = timer()
print(len(trainset))
for epoch in range(NUM_EPOCHS):
  model.train()
  # let's slide over our dataset
  for beg_t, end_t in zip(range(0, CHUNK_SIZE - SEQ_LEN - 1, SEQ_LEN + 1),
                          range(SEQ_LEN + 1, CHUNK_SIZE, SEQ_LEN + 1)):
    # Step 1. Remember that Pytorch accumulates gradients.
    # We need to clear them out before each instance
    optimizer.zero_grad()
    
    dataX = []
    dataY = []
    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Tensors of one-hot sequences. 
    for sent in trainset:
      # chunk the sentence
      chunk = sent[beg_t:end_t]
      # get X and Y with a shift of 1
      X = chunk[:-1]
      Y = chunk[1:]
      # convert each sequence to one-hots and labels respectively
      X = prepare_sequence(X, char2idx)
      Y = prepare_sequence(Y, char2idx, onehot=False)
      dataX.append(X.unsqueeze(0)) # create batch-dim
      dataY.append(Y.unsqueeze(0)) # create batch-dim
      
    dataX = torch.cat(dataX, dim=0).to(device)
    dataY = torch.cat(dataY, dim=0).to(device)
    

    # Step 3. Run our forward pass.
    # Forward through model and carry the previous state forward in time (statefulness)
    y_, state = model(dataX, state)
    print(y_.shape)
    # detach the previous state graph to not backprop gradients further than the BPTT span
    state = (state[0].detach(), # detach c[t]
             state[1].detach()) # detach h[t]

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function(y_, dataY.view(-1))
    loss.backward()
    optimizer.step()
    tr_loss.append(loss.item())
  timer_end = timer()  
  if (epoch + 1) % 50 == 0:
    # Generate a seed sentence to play around
    model.to('cpu')
    print('-' * 30) 
    print(gen_text(model, 'They ', char2idx))
    print('-' * 30)
    model.to(device)
    print('Finished epoch {} in {:.1f} s: loss: {:.6f}'.format(epoch + 1, 
                                                               timer_end - timer_beg,
                                                               np.mean(tr_loss[-10:])))
  timer_beg = timer()

plt.plot(tr_loss)
plt.xlabel('Epoch')
plt.ylabel('NLLLoss')
