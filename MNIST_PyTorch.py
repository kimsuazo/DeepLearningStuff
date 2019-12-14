import numpy as np
np.random.seed(1)
import torch
import torch.optim as optim
torch.manual_seed(1)

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer


# Let's define some hyper-parameters ----------------------------------------------------------
hparams = {
    'batch_size':64,
    'num_epochs':10,
    'test_batch_size':64,
    'hidden_size':128,
    'num_classes':10,
    'num_inputs':784,
    'learning_rate':1e-3,
    'log_interval':100,
}

# we select to work on GPU if it is available in the machine, otherwise
# will run on CPU
hparams['device'] = 'cpu' #'cuda' if torch.cuda.is_available() else 'cpu'
print(hparams['device'])

# whenever we send something to the selected device (X.to(device)) we already use
# either CPU or CUDA (GPU). Importantly...
# The .to() operation is in-place for nn.Module's, so network.to(device) suffices
# The .to() operation is NOT in.place for tensors, so we must assign the result
# to some tensor, like: X = X.to(device)

# Import Datasets -----------------------------------------------------------------------------

mnist_trainset = datasets.MNIST('data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
mnist_testset = datasets.MNIST('data', train=False, 
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))

train_loader = torch.utils.data.DataLoader(
    mnist_trainset,
    batch_size=hparams['batch_size'], 
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    mnist_testset,
    batch_size=hparams['test_batch_size'], 
    shuffle=False)


NUM_BITS_FLOAT32 = 32


class ConvBlock(nn.Module):

  def __init__(self, num_inp_channels, num_out_fmaps, 
               kernel_size, pool_size=2):
    super().__init__()
    # TODO: define the 3 modules needed
    self.conv = nn.Conv2d(num_inp_channels, num_out_fmaps, kernel_size)
    self.relu = nn.ReLU()
    self.maxpool = nn.MaxPool2d(pool_size)
  
  def forward(self, x):
    return self.maxpool(self.relu(self.conv(x)))

x = torch.randn(1, 1, 32, 32)
y = ConvBlock(1, 6, 5, 2)(x)
assert y.shape[1] == 6, 'The amount of feature maps is not correct!'
assert y.shape[2] == 14 and y.shape[3] == 14, 'The spatial dimensions are not correct!'
print('Input shape: {}'.format(x.shape))
print('ConvBlock output shape (S2 level in Figure): {}'.format(y.shape))


class PseudoLeNet(nn.Module):

  def __init__(self):
    super().__init__()
    # TODO: Define the padding
    self.pad = nn.ConstantPad2d(2, 0)
    self.conv1 = ConvBlock(1, 6, 5)
    self.conv2 = ConvBlock(6, 16, 5)
    # TODO: Define the MLP at the deepest layers
    self.mlp = nn.Sequential(
        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10),
        nn.LogSoftmax(dim = 1)
    )

  def forward(self, x):
    x = self.pad(x)
    x = self.conv1(x)
    x = self.conv2(x)
    # Obtain the parameters of the tensor in terms of:
    # 1) batch size
    # 2) number of channels
    # 3) spatial "height"
    # 4) spatial "width"
    bsz, nch, height, width = x.shape
    # TODO: Flatten the feature map with the view() operator 
    # within each batch sample    
    x = x.view(bsz, nch * height * width)
    y = self.mlp(x)
    return y

# Let's forward a toy example emulating the MNIST image size
plenet = PseudoLeNet()
y = plenet(torch.randn(1, 1, 28, 28))
print(y.shape)

def correct_predictions(predicted_batch, label_batch):
  pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
  acum = pred.eq(label_batch.view_as(pred)).sum().item()
  return acum

def train_epoch(train_loader, network, optimizer, criterion, hparams):
  # Activate the train=True flag inside the model
  network.train()
  device = hparams['device']
  avg_loss = None
  avg_weight = 0.1
  for batch_idx, (data, target) in enumerate(train_loader):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = network(data)
      loss = criterion(output, target)
      loss.backward()
      if avg_loss:
        avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
      else:
        avg_loss = loss.item()
      optimizer.step()
      if batch_idx % hparams['log_interval'] == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item()))
  return avg_loss

def test_epoch(test_loader, network, hparams):
    network.eval()
    device = hparams['device']
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += criterion(output, target, reduction='sum').item() # sum up batch loss
            # compute number of correct predictions in the batch
            acc += correct_predictions(output, target)
    # Average acc across all correct predictions batches now
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * acc / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))
    return test_loss, test_acc


#Run over few Epochs ---------------------------------------------------------
tr_losses = []
te_losses = []
te_accs = []
network = PseudoLeNet()
network.to(hparams['device'])
optimizer = optim.RMSprop(network.parameters(), lr=hparams['learning_rate'])
criterion = F.nll_loss

for epoch in range(1, hparams['num_epochs'] + 1):
  tr_losses.append(train_epoch(train_loader, network, optimizer, criterion, hparams))
  te_loss, te_acc = test_epoch(test_loader, network, hparams)
  te_losses.append(te_loss)
  te_accs.append(te_acc)

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.xlabel('Epoch')
plt.ylabel('NLLLoss')
plt.plot(tr_losses, label='train')
plt.plot(te_losses, label='test')
plt.legend()
plt.subplot(2,1,2)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy [%]')
plt.plot(te_accs)