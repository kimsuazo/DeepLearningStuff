import numpy as np
SEED = 1
np.random.seed(SEED)
import torch
import torch.optim as optim
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(SEED)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import random
random.seed(SEED)
from random import shuffle


# Let's define some hyper-parameters
hparams = {
    'batch_size':100,
    'num_epochs':12,
    'val_batch_size':100,
    'num_classes':10,
    'learning_rate':1e-3,
    'log_interval':100,
}

# we select to work on GPU if it is available in the machine, otherwise
# will run on CPU
hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# whenever we send something to the selected device (X.to(device)) we already use
# either CPU or CUDA (GPU). Importantly...
# The .to() operation is in-place for nn.Module's, so network.to(device) suffices
# The .to() operation is NOT in.place for tensors, so we must assign the result
# to some tensor, like: X = X.to(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = datasets.CIFAR10('data', train=True, download=True,
                            transform=transform)

trainset.data = trainset.data[:10000]
trainset.targets = trainset.targets[:10000]

evalset = datasets.CIFAR10('data', train=False, 
                           transform=transform)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=hparams['batch_size'], 
    shuffle=True)

eval_loader = torch.utils.data.DataLoader(
    evalset,
    batch_size=hparams['val_batch_size'], 
    shuffle=False)

# We can retrieve a sample from the dataset by simply indexing it
img, label = trainset[0]
print('Img shape: ', img.shape)
print('Label: ', label)

# Similarly, we can sample a BATCH from the dataloader by running over its iterator
iter_ = iter(train_loader)
bimg, blabel = next(iter_)
print('Batch Img shape: ', bimg.shape)
print('Batch Label shape: ', blabel.shape)
print('The Batched tensors return a collection of {} RGB images ({} channel, {} height pixels, {} width pixels)'.format(bimg.shape[0],
                                                                                                                        bimg.shape[1],
                                                                                                                        bimg.shape[2],
                                                                                                                        bimg.shape[3]))
print('In the case of the labels, we obtain {} batched integers, one per image'.format(blabel.shape[0]))

def plot_samples(images,N=5):
    # Randomly select NxN images and save them in ps
    ps = random.sample(range(0,images.shape[0]), N**2)
    # Allocates figure f divided in subplots contained in an NxN axarr
    # https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.subplots.html
    f, axarr = plt.subplots(N, N)
    # Index for the images in ps to be plotted
    p = 0
    # Scan the NxN positions of the grid
    for i in range(N):
        for j in range(N):
          
            # Load the image pointed by p
            im = images[ps[p]].transpose(1,2,0)
            axarr[i,j].imshow(im)
            # Remove axis
            axarr[i,j].axis('off')
            # Point to the next image from the random selection
            p+=1
    # Show the plotted figure         
    plt.show()

#make a toy CIFAR sampleset without normalization for the sake of visualizing
# some samples
sampleset = datasets.CIFAR10('data', train=True, download=True, 
                             transform=transforms.Compose([
                                 transforms.ToTensor()
                             ]))
img = []
for n in range(50):
  img.append(sampleset[n][0][None, ...])
img = torch.cat(img, dim=0).data.numpy()
plot_samples(np.array(img))

def correct_predictions(predicted_batch, label_batch):
  pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
  acum = pred.eq(label_batch.view_as(pred)).sum().item()
  return acum

def train_epoch(train_loader, network, optimizer, criterion, hparams, epoch):
  # Activate the train=True flag inside the model
  network.train()
  device = hparams['device']
  losses = []
  accs = []
  for batch_idx, (data, target) in enumerate(train_loader, 1):
      data, target = data.to(device), target.to(device)
      optimizer.zero_grad()
      output = network(data)
      loss = criterion(output, target)
      loss.backward()
      acc = 100 * (correct_predictions(output, target) / data.shape[0])
      losses.append(loss.item())
      accs.append(acc)
      optimizer.step()
      if batch_idx % hparams['log_interval'] == 0 or batch_idx >= len(train_loader):
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.1f}'.format(
              epoch, batch_idx * len(data), len(train_loader.dataset),
              100. * batch_idx / len(train_loader), loss.item(),
              acc))
  return np.mean(losses), np.mean(accs)

def eval_epoch(val_loader, network, criterion, hparams):
    network.eval()
    device = hparams['device']
    eval_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            eval_loss += criterion(output, target, reduction='sum').item() # sum up batch loss
            # compute number of correct predictions in the batch
            acc += correct_predictions(output, target)
    # Average acc across all correct predictions batches now
    eval_loss /= len(eval_loader.dataset)
    eval_acc = 100. * acc / len(eval_loader.dataset)
    print('Eval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        eval_loss, acc, len(eval_loader.dataset), eval_acc,
        ))
    return eval_loss, eval_acc

def train_net(network, train_loader, optimizer, num_epochs, plot=True):
  """ Function that trains and evals a network for num_epochs,
      showing the plot of losses and accs and returning them.
  """
  tr_losses = []
  tr_accs = []
  te_losses = []
  te_accs = []

  network.to(hparams['device'])
  criterion = F.nll_loss

  for epoch in range(1, num_epochs + 1):
    tr_loss, tr_acc = train_epoch(train_loader, network, optimizer, criterion, hparams, epoch)
    te_loss, te_acc = eval_epoch(eval_loader, network, criterion, hparams)
    te_losses.append(te_loss)
    te_accs.append(te_acc)
    tr_losses.append(tr_loss)
    tr_accs.append(tr_acc)
  rets = {'tr_losses':tr_losses, 'te_losses':te_losses,
          'tr_accs':tr_accs, 'te_accs':te_accs}
  if plot:
    plt.figure(figsize=(10, 8))
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('NLLLoss')
    plt.plot(tr_losses, label='train')
    plt.plot(te_losses, label='eval')
    plt.legend()
    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Eval Accuracy [%]')
    plt.plot(tr_accs, label='train')
    plt.plot(te_accs, label='eval')
    plt.legend()
  return rets

  class ConvBlock(nn.Module):

  def __init__(self, num_inp_channels, num_out_fmaps, 
               kernel_size, stride=1):
    super().__init__()
    self.kernel_size = kernel_size
    self.stride = stride
    self.conv = nn.Conv2d(num_inp_channels, num_out_fmaps, kernel_size,
                          stride=stride)
    self.relu = nn.ReLU(inplace=True)
  
def forward(self, x):
    P_ = self.kernel_size // 2
    if self.stride > 1:
      P = (P_ - 1, P_, P_  - 1, P_ )
    else:
      P = (P_, P_, P_, P_)
    x = self.conv(F.pad(x, P, mode='constant'))
    return self.relu(x)

    def model_params(model):
    # from: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class BigNet(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv1 = ConvBlock(3, 512, 3, stride=4)
    self.conv2 = ConvBlock(512, 1024, 3, stride=4)
    self.mlp = nn.Sequential(
        nn.Linear(1024 * 2 * 2, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, hparams['num_classes']),
        nn.LogSoftmax(dim=-1)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    bsz, nch, height, width = x.shape
    x = x.view(bsz, -1)
    y = self.mlp(x)
    return y

bignet = BigNet()
# let's check the num of params
nparams = model_params(bignet)
print('BigNet number of params: ', nparams)
optimizer = optim.Adam(bignet.parameters(), lr=hparams['learning_rate'])

bigmodel_log = train_net(bignet, train_loader, optimizer, hparams['num_epochs'])


class SmallNet(BigNet):

  def __init__(self):
    super().__init__()
    self.conv1 = ConvBlock(3, 128, 3, stride=4)
    self.conv2 = ConvBlock(128, 256, 3, stride=4)
    self.mlp = nn.Sequential(
        nn.Linear(256 * 2 * 2, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, hparams['num_classes']),
        nn.LogSoftmax(dim=-1)
    )
    
smallnet = SmallNet()
# let's check the num of params
nparams = model_params(smallnet)
print('SmallNet number of params: ', nparams)
optimizer = optim.Adam(smallnet.parameters(), lr=hparams['learning_rate'])

smallmodel_log = train_net(smallnet, train_loader, optimizer, hparams['num_epochs'])

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(bigmodel_log['te_losses'], label='bigmodel')
plt.plot(smallmodel_log['te_losses'], label='bigmodel')
plt.xlabel('Epoch')
plt.ylabel('Eval Loss')
plt.subplot(2,1,2)
plt.plot(bigmodel_log['te_accs'], label='bigmodel')
plt.plot(smallmodel_log['te_accs'], label='bigmodel')
plt.xlabel('Epoch')
plt.ylabel('Eval Accs')

# TO DO
bignet = BigNet()
optimizer = optim.Adam(bignet.parameters(), lr=hparams['learning_rate'], weight_decay = 0.001)
bigmodel_l2_log = train_net(bignet, train_loader, optimizer, hparams['num_epochs'])

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(bigmodel_log['te_losses'], label='bigmodel')
plt.plot(smallmodel_log['te_losses'], label='smallmodel')
plt.plot(bigmodel_l2_log['te_losses'], label='bigmodel L2')
plt.xlabel('Epoch')
plt.ylabel('Eval Loss')
plt.subplot(2,1,2)
plt.plot(bigmodel_log['te_accs'], label='bigmodel')
plt.plot(smallmodel_log['te_accs'], label='smallmodel')
plt.plot(bigmodel_l2_log['te_accs'], label='bigmodel L2')
plt.xlabel('Epoch')
plt.ylabel('Eval Accs')

# TO DO
class BigNetDropout(nn.Module):

  def __init__(self, dropout):
    super().__init__()
    self.conv1 = ConvBlock(3, 512, 3, stride=4)
    self.drop1 = nn.Dropout(dropout)
    self.conv2 = ConvBlock(512, 1024, 3, stride=4)
    self.drop2 = nn.Dropout(dropout)
    self.mlp = nn.Sequential(
        nn.Linear(1024 * 2 * 2, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(1024, hparams['num_classes']),
        nn.LogSoftmax(dim=-1)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.drop1(x)
    x = self.conv2(x)
    x = self.drop2(x)
    bsz, nch, height, width = x.shape
    x = x.view(bsz, -1)
    y = self.mlp(x)
    return y

dropout = 0.5
# TO DO
bignet = BigNetDropout(dropout)
optimizer = optimizer = optim.Adam(bignet.parameters(), lr=hparams['learning_rate'])
bigmodel_dout_log = train_net(bignet, train_loader, optimizer, hparams['num_epochs'])


plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(bigmodel_log['te_losses'], label='bigmodel')
plt.plot(smallmodel_log['te_losses'], label='smallmodel')
plt.plot(bigmodel_l2_log['te_losses'], label='bigmodel L2')
plt.plot(bigmodel_dout_log['te_losses'], label='bigmodel Dropout')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Eval Loss')
plt.subplot(2,1,2)
plt.plot(bigmodel_log['te_accs'], label='bigmodel')
plt.plot(smallmodel_log['te_accs'], label='smallmodel')
plt.plot(bigmodel_l2_log['te_accs'], label='bigmodel L2')
plt.plot(bigmodel_dout_log['te_accs'], label='bigmodel Dropout')
plt.legend()
plt.xlabel('Epoch')
_ = plt.ylabel('Eval Accs')

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(bigmodel_log['tr_losses'], label='bigmodel')
plt.plot(smallmodel_log['tr_losses'], label='smallmodel')
plt.plot(bigmodel_l2_log['tr_losses'], label='bigmodel L2')
plt.plot(bigmodel_dout_log['tr_losses'], label='bigmodel Dropout')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.subplot(2,1,2)
plt.plot(bigmodel_log['tr_accs'], label='bigmodel')
plt.plot(smallmodel_log['tr_accs'], label='smallmodel')
plt.plot(bigmodel_l2_log['tr_accs'], label='bigmodel L2')
plt.plot(bigmodel_dout_log['tr_accs'], label='bigmodel Dropout')
plt.legend()
plt.xlabel('Epoch')
_ = plt.ylabel('Train Accs')

# TO DO

big_trainset = datasets.CIFAR10('data', train=True, download=True,
                            transform=transform)

big_loader = torch.utils.data.DataLoader(
    big_trainset,
    batch_size=hparams['batch_size'], 
    shuffle=True)

bignet = BigNetDropout(dropout)
optimizer = optimizer = optim.Adam(bignet.parameters(), lr=hparams['learning_rate'])
bigmodel_bigtrainset_dout_log = train_net(bignet, big_loader, optimizer, hparams['num_epochs'])

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(bigmodel_log['te_losses'], label='bigmodel')
plt.plot(bigmodel_dout_log['te_losses'], label='bigmodel Dropout')
plt.plot(bigmodel_bigtrainset_dout_log['te_losses'], label='bigmodel bigtrainset Dropout')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.subplot(2,1,2)
plt.plot(bigmodel_log['te_accs'], label='bigmodel')
plt.plot(bigmodel_dout_log['te_accs'], label='bigmodel Dropout')
plt.plot(bigmodel_bigtrainset_dout_log['te_accs'], label='bigmodel bigtrainset Dropout')
plt.legend()
plt.xlabel('Epoch')
_ = plt.ylabel('Train Accs')