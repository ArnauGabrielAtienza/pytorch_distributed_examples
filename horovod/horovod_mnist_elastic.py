import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim

import itertools

import horovod.torch as hvd

# HYPERPARAMETERS
epochs = 30
batches_per_commit = 30
lr = 0.01

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

# Initialize Horovod
hvd.init()

# Initialize model
model = Net()

# Initialize optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01 * hvd.size())
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

def get_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('mnist_data/', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, sampler=train_sampler)
    return train_loader

@hvd.elastic.run
def train(state):
    print(f'Loading Dataset')
    train_loader = get_dataset()
    
    print(f'Starting training')
    batch_offset = state.batch
    for state.epoch in range(state.epoch, epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            if(batch_idx >= batch_offset):
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                if state.batch % batches_per_commit == 0:
                    state.commit()
                    print(f'Processed batch {state.batch}/{len(train_loader)}')
                    print(f"Commited batch {state.batch} at epoch {state.epoch} with loss {loss.item()}")
                    print(f'Number of processes working: {hvd.size()} Current Rank: {hvd.rank()}')
                    
                state.batch = batch_idx
        state.batch = 0
        batch_offset=0
        
        
def on_state_reset():
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * hvd.size()

state = hvd.elastic.TorchState(model, optimizer, batch=0, epoch=0)
state.register_reset_callbacks([on_state_reset])
train(state)

# Example: horovodrun -np 2 --min-np=1 --blacklist-cooldown-range 15 30 --host-discovery-script ~/scripts/discover_hosts.sh python3 horovod_mnist_elastic.py