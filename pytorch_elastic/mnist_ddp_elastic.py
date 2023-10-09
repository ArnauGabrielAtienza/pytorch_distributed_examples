"""
Implementation of a Distributed Data Parallel training using Torch Elastic
https://pytorch.org/docs/stable/distributed.elastic.html

Example for running on 2 different machines: 
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=172.31.76.253:29603 mnist_ddp_elastic.py 10 5
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms, datasets
import os
import time


def ddp_setup():
    """
    Initialize the communication backend for CPU.
    """
    init_process_group(backend="gloo")
    os.environ["OMP_NUM_THREADS"] = "1"


class Trainer:
    """
    Trainer class that implements the training methods, optimizer, criterion, data...
    """
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        test_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.global_rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optimizer
        self.criterion = criterion
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=None)


    def _load_snapshot(self, snapshot_path):
        """
        Load previous backup of the model parameters
        """
        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")


    def _run_batch(self, source, targets):
        """
        Run training on a given batch of data
        """
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()


    def _run_epoch(self, epoch):
        """
        Run an entire epoch with the data given during the Trainer initialization
        """
        self.model.train()
        b_sz = len(next(iter(self.train_data))[0])
        print(f"Local Rank: {self.local_rank} | Global Rank: {self.global_rank} | Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            self._run_batch(source, targets)
        self.test()


    def _save_snapshot(self, epoch):
        """
        Make a backup of the model parameters
        """
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")


    def train(self, max_epochs: int):
        """
        Train the model for a given number of epochs
        """
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
    
    
    def test(self):
        """
        Test the model's accuracy
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.test_data:
                outputs = self.model(images)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'Test accuracy: {(correct/total)*100:.2f}%')
        

class Model(nn.Module):
    """
    Simple model consisting on fully connected layers
    """
    def __init__(self, hidden_layers=1, features=128):
        super().__init__()
        
        self.input_layer = nn.Linear(784, features)
        
        layers = []
        for _ in range(hidden_layers):
            layers.append(nn.Linear(features, features))
        self.hidden_layers = nn.ModuleList(layers)
        
        self.final_layer = nn.Linear(features, out_features=10)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        intermediate_result = self.relu(self.input_layer(x))
        
        for layer in self.hidden_layers:
            intermediate_result = self.relu(layer(intermediate_result))
        
        return self.final_layer(intermediate_result)


def load_train_objs():
    """
    Initialize the model, the data, the optimizer and the criterion
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST('mnist_data/', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('mnist_data/', train=False, download=True, transform=transform)
    model = Model(hidden_layers=5, features=1024)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    return train_set, test_set, model, optimizer, criterion


def prepare_dataloader(dataset: Dataset, batch_size: int):
    """
    Create the DataLoader for a given DataSet. Use a Distributed Sampler
    to give each worker a different sample of data during training.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    dataset, test_dataset, model, optimizer, criterion = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    test_data = prepare_dataloader(test_dataset, batch_size)
    trainer = Trainer(model, train_data, test_data, optimizer, criterion, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=128, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    start = time.time()
    main(args.save_every, args.total_epochs, args.batch_size)
    end = time.time()
    print(f'Execution time: {end-start}')