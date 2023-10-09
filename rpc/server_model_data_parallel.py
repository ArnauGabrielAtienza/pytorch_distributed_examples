"""
Implementation of the following Distributed Training paradigms using RPC:
    1. Model Parallelism: The model is split between a worker device, which holds a fully connected layer and 
a separate worker (parameter server) which holds an embedding table.
    2. Data parallelism: Each workers ingests a diferent subset of the training data.
    3. Parameter Server: To access the sparse embedding table, all workers connect to this special worker.
    
Execution example: "python3 server_mode_data_parallel.py"

TODO: Modify the script to work across multiple machines.
"""


import random

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.nn import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.rpc import TensorPipeRpcBackendOptions
from torch.nn.parallel import DistributedDataParallel as DDP


# Embedding table hyperparameters
NUM_EMBEDDINGS = 100
EMBEDDING_DIM = 16


class HybridModel(torch.nn.Module):
    """
    Hybrid Model consisting on a local Fully Connected layer and a remote Embedding Layer.
    """
    def __init__(self, remote_emb_module, device):
        super(HybridModel, self).__init__()
        self.remote_emb_module = remote_emb_module
        self.fc = DDP(torch.nn.Linear(16, 8), device_ids=None)
        self.device = device

    def forward(self, indices, offsets):
        emb_lookup = self.remote_emb_module.forward(indices, offsets)
        return self.fc(emb_lookup)


def get_next_batch():
        """
        Function to generate random data to feed the model
        """
        for _ in range(10):
            num_indices = random.randint(20, 50)
            indices = torch.LongTensor(num_indices).random_(0, NUM_EMBEDDINGS)

            # Generate offsets.
            offsets = []
            start = 0
            batch_size = 0
            while start < num_indices:
                offsets.append(start)
                start += random.randint(1, 10)
                batch_size += 1

            offsets_tensor = torch.LongTensor(offsets)
            target = torch.LongTensor(batch_size).random_(8)
            yield indices, offsets_tensor, target


def _run_trainer(remote_emb_module, rank):
    # 1) Setup the model.
    model = HybridModel(remote_emb_module, rank)

    # 2) Retrieve all model parameters as RRefs for DistributedOptimizer.
    
    # 2.1) Retrieve RRef to the parameters from embedding table.
    model_parameter_rrefs = model.remote_emb_module.remote_parameters()

    # 2.2) Retrieve local parameters as RRefs
    for param in model.fc.parameters():
        model_parameter_rrefs.append(RRef(param))

    # Setup distributed optimizer
    opt = DistributedOptimizer(
        optim.SGD,
        model_parameter_rrefs,
        lr=0.05,
    )

    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        for indices, offsets, target in get_next_batch(rank):
            # Train with Distributed Autograd
            with dist_autograd.context() as context_id:
                # Run forward pass and calculate the loss
                output = model(indices, offsets)
                loss = criterion(output, target)

                # Run distributed backward pass
                dist_autograd.backward(context_id, [loss])

                # Run distributed optimizer
                opt.step(context_id)

                # Not necessary to zero grads as each iteration creates a different
                # distributed autograd context which hosts different grads
                
        if epoch % 5 == 0:
            print("Training done for epoch {}".format(epoch))


def run_worker(rank, world_size):
    r"""
    A wrapper function that initializes RPC, calls the function, and shuts down
    RPC.
    """
    # We need to use different port numbers in TCP init_method for init_rpc and
    # init_process_group to avoid port conflicts.
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # Rank 2 is master, 3 is ps and 0 and 1 are trainers.
    if rank == 2:
        # Initialize master node
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        remote_emb_module = RemoteModule(
            "ps",
            torch.nn.EmbeddingBag,
            args=(NUM_EMBEDDINGS, EMBEDDING_DIM),
            kwargs={"mode": "sum"},
        )

        # Run the training loop on trainers.
        futs = []
        for trainer_rank in [0, 1]:
            trainer_name = "trainer{}".format(trainer_rank)
            fut = rpc.rpc_async(
                trainer_name, _run_trainer, args=(remote_emb_module, trainer_rank)
            )
            futs.append(fut)

        # Wait for all training to finish.
        for fut in futs:
            fut.wait()
    elif rank <= 1:
        # Initialize process group for Distributed DataParallel on trainers.
        dist.init_process_group(
            backend="gloo", rank=rank, world_size=2, init_method="tcp://localhost:29500"
        )

        # Initialize RPC.
        trainer_name = "trainer{}".format(rank)
        rpc.init_rpc(
            trainer_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )

        # Trainer just waits for RPCs from master.
    else:
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # parameter server do nothing
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 4
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)