# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Hello Kubeflow
#     language: python
#     name: hello-kubeflow
# ---

# %% [markdown]
# # constants

# %%
BATCH_SIZE = 10
CSV_FILE = '../data/mpg-pounds.csv'
EPOCHS = 100
LEARNING_RATE = 0.05


# %% [markdown]
# # train_pytorch

# %%
def train_pytorch():
    import os

    import torch
    from torch import nn
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    import pandas as pd
    from torch.utils.data import Dataset


    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    dist.init_process_group(backend=backend)
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    print(
        "Distributed Training with WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}.".format(
            dist.get_world_size(),
            dist.get_rank(),
            local_rank,
        )
    )

    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            
        def forward(self, x):
            return self.linear(x)

    device = torch.device(f"{device}:{local_rank}")
    model = nn.parallel.DistributedDataParallel(NeuralNetwork().to(device))
    model.train()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    class CustomDataset(Dataset):
        def __init__(self, csv_file):
            df = pd.read_csv(csv_file)
            self.pounds = torch.tensor(df["pounds"].values, dtype=torch.float32)
            self.mpg = torch.tensor(df["mpg"].values, dtype=torch.float32)
            
        def __len__(self):
            return self.pounds.shape[0]
        
        def __getitem__(self, idx):
            return self.pounds[idx].view(-1, 1), self.mpg[idx].view(-1, 1)
    
    dataset = CustomDataset(CSV_FILE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=DistributedSampler(dataset))
    for epoch in range(EPOCHS):
        for batch_idx, (batch_pounds, batch_mpg) in enumerate(dataloader):
            batch_pounds, batch_mpg = batch_pounds.to(device), batch_mpg.to(device)
            pred_mpg = model(batch_pounds)
            loss = loss_fn(pred_mpg, batch_mpg)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 10 == 0 and dist.get_rank() == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(batch_pounds),
                        len(dataloader.dataset),
                        100.0 * batch_idx / len(dataloader),
                        loss.item(),
                    )
                ) 
    dist.barrier()
    if dist.get_rank() == 0:
        print("Training is finished")
    dist.destroy_process_group()

# %% [markdown]
# # TODO

# %%
from kubeflow.trainer import TrainerClient, CustomTrainer
for r in TrainerClient().list_runtimes():
    print(f"Runtime: {r.name}")

# %% [markdown]
# # TODO

# %%
job_id = TrainerClient().train(
    trainer=CustomTrainer(
        func=train_pytorch,
        num_nodes=2,
        resources_per_node={
            "cpu": 5,
            "memory": "4Gi",
            # "gpu": 1, # Comment this line if you don't have GPUs.
        },
    ),
    runtime=TrainerClient().get_runtime("torch-distributed"),
)
