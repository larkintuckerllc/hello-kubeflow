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
# # imports

# %%
import os

from kubeflow.trainer import TrainerClient, CustomTrainer


# %% [markdown]
# # training function

# %%
def train_pytorch():
    import os

    import pandas as pd
    import s3fs # s3fs is implicitly used by pandas for s3 paths
    import torch
    from torch import nn
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.utils.data import Dataset

    BATCH_SIZE = 10
    BUCKET_NAME = "hello-kubeflow"
    EPOCHS = 100
    LEARNING_RATE = 0.05
    OBJECT_PATH = "linear-regression/mpg-pounds.csv"

    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = torch.device(f"{device}:{local_rank}")
    print(f"Distributed Training with WORLD_SIZE: {world_size}, RANK: {rank}, LOCAL_RANK: {local_rank}.")

    class CustomDataset(Dataset):
        def __init__(self, bucket_name, object_path):
            df = pd.read_csv(f"s3://{bucket_name}/{object_path}")
            self.pounds = torch.tensor(df["pounds"].values, dtype=torch.float32)
            self.mpg = torch.tensor(df["mpg"].values, dtype=torch.float32)
            
        def __len__(self):
            return self.pounds.shape[0]
        
        def __getitem__(self, idx):
            return self.pounds[idx].view(-1, 1), self.mpg[idx].view(-1, 1) 

    distributed_batch_size = BATCH_SIZE // world_size
    dataset = CustomDataset(BUCKET_NAME, OBJECT_PATH)
    dataloader = DataLoader(dataset, batch_size=distributed_batch_size, sampler=DistributedSampler(dataset))

    class NeuralNetwork(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            
        def forward(self, x):
            return self.linear(x)

    model = nn.parallel.DistributedDataParallel(NeuralNetwork().to(device))
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_pounds, batch_mpg in dataloader:
            batch_pounds, batch_mpg = batch_pounds.to(device), batch_mpg.to(device)
            pred_mpg = model(batch_pounds)
            loss = loss_fn(pred_mpg, batch_mpg)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0 and rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.4f}')
    dist.barrier()
    if rank == 0:
        print("Training is finished")
    dist.destroy_process_group()

# %% [markdown]
# # validate runtime

# %%
for r in TrainerClient().list_runtimes():
    print(f"Runtime: {r.name}")

# %% [markdown]
# # create training job

# %%
job_id = TrainerClient().train(
    trainer=CustomTrainer(
        env={
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "AWS_SESSION_TOKEN": os.getenv("AWS_SESSION_TOKEN"),
        },
        func=train_pytorch,
        num_nodes=2,
        packages_to_install=[
            "s3fs==2025.7.0",
            "pandas==2.3.2",
        ],
        resources_per_node={
            "cpu": 2,
            "memory": "4Gi",
            # "gpu": 1, # Comment this line if you don't have GPUs.
        },
    ),
    runtime=TrainerClient().get_runtime("torch-distributed"),
)

# %% [markdown]
# # training job status

# %%
for s in TrainerClient().get_job(name=job_id).steps:
    print(f"Step: {s.name}, Status: {s.status}, Devices: {s.device} x {s.device_count}")


# %% [markdown]
# # training job output

# %%
logs = TrainerClient().get_job_logs(name=job_id)
print(logs["node-0"])
