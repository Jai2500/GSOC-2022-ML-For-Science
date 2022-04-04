from turtle import forward
import torch
import wandb
from tqdm.auto import tqdm
from torchmetrics import AUROC
from dataset import ImageDatasetFromParquet
import torchvision.transforms as T
import torchvision

DEVICE = "cuda"
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64

wandb.init(project="gsoc")


required_transform = [
    # T.Resize(224),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    # T.RandomAdjustSharpness(0.5, p=0.1),
]


run_0_path = "/scratch/gsoc/parquet_ds/QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet"
run_1_path = "/scratch/gsoc/parquet_ds/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540.test.snappy.parquet"
run_2_path = "/scratch/gsoc/parquet_ds/QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494.test.snappy.parquet"


run_0_ds = ImageDatasetFromParquet(run_0_path, transforms=required_transform, return_regress=True)
run_1_ds = ImageDatasetFromParquet(run_1_path, transforms=required_transform, return_regress=True)
run_2_ds = ImageDatasetFromParquet(run_2_path, transforms=required_transform, return_regress=True)

combined_dset = torch.utils.data.ConcatDataset([run_0_ds, run_1_ds])

TEST_SIZE = 0.15
VAL_SIZE = 0.15

test_size = int(len(combined_dset) * TEST_SIZE)
val_size = int(len(combined_dset) * VAL_SIZE)
train_size = len(combined_dset) - val_size - test_size

train_dset, val_dset, test_dset = torch.utils.data.random_split(
    combined_dset,
    [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42),
)


train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=TRAIN_BATCH_SIZE, pin_memory=True, num_workers=10)
val_loader = torch.utils.data.DataLoader(val_dset, shuffle=False, batch_size=VAL_BATCH_SIZE, pin_memory=True, num_workers=10)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=TEST_BATCH_SIZE, num_workers=10)

####################### MODEL #########################

# model = torch.hub.load(
#     "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False
# )

class RegressModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        in_features = self.model.fc.in_features
        self.model.fc = torch.nn.Identity()

        self.out_lin = torch.nn.Linear(in_features + 1, 1, bias=True)

    def forward(self, X, pt):
        out = self.model(X)
        out = torch.cat([out, pt.unsqueeze(-1)], dim=1)
        return self.out_lin(out)

model = RegressModel(
    model=torchvision.models.resnet34(pretrained=True)
)

model = model.to(DEVICE)

####################### OPTIMIZER #####################

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

######################## CRITERION ####################

criterion = torch.nn.MSELoss()

######################## TRAIN #######################

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    model.train()
    tqdm_iter = tqdm(train_loader, total=len(train_loader))
    tqdm_iter.set_description(f"Epoch {epoch}")

    for it, batch in enumerate(tqdm_iter):
        optimizer.zero_grad()

        X, pt, m0 = batch['X_jets'].float(), batch['pt'].float(), batch['m0'].float()

        X = X.to(DEVICE, non_blocking=True)
        pt = pt.to(DEVICE, non_blocking=True)
        m0 = m0.to(DEVICE, non_blocking=True)

        out = model(X, pt)

        loss = criterion(out, m0.unsqueeze(-1))

        tqdm_iter.set_postfix(loss=loss.item())
        wandb.log({
            "train_mse_loss": loss.item(),
            "train_step": (it * TRAIN_BATCH_SIZE) + epoch * train_size
        })

        loss.backward()
        optimizer.step()

    model.eval()
    val_tqdm_iter = tqdm(val_loader, total=len(val_loader))
    val_tqdm_iter.set_description(f"Validation Epoch {epoch}")

    for it, batch in enumerate(val_tqdm_iter):
        with torch.no_grad():
            X, pt, m0 = batch['X_jets'].float(), batch['pt'].float(), batch['m0'].float()

            X = X.to(DEVICE, non_blocking=True)
            pt = pt.to(DEVICE, non_blocking=True)
            m0 = m0.to(DEVICE, non_blocking=True)

            out = model(X, pt)

            loss = criterion(out, m0.unsqueeze(-1))

            val_tqdm_iter.set_postfix(loss=loss.item())
            wandb.log({
                "val_mse_loss": loss.item(),
                "val_step": (it * VAL_BATCH_SIZE) + epoch * val_size
            })
