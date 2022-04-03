import torch
from tqdm.auto import tqdm
import torchvision.transforms as T
import wandb
from torchmetrics import AUROC

from dataset import ImageDatasetFromHDF5

DEVICE = "cuda"
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

wandb.init(project="gsoc")

######################## DATASET ########################

required_transform = []

photon_file_path = "/scratch/gsoc/SinglePhotonPt50_IMGCROPS_n249k_RHv1.hdf5"
electron_file_path = "/scratch/gsoc/SingleElectronPt50_IMGCROPS_n249k_RHv1.hdf5"

photon_dset = ImageDatasetFromHDF5(
    photon_file_path, required_transforms=required_transform
)
electron_dset = ImageDatasetFromHDF5(
    electron_file_path, required_transforms=required_transform
)

combined_dset = torch.utils.data.ConcatDataset([photon_dset, electron_dset])

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


train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_dset, shuffle=False, batch_size=VAL_BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=TEST_BATCH_SIZE)

####################### MODEL #########################

model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False
)

model.conv1 = torch.nn.Conv2d(
    2, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
)

in_features = model.fc.in_features

model.fc = torch.nn.Linear(in_features, 1, bias=True)

model = model.to(DEVICE)

####################### OPTIMIZER #####################

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

######################## CRITERION ####################

criterion = torch.nn.BCEWithLogitsLoss()
train_auroc = AUROC(num_classes=None)
val_auroc = AUROC(num_classes=None)

######################## TRAIN #######################

NUM_EPOCHS = 10

for epoch in range(NUM_EPOCHS):
    model.train()
    tqdm_iter = tqdm(train_loader, total=len(train_loader))
    tqdm_iter.set_description(f"Epoch {epoch}")

    for it, batch in enumerate(tqdm_iter):
        optimizer.zero_grad()

        X, y = batch[0], batch[1]

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        out = model(X)

        loss = criterion(out, y.unsqueeze(-1))

        pred = torch.sigmoid(out.detach()).squeeze()
        acc = ((pred >= 0.5) == y).float().mean()
        train_auroc.update(pred, y.detach().long())

        tqdm_iter.set_postfix(loss=loss.item(), acc=acc.item())
        wandb.log({
            "train_loss": loss.item(),
            "train_acc": acc.item(),
            "train_step": (it * TRAIN_BATCH_SIZE) + epoch * train_size
        })

        loss.backward()
        optimizer.step()

    wandb.log({
        "train_auroc": train_auroc.compute(),
        "train_epoch": epoch
    })
    train_auroc.reset()

    model.eval()
    val_tqdm_iter = tqdm(val_loader, total=len(val_loader))
    val_tqdm_iter.set_description(f"Validation Epoch {epoch}")

    for it, batch in enumerate(val_tqdm_iter):
        with torch.no_grad():
            X, y = batch[0], batch[1]

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            out = model(X)

            loss = criterion(out, y.unsqueeze(-1))

            pred = torch.sigmoid(out.detach()).squeeze() 
            acc = ((pred >= 0.5) == y).float().mean()
            val_auroc.update(pred, y.long())

            val_tqdm_iter.set_postfix(loss=loss.item(), acc=acc.item())
            wandb.log({
                "val_loss": loss.item(),
                "val_acc": acc.item(),
                "val_step": (it * VAL_BATCH_SIZE) + epoch * val_size
            })

    wandb.log({
        "val_auroc": val_auroc.compute(),
        "val_epoch": epoch
    })
    val_auroc.reset()



    