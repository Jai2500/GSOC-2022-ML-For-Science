import torch
import wandb
from tqdm.auto import tqdm
from torchmetrics import AUROC
from dataset import ImageDatasetFromParquet

DEVICE = "cuda"
TRAIN_BATCH_SIZE = 128
VAL_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128

wandb.init(project="gsoc")

run_0_path = "/scratch/gsoc/parquet_ds/QCDToGGQQ_IMGjet_RH1all_jet0_run0_n36272.test.snappy.parquet"
run_1_path = "/scratch/gsoc/parquet_ds/QCDToGGQQ_IMGjet_RH1all_jet0_run1_n47540.test.snappy.parquet"
run_2_path = "/scratch/gsoc/parquet_ds/QCDToGGQQ_IMGjet_RH1all_jet0_run2_n55494.test.snappy.parquet"


run_0_ds = ImageDatasetFromParquet(run_0_path)
run_1_ds = ImageDatasetFromParquet(run_1_path)
run_2_ds = ImageDatasetFromParquet(run_2_path)

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


train_loader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_dset, shuffle=False, batch_size=VAL_BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_dset, shuffle=False, batch_size=TEST_BATCH_SIZE)

####################### MODEL #########################

model = torch.hub.load(
    "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False
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
