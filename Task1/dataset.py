import torch
import torchvision.transforms as T
import h5py
import numpy as np

class ImageDatasetFromHDF5(torch.utils.data.Dataset):
    def __init__(self, file_path, required_transforms=[]):
        super().__init__()
        self.file_path = file_path

        f = h5py.File(file_path)
        self.X = np.array(f["X"])
        self.y = np.array(f["y"])

        self.transforms = T.Compose([T.ToTensor(), *required_transforms])

    def __getitem__(self, idx):
        return (self.transforms(self.X[idx]), self.y[idx])

    def __len__(self):
        return len(self.X)
