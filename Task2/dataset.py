import pyarrow.parquet as pq
import numpy as np
import torch 
import torchvision.transforms as T

class ImageDatasetFromParquet(torch.utils.data.Dataset):
    def __init__(self, filename, transforms=[]) -> None:
        super().__init__()

        self.file = pq.ParquetFile(filename)

        self.transforms = T.Compose([
            T.ToTensor(),
            *transforms
        ])
    
    def __getitem__(self, idx, return_regress=False):
        row = self.file.read_row_group(idx).to_pydict()
        to_return = {
            "X_jets": self.transforms(np.array(row['X_jets'])),
            "y": row['y'][0]
        }
        
        if return_regress:
            to_return['pt'] = row['pt'][0]
            to_return['m0'] = row['m0'][0]

        return to_return

    def __len__(self):
        return self.file.num_row_groups
