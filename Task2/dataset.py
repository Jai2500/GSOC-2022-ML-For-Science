import pyarrow.parquet as pq
import numpy as np
import torch 
import torchvision.transforms as T

class ImageDatasetFromParquet(torch.utils.data.Dataset):
    def __init__(self, filename, transforms=[], return_regress=False) -> None:
        super().__init__()

        self.file = pq.ParquetFile(filename)

        self.transforms = T.Compose([
            T.ToTensor(),
            *transforms
        ])
        self.return_regress = return_regress
    
    def __getitem__(self, idx, ):
        row = self.file.read_row_group(idx).to_pydict()
        to_return = {
            "X_jets": self.transforms(np.array(row['X_jets'][0])),
            "y": row['y'][0]
        }
        
        if self.return_regress:
            to_return['pt'] = row['pt'][0]
            to_return['m0'] = row['m0'][0]

        return to_return

    def __len__(self):
        return self.file.num_row_groups
