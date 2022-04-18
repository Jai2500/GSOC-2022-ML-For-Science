import pyarrow.parquet as pq
import numpy as np
import torch 
import torch_geometric

class PointCloudFromParquetDataset(torch.utils.data.Dataset):
    def __init__(self, filename) -> None:
        super().__init__()

        self.file = pq.ParquetFile(filename)

    def __getitem__(self, idx, ):
        row = self.file.read_row_group(idx).to_pydict()
        
        arr = np.array(row['X_jets'][0])
        idx = np.where(arr.sum(axis=0) > 0)
        pos = np.array(idx).T / arr.shape[1]
        x = arr[:, idx[0], idx[1]].T
        x = np.concatenate([x, pos], axis=1)
        
        y = row['y'][0]
        pt = row['pt'][0]
        m0 = row['m0'][0]

        data = torch_geometric.data.Data(pos=pos, x=x, y=y, pt=pt, m0=m0)

        return data

    def __len__(self):
        return self.file.num_row_groups
