import torch_geometric
import torch

class MLPStack(torch.nn.Module):
    def __init__(self, layers, bn=True, act=True):
        super().__init__()
        assert len(layers) > 1, "At least input and output channels must be provided"

        modules = []
        for i in range(1, len(layers)):
            modules.append(
                torch.nn.Linear(layers[i-1], layers[i])
            )
            modules.append(
                torch.nn.BatchNorm1d(layers[i]) if bn == True else torch.nn.Identity()
            )
            modules.append(
                torch.nn.SiLU() if bn == True else torch.nn.Identity()
            )

        self.mlp_stack = torch.nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp_stack(x)

class DynamicEdgeConvPN(torch.nn.Module):
    def __init__(self, edge_nn, nn, k=7, aggr='max', flow='source_to_target') -> None:
        super().__init__()
        self.nn = nn
        self.k = k
        self.edge_conv = torch_geometric.nn.EdgeConv(nn=edge_nn, aggr=aggr)
        self.flow = flow

    def forward(self, x, pos, batch):
        edge_index = torch_geometric.nn.knn_graph(x=pos, k=self.k, batch=batch, flow=self.flow)

        edge_out = self.edge_conv(x, edge_index)

        x_out = self.nn(x)

        return edge_out + x_out


class DGCNN(torch.nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.dynamic_conv_1 = DynamicEdgeConvPN(
            edge_nn=MLPStack(
                [10, 32, 32, 32], bn=True, act=True
            ),
            nn=MLPStack(
                [5, 32, 32, 32], bn=True, act=True
            ),
            k=k
        )

        self.dynamic_conv_2 = DynamicEdgeConvPN(
            edge_nn=MLPStack(
                [64, 64, 64, 64], bn=True, act=True
            ),
            nn=MLPStack(
                [32, 64, 64, 64], bn=True, act=True
            ),
            k=k
        )

        self.out_nn = torch.nn.Linear(64, 1)

    def forward(self, data):
        x = data.x
        pos = data.pos
        batch = data.batch

        x_out = self.dynamic_conv_1(
            x, pos, batch
        )
        x_out = self.dynamic_conv_2(
            x_out, x_out, batch
        )

        x_out = torch_geometric.nn.global_mean_pool(x_out, batch)

        return self.out_nn(x_out)

class SimpleGAT(torch.nn.Module):
    def __init__(self):
        super().__init__()