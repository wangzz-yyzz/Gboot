from torch_geometric.data import Data, Dataset


class PEMSDataset(Dataset):
    def __init__(self, x, y, norm_y, edge_index, ext, mean=None, std=None, trend_x=None, trend_y=None):
        super().__init__()
        # shape: [T, F, N]
        self.x = x
        self.y = y
        self.norm_y = norm_y
        self.edge_index = edge_index
        self.mean = mean
        self.std = std
        self.ext = ext
        self.trend_x = trend_x
        self.trend_y = trend_y

    def len(self):
        return self.x.shape[0]

    def get(self, index):
        x = self.x[index].T
        y = self.y[index].T
        norm_y = self.norm_y[index].T
        mean = self.mean
        std = self.std
        ext = self.ext[index]
        if self.trend_x is not None:
            trend_x = self.trend_x[index].mT
            trend_y = self.trend_y[index].mT
            return Data(x=x, y=y, edge_index=self.edge_index, norm_y=norm_y,
                        mean=mean, std=std, ext=ext, trend_x=trend_x, trend_y=trend_y)
        else:
            return Data(x=x, y=y, edge_index=self.edge_index, norm_y=norm_y,
                        mean=mean, std=std, ext=ext)
