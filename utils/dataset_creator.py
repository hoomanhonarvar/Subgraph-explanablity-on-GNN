

import torch
import numpy as np
import pickle as pkl
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.datasets.motif_generator import HouseMotif, CycleMotif
from typing import Optional, Callable, List
import os


class BA2MotifWithGroundTruth(InMemoryDataset):
    


    filename = 'BA-2motif-with-gt.pkl'
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.force_reload = force_reload
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return self.filename

    @property
    def processed_file_names(self) -> List[str]:
        return 'data.pt'

    def process(self):
        dataset1 = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=25, num_edges=1),
            motif_generator=HouseMotif(),
            num_motifs=1,
            num_graphs=500,
        )
        dataset2 = ExplainerDataset(
            graph_generator=BAGraph(num_nodes=25, num_edges=1),
            motif_generator=CycleMotif(5),
            num_motifs=1,
            num_graphs=500,
        )

        data_list: List[Data] = []
        pkl_data_list = []

        for exp in dataset1:
            if exp.x is None:
                num_nodes = exp.edge_index.max().item() + 1
                x = torch.ones(num_nodes, 10)
            else:
                x = exp.x

            y_val = exp.y
            if y_val.dim() == 0:
                y_val = y_val.item()
            elif y_val.numel() == 1:
                y_val = y_val.item()
            else:
                y_val = y_val.argmax().item()

            data = Data(
                x=x,
                edge_index=exp.edge_index,
                y=y_val,
                node_mask=exp.node_mask.bool(),
                edge_mask=exp.edge_mask.bool(),
                motif_nodes=torch.where(exp.node_mask)[0],
            )
            if self.pre_transform:
                data = self.pre_transform(data)
            data_list.append(data)

            adj = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float)
            adj[data.edge_index[0], data.edge_index[1]] = 1.0
            pkl_data_list.append({
                'adj': adj.numpy(),
                'x': data.x.numpy(),
                'y': data.y.item() if isinstance(data.y, torch.Tensor) else data.y,
                'node_mask': data.node_mask.numpy(),
                'edge_mask': data.edge_mask.numpy(),
            })

        for exp in dataset2:
            if exp.x is None:
                num_nodes = exp.edge_index.max().item() + 1
                x = torch.ones(num_nodes, 10)
            else:
                x = exp.x

            y_val = exp.y
            if y_val.dim() == 0:
                y_val = y_val.item()
            elif y_val.numel() == 1:
                y_val = y_val.item()
            else:
                y_val = y_val.argmax().item()

            data = Data(
                x=x,
                edge_index=exp.edge_index,
                y=y_val,
                node_mask=exp.node_mask.bool(),
                edge_mask=exp.edge_mask.bool(),
                motif_nodes=torch.where(exp.node_mask)[0],
            )
            if self.pre_transform:
                data = self.pre_transform(data)
            data_list.append(data)

            adj = torch.zeros(data.num_nodes, data.num_nodes, dtype=torch.float)
            adj[data.edge_index[0], data.edge_index[1]] = 1.0
            pkl_data_list.append({
                'adj': adj.numpy(),
                'x': data.x.numpy(),
                'y': data.y.item() if isinstance(data.y, torch.Tensor) else data.y,
                'node_mask': data.node_mask.numpy(),
                'edge_mask': data.edge_mask.numpy(),
            })

        torch.save(self.collate(data_list), self.processed_paths[0])

        raw_pkl_path = os.path.join(self.raw_dir, 'BA-2motif-with-gt.pkl')
        with open(raw_pkl_path, 'wb') as f:
            pkl.dump(pkl_data_list, f)

    def download(self):
        pass

if __name__ == "__main__":
    dataset = BA2MotifWithGroundTruth(root='./data/BA2Motif_GT')
