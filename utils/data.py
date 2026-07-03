import torch
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import pickle
import os

def load_data(dataset_root="./../data/BA2Motif"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(base_dir, dataset_root) if not os.path.isabs(dataset_root) else dataset_root

    return BA2MotifDataset(root=dataset_root)




def dataset_description(dataset,sample_index):
    print(f"lenght of dataset : {len(dataset)}")
    print(f"number of class : {dataset.num_classes}")
    print(f"number of features : {dataset.num_features}")


    data=dataset[sample_index]
    print(" construction of first graph :")
    print(f" edge matrix : {data.edge_index.shape}")
    print(f" nodes features : {data.x.shape}")
    print(f" label : {data.y.item()}")


    data = dataset[sample_index]
    print(f"\n{'='*50}")
    print(f"Sample {sample_index}:")
    print(f"{'='*50}")
    
    print(f"1. Node features (x): shape = {data.x.shape}")
    print(f"   First 2 rows (if enough nodes):\n{data.x[:2]}")  
    
    print(f"2. Edge index (edge_index): shape = {data.edge_index.shape}")
    print(f"   Number of edges: {data.edge_index.size(1)}")
    print(f"   First 5 edges (source -> target):\n{data.edge_index[:, :5].t()}")  
    
    print(f"3. Graph label (y): {data.y.item()} -> {'Positive (House motif)' if data.y.item() == 1 else 'Negative (Cycle motif)'}")
    
    if hasattr(data, 'motif_mask'):
        motif_mask = data.motif_mask
        print(f"4. Motif mask (motif_mask): shape = {motif_mask.shape}")
        num_motif_nodes = motif_mask.sum().item()
        print(f"   Number of nodes in motif: {num_motif_nodes}")
        print(f"   Indices of motif nodes: {motif_mask.nonzero(as_tuple=True)[0].tolist()}")
    elif hasattr(data, 'ground_truth_mask'):
        print(f"4. Ground truth mask: shape = {data.ground_truth_mask.shape}")
    else:
        print("4. No motif mask found in this dataset sample.")

    exclude_attrs = ['stores','node_stores','edge_stores', '__dict__', '__module__', '__weakref__']

    print("\nAll attributes of this data object:")
    for attr in dir(data):
        if attr.startswith('_') or attr in exclude_attrs:
            continue
        val = getattr(data, attr)
        if not callable(val):
            if isinstance(val, torch.Tensor):
                print(f"   - {attr}: tensor of shape {val.shape}")
            else:
                print(f"   - {attr}: {val}")




def train_test_split(dataset,saving_root="./data/splits.pkl",train_ratio=0.8,test_ratio=0.1,gt=False):
    if gt:
        saving_root="./data/splits_with_gt.pkl"
    total_len=len(dataset)
    train_len=int(train_ratio*total_len)
    test_len=int(test_ratio*total_len)
    val_len=total_len-train_len-test_len
    print(train_len,test_len,val_len)



    train_dataset,val_dataset,test_dataset=random_split(
        dataset,[train_len,val_len,test_len]
        ,generator=torch.Generator().manual_seed(404131029))



    train_indices=train_dataset.indices
    val_indices=val_dataset.indices
    test_indices=test_dataset.indices

    splits={
        'train':train_indices,
        'val':val_indices,
        'test':test_indices
    }

    with open(saving_root,'wb') as f:
        pickle.dump(splits,f)

    print(f"train,test,validation indices saved in {saving_root}")


if __name__=="__main__":
    train_test_split(load_data(with_gt=True))
