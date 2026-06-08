import torch
from torch_geometric.data import Data
import networkx as ntx
from torch_geometric.utils import to_networkx,subgraph
import matplotlib.pyplot as plt

def remove_node(data, nodes_to_remove):

    if isinstance(nodes_to_remove, list):
        nodes_to_remove = torch.tensor(nodes_to_remove, dtype=torch.long)
    num_nodes = data.num_nodes
    keep_mask = torch.ones(num_nodes, dtype=torch.bool)
    keep_mask[nodes_to_remove] = False
    new_x = data.x[keep_mask]
    
    new_index = torch.zeros(num_nodes, dtype=torch.long)
    new_index[keep_mask] = torch.arange(keep_mask.sum().item())
    
    edge_index = data.edge_index
    edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    new_edge_index = edge_index[:, edge_mask]
    new_edge_index = new_index[new_edge_index]
    
    new_data = Data(x=new_x, edge_index=new_edge_index)
    
    for key in data.keys():   
        if key in ['x', 'edge_index', 'num_nodes']:
            continue
        attr = data[key]
        if torch.is_tensor(attr) and attr.size(0) == num_nodes:
            new_data[key] = attr[keep_mask]
        else:
            new_data[key] = attr
    return new_data



def subgraph_connection_check(graph,node_set):
    if len(node_set) <= 1:
        return True
    sub_edge_index, _ = subgraph(node_set, graph.edge_index, relabel_nodes=True)
    sub_data = Data(edge_index=sub_edge_index, num_nodes=len(node_set))
    G = to_networkx(sub_data, to_undirected=True)
    return ntx.is_connected(G)


def compute_score(model,data,target_class,device):
    model.eval()
    with torch.no_grad():
        if data.batch is not None:
            out=model(data.x,data.edge_index,data.batch)
        else:
            batch=torch.zeros(data.num_nodes,dtype=torch.long,device=device)
            out = model(data.x, data.edge_index, batch)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        score = out[0, target_class]  
    return score 



def induced_subgraph(data, node_indices):
    
    if not isinstance(node_indices, torch.Tensor):
        node_indices = torch.tensor(node_indices, dtype=torch.long)
    
    device = data.x.device
    node_indices = node_indices.to(device)
    
    edge_index, _ = subgraph(node_indices, data.edge_index, relabel_nodes=True)
    
    x = data.x[node_indices]
    
    sub_data = Data(x=x, edge_index=edge_index)
    
    for key in data.keys():
        if key in ['x', 'edge_index', 'num_nodes']:
            continue
        attr = data[key]
        if torch.is_tensor(attr) and attr.size(0) == data.num_nodes:
            sub_data[key] = attr[node_indices]
        else:
            sub_data[key] = attr
    
    return sub_data


def is_graph_connected(data):
    
    G = to_networkx(data, to_undirected=True)
    return ntx.is_connected(G)


def subgraph_explanation(model, data, target_class,device, budget=0.2, ):
    
    
    initial_num_nodes = data.num_nodes
    target_size = budget if isinstance(budget, int) else int(initial_num_nodes * budget)
    target_size = max(1, min(target_size, initial_num_nodes))
    
    if not is_graph_connected(data):
        G_full = to_networkx(data, to_undirected=True)
        components = [list(comp) for comp in ntx.connected_components(G_full)]
        best_comp = None
        best_score = -float('inf')
        for comp in components:
            sub = induced_subgraph(data, comp)
            score = compute_score(model, sub, target_class, device)
            if score > best_score:
                best_score = score
                best_comp = comp
        removed_nodes = list(set(range(data.num_nodes)) - set(best_comp))
        data = induced_subgraph(data, best_comp)
        current_nodes = set(range(data.num_nodes))
    else:
        current_nodes = set(range(data.num_nodes))
        removed_nodes = []
    


    while len(current_nodes) > target_size:
        best_node = None
        best_delta = None
        best_new_data = None
        
        for v in list(current_nodes):
            candidate_nodes = current_nodes - {v}
            if not subgraph_connection_check(data, list(candidate_nodes)):
                continue
            
            current_subgraph = remove_node(data, list(set(range(data.num_nodes)) - current_nodes))
            candidate_subgraph = remove_node(data, list(set(range(data.num_nodes)) - candidate_nodes))
            
            score_current = compute_score(model, current_subgraph, target_class, device)
            score_candidate = compute_score(model, candidate_subgraph, target_class, device)
            
            delta = score_current - score_candidate
            
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_node = v
                best_new_data = candidate_subgraph
        
        if best_node is None:
            print("there is no node to remove! perhaps graph is not connected at all")
            break
        
        current_nodes.remove(best_node)
        removed_nodes.append(best_node)

    final_subgraph = remove_node(data, list(set(range(data.num_nodes)) - current_nodes))
    fidelity_score = compute_score(model, data, target_class, device) - compute_score(model, final_subgraph, target_class, device)
    
    return final_subgraph, removed_nodes, fidelity_score


def compute_node_saliency(model, data, target_class, device):

    model.eval()
    data = data.to(device)
    data.x.requires_grad_(True)
    
    batch = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    out = model(data.x, data.edge_index, batch)  # [1, num_classes]
    score = out[0, target_class]  
    
    score.backward()
    grad = data.x.grad  # [num_nodes, num_features]
    
    saliency = torch.norm(grad, dim=1)  # [num_nodes]

    data.x.grad = None
    data.x.requires_grad_(False)
    
    return saliency.cpu()


def ensure_connectivity(sub_data, original_data):
    
    G_sub = to_networkx(sub_data, to_undirected=True)
    if ntx.is_connected(G_sub):
        return sub_data, list(G_sub.nodes)
    
    components = list(ntx.connected_components(G_sub))

    largest_comp = max(components, key=len)
    current_nodes = set(largest_comp)
    
    final_nodes = sorted(list(current_nodes))
    connected_data = induced_subgraph(original_data, final_nodes)
    return connected_data, final_nodes


def ensure_connected_with_shortest_paths(original_data, selected_nodes):
    
    G_full = to_networkx(original_data, to_undirected=True)
    
    sub_G = G_full.subgraph(selected_nodes).copy()
    
    if ntx.is_connected(sub_G):
        return selected_nodes, induced_subgraph(original_data, selected_nodes)
    
    components = list(ntx.connected_components(sub_G))
    components.sort(key=len, reverse=True)
    
    final_nodes = set(components[0])
    
    for comp in components[1:]:
        min_path = None
        min_path_len = float('inf')
        for u in comp:
            for v in final_nodes:
                try:
                    path = ntx.shortest_path(G_full, source=u, target=v)
                    if len(path) < min_path_len:
                        min_path_len = len(path)
                        min_path = path
                except ntx.NetworkXNoPath:
                    continue
        if min_path is not None:
            final_nodes.update(min_path) 
        else:
            final_nodes.update(comp)
    
    final_nodes = sorted(list(final_nodes))
    final_subgraph = induced_subgraph(original_data, final_nodes)
    return final_nodes, final_subgraph

def baseline_explanation(model, data, target_class, budget, device):

    saliency = compute_node_saliency(model, data, target_class, device)
    num_nodes = data.num_nodes
    if isinstance(budget, float):
        k = max(1, int(num_nodes * budget))
    else:
        k = min(budget, num_nodes)
    
    _, top_indices = torch.topk(saliency, k)
    selected_nodes = top_indices.tolist()
    
    data_cpu = data.cpu()
    sub_data_initial = induced_subgraph(data_cpu, selected_nodes)
    
    final_nodes, explanation_graph = ensure_connected_with_shortest_paths(
        data_cpu, selected_nodes
    )
    
    original_data_score=compute_score(model,data,target_class,device=device)
    subgraph_score=compute_score(model,explanation_graph,target_class,device=device)

    fidelity_score=original_data_score-subgraph_score
    
    return explanation_graph, final_nodes,round(fidelity_score.item(),4)