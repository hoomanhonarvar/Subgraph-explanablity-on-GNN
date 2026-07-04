import torch
from torch_geometric.data import Data
import networkx as ntx
from torch_geometric.utils import to_networkx,subgraph
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

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


def is_connected_subset(original_graph, node_set):

    if len(node_set) <= 1:
        return True

    G = to_networkx(original_graph, to_undirected=True)

    subgraph = G.subgraph(node_set)

    return ntx.is_connected(subgraph)




def compute_score(model, data, target_class, device, score_func="logit"):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(
                data.num_nodes,
                dtype=torch.long,
                device=device,
            )
        out = model(data.x, data.edge_index, batch)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        if score_func == "prob":
            out = F.softmax(out, dim=-1)
        return out[0, target_class].item()


def induced_subgraph(data, node_indices):

    if not isinstance(node_indices, torch.Tensor):
        node_indices = torch.tensor(
            node_indices,
            dtype=torch.long,
            device=data.x.device,
        )

    edge_index, _ = subgraph(
        node_indices,
        data.edge_index,
        relabel_nodes=True,
    )

    new_data = Data(
        x=data.x[node_indices],
        edge_index=edge_index,
    )

    for key in data.keys():

        if key in ["x", "edge_index", "num_nodes"]:
            continue

        value = data[key]

        if (
            torch.is_tensor(value)
            and value.size(0) == data.num_nodes
        ):
            new_data[key] = value[node_indices]
        else:
            new_data[key] = value

    return new_data

def is_graph_connected(data):
    
    G = to_networkx(data, to_undirected=True)
    return ntx.is_connected(G)


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



def subgraph_explanation(
    model,
    original_data,
    target_class,
    device,
    budget=0.2,
    score_func="logit",
):

    model.eval()

    initial_num_nodes = original_data.num_nodes

    if isinstance(budget, float):
        target_size = max(1, int(initial_num_nodes * budget))
    else:
        target_size = budget

    target_size = min(target_size, initial_num_nodes)



    G = to_networkx(original_data, to_undirected=True)

    if ntx.is_connected(G):
        current_nodes = set(range(initial_num_nodes))

    else:

        components = list(ntx.connected_components(G))

        best_score = -float("inf")
        best_component = None

        for comp in components:

            comp = sorted(comp)

            sub = induced_subgraph(original_data, comp)

            score = compute_score(
                model,
                sub,
                target_class,
                device,
                score_func,
            )

            if score > best_score:
                best_score = score
                best_component = comp

        current_nodes = set(best_component)

    while len(current_nodes) > target_size:

        current_subgraph = induced_subgraph(
            original_data,
            sorted(current_nodes),
        )

        current_score = compute_score(
            model,
            current_subgraph,
            target_class,
            device,
            score_func,
        )

        best_node = None
        smallest_delta = float("inf")

        for node in list(current_nodes):

            candidate_nodes = current_nodes - {node}

            if not is_connected_subset(
                original_data,
                candidate_nodes,
            ):
                continue

            candidate_subgraph = induced_subgraph(
                original_data,
                sorted(candidate_nodes),
            )

            candidate_score = compute_score(
                model,
                candidate_subgraph,
                target_class,
                device,
                score_func,
            )

            delta = current_score - candidate_score

            if delta < smallest_delta:
                smallest_delta = delta
                best_node = node

        if best_node is None:
            break

        current_nodes.remove(best_node)

    final_nodes = sorted(current_nodes)

    explanation = induced_subgraph(
        original_data,
        final_nodes,
    )

    original_score = compute_score(
        model,
        original_data,
        target_class,
        device,
        score_func,
    )

    explanation_score = compute_score(
        model,
        explanation,
        target_class,
        device,
        score_func,
    )

    fidelity = original_score - explanation_score

    return explanation, final_nodes, fidelity

def baseline_explanation(
    model,
    data,
    target_class,
    budget,
    device,
    score_func="logit",
):

    saliency = compute_node_saliency(
        model,
        data,
        target_class,
        device,
    )

    num_nodes = data.num_nodes

    if isinstance(budget, float):
        k = max(1, int(num_nodes * budget))
    else:
        k = min(budget, num_nodes)

    _, top_idx = torch.topk(saliency, k)

    selected_nodes = top_idx.tolist()

    G = to_networkx(data.cpu(), to_undirected=True)

    subgraph = G.subgraph(selected_nodes)

    if not ntx.is_connected(subgraph):

        largest_component = max(
            ntx.connected_components(subgraph),
            key=len,
        )

        selected_nodes = list(largest_component)

    explanation = induced_subgraph(
        data.cpu(),
        selected_nodes,
    )

    original_score = compute_score(
        model,
        data,
        target_class,
        device,
        score_func,
    )

    explanation_score = compute_score(
        model,
        explanation,
        target_class,
        device,
        score_func,
    )

    fidelity = original_score - explanation_score

    return explanation, selected_nodes, fidelity