import networkx as ntx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt




def visualize_explanation(original_data, explanation_node_indices, title="Explanation Subgraph", save_path=None):

    G = to_networkx(original_data, to_undirected=True)
    pos = ntx.spring_layout(G, seed=404131029)  
    
    node_colors = []
    for node in G.nodes():
        if node in explanation_node_indices:
            node_colors.append('red')
        else:
            node_colors.append('skyblue')
    
    plt.figure(figsize=(10, 8))
    ntx.draw(G, pos, node_color=node_colors, with_labels=True, 
            node_size=500, font_size=10, font_weight='bold', 
            edge_color='gray', alpha=0.8)
    
    plt.title(f"{title}\n(Explanation nodes: {len(explanation_node_indices)} / {original_data.num_nodes})")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
