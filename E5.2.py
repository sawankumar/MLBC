import networkx as nx

G = nx.Graph()
G.add_edges_from([('A', 'B'), ('A', 'K'), ('B', 'K'), ('A', 'C'),
 ('B', 'C'), ('C', 'F'), ('F', 'G'), ('C', 'E'),
 ('E', 'F'), ('E', 'D'), ('E', 'H'), ('I', 'J')])

# Visualize the graph
nx.draw_networkx(G, with_labels=True, node_color='green')

# Check if the graph is connected (True or False)
print("Is the graph connected?", nx.is_connected(G))

# Calculate the number of different connected components
print("Number of connected components:", nx.number_connected_components(G))

# List nodes in different connected components
print("Nodes in connected components:")
connected_components = list(nx.connected_components(G))
for component in connected_components:
    print(component)

# List nodes of the component containing the given node ('I')
print("Nodes in the connected component of node 'I':", list(nx.node_connected_component(G, 'I')))

# Calculate the number of nodes to be removed so that the graph becomes disconnected
print("Node connectivity:", nx.node_connectivity(G))

# Calculate the number of edges to be removed so that the graph becomes disconnected
print("Edge connectivity:", nx.edge_connectivity(G))
