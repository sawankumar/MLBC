import networkx as nx

G = nx.Graph()
G.add_edges_from([
    ('A', 'B'), ('A', 'K'), ('B', 'K'), ('A', 'C'),
    ('B', 'C'), ('C', 'F'), ('F', 'G'), ('C', 'E'),
    ('E', 'F'), ('E', 'D'), ('E', 'H'), ('I', 'J')
])

# Calculate clustering value for each node in the graph
clustering_values = nx.clustering(G)

# Print the clustering value for each node on new lines
for node, clustering_value in clustering_values.items():
    print(f"Clustering value for node '{node}': {clustering_value}")

# Calculate clustering value for a specific node (e.g., 'C')
clustering_value_C = nx.clustering(G, 'C')

# Print the clustering value for the specified node
print(f"\nClustering value for node 'C': {clustering_value_C}")
