import matplotlib
import networkx as nx
import matplotlib.pyplot as plt

# Set the Matplotlib backend
matplotlib.use("TkAgg")

# Example arcs list (replace with expanded_pn.arcs)
arcs = [
    ("arrival.0", "arrive"),
    ("arrival.1", "arrive"),
    ("arrive", "arrival.0"),
    ("arrive", "arrival.1"),
    ("arrive", "waiting1.0"),
    ("arrive", "waiting1.1"),
    ("busy1.0", "done1"),
    ("done1", "employee1.0"),
    ("done1", "employee1.1"),
    ("done1", "employee1.2"),
    ("done1", "waiting2.0"),
    ("busy2.0", "done2"),
    ("done2", "employee2.0"),
    ("done2", "employee2.1"),
    ("done2", "employee2.2"),
    ("waiting1.0", "start1.0"),
    ("employee1.0", "start1.0"),
    ("start1.0", "busy1.0"),
    ("waiting1.0", "start1.1"),
    ("employee1.1", "start1.1"),
    ("start1.1", "busy1.0"),
    ("waiting1.0", "start1.2"),
    ("employee1.2", "start1.2"),
    ("start1.2", "busy1.0"),
    ("waiting1.1", "start1.3"),
    ("employee1.0", "start1.3"),
    ("start1.3", "busy1.0"),
    ("waiting1.1", "start1.4"),
    ("employee1.1", "start1.4"),
    ("start1.4", "busy1.0"),
    ("waiting1.1", "start1.5"),
    ("employee1.2", "start1.5"),
    ("start1.5", "busy1.0"),
    ("waiting2.0", "start2.0"),
    ("employee2.0", "start2.0"),
    ("start2.0", "busy2.0"),
    ("waiting2.0", "start2.1"),
    ("employee2.1", "start2.1"),
    ("start2.1", "busy2.0"),
    ("waiting2.0", "start2.2"),
    ("employee2.2", "start2.2"),
    ("start2.2", "busy2.0"),
]

# Create a directed graph
graph = nx.DiGraph()

# Add edges to the graph
graph.add_edges_from(arcs)

# Calculate positions using spring layout
pos = nx.spring_layout(graph, k=0.5, seed=42)  # Adjust k to control spacing

# Draw the graph
plt.figure(figsize=(12, 8))
nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, font_weight='bold')
plt.title("Graph Visualization with Improved Layout")
plt.show()