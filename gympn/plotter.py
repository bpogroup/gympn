import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from torch_geometric.data import HeteroData
matplotlib.use("TkAgg")

class GraphPlotter:
    """
    A singleton class for plotting graphs side by side using Matplotlib and NetworkX.
    Used as a debugging tool when applying graph neural networks to Petri nets.
    """


    _instance = None  # Class-level variable to hold the singleton instance

    def __new__(cls, *args, **kwargs):
        """
        Override __new__ to implement the singleton pattern.
        This ensures that only one instance of GraphPlotter is created.
        """
        if cls._instance is None:
            cls._instance = super(GraphPlotter, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the GraphPlotter instance. This method will only run once due to the singleton pattern.
        """

        if self._initialized:
            return  # Prevent reinitialization
        self._initialized = True

        # Enable interactive mode
        plt.ion()
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))  # Create a single figure with two subplots

    def plot_hetero_data(self, data: HeteroData):
        """
        Convert a HeteroData object to a NetworkX directed graph and return it.

        Parameters
        ----------
        :param data: HeteroData object containing node and edge information.

        Returns
        ----------
        :graph: A NetworkX directed graph representing the HeteroData.
        """
        # Create a directed graph
        graph = nx.DiGraph()

        # Add nodes for each node type
        for node_type, node_data in data.node_items():
            num_nodes = node_data.x.size(0) if 'x' in node_data else 0
            for i in range(num_nodes):
                graph.add_node(f"{node_type}_{i}", type=node_type)

        # Add edges for each edge type
        for edge_type, edge_data in data.edge_items():
            src_type, _, dst_type = edge_type
            if 'edge_index' in edge_data:
                src, dst = edge_data.edge_index
                for s, d in zip(src.tolist(), dst.tolist()):
                    graph.add_edge(f"{src_type}_{s}", f"{dst_type}_{d}", type=edge_type)

        return graph

    def plot_side_by_side(self, expanded_pn, ret_graph):
        """
        Plot two graphs side by side for comparison.
        Parameters
        ----------
        :param expanded_pn: The expanded Petri net object containing arcs.
        :param ret_graph: The HeteroData object representing the ret graph.

        Returns
        ----------
        None
        """

        def grid_layout(graph):
            """Generate a grid layout for the nodes.

            Parameters
            ----------
            :param graph: The NetworkX graph to layout.

            Returns
            ----------
            :return: A dictionary mapping nodes to positions.
            """
            num_nodes = len(graph.nodes)
            cols = int(num_nodes ** 0.5) + 1  # Number of columns in the grid
            positions = {}
            for i, node in enumerate(graph.nodes):
                row, col = divmod(i, cols)
                positions[node] = (col, -row)  # Arrange in a grid
            return positions

        # Clear the axes to prepare for new plots
        for ax in self.axes:
            ax.clear()

        # Plot the expanded_pn graph
        arcs_str = [(a[0]._id, a[1]._id) for a in expanded_pn.arcs]
        graph_expanded = nx.DiGraph()
        graph_expanded.add_edges_from(arcs_str)
        pos_expanded = grid_layout(graph_expanded)
        edge_colors_expanded = list(mcolors.TABLEAU_COLORS.values())[:len(graph_expanded.edges)]
        nx.draw(
            graph_expanded, pos_expanded, ax=self.axes[0], with_labels=True, node_color='lightblue',
            edge_color=edge_colors_expanded, edge_cmap=plt.cm.rainbow, node_size=2000, font_size=10, font_weight='bold'
        )
        self.axes[0].set_title("Expanded PN")

        # Plot the ret_graph
        graph_ret = self.plot_hetero_data(ret_graph)
        pos_ret = grid_layout(graph_ret)
        edge_colors_ret = list(mcolors.TABLEAU_COLORS.values())[:len(graph_ret.edges)]
        nx.draw(
            graph_ret, pos_ret, ax=self.axes[1], with_labels=True, node_color='lightgreen',
            edge_color=edge_colors_ret, edge_cmap=plt.cm.rainbow, node_size=2000, font_size=10, font_weight='bold'
        )
        self.axes[1].set_title("Ret Graph")

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()