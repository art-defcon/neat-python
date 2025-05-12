import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch
from collections import deque
import networkx as nx
import numpy as np
import random
from PyQt5.QtGui import QFont, QFontDatabase, QPainter, QPixmap
from PyQt5.QtCore import Qt

class NEATVisualization:
    def __init__(self, main_layout, config):
        self.main_layout = main_layout
        self.config = config
        self.viz_widget = None
        self.viz_layout = None
        self.fig = None
        self.ax_grid = None
        self.ax_input = None
        self.ax_network = None
        self.ax_output = None
        self.ax_pred = None
        self.canvas = None
        self.current_letter = None # This should ideally come from the main app logic
        self.inter_ax_patches = [] # To store ConnectionPatch objects

    def _calculate_node_levels(self, genome, config_input_keys, config_output_keys):
        """Calculates the 'level' of each node based on its shortest path from an input node."""
        node_levels = {}
        all_genome_nodes = set(genome.nodes.keys())

        for node_id in all_genome_nodes:
            node_levels[node_id] = float('inf')

        input_nodes = sorted([n_id for n_id in all_genome_nodes if n_id in config_input_keys])
        for node_id in input_nodes:
            node_levels[node_id] = 0

        # Build adjacency list for BFS (only forward, enabled connections)
        adj = {node_id: [] for node_id in all_genome_nodes}
        if genome.connections: # Only build adj list if there are connections
            for conn_key, conn in genome.connections.items():
                if getattr(conn, 'enabled', True):
                    u, v = conn.key
                    # Ensure nodes in connection are part of the genome's node set
                    if u in all_genome_nodes and v in all_genome_nodes:
                        adj[u].append(v)
        
        queue = deque()
        for node_id in input_nodes:
            queue.append((node_id, 0))

        max_level = 0
        # BFS to calculate levels
        visited_during_bfs = set(input_nodes) # Keep track of nodes added to queue to avoid cycles / redundant processing

        while queue:
            u, level_u = queue.popleft()
            max_level = max(max_level, level_u)

            # Sort neighbors for deterministic layout, though BFS itself handles shortest path
            sorted_neighbors = sorted(adj.get(u, []))

            for v in sorted_neighbors:
                # If we find a path to v, and it's shorter or the first time, update its level
                if node_levels[v] > level_u + 1: # Check if new path is shorter
                    node_levels[v] = level_u + 1
                    if v not in visited_during_bfs:
                         queue.append((v, level_u + 1))
                         visited_during_bfs.add(v)
                elif node_levels[v] == float('inf'): # First time reaching this node
                    node_levels[v] = level_u + 1
                    if v not in visited_during_bfs:
                        queue.append((v, level_u + 1))
                        visited_during_bfs.add(v)


        # Handle unreachable nodes or nodes in a genome with no connections
        # Output nodes not reached by BFS: place them after the max_level found.
        # Hidden nodes not reached: place them similarly.
        
        # Determine max_level from reachable nodes first
        current_max_reachable_level = 0
        has_reachable_nodes = False
        for node_id in all_genome_nodes:
            if node_levels[node_id] != float('inf'):
                current_max_reachable_level = max(current_max_reachable_level, node_levels[node_id])
                has_reachable_nodes = True
        
        if not has_reachable_nodes and not input_nodes: # Completely empty or isolated graph with no inputs
             max_level = 0 # Default max_level if no nodes were processed (e.g. no inputs)
        else:
             max_level = current_max_reachable_level


        # Assign levels to unreachable output/hidden nodes
        # These will be placed one level beyond the max reachable level.
        output_node_ids = sorted([n_id for n_id in all_genome_nodes if n_id in config_output_keys])
        hidden_node_ids = sorted([n_id for n_id in all_genome_nodes if n_id not in config_input_keys and n_id not in config_output_keys])

        unreachable_level_offset = 1

        for node_id in output_node_ids:
            if node_levels[node_id] == float('inf'):
                # If no connections at all, and no inputs, place outputs at level 1
                if not genome.connections and not input_nodes:
                    node_levels[node_id] = 1
                    max_level = max(max_level, 1)
                else: # Otherwise, place after all reachable nodes
                    node_levels[node_id] = max_level + unreachable_level_offset
                    max_level = max(max_level, node_levels[node_id])


        for node_id in hidden_node_ids:
            if node_levels[node_id] == float('inf'):
                if not genome.connections and not input_nodes: # Should be rare for hidden nodes
                    node_levels[node_id] = 1 
                    max_level = max(max_level, 1)
                else:
                    node_levels[node_id] = max_level + unreachable_level_offset # Could be same level as unreachable outputs or further
                    max_level = max(max_level, node_levels[node_id])
        
        # Final fallback for any node still at infinity (should not happen with above logic)
        for node_id in all_genome_nodes:
            if node_levels[node_id] == float('inf'):
                # print(f"[VIS-WARN] Node {node_id} remained at level inf. Defaulting to 0.")
                node_levels[node_id] = 0 # Default to level 0 as a last resort
                max_level = max(max_level, 0)

        return node_levels, max_level

    def create_visualization(self):
        """Center pane - Network Visualization"""
        from PyQt5.QtWidgets import QWidget, QVBoxLayout # Import here to avoid circular dependency if NEATVisualization is in a separate file

        self.viz_widget = QWidget()
        self.viz_widget.setStyleSheet("background-color: #1e1e1e;")
        self.viz_layout = QVBoxLayout(self.viz_widget)

        # Create matplotlib figure with multiple subplots
        self.fig = Figure(figsize=(10, 6), facecolor='#1e1e1e') # Adjusted figsize
        # Adjusted grid_spec to remove ax_output and expand ax_network
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1, 4, 1])

        # Create axes
        self.ax_input = self.fig.add_subplot(gs[0])   # Input neurons
        self.ax_network = self.fig.add_subplot(gs[1]) # Network topology
        self.ax_pred = self.fig.add_subplot(gs[2])    # Prediction text

        # Set aspect for square plots
        self.ax_input.set_aspect('equal', adjustable='box')

        # Configure axes
        for ax_obj in [self.ax_input, self.ax_network, self.ax_pred]: # Removed ax_output
            ax_obj.set_facecolor('#1e1e1e')
            ax_obj.tick_params(axis='both', which='both', length=0)
            ax_obj.set_xticks([])
            ax_obj.set_yticks([])
            # Style spines for a "box" around subplot
            for spine in ax_obj.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('gray') # Light gray for subtle box
                spine.set_linewidth(0.5)


        # Adjust subplot spacing (margins)
        # Assuming 100 DPI, 12-inch width (1200px). 20px is 0.2 inches.
        # Left/Right margin: 0.2in / 12in = 0.0167. Let's use 0.02 for a bit more.
        # Wspace: 0.2in. Avg subplot width approx (12 - 2*0.2 - 4*0.2)/5 = 2.16in. wspace_frac = 0.2/2.16 = 0.092. Let's use 0.1.
        self.fig.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.90, wspace=0.15, hspace=0.15)


        self.canvas = FigureCanvasQTAgg(self.fig)
        self.viz_layout.addWidget(self.canvas)
        plt.close(self.fig) # Close the figure to prevent it from opening in a separate window

        self.main_layout.addWidget(self.viz_widget, stretch=1)

    def draw_mock_network_wrapper(self):
        # This wrapper will call draw_network with mock data
        # The actual mock data generation will be in _get_mock_visualization_data
        # and draw_network will be modified to use it.
        # For now, let's assume draw_network can handle a None or a specific mock genome.
        # We will create _get_mock_visualization_data and adapt draw_network later.
        # This method is called when mock_data_enabled is True.
        mock_data_tuple = self._get_mock_visualization_data()
        self.draw_network(
            genome=mock_data_tuple[0],
            is_mock=True,
            mock_letter_pattern=mock_data_tuple[1],
            mock_output_activations=mock_data_tuple[2],
            mock_prediction=mock_data_tuple[3]
        )

    def _get_mock_visualization_data(self):
        """Generates a consistent set of mock data for visualization."""
        # 1. Mock Rasterized Letter (e.g., an 'A')
        # Updated to 16x16
        mock_letter_pattern = np.zeros((16, 16), dtype=int)
        # Simple 'A' like pattern for 16x16
        mock_letter_pattern[2:14, 7:9] = 1 # Vertical bar
        mock_letter_pattern[2, 4:12] = 1   # Top bar
        mock_letter_pattern[7, 4:12] = 1   # Middle bar
        mock_letter_pattern[2:8, 4] = 1    # Left leg part 1
        mock_letter_pattern[2:8, 11] = 1   # Right leg part 1

        # 2. Mock Network Genome
        # Create a new genome config if not available (should be from self.config)
        genome_config = self.config.genome_config
        mock_genome = self.config.genome_type(0) # Key 0
        mock_genome.configure_new(genome_config)

        # Node IDs from config. `mock_genome.configure_new(genome_config)` already populates
        # mock_genome.nodes with input and output nodes using neat-python conventions.
        # Input keys: e.g., [-1, -2, ..., -num_inputs]
        # Output keys: e.g., [0, 1, ..., num_outputs-1]

        # Add mock hidden nodes. Their IDs start after the highest output node ID.
        hidden_node_ids_mock = []
        next_hidden_id = genome_config.num_outputs # Assumes output keys are 0 to num_outputs-1
        for _ in range(3): # Create 3 mock hidden nodes
            node_id = next_hidden_id
            hidden_node_ids_mock.append(node_id)
            # Add new hidden node gene if it doesn't exist
            if node_id not in mock_genome.nodes:
                mock_genome.nodes[node_id] = genome_config.node_gene_type(node_id)
            # Set attributes for the hidden node
            mock_genome.nodes[node_id].bias = random.uniform(-0.5, 0.5)
            mock_genome.nodes[node_id].response = 1.0
            mock_genome.nodes[node_id].activation = 'tanh'
            mock_genome.nodes[node_id].aggregation = 'sum'
            next_hidden_id += 1

        # Ensure output nodes (created by configure_new) have specific mock attributes
        for node_id in genome_config.output_keys:
            if node_id in mock_genome.nodes: # Should exist
                mock_genome.nodes[node_id].bias = 0.0 # Explicitly set for mock
                mock_genome.nodes[node_id].response = 1.0
                mock_genome.nodes[node_id].activation = 'tanh'
                mock_genome.nodes[node_id].aggregation = 'sum'

        # Define connections using neat-python node ID conventions
        input_keys = list(genome_config.input_keys) # e.g., [-1, -2,...]
        output_keys = list(genome_config.output_keys) # e.g., [0, 1, 2]
        # hidden_node_ids_mock was defined in the node creation block above.

        # Clear existing connections for a fresh mock setup
        mock_genome.connections.clear()

        # Connection 1: First input to first hidden
        if input_keys and hidden_node_ids_mock:
            conn_key1 = (input_keys[0], hidden_node_ids_mock[0])
            cg1 = genome_config.connection_gene_type(conn_key1)
            cg1.weight = 0.5
            cg1.enabled = True
            mock_genome.connections[cg1.key] = cg1

        # Connection 2: An input (e.g., 10th if exists, using index 9) to second hidden
        if len(input_keys) > 9 and len(hidden_node_ids_mock) > 1: # input_keys[9] is the 10th input
            conn_key2 = (input_keys[9], hidden_node_ids_mock[1])
            cg2 = genome_config.connection_gene_type(conn_key2)
            cg2.weight = -0.3
            cg2.enabled = True
            mock_genome.connections[cg2.key] = cg2

        # Connection 3: First hidden to first output
        if hidden_node_ids_mock and output_keys:
            conn_key3 = (hidden_node_ids_mock[0], output_keys[0])
            cg3 = genome_config.connection_gene_type(conn_key3)
            cg3.weight = 0.8
            cg3.enabled = True
            mock_genome.connections[cg3.key] = cg3

        # Connection 4: Second hidden to second output
        if len(hidden_node_ids_mock) > 1 and len(output_keys) > 1:
            conn_key4 = (hidden_node_ids_mock[1], output_keys[1])
            cg4 = genome_config.connection_gene_type(conn_key4)
            cg4.weight = 0.2
            cg4.enabled = True
            mock_genome.connections[cg4.key] = cg4
        
        # Connection 5: Third hidden to third output (this hidden node has no direct input from an input node in this mock)
        if len(hidden_node_ids_mock) > 2 and len(output_keys) > 2:
            conn_key5 = (hidden_node_ids_mock[2], output_keys[2])
            cg5 = genome_config.connection_gene_type(conn_key5)
            cg5.weight = 0.4
            cg5.enabled = True
            mock_genome.connections[cg5.key] = cg5

        # Connection 6: Example of a disabled connection (e.g., 5th input to first output)
        if len(input_keys) > 4 and output_keys: # input_keys[4] is the 5th input
            conn_key_disabled = (input_keys[4], output_keys[0])
            if conn_key_disabled not in mock_genome.connections: # Avoid overwriting for this example
                cg_disabled = genome_config.connection_gene_type(conn_key_disabled)
                cg_disabled.weight = -0.7
                cg_disabled.enabled = False # This connection is disabled
                mock_genome.connections[cg_disabled.key] = cg_disabled


        # 3. Mock Output Activations
        mock_output_activations = [0.75, 0.20, 0.05]

        # 4. Mock Predicted Letter
        mock_prediction = "A"

        return mock_genome, mock_letter_pattern, mock_output_activations, mock_prediction

    def draw_network(self, genome, is_mock=False,
                       mock_letter_pattern=None, mock_output_activations=None, mock_prediction=None,
                       actual_letter_pattern=None, actual_output_activations=None, actual_prediction=None,
                       actual_letter=None, # Added actual_letter parameter
                       is_correct=None):
        """Draw complete visualization including raster grid, network, and prediction"""
        # Clear previous inter-axes connection patches
        if hasattr(self, 'inter_ax_patches'):
            for patch in self.inter_ax_patches:
                if patch in self.fig.artists:
                    patch.remove()
        self.inter_ax_patches = []

        # Clear all axes
        for ax in [self.ax_input, self.ax_network, self.ax_pred]: # Removed ax_output
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure square aspect ratio after clearing
        self.ax_input.set_aspect('equal', adjustable='box')

        # Style spines for each subplot on redraw as clear() might reset them
        for ax_obj in [self.ax_input, self.ax_network, self.ax_pred]: # Removed ax_output
            for spine in ax_obj.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('gray')
                spine.set_linewidth(0.5)

        current_letter_pattern = None
        current_output_activations = None
        current_prediction = None

        if is_mock:
            current_letter_pattern = mock_letter_pattern
            current_output_activations = mock_output_activations
            current_prediction = mock_prediction
            # genome is already the mock_genome passed in
        else:
            # This is for the real NEAT data path
            # Use the directly passed actual_... parameters
            current_letter_pattern = actual_letter_pattern
            current_output_activations = actual_output_activations
            current_prediction = actual_prediction
            # is_correct is passed directly and will be used for coloring the prediction text

        # Ensure current_letter_pattern is not None before imshow
        if current_letter_pattern is None:
             current_letter_pattern = np.zeros((16,16)) # Default to blank if None

        # 1. Draw raster grid (16x16) - REMOVED
        # self.ax_grid.imshow(current_letter_pattern, cmap='gray', vmin=0, vmax=1)

        # 2. Draw input neurons (256 nodes)
        G_input = nx.DiGraph()
        # Ensure input nodes are added based on the actual config
        num_inputs_cfg = self.config.genome_config.num_inputs
        G_input.add_nodes_from(range(num_inputs_cfg))

        # Calculate positions for input neurons in an 8x6 grid
        input_pos = {}
        rows, cols = 8, 6 # Updated to 8x6 grid
        # Adjust spacing and centering for the grid
        x_spacing = 1.0 / (cols - 1) if cols > 1 else 0
        y_spacing = 1.0 / (rows - 1) if rows > 1 else 0

        for i in range(num_inputs_cfg):
            # Assuming row-major order for flattened image data
            row = i // cols
            col = i % cols
            # Map grid coordinates to plot coordinates (adjusting for origin and scaling)
            # Reverse vertical order, keep horizontal order direct
            input_pos[i] = (col * x_spacing, (rows - 1 - row) * y_spacing)

        # Determine node colors based on activation (pixel value)
        input_node_colors = []
        if current_letter_pattern is not None:
            flattened_pattern = current_letter_pattern.flatten()
            for i in range(num_inputs_cfg):
                if i < len(flattened_pattern):
                    # Color based on pixel value (0 or 1)
                    color = '#4e79a7' if flattened_pattern[i] == 0 else '#f28e2b' # Blue for 0, Orange for 1
                    input_node_colors.append(color)
                else:
                    input_node_colors.append('#4e79a7') # Default color if pattern is smaller

        nx.draw_networkx_nodes(G_input, input_pos, ax=self.ax_input,
                                 nodelist=list(range(num_inputs_cfg)), # Explicitly pass nodelist
                                 node_color=input_node_colors, node_size=10, # Reduced node size for 16x16 grid
                                 edgecolors='dimgray', linewidths=0.5) 
        self.ax_input.set_title('Input Neurons', color='white', fontsize=10, bbox=dict(facecolor='none', edgecolor='dimgray', boxstyle='round,pad=0.3', lw=0.5))

        # Set limits to encompass the grid
        self.ax_input.set_xlim(-0.1, 1.1)
        self.ax_input.set_ylim(-0.1, 1.1)
        # Removed self.ax_input.invert_yaxis()

        # 3. Draw main network topology
        G = nx.DiGraph()

        # Force initial topology if empty (This logic might need to be in the main app or passed in)
        # For now, assuming genome has nodes and connections
        if not genome.connections and not genome.nodes:
             # Handle case where genome is truly empty, maybe draw a default structure or nothing
             pass # Or add some default nodes/connections for visualization purposes

        # Add nodes and define layout/styling
        pos = {}
        node_colors_list = [] # Renamed to avoid conflict
        node_sizes_list = []  # Renamed to avoid conflict

        num_inputs = self.config.genome_config.num_inputs

        color_map = {
            'input': '#4e79a7', 'hidden': '#f28e2b',
            'output': '#e15759', 'bias': '#76b7b2'
        }

        # Use config keys for proper node identification based on neat-python conventions
        config_input_keys = set(self.config.genome_config.input_keys)
        config_output_keys = set(self.config.genome_config.output_keys)
        
        print(f"[VIS] draw_network called. Genome key: {genome.key if genome else 'None'}")
        if genome:
            print(f"[VIS] Genome has {len(genome.nodes)} nodes and {len(genome.connections)} connections.")
        else:
            print("[VIS] Genome is None.")
            # Handle drawing for None genome if necessary, or return
            self.canvas.draw()
            plt.pause(0.01)
            return

        current_genome_node_ids = set(genome.nodes.keys())

        input_node_ids = sorted([n_id for n_id in current_genome_node_ids if n_id in config_input_keys])
        output_node_ids = sorted([n_id for n_id in current_genome_node_ids if n_id in config_output_keys])
        hidden_node_ids = sorted([n_id for n_id in current_genome_node_ids if n_id not in config_input_keys and n_id not in config_output_keys])
        
        print(f"[VIS] Categorized nodes: Inputs({len(input_node_ids)}): {input_node_ids[:5]}..., Hidden({len(hidden_node_ids)}): {hidden_node_ids[:5]}..., Outputs({len(output_node_ids)}): {output_node_ids}")

        # Calculate node levels for layered layout
        node_levels, max_level = self._calculate_node_levels(genome, config_input_keys, config_output_keys)
        # print(f"[VIS] Node levels calculated: {node_levels}, Max level: {max_level}")

        # Define get_y_pos locally
        def get_y_pos_local(nodes_in_list, current_idx_in_list):
            if not nodes_in_list: return 0
            num_nodes_in_list = len(nodes_in_list)
            if num_nodes_in_list == 0: return 0
            max_spread = 10.0  # Max vertical spread for a layer
            node_spacing_val = max_spread / float(num_nodes_in_list) if num_nodes_in_list > 1 else 0
            return -(current_idx_in_list * node_spacing_val - (num_nodes_in_list - 1) * node_spacing_val / 2.0)

        all_nodes_for_graph = []
        pos = {} # Reset pos
        node_colors_list = [] # Reset
        node_sizes_list = [] # Reset

        input_x_coord = 0.0
        x_increment_per_level = 1.5 # Horizontal spacing between layers

        nodes_by_level = {}
        for node_id_val, level_val in node_levels.items():
            if level_val != float('inf') and node_id_val in current_genome_node_ids:
                 nodes_by_level.setdefault(level_val, []).append(node_id_val)
        
        # print(f"[VIS] Nodes grouped by level: {nodes_by_level}")


        for level_idx in sorted(nodes_by_level.keys()): # Iterate through levels in order
            nodes_in_this_level = sorted(nodes_by_level[level_idx]) # Sort nodes within a level by ID
            
            current_x = input_x_coord + level_idx * x_increment_per_level
            if not nodes_in_this_level: continue # Skip if a level somehow has no nodes (shouldn't happen with setdefault)

            for i, node_id in enumerate(nodes_in_this_level):
                y_pos_val = get_y_pos_local(nodes_in_this_level, i)
                pos[node_id] = (current_x, y_pos_val)
                
                if node_id not in all_nodes_for_graph: # Ensure node is added only once
                    all_nodes_for_graph.append(node_id)
                    # Determine color and size
                    if node_id in config_input_keys:
                        node_colors_list.append(color_map['input'])
                        node_sizes_list.append(30)
                    elif node_id in config_output_keys:
                        node_colors_list.append(color_map['output'])
                        node_sizes_list.append(35)
                    else:  # Hidden node
                        bias_val = 0.0
                        if node_id in genome.nodes and hasattr(genome.nodes[node_id], 'bias'):
                            bias_val = genome.nodes[node_id].bias if genome.nodes[node_id].bias is not None else 0.0
                        node_colors_list.append(color_map['hidden'])
                        node_sizes_list.append(40 + 20 * abs(bias_val))
        
        # Ensure all_nodes_for_graph matches node_colors_list and node_sizes_list
        # This might require rebuilding these lists based on the final all_nodes_for_graph order if nodes were added out of sync.
        # The current logic appends to all three lists together, so they should be synced.

        G.add_nodes_from(all_nodes_for_graph) # Add all nodes that have positions
        # print(f"[VIS] Nodes added to G for ax_network: {len(all_nodes_for_graph)}. Sample: {all_nodes_for_graph[:5]}")
        # print(f"[VIS] Positions calculated (pos dict size {len(pos)}).")


        # Add connections
        edge_widths = []
        edge_colors_list_for_edges = []
        edges_to_draw = []

        for conn_key, conn in genome.connections.items():
            # Correctly access input and output node IDs from conn.key
            in_node_id = conn.key[0]
            out_node_id = conn.key[1]
            # Use getattr for 'enabled' as a safeguard
            if getattr(conn, 'enabled', True) and in_node_id in G and out_node_id in G:
                edges_to_draw.append(conn_key) # conn_key is (in_node_id, out_node_id)
                edge_widths.append(max(0.3, abs(conn.weight) * 1.5))
                edge_colors_list_for_edges.append('#59a14f' if conn.weight > 0 else '#e15759')

        nx.draw_networkx_nodes(G, pos, ax=self.ax_network, nodelist=all_nodes_for_graph,
                               node_color=node_colors_list, node_size=node_sizes_list,
                               edgecolors='black', linewidths=1.0)
        if edges_to_draw: # Only draw edges if there are any
            nx.draw_networkx_edges(G, pos, ax=self.ax_network, edgelist=edges_to_draw,
                                   width=edge_widths, edge_color=edge_colors_list_for_edges,
                                   arrowsize=7, arrowstyle='->', alpha=0.6)
        self.ax_network.set_title('Network Topology', color='white', fontsize=10, bbox=dict(facecolor='none', edgecolor='dimgray', boxstyle='round,pad=0.3', lw=0.5))
        self.ax_network.autoscale_view()
        self.ax_network.set_xticks([]) # Ensure ticks are off
        self.ax_network.set_yticks([])
        # Set margins for the network plot to give some space
        self.ax_network.margins(0.1)

        # --- Add ConnectionPatches between subplots for individual nodes ---

        # 1. Connect each visual input neuron in ax_input to its corresponding genome input node in ax_network
        # input_pos is for ax_input, pos is for ax_network
        # self.config.genome_config.input_keys are sorted, e.g., [-1, -2, ..., -36]
        # We assume the 0th visual input neuron (input_pos[0]) corresponds to input_keys[0]
        genome_input_keys = sorted(list(self.config.genome_config.input_keys)) # Ensure sorted
        if input_node_ids and len(input_pos) == len(genome_input_keys):
            for i in range(len(genome_input_keys)):
                ax_input_coord = input_pos[i] # Coordinate of the i-th visual input node in ax_input
                network_input_node_id = genome_input_keys[i]
                if network_input_node_id in pos: # Ensure the node exists in the network drawing
                    ax_network_coord = pos[network_input_node_id]

                    con = ConnectionPatch(xyA=ax_input_coord, xyB=ax_network_coord,
                                          coordsA="data", coordsB="data",
                                          axesA=self.ax_input, axesB=self.ax_network,
                                          color="gray", linestyle=":", alpha=0.4, linewidth=0.5, zorder=-1)
                    self.fig.add_artist(con)
                    self.inter_ax_patches.append(con)

        # 4. Draw prediction text
        # Use current_prediction which is set based on mock or real mode
        if current_prediction is None:
            current_prediction = "?"

        prediction_color = 'white' # Default
        if not is_mock and is_correct is not None: # Only apply color logic for non-mock with correctness info
            prediction_color = 'green' if is_correct else 'red'
        
        # Determine prediction text based on user's desired format
        if not is_mock:
            # Construct the base prediction text
            prediction_text = f"Predicted:\n{current_prediction}"

            # Add the actual letter in parentheses if available
            if actual_letter is not None:
                 prediction_text += f" ({actual_letter})"

            # The color logic based on is_correct remains the same
            # prediction_color is set earlier based on is_correct

        else: # Mock data
            # For mock data, just show the mock prediction
            prediction_text = f"Predicted:\n{current_prediction}"

        # For ax_pred, the title is effectively the text itself. We can draw a box around the text area.
        # However, set_title is not typically used for the main content of ax_pred.
        # The text is drawn directly. To put a box around this text, we'd need to draw a patch.
        # For now, let's skip boxing the prediction text itself, as it's not a "title".
        # If a box is needed around the ax_pred subplot, the spine styling handles that.
        self.ax_pred.set_title('Prediction', color='white', fontsize=10, bbox=dict(facecolor='none', edgecolor='dimgray', boxstyle='round,pad=0.3', lw=0.5))


        self.ax_pred.text(0.5, 0.5, prediction_text,
                         color=prediction_color, ha='center', va='center', fontsize=10) # Prediction text fontsize was already 10

        self.canvas.draw()
        plt.pause(0.01)

    def _pixmap_to_matrix(self, pixmap, actual_letter):
        """Convert QPixmap to 16x16 binary matrix"""
        image = pixmap.toImage()
        matrix = []
        for y in range(16):
            row = []
            for x in range(16):
                pixel = image.pixelColor(x, y)
                row.append(0 if pixel.lightness() > 127 else 1)
            matrix.append(row)
        return np.array(matrix), actual_letter

    def generate_letter_pattern(self, letter):
        """Generate 16x16 pattern from random system font for a given letter"""
        from PyQt5.QtGui import QFont, QFontDatabase, QPainter, QPixmap
        from PyQt5.QtCore import Qt

        # Get available fonts (excluding symbol fonts)
        fonts = [f for f in QFontDatabase().families()
                if not any(x in f.lower() for x in ['symbol','dingbat','emoji'])]
        selected_font = QFont(random.choice(fonts[:20]))  # Use first 20 for consistency

        # Render letter at high resolution
        selected_font.setPixelSize(64)
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setFont(selected_font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, letter)
        painter.end()

        # Scale down to 16x16 and convert to matrix
        return self._pixmap_to_matrix(pixmap.scaled(
            16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation), letter)

    def classify_letter(self, genome, letter_pattern):
        """Classify letter using NEAT network based on a given letter_pattern"""
        if not genome or letter_pattern is None: return "?"

        try:
            import neat # Assuming neat is available
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            input_data = letter_pattern.flatten().astype(float)

            if len(input_data) != self.config.genome_config.num_inputs:
                return "?"

            output_activations = net.activate(input_data)
            predicted_idx = output_activations.index(max(output_activations))
            return ['A', 'B', 'C'][predicted_idx % 3]
        except Exception as e:
            # print(f"Error classifying with genome {genome.key if genome else 'None'}: {e}")
            return "?"
