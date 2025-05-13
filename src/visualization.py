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

    def get_activation_color(self, value, min_val=0.0, max_val=1.0, color_low=(0.2, 0.2, 0.8), color_mid=(0.9, 0.9, 0.9), color_high=(0.8, 0.2, 0.2)):
        """Calculates a color based on an activation value within a range."""
        # Ensure value is clamped
        value = max(min_val, min(value, max_val))
        if abs(max_val - min_val) < 1e-6: # Avoid division by zero if range is effectively zero
            return color_mid if value >= min_val else color_low

        # Normalize value to 0-1 range
        norm_value = (value - min_val) / (max_val - min_val)

        if norm_value < 0.5:
            # Interpolate between low and mid
            interp = norm_value * 2.0 # Scale to 0-1 for this half
            r = color_low[0] * (1 - interp) + color_mid[0] * interp
            g = color_low[1] * (1 - interp) + color_mid[1] * interp
            b = color_low[2] * (1 - interp) + color_mid[2] * interp
        else:
            # Interpolate between mid and high
            interp = (norm_value - 0.5) * 2.0 # Scale to 0-1 for this half
            r = color_mid[0] * (1 - interp) + color_high[0] * interp
            g = color_mid[1] * (1 - interp) + color_high[1] * interp
            b = color_mid[2] * (1 - interp) + color_high[2] * interp
        return (r, g, b)

    def _calculate_node_levels(self, genome, config_input_keys, config_output_keys):
        """
        Calculates the 'level' of each node based on the new layout strategy:
        - Inputs on layer 0.
        - Connected hidden nodes layered by distance from inputs.
        - Disconnected hidden nodes on their own layer before outputs.
        - Outputs on the rightmost layer.
        """
        node_levels = {}  # node_id -> level
        all_genome_nodes = set(genome.nodes.keys())
        
        # Initialize levels to None (or a placeholder indicating not yet assigned)
        for node_id in all_genome_nodes:
            node_levels[node_id] = None

        # 1. Assign Input Layer (Level 0)
        input_node_ids = sorted([n_id for n_id in all_genome_nodes if n_id in config_input_keys])
        for node_id in input_node_ids:
            node_levels[node_id] = 0
        
        max_level_so_far = 0
        if input_node_ids: # Only update if there are inputs
            max_level_so_far = 0

        # Build adjacency list for BFS (only enabled connections)
        adj = {node_id: [] for node_id in all_genome_nodes}
        # Also, count degrees for identifying disconnected hidden nodes
        node_degrees = {node_id: 0 for node_id in all_genome_nodes}

        if genome.connections:
            for conn_key, conn in genome.connections.items():
                if getattr(conn, 'enabled', True):
                    u, v = conn.key
                    if u in all_genome_nodes and v in all_genome_nodes:
                        adj[u].append(v)
                        node_degrees[u] += 1
                        node_degrees[v] += 1
        
        # 2. Layer Connected Hidden Nodes (BFS from inputs)
        # Only consider paths to other hidden nodes or output nodes for layering connected hidden nodes
        queue = deque()
        visited_for_bfs = set()

        for node_id in input_node_ids:
            queue.append((node_id, 0)) # (node, current_level_from_input)
            visited_for_bfs.add(node_id)

        temp_hidden_levels = {} # Store levels for hidden nodes found via BFS

        while queue:
            u, level_u = queue.popleft()

            # Iterate through neighbors
            sorted_neighbors = sorted(adj.get(u, []))
            for v in sorted_neighbors:
                if v in config_input_keys: # Don't go backward to an input node
                    continue

                new_level_v = level_u + 1
                
                if v in config_output_keys: # Path reaches an output node
                    # We don't assign levels to outputs here, but this path contributes to max_level_so_far
                    max_level_so_far = max(max_level_so_far, new_level_v -1) # -1 because output is next level
                    continue # Stop BFS path here for outputs

                # If v is a hidden node
                if v not in config_input_keys and v not in config_output_keys:
                    if v not in temp_hidden_levels or new_level_v < temp_hidden_levels[v]:
                        temp_hidden_levels[v] = new_level_v
                        max_level_so_far = max(max_level_so_far, new_level_v)
                        if v not in visited_for_bfs: # Add to queue only if not visited or found shorter path
                             queue.append((v, new_level_v))
                             visited_for_bfs.add(v) # Mark as visited to handle cycles/multiple paths

        # Assign BFS-derived levels to connected hidden nodes
        for h_node, h_level in temp_hidden_levels.items():
            node_levels[h_node] = h_level


        # 3. Identify Disconnected Hidden Nodes
        disconnected_hidden_node_ids = []
        all_hidden_node_ids = [n_id for n_id in all_genome_nodes if n_id not in config_input_keys and n_id not in config_output_keys]

        for h_node_id in all_hidden_node_ids:
            is_disconnected = False
            # Condition 1: Not reached by BFS from any input (level is still None)
            if node_levels[h_node_id] is None:
                is_disconnected = True
            # Condition 2: Degree < 2 (even if reached by BFS, but user wants it separate)
            if node_degrees.get(h_node_id, 0) < 2:
                is_disconnected = True
                if node_levels[h_node_id] is not None: # Was assigned a level by BFS
                    # print(f"[VIS-INFO] Hidden node {h_node_id} has degree < 2, re-classifying as disconnected.")
                    node_levels[h_node_id] = None # Reset its level, will be placed in disconnected layer

            if is_disconnected:
                disconnected_hidden_node_ids.append(h_node_id)
        
        disconnected_hidden_node_ids = sorted(list(set(disconnected_hidden_node_ids))) # Unique and sorted

        # 4. Determine Placement for Disconnected and Output Layers
        # `max_level_so_far` currently holds the max level of connected hidden nodes (or 0 if no inputs/hidden)
        
        final_disconnected_layer = -1 # Placeholder if no disconnected nodes
        final_output_layer = -1       # Placeholder

        if not disconnected_hidden_node_ids:
            # No disconnected hidden nodes, outputs go directly after connected hidden nodes
            final_output_layer = max_level_so_far + 1
        else:
            # There are disconnected hidden nodes
            tentative_disconnected_layer = max_level_so_far + 1
            tentative_output_layer = tentative_disconnected_layer + 1 # Outputs are one step after disconnected

            # Check for conflict: if tentative_disconnected_layer is already occupied by a *connected* hidden node
            # This should not happen if max_level_so_far was calculated correctly from connected hidden nodes.
            # However, as a safeguard or if logic changes:
            is_conflict = False
            for h_node_id, h_level in node_levels.items():
                if h_level == tentative_disconnected_layer and h_node_id not in disconnected_hidden_node_ids and h_node_id in all_hidden_node_ids:
                    is_conflict = True
                    # print(f"[VIS-WARN] Conflict: Tentative disconnected layer {tentative_disconnected_layer} occupied by connected hidden node {h_node_id}.")
                    break
            
            if is_conflict: # Should be rare with current logic
                final_disconnected_layer = tentative_disconnected_layer + 1
                final_output_layer = final_disconnected_layer + 1
            else:
                final_disconnected_layer = tentative_disconnected_layer
                final_output_layer = tentative_output_layer

        # 5. Assign Levels
        if final_disconnected_layer != -1:
            for dh_node_id in disconnected_hidden_node_ids:
                node_levels[dh_node_id] = final_disconnected_layer
        
        output_node_ids = sorted([n_id for n_id in all_genome_nodes if n_id in config_output_keys])
        for o_node_id in output_node_ids:
            node_levels[o_node_id] = final_output_layer


        # 6. Handle any remaining unassigned nodes (fallback, should be rare)
        for node_id, level in node_levels.items():
            if level is None:
                # print(f"[VIS-WARN] Node {node_id} remained unassigned. Placing in default layer.")
                if node_id in all_hidden_node_ids:
                    node_levels[node_id] = final_disconnected_layer if final_disconnected_layer != -1 else max_level_so_far + 1
                elif node_id in output_node_ids:
                     node_levels[node_id] = final_output_layer
                else: # Should not happen for inputs
                    node_levels[node_id] = 0 # Default to input layer as last resort

        # 7. Normalize Levels to be contiguous (0, 1, 2, ...)
        # This ensures that layers are compact if some intermediate layers ended up empty.
        unique_sorted_levels = sorted(list(set(l for l in node_levels.values() if l is not None and l != float('inf'))))
        
        level_map = {old_level: new_level for new_level, old_level in enumerate(unique_sorted_levels)}
        
        final_node_levels = {}
        current_max_normalized_level = 0
        if not level_map: # Handles empty graph or graph with no assigned levels
            if not all_genome_nodes: # Truly empty genome
                 return {}, 0
            else: # Nodes exist but no levels assigned (e.g. only outputs, no inputs)
                # Fallback: assign all to level 0 or handle as per specific rules for such cases
                # For now, if only outputs, they should have been assigned final_output_layer.
                # This case implies something went very wrong or an edge case not fully covered.
                # print("[VIS-WARN] Level map is empty, but nodes exist. Defaulting levels.")
                # Default all to 0 if not already set.
                for node_id in all_genome_nodes:
                    final_node_levels[node_id] = node_levels.get(node_id, 0) # Use existing if somehow set, else 0
                current_max_normalized_level = 0
        else:
            for node_id, old_level in node_levels.items():
                if old_level is not None and old_level != float('inf'):
                    new_level = level_map.get(old_level, 0) # Default to 0 if old_level not in map (should not happen)
                    final_node_levels[node_id] = new_level
                    current_max_normalized_level = max(current_max_normalized_level, new_level)
                else: # Handle nodes that might still be inf or None (should be resolved by fallback)
                    # print(f"[VIS-WARN] Node {node_id} has problematic level {old_level} before normalization.")
                    final_node_levels[node_id] = 0 # Default to 0

        # Ensure all genome nodes are in final_node_levels
        for node_id in all_genome_nodes:
            if node_id not in final_node_levels:
                # print(f"[VIS-WARN] Node {node_id} was missing from final_node_levels. Defaulting to level 0.")
                final_node_levels[node_id] = 0
                current_max_normalized_level = max(current_max_normalized_level, 0)


        return final_node_levels, current_max_normalized_level

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

    def draw_network(self, genome,
                       actual_letter_pattern=None,
                       hidden_node_activations=None, # NEW: Expect hidden activations
                       actual_output_activations=None,
                       actual_prediction=None,
                       actual_letter=None,
                       is_correct=None):
        """Draw complete visualization including raster grid, network, and prediction with new layout and visuals."""
        
        # --- Setup & Clearing ---
        if hasattr(self, 'inter_ax_patches'):
            for patch in self.inter_ax_patches:
                if patch in self.fig.artists:
                    patch.remove()
        self.inter_ax_patches = []

        for ax in [self.ax_input, self.ax_network, self.ax_pred]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

        self.ax_input.set_aspect('equal', adjustable='box')
        for ax_obj in [self.ax_input, self.ax_network, self.ax_pred]:
            for spine in ax_obj.spines.values():
                spine.set_visible(False)
        self.ax_input.set_facecolor('#1e1e1e')
        self.ax_network.set_facecolor('none')
        self.ax_pred.set_facecolor('none')

        # --- Data Preparation ---
        current_letter_pattern = actual_letter_pattern if actual_letter_pattern is not None else np.zeros((8, 10)) # Adjusted default size
        flattened_pattern = current_letter_pattern.flatten()
        
        # Default hidden activations if not provided
        current_hidden_activations = hidden_node_activations if hidden_node_activations is not None else {}
        
        # Default output activations if not provided
        num_outputs_cfg = self.config.genome_config.num_outputs
        current_output_activations = actual_output_activations if actual_output_activations is not None else [0.0] * num_outputs_cfg
        
        current_prediction = actual_prediction
        
        # Map output node IDs to letters (assuming alphabetical order A, B, C...)
        # Ensure output_keys are sorted numerically before assigning letters
        sorted_output_keys = sorted(list(self.config.genome_config.output_keys))
        output_labels = {key: chr(ord('A') + i) for i, key in enumerate(sorted_output_keys)}
        predicted_output_node_id = None
        if current_prediction:
            try:
                pred_index = ['A', 'B', 'C'].index(current_prediction) # Assuming A, B, C
                if pred_index < len(sorted_output_keys):
                    predicted_output_node_id = sorted_output_keys[pred_index]
            except ValueError:
                pass # Prediction not in A, B, C

        # --- 1. Draw Input Neurons (ax_input) ---
        G_input = nx.DiGraph()
        num_inputs_cfg = self.config.genome_config.num_inputs
        G_input.add_nodes_from(range(num_inputs_cfg)) # Use range for visual indexing

        input_pos = {}
        rows, cols = 8, 10 # Grid size
        x_spacing = 1.0 / (cols - 1) if cols > 1 else 0
        y_spacing = 1.0 / (rows - 1) if rows > 1 else 0
        for i in range(num_inputs_cfg):
            row, col = i // cols, i % cols
            input_pos[i] = (col * x_spacing, (rows - 1 - row) * y_spacing)

        input_node_colors = []
        for i in range(num_inputs_cfg):
            activation = flattened_pattern[i] if i < len(flattened_pattern) else 0.0
            # Use get_activation_color for input nodes in ax_input
            color = self.get_activation_color(activation, min_val=0.0, max_val=1.0, color_low=(0.2, 0.2, 0.8), color_high=(0.8, 0.5, 0.2)) # Blue to Orange
            input_node_colors.append(color)

        nx.draw_networkx_nodes(G_input, input_pos, ax=self.ax_input,
                                 nodelist=list(range(num_inputs_cfg)),
                                 node_color=input_node_colors, node_size=10,
                                 edgecolors='dimgray', linewidths=0.5)
        self.ax_input.set_title('Input Neurons', color='white', fontsize=10, bbox=dict(facecolor='none', edgecolor='dimgray', boxstyle='round,pad=0.3', lw=0.5))
        self.ax_input.set_xlim(-0.1, 1.1)
        self.ax_input.set_ylim(-0.1, 1.1)

        # --- 2. Draw Main Network Topology (ax_network) ---
        G = nx.DiGraph()
        if not genome:
            print("[VIS] Genome is None. Cannot draw network.")
            self.canvas.draw()
            plt.pause(0.01)
            return

        config_input_keys = set(self.config.genome_config.input_keys)
        config_output_keys = set(self.config.genome_config.output_keys)
        current_genome_node_ids = set(genome.nodes.keys())

        # Calculate node levels using the updated method
        node_levels, max_level = self._calculate_node_levels(genome, config_input_keys, config_output_keys)

        # --- Position Calculation ---
        pos = {}
        nodes_by_level = {}
        for node_id_val, level_val in node_levels.items():
            if node_id_val in current_genome_node_ids: # Ensure node exists in genome
                 nodes_by_level.setdefault(level_val, []).append(node_id_val)

        input_x_coord = 0.0
        x_increment_per_level = 1.5 # Horizontal spacing

        def get_y_pos_local(nodes_in_list, current_idx_in_list):
            num_nodes_in_list = len(nodes_in_list)
            if num_nodes_in_list <= 1: return 0
            max_spread = 10.0
            node_spacing_val = max_spread / float(num_nodes_in_list - 1)
            return -(current_idx_in_list * node_spacing_val - (num_nodes_in_list - 1) * node_spacing_val / 2.0)

        all_nodes_for_graph = []
        node_colors_list = []
        node_sizes_list = []
        node_edgecolors_list = []
        node_linewidths_list = []
        node_labels = {} # For output node letters

        # Combine all activations for easier lookup
        all_activations = {}
        # Input activations (map genome input keys to flattened pattern)
        genome_input_keys_sorted = sorted(list(config_input_keys))
        for i, key in enumerate(genome_input_keys_sorted):
             all_activations[key] = flattened_pattern[i] if i < len(flattened_pattern) else 0.0
        # Hidden activations
        all_activations.update(current_hidden_activations)
        # Output activations (map genome output keys to output activation list)
        for i, key in enumerate(sorted_output_keys):
             all_activations[key] = current_output_activations[i] if i < len(current_output_activations) else 0.0

        for level_idx in sorted(nodes_by_level.keys()):
            nodes_in_this_level = nodes_by_level[level_idx]
            
            # --- Sort output nodes alphabetically for vertical positioning ---
            if any(n in config_output_keys for n in nodes_in_this_level):
                 nodes_in_this_level = sorted(nodes_in_this_level, key=lambda n: output_labels.get(n, 'Z')) # Sort by letter, fallback 'Z'
            else:
                 nodes_in_this_level = sorted(nodes_in_this_level) # Sort others by ID

            current_x = input_x_coord + level_idx * x_increment_per_level
            if not nodes_in_this_level: continue

            for i, node_id in enumerate(nodes_in_this_level):
                y_pos_val = get_y_pos_local(nodes_in_this_level, i)
                pos[node_id] = (current_x, y_pos_val)

                if node_id not in all_nodes_for_graph:
                    all_nodes_for_graph.append(node_id)
                    
                    # Determine activation and color
                    activation = all_activations.get(node_id, 0.0) # Default to 0 if activation missing
                    node_color = self.get_activation_color(activation)
                    node_colors_list.append(node_color)

                    # Determine size and labels
                    edge_color = 'black'
                    line_width = 1.0
                    if node_id in config_input_keys:
                        node_sizes_list.append(30)
                    elif node_id in config_output_keys:
                        node_sizes_list.append(50) # Slightly larger for labels
                        node_labels[node_id] = output_labels.get(node_id, '?') # Add letter label
                        # Highlight predicted output node
                        if node_id == predicted_output_node_id:
                            edge_color = 'lime' # Bright green highlight
                            line_width = 2.0
                    else: # Hidden node
                        node_sizes_list.append(40) # Default size for hidden

                    node_edgecolors_list.append(edge_color)
                    node_linewidths_list.append(line_width)

        G.add_nodes_from(all_nodes_for_graph)

        # --- Draw Nodes ---
        nx.draw_networkx_nodes(G, pos, ax=self.ax_network, nodelist=all_nodes_for_graph,
                               node_color=node_colors_list, node_size=node_sizes_list,
                               edgecolors=node_edgecolors_list, linewidths=node_linewidths_list)

        # --- Draw Node Labels (Output Nodes) ---
        nx.draw_networkx_labels(G, pos, ax=self.ax_network, labels=node_labels, font_size=8, font_color='black')


        # --- Draw Connections ---
        edge_colors_list_for_edges = []
        edges_to_draw = []
        fixed_edge_width = 0.5 # Thin connections

        if genome.connections:
            for conn_key, conn in genome.connections.items():
                in_node_id, out_node_id = conn.key
                if getattr(conn, 'enabled', True) and in_node_id in G and out_node_id in G:
                    edges_to_draw.append(conn_key)
                    # Color based on target node activation
                    target_activation = all_activations.get(out_node_id, 0.0)
                    edge_color = self.get_activation_color(target_activation)
                    edge_colors_list_for_edges.append(edge_color)

        if edges_to_draw:
            nx.draw_networkx_edges(G, pos, ax=self.ax_network, edgelist=edges_to_draw,
                                   width=fixed_edge_width, # Use fixed thin width
                                   edge_color=edge_colors_list_for_edges,
                                   arrowsize=7, arrowstyle='->', alpha=0.7) # Slightly increased alpha

        self.ax_network.set_title('Network Topology', color='white', fontsize=10, bbox=dict(facecolor='none', edgecolor='dimgray', boxstyle='round,pad=0.3', lw=0.5))
        self.ax_network.autoscale_view()
        self.ax_network.margins(0.1)

        # --- 3. Add ConnectionPatches between subplots ---
        # Define input_node_ids for this scope
        input_node_ids = sorted([n_id for n_id in current_genome_node_ids if n_id in config_input_keys])

        if input_node_ids and len(input_pos) == len(genome_input_keys_sorted):
            for i in range(len(genome_input_keys_sorted)):
                ax_input_idx = i # Visual index in ax_input
                network_input_node_id = genome_input_keys_sorted[i]

                if network_input_node_id in pos and ax_input_idx in input_pos:
                    ax_input_coord = input_pos[ax_input_idx]
                    ax_network_coord = pos[network_input_node_id]
                    
                    # Color based on input activation
                    activation = flattened_pattern[i] if i < len(flattened_pattern) else 0.0
                    patch_color = self.get_activation_color(activation, min_val=0.0, max_val=1.0, color_low=(0.2, 0.2, 0.8), color_high=(0.8, 0.5, 0.2)) # Blue to Orange
                    
                    con = ConnectionPatch(xyA=ax_input_coord, xyB=ax_network_coord,
                                          coordsA="data", coordsB="data",
                                          axesA=self.ax_input, axesB=self.ax_network,
                                          color=patch_color, linestyle=":",
                                          alpha=0.5 if activation > 0 else 0.2, # More visible if active
                                          linewidth=0.5, zorder=-1)
                    self.fig.add_artist(con)
                    self.inter_ax_patches.append(con)


        # --- 4. Draw Prediction Text (ax_pred) ---
        if current_prediction is None: current_prediction = "?"
        prediction_color = 'green' if is_correct else ('red' if is_correct is not None else 'white')
        prediction_text = f"Predicted:\n{current_prediction}"
        if actual_letter is not None: prediction_text += f" ({actual_letter})"

        self.ax_pred.set_title('Prediction', color='white', fontsize=10, bbox=dict(facecolor='none', edgecolor='dimgray', boxstyle='round,pad=0.3', lw=0.5))
        self.ax_pred.text(0.5, 0.5, prediction_text, color=prediction_color, ha='center', va='center', fontsize=10)

        # --- Final Draw ---
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
