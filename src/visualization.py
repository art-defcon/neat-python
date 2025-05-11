import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
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

    def create_visualization(self):
        """Center pane - Network Visualization"""
        from PyQt5.QtWidgets import QWidget, QVBoxLayout # Import here to avoid circular dependency if NEATVisualization is in a separate file

        self.viz_widget = QWidget()
        self.viz_widget.setStyleSheet("background-color: #1e1e1e;")
        self.viz_layout = QVBoxLayout(self.viz_widget)

        # Create matplotlib figure with multiple subplots
        self.fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        gs = self.fig.add_gridspec(1, 5, width_ratios=[1, 1, 3, 1, 1])

        # Create axes
        self.ax_grid = self.fig.add_subplot(gs[0])    # Raster grid
        self.ax_input = self.fig.add_subplot(gs[1])   # Input neurons
        self.ax_network = self.fig.add_subplot(gs[2]) # Network topology
        self.ax_output = self.fig.add_subplot(gs[3])  # Output neurons
        self.ax_pred = self.fig.add_subplot(gs[4])    # Prediction text

        # Configure axes
        for ax in [self.ax_grid, self.ax_input, self.ax_network, self.ax_output, self.ax_pred]:
            ax.set_facecolor('#1e1e1e')
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_xticks([])
            ax.set_yticks([])

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
        mock_letter_pattern = np.array([
            [0, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1]
        ]) # 6x6

        # 2. Mock Network Genome
        # Create a new genome config if not available (should be from self.config)
        genome_config = self.config.genome_config
        mock_genome = self.config.genome_type(0) # Key 0
        mock_genome.configure_new(genome_config)

        # Define nodes: 36 inputs (0-35), 3 hidden (36,37,38), 3 outputs (-1,-2,-3)
        # Inputs are implicitly defined by num_inputs in config
        # Hidden nodes
        for i in range(3):
            node_id = genome_config.num_inputs + i
            mock_genome.nodes[node_id] = genome_config.node_gene_type(node_id)
            mock_genome.nodes[node_id].bias = random.uniform(-0.5, 0.5)
            mock_genome.nodes[node_id].response = 1.0
            mock_genome.nodes[node_id].activation = 'tanh'
            mock_genome.nodes[node_id].aggregation = 'sum'
        # Output nodes
        for i in range(genome_config.num_outputs):
            node_id = -i - 1
            mock_genome.nodes[node_id] = genome_config.node_gene_type(node_id)
            mock_genome.nodes[node_id].bias = 0.0 # Outputs often have 0 bias
            mock_genome.nodes[node_id].response = 1.0
            mock_genome.nodes[node_id].activation = 'tanh'
            mock_genome.nodes[node_id].aggregation = 'sum'

        # Define connections
        # Input 0 to Hidden 36
        cg1 = genome_config.connection_gene_type((0, genome_config.num_inputs + 0))
        cg1.weight = 0.5
        mock_genome.connections[cg1.key] = cg1
        # Input 10 to Hidden 37
        cg2 = genome_config.connection_gene_type((10, genome_config.num_inputs + 1))
        cg2.weight = -0.3
        mock_genome.connections[cg2.key] = cg2
        # Hidden 36 to Output -1
        cg3 = genome_config.connection_gene_type((genome_config.num_inputs + 0, -1))
        cg3.weight = 0.8
        mock_genome.connections[cg3.key] = cg3
        # Hidden 37 to Output -2
        cg4 = genome_config.connection_gene_type((genome_config.num_inputs + 1, -2))
        cg4.weight = 0.2
        mock_genome.connections[cg4.key] = cg4
         # Hidden 38 to Output -3 (no input to hidden 38, so it's a disconnected part)
        cg5 = genome_config.connection_gene_type((genome_config.num_inputs + 2, -3))
        cg5.weight = 0.4
        mock_genome.connections[cg5.key] = cg5


        # 3. Mock Output Activations
        mock_output_activations = [0.75, 0.20, 0.05]

        # 4. Mock Predicted Letter
        mock_prediction = "A"

        return mock_genome, mock_letter_pattern, mock_output_activations, mock_prediction

    def draw_network(self, genome, is_mock=False, mock_letter_pattern=None, mock_output_activations=None, mock_prediction=None):
        """Draw complete visualization including raster grid, network, and prediction"""
        # Clear all axes
        for ax in [self.ax_grid, self.ax_input, self.ax_network, self.ax_output, self.ax_pred]:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])

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
            # Need to get the current letter pattern from the main app or pass it in
            # For now, using a placeholder or assuming self.current_letter is set
            if self.current_letter:
                 current_letter_pattern, _ = self.generate_letter_pattern(self.current_letter)
            else:
                 current_letter_pattern = np.zeros((6,6)) # Default to blank if no letter

            # Calculate real activations and prediction if needed (or they are passed if pre-calculated)
            # This part needs access to the NEAT population/config, which should be passed or accessed differently
            # For now, using placeholder logic
            try:
                import neat # Assuming neat is available
                net = neat.nn.FeedForwardNetwork.create(genome, self.config)
                input_pattern_flat = current_letter_pattern.flatten().astype(float)
                if len(input_pattern_flat) == self.config.genome_config.num_inputs:
                    current_output_activations = net.activate(input_pattern_flat)
                else:
                    current_output_activations = [0.0] * self.config.genome_config.num_outputs
            except Exception:
                current_output_activations = [0.0] * self.config.genome_config.num_outputs

            # This also needs access to the classification logic, which should be passed or accessed
            current_prediction = self.classify_letter(genome, current_letter_pattern) # Needs letter pattern

        # Ensure current_letter_pattern is not None before imshow
        if current_letter_pattern is None:
             current_letter_pattern = np.zeros((6,6)) # Default to blank if None

        # 1. Draw raster grid (6x6)
        self.ax_grid.imshow(current_letter_pattern, cmap='gray', vmin=0, vmax=1)
        self.ax_grid.set_title('Input Pattern (6x6)', color='white', fontsize=12)

        # 2. Draw input neurons (36 nodes)
        G_input = nx.DiGraph()
        # Ensure input nodes are added based on the actual config
        num_inputs_cfg = self.config.genome_config.num_inputs
        G_input.add_nodes_from(range(num_inputs_cfg))

        # Calculate positions for input neurons in an 6x6 grid
        input_pos = {}
        rows, cols = 6, 6
        # Adjust spacing and centering for the grid
        x_spacing = 1.0 / (cols - 1) if cols > 1 else 0
        y_spacing = 1.0 / (rows - 1) if rows > 1 else 0

        for i in range(num_inputs_cfg):
            # Assuming row-major order for flattened image data
            row = i // cols
            col = i % cols
            # Map grid coordinates to plot coordinates (adjusting for origin and scaling)
            # Reverse both horizontal and vertical order
            input_pos[i] = ((cols - 1 - col) * x_spacing, (rows - 1 - row) * y_spacing)

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
                                 node_color=input_node_colors, node_size=30) # Smaller nodes
        self.ax_input.set_title('Input Neurons', color='white', fontsize=12)

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

        input_node_ids = sorted([n for n in genome.nodes if n < num_inputs])
        output_node_ids = sorted([n for n in genome.nodes if n < 0])
        hidden_node_ids = sorted([n for n in genome.nodes if n >= num_inputs])

        layer_x_coords = {'input': 0, 'hidden': 1.5, 'output': 3.0}

        def get_y_pos(nodes_in_layer_list, current_idx, total_nodes_in_ref_layer=len(input_node_ids) if input_node_ids else 1):
            if not nodes_in_layer_list: return 0
            num_nodes_in_layer = len(nodes_in_layer_list)
            if num_nodes_in_layer == 0: return 0 # Avoid division by zero
            # Spread nodes, aiming for a total spread similar to input layer's potential spread
            # Max spread of 10 units (like input layer if it had 10 items with 1 unit spacing)
            max_spread = 10.0
            node_spacing = max_spread / float(num_nodes_in_layer) if num_nodes_in_layer > 1 else 0
            # Center the group of nodes around y=0
            return -(current_idx * node_spacing - (num_nodes_in_layer - 1) * node_spacing / 2.0)

        all_nodes_for_graph = []
        for i, node_id in enumerate(input_node_ids):
            pos[node_id] = (layer_x_coords['input'], get_y_pos(input_node_ids, i))
            all_nodes_for_graph.append(node_id)
            node_colors_list.append(color_map['input'])
            node_sizes_list.append(30)

        for i, node_id in enumerate(hidden_node_ids):
            pos[node_id] = (layer_x_coords['hidden'], get_y_pos(hidden_node_ids, i))
            all_nodes_for_graph.append(node_id)
            node_colors_list.append(color_map['hidden'])
            node_sizes_list.append(40 + 20 * abs(genome.nodes[node_id].bias if node_id in genome.nodes and genome.nodes[node_id].bias else 0)) # Check if node exists

        for i, node_id in enumerate(output_node_ids):
            pos[node_id] = (layer_x_coords['output'], get_y_pos(output_node_ids, i))
            all_nodes_for_graph.append(node_id)
            node_colors_list.append(color_map['output'])
            node_sizes_list.append(35)

        G.add_nodes_from(all_nodes_for_graph)

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
                               node_color=node_colors_list, node_size=node_sizes_list)
        if edges_to_draw: # Only draw edges if there are any
            nx.draw_networkx_edges(G, pos, ax=self.ax_network, edgelist=edges_to_draw,
                                   width=edge_widths, edge_color=edge_colors_list_for_edges,
                                   arrowsize=7, arrowstyle='->', alpha=0.6)
        self.ax_network.set_title('Network Topology', color='white', fontsize=12)
        self.ax_network.autoscale_view()
        self.ax_network.set_xticks([]) # Ensure ticks are off
        self.ax_network.set_yticks([])
        # Set margins for the network plot to give some space
        self.ax_network.margins(0.1)


        # 4. Draw output neurons with activation visualization
        # Use current_output_activations which is set based on mock or real mode
        if current_output_activations is None or len(current_output_activations) != self.config.genome_config.num_outputs:
            current_output_activations = [0.0] * self.config.genome_config.num_outputs

        num_outputs_cfg = self.config.genome_config.num_outputs
        for i, act_val in enumerate(current_output_activations):
            act = max(0, min(1, act_val))
            alpha_val = int(act * 200 + 55)
            if alpha_val > 255: alpha_val = 255
            if alpha_val < 0: alpha_val = 0
            color_hex = f"{alpha_val:02x}"
            color = f'#59a14f{color_hex}'

            node_radius = 0.10 # Relative radius
            # Spread out vertically, centered in the 0-1 box
            # Total height for nodes: num_outputs_cfg * (2*node_radius + spacing_factor)
            # Let's use a fixed spacing for simplicity
            spacing_between_nodes = node_radius * 0.5
            total_height_nodes = num_outputs_cfg * (2*node_radius) + max(0, num_outputs_cfg - 1) * spacing_between_nodes
            start_y = 0.5 + total_height_nodes/2.0 - node_radius # Top of highest node

            y_pos = start_y - i * (2*node_radius + spacing_between_nodes)

            self.ax_output.add_patch(plt.Circle(
                (0.5, y_pos), node_radius,
                color=color,
                alpha=0.7 + act * 0.3
            ))

            self.ax_output.text(
                0.5, y_pos,
                f"{['A','B','C'][i % 3]}\n{act_val:.2f}",
                ha='center', va='center',
                color='white', fontsize=8
            )

        self.ax_output.set_title('Outputs', color='white', fontsize=12)
        self.ax_output.set_xlim(0, 1)
        self.ax_output.set_ylim(0, 1)

        # 5. Draw prediction text
        # Use current_prediction which is set based on mock or real mode
        if current_prediction is None:
            current_prediction = "?"
        self.ax_pred.text(0.5, 0.5, f"Predicted:\n{current_prediction}",
                         color='white', ha='center', va='center', fontsize=12)

        self.canvas.draw()
        plt.pause(0.01)

    def _pixmap_to_matrix(self, pixmap, actual_letter):
        """Convert QPixmap to 6x6 binary matrix"""
        image = pixmap.toImage()
        matrix = []
        for y in range(6):
            row = []
            for x in range(6):
                pixel = image.pixelColor(x, y)
                row.append(0 if pixel.lightness() > 127 else 1)
            matrix.append(row)
        return np.array(matrix), actual_letter

    def generate_letter_pattern(self, letter):
        """Generate 6x6 pattern from random system font for a given letter"""
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

        # Scale down to 6x6 and convert to matrix
        return self._pixmap_to_matrix(pixmap.scaled(
            6, 6, Qt.KeepAspectRatio, Qt.SmoothTransformation), letter)

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
