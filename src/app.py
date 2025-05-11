import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QSlider, QCheckBox, QPushButton)
from PyQt5.QtCore import Qt, QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import neat
import os
import random
import numpy as np
import networkx as nx

class NEATLetterClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEAT Letter Classifier")
        self.setStyleSheet("background-color: #1e1e1e;")
        
        # Configuration
        self.config = {
            'mutation_rate': 0.1,
            'population_size': 150,
            'fitness_threshold': 0.95,
            'generation': 0,
            'best_fitness': 0,
            'training_samples': 0
        }
        
        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        self.mock_data_enabled = True # Initialize state variable
        
        # Setup UI
        self.create_controls()
        self.create_visualization()
        self.create_stats_panel()
        
        # NEAT setup
        self.setup_neat()
        
        # Network visualization setup
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.viz_layout.addWidget(self.canvas)
        # Removed plt.ion() as it might cause an extra window
        
    def create_controls(self):
        """Left pane - Controls"""
        control_widget = QWidget()
        control_widget.setStyleSheet("background-color: #1e1e1e;")
        control_layout = QVBoxLayout(control_widget)
        
        # Parameters
        mutation_label = QLabel("Mutation Rate:")
        mutation_label.setStyleSheet("color: white;")
        control_layout.addWidget(mutation_label)
        
        self.mutation_slider = QSlider(Qt.Horizontal)
        self.mutation_slider.setRange(1, 50)
        self.mutation_slider.setValue(int(self.config['mutation_rate'] * 100))
        self.mutation_slider.setStyleSheet("background-color: #2d2d2d;")
        control_layout.addWidget(self.mutation_slider)
        
        pop_label = QLabel("Population Size:")
        pop_label.setStyleSheet("color: white;")
        control_layout.addWidget(pop_label)
        
        self.pop_slider = QSlider(Qt.Horizontal)
        self.pop_slider.setRange(50, 300)
        self.pop_slider.setSingleStep(10)
        self.pop_slider.setValue(self.config['population_size'])
        self.pop_slider.setStyleSheet("background-color: #2d2d2d;")
        control_layout.addWidget(self.pop_slider)
        
        fitness_label = QLabel("Fitness Threshold:")
        fitness_label.setStyleSheet("color: white;")
        control_layout.addWidget(fitness_label)
        
        self.fitness_slider = QSlider(Qt.Horizontal)
        self.fitness_slider.setRange(80, 99)
        self.fitness_slider.setValue(int(self.config['fitness_threshold'] * 100))
        self.fitness_slider.setStyleSheet("background-color: #2d2d2d;")
        control_layout.addWidget(self.fitness_slider)
        
        # Auto-Evolve toggle
        self.auto_evolve = QCheckBox("Auto-Evolve")
        self.auto_evolve.setStyleSheet("color: white;")
        self.auto_evolve.setChecked(False) # Default to False when mock is on
        control_layout.addWidget(self.auto_evolve)

        # Mock Data toggle
        self.mock_data_checkbox = QCheckBox("Mock Data")
        self.mock_data_checkbox.setStyleSheet("color: white;")
        self.mock_data_checkbox.setChecked(True) # Default to True
        self.mock_data_checkbox.stateChanged.connect(self.on_mock_data_toggled)
        control_layout.addWidget(self.mock_data_checkbox)
        
        self.main_layout.addWidget(control_widget)

    def on_mock_data_toggled(self, state):
        self.mock_data_enabled = (state == Qt.Checked)
        if self.mock_data_enabled:
            self.auto_evolve.setChecked(False)
            self.auto_evolve.setEnabled(False)
            # If a timer for evolution is running, stop it (more robust timer management might be needed)
            # For now, run_evolution will check this flag.
            if hasattr(self, 'evolution_timer') and self.evolution_timer:
                self.evolution_timer.stop()
            self.draw_mock_network_wrapper() # Redraw with mock data
        else:
            self.auto_evolve.setEnabled(True)
            # Potentially restart evolution if auto_evolve was checked
            if self.auto_evolve.isChecked():
                self.run_evolution()
            else: # Or just draw the current best real genome if not auto-evolving
                if hasattr(self, 'p') and self.p and hasattr(self.p, 'best_genome') and self.p.best_genome:
                    self.draw_network(self.p.best_genome)
                else: # If no best_genome, maybe draw an empty real network or a default one
                    self.draw_network(self.config.genome_type(0)) # Draw a default empty genome

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
            [1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]) # 8x6

        # 2. Mock Network Genome
        # Create a new genome config if not available (should be from self.config)
        genome_config = self.config.genome_config
        mock_genome = self.config.genome_type(0) # Key 0
        mock_genome.configure_new(genome_config)

        # Define nodes: 48 inputs (0-47), 3 hidden (48,49,50), 3 outputs (-1,-2,-3)
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
        # Input 0 to Hidden 48
        cg1 = genome_config.connection_gene_type((0, genome_config.num_inputs + 0))
        cg1.weight = 0.5
        mock_genome.connections[cg1.key] = cg1
        # Input 10 to Hidden 49
        cg2 = genome_config.connection_gene_type((10, genome_config.num_inputs + 1))
        cg2.weight = -0.3
        mock_genome.connections[cg2.key] = cg2
        # Hidden 48 to Output -1
        cg3 = genome_config.connection_gene_type((genome_config.num_inputs + 0, -1))
        cg3.weight = 0.8
        mock_genome.connections[cg3.key] = cg3
        # Hidden 49 to Output -2
        cg4 = genome_config.connection_gene_type((genome_config.num_inputs + 1, -2))
        cg4.weight = 0.2
        mock_genome.connections[cg4.key] = cg4
         # Hidden 50 to Output -3 (no input to hidden 50, so it's a disconnected part)
        cg5 = genome_config.connection_gene_type((genome_config.num_inputs + 2, -3))
        cg5.weight = 0.4
        mock_genome.connections[cg5.key] = cg5


        # 3. Mock Output Activations
        mock_output_activations = [0.75, 0.20, 0.05]

        # 4. Mock Predicted Letter
        mock_prediction = "A"

        return mock_genome, mock_letter_pattern, mock_output_activations, mock_prediction

    def create_visualization(self):
        """Center pane - Network Visualization"""
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
    
    def create_stats_panel(self):
        """Right pane - Stats & Info"""
        stats_widget = QWidget()
        stats_widget.setStyleSheet("background-color: #1e1e1e;")
        stats_layout = QVBoxLayout(stats_widget)
        
        # Generation info
        gen_label = QLabel("Generation:")
        gen_label.setStyleSheet("color: white;")
        stats_layout.addWidget(gen_label)
        
        self.gen_label = QLabel("0")
        self.gen_label.setStyleSheet("color: cyan; font: 10pt 'Consolas';")
        stats_layout.addWidget(self.gen_label)
        
        # Best fitness
        fitness_label = QLabel("Best Fitness:")
        fitness_label.setStyleSheet("color: white;")
        stats_layout.addWidget(fitness_label)
        
        self.fitness_label = QLabel("0.00")
        self.fitness_label.setStyleSheet("color: cyan; font: 10pt 'Consolas';")
        stats_layout.addWidget(self.fitness_label)
        
        # Training samples
        samples_label = QLabel("Training Samples:")
        samples_label.setStyleSheet("color: white;")
        stats_layout.addWidget(samples_label)
        
        self.samples_label = QLabel("0")
        self.samples_label.setStyleSheet("color: cyan; font: 10pt 'Consolas';")
        stats_layout.addWidget(self.samples_label)
        
        # Fitness history graph
        fig, self.ax = plt.subplots(figsize=(1.5, 1.5))
        self.line, = self.ax.plot([], [], 'g-')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        self.ax.set_title("Fitness History")
        
        canvas = FigureCanvasQTAgg(fig)
        stats_layout.addWidget(canvas)
        plt.close(fig) # Close the figure to prevent it from opening in a separate window
        
        self.main_layout.addWidget(stats_widget)
    
    def setup_neat(self):
        """Initialize NEAT configuration"""
        self.config_path = os.path.join(os.path.dirname(__file__), 'neat_config')
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 self.config_path)
        
        # Create population
        self.p = neat.Population(self.config)
        self.p.add_reporter(neat.StdOutReporter(True))
        self.p.add_reporter(neat.StatisticsReporter())
        
        # Run evolution (or initial mock draw)
        self.on_mock_data_toggled(Qt.Checked if self.mock_data_enabled else Qt.Unchecked) # Initial setup based on mock_data_enabled
    
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
            current_letter_pattern, _ = self.generate_letter_pattern()
            # Calculate real activations and prediction if needed (or they are passed if pre-calculated)
            if hasattr(self, 'p') and self.p.best_genome and genome == self.p.best_genome:
                try:
                    net = neat.nn.FeedForwardNetwork.create(self.p.best_genome, self.config)
                    input_pattern_flat = current_letter_pattern.flatten().astype(float)
                    if len(input_pattern_flat) == self.config.genome_config.num_inputs:
                        current_output_activations = net.activate(input_pattern_flat)
                    else:
                        current_output_activations = [0.0] * self.config.genome_config.num_outputs
                except Exception:
                    current_output_activations = [0.0] * self.config.genome_config.num_outputs
            else: # Fallback for genomes not yet fully processed or if p.best_genome is not set
                 current_output_activations = [0.0] * self.config.genome_config.num_outputs
            
            current_prediction = self.classify_letter(genome) # Uses current_letter_pattern internally

        # Ensure current_letter_pattern is not None before imshow
        if current_letter_pattern is None:
             current_letter_pattern = np.zeros((8,6)) # Default to blank if None

        # 1. Draw raster grid (8x6)
        self.ax_grid.imshow(current_letter_pattern, cmap='gray', vmin=0, vmax=1)
        self.ax_grid.set_title('Input Pattern', color='white')
        
        # 2. Draw input neurons (48 nodes)
        G_input = nx.DiGraph()
        # Ensure input nodes are added based on the actual config
        num_inputs_cfg = self.config.genome_config.num_inputs
        G_input.add_nodes_from(range(num_inputs_cfg))
        
        # Calculate positions for input neurons in an 8x6 grid
        input_pos = {}
        rows, cols = 8, 6
        # Adjust spacing and centering for the grid
        x_spacing = 1.0 / (cols - 1) if cols > 1 else 0
        y_spacing = 1.0 / (rows - 1) if rows > 1 else 0
        
        for i in range(num_inputs_cfg):
            # Assuming row-major order for flattened image data
            row = i // cols
            col = i % cols
            # Map grid coordinates to plot coordinates (adjusting for origin and scaling)
            # Invert y-axis to match image convention (origin top-left)
            input_pos[i] = (col * x_spacing, 1.0 - row * y_spacing) 

        nx.draw_networkx_nodes(G_input, input_pos, ax=self.ax_input,
                                 nodelist=list(range(num_inputs_cfg)), # Explicitly pass nodelist
                                 node_color='#4e79a7', node_size=30) # Smaller nodes
        self.ax_input.set_title('Input Neurons', color='white', fontsize=9)
        
        # Set limits to encompass the grid
        self.ax_input.set_xlim(-0.1, 1.1)
        self.ax_input.set_ylim(-0.1, 1.1)
        self.ax_input.invert_yaxis() # Invert y-axis to match image
        
        # 3. Draw main network topology
        G = nx.DiGraph()
        
        # Force initial topology if empty
        if not genome.connections:
            num_inputs = self.config.genome_config.num_inputs
            num_outputs = self.config.genome_config.num_outputs

            # Add 3 hidden nodes (e.g., IDs 48, 49, 50 if num_inputs is 48)
            hidden_node_ids = [num_inputs + i for i in range(3)] 
            for h_id in hidden_node_ids:
                if h_id not in genome.nodes: # Check if node already exists
                    node_gene = self.config.genome_config.node_gene_type(h_id)
                    node_gene.bias = random.uniform(-1, 1)
                    node_gene.response = self.config.genome_config.response_init_mean
                    node_gene.activation = self.config.genome_config.activation_default
                    node_gene.aggregation = self.config.genome_config.aggregation_default
                    genome.nodes[h_id] = node_gene

            # Output node IDs (e.g., -1, -2, -3 for num_outputs = 3)
            output_node_ids = [-i - 1 for i in range(num_outputs)]
            for o_id in output_node_ids:
                if o_id not in genome.nodes: # Check if node already exists
                    node_gene = self.config.genome_config.node_gene_type(o_id)
                    node_gene.bias = 0.0 
                    node_gene.response = self.config.genome_config.response_init_mean
                    node_gene.activation = self.config.genome_config.activation_default
                    node_gene.aggregation = self.config.genome_config.aggregation_default
                    genome.nodes[o_id] = node_gene

            # Connect inputs to hidden
            for i in range(num_inputs):
                for h_id in hidden_node_ids:
                    cg = self.config.genome_config.connection_gene_type((i, h_id))
                    cg.weight = random.uniform(-1, 1)
                    genome.connections[(i, h_id)] = cg
            
            # Connect hidden to outputs
            for h_id in hidden_node_ids:
                for o_id in output_node_ids:
                    cg = self.config.genome_config.connection_gene_type((h_id, o_id))
                    cg.weight = random.uniform(-1, 1)
                    genome.connections[(h_id, o_id)] = cg

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
            node_sizes_list.append(40 + 20 * abs(genome.nodes[node_id].bias if genome.nodes[node_id].bias else 0))

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
        self.ax_network.set_title('Network Topology', color='white', fontsize=9)
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
        
        self.ax_output.set_title('Outputs', color='white', fontsize=9)
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
    
    def run_evolution(self):
        """Main evolution loop"""
        if self.mock_data_enabled:
            # If mock data is enabled, ensure no evolution timer runs and UI reflects this
            if hasattr(self, 'evolution_timer') and self.evolution_timer:
                self.evolution_timer.stop()
            self.auto_evolve.setChecked(False)
            self.auto_evolve.setEnabled(False)
            return

        if not hasattr(self, 'generation'):
            self.generation = 0
        self.generation += 1
        self.gen_label.setText(str(self.generation))
        
        # Store current letter for this generation
        self.current_letter = self.generate_letter()
        
        # Evaluate genomes and find best
        for genome_id, genome in self.p.population.items():
            genome.fitness = self.evaluate_genome(genome)
        
        # Set best genome
        self.p.best_genome = max(self.p.population.values(), key=lambda g: g.fitness)
        
        # Check fitness threshold
        if hasattr(self.p, 'best_genome') and self.p.best_genome and \
           self.p.best_genome.fitness >= self.config.fitness_threshold:
            self.best_fitness = self.p.best_genome.fitness
            self.fitness_label.setText(f"{self.best_fitness:.2f}")
            return
        
        # Evolve
        self.p.run(self.evaluate, 1)
        
        # Update visualization with best genome
        self.draw_network(self.p.best_genome)
        
        # Auto-evolve if enabled (2000ms interval)
        if self.auto_evolve.isChecked() and not self.mock_data_enabled:
            if hasattr(self, 'evolution_timer') and self.evolution_timer: # Clear existing timer
                self.evolution_timer.stop()
            self.evolution_timer = QTimer()
            self.evolution_timer.setSingleShot(True)
            self.evolution_timer.timeout.connect(self.run_evolution)
            self.evolution_timer.start(2000)
    
    def evaluate_genome(self, genome):
        """Evaluate a single genome based on the current_letter"""
        if not genome: return 0.0
        
        # Get the current letter and its pattern
        # self.current_letter should be set by run_evolution before this is called
        letter_pattern, actual_letter = self.generate_letter_pattern() # Uses self.current_letter
        
        if letter_pattern is None: return 0.0

        try:
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            input_data = letter_pattern.flatten().astype(float)
            
            if len(input_data) != self.config.genome_config.num_inputs:
                # print(f"Warning: Input data size {len(input_data)} does not match num_inputs {self.config.genome_config.num_inputs}")
                return 0.0 # Cannot evaluate if input size is wrong

            output_activations = net.activate(input_data)
            predicted_idx = output_activations.index(max(output_activations))
            predicted_letter = ['A', 'B', 'C'][predicted_idx % 3]
            
            return 1.0 if predicted_letter == actual_letter else 0.0
        except Exception as e:
            # print(f"Error evaluating genome {genome.key if genome else 'None'}: {e}")
            return 0.0
    
    def generate_letter(self):
        """Generate random A/B/C pattern"""
        return random.choice(['A', 'B', 'C'])
        
    def _pixmap_to_matrix(self, pixmap):
        """Convert QPixmap to 8x6 binary matrix"""
        image = pixmap.toImage()
        matrix = []
        for y in range(8):
            row = []
            for x in range(6):
                pixel = image.pixelColor(x, y)
                row.append(0 if pixel.lightness() > 127 else 1)
            matrix.append(row)
        return np.array(matrix), self.current_letter

    def generate_letter_pattern(self):
        """Generate 8x6 pattern from random system font"""
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
        painter.drawText(pixmap.rect(), Qt.AlignCenter, self.current_letter)
        painter.end()
        
        # Scale down to 8x6 and convert to matrix
        return self._pixmap_to_matrix(pixmap.scaled(
            6, 8, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def classify_letter(self, genome):
        """Classify letter using NEAT network based on current_letter_pattern"""
        if not genome: return "?"
        
        letter_pattern, _ = self.generate_letter_pattern() # Uses self.current_letter
        if letter_pattern is None: return "?"

        try:
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
    
    def evaluate(self, genomes, config):
        """NEAT evaluation function - evaluates each genome against the current letter"""
        # self.current_letter is set once per generation in run_evolution
        # For simplicity, all genomes in this generation are evaluated against this single current_letter
        letter_pattern, actual_letter = self.generate_letter_pattern() # Uses self.current_letter
        
        if letter_pattern is None:
            for genome_id, genome in genomes:
                genome.fitness = 0.0
            return

        input_data = letter_pattern.flatten().astype(float)
        if len(input_data) != config.genome_config.num_inputs:
            # print(f"Warning: Input data size {len(input_data)} for batch eval does not match num_inputs {config.genome_config.num_inputs}")
            for genome_id, genome in genomes:
                genome.fitness = 0.0
            return

        for genome_id, genome in genomes:
            try:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                output_activations = net.activate(input_data)
                predicted_idx = output_activations.index(max(output_activations))
                predicted_letter = ['A', 'B', 'C'][predicted_idx % 3]
                genome.fitness = 1.0 if predicted_letter == actual_letter else 0.0
            except Exception as e:
                # print(f"Error in batch evaluating genome {genome_id}: {e}")
                genome.fitness = 0.0

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NEATLetterClassifier()
    window.show()
    sys.exit(app.exec_())
