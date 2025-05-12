import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QSlider, QCheckBox, QPushButton, QGroupBox)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import os

# Import refactored modules
from visualization import NEATVisualization
from neat_logic import NEATLogic

class NEATLetterClassifier(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEAT Letter Classifier")
        self.setStyleSheet("background-color: #1e1e1e;")
        self.resize(1400, self.height()) # Set default width to 1400px

        # Configuration (initial values, will be updated by NEATLogic)
        # 'mutation_rate' is removed as it's replaced by specific mutation sliders
        self.config_values = {
            'population_size': 150, # Default, will be configurable
            'fitness_threshold': 0.95, # Default, will be configurable
            'generation': 0,
            'best_fitness': 0,
            'training_samples': 0 # This will be updated/repurposed
        }

        # Main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        # Instantiate refactored components
        self.neat_logic = NEATLogic(os.path.join(os.path.dirname(__file__), 'neat_config'))
        # NEAT setup
        self.neat_logic.setup_neat() # Use neat_logic component

        # Pass the main_layout and neat_logic config to visualization
        self.visualization = NEATVisualization(self.main_layout, self.neat_logic.config)


        # Setup UI
        self.create_controls()
        self.visualization.create_visualization() # Use visualization component
        self.create_stats_panel()

        # --- Initial Network Visualization ---
        # Generate a random letter and classify it with the initial genome
        initial_letter = self.neat_logic.generate_letter()
        initial_letter_pattern, _ = self.neat_logic.generate_letter_pattern(initial_letter)

        initial_genome = None
        if self.neat_logic.p and self.neat_logic.p.population:
            initial_genome = next(iter(self.neat_logic.p.population.values()), None)

        if initial_genome and initial_letter_pattern is not None:
            predicted_letter, output_activations = self.neat_logic.classify_letter(initial_genome, initial_letter_pattern)
            is_correct = (predicted_letter == initial_letter)

            self.visualization.draw_network(
                genome=initial_genome,
                actual_letter_pattern=initial_letter_pattern,
                actual_output_activations=output_activations,
                actual_prediction=predicted_letter,
                actual_letter=initial_letter,
                is_correct=is_correct
            )
        elif initial_genome:
             # Handle case where letter_pattern could not be generated
             self.visualization.draw_network(genome=initial_genome, actual_prediction="Error: No Pattern", actual_letter=None)
        else:
             # Handle case: No initial genome found
             self.visualization.draw_network(genome=None, actual_prediction="Error: No Genome", actual_letter=None)

        self.update_stats_display() # Update stats for the initial state
        # --- End of Initial Network Visualization ---

        # Network visualization setup (This part is now handled by NEATVisualization)
        # self.fig, self.ax = plt.subplots(figsize=(6, 4))
        # self.canvas = FigureCanvasQTAgg(self.fig)
        # self.viz_layout.addWidget(self.canvas)
        # Removed plt.ion() as it might cause an extra window

    def create_controls(self):
        """Left pane - Controls"""
        control_widget = QWidget()
        control_widget.setStyleSheet("background-color: #1e1e1e;")
        top_control_layout = QVBoxLayout(control_widget) # Renamed to avoid conflict

        # --- Evolution Settings Group ---
        evolution_settings_group = QGroupBox("Evolution Settings")
        evolution_settings_group.setStyleSheet("QGroupBox { color: white; border: 1px solid gray; margin-top: 0.5em; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; }")
        evolution_layout = QVBoxLayout()

        instruction_label = QLabel("Configure parameters before starting. Settings are locked during auto-evolution.")
        instruction_label.setStyleSheet("color: #aaa; font-style: italic;")
        instruction_label.setWordWrap(True)
        evolution_layout.addWidget(instruction_label)

        # Helper function to create sliders with labels and descriptions
        def create_slider_with_labels(name, description_text, min_val, max_val, step_val, default_val, is_percentage=False):
            slider_layout = QVBoxLayout()
            
            name_label = QLabel(name)
            name_label.setStyleSheet("color: white;")
            slider_layout.addWidget(name_label)

            value_label = QLabel(f"{default_val / 100.0 if is_percentage else default_val}") # Displays current value
            value_label.setStyleSheet("color: #50fa7b; font-size: 8pt;") # Greenish color for value

            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setSingleStep(step_val)
            slider.setValue(default_val)
            slider.setStyleSheet("background-color: #2d2d2d;")
            
            def update_value_label(value):
                if is_percentage:
                    value_label.setText(f"{value / 100.0:.2f}")
                else:
                    value_label.setText(str(value))

            slider.valueChanged.connect(update_value_label)
            
            slider_layout.addWidget(slider)
            slider_layout.addWidget(value_label) # Add value label below slider

            description_label = QLabel(description_text)
            description_label.setStyleSheet("color: #ccc; font-size: 8pt;")
            description_label.setWordWrap(True)
            slider_layout.addWidget(description_label)
            
            evolution_layout.addLayout(slider_layout)
            return slider, value_label # Return slider and its value label

        # Population Size
        self.pop_slider, self.pop_value_label = create_slider_with_labels(
            "Population Size (`pop_size`):",
            "Number of neural networks (genomes) in each generation.",
            10, 100, 10, self.config_values['population_size']
        )

        # Fitness Threshold
        self.fitness_slider, self.fitness_value_label = create_slider_with_labels(
            "Fitness Threshold (`fitness_threshold`):",
            "Target fitness score; evolution stops if a genome reaches this.",
            80, 100, 1, int(self.config_values['fitness_threshold'] * 100), is_percentage=True
        )

        # Number of Hidden Nodes
        self.num_hidden_slider, self.num_hidden_value_label = create_slider_with_labels(
            "Number of Hidden Nodes (`num_hidden`):",
            "Initial number of hidden neurons in the network. NEAT can add more during evolution.",
            0, 10, 1, 0 # Min=0, Max=10, Step=1, Default=0
        )
        
        # Evaluation Trials per Network
        self.eval_trials_slider, self.eval_trials_value_label = create_slider_with_labels(
            "Evaluation Trials per Network:",
            "Number of random letters each network is tested on per generation to calculate its fitness.",
            1, 100, 1, 10 # Default to 10 trials
        )

        # Weight Mutate Rate
        self.weight_mutate_rate_slider, self.weight_mutate_rate_value_label = create_slider_with_labels(
            "Weight Mutate Rate (`weight_mutate_rate`):",
            "Probability that existing connection weights are perturbed.",
            0, 100, 1, 80, is_percentage=True # Default 0.8 (80%)
        )

        # Weight Replace Rate
        self.weight_replace_rate_slider, self.weight_replace_rate_value_label = create_slider_with_labels(
            "Weight Replace Rate (`weight_replace_rate`):",
            "Probability that a connection weight is replaced with a new random value.",
            0, 100, 1, 10, is_percentage=True # Default 0.1 (10%)
        )

        # Connection Add Probability
        self.conn_add_prob_slider, self.conn_add_prob_value_label = create_slider_with_labels(
            "Connection Add Probability (`conn_add_prob`):",
            "Probability of adding a new connection between neurons.",
            0, 100, 1, 50, is_percentage=True # Default 0.5 (50%)
        )

        # Node Add Probability
        self.node_add_prob_slider, self.node_add_prob_value_label = create_slider_with_labels(
            "Node Add Probability (`node_add_prob`):",
            "Probability of adding a new neuron (node) to the network.",
            0, 100, 1, 20, is_percentage=True # Default 0.2 (20%)
        )

        # Auto-Evolve Button
        self.auto_evolve_button = QPushButton("Start Auto-Evolve")
        self.auto_evolve_button.setStyleSheet("color: white; background-color: #007acc; padding: 5px;")
        self.auto_evolve_button.setCheckable(True) # To manage Start/Stop state
        self.auto_evolve_button.toggled.connect(self.on_auto_evolve_toggled)
        evolution_layout.addWidget(self.auto_evolve_button)

        evolution_settings_group.setLayout(evolution_layout)
        top_control_layout.addWidget(evolution_settings_group)
        # --- End of Evolution Settings Group ---

        # Randomize Letter Button (remains outside the group)
        self.randomize_button = QPushButton("Randomize New Letter")
        self.randomize_button.setStyleSheet("color: white; background-color: #555; padding: 5px;")
        self.randomize_button.clicked.connect(self.on_randomize_letter_clicked)
        top_control_layout.addWidget(self.randomize_button)
        
        top_control_layout.addStretch() # Pushes controls to the top

        self.main_layout.addWidget(control_widget)

    def on_auto_evolve_toggled(self, checked):
        """Handles Start/Stop Auto-Evolve button clicks and parameter locking."""
        self.evolution_sliders = [
            self.pop_slider, self.fitness_slider, self.num_hidden_slider, self.eval_trials_slider,
            self.weight_mutate_rate_slider, self.weight_replace_rate_slider,
            self.conn_add_prob_slider, self.node_add_prob_slider
        ]
        if checked:  # "Start Auto-Evolve" was clicked
            self.auto_evolve_button.setText("Stop Auto-Evolve")
            self.auto_evolve_button.setStyleSheet("color: white; background-color: #c70000; padding: 5px;")  # Red for stop

            # Disable sliders
            for slider in self.evolution_sliders:
                slider.setEnabled(False)

            current_params = self.get_current_neat_parameters()
            self.neat_logic.reconfigure_neat(current_params)  # Pass params to NEATLogic

            # Evolution loop
            while self.auto_evolve_button.isChecked():
                # This is the core logic previously in run_evolution(), minus timer parts
                evolution_finished = self.neat_logic.run_evolution_step()
                self.update_stats_display()

                # Update visualization with best genome from NEATLogic
                if self.neat_logic.p and self.neat_logic.p.best_genome:
                    current_eval_letter = self.neat_logic.generate_letter()
                    letter_pattern, _ = self.neat_logic.generate_letter_pattern(current_eval_letter)

                    if letter_pattern is not None:
                        predicted_letter, output_activations = self.neat_logic.classify_letter(self.neat_logic.p.best_genome, letter_pattern)
                        is_correct = (predicted_letter == current_eval_letter)
                        self.visualization.draw_network(
                            genome=self.neat_logic.p.best_genome,
                            actual_letter_pattern=letter_pattern,
                            actual_output_activations=output_activations,
                            actual_prediction=predicted_letter,
                            actual_letter=current_eval_letter,
                            is_correct=is_correct
                        )
                    else:
                         self.visualization.draw_network(
                            genome=self.neat_logic.p.best_genome,
                            actual_prediction="Error: No Pattern",
                            actual_letter=None
                        )
                else:
                    pass  # No best genome to visualize

                QApplication.processEvents()  # Allow UI to update and respond

                if evolution_finished:
                    if self.auto_evolve_button.isChecked():
                        self.auto_evolve_button.setChecked(False)
                    break  # Exit the evolution loop
        else:  # "Stop Auto-Evolve" was clicked or evolution finished
            self.auto_evolve_button.setText("Start Auto-Evolve")
            self.auto_evolve_button.setStyleSheet("color: white; background-color: #007acc; padding: 5px;")  # Blue for start

            # Enable sliders
            for slider in self.evolution_sliders:
                slider.setEnabled(True)
            # No timer to stop, the loop condition handles termination.

    def on_randomize_letter_clicked(self):
        """Randomize a new letter and redraw the network visualization."""
        # 1. Randomize a letter
        random_actual_letter = self.neat_logic.generate_letter()

        # 2. Rasterize it
        letter_pattern, _ = self.neat_logic.generate_letter_pattern(random_actual_letter)

        # Get the current best genome
        current_genome = None
        if self.neat_logic.p and self.neat_logic.p.best_genome:
             current_genome = self.neat_logic.p.best_genome
        elif self.neat_logic.p and self.neat_logic.p.population:
             # If no best genome yet, use the first one in the population
             current_genome = next(iter(self.neat_logic.p.population.values()), None)


        if current_genome and letter_pattern is not None:
            # 3. Get network output and classification
            predicted_letter, output_activations = self.neat_logic.classify_letter(current_genome, letter_pattern)

            # 4. Determine correctness
            is_correct = (predicted_letter == random_actual_letter)

            # 5. Update visualization
            self.visualization.draw_network(
                genome=current_genome,
                actual_letter_pattern=letter_pattern,
                actual_output_activations=output_activations,
                actual_prediction=predicted_letter,
                actual_letter=random_actual_letter,
                is_correct=is_correct
            )

            # 6. Update stats display (optional, as stats are generation-based, but good for consistency)
            self.update_stats_display()
        elif current_genome:
             # Handle case where letter_pattern could not be generated
             self.visualization.draw_network(genome=current_genome, actual_prediction="Error: No Pattern", actual_letter=None)
             self.update_stats_display()
        else:
             # Handle case: No genome found
             self.visualization.draw_network(genome=None, actual_prediction="Error: No Genome", actual_letter=None)
             self.update_stats_display()


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

        # Best fitness
        fitness_label = QLabel("Best Fitness:")
        fitness_label.setStyleSheet("color: white;")
        stats_layout.addWidget(fitness_label)

        self.fitness_label = QLabel("0.00")
        self.fitness_label.setStyleSheet("color: cyan; font: 10pt 'Consolas';")
        stats_layout.addWidget(self.fitness_label)

        # Average Population Fitness
        avg_fitness_title_label = QLabel("Average Population Fitness:")
        avg_fitness_title_label.setStyleSheet("color: white;")
        stats_layout.addWidget(avg_fitness_title_label)

        self.avg_fitness_label = QLabel("0.00")
        self.avg_fitness_label.setStyleSheet("color: cyan; font: 10pt 'Consolas';")
        stats_layout.addWidget(self.avg_fitness_label)

        # Total Evaluations (Last Gen)
        self.total_evals_title_label = QLabel("Total Evaluations (Last Gen):") # Renamed
        self.total_evals_title_label.setStyleSheet("color: white;")
        stats_layout.addWidget(self.total_evals_title_label)

        self.total_evals_label = QLabel("0") # Renamed from samples_label
        self.total_evals_label.setStyleSheet("color: cyan; font: 10pt 'Consolas';")
        stats_layout.addWidget(self.total_evals_label)

        self.main_layout.addWidget(stats_widget)

    def update_stats_display(self):
        """Update the stats labels based on NEAT logic"""
        self.gen_label.setText(str(self.neat_logic.generation))
        self.fitness_label.setText(f"{self.neat_logic.best_fitness:.2f}")
        
        # Update average population fitness
        avg_fitness = 0.0
        if self.neat_logic.stats_reporter:
            fitness_means = self.neat_logic.stats_reporter.get_fitness_mean()
            if fitness_means:
                avg_fitness = fitness_means[-1] # Get the last one
        self.avg_fitness_label.setText(f"{avg_fitness:.2f}")

        # Update Total Evaluations (Last Gen)
        # This requires knowing pop_size and eval_trials used for the *last completed* generation
        # For now, let's use the current slider values if available, or defaults from neat_logic
        pop_size = self.pop_slider.value()
        eval_trials = self.eval_trials_slider.value()
        if self.neat_logic.p and self.neat_logic.p.config: # If evolution has run, use actual config
            pop_size = self.neat_logic.p.config.pop_size
        # eval_trials is stored in neat_logic.num_evaluation_trials after reconfigure
        if hasattr(self.neat_logic, 'num_evaluation_trials') and self.neat_logic.num_evaluation_trials is not None:
             eval_trials = self.neat_logic.num_evaluation_trials

        total_evals = pop_size * eval_trials if self.neat_logic.generation > 0 else 0 # Only show if evolution has run
        self.total_evals_label.setText(str(total_evals))


    # Removed setup_neat as it's in neat_logic.py

    # Removed draw_network as it's in visualization.py

    def get_current_neat_parameters(self):
        """Reads current NEAT parameters from UI sliders and converts them."""
        params = {
            'population_size': self.pop_slider.value(),
            'fitness_threshold': self.fitness_slider.value() / 100.0,
            'num_hidden': self.num_hidden_slider.value(),
            'num_evaluation_trials': self.eval_trials_slider.value(),
            'weight_mutate_rate': self.weight_mutate_rate_slider.value() / 100.0,
            'weight_replace_rate': self.weight_replace_rate_slider.value() / 100.0,
            'conn_add_prob': self.conn_add_prob_slider.value() / 100.0,
            'node_add_prob': self.node_add_prob_slider.value() / 100.0,
        }
        return params

    # Removed evaluate_genome, generate_letter, _pixmap_to_matrix, generate_letter_pattern, classify_letter, evaluate
    # as they are in neat_logic.py or visualization.py

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NEATLetterClassifier()
    window.show()
    app.lastWindowClosed.connect(app.quit)
sys.exit(app.exec_())
