import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                            QLabel, QSlider, QCheckBox, QPushButton)
from PyQt5.QtCore import Qt, QTimer
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

        # Configuration (initial values, will be updated by NEATLogic)
        self.config_values = {
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

        self.mock_data_enabled = False # Initialize state variable - Changed to False
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

        # Network visualization setup (This part is now handled by NEATVisualization)
        # self.fig, self.ax = plt.subplots(figsize=(6, 4))
        # self.canvas = FigureCanvasQTAgg(self.fig)
        # self.viz_layout.addWidget(self.canvas)
        # Removed plt.ion() as it might cause an extra window

        # Initial draw based on mock data state
        self.on_mock_data_toggled(Qt.Checked if self.mock_data_enabled else Qt.Unchecked)

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
        self.mutation_slider.setValue(int(self.config_values['mutation_rate'] * 100))
        self.mutation_slider.setStyleSheet("background-color: #2d2d2d;")
        control_layout.addWidget(self.mutation_slider)

        pop_label = QLabel("Population Size:")
        pop_label.setStyleSheet("color: white;")
        control_layout.addWidget(pop_label)

        self.pop_slider = QSlider(Qt.Horizontal)
        self.pop_slider.setRange(50, 300)
        self.pop_slider.setSingleStep(10)
        self.pop_slider.setValue(self.config_values['population_size'])
        self.pop_slider.setStyleSheet("background-color: #2d2d2d;")
        control_layout.addWidget(self.pop_slider)

        fitness_label = QLabel("Fitness Threshold:")
        fitness_label.setStyleSheet("color: white;")
        control_layout.addWidget(fitness_label)

        self.fitness_slider = QSlider(Qt.Horizontal)
        self.fitness_slider.setRange(80, 99)
        self.fitness_slider.setValue(int(self.config_values['fitness_threshold'] * 100))
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
        self.mock_data_checkbox.setChecked(False) # Default to False - Changed
        self.mock_data_checkbox.stateChanged.connect(self.on_mock_data_toggled)
        control_layout.addWidget(self.mock_data_checkbox)

        self.main_layout.addWidget(control_widget)

    def on_mock_data_toggled(self, state):
        self.mock_data_enabled = (state == Qt.Checked)
        if self.mock_data_enabled:
            self.auto_evolve.setChecked(False)
            self.auto_evolve.setEnabled(False)
            # If a timer for evolution is running, stop it (more robust timer management might be needed)
            if hasattr(self, 'evolution_timer') and self.evolution_timer:
                self.evolution_timer.stop()
            self.visualization.draw_mock_network_wrapper() # Redraw with mock data
            self.update_stats_display(is_mock=True) # Update stats for mock data
        else:
            self.auto_evolve.setEnabled(True)
            
            # --- New Startup Sequence Logic ---
            initial_genome = None
            if self.neat_logic.p and self.neat_logic.p.population:
                # Get the first genome from the initialized population as the "starter network"
                initial_genome = next(iter(self.neat_logic.p.population.values()), None)

            if initial_genome:
                # 1. Randomize a letter
                random_actual_letter = self.neat_logic.generate_letter()

                # 2. Rasterize it
                letter_pattern, _ = self.neat_logic.generate_letter_pattern(random_actual_letter)

                if letter_pattern is not None:
                    # 3. Get network output and classification (using modified neat_logic.classify_letter)
                    # classify_letter now returns (predicted_char, activations_list)
                    predicted_letter, output_activations = self.neat_logic.classify_letter(initial_genome, letter_pattern)
                    
                    # 4. Determine correctness
                    is_correct = (predicted_letter == random_actual_letter)

                    # 5. Update visualization (using modified draw_network)
                    self.visualization.draw_network(
                        genome=initial_genome,
                        is_mock=False,
                        actual_letter_pattern=letter_pattern,
                        actual_output_activations=output_activations,
                        actual_prediction=predicted_letter,
                        actual_letter=random_actual_letter, # Pass the actual letter
                        is_correct=is_correct
                    )
                else:
                    # Handle case where letter_pattern could not be generated
                    self.visualization.draw_network(genome=initial_genome, is_mock=False, actual_prediction="Error: No Pattern", actual_letter=None) # Pass None for actual letter

                self.update_stats_display() # Update stats for real data
            else:
                # Handle case: No initial genome found
                self.visualization.draw_network(genome=None, is_mock=False, actual_prediction="Error: No Genome", actual_letter=None) # Pass None for actual letter
                self.update_stats_display(is_mock=True) # Show mock stats if no network

            # Potentially restart evolution if auto_evolve was checked (existing logic)
            if self.auto_evolve.isChecked():
                self.run_evolution()
            # --- End of New Startup Sequence Logic ---

    # Removed draw_mock_network_wrapper and _get_mock_visualization_data as they are in visualization.py

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
        self.fig_fitness, self.ax_fitness = plt.subplots(figsize=(1.5, 1.5))
        self.line_fitness, = self.ax_fitness.plot([], [], 'g-')
        self.ax_fitness.set_xlim(0, 100)
        self.ax_fitness.set_ylim(0, 1)
        self.ax_fitness.set_title("Fitness History")
        self.ax_fitness.set_facecolor('#1e1e1e') # Match background
        self.ax_fitness.tick_params(axis='both', which='both', colors='white') # White ticks
        self.ax_fitness.xaxis.label.set_color('white') # White labels
        self.ax_fitness.yaxis.label.set_color('white')
        self.ax_fitness.title.set_color('white') # White title


        canvas = FigureCanvasQTAgg(self.fig_fitness)
        stats_layout.addWidget(canvas)
        plt.close(self.fig_fitness) # Close the figure to prevent it from opening in a separate window

        self.main_layout.addWidget(stats_widget)

    def update_stats_display(self, is_mock=False):
        """Update the stats labels based on NEAT logic or mock data"""
        if is_mock:
            self.gen_label.setText("Mock")
            self.fitness_label.setText("N/A")
            self.samples_label.setText("N/A")
            # Clear fitness history graph for mock data
            self.line_fitness.set_data([], [])
            self.ax_fitness.set_xlim(0, 100)
            self.ax_fitness.set_ylim(0, 1)
            self.fig_fitness.canvas.draw()
        else:
            self.gen_label.setText(str(self.neat_logic.generation))
            self.fitness_label.setText(f"{self.neat_logic.best_fitness:.2f}")
            self.samples_label.setText(str(self.neat_logic.training_samples)) # Assuming neat_logic tracks this

            # Update fitness history graph
            # Need to get fitness history data from neat_logic
            # Access the stored StatisticsReporter
            stats_reporter = self.neat_logic.stats_reporter

            if stats_reporter and hasattr(stats_reporter, 'num_generations') and hasattr(stats_reporter, 'most_fit_genomes'):
                 generations = range(stats_reporter.num_generations)
                 best_fitness_history = [c.fitness for c in stats_reporter.most_fit_genomes]

                 if generations and best_fitness_history:
                     self.line_fitness.set_data(generations, best_fitness_history)
                     self.ax_fitness.set_xlim(0, max(100, len(generations))) # Adjust x-limit
                     self.ax_fitness.set_ylim(0, max(1, max(best_fitness_history) * 1.1)) # Adjust y-limit
                     self.fig_fitness.canvas.draw()


    # Removed setup_neat as it's in neat_logic.py

    # Removed draw_network as it's in visualization.py

    def run_evolution(self):
        """Main evolution loop"""
        if self.mock_data_enabled:
            # If mock data is enabled, ensure no evolution timer runs and UI reflects this
            if hasattr(self, 'evolution_timer') and self.evolution_timer:
                self.evolution_timer.stop()
            self.auto_evolve.setChecked(False)
            self.auto_evolve.setEnabled(False)
            return

        # Run one step of evolution using NEATLogic
        evolution_finished = self.neat_logic.run_evolution_step()

        # Update UI with new stats
        self.update_stats_display()

        # Update visualization with best genome from NEATLogic
        if self.neat_logic.p and self.neat_logic.p.best_genome:
            # Generate a letter for this evolution step
            current_eval_letter = self.neat_logic.generate_letter() # Use a new variable to avoid confusion with self.neat_logic.current_letter used in eval
            letter_pattern, _ = self.neat_logic.generate_letter_pattern(current_eval_letter)

            if letter_pattern is not None:
                # Get activations and prediction using the best_genome
                # classify_letter now returns (predicted_char, activations_list)
                predicted_letter, output_activations = self.neat_logic.classify_letter(self.neat_logic.p.best_genome, letter_pattern)
                
                is_correct = (predicted_letter == current_eval_letter)

                self.visualization.draw_network(
                    genome=self.neat_logic.p.best_genome,
                    is_mock=False,
                    actual_letter_pattern=letter_pattern,
                    actual_output_activations=output_activations,
                    actual_prediction=predicted_letter,
                    actual_letter=current_eval_letter, # Pass the actual letter
                    is_correct=is_correct
                )
            else:
                # Handle error if letter_pattern is None
                 self.visualization.draw_network(
                    genome=self.neat_logic.p.best_genome,
                    is_mock=False,
                    actual_prediction="Error: No Pattern",
                    actual_letter=None # Pass None for actual letter
                )


        # Check if evolution finished
        if evolution_finished:
            print("Evolution finished successfully!")
            if hasattr(self, 'evolution_timer') and self.evolution_timer:
                self.evolution_timer.stop()
            # Optionally, display a message or take further action

        # Auto-evolve if enabled (2000ms interval) and not finished
        if self.auto_evolve.isChecked() and not self.mock_data_enabled and not evolution_finished:
            if hasattr(self, 'evolution_timer') and self.evolution_timer: # Clear existing timer
                self.evolution_timer.stop()
            self.evolution_timer = QTimer()
            self.evolution_timer.setSingleShot(True)
            self.evolution_timer.timeout.connect(self.run_evolution)
            self.evolution_timer.start(2000)

    # Removed evaluate_genome, generate_letter, _pixmap_to_matrix, generate_letter_pattern, classify_letter, evaluate
    # as they are in neat_logic.py or visualization.py

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = NEATLetterClassifier()
    window.show()
    app.lastWindowClosed.connect(app.quit)
sys.exit(app.exec_())
