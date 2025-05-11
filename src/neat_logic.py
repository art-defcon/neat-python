import neat
import os
import random
import numpy as np

class NEATLogic:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None
        self.p = None
        self.generation = 0
        self.best_fitness = 0
        self.training_samples = 0 # This will be effectively population_size * num_evaluation_trials
        self.current_letter = None # May not be needed if evaluate handles its own letters
        self.stats_reporter = None
        self.num_evaluation_trials = 1 # Default, will be overridden by reconfigure_neat

    def setup_neat(self):
        """Initialize NEAT configuration and population with default settings."""
        # Load base configuration
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 self.config_path)
        # Create population
        self.p = neat.Population(self.config)
        
        # Add reporters
        self.p.add_reporter(neat.StdOutReporter(True)) # For console output
        self.stats_reporter = neat.StatisticsReporter()
        self.p.add_reporter(self.stats_reporter)
        
        self.generation = 0
        self.best_fitness = 0.0


    def reconfigure_neat(self, params):
        """Reconfigures NEAT based on parameters from the UI and resets evolution."""
        print(f"Reconfiguring NEAT with params: {params}") # Debug print
        # Load the base configuration again to ensure a clean slate
        new_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 self.config_path)

        # Override parameters from the params dictionary
        new_config.pop_size = params.get('population_size', new_config.pop_size)
        new_config.fitness_threshold = params.get('fitness_threshold', new_config.fitness_threshold)
        
        # Genome specific configurations
        new_config.genome_config.weight_mutate_rate = params.get('weight_mutate_rate', new_config.genome_config.weight_mutate_rate)
        new_config.genome_config.weight_replace_rate = params.get('weight_replace_rate', new_config.genome_config.weight_replace_rate)
        # Compatibility for older neat-python versions for conn_add_prob and node_add_prob
        if hasattr(new_config.genome_config, 'conn_add_prob'):
            new_config.genome_config.conn_add_prob = params.get('conn_add_prob', new_config.genome_config.conn_add_prob)
            new_config.genome_config.node_add_prob = params.get('node_add_prob', new_config.genome_config.node_add_prob)
        else: # For newer versions, these might be capabilities
            new_config.genome_config.compatibility_disjoint_coefficient = params.get('compatibility_disjoint_coefficient', new_config.genome_config.compatibility_disjoint_coefficient) # Example, adjust if needed
            # For new attributes like 'enabled_mutate_rate' or specific mutation types:
            # new_config.genome_config.enabled_mutate_rate = params.get('enabled_mutate_rate', new_config.genome_config.enabled_mutate_rate)
            # new_config.genome_config.node_add_prob = params.get('node_add_prob', new_config.genome_config.node_add_prob)
            # new_config.genome_config.conn_add_prob = params.get('conn_add_prob', new_config.genome_config.conn_add_prob)
            # The implementation plan uses conn_add_prob and node_add_prob directly on genome_config.
            # Let's assume they are direct attributes for now as per the plan.
            # If these cause errors, it means the config structure is different.
            # For now, sticking to the plan:
            new_config.genome_config.conn_add_prob = params.get('conn_add_prob', 0.5) # Default if not in config
            new_config.genome_config.node_add_prob = params.get('node_add_prob', 0.2) # Default if not in config


        self.config = new_config
        self.num_evaluation_trials = params.get('num_evaluation_trials', 10) # Default to 10 if not provided

        # Create a new population with the reconfigured settings
        self.p = neat.Population(self.config)
        
        # Add reporters to the new population
        self.p.add_reporter(neat.StdOutReporter(True))
        self.stats_reporter = neat.StatisticsReporter()
        self.p.add_reporter(self.stats_reporter)
        
        # Reset evolution state
        self.generation = 0
        self.best_fitness = 0.0
        self.p.best_genome = None
        print(f"NEAT reconfigured. Pop size: {self.config.pop_size}, Fitness threshold: {self.config.fitness_threshold}, Eval Trials: {self.num_evaluation_trials}")
        print(f"Mutation rates: WMR={self.config.genome_config.weight_mutate_rate}, WRR={self.config.genome_config.weight_replace_rate}, CAP={self.config.genome_config.conn_add_prob}, NAP={self.config.genome_config.node_add_prob}")


    def run_evolution_step(self):
        """Run one generation of evolution."""
        if not self.p:
            print("Population not initialized. Call setup_neat() or reconfigure_neat() first.")
            return True # Indicate finished if not set up

        # Run NEAT's evolution for one generation.
        # The `self.evaluate` function will be called by `p.run` for all genomes.
        winner = self.p.run(self.evaluate, 1) # Run for 1 generation

        self.generation = self.p.generation
        if self.p.best_genome:
            self.best_fitness = self.p.best_genome.fitness
        else:
            # This case should ideally not happen if population is not empty
            # and p.run completed.
            # Try to get best from stats if available
            if self.stats_reporter and self.stats_reporter.most_fit_genomes:
                self.best_fitness = self.stats_reporter.most_fit_genomes[-1].fitness
            else:
                self.best_fitness = 0.0


        # Update training samples based on the new logic
        self.training_samples = self.config.pop_size * self.num_evaluation_trials

        # Check if the best genome's fitness meets the threshold
        if self.best_fitness >= self.config.fitness_threshold:
            print(f"Solution found in generation {self.generation} with fitness {self.best_fitness:.4f}")
            return True  # Evolution finished

        if winner: # p.run can return the best genome if solution is found
             print(f"Winner found by p.run: {winner.fitness}")
             # self.p.best_genome should already be set by p.run
             return True


        return False # Evolution not finished

    # evaluate_genome is no longer directly called by run_evolution_step,
    # but might be useful for single evaluations (e.g. randomize button).
    # For now, the main evaluation path is self.evaluate.
    # Let's keep evaluate_genome for now.
    def evaluate_genome(self, genome, letter_to_eval=None):
        """Evaluate a single genome based on a specific letter or a random one."""
        if not genome: return 0.0

        if letter_to_eval is None:
            letter_to_eval = self.generate_letter()
        
        letter_pattern, actual_letter = self.generate_letter_pattern(letter_to_eval)

        if letter_pattern is None: return 0.0

        try:
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            input_data = letter_pattern.flatten().astype(float)

            if len(input_data) != self.config.genome_config.num_inputs:
                return 0.0

            output_activations = net.activate(input_data)
            predicted_idx = output_activations.index(max(output_activations))
            predicted_letter = ['A', 'B', 'C'][predicted_idx % 3]

            return 1.0 if predicted_letter == actual_letter else 0.0
        except Exception:
            return 0.0

    def generate_letter(self):
        """Generate random A/B/C pattern"""
        return random.choice(['A', 'B', 'C'])

    def _pixmap_to_matrix(self, pixmap, actual_letter):
        """Convert QPixmap to 16x16 binary matrix"""
        from PyQt5.QtGui import QImage # Import here to avoid circular dependency
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
        from PyQt5.QtGui import QFont, QFontDatabase, QPainter, QPixmap # Import here to avoid circular dependency
        from PyQt5.QtCore import Qt # Import here to avoid circular dependency

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
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            input_data = letter_pattern.flatten().astype(float)

            if len(input_data) != self.config.genome_config.num_inputs:
                return "?", []

            output_activations = net.activate(input_data)
            predicted_idx = output_activations.index(max(output_activations))
            return ['A', 'B', 'C'][predicted_idx % 3], output_activations
        except Exception as e:
            # print(f"Error classifying with genome {genome.key if genome else 'None'}: {e}")
            return "?", [] # Return empty list for activations on error

    def evaluate(self, genomes, config):
        """
        NEAT evaluation function.
        Evaluates each genome in the `genomes` list based on its performance
        over `self.num_evaluation_trials` random letter classifications.
        The fitness is the average success rate.
        """
        for genome_id, genome in genomes:
            successful_trials = 0.0
            if self.num_evaluation_trials <= 0:
                genome.fitness = 0.0
                continue

            for _ in range(self.num_evaluation_trials):
                current_challenge_letter = self.generate_letter()
                letter_pattern, actual_letter = self.generate_letter_pattern(current_challenge_letter)

                if letter_pattern is None:
                    # If pattern generation fails, this trial is not successful.
                    # Depending on desired behavior, could skip or count as failure.
                    # For now, let's assume it doesn't contribute to success.
                    continue

                input_data = letter_pattern.flatten().astype(float)
                # Ensure input data size matches network's expected input size
                if len(input_data) != config.genome_config.num_inputs:
                    # print(f"Warning: Input data size {len(input_data)} for genome {genome_id} does not match num_inputs {config.genome_config.num_inputs}")
                    continue # Skip this trial if input size is wrong

                try:
                    net = neat.nn.FeedForwardNetwork.create(genome, config)
                    output_activations = net.activate(input_data)
                    predicted_idx = output_activations.index(max(output_activations))
                    predicted_letter = ['A', 'B', 'C'][predicted_idx % 3] # Assuming 3 output classes

                    if predicted_letter == actual_letter:
                        successful_trials += 1.0
                except Exception as e:
                    # print(f"Error evaluating genome {genome_id} during trial: {e}")
                    # Count as a failed trial or skip, here we just don't increment success
                    pass
            
            genome.fitness = successful_trials / self.num_evaluation_trials
