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
        self.training_samples = 0
        self.current_letter = None
        self.stats_reporter = None # Add attribute to store StatisticsReporter

    def setup_neat(self):
        """Initialize NEAT configuration and population"""
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 self.config_path)

        # Create population
        self.p = neat.Population(self.config)
        self.p.add_reporter(neat.StdOutReporter(True))
        self.stats_reporter = neat.StatisticsReporter() # Create and store instance
        self.p.add_reporter(self.stats_reporter) # Add the stored instance

    def run_evolution_step(self):
        """Run one step of the evolution"""
        if not hasattr(self, 'generation'):
            self.generation = 0
        self.generation += 1

        # Store current letter for this generation
        self.current_letter = self.generate_letter()

        # Evaluate genomes and find best
        for genome_id, genome in self.p.population.items():
            genome.fitness = self.evaluate_genome(genome)

        # Set best genome
        self.p.best_genome = max(self.p.population.values(), key=lambda g: g.fitness)
        self.best_fitness = self.p.best_genome.fitness

        # Check fitness threshold
        if self.best_fitness >= self.config.fitness_threshold:
            return True # Evolution finished

        # Evolve
        self.p.run(self.evaluate, 1)

        return False # Evolution not finished

    def evaluate_genome(self, genome):
        """Evaluate a single genome based on the current_letter"""
        if not genome: return 0.0

        # Get the current letter and its pattern
        # self.current_letter should be set by run_evolution_step before this is called
        letter_pattern, actual_letter = self.generate_letter_pattern(self.current_letter)

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
        """NEAT evaluation function - evaluates each genome against the current letter"""
        # self.current_letter is set once per generation in run_evolution_step
        # For simplicity, all genomes in this generation are evaluated against this single current_letter
        letter_pattern, actual_letter = self.generate_letter_pattern(self.current_letter)

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
