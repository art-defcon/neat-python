import neat
import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont # Import Pillow for image generation

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
        self.letter_options = ['A', 'B', 'C', 'D', 'E', 'F'] #, 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'] # Define possible letters
        self.image_size = (16, 16) # Define image size for letter patterns

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

        # WORKAROUND: Manually add input nodes to initial genomes if they are missing
        if self.p and self.p.population:
            for genome in self.p.population.values():
                for input_key in self.config.genome_config.input_keys:
                    if input_key not in genome.nodes:
                        # Create and initialize the node gene properly
                        node_gene = self.config.genome_config.node_gene_type(input_key)
                        node_gene.init_attributes(self.config.genome_config) # Initialize attributes like bias, response, etc.
                        genome.nodes[input_key] = node_gene


    def reconfigure_neat(self, params):
        """Reconfigures NEAT based on parameters from the UI and resets evolution."""
        # Load the base configuration again to ensure a clean slate
        new_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                 neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                 self.config_path)

        # Override parameters from the params dictionary
        new_config.pop_size = params.get('population_size', new_config.pop_size)
        new_config.fitness_threshold = params.get('fitness_threshold', new_config.fitness_threshold)
        new_config.genome_config.num_hidden = params.get('num_hidden', new_config.genome_config.num_hidden)
        
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

        # WORKAROUND: Manually add input nodes to initial genomes if they are missing
        if self.p and self.p.population:
            for genome in self.p.population.values():
                for input_key in self.config.genome_config.input_keys:
                    if input_key not in genome.nodes:
                        # Create and initialize the node gene properly
                        node_gene = self.config.genome_config.node_gene_type(input_key)
                        node_gene.init_attributes(self.config.genome_config) # Initialize attributes like bias, response, etc.
                        genome.nodes[input_key] = node_gene


    def run_evolution_step(self):
        """Run one generation of evolution."""
        if not self.p:
            return True # Indicate finished if not set up
        
        # Run NEAT's evolution for one generation.
        # The `self.evaluate` function will be called by `p.run` for all genomes.
        try:
            winner = self.p.run(self.evaluate, 1) # Run for 1 generation
        except neat.CompleteExtinctionException:
            print("Complete extinction exception encountered. Stopping auto-evolve.")
            return False # Indicate stopping evolution
            
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
        if self.p.best_genome and self.p.best_genome.fitness is not None and self.p.best_genome.fitness >= self.config.fitness_threshold:
            self.best_fitness = self.p.best_genome.fitness # Ensure self.best_fitness is up-to-date
            return True  # Evolution finished

        # If p.run returned a 'winner' (best of the generation), it doesn't mean the overall evolution is done
        # unless the fitness threshold is met (checked above).
        # The 'winner' variable from p.run(..., 1) is just the best genome of that single generation.
        # So, we don't return True based on 'winner' alone here.

        return False # Evolution not finished for the auto-evolve sequence

    def generate_letter(self):
        """Generates a random letter from the predefined options."""
        return random.choice(self.letter_options)

    def generate_letter_pattern(self, letter):
        """Generates a simple pixel pattern for a given letter."""
        try:
            img = Image.new('L', self.image_size, color=0) # 'L' for grayscale, 0 for black background
            d = ImageDraw.Draw(img)
            
            # Try to load a font. Use a default if 'arial.ttf' is not found.
            try:
                # Adjust font path as needed for your system
                font_path = "/Library/Fonts/Arial.ttf" if os.path.exists("/Library/Fonts/Arial.ttf") else None
                if font_path:
                    font = ImageFont.truetype(font_path, int(self.image_size[1] * 0.7)) # Adjust size
                else:
                    font = ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()

            # Calculate text size and position to center it
            text_bbox = d.textbbox((0, 0), letter, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            x = (self.image_size[0] - text_width) / 2
            y = (self.image_size[1] - text_height) / 2 - text_bbox[1] # Adjust for baseline

            d.text((x, y), letter, fill=255, font=font) # 255 for white letter

            # Convert image to numpy array (normalize to 0-1)
            pattern = np.array(img).astype(float) / 255.0

            # Crop the pattern: remove first 4 rows, last 4 rows, first 5 columns, last 5 columns
            # Original size: 16x16
            # Rows to keep: 4 to 12 (inclusive)
            # Columns to keep: 4 to 12 (inclusive)
            
            cropped_pattern = pattern[4:12, 3:13]
            return cropped_pattern, letter # Return the cropped pattern and the actual letter
            
            #return pattern, letter # Return the pattern and the actual letter
        
        except Exception as e:
            print(f"Error generating letter pattern for '{letter}': {e}")
            return None, letter # Return None pattern on error

    def classify_letter(self, genome, letter_pattern):
        """Classifies a letter pattern using a given genome."""
        if not genome or letter_pattern is None:
            return "Error", [0.0] * len(self.letter_options) # Return default error

        try:
            net = neat.nn.FeedForwardNetwork.create(genome, self.config)
            input_data = letter_pattern.flatten().astype(float)

            if len(input_data) != self.config.genome_config.num_inputs:
                 print(f"Input data size mismatch: Expected {self.config.genome_config.num_inputs}, got {len(input_data)}")
                 return "Error", [0.0] * len(self.letter_options)

            output_activations = net.activate(input_data)
            
            # Ensure output_activations has the expected number of elements
            if len(output_activations) != len(self.letter_options):
                 print(f"Output activations size mismatch: Expected {len(self.letter_options)}, got {len(output_activations)}")
                 # Pad or truncate output_activations to match expected size
                 if len(output_activations) < len(self.letter_options):
                     output_activations.extend([0.0] * (len(self.letter_options) - len(output_activations)))
                 elif len(output_activations) > len(self.letter_options):
                     output_activations = output_activations[:len(self.letter_options)]


            predicted_idx = output_activations.index(max(output_activations))
            predicted_letter = self.letter_options[predicted_idx]

            return predicted_letter, output_activations
        except Exception as e:
            print(f"Error classifying letter: {e}")
            return "Error", [0.0] * len(self.letter_options) # Return default error and zero activations

    def evaluate(self, genomes, config):
        """Fitness function for NEAT evolution."""
        for genome_id, genome in genomes:
            successful_trials = 0
            for _ in range(self.num_evaluation_trials):
                letter_to_eval = self.generate_letter()
                letter_pattern, actual_letter = self.generate_letter_pattern(letter_to_eval)

                if letter_pattern is None:
                    continue # Skip if pattern generation failed

                predicted_letter, _ = self.classify_letter(genome, letter_pattern)

                if predicted_letter == actual_letter:
                    successful_trials += 1

            # Fitness is the proportion of successful trials
            genome.fitness = successful_trials / self.num_evaluation_trials if self.num_evaluation_trials > 0 else 0.0

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
            predicted_letter = self.letter_options[predicted_idx] # Use self.letter_options

            if predicted_letter == actual_letter:
                return 1.0 # Return 1.0 for a successful single trial
            else:
                return 0.0 # Return 0.0 for a failed single trial

        except Exception as e: # Corrected indentation
            print(f"Error in evaluate_genome: {e}")
            return 0.0 # Return 0.0 on error
