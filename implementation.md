# NEAT Parameter Control and UI Enhancement Plan

## 1. Overview of Changes

The goal of these changes is to provide users with more granular control over the NEAT algorithm's parameters, improve the usability of these controls, and ensure that the settings chosen by the user are effectively applied to the evolution process. This involves:
*   Adding new UI sliders for specific NEAT parameters.
*   Grouping related settings for clarity.
*   Implementing logic to lock settings during an "Auto-Evolve" run.
*   Ensuring the NEAT system re-initializes with user-defined parameters before each run.
*   Enhancing the statistics display.

## 2. UI Enhancements (Target File: `src/app.py`)

### 2.1. "Evolution Settings" Group

*   A `QGroupBox` titled "Evolution Settings" will be created in the left control pane.
*   It will contain all NEAT parameter sliders, their descriptions, and the Auto-Evolve button.
*   An instructional `QLabel` will be added within this group: "Configure parameters before starting. Settings are locked during auto-evolution."

### 2.2. Parameter Sliders and Descriptions

The following sliders will be implemented or updated within the "Evolution Settings" group. Each will have a `QLabel` for its name and another `QLabel` for its description.

*   **Population Size (`pop_size`):**
    *   Slider Range: 10-100, Step: 10.
    *   Description: "Number of neural networks (genomes) in each generation."

*   **Fitness Threshold (`fitness_threshold`):**
    *   Slider Range: 80-100 (representing 0.80 to 1.00).
    *   Description: "Target fitness score; evolution stops if a genome reaches this."

*   **Evaluation Trials per Network:**
    *   Slider Range: 1-100, Step: 1.
    *   Description: "Number of random letters each network is tested on per generation to calculate its fitness."

*   **Weight Mutate Rate (`weight_mutate_rate`):**
    *   Slider Range: 0-100 (representing 0.0 to 1.0).
    *   Description: "Probability that existing connection weights are perturbed."

*   **Weight Replace Rate (`weight_replace_rate`):**
    *   Slider Range: 0-100 (representing 0.0 to 1.0).
    *   Description: "Probability that a connection weight is replaced with a new random value."

*   **Connection Add Probability (`conn_add_prob`):**
    *   Slider Range: 0-100 (representing 0.0 to 1.0).
    *   Description: "Probability of adding a new connection between neurons."

*   **Node Add Probability (`node_add_prob`):**
    *   Slider Range: 0-100 (representing 0.0 to 1.0).
    *   Description: "Probability of adding a new neuron (node) to the network."

### 2.3. Auto-Evolve Button

*   The current "Auto-Evolve" `QCheckBox` will be replaced with a `QPushButton`.
*   Button Text: Will toggle between "Start Auto-Evolve" and "Stop Auto-Evolve".
*   This button will be located within the "Evolution Settings" group.

## 3. Functional Changes

### 3.1. Parameter Locking Logic (in `src/app.py`)

*   When "Start Auto-Evolve" is clicked:
    *   All sliders within the "Evolution Settings" group will be disabled (`setEnabled(False)`).
    *   The button text changes to "Stop Auto-Evolve".
    *   The auto-evolution process begins.
*   When "Stop Auto-Evolve" is clicked (or evolution finishes):
    *   All sliders in the group will be re-enabled (`setEnabled(True)`).
    *   The button text changes back to "Start Auto-Evolve".

### 3.2. Making Parameters Effective (in `src/app.py` and `src/neat_logic.py`)

*   **In `src/app.py`:**
    *   When "Start Auto-Evolve" is clicked (before `self.run_evolution()` is called for the new sequence):
        1.  Read current values from all parameter sliders (Population Size, Fitness Threshold, Evaluation Trials, all mutation sliders).
        2.  Convert UI slider values (e.g., 0-100) to the actual scales required by `neat-python` (e.g., 0.0-1.0 for probabilities, integers for counts).
        3.  Pass these processed values to a method in `NEATLogic` (e.g., `reconfigure_neat(params)` or pass to `setup_neat` if it's called anew).
*   **In `src/neat_logic.py`:**
    1.  `NEATLogic` will store the `num_evaluation_trials` value (e.g., as `self.num_evaluation_trials`).
    2.  The `setup_neat()` method (or a new reconfiguration method) will:
        *   Load the base configuration from `src/neat_config`.
        *   Override specific values in the loaded `self.config` object using the parameters passed from `app.py` *before* `self.p = neat.Population(self.config)` is called.
        *   Example overrides:
            ```python
            self.config.pop_size = params['population_size']
            self.config.fitness_threshold = params['fitness_threshold']
            self.config.genome_config.weight_mutate_rate = params['weight_mutate_rate']
            # ... and so on for other mutation params.
            ```
        *   The `self.num_evaluation_trials` will be stored locally in `NEATLogic` for use by the `evaluate` method.
        *   A new `neat.Population` (`self.p`) will be created with these updated settings, effectively restarting evolution from Generation 0 with the new parameters.

### 3.3. Modified `NEATLogic.evaluate()` Method

*   The `evaluate(self, genomes, config)` method in `NEATLogic` will be updated to implement the "Evaluation Trials per Network":
    ```python
    # In NEATLogic class
    # self.num_evaluation_trials will be set from app.py

    def evaluate(self, genomes, config):
        for genome_id, genome in genomes: # Outer loop: iterates through each network
            successful_trials = 0.0
            for _ in range(self.num_evaluation_trials): # Inner loop: ALL trials for CURRENT genome
                current_challenge_letter = self.generate_letter()
                letter_pattern, actual_letter = self.generate_letter_pattern(current_challenge_letter)
                if letter_pattern is None: continue
                input_data = letter_pattern.flatten().astype(float)
                if len(input_data) != config.genome_config.num_inputs: continue

                try:
                    net = neat.nn.FeedForwardNetwork.create(genome, config)
                    output_activations = net.activate(input_data)
                    predicted_idx = output_activations.index(max(output_activations))
                    predicted_letter = ['A', 'B', 'C'][predicted_idx % 3]
                    if predicted_letter == actual_letter:
                        successful_trials += 1.0
                except Exception:
                    pass # Error during activation
            
            # Calculate average fitness
            genome.fitness = successful_trials / self.num_evaluation_trials if self.num_evaluation_trials > 0 else 0.0
    ```

## 4. Stats Panel Updates (Target File: `src/app.py`)

*   In `create_stats_panel()`:
    *   Add a `QLabel` for "Average Population Fitness:" and a corresponding value `QLabel` (e.g., `self.avg_fitness_label`).
*   In `update_stats_display()`:
    *   Fetch the mean fitness of the latest generation from `self.neat_logic.stats_reporter.get_fitness_mean()[-1]` (if available).
    *   Update `self.avg_fitness_label` with this value, formatted to two decimal places.
    *   The "Training Samples" label will be repurposed or renamed. Suggestion: Rename to "Total Evaluations (Last Gen)" and display `population_size * evaluation_trials`.

## 5. Workflow Summary

1.  User adjusts sliders in the "Evolution Settings" group.
2.  User clicks "Start Auto-Evolve".
3.  `app.py` reads slider values and passes them to `NEATLogic`.
4.  `NEATLogic` re-initializes its NEAT `config` and creates a new `Population` (Generation 0).
5.  UI sliders are locked.
6.  Auto-evolution begins:
    *   For each generation:
        *   For each genome in the population:
            *   The genome is evaluated on `N` (Evaluation Trials) random letters.
            *   Its fitness is the average success rate over these trials.
        *   Statistics (best fitness, average fitness) are updated.
        *   A new generation is produced.
7.  Process continues until "Stop Auto-Evolve" is clicked or fitness threshold is met. Sliders unlock.
