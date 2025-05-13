# NEAT Letter Classifier

## What is NEAT?
NEAT (NeuroEvolution of Augmenting Topologies) is a genetic algorithm for evolving artificial neural networks. It starts with a simple network and complexifies it over generations by adding new neurons and connections through mutation, while also optimizing the weights of existing connections. This allows NEAT to discover novel and increasingly complex network structures.

## Project Overview
This application demonstrates the NeuroEvolution of Augmenting Topologies (NEAT) algorithm applied to a letter classification task. While NEAT might not be the the best (or even good;)  or efficient approach for my own curiosity to explore NEAT's capabilities and visualize its evolutionary process in a tangible way.

The application features interactive interface for controlling NEAT parameters, visualizing the evolving neural network in real-time, and observing its performance. Users can configure settings like population size, mutation rates, and evaluation trials before starting an automated evolution process.

## Screenshot
![Screenshot of the NEAT Letter Classifier application](https://github.com/art-defcon/neat-python/blob/main/public/screenshot.png?raw=true)

## Technology Stack and Libraries

This project utilizes the following technologies and libraries:

-   **Python 3.10+**: The primary programming language.
-   **PyQt5**: Used for building the graphical user interface.
-   **matplotlib**: Used for plotting and visualization, specifically for the fitness history graph and network visualization.
-   **networkx**: Used for representing and manipulating the neural network graph structure for visualization.
-   **numpy**: Used for numerical operations, particularly in handling the pixel grid representation of letters.
-   **NEAT Algorithm**: The core of this project is the NeuroEvolution of Augmenting Topologies (NEAT) algorithm, implemented using the excellent [neat-python](https://github.com/CodeReclaimers/neat-python) library by CodeReclaimers. **DISCLAIMER: I have modified neat-python slightly will publish diff or fork later**

### Installation and Running

To get started with the NEAT Letter Classifier, follow these steps:

**1. Clone the repository:**

```bash
git clone https://github.com/art-defcon/neat-python.git
cd neat-python
```

**2. Install the required libraries:**

```bash
pip install PyQt5 matplotlib neat-python networkx numpy
```

**3. Run the application:**

```bash
python src/app.py
```

### Basic Usage

Once the application is running, you can:

- Configure NEAT parameters using the sliders in the left pane.
- Click "Start Auto-Evolve" to begin the evolutionary process.
- Observe the evolving network topology and performance in the center and right panes.
- Click "Randomize New Letter" to test the current best network on a new letter (when not in auto-evolve mode).
- BUG: Closing app/window sometimes leaves "Auto-Evolve" running in background and you might need to kill process (for instance by running "pkill python")
- TODO: I dont really handle end state when it solves the task very graceful 

## Key Features
- **Interactive NEAT Parameters**:
    - Grouped "Evolution Settings" in the left pane.
    - Sliders for Population Size, Fitness Threshold, Evaluation Trials per Network.
    - Sliders for specific mutation rates: Weight Mutate, Weight Replace, Connection Add, Node Add.
    - Descriptive text for each parameter slider.
    - Settings are locked during auto-evolution.
- **Three-Pane Layout**:
    - **Left Pane ("Evolution Settings" & Controls)**:
        - NEAT parameter configuration group.
        - "Start/Stop Auto-Evolve" button for continuous evolution.
        - "Randomize New Letter" button (for manual stepping when not auto-evolving).
        - "Mock Data" toggle.
    - **Center Pane (Visualization Details)**:
        1.  **Input Neuron Layer (To the right of Rasterized Letter):** 256 input nodes.
        2.  **Network Topology - Hidden Layers & Connections (Center Area):** Dynamic area showing hidden neurons and connections. Connection weights visualized by line thickness/color; neuron activations by color intensity.
        3.  **Live Classification Result (Far Right):** Text display of the predicted letter.
    - **Right Pane (Stats & Info)**:
        - Display of Generation, Best Fitness (individual), Average Population Fitness.
        - Display of "Total Evaluations (Last Gen)".
        - Compact Fitness History graph.

## ASCII Wireframe Layout (Conceptual Update)
```
+-----------------------------------------------------------------+
| NEAT Letter Classifier (Dark Theme)                             |
+---------------------------+-----------------------+-------------+
| Evolution Settings & Ctrl | Network Visualization | Stats & Info|
|---------------------------|                       |-------------|
| [GroupBox: Evo Settings]  | +-----------------+   | Generation: |
|  Population Size: [Sl]    | | Input Neurons     |   | Best Fit:   |
|  Fitness Thresh: [Sl]     | | Input (16x16)   |   | Avg Pop Fit:|
|  Eval Trials: [Sl]        | +--------+--------+   | Tot Evals:  |
|  Weight Mutate: [Sl]      |          |           |             |
|  Weight Replace: [Sl]     |          v           | [Graph]     |
|  Conn Add Prob: [Sl]      | +-----------------+   | Fitness     |
|  Node Add Prob: [Sl]      | | Hidden Layers   |   | History     |
|  [Instructional Text]     | | (Dynamic Nodes)     |   |             |
|                           | +--------+--------+   |             |
| [Button:Start/Stop AutoEV]|          |           |             |
|---------------------------|          v           |             |
| [Button:Randomize Letter] | +-----------------+   |             |
|                           | | Prediction Text   |   |             |
|                           | +--------+--------+   |             |
|                           |                      |             |
+---------------------------+-----------------------+-------------+
```