# NEAT Letter Classifier

## Project Overview
This application demonstrates the NeuroEvolution of Augmenting Topologies (NEAT) algorithm applied to a letter classification task. The application features a dynamic, interactive interface for controlling NEAT parameters, visualizing the evolving neural network in real-time, and observing its performance. Users can configure settings like population size, mutation rates, and evaluation trials before starting an automated evolution process.

## Technology Stack
- **Core Logic**:
    - **NEAT Algorithm**: Implemented using [neat-python](https://github.com/CodeReclaimers/neat-python) library
    - **Letter Generation**: Randomized font selection + random A/B/C generation
    - **Rasterization**: 16x16 pixel grid representation of letters (Note: `neat_logic.py` uses 16x16, `neat_config` num_inputs is 256).
    - **Classification**: NEAT network evaluates input patterns and outputs A/B/C predictions
- **Frontend**:
    - **Framework/Library**: PyQt5
    - **Language**: Python 3.10+
    - **UI Components**:
        - Three-pane layout with interactive controls.
        - Grouped "Evolution Settings" with sliders for Population Size, Fitness Threshold, Evaluation Trials per Network, and various Mutation Rates (Weight Mutate, Weight Replace, Connection Add, Node Add).
        - Real-time network visualization.
        - Live classification results.
    - **Styling**: Dark theme with VS Code-like aesthetics.
- **Visualization**:
    - **Network Graph**: Using NetworkX + Matplotlib for dynamic node/edge rendering.
    - **Pixel Grid**: 16x16 rasterized letter display.
    - **Activation Visualization**: Color intensity for neuron activations.

### Requirements
To run the application, you'll need:
```bash
pip install PyQt5 matplotlib neat-python networkx numpy
```

### Basic Usage
The PyQt version offers:
- Interactive configuration of NEAT parameters before starting evolution.
- Smoother animations for network visualization.
- More responsive UI controls.
- Better high-DPI display support.

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
- **Dark Theme**: Professional, VS Code-like appearance.
- **Data Correction**: Modal dialog for users to input and label training samples (if this feature is still active/relevant).

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

For more details on the NEAT algorithm concepts like population and generations, and the planned implementation changes, please refer to `about_neat.md` and `implementation.md` respectively.
