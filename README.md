NEAT Letter Classifier

## Project Overview
This application demonstrates the NeuroEvolution of Augmenting Topologies (NEAT) algorithm applied to a letter classification task. The application features a dynamic, interactive interface for controlling NEAT parameters, visualizing the evolving neural network in real-time, and observing its performance.

## Technology Stack
- **Core Logic**:
    - **NEAT Algorithm**: Implemented using [neat-python](https://github.com/CodeReclaimers/neat-python) library
    - **Letter Generation**: Randomized font selection + random A/B/C generation
    - **Rasterization**: 8x6 pixel grid representation of letters
    - **Classification**: NEAT network evaluates input patterns and outputs A/B/C predictions
- **Frontend**:
    - **Framework/Library**: Tkinter (Python's standard GUI library)
    - **Language**: Python 3.10+
    - **UI Components**: 
        - Three-pane layout with interactive controls
        - Sliders for Mutation Rate, Population Size, Fitness Threshold
        - Real-time network visualization
        - Live classification results
    - **Styling**: Dark theme with VS Code-like aesthetics
- **Visualization**:
    - **Network Graph**: Using NetworkX + Matplotlib for dynamic node/edge rendering
    - **Pixel Grid**: 8x6 rasterized letter display
    - **Activation Visualization**: Color intensity for neuron activations

## Key Features
- **Interactive NEAT Parameters**: Sliders and inputs for Mutation Rate, Population Size, and Fitness Threshold.
- **Three-Pane Layout**:
    - **Left Pane**: NEAT parameter controls.
    - **Center Pane Visualization Details**:
        The center pane provides a left-to-right visualization of the network's processing flow:
        1.  **Rasterized Letter Input (Far Left):**
            *   Displays the current input pattern (e.g., a simplified 'A' 'B or 'C') as an 8x6 grid of pixels. Active pixels are colored to form the letter's shape.
        2.  **Input Neuron Layer (To the right of Rasterized Letter):**
            *   The first layer of the neural network graph, consisting of 48 nodes, one for each pixel in the input grid.
            *   These nodes visually represent the input neurons, and their activation (corresponding to pixel values) can be indicated by color intensity.
        3.  **Network Topology - Hidden Layers & Connections (Center Area):**
            *   The main dynamic area showing all evolved hidden neurons and the connections between all neuron types (input-hidden, hidden-hidden, hidden-output, input-output).
            *   **Connection Weights:** Visualized by line thickness (strength) and color (positive/negative).
            *   **Neuron Activations:** Hidden neuron activation levels are shown by color intensity.
        4.  **Output Neuron Layer (To the right of Network Topology):**
            *   The final layer of the graph, with 3 nodes for the A, B, C classification task.
            *   Output neuron activations are clearly visualized (e.g., by color intensity or size) to indicate the network's decision-making process.
        5.  **Live Classification Result (Far Right):**
            *   A text display showing the final predicted letter (e.g., "Predicted: A"), based on the output neuron with the highest activation.
        This entire visualization updates in real-time as the network evolves and processes new inputs.
    - **Right Pane**:
        - "Auto-Evolute" toggle switch for continuous evolution and classification.
        - Display of Generation, Best Fitness, and Training Samples.
        - Compact Fitness History graph.
- **Dark Theme**: Professional, VS Code-like appearance.
- **Data Correction**: Modal dialog for users to input and label training samples.

## ASCII Wireframe Layout
```
+---------------------------------------------------+
| NEAT Letter Classifier (Dark Theme)              |
+-----------+-----------------------+---------------+
| Controls  | Network Visualization | Stats & Info  |
|           |                       |               |
| [Slider]  | +-----------------+   | [Toggle]      |
| Mutation  | | Rasterized      |   | Generation:   |
| Rate      | | Input Grid      |   | Best Fitness: |
|           | +--------+--------+   | Training Samples: |
| [Slider]  |          |           |               |
| Population|          v           | [Graph]       |
| Size      | +-----------------+   |               |
|           | | Input Neurons   |   |               |
| [Slider]  | | (48 Nodes)      |   |               |
| Fitness   | +--------+--------+   |               |
| Threshold |          |           |               |
|           |          v           |               |
|           | +-----------------+   |               |
|           | | Hidden Layers   |   |               |
|           | | (Dynamic Nodes) |   |               |
|           | +--------+--------+   |               |
|           |          |           |               |
|           |          v           |               |
|           | +-----------------+   |               |
|           | | Output Neurons  |   |               |
|           | | (A/B/C - 3 Nodes)|  |               |
|           | +--------+--------+   |               |
|           |          |           |               |
|           |          v           |               |
|           | +-----------------+   |               |
