# Network Simulation with Reinforcement Learning

This project implements a network simulation environment using reinforcement learning techniques, specifically Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO). The simulation allows for the visualization of network metrics and the application of RL algorithms to optimize network performance.

## Features

- **Network Simulation Environment**: A customizable environment that simulates a network with nodes and edges, allowing for various routing decisions.
- **Reinforcement Learning Agents**: Implementations of DQN and PPO agents that learn to optimize network performance based on defined metrics.
- **Graphical User Interface (GUI)**: A user-friendly interface built with Tkinter that visualizes network statistics and allows users to toggle between different RL strategies.
- **Real-time Updates**: The GUI updates network statistics and visualizations in real-time, providing immediate feedback on the performance of the RL agents.

## Installation

To run this project, ensure you have Python 3.x installed along with the required libraries. You can install the necessary packages using pip:

bash
pip install numpy matplotlib networkx gym torch


## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/network-simulation.git
   cd network-simulation
   ```

2. Run the main script to start the simulation:
   ```bash
   python main.py
   ```

3. Use the GUI to enable or disable reinforcement learning, and observe how the network metrics change in response to the agents' actions.

## Code Structure

- `main.py`: The entry point of the application that initializes the environment and agents, and starts the GUI.
- `environment.py`: Contains the `NetworkSimEnvironment` class that defines the network simulation environment.
- `agents.py`: Implements the DQN and PPO agents used for reinforcement learning.
- `gui.py`: Contains the `NetworkVisualizerGUI` class that creates the graphical interface for the simulation.


