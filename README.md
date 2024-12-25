# Python Network Simulation Environment

A sophisticated network simulation environment built with Python, featuring reinforcement learning agents, traffic classification, anomaly detection, and predictive maintenance capabilities.

## 🌟 Features

### Core Components
- **Network Environment**: Customizable network topology with dynamic packet routing
- **Multiple RL Agents**: DQN and PPO implementations for network optimization
- **Traffic Classification**: Real-time classification of network traffic (Gaming, Video, Web, etc.)
- **Anomaly Detection**: Detection and mitigation of network anomalies (DoS, Port Scanning, Data Exfiltration)
- **Predictive Maintenance**: Proactive network maintenance scheduling based on performance metrics

### Interactive GUI
- Real-time network visualization
- Performance metrics monitoring
- Traffic pattern analysis
- Anomaly detection status
- Maintenance scheduling interface

### Network Metrics
- Bandwidth utilization
- Latency monitoring
- Packet loss tracking
- Network throughput
- Custom performance thresholds

## 🚀 Getting Started

### Prerequisites

pip install gym numpy networkx torch matplotlib tkinter joblib


### Project Structure

network_sim/
├── environment.py
├── agents.py
├── gui.py
├── main.py
└── README.md

### Running the Application
Save all files in your project directory:
environment.py - Contains the network simulation environment
agents.py - Contains the AI agents (DQN, PPO, etc.)
gui.py - Contains the visualization interface
main.py - The entry point of the application

Run the application by executing main.py:
python main.py


## 🛠️ Components

### Network Environment
- Configurable number of nodes
- Dynamic packet generation and routing
- Realistic network conditions simulation
- Support for various traffic types

### RL Agents
- **DQN Agent**: Deep Q-Network implementation for routing optimization
- **PPO Agent**: Proximal Policy Optimization for network management

### Traffic Classification
- Real-time traffic pattern analysis
- Multiple traffic categories (Gaming, Video, Web, Other)
- Confidence-based classification

### Anomaly Detection
- Multiple anomaly types (DoS, Port Scan, Data Exfiltration)
- Automated mitigation strategies
- Real-time threat monitoring

### Predictive Maintenance
- Performance degradation monitoring
- Scheduled maintenance planning
- Maintenance impact analysis

## 📊 Visualization

The GUI provides comprehensive network monitoring capabilities:
- Network topology visualization
- Real-time performance metrics
- Traffic pattern analysis
- Anomaly detection status
- Maintenance scheduling interface

## Screenshots

![Ekran Görüntüsü (470)](https://github.com/user-attachments/assets/7a876f3e-8198-4738-930e-537fbf68bba4)
![Ekran Görüntüsü (469)](https://github.com/user-attachments/assets/7f45c7ff-a599-4188-ac84-9916eeca62e3)
![Ekran Görüntüsü (468)](https://github.com/user-attachments/assets/7ba4cc31-6f0d-4636-8e5e-bff91bb265bd)

  
## Note
-The simulation starts with 24 nodes by default. If you want to change this, modify the num_nodes parameter in main.py:
env = NetworkSimEnvironment(num_nodes=12)  # Change to desired number

