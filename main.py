# main.py
from environment import NetworkSimEnvironment
from agents import (DQNAgent, PPOAgent, TrafficClassifierAgent, 
                   AnomalyDetectorAgent, PredictiveMaintenanceAgent)
from gui import NetworkVisualizerGUI

def main():
    # Create environment and controllers with more nodes
    env = NetworkSimEnvironment(num_nodes=24)  
    dqn_agent = DQNAgent(env)
    ppo_agent = PPOAgent(env)
    
    # Create and run GUI
    gui = NetworkVisualizerGUI(env, dqn_agent, ppo_agent)
    gui.run()

if __name__ == "__main__":
    main()