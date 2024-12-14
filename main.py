# main.py
from environment import NetworkSimEnvironment
from agents import DQNAgent, PPOAgent
from gui import NetworkVisualizerGUI

def main():
    # Create environment and controllers
    env = NetworkSimEnvironment()
    dqn_agent = DQNAgent(env)
    ppo_agent = PPOAgent(env)
    
    # Create and run GUI
    gui = NetworkVisualizerGUI(env, dqn_agent, ppo_agent)
    gui.run()

if __name__ == "__main__":
    main()