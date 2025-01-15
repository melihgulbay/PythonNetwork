# main.py
import logging
import sys
from typing import Optional
import torch
import numpy as np
from environment import NetworkSimEnvironment
from agents import DQNAgent, PPOAgent, A3CAgent, REINFORCEAgent, HybridAgent
from gui1 import NetworkVisualizerGUI as GUI1
from intro_gui1 import IntroGUI
from gpu_utils import get_device_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_sim.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class NetworkSimulation:
    def __init__(self, num_nodes: int = 10, seed: Optional[int] = None):
        """
        Initialize the network simulation with specified parameters.
        
        Args:
            num_nodes: Number of nodes in the network
            seed: Random seed for reproducibility
        """
        self.logger = logging.getLogger(__name__)
        self.set_random_seed(seed)
        
        try:
            self.env = NetworkSimEnvironment(num_nodes=num_nodes)
            self.logger.info(f"Created environment with {num_nodes} nodes")
            
            # Initialize all agents
            self.dqn_agent = self.initialize_dqn()
            self.ppo_agent = self.initialize_ppo()
            self.a3c_agent = self.initialize_a3c()
            self.reinforce_agent = self.initialize_reinforce()
            self.hybrid_agent = self.initialize_hybrid()
            
            # Initialize GUI with all agents
            self.gui = GUI1(self.env, self.dqn_agent, self.ppo_agent, 
                          self.a3c_agent, self.reinforce_agent, self.hybrid_agent)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize simulation: {str(e)}")
            raise

    def set_random_seed(self, seed: Optional[int]) -> None:
        """Set random seed for reproducibility"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            self.logger.info(f"Set random seed to {seed}")

    def initialize_dqn(self) -> DQNAgent:
        """Initialize DQN agent with error handling"""
        try:
            agent = DQNAgent(self.env)
            self.logger.info("DQN agent initialized successfully")
            return agent
        except Exception as e:
            self.logger.error(f"Failed to initialize DQN agent: {str(e)}")
            raise

    def initialize_ppo(self) -> PPOAgent:
        """Initialize PPO agent with error handling"""
        try:
            agent = PPOAgent(self.env)
            self.logger.info("PPO agent initialized successfully")
            return agent
        except Exception as e:
            self.logger.error(f"Failed to initialize PPO agent: {str(e)}")
            raise

    def initialize_a3c(self) -> A3CAgent:
        """Initialize A3C agent with error handling"""
        try:
            agent = A3CAgent(self.env)
            self.logger.info("A3C agent initialized successfully")
            return agent
        except Exception as e:
            self.logger.error(f"Failed to initialize A3C agent: {str(e)}")
            raise

    def initialize_reinforce(self) -> REINFORCEAgent:
        """Initialize REINFORCE agent with error handling"""
        try:
            agent = REINFORCEAgent(self.env)
            self.logger.info("REINFORCE agent initialized successfully")
            return agent
        except Exception as e:
            self.logger.error(f"Failed to initialize REINFORCE agent: {str(e)}")
            raise

    def initialize_hybrid(self) -> HybridAgent:
        """Initialize Hybrid agent with error handling"""
        try:
            agent = HybridAgent(self.env)
            self.logger.info("Hybrid agent initialized successfully")
            return agent
        except Exception as e:
            self.logger.error(f"Failed to initialize Hybrid agent: {str(e)}")
            raise

    def run(self) -> None:
        """Run the simulation"""
        try:
            self.logger.info("Starting simulation...")
            self.gui.run()
        except Exception as e:
            self.logger.error(f"Error during simulation: {str(e)}")
            raise
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            # Add cleanup code here (e.g., saving models, closing connections)
            self.logger.info("Cleaning up resources...")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Main entry point with configuration options"""
    try:
        # Get GPU information
        gpu_info = get_device_info()
        
        # Configuration parameters
        CONFIG = {
            'num_nodes': 20,
            'seed': None,  # Set to None for random behavior
            'gpu_enabled': gpu_info['available'],
            'gpu_type': gpu_info['type']
        }
        
        # Log system information
        logging.info(f"Starting simulation with config: {CONFIG}")
        logging.info(f"GPU info: {gpu_info['type']} (Available: {gpu_info['available']})")
        
        # Show intro GUI
        intro = IntroGUI(CONFIG['num_nodes'], CONFIG['seed'])
        intro.run()
        
        # Create and run simulation
        sim = NetworkSimulation(
            num_nodes=CONFIG['num_nodes'],
            seed=CONFIG['seed']
        )
        sim.run()
        
    except KeyboardInterrupt:
        logging.info("Simulation interrupted by user")
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise
    finally:
        logging.info("Simulation ended")

if __name__ == "__main__":
    main()

