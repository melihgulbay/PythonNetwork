# environment.py
import gym
from gym import spaces
import numpy as np
import networkx as nx
import random

class NetworkSimEnvironment(gym.Env):
    def __init__(self, num_nodes=6):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(num_nodes * (num_nodes - 1))  # Possible routing decisions
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_nodes * num_nodes + 4,), dtype=np.float32
        )
        
        # Initialize network topology
        self.network = nx.complete_graph(num_nodes)
        self.reset()
        
        # Add performance thresholds
        self.max_latency = 20.0
        self.max_packet_loss = 0.3
        self.min_throughput = 0.2
        
    def reset(self):
        # Reset network state
        edges = list(self.network.edges())
        # Initialize metrics for both directions of each edge
        self.bandwidth_utilization = {}
        self.latency = {}
        self.packet_loss = {}
        self.throughput = {}
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    # Initialize metrics for both (i,j) and (j,i)
                    if (i, j) in edges or (j, i) in edges:
                        # Start with poor network conditions
                        self.bandwidth_utilization[(i, j)] = random.uniform(0.7, 0.9)  # High utilization
                        self.latency[(i, j)] = random.uniform(15, 19)  # High latency
                        self.packet_loss[(i, j)] = random.uniform(0.2, 0.25)  # High packet loss
                        self.throughput[(i, j)] = random.uniform(0.2, 0.4)  # Low throughput
        
        return self._get_state()
    
    def _get_state(self):
        # Flatten network metrics into state vector
        state = []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    # Check both (i,j) and (j,i) since edges can be stored in either direction
                    if (i, j) in self.bandwidth_utilization:
                        state.append(self.bandwidth_utilization[(i, j)])
                    elif (j, i) in self.bandwidth_utilization:
                        state.append(self.bandwidth_utilization[(j, i)])
                    else:
                        state.append(0)
                else:
                    state.append(0)
                    
        # Add global metrics
        state.extend([
            np.mean(list(self.bandwidth_utilization.values())),
            np.mean(list(self.latency.values())),
            np.mean(list(self.packet_loss.values())),
            np.mean(list(self.throughput.values()))
        ])
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        # Convert action to source-destination pair
        src = action // (self.num_nodes - 1)
        dst = action % (self.num_nodes - 1)
        if dst >= src:
            dst += 1
            
        # Apply routing decision and update metrics
        edge = (src, dst) if (src, dst) in self.network.edges() else (dst, src)
        if edge in self.network.edges():
            # Reduce utilization when the path is chosen
            self.bandwidth_utilization[edge] = max(0.1, self.bandwidth_utilization[edge] - 0.15)
            # Improve latency
            self.latency[edge] = max(1.0, self.latency[edge] * 0.85)
            # Reduce packet loss
            self.packet_loss[edge] = max(0.01, self.packet_loss[edge] * 0.85)
            # Improve throughput
            self.throughput[edge] = min(1.0, self.throughput[edge] * 1.2)
            
        # Adjust reward function to more strongly incentivize improvements
        bandwidth_reward = (1 - np.mean(list(self.bandwidth_utilization.values()))) * 0.3
        latency_reward = (1 - np.clip(np.mean(list(self.latency.values())) / self.max_latency, 0, 1)) * 0.3
        packet_loss_reward = (1 - np.clip(np.mean(list(self.packet_loss.values())) / self.max_packet_loss, 0, 1)) * 0.2
        throughput_reward = np.clip(np.mean(list(self.throughput.values())) / self.min_throughput, 0, 1) * 0.2
        
        reward = bandwidth_reward + latency_reward + packet_loss_reward + throughput_reward
        
        # Add episode termination conditions
        done = (np.mean(list(self.latency.values())) > self.max_latency or
                np.mean(list(self.packet_loss.values())) > self.max_packet_loss or
                np.mean(list(self.throughput.values())) < self.min_throughput)
        
        info = {}
        
        return self._get_state(), reward, done, info