#rl_agent.py

import random
import numpy as np

class RLAgent:
    def __init__(self, links, alpha=0.1, gamma=0.9, epsilon=0.1, min_capacity=1, max_capacity=100):
        self.links = links
        self.state_size = len(links)
        self.action_space = [-5, 0, 5]  # Decrease, no change, increase capacity
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        self.q_table = {}

    def get_state(self):
        """Get the current state (link capacities and load)."""
        return tuple(np.array([link.capacity for link in self.links] + [link.load for link in self.links]))

    def choose_action(self, state):
        """Choose action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # Exploration: Choose random actions
            return [random.choice(self.action_space) for _ in range(len(state) // 2)]
        else:
            # Exploitation: Choose the best action based on the Q-table
            if state not in self.q_table:
                self.q_table[state] = [0] * len(self.action_space)
            return [self.action_space[max(range(len(self.action_space)), key=lambda x: self.q_table[state][x])] for _ in range(len(state) // 2)]

    def calculate_reward(self, state, actions):
        """Calculate the reward based on link load and capacity."""
        total_capacity = sum(state[:len(self.links)])  # State holds capacities in the first half
        total_load = sum(link.load for link in self.links)
        reward = -total_load / total_capacity  # Negative reward if the load is high relative to capacity
        return reward

    def update_q_table(self, state, actions, reward, next_state):
        """Update the Q-table using the Q-learning update rule."""
        if state not in self.q_table:
            self.q_table[state] = [0] * len(self.action_space)
        if next_state not in self.q_table:
            self.q_table[next_state] = [0] * len(self.action_space)
        
        # Find the action taken and its index
        action_idx = [self.action_space.index(action) for action in actions]
        
        # Compute the best future action (max Q-value)
        best_next_action = max(self.q_table[next_state])

        # Update Q-table for each action
        for idx, action in zip(action_idx, actions):
            self.q_table[state][idx] = self.q_table[state][idx] + self.alpha * (
                reward + self.gamma * best_next_action - self.q_table[state][idx])

    def simulate_network(self, steps=1):
        """Simulate the network over a given number of steps."""
        for _ in range(steps):
            state = self.get_state()
            actions = self.choose_action(state)
            reward = self.calculate_reward(state, actions)
            next_state = self.get_state()  # Next state could be based on network changes
            self.update_q_table(state, actions, reward, next_state)

