import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import logging
import random
from collections import deque
import torch.multiprocessing as mp
from torch.distributions import Normal
import threading
import torch.nn.functional as F
import platform
from gpu_utils import get_device_info

class PPONetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(PPONetwork, self).__init__()
        
        # Actor network with batch normalization and larger architecture
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        # Critic network with batch normalization and larger architecture
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state):
        # Ensure state is 2D for LayerNorm
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Get raw logits from actor
        logits = self.actor(state)
        
        # Apply softmax with numerical stability
        logits = logits - logits.max(dim=-1, keepdim=True)[0]  # Subtract max for numerical stability
        exp_logits = torch.exp(logits)
        probs = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Ensure no NaN values
        probs = torch.where(torch.isnan(probs), torch.ones_like(probs) / probs.size(-1), probs)
        
        return probs, self.critic(state)

class PPOAgent:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, epsilon=0.2, 
                 epochs=4, batch_size=64, gae_lambda=0.95):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        
        # Initialize networks with modified learning rate
        self.input_dim = env.observation_space.shape[0]
        self.output_dim = env.action_space.n
        gpu_info = get_device_info()
        self.device = gpu_info['device']  # Use detected device
        self.network = PPONetwork(self.input_dim, self.output_dim).to(self.device)
        
        # Use separate optimizers with different learning rates
        self.actor_optimizer = optim.Adam(self.network.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.network.critic.parameters(), lr=learning_rate * 3)
        
        # Initialize memory buffers
        self.reset_memory()
        
        # Setup logging and device
        self.logger = logging.getLogger(__name__)
        self.network.to(self.device)
        
        # Increase entropy coefficient for better exploration
        self.entropy_coef = 0.02
        
        # Add experience replay buffer for better sample efficiency
        self.replay_buffer_size = 10000
        self.replay_buffer = []
        
    def select_action(self, state):
        """Select action using current policy with added safety checks"""
        state = torch.FloatTensor(state).to(self.device)
        action_probs, value = self.network(state)
        
        # Additional safety checks
        if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
            # If we still get NaN/Inf, fall back to uniform distribution
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
        
        # Ensure probabilities sum to 1
        action_probs = F.normalize(action_probs, p=1, dim=-1)
        
        try:
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Store trajectory information
            self.states.append(state)
            self.actions.append(action)
            self.values.append(value)
            self.log_probs.append(log_prob)
            
            return action.item(), log_prob.item()
        except ValueError as e:
            # Fallback to random action if distribution creation fails
            action = torch.randint(0, self.output_dim, (1,))[0]
            log_prob = torch.tensor(0.0)  # Default log prob
            
            # Store trajectory information
            self.states.append(state)
            self.actions.append(action)
            self.values.append(value)
            self.log_probs.append(log_prob)
            
            return action.item(), log_prob.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        """Store transition in memory"""
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def compute_advantages(self):
        """Compute advantages using GAE with lambda parameter"""
        advantages = []
        gae = 0
        next_value = 0
        
        # Move values to CPU for computation
        values = [v.cpu() for v in self.values]
        values = torch.cat(values).detach()
        
        for reward, value, done in zip(reversed(self.rewards), 
                                     reversed(values), 
                                     reversed(self.dones)):
            if done:
                delta = reward - value
                gae = delta
            else:
                delta = reward + self.gamma * next_value - value
                gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            next_value = value
        
        # Create tensor on CPU first
        advantages = torch.FloatTensor(advantages)
        # Then move to the appropriate device
        return advantages.to(self.device)
    
    def train(self):
        if len(self.states) < self.batch_size:
            return
        
        # Store experience in replay buffer
        for i in range(len(self.states)):
            self.replay_buffer.append({
                'state': self.states[i].cpu(),  # Move to CPU before storing
                'action': self.actions[i].cpu(),
                'reward': self.rewards[i],
                'log_prob': self.log_probs[i].cpu(),
                'value': self.values[i].cpu(),
                'done': self.dones[i]
            })
        
        # Keep only the most recent experiences
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]
        
        # Sample batch
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        # Process each sample individually to avoid scatter operations
        total_policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_value_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        total_entropy = torch.tensor(0.0, device=self.device)
        
        for idx in batch_indices:
            # Get sample data
            state = self.replay_buffer[idx]['state'].to(self.device)
            action = self.replay_buffer[idx]['action'].to(self.device)
            old_log_prob = self.replay_buffer[idx]['log_prob'].to(self.device)
            old_value = self.replay_buffer[idx]['value'].to(self.device)
            reward = self.replay_buffer[idx]['reward']
            done = self.replay_buffer[idx]['done']
            
            # Get current policy and value predictions
            action_probs, value = self.network(state)
            dist = Categorical(action_probs)
            curr_log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            
            # Compute advantage for this sample
            next_value = 0 if done else self.network(state)[1].detach()
            advantage = reward + (1 - done) * self.gamma * next_value - old_value
            
            # Normalize advantage
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            # Compute policy ratio and losses
            ratio = (curr_log_prob - old_log_prob).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage
            
            # Accumulate losses
            policy_loss = -torch.min(surr1, surr2)
            value_loss = F.mse_loss(value.squeeze(), reward + (1 - done) * self.gamma * next_value)
            
            total_policy_loss = total_policy_loss + policy_loss
            total_value_loss = total_value_loss + value_loss
            total_entropy = total_entropy + entropy
        
        # Calculate average losses
        avg_policy_loss = total_policy_loss / batch_size
        avg_value_loss = total_value_loss / batch_size
        avg_entropy = total_entropy / batch_size
        
        # Final loss with entropy bonus
        final_policy_loss = avg_policy_loss - self.entropy_coef * avg_entropy
        
        # Update actor
        self.actor_optimizer.zero_grad()
        final_policy_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.network.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        avg_value_loss.backward()
        nn.utils.clip_grad_norm_(self.network.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Clear memory buffers
        self.reset_memory()
    
    def reset_memory(self):
        """Clear memory buffers"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def reset_internal_state(self):
        """Reset any internal states (if needed)"""
        self.reset_memory()
    
    def save_model(self, path):
        """Save model to disk"""
        torch.save(self.network.state_dict(), path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model from disk"""
        try:
            self.network.load_state_dict(torch.load(path))
            self.logger.info(f"Model loaded from {path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DQNNetwork, self).__init__()
        
        # Replace BatchNorm1d with LayerNorm for better single sample handling
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights using Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Handle single state case
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

class DQNAgent:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, memory_size=10000,
                 batch_size=64):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        gpu_info = get_device_info()
        self.device = gpu_info['device']  # Use detected device
        
        # Simplified network architecture
        self.policy_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n)
        ).to(self.device)
        
        self.target_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n)
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                return self.policy_net(state).argmax().item()
        return random.randrange(self.env.action_space.n)

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_net(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = F.smooth_l1_loss(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset_internal_state(self):
        """Reset internal state while preserving learned parameters"""
        # Clear memory buffer but keep network weights
        self.memory.clear()
        # Reset epsilon to allow for more exploration after reset
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

class A3CNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(A3CNetwork, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        shared_features = self.shared(x)
        return self.actor(shared_features), self.critic(shared_features)

class A3CWorker:
    def __init__(self, global_network, optimizer, env, worker_id, global_episode, global_step, lock, device):
        self.global_network = global_network
        self.optimizer = optimizer
        self.env = env
        self.worker_id = worker_id
        self.global_episode = global_episode
        self.global_step = global_step
        self.lock = lock
        self.device = device
        
        
        self.local_network = A3CNetwork(
            input_dim=env.observation_space.shape[0],
            output_dim=env.action_space.n
        ).to(self.device)
        
        self.sync_with_global()

    def sync_with_global(self):
        """Synchronize local network with global network"""
        self.local_network.load_state_dict(self.global_network.state_dict())

    def train(self):
        """Training loop for the worker"""
        try:
            while True:  # Continuous training loop
                state = self.env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    # Get action probabilities and value
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    action_probs, value = self.local_network(state_tensor)
                    
                    # Sample action
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    # Take action in environment
                    next_state, reward, done, _ = self.env.step(action.item())
                    episode_reward += reward
                    
                    # Calculate advantage
                    next_state_tensor = torch.FloatTensor(next_state).to(self.device)
                    _, next_value = self.local_network(next_state_tensor)
                    advantage = reward + (0.99 * next_value * (1 - done)) - value
                    
                    # Calculate losses
                    actor_loss = -log_prob * advantage.detach()
                    critic_loss = F.smooth_l1_loss(value, reward + (0.99 * next_value * (1 - done)).detach())
                    total_loss = actor_loss + 0.5 * critic_loss
                    
                    # Update global network
                    with self.lock:
                        self.optimizer.zero_grad()
                        total_loss.backward()
                        # Clip gradients to avoid exploding gradients
                        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), max_norm=0.5)
                        self.optimizer.step()
                    
                    # Sync with global network
                    self.sync_with_global()
                    state = next_state
                    
        except Exception as e:
            print(f"Worker {self.worker_id} encountered error: {str(e)}")

class A3CAgent:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99, num_workers=4):
        self.env = env
        self.gamma = gamma
        self.global_episode = 0
        self.global_step = 0
        self.lock = threading.Lock()  # Use threading.Lock instead of mp.Lock
        
        gpu_info = get_device_info()
        self.device = gpu_info['device']  # Use detected device
        
        # Initialize global network without shared memory
        self.global_network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n + 1)  # Actions + Value
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.global_network.parameters(), lr=learning_rate)
        
        self.running = False
        self.workers = []
        self.threads = []

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            output = self.global_network(state)
            probs = F.softmax(output[:-1], dim=0)
            action = Categorical(probs).sample()
            return action.item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition for training"""
        pass  # A3C doesn't use a replay buffer

    def train(self):
        """Start training threads if not already running"""
        if not self.running:
            self.running = True
            # Start threads if they're not running
            for i, worker in enumerate(self.workers):
                if i >= len(self.threads) or not self.threads[i].is_alive():
                    thread = threading.Thread(target=worker.train)
                    thread.daemon = True
                    self.threads.append(thread)
                    thread.start()

    def reset_internal_state(self):
        """Reset internal state while preserving learned parameters"""
        self.running = False
        # Create new threads
        self.threads = []
        # Sync workers with global network
        for worker in self.workers:
            worker.sync_with_global()

    def __del__(self):
        """Cleanup method"""
        self.running = False

class REINFORCEAgent:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99):
        self.env = env
        gpu_info = get_device_info()
        self.device = gpu_info['device']  # Use detected device
        
        # Simplified policy network architecture
        self.policy_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Linear(128, env.action_space.n),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # Initialize weights
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.rewards = []
        self.states = []
        self.actions = []
        self.eps = 1e-8

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            probs = self.policy_net(state)
            
        try:
            dist = Categorical(probs)
            action = dist.sample()
            
            self.states.append(state)
            self.actions.append(action)
            
            return action.item()
        except ValueError as e:
            print(f"Warning: Invalid probability distribution. Using random action. Error: {e}")
            return self.env.action_space.sample()

    def store_transition(self, state, action, reward, next_state, done):
        self.rewards.append(reward)

    def train(self):
        if len(self.rewards) == 0 or len(self.states) == 0:
            return

        # Calculate discounted rewards using numpy
        discounted_rewards = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        
        # Convert to numpy array for normalization
        discounted_rewards = np.array(discounted_rewards)
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + self.eps)
        
        # Initialize total loss as a scalar tensor on CPU
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Process each state-action pair individually
        for i in range(len(self.states)):
            state = self.states[i]
            action = self.actions[i]
            reward = torch.tensor(discounted_rewards[i], device=self.device)
            
            # Forward pass for single state
            probs = self.policy_net(state)
            dist = Categorical(probs)
            log_prob = dist.log_prob(action)
            
            # Add to total loss using basic arithmetic
            loss = -log_prob * reward
            total_loss = total_loss + loss
        
        # Calculate average loss
        if len(self.states) > 0:
            policy_loss = total_loss / len(self.states)
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        self.reset_internal_state()

    def reset_internal_state(self):
        """Reset internal state while preserving learned parameters"""
        self.rewards = []
        self.states = []
        self.actions = []

class HybridAgent:
    def __init__(self, env):
        """Initialize all sub-agents and metric assignments"""
        self.env = env
        
        # Initialize all possible agents
        self.dqn_agent = DQNAgent(env)
        self.ppo_agent = PPOAgent(env)
        self.a3c_agent = A3CAgent(env)
        self.reinforce_agent = REINFORCEAgent(env)
        
        # Default metric assignments
        self.metric_agents = {
            'bandwidth': self.dqn_agent,
            'latency': self.ppo_agent,
            'packet_loss': self.a3c_agent,
            'throughput': self.reinforce_agent
        }
        
        # Available agents for each metric
        self.available_agents = {
            'dqn': self.dqn_agent,
            'ppo': self.ppo_agent,
            'a3c': self.a3c_agent,
            'reinforce': self.reinforce_agent
        }
        
        # Initialize metric weights based on environment's reward weights
        self.metric_weights = {
            'bandwidth': 2.0,    # Matches reward weight in env.step()
            'latency': 1.5,      # Matches reward weight in env.step()
            'packet_loss': 1.5,  # Matches reward weight in env.step()
            'throughput': 1.0    # Matches reward weight in env.step()
        }
        
        # Normalize weights
        total_weight = sum(self.metric_weights.values())
        self.metric_weights = {k: v/total_weight for k, v in self.metric_weights.items()}

    def set_agent_for_metric(self, metric: str, agent_type: str) -> bool:
        """
        Assign an agent type to a specific metric
        
        Args:
            metric: The metric to assign an agent to ('bandwidth', 'latency', etc.)
            agent_type: The type of agent to assign ('dqn', 'ppo', etc.)
        
        Returns:
            bool: True if assignment was successful, False otherwise
        """
        if agent_type in self.available_agents and metric in self.metric_agents:
            self.metric_agents[metric] = self.available_agents[agent_type]
            return True
        return False

    def select_action(self, state):
        """
        Select action by combining recommendations from all agents
        
        Args:
            state: The current environment state
        
        Returns:
            int: The selected action
        """
        # Get action recommendations from each agent
        actions = {}
        for metric, agent in self.metric_agents.items():
            if isinstance(agent, PPOAgent):
                action, _ = agent.select_action(state)
            else:
                action = agent.select_action(state)
            actions[metric] = action
        
        # Weight the actions based on current network conditions
        weights = self._analyze_network_state(state)
        final_action = self._combine_actions(actions, weights)
        
        return final_action

    def _analyze_network_state(self, state):
        """
        Analyze current network state to determine metric importance
        Uses environment's metric bounds and congestion levels
        """
        metrics = {
            'bandwidth': self.env.bandwidth_utilization,
            'latency': self.env.latency,
            'packet_loss': self.env.packet_loss,
            'throughput': self.env.throughput
        }
        
        # Calculate normalized urgency for each metric
        urgency = {}
        for metric, values in metrics.items():
            if values:  # Check if there are any values
                current_value = np.mean(list(values.values()))
                
                if metric == 'bandwidth':
                    # Higher utilization = more urgent
                    urgency[metric] = current_value / self.env.metric_bounds[metric][1]
                elif metric == 'latency':
                    # Higher latency = more urgent
                    urgency[metric] = current_value / self.env.max_latency
                elif metric == 'packet_loss':
                    # Higher loss = more urgent
                    urgency[metric] = current_value / self.env.max_packet_loss
                elif metric == 'throughput':
                    # Lower throughput = more urgent
                    urgency[metric] = 1 - (current_value / self.env.metric_bounds[metric][1])
            else:
                urgency[metric] = 0.25  # Default weight if no data
        
        # Normalize urgency scores
        total_urgency = sum(urgency.values())
        if total_urgency == 0:
            return self.metric_weights  # Use default weights if all urgencies are 0
        
        normalized_weights = {k: v/total_urgency for k, v in urgency.items()}
        
        # Combine with base weights
        final_weights = {}
        for metric in self.metric_weights:
            final_weights[metric] = (
                0.7 * normalized_weights[metric] +  # 70% based on current state
                0.3 * self.metric_weights[metric]   # 30% based on base weights
            )
        
        # Normalize final weights
        total_weight = sum(final_weights.values())
        return {k: v/total_weight for k, v in final_weights.items()}

    def _combine_actions(self, actions, weights):
        """
        Combine actions using weighted voting and environment constraints
        """
        action_votes = {}
        
        # Initialize votes for all possible actions
        num_nodes = self.env.num_nodes
        for i in range(num_nodes * (num_nodes - 1)):  # All possible actions
            action_votes[i] = 0
        
        # Count weighted votes for each action
        for metric, action in actions.items():
            weight = weights[metric]
            action_votes[action] = action_votes.get(action, 0) + weight
        
        # Filter invalid actions (non-existent edges)
        valid_actions = []
        for action, votes in action_votes.items():
            src = action // (num_nodes - 1)
            dst = action % (num_nodes - 1)
            if dst >= src:
                dst += 1
            edge = (src, dst) if (src, dst) in self.env.network.edges() else (dst, src)
            
            if edge in self.env.network.edges():
                valid_actions.append((action, votes))
        
        if not valid_actions:
            # If no valid actions found, return random valid action
            valid_edges = list(self.env.network.edges())
            if valid_edges:
                edge = random.choice(valid_edges)
                src, dst = edge
                return src * (num_nodes - 1) + (dst if dst < src else dst - 1)
            return 0
        
        # Return action with highest weighted votes among valid actions
        return max(valid_actions, key=lambda x: x[1])[0]

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition in all agents with metric-specific rewards
        """
        metrics_before = {
            'bandwidth': np.mean(list(self.env.bandwidth_utilization.values())),
            'latency': np.mean(list(self.env.latency.values())),
            'packet_loss': np.mean(list(self.env.packet_loss.values())),
            'throughput': np.mean(list(self.env.throughput.values()))
        }
        
        metrics_after = {
            'bandwidth': np.mean(list(self.env.bandwidth_utilization.values())),
            'latency': np.mean(list(self.env.latency.values())),
            'packet_loss': np.mean(list(self.env.packet_loss.values())),
            'throughput': np.mean(list(self.env.throughput.values()))
        }
        
        # Calculate metric-specific rewards
        metric_rewards = {
            'bandwidth': (metrics_before['bandwidth'] - metrics_after['bandwidth']) * 2.0,
            'latency': (metrics_before['latency'] - metrics_after['latency']) * 1.5,
            'packet_loss': (metrics_before['packet_loss'] - metrics_after['packet_loss']) * 1.5,
            'throughput': (metrics_after['throughput'] - metrics_before['throughput']) * 1.0
        }
        
        # Store transitions with metric-specific rewards
        for metric, agent in self.metric_agents.items():
            metric_reward = metric_rewards[metric]
            if isinstance(agent, PPOAgent):
                agent.store_transition(state, action, metric_reward, next_state, done, None)
            else:
                agent.store_transition(state, action, metric_reward, next_state, done)

    def train(self):
        """Train all agents"""
        for agent in set(self.metric_agents.values()):
            agent.train()

    def reset_internal_state(self):
        """Reset internal state of all agents"""
        for agent in set(self.metric_agents.values()):
            agent.reset_internal_state()
