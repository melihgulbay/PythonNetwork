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
            nn.Softmax(dim=-1)
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
        return self.actor(state), self.critic(state)

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
        self.network = PPONetwork(self.input_dim, self.output_dim)
        
        # Use separate optimizers with different learning rates
        self.actor_optimizer = optim.Adam(self.network.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.network.critic.parameters(), lr=learning_rate * 3)
        
        # Initialize memory buffers
        self.reset_memory()
        
        # Setup logging and device
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network.to(self.device)
        
        # Increase entropy coefficient for better exploration
        self.entropy_coef = 0.02
        
        # Add experience replay buffer for better sample efficiency
        self.replay_buffer_size = 10000
        self.replay_buffer = []
        
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.FloatTensor(state).to(self.device)
        action_probs, value = self.network(state)
        
        # Create distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Store trajectory information
        self.states.append(state)
        self.actions.append(action)
        self.values.append(value)
        self.log_probs.append(dist.log_prob(action))
        
        return action.item(), dist.log_prob(action).item()
    
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
        
        values = torch.cat(self.values).detach()
        
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
        
        return torch.FloatTensor(advantages).to(self.device)
    
    def train(self):
        if len(self.states) < self.batch_size:
            return
        
        # Store experience in replay buffer
        for i in range(len(self.states)):
            self.replay_buffer.append({
                'state': self.states[i],
                'action': self.actions[i],
                'reward': self.rewards[i],
                'log_prob': self.log_probs[i],
                'value': self.values[i],
                'done': self.dones[i]
            })
        
        # Keep only the most recent experiences if buffer is too large
        if len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.replay_buffer_size:]
        
        # Sample batch size number of experiences
        batch_size = min(self.batch_size, len(self.replay_buffer))
        batch_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        
        # Prepare batches
        states = torch.stack([self.replay_buffer[i]['state'] for i in batch_indices]).detach()
        actions = torch.stack([self.replay_buffer[i]['action'] for i in batch_indices]).detach()
        old_log_probs = torch.stack([self.replay_buffer[i]['log_prob'] for i in batch_indices]).detach()
        
        # Compute advantages and returns
        advantages = self.compute_advantages()
        # Ensure advantages match batch size
        advantages = advantages[-batch_size:] if len(advantages) >= batch_size else advantages
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Get current values for returns calculation
        current_values = torch.cat([self.replay_buffer[i]['value'] for i in batch_indices]).detach()
        # Ensure current_values is the right shape
        current_values = current_values.view(-1)  # Flatten to 1D
        
        # Calculate returns
        returns = advantages + current_values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update with mini-batches
        for _ in range(self.epochs):
            # Get current policy and value predictions
            action_probs, values = self.network(states)
            dist = Categorical(action_probs)
            curr_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Ensure values is the right shape
            values = values.squeeze()  # Remove extra dimensions
            
            # Compute policy ratio and surrogate loss
            ratios = (curr_log_probs - old_log_probs).exp()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
            
            # Calculate losses with entropy bonus
            policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            value_loss = nn.MSELoss()(values, returns)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(self.network.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            value_loss.backward()
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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

class A3CWorker(mp.Process):
    def __init__(self, global_network, optimizer, env, worker_id, 
                 global_episode, global_step, lock, device):
        super(A3CWorker, self).__init__()
        self.global_network = global_network
        self.optimizer = optimizer
        self.env = env
        self.worker_id = worker_id
        self.global_episode = global_episode
        self.global_step = global_step
        self.lock = lock
        self.device = device
        
        # Initialize local network
        self.local_network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n + 1)
        ).to(device)
        
        # Copy global parameters to local
        self.sync_with_global()
        
    def sync_with_global(self):
        self.local_network.load_state_dict(self.global_network.state_dict())
    
    def train(self):
        try:
            while self.global_episode.value < 1000:  # Training episodes limit
                state = self.env.reset()
                done = False
                episode_reward = 0
                
                # Store episode history
                states, actions, rewards = [], [], []
                values = []
                
                while not done:
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    output = self.local_network(state_tensor)
                    action_logits = output[:-1]
                    value = output[-1]
                    
                    action_probs = F.softmax(action_logits, dim=0)
                    dist = Categorical(action_probs)
                    action = dist.sample()
                    
                    next_state, reward, done, _ = self.env.step(action.item())
                    
                    states.append(state_tensor)
                    actions.append(action)
                    rewards.append(reward)
                    values.append(value)
                    
                    state = next_state
                    episode_reward += reward
                    
                    with self.lock:
                        self.global_step.value += 1
                
                # Process episode data and update global network
                self.process_episode(states, actions, rewards, values)
                
        except Exception as e:
            print(f"Worker {self.worker_id} failed: {str(e)}")

    def process_episode(self, states, actions, rewards, values):
        if len(states) == 0:
            return

        # Calculate returns and advantages
        R = 0
        returns = []
        advantages = []
        
        for r, v in zip(reversed(rewards), reversed(values)):
            R = r + 0.99 * R
            advantage = R - v.item()
            returns.insert(0, R)
            advantages.insert(0, advantage)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Calculate losses
        outputs = self.local_network(states)
        action_logits = outputs[:, :-1]
        values_pred = outputs[:, -1]
        
        action_probs = F.softmax(action_logits, dim=1)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = 0.5 * (returns - values_pred).pow(2).mean()
        total_loss = actor_loss + critic_loss - 0.01 * entropy
        
        # Update global network
        with self.lock:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_network.parameters(), 40.0)
            
            for global_param, local_param in zip(
                self.global_network.parameters(),
                self.local_network.parameters()
            ):
                if global_param.grad is None:
                    global_param.grad = local_param.grad
                else:
                    global_param.grad += local_param.grad
            
            self.optimizer.step()
            self.global_episode.value += 1
            
        self.sync_with_global()

class A3CAgent:
    def __init__(self, env, num_workers=4, learning_rate=1e-4):
        super(A3CAgent, self).__init__()
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enhanced parameters
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5
        self.max_grad_norm = 0.5
        
        # Initialize shared counters
        self.global_episode = mp.Value('i', 0)
        self.global_step = mp.Value('i', 0)
        self.lock = mp.Lock()
        
        self.global_network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, env.action_space.n + 1)  # Actions + Value
        ).to(self.device)
        
        self.global_network.share_memory()
        self.optimizer = optim.Adam(self.global_network.parameters(), lr=learning_rate)
        
        # Initialize workers
        self.workers = []
        self.processes = []
        for i in range(num_workers):
            worker = A3CWorker(
                global_network=self.global_network,
                optimizer=self.optimizer,
                env=env,
                worker_id=i,
                global_episode=self.global_episode,
                global_step=self.global_step,
                lock=self.lock,
                device=self.device
            )
            self.workers.append(worker)
            # Create process but don't start it yet
            process = mp.Process(target=worker.train)
            self.processes.append(process)

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
        """Start training processes if not already running"""
        # Start processes if they're not running
        for p in self.processes:
            if not p.is_alive():
                p.start()

    def reset_internal_state(self):
        """Reset internal state while preserving learned parameters"""
        # Stop all processes
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join()
        
        # Create new processes
        self.processes = []
        for worker in self.workers:
            worker.sync_with_global()
            process = mp.Process(target=worker.train)
            self.processes.append(process)

    def __del__(self):
        """Cleanup method"""
        for p in self.processes:
            if p.is_alive():
                p.terminate()
                p.join()

class REINFORCEAgent:
    def __init__(self, env, learning_rate=3e-4, gamma=0.99):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Simple policy network with proper initialization
        self.policy_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
            nn.Softmax(dim=-1)
        ).to(self.device)
        
        # Initialize weights properly
        for layer in self.policy_net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.rewards = []
        self.log_probs = []
        self.eps = 1e-8

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Enable gradients for action selection
        probs = self.policy_net(state)
        probs = F.softmax(probs, dim=-1)
        probs = probs + self.eps
        probs = probs / probs.sum()
        
        try:
            dist = Categorical(probs)
            action = dist.sample()
            # Store log probability with gradient tracking
            self.log_probs.append(dist.log_prob(action))
            return action.item()
        except ValueError as e:
            print(f"Warning: Invalid probability distribution. Using random action. Error: {e}")
            return self.env.action_space.sample()

    def store_transition(self, state, action, reward, next_state, done):
        self.rewards.append(reward)

    def train(self):
        if len(self.rewards) == 0:
            return

        # Calculate returns with gradient tracking
        returns = []
        R = 0
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # Ensure log_probs have gradients
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            if log_prob is not None and log_prob.requires_grad:  # Check if gradient tracking is enabled
                policy_loss.append(-log_prob * R.detach())  # Detach returns to only update policy
        
        if not policy_loss:  # If no valid log probabilities, skip update
            self.reset_internal_state()
            return

        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.reset_internal_state()

    def reset_internal_state(self):
        """Reset internal state while preserving learned parameters"""
        self.rewards = []
        self.log_probs = []

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
