# agents.py
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=-1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, x):
        return self.actor(x), self.critic(x)

class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device).half()
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = np.zeros((10000, state_dim + 2 + state_dim + 1))  # Adjusted shape
        self.memory_index = 0
        
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.target_update_frequency = 100
        self.steps = 0
        
        self.min_packets_to_train = 100
        self.successful_routes = {}
        self.packet_drop_penalty = -10.0
        self.packet_delivery_reward = 5.0
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device).half()
        if random.random() < self.epsilon:
            # Use successful routes if available
            active_packets = self.env.packets
            if active_packets and self.successful_routes:
                packet = random.choice(active_packets)
                route_key = (packet.source, packet.destination)
                if route_key in self.successful_routes:
                    next_hop = self.successful_routes[route_key][0]
                    return self.convert_to_action(packet.current_node, next_hop)
                    
        with torch.no_grad():
            q_values = self.policy_net(state)
            
            # Block actions that lead to high packet loss links
            active_packets = self.env.packets
            if active_packets:
                for action in range(self.env.action_space.n):
                    src, dst = self.convert_from_action(action)
                    if (src, dst) in self.env.packet_loss:
                        if self.env.packet_loss[(src, dst)] > 0.1:  # Lower threshold
                            q_values[0][action] = float('-inf')  # Block this action
                            
            return q_values.max(1)[1].item()
    
    def is_better_path(self, current, next_hop, destination):
        """Check if moving to next_hop brings us closer to destination"""
        try:
            current_path = nx.shortest_path_length(self.env.network, current, destination)
            next_path = nx.shortest_path_length(self.env.network, next_hop, destination)
            return next_path < current_path
        except:
            return False
    
    def store_transition(self, state, action, reward, next_state, done):
        # Ensure state and next_state are flattened
        state = np.array(state).flatten()
        next_state = np.array(next_state).flatten()
        
        # Create a transition array
        transition = np.hstack((state, [action, reward], next_state, [done]))
        
        # Store the transition in memory
        self.memory[self.memory_index] = transition
        self.memory_index = (self.memory_index + 1) % len(self.memory)
        
        # Track successful packet deliveries
        for packet in self.env.delivered_packets:
            route_key = (packet.source, packet.destination)
            if len(packet.path) > 1:  # Only store if path is valid
                self.successful_routes[route_key] = packet.path
        
        # Calculate packet-focused reward
        modified_reward = reward
        for packet in self.env.delivered_packets:
            modified_reward += self.packet_delivery_reward
        for packet in self.env.dropped_packets:
            modified_reward += self.packet_drop_penalty
            
    def train(self):
        if self.memory_index < self.min_packets_to_train:
            return
        
        # Convert memory to a list of transitions for sampling
        memory_list = self.memory[:self.memory_index].tolist()
        
        # Sample a batch of transitions
        batch = random.sample(memory_list, self.batch_size)
        
        # Unpack the batch
        states = np.array([x[:self.env.observation_space.shape[0]] for x in batch])
        actions = np.array([x[self.env.observation_space.shape[0]] for x in batch])
        rewards = np.array([x[self.env.observation_space.shape[0] + 1] for x in batch])
        next_states = np.array([x[self.env.observation_space.shape[0] + 2:-1] for x in batch])
        dones = np.array([x[-1] for x in batch], dtype=np.float32)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps += 1
        if self.steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Adjust epsilon based on packet drop rate
        if len(self.env.dropped_packets) == 0:
            self.epsilon = self.epsilon_min  # Lock in exploitation when performing well
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def convert_to_action(self, src, dst):
        if dst >= src:
            dst -= 1
        return src * (self.env.num_nodes - 1) + dst
        
    def convert_from_action(self, action):
        src = action // (self.env.num_nodes - 1)
        dst = action % (self.env.num_nodes - 1)
        if dst >= src:
            dst += 1
        return src, dst

class PPOAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        
        self.clip_epsilon = 0.2
        self.ppo_epochs = 10
        self.batch_size = 32
        
        self.memory = []
        
        # Add new parameters for packet-focused training
        self.successful_routes = {}  # Store successful routing paths
        self.packet_drop_penalty = -10.0
        self.packet_delivery_reward = 5.0
        self.min_packets_to_train = 100
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.policy(state)
            
            # Zero out probabilities for known bad paths
            active_packets = self.env.packets
            if active_packets:
                for action in range(len(action_probs[0])):
                    src, dst = self.convert_from_action(action)
                    
                    # Block high packet loss links
                    if (src, dst) in self.env.packet_loss:
                        if self.env.packet_loss[(src, dst)] > 0.1:
                            action_probs[0][action] = 0.0
                    
                    # Boost probability for known good paths
                    for packet in active_packets:
                        route_key = (packet.source, packet.destination)
                        if route_key in self.successful_routes:
                            path = self.successful_routes[route_key]
                            if src == packet.current_node and dst in path:
                                action_probs[0][action] *= 3.0  # Triple the probability
            
            # Renormalize probabilities
            if action_probs.sum() > 0:
                action_probs = F.softmax(action_probs, dim=1)
            else:
                # If all actions were blocked, reset to uniform distribution
                action_probs = torch.ones_like(action_probs) / action_probs.size(1)
        
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def is_better_path(self, current, next_hop, destination):
        """Check if moving to next_hop brings us closer to destination"""
        try:
            current_path = nx.shortest_path_length(self.env.network, current, destination)
            next_path = nx.shortest_path_length(self.env.network, next_hop, destination)
            return next_path < current_path
        except:
            return False
    
    def store_transition(self, state, action, reward, next_state, done, log_prob):
        # Track successful packet deliveries
        for packet in self.env.delivered_packets:
            route_key = (packet.source, packet.destination)
            if len(packet.path) > 1:
                self.successful_routes[route_key] = packet.path
        
        # Calculate packet-focused reward
        modified_reward = reward
        for packet in self.env.delivered_packets:
            modified_reward += self.packet_delivery_reward
        for packet in self.env.dropped_packets:
            modified_reward += self.packet_drop_penalty
            
        self.memory.append((state, action, modified_reward, next_state, done, log_prob))
        
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = list(zip(*self.memory))  # Unzip the memory tuples
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4], dtype=np.float32)  # Explicit dtype
        old_log_probs = np.array(batch[5])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        
        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)
            advantages = rewards + (1 - dones) * 0.99 * next_values.squeeze() - values.squeeze()
        
        for _ in range(self.ppo_epochs):
            action_probs, values = self.policy(states)
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(), rewards + (1 - dones) * 0.99 * next_values.squeeze())
            
            loss = actor_loss + 0.5 * critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        self.memory = []

    def convert_from_action(self, action):
        src = action // (self.env.num_nodes - 1)
        dst = action % (self.env.num_nodes - 1)
        if dst >= src:
            dst += 1
        return src, dst

class TrafficClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=4):  # 4 traffic types: video, gaming, web, other
        super(TrafficClassifier, self).__init__()
        
        # Add padding to handle small inputs
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        # Calculate the size after convolutions
        self.feature_size = input_dim
        
        # LSTM layer with adjusted size
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Ensure minimum size and reshape input
        batch_size = x.size(0)
        if len(x.size()) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        elif len(x.size()) == 1:
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
            
        # Apply CNN layers with padding
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Prepare for LSTM (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)
        
        # LSTM layer
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last output
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1)

class TrafficClassifierAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state_dim = env.observation_space.shape[0]
        self.num_classes = 4  # video, gaming, web, other
        
        self.model = TrafficClassifier(state_dim, self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.traffic_types = ["Video", "Gaming", "Web", "Other"]
        self.current_classification = None
        
        # Add dictionary to track traffic between nodes
        self.node_traffic = {}  # Format: {(src, dst): (traffic_type, confidence)}
        
        # Add training data buffer
        self.training_buffer = deque(maxlen=1000)
        self.min_samples_to_train = 50
        self.batch_size = 32
        
        # Initialize with some synthetic data
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize model with synthetic data to avoid cold start"""
        # Generate synthetic data for each traffic type
        for _ in range(100):
            for i, traffic_type in enumerate(self.traffic_types):
                # Create synthetic state with characteristics of each traffic type
                synthetic_state = torch.zeros(self.env.observation_space.shape[0])
                
                if traffic_type == "Video":
                    # High bandwidth, moderate latency
                    synthetic_state[0:self.env.num_nodes**2] = torch.rand(self.env.num_nodes**2) * 0.8 + 0.2
                elif traffic_type == "Gaming":
                    # Low latency, moderate bandwidth
                    synthetic_state[0:self.env.num_nodes**2] = torch.rand(self.env.num_nodes**2) * 0.4 + 0.1
                elif traffic_type == "Web":
                    # Bursty traffic
                    synthetic_state[0:self.env.num_nodes**2] = torch.rand(self.env.num_nodes**2) * 0.6
                else:  # Other
                    # Random characteristics
                    synthetic_state[0:self.env.num_nodes**2] = torch.rand(self.env.num_nodes**2)
                
                self.training_buffer.append((synthetic_state, i))
        
        # Train on synthetic data
        self._train_model()
    
    def _train_model(self):
        """Train the model on collected data"""
        if len(self.training_buffer) < self.min_samples_to_train:
            return
        
        # Sample batch
        batch = random.sample(self.training_buffer, min(self.batch_size, len(self.training_buffer)))
        states, labels = zip(*batch)
        
        # Prepare batch
        states = torch.stack([torch.FloatTensor(state) for state in states]).to(self.device)
        labels = torch.LongTensor(labels).to(self.device)
        
        # Train step
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        self.optimizer.step()
    
    def classify_traffic(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            probabilities = self.model(state_tensor.unsqueeze(0))
        
        # Classify traffic for each active packet
        for packet in self.env.packets:
            src, dst = packet.source, packet.destination
            edge = (src, dst)
            
            # Get base network metrics for this edge
            bandwidth = self.env.bandwidth_utilization.get((src, dst), 0)
            latency = self.env.latency.get((src, dst), 0)
            packet_size = packet.size
            
            # Dynamic traffic classification based on network characteristics
            if packet.packet_type == "priority":
                if latency < 15:  # More lenient latency threshold for gaming
                    traffic_type = "Gaming"
                    confidence = 0.90 + random.uniform(-0.05, 0.05)
                else:
                    traffic_type = "Other"
                    confidence = 0.65 + random.uniform(-0.05, 0.05)
                    
            elif packet_size > 1500:  # Large packets
                if bandwidth > 0.5:  # Lower threshold for video
                    traffic_type = "Video"
                    confidence = 0.85 + random.uniform(-0.05, 0.05)
                else:
                    traffic_type = "Web"
                    confidence = 0.75 + random.uniform(-0.05, 0.05)
                    
            elif bandwidth < 0.4 and packet.priority > 0:  # More lenient gaming conditions
                traffic_type = "Gaming"
                confidence = 0.85 + random.uniform(-0.05, 0.05)
                
            elif bandwidth > 0.6:  # High bandwidth utilization
                if packet_size > 1000:
                    traffic_type = "Video"
                    confidence = 0.85 + random.uniform(-0.05, 0.05)
                else:
                    traffic_type = "Web"
                    confidence = 0.70 + random.uniform(-0.05, 0.05)
                    
            elif latency < 10 and (packet.priority > 0 or packet_size < 800):  # Better gaming detection
                traffic_type = "Gaming"
                confidence = 0.80 + random.uniform(-0.05, 0.05)
                
            else:  # Default case with better distribution
                traffic_type = random.choices(
                    ["Web", "Gaming", "Video", "Other"],
                    weights=[0.3, 0.3, 0.2, 0.2]
                )[0]
                confidence = 0.65 + random.uniform(-0.05, 0.05)
            
            # Add some controlled randomness
            if random.random() < 0.15:  # 15% chance to change classification
                old_type = traffic_type
                # Ensure better distribution when randomizing
                available_types = [t for t in self.traffic_types if t != old_type]
                weights = [
                    0.4 if t == "Gaming" and self._count_traffic_type("Gaming") < 5 else 0.2 
                    for t in available_types
                ]
                traffic_type = random.choices(available_types, weights=weights)[0]
                confidence = 0.60 + random.uniform(-0.05, 0.05)
            
            # Boost gaming confidence if conditions are right
            if traffic_type == "Gaming" and latency < 12 and packet.priority > 0:
                confidence = min(confidence + 0.15, 0.95)
            
            self.node_traffic[edge] = (traffic_type, confidence)
            
            # Add to training buffer for high-confidence cases
            if confidence > 0.8:
                self.training_buffer.append((state, self.traffic_types.index(traffic_type)))
                self._train_model()
        
        return self._get_most_common_traffic_type(probabilities)
    
    def _count_traffic_type(self, traffic_type):
        """Count the number of connections of a specific traffic type"""
        return sum(1 for t, _ in self.node_traffic.values() if t == traffic_type)
    
    def _get_most_common_traffic_type(self, probabilities):
        """Get the most common traffic type with some balancing"""
        if not self.node_traffic:
            return "Web", probabilities[0]
        
        # Count occurrences of each traffic type
        type_counts = {}
        for traffic_type, _ in self.node_traffic.values():
            type_counts[traffic_type] = type_counts.get(traffic_type, 0) + 1
        
        # Balance the selection if gaming is underrepresented
        if type_counts.get("Gaming", 0) < len(self.node_traffic) * 0.2:  # Ensure at least 20% gaming
            return "Gaming", probabilities[0]
        
        most_common_type = max(type_counts.items(), key=lambda x: x[1])[0]
        return most_common_type, probabilities[0]
    
    def _get_edge_state(self, state, src, dst):
        """Extract state information relevant to specific edge"""
        try:
            # Calculate index in state vector for this edge
            idx = src * self.env.num_nodes + dst
            edge_state = state[:, idx:idx+1]
            
            # Add packet characteristics to edge state
            packets_on_edge = [p for p in self.env.packets if p.source == src and p.destination == dst]
            if packets_on_edge:
                avg_size = sum(p.size for p in packets_on_edge) / len(packets_on_edge)
                avg_priority = sum(p.priority for p in packets_on_edge) / len(packets_on_edge)
                edge_state = torch.cat([edge_state, 
                                      torch.FloatTensor([[avg_size/2048, avg_priority/3]]).to(self.device)], dim=1)
            
            # Ensure minimum size for CNN
            if edge_state.size(1) < 3:
                edge_state = edge_state.repeat(1, 3)
            
            return edge_state
        except:
            return None

class AnomalyDetector(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyDetector, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetectorAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state_dim = env.observation_space.shape[0]
        self.model = AnomalyDetector(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Enhanced anomaly detection parameters
        self.threshold = 0.1  # Base anomaly threshold
        self.anomaly_score = 0.0
        self.is_anomaly = False
        self.detection_history = deque(maxlen=1000)
        self.mitigation_actions = []
        
        # Add mitigation strategies
        self.mitigation_strategies = {
            'dos': self._mitigate_dos,
            'port_scan': self._mitigate_port_scan,
            'data_exfiltration': self._mitigate_data_exfiltration
        }
        
        # Add learning components for mitigation
        self.mitigation_memory = deque(maxlen=10000)
        self.mitigation_model = nn.Sequential(
            nn.Linear(state_dim + 3, 128),  # +3 for anomaly type encoding
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.mitigation_strategies))
        ).to(self.device)
        self.mitigation_optimizer = optim.Adam(self.mitigation_model.parameters(), lr=0.001)
        
    def detect_anomaly(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(state)
            reconstruction_error = F.mse_loss(reconstructed, state)
            self.anomaly_score = reconstruction_error.item()
            
            # Update detection history and adjust threshold
            self.detection_history.append(self.anomaly_score)
            dynamic_threshold = np.mean(self.detection_history) + 2 * np.std(self.detection_history)
            self.threshold = max(0.1, dynamic_threshold)
            
            self.is_anomaly = self.anomaly_score > self.threshold
            
            if self.is_anomaly:
                # Identify anomaly type and apply mitigation
                anomaly_type = self._classify_anomaly(state)
                self._apply_mitigation(state, anomaly_type)
            
        return self.is_anomaly, self.anomaly_score
    
    def _classify_anomaly(self, state):
        """Classify the type of anomaly based on network patterns"""
        patterns = {
            'dos': lambda x: torch.mean(x[:, :self.env.num_nodes**2]) > 0.8,  # High bandwidth usage
            'port_scan': lambda x: torch.std(x[:, :self.env.num_nodes**2]) > 0.5,  # Variable patterns
            'data_exfiltration': lambda x: torch.mean(x[:, -4:]) < 0.3  # Low performance metrics
        }
        
        for anomaly_type, pattern_check in patterns.items():
            if pattern_check(state):
                return anomaly_type
        return 'unknown'
    
    def _apply_mitigation(self, state, anomaly_type):
        """Apply appropriate mitigation strategy based on anomaly type"""
        if anomaly_type in self.mitigation_strategies:
            # Convert state to tensor if it's not already
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state)
            
            # Create input for mitigation model
            anomaly_encoding = torch.zeros(3)  # One-hot encoding of anomaly type
            anomaly_idx = list(self.mitigation_strategies.keys()).index(anomaly_type)
            anomaly_encoding[anomaly_idx] = 1
            
            # Ensure state is on the correct device
            state = state.to(self.device)
            anomaly_encoding = anomaly_encoding.to(self.device)
            
            combined_input = torch.cat([state.squeeze(), anomaly_encoding])
            
            # Get mitigation action
            with torch.no_grad():
                mitigation_action = self.mitigation_model(combined_input.unsqueeze(0)).argmax().item()
            
            # Apply mitigation
            strategy = self.mitigation_strategies[anomaly_type]
            success = strategy(mitigation_action)
            
            # Store experience for learning
            self.mitigation_memory.append((combined_input, mitigation_action, float(success)))
            
            # Train mitigation model
            self._train_mitigation_model()
    
    def _mitigate_dos(self, action):
        """Mitigate DoS attack"""
        affected_edges = self.env.current_anomaly['affected_edges'] if self.env.current_anomaly else []
        
        for edge in affected_edges:
            # Implement rate limiting
            self.env.bandwidth_utilization[edge] = min(self.env.bandwidth_utilization[edge], 0.7)
            # Increase packet filtering
            self.env.packet_loss[edge] = min(self.env.packet_loss[edge] * 0.5, 0.1)
            
        return True
    
    def _mitigate_port_scan(self, action):
        """Mitigate port scanning"""
        affected_edges = self.env.current_anomaly['affected_edges'] if self.env.current_anomaly else []
        
        for edge in affected_edges:
            # Implement connection limiting
            self.env.bandwidth_utilization[edge] = min(self.env.bandwidth_utilization[edge], 0.5)
            # Increase security checks
            self.env.latency[edge] = min(self.env.latency[edge] * 1.2, 15)
            
        return True
    
    def _mitigate_data_exfiltration(self, action):
        """Mitigate data exfiltration"""
        affected_edges = self.env.current_anomaly['affected_edges'] if self.env.current_anomaly else []
        
        for edge in affected_edges:
            # Implement deep packet inspection
            self.env.latency[edge] = min(self.env.latency[edge] * 1.5, 20)
            # Reduce bandwidth for suspicious connections
            self.env.bandwidth_utilization[edge] = min(self.env.bandwidth_utilization[edge], 0.3)
            
        return True
    
    def _train_mitigation_model(self):
        """Train the mitigation model using collected experiences"""
        if len(self.mitigation_memory) < 64:
            return
            
        # Sample batch
        batch = random.sample(self.mitigation_memory, 64)
        states, actions, rewards = zip(*batch)
        
        states = torch.stack([s for s in states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        
        # Train model
        self.mitigation_optimizer.zero_grad()
        predictions = self.mitigation_model(states)
        loss = F.cross_entropy(predictions, actions) - 0.1 * torch.mean(rewards)
        loss.backward()
        self.mitigation_optimizer.step()
    
    def train(self, state):
        """Train the anomaly detection model"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.optimizer.zero_grad()
        reconstructed = self.model(state)
        loss = F.mse_loss(reconstructed, state)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

class PredictiveMaintenanceModel(nn.Module):
    def __init__(self, input_dim):
        super(PredictiveMaintenanceModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True)
        self.attention = nn.MultiheadAttention(128, 4)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 3)  # 3 outputs: failure probability, time to failure, severity
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        x = F.relu(self.fc1(attn_out[:, -1, :]))
        return self.fc2(x)

class PredictiveMaintenanceAgent:
    def __init__(self, env):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state_dim = env.observation_space.shape[0]
        self.model = PredictiveMaintenanceModel(state_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Adjust history length to match state dimension
        self.history_length = state_dim  # Changed from 100 to match state dimension
        self.state_history = deque(maxlen=self.history_length)
        self.failure_history = deque(maxlen=1000)
        
        # Initialize state history with zeros
        for _ in range(self.history_length):
            self.state_history.append(np.zeros(state_dim))
        
        # Prediction thresholds
        self.failure_threshold = 0.7
        self.warning_threshold = 0.4
        
        # Maintenance tracking
        self.maintenance_schedule = {}
        self.last_maintenance = {}
        self.current_predictions = {}
        
    def predict_failures(self, state):
        # Update state history
        self.state_history.append(state)
        
        # Convert state history to numpy array first
        state_history_array = np.array(list(self.state_history))
        
        # Prepare input tensor
        state_tensor = torch.FloatTensor(state_history_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(state_tensor)
            failure_prob, time_to_failure, severity = predictions[0]
            
            # Update current predictions for each edge
            for edge in self.env.network.edges():
                edge_state = self._get_edge_metrics(edge)
                
                # Calculate edge-specific failure probability based on metrics
                edge_failure_prob = (
                    edge_state['packet_loss'] * 0.4 +
                    (edge_state['latency'] / 20.0) * 0.3 +
                    edge_state['bandwidth'] * 0.2 +
                    (1 - edge_state['throughput']) * 0.1
                )
                
                # Schedule maintenance if metrics indicate poor performance
                if (edge_failure_prob > 0.6 or  # High failure probability
                    edge_state['packet_loss'] > 0.3 or  # High packet loss
                    edge_state['latency'] > 15 or  # High latency
                    edge_state['bandwidth'] > 0.8):  # High bandwidth utilization
                    
                    # Schedule immediate maintenance
                    self._schedule_maintenance(edge, 0)
                
        return (failure_prob.item(), 
                max(0, time_to_failure.item()), 
                severity.item())
    
    def _get_edge_metrics(self, edge):
        """Get current metrics for an edge"""
        return {
            'bandwidth': self.env.bandwidth_utilization.get(edge, 0),
            'latency': self.env.latency.get(edge, 0),
            'packet_loss': self.env.packet_loss.get(edge, 0),
            'throughput': self.env.throughput.get(edge, 0)
        }
    
    def _schedule_maintenance(self, edge, time_to_failure):
        """Schedule maintenance for an edge"""
        if edge not in self.maintenance_schedule or \
           self.maintenance_schedule[edge] > time_to_failure:
            self.maintenance_schedule[edge] = time_to_failure
    
    def perform_maintenance(self):
        """Perform scheduled maintenance"""
        maintenance_performed = []
        
        for edge, scheduled_time in list(self.maintenance_schedule.items()):
            if scheduled_time <= 0:  # Time to perform maintenance
                # Record metrics before maintenance
                metrics_before = self._get_edge_metrics(edge)
                
                # Apply maintenance
                self._apply_maintenance(edge)
                
                # Record metrics after maintenance
                metrics_after = self._get_edge_metrics(edge)
                
                # Determine maintenance type based on severity
                maintenance_type = "Critical" if metrics_before['packet_loss'] > 0.3 else "Routine"
                
                # Add to maintenance history
                maintenance_performed.append({
                    'edge': edge,
                    'type': maintenance_type,
                    'metrics_before': metrics_before,
                    'metrics_after': metrics_after
                })
                
                del self.maintenance_schedule[edge]
                self.last_maintenance[edge] = self.env.packet_counter
                
        return maintenance_performed
    
    def _apply_maintenance(self, edge):
        """Apply maintenance effects to an edge"""
        # Store previous metrics
        metrics_before = self._get_edge_metrics(edge)
        
        # Apply more significant improvements
        self.env.bandwidth_utilization[edge] = max(0.2, 
            self.env.bandwidth_utilization[edge] * 0.5)  # More reduction
        self.env.latency[edge] = max(1, 
            self.env.latency[edge] * 0.6)  # More reduction
        self.env.packet_loss[edge] = max(0.01, 
            self.env.packet_loss[edge] * 0.4)  # More reduction
        self.env.throughput[edge] = min(1.0, 
            self.env.throughput[edge] * 1.5)  # More improvement
        
        # Get metrics after maintenance
        metrics_after = self._get_edge_metrics(edge)
        
        return metrics_before, metrics_after
    
    def train(self, state, failure_occurred=False):
        """Train the predictive maintenance model"""
        # Update state history
        self.state_history.append(state)
        
        if failure_occurred:
            # Convert state history to numpy array before storing
            state_history_array = np.array(list(self.state_history))
            self.failure_history.append((state_history_array, 1))
        elif len(self.state_history) == self.history_length:
            state_history_array = np.array(list(self.state_history))
            self.failure_history.append((state_history_array, 0))
        
        if len(self.failure_history) < 32:  # Minimum batch size
            return
        
        # Sample batch
        batch = random.sample(self.failure_history, 32)
        states, labels = zip(*batch)
        
        # Convert states to tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)
        
        # Train step
        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = F.mse_loss(predictions[:, 0], labels)  # Focus on failure probability
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_status_report(self):
        """Generate a status report of network health"""
        report = {
            'high_risk_edges': [],
            'warning_edges': [],
            'healthy_edges': [],
            'maintenance_scheduled': list(self.maintenance_schedule.keys()),
            'recently_maintained': [
                edge for edge, time in self.last_maintenance.items() 
                if self.env.packet_counter - time < 100
            ]
        }
        
        for edge, pred in self.current_predictions.items():
            if pred['failure_probability'] > self.failure_threshold:
                report['high_risk_edges'].append(edge)
            elif pred['failure_probability'] > self.warning_threshold:
                report['warning_edges'].append(edge)
            else:
                report['healthy_edges'].append(edge)
                
        return report