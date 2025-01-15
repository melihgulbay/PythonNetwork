import numpy as np
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from collections import deque
import random
from environment import SDNLayer

class MitigationNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(MitigationNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 4),  # 4 different mitigation actions
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class NetworkAnomalyDetector:
    def __init__(self, num_features=4):
        # Anomaly Detection
        self.detector = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        self.num_features = num_features
        self.training_data = []
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
        
        # Mitigation
        self.mitigation_network = MitigationNetwork(num_features)
        self.optimizer = optim.Adam(self.mitigation_network.parameters(), lr=0.005)
        self.memory = deque(maxlen=1000)
        self.batch_size = 64
        self.gamma = 0.99
        self.min_memory_size = 100
        
        # Prioritized experience replay
        self.priorities = deque(maxlen=1000)
        self.priority_scale = 0.7
        
        # Mitigation success tracking
        self.mitigation_success = deque(maxlen=100)
        self.current_mitigation = None
        
        # Mitigation strategies
        self.mitigation_strategies = [
            self.reduce_load,
            self.reroute_traffic,
            self.adjust_qos,
            self.isolate_affected_area
        ]
        
    def collect_sample(self, metrics):
        """Collect a sample of network metrics for training/detection"""
        features = np.array([
            metrics.get('bandwidth', 0),
            metrics.get('latency', 0),
            metrics.get('packet_loss', 0),
            metrics.get('throughput', 0)
        ]).reshape(1, -1)
        
        if not self.is_trained:
            self.training_data.append(features[0])
            if len(self.training_data) >= 100:
                self.train()
                
        return features
        
    def train(self):
        """Train the anomaly detector"""
        if len(self.training_data) < 50:
            return False
            
        try:
            X = np.array(self.training_data)
            self.detector.fit(X)
            self.is_trained = True
            self.logger.info("Anomaly detector trained successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to train anomaly detector: {str(e)}")
            return False
            
    def detect(self, metrics):
        """Detect anomalies in current network state"""
        if not self.is_trained:
            return False
            
        features = self.collect_sample(metrics)
        try:
            prediction = self.detector.predict(features)
            return prediction[0] == -1
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {str(e)}")
            return False

    def select_mitigation(self, state):
        """Select mitigation strategy using the neural network with exploration"""
        # Add exploration factor
        if random.random() < 0.2:  # 20% chance to explore
            return random.randint(0, len(self.mitigation_strategies) - 1)
            
        state_tensor = torch.FloatTensor(state)
        with torch.no_grad():
            action_probs = self.mitigation_network(state_tensor)
            # Add some randomness to prevent always selecting the same action
            action_probs = action_probs + torch.randn_like(action_probs) * 0.1
            action = torch.argmax(action_probs).item()
        return action

    def mitigate(self, metrics, env):
        """Apply mitigation strategy to detected anomaly"""
        if not env.anomaly_active:
            return False

        # Get current state
        state = self.collect_sample(metrics)[0]
        
        # Select mitigation strategy
        action = self.select_mitigation(state)
        
        # Store initial metrics before mitigation
        initial_metrics = {
            'bandwidth': np.mean([v for v in env.bandwidth_utilization.values() if v is not None]),
            'latency': np.mean([v for v in env.latency.values() if v is not None]),
            'packet_loss': np.mean([v for v in env.packet_loss.values() if v is not None]),
            'throughput': np.mean([v for v in env.throughput.values() if v is not None])
        }
        
        # Apply strategy
        try:
            self.mitigation_strategies[action](env)
        except Exception as e:
            self.logger.error(f"Mitigation strategy encountered error: {str(e)}")
        
        # Get final metrics after mitigation
        final_metrics = {
            'bandwidth': np.mean([v for v in env.bandwidth_utilization.values() if v is not None]),
            'latency': np.mean([v for v in env.latency.values() if v is not None]),
            'packet_loss': np.mean([v for v in env.packet_loss.values() if v is not None]),
            'throughput': np.mean([v for v in env.throughput.values() if v is not None])
        }
        
        # Calculate improvement with stronger scaling
        improvement = 0
        if env.current_anomaly['type'] == 'bandwidth_surge':
            improvement = (initial_metrics['bandwidth'] - final_metrics['bandwidth']) / initial_metrics['bandwidth']
        elif env.current_anomaly['type'] == 'latency_spike':
            improvement = (initial_metrics['latency'] - final_metrics['latency']) / initial_metrics['latency']
        elif env.current_anomaly['type'] == 'packet_loss_burst':
            improvement = (initial_metrics['packet_loss'] - final_metrics['packet_loss']) / initial_metrics['packet_loss']
        elif env.current_anomaly['type'] == 'throughput_drop':
            improvement = (final_metrics['throughput'] - initial_metrics['throughput']) / initial_metrics['throughput']
        
        # Enhanced reward calculation with stronger scaling
        success_percentage = max(0.3, min(1.0, 0.5 + improvement * 2.0))
        
        # Calculate priority for experience
        priority = abs(success_percentage - 0.5) + 0.01
        
        # Store experience with priority
        self.memory.append((state, action, success_percentage))
        self.priorities.append(priority)
        
        # Update success tracking
        self.mitigation_success.append(success_percentage)
        self.current_mitigation = self.mitigation_strategies[action].__name__
        
        # Train more frequently
        if len(self.memory) >= self.min_memory_size:
            for _ in range(3):  # Train multiple times per mitigation
                self.train_mitigation()
        
        return True

    def train_mitigation(self):
        """Train the mitigation network using prioritized experience replay"""
        if len(self.memory) < self.min_memory_size:
            return

        # Calculate sampling probabilities based on priorities
        probs = np.array(self.priorities) ** self.priority_scale
        probs /= probs.sum()
        
        # Sample batch with priorities
        batch_indices = np.random.choice(len(self.memory), 
                                       size=min(self.batch_size, len(self.memory)), 
                                       p=probs)
        
        # Prepare batch
        batch = [self.memory[idx] for idx in batch_indices]
        states, actions, rewards = zip(*batch)

        # Convert to tensors with stronger reward scaling
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards) * 2.0

        # Get current action probabilities
        action_probs = self.mitigation_network(states)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1))

        # Calculate loss with importance sampling weights
        importance = 1.0 / (len(self.memory) * probs[batch_indices])
        importance = importance / importance.max()
        loss = -(torch.FloatTensor(importance).unsqueeze(1) * 
                torch.log(selected_action_probs) * 
                rewards.unsqueeze(1)).mean()

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mitigation_network.parameters(), 1.0)
        self.optimizer.step()

    # Mitigation Strategies
    def reduce_load(self, env):
        """Reduce load on affected edges"""
        try:
            if env.current_anomaly and env.current_anomaly['type'] in ['bandwidth_surge', 'throughput_drop']:
                affected_edges = env.current_anomaly['edges']
                changes_made = False
                
                for edge in affected_edges:
                    current_util = env.bandwidth_utilization.get(edge, 0)
                    if current_util > 0.4:  # Lower threshold for intervention
                        # More aggressive reduction based on current utilization
                        reduction_factor = 0.5 if current_util > 0.8 else 0.7
                        env.bandwidth_utilization[edge] = max(0.3, current_util * reduction_factor)
                        changes_made = True
                        
                self.logger.info(f"Load reduction applied to {len(affected_edges)} edges")
                return changes_made
            return False
        except Exception as e:
            self.logger.error(f"Load reduction failed: {str(e)}")
            return False

    def reroute_traffic(self, env):
        """Reroute traffic around affected area"""
        try:
            if env.current_anomaly and env.current_anomaly['type'] in ['latency_spike', 'packet_loss_burst']:
                affected_edges = set(env.current_anomaly['edges'])
                changes_made = False
                
                for src, dst in affected_edges:
                    # Clear existing flow table entries
                    if src in env.sdn_layers[SDNLayer.CONTROL]['flow_tables']:
                        env.sdn_layers[SDNLayer.CONTROL]['flow_tables'][src] = []
                        changes_made = True
                    if dst in env.sdn_layers[SDNLayer.CONTROL]['flow_tables']:
                        env.sdn_layers[SDNLayer.CONTROL]['flow_tables'][dst] = []
                        changes_made = True
                        
                    # Reduce bandwidth utilization to discourage traffic
                    edge = (src, dst)
                    if edge in env.bandwidth_utilization:
                        env.bandwidth_utilization[edge] = max(0.2, env.bandwidth_utilization[edge] * 0.5)
                        changes_made = True
                        
                return changes_made
            return False
        except Exception as e:
            self.logger.error(f"Traffic rerouting failed: {str(e)}")
            return False

    def adjust_qos(self, env):
        """Adjust QoS parameters"""
        try:
            if env.current_anomaly and env.current_anomaly['type'] in ['latency_spike', 'packet_loss_burst']:
                for edge in env.current_anomaly['edges']:
                    affected_packets = [p for p in env.packets if p.current_node in edge]
                    for packet in affected_packets:
                        if packet.priority < 2:  # Only increase if priority is low
                            packet.priority += 1
                return bool(affected_packets)  # Return True only if packets were affected
            return False
        except Exception as e:
            self.logger.error(f"QoS adjustment failed: {str(e)}")
            return False

    def isolate_affected_area(self, env):
        """Temporarily isolate affected network segment"""
        try:
            if env.current_anomaly and env.current_anomaly['type'] in ['bandwidth_surge', 'packet_loss_burst']:
                total_edges = len(env.network.edges())
                affected_edges = env.current_anomaly['edges']
                # Only isolate if it won't disconnect too much of the network
                if len(affected_edges) < total_edges * 0.3:
                    for edge in affected_edges:
                        env.bandwidth_utilization[edge] = 0.1  # Reduce to minimum instead of complete isolation
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Area isolation failed: {str(e)}")
            return False

    def get_success_rate(self):
        """Get the current mitigation success rate"""
        if not self.mitigation_success:
            return 0.0
        return sum(self.mitigation_success) / len(self.mitigation_success)

    def get_current_strategy(self):
        """Get the name of the current mitigation strategy"""
        return self.current_mitigation if self.current_mitigation else "None" 