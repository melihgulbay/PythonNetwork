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
    def __init__(self, input_dim, hidden_dim=64):
        super(MitigationNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 4),  # 4 different mitigation actions
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
        self.optimizer = optim.Adam(self.mitigation_network.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        
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
        
        # Add logging to debug strategy selection
        self.logger.info(f"Selected mitigation strategy: {self.mitigation_strategies[action].__name__}")
        
        # Apply strategy and evaluate its success
        initial_metrics = {
            'bandwidth': np.mean(list(env.bandwidth_utilization.values())),
            'latency': np.mean(list(env.latency.values())),
            'packet_loss': np.mean(list(env.packet_loss.values())),
            'throughput': np.mean(list(env.throughput.values()))
        }
        
        # Apply strategy with error handling
        try:
            strategy_applied = self.mitigation_strategies[action](env)
            if not strategy_applied:
                self.logger.warning(f"Strategy {self.mitigation_strategies[action].__name__} failed to apply")
                self.mitigation_success.append(False)
                return False
        except Exception as e:
            self.logger.error(f"Mitigation strategy failed: {str(e)}")
            self.mitigation_success.append(False)
            return False
            
        # Evaluate success by comparing metrics
        final_metrics = {
            'bandwidth': np.mean(list(env.bandwidth_utilization.values())),
            'latency': np.mean(list(env.latency.values())),
            'packet_loss': np.mean(list(env.packet_loss.values())),
            'throughput': np.mean(list(env.throughput.values()))
        }
        
        # Calculate success based on improvement in relevant metrics
        success = False
        if env.current_anomaly['type'] == 'bandwidth_surge':
            success = final_metrics['bandwidth'] < initial_metrics['bandwidth']
        elif env.current_anomaly['type'] == 'latency_spike':
            success = final_metrics['latency'] < initial_metrics['latency']
        elif env.current_anomaly['type'] == 'packet_loss_burst':
            success = final_metrics['packet_loss'] < initial_metrics['packet_loss']
        elif env.current_anomaly['type'] == 'throughput_drop':
            success = final_metrics['throughput'] > initial_metrics['throughput']
        
        # Store experience for training
        reward = 1.0 if success else -0.1
        self.memory.append((state, action, reward))
        
        # Update success tracking
        self.mitigation_success.append(success)
        self.current_mitigation = self.mitigation_strategies[action].__name__
        
        # Train mitigation network
        self.train_mitigation()
        
        return success

    def train_mitigation(self):
        """Train the mitigation network using collected experiences"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)

        # Get current action probabilities
        action_probs = self.mitigation_network(states)
        selected_action_probs = action_probs.gather(1, actions.unsqueeze(1))

        # Calculate loss (using policy gradient)
        loss = -(torch.log(selected_action_probs) * rewards.unsqueeze(1)).mean()

        # Update network
        self.optimizer.zero_grad()
        loss.backward()
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
                    # Lower the threshold and increase reduction
                    if current_util > 0.5:  # Changed from 0.7
                        env.bandwidth_utilization[edge] = max(0.3, current_util * 0.6)  # More aggressive reduction
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
                for src, dst in affected_edges:
                    # Clear existing flow table entries for affected nodes
                    env.sdn_layers[SDNLayer.CONTROL]['flow_tables'][src] = []
                    env.sdn_layers[SDNLayer.CONTROL]['flow_tables'][dst] = []
                return True
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