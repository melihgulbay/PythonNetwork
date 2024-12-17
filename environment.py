# environment.py
import gym
from gym import spaces
import numpy as np
import networkx as nx
import random

class Packet:
    def __init__(self, source, destination, packet_type="data", size=1024, priority=0):
        self.source = source
        self.destination = destination
        self.packet_type = packet_type  # data, control, or priority
        self.size = size  # in bytes
        self.priority = priority  # 0-3, higher means more priority
        self.creation_time = 0
        self.current_node = source
        self.path = [source]
        self.dropped = False
        self.delivered = False
        
        # Packet statistics
        self.latency = 0
        self.hops = 0

class NetworkSimEnvironment(gym.Env):
    def __init__(self, num_nodes=6):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(num_nodes * (num_nodes - 1))  # Possible routing decisions
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_nodes * num_nodes + 4,), dtype=np.float32
        )
        
        # Initialize network topology with random geometric graph instead of complete graph
        # This creates a more realistic network where not all nodes are directly connected
        pos = {i: (random.random(), random.random()) for i in range(num_nodes)}
        self.network = nx.random_geometric_graph(num_nodes, radius=0.5, pos=pos)
        
        # Ensure the graph is connected (all nodes can reach each other)
        while not nx.is_connected(self.network):
            # Add edges between closest disconnected components until graph is connected
            components = list(nx.connected_components(self.network))
            if len(components) > 1:
                comp1 = list(components[0])
                comp2 = list(components[1])
                self.network.add_edge(comp1[0], comp2[0])
        
        self.reset()
        
        # Add performance thresholds
        self.max_latency = 20.0
        self.max_packet_loss = 0.3
        self.min_throughput = 0.2
        
        # Add metric bounds
        self.metric_bounds = {
            'bandwidth': (0.0, 1.0),    # Utilization between 0-100%
            'latency': (1.0, 20.0),     # Latency between 1-20ms
            'packet_loss': (0.0, 0.3),  # Loss rate between 0-30%
            'throughput': (0.2, 1.0)    # Throughput between 20-100%
        }
        
        # Add anomaly-related parameters
        self.anomaly_probability = 0.05  # 5% chance of anomaly per step
        self.current_anomaly = None
        self.anomaly_duration = 0
        self.anomaly_types = {
            'dos': {
                'duration': (20, 50),  # Duration range in steps
                'effects': {
                    'bandwidth': (0.8, 1.0),  # High utilization
                    'latency': (2.0, 4.0),    # Multiplier for increased latency
                    'packet_loss': (0.2, 0.4), # High packet loss
                    'throughput': (0.2, 0.4)   # Reduced throughput
                }
            },
            'port_scan': {
                'duration': (10, 30),
                'effects': {
                    'bandwidth': (0.3, 0.5),
                    'latency': (1.2, 1.5),
                    'packet_loss': (0.05, 0.1),
                    'throughput': (0.6, 0.8)
                }
            },
            'data_exfiltration': {
                'duration': (30, 60),
                'effects': {
                    'bandwidth': (0.6, 0.8),
                    'latency': (1.1, 1.3),
                    'packet_loss': (0.01, 0.05),
                    'throughput': (0.7, 0.9)
                }
            }
        }
        
        self.anomaly_detection_enabled = False
        
        # Add packet-related attributes
        self.packets = []
        self.max_packets = 1000
        self.packet_types = ["data", "control", "priority"]
        self.packet_counter = 0
        self.delivered_packets = []
        self.dropped_packets = []
        
        # Packet generation parameters
        self.packet_generation_rate = 0.7  # Increased from 0.3
        self.min_packets_per_step = 2  # Minimum packets to generate per step
        self.max_packets_per_step = 5  # Maximum packets to generate per step
        self.packet_size_range = (512, 2048)  # bytes

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
    
    def _generate_anomaly(self):
        """Randomly generate network anomalies"""
        if self.current_anomaly is None and random.random() < self.anomaly_probability:
            # Start new anomaly
            anomaly_type = random.choice(list(self.anomaly_types.keys()))
            duration_range = self.anomaly_types[anomaly_type]['duration']
            self.current_anomaly = {
                'type': anomaly_type,
                'duration': random.randint(*duration_range),
                'affected_edges': random.sample(list(self.network.edges()), 
                                             k=random.randint(1, len(self.network.edges())//2))
            }
            self.anomaly_duration = 0
            return True
        return False

    def _apply_anomaly_effects(self):
        """Apply effects of current anomaly to the network"""
        if self.current_anomaly:
            anomaly_type = self.current_anomaly['type']
            effects = self.anomaly_types[anomaly_type]['effects']
            
            for edge in self.current_anomaly['affected_edges']:
                # Apply anomaly effects with bounds
                self.bandwidth_utilization[edge] = min(self.metric_bounds['bandwidth'][1], 
                    max(self.metric_bounds['bandwidth'][0],
                        random.uniform(*effects['bandwidth'])))
                
                self.latency[edge] = min(self.metric_bounds['latency'][1],
                    max(self.metric_bounds['latency'][0],
                        self.latency[edge] * random.uniform(*effects['latency'])))
                
                self.packet_loss[edge] = min(self.metric_bounds['packet_loss'][1],
                    max(self.metric_bounds['packet_loss'][0],
                        random.uniform(*effects['packet_loss'])))
                
                self.throughput[edge] = min(self.metric_bounds['throughput'][1],
                    max(self.metric_bounds['throughput'][0],
                        random.uniform(*effects['throughput'])))

            self.anomaly_duration += 1
            
            # Check if anomaly should end
            if self.anomaly_duration >= self.current_anomaly['duration']:
                self.current_anomaly = None

    def set_anomaly_detection(self, enabled):
        self.anomaly_detection_enabled = enabled

    def generate_packet(self):
        """Generate multiple new packets with random source and destination"""
        # Determine number of packets to generate this step
        num_packets = random.randint(self.min_packets_per_step, self.max_packets_per_step)
        
        for _ in range(num_packets):
            if len(self.packets) < self.max_packets and random.random() < self.packet_generation_rate:
                source = random.randint(0, self.num_nodes - 1)
                destination = random.randint(0, self.num_nodes - 1)
                while destination == source:
                    destination = random.randint(0, self.num_nodes - 1)
                
                packet_type = random.choice(self.packet_types)
                size = random.randint(*self.packet_size_range)
                priority = random.randint(0, 3)
                
                packet = Packet(source, destination, packet_type, size, priority)
                packet.creation_time = self.packet_counter
                self.packets.append(packet)
                self.packet_counter += 1

    def process_packets(self):
        """Process all active packets in the network"""
        for packet in self.packets[:]:  # Create a copy to allow modification during iteration
            if packet.dropped or packet.delivered:
                continue
            
            # Calculate next hop based on current routing policy
            current_node = packet.current_node
            if current_node == packet.destination:
                packet.delivered = True
                packet.latency = self.packet_counter - packet.creation_time
                self.delivered_packets.append(packet)
                self.packets.remove(packet)
                continue
            
            # Check for packet loss
            if random.random() < self.packet_loss.get((current_node, packet.destination), 0):
                packet.dropped = True
                self.dropped_packets.append(packet)
                self.packets.remove(packet)
                continue
            
            # Move packet to next hop
            next_hop = self._get_next_hop(current_node, packet.destination)
            if next_hop is not None:
                packet.current_node = next_hop
                packet.path.append(next_hop)
                # Increment hops counter when packet moves to a new node
                if next_hop != packet.source:  # Don't count the source node as a hop
                    packet.hops += 1
            
    def _get_next_hop(self, current_node, destination):
        """Determine next hop for packet routing"""
        try:
            # If there's a direct connection, use it
            if (current_node, destination) in self.network.edges() or (destination, current_node) in self.network.edges():
                return destination
            
            # Otherwise, get the next hop from the shortest path
            path = nx.shortest_path(self.network, current_node, destination)
            if len(path) > 1:
                return path[1]
            return None
            
        except nx.NetworkXNoPath:
            return None

    def step(self, action):
        # Generate and process packets
        self.generate_packet()
        self.process_packets()
        
        # Only generate anomalies if anomaly detection is enabled
        if self.anomaly_detection_enabled:
            self._generate_anomaly()
            self._apply_anomaly_effects()
        
        # Convert action to source-destination pair
        src = action // (self.num_nodes - 1)
        dst = action % (self.num_nodes - 1)
        if dst >= src:
            dst += 1
            
        edge = (src, dst) if (src, dst) in self.network.edges() else (dst, src)
        if edge in self.network.edges():
            # Only apply improvements if edge is not affected by anomaly
            if not self.current_anomaly or edge not in self.current_anomaly['affected_edges']:
                # Apply changes with bounds
                self.bandwidth_utilization[edge] = min(self.metric_bounds['bandwidth'][1],
                    max(self.metric_bounds['bandwidth'][0],
                        self.bandwidth_utilization[edge] - 0.15))
                
                self.latency[edge] = min(self.metric_bounds['latency'][1],
                    max(self.metric_bounds['latency'][0],
                        self.latency[edge] * 0.85))
                
                self.packet_loss[edge] = min(self.metric_bounds['packet_loss'][1],
                    max(self.metric_bounds['packet_loss'][0],
                        self.packet_loss[edge] * 0.85))
                
                self.throughput[edge] = min(self.metric_bounds['throughput'][1],
                    max(self.metric_bounds['throughput'][0],
                        self.throughput[edge] * 1.2))
            
        # Calculate reward
        bandwidth_reward = (1 - np.mean(list(self.bandwidth_utilization.values()))) * 0.3
        latency_reward = (1 - np.clip(np.mean(list(self.latency.values())) / self.max_latency, 0, 1)) * 0.3
        packet_loss_reward = (1 - np.clip(np.mean(list(self.packet_loss.values())) / self.max_packet_loss, 0, 1)) * 0.2
        throughput_reward = np.clip(np.mean(list(self.throughput.values())) / self.min_throughput, 0, 1) * 0.2
        
        reward = bandwidth_reward + latency_reward + packet_loss_reward + throughput_reward
        
        # Add anomaly penalty to reward
        if self.current_anomaly:
            reward *= 0.5  # Reduce reward during anomalies
        
        done = (np.mean(list(self.latency.values())) > self.max_latency or
                np.mean(list(self.packet_loss.values())) > self.max_packet_loss or
                np.mean(list(self.throughput.values())) < self.min_throughput)
        
        info = {
            'anomaly': self.current_anomaly['type'] if self.current_anomaly else None,
            'affected_edges': self.current_anomaly['affected_edges'] if self.current_anomaly else [],
            'active_packets': len(self.packets),
            'delivered_packets': len(self.delivered_packets),
            'dropped_packets': len(self.dropped_packets),
            'average_latency': np.mean([p.latency for p in self.delivered_packets]) if self.delivered_packets else 0,
            'average_hops': np.mean([p.hops for p in self.delivered_packets]) if self.delivered_packets else 0
        }
        
        return self._get_state(), reward, done, info