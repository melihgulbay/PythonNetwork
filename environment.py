# environment.py
import gym
from gym import spaces
import numpy as np
import networkx as nx
import random
from enum import Enum
import copy

class SDNLayer(Enum):
    APPLICATION = "application"
    CONTROL = "control"
    INFRASTRUCTURE = "infrastructure"
    def get_state(self):
        """Return a copy of the current environment state"""
        return {
            'packets': copy.deepcopy(self.packets),
            'delivered_packets': copy.deepcopy(self.delivered_packets),
            'dropped_packets': copy.deepcopy(self.dropped_packets),
            'bandwidth_utilization': copy.deepcopy(self.bandwidth_utilization),
            'latency': copy.deepcopy(self.latency),
            'packet_loss': copy.deepcopy(self.packet_loss),
            'throughput': copy.deepcopy(self.throughput),
            'anomaly_active': self.anomaly_active,
            'current_anomaly': copy.deepcopy(self.current_anomaly)
        }

    def restore_state(self, state):
        """Restore environment to a previous state"""
        self.packets = state['packets']
        self.delivered_packets = state['delivered_packets']
        self.dropped_packets = state['dropped_packets']
        self.bandwidth_utilization = state['bandwidth_utilization']
        self.latency = state['latency']
        self.packet_loss = state['packet_loss']
        self.throughput = state['throughput']
        self.anomaly_active = state['anomaly_active']
        self.current_anomaly = state['current_anomaly']

class Protocol(Enum):
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    ICMP = "icmp"

class Packet:
    def __init__(self, source, destination, packet_type="data", size=1024, priority=0, protocol=Protocol.TCP):
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

        self.protocol = protocol
        
        # Protocol-specific attributes
        self.protocol_state = {
            'sequence_number': 0,
            'ack_number': 0,
            'window_size': 65535,
            'retransmission_count': 0,
            'timeout': False,
            'connection_established': False
        }
        
        # Protocol-specific configurations
        self.protocol_config = {
            Protocol.TCP: {
                'requires_ack': True,
                'window_size': 65535,
                'congestion_control': True,
                'max_retransmissions': 3,
                'timeout_ms': 1000
            },
            Protocol.UDP: {
                'requires_ack': False,
                'reliable': False,
                'max_datagram_size': 65507
            },
            Protocol.HTTP: {
                'method': 'GET',
                'response_code': 200,
                'headers': {},
                'requires_tcp': True
            }
        }

class SDNPacket(Packet):
    def __init__(self, source, destination, packet_type="data", size=1024, priority=0, sdn_layer=SDNLayer.INFRASTRUCTURE, protocol=Protocol.TCP):
        super().__init__(source, destination, packet_type, size, priority, protocol)
        self.sdn_layer = sdn_layer
        self.flow_id = None  # For flow table matching
        self.qos_requirements = {
            'min_bandwidth': 0,
            'max_latency': float('inf'),
            'priority': priority
        }

class FlowTableEntry:
    def __init__(self, match_fields, actions, priority=0, timeout=None):
        self.match_fields = match_fields  # Dict of fields to match
        self.actions = actions  # List of actions to take
        self.priority = priority
        self.timeout = timeout
        self.packet_count = 0
        self.byte_count = 0
        self.last_used = 0

class SDNSwitch:
    def __init__(self, switch_id):
        self.switch_id = switch_id
        self.flow_table = []
        self.buffer = []
        self.controller_connection = True
        self.stats = {
            'processed_packets': 0,
            'dropped_packets': 0,
            'forwarded_packets': 0
        }

class NetworkSimEnvironment(gym.Env):
    def __init__(self, num_nodes=20):
        super().__init__()
        self.num_nodes = num_nodes
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(num_nodes * (num_nodes - 1))
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_nodes * num_nodes + 4,), dtype=np.float32
        )
        
        # Calculate radius based on number of nodes to maintain reasonable connectivity
        radius = min(0.5, 1.5 / np.sqrt(num_nodes))
        
        # Initialize node positions with more spread (using a larger area)
        pos = {}
        for i in range(num_nodes):
            # Try to place each node with minimum distance from others
            while True:
                # Scale positions to [0.1, 0.9] instead of [0, 1] to avoid edge clustering
                x = random.uniform(0.1, 0.9)
                y = random.uniform(0.1, 0.9)
                
                # Check distance from existing nodes
                min_distance = float('inf')
                for j in range(i):
                    dx = x - pos[j][0]
                    dy = y - pos[j][1]
                    distance = (dx*dx + dy*dy) ** 0.5
                    min_distance = min(min_distance, distance)
                
                # Accept position if it's far enough from other nodes
                if i == 0 or min_distance > 0.15:  # Minimum distance threshold
                    pos[i] = (x, y)
                    break
        
        # Create network with the spread-out positions
        self.network = nx.random_geometric_graph(num_nodes, radius=radius, pos=pos)
        
        # Ensure the graph is connected but not over-connected
        # First, ensure basic connectivity
        while not nx.is_connected(self.network):
            components = list(nx.connected_components(self.network))
            if len(components) > 1:
                comp1 = list(components[0])
                comp2 = list(components[1])
                self.network.add_edge(comp1[0], comp2[0])
        
        # Then, optionally remove excess edges if the graph is too dense
        target_avg_degree = 4  # Desired average number of connections per node
        current_avg_degree = sum(dict(self.network.degree()).values()) / num_nodes
        
        while current_avg_degree > target_avg_degree:
            # Get edges sorted by their geometric distance
            edges = list(self.network.edges())
            if not edges:
                break
                
            # Remove an edge that doesn't disconnect the graph
            for edge in sorted(edges, 
                             key=lambda e: ((pos[e[0]][0] - pos[e[1]][0])**2 + 
                                          (pos[e[0]][1] - pos[e[1]][1])**2)**0.5,
                             reverse=True):  # Start with longest edges
                if len(list(nx.edge_disjoint_paths(self.network, edge[0], edge[1]))) > 1:
                    self.network.remove_edge(*edge)
                    break
            
            current_avg_degree = sum(dict(self.network.degree()).values()) / num_nodes

        # Add performance thresholds
        self.max_latency = 20.0
        self.max_packet_loss = 0.5
        self.min_throughput = 0.2
        
        # Add metric bounds
        self.metric_bounds = {
            'bandwidth': (0.0, 1.0),
            'latency': (1.0, 20.0),
            'packet_loss': (0.0, 0.3),
            'throughput': (0.2, 1.0)
        }
        
        # Add packet-related attributes
        self.packets = []
        self.max_packets = 1000
        self.packet_types = ["data", "control", "priority"]
        self.packet_counter = 0
        self.delivered_packets = []
        self.dropped_packets = []
        
        # Packet generation parameters
        self.packet_generation_rate = 0.4
        self.min_packets_per_step = 2
        self.max_packets_per_step = 4
        self.packet_size_range = (64, 1500)
        
        # Modify congestion thresholds to be more lenient
        self.congestion_levels = {
            'normal': 0.4,      # Was 0.3
            'moderate': 0.6,    # Was 0.5
            'high': 0.75,       # Was 0.65
            'critical': 0.9     # Was 0.8
        }
        
        # Reduce base drop probabilities
        self.drop_probabilities = {
            'normal': 0.05,     # Was 0.1
            'moderate': 0.15,   # Was 0.3
            'high': 0.25,       # Was 0.5
            'critical': 0.4     # Was 0.9
        }
        
        # Adjust congestion factors to give more weight to agent actions
        self.congestion_factors = {
            'packet_size_weight': 0.4,      # Was 0.3
            'priority_impact': 0.3,         # Was 0.2 (increased priority impact)
            'path_length_penalty': 0.3,     # Was 0.15
            'buffer_threshold': 75,         # Was 50 (increased buffer size)
            'congestion_memory': 0.7,       # Was 0.9 (reduced memory effect)
            'agent_action_impact': 0.5      # New factor for RL agent actions
        }
        
        # Add congestion recovery rate
        self.congestion_recovery_rate = 0.05  # How much congestion can be reduced per action
        
        # Initialize SDN layers before reset
        self.sdn_layers = {
            SDNLayer.APPLICATION: {
                'services': ['routing', 'load_balancing', 'security'],
                'policies': {}
            },
            SDNLayer.CONTROL: {
                'topology': self.network,
                'flow_tables': {},
                'stats': {}
            },
            SDNLayer.INFRASTRUCTURE: {
                'switches': {},
                'links': {}
            }
        }
        
        # Initialize SDN infrastructure
        self._initialize_sdn_infrastructure()
        
        # Now we can safely call reset
        self.reset()
        
        # Add anomaly-related attributes
        self.anomaly_active = False
        self.anomaly_detector_enabled = False
        self.current_anomaly = None
        self.anomaly_types = [
            'bandwidth_surge',
            'latency_spike',
            'packet_loss_burst',
            'throughput_drop'
        ]

        # Add protocol-specific configurations
        self.protocol_configs = {
            Protocol.TCP: {
                'retry_timeout': 5,
                'max_window_size': 65535,
                'min_window_size': 1024,
                'congestion_threshold': 0.8
            },
            Protocol.UDP: {
                'max_packet_loss': 0.3,
                'datagram_timeout': 2
            },
            Protocol.HTTP: {
                'request_timeout': 10,
                'max_redirects': 3
            }
        }

    def _initialize_sdn_infrastructure(self):
        """Initialize SDN infrastructure layer"""
        # Create SDN switches for each node
        for node in range(self.num_nodes):
            self.sdn_layers[SDNLayer.INFRASTRUCTURE]['switches'][node] = SDNSwitch(node)
            
            # Initialize flow tables in control layer
            self.sdn_layers[SDNLayer.CONTROL]['flow_tables'][node] = []

        # Initialize link properties
        for edge in self.network.edges():
            self.sdn_layers[SDNLayer.INFRASTRUCTURE]['links'][edge] = {
                'capacity': 1.0,  # Maximum bandwidth
                'status': 'up',
                'qos_config': {
                    'priority_queues': 4,
                    'queue_sizes': [64, 128, 256, 512]  # Different sizes for different priorities
                }
            }

    def reset(self):
        # Reset network state
        edges = list(self.network.edges())
        # Initialize metrics for both directions of each edge
        self.bandwidth_utilization = {}
        self.latency = {}
        self.packet_loss = {}
        self.throughput = {}
        
        # Randomize initial network conditions
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    if (i, j) in edges or (j, i) in edges:
                        # Completely random initial conditions
                        self.bandwidth_utilization[(i, j)] = random.uniform(0.8, 0.9)  # 80-90% utilization
                        self.latency[(i, j)] = random.uniform(1, 20)   # 1-20 units latency
                        self.packet_loss[(i, j)] = random.uniform(0.01, 0.5)  #This method is no longer used check def _update_packet_status for how packet loss is calculated
                        self.throughput[(i, j)] = random.uniform(0.2, 1.0)  # 20-100% throughput
        
        # Reset packet-related attributes
        self.packets = []
        self.delivered_packets = []
        self.dropped_packets = []
        self.packet_counter = 0
        
        # Randomize packet generation rate
        self.packet_generation_rate = random.uniform(0.2, 0.6)
        
        # Reset SDN switches
        for switch in self.sdn_layers[SDNLayer.INFRASTRUCTURE]['switches'].values():
            switch.stats = {
                'processed_packets': 0,
                'dropped_packets': 0,
                'forwarded_packets': 0
            }
            switch.flow_table = []
            switch.buffer = []
        
        return self._get_state()
    
    def _get_state(self):
        # Calculate actual packet loss rate based on delivered vs dropped packets
        total_packets = len(self.delivered_packets) + len(self.dropped_packets)
        actual_loss_rate = (len(self.dropped_packets) / total_packets) if total_packets > 0 else 0.0
        
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
                    
        # Add global metrics with actual packet loss rate
        state.extend([
            np.mean(list(self.bandwidth_utilization.values())),
            np.mean(list(self.latency.values())),
            actual_loss_rate,  # Use actual packet loss rate instead of link-level averages
            np.mean(list(self.throughput.values()))
        ])
        
        return np.array(state, dtype=np.float32)
    
    def generate_packet(self):
        """Generate new packets with guaranteed minimum spawning"""
        # Track packets generated this step
        packets_generated = 0
        
        # First pass: Generate minimum required packets
        while packets_generated < self.min_packets_per_step:
            # Randomly select source and destination
            source = random.randint(0, self.num_nodes - 1)
            destination = random.randint(0, self.num_nodes - 1)
            while destination == source:
                destination = random.randint(0, self.num_nodes - 1)
            
            # Randomly select protocol and priority
            protocol = random.choice(list(Protocol))
            priority = random.randint(0, 3)
            
            # Create packet
            packet = SDNPacket(
                source=source,
                destination=destination,
                packet_type="data",
                size=random.randint(*self.packet_size_range),
                priority=priority,
                protocol=protocol
            )
            
            self.packets.append(packet)
            self.packet_counter += 1
            packets_generated += 1
        
        # Second pass: Additional probabilistic packet generation
        remaining_slots = self.max_packets_per_step - packets_generated
        for _ in range(remaining_slots):
            if random.random() < self.packet_generation_rate:
                source = random.randint(0, self.num_nodes - 1)
                destination = random.randint(0, self.num_nodes - 1)
                while destination == source:
                    destination = random.randint(0, self.num_nodes - 1)
                
                protocol = random.choice(list(Protocol))
                priority = random.randint(0, 3)
                
                packet = SDNPacket(
                    source=source,
                    destination=destination,
                    packet_type="data",
                    size=random.randint(*self.packet_size_range),
                    priority=priority,
                    protocol=protocol
                )
                
                self.packets.append(packet)
                self.packet_counter += 1
                packets_generated += 1

    def process_packets(self):
        """Enhanced packet processing with SDN layers"""
        for packet in self.packets[:]:
            if packet.dropped or packet.delivered:
                continue

            # Get current switch
            current_switch = self.sdn_layers[SDNLayer.INFRASTRUCTURE]['switches'][packet.current_node]

            # Check flow table for matching rule
            flow_rule = self._match_flow_rule(current_switch, packet)

            if flow_rule:
                # Apply flow rule actions
                self._apply_flow_actions(packet, flow_rule)
            else:
                # No matching rule - send to controller
                self._send_to_controller(packet, current_switch)

            # Update metrics and check delivery/dropping
            self._update_packet_status(packet)

    def _match_flow_rule(self, switch, packet):
        """Match packet against flow table rules"""
        flow_table = self.sdn_layers[SDNLayer.CONTROL]['flow_tables'][switch.switch_id]
        
        for entry in sorted(flow_table, key=lambda x: x.priority, reverse=True):
            if self._matches_rule(packet, entry.match_fields):
                entry.packet_count += 1
                entry.byte_count += packet.size
                entry.last_used = self.packet_counter
                return entry
        return None

    def _matches_rule(self, packet, match_fields):
        """Check if packet matches flow rule fields"""
        for field, value in match_fields.items():
            if field == 'source' and packet.source != value:
                return False
            elif field == 'destination' and packet.destination != value:
                return False
            elif field == 'packet_type' and packet.packet_type != value:
                return False
            elif field == 'priority' and packet.priority != value:
                return False
        return True

    def _apply_flow_actions(self, packet, flow_rule):
        """Apply flow rule actions to packet"""
        for action in flow_rule.actions:
            if action['type'] == 'forward':
                next_hop = action['port']
                if self._is_valid_next_hop(packet.current_node, next_hop):
                    packet.current_node = next_hop
                    packet.path.append(next_hop)
                    packet.hops += 1
            elif action['type'] == 'drop':
                packet.dropped = True
            elif action['type'] == 'modify':
                setattr(packet, action['field'], action['value'])

    def _send_to_controller(self, packet, switch):
        """Send packet to SDN controller for decision"""
        # Generate new flow rule based on current network state
        next_hop = self._get_next_hop(packet.current_node, packet.destination)
        
        if next_hop is not None:
            # Create new flow rule
            new_rule = FlowTableEntry(
                match_fields={
                    'source': packet.source,
                    'destination': packet.destination,
                    'packet_type': packet.packet_type
                },
                actions=[{'type': 'forward', 'port': next_hop}],
                priority=packet.priority,
                timeout=100  # Flow rule expires after 100 steps
            )
            
            # Add rule to flow table
            self.sdn_layers[SDNLayer.CONTROL]['flow_tables'][switch.switch_id].append(new_rule)
            
            # Forward packet
            packet.current_node = next_hop
            packet.path.append(next_hop)
            packet.hops += 1

    def _update_packet_status(self, packet):
        """Enhanced packet status update with protocol handling"""
        if packet.current_node == packet.destination:
            packet.delivered = True
            packet.latency = self.packet_counter - packet.creation_time
            self.delivered_packets.append(packet)
            self.packets.remove(packet)
            return

        current_node = packet.current_node
        next_hop = self._get_next_hop(current_node, packet.destination)
        
        if next_hop is None:
            packet.dropped = True
            self.dropped_packets.append(packet)
            self.packets.remove(packet)
            return

        # Get link congestion (both directions)
        link = (current_node, next_hop)
        reverse_link = (next_hop, current_node)
        
        # Calculate current utilization with reduced memory effect
        current_util = max(
            self.bandwidth_utilization.get(link, 0),
            self.bandwidth_utilization.get(reverse_link, 0)
        )
        
        # Add historical congestion impact with less weight
        historical_util = getattr(self, '_previous_util', current_util)
        utilization = (current_util * (1 - self.congestion_factors['congestion_memory']) +
                      historical_util * self.congestion_factors['congestion_memory'])
        self._previous_util = utilization

        # Determine congestion level
        congestion_level = 'normal'
        for level, threshold in sorted(self.congestion_levels.items(), key=lambda x: x[1]):
            if utilization > threshold:
                congestion_level = level

        # Calculate base drop probability
        base_drop_prob = self.drop_probabilities[congestion_level]
        
        # Factor in packet size with reduced impact
        size_factor = (packet.size / 1024) * self.congestion_factors['packet_size_weight']
        
        # Increased priority impact (higher priority = much lower drop chance)
        priority_factor = -((3 - packet.priority) / 3) * self.congestion_factors['priority_impact']
        
        # Reduced path length penalty
        path_penalty = (len(packet.path) / self.num_nodes) * self.congestion_factors['path_length_penalty']
        
        # More forgiving buffer occupancy check
        current_buffer = len([p for p in self.packets if p.current_node == current_node])
        buffer_factor = max(0, (current_buffer - self.congestion_factors['buffer_threshold'] / 2) / 
                          self.congestion_factors['buffer_threshold'])

        # Calculate agent effectiveness (based on network metrics)
        agent_effectiveness = 1.0 - (
            self.bandwidth_utilization.get(link, 0) +
            self.packet_loss.get(link, 0) +
            (self.latency.get(link, 1.0) / self.max_latency)
        ) / 3.0

        # Apply agent action impact
        agent_factor = -agent_effectiveness * self.congestion_factors['agent_action_impact']
        
        # Calculate final drop probability with possibility of reaching zero
        final_drop_prob = max(0.0, min(0.95,
            base_drop_prob * (1 + size_factor + priority_factor + path_penalty + buffer_factor + agent_factor)
        ))

        # Perfect conditions can lead to zero drop probability
        if (agent_effectiveness > 0.9 and  # High agent performance
            packet.priority >= 2 and       # High priority packet
            current_buffer < self.congestion_factors['buffer_threshold'] / 2 and  # Low buffer usage
            utilization < self.congestion_levels['moderate']):  # Low congestion
            final_drop_prob = 0.0

        # Handle protocol-specific behaviors
        if not self._handle_protocol_behavior(packet):
            packet.dropped = True
            self.dropped_packets.append(packet)
            self.packets.remove(packet)
            return

        # Apply drop decision
        if random.random() < final_drop_prob:
            packet.dropped = True
            self.dropped_packets.append(packet)
            self.packets.remove(packet)
            self._update_edge_metrics(link, success=False)
        else:
            packet.current_node = next_hop
            packet.path.append(next_hop)
            packet.hops += 1
            self._update_edge_metrics(link, success=True)

    def _update_edge_metrics(self, edge, success):
        """Updated edge metrics with more direct traffic impact and no natural recovery"""
        if edge in self.network.edges() or (edge[1], edge[0]) in self.network.edges():
            edge = edge if edge in self.network.edges() else (edge[1], edge[0])
            
            # Calculate traffic intensity based on number of active packets on this edge
            edge_packets = len([p for p in self.packets if 
                             len(p.path) > 1 and 
                             ((p.path[-2], p.path[-1]) == edge or 
                              (p.path[-1], p.path[-2]) == edge)])
            
            # Factor in the packet generation rate for more dynamic bandwidth utilization
            traffic_intensity = edge_packets * self.packet_generation_rate
            
            if success:
                # More significant impact from traffic intensity
                self.bandwidth_utilization[edge] = min(0.95, max(0.1,
                    self.bandwidth_utilization[edge] * 0.95 + traffic_intensity * 0.1))
                
                # Other metrics remain similar but slightly adjusted
                self.latency[edge] = max(1.0, 
                    self.latency[edge] * (0.95 + traffic_intensity * 0.05))
                self.packet_loss[edge] = max(0.01, 
                    self.packet_loss[edge] * (0.9 + traffic_intensity * 0.05))
                self.throughput[edge] = min(1.0, 
                    self.throughput[edge] * (1.1 - traffic_intensity * 0.05))
            else:
                # Failed transmissions have stronger impact with higher traffic
                self.bandwidth_utilization[edge] = min(0.95,
                    self.bandwidth_utilization[edge] * (1.1 + traffic_intensity * 0.05))
                self.latency[edge] = min(20.0,
                    self.latency[edge] * (1.05 + traffic_intensity * 0.05))
                self.packet_loss[edge] = min(0.5,
                    self.packet_loss[edge] * (1.05 + traffic_intensity * 0.05))
                self.throughput[edge] = max(0.2,
                    self.throughput[edge] * (0.95 - traffic_intensity * 0.05))

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

    def _is_valid_next_hop(self, current_node, next_hop):
        """Check if next_hop is a valid neighbor of current_node"""
        # Check if edge exists in either direction (undirected graph)
        return (current_node, next_hop) in self.network.edges() or \
               (next_hop, current_node) in self.network.edges()

    def step(self, action):
        """Execute one step in the environment"""
        # Gradually increase difficulty over time
        if self.packet_counter > 100:  # After some initial learning period
            # Gradually increase packet generation rate
            self.packet_generation_rate = min(0.4, 0.2 + (self.packet_counter - 1000) * 0.0001)
            
            # Gradually increase congestion and packet loss
            for edge in self.network.edges():
                # Add small random fluctuations to make it more challenging
                if random.random() < 0.1:  # 10% chance per edge per step
                    self.bandwidth_utilization[edge] = min(0.9, 
                        self.bandwidth_utilization[edge] + random.uniform(0, 0.05))
                    self.packet_loss[edge] = min(0.3, 
                        self.packet_loss[edge] + random.uniform(0, 0.02))
        
        # Generate and process packets
        self.generate_packet()
        self.process_packets()
        
        # Convert action to source-destination pair
        src = action // (self.num_nodes - 1)
        dst = action % (self.num_nodes - 1)
        if dst >= src:
            dst += 1
            
        edge = (src, dst) if (src, dst) in self.network.edges() else (dst, src)
        
        # Track metrics before action
        metrics_before = {
            'bandwidth': np.mean(list(self.bandwidth_utilization.values())),
            'latency': np.mean(list(self.latency.values())),
            'packet_loss': np.mean(list(self.packet_loss.values())),
            'throughput': np.mean(list(self.throughput.values()))
        }
        
        # Apply action effects if valid edge
        if edge in self.network.edges():
            # Update bandwidth (reduce utilization)
            self.bandwidth_utilization[edge] = max(
                self.metric_bounds['bandwidth'][0],
                min(self.metric_bounds['bandwidth'][1],
                    self.bandwidth_utilization.get(edge, 0.5) - 0.1)
            )
            
            # Update latency (improve)
            self.latency[edge] = max(
                self.metric_bounds['latency'][0],
                min(self.metric_bounds['latency'][1],
                    self.latency.get(edge, 10.0) * 0.9)
            )
            
            # Update packet loss (reduce)
            self.packet_loss[edge] = max(
                self.metric_bounds['packet_loss'][0],
                min(self.metric_bounds['packet_loss'][1],
                    self.packet_loss.get(edge, 0.1) * 0.9)
            )
            
            # Update throughput (improve)
            self.throughput[edge] = max(
                self.metric_bounds['throughput'][0],
                min(self.metric_bounds['throughput'][1],
                    self.throughput.get(edge, 0.5) * 1.1)
            )
        
        # Update metrics based on packet processing
        for packet in self.delivered_packets[-10:]:  # Look at recent deliveries
            path = packet.path
            for i in range(len(path) - 1):
                edge = (path[i], path[i+1])
                if edge not in self.network.edges():
                    edge = (path[i+1], path[i])
                if edge in self.network.edges():
                    # Improve metrics for successful paths
                    self.bandwidth_utilization[edge] = max(
                        self.metric_bounds['bandwidth'][0],
                        self.bandwidth_utilization[edge] * 0.95
                    )
                    self.latency[edge] = max(
                        self.metric_bounds['latency'][0],
                        self.latency[edge] * 0.95
                    )
                    self.packet_loss[edge] = max(
                        self.metric_bounds['packet_loss'][0],
                        self.packet_loss[edge] * 0.95
                    )
                    self.throughput[edge] = min(
                        self.metric_bounds['throughput'][1],
                        self.throughput[edge] * 1.05
                    )
        
        # Calculate reward
        metrics_after = {
            'bandwidth': np.mean(list(self.bandwidth_utilization.values())),
            'latency': np.mean(list(self.latency.values())),
            'packet_loss': np.mean(list(self.packet_loss.values())),
            'throughput': np.mean(list(self.throughput.values()))
        }
        
        # Calculate reward components
        bandwidth_improvement = metrics_before['bandwidth'] - metrics_after['bandwidth']
        latency_improvement = metrics_before['latency'] - metrics_after['latency']
        packet_loss_improvement = metrics_before['packet_loss'] - metrics_after['packet_loss']
        throughput_improvement = metrics_after['throughput'] - metrics_before['throughput']
        
        # Modify the reward calculation to more strongly consider packet drops
        drop_penalty = len(self.dropped_packets) * 0.5  # Increased penalty for drops
        delivery_reward = len(self.delivered_packets) * 0.3  # Moderate reward for deliveries
        
        # Add congestion management reward
        congestion_reward = 0
        for edge in self.network.edges():
            util = self.bandwidth_utilization[edge]
            if util < self.congestion_levels['moderate']:
                congestion_reward += 0.1  # Reward for maintaining low congestion
            elif util > self.congestion_levels['high']:
                congestion_reward -= 0.2  # Penalty for high congestion
        
        # Combine rewards
        reward = (
            bandwidth_improvement * 2.0 +
            latency_improvement * 1.5 +
            packet_loss_improvement * 1.5 +
            throughput_improvement * 1.0 +
            delivery_reward +
            congestion_reward
        ) - drop_penalty
        
        # Check if done
        done = (metrics_after['latency'] > self.max_latency or
                metrics_after['packet_loss'] > self.max_packet_loss or
                metrics_after['throughput'] < self.min_throughput)
        
        # Prepare info dict
        info = {
            'active_packets': len(self.packets),
            'delivered_packets': len(self.delivered_packets),
            'dropped_packets': len(self.dropped_packets),
            'average_latency': np.mean([p.latency for p in self.delivered_packets]) if self.delivered_packets else 0,
            'average_hops': np.mean([p.hops for p in self.delivered_packets]) if self.delivered_packets else 0
        }
        
        return self._get_state(), reward, done, info

    def _spawn_anomaly(self):
        """Spawn a network anomaly if detector is enabled"""
        if not self.anomaly_detector_enabled:
            return
            
        if random.random() < 0.05:  # 5% chance per step
            anomaly_type = random.choice(self.anomaly_types)
            affected_edges = random.sample(list(self.network.edges()), 
                                        k=random.randint(1, 3))
            
            self.current_anomaly = {
                'type': anomaly_type,
                'edges': affected_edges,
                'duration': random.randint(20, 50),  # steps
                'severity': random.uniform(0.5, 1.0)
            }
            self.anomaly_active = True
            
    def _apply_anomaly_effects(self):
        """Apply effects of active anomaly"""
        if not self.anomaly_active or not self.current_anomaly:
            return
            
        for edge in self.current_anomaly['edges']:
            severity = self.current_anomaly['severity']
            
            if self.current_anomaly['type'] == 'bandwidth_surge':
                self.bandwidth_utilization[edge] = min(1.0, 
                    self.bandwidth_utilization[edge] * (1 + severity))
            elif self.current_anomaly['type'] == 'latency_spike':
                self.latency[edge] = min(20.0, 
                    self.latency[edge] * (1 + severity))
            elif self.current_anomaly['type'] == 'packet_loss_burst':
                self.packet_loss[edge] = min(1.0, 
                    self.packet_loss[edge] + severity * 0.5)
            elif self.current_anomaly['type'] == 'throughput_drop':
                self.throughput[edge] = max(0.1, 
                    self.throughput[edge] * (1 - severity * 0.5))
                    
        self.current_anomaly['duration'] -= 1
        if self.current_anomaly['duration'] <= 0:
            self.anomaly_active = False
            self.current_anomaly = None

    def _handle_protocol_behavior(self, packet):
        """Handle protocol-specific behaviors and requirements"""
        if packet.protocol == Protocol.TCP:
            return self._handle_tcp_behavior(packet)
        elif packet.protocol == Protocol.UDP:
            return self._handle_udp_behavior(packet)
        elif packet.protocol == Protocol.HTTP:
            return self._handle_http_behavior(packet)
        return True

    def _handle_tcp_behavior(self, packet):
        """Simulate TCP protocol behavior"""
        # Get link congestion
        current_node = packet.current_node
        next_hop = self._get_next_hop(current_node, packet.destination)
        if not next_hop:
            return False

        link = (current_node, next_hop)
        congestion = self.bandwidth_utilization.get(link, 0)
        
        # TCP Congestion Control
        if congestion > self.protocol_configs[Protocol.TCP]['congestion_threshold']:
            # Reduce window size
            packet.protocol_state['window_size'] //= 2
            packet.protocol_state['window_size'] = max(
                self.protocol_configs[Protocol.TCP]['min_window_size'],
                packet.protocol_state['window_size']
            )
        else:
            # Increase window size (additive increase)
            packet.protocol_state['window_size'] = min(
                packet.protocol_state['window_size'] + 1024,
                self.protocol_configs[Protocol.TCP]['max_window_size']
            )
        
        # Handle retransmissions
        if packet.dropped and packet.protocol_state['retransmission_count'] < packet.protocol_config[Protocol.TCP]['max_retransmissions']:
            packet.protocol_state['retransmission_count'] += 1
            packet.dropped = False
            return True
        
        return not packet.dropped

    def _handle_udp_behavior(self, packet):
        """Simulate UDP protocol behavior"""
        # UDP is unreliable - no retransmission
        if packet.size > packet.protocol_config[Protocol.UDP]['max_datagram_size']:
            packet.dropped = True
            return False
        
        # Higher tolerance for packet loss
        if packet.dropped and random.random() < self.protocol_configs[Protocol.UDP]['max_packet_loss']:
            return False
        
        return True

    def _handle_http_behavior(self, packet):
        """Simulate HTTP protocol behavior"""
        # HTTP requires TCP
        if not packet.protocol_state['connection_established']:
            # Simulate TCP handshake
            if random.random() < 0.95:  # 95% success rate for connection establishment
                packet.protocol_state['connection_established'] = True
            else:
                return False
        
        # Handle HTTP-specific timeouts
        if packet.latency > self.protocol_configs[Protocol.HTTP]['request_timeout']:
            packet.dropped = True
            return False
        
        return True

    def get_state(self):
        """Return a copy of the current environment state"""
        return {
            'packets': copy.deepcopy(self.packets),
            'delivered_packets': copy.deepcopy(self.delivered_packets),
            'dropped_packets': copy.deepcopy(self.dropped_packets),
            'bandwidth_utilization': copy.deepcopy(self.bandwidth_utilization),
            'latency': copy.deepcopy(self.latency),
            'packet_loss': copy.deepcopy(self.packet_loss),
            'throughput': copy.deepcopy(self.throughput),
            'anomaly_active': self.anomaly_active,
            'current_anomaly': copy.deepcopy(self.current_anomaly),
            'packet_counter': self.packet_counter
        }

    def restore_state(self, state):
        """Restore environment to a previous state"""
        self.packets = state['packets']
        self.delivered_packets = state['delivered_packets']
        self.dropped_packets = state['dropped_packets']
        self.bandwidth_utilization = state['bandwidth_utilization']
        self.latency = state['latency']
        self.packet_loss = state['packet_loss']
        self.throughput = state['throughput']
        self.anomaly_active = state['anomaly_active']
        self.current_anomaly = state['current_anomaly']
        self.packet_counter = state['packet_counter']

    def save_state(self):
        """Return a copy of the complete environment state"""
        return {
            'packets': copy.deepcopy(self.packets),
            'delivered_packets': copy.deepcopy(self.delivered_packets),
            'dropped_packets': copy.deepcopy(self.dropped_packets),
            'bandwidth_utilization': copy.deepcopy(self.bandwidth_utilization),
            'latency': copy.deepcopy(self.latency),
            'packet_loss': copy.deepcopy(self.packet_loss),
            'throughput': copy.deepcopy(self.throughput),
            'anomaly_active': self.anomaly_active,
            'current_anomaly': copy.deepcopy(self.current_anomaly),
            'packet_counter': self.packet_counter
        }
