# gui.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
import networkx as nx
from agents import DQNAgent, PPOAgent, TrafficClassifierAgent, AnomalyDetectorAgent, PredictiveMaintenanceAgent
from collections import deque
import time

class NetworkVisualizerGUI:
    def __init__(self, env, dqn_agent, ppo_agent):
        self.env = env
        self.dqn_controller = dqn_agent
        self.ppo_controller = ppo_agent
        self.traffic_classifier = TrafficClassifierAgent(env)
        self.anomaly_detector = AnomalyDetectorAgent(env)
        self.predictive_maintenance = PredictiveMaintenanceAgent(env)
        self.rl_enabled = False
        self.rl_type = "none"  # Can be "none", "dqn", "ppo", "classifier", "anomaly", or "predictive"
        
        self.root = tk.Tk()
        self.root.title("SDN Controller with RL")
        
        # Set window size to 80% of screen size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        
        # Calculate position for center of screen
        position_x = (screen_width - window_width) // 2
        position_y = (screen_height - window_height) // 2
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        
        # Create main frame with scrollbar
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure root grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Add canvas and scrollbar for main content
        self.tk_canvas = tk.Canvas(self.main_frame)
        self.scrollbar = ttk.Scrollbar(self.main_frame, orient="vertical", command=self.tk_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.tk_canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.tk_canvas.configure(scrollregion=self.tk_canvas.bbox("all"))
        )
        
        self.tk_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.tk_canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Grid layout for scrollable components
        self.tk_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure main frame grid weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Add control panel to scrollable frame
        self.control_panel = ttk.Frame(self.scrollable_frame)
        self.control_panel.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # For DQN, PPO, and disable RL (mutually exclusive)
        self.rl_control_var = tk.StringVar(value="none")

        self.dqn_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable DQN RL",
            variable=self.rl_control_var,
            value="dqn",
            command=self.toggle_rl
        )
        self.dqn_toggle.grid(row=0, column=0, padx=5)

        self.ppo_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable PPO RL",
            variable=self.rl_control_var,
            value="ppo",
            command=self.toggle_rl
        )
        self.ppo_toggle.grid(row=0, column=1, padx=5)

        self.disable_rl = ttk.Radiobutton(
            self.control_panel,
            text="Disable RL",
            variable=self.rl_control_var,
            value="none",
            command=self.toggle_rl
        )
        self.disable_rl.grid(row=0, column=2, padx=5)

        # For other features (can be enabled simultaneously)
        self.classifier_var = tk.BooleanVar(value=False)
        self.anomaly_var = tk.BooleanVar(value=False)
        self.predictive_var = tk.BooleanVar(value=False)

        self.classifier_toggle = ttk.Checkbutton(
            self.control_panel,
            text="Enable Traffic Classification",
            variable=self.classifier_var,
            command=self.toggle_features
        )
        self.classifier_toggle.grid(row=0, column=3, padx=5)

        self.anomaly_toggle = ttk.Checkbutton(
            self.control_panel,
            text="Enable Anomaly Detection",
            variable=self.anomaly_var,
            command=self.toggle_features
        )
        self.anomaly_toggle.grid(row=0, column=4, padx=5)

        self.predictive_toggle = ttk.Checkbutton(
            self.control_panel,
            text="Enable Predictive Maintenance",
            variable=self.predictive_var,
            command=self.toggle_features
        )
        self.predictive_toggle.grid(row=0, column=5, padx=5)
        
        # Create statistics display
        self.stats_frame = ttk.LabelFrame(self.scrollable_frame, text="Network Statistics", padding="5")
        self.stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.stats_labels = {}
        stats = ["Bandwidth Utilization", "Latency", "Packet Loss Rate", "Throughput"]
        for i, stat in enumerate(stats):
            ttk.Label(self.stats_frame, text=f"{stat}:").grid(row=i, column=0, sticky=tk.W)
            self.stats_labels[stat] = ttk.Label(self.stats_frame, text="0.0")
            self.stats_labels[stat].grid(row=i, column=1, padx=5)
        
        # Create network visualization with larger figure
        self.figure, self.ax = plt.subplots(figsize=(12, 8))  # Increased figure size
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=self.scrollable_frame)
        self.plot_canvas.get_tk_widget().grid(row=2, column=0, pady=10, padx=10)
        
        # Initialize colorbar
        self.sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
        self.colorbar = plt.colorbar(self.sm, ax=self.ax, label='Bandwidth Utilization')
        
        # Modify traffic classification display
        self.traffic_frame = ttk.LabelFrame(self.scrollable_frame, text="Traffic Classification", padding="5")
        self.traffic_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)

        # Add text widget for detailed traffic information with increased width
        self.traffic_text = tk.Text(self.traffic_frame, height=8, width=80)  # Increased width
        self.traffic_text.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Add scrollbar for traffic text
        traffic_scrollbar = ttk.Scrollbar(self.traffic_frame, orient="vertical", 
                                        command=self.traffic_text.yview)
        traffic_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.traffic_text.configure(yscrollcommand=traffic_scrollbar.set)
        
        # Add packet information frame
        self.packet_frame = ttk.LabelFrame(self.scrollable_frame, text="Packet Information", padding="5")
        self.packet_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # Add text widget for packet information with increased width
        self.packet_text = tk.Text(self.packet_frame, height=8, width=80)  # Increased width
        self.packet_text.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Add scrollbar for packet text
        packet_scrollbar = ttk.Scrollbar(self.packet_frame, orient="vertical", 
                                       command=self.packet_text.yview)
        packet_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.packet_text.configure(yscrollcommand=packet_scrollbar.set)
        
        # Bind mousewheel to scroll
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # Configure style for better appearance
        style = ttk.Style()
        style.configure('TLabelframe', padding=10)
        style.configure('TFrame', padding=5)
        
        # Add anomaly detection display with more details and mitigation status
        self.anomaly_frame = ttk.LabelFrame(self.scrollable_frame, text="Security Status", padding="5")
        self.anomaly_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=10)

        self.anomaly_label = ttk.Label(self.anomaly_frame, text="Status: Normal", font=('TkDefaultFont', 10, 'bold'))
        self.anomaly_label.grid(row=0, column=0, sticky=tk.W)

        self.anomaly_score_label = ttk.Label(self.anomaly_frame, text="Anomaly Score: 0.0")
        self.anomaly_score_label.grid(row=1, column=0, sticky=tk.W)

        self.anomaly_type_label = ttk.Label(self.anomaly_frame, text="Anomaly Type: None")
        self.anomaly_type_label.grid(row=2, column=0, sticky=tk.W)

        self.affected_edges_label = ttk.Label(self.anomaly_frame, text="Affected Edges: None")
        self.affected_edges_label.grid(row=3, column=0, sticky=tk.W)

        # Add new labels for mitigation status
        self.mitigation_status_label = ttk.Label(self.anomaly_frame, text="Mitigation Status: Inactive", 
                                                font=('TkDefaultFont', 9, 'italic'))
        self.mitigation_status_label.grid(row=4, column=0, sticky=tk.W)

        self.mitigation_action_label = ttk.Label(self.anomaly_frame, text="Current Action: None")
        self.mitigation_action_label.grid(row=5, column=0, sticky=tk.W)

        self.mitigation_effect_label = ttk.Label(self.anomaly_frame, text="Mitigation Effect: -")
        self.mitigation_effect_label.grid(row=6, column=0, sticky=tk.W)

        # Add a progress bar for mitigation progress
        self.mitigation_progress = ttk.Progressbar(self.anomaly_frame, length=200, mode='determinate')
        self.mitigation_progress.grid(row=7, column=0, sticky=tk.W, pady=5)

        # Add predictive maintenance frame
        self.maintenance_frame = ttk.LabelFrame(self.scrollable_frame, text="Predictive Maintenance", padding="5")
        self.maintenance_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=10)

        self.maintenance_text = tk.Text(self.maintenance_frame, height=8, width=80)
        self.maintenance_text.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)

        maintenance_scrollbar = ttk.Scrollbar(self.maintenance_frame, orient="vertical",
                                            command=self.maintenance_text.yview)
        maintenance_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.maintenance_text.configure(yscrollcommand=maintenance_scrollbar.set)

        # Add maintenance history frame
        self.maintenance_history_frame = ttk.LabelFrame(self.scrollable_frame, 
            text="Maintenance History", padding="5")
        self.maintenance_history_frame.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=10)

        # Create a treeview for maintenance history
        self.maintenance_tree = ttk.Treeview(self.maintenance_history_frame, 
            columns=("Time", "Edge", "Type", "Metrics Before", "Metrics After", "Impact"), 
            show="headings", height=6)

        # Configure columns
        self.maintenance_tree.heading("Time", text="Time")
        self.maintenance_tree.heading("Edge", text="Edge")
        self.maintenance_tree.heading("Type", text="Maintenance Type")
        self.maintenance_tree.heading("Metrics Before", text="Metrics Before")
        self.maintenance_tree.heading("Metrics After", text="Metrics After")
        self.maintenance_tree.heading("Impact", text="Performance Impact")

        # Set column widths
        self.maintenance_tree.column("Time", width=100)
        self.maintenance_tree.column("Edge", width=100)
        self.maintenance_tree.column("Type", width=150)
        self.maintenance_tree.column("Metrics Before", width=200)
        self.maintenance_tree.column("Metrics After", width=200)
        self.maintenance_tree.column("Impact", width=150)

        # Add scrollbar for the treeview
        maintenance_scroll = ttk.Scrollbar(self.maintenance_history_frame, 
            orient="vertical", command=self.maintenance_tree.yview)
        self.maintenance_tree.configure(yscrollcommand=maintenance_scroll.set)

        # Grid layout
        self.maintenance_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        maintenance_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        # Add filter controls
        filter_frame = ttk.Frame(self.maintenance_history_frame)
        filter_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(filter_frame, text="Filter by:").pack(side=tk.LEFT, padx=5)
        self.filter_var = tk.StringVar(value="all")
        ttk.Radiobutton(filter_frame, text="All", variable=self.filter_var, 
            value="all", command=self.filter_maintenance_history).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(filter_frame, text="Critical", variable=self.filter_var, 
            value="critical", command=self.filter_maintenance_history).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(filter_frame, text="Routine", variable=self.filter_var, 
            value="routine", command=self.filter_maintenance_history).pack(side=tk.LEFT, padx=5)

        # Add performance comparison frame
        self.performance_frame = ttk.LabelFrame(self.scrollable_frame, text="Agent Performance Comparison", padding="5")
        self.performance_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=10)

        # Create performance plot with 4 subplots
        self.perf_figure, ((self.reward_ax, self.delivery_ax), 
                           (self.latency_ax, self.throughput_ax)) = plt.subplots(2, 2, figsize=(12, 6))
        self.perf_canvas = FigureCanvasTkAgg(self.perf_figure, master=self.performance_frame)
        self.perf_canvas.get_tk_widget().grid(row=0, column=0, pady=10, padx=10)

        # Add performance data buffers with fixed size
        self.buffer_size = 1000  # Adjust based on needed history
        self.dqn_performance = {
            'rewards': deque(maxlen=self.buffer_size),
            'packet_delivery_rate': deque(maxlen=self.buffer_size),
            'average_latency': deque(maxlen=self.buffer_size),
            'throughput': deque(maxlen=self.buffer_size)
        }
        
        self.ppo_performance = {
            'rewards': deque(maxlen=self.buffer_size),
            'packet_delivery_rate': deque(maxlen=self.buffer_size),
            'average_latency': deque(maxlen=self.buffer_size),
            'throughput': deque(maxlen=self.buffer_size)
        }

        # Add update rate control
        self.last_update_time = time.time()
        self.update_interval = 0.2  # 200ms between updates
        self.plot_update_interval = 1.0  # 1 second between plot updates
        self.last_plot_update = time.time()

        self.update()
        
    def _on_mousewheel(self, event):
        self.tk_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def toggle_rl(self):
        """Handle toggling of DQN/PPO/disable RL"""
        previous_rl_type = self.rl_control_var.get()
        self.rl_type = self.rl_control_var.get()
        self.rl_enabled = self.rl_type != "none"
        
        # Reset environment when changing RL type
        if previous_rl_type != self.rl_type:
            self.env.reset()
        
        # Additional reset when RL is disabled
        if self.rl_type == "none":
            self.env.reset()

    def toggle_features(self):
        """Handle toggling of additional features"""
        # Update environment settings based on feature toggles
        self.env.set_anomaly_detection(self.anomaly_var.get())

    def update(self):
        current_time = time.time()
        
        # Check if enough time has passed since last update
        if current_time - self.last_update_time < self.update_interval:
            self.root.after(10, self.update)  # Check again in 10ms
            return
            
        # Update network state and metrics
        if self.rl_enabled:
            state = self.env._get_state()
            if self.rl_type == "dqn":
                action = self.dqn_controller.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.dqn_controller.store_transition(state, action, reward, next_state, done)
                self.dqn_controller.train()
                self.update_performance_metrics(info, reward)
            elif self.rl_type == "ppo":
                action, log_prob = self.ppo_controller.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.ppo_controller.store_transition(state, action, reward, next_state, done, log_prob)
                self.ppo_controller.train()
                self.update_performance_metrics(info, reward)

        # Update basic displays more frequently
        self.update_packet_info()
        self.update_stats()
        
        # Update visualization and plots less frequently
        if current_time - self.last_plot_update >= self.plot_update_interval:
            self.update_network_visualization()
            self.plot_performance_comparison()
            self.last_plot_update = current_time
        
        self.last_update_time = current_time
        self.root.after(10, self.update)

    def update_stats(self):
        stats = {
            "Bandwidth Utilization": np.mean(list(self.env.bandwidth_utilization.values())),
            "Latency": np.mean(list(self.env.latency.values())),
            "Packet Loss Rate": np.mean(list(self.env.packet_loss.values())),
            "Throughput": np.mean(list(self.env.throughput.values()))
        }
        
        for stat, value in stats.items():
            self.stats_labels[stat].config(text=f"{value:.3f}")
            
    def update_network_visualization(self):
        self.ax.clear()
        
        # Update network layout periodically (every 10 steps)
        if not hasattr(self, 'layout_counter'):
            self.layout_counter = 0
        self.layout_counter += 1
        
        if not hasattr(self, 'pos') or self.layout_counter >= 10:
            self.pos = nx.spring_layout(self.env.network, k=1, iterations=50)
            self.layout_counter = 0
        
        # Separate edges into active and inactive
        active_edges = []
        inactive_edges = []
        inactive_colors = []
        
        for edge in self.env.network.edges():
            # Check if there's a packet currently traveling on this edge
            has_packet = any(
                (edge[0] in packet.path and edge[1] in packet.path and 
                 abs(packet.path.index(edge[0]) - packet.path.index(edge[1])) == 1)
                for packet in self.env.packets
            )
            
            if has_packet:
                active_edges.append(edge)
            else:
                inactive_edges.append(edge)
                inactive_colors.append(self.env.bandwidth_utilization.get(edge, 0))
        
        # Draw inactive edges first
        if inactive_edges:
            nx.draw_networkx_edges(self.env.network, self.pos, 
                                 edgelist=inactive_edges,
                                 edge_color=inactive_colors,
                                 width=1.0,
                                 edge_cmap=plt.cm.RdYlGn_r,
                                 ax=self.ax)
        
        # Draw active edges with blue color
        if active_edges:
            nx.draw_networkx_edges(self.env.network, self.pos, 
                                 edgelist=active_edges,
                                 edge_color='blue',
                                 width=2.0,
                                 ax=self.ax)
        
        # Draw nodes
        nodes = nx.draw_networkx_nodes(self.env.network, self.pos, 
                                     node_color='lightblue', 
                                     node_size=500, ax=self.ax)
        
        # Update labels only if changed
        if not hasattr(self, 'prev_labels') or self.prev_labels != self.env.network.nodes():
            labels = {i: f'Node {i}' for i in self.env.network.nodes()}
            nx.draw_networkx_labels(self.env.network, self.pos, labels, ax=self.ax)
            self.prev_labels = set(self.env.network.nodes())
        
        # Update colorbar with only the inactive edges' colors
        self.sm.set_array(inactive_colors)
        
        self.ax.set_title('Network Topology')
        self.plot_canvas.draw()
        
    def update_packet_info(self):
        """Update packet-related information in the GUI"""
        # Update packet statistics
        active_packets = len(self.env.packets)
        delivered_packets = len(self.env.delivered_packets)
        dropped_packets = len(self.env.dropped_packets)
        
        # Calculate delivery ratio
        total_packets = delivered_packets + dropped_packets
        delivery_ratio = (delivered_packets / total_packets * 100) if total_packets > 0 else 0.0
        
        # Calculate averages
        if self.env.delivered_packets:
            avg_latency = np.mean([p.latency for p in self.env.delivered_packets])
            avg_hops = np.mean([p.hops for p in self.env.delivered_packets])
        else:
            avg_latency = 0
            avg_hops = 0
        
        # Update labels
        self.packet_text.delete(1.0, tk.END)
        self.packet_text.insert(tk.END, "Active Packets:\n")
        self.packet_text.insert(tk.END, "-" * 40 + "\n")
        
        for packet in self.env.packets:
            text = (f"Packet {packet.source}→{packet.destination}: "
                   f"Type={packet.packet_type}, Size={packet.size}, "
                   f"Priority={packet.priority}, Hops={packet.hops}\n")
            self.packet_text.insert(tk.END, text)
        
        # Add packet statistics
        self.packet_text.insert(tk.END, "\nPacket Statistics:\n")
        self.packet_text.insert(tk.END, "-" * 40 + "\n")
        self.packet_text.insert(tk.END, f"Active Packets: {len(self.env.packets)}\n")
        self.packet_text.insert(tk.END, f"Delivered: {len(self.env.delivered_packets)}\n")
        self.packet_text.insert(tk.END, f"Dropped: {len(self.env.dropped_packets)}\n")
        
    def update(self):
        # First handle RL control (DQN/PPO)
        if self.rl_enabled:
            state = self.env._get_state()
            if self.rl_type == "dqn":
                action = self.dqn_controller.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.dqn_controller.store_transition(state, action, reward, next_state, done)
                self.dqn_controller.train()
                self.update_performance_metrics(info, reward)
            elif self.rl_type == "ppo":
                action, log_prob = self.ppo_controller.select_action(state)
                next_state, reward, done, info = self.env.step(action)
                self.ppo_controller.store_transition(state, action, reward, next_state, done, log_prob)
                self.ppo_controller.train()
                self.update_performance_metrics(info, reward)

        # Handle additional features (can run simultaneously)
        if self.classifier_var.get():
            state = self.env._get_state()
            traffic_type, probabilities = self.traffic_classifier.classify_traffic(state)
            
            # Update traffic display
            self.traffic_text.delete(1.0, tk.END)
            
            # Display active traffic classifications
            self.traffic_text.insert(tk.END, "Active Traffic Classifications:\n")
            self.traffic_text.insert(tk.END, "-" * 40 + "\n")
            
            for (src, dst), (traffic_type, confidence) in self.traffic_classifier.node_traffic.items():
                text = f"Node {src} → Node {dst}: {traffic_type} (Confidence: {confidence:.2f})\n"
                
                # Color code based on traffic type
                color = {
                    "Video": "red",
                    "Gaming": "green",
                    "Web": "blue",
                    "Other": "purple"
                }.get(traffic_type, "black")
                
                self.traffic_text.insert(tk.END, text)
                
                # Apply color to the last inserted line
                last_line_start = self.traffic_text.index("end-2c linestart")
                last_line_end = self.traffic_text.index("end-1c")
                self.traffic_text.tag_add(traffic_type, last_line_start, last_line_end)
                self.traffic_text.tag_config(traffic_type, foreground=color)
            
            # Add traffic statistics
            self.traffic_text.insert(tk.END, "\nTraffic Statistics:\n")
            self.traffic_text.insert(tk.END, "-" * 40 + "\n")
            
            traffic_counts = {}
            for _, (traffic_type, _) in self.traffic_classifier.node_traffic.items():
                traffic_counts[traffic_type] = traffic_counts.get(traffic_type, 0) + 1
            
            for traffic_type, count in traffic_counts.items():
                self.traffic_text.insert(tk.END, f"{traffic_type}: {count} connections\n")

            # Add network and packet updates
            self.env.generate_packet()
            self.env.process_packets()
            
            # Update network metrics
            for edge in self.env.network.edges():
                self.env.bandwidth_utilization[edge] = min(1.0, max(0, 
                    self.env.bandwidth_utilization[edge] + random.uniform(-0.05, 0.05)))
                self.env.latency[edge] = max(1, 
                    self.env.latency[edge] + random.uniform(-0.5, 0.5))
                self.env.packet_loss[edge] = min(1.0, max(0, 
                    self.env.packet_loss[edge] + random.uniform(-0.01, 0.01)))
                self.env.throughput[edge] = min(1.0, max(0, 
                    self.env.throughput[edge] + random.uniform(-0.05, 0.05)))

        if self.anomaly_var.get():
            state = self.env._get_state()
            is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(state)
            
            # Get actual anomaly info from environment and process network updates
            env_info = self.env.step(0)[3]  # Get info dict from env step
            actual_anomaly = env_info['anomaly']
            affected_edges = env_info['affected_edges']
            
            # Always process packets and update network metrics
            self.env.generate_packet()  # Generate new packets
            self.env.process_packets()  # Process existing packets
            
            # Update network metrics even when no anomaly is present
            for edge in self.env.network.edges():
                if not actual_anomaly or edge not in affected_edges:
                    # Normal network fluctuations
                    self.env.bandwidth_utilization[edge] = min(1.0, max(0, 
                        self.env.bandwidth_utilization[edge] + random.uniform(-0.05, 0.05)))
                    self.env.latency[edge] = max(1, 
                        self.env.latency[edge] + random.uniform(-0.5, 0.5))
                    self.env.packet_loss[edge] = min(1.0, max(0, 
                        self.env.packet_loss[edge] + random.uniform(-0.01, 0.01)))
                    self.env.throughput[edge] = min(1.0, max(0, 
                        self.env.throughput[edge] + random.uniform(-0.05, 0.05)))
            
            # Update display based on both detection and actual anomaly
            if is_anomaly or actual_anomaly:
                status_text = "Status: ANOMALY DETECTED!"
                status_color = "red"
                mitigation_status = "ACTIVE - Defending Network"
                self.mitigation_progress['value'] = 100
                
                # Ensure mitigation is being applied
                if actual_anomaly:
                    self.anomaly_detector._apply_mitigation(state, actual_anomaly)
            else:
                status_text = "Status: Normal"
                status_color = "green"
                mitigation_status = "Inactive - Monitoring"
                self.mitigation_progress['value'] = 0
            
            self.anomaly_label.config(
                text=status_text,
                foreground=status_color
            )
            
            self.anomaly_score_label.config(
                text=f"Anomaly Score: {anomaly_score:.6f}"
            )
            
            if actual_anomaly:
                self.anomaly_type_label.config(
                    text=f"Anomaly Type: {actual_anomaly}",
                    foreground="red"
                )
                
                # Show mitigation details
                mitigation_action = self.anomaly_detector.mitigation_strategies.get(actual_anomaly, "Unknown")
                if callable(mitigation_action):
                    mitigation_action = mitigation_action.__name__.replace('_', ' ').title()
                
                self.mitigation_status_label.config(
                    text=f"Mitigation Status: {mitigation_status}",
                    foreground="blue"  # Always blue when there's an actual anomaly
                )
                
                self.mitigation_action_label.config(
                    text=f"Current Action: {mitigation_action}"
                )
                
                # Calculate and show mitigation effect
                if affected_edges:
                    avg_improvement = sum(
                        1 - self.env.bandwidth_utilization[edge] 
                        for edge in affected_edges
                    ) / len(affected_edges)
                    
                    self.mitigation_effect_label.config(
                        text=f"Mitigation Effect: {avg_improvement:.1%} Improvement"
                    )
                    
                    # Update progress bar based on mitigation effect
                    self.mitigation_progress['value'] = min(100, max(0, avg_improvement * 100))
            else:
                self.anomaly_type_label.config(
                    text="Anomaly Type: None",
                    foreground="black"
                )
                self.mitigation_status_label.config(
                    text="Mitigation Status: Inactive - No Threats",
                    foreground="black"
                )
                self.mitigation_action_label.config(
                    text="Current Action: None"
                )
                self.mitigation_effect_label.config(
                    text="Mitigation Effect: -"
                )
                self.mitigation_progress['value'] = 0
            
            self.affected_edges_label.config(
                text=f"Affected Edges: {affected_edges if affected_edges else 'None'}"
            )
            
            # Train the detector
            self.anomaly_detector.train(state)

        if self.predictive_var.get():
            state = self.env._get_state()
            failure_prob, time_to_failure, severity = self.predictive_maintenance.predict_failures(state)
            status_report = self.predictive_maintenance.get_status_report()
            
            # Perform scheduled maintenance and get maintenance records
            maintained_edges = self.predictive_maintenance.perform_maintenance()
            
            # Update maintenance history with any new maintenance records
            if maintained_edges:
                for maintenance in maintained_edges:
                    self.add_maintenance_record(
                        maintenance['edge'],
                        maintenance['type'],
                        maintenance['metrics_before'],
                        maintenance['metrics_after']
                    )
            
            # Update maintenance display
            self.maintenance_text.delete(1.0, tk.END)
            self.maintenance_text.insert(tk.END, "Network Health Status:\n")
            self.maintenance_text.insert(tk.END, "-" * 40 + "\n")
            
            # Add overall network health metrics
            if failure_prob is not None:
                self.maintenance_text.insert(tk.END, f"\nOverall Network Health:\n")
                self.maintenance_text.insert(tk.END, f"Failure Probability: {failure_prob:.2%}\n")
                self.maintenance_text.insert(tk.END, f"Estimated Time to Failure: {time_to_failure:.1f} steps\n")
                self.maintenance_text.insert(tk.END, f"Severity Level: {severity:.2f}\n\n")
            
            if status_report['high_risk_edges']:
                self.maintenance_text.insert(tk.END, "High Risk Edges:\n")
                for edge in status_report['high_risk_edges']:
                    self.maintenance_text.insert(tk.END, f"Edge {edge}: Immediate attention needed\n", "high_risk")
                    
            if status_report['warning_edges']:
                self.maintenance_text.insert(tk.END, "\nWarning Edges:\n")
                for edge in status_report['warning_edges']:
                    self.maintenance_text.insert(tk.END, f"Edge {edge}: Monitor closely\n", "warning")
                    
            if maintained_edges:
                self.maintenance_text.insert(tk.END, "\nMaintenance Performed:\n")
                for edge in maintained_edges:
                    self.maintenance_text.insert(tk.END, f"Edge {edge}: Maintenance complete\n", "maintenance")
                    
            if status_report['maintenance_scheduled']:
                self.maintenance_text.insert(tk.END, "\nScheduled Maintenance:\n")
                for edge in status_report['maintenance_scheduled']:
                    # Add check for edge in maintenance_schedule
                    if edge in self.predictive_maintenance.maintenance_schedule:
                        time = self.predictive_maintenance.maintenance_schedule[edge]
                        self.maintenance_text.insert(tk.END, f"Edge {edge}: Scheduled in {time:.1f} steps\n")
            
            # Configure text colors
            self.maintenance_text.tag_configure("high_risk", foreground="red")
            self.maintenance_text.tag_configure("warning", foreground="orange")
            self.maintenance_text.tag_configure("maintenance", foreground="green")
            
            # Continue normal network operations
            self.env.generate_packet()
            self.env.process_packets()
            
            # Update network metrics with some degradation to simulate wear
            for edge in self.env.network.edges():
                # Gradually degrade network performance
                self.env.bandwidth_utilization[edge] = min(1.0, max(0, 
                    self.env.bandwidth_utilization[edge] + random.uniform(-0.02, 0.05)))
                self.env.latency[edge] = max(1, 
                    self.env.latency[edge] + random.uniform(-0.2, 0.5))
                self.env.packet_loss[edge] = min(1.0, max(0, 
                    self.env.packet_loss[edge] + random.uniform(-0.01, 0.02)))
                self.env.throughput[edge] = min(1.0, max(0, 
                    self.env.throughput[edge] + random.uniform(-0.05, 0.02)))
            
            # Train the predictive maintenance model
            self.predictive_maintenance.train(state, 
                failure_occurred=any(self.env.packet_loss[e] > 0.5 for e in self.env.network.edges()))
            
            # Handle maintenance records
            maintained_edges = self.predictive_maintenance.perform_maintenance()
            for maintenance in maintained_edges:
                self.add_maintenance_record(
                    maintenance['edge'],
                    maintenance['type'],
                    maintenance['metrics_before'],
                    maintenance['metrics_after']
                )

        # Always perform these operations
        if not self.rl_enabled:
            self.env.generate_packet()
            self.env.process_packets()
            
            # Random network updates when RL is disabled
            for edge in self.env.network.edges():
                # Randomly fluctuate network metrics
                self.env.bandwidth_utilization[edge] = min(1.0, max(0, 
                    self.env.bandwidth_utilization[edge] + random.uniform(-0.05, 0.05)))
                self.env.latency[edge] = max(1, 
                    self.env.latency[edge] + random.uniform(-0.5, 0.5))
                self.env.packet_loss[edge] = min(1.0, max(0, 
                    self.env.packet_loss[edge] + random.uniform(-0.01, 0.01)))
                self.env.throughput[edge] = min(1.0, max(0, 
                    self.env.throughput[edge] + random.uniform(-0.05, 0.05)))
        
        # Always update packet info, stats, and visualization regardless of mode
        self.update_packet_info()
        self.update_stats()
        self.update_network_visualization()
        self.root.after(200, lambda: self.update())
        
        # Update performance plots
        self.plot_performance_comparison()
        
    def run(self):
        self.root.mainloop()

    def add_maintenance_record(self, edge, maintenance_type, metrics_before, metrics_after):
        """Add a new maintenance record to the history"""
        current_time = self.env.packet_counter
        
        # Calculate performance impact
        impact = self._calculate_maintenance_impact(metrics_before, metrics_after)
        
        # Format metrics for display
        metrics_before_str = self._format_metrics(metrics_before)
        metrics_after_str = self._format_metrics(metrics_after)
        
        # Insert new record at the top of the tree
        self.maintenance_tree.insert('', 0, values=(
            f"Step {current_time}",
            f"Edge {edge}",
            maintenance_type,
            metrics_before_str,
            metrics_after_str,
            f"{impact:+.2f}%"
        ))
        
        # Keep only the last 100 records
        if len(self.maintenance_tree.get_children()) > 100:
            self.maintenance_tree.delete(self.maintenance_tree.get_children()[-1])

    def _calculate_maintenance_impact(self, before, after):
        """Calculate the overall performance impact of maintenance"""
        # Calculate weighted improvement across all metrics
        weights = {
            'bandwidth': 0.3,
            'latency': 0.3,
            'packet_loss': 0.2,
            'throughput': 0.2
        }
        
        total_impact = 0
        for metric in weights:
            if metric in before and metric in after:
                # Avoid division by zero
                if before[metric] == 0:
                    if after[metric] == 0:
                        improvement = 0  # No change
                    else:
                        improvement = 100  # Improvement from 0 to something
                else:
                    if metric == 'latency' or metric == 'packet_loss':
                        # Lower is better for these metrics
                        improvement = (before[metric] - after[metric]) / before[metric] * 100
                    else:
                        # Higher is better for these metrics
                        improvement = (after[metric] - before[metric]) / before[metric] * 100
                total_impact += improvement * weights[metric]
        
        return total_impact

    def _format_metrics(self, metrics):
        """Format metrics dictionary for display"""
        return ", ".join([f"{k}: {v:.2f}" for k, v in metrics.items()])

    def filter_maintenance_history(self):
        """Filter maintenance history based on selected filter"""
        filter_type = self.filter_var.get()
        
        # Show all items
        for item in self.maintenance_tree.get_children():
            self.maintenance_tree.reattach(item, '', 'end')
        
        if filter_type != "all":
            for item in self.maintenance_tree.get_children():
                values = self.maintenance_tree.item(item)['values']
                impact = float(values[5].rstrip('%'))
                
                if filter_type == "critical" and abs(impact) < 10:
                    self.maintenance_tree.detach(item)
                elif filter_type == "routine" and abs(impact) >= 10:
                    self.maintenance_tree.detach(item)

    def update_performance_metrics(self, info, reward):
        """Update performance metrics based on the agent type"""
        if self.rl_type == "dqn":
            metrics = self.dqn_performance
        elif self.rl_type == "ppo":
            metrics = self.ppo_performance
        else:
            return
        
        metrics['rewards'].append(reward)
        metrics['packet_delivery_rate'].append(
            info['delivered_packets'] / (info['delivered_packets'] + info['dropped_packets']) 
            if (info['delivered_packets'] + info['dropped_packets']) > 0 else 0
        )
        metrics['average_latency'].append(info['average_latency'])
        metrics['throughput'].append(info['delivered_packets'])

    def plot_performance_comparison(self):
        # Skip if no new data
        if not self.dqn_performance['rewards'] and not self.ppo_performance['rewards']:
            return
            
        # Clear all axes
        self.reward_ax.clear()
        self.delivery_ax.clear()
        self.latency_ax.clear()
        self.throughput_ax.clear()
        
        # Track if we have data to show legends
        has_dqn_data = len(self.dqn_performance['rewards']) > 0
        has_ppo_data = len(self.ppo_performance['rewards']) > 0
        
        # Plot data with reduced number of points
        def downsample(data, target_size=100):
            if len(data) > target_size:
                step = len(data) // target_size
                return list(data)[::step]
            return data
        
        # Plot rewards
        if has_dqn_data:
            dqn_rewards = downsample(self.dqn_performance['rewards'])
            self.reward_ax.plot(dqn_rewards, label='DQN', color='blue')
        if has_ppo_data:
            ppo_rewards = downsample(self.ppo_performance['rewards'])
            self.reward_ax.plot(ppo_rewards, label='PPO', color='red')
        self.reward_ax.set_title('Cumulative Rewards', fontsize=8)
        self.reward_ax.set_xlabel('Steps', fontsize=8)
        self.reward_ax.set_ylabel('Reward', fontsize=8)
        self.reward_ax.tick_params(labelsize=8)
        if has_dqn_data or has_ppo_data:
            self.reward_ax.legend(fontsize=8)
        
        # Plot delivery rates
        if has_dqn_data:
            dqn_delivery = downsample(self.dqn_performance['packet_delivery_rate'])
            self.delivery_ax.plot(dqn_delivery, label='DQN', color='blue')
        if has_ppo_data:
            ppo_delivery = downsample(self.ppo_performance['packet_delivery_rate'])
            self.delivery_ax.plot(ppo_delivery, label='PPO', color='red')
        self.delivery_ax.set_title('Packet Delivery Rate', fontsize=8)
        self.delivery_ax.set_xlabel('Steps', fontsize=8)
        self.delivery_ax.set_ylabel('Delivery Rate', fontsize=8)
        self.delivery_ax.tick_params(labelsize=8)
        if has_dqn_data or has_ppo_data:
            self.delivery_ax.legend(fontsize=8)
        
        # Plot latency
        if has_dqn_data:
            dqn_latency = downsample(self.dqn_performance['average_latency'])
            self.latency_ax.plot(dqn_latency, label='DQN', color='blue')
        if has_ppo_data:
            ppo_latency = downsample(self.ppo_performance['average_latency'])
            self.latency_ax.plot(ppo_latency, label='PPO', color='red')
        self.latency_ax.set_title('Average Latency', fontsize=8)
        self.latency_ax.set_xlabel('Steps', fontsize=8)
        self.latency_ax.set_ylabel('Latency', fontsize=8)
        self.latency_ax.tick_params(labelsize=8)
        if has_dqn_data or has_ppo_data:
            self.latency_ax.legend(fontsize=8)
        
        # Plot throughput
        if has_dqn_data:
            dqn_throughput = downsample(self.dqn_performance['throughput'])
            self.throughput_ax.plot(dqn_throughput, label='DQN', color='blue')
        if has_ppo_data:
            ppo_throughput = downsample(self.ppo_performance['throughput'])
            self.throughput_ax.plot(ppo_throughput, label='PPO', color='red')
        self.throughput_ax.set_title('Throughput', fontsize=8)
        self.throughput_ax.set_xlabel('Steps', fontsize=8)
        self.throughput_ax.set_ylabel('Throughput', fontsize=8)
        self.throughput_ax.tick_params(labelsize=8)
        if has_dqn_data or has_ppo_data:
            self.throughput_ax.legend(fontsize=8)
        
        # Use tight_layout only when necessary
        if time.time() - self.last_plot_update >= self.plot_update_interval:
            self.perf_figure.tight_layout()
        
        # Update canvas
        self.perf_canvas.draw_idle()