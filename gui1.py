import tkinter as tk
from tkinter import ttk
import numpy as np
import random
from collections import deque
import time
import logging
import tkinter.messagebox as messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from environment import SDNLayer
from anomaly_detector import NetworkAnomalyDetector
from agents import DQNAgent, PPOAgent, A3CAgent, REINFORCEAgent, HybridAgent
from traffic_classification import TrafficClassifier
from gui1_visualizations import NetworkVisualizer, PerformanceVisualizer, TrafficVisualizer
from network_editor import NetworkEditor
from tutorial import TutorialWindow
from agent_profiler import AgentProfiler

class NetworkVisualizerGUI:
    def __init__(self, env, dqn_agent, ppo_agent, a3c_agent, reinforce_agent, hybrid_agent):
        self.env = env
        self.dqn_controller = dqn_agent
        self.ppo_controller = ppo_agent
        self.a3c_controller = a3c_agent
        self.reinforce_controller = reinforce_agent
        self.hybrid_controller = hybrid_agent
        self.rl_enabled = False
        self.rl_type = "none"  # Can be "none", "dqn", "ppo", "a3c", "reinforce", or "hybrid"
        
        self.root = tk.Tk()
        self.root.title("SDN Controller with RL")
        
        # Set window size to 90% of screen size (increased from 80%)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.9)  # Increased from 0.8
        window_height = int(screen_height * 0.9)  # Increased from 0.8
        
        # Calculate position for center of screen
        position_x = (screen_width - window_width) // 2
        position_y = (screen_height - window_height) // 2
        
        # Set window size and position
        self.root.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        
        # Create main frame first
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
        
        # Create simulation control frame at the top of scrollable frame
        self.sim_control_frame = ttk.Frame(self.scrollable_frame)
        self.sim_control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # Add simulation control button
        self.sim_running = True
        self.sim_button = ttk.Button(
            self.sim_control_frame,
            text="Stop Simulation",
            command=self.toggle_simulation
        )
        self.sim_button.grid(row=0, column=0, padx=5)
        
        # Add tutorial button
        self.tutorial_button = ttk.Button(
            self.sim_control_frame,
            text="Tutorial",
            command=self.open_tutorial
        )
        self.tutorial_button.grid(row=0, column=2, padx=5)
        
        # Update grid configuration to maintain layout
        self.sim_control_frame.columnconfigure(2, weight=0)  # Don't expand tutorial button
        self.sim_control_frame.columnconfigure(3, weight=1)  # Make the space after buttons expand
        
        # Add step slider
        self.step_frame = ttk.Frame(self.sim_control_frame)
        self.step_frame.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        ttk.Label(self.step_frame, text="Revert Steps:").grid(row=0, column=0, padx=5)
        
        self.step_var = tk.IntVar(value=0)
        self.step_slider = ttk.Scale(
            self.step_frame,
            from_=0,
            to=100,  # Maximum steps to revert
            orient='horizontal',
            variable=self.step_var,
            command=self.update_step_display,
            state='disabled'
        )
        self.step_slider.grid(row=0, column=1, padx=5)
        
        self.step_label = ttk.Label(self.step_frame, text="0")
        self.step_label.grid(row=0, column=2, padx=5)
        
        self.revert_button = ttk.Button(
            self.step_frame,
            text="Revert",
            command=self.revert_steps,
            state='disabled'
        )
        self.revert_button.grid(row=0, column=3, padx=5)
        
        # Make step frame expandable
        self.sim_control_frame.columnconfigure(1, weight=1)
        self.step_frame.columnconfigure(1, weight=1)
        
        # Store simulation history
        self.history_buffer_size = 100
        self.sim_history = deque(maxlen=self.history_buffer_size)
        
        # Adjust main_frame grid position to row=1 (instead of 0)
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add control panel to scrollable frame (now after sim controls)
        self.control_panel = ttk.Frame(self.scrollable_frame)
        self.control_panel.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # Add RL controls
        self.rl_control_var = tk.StringVar(value="none")
        
        # First row of controls
        self.dqn_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable DQN",
            variable=self.rl_control_var,
            value="dqn",
            command=self.toggle_rl
        )
        self.dqn_toggle.grid(row=0, column=0, padx=5)

        self.ppo_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable PPO",
            variable=self.rl_control_var,
            value="ppo",
            command=self.toggle_rl
        )
        self.ppo_toggle.grid(row=0, column=1, padx=5)

        self.a3c_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable A3C",
            variable=self.rl_control_var,
            value="a3c",
            command=self.toggle_rl
        )
        self.a3c_toggle.grid(row=0, column=2, padx=5)

        # Second row of controls
        self.reinforce_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable REINFORCE",
            variable=self.rl_control_var,
            value="reinforce",
            command=self.toggle_rl
        )
        self.reinforce_toggle.grid(row=1, column=0, padx=5)

        self.hybrid_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable Hybrid",
            variable=self.rl_control_var,
            value="hybrid",
            command=self.toggle_rl
        )
        self.hybrid_toggle.grid(row=1, column=1, padx=5)

        self.disable_rl = ttk.Radiobutton(
            self.control_panel,
            text="Disable RL",
            variable=self.rl_control_var,
            value="none",
            command=self.toggle_rl
        )
        self.disable_rl.grid(row=1, column=2, padx=5)

        # Configure grid weights
        for i in range(3):  # Three columns
            self.control_panel.columnconfigure(i, weight=1)
        
        # Add reset button to control panel with a more descriptive name
        self.reset_button = ttk.Button(
            self.control_panel,
            text="Reset Network (Keep Metrics)",
            command=self.reset_network_keep_metrics
        )
        self.reset_button.grid(row=0, column=3, padx=5)  # Add after the RL control buttons
        
        # Add traffic control slider after the control panel
        self.traffic_frame = ttk.LabelFrame(self.scrollable_frame, text="Traffic Control", padding="5")
        self.traffic_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # Traffic multiplier slider
        self.traffic_multiplier = tk.DoubleVar(value=1.0)
        self.traffic_slider = ttk.Scale(
            self.traffic_frame,
            from_=0.1,
            to=5.0,
            orient='horizontal',
            variable=self.traffic_multiplier,
            command=self.update_traffic_rate
        )
        self.traffic_slider.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        # Label for the slider
        self.traffic_label = ttk.Label(self.traffic_frame, text="Traffic Multiplier: 1.0x")
        self.traffic_label.grid(row=0, column=0, padx=5)
        
        # Make the slider expand horizontally
        self.traffic_frame.columnconfigure(1, weight=1)
        
        # Store original packet generation parameters
        self.original_packet_params = {
            'rate': self.env.packet_generation_rate,
            'min_packets': self.env.min_packets_per_step,
            'max_packets': self.env.max_packets_per_step
        }

        # Create statistics display
        self.stats_frame = ttk.LabelFrame(self.scrollable_frame, text="Network Statistics", padding="5")
        self.stats_frame.grid(row=6, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10, padx=10)
        
        self.stats_labels = {}
        stats = [
            "Bandwidth Utilization",
            "Latency",
            "Packet Loss Rate",
            "Throughput",
            "Priority 0 Success Rate",   # Low priority
            "Priority 1 Success Rate",   # Medium priority
            "Priority 2 Success Rate",   # High priority
            "Priority 3 Success Rate",   # Critical priority
            "Average Path Length",       # Routing efficiency
            "Queue Length (P0)",        # Queue lengths by priority
            "Queue Length (P1)",
            "Queue Length (P2)", 
            "Queue Length (P3)"
        ]
        
        # Create a frame inside stats_frame to hold the labels
        stats_content = ttk.Frame(self.stats_frame)
        stats_content.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        
        # Configure grid weights for stats_frame
        self.stats_frame.columnconfigure(0, weight=1)
        
        # Add labels with better spacing and alignment
        for i, stat in enumerate(stats):
            ttk.Label(stats_content, text=f"{stat}:", anchor="w").grid(
                row=i, column=0, sticky=tk.W, padx=(5,10), pady=2
            )
            self.stats_labels[stat] = ttk.Label(stats_content, text="0.0", anchor="e")
            self.stats_labels[stat].grid(
                row=i, column=1, sticky=tk.E, padx=5, pady=2
            )
            
        # Configure grid weights for stats_content
        stats_content.columnconfigure(1, weight=1)

        # Create network visualization
        self.network_viz = NetworkVisualizer(self.scrollable_frame)
        # The plot_canvas is already created in NetworkVisualizer
        # No need to call grid() on network_viz itself
        
        # Add packet information frame
        self.packet_frame = ttk.LabelFrame(self.scrollable_frame, text="Packet Information", padding="5")
        self.packet_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # Add text widget for packet information
        self.packet_text = tk.Text(self.packet_frame, height=8, width=80)
        self.packet_text.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Add scrollbar for packet text
        packet_scrollbar = ttk.Scrollbar(self.packet_frame, orient="vertical", 
                                       command=self.packet_text.yview)
        packet_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.packet_text.configure(yscrollcommand=packet_scrollbar.set)

        # Add performance comparison frame
        self.performance_frame = ttk.LabelFrame(self.scrollable_frame, text="Agent Performance Comparison", padding="5")
        self.performance_frame.grid(row=9, column=0, sticky=(tk.W, tk.E), pady=10)

        # Create performance plot with 4 subplots
        self.perf_figure, ((self.reward_ax, self.delivery_ax), 
                          (self.latency_ax, self.throughput_ax)) = plt.subplots(2, 2, figsize=(12, 6))
        self.perf_canvas = FigureCanvasTkAgg(self.perf_figure, master=self.performance_frame)
        self.perf_canvas.get_tk_widget().grid(row=0, column=0, pady=10, padx=10)

        # Bind mousewheel to scroll
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)

        # Initialize performance tracking (existing code remains the same)
        self.buffer_size = 1000
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

        # Add performance tracking for new agents
        self.a3c_performance = {
            'rewards': deque(maxlen=self.buffer_size),
            'packet_delivery_rate': deque(maxlen=self.buffer_size),
            'average_latency': deque(maxlen=self.buffer_size),
            'throughput': deque(maxlen=self.buffer_size)
        }
        
        self.reinforce_performance = {
            'rewards': deque(maxlen=self.buffer_size),
            'packet_delivery_rate': deque(maxlen=self.buffer_size),
            'average_latency': deque(maxlen=self.buffer_size),
            'throughput': deque(maxlen=self.buffer_size)
        }

        # Add update rate control
        self.last_update_time = time.time()
        self.update_interval = 0.5
        self.plot_update_interval = 2.0
        self.last_plot_update = time.time()

        # Add Hybrid Agent controls
        self.hybrid_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable Hybrid",
            variable=self.rl_control_var,
            value="hybrid",
            command=self.toggle_rl
        )
        self.hybrid_toggle.grid(row=0, column=4, padx=5)

        self.disable_rl.grid(row=0, column=5, padx=5)  # Move disable button to end
        
        # Create hybrid agent configuration frame
        self.hybrid_frame = ttk.LabelFrame(self.scrollable_frame, text="Hybrid Agent Configuration", padding="5")
        self.hybrid_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        self.hybrid_frame.grid_remove()  # Hide by default
        
        # Create agent selection dropdowns for each metric
        self.hybrid_controls = {}
        metrics = ['bandwidth', 'latency', 'packet_loss', 'throughput']
        agent_types = ['dqn', 'ppo', 'a3c', 'reinforce']
        
        for i, metric in enumerate(metrics):
            # Create frame for each metric
            metric_frame = ttk.Frame(self.hybrid_frame)
            metric_frame.grid(row=i, column=0, sticky=(tk.W, tk.E), pady=2)
            
            # Add label
            ttk.Label(metric_frame, text=f"{metric.title()}:").grid(row=0, column=0, padx=5)
            
            # Add dropdown
            var = tk.StringVar(value='dqn')  # Default to DQN
            dropdown = ttk.Combobox(metric_frame, textvariable=var, values=agent_types, state='readonly', width=15)
            dropdown.grid(row=0, column=1, padx=5)
            
            # Store control reference
            self.hybrid_controls[metric] = var
            
            # Bind selection event
            dropdown.bind('<<ComboboxSelected>>', lambda e, m=metric: self.update_hybrid_agent(m))
        
        # Configure grid weights for hybrid frame
        self.hybrid_frame.columnconfigure(0, weight=1)

        # Add anomaly detection frame with more detailed information
        self.anomaly_frame = ttk.LabelFrame(self.scrollable_frame, text="Anomaly Detection", padding="5")
        self.anomaly_frame.grid(row=10, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # Create a frame for anomaly controls and basic status
        self.anomaly_controls = ttk.Frame(self.anomaly_frame)
        self.anomaly_controls.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Add anomaly detection toggle
        self.anomaly_var = tk.BooleanVar(value=False)
        self.anomaly_toggle = ttk.Checkbutton(
            self.anomaly_controls,
            text="Enable Anomaly Detection",
            variable=self.anomaly_var,
            command=self.toggle_anomaly_detection
        )
        self.anomaly_toggle.grid(row=0, column=0, padx=5)
        
        # Add anomaly status label
        self.anomaly_status = ttk.Label(self.anomaly_controls, text="Status: No anomalies detected")
        self.anomaly_status.grid(row=0, column=1, padx=5)
        
        # Add text widget for detailed anomaly information
        self.anomaly_text = tk.Text(self.anomaly_frame, height=4, width=80)
        self.anomaly_text.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Add scrollbar for anomaly text
        anomaly_scrollbar = ttk.Scrollbar(self.anomaly_frame, orient="vertical", 
                                        command=self.anomaly_text.yview)
        anomaly_scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.anomaly_text.configure(yscrollcommand=anomaly_scrollbar.set)

        # Initialize anomaly detector
        self.anomaly_detector = NetworkAnomalyDetector()

        # Add mitigation controls to anomaly frame
        self.mitigation_frame = ttk.LabelFrame(self.anomaly_frame, text="Anomaly Mitigation", padding="5")
        self.mitigation_frame.grid(row=11, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Add auto-mitigation toggle
        self.auto_mitigate_var = tk.BooleanVar(value=False)
        self.auto_mitigate_toggle = ttk.Checkbutton(
            self.mitigation_frame,
            text="Enable Auto-Mitigation",
            variable=self.auto_mitigate_var
        )
        self.auto_mitigate_toggle.grid(row=0, column=0, padx=5)
        
        # Add manual mitigation button
        self.manual_mitigate_btn = ttk.Button(
            self.mitigation_frame,
            text="Mitigate Now",
            command=self.manual_mitigate
        )
        self.manual_mitigate_btn.grid(row=0, column=1, padx=5)
        
        # Add mitigation status labels
        self.mitigation_status = ttk.Label(self.mitigation_frame, text="Status: Idle")
        self.mitigation_status.grid(row=0, column=2, padx=5)
        
        self.strategy_label = ttk.Label(self.mitigation_frame, text="Current Strategy: None")
        self.strategy_label.grid(row=1, column=0, columnspan=2, padx=5, pady=2)
        
        self.success_rate_label = ttk.Label(self.mitigation_frame, text="Success Rate: N/A")
        self.success_rate_label.grid(row=1, column=2, padx=5, pady=2)

        # Initialize traffic classifier
        self.traffic_classifier = TrafficClassifier()
        
        # Add traffic classification frame with pie charts
        self.traffic_frame = ttk.LabelFrame(self.scrollable_frame, text="Traffic Classification", padding="5")
        self.traffic_frame.grid(row=12, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # Create frame for controls and pie charts
        self.traffic_controls = ttk.Frame(self.traffic_frame)
        self.traffic_controls.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Add traffic classification toggle
        self.traffic_var = tk.BooleanVar(value=False)
        self.traffic_toggle = ttk.Checkbutton(
            self.traffic_controls,
            text="Enable Traffic Classification",
            variable=self.traffic_var,
            command=self.toggle_traffic_classification
        )
        self.traffic_toggle.grid(row=0, column=0, padx=5)
        
        # Add edge selection combobox
        self.edge_var = tk.StringVar()
        self.edge_selector = ttk.Combobox(
            self.traffic_controls,
            textvariable=self.edge_var,
            state='readonly',
            width=30
        )
        self.edge_selector.grid(row=0, column=1, padx=5)
        self.edge_selector.bind('<<ComboboxSelected>>', self.update_traffic_charts)
        
        # Create figures for pie charts
        self.traffic_figure = Figure(figsize=(12, 4))
        self.overall_ax = self.traffic_figure.add_subplot(121)
        self.edge_ax = self.traffic_figure.add_subplot(122)
        
        # Add canvas for pie charts
        self.traffic_canvas = FigureCanvasTkAgg(self.traffic_figure, master=self.traffic_frame)
        self.traffic_canvas.get_tk_widget().grid(row=1, column=0, pady=5)
        
        # Add text widget for detailed stats
        self.traffic_text = tk.Text(self.traffic_frame, height=6, width=80)
        self.traffic_text.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Add scrollbar for traffic text
        traffic_scrollbar = ttk.Scrollbar(self.traffic_frame, orient="vertical", 
                                        command=self.traffic_text.yview)
        traffic_scrollbar.grid(row=2, column=1, sticky=(tk.N, tk.S))
        self.traffic_text.configure(yscrollcommand=traffic_scrollbar.set)

        # Replace the matplotlib/visualization setup with:
        self.network_viz = NetworkVisualizer(self.scrollable_frame)
        self.performance_viz = PerformanceVisualizer(self.performance_frame)
        self.traffic_viz = TrafficVisualizer(self.traffic_frame)
        
        # Add Node History frame after traffic frame
        self.node_history_frame = ttk.LabelFrame(self.scrollable_frame, text="Node History", padding="5")
        self.node_history_frame.grid(row=13, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # Add node selection controls
        self.node_controls = ttk.Frame(self.node_history_frame)
        self.node_controls.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Add node selector
        ttk.Label(self.node_controls, text="Select Node:").grid(row=0, column=0, padx=5)
        self.node_var = tk.StringVar()
        self.node_selector = ttk.Combobox(
            self.node_controls,
            textvariable=self.node_var,
            state='readonly',
            width=10
        )
        self.node_selector.grid(row=0, column=1, padx=5)
        self.node_selector['values'] = list(range(self.env.num_nodes))
        self.node_selector.bind('<<ComboboxSelected>>', self.update_node_history)
        
        # Add history length control
        ttk.Label(self.node_controls, text="History Length:").grid(row=0, column=2, padx=5)
        self.history_length_var = tk.StringVar(value="100")
        history_length = ttk.Entry(
            self.node_controls,
            textvariable=self.history_length_var,
            width=10
        )
        history_length.grid(row=0, column=3, padx=5)
        
        # Create figure for node history
        self.node_figure = Figure(figsize=(12, 6))
        self.node_figure.subplots_adjust(hspace=0.3)
        
        # Create subplots for different metrics
        self.bandwidth_ax = self.node_figure.add_subplot(221)
        self.latency_ax = self.node_figure.add_subplot(222)
        self.packet_loss_ax = self.node_figure.add_subplot(223)
        self.throughput_ax = self.node_figure.add_subplot(224)
        
        # Add canvas for node history plots
        self.node_canvas = FigureCanvasTkAgg(self.node_figure, master=self.node_history_frame)
        self.node_canvas.get_tk_widget().grid(row=1, column=0, pady=5)
        
        # Initialize node history storage
        self.node_history = {
            node: {
                'bandwidth': deque(maxlen=1000),
                'latency': deque(maxlen=1000),
                'packet_loss': deque(maxlen=1000),
                'throughput': deque(maxlen=1000),
                'timestamp': deque(maxlen=1000)
            } for node in range(self.env.num_nodes)
        }
        
        # Initialize timestamp counter
        self.history_timestamp = 0

        # Add network editor after node history frame
        self.network_editor = NetworkEditor(self)

        # Add agent profiler
        self.agent_profiler = AgentProfiler()

        self.update()

    def _on_mousewheel(self, event):
        self.tk_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def toggle_rl(self):
        """Handle toggling of RL agents"""
        previous_rl_type = self.rl_type
        self.rl_type = self.rl_control_var.get()
        self.rl_enabled = self.rl_type != "none"
        
        # Show/hide hybrid controls
        if self.rl_type == "hybrid":
            self.hybrid_frame.grid()
            # Initialize hybrid agent if not already done
            if hasattr(self, 'hybrid_controller'):
                # Update all metric assignments
                for metric, var in self.hybrid_controls.items():
                    self.update_hybrid_agent(metric)
        else:
            self.hybrid_frame.grid_remove()
        
        # Update GUI elements
        self.update_network_visualization()
        self.update_stats()
        self.update_packet_info()
        
        # Log the change
        if previous_rl_type != self.rl_type:
            logging.info(f"Switched RL control from {previous_rl_type} to {self.rl_type}")

        # Start/stop profiling based on RL state
        if self.rl_type != "none":
            self.agent_profiler.start_profiling(self.rl_type)
        else:
            self.agent_profiler.stop_profiling()

    def update_hybrid_agent(self, metric):
        """Update hybrid agent's metric assignment"""
        if hasattr(self, 'hybrid_controller'):
            agent_type = self.hybrid_controls[metric].get()
            self.hybrid_controller.set_agent_for_metric(metric, agent_type)
            logging.info(f"Updated hybrid agent: {metric} -> {agent_type}")

    def update(self):
        """Main update loop"""
        current_time = time.time()
        
        if current_time - self.last_update_time < self.update_interval:
            self.root.after(10, self.update)
            return
        
        if self.sim_running:
            # Store current state in history before updating
            self.sim_history.append(self.env.save_state())  # Use save_state instead of _get_state
            
            # Get observation state for RL
            state = self.env._get_state()
            
            if self.rl_enabled:
                # Record metrics before processing RL actions
                self.agent_profiler.record_metrics()
                
                if self.rl_type == "hybrid":
                    action = self.hybrid_controller.select_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    self.hybrid_controller.store_transition(state, action, reward, next_state, done)
                    self.hybrid_controller.train()
                    self.update_performance_metrics(info, reward)
                elif self.rl_type == "dqn":
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
                elif self.rl_type == "a3c":
                    action = self.a3c_controller.select_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    self.a3c_controller.store_transition(state, action, reward, next_state, done)
                    self.a3c_controller.train()
                    self.update_performance_metrics(info, reward)
                elif self.rl_type == "reinforce":
                    action = self.reinforce_controller.select_action(state)
                    next_state, reward, done, info = self.env.step(action)
                    self.reinforce_controller.store_transition(state, action, reward, next_state, done)
                    self.reinforce_controller.train()
                    self.update_performance_metrics(info, reward)
            else:
                # Even when RL is disabled, we should still step the environment
                _, _, _, info = self.env.step(0)  # Use default action or random action
                self.update_performance_metrics(info, 0)  # Update metrics with 0 reward
            
            # Always update visualizations regardless of RL state
            self.update_network_visualization()
            self.update_stats()
            self.update_packet_info()
            self.plot_performance_comparison()
            
            # Update anomaly status
            self.update_anomaly_status()
            
            # Spawn and apply anomalies
            if self.env.anomaly_detector_enabled:
                self.env._spawn_anomaly()
                if self.env.anomaly_active:
                    self.env._apply_anomaly_effects()
            
            # Handle auto-mitigation
            if self.env.anomaly_active and self.auto_mitigate_var.get():
                metrics = {
                    'bandwidth': np.mean(list(self.env.bandwidth_utilization.values())),
                    'latency': np.mean(list(self.env.latency.values())),
                    'packet_loss': np.mean(list(self.env.packet_loss.values())),
                    'throughput': np.mean(list(self.env.throughput.values()))
                }
                success = self.anomaly_detector.mitigate(metrics, self.env)
                status = "Auto-mitigation active" if success else "Auto-mitigation failed"
                self.mitigation_status.config(text=f"Status: {status}")
            
            # Add traffic classification update
            if self.traffic_var.get():
                for packet in self.env.packets:
                    if len(packet.path) >= 2:
                        edge = (packet.path[-2], packet.path[-1])
                        if edge not in self.env.network.edges():
                            edge = (edge[1], edge[0])
                        if edge in self.env.network.edges():
                            self.traffic_classifier.update_edge_traffic(edge, packet)
            
            # Update traffic visualization directly without requiring edge reselection
            selected_edge = None
            if self.edge_var.get():
                edge_str = self.edge_var.get()
                u, v = map(int, edge_str.split('->'))
                selected_edge = (u, v)
            self.traffic_viz.update_visualization(self.traffic_classifier, self.env, selected_edge)
            self.update_traffic_info()
            
            # Update node history
            self.update_node_metrics()
            if self.node_var.get():
                self.update_node_history()
        
        self.last_update_time = current_time
        self.root.after(10, self.update)

    def update_stats(self):
        """Update network statistics display"""
        # Calculate actual packet loss rate
        total_packets = len(self.env.delivered_packets) + len(self.env.dropped_packets)
        actual_loss_rate = (len(self.env.dropped_packets) / total_packets) if total_packets > 0 else 0.0
        
        # Calculate success rates for each priority level
        priority_success = {0: 0, 1: 0, 2: 0, 3: 0}
        priority_total = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # Count delivered packets by priority
        for packet in self.env.delivered_packets:
            priority_success[packet.priority] = priority_success.get(packet.priority, 0) + 1
            priority_total[packet.priority] = priority_total.get(packet.priority, 0) + 1
        
        # Count dropped packets by priority
        for packet in self.env.dropped_packets:
            priority_total[packet.priority] = priority_total.get(packet.priority, 0) + 1
        
        # Calculate success rates for each priority
        priority_rates = {
            f"Priority {p} Success Rate": (priority_success[p] / priority_total[p]) if priority_total[p] > 0 else 0.0
            for p in range(4)
        }
        
        # Calculate average path length for delivered packets
        path_lengths = [len(p.path) - 1 for p in self.env.delivered_packets]  # -1 because path includes source
        avg_path_length = np.mean(path_lengths) if path_lengths else 0.0
        
        # Calculate queue lengths by priority
        queue_lengths = {0: 0, 1: 0, 2: 0, 3: 0}
        for packet in self.env.packets:
            if not packet.dropped and not packet.delivered:
                queue_lengths[packet.priority] += 1
        
        # Update all stats
        stats = {
            "Bandwidth Utilization": np.mean(list(self.env.bandwidth_utilization.values())),
            "Latency": np.mean(list(self.env.latency.values())),
            "Packet Loss Rate": actual_loss_rate,
            "Throughput": np.mean(list(self.env.throughput.values())),
            **priority_rates,  # Priority success rates
            "Average Path Length": avg_path_length,
            "Queue Length (P0)": queue_lengths[0],
            "Queue Length (P1)": queue_lengths[1],
            "Queue Length (P2)": queue_lengths[2],
            "Queue Length (P3)": queue_lengths[3]
        }
        
        # Update labels
        for stat, value in stats.items():
            if "Queue Length" in stat:
                self.stats_labels[stat].config(text=f"{int(value)}")  # Show queue lengths as integers
            else:
                self.stats_labels[stat].config(text=f"{value:.3f}")

    def update_packet_info(self):
        """Update packet information display"""
        self.packet_text.delete(1.0, tk.END)
        
        # Display active packets
        self.packet_text.insert(tk.END, "Active Packets:\n")
        for packet in self.env.packets[:10]:  # Show first 10 packets
            # Get protocol name (remove 'Protocol.' prefix)
            protocol_name = packet.protocol.value.upper()
            
            # Add protocol info and format with more details
            self.packet_text.insert(tk.END, 
                f"Packet {packet.creation_time}: {packet.source}->{packet.destination} "
                f"({packet.packet_type}, {protocol_name}, Size: {packet.size} bytes, Priority: {packet.priority})\n"
                f"    Protocol State: Window={packet.protocol_state['window_size']}, "
                f"Retries={packet.protocol_state['retransmission_count']}, "
                f"Connected={packet.protocol_state['connection_established']}\n"
            )
        
        # Display statistics
        self.packet_text.insert(tk.END, "\nStatistics:\n")
        self.packet_text.insert(tk.END, 
            f"Total Active: {len(self.env.packets)}\n"
            f"Delivered: {len(self.env.delivered_packets)}\n"
            f"Dropped: {len(self.env.dropped_packets)}\n"
        )

    def update_network_visualization(self):
        """Update network visualization"""
        self.network_viz.update_visualization(self.env)

    def update_performance_metrics(self, info, reward):
        """Update performance metrics based on the agent type"""
        if self.rl_type == "dqn":
            metrics = self.dqn_performance
        elif self.rl_type == "ppo":
            metrics = self.ppo_performance
        elif self.rl_type == "a3c":
            metrics = self.a3c_performance
        elif self.rl_type == "reinforce":
            metrics = self.reinforce_performance
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
        """Plot performance comparison between all agents"""
        # Skip if no new data
        if not self.dqn_performance['rewards'] and not self.ppo_performance['rewards'] and \
           not self.a3c_performance['rewards'] and not self.reinforce_performance['rewards']:
            return
        
        performance_data = {
            'DQN': self.dqn_performance,
            'PPO': self.ppo_performance,
            'A3C': self.a3c_performance,
            'REINFORCE': self.reinforce_performance
        }
        
        max_length = max(
            len(self.dqn_performance['rewards']),
            len(self.ppo_performance['rewards']),
            len(self.a3c_performance['rewards']),
            len(self.reinforce_performance['rewards'])
        )
        
        self.performance_viz.update_visualization(performance_data, max_length)

    def reset_network_keep_metrics(self):
        """Reset the network to initial state while preserving performance metrics"""
        try:
            # Store current RL state
            previous_rl_type = self.rl_type
            previous_rl_enabled = self.rl_enabled
            
            # Reset environment
            self.env.reset()
            
            # Reset agents' internal states but keep their learned parameters
            if self.dqn_controller:
                self.dqn_controller.reset_internal_state()
            if self.ppo_controller:
                self.ppo_controller.reset_internal_state()
            
            # Restore previous RL state
            self.rl_type = previous_rl_type
            self.rl_enabled = previous_rl_enabled
            self.rl_control_var.set(previous_rl_type)
            
            # Force update of visualizations
            self.update_network_visualization()
            self.update_stats()
            self.update_packet_info()
            self.plot_performance_comparison()
            
            # Log reset
            logging.info("Network reset to initial state (metrics preserved)")
            
            # Optional: Add visual feedback that reset was successful
            self.reset_button.state(['disabled'])  # Temporarily disable button
            self.root.after(500, lambda: self.reset_button.state(['!disabled']))  # Re-enable after 500ms
            
            # Log summary stats before resetting
            if self.rl_enabled:
                self.agent_profiler.log_summary_stats()
            
        except Exception as e:
            logging.error(f"Error resetting network: {str(e)}")
            messagebox.showerror("Reset Error", f"Failed to reset network: {str(e)}")

    def update_traffic_rate(self, *args):
        """Update traffic generation parameters based on slider value"""
        multiplier = self.traffic_multiplier.get()
        
        # Update the label
        self.traffic_label.config(text=f"Traffic Multiplier: {multiplier:.1f}x")
        
        # Update environment parameters
        self.env.packet_generation_rate = min(1.0, self.original_packet_params['rate'] * multiplier)
        self.env.min_packets_per_step = max(1, int(self.original_packet_params['min_packets'] * multiplier))
        self.env.max_packets_per_step = max(2, int(self.original_packet_params['max_packets'] * multiplier))

    def toggle_anomaly_detection(self):
        """Toggle anomaly detection on/off"""
        self.env.anomaly_detector_enabled = self.anomaly_var.get()
        if not self.env.anomaly_detector_enabled:
            self.env.anomaly_active = False
            self.env.current_anomaly = None
        logging.info(f"Anomaly detection {'enabled' if self.env.anomaly_detector_enabled else 'disabled'}")

    def update_anomaly_status(self):
        """Update anomaly and mitigation status display"""
        if not self.env.anomaly_detector_enabled:
            status = "Anomaly detection disabled"
            self.anomaly_text.delete('1.0', tk.END)
            self.manual_mitigate_btn.state(['disabled'])
        elif self.env.anomaly_active:
            anomaly = self.env.current_anomaly
            status = f"Active anomaly detected!"
            
            details = (f"Type: {anomaly['type']}\n"
                     f"Severity: {anomaly['severity']:.2f}\n"
                     f"Duration: {anomaly['duration']} steps remaining\n"
                     f"Affected edges: {', '.join([f'{e[0]}->{e[1]}' for e in anomaly['edges']])}\n"
                     f"Mitigation:\n")
            
            # Add more detailed mitigation status
            if self.auto_mitigate_var.get():
                details += f"- Auto-mitigation: Active\n"
                details += f"- Last attempt result: {self.mitigation_status.cget('text')}\n"
            details += f"- Current Strategy: {self.anomaly_detector.get_current_strategy()}\n"
            details += f"- Success Rate: {self.anomaly_detector.get_success_rate():.1%}\n"
            
            self.anomaly_text.delete('1.0', tk.END)
            self.anomaly_text.insert('1.0', details)
        else:
            status = "Status: No anomalies detected"
            self.anomaly_text.delete('1.0', tk.END)
            self.manual_mitigate_btn.state(['disabled'])
            
        self.anomaly_status.config(text=status)
        self.strategy_label.config(text=f"Current Strategy: {self.anomaly_detector.get_current_strategy()}")
        self.success_rate_label.config(text=f"Success Rate: {self.anomaly_detector.get_success_rate():.1%}")

    def manual_mitigate(self):
        """Handle manual mitigation request"""
        if self.env.anomaly_active:
            metrics = {
                'bandwidth': np.mean(list(self.env.bandwidth_utilization.values())),
                'latency': np.mean(list(self.env.latency.values())),
                'packet_loss': np.mean(list(self.env.packet_loss.values())),
                'throughput': np.mean(list(self.env.throughput.values()))
            }
            success = self.anomaly_detector.mitigate(metrics, self.env)
            status = "Mitigation successful" if success else "Mitigation failed"
            self.mitigation_status.config(text=f"Status: {status}")
            self.update_anomaly_status()

    def toggle_traffic_classification(self):
        """Toggle traffic classification on/off"""
        if self.traffic_var.get():
            if not self.traffic_classifier.trained:
                self.traffic_classifier.train()
            self.update_edge_selector()
        else:
            self.traffic_classifier.reset_stats()
        self.update_traffic_charts()
        self.update_traffic_info()

    def update_traffic_info(self):
        """Update traffic classification information display with confidence levels"""
        if not self.traffic_var.get():
            self.traffic_text.delete(1.0, tk.END)
            self.traffic_text.insert(tk.END, "Traffic classification disabled")
            return
            
        self.traffic_text.delete(1.0, tk.END)
        self.traffic_text.insert(tk.END, "Traffic Classification Results:\n\n")
        
        # Get traffic stats for each edge
        for edge in self.env.network.edges():
            stats = self.traffic_classifier.get_edge_traffic_stats(edge)
            if stats:
                self.traffic_text.insert(tk.END, f"Edge {edge}:\n")
                for traffic_type, data in sorted(
                    stats.items(), 
                    key=lambda x: x[1]['percentage'], 
                    reverse=True
                ):
                    self.traffic_text.insert(
                        tk.END, 
                        f"  {traffic_type}: {data['percentage']:.1%} "
                        f"(confidence: {data['confidence']:.2f})\n"
                    )
                self.traffic_text.insert(tk.END, "\n")

    def update_edge_selector(self):
        """Update the edge selector combobox with current edges"""
        edges = list(self.env.network.edges())
        edge_strings = [f"{u}->{v}" for u, v in edges]
        self.edge_selector['values'] = edge_strings
        if edge_strings and not self.edge_var.get():
            self.edge_selector.set(edge_strings[0])

    def update_traffic_charts(self, *args):
        """Update traffic classification charts"""
        if not self.traffic_var.get():
            self.traffic_viz.update_visualization(self.traffic_classifier, self.env)
            return

        if self.edge_var.get():
            edge_str = self.edge_var.get()
            u, v = map(int, edge_str.split('->'))
            selected_edge = (u, v)
        else:
            selected_edge = None

        self.traffic_viz.update_visualization(self.traffic_classifier, self.env, selected_edge)

    def update_node_history(self, *args):
        """Update node history plots"""
        if not self.node_var.get():
            return
            
        node = int(self.node_var.get())
        history_length = min(1000, max(10, int(self.history_length_var.get())))
        
        # Clear all axes
        self.bandwidth_ax.clear()
        self.latency_ax.clear()
        self.packet_loss_ax.clear()
        self.throughput_ax.clear()
        
        # Get node history data
        history = self.node_history[node]
        timestamps = list(history['timestamp'])[-history_length:]
        
        # Plot each metric
        self.bandwidth_ax.plot(timestamps, 
                             list(history['bandwidth'])[-history_length:], 
                             'b-', label='Bandwidth Utilization')
        self.latency_ax.plot(timestamps, 
                            list(history['latency'])[-history_length:], 
                            'r-', label='Latency')
        self.packet_loss_ax.plot(timestamps, 
                                list(history['packet_loss'])[-history_length:], 
                                'g-', label='Packet Loss')
        self.throughput_ax.plot(timestamps, 
                               list(history['throughput'])[-history_length:], 
                               'y-', label='Throughput')
        
        # Set titles and labels
        self.bandwidth_ax.set_title('Bandwidth Utilization')
        self.latency_ax.set_title('Latency')
        self.packet_loss_ax.set_title('Packet Loss')
        self.throughput_ax.set_title('Throughput')
        
        # Add legends
        self.bandwidth_ax.legend()
        self.latency_ax.legend()
        self.packet_loss_ax.legend()
        self.throughput_ax.legend()
        
        # Update canvas
        self.node_figure.tight_layout()
        self.node_canvas.draw()

    def update_node_metrics(self):
        """Update stored metrics for each node"""
        self.history_timestamp += 1
        
        for node in range(self.env.num_nodes):
            # Calculate average metrics for all edges connected to this node
            connected_edges = [
                (u, v) for (u, v) in self.env.network.edges()
                if u == node or v == node
            ]
            
            if connected_edges:
                # Calculate averages for all connected edges
                avg_bandwidth = np.mean([
                    self.env.bandwidth_utilization.get((u, v), 0)
                    for (u, v) in connected_edges
                ])
                avg_latency = np.mean([
                    self.env.latency.get((u, v), 0)
                    for (u, v) in connected_edges
                ])
                avg_packet_loss = np.mean([
                    self.env.packet_loss.get((u, v), 0)
                    for (u, v) in connected_edges
                ])
                avg_throughput = np.mean([
                    self.env.throughput.get((u, v), 0)
                    for (u, v) in connected_edges
                ])
                
                # Store metrics in history
                self.node_history[node]['bandwidth'].append(avg_bandwidth)
                self.node_history[node]['latency'].append(avg_latency)
                self.node_history[node]['packet_loss'].append(avg_packet_loss)
                self.node_history[node]['throughput'].append(avg_throughput)
                self.node_history[node]['timestamp'].append(self.history_timestamp)

    def toggle_simulation(self):
        """Toggle simulation running state"""
        self.sim_running = not self.sim_running
        
        if self.sim_running:
            self.sim_button.config(text="Stop Simulation")
            self.step_slider.state(['disabled'])
            self.revert_button.state(['disabled'])
            self.network_editor.disable_controls()  # Disable network editing
        else:
            self.sim_button.config(text="Start Simulation")
            self.step_slider.state(['!disabled'])
            self.revert_button.state(['!disabled'])
            self.network_editor.enable_controls()  # Enable network editing
            # Update slider maximum based on available history
            self.step_slider.configure(to=len(self.sim_history))
            self.step_var.set(0)
            self.update_step_display()

    def update_step_display(self, *args):
        """Update the step display label"""
        self.step_label.config(text=str(self.step_var.get()))

    def revert_steps(self):
        """Revert the simulation by the specified number of steps"""
        if not self.sim_running and self.sim_history:
            steps = self.step_var.get()
            if steps > 0 and steps <= len(self.sim_history):
                # Restore state from history
                state = self.sim_history[-steps]
                self.env.restore_state(state)
                
                # Update visualizations
                self.update_network_visualization()
                self.update_stats()
                self.update_packet_info()
                self.plot_performance_comparison()
                
                # Remove reverted states from history
                for _ in range(steps):
                    self.sim_history.pop()
                
                # Update slider maximum
                self.step_slider.configure(to=len(self.sim_history))
                self.step_var.set(0)
                self.update_step_display()

    def open_tutorial(self):
        """Open the tutorial window"""
        tutorial_window = TutorialWindow(self.root)
        tutorial_window.grab_set()  # Make the tutorial window modal

    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()