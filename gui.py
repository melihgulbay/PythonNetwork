# gui.py
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import random
import networkx as nx

class NetworkVisualizerGUI:
    def __init__(self, env, dqn_agent, ppo_agent):
        self.env = env
        self.dqn_controller = dqn_agent
        self.ppo_controller = ppo_agent
        self.rl_enabled = False
        self.rl_type = "none"  # Can be "none", "dqn", or "ppo"
        
        self.root = tk.Tk()
        self.root.title("SDN Controller with RL")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create control panel
        self.control_panel = ttk.Frame(self.main_frame)
        self.control_panel.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Add RL toggle button
        self.rl_var = tk.StringVar(value="none")
        
        self.dqn_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable DQN RL",
            variable=self.rl_var,
            value="dqn",
            command=self.toggle_rl
        )
        self.dqn_toggle.grid(row=0, column=0, padx=5)
        
        self.ppo_toggle = ttk.Radiobutton(
            self.control_panel,
            text="Enable PPO RL",
            variable=self.rl_var,
            value="ppo",
            command=self.toggle_rl
        )
        self.ppo_toggle.grid(row=0, column=1, padx=5)
        
        self.disable_rl = ttk.Radiobutton(
            self.control_panel,
            text="Disable RL",
            variable=self.rl_var,
            value="none",
            command=self.toggle_rl
        )
        self.disable_rl.grid(row=0, column=2, padx=5)
        
        # Create statistics display
        self.stats_frame = ttk.LabelFrame(self.main_frame, text="Network Statistics", padding="5")
        self.stats_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.stats_labels = {}
        stats = ["Bandwidth Utilization", "Latency", "Packet Loss Rate", "Throughput"]
        for i, stat in enumerate(stats):
            ttk.Label(self.stats_frame, text=f"{stat}:").grid(row=i, column=0, sticky=tk.W)
            self.stats_labels[stat] = ttk.Label(self.stats_frame, text="0.0")
            self.stats_labels[stat].grid(row=i, column=1, padx=5)
        
        # Create network visualization
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=2, column=0, pady=10)
        
        # Initialize colorbar
        self.sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
        self.colorbar = plt.colorbar(self.sm, ax=self.ax, label='Bandwidth Utilization')
        
        self.update()
        
    def toggle_rl(self):
        self.rl_type = self.rl_var.get()
        self.rl_enabled = self.rl_type != "none"
        
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
        pos = nx.spring_layout(self.env.network)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.env.network, pos, node_color='lightblue', 
                              node_size=500, ax=self.ax)
        
        # Update node labels to be more descriptive
        labels = {i: f'Node {i}' for i in self.env.network.nodes()}
        nx.draw_networkx_labels(self.env.network, pos, labels, ax=self.ax)
        
        # Draw edges with colors based on utilization
        edges = self.env.network.edges()
        edge_colors = [self.env.bandwidth_utilization[e] for e in edges]
        nx.draw_networkx_edges(self.env.network, pos, edge_color=edge_colors, 
                               edge_cmap=plt.cm.RdYlGn_r, ax=self.ax)
        
        # Update colorbar
        self.sm.set_array(edge_colors)
        
        self.ax.set_title('Network Topology')
        self.canvas.draw()
        
    def update(self):
        if self.rl_enabled:
            state = self.env._get_state()
            if self.rl_type == "dqn":
                action = self.dqn_controller.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.dqn_controller.store_transition(state, action, reward, next_state, done)
                self.dqn_controller.train()
            else:  # PPO
                action, log_prob = self.ppo_controller.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.ppo_controller.store_transition(state, action, reward, next_state, done, log_prob)
                self.ppo_controller.train()
        else:
            # Random network updates when RL is disabled
            for edge in self.env.network.edges():
                # Randomly fluctuate network metrics
                self.env.bandwidth_utilization[edge] = min(1.0, max(0, self.env.bandwidth_utilization[edge] + random.uniform(-0.05, 0.05)))
                self.env.latency[edge] = max(1, self.env.latency[edge] + random.uniform(-0.5, 0.5))
                self.env.packet_loss[edge] = min(1.0, max(0, self.env.packet_loss[edge] + random.uniform(-0.01, 0.01)))
                self.env.throughput[edge] = min(1.0, max(0, self.env.throughput[edge] + random.uniform(-0.05, 0.05)))
        
        self.update_stats()
        self.update_network_visualization()
        self.root.after(100, lambda: self.update())
        
    def run(self):
        self.root.mainloop()