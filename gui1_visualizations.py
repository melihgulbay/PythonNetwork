import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import networkx as nx
from matplotlib.figure import Figure
import time

class NetworkVisualizer:
    def __init__(self, scrollable_frame):
        # Create network visualization
        self.figure, self.ax = plt.subplots(figsize=(12, 8))
        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=scrollable_frame)
        self.plot_canvas.get_tk_widget().grid(row=3, column=0, pady=10, padx=10)
        
        # Initialize colorbar
        self.sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r, norm=plt.Normalize(0, 1))
        self.colorbar = plt.colorbar(self.sm, ax=self.ax, label='Bandwidth Utilization')

    def update_visualization(self, env):
        """Update network visualization with anomaly highlighting"""
        self.ax.clear()
        
        # Get node positions
        pos = nx.get_node_attributes(env.network, 'pos')
        if not pos:
            pos = nx.spring_layout(env.network)
        
        # Draw normal edges first
        normal_edges = []
        anomaly_edges = []
        
        if env.anomaly_active and env.current_anomaly:
            anomaly_edges = env.current_anomaly['edges']
            normal_edges = [e for e in env.network.edges() if e not in anomaly_edges]
        else:
            normal_edges = list(env.network.edges())
        
        # Draw normal edges
        edge_colors = []
        for edge in normal_edges:
            utilization = env.bandwidth_utilization.get(edge, 0.5)
            edge_colors.append(utilization)
        
        nx.draw_networkx_edges(env.network, pos, ax=self.ax,
                             edgelist=normal_edges,
                             edge_color=edge_colors,
                             edge_cmap=plt.cm.RdYlGn_r,
                             width=2)
        
        # Draw anomaly edges with red color and dashed style
        if anomaly_edges:
            nx.draw_networkx_edges(env.network, pos, ax=self.ax,
                                 edgelist=anomaly_edges,
                                 edge_color='red',
                                 style='dashed',
                                 width=3)
        
        # Draw active paths
        active_edges = set()
        for packet in env.packets:
            if len(packet.path) > 1:
                for i in range(len(packet.path) - 1):
                    active_edges.add((packet.path[i], packet.path[i + 1]))
                    active_edges.add((packet.path[i + 1], packet.path[i]))
        
        # Draw active paths with blue color
        active_path_edges = [(u, v) for (u, v) in env.network.edges() 
                           if (u, v) in active_edges or (v, u) in active_edges]
        if active_path_edges:
            nx.draw_networkx_edges(env.network, pos, ax=self.ax,
                                 edgelist=active_path_edges,
                                 edge_color='blue',
                                 width=2,
                                 alpha=0.7)
        
        # Draw nodes and labels
        nx.draw_networkx_nodes(env.network, pos, ax=self.ax,
                             node_color='lightblue',
                             node_size=500,
                             edgecolors='black',
                             linewidths=1)
        
        nx.draw_networkx_labels(env.network, pos, ax=self.ax,
                              font_size=10,
                              font_weight='bold')
        
        # Update title with anomaly information
        title = "Network Topology\n"
        if env.anomaly_active and env.current_anomaly:
            title += f"Anomaly detected: {env.current_anomaly['type']}\n"
            title += "Red dashed lines indicate affected edges"
        else:
            title += "Blue lines indicate active packet transfers"
        
        self.ax.set_title(title, pad=20, fontsize=12)
        self.ax.set_axis_off()
        
        # Update colorbar
        self.sm.set_array([])
        self.colorbar.update_normal(self.sm)
        
        self.plot_canvas.draw()

class PerformanceVisualizer:
    def __init__(self, scrollable_frame):
        # Create performance plot with 4 subplots
        self.perf_figure, ((self.reward_ax, self.delivery_ax), 
                          (self.latency_ax, self.throughput_ax)) = plt.subplots(2, 2, figsize=(12, 6))
        self.perf_canvas = FigureCanvasTkAgg(self.perf_figure, master=scrollable_frame)
        self.perf_canvas.get_tk_widget().grid(row=0, column=0, pady=10, padx=10)
        
        self.last_plot_update = time.time()
        self.plot_update_interval = 2.0

    def update_visualization(self, performance_data, max_length):
        """Plot performance comparison between all agents"""
        def downsample(data, target_size=100):
            if len(data) == 0:
                return []
            data_list = list(data)
            if len(data_list) > target_size:
                indices = np.linspace(0, len(data_list) - 1, target_size, dtype=int)
                return [data_list[i] for i in indices]
            return data_list

        # Clear all axes
        self.reward_ax.clear()
        self.delivery_ax.clear()
        self.latency_ax.clear()
        self.throughput_ax.clear()

        # Plot data for each agent type
        agent_colors = {'DQN': 'blue', 'PPO': 'red', 'A3C': 'green', 'REINFORCE': 'purple'}
        
        for agent_name, data in performance_data.items():
            if len(data['rewards']) > 0:
                x = np.linspace(0, max_length-1, len(downsample(data['rewards'])))
                color = agent_colors[agent_name]
                
                self.reward_ax.plot(x, downsample(data['rewards']), label=agent_name, color=color)
                self.delivery_ax.plot(x, downsample(data['packet_delivery_rate']), label=agent_name, color=color)
                self.latency_ax.plot(x, downsample(data['average_latency']), label=agent_name, color=color)
                self.throughput_ax.plot(x, downsample(data['throughput']), label=agent_name, color=color)

        # Set titles and labels
        for ax, title in [(self.reward_ax, 'Cumulative Rewards'),
                         (self.delivery_ax, 'Packet Delivery Rate'),
                         (self.latency_ax, 'Average Latency'),
                         (self.throughput_ax, 'Throughput')]:
            ax.set_title(title, fontsize=8)
            ax.set_xlabel('Steps', fontsize=8)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8)

        # Adjust layout and update canvas
        if time.time() - self.last_plot_update >= self.plot_update_interval:
            self.perf_figure.tight_layout()
            self.last_plot_update = time.time()
        
        self.perf_canvas.draw_idle()

class TrafficVisualizer:
    def __init__(self, scrollable_frame):
        # Create traffic visualization
        self.traffic_figure = Figure(figsize=(12, 4))
        self.overall_ax = self.traffic_figure.add_subplot(121)
        self.edge_ax = self.traffic_figure.add_subplot(122)
        self.traffic_canvas = FigureCanvasTkAgg(self.traffic_figure, master=scrollable_frame)
        self.traffic_canvas.get_tk_widget().grid(row=1, column=0, pady=5)

    def update_visualization(self, traffic_classifier, env, selected_edge=None):
        """Update traffic classification pie charts"""
        self.overall_ax.clear()
        self.edge_ax.clear()
        
        # Get overall traffic stats
        overall_stats = {}
        overall_confidence = {}
        for edge in env.network.edges():
            edge_stats = traffic_classifier.get_edge_traffic_stats(edge)
            for traffic_type, data in edge_stats.items():
                if traffic_type not in overall_stats:
                    overall_stats[traffic_type] = 0
                    overall_confidence[traffic_type] = []
                overall_stats[traffic_type] += data['percentage']
                overall_confidence[traffic_type].append(data['confidence'])
        
        # Plot overall traffic
        if overall_stats:
            self._plot_traffic_pie(
                self.overall_ax,
                overall_stats,
                overall_confidence,
                'Overall Network Traffic Distribution'
            )
        
        # Plot selected edge traffic
        if selected_edge:
            edge_stats = traffic_classifier.get_edge_traffic_stats(selected_edge)
            if edge_stats:
                self._plot_traffic_pie(
                    self.edge_ax,
                    {k: v['percentage'] for k, v in edge_stats.items()},
                    {k: v['confidence'] for k, v in edge_stats.items()},
                    f'Traffic Distribution for Edge {selected_edge[0]}->{selected_edge[1]}'
                )
        
        self.traffic_figure.tight_layout()
        self.traffic_canvas.draw()

    def _plot_traffic_pie(self, ax, stats, confidence, title):
        values = list(stats.values())
        labels = [f"{k}\n(conf: {np.mean(confidence[k]) if isinstance(confidence[k], list) else confidence[k]:.2f})" 
                 for k in stats.keys()]
        
        ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title(title) 