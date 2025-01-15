import tkinter as tk
from tkinter import ttk
import platform
from tkinter import font

class TutorialWindow:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("SDN Simulation Tutorial")
        
        # Set window size to 60% of parent window
        parent_width = parent.winfo_width()
        parent_height = parent.winfo_height()
        window_width = int(parent_width * 0.6)
        window_height = int(parent_height * 0.6)
        
        # Center the window
        position_x = parent.winfo_x() + (parent_width - window_width) // 2
        position_y = parent.winfo_y() + (parent_height - window_height) // 2
        
        self.window.geometry(f"{window_width}x{window_height}+{position_x}+{position_y}")
        
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.window)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Create modern font
        self.modern_font = font.Font(family="Helvetica", size=10)
        
        # Create tabs
        self.create_overview_tab()
        self.create_network_visualization_tab()
        self.create_performance_tab()
        self.create_traffic_tab()
        self.create_editor_tab()
        self.create_agents_tab()
        
        # Add close button at bottom
        close_button = ttk.Button(
            self.window,
            text="Close Tutorial",
            command=self.window.destroy
        )
        close_button.pack(pady=10)
        
        # Make window modal
        self.window.transient(parent)
        self.window.grab_set()
        
        # Bind key events
        self.window.bind('<Escape>', lambda e: self.window.destroy())
        if platform.system() == 'Darwin':  # macOS
            self.window.bind('<Command-w>', lambda e: self.window.destroy())
        else:
            self.window.bind('<Control-w>', lambda e: self.window.destroy())

    def create_overview_tab(self):
        """Create the overview tab with general information"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Overview')
        
        # Add scrollable text widget with modern font
        text = tk.Text(tab, wrap=tk.WORD, padx=10, pady=10, font=self.modern_font)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        text.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        # Add content
        overview_text = """Welcome to the SDN Network Simulation!

This simulation allows you to explore Software-Defined Networking (SDN) concepts through an interactive interface. Here's what you can do:

1. Network Visualization
   • View the network topology in real-time
   • Monitor packet flows and network status
   • Observe anomaly detection and responses

2. Performance Monitoring
   • Track rewards, packet delivery rates, latency, and throughput
   • Compare different RL agent performances
   • Monitor network metrics in real-time

3. Traffic Analysis
   • View traffic distribution across the network
   • Analyze traffic patterns per connection
   • Monitor protocol-specific behaviors

4. Network Editor
   • Add or remove nodes and connections
   • Modify network topology in real-time
   • Test different network configurations

5. RL Agents
   • Switch between different RL algorithms
   • Compare agent performances
   • Observe learning behaviors

Use the tabs above to learn more about each component."""

        text.insert('1.0', overview_text)
        text.configure(state='disabled')

    def create_network_visualization_tab(self):
        """Create the network visualization tutorial tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Network View')
        
        text = tk.Text(tab, wrap=tk.WORD, padx=10, pady=10, font=self.modern_font)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        text.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        viz_text = """Network Visualization Guide

The network topology display shows:

• Nodes (Network Devices)
  - Represented as circles
  - Color indicates device status
  - Size reflects traffic volume

• Connections
  - Blue lines: Active packet transfers
  - Red dashed lines: Anomaly-affected paths
  - Line thickness: Bandwidth utilization

• Color Scale
  - Green: Low utilization
  - Yellow: Moderate utilization
  - Red: High utilization/congestion

• Anomaly Detection
  - The title will show active anomalies
  - Affected paths are highlighted
  - Anomaly type and severity are displayed

Interaction:
• You can click on nodes to see details
• Hover over connections to view metrics
• The visualization updates in real-time"""

        text.insert('1.0', viz_text)
        text.configure(state='disabled')

    def create_performance_tab(self):
        """Create the performance metrics tutorial tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Performance')
        
        text = tk.Text(tab, wrap=tk.WORD, padx=10, pady=10, font=self.modern_font)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        text.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        perf_text = """Performance Metrics Guide

The performance visualization shows four key metrics:

1. Cumulative Rewards
   • Shows the learning progress of agents
   • Higher values indicate better performance
   • Compare different algorithms' effectiveness

2. Packet Delivery Rate
   • Percentage of successfully delivered packets
   • Higher is better
   • Indicates network reliability

3. Average Latency
   • Time taken for packet delivery
   • Lower is better
   • Measures network responsiveness

4. Throughput
   • Network capacity utilization
   • Higher values indicate better efficiency
   • Shows network performance under load

Each metric is color-coded by agent type:
• Blue: DQN
• Red: PPO
• Green: A3C
• Purple: REINFORCE

The graphs update automatically and show the most recent performance window."""

        text.insert('1.0', perf_text)
        text.configure(state='disabled')

    def create_traffic_tab(self):
        """Create the traffic analysis tutorial tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Traffic')
        
        text = tk.Text(tab, wrap=tk.WORD, padx=10, pady=10, font=self.modern_font)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        text.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        traffic_text = """Traffic Analysis Guide

The traffic visualization shows two pie charts:

1. Overall Network Traffic
   • Distribution of traffic types
   • Percentage of each protocol
   • Network-wide traffic patterns

2. Selected Edge Traffic
   • Click an edge to see its traffic
   • Protocol distribution for that connection
   • Local traffic patterns

Protocols Monitored:
• TCP: Reliable, connection-oriented
• UDP: Fast, connectionless
• HTTP: Web traffic
• HTTPS: Secure web traffic
• ICMP: Network control messages

Each slice shows:
• Protocol type
• Percentage of total traffic
• Confidence level of classification

The charts update in real-time as traffic flows through the network."""

        text.insert('1.0', traffic_text)
        text.configure(state='disabled')

    def create_editor_tab(self):
        """Create the network editor tutorial tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Editor')
        
        text = tk.Text(tab, wrap=tk.WORD, padx=10, pady=10, font=self.modern_font)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        text.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        editor_text = """Network Editor Guide

The network editor allows you to modify the network topology:

Node Operations:
• Add Node: Creates a new network device
• Remove Node: Deletes selected node
  - All connections to node are removed
  - Existing traffic is rerouted

Connection Operations:
• Add Edge: Create new connection
  1. Select source node
  2. Select destination node
  3. Click Add
• Remove Edge: Delete connection
  - Select from dropdown
  - Click Remove

Important Notes:
• Changes take effect immediately
• The network maintains connectivity
• Agents adapt to topology changes
• Performance might temporarily decrease
• Network metrics are updated automatically

Use the editor to:
• Test different topologies
• Simulate network growth
• Create failure scenarios
• Optimize network layout"""

        text.insert('1.0', editor_text)
        text.configure(state='disabled')

    def create_agents_tab(self):
        """Create the RL agents tutorial tab"""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Agents')
        
        text = tk.Text(tab, wrap=tk.WORD, padx=10, pady=10, font=self.modern_font)
        scrollbar = ttk.Scrollbar(tab, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        
        text.grid(row=0, column=0, sticky='nsew')
        scrollbar.grid(row=0, column=1, sticky='ns')
        
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(0, weight=1)
        
        agents_text = """Reinforcement Learning Agents Guide

Available Agents:

1. DQN (Deep Q-Network)
   • Good for discrete action spaces
   • Stable learning behavior
   • Memory-based learning

2. PPO (Proximal Policy Optimization)
   • Balanced exploration/exploitation
   • Good for continuous control
   • Stable policy updates

3. A3C (Asynchronous Advantage Actor-Critic)
   • Parallel learning
   • Good for complex environments
   • Fast convergence

4. REINFORCE
   • Policy gradient method
   • Good for simple policies
   • Direct policy optimization

5. Hybrid
   • Combines multiple approaches
   • Adaptive strategy selection
   • Best for dynamic environments

To use agents:
1. Select agent type using radio buttons
2. Monitor performance in graphs
3. Compare different agents
4. Observe adaptation to changes

The agents learn continuously and adapt to:
• Network changes
• Traffic patterns
• Anomalies
• Performance requirements"""

        text.insert('1.0', agents_text)
        text.configure(state='disabled') 