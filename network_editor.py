import tkinter as tk
from tkinter import ttk, messagebox
import networkx as nx
import random
from environment import SDNLayer

class NetworkEditor:
    def __init__(self, gui):
        self.gui = gui
        self.env = gui.env
        self.create_editor_frame()
        # Initialize selections
        self.update_node_lists()
        self.update_edge_list()
        
    def create_editor_frame(self):
        """Create the network editor frame"""
        self.editor_frame = ttk.LabelFrame(
            self.gui.scrollable_frame, 
            text="Network Editor", 
            padding="5"
        )
        self.editor_frame.grid(row=14, column=0, sticky=(tk.W, tk.E), pady=5, padx=10)
        
        # Node controls
        self.node_frame = ttk.Frame(self.editor_frame)
        self.node_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Add node
        ttk.Label(self.node_frame, text="Add Node:").grid(row=0, column=0, padx=5)
        self.add_node_button = ttk.Button(
            self.node_frame,
            text="Add",
            command=self.add_node,
            state='disabled'
        )
        self.add_node_button.grid(row=0, column=1, padx=5)
        
        # Remove node
        ttk.Label(self.node_frame, text="Remove Node:").grid(row=0, column=2, padx=5)
        self.remove_node_var = tk.StringVar()
        self.remove_node_combo = ttk.Combobox(
            self.node_frame,
            textvariable=self.remove_node_var,
            state='disabled',
            width=10,
            values=[]  # Initialize with empty list
        )
        self.remove_node_combo.grid(row=0, column=3, padx=5)
        self.remove_node_button = ttk.Button(
            self.node_frame,
            text="Remove",
            command=self.remove_node,
            state='disabled'
        )
        self.remove_node_button.grid(row=0, column=4, padx=5)
        
        # Edge controls
        self.edge_frame = ttk.Frame(self.editor_frame)
        self.edge_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Add edge
        ttk.Label(self.edge_frame, text="Add Edge:").grid(row=0, column=0, padx=5)
        self.add_edge_source_var = tk.StringVar()  # Add variable for source
        self.add_edge_source = ttk.Combobox(
            self.edge_frame,
            textvariable=self.add_edge_source_var,
            state='disabled',
            width=10,
            values=[]  # Initialize with empty list
        )
        self.add_edge_source.grid(row=0, column=1, padx=5)
        
        ttk.Label(self.edge_frame, text="to").grid(row=0, column=2, padx=5)
        self.add_edge_target_var = tk.StringVar()  # Add variable for target
        self.add_edge_target = ttk.Combobox(
            self.edge_frame,
            textvariable=self.add_edge_target_var,
            state='disabled',
            width=10,
            values=[]  # Initialize with empty list
        )
        self.add_edge_target.grid(row=0, column=3, padx=5)
        
        self.add_edge_button = ttk.Button(
            self.edge_frame,
            text="Add",
            command=self.add_edge,
            state='disabled'
        )
        self.add_edge_button.grid(row=0, column=4, padx=5)
        
        # Remove edge
        ttk.Label(self.edge_frame, text="Remove Edge:").grid(row=1, column=0, padx=5)
        self.remove_edge_var = tk.StringVar()
        self.remove_edge_combo = ttk.Combobox(
            self.edge_frame,
            textvariable=self.remove_edge_var,
            state='disabled',
            width=30,
            values=[]  # Initialize with empty list
        )
        self.remove_edge_combo.grid(row=1, column=1, columnspan=3, padx=5)
        self.remove_edge_button = ttk.Button(
            self.edge_frame,
            text="Remove",
            command=self.remove_edge,
            state='disabled'
        )
        self.remove_edge_button.grid(row=1, column=4, padx=5)
        
        # Bind dropdown events to handle selection
        self.remove_node_combo.bind('<<ComboboxSelected>>', lambda e: self.remove_node_var.set(self.remove_node_combo.get()))
        self.add_edge_source.bind('<<ComboboxSelected>>', lambda e: self.add_edge_source_var.set(self.add_edge_source.get()))
        self.add_edge_target.bind('<<ComboboxSelected>>', lambda e: self.add_edge_target_var.set(self.add_edge_target.get()))
        self.remove_edge_combo.bind('<<ComboboxSelected>>', lambda e: self.remove_edge_var.set(self.remove_edge_combo.get()))
        
    def enable_controls(self):
        """Enable network editing controls"""
        # First update the lists to ensure valid options
        self.update_node_lists()
        self.update_edge_list()
        
        # Enable buttons
        self.add_node_button.configure(state='normal')
        self.remove_node_button.configure(state='normal')
        self.add_edge_button.configure(state='normal')
        self.remove_edge_button.configure(state='normal')
        
        # Enable Comboboxes
        self.remove_node_combo.configure(state='normal')
        self.add_edge_source.configure(state='normal')
        self.add_edge_target.configure(state='normal')
        self.remove_edge_combo.configure(state='normal')
        
    def disable_controls(self):
        """Disable network editing controls"""
        # Disable buttons
        self.add_node_button.configure(state='disabled')
        self.remove_node_button.configure(state='disabled')
        self.add_edge_button.configure(state='disabled')
        self.remove_edge_button.configure(state='disabled')
        
        # Disable Comboboxes
        self.remove_node_combo.configure(state='disabled')
        self.add_edge_source.configure(state='disabled')
        self.add_edge_target.configure(state='disabled')
        self.remove_edge_combo.configure(state='disabled')
        
    def update_node_lists(self):
        """Update all node selection dropdowns"""
        nodes = sorted(list(self.env.network.nodes()))
        self.remove_node_combo['values'] = nodes
        self.add_edge_source['values'] = nodes
        self.add_edge_target['values'] = nodes
        
        # Clear current selections if they're invalid
        if self.remove_node_var.get() and int(self.remove_node_var.get()) not in nodes:
            self.remove_node_var.set('')
        if self.add_edge_source.get() and int(self.add_edge_source.get()) not in nodes:
            self.add_edge_source.set('')
        if self.add_edge_target.get() and int(self.add_edge_target.get()) not in nodes:
            self.add_edge_target.set('')
            
    def update_edge_list(self):
        """Update edge selection dropdown"""
        edges = sorted([f"{u}->{v}" for u, v in self.env.network.edges()])
        self.remove_edge_combo['values'] = edges
        
        # Clear current selection if it's invalid
        if self.remove_edge_var.get() and self.remove_edge_var.get() not in edges:
            self.remove_edge_var.set('')
            
    def add_node(self):
        """Add a new node to the network"""
        try:
            # Get new node ID
            new_node = len(self.env.network.nodes())
            
            # Add node to network
            self.env.network.add_node(new_node)
            
            # Position the new node randomly but away from others
            pos = nx.get_node_attributes(self.env.network, 'pos')
            if not pos:
                pos = nx.spring_layout(self.env.network)
            
            # Find position away from existing nodes
            while True:
                new_pos = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
                min_dist = float('inf')
                for p in pos.values():
                    dist = ((p[0] - new_pos[0])**2 + (p[1] - new_pos[1])**2)**0.5
                    min_dist = min(min_dist, dist)
                if min_dist > 0.15 or len(pos) == 0:
                    break
            
            pos[new_node] = new_pos
            nx.set_node_attributes(self.env.network, pos, 'pos')
            
            # Initialize SDN infrastructure for new node
            self.env.sdn_layers[self.env.SDNLayer.INFRASTRUCTURE]['switches'][new_node] = self.env.SDNSwitch(new_node)
            self.env.sdn_layers[self.env.SDNLayer.CONTROL]['flow_tables'][new_node] = []
            
            # Update GUI elements
            self.update_node_lists()
            self.gui.update_network_visualization()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add node: {str(e)}")
            
    def remove_node(self):
        """Remove selected node from the network"""
        try:
            node = int(self.remove_node_var.get())
            
            # Remove node and its associated edges
            self.env.network.remove_node(node)
            
            # Clean up SDN infrastructure
            self.env.sdn_layers[self.env.SDNLayer.INFRASTRUCTURE]['switches'].pop(node, None)
            self.env.sdn_layers[self.env.SDNLayer.CONTROL]['flow_tables'].pop(node, None)
            
            # Remove associated metrics
            edges_to_remove = []
            for edge in self.env.bandwidth_utilization:
                if node in edge:
                    edges_to_remove.append(edge)
            for edge in edges_to_remove:
                self.env.bandwidth_utilization.pop(edge, None)
                self.env.latency.pop(edge, None)
                self.env.packet_loss.pop(edge, None)
                self.env.throughput.pop(edge, None)
            
            # Update GUI elements
            self.update_node_lists()
            self.update_edge_list()
            self.gui.update_network_visualization()
            
        except ValueError:
            messagebox.showerror("Error", "Please select a node to remove")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove node: {str(e)}")
            
    def add_edge(self):
        """Add new edge between selected nodes"""
        try:
            source = int(self.add_edge_source.get())
            target = int(self.add_edge_target.get())
            
            if source == target:
                messagebox.showerror("Error", "Cannot create self-loop")
                return
                
            if self.env.network.has_edge(source, target):
                messagebox.showerror("Error", "Edge already exists")
                return
            
            # Add edge to network
            self.env.network.add_edge(source, target)
            
            # Initialize metrics for the edge
            edge = (source, target)
            
            # Initialize all required metrics in the environment
            self.env.bandwidth_utilization[edge] = 0.5
            self.env.latency[edge] = 10.0
            self.env.packet_loss[edge] = 0.1
            self.env.throughput[edge] = 0.5
            
            # Initialize metrics for the reverse direction
            reverse_edge = (target, source)
            self.env.bandwidth_utilization[reverse_edge] = 0.5
            self.env.latency[reverse_edge] = 10.0
            self.env.packet_loss[reverse_edge] = 0.1
            self.env.throughput[reverse_edge] = 0.5
            
            # Initialize SDN link properties for both directions
            for e in [edge, reverse_edge]:
                self.env.sdn_layers[SDNLayer.INFRASTRUCTURE]['links'][e] = {
                    'capacity': 1.0,
                    'status': 'up',
                    'qos_config': {
                        'priority_queues': 4,
                        'queue_sizes': [64, 128, 256, 512]
                    }
                }
            
            # Update GUI elements
            self.update_edge_list()
            self.gui.update_network_visualization()
            
            # Clear the selection after adding the edge
            self.add_edge_source.set('')
            self.add_edge_target.set('')
            
        except ValueError:
            messagebox.showerror("Error", "Please select both source and target nodes")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add edge: {str(e)}")
            # Print the full error for debugging
            import traceback
            print(traceback.format_exc())
            
    def remove_edge(self):
        """Remove selected edge from the network"""
        try:
            edge_str = self.remove_edge_var.get()
            source, target = map(int, edge_str.split('->'))
            
            # Remove edge from network
            self.env.network.remove_edge(source, target)
            
            # Remove edge metrics
            edge = (source, target)
            self.env.bandwidth_utilization.pop(edge, None)
            self.env.latency.pop(edge, None)
            self.env.packet_loss.pop(edge, None)
            self.env.throughput.pop(edge, None)
            
            # Remove SDN link properties
            self.env.sdn_layers[self.env.SDNLayer.INFRASTRUCTURE.value]['links'].pop(edge, None)
            
            # Update GUI elements
            self.update_edge_list()
            self.gui.update_network_visualization()
            
        except ValueError:
            messagebox.showerror("Error", "Please select an edge to remove")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove edge: {str(e)}") 