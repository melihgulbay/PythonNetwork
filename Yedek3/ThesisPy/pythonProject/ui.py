# ui.py
import tkinter as tk
import random
import simpy
from packet import Packet
from link import Link
from router import Router
from rl_agent import RLAgent  # Import the RLAgent

class NetworkSimulatorUI:
    def __init__(self, root, env, routers, rl_agent):
        self.root = root
        self.env = env  # Store the environment in self.env
        self.routers = routers
        self.rl_agent = rl_agent  # Store the RL agent
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Network Simulator")

        # Start and Stop buttons
        start_button = tk.Button(self.root, text="Start Simulation", command=self.start_simulation)
        start_button.pack()

        stop_button = tk.Button(self.root, text="Stop Simulation", command=self.stop_simulation)
        stop_button.pack()

        # Router metrics labels
        self.metric_labels = {}
        for router in self.routers:
            label = tk.Label(self.root, text=f"Router {router.name}: Latency: -, Drop Rate: -")
            label.pack()
            self.metric_labels[router.name] = label

    def start_simulation(self):
        """Start the simulation and RL agent when the simulation starts."""
        self.running = True
        self.run_simulation_step()  # Start simulation steps

        # Start periodic packet generation (every 2 seconds)
        self.env.process(self.generate_packets())

        # Start the RL agent simulation loop, passing steps as needed
        self.root.after(1000, self.run_rl_step)  # Start RL agent after 1 second

    def stop_simulation(self):
        """Stop the simulation."""
        self.running = False
        print("Stopping simulation...")
        self.env.process(self.stop_rl_agent())  # Stop the RL agent if necessary

    def stop_rl_agent(self):
        """Stop the RL agent simulation loop if required."""
        self.rl_agent.simulate_network(steps=0)  # Ensure RL agent stops


    def run_simulation_step(self):
        """Perform one simulation step."""
        if self.running:
            try:
                self.env.step()  # Perform one simulation step
            except simpy.core.EmptySchedule:
                # No more events to process
                self.running = False
                print("Simulation has finished.")
            self.update_metrics()  # Update the UI metrics
            if self.running:  # Only call again if running
                self.root.after(100, self.run_simulation_step)


    def update_metrics(self):
        """Update router metrics on the UI."""
        for router in self.routers:
            metrics = router.get_metrics()
            self.metric_labels[router.name].config(
                text=f"Router {router.name}: Latency: {metrics['latency']:.2f}s, "
                     f"Drop Rate: {metrics['drop_rate']:.2%}, Sent: {metrics['sent_packets']}, "
                     f"Received: {metrics['received_packets']}, Lost: {metrics['lost_packets']}, "
                     f"Avg Delay: {metrics['avg_transmission_time']:.2f}s"
            )

    def generate_packets(self):
        """Periodically generate packets and send them."""
        packet_id = 1
        while self.running:
            source = random.choice(self.routers)
            destination = random.choice([r for r in self.routers if r != source])  # Choose a different router
            packet = Packet(packet_id, source.name, destination.name, self.env.now)
            source.send_packet(packet)
            packet_id += 1
            yield self.env.timeout(2)  # Generate a packet every 2 seconds

    def run_rl_step(self):
        """Run one step of the RL agent."""
        self.rl_agent.simulate_network(steps=1)  # Run the RL agent for 1 step
        if self.running:  # Only repeat after 1 second if running
            self.root.after(1000, self.run_rl_step)

