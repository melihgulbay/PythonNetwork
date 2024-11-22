# link.py
import random
import simpy

class Link:
    def __init__(self, env, source, target, capacity, delay, loss_probability=0.01):
        self.env = env
        self.source = source          # Source router of the link
        self.target = target          # Target router of the link
        self.capacity = capacity      # Maximum bandwidth of the link
        self.base_delay = delay       # Base delay for the link
        self.loss_probability = loss_probability  # Probability of packet loss
        self.queue = simpy.Store(env) # Packet queue
        self.current_load = 0         # Current load (how much of the capacity is being used)

    @property
    def load(self):
        """Returns the current load of the link as a percentage of its capacity."""
        return self.current_load

    def transmit(self, packet, packet_size):
        """Simulate packet transmission over the link."""
        actual_delay = self.base_delay + random.uniform(-self.base_delay * 0.1, self.base_delay * 0.1)

        # Simulate packet loss based on the loss probability
        if random.random() < self.loss_probability:
            print(f"Packet {packet.id} lost in transmission at {self.env.now}")
            self.lost_packets += 1  # Increment lost packet count
            return False

        # Check if the link is overloaded before transmitting the packet
        if self.current_load + packet_size > self.capacity:
            print(f"Packet {packet.id} dropped due to overload at {self.env.now}")
            self.dropped_packets += 1  # Increment dropped packet count
            return False
        else:
            # Transmit the packet (increase the load)
            self.current_load += packet_size
            print(f"Packet {packet.id} transmitted at {self.env.now} with size {packet_size}. Current load: {self.current_load}/{self.capacity}")

            yield self.queue.put(packet)  # Simulate packet being placed in the queue
            yield self.env.timeout(packet_size / self.capacity)  # Simulate the time to transmit the packet
            self.current_load -= packet_size  # Decrease the load after transmission
            return True


    def adjust_capacity(self, new_capacity):
        """Adjust the link's bandwidth capacity."""
        new_capacity = min(max(new_capacity, self.min_capacity), self.max_capacity)  # Ensure within bounds
        if new_capacity != self.capacity:
            print(f"Link capacity updated from {self.capacity} to {new_capacity} at {self.env.now}")
            self.capacity = new_capacity

