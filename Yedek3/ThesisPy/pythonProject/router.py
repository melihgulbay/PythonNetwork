# router.py
import simpy

class Router:
    def __init__(self, env, name, sdn_controller, buffer_capacity=100):
        self.env = env
        self.name = name
        self.sdn_controller = sdn_controller  # Reference to the SDN controller
        self.routing_table = {}  # Will be updated by SDN controller
        self.buffer = simpy.Store(env, capacity=buffer_capacity)
        self.process = env.process(self.route_packets())
        self.received_packets = 0
        self.sent_packets = 0
        self.lost_packets = 0
        self.dropped_packets = 0
        self.total_latency = 0
        self.total_transmission_time = 0
        self.total_packet_count = 0

    def route_packets(self):
        while True:
            packet = yield self.buffer.get()
            # Query the SDN controller for the next hop
            next_hop, link = self.routing_table.get(packet.destination, (None, None))
            if next_hop:
                start_time = self.env.now
                self.total_packet_count += 1
                if (yield self.env.process(link.transmit(packet, 1))):
                    end_time = self.env.now
                    self.total_latency += end_time - start_time
                    self.total_transmission_time += end_time - start_time
                    self.received_packets += 1
                else:
                    self.lost_packets += 1
                    self.dropped_packets += 1
            else:
                self.dropped_packets += 1

    def send_packet(self, packet):
        self.sent_packets += 1
        try:
            self.buffer.put(packet)
        except simpy.StoreFull:
            self.dropped_packets += 1

    def get_metrics(self):
        latency = self.total_latency / max(1, self.received_packets)
        drop_rate = self.dropped_packets / max(1, self.received_packets + self.dropped_packets)
        avg_transmission_time = self.total_transmission_time / max(1, self.received_packets)
        return {
            "latency": latency,
            "drop_rate": drop_rate,
            "sent_packets": self.sent_packets,
            "received_packets": self.received_packets,
            "lost_packets": self.lost_packets,
            "dropped_packets": self.dropped_packets,
            "avg_transmission_time": avg_transmission_time
        }
