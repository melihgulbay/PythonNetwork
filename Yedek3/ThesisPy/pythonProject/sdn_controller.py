# sdn_controller.py
import random

class SDNController:
    def __init__(self, env, routers, links):
        self.env = env
        self.routers = routers
        self.links = links
        self.routing_table = {}

    def compute_routes(self):
        """Compute the best routes for all routers based on current network conditions."""
        for router in self.routers:
            # Example: Compute routes based on available links
            router.routing_table = {}  # Clear the previous routing table
            for link in self.links:
                # Create routing table based on source and target router names
                if router.name == link.source.name:
                    router.routing_table[link.target.name] = (link.target.name, link)
                elif router.name == link.target.name:
                    router.routing_table[link.source.name] = (link.source.name, link)

            print(f"Updated routing table for Router {router.name}: {router.routing_table}")
    
    def update_routing_tables(self):
        """Periodically update all routers with new routing tables."""
        while True:
            self.compute_routes()  # Compute the routes periodically
            yield self.env.timeout(10)  # Update every 10 seconds
