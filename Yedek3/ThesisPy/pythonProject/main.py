#main.py
import tkinter as tk
import simpy
from router import Router
from link import Link
from sdn_controller import SDNController
from rl_agent import RLAgent
from ui import NetworkSimulatorUI

# Initialize the Tkinter window and environment
root = tk.Tk()
env = simpy.Environment()

# Define routers
router_A = Router(env, "A", None)
router_B = Router(env, "B", None)
router_C = Router(env, "C", None)

# Define links
link_AB = Link(env, router_A, router_B, capacity=10, delay=10)
link_BC = Link(env, router_B, router_C, capacity=20, delay=5)

links = [link_AB, link_BC]
routers = [router_A, router_B, router_C]

# Set up SDN controller
sdn_controller = SDNController(env, routers, links)

# Assign SDN controller to routers
for router in routers:
    router.sdn_controller = sdn_controller

# Set up RL Agent
rl_agent = RLAgent(links)

# SDN controller process to periodically update routing tables
env.process(sdn_controller.update_routing_tables())

# Set up the UI, passing the environment, routers, and RL agent
ui = NetworkSimulatorUI(root, env, routers, rl_agent)
root.mainloop()
