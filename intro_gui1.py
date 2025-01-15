import tkinter as tk
from tkinter import ttk
import time
import threading
import sys
import platform
import psutil
import torch
import numpy as np
from gpu_utils import get_device_info

class IntroGUI:
    def __init__(self, num_nodes, seed):
        self.root = tk.Tk()
        self.root.title("SDN Simulation Initializing")
        
        # Set window size and position
        window_width = 1280
        window_height = 720  # Reduced height since we removed agents section
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # Make window non-resizable
        self.root.resizable(False, False)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Add title
        self.title_label = ttk.Label(
            self.main_frame,
            text="SDN Network Simulation",
            font=('Helvetica', 24, 'bold')
        )
        self.title_label.grid(row=0, column=0, pady=(0, 40))
        
        # Add system info frame
        self.info_frame = ttk.LabelFrame(self.main_frame, text="System Information", padding="20")
        self.info_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 40))
        
        # Gather system information
        system_info = self._get_system_info()
        
        # Add system info labels
        for i, (key, value) in enumerate(system_info.items()):
            ttk.Label(self.info_frame, text=f"{key}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 10))
            ttk.Label(self.info_frame, text=value).grid(row=i, column=1, sticky=tk.W)
        
        # Add configuration frame
        self.config_frame = ttk.LabelFrame(self.main_frame, text="Simulation Configuration", padding="20")
        self.config_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 40))
        
        # Add configuration info with more details
        ttk.Label(self.config_frame, text="Number of Nodes:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(self.config_frame, text=str(num_nodes)).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(self.config_frame, text="Random Seed:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        ttk.Label(self.config_frame, text=str(seed if seed is not None else "Random")).grid(row=1, column=1, sticky=tk.W)
        
        # Add network parameters frame
        self.network_frame = ttk.LabelFrame(self.main_frame, text="Network Parameters", padding="20")
        self.network_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 40))
        
        # Add network parameters
        network_params = {
            "Packet Generation Rate": "0.2 - 0.6 packets/step",
            "Packet Size Range": "64 - 1500 bytes",
            "Max Latency": "20.0 ms",
            "Min Throughput": "20%",
            "Max Packet Loss": "50%",
            "Protocols": "TCP, UDP, HTTP, HTTPS, ICMP"
        }
        
        for i, (key, value) in enumerate(network_params.items()):
            ttk.Label(self.network_frame, text=f"{key}:").grid(row=i, column=0, sticky=tk.W, padx=(0, 10))
            ttk.Label(self.network_frame, text=value).grid(row=i, column=1, sticky=tk.W)
        
        # Move progress frame to bottom
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        
        # Add progress bar
        self.progress = ttk.Progressbar(
            self.progress_frame,
            length=800,
            mode='determinate'
        )
        self.progress.grid(row=0, column=0, pady=(0, 10))
        
        # Add status label
        self.status_label = ttk.Label(
            self.progress_frame,
            text="Initializing...",
            font=('Helvetica', 12)
        )
        self.status_label.grid(row=1, column=0)
        
        # Initialize progress variables
        self.progress_value = 0
        self.loading_complete = False
        
        # Start progress update in separate thread
        self.progress_thread = threading.Thread(target=self._update_progress)
        self.progress_thread.daemon = True
        self.progress_thread.start()

    def _get_system_info(self):
        """Gather system information"""
        
        # Get CPU info
        try:
            if platform.system() == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
                winreg.CloseKey(key)
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            break
            else:  # macOS and others
                cpu_name = platform.processor()
        except:
            cpu_name = platform.processor()

        # Get GPU info using the utility function
        gpu_info = get_device_info()
        
        return {
            "OS": f"{platform.system()} {platform.release()}",
            "CPU": cpu_name,
            "RAM": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
            "Python": sys.version.split()[0],
            "PyTorch": torch.__version__,
            "NumPy": np.__version__,
            "GPU": f"{gpu_info['type']} ({gpu_info['available']})"
        }

    def _update_progress(self):
        """Update progress bar and status messages"""
        status_messages = [
            "Initializing system...",
            "Loading network configuration...",
            "Initializing SDN layers...",
            "Setting up network parameters...",
            "Preparing visualization components...",
            "Starting simulation..."
        ]
        
        # Increase step duration from 0.5 to 1.3 seconds
        step_duration = 1.3  # seconds
        
        for i, message in enumerate(status_messages):
            self.status_label.config(text=message)
            start_progress = (i / len(status_messages)) * 100
            end_progress = ((i + 1) / len(status_messages)) * 100
            
            start_time = time.time()
            
            while self.progress_value < end_progress:
                elapsed = (time.time() - start_time) / step_duration
                self.progress_value = start_progress + (end_progress - start_progress) * min(1, elapsed)
                self.progress['value'] = self.progress_value
                time.sleep(0.01)
        
        self.loading_complete = True
        self.root.after(1000, self.root.destroy)  # Close after 1 second delay instead of 500ms

    def run(self):
        """Start the intro GUI"""
        self.root.mainloop() 