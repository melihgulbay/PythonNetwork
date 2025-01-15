import psutil
import time
import logging
from collections import deque
import numpy as np
from datetime import datetime
import os

class AgentProfiler:
    def __init__(self, log_directory="agent_logs"):
        """
        Initialize the agent profiler.
        
        Args:
            log_directory (str): Directory where log files will be stored
        """
        self.log_directory = log_directory
        self.ensure_log_directory()
        
        # Get number of CPU cores
        self.num_cores = psutil.cpu_count()
        
        # Initialize performance tracking dictionaries
        self.cpu_usage = {
            'dqn': deque(maxlen=1000),
            'ppo': deque(maxlen=1000),
            'a3c': deque(maxlen=1000),
            'reinforce': deque(maxlen=1000),
            'hybrid': deque(maxlen=1000)
        }
        
        # Add per-core CPU tracking
        self.cpu_per_core = {
            'dqn': {i: deque(maxlen=1000) for i in range(self.num_cores)},
            'ppo': {i: deque(maxlen=1000) for i in range(self.num_cores)},
            'a3c': {i: deque(maxlen=1000) for i in range(self.num_cores)},
            'reinforce': {i: deque(maxlen=1000) for i in range(self.num_cores)},
            'hybrid': {i: deque(maxlen=1000) for i in range(self.num_cores)}
        }
        
        self.memory_usage = {
            'dqn': deque(maxlen=1000),
            'ppo': deque(maxlen=1000),
            'a3c': deque(maxlen=1000),
            'reinforce': deque(maxlen=1000),
            'hybrid': deque(maxlen=1000)
        }
        
        self.execution_times = {
            'dqn': deque(maxlen=1000),
            'ppo': deque(maxlen=1000),
            'a3c': deque(maxlen=1000),
            'reinforce': deque(maxlen=1000),
            'hybrid': deque(maxlen=1000)
        }
        
        # Initialize loggers for each agent
        self.loggers = {}
        self.setup_loggers()
        
        # Track active agents
        self.active_agent = None
        self.start_time = None
        
        # Initialize process tracking
        self.process = psutil.Process()
        self.last_per_core_times = None

    def ensure_log_directory(self):
        """Create log directory if it doesn't exist."""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)

    def setup_loggers(self):
        """Set up individual loggers for each agent."""
        agent_types = ['dqn', 'ppo', 'a3c', 'reinforce', 'hybrid']
        
        for agent_type in agent_types:
            logger = logging.getLogger(f'agent_profiler_{agent_type}')
            logger.setLevel(logging.INFO)
            
            # Create file handler
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fh = logging.FileHandler(
                os.path.join(self.log_directory, f'{agent_type}_profile_{timestamp}.log')
            )
            fh.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(fh)
            self.loggers[agent_type] = logger

    def start_profiling(self, agent_type):
        """
        Start profiling a specific agent.
        
        Args:
            agent_type (str): Type of agent being profiled ('dqn', 'ppo', etc.)
        """
        self.active_agent = agent_type.lower()
        self.start_time = time.time()
        
        # Initialize CPU tracking
        cpu_times = self.process.cpu_times()
        self.last_cpu_time = cpu_times.user + cpu_times.system
        self.last_time = time.time()
        
        # Log start of profiling
        if self.active_agent in self.loggers:
            self.loggers[self.active_agent].info(f"Started profiling {agent_type} agent")

    def stop_profiling(self):
        """Stop profiling the current agent."""
        if self.active_agent and self.active_agent in self.loggers:
            self.loggers[self.active_agent].info(f"Stopped profiling {self.active_agent} agent")
        self.active_agent = None
        self.start_time = None

    def record_metrics(self):
        """Record current CPU and memory metrics for the active agent."""
        if not self.active_agent or not self.start_time:
            return

        try:
            # Get CPU times
            cpu_times = self.process.cpu_times()
            total_cpu_time = cpu_times.user + cpu_times.system
            
            # Calculate overall CPU percentage
            current_time = time.time()
            if hasattr(self, 'last_cpu_time'):
                time_diff = current_time - self.last_time
                cpu_diff = total_cpu_time - self.last_cpu_time
                cpu_percent = (cpu_diff / time_diff) * 100
            else:
                cpu_percent = 0
                
            # Get per-core CPU times
            current_per_core = psutil.cpu_percent(percpu=True)
            
            # Store current values for next calculation
            self.last_cpu_time = total_cpu_time
            self.last_time = current_time
            
            # Record memory usage (MB)
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            
            # Record execution time since start
            execution_time = time.time() - self.start_time
            
            # Store metrics
            self.cpu_usage[self.active_agent].append(cpu_percent)
            self.memory_usage[self.active_agent].append(memory_mb)
            self.execution_times[self.active_agent].append(execution_time)
            
            # Store per-core CPU usage
            for core_id, core_percent in enumerate(current_per_core):
                self.cpu_per_core[self.active_agent][core_id].append(core_percent)
            
            # Calculate average per-core usage for logging
            core_averages = [
                np.mean(list(self.cpu_per_core[self.active_agent][i])) 
                for i in range(self.num_cores)
            ]
            
            # Log metrics
            logger = self.loggers.get(self.active_agent)
            if logger:
                logger.info(
                    f"Metrics - Overall CPU: {cpu_percent:.1f}%, "
                    f"Memory: {memory_mb:.1f}MB, "
                    f"Execution Time: {execution_time:.3f}s"
                )
                logger.info(
                    f"Per-Core CPU Usage: " + 
                    ", ".join([f"Core {i}: {usage:.1f}%" 
                             for i, usage in enumerate(current_per_core)])
                )
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
            logger = self.loggers.get(self.active_agent)
            if logger:
                logger.error(f"Error recording metrics: {str(e)}")

    def get_agent_stats(self, agent_type):
        """
        Get statistics for a specific agent.
        
        Args:
            agent_type (str): Type of agent to get stats for
        
        Returns:
            dict: Dictionary containing agent statistics
        """
        if not self.cpu_usage[agent_type] or not self.memory_usage[agent_type]:
            return None
            
        # Get per-core CPU stats
        core_stats = {}
        for core in range(self.num_cores):
            core_data = self.cpu_per_core[agent_type][core]
            if core_data:
                core_stats[f'core_{core}'] = {
                    'mean': np.mean(core_data),
                    'max': np.max(core_data),
                    'min': np.min(core_data)
                }
            
        return {
            'cpu_usage': {
                'mean': np.mean(self.cpu_usage[agent_type]),
                'max': np.max(self.cpu_usage[agent_type]),
                'min': np.min(self.cpu_usage[agent_type]),
                'per_core': core_stats
            },
            'memory_usage': {
                'mean': np.mean(self.memory_usage[agent_type]),
                'max': np.max(self.memory_usage[agent_type]),
                'min': np.min(self.memory_usage[agent_type])
            },
            'execution_time': {
                'mean': np.mean(self.execution_times[agent_type]),
                'max': np.max(self.execution_times[agent_type]),
                'min': np.min(self.execution_times[agent_type])
            }
        }

    def log_summary_stats(self):
        """Log summary statistics for all agents that have been profiled."""
        for agent_type in self.loggers.keys():
            stats = self.get_agent_stats(agent_type)
            if stats:
                logger = self.loggers[agent_type]
                logger.info("Summary Statistics:")
                logger.info(f"Overall CPU Usage - Mean: {stats['cpu_usage']['mean']:.1f}%, "
                          f"Max: {stats['cpu_usage']['max']:.1f}%, "
                          f"Min: {stats['cpu_usage']['min']:.1f}%")
                
                # Log per-core statistics
                for core_id, core_stats in stats['cpu_usage']['per_core'].items():
                    logger.info(f"{core_id} - Mean: {core_stats['mean']:.1f}%, "
                              f"Max: {core_stats['max']:.1f}%, "
                              f"Min: {core_stats['min']:.1f}%")
                
                logger.info(f"Memory Usage - Mean: {stats['memory_usage']['mean']:.1f}MB, "
                          f"Max: {stats['memory_usage']['max']:.1f}MB, "
                          f"Min: {stats['memory_usage']['min']:.1f}MB")
                logger.info(f"Execution Time - Mean: {stats['execution_time']['mean']:.3f}s, "
                          f"Max: {stats['execution_time']['max']:.3f}s, "
                          f"Min: {stats['execution_time']['min']:.3f}s")

    def reset_stats(self, agent_type=None):
        """
        Reset statistics for a specific agent or all agents.
        
        Args:
            agent_type (str, optional): Agent type to reset. If None, reset all.
        """
        if agent_type:
            self.cpu_usage[agent_type].clear()
            self.memory_usage[agent_type].clear()
            self.execution_times[agent_type].clear()
        else:
            for agent in self.cpu_usage.keys():
                self.cpu_usage[agent].clear()
                self.memory_usage[agent].clear()
                self.execution_times[agent].clear() 