import psutil
import time
import os
import platform
import csv
from datetime import datetime
from typing import Dict, List, Optional, Any


class HardwareMonitor:
 #Monitors and records hardware information during program execution.
   
    
    def __init__(self, run_name: str = "training_run", steps_per_log: int = 5000, testing: bool = False):
        """
        Initialize the HardwareMonitor.
        
        Args:
            run_name (str): Name of the current training run. Default is "training_run".
            steps_per_log (int): Log RAM usage every N steps. Default is 5000.
            testing (bool): If True, save Hardware_metadata to CSV file. If False, only keep in dictionary. Default is False.
        """
        self.run_name = run_name
        self.steps_per_log = steps_per_log
        self.testing = testing
        self.step_count = 0
        
        # Dictionaries to store hardware information with understandable names
        self.initial_hardware_info = {}
        self.final_hardware_info = {}
        self.ram_usage_per_step = []  # List of RAM% values every steps_per_log steps
        
        # Training metrics captured from mlagents-learn output
        self.training_metrics = {}
        
        self.start_time = None
        self.end_time = None
        self._is_monitoring = False
        
    def record_initial_state(self) -> Dict[str, Any]:
        """
        Record initial hardware state before training starts inside of a dictionary
        Captures:
        - Operating System name
        - CPU physical cores
        - CPU logical cores
        - CPU clock speed (GHz)
        - Total RAM (MB)
        - Current RAM usage percentage
        """
        self.start_time = time.time()
        
        # Get OS information
        os_name = platform.system()
        # Convert Darwin to macOS for better readability
        if os_name == "Darwin":
            os_name = "macOS"
        
        # Get CPU information
        cpu_physical_cores = psutil.cpu_count(logical=False)
        cpu_logical_cores = psutil.cpu_count(logical=True)
        
        # Get CPU frequency in GHz
        try:
            cpu_clock_ghz = psutil.cpu_freq().current / 1000  # Convert MHz to GHz
        except (AttributeError, TypeError):
            cpu_clock_ghz = None
        
        # Get memory information
        memory_info = psutil.virtual_memory()
        ram_total_mb = memory_info.total / (1024 ** 2)
        ram_usage_percent_initial = memory_info.percent
        
        # Store in dictionary with understandable names
        self.initial_hardware_info = {
            "timestamp": datetime.now().isoformat(),
            "operating_system": os_name,
            "cpu_physical_cores_count": cpu_physical_cores,
            "cpu_logical_cores_count": cpu_logical_cores,
            "cpu_clock_speed_ghz": cpu_clock_ghz,
            "total_ram_mb": round(ram_total_mb, 2),
            "initial_ram_usage_percent": ram_usage_percent_initial
        }
        
        # Record initial memory usage
        self.ram_usage_per_step.append(ram_usage_percent_initial)
        
        print("\n" + "="*60)
        print("HARDWARE INFORMATION (Pre-Training)")
        print("="*60)
        print(f"Operating System: {os_name}")
        print(f"CPU Physical Cores: {cpu_physical_cores}")
        print(f"CPU Logical Cores: {cpu_logical_cores}")
        if cpu_clock_ghz:
            print(f"CPU Clock Speed: {cpu_clock_ghz:.2f} GHz")
        print(f"Total RAM: {ram_total_mb:.2f} MB ({ram_total_mb/1024:.2f} GB)")
        print(f"RAM Usage: {ram_usage_percent_initial}%")
        print("="*60 + "\n")
        
        return self.initial_hardware_info
    
    def start_continuous_monitoring(self) -> None:
        """
        Start continuous monitoring of RAM usage during training.
        
        This method should be called after record_initial_state() to begin
        tracking RAM usage throughout the training process.
        """
        self._is_monitoring = True
        if not self.start_time:
            self.start_time = time.time()
    
    def stop_continuous_monitoring(self) -> None:
        """Stop continuous monitoring of RAM usage."""
        self._is_monitoring = False
    
    def log_step(self) -> None:
        """
        Log a training step. Call this for each training step/iteration.
        
        Records RAM usage and increments step count.
        """
        if not self._is_monitoring:
            return
        
        # Record RAM usage for this step
        memory_info = psutil.virtual_memory()
        self.ram_usage_per_step.append(memory_info.percent)
    
    def record_final_state(self) -> Dict[str, Any]:
        """
        Record final hardware state after training completes.
        
        Calculates:
        - RAM usage percentage at end
        - Total elapsed time
        - Peak RAM usage during training
        - Average RAM usage during training
        
        Returns:
            Dictionary with easily understandable final hardware information.
        """
        self.end_time = time.time()
        self.stop_continuous_monitoring()
        
        memory_info = psutil.virtual_memory()
        ram_usage_percent_final = memory_info.percent
        
        elapsed_time = self.end_time - self.start_time
        
        # Calculate statistics from ram usage history
        if self.ram_usage_per_step:
            peak_ram_percent = max(self.ram_usage_per_step)
            average_ram_percent = sum(self.ram_usage_per_step) / len(self.ram_usage_per_step)
        else:
            peak_ram_percent = ram_usage_percent_final
            average_ram_percent = ram_usage_percent_final
        
        self.final_hardware_info = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "total_training_steps": self.step_count,
            "final_ram_usage_percent": ram_usage_percent_final,
            "peak_ram_usage_percent": peak_ram_percent,
            "average_ram_usage_percent": round(average_ram_percent, 2)
        }
        
        print("\n" + "="*60)
        print("HARDWARE INFORMATION (Post-Training)")
        print("="*60)
        print(f"Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Total Training Steps: {self.step_count}")
        print(f"RAM Usage (Final): {ram_usage_percent_final}%")
        print(f"RAM Usage (Peak): {peak_ram_percent}%")
        print(f"RAM Usage (Average): {average_ram_percent:.2f}%")
        print("="*60 + "\n")
        
        return self.final_hardware_info
    
    def record_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Record training metrics from mlagents-learn output.
            - steps: Current training step number
            - time_elapsed: Elapsed time in seconds
            - mean_reward: Mean reward for the environment
            - std_of_reward: Standard deviation of reward
        """
        self.training_metrics = metrics
    
    def get_hardware_data(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "initial_hardware_info": self.initial_hardware_info,
            "final_hardware_info": self.final_hardware_info,
            "ram_usage_per_step": self.ram_usage_per_step,
            "training_metrics": self.training_metrics
        }
    
    def save_to_csv(self, output_dir: str = "results") -> str:
        """
        Save hardware monitoring data to organized folder structure.
        
        Creates:
        - results/run_data/{run_name}_ram_usage.csv with RAM readings per step
        - results/all_runs_metadata.csv with all run information (only in testing mode)
        """
        # Create base directories
        os.makedirs(output_dir, exist_ok=True)
        run_data_dir = os.path.join(output_dir, "run_data")
        os.makedirs(run_data_dir, exist_ok=True)
        
        # Save RAM usage data with run_name prefix directly in run_data
        ram_csv_path = os.path.join(run_data_dir, f"{self.run_name}_ram_usage.csv")
        
        with open(ram_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['step_index', 'ram_percent'])
            # Data - each RAM reading on a separate row
            for idx, ram_val in enumerate(self.ram_usage_per_step):
                writer.writerow([idx, ram_val])
        
        print(f"\n✓ RAM usage data saved: {ram_csv_path}")
        
        # Update central metadata file for all runs
        metadata_path = os.path.join(output_dir, "all_runs_metadata.csv")
        
        # Prepare hardware-only metadata dictionary (excludes training metrics)
        Hardware_metadata = {
            'run_name': self.run_name,
            'timestamp': self.initial_hardware_info.get('timestamp', 'N/A'),
            'operating_system': self.initial_hardware_info.get('operating_system', 'N/A'),
            'cpu_physical_cores': self.initial_hardware_info.get('cpu_physical_cores_count', 'N/A'),
            'cpu_logical_cores': self.initial_hardware_info.get('cpu_logical_cores_count', 'N/A'),
            'cpu_clock_ghz': self.initial_hardware_info.get('cpu_clock_speed_ghz', 'N/A'),
            'total_ram_mb': self.initial_hardware_info.get('total_ram_mb', 'N/A'),
            'initial_ram_percent': self.initial_hardware_info.get('initial_ram_usage_percent', 'N/A'),
            'elapsed_time_seconds': self.final_hardware_info.get('elapsed_time_seconds', 'N/A'),
            'peak_ram_percent': self.final_hardware_info.get('peak_ram_usage_percent', 'N/A'),
            'average_ram_percent': self.final_hardware_info.get('average_ram_usage_percent', 'N/A'),
            'ram_readings_count': len(self.ram_usage_per_step),
            'final_ram_percent': self.final_hardware_info.get('final_ram_usage_percent', 'N/A'),
            'data_folder': ram_csv_path
        }
        
        # Only save to CSV if testing is True
        if self.testing:
            # Check if file exists to determine if we need to write header
            metadata_path = os.path.join(output_dir, "all_runs_metadata.csv")
            file_exists = os.path.exists(metadata_path)
            
            with open(metadata_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=Hardware_metadata.keys())
                
                # Write header only if file is new
                if not file_exists:
                    writer.writeheader()
                
                # Write metadata row
                writer.writerow(Hardware_metadata)
            
            print(f"✓ Metadata updated: {metadata_path}")
            print(f"✓ Testing mode enabled - Hardware_metadata saved to CSV")
        else:
            print(f"✓ Testing mode disabled - Hardware_metadata kept in dictionary only")
            print(f"✓ RAM data file: {ram_csv_path}")
        
        return ram_csv_path
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage information.
        
        Returns:
            Dictionary with easily understandable memory info:
            - used_ram_mb: RAM currently used in MB
            - available_ram_mb: RAM available in MB
            - ram_usage_percent: RAM usage percentage
        """
        memory_info = psutil.virtual_memory()
        return {
            "used_ram_mb": round(memory_info.used / (1024 ** 2), 2),
            "available_ram_mb": round(memory_info.available / (1024 ** 2), 2),
            "ram_usage_percent": memory_info.percent
        }
    
    def __init__(self, run_name: str = "training_run", steps_per_log: int = 5000):
        """
        Initialize the context manager.
        
        Args:
            run_name (str): Name of the training run.
            steps_per_log (int): Log RAM usage every N steps.
        """
        self.monitor = HardwareMonitor(run_name, steps_per_log)
    
    def __enter__(self) -> HardwareMonitor:
        """Enter context: record initial state and start monitoring."""
        self.monitor.record_initial_state()
        self.monitor.start_continuous_monitoring()
        return self.monitor
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context: record final state and stop monitoring."""
        self.monitor.record_final_state()
        return False


