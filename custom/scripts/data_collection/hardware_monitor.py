import csv
import time
import platform
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union


class HardwareMonitor:
    
    def __init__(self, run_name: str = "training_run", steps_per_log: int = 10000, testing: bool = False, results_dir: Optional[Path] = None):
        """
        sets up all the variables for monitoring
        parameters: run_name, steps_per_log, testing, results_dir
        outputs: nothing, just initializes the object
        """
        # store all the parameters so we can use them later
        self.run_name = run_name
        self.steps_per_log = steps_per_log
        self.testing = testing
        self.step_count = 0  # count how many steps we logged
        
        # figure out where to save
        # if no directory given, we need to import it from SingleRunner
        # i do import here not at top to avoid circular import errors (learned this the hard way)
        if results_dir is None:
            from custom.scripts.training_runner.SingleRunner import RESULTS_DIR
            self.results_dir = RESULTS_DIR
        else:
            # convert to Path object if its not already
            self.results_dir = Path(results_dir)
        
        # these dicts store all the hardware info
        self.initial_hardware_info = {}  # info before training starts
        self.final_hardware_info = {}    # info after training ends
        
        # this list stores data for each step
        # each item is a dict with: step_number, time_elapsed, mean_reward,std_of_reward, cpu_percent, ram_percent
        self.step_data = []
        
        # we can store training metrics seperately too if needed
        self.training_metrics = {}
        
        # this holds the process object to monitor cpu for specific process
        # i use _process with underscore cause its kinda internal/private
        self._process = None
        
        # timestamps for when monitoring started and ended
        self.start_time = None
        self.end_time = None
        self._is_monitoring = False  # flag to know if were monitoring right now
        
    def record_initial_state(self) -> Dict[str, Any]:
        """
        records hardware info before training starts (os, cpu, ram)
        outputs: dict with initial hardware info
        """
        # record when we started
        self.start_time = time.time()
        
        # get os name using platform module
        os_name = platform.system()
        # on mac, platform.system() returns "Darwin" which is confusing
        # so i convert it to "macOS" which makes more sense
        if os_name == "Darwin":
            os_name = "macOS"
        
        # get cpu core info
        # logical=False gives physical cores (actual cpu chips)
        # logical=True gives logical cores (includes hyperthreading)
        cpu_physical_cores = psutil.cpu_count(logical=False)
        cpu_logical_cores = psutil.cpu_count(logical=True)
        
        # try to get cpu frequency (speed)
        # sometimes this doesnt work, so i wrap it in try/except
        try:
            cpu_freq = psutil.cpu_freq()
            # cpu frequency is in mhz, so divide by 1000 to get ghz
            # also round to 2 decimals to make it readable
            cpu_clock_ghz = round(cpu_freq.current / 1000, 2) if cpu_freq and cpu_freq.current else None
        except (AttributeError, TypeError):
            # if we cant get frequency, just set to None
            cpu_clock_ghz = None
        
        # get memory (ram) info
        memory_info = psutil.virtual_memory()
        # total ram is in bytes, so divide by (1024^2) to convert to megabytes
        ram_total_mb = memory_info.total / (1024 ** 2)
        # get the percent of ram currently in use
        ram_usage_percent_initial = memory_info.percent
        
        # now store everything in a dict with clear names
        # i use descriptive names so its obvious what each value is
        self.initial_hardware_info = {
            "timestamp": datetime.now().isoformat(),  # when we recorded this
            "operating_system": os_name,
            "cpu_physical_cores_count": cpu_physical_cores,
            "cpu_logical_cores_count": cpu_logical_cores,
            "cpu_clock_speed_ghz": cpu_clock_ghz,
            "total_ram_mb": round(ram_total_mb, 2),  # round to 2 decimals
            "initial_ram_usage_percent": ram_usage_percent_initial
        }
        
        # also add initial entry to step_data
        # this is step 0 (before training), so we dont have training metrics yet
        # thats why mean_reward, std_of_reward, and cpu_percent are None
        self.step_data.append({
            'step_number': 0,  # step 0 = before training
            'time_elapsed': 0.0,  # no time passed yet
            'mean_reward': None,  # no training data yet
            'std_of_reward': None,  # no training data yet
            'cpu_percent': None,  # we'll get this during training
            'cpu_source': 'system',  # cpu source (process or system)
            'ram_percent': ram_usage_percent_initial  # but we do have initial ram usage
        })
        
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
    
    def start_continuous_monitoring(self, process: Optional[psutil.Process] = None) -> None:
        """
        starts monitoring during training
        if u give it a process, it monitors that specific process cpu, otherwise system-wide
        outputs: nothing
        """
        # set flag to True so we know were monitoring
        self._is_monitoring = True
        self._process = process
        
        # make sure we have start time recorded
        if not self.start_time:
            self.start_time = time.time()
        
        # if we have specific process to monitor, we need to "prime" cpu measurement
        # this is a quirk of psutil - first call to cpu_percent() returns 0.0
        # so we call it once here with interval=None to get it ready
        if self._process:
            try:
                self._process.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # if process doesnt exist or we cant access it, just set to None
                # we'll fall back to system-wide monitoring
                self._process = None
    
    def stop_continuous_monitoring(self) -> None:
        """
        stops monitoring
        outputs: nothing
        """
        self._is_monitoring = False
    
    def log_step(self, step_number: int, time_elapsed: float, mean_reward: float, std_of_reward: float) -> None:
        """
        records metrics for one training step (cpu, ram, rewards, time)
        parameters: step_number, time_elapsed, mean_reward, std_of_reward
        outputs: nothing, saves data to step_data list
        """
        # if were not monitoring, dont do anything
        if not self._is_monitoring:
            return
        
        # increment step counter
        self.step_count += 1
        
        # get current ram usage from system
        memory_info = psutil.virtual_memory()
        ram_percent = memory_info.percent
        
        # get cpu usage
        cpu_source = "system"  # default is system-wide
        if self._process:
            # if we have specific process to monitor, try to get its cpu usage
            try:
                cpu_percent = self._process.cpu_percent(interval=None)
                cpu_source = "process"  # we got it from specific process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # if process disappeared or we lost access, fall back to system-wide
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_source = "system"  # had to fall back to system
        else:
            # no specific process, so just get overall system cpu usage
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_source = "system"  # system-wide monitoring
        
        # now save all this data in a dict and add to our list
        # this way we have record of everything that happened at this step
        self.step_data.append({
            'step_number': step_number,  # actual training step (10000, 20000, etc)
            'time_elapsed': time_elapsed,  # seconds since training started
            'mean_reward': mean_reward,  # average reward
            'std_of_reward': std_of_reward,  # reward consistency
            'cpu_percent': cpu_percent,  # cpu usage percent
            'cpu_source': cpu_source,  # whether cpu is from "process" (specific program) or "system" (overall)
            'ram_percent': ram_percent  # ram usage percent
        })
    
    def record_final_state(self) -> Dict[str, Any]:
        """
        records final state after training and calculates stats (peak/avg cpu/ram, elapsed time)
        outputs: dict with final hardware info and statistics
        """
        # record when we finished
        self.end_time = time.time()
        # stop monitoring
        self.stop_continuous_monitoring()
        
        # get final ram usage
        memory_info = psutil.virtual_memory()
        ram_usage_percent_final = memory_info.percent
        
        # calculate total time elapsed
        # make sure start_time exists to avoid errors
        elapsed_time = self.end_time - self.start_time if self.start_time else 0
        
        # now calculate some stats from all the data we collected
        if self.step_data:
            # extract all ram percentages from step data
            # filter out None values just in case
            ram_values = [step['ram_percent'] for step in self.step_data if step['ram_percent'] is not None]
            if ram_values:
                # find maximum (peak) ram usage
                peak_ram_percent = max(ram_values)
                # calculate average by summing all values and dividing by count
                average_ram_percent = sum(ram_values) / len(ram_values)
            else:
                # if we somehow have no ram values, just use final value
                peak_ram_percent = ram_usage_percent_final
                average_ram_percent = ram_usage_percent_final
            
            # extract all cpu percentages from step data
            # filter out None values just in case
            cpu_values = [step['cpu_percent'] for step in self.step_data if step['cpu_percent'] is not None]
            if cpu_values:
                # find maximum (peak) cpu usage
                peak_cpu_percent = max(cpu_values)
                # calculate average by summing all values and dividing by count
                average_cpu_percent = sum(cpu_values) / len(cpu_values)
                # get final cpu from last step that has cpu data
                final_cpu_percent = cpu_values[-1] if cpu_values else None
            else:
                # if we have no cpu values, set to None
                peak_cpu_percent = None
                average_cpu_percent = None
                final_cpu_percent = None
        else:
            # if we have no step data at all, use final values
            peak_ram_percent = ram_usage_percent_final
            average_ram_percent = ram_usage_percent_final
            peak_cpu_percent = None
            average_cpu_percent = None
            final_cpu_percent = None
        
        self.final_hardware_info = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "total_training_steps": self.step_count,
            "final_ram_usage_percent": ram_usage_percent_final,
            "peak_ram_usage_percent": peak_ram_percent,
            "average_ram_usage_percent": round(average_ram_percent, 2),
            "final_cpu_usage_percent": round(final_cpu_percent, 2) if final_cpu_percent is not None else None,
            "peak_cpu_usage_percent": round(peak_cpu_percent, 2) if peak_cpu_percent is not None else None,
            "average_cpu_usage_percent": round(average_cpu_percent, 2) if average_cpu_percent is not None else None
        }
        
        print("\n" + "="*60)
        print("HARDWARE INFORMATION (Post-Training)")
        print("="*60)
        print(f"Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        print(f"Total Training Steps: {self.step_count}")
        print(f"RAM Usage (Final): {ram_usage_percent_final}%")
        print(f"RAM Usage (Peak): {peak_ram_percent}%")
        print(f"RAM Usage (Average): {average_ram_percent:.2f}%")
        if final_cpu_percent is not None:
            print(f"CPU Usage (Final): {final_cpu_percent:.2f}%")
            print(f"CPU Usage (Peak): {peak_cpu_percent:.2f}%")
            print(f"CPU Usage (Average): {average_cpu_percent:.2f}%")
        else:
            print(f"CPU Usage: No data collected")
        print("="*60 + "\n")
        
        return self.final_hardware_info
    
    def record_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        stores training metrics in a dict (steps, time, rewards, etc)
        outputs: nothing, just saves metrics
        """
        self.training_metrics = metrics
    
    def get_hardware_data(self) -> Dict[str, Any]:
        """
        returns all collected data as a dict
        outputs: dict with all hardware info and step data
        """
        return {
            "run_name": self.run_name,
            "initial_hardware_info": self.initial_hardware_info,
            "final_hardware_info": self.final_hardware_info,
            "step_data": self.step_data,
            "training_metrics": self.training_metrics
        }
    
    def save_to_csv(self, output_dir: Optional[Path] = None) -> Path:
        """
        saves all collected data to csv files (step data and metadata if testing=True)
        outputs: path to the step data csv file
        """
        # figure out where to save
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)
        
        # create folder for this run
        # folder name is the run_name
        run_data_dir = output_dir / self.run_name
        # mkdir with parents=True creates all parent dirs if they dont exist
        # exist_ok=True means it wont error if folder already exists
        run_data_dir.mkdir(parents=True, exist_ok=True)
        
        # create csv file path
        ram_csv_path = run_data_dir / "hardware_ram_usage.csv"
        
        # open file for writing
        # 'w' means write mode (overwrites if file exists)
        # newline='' needed for csv on windows (doesnt hurt on mac/linux)
        # encoding='utf-8' makes sure it works with special characters
        with ram_csv_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # write header row with column names
            writer.writerow(['step_number', 'time_elapsed', 'mean_reward', 'std_of_reward', 'cpu_percent', 'cpu_source', 'ram_percent'])
            
            # now write each step as a row
            for step in self.step_data:
                # handle None values - if value is None, write empty string instead
                # this makes csv cleaner and easier to read
                writer.writerow([
                    step['step_number'],
                    step['time_elapsed'],
                    step['mean_reward'] if step['mean_reward'] is not None else '',
                    step['std_of_reward'] if step['std_of_reward'] is not None else '',
                    step['cpu_percent'] if step['cpu_percent'] is not None else '',
                    step.get('cpu_source', 'system'),  # cpu_source: "process" if from specific program, "system" if system-wide
                    step['ram_percent']
                ])
        
        print(f"\n✓ RAM usage data saved: {ram_csv_path}")
        
        # now handle metadata file (summary of all runs)
        # this file collects info from multiple training runs
        # when testing mode enabled, we save to TESTING_all_runs_metadata.csv
        metadata_path = output_dir / "TESTING_all_runs_metadata.csv"
        
        # create dict with all summary info
        # this is different from step_data - this is just one row per run
        hardware_metadata = {
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
            'ram_readings_count': len(self.step_data),
            'final_ram_percent': self.final_hardware_info.get('final_ram_usage_percent', 'N/A'),
            'final_cpu_percent': self.final_hardware_info.get('final_cpu_usage_percent', 'N/A'),
            'peak_cpu_percent': self.final_hardware_info.get('peak_cpu_usage_percent', 'N/A'),
            'average_cpu_percent': self.final_hardware_info.get('average_cpu_usage_percent', 'N/A'),
            'data_folder': str(ram_csv_path)
        }
        
        # only save metadata csv if testing mode is enabled
        # this is cause metadata file can get big if u run lots of experiments
        if self.testing:
            # check if file already exists
            # if it does, we dont want to write header again
            file_exists = metadata_path.exists()
            
            # open in append mode ('a') so we add to file instead of overwriting
            with metadata_path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=hardware_metadata.keys())
                
                # only write header if this is new file
                if not file_exists:
                    writer.writeheader()
                
                # write this run's metadata as new row
                writer.writerow(hardware_metadata)
            
            print(f"✓ Metadata updated: {metadata_path}")
            print(f"✓ Testing mode enabled - Hardware_metadata saved to CSV")
        else:
            # if testing mode is off, we just keep data in memory
            print(f"✓ Testing mode disabled - Hardware_metadata kept in dictionary only")
            print(f"✓ RAM data file: {ram_csv_path}")
        
        return ram_csv_path
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        gets current memory usage info (used, available, percent)
        outputs: dict with current ram usage info
        """
        # get memory info from system
        memory_info = psutil.virtual_memory()
        
        # convert bytes to megabytes and round to 2 decimals
        return {
            "used_ram_mb": round(memory_info.used / (1024 ** 2), 2),
            "available_ram_mb": round(memory_info.available / (1024 ** 2), 2),
            "ram_usage_percent": memory_info.percent
        }


class HardwareMonitorContext:
    """
    context manager for HardwareMonitor, handles setup/cleanup automatically
    use with "with" statement to automatically record initial/final state
    """
    
    def __init__(self, run_name: str = "training_run", steps_per_log: int = 5000, testing: bool = False, results_dir: Optional[Path] = None):
        """
        creates context manager with same parameters as HardwareMonitor
        outputs: nothing, just initializes
        """
        # create HardwareMonitor instance
        self.monitor = HardwareMonitor(run_name, steps_per_log, testing, results_dir)
    
    def __enter__(self) -> HardwareMonitor:
        """
        called when entering "with" block, records initial state and starts monitoring
        outputs: HardwareMonitor instance
        """
        self.monitor.record_initial_state()
        self.monitor.start_continuous_monitoring()
        return self.monitor
    
    def set_process(self, process: Optional[psutil.Process]) -> None:
        """
        sets which process to monitor for cpu usage
        outputs: nothing
        """
        self.monitor._process = process
        if process:
            try:
                # prime cpu measurement (same as in start_continuous_monitoring)
                process.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # if we cant access it, just ignore it
                pass
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        called when exiting "with" block, records final state and saves to csv
        outputs: nothing
        """
        self.monitor.record_final_state()
        # also save to csv automatically
        self.monitor.save_to_csv()
        return False

