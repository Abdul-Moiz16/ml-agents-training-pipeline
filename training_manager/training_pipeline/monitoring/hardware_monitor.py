import csv
import time
import platform
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any


class HardwareMonitor:
    """Monitors CPU and RAM usage during training runs."""
    
    def __init__(self, run_name: str = "training_run", steps_per_log: int = 10000, testing: bool = False, results_dir: Optional[Path] = None):
        self.run_name = run_name
        self.steps_per_log = steps_per_log
        self.testing = testing
        self.step_count = 0
        
        if results_dir is None:
            from training_manager.training_pipeline.io_utils.paths import Paths
            self.results_dir = Paths().results_dir
        else:
            self.results_dir = Path(results_dir)
        
        self.initial_hardware_info: Dict[str, Any] = {}
        self.final_hardware_info: Dict[str, Any] = {}
        self.step_data = []
        self.training_metrics = {}
        
        self._process: Optional[psutil.Process] = None
        self.start_time = None
        self.end_time = None
        self._is_monitoring = False

        self._csv_file: Optional[Any] = None
        self._csv_writer: Optional[csv.writer] = None
        self._csv_path: Optional[Path] = None
        self._existing_log: bool = False
        self.time_offset: float = 0.0  # used when resuming to keep elapsed time monotonic

    def _ensure_csv_writer(self) -> None:
        """Open CSV file and write header if not already open."""
        if self._csv_writer is not None:
            return

        # Expect run_name to already include machine_id/run_id when invoked from Runner
        run_data_dir = self.results_dir / self.run_name / "run_logs"
        run_data_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = run_data_dir / "run_log.csv"


        # this is when we resume the trainig it was losingteh initial runlog and only started new runlog from the new step now its fixed and it keeps teh previous runlog 
        file_exists = self._csv_path.exists()

        if file_exists:
            try:
                with self._csv_path.open('r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    last_row = None
                    for row in reader:
                        last_row = row
                    if last_row:
                        try:
                            self.time_offset = float(last_row.get("time_elapsed", 0) or 0)
                        except (TypeError, ValueError):
                            self.time_offset = 0.0
                        try:
                            self.step_count = int(last_row.get("step_number") or last_row.get("steps") or 0)
                        except (TypeError, ValueError):
                            pass
                self._existing_log = True
            except Exception:
                self.time_offset = 0.0
                self._existing_log = False

        self._csv_file = self._csv_path.open('a', newline='', encoding='utf-8')
        self._csv_writer = csv.writer(self._csv_file)
        if not file_exists:
            self._csv_writer.writerow([
                'step_number', 'time_elapsed', 'mean_reward', 'std_of_reward', 
                'cpu_percent', 'cpu_source', 'ram_percent', 'ram_source', 'ram_mb'
            ])

    def _write_step_row(self, step: Dict[str, Any]) -> None:
        """Append a step row to CSV and flush to disk."""
        self._ensure_csv_writer()
        if not self._csv_writer:
            return
        # avoid duplicating the baseline row when resuming
        if self._existing_log and step.get('step_number') == 0:
            return

        self._csv_writer.writerow([
            step.get('step_number'),
            step.get('time_elapsed'),
            step.get('mean_reward') if step.get('mean_reward') is not None else '',
            step.get('std_of_reward') if step.get('std_of_reward') is not None else '',
            step.get('cpu_percent') if step.get('cpu_percent') is not None else '',
            step.get('cpu_source', 'system'),
            step.get('ram_percent'),
            step.get('ram_source', 'system'),
            step.get('ram_mb') if step.get('ram_mb') is not None else '',
        ])
        if self._csv_file:
            self._csv_file.flush()

    def _close_csv_writer(self) -> None:
        """Close the CSV file handle."""
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass
        self._csv_file = None
        self._csv_writer = None
        
    def record_initial_state(self) -> Dict[str, Any]:
        """Record hardware specs and initial RAM usage before training starts."""
        self.start_time = time.time()
        
        os_name = platform.system()
        if os_name == "Darwin":
            os_name = "macOS"
        
        cpu_physical_cores = psutil.cpu_count(logical=False)
        cpu_logical_cores = psutil.cpu_count(logical=True)
        
        try:
            cpu_freq = psutil.cpu_freq()
            cpu_clock_ghz = round(cpu_freq.current / 1000, 2) if cpu_freq and cpu_freq.current else None
        except (AttributeError, TypeError):
            cpu_clock_ghz = None
        
        memory_info = psutil.virtual_memory()
        ram_total_mb = memory_info.total / (1024 ** 2)
        ram_usage_percent_initial = memory_info.percent
        
        self.initial_hardware_info = {
            "timestamp": datetime.now().isoformat(),
            "operating_system": os_name,
            "cpu_physical_cores_count": cpu_physical_cores,
            "cpu_logical_cores_count": cpu_logical_cores,
            "cpu_clock_speed_ghz": cpu_clock_ghz,
            "total_ram_mb": round(ram_total_mb, 2),
            "initial_ram_usage_percent": ram_usage_percent_initial
        }
        
        # Step 0: system baseline before training process starts
        initial_step = {
            'step_number': 0,
            'time_elapsed': 0.0,
            'mean_reward': None,
            'std_of_reward': None,
            'cpu_percent': None,
            'cpu_source': 'system',
            'ram_percent': ram_usage_percent_initial,
            'ram_source': 'system',
            'ram_mb': None
        }
        self.step_data.append(initial_step)
        self._write_step_row(initial_step)
        
        return self.initial_hardware_info
    
    def start_continuous_monitoring(self, process: Optional[psutil.Process] = None) -> None:
        """Start monitoring. Optionally set a specific process to track."""
        self._is_monitoring = True
        self._process = process
        
        if not self.start_time:
            self.start_time = time.time()
        
        if self._process:
            try:
                self._process.cpu_percent(interval=None)
                # prime children too so first real sample isn't zero
                for child in self._process.children(recursive=True):
                    try:
                        child.cpu_percent(interval=None)
                    except psutil.Error:
                        continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self._process = None
    
    def stop_continuous_monitoring(self) -> None:
        """Stop monitoring."""
        self._is_monitoring = False
    
    def log_step(self, step_number: int, time_elapsed: float, mean_reward: float, std_of_reward: float) -> None:
        """Log hardware metrics for a training step."""
        if not self._is_monitoring:
            return
        
        self.step_count += 1
        system_memory = psutil.virtual_memory()
        total_ram_bytes = system_memory.total

        cpu_percent = None
        ram_percent = system_memory.percent
        ram_mb = None
        cpu_source = "system"
        ram_source = "system"

        if self._process:
            try:
                procs = [self._process] + self._process.children(recursive=True)
                cpu_vals = []
                ram_vals = []
                for p in procs:
                    try:
                        cpu_vals.append(p.cpu_percent(interval=0.1))
                        ram_vals.append(p.memory_info().rss)
                    except psutil.Error:
                        continue
                if cpu_vals:
                    raw_cpu = sum(cpu_vals)
                    cpu_percent = raw_cpu / psutil.cpu_count(logical=True)
                    cpu_source = "process_tree"
                if ram_vals:
                    ram_bytes = sum(ram_vals)
                    ram_mb = round(ram_bytes / (1024 ** 2), 2)
                    ram_percent = round((ram_bytes / total_ram_bytes) * 100, 2)
                    ram_source = "process_tree"
            except psutil.Error:
                pass

        if cpu_percent is None:
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_source = "system"
        
        adjusted_time = time_elapsed + self.time_offset

        step = {
            'step_number': step_number,
            'time_elapsed': adjusted_time,
            'mean_reward': mean_reward,
            'std_of_reward': std_of_reward,
            'cpu_percent': cpu_percent,
            'cpu_source': cpu_source,
            'ram_percent': ram_percent,
            'ram_source': ram_source,
            'ram_mb': ram_mb
        }
        self.step_data.append(step)
        self._write_step_row(step)
    
    def record_final_state(self) -> Dict[str, Any]:
        """Record final state and compute summary statistics."""
        self.end_time = time.time()
        self.stop_continuous_monitoring()
        
        memory_info = psutil.virtual_memory()
        ram_usage_percent_final = memory_info.percent
        elapsed_time = self.end_time - self.start_time if self.start_time else 0
        
        # Process vs system stats
        if self.step_data:
            process_ram = [s['ram_percent'] for s in self.step_data 
                          if s['ram_percent'] is not None and s.get('ram_source') == 'process']
            system_ram = [s['ram_percent'] for s in self.step_data 
                          if s['ram_percent'] is not None and s.get('ram_source') == 'system']
            if process_ram:
                peak_ram_percent = max(process_ram)
                average_ram_percent = sum(process_ram) / len(process_ram)
            elif system_ram:
                peak_ram_percent = max(system_ram)
                average_ram_percent = sum(system_ram) / len(system_ram)
            else:
                peak_ram_percent = ram_usage_percent_final
                average_ram_percent = ram_usage_percent_final
            
            process_cpu = [s['cpu_percent'] for s in self.step_data 
                           if s['cpu_percent'] is not None and s.get('cpu_source') == 'process']
            system_cpu = [s['cpu_percent'] for s in self.step_data 
                          if s['cpu_percent'] is not None and s.get('cpu_source') == 'system']
            cpu_values = process_cpu if process_cpu else system_cpu
            if cpu_values:
                peak_cpu_percent = max(cpu_values)
                average_cpu_percent = sum(cpu_values) / len(cpu_values)
                final_cpu_percent = cpu_values[-1]
            else:
                peak_cpu_percent = None
                average_cpu_percent = None
                final_cpu_percent = None
        else:
            peak_ram_percent = ram_usage_percent_final
            average_ram_percent = ram_usage_percent_final
            peak_cpu_percent = None
            average_cpu_percent = None
            final_cpu_percent = None
        
        # RAM MB stats (process-only)
        if self.step_data:
            ram_mb_values = [s.get('ram_mb') for s in self.step_data if s.get('ram_mb') is not None]
            if ram_mb_values:
                peak_ram_mb = max(ram_mb_values)
                average_ram_mb = sum(ram_mb_values) / len(ram_mb_values)
                final_ram_mb = ram_mb_values[-1]
            else:
                peak_ram_mb = None
                average_ram_mb = None
                final_ram_mb = None
        else:
            peak_ram_mb = None
            average_ram_mb = None
            final_ram_mb = None
        
        self.final_hardware_info = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_time_seconds": round(elapsed_time, 2),
            "total_training_steps": self.step_count,
            "final_ram_usage_percent": ram_usage_percent_final,
            "peak_ram_usage_percent": peak_ram_percent,
            "average_ram_usage_percent": round(average_ram_percent, 2),
            "final_cpu_usage_percent": round(final_cpu_percent, 2) if final_cpu_percent is not None else None,
            "peak_cpu_usage_percent": round(peak_cpu_percent, 2) if peak_cpu_percent is not None else None,
            "average_cpu_usage_percent": round(average_cpu_percent, 2) if average_cpu_percent is not None else None,
            "final_ram_mb": round(final_ram_mb, 2) if final_ram_mb is not None else None,
            "peak_ram_mb": round(peak_ram_mb, 2) if peak_ram_mb is not None else None,
            "average_ram_mb": round(average_ram_mb, 2) if average_ram_mb is not None else None,
            # training metrics from last step if available
            "final_mean_reward": self.step_data[-1].get("mean_reward") if self.step_data else None,
            "final_std_reward": self.step_data[-1].get("std_of_reward") if self.step_data else None,
        }
        
        return self.final_hardware_info
    
    def record_training_metrics(self, metrics: Dict[str, Any]) -> None:
        self.training_metrics = metrics
    
    def get_hardware_data(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "initial_hardware_info": self.initial_hardware_info,
            "final_hardware_info": self.final_hardware_info,
            "step_data": self.step_data,
            "training_metrics": self.training_metrics
        }
    
    def save_to_csv(self, output_dir: Optional[Path] = None) -> Path:
        """Finalize and close the csv file."""
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)

        # Write all data if streaming didn't happen
        if self._csv_path is None:
            run_data_dir = output_dir / self.run_name / "run_logs"
            run_data_dir.mkdir(parents=True, exist_ok=True)
            self._csv_path = run_data_dir / "run_log.csv"
            with self._csv_path.open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step_number', 'time_elapsed', 'mean_reward', 'std_of_reward',
                    'cpu_percent', 'cpu_source', 'ram_percent', 'ram_source', 'ram_mb'
                ])
                for step in self.step_data:
                    writer.writerow([
                        step['step_number'],
                        step['time_elapsed'],
                        step['mean_reward'] if step['mean_reward'] is not None else '',
                        step['std_of_reward'] if step['std_of_reward'] is not None else '',
                        step['cpu_percent'] if step['cpu_percent'] is not None else '',
                        step.get('cpu_source', 'system'),
                        step['ram_percent'],
                        step.get('ram_source', 'system'),
                        step.get('ram_mb') if step.get('ram_mb') is not None else '',
                    ])

        self._close_csv_writer()
        ram_csv_path = self._csv_path

        print(f"\n? RAM usage data saved: {ram_csv_path}")
        
        metadata_path = output_dir / "TESTING_all_runs_metadata.csv"
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
            'final_ram_mb': self.final_hardware_info.get('final_ram_mb', 'N/A'),
            'peak_ram_mb': self.final_hardware_info.get('peak_ram_mb', 'N/A'),
            'average_ram_mb': self.final_hardware_info.get('average_ram_mb', 'N/A'),
            'final_cpu_percent': self.final_hardware_info.get('final_cpu_usage_percent', 'N/A'),
            'peak_cpu_percent': self.final_hardware_info.get('peak_cpu_usage_percent', 'N/A'),
            'average_cpu_percent': self.final_hardware_info.get('average_cpu_usage_percent', 'N/A'),
            'data_folder': str(ram_csv_path)
        }
        
        if self.testing:
            file_exists = metadata_path.exists()
            with metadata_path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=hardware_metadata.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(hardware_metadata)
            print(f"? Metadata updated: {metadata_path}")
            print("? Testing mode enabled - Hardware metadata saved to CSV")
        else:
            print("? Testing mode disabled - Hardware metadata kept in dictionary only")
            print(f"? RAM data file: {ram_csv_path}")
        
        return ram_csv_path
    
    def get_current_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage snapshot."""
        memory_info = psutil.virtual_memory()
        
        result = {
            "system_used_ram_mb": round(memory_info.used / (1024 ** 2), 2),
            "system_available_ram_mb": round(memory_info.available / (1024 ** 2), 2),
            "system_ram_usage_percent": memory_info.percent,
            "source": "system"
        }
        
        if self._process:
            try:
                proc_memory = self._process.memory_info()
                proc_ram_mb = round(proc_memory.rss / (1024 ** 2), 2)
                proc_ram_percent = round((proc_memory.rss / memory_info.total) * 100, 2)
                result.update({
                    "process_ram_mb": proc_ram_mb,
                    "process_ram_percent": proc_ram_percent,
                    "source": "process"
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return result


class HardwareMonitorContext:
    """Context manager wrapper for HardwareMonitor."""
    
    def __init__(self, run_name: str = "training_run", steps_per_log: int = 5000, testing: bool = False, results_dir: Optional[Path] = None):
        self.monitor = HardwareMonitor(run_name, steps_per_log, testing, results_dir)
    
    def __enter__(self) -> HardwareMonitor:
        self.monitor.record_initial_state()
        self.monitor.start_continuous_monitoring()
        return self.monitor
    
    def set_process(self, process: Optional[psutil.Process]) -> None:
        """Set the target process for monitoring."""
        self.monitor._process = process
        if process:
            try:
                process.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.monitor.record_final_state()
        self.monitor.save_to_csv()
        return False
