#!/usr/bin/env python3
"""
simple training script with hardware monitoring
this script runs ml-agents training and monitors hardware during training
it saves all data to csv files so we can analyze later
"""

import argparse
import os
import sys
import subprocess
import time
import re
from pathlib import Path

# add project root to python path so we can import from custom
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# import hardware monitor from our custom scripts
from custom.scripts.data_collection.hardware_monitor import HardwareMonitor


def parse_training_output(output_line: str) -> dict:
    """
    parse training metrics from mlagents-learn output
    
    looks for patterns like:
    "[INFO] 3DBall. Step: 10000. Time Elapsed: 15.400 s. Mean Reward: 1.250. Std of Reward: 0.753. Training."
    
    returns dict with extracted metrics or None if pattern doesnt match
    """
    # pattern to match training step output
    # this matches format we saw in terminal output
    pattern = r"Step:\s+(\d+)\.\s+Time\s+Elapsed:\s+([\d.]+)\s+s\.\s+Mean\s+Reward:\s+([-\d.]+)\.\s+Std\s+of\s+Reward:\s+([\d.]+)\."
    
    match = re.search(pattern, output_line)
    if match:
        return {
            'steps': int(match.group(1)),
            'time_elapsed': float(match.group(2)),
            'mean_reward': float(match.group(3)),
            'std_of_reward': float(match.group(4))
        }
    return None


def main():
    parser = argparse.ArgumentParser(description='Train 3DBall with hardware monitoring')
    parser.add_argument(
        '--config',
        type=str,
        default='custom/data/configs_for_training/ppo_c1e1bec196.yaml',
        help='Path to training config (default: custom/data/configs_for_training/ppo_c1e1bec196.yaml)'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default='3DBall_training',
        help='Name for this training run'
    )
    parser.add_argument(
        '--log-steps',
        type=int,
        default=5000,
        help='Log hardware every N steps (default: 5000) - NOTE: This is now ignored, we log every time we see a step update'
    )
    
    parser.add_argument(
        '--testing',
        action='store_true',
        help='Enable testing mode: save Hardware_metadata to CSV file (default: False)'
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        default='custom/data/results',
        help='Directory to save results (default: custom/data/results)'
    )
    
    args = parser.parse_args()
    
    # check if config exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {args.config}")
        sys.exit(1)
    
    # create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("3DBALL TRAINING WITH HARDWARE MONITORING")
    print("="*70)
    print(f"Run: {args.run_name}")
    print(f"Config: {args.config}")
    print(f"Results dir: {args.results_dir}")
    print(f"Testing mode: {args.testing}")
    print("="*70 + "\n")
    
    # initialize hardware monitor
    # note: steps_per_log is kept for compatibility but we actually log every step update now
    monitor = HardwareMonitor(
        run_name=args.run_name,
        steps_per_log=args.log_steps,  # not really used anymore but kept for compatibility
        testing=args.testing,
        results_dir=results_dir
    )
    
    # record pre-training hardware state
    print("Recording initial hardware state...\n")
    monitor.record_initial_state()
    
    # start monitoring (we'll set process later when we start training)
    monitor.start_continuous_monitoring()
    
    latest_metrics = {}
    training_process = None
    
    try:
        # build training command
        # we dont use --env flag so it connects to unity editor
        cmd = [
            'mlagents-learn',
            str(config_path),
            f'--run-id={args.run_name}',
            f'--results-dir={results_dir}',
            '--force'
        ]
        
        print("\nStarting training...\n")
        print(f"Command: {' '.join(cmd)}\n")
        print("~i Make sure Unity Editor is running with the 3DBall scene open!")
        print("~i When you see 'Start training by pressing the Play button', press Play in Unity\n")
        
        # start training with output capturing
        training_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,  # use text mode instead of universal_newlines
            bufsize=1
        )
        
        # set process for cpu monitoring
        # this lets us monitor cpu usage specifically for mlagents-learn process
        try:
            import psutil
            ps_proc = psutil.Process(training_process.pid)
            # update monitor to use this specific process
            monitor._process = ps_proc
            # prime cpu measurement (first call returns 0.0, so we call it once)
            ps_proc.cpu_percent(interval=None)
        except Exception as e:
            print(f"~i Could not set process monitoring, using system-wide CPU: {e}\n")
        
        print("✓ Training running (Press Ctrl+C to stop)\n")
        print("="*70)
        print("TRAINING OUTPUT (with metrics capture)")
        print("="*70 + "\n")
        
        # read output line by line to capture metrics
        if training_process.stdout:
            for line in training_process.stdout:
                print(line, end='')  # print to console
                
                # try to parse training metrics from output line
                metrics = parse_training_output(line)
                if metrics:
                    latest_metrics = metrics
                    
                    # log the step with all metrics
                    # this is the new way - we pass all data to log_step()
                    monitor.log_step(
                        step_number=metrics['steps'],
                        time_elapsed=metrics['time_elapsed'],
                        mean_reward=metrics['mean_reward'],
                        std_of_reward=metrics['std_of_reward']
                    )
                    
                    # print parsed metrics so we can see what was captured
                    print(f"\n   [CAPTURED] Step={metrics['steps']}, " +
                          f"Time={metrics['time_elapsed']:.1f}s, " +
                          f"Mean_Reward={metrics['mean_reward']:.2f}, " +
                          f"Std_Reward={metrics['std_of_reward']:.2f}\n")
        
        # wait for process to finish
        training_process.wait()
        
        print("\n✓ Training completed!")
        
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        if training_process:
            training_process.terminate()
            try:
                training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                training_process.kill()
    
    finally:
        # record final hardware state
        print("\n Recording final hardware state...\n")
        monitor.record_final_state()
        
        # record latest training metrics if captured
        if latest_metrics:
            monitor.record_training_metrics(latest_metrics)
        
        # get and display results
        data = monitor.get_hardware_data()
        
        print("\n" + "="*70)
        print("TRAINING & HARDWARE MONITORING RESULTS")
        print("="*70)
        
        print("\n TRAINING METRICS (Latest captured):")
        if latest_metrics:
            for key, value in latest_metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print("     No training metrics captured")
        
        print("\n HARDWARE - Initial State:")
        for key, value in data['initial_hardware_info'].items():
            print(f"   {key}: {value}")
        
        print("\n HARDWARE - Final State:")
        for key, value in data['final_hardware_info'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n HARDWARE - Step Data Collected:")
        print(f"   Total measurements: {len(data['step_data'])}")
        if data['step_data']:
            # show a few examples
            print(f"   First step: {data['step_data'][0]}")
            if len(data['step_data']) > 1:
                print(f"   Last step: {data['step_data'][-1]}")
        
        # save to csv files
        ram_file = monitor.save_to_csv()
        
        print("\n" + "="*70)
        print("✓ RESULTS SAVED")
        print("="*70)
        print(f"Step data CSV: {ram_file}")
        print(f"  └─ Contains: step_number, time_elapsed, mean_reward, std_of_reward, cpu_percent, ram_percent")
        if args.testing:
            print(f"\nCentral metadata: {results_dir}/TESTING_all_runs_metadata.csv")
            print(f"  └─ All run information (OS, CPU, performance, etc.)")
        print("="*70 + "\n")


if __name__ == '__main__':
    main()

