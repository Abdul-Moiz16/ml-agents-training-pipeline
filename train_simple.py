#!/usr/bin/env python3
import argparse
import os
import sys
import subprocess
import time
import re
from pathlib import Path

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from hardware_monitor import HardwareMonitor


def parse_training_output(output_line: str) -> dict:
    """
    Parse training metrics from mlagents-learn output.
    
    Looks for patterns like:
    "Step: 12000. Time Elapsed: 52.095 s. Mean Reward: 1.192. Std of Reward: 0.708"
    
    Returns dict with extracted metrics or None if pattern doesn't match.
    """
    # Pattern to match training step output
    # Example: "[INFO] 3DBall. Step: 12000. Time Elapsed: 52.095 s. Mean Reward: 1.192. Std of Reward: 0.708. Training."
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
        default='config/ppo/3DBall.yaml',
        help='Path to training config (default: config/ppo/3DBall.yaml)'
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
        help='Log hardware every N steps (default: 5000)'
    )
    
    parser.add_argument(
        '--testing',
        action='store_true',
        help='Enable testing mode: save Hardware_metadata to CSV file (default: False)'
    )
    
    args = parser.parse_args()
    
    # Check if config exists
    if not os.path.exists(args.config):
        print(f"Config not found: {args.config}")
        sys.exit(1)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    print("\n" + "="*70)
    print("3DBALL TRAINING WITH HARDWARE MONITORING")
    print("="*70)
    print(f"Run: {args.run_name}")
    print(f"Config: {args.config}")
    print(f"Log every: {args.log_steps} steps")
    print("="*70 + "\n")
    
    # Initialize hardware monitor
    monitor = HardwareMonitor(
        run_name=args.run_name,
        steps_per_log=args.log_steps,
        testing=args.testing
    )
    
    # Record pre-training hardware state
    print("Recording initial hardware state...\n")
    monitor.record_initial_state()
    
    # Start monitoring
    monitor.start_continuous_monitoring()
    
    latest_metrics = {}
    
    try:
        # Build training command using full path to mlagents-learn
        import shutil
        mlagents_path = shutil.which('mlagents-learn') or '/Users/filippomorini/GitHub/Group8-AI-ML/venv_310/bin/mlagents-learn'
        
        cmd = [
            mlagents_path,
            args.config,
            '--run-id', args.run_name,
            '--results-dir', 'results'
        ]
        
        print("\nStarting training...\n")
        print(f"Command: {' '.join(cmd)}\n")
        
        # Start training with output capturing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        print("✓ Training running (Press Ctrl+C to stop)\n")
        print("="*70)
        print("TRAINING OUTPUT (with metrics capture)")
        print("="*70 + "\n")
        
        # Read output line by line to capture metrics
        for line in process.stdout:
            print(line, end='')  # Print to console
            
            # Try to parse training metrics
            metrics = parse_training_output(line)
            if metrics:
                latest_metrics = metrics
                # Log RAM usage when we capture training step
                monitor.step_count = metrics['steps']
                monitor.log_step()
                # Also print parsed metrics clearly
                print(f"\n   Captured: Steps={metrics['steps']}, " +
                      f"Time={metrics['time_elapsed']:.1f}s, " +
                      f"Mean_Reward={metrics['mean_reward']:.2f}, " +
                      f"Std_Reward={metrics['std_of_reward']:.2f}\n")
        
        process.wait()
        
        print("\n✓ Training completed!")
        
    except KeyboardInterrupt:
        print("\n\n  Training interrupted by user")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    
    finally:
        # Record final hardware state
        print("\n Recording final hardware state...\n")
        monitor.record_final_state()
        
        # Record latest training metrics if captured
        if latest_metrics:
            monitor.record_training_metrics(latest_metrics)
        
        # Get and display results
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
        
        print("\n HARDWARE - RAM Usage During Training:")
        print(f"   Measurements: {len(data['ram_usage_per_step'])}")
        if data['ram_usage_per_step']:
            print(f"   Values (%): {[round(x, 2) for x in data['ram_usage_per_step']]}")
        
        # Save to organized folder structure
        ram_file = monitor.save_to_csv('results')
        
        print("\n" + "="*70)
        print("✓ RESULTS SAVED")
        print("="*70)
        print(f"RAM usage file: {ram_file}")
        print(f"  └─ Step-by-step RAM measurements")
        print(f"\nCentral metadata: results/all_runs_metadata.csv")
        print(f"  └─ All run information (OS, CPU, performance, etc.)")
        print("="*70 + "\n")


if __name__ == '__main__':
    main()
