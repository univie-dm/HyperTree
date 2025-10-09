#!/usr/bin/env python3
"""
SLURM Parameter Sweep Tool

Submits SLURM jobs with parameter combinations.
Uses Hydra's config.yaml as base, overriding only specified parameters.

Usage:
    python sweep.py --sweep-config sweep.yaml [--sbatch-config sbatch_config.yaml] [--dry-run]
    
    # Quick run with default configs
    python sweep.py
    
    # With custom configs
    python sweep.py --sweep-config my_sweep.yaml
    
    # Test without submitting
    python sweep.py --dry-run
"""

import argparse
import itertools
import subprocess
import yaml
import os
from pathlib import Path
from datetime import datetime
import hashlib


def load_config(config_path):
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_combinations(sweep_config):
    """
    Generate all parameter combinations from sweep config.
    
    Returns:
        List of dicts, each containing one parameter combination
    """
    parameters = sweep_config.get('parameters', {})
    
    # Parameters that should NOT be swept even if they are lists
    # These are parameters that expect list values
    NO_SWEEP_PARAMS = {'wandb.tags', 'wandb.group', 'wandb.notes'}
    
    # Separate sweep parameters (lists) from fixed parameters (single values)
    sweep_params = {}
    fixed_params = {}
    
    for param_name, param_value in parameters.items():
        # Check if it's a list AND not in the no-sweep list
        if isinstance(param_value, list) and param_name not in NO_SWEEP_PARAMS:
            sweep_params[param_name] = param_value
        else:
            fixed_params[param_name] = param_value
    
    # Generate all combinations
    if sweep_params:
        keys = list(sweep_params.keys())
        values = [sweep_params[k] for k in keys]
        combinations = []
        
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            param_dict.update(fixed_params)
            combinations.append(param_dict)
    else:
        # Only fixed parameters
        combinations = [fixed_params] if fixed_params else []
    
    return combinations


def create_sbatch_script(params, sweep_config, sbatch_config, output_path):
    """
    Create a SLURM batch script.
    
    Args:
        params: Dict of Hydra parameters for this job
        sweep_config: Sweep configuration
        sbatch_config: SBATCH configuration
        output_path: Where to save the script
    """
    lines = []
    
    # SBATCH header
    lines.append("#!/bin/bash")
    
    # Create job name with parameter suffix
    param_suffix = '_'.join([f"{k.split('.')[-1]}{v}" for k, v in sorted(params.items()) 
                             if isinstance(v, (int, float)) or len(str(v)) < 10])
    if len(param_suffix) > 40:
        param_suffix = hashlib.md5(param_suffix.encode()).hexdigest()[:8]
    
    job_name = f"{sweep_config.get('sweep_name', 'sweep')}_{param_suffix}"
    lines.append(f"#SBATCH --job-name={job_name}")
    
    # SBATCH directives
    lines.append(f"#SBATCH --time={sbatch_config['time']}")
    lines.append(f"#SBATCH -c {sbatch_config['cpus']}")
    lines.append(f"#SBATCH --mem={sbatch_config['mem']}")
    lines.append(f"#SBATCH --gres={sbatch_config['gres']}")
    lines.append(f"#SBATCH --constraint={sbatch_config.get('constraint', '')}")
    
    # Output log
    output_dir = sbatch_config.get('output_dir', 'logs')
    lines.append(f"#SBATCH --output={output_dir}/{job_name}.log")
    lines.append("")
    
    # Setup commands
    for cmd in sbatch_config.get('setup_commands', []):
        lines.append(cmd)
    lines.append("")
    
    # Python training command
    param_strs = []
    for k, v in params.items():
        if isinstance(v, list):
            # Format list as Hydra expects: [item1,item2,item3]
            formatted_list = '[' + ','.join(f"'{item}'" if isinstance(item, str) else str(item) for item in v) + ']'
            param_strs.append(f"{k}={formatted_list}")
        else:
            param_strs.append(f"{k}={v}")
    
    lines.append(f"# Training with parameters: {', '.join(param_strs[:3])}...")
    lines.append(f"python train.py {' '.join(param_strs)}")
    lines.append("")
    
    # Cleanup commands
    for cmd in sbatch_config.get('cleanup_commands', []):
        lines.append(cmd)
    
    # Write script
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    os.chmod(output_path, 0o755)
    return job_name


def create_array_script(combinations, sweep_config, sbatch_config, output_path):
    """
    Create a SLURM array job script.
    
    Args:
        combinations: List of parameter dicts
        sweep_config: Sweep configuration
        sbatch_config: SBATCH configuration
        output_path: Where to save the script
    """
    lines = []
    
    # SBATCH header
    lines.append("#!/bin/bash")
    
    job_name = sweep_config.get('sweep_name', 'sweep')
    lines.append(f"#SBATCH --job-name={job_name}")
    lines.append(f"#SBATCH --time={sbatch_config['time']}")
    lines.append(f"#SBATCH -c {sbatch_config['cpus']}")
    lines.append(f"#SBATCH --mem={sbatch_config['mem']}")
    lines.append(f"#SBATCH --gres={sbatch_config['gres']}")
    lines.append(f"#SBATCH --array=0-{len(combinations)-1}")
    
    output_dir = sbatch_config.get('output_dir', 'logs')
    lines.append(f"#SBATCH --output={output_dir}/{job_name}_%A_%a.log")
    lines.append("")
    
    # Setup commands
    for cmd in sbatch_config.get('setup_commands', []):
        lines.append(cmd)
    lines.append("")
    
    # Parameter selection based on array task ID
    lines.append("# Select parameters based on SLURM_ARRAY_TASK_ID")
    lines.append("TASK_ID=${SLURM_ARRAY_TASK_ID:-0}")
    lines.append("")
    lines.append("# Parameter combinations")
    lines.append("case $TASK_ID in")
    
    for i, params in enumerate(combinations):
        param_strs = []
        for k, v in params.items():
            if isinstance(v, list):
                # Format list as Hydra expects: [item1,item2,item3]
                formatted_list = '[' + ','.join(f"'{item}'" if isinstance(item, str) else str(item) for item in v) + ']'
                param_strs.append(f"{k}={formatted_list}")
            else:
                param_strs.append(f"{k}={v}")
        
        lines.append(f"  {i})")
        lines.append(f"    PARAMS='{' '.join(param_strs)}'")
        lines.append(f"    ;;")
    
    lines.append("  *)")
    lines.append("    echo \"Error: Invalid TASK_ID $TASK_ID\"")
    lines.append("    exit 1")
    lines.append("    ;;")
    lines.append("esac")
    lines.append("")
    
    lines.append("echo \"Running task $TASK_ID with: $PARAMS\"")
    lines.append("")
    
    # Python command
    lines.append("python train.py $PARAMS")
    lines.append("")
    
    # Cleanup commands
    for cmd in sbatch_config.get('cleanup_commands', []):
        lines.append(cmd)
    
    # Write script
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    os.chmod(output_path, 0o755)
    
    return job_name


def submit_job(script_path, dry_run=False):
    """Submit a job using sbatch."""
    if dry_run:
        print(f"  [DRY RUN] Would submit: sbatch {script_path}")
        return None
    else:
        result = subprocess.run(['sbatch', str(script_path)], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"  ✓ Submitted job {job_id}")
            return job_id
        else:
            print(f"  ✗ Error: {result.stderr.strip()}")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='Submit SLURM jobs with Hydra parameter sweeps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sweep.py                                    # Use default configs
  python sweep.py --sweep-config my_sweep.yaml       # Custom sweep
  python sweep.py --dry-run                          # Test without submitting
  python sweep.py --method array                     # Force array method
        """
    )
    parser.add_argument('--sweep-config', default='sweep.yaml',
                       help='Sweep configuration file (default: sweep.yaml)')
    parser.add_argument('--sbatch-config', default='sbatch_config.yaml',
                       help='SBATCH configuration file (default: sbatch_config.yaml)')
    parser.add_argument('--method', choices=['individual', 'array'],
                       help='Override method from sweep config')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate scripts but do not submit jobs')
    parser.add_argument('--output-dir', default='generated_jobs',
                       help='Directory for generated scripts (default: generated_jobs)')
    
    args = parser.parse_args()
    
    # Load configurations
    print(f"Loading sweep config: {args.sweep_config}")
    sweep_config = load_config(args.sweep_config)
    
    print(f"Loading SBATCH config: {args.sbatch_config}")
    sbatch_config = load_config(args.sbatch_config)
    
    # Determine method
    method = args.method or sweep_config.get('method', 'individual')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Ensure log directory exists
    log_dir = Path(sbatch_config.get('output_dir', 'logs'))
    log_dir.mkdir(exist_ok=True)
    
    # Generate combinations
    combinations = generate_combinations(sweep_config)
    
    print(f"\nSweep: {sweep_config.get('sweep_name', 'unnamed')}")
    print(f"Method: {method}")
    print(f"Combinations: {len(combinations)}")
    
    if len(combinations) == 0:
        print("Warning: No parameter combinations generated!")
        return
    
    # Show first few combinations
    print(f"\nFirst {min(3, len(combinations))} combinations:")
    for i, combo in enumerate(combinations[:3], 1):
        print(f"  {i}. {combo}")
    if len(combinations) > 3:
        print(f"  ... and {len(combinations) - 3} more")
    
    print()
    
    if method == 'array':
        # Create array job script
        script_path = output_dir / f"{sweep_config.get('sweep_name', 'sweep')}_array.sh"
        job_name = create_array_script(combinations, sweep_config, sbatch_config, script_path)
        
        print(f"Generated array job script: {script_path}")
        
        if args.dry_run:
            print(f"[DRY RUN] Would submit: sbatch {script_path}")
        else:
            print(f"\nTo submit the array job, run:")
            print(f"  sbatch {script_path}")
            print(f"\nOr to submit now:")
            response = input("Submit now? [y/N]: ")
            if response.lower() == 'y':
                submit_job(script_path, dry_run=False)
    
    else:  # individual
        # Create and submit individual jobs
        submitted_jobs = []
        
        print(f"Submitting {len(combinations)} individual jobs...\n")
        
        for i, params in enumerate(combinations, 1):
            script_path = output_dir / f"job_{i:03d}.sh"
            
            try:
                job_name = create_sbatch_script(params, sweep_config, sbatch_config, script_path)
                print(f"[{i}/{len(combinations)}] {job_name}")
                
                job_id = submit_job(script_path, args.dry_run)
                
                if job_id:
                    submitted_jobs.append({
                        'job_id': job_id,
                        'job_name': job_name,
                        'script': str(script_path),
                        'params': params
                    })
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        # Save job tracking info
        if not args.dry_run and submitted_jobs:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tracking_file = output_dir / f"submitted_jobs_{timestamp}.yaml"
            with open(tracking_file, 'w') as f:
                yaml.dump(submitted_jobs, f, default_flow_style=False)
            print(f"\nJob tracking saved to: {tracking_file}")
        
        print(f"\n{'Would submit' if args.dry_run else 'Submitted'} {len(submitted_jobs)} jobs")


if __name__ == '__main__':
    main()
