import json
import numpy as np
import argparse
from pathlib import Path
import sys
from sklearn.preprocessing import RobustScaler

# Add parent directory to sys.path to import config
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DataConfig

def normalize_time_intervals(delta_times, apply_norm=True):
    """
    Standardize time intervals using RobustScaler.
    Best for Tor traffic as it is robust to outliers (long idle times).
    """
    if not delta_times or len(delta_times) == 0 or not apply_norm:
        return delta_times
    
    # Reshape for sklearn scaler
    delta_times = np.array(delta_times, dtype=np.float64).reshape(-1, 1)
    delta_times = np.clip(delta_times, 0, None) 
    
    # Initialize and fit RobustScaler
    scaler = RobustScaler()
    normalized_intervals = scaler.fit_transform(delta_times).flatten()
    
    # Clip values to range [-5, 5] to ensure stable training on A100
    normalized = np.clip(normalized_intervals, -5, 5)
    
    return normalized.tolist()

def analyze_delta_statistics(delta_times):
    """
    Calculate basic statistics for time intervals.
    """
    if not delta_times or len(delta_times) == 0:
        return {}
    
    actual_deltas = np.array(delta_times)
    
    return {
        "count": len(actual_deltas),
        "min": float(actual_deltas.min()),
        "max": float(actual_deltas.max()),
        "mean": float(actual_deltas.mean()),
        "std": float(actual_deltas.std()),
        "median": float(np.median(actual_deltas)),
    }

def filter_and_process(label: str, min_packets: int, apply_norm: bool = True, 
                       analyze_only: bool = False):
    """
    Filters sessions by packet count and optionally applies Robust Scaling.
    """
    cfg = DataConfig()
    label_dir = cfg.interim_dir / label
    
    if not label_dir.exists():
        print(f"[Error] Directory not found: {label_dir}")
        return

    jsonl_files = list(label_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[Warning] No .jsonl files found in {label}")
        return

    print("\n" + "="*70)
    print(f"PROCESS CATEGORY: {label}")
    print(f"PACKET THRESHOLD: {min_packets}")
    print(f"ROBUST NORMALIZATION: {'ENABLED' if apply_norm else 'DISABLED'}")
    print(f"EXECUTION MODE: {'ANALYZE_ONLY' if analyze_only else 'UPDATE_IN_PLACE'}")
    print("="*70)

    summary = {
        "raw_sessions": 0,
        "filtered_sessions": 0,
        "raw_packets": 0,
        "filtered_packets": 0,
    }

    for file_path in jsonl_files:
        kept_sessions = []
        file_raw_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                
                session = json.loads(line)
                file_raw_count += 1
                summary["raw_sessions"] += 1
                
                pkts = session.get("packet_count", 0)
                summary["raw_packets"] += pkts
                
                # Apply Packet Count Filter
                if pkts >= min_packets:
                    # Apply Robust Normalization if requested
                    if apply_norm and "delta_time" in session:
                        session["delta_time"] = normalize_time_intervals(session["delta_time"], True)
                    
                    kept_sessions.append(session)
                    summary["filtered_packets"] += pkts
        
        summary["filtered_sessions"] += len(kept_sessions)
        
        if not analyze_only:
            with open(file_path, 'w', encoding='utf-8') as f:
                for s in kept_sessions:
                    f.write(json.dumps(s) + '\n')
            status = "SUCCESS" if len(kept_sessions) > 0 else "EMPTY"
        else:
            status = "ANALYZED"
        
        print(f"File: {file_path.name:<30} | {file_raw_count:>4} -> {len(kept_sessions):>4} sessions [{status}]")

    print("\n" + "-" * 70)
    print(f"GLOBAL SUMMARY for [{label}]:")
    print(f" - Raw Sessions:      {summary['raw_sessions']}")
    print(f" - Filtered Sessions: {summary['filtered_sessions']}")
    print(f" - Dropped Sessions:  {summary['raw_sessions'] - summary['filtered_sessions']}")
    
    if summary['raw_sessions'] > 0:
        rate = (summary['filtered_sessions'] / summary['raw_sessions']) * 100
        print(f" - Retention Rate:    {rate:.2f}%")
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Tor Traffic Sequence Balancing & Normalization")
    parser.add_argument("--categories", type=str, required=True, 
                        help="Target category name (e.g., obfs4, meek)")
    parser.add_argument("--min_pkts", type=int, default=10, 
                        help="Minimum packet count per session (Default: 10)")
    parser.add_argument("--normalize", action="store_true",
                        help="Enable Robust Scaling for time intervals")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Run analysis without modifying local files")
    
    args = parser.parse_args()
    
    filter_and_process(
        label=args.categories,
        min_packets=args.min_pkts,
        apply_norm=args.normalize,
        analyze_only=args.analyze_only
    )

if __name__ == "__main__":
    
    main()