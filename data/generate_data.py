import json
import numpy as np
from pathlib import Path
import sys
import random

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DataConfig

def main():
    cfg = DataConfig()
    MAX_SEQ_LEN = cfg.max_sequence_length

    # Initialize directory structure in the processed folder
    for label in cfg.traffic_labels:
        (cfg.processed_dir / label / "sequences").mkdir(parents=True, exist_ok=True)
        (cfg.processed_dir / label / "images").mkdir(parents=True, exist_ok=True)

    print(f">>> Stage 1: Extracting sequence features from interim and saving to processed.")
    
    matched_data = {label: {} for label in cfg.traffic_labels}

    for label in cfg.traffic_labels:
        interim_label_dir = cfg.interim_dir / label
        if not interim_label_dir.exists():
            continue

        print(f"Processing sequence category: {label}")

        for jsonl_path in interim_label_dir.glob("*.jsonl"):
            pcap_stem = jsonl_path.stem  
            
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    session_dict = json.loads(line)
                    session_name = session_dict['pcap_name']
                    session_stem = Path(session_name).stem 
                    
                    # Sequence Processing: Truncate to MAX_SEQ_LEN
                    direction = session_dict['direction'][:MAX_SEQ_LEN]
                    delta_time = session_dict['delta_time'][:MAX_SEQ_LEN]
                    payload_size = session_dict['payload_size'][:MAX_SEQ_LEN]

                    # Stack features into a (3, Length) matrix
                    seq_matrix = np.array([direction, delta_time, payload_size], dtype=np.float32)
                    
                    # Right-side zero padding if length is less than MAX_SEQ_LEN
                    current_len = seq_matrix.shape[1]
                    if current_len < MAX_SEQ_LEN:
                        pad_width = MAX_SEQ_LEN - current_len
                        seq_matrix = np.pad(seq_matrix, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
                    
                    # Save sequence feature as .npy
                    seq_sub_dir = cfg.processed_dir / label / "sequences" / pcap_stem
                    seq_sub_dir.mkdir(parents=True, exist_ok=True)
                    seq_file_path = seq_sub_dir / f"{session_stem}.npy"
                    np.save(seq_file_path, seq_matrix)

                    # Modality Matching Check: Ensure the corresponding image feature exists
                    img_file_path = cfg.processed_dir / label / "images" / pcap_stem / f"{session_stem}.npy"
                    
                    if img_file_path.exists():
                        rel_seq = str(seq_file_path.relative_to(cfg.processed_dir))
                        rel_img = str(img_file_path.relative_to(cfg.processed_dir))
                        
                        unique_id = f"{pcap_stem}_{session_stem}"
                        matched_data[label][unique_id] = {
                            "sequence_path": rel_seq,
                            "image_path": rel_img,
                            "label": label
                        }

    print("\n>>> Stage 2: Performing dataset split based on Bi-flow")
    
    train_manifest, val_manifest, test_manifest = [], [], []
    split_ratio = (0.7, 0.15, 0.15) 
    random.seed(42)

    stats = {label: {"total": 0, "train": 0, "val": 0, "test": 0} for label in cfg.traffic_labels}

    for label, sessions in matched_data.items():
        if not sessions:
            continue
            
        session_keys = list(sessions.keys())
        random.shuffle(session_keys)
        
        n = len(session_keys)
        n_train = int(n * split_ratio[0])
        n_val = int(n * split_ratio[1])
        
        train_keys = session_keys[:n_train]
        val_keys = session_keys[n_train : n_train + n_val]
        test_keys = session_keys[n_train + n_val:]
        
        for k in train_keys: 
            train_manifest.append(sessions[k])
            stats[label]["train"] += 1
        for k in val_keys: 
            val_manifest.append(sessions[k])
            stats[label]["val"] += 1
        for k in test_keys: 
            test_manifest.append(sessions[k])
            stats[label]["test"] += 1
        
        stats[label]["total"] = n

    # Save manifest files
    manifest_map = {
        "train_manifest.json": train_manifest,
        "val_manifest.json": val_manifest,
        "test_manifest.json": test_manifest,
        "manifest.json": train_manifest + val_manifest + test_manifest
    }

    for name, data in manifest_map.items():
        with open(cfg.processed_dir / name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    
    # Print summary statistics
    print(f"\n{'Category':<15} {'Total':<8} {'Train':<8} {'Validation':<8} {'Test':<8}")
    print("-" * 55)
    for label, s in stats.items():
        print(f"{label:<15} {s['total']:<10} {s['train']:<10} {s['val']:<10} {s['test']:<10}")
    
    print(f"\nProcessing complete! Manifest files generated at: {cfg.processed_dir}")

if __name__ == "__main__":
    main()
