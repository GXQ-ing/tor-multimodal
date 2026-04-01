import sys
import logging
import argparse
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Sequence
from collections import Counter
import numpy as np
from scapy.all import Packet, rdpcap, IP, conf
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
conf.max_list_count = 99999999

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DataConfig

@dataclass(frozen=True)
class PacketRecord:
    timestamp: float
    direction: int
    payload_size: int

def direction_sequence(segment: Sequence[PacketRecord], max_length: int) -> np.ndarray:
    """
    Generates a direction sequence from packet segments.
    Pads with zeros if length is less than max_length.
    """
    seq = np.array([p.direction for p in segment], dtype=np.int16)
    if seq.size >= max_length:
        return seq[:max_length]
    padded = np.zeros(max_length, dtype=np.int16)
    padded[: seq.size] = seq
    return padded

def direction_resolver(packet: Packet, local_ip: str) -> int:
    """
    Determines packet direction relative to the local IP.
    Returns 1 for Outgoing, -1 for Incoming, and 0 for others.
    """
    if packet.haslayer(IP):
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        if src_ip == local_ip:
            return 1
        elif dst_ip == local_ip:
            return -1
    return 0

def detect_local_ip(pcap_folder: Path) -> str:
    """
    Automatically detects the local IP by analyzing filenames in the folder.
    Expects flow-based filenames like 'srcIP_dstIP_...pcap'.
    """
    ip_counter = Counter()
    pcap_files = list(pcap_folder.glob("*.pcap*"))
    
    if not pcap_files:
        return None

    for pcap_file in pcap_files:
        parts = pcap_file.stem.split('_')
        if len(parts) >= 2:
            ip_counter.update([parts[0], parts[1]])

    if not ip_counter:
        return None

    most_common_ip = ip_counter.most_common(1)[0][0]
    return most_common_ip

def get_single_session_features(pcap_path, local_ip):
    """Extracts feature sequences (direction, time delta, size) from a single PCAP session."""
    try:
        packets = rdpcap(str(pcap_path))
        if len(packets) == 0:
            return None
        
        pkts_data = []
        for pkt in packets:
            if IP not in pkt: continue

            # Extract Timestamp
            ts = float(pkt.time)
            # Resolve Direction 
            direction = direction_resolver(pkt, local_ip)
            # Extract Payload Size 
            size = len(pkt[IP].payload) if pkt[IP].payload else 0
            
            pkts_data.append({"ts": ts, "dir": direction, "size": size})

        if not pkts_data:
            return None

        # Sort packets by timestamp to ensure chronological order
        pkts_data.sort(key=lambda x: x["ts"])

        # Generate sequences
        dir_seq = [p["dir"] for p in pkts_data]
        delta_seq = [0.0] + list(np.diff([p["ts"] for p in pkts_data]))
        # Normalize packet size
        size_seq = [min(p["size"] / 512.0, 1.0) for p in pkts_data]

        return {
            "pcap_name": pcap_path.name,
            "packet_count": len(pkts_data),
            "direction": dir_seq,
            "delta_time": [float(d) for d in delta_seq],
            "payload_size": [float(s) for s in size_seq]
        }
    except Exception as e:
        print(f"Error processing file {pcap_path}: {e}")
        return None

def process_label_data(label: str, cfg, manual_local_ip: str = None):
    raw_label_dir = Path(cfg.raw_dir) / label
    interim_label_dir = Path(cfg.interim_dir) / label
    interim_label_dir.mkdir(parents=True, exist_ok=True)

    if not raw_label_dir.exists():
        print(f"Path does not exist: {raw_label_dir}")
        return

    for pcap_folder in raw_label_dir.iterdir():
        if not pcap_folder.is_dir():
            continue
        
        # --- 自动检测 Local IP ---
        if manual_local_ip:
            local_ip = manual_local_ip
        else:
            local_ip = detect_local_ip(pcap_folder)
            
        if not local_ip:
            print(f"Warning: No valid PCAPs or unable to detect Local IP in {pcap_folder.name}. Skipping.")
            continue
            
        print(f"Processing session collection: {pcap_folder.name} ")
        
        output_jsonl = interim_label_dir / f"{pcap_folder.name}.jsonl"
        
        with open(output_jsonl, "w", encoding="utf-8") as f:
            session_count = 0
            for session_pcap in pcap_folder.glob("*.pcap*"):
                features = get_single_session_features(session_pcap, local_ip)
                
                if features:
                    features["label"] = label
                    f.write(json.dumps(features) + "\n")
                    session_count += 1
            
            print(f"Done! Wrote {session_count} session features to {output_jsonl.name}")

def main():
    parser = argparse.ArgumentParser(description="Sequence Feature Extraction")
    parser.add_argument("--categories", type=str, default="all", help="Traffic categories to process (comma-separated or 'all')")
    args = parser.parse_args()

    cfg = DataConfig()
    all_labels = cfg.traffic_labels
    
    target_categories = all_labels if args.categories.lower() == "all" else \
                        [s.strip() for s in args.categories.split(",") if s.strip() in all_labels]

    for label in target_categories:
        process_label_data(label, cfg, manual_local_ip=None)

    print("\n[Task Finished]")

if __name__ == "__main__":
    main()
