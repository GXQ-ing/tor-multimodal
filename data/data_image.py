import sys
import argparse
import numpy as np
from pathlib import Path
from scapy.all import rdpcap, IP, conf
import logging

logging.getLogger("scapy.runtime").setLevel(logging.ERROR)
conf.max_list_count = 99999999

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DataConfig

def extract_mfr_image_matrix(pcap_path, num_packets=16, payload_len=256, min_pkts=10):
    """
    Extracts a feature matrix where each packet contributes 256 bytes of pure payload.
    The final matrix size is (num_packets * payload_len) reshaped to 64x64.
    """
    try:
        packets = rdpcap(str(pcap_path))
        if len(packets) == 0: return None
        
        payload_list = []
        for pkt in packets:
            if IP in pkt:
                layer_payload = pkt[IP].payload 
                raw_payload = bytes(layer_payload)

                # Pad individual packet to 256 bytes using zero-padding
                if len(raw_payload) < payload_len:
                    p_data = raw_payload.ljust(payload_len, b'\x00')
                else:
                    p_data = raw_payload[:payload_len]
                
                payload_list.append(p_data)
        
        # Validate minimum packet count threshold
        if len(payload_list) < min_pkts: return None

        # Take the first 16 packets
        valid_payloads = payload_list[:num_packets]
        
        all_bytes = []
        for p in valid_payloads:
            all_bytes.extend(list(p))

        # If fewer than 16 packets, pad missing packets with zeros to maintain 64x64 shape
        if len(valid_payloads) < num_packets:
            needed_bytes = (num_packets - len(valid_payloads)) * payload_len
            all_bytes.extend([0] * needed_bytes)

        return np.array(all_bytes, dtype=np.uint8).reshape(64, 64)

    except Exception as e:
        print(f"Error parsing {pcap_path}: {e}")
        return None

def process_label_images(label: str, cfg, min_pkts: int):
    raw_label_dir = Path(cfg.raw_dir) / label
    interim_label_dir = Path(cfg.processed_dir) / label
    
    if not raw_label_dir.exists():
        return

    for pcap_folder in raw_label_dir.iterdir():
        if not pcap_folder.is_dir():
            continue
        
        target_sub_dir = interim_label_dir / "images" / pcap_folder.name
        print(f"Processing {label} : {pcap_folder.name} (min_pkts={min_pkts})")
        
        session_count = 0
        skip_count = 0
        
        for session_pcap in pcap_folder.glob("*.pcap*"):
            matrix = extract_mfr_image_matrix(session_pcap, min_pkts=min_pkts)
            if matrix is not None:
                target_sub_dir.mkdir(parents=True, exist_ok=True)
                output_path = target_sub_dir / f"{session_pcap.stem}.npy"
                np.save(output_path, matrix)
                session_count += 1
            else:
                skip_count += 1
        
        print(f"Done! Valid: {session_count}, Filtered: {skip_count}")

def main():
    parser = argparse.ArgumentParser(description="Traffic Image Feature Extraction")

    parser.add_argument("--categories", type=str, default="all", help="Traffic categories to process (comma-separated or 'all')")
    parser.add_argument("--min-pkts", type=int, default=0, help="Minimum packet count threshold")
    args = parser.parse_args()
        
    cfg = DataConfig()

    all_labels = cfg.traffic_labels
    target_categories = all_labels if args.categories.lower() == "all" else \
                        [s.strip() for s in args.categories.split(",") if s.strip() in all_labels]

    for label in target_categories:
        process_label_images(label, cfg, args.min_pkts)

if __name__ == "__main__":
    main()