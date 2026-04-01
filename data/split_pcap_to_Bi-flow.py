"""
Split original PCAP files into independent PCAP files based on flows (IP pairs, port pairs, protocol).
Input: Directory containing PCAP files.
Output: Subdirectories for each PCAP file, containing separate PCAP files for each identified flow.
"""
import os
from pathlib import Path
from scapy.all import rdpcap, wrpcap
from scapy.layers.inet import IP, TCP, UDP
from collections import defaultdict
from tqdm import tqdm
from typing import Tuple, Dict, List
import sys
    
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DataConfig


def create_directory(path: str) -> None:
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def generate_flow_key(pkt) -> Tuple[Tuple[str, str], Tuple[int, int], str]:
    """Generate a unique flow identifier (IP pair, port pair, protocol)."""
    if IP not in pkt:
        raise ValueError("Packet does not contain IP layer")

    ip_src = pkt[IP].src
    ip_dst = pkt[IP].dst

    if TCP in pkt:
        proto = "TCP"
        src_port = pkt[TCP].sport
        dst_port = pkt[TCP].dport
    elif UDP in pkt:
        proto = "UDP"
        src_port = pkt[UDP].sport
        dst_port = pkt[UDP].dport
    else:
        raise ValueError("Packet is not TCP or UDP")

    # Sort IPs and ports to ensure bidirectional traffic belongs to the same flow
    sorted_ips = tuple(sorted([ip_src, ip_dst]))
    sorted_ports = tuple(sorted([src_port, dst_port]))
    return (sorted_ips, sorted_ports, proto)


def process_pcap(pcap_file: str, output_dir: str) -> None:
    """Process a single PCAP file, split by flow, and save."""
    try:
        packets = rdpcap(pcap_file)
    except Exception as e:
        print(f"Error reading {pcap_file}: {str(e)}")
        return

    flows: Dict[Tuple, List] = defaultdict(list)
    total_packets = len(packets)

    # Extract and group flows
    for pkt in tqdm(packets, total=total_packets, desc=f"Processing {os.path.basename(pcap_file)}", unit="pkt"):
        try:
            flow_key = generate_flow_key(pkt)
            flows[flow_key].append(pkt)
        except ValueError:
            continue  # Skip non-TCP/UDP packets or packets without IP layer

    # Save each flow
    for flow_key, flow_packets in flows.items():
        (ip1, ip2), (port1, port2), proto = flow_key
        flow_name = f"{ip1}_{ip2}_{port1}_{port2}_{proto}"
        flow_path= os.path.join(output_dir, f"{flow_name}.pcap")

        try:
            wrpcap(flow_path, flow_packets)
        except Exception as e:
            print(f"Error writing {flow_path}: {str(e)}")


def process_directory(input_dir: str) -> None:
    """Traverse all PCAP files in the directory and process them."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory {input_dir} does not exist")

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(".pcap")or file.lower().endswith(".pcapng"):
                pcap_path = os.path.join(root, file)
                # Output directory: Subdirectory named after the PCAP file in the same root
                output_dir = os.path.join(root, os.path.splitext(file)[0])
                create_directory(output_dir)
                process_pcap(pcap_path, output_dir)


if __name__ == "__main__":

    cfg = DataConfig()
    categories = cfg.traffic_labels
    for category in categories:
        category_dir = os.path.join(cfg.raw_dir, category)
        if os.path.isdir(category_dir):
            print(f"Processing category: {category}")
            process_directory(category_dir)
            print(f"Finished processing category: {category}\n")
        else:
            print(f"Category directory {category_dir} does not exist, skipping.")
