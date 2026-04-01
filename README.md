# A Multimodal Framework for Heterogeneous Traffic Identification at Tor Entry Nodes

This project implements a deep learning framework designed to classify Tor anonymous communication traffic (e.g., obfs4, meek, Snowflake) by fusing **Sequence features** and **Image features** into a unified multi-modal architecture.

## Core Components
* **Temporal Backbone**: Uses `df.py` (Deep Fingerprinting 1D-CNN) to extract packet-level timing and direction features.
* **Spatial Backbone**: Uses `resnet.py` (Residual Network) to extract spatial patterns from traffic-to-image representations.
* **Feature Fusion**: A joint architecture in `multimodal.py` that integrates both modalities for high-accuracy classification.
* **Pipeline Automation**: Includes scripts for PCAP splitting, dual-modality feature engineering, and data alignment.

---

## Environment Setup
Ensure you have Python 3.8+ and the required deep learning libraries installed.

```bash
# Create a new conda environment named 'tor-classification'
conda create -n tor-classification python=3.11 -y

# Activate the environment
conda activate tor-classification

# Install dependencies
pip install -r requirements.txt
```

## Step-by-Step Reproduction Guide
Follow the commands in this specific order to process the raw traffic and train the model:

1. Traffic Preprocessing

The first step converts raw PCAP files into bidirectional flows (Bi-flows):

```bash
python data/split_pcap_to_Bi-flow.py
```

2. Dual-Modality Feature Extraction

Generate the sequence and image features separately:

```bash
# Extract temporal sequence features (Direction, Time, Size)
python data/data_sequence.py --categories <*>

# Generate spatial grayscale image features
python data/data_image.py --categories <*> --min-pkts <*>
```

3. Data Balancing (Optional)

If your dataset classes are imbalanced (e.g., more obfs4 than Snowflake), run this to equalize samples:

```bash
python data/balance_data.py --categories <*> --min_pkts <*> --normalize
```
4. Data Alignment and Manifest Generation
   
This step aligns the two modalities and creates the final dataset manifest for the model:

```bash
python data/generate_data.py
```

5. Model Training
   
Train the multi-modal network. The best performing weights will be saved automatically.

```bash
python models/train.py
```

6. Performance Evaluation
   
Run the evaluation script to generate a full classification report (Precision, Recall, F1-Score) on the test set:

```bash
python models/evaluate.py
```
