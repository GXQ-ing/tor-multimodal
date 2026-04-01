import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
from sklearn.metrics import classification_report, accuracy_score

# Add parent directory to path for config and model imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DataConfig
from data.dataset import TorTrafficDataset, LabelEncoder
from models.multimodal import MultiModalClassifier

def evaluate_model():
    """
    Evaluates the trained Multi-Modal model on the validation/test set 
    and generates a comprehensive classification report.
    """
    d_cfg = DataConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*20} Starting Model Evaluation {'='*20}")

    # 1. Load Test/Validation Dataset
    # We use the label encoder to ensure index-to-string mapping is consistent
    encoder = LabelEncoder(labels=d_cfg.traffic_labels)
    
    # Typically, final evaluation is done on val_manifest or a separate test_manifest
    test_ds = TorTrafficDataset(
        d_cfg.processed_dir / "val_manifest.json", 
        encoder, 
        cache_in_memory=True
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)

    # 2. Initialize Model and Load Trained Weights
    num_classes = len(d_cfg.traffic_labels)
    model = MultiModalClassifier(
        input_length=d_cfg.max_sequence_length, 
        num_classes=num_classes
    )
    
    # Path to the best weights saved during the training phase
    model_path = Path("artifacts/best_multimodal_model.pth")  

    if not model_path.exists():
        print(f"[!] Error: Weight file not found at {model_path}")
        return

    # Load state dict and move to evaluation mode (disables Dropout/BatchNorm updates)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 3. Inference Phase
    all_preds, all_labels = [], []
    
    print(f"[*] Running inference on {len(test_ds)} samples...")
    
    with torch.no_grad():
        for batch in test_loader:
            # Transfer all modalities to the GPU/CPU
            seq = batch["sequence"].to(device)
            img = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(seq, img)
            
            # Get the index of the highest logit
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Generate Performance Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n[Results]")
    print(f"Overall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nDetailed Classification Report:")
    
    # This report provides Precision, Recall, and F1 for each individual class
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=d_cfg.traffic_labels, 
        digits=4
    )
    print(report)

if __name__ == "__main__":
    evaluate_model()