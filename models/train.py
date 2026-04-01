import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DataConfig, TrainingConfig
from data.dataset import TorTrafficDataset, LabelEncoder
from models.multimodal import MultiModalClassifier

def training():
    # 1. Initialize Configurations
    d_cfg = DataConfig()
    t_cfg = TrainingConfig()
    
    output_dir = t_cfg.save_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = output_dir / "best_multimodal_model.pth"

    # Set device (Optimized for your NVIDIA A100 environment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # 2. Load Datasets
    # Initialize encoder with predefined traffic labels (meek, obfs4, etc.)
    encoder = LabelEncoder(labels=d_cfg.traffic_labels)
    
    # Using cache_in_memory=True to leverage A100 system RAM for faster training
    train_ds = TorTrafficDataset(d_cfg.processed_dir / "train_manifest.json", encoder, cache_in_memory=True)
    val_ds = TorTrafficDataset(d_cfg.processed_dir / "val_manifest.json", encoder, cache_in_memory=True)

    train_loader = DataLoader(train_ds, batch_size=t_cfg.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=t_cfg.batch_size, shuffle=False, num_workers=4)

    # 3. Initialize Multi-Modal Model
    # Dynamically determine sequence length from the first sample
    seq_len = train_ds[0]["sequence"].shape[1] 
    model = MultiModalClassifier(
        input_length=seq_len,
        num_classes=len(d_cfg.traffic_labels)
    ).to(device)

    # 4. Define Optimizer and Loss Function
    # Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=t_cfg.learning_rate, 
        weight_decay=t_cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    # 5. Training Loop
    best_acc = 0.0
    print(f"\n[!] Starting training for {t_cfg.max_epochs} epochs...")
    
    for epoch in range(t_cfg.max_epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            seq = batch["sequence"].to(device)
            img = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            
            # Forward pass through Multi-Modal Gated architecture
            outputs = model(seq, img)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * seq.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += seq.size(0)

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["sequence"].to(device)
                img = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                outputs = model(seq, img)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * seq.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += seq.size(0)

        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total

        # Progress update
        print(f"Epoch {epoch+1:03d}/{t_cfg.max_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc*100:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} Val Acc: {avg_val_acc*100:.2f}%")

        # Save model if validation accuracy improves
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"  --> Best model weights saved (Accuracy: {best_acc*100:.2f}%)")

    print(f"\n[Task Finished] Training complete. Best Accuracy: {best_acc*100:.2f}%")

if __name__ == "__main__":
    training()