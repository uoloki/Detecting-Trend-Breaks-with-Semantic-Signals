# stock_predictor/trainer.py

import numpy as np
import torch
import torch.nn as nn
from . import config
from .model import BiTALSTM

def train_model(model, train_dl, test_dl, y_test, model_name):
    """
    Trains a given PyTorch model.
    
    Args:
        model (nn.Module): The model instance to train.
        train_dl (DataLoader): Training dataloader.
        test_dl (DataLoader): Testing dataloader.
        y_test (np.array): True labels for the test set for validation loss.
        model_name (str): Name for saving the model weights.

    Returns:
        Trained PyTorch model.
    """
    model.to(config.DEVICE) # Ensure model is on the correct device
    
    loss_fn = nn.SmoothL1Loss(beta=0.003)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, factor=0.5)
    
    best_loss, patience_counter = np.inf, 0
    
    # Create model directory if it doesn't exist
    config.MODEL_DIR.mkdir(exist_ok=True)
    
    for epoch in range(1, config.EPOCHS + 1):
        # -- Training --
        model.train()
        total_train_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(config.DEVICE), yb.to(config.DEVICE)
            
            pred, _ = model(xb)
            loss = loss_fn(pred, yb.squeeze()) # Ensure target is correct shape
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * xb.size(0)
        
        avg_train_loss = total_train_loss / len(train_dl.dataset)

        # -- Validation --
        model.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in test_dl:
                pred, _ = model(xb.to(config.DEVICE))
                preds.append(pred.cpu())
        
        val_loss = loss_fn(torch.cat(preds), torch.tensor(y_test.squeeze())).item()
        scheduler.step(val_loss)
        
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_DIR / f"best_{model_name}.pt")
        else:
            patience_counter += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break
            
    # Load best model
    model.load_state_dict(torch.load(config.MODEL_DIR / f"best_{model_name}.pt"))
    return model