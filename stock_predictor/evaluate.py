# stock_predictor/evaluate.py

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

def predict(model, test_dl, device):
    """Generates predictions from a trained model."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for xb, _ in test_dl:
            pred, _ = model(xb.to(device))
            predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions).ravel()

def print_metrics(true_prices, pred_prices_p, pred_prices_f, pred_prices_e, last_prices_p):
    """Calculates and prints MSE and MAE for all models."""
    naive_prices = last_prices_p # Random-walk prediction
    
    mse_nv, mae_nv = mean_squared_error(true_prices, naive_prices), mean_absolute_error(true_prices, naive_prices)
    mse_p, mae_p = mean_squared_error(true_prices, pred_prices_p), mean_absolute_error(true_prices, pred_prices_p)
    mse_f, mae_f = mean_squared_error(true_prices, pred_prices_f), mean_absolute_error(true_prices, pred_prices_f)
    mse_e, mae_e = mean_squared_error(true_prices, pred_prices_e), mean_absolute_error(true_prices, pred_prices_e)
    
    print("\n────────  TEST METRICS  ────────")
    print(f"Naïve             MSE {mse_nv:,.2f} | MAE {mae_nv:,.2f}")
    print(f"Price-only LSTM   MSE {mse_p:,.2f} | MAE {mae_p:,.2f}  "
          f"({100*(mae_nv-mae_p)/mae_nv:+.1f} % vs naïve)")
    print(f"Price+TEXT LSTM   MSE {mse_f:,.2f} | MAE {mae_f:,.2f}  "
          f"({100*(mae_nv-mae_f)/mae_nv:+.1f} % vs naïve) "
          f"({100*(mae_p-mae_f)/mae_p:+.1f} % vs price)")
    print(f"ENSEMBLE          MSE {mse_e:,.2f} | MAE {mae_e:,.2f}  "
          f"({100*(mae_nv-mae_e)/mae_nv:+.1f} % vs naïve) "
          f"({100*(mae_f-mae_e)/mae_f:+.1f} % vs price+text)")