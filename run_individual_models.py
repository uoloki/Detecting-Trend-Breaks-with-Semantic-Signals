"""
Individual Model Analysis

Comprehensive evaluation of deep learning models for S&P 500 price prediction
using multi-seed analysis for robust results.
"""

import pandas as pd
import numpy as np
import warnings
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import random

warnings.filterwarnings('ignore')

# Import project modules
from stock_predictor import data_processing, config, dataset, evaluate
from stock_predictor.model_advanced import (
    CNNLSTMAttention, TransformerPredictor, HybridCNNTransformer, 
    ResidualLSTM
)
from stock_predictor.model import BiTALSTM

def set_random_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_random_seeds(num_seeds, base_seed=42):
    """Generate random seeds for reproducible multi-seed analysis."""
    # Use a fixed base seed to ensure reproducible seed generation
    np.random.seed(base_seed)
    
    # Generate random seeds in a reasonable range
    seeds = np.random.randint(1, 10000, size=num_seeds).tolist()
    
    # Ensure seeds are unique
    seeds = list(set(seeds))
    
    # If we lost some due to duplicates, generate more
    while len(seeds) < num_seeds:
        additional_seeds = np.random.randint(1, 10000, size=num_seeds - len(seeds))
        seeds.extend(additional_seeds.tolist())
        seeds = list(set(seeds))
    
    # Take exactly the number we need
    seeds = seeds[:num_seeds]
    seeds.sort()  # Sort for consistent ordering
    
    return seeds

# Simple BiGRU model for comparison
class BiGRUModel(torch.nn.Module):
    """Bidirectional GRU model for comparison."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)
        
    def forward(self, x):
        _, hidden = self.gru(x)  # hidden: (2, batch, hidden_dim)
        # Concatenate forward and backward hidden states
        hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # (batch, hidden_dim * 2)
        output = self.fc(hidden).squeeze(-1)
        return output, torch.ones(x.size(0), x.size(1), device=x.device)

def get_model_class(model_name):
    """Return model class based on name."""
    model_classes = {
        'bilstm': BiTALSTM,
        'bigru': BiGRUModel,
        'cnn_lstm': CNNLSTMAttention,
        'transformer': TransformerPredictor,
        'hybrid_cnn_transformer': HybridCNNTransformer,
        'residual_lstm': ResidualLSTM
    }
    return model_classes.get(model_name)

def train_individual_model(model_name, df_train, df_val, features, model_params=None, seed=42):
    """Train a single model with specified configuration and seed."""
    
    try:
        # Set random seed for reproducibility
        set_random_seeds(seed)
        
        # Get model class
        ModelClass = get_model_class(model_name)
        if ModelClass is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Scale features
        X_train_scaled, X_val_scaled = data_processing.scale_features(df_train, df_val, features)
        
        # Create sequences
        X_train_seq, y_train_seq, _ = data_processing.create_sequences(
            X_train_scaled, df_train['Close'].values, config.LOOKBACK)
        X_val_seq, y_val_seq, _ = data_processing.create_sequences(
            X_val_scaled, df_val['Close'].values, config.LOOKBACK)
        
        # Create data loaders
        train_ds = dataset.PriceSequenceDataset(X_train_seq, y_train_seq)
        val_ds = dataset.PriceSequenceDataset(X_val_seq, y_val_seq)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Initialize model
        if model_params is None:
            model_params = {}
        
        # Set default parameters based on model type
        if model_name == 'bilstm':
            model = ModelClass(len(features), config.HIDDEN, config.LAYERS, config.DROPOUT)
        elif model_name == 'bigru':
            model = ModelClass(len(features), config.HIDDEN)
        elif model_name == 'cnn_lstm':
            model = ModelClass(len(features), config.HIDDEN, config.LAYERS, config.DROPOUT)
        elif model_name == 'transformer':
            embed_dim = model_params.get('embed_dim', 64)
            num_heads = model_params.get('num_heads', 4)
            # Ensure embed_dim is divisible by num_heads
            embed_dim = (embed_dim // num_heads) * num_heads
            model = ModelClass(len(features), embed_dim, config.LAYERS, config.DROPOUT, num_heads)
        elif model_name == 'hybrid_cnn_transformer':
            model = ModelClass(len(features), config.HIDDEN, config.LAYERS, config.DROPOUT)
        elif model_name == 'residual_lstm':
            model = ModelClass(len(features), config.HIDDEN, config.LAYERS, config.DROPOUT)
        
        # Train model
        from stock_predictor.trainer import train_model
        trained_model = train_model(model, train_dl, val_dl, y_val_seq, f"{model_name}_seed{seed}")
        
        return trained_model, X_val_scaled, y_val_seq
        
    except Exception as e:
        print(f"❌ Failed to train {model_name} (seed {seed}): {e}")
        return None, None, None

def evaluate_individual_model(model, model_name, df_test, features, df_val=None, seed=42):
    """Evaluate a single model on test set."""
    try:
        # Scale features (using validation data for scaling reference)
        if df_val is not None:
            X_scaled, X_test_scaled = data_processing.scale_features(df_val, df_test, features)
        else:
            # Fallback: scale test independently
            from sklearn.preprocessing import MinMaxScaler
            X_test_scaled = np.zeros((len(df_test), len(features)), dtype=np.float32)
            for i, col in enumerate(features):
                scaler = MinMaxScaler().fit(df_test[[col]])
                X_test_scaled[:, i] = scaler.transform(df_test[[col]]).ravel()
        
        # Create sequences
        X_test_seq, y_test_seq, last_prices = data_processing.create_sequences(
            X_test_scaled, df_test['Close'].values, config.LOOKBACK)
        
        # Create data loader
        test_ds = dataset.PriceSequenceDataset(X_test_seq, y_test_seq)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Get predictions
        pred_log_ret = evaluate.predict(model, test_dl, config.DEVICE)
        pred_prices = last_prices[:len(pred_log_ret)].ravel() * np.exp(pred_log_ret)
        true_prices = last_prices[:len(pred_log_ret)].ravel() * np.exp(y_test_seq[:len(pred_log_ret)].ravel())
        
        # Calculate metrics
        mse = mean_squared_error(true_prices, pred_prices)
        mae = mean_absolute_error(true_prices, pred_prices)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((true_prices - pred_prices) / true_prices)) * 100
        
        return {
            'model_name': model_name,
            'seed': seed,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'predictions': pred_prices,
            'true_prices': true_prices,
            'last_prices': last_prices[:len(pred_log_ret)].ravel(),
            'features_used': features
        }
        
    except Exception as e:
        print(f"❌ Failed to evaluate {model_name} (seed {seed}): {e}")
        return None

def train_and_evaluate_with_seeds(config_dict, df_train, df_val, df_test, seeds):
    """Train and evaluate a model configuration with multiple seeds."""
    results = []
    model_label = config_dict['label']
    
    print(f"\n🔄 {model_label} - Multi-seed evaluation ({len(seeds)} runs)")
    print(f"  Seeds: {seeds[:5]}{'...' if len(seeds) > 5 else ''}")
    
    for i, seed in enumerate(seeds):
        print(f"   Run {i+1}/{len(seeds)} (seed {seed})...", end=" ")
        
        # Train model
        model, X_val_scaled, y_val_seq = train_individual_model(
            config_dict['name'], df_train, df_val, config_dict['features'], 
            config_dict.get('params', {}), seed=seed
        )
        
        if model is not None:
            # Evaluate model
            result = evaluate_individual_model(
                model, model_label, df_test, config_dict['features'], df_val, seed=seed
            )
            
            if result is not None:
                results.append(result)
                print(f"MSE={result['mse']:.1f}")
            else:
                print("Evaluation failed")
        else:
            print("Training failed")
    
    return results

def calculate_proper_naive_baseline(df_test):
    """Calculate proper naive baseline (yesterday's price = today's prediction)."""
    test_prices = df_test['Close'].values
    naive_predictions = test_prices[:-1]
    true_prices = test_prices[1:]
    
    if len(naive_predictions) > config.LOOKBACK:
        naive_predictions = naive_predictions[config.LOOKBACK:]
        true_prices = true_prices[config.LOOKBACK:]
    
    naive_mse = mean_squared_error(true_prices, naive_predictions)
    return naive_mse

def analyze_multi_seed_results(all_results, naive_mse):
    """Analyze results across multiple seeds for each model."""
    summary_results = {}
    
    for model_results in all_results:
        if not model_results:
            continue
            
        model_name = model_results[0]['model_name']
        mse_values = [r['mse'] for r in model_results]
        
        if mse_values:
            best_mse = min(mse_values)
            mean_mse = np.mean(mse_values)
            std_mse = np.std(mse_values)
            median_mse = np.median(mse_values)
            worst_mse = max(mse_values)
            
            best_improvement = ((naive_mse - best_mse) / naive_mse) * 100
            mean_improvement = ((naive_mse - mean_mse) / naive_mse) * 100
            median_improvement = ((naive_mse - median_mse) / naive_mse) * 100
            
            stability_score = 1 - (std_mse / mean_mse) if mean_mse > 0 else 0
            
            # Find best seed
            best_result = min(model_results, key=lambda x: x['mse'])
            best_seed = best_result['seed']
            
            summary_results[model_name] = {
                'best_mse': best_mse,
                'mean_mse': mean_mse,
                'std_mse': std_mse,
                'median_mse': median_mse,
                'worst_mse': worst_mse,
                'best_improvement': best_improvement,
                'mean_improvement': mean_improvement,
                'median_improvement': median_improvement,
                'stability_score': stability_score,
                'num_runs': len(mse_values),
                'best_seed': best_seed,
                'mse_range': worst_mse - best_mse
            }
    
    return summary_results

def create_results_table(summary_results, naive_mse, num_seeds):
    """Create and display results table."""
    print(f"\nMULTI-SEED ANALYSIS RESULTS ({num_seeds} random seeds)")
    print("=" * 90)
    print(f"Naive Baseline MSE: {naive_mse:.2f}")
    print()
    
    print(f"{'Model':<20} {'Best MSE':<10} {'Mean±Std':<15} {'Best Imp%':<10} {'Stability':<10} {'Range':<10}")
    print("-" * 90)
    
    # Sort by best improvement
    sorted_results = sorted(summary_results.items(), key=lambda x: x[1]['best_improvement'], reverse=True)
    
    for model_name, stats in sorted_results:
        print(f"{model_name:<20} {stats['best_mse']:<10.1f} "
              f"{stats['mean_mse']:.1f}±{stats['std_mse']:.1f}"f"{'':>3} "
              f"{stats['best_improvement']:>+7.2f}%"f"{'':>2} "
              f"{stats['stability_score']:.3f}"f"{'':>6} "
              f"{stats['mse_range']:.1f}")

def save_detailed_results(summary_results, all_results, naive_mse, num_seeds):
    """Save detailed results to CSV files."""
    output_dir = "outputs/individual_models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary results
    summary_data = []
    for model_name, stats in summary_results.items():
        summary_data.append({
            'Model': model_name,
            'Num_Seeds': num_seeds,
            'Best_MSE': stats['best_mse'],
            'Best_Seed': stats['best_seed'],
            'Mean_MSE': stats['mean_mse'],
            'Std_MSE': stats['std_mse'],
            'Median_MSE': stats['median_mse'],
            'Worst_MSE': stats['worst_mse'],
            'MSE_Range': stats['mse_range'],
            'Stability_Score': stats['stability_score'],
            'Best_Improvement': stats['best_improvement'],
            'Mean_Improvement': stats['mean_improvement'],
            'Median_Improvement': stats['median_improvement'],
            'Naive_Baseline_MSE': naive_mse
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, f"model_summary_{num_seeds}seeds.csv")
    summary_df.to_csv(summary_file, index=False)
    
    # Save detailed results
    detailed_data = []
    for model_results in all_results:
        for result in model_results:
            if result is not None:
                improvement = ((naive_mse - result['mse']) / naive_mse) * 100
                detailed_data.append({
                    'Model': result['model_name'],
                    'Seed': result['seed'],
                    'MSE': result['mse'],
                    'MAE': result['mae'],
                    'RMSE': result['rmse'],
                    'MAPE': result['mape'],
                    'Improvement': improvement,
                    'Naive_Baseline_MSE': naive_mse
                })
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_file = os.path.join(output_dir, f"model_detailed_{num_seeds}seeds.csv")
    detailed_df.to_csv(detailed_file, index=False)
    
    print(f"\nResults saved:")
    print(f"  Summary: {summary_file}")
    print(f"  Detailed: {detailed_file}")

def run_model_analysis(num_seeds=10):
    """Run comprehensive model analysis with multiple random seeds."""
    if num_seeds < 1 or num_seeds > 100:
        raise ValueError("Number of seeds must be between 1 and 100")
    
    print("INDIVIDUAL MODEL ANALYSIS")
    print("=" * 50)
    
    # Load data
    print("Loading and processing data...")
    df_train, df_test, feat_price, feat_full = data_processing.load_and_process_data()
    
    # Create 60/20/20 split
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    n = len(df_combined)
    train_size = int(0.6 * n)
    val_size = int(0.2 * n)
    
    df_train = df_combined[:train_size].copy()
    df_val = df_combined[train_size:train_size + val_size].copy()
    df_test = df_combined[train_size + val_size:].copy()
    
    print(f"Data split: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test")
    
    # Calculate baseline
    naive_mse = calculate_proper_naive_baseline(df_test)
    
    # Generate random seeds
    seeds = generate_random_seeds(num_seeds)
    print(f"Generated {num_seeds} random seeds: {seeds[:10]}{'...' if num_seeds > 10 else ''}")
    
    # Model configurations
    model_configs = [
        {'name': 'cnn_lstm', 'label': 'CNN-LSTM (Price)', 'features': feat_price},
        {'name': 'cnn_lstm', 'label': 'CNN-LSTM (Full)', 'features': feat_full},
        {'name': 'bilstm', 'label': 'BiLSTM (Price)', 'features': feat_price},
        {'name': 'bilstm', 'label': 'BiLSTM (Full)', 'features': feat_full},
        {'name': 'transformer', 'label': 'Transformer (Price)', 'features': feat_price},
        {'name': 'transformer', 'label': 'Transformer (Full)', 'features': feat_full},
    ]
    
    # Run analysis
    print(f"\nRunning analysis with {num_seeds} random seeds per model...")
    start_time = time.time()
    
    all_results = []
    for config in model_configs:
        results = train_and_evaluate_with_seeds(config, df_train, df_val, df_test, seeds)
        all_results.append(results)
    
    # Analyze results
    summary_results = analyze_multi_seed_results(all_results, naive_mse)
    create_results_table(summary_results, naive_mse, num_seeds)
    
    # Save results
    save_detailed_results(summary_results, all_results, naive_mse, num_seeds)
    
    # Find best model
    if summary_results:
        best_model = max(summary_results.items(), key=lambda x: x[1]['best_improvement'])
        print(f"\nBest Model: {best_model[0]}")
        print(f"Best Improvement: +{best_model[1]['best_improvement']:.2f}%")
        print(f"Best Seed: {best_model[1]['best_seed']}")
        print(f"Stability Score: {best_model[1]['stability_score']:.3f}")
        print(f"Mean±Std MSE: {best_model[1]['mean_mse']:.1f}±{best_model[1]['std_mse']:.1f}")
    
    elapsed_time = time.time() - start_time
    print(f"\nAnalysis completed in {elapsed_time:.1f} seconds")
    print(f"Total model training runs: {len(model_configs) * num_seeds}")
    
    return summary_results, naive_mse

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Individual Model Analysis')
    parser.add_argument('--seeds', type=int, default=10, 
                       help='Number of random seeds to use (1-100, default: 10)')
    
    args = parser.parse_args()
    
    run_model_analysis(num_seeds=args.seeds) 