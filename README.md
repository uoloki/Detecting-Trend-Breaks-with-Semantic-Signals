# Detecting Trend Breaks with Semantic Signals with Deep Learning and Sentiment Analysis

## Problem Definition & Motivation

This project addresses the challenge of **short-term stock price prediction** for the S&P 500 index using a novel combination of:
- **Historical price data** with engineered technical indicators
- **Real-time sentiment analysis** from financial data on X (Twitter)  
- **Advanced deep learning architectures** with attention mechanisms
- **Ensemble methods** with stability-based weighting

**Research Question**: Can combining financial sentiment with technical indicators improve S&P 500 price prediction accuracy compared to traditional approaches of time series?

**Relevance**: Stock price prediction is crucial for algorithmic trading, risk management, and investment decision-making. The integration of social media sentiment represents a modern approach to capturing market psychology.

**References**: 
We used the dataset from:
 - StockEmotions: Discover Investor Emotions for Financial Sentiment Analysis and Multivariate Time Series [https://arxiv.org/abs/2301.09279]

The following works inspired parts of this project : 
 - SENN: Stock Ensemble-based Neural Network for Stock Prediction [https://github.com/louisowen6/SENN] 
 - Investigating Twitter Sentiment in Cryptocurrency Price Prediction [https://github.com/BaharehAm/Cryptocurrency-Price-Prediction]

## Key Results & Innovation

![Bar chart of model improvements](https://github.com/uoloki/Detecting-Trend-Breaks-with-Semantic-Signals/blob/main/results/bar%20chart.png)

- **Best Overall Model**: Weighted Ensemble achieving **29.5% improvement** over naive baseline (MSE: 674.5)
- **Best Individual Model**: CNN-LSTM with full features achieving **27.4% improvement** (MSE: 695.4)
- **Comprehensive Analysis**: Multi-seed stability testing across 50 random seeds
- **Methodology Validation**: Rigorous data leakage audit ensuring temporal integrity

### Creative Approach
- **Novel Architecture**: CNN-LSTM combining convolutional feature extraction with LSTM temporal modeling
- **Intelligent Ensembling**: Stability-weighted ensemble strategy based on multi-seed performance variance
- **Dual Feature Engineering**: Integration of 13 technical indicators with 16 sentiment metrics
- **Robust Evaluation**: Multi-seed analysis with stability scoring for reliable model selection

## Getting Started

### Installation

```bash
python -m venv venv #activate virtual environment

venv\Scripts\activate #for Windows
source venv/bin/activate #for macOS/Linux

#install dependencies 
pip install -r requirements.txt
```

### Usage

#### Interactive Mode
```bash
python main.py
```

#### Command Line Interface
```bash
python main.py --individual #run individual model analysis with default - 10 random seeds

python main.py --individual --seeds 25 #run with custom number of seeds (1 - 100)
 
python main.py --ensemble #ensemble analysis

python main.py --all --seeds 50 #complete analsis suite (50 seeds)
```

## Model Architecture & Optimization

### Deep Learning Models

#### 1. CNN-LSTM with Attention **Best Performer**
- **Architecture**: Convolutional layers → LSTM → Attention → Dense
- **Innovation**: Combines spatial feature extraction with temporal modeling
- **Performance**: 27.4% improvement over baseline (full features)
- **Stability**: 0.945 (this is the highest among individual models)

#### 2. BiLSTM with Attention
- **Architecture**: Bidirectional LSTM → Attention → Dense  
- **Advantage**: Captures both forward and backward temporal dependencies
- **Performance**: 26.9% improvement (full features)
- **Stability**: 0.854

#### 3. Transformer
- **Architecture**: Multi-head self-attention → Feed-forward → Dense
- **Challenge**: Parameter explosion with full feature set
- **Best Configuration**: Price-only features (22.9% improvement)
- **Stability**: 0.686 (price), 0.790 (full)

### Optimization Strategy
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate - 0.0015
- **Regularization**: Dropout (0.15), early stopping (patience=14)
- **Evaluation Metrics**: MSE, MAE, RMSE, MAPE, % improvement over baseline

## Data Quality & Processing

![Processing pipeline diagram](https://github.com/uoloki/Detecting-Trend-Breaks-with-Semantic-Signals/blob/main/results/graph14.svg)

### Data Sources
- **S&P 500 Price Data**: Daily OHLCV data from Yahoo Finance (2020)
- **Financial Tweets**: StockEmotion dataset with sentiment labels
- **Total Samples**: approximately 250 trading days with aligned sentiment data

![Train / validation / test split](https://github.com/uoloki/Detecting-Trend-Breaks-with-Semantic-Signals/blob/main/results/graph21.svg)

### Feature Engineering

#### Price Features (13 dimensions)
- **Returns**: 1-day, 2-day, 3-day log returns
- **Price Ratios**: high/close, low/close, close/open 
- **Technical Indicators**: price range, volume Z-score
- **Lag Features**: previous day technical values

#### Sentiment Features (16 dimensions)
- **Sentiment Ratios**: bullish/bearish tweet proportions
- **Aggregated Metrics**: daily sentiment means, standard deviations
- **Volume Indicators**: tweet count, engagement metrics
- **Strength Measures**: sentiment intensity scores

### Data Integrity
- **Temporal Splits**: Chronological 60% train, 20% validation, 20% test
- **No Data Leakage**: Strict temporal ordering maintained
- **Proper Scaling**: StandardScaler fit on training data only
- **Sequence Creation**: 6-day lookback windows with sliding approach

## Project Structure

```
Repo/
├── main.py                     # Main entry point with CLI
├── run_individual_models.py    # Individual model analysis
├── run_ensemble.py            # Ensemble analysis  
├── stock_predictor/           # Core package
│   ├── config.py             # Configuration parameters
│   ├── data_processing.py    # Data loading and preprocessing
│   ├── dataset.py            # PyTorch dataset implementation
│   ├── model.py              # BiLSTM model
│   ├── model_advanced.py     # CNN-LSTM and Transformer models
│   ├── trainer.py            # Model training logic
│   └── evaluate.py           # Model evaluation utilities
├── data/                      # Raw and processed data files
│   ├── ^GSPC.csv             # S&P 500 price data
│   ├── processed_stockemo_masked.csv  # Tweet sentiment data
│   ├── train_stockemo_masked.csv      # Training split
│   ├── val_stockemo_masked.csv        # Validation split
│   └── test_stockemo_masked.csv       # Test split
├── models/                    # Saved model checkpoints
└── outputs/                   # Generated results and plots
```

## Results Analysis

![Time-series prediction vs. actual](https://github.com/uoloki/Detecting-Trend-Breaks-with-Semantic-Signals/blob/main/results/model%20performance%20graph.png)

### Performance Comparison (50-Seed Analysis)

| Model | Best MSE | Mean±Std | Improvement | Stability | Range |
|-------|----------|----------|-------------|-----------|-------|
| **Weighted Ensemble** | **674.5** | - | **29.5%** | - | - |
| CNN-LSTM (Full) | 695.4 | 779.3±42.6 | 27.4% | 0.945 | 157.7 |
| BiLSTM (Full) | 699.7 | 808.4±118.2 | 26.9% | 0.854 | 686.5 |
| CNN-LSTM (Price) | 702.8 | 823.4±57.2 | 26.6% | 0.930 | 346.9 |
| BiLSTM (Price) | 727.7 | 894.8±168.5 | 24.0% | 0.812 | 1156.2 |
| Transformer (Price) | 738.5 | 1004.1±315.1 | 22.9% | 0.686 | 1499.7 |
| Transformer (Full) | 752.3 | 981.5±206.2 | 21.4% | 0.790 | 991.7 |

**Baseline**: Naive predictor (previous day's price) - MSE: 957.4

### Key Findings

1. **Ensemble Superiority**: Weighted ensemble achieves best performance (29.5% improvement)
2. **Feature Importance**: Full features generally outperform price-only (except Transformer)
3. **CNN-LSTM Excellence**: Best individual model with high stability
4. **Stability vs Performance**: CNN-LSTM offers best stability-performance trade-off
5. **Transformer Challenges**: Struggles with high-dimensional feature spaces

## Plots & Visualizations

The project generates comprehensive visualizations:
- **Model Performance Comparison**: Bar charts comparing MSE across models
- **Improvement Analysis**: Percentage improvement over naive baseline  
- **Time Series Predictions**: Actual vs predicted price trajectories
- **Multi-seed Stability**: Performance distribution across random seeds
- **Architecture Diagrams**: Data flow and model pipeline visualizations

## Technical Implementation

- **Framework**: PyTorch with CUDA acceleration
- **Training**: GPU-optimized with mixed precision
- **Reproducibility**: Fixed base seed (42) with deterministic seed generation
- **Hardware**: Optimized for both CPU and GPU execution
- **Memory Management**: Efficient batch processing and gradient accumulation

## Conclusions & Future Work

### Key Insights
1. **Ensemble methods** consistently outperform individual models
2. **CNN-LSTM architecture** effectively captures both spatial and temporal patterns
3. **Sentiment features** provide measurable improvements for most architectures
4. **Model stability** is crucial for reliable deployment in financial applications

### Evaluation Metrics Justification
- **MSE**: Primary metric for regression tasks, penalizes large errors
- **Improvement percentage**: Contextualizes performance relative to baseline
- **Stability Score**: Critical for financial applications requiring consistent performance
- **Multi-seed Analysis**: Ensures results are not dependent on random initialization

### Future Directions
1. **Extended Time Horizons**: Multi-step ahead predictions
2. **Alternative Data Sources**: Macroeconomic indicators, options flow, insider trading
3. **Advanced Architectures**: Graph neural networks for market relationships
4. **Risk-Adjusted Metrics**: Sharpe ratio, maximum drawdown optimization
5. **Real-time Deployment**: Low-latency prediction system for live trading

### Limitations
- Limited to 2020 data (single market regime)
- Sentiment data may contain noise from non-financial discussions
- Model performance may degrade in different market conditions
- Computational requirements limit real-time deployment feasibility

## Requirements

**Note**: You'll need to create a `requirements.txt` file with the following key dependencies: 
```
 - matplotlib==3.10.3
 - numpy==2.2.6
 - pandas==2.3.0
 - scikit-learn==1.7.0
 - sentence-transformers==4.1.0
 - torch==2.7.1
 - umap-learn==0.5.7
```

## License

This project is for educational and research purposes.
