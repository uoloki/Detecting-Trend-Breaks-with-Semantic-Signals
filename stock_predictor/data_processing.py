# stock_predictor/data_processing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.model_selection import TimeSeriesSplit
import pickle
import hashlib
from . import config

def get_cache_path(cache_name):
    """Generate cache file path"""
    cache_dir = config.DATA_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{cache_name}.pkl"

def get_data_hash(data):
    """Generate hash for data to detect changes"""
    if isinstance(data, pd.DataFrame):
        data_str = data.to_string()
    elif isinstance(data, list):
        data_str = ''.join(str(x) for x in data)
    else:
        data_str = str(data)
    return hashlib.md5(data_str.encode()).hexdigest()

def load_cached_embeddings(texts, cache_name="finbert_embeddings_masked"):
    """Load cached embeddings or compute if not available"""
    cache_path = get_cache_path(cache_name)
    text_hash = get_data_hash(texts)
    
    # Try to load from cache
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Handle old cache format (direct numpy array)
            if isinstance(cached_data, np.ndarray):
                print(f"Loading embeddings from old cache format: {cache_path}")
                print("Note: Cache will be updated to new format on next save")
                return cached_data
            
            # Handle new cache format (structured dictionary)
            elif isinstance(cached_data, dict) and 'text_hash' in cached_data:
                if cached_data['text_hash'] == text_hash:
                    print(f"Loading embeddings from cache: {cache_path}")
                    return cached_data['embeddings']
                else:
                    print("Text data changed, recomputing embeddings...")
            else:
                print("Unrecognized cache format, recomputing embeddings...")
                
        except Exception as e:
            print(f"Error loading cache: {e}, recomputing embeddings...")
    
    # Compute embeddings
    print("Computing FinBERT embeddings...")
    encoder = SentenceTransformer("yiyanghkust/finbert-tone")
    embeddings = encoder.encode(texts, show_progress_bar=True)
    
    # Save to cache
    try:
        cache_data = {
            'embeddings': embeddings,
            'text_hash': text_hash,
            'model_name': "yiyanghkust/finbert-tone"
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Embeddings cached to: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not cache embeddings: {e}")
    
    return embeddings

def load_cached_umap(embeddings, cache_name="umap_features_masked"):
    """Load cached UMAP transformation or compute if not available"""
    cache_path = get_cache_path(cache_name)
    emb_hash = get_data_hash(embeddings.tolist())
    
    # Try to load from cache
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            # Handle old cache format (direct numpy array)
            if isinstance(cached_data, np.ndarray):
                print(f"Loading UMAP features from old cache format: {cache_path}")
                print("Note: Cache will be updated to new format on next save")
                # For old format, we can't verify hash match, so we'll use it but recompute UMAP model
                umap_features = cached_data
                # Still need to recompute UMAP model for future transforms
                print("Recomputing UMAP model (not cached in old format)...")
                tscv = TimeSeriesSplit(n_splits=5)
                train_idx, _ = list(tscv.split(embeddings))[-1]
                umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, 
                                  metric="cosine", random_state=config.RAND)
                umap_model.fit(embeddings[train_idx])
                return umap_features, umap_model
            
            # Handle new cache format (structured dictionary)
            elif isinstance(cached_data, dict) and 'embeddings_hash' in cached_data:
                if cached_data['embeddings_hash'] == emb_hash:
                    print(f"Loading UMAP features from cache: {cache_path}")
                    return cached_data['umap_features'], cached_data['umap_model']
                else:
                    print("Embeddings changed, recomputing UMAP...")
            else:
                print("Unrecognized UMAP cache format, recomputing...")
                
        except Exception as e:
            print(f"Error loading UMAP cache: {e}, recomputing...")
    
    # Compute UMAP
    print("Computing UMAP transformation...")
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, _ = list(tscv.split(embeddings))[-1]
    
    umap_model = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, 
                      metric="cosine", random_state=config.RAND)
    umap_model.fit(embeddings[train_idx])
    umap_features = umap_model.transform(embeddings)
    
    # Save to cache
    try:
        cache_data = {
            'umap_features': umap_features,
            'umap_model': umap_model,
            'embeddings_hash': emb_hash,
            'umap_params': {
                'n_neighbors': 10,
                'n_components': 2,
                'min_dist': 0.0,
                'metric': 'cosine',
                'random_state': config.RAND
            }
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"UMAP features cached to: {cache_path}")
    except Exception as e:
        print(f"Warning: Could not cache UMAP features: {e}")
    
    return umap_features, umap_model

def load_and_process_data():
    """
    Loads raw data, engineers features for both tweet and price data,
    merges them, and splits into training and testing sets.
    """
    print("Processing tweet data...")
    # ─── TWEET DATA ───────────────────────────────────────────────────────────
    tweets = (pd.read_csv(config.TWEET_CSV).rename(columns=str.lower))
    tweets["date"] = pd.to_datetime(tweets["date"]).dt.normalize()
    tweets["bull"] = (tweets["senti_label"] == "Bullish").astype(int)

    # FinBERT embeddings → 2-D UMAP geometry (with caching)
    texts = tweets["processed"].tolist()
    emb = load_cached_embeddings(texts)
    umap_features, umap_model = load_cached_umap(emb)
    tweets[["u0", "u1"]] = umap_features

    # Daily aggregates + sentiment/UMAP lags + z-score
    daily = (tweets.groupby("date")
                   .agg(bull_ratio=("bull", "mean"), u0=("u0", "mean"), u1=("u1", "mean")))
    for k in [1, 2, 3, 5]:
        daily[[f"bull_lag{k}", f"u0_lag{k}", f"u1_lag{k}"]] = daily[["bull_ratio", "u0", "u1"]].shift(k)
    roll = daily["bull_ratio"].rolling(5, min_periods=5)
    daily["bull_z5"] = (daily["bull_ratio"] - roll.mean()) / roll.std(ddof=0)
    daily = daily.dropna()

    print("Processing price data...")
    # ─── PRICE DATA ───────────────────────────────────────────────────────────
    px = (pd.read_csv(config.PRICE_CSV, parse_dates=["Date"])
            .sort_values("Date")
            .set_index("Date"))

    px["ret_log"] = np.log(px["Close"] / px["Close"].shift(1))
    px["ret1"] = np.log(px["Close"] / px["Close"].shift(1))
    px["ret2"] = np.log(px["Close"] / px["Close"].shift(2))
    px["ret3"] = np.log(px["Close"] / px["Close"].shift(3))
    px["range"] = (px["High"] - px["Low"]) / px["Close"]
    px["co"] = px["Close"] / px["Open"]
    px["hc"] = px["High"] / px["Close"]
    px["lc"] = px["Low"] / px["Close"]
    px["vol_z"] = (px["Volume"] - px["Volume"].rolling(3).mean()) / px["Volume"].rolling(3).std()
    px["vol_z"].replace([np.inf, -np.inf], 0, inplace=True)
    px["vol_z"].fillna(0, inplace=True)
    px = px.dropna()

    # ─── MERGE & SPLIT ────────────────────────────────────────────────────────
    df = (px.join(daily, how="inner").dropna()).reset_index(drop=True)
    
    price_cols = ["Close", "Open", "High", "Low", "Volume", "ret1", "ret2", "ret3", "range", "co", "hc", "lc", "vol_z"]
    text_cols = [c for c in daily.columns]
    feat_price = price_cols
    feat_full = price_cols + text_cols

    split_idx = int(len(df) * (1 - config.TEST_RATIO))
    df_train, df_test = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()

    return df_train, df_test, feat_price, feat_full

def scale_features(df_train, df_test, feature_cols):
    """ Scales features using MinMaxScaler fit only on training data. """
    mat_train = np.zeros((len(df_train), len(feature_cols)), dtype=np.float32)
    mat_test = np.zeros((len(df_test), len(feature_cols)), dtype=np.float32)

    for i, col in enumerate(feature_cols):
        scaler = MinMaxScaler().fit(df_train[[col]])
        mat_train[:, i] = scaler.transform(df_train[[col]]).ravel()
        mat_test[:, i] = scaler.transform(df_test[[col]]).ravel()
    return mat_train, mat_test

def create_sequences(X_matrix, close_prices, lookback):
    """Converts a matrix of features into sequences for LSTM."""
    X_seq, y, last_prices = [], [], []
    for i in range(len(X_matrix) - lookback - 1):
        window = X_matrix[i : i + lookback]
        p_t = close_prices[i + lookback]
        p_t1 = close_prices[i + lookback + 1]
        
        y.append(np.log(p_t1 / p_t))
        X_seq.append(window)
        last_prices.append(p_t)
        
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(y, dtype=np.float32).reshape(-1, 1),
        np.array(last_prices, dtype=np.float32).reshape(-1, 1)
    )