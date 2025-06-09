from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
# Enhanced ML imports with fallbacks
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    import numpy as np
    import pandas as pd
    import ta
    import ta.momentum
    import ta.trend
    import ta.volatility
    import logging
    import warnings
    import time
    import random
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback classes for when sklearn is not available
    class KMeans:
        def __init__(self, n_clusters=3, random_state=42, *args, **kwargs):
            self.n_clusters = n_clusters
            self.random_state = random_state
            
        def fit_predict(self, X):
            """Simple fallback clustering based on value ranges"""
            if len(X) == 0:
                return []
            
            # Convert to numpy array if needed
            if hasattr(X, 'values'):
                X = X.values
            
            # Simple clustering based on percentiles
            flat_values = np.array(X).flatten()
            if len(flat_values) == 0:
                return [0] * len(X)
                
            # Create clusters based on percentiles
            p33 = np.percentile(flat_values, 33)
            p67 = np.percentile(flat_values, 67)
            
            clusters = []
            for row in X:
                avg_val = np.mean(row) if hasattr(row, '__iter__') else row
                if avg_val <= p33:
                    clusters.append(0)  # Low volatility/trending
                elif avg_val <= p67:
                    clusters.append(1)  # Medium volatility/mixed
                else:
                    clusters.append(2)  # High volatility/ranging
            
            return clusters
    
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None
            
        def fit_transform(self, X):
            """Simple standardization fallback"""
            if len(X) == 0:
                return X
                
            X_array = np.array(X)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
                
            # Calculate mean and std for each column
            self.mean_ = np.mean(X_array, axis=0)
            self.scale_ = np.std(X_array, axis=0)
            
            # Avoid division by zero
            self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)
            
            # Standardize
            return (X_array - self.mean_) / self.scale_
    
    class IsolationForest:
        def __init__(self, contamination=0.1, random_state=42, *args, **kwargs):
            self.contamination = contamination
            self.random_state = random_state
            
        def fit_predict(self, X):
            """Simple outlier detection fallback using IQR method"""
            if len(X) == 0:
                return []
                
            X_array = np.array(X)
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            
            results = []
            for i, row in enumerate(X_array):
                # Calculate z-score or use IQR method
                if len(X_array) > 4:
                    # Use IQR method for outlier detection
                    q1 = np.percentile(X_array, 25)
                    q3 = np.percentile(X_array, 75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    row_mean = np.mean(row)
                    if lower_bound <= row_mean <= upper_bound:
                        results.append(1)  # Normal
                    else:
                        results.append(-1)  # Outlier
                else:
                    results.append(1)  # Not enough data, assume normal
                    
            return results
    
    SKLEARN_AVAILABLE = False
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OrderFlowAnalyzer:
    """Analyzes order flow imbalance to detect institutional activity"""
    
    def __init__(self):
        self.lookback_period = 14
        self.volume_threshold = 1.5  # Threshold for large volume bars
        self.imbalance_threshold = 0.65  # Threshold for significant order imbalance
        
    def calculate_order_flow_imbalance(self, df):
        """
        Calculate Order Flow Imbalance (OFI) to detect institutional vs retail activity
        
        Returns DataFrame with added OFI metrics
        """
        if len(df) < self.lookback_period:
            logger.warning(f"Not enough data for order flow analysis: {len(df)} < {self.lookback_period}")
            return df
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Calculate buying and selling volume
        result_df['bar_range'] = result_df['high'] - result_df['low']
        result_df['price_move'] = result_df['close'] - result_df['open']
        
        # Avoid division by zero in bar_range calculations
        safe_bar_range = result_df['bar_range'].where(result_df['bar_range'] > 0, 0.0001)
        
        # Calculate buying and selling volume
        result_df['buy_volume'] = np.where(
            result_df['close'] >= result_df['open'],
            result_df['volume'],
            result_df['volume'] * (result_df['close'] - result_df['low']) / safe_bar_range
        )
        result_df['sell_volume'] = np.where(
            result_df['close'] < result_df['open'],
            result_df['volume'],
            result_df['volume'] * (result_df['high'] - result_df['close']) / safe_bar_range
        )
        
        # Fix potential NaN values when bar_range is 0
        result_df['buy_volume'] = result_df['buy_volume'].fillna(0)
        result_df['sell_volume'] = result_df['sell_volume'].fillna(0)
        
        # Calculate Delta (buy volume - sell volume)
        result_df['delta'] = result_df['buy_volume'] - result_df['sell_volume']
        
        # Calculate Cumulative Delta for trend identification
        result_df['cumulative_delta'] = result_df['delta'].cumsum()
        
        # Calculate OFI (Order Flow Imbalance)
        result_df['ofi'] = result_df['delta'] / result_df['volume'].where(result_df['volume'] > 0, 1)
        
        # Calculate rolling average volume for baseline
        result_df['avg_volume'] = result_df['volume'].rolling(window=self.lookback_period).mean()
        
        # Identify large volume bars (potential institutional activity)
        result_df['large_volume'] = result_df['volume'] > (result_df['avg_volume'] * self.volume_threshold)
        
        # Identify strong imbalance bars
        result_df['buy_imbalance'] = (result_df['ofi'] > self.imbalance_threshold) & result_df['large_volume']
        result_df['sell_imbalance'] = (result_df['ofi'] < -self.imbalance_threshold) & result_df['large_volume']
        
        # Calculate institutional activity score (-100 to 100)
        result_df['inst_score'] = np.where(
            result_df['large_volume'],
            result_df['ofi'] * 100,
            result_df['ofi'] * 50  # Lower weight for normal volume
        )
        
        # Identify accumulation/distribution
        result_df['accumulation'] = self._detect_accumulation(result_df)
        result_df['distribution'] = self._detect_distribution(result_df)
        
        # Calculate VWAP (Volume Weighted Average Price) for reference
        result_df['vwap'] = self._calculate_vwap(result_df)
        
        # Divergence between price and delta (hidden institutional activity)
        result_df['delta_divergence'] = self._calculate_delta_divergence(result_df)
        
        return result_df
    
    def _calculate_vwap(self, df):
        """Calculate VWAP from OHLCV data"""
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['price_volume'] = df['typical_price'] * df['volume']
        df['cumulative_volume'] = df['volume'].cumsum()
        df['cumulative_price_volume'] = df['price_volume'].cumsum()
        # Fix division by zero - use where() method with proper condition
        vwap = df['cumulative_price_volume'] / df['cumulative_volume'].where(df['cumulative_volume'] > 0, 1)
        return vwap
    
    def _detect_accumulation(self, df):
        """
        Detect institutional accumulation patterns
        Returns 1 for accumulation, 0 for no accumulation
        """
        accumulation = np.zeros(len(df))
        
        # Need at least 5 bars to detect patterns
        if len(df) < 5:
            return accumulation
        
        # Vectorized calculation for price changes
        close_shifted = df['close'].shift(4)
        price_changes = np.where(close_shifted != 0, 
                                (df['close'] - close_shifted) / close_shifted, 
                                0)
        
        # Vectorized rolling sum of delta over 5 periods
        delta_sums = df['delta'].rolling(window=5, min_periods=5).sum()
        
        # Pattern 1: Price sideways/down with positive delta
        pattern1_mask = ((price_changes >= -0.02) & (price_changes <= 0.005) & 
                        (delta_sums > 0))
        
        # Strong accumulation check (vectorized)
        low_min_rolling = df['low'].rolling(window=5, min_periods=5).min()
        close_threshold = close_shifted * 0.99
        price_trying_down = low_min_rolling < close_threshold
        close_recovery = df['close'] >= close_shifted * 0.995
        
        strong_accumulation = pattern1_mask & price_trying_down & close_recovery
        accumulation[strong_accumulation] = 1
        
        # Pattern 2: Higher lows with increasing delta (for indices >= 10)
        if len(df) >= 11:
            for i in range(10, len(df)):
                last_5_lows = df['low'].iloc[i-5:i+1]
                last_5_delta = df['delta'].iloc[i-5:i+1]
                
                # Check for higher lows and positive delta sum
                if last_5_lows.is_monotonic_increasing and last_5_delta.sum() > 0:
                    accumulation[i] = 1
        
        return accumulation
    
    def _detect_distribution(self, df):
        """
        Detect institutional distribution patterns
        Returns 1 for distribution, 0 for no distribution
        """
        distribution = np.zeros(len(df))
        
        # Need at least 5 bars to detect patterns
        if len(df) < 5:
            return distribution
        
        # Vectorized calculation for price changes
        close_shifted = df['close'].shift(4)
        price_changes = np.where(close_shifted != 0, 
                                (df['close'] - close_shifted) / close_shifted, 
                                0)
        
        # Vectorized rolling sum of delta over 5 periods
        delta_sums = df['delta'].rolling(window=5, min_periods=5).sum()
        
        # Pattern 1: Price sideways/up with negative delta
        pattern1_mask = ((price_changes >= -0.005) & (price_changes <= 0.02) & 
                        (delta_sums < 0))
        
        # Strong distribution check (vectorized)
        high_max_rolling = df['high'].rolling(window=5, min_periods=5).max()
        close_threshold = close_shifted * 1.01
        price_trying_up = high_max_rolling > close_threshold
        close_decline = df['close'] <= close_shifted * 1.005
        
        strong_distribution = pattern1_mask & price_trying_up & close_decline
        distribution[strong_distribution] = 1
        
        # Pattern 2: Lower highs with decreasing delta (for indices >= 10)
        if len(df) >= 11:
            for i in range(10, len(df)):
                last_5_highs = df['high'].iloc[i-5:i+1]
                last_5_delta = df['delta'].iloc[i-5:i+1]
                
                # Check for lower highs and negative delta sum
                if last_5_highs.is_monotonic_decreasing and last_5_delta.sum() < 0:
                    distribution[i] = 1
        
        return distribution
    
    def _calculate_delta_divergence(self, df):
        """
        Calculate divergence between price movement and delta
        Positive values indicate hidden buying pressure despite price decline
        Negative values indicate hidden selling pressure despite price rise
        """
        divergence = np.zeros(len(df))
        
        # Need at least 3 bars to detect divergence
        if len(df) < 3:
            return divergence
        
        # Vectorized calculations
        close_shifted = df['close'].shift(2)
        price_changes = np.where(close_shifted != 0, 
                                (df['close'] - close_shifted) / close_shifted, 
                                0)
        
        # Rolling sum of delta over 3 periods
        delta_sums = df['delta'].rolling(window=3, min_periods=3).sum()
        
        # Rolling average volume for normalization
        avg_volumes = df['volume'].rolling(window=3, min_periods=3).mean()
        norm_deltas = np.where(avg_volumes > 0, delta_sums / avg_volumes, 0)
        
        # Bullish divergence: Price down, delta up (hidden buying)
        bullish_mask = (price_changes < -0.005) & (norm_deltas > 0.1)
        divergence[bullish_mask] = norm_deltas[bullish_mask] * 100
        
        # Bearish divergence: Price up, delta down (hidden selling)
        bearish_mask = (price_changes > 0.005) & (norm_deltas < -0.1)
        divergence[bearish_mask] = norm_deltas[bearish_mask] * 100
                
        return divergence

class AdvancedMarketAnalyzer:
    """Advanced market analyzer with ML-based regime detection and comprehensive fallbacks"""
    
    def __init__(self):
        """Initialize market analyzer with optimal default parameters"""
        self.min_volatility_lookback = 14
        self.max_volatility_lookback = 50
        self.volume_window = 20
        self.volatility_threshold = 0.02
        self.trend_strength_window = 10
        
        # Add order flow analyzer
        self.order_flow_analyzer = OrderFlowAnalyzer()
        
    def detect_market_regime(self, df):
        """Detect market regime with ML clustering and statistical fallbacks"""
        try:
            if SKLEARN_AVAILABLE and len(df) >= 50:
                return self._ml_regime_detection(df)
            else:
                return self._statistical_regime_detection(df)
        except Exception as e:
            logger.warning(f"Market regime detection failed: {e}")
            return self._simple_regime_detection(df)
    
    def _ml_regime_detection(self, df):
        """ML-based market regime detection using clustering"""
        try:
            # Prepare features for clustering
            features = self._prepare_ml_features(df)
            if len(features) < 10:
                return self._statistical_regime_detection(df)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply KMeans clustering for regime detection
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Map clusters to market regimes based on volatility and trend
            regime_mapping = self._map_clusters_to_regimes(features, clusters)
            
            # Extend regime to full DataFrame length
            full_regimes = ['neutral'] * len(df)
            start_idx = len(df) - len(clusters)
            for i, regime in enumerate(clusters):
                full_regimes[start_idx + i] = regime_mapping.get(regime, 'neutral')
            
            return full_regimes
            
        except Exception as e:
            logger.warning(f"ML regime detection failed: {e}")
            return self._statistical_regime_detection(df)
    
    def _prepare_ml_features(self, df):
        """Prepare features for ML analysis"""
        window = min(50, len(df))
        recent_df = df.tail(window).copy()
        
        # Price-based features
        recent_df['returns'] = recent_df['close'].pct_change()
        recent_df['volatility'] = recent_df['returns'].rolling(10).std()
        recent_df['price_trend'] = recent_df['close'].rolling(10).apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, raw=False
        )
        
        # Volume-based features
        recent_df['volume_trend'] = recent_df['volume'].rolling(10).mean()
        recent_df['volume_volatility'] = recent_df['volume'].rolling(10).std()
        
        # High-low spread
        recent_df['hl_spread'] = (recent_df['high'] - recent_df['low']) / recent_df['close']
        
        # Add Order Flow Imbalance features if available
        if 'delta' not in recent_df.columns:
            # Add order flow metrics
            try:
                recent_df = self.order_flow_analyzer.calculate_order_flow_imbalance(recent_df)
            except Exception as e:
                logger.warning(f"Failed to add order flow features: {e}")
                # Add safe defaults
                recent_df['delta'] = 0.0
                recent_df['inst_score'] = 0.0
            
        # Calculate derived order flow features
        recent_df['norm_delta'] = recent_df['delta'] / recent_df['volume'].where(recent_df['volume'] > 0, 1)
        recent_df['delta_trend'] = recent_df['delta'].rolling(10).mean()
        recent_df['delta_volatility'] = recent_df['delta'].rolling(10).std()
        
        # Select features and drop NaN
        feature_cols = ['returns', 'volatility', 'price_trend', 'volume_trend', 'hl_spread', 
                        'norm_delta', 'delta_trend', 'delta_volatility']
        features = recent_df[feature_cols].dropna()
        
        return features.values
    
    def _map_clusters_to_regimes(self, features, clusters):
        """Map cluster numbers to meaningful market regimes"""
        df_features = pd.DataFrame(features, columns=['returns', 'volatility', 'price_trend', 'volume_trend', 
                                                     'hl_spread', 'norm_delta', 'delta_trend', 'delta_volatility'])
        
        regime_mapping = {}
        for cluster_id in set(clusters):
            cluster_mask = clusters == cluster_id
            cluster_data = df_features[cluster_mask]
            
            avg_volatility = cluster_data['volatility'].mean()
            avg_trend = cluster_data['price_trend'].mean()
            avg_delta = cluster_data['norm_delta'].mean() if 'norm_delta' in cluster_data else 0
            
            if avg_volatility > df_features['volatility'].quantile(0.7):
                if avg_delta > 0.1:
                    regime_mapping[cluster_id] = 'volatile_bullish'
                elif avg_delta < -0.1:
                    regime_mapping[cluster_id] = 'volatile_bearish'
                else:
                    regime_mapping[cluster_id] = 'volatile'
            elif abs(avg_trend) > df_features['price_trend'].std():
                if avg_trend > 0:
                    regime_mapping[cluster_id] = 'bullish' if avg_delta > 0 else 'weak_bullish'
                else:
                    regime_mapping[cluster_id] = 'bearish' if avg_delta < 0 else 'weak_bearish'
            else:
                if abs(avg_delta) > 0.05:
                    regime_mapping[cluster_id] = 'accumulation' if avg_delta > 0 else 'distribution'
                else:
                    regime_mapping[cluster_id] = 'neutral'
        
        return regime_mapping
    
    def _statistical_regime_detection(self, df):
        """Statistical fallback for regime detection"""
        regimes = []
        window = min(20, len(df) // 2)
        
        # Get order flow features if not present
        df_with_flow = df.copy()
        if 'delta' not in df_with_flow.columns:
            try:
                df_with_flow = self.order_flow_analyzer.calculate_order_flow_imbalance(df_with_flow)
            except Exception as e:
                logger.warning(f"Failed to add order flow data in regime detection: {e}")
                # Add safe defaults
                df_with_flow['delta'] = 0.0
                df_with_flow['accumulation'] = 0
                df_with_flow['distribution'] = 0
        
        # Vectorized regime detection for better performance
        if len(df_with_flow) < window:
            return ['neutral'] * len(df_with_flow)
            
        # Pre-calculate rolling statistics
        rolling_returns = df_with_flow['close'].pct_change()
        rolling_volatility = rolling_returns.rolling(window=window, min_periods=5).std().fillna(0)
        
        # Calculate price changes over the window
        price_changes = (df_with_flow['close'] - df_with_flow['close'].shift(window)) / df_with_flow['close'].shift(window)
        price_changes = price_changes.fillna(0)
        
        # Calculate rolling averages for delta and volume
        if 'delta' in df_with_flow.columns:
            rolling_delta = df_with_flow['delta'].rolling(window=window, min_periods=5).mean().fillna(0)
            rolling_volume = df_with_flow['volume'].rolling(window=window, min_periods=5).mean()
            avg_delta = rolling_delta / rolling_volume.where(rolling_volume > 0, 1)
            
            # Rolling sum for accumulation/distribution detection
            if 'accumulation' in df_with_flow.columns:
                rolling_accumulation = df_with_flow['accumulation'].rolling(window=5, min_periods=1).sum().fillna(0)
            else:
                rolling_accumulation = pd.Series(0, index=df_with_flow.index)
                
            if 'distribution' in df_with_flow.columns:
                rolling_distribution = df_with_flow['distribution'].rolling(window=5, min_periods=1).sum().fillna(0)
            else:
                rolling_distribution = pd.Series(0, index=df_with_flow.index)
        else:
            avg_delta = pd.Series(0, index=df_with_flow.index)
            rolling_accumulation = pd.Series(0, index=df_with_flow.index)
            rolling_distribution = pd.Series(0, index=df_with_flow.index)
        
        # Vectorized regime classification
        regimes = []
        for i in range(len(df_with_flow)):
            if i < window - 1:
                regimes.append('neutral')
                continue
                
            vol = rolling_volatility.iloc[i]
            price_chg = price_changes.iloc[i]
            delta_val = avg_delta.iloc[i]
            has_accumulation = rolling_accumulation.iloc[i] > 0
            has_distribution = rolling_distribution.iloc[i] > 0
            
            # Regime classification logic (vectorized where possible)
            if vol > self.volatility_threshold:
                if price_chg > 0.02 and delta_val > 0:
                    regimes.append('volatile_bullish')
                elif price_chg < -0.02 and delta_val < 0:
                    regimes.append('volatile_bearish')
                else:
                    regimes.append('volatile')
            elif abs(price_chg) > 0.05:
                if price_chg > 0:
                    if delta_val > 0 or has_accumulation:
                        regimes.append('bullish')
                    else:
                        regimes.append('weak_bullish')
                else:
                    if delta_val < 0 or has_distribution:
                        regimes.append('bearish')
                    else:
                        regimes.append('weak_bearish')
            else:
                if has_accumulation:
                    regimes.append('accumulation')
                elif has_distribution:
                    regimes.append('distribution')
                else:
                    regimes.append('neutral')
        
        return regimes
    
    def _simple_regime_detection(self, df):
        """Simple fallback regime detection"""
        regimes = ['neutral'] * len(df)
        
        if len(df) >= 10:
            # Simple volatility-based classification
            recent_returns = df['close'].pct_change().tail(10)
            volatility = recent_returns.std()
            
            if volatility > 0.03:
                if df['close'].iloc[-10] != 0:
                    recent_price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
                else:
                    recent_price_change = 0
                if recent_price_change > 0:
                    regimes = ['volatile_bullish'] * len(df)
                else:
                    regimes = ['volatile_bearish'] * len(df)
            else:
                price_direction = df['close'].iloc[-1] > df['close'].iloc[-5]
                if price_direction:
                    regimes = ['weak_bullish'] * len(df)
                else:
                    regimes = ['weak_bearish'] * len(df)
        
        return regimes
    
class EnhancedSupertrendIndicator:
    """Enhanced Supertrend with adaptive parameters and multiple timeframe support"""
    
    def __init__(self, period=10, multiplier=3.0, adaptive=True):
        self.period = period
        self.multiplier = multiplier
        self.adaptive = adaptive
        self.min_period = max(7, period - 5)
        self.max_period = period + 10
    
    def calculate(self, df):
        """Calculate enhanced Supertrend with adaptive parameters"""
        if len(df) < self.period:
            logger.warning(f"Insufficient data for Supertrend calculation: {len(df)} < {self.period}")
            return self._basic_supertrend(df)
        
        try:
            if self.adaptive and len(df) >= 50:
                return self._adaptive_supertrend(df)
            else:
                return self._basic_supertrend(df)
        except Exception as e:
            logger.error(f"Enhanced Supertrend calculation failed: {e}")
            return self._basic_supertrend(df)
    
    def _adaptive_supertrend(self, df):
        """Adaptive Supertrend with dynamic parameters"""
        # Calculate market volatility
        returns = df['close'].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else returns.std()
        
        # Adapt parameters based on volatility
        if volatility > 0.03:  # High volatility
            adapted_period = max(self.min_period, self.period - 3)
            adapted_multiplier = self.multiplier * 1.2
        elif volatility < 0.01:  # Low volatility
            adapted_period = min(self.max_period, self.period + 5)
            adapted_multiplier = self.multiplier * 0.8
        else:  # Normal volatility
            adapted_period = self.period
            adapted_multiplier = self.multiplier
        
        return self._calculate_supertrend(df, adapted_period, adapted_multiplier)
    
    def _basic_supertrend(self, df):
        """Basic Supertrend calculation"""
        return self._calculate_supertrend(df, self.period, self.multiplier)
    
    def _calculate_supertrend(self, df, period, multiplier):
        """Core Supertrend calculation logic"""
        # Calculate ATR
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=period
        )
        
        # Calculate basic upper and lower bands
        df['basic_upper'] = (df['high'] + df['low']) / 2 + (multiplier * df['atr'])
        df['basic_lower'] = (df['high'] + df['low']) / 2 - (multiplier * df['atr'])
        
        # Initialize columns
        df['supertrend'] = np.nan
        df['supertrend_direction'] = np.nan
        df['final_upper'] = df['basic_upper'].copy()
        df['final_lower'] = df['basic_lower'].copy()
        
        # Calculate final bands and supertrend
        for i in range(period, len(df)):
            # Upper band logic
            if (df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1] or 
                df['close'].iloc[i-1] > df['final_upper'].iloc[i-1]):
                df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
            else:
                df.loc[df.index[i], 'final_upper'] = df['final_upper'].iloc[i-1]
            
            # Lower band logic
            if (df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1] or 
                df['close'].iloc[i-1] < df['final_lower'].iloc[i-1]):
                df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
            else:
                df.loc[df.index[i], 'final_lower'] = df['final_lower'].iloc[i-1]
            
            # Supertrend logic
            if i == period:
                if df['close'].iloc[i] <= df['final_upper'].iloc[i]:
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1
                else:
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1
            else:
                if df['close'].iloc[i] <= df['final_upper'].iloc[i]:
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1
                else:
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1
        
        # Add trend signal for compatibility
        df['trend'] = np.where(df['supertrend_direction'] == 1, 1, -1)
        
        return df

class SupertrendIndicator:
    """Enhanced Supertrend indicator implementation for faster trend detection"""
    def __init__(self, period=10, multiplier=3.0):
        self.period = period
        self.multiplier = multiplier
        self.enhanced_indicator = EnhancedSupertrendIndicator(period, multiplier, adaptive=True)
        
    def calculate(self, df):
        """Calculate Supertrend indicator using enhanced version with fallback"""
        try:
            # Try enhanced calculation first
            return self.enhanced_indicator.calculate(df)
        except Exception as e:
            logger.warning(f"Enhanced Supertrend failed, using fallback: {e}")
            return self._fallback_calculate(df)
    
    def _fallback_calculate(self, df):
        """Fallback original Supertrend calculation"""
        # Calculate ATR
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=self.period
        )
        
        # Calculate basic upper and lower bands
        df['basic_upper'] = (df['high'] + df['low']) / 2 + (self.multiplier * df['atr'])
        df['basic_lower'] = (df['high'] + df['low']) / 2 - (self.multiplier * df['atr'])
        
        # Initialize Supertrend columns
        df['supertrend'] = np.nan
        df['supertrend_direction'] = np.nan
        df['final_upper'] = np.nan
        df['final_lower'] = np.nan
        
        # Calculate final upper and lower bands
        for i in range(self.period, len(df)):
            if i == self.period:
                # Using .loc to properly set values
                df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
                df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
                
                # Initial trend direction
                if df['close'].iloc[i] <= df['final_upper'].iloc[i]:
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
                else:
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
            else:
                # Calculate upper band
                if (df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1] or 
                    df['close'].iloc[i-1] > df['final_upper'].iloc[i-1]):
                    df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
                else:
                    df.loc[df.index[i], 'final_upper'] = df['final_upper'].iloc[i-1]
                
                # Calculate lower band
                if (df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1] or 
                    df['close'].iloc[i-1] < df['final_lower'].iloc[i-1]):
                    df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
                else:
                    df.loc[df.index[i], 'final_lower'] = df['final_lower'].iloc[i-1]
                
                # Calculate Supertrend value
                if (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
                    df['close'].iloc[i] <= df['final_upper'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
                elif (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
                      df['close'].iloc[i] > df['final_upper'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
                elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
                      df['close'].iloc[i] >= df['final_lower'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
                elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
                      df['close'].iloc[i] < df['final_lower'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
        
        # Add trend signal for compatibility
        df['trend'] = np.where(df['supertrend_direction'] == 1, 1, -1)
        
        return df

class TradingStrategy:
    """Base class for trading strategies"""
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.risk_manager = None
        # Add cache related attributes
        self._cache = {}
        self._max_cache_entries = 10  # Limit cache size
        self._cache_expiry = 3600  # Cache expiry in seconds (1 hour)
        self._last_kline_time = None
        self._cached_dataframe = None
        
    def prepare_data(self, klines):
        """Convert raw klines to a DataFrame with OHLCV data"""
        if not klines or len(klines) == 0:
            logger.warning("No klines data provided to prepare_data")
            return pd.DataFrame()
            
        # Check if we can use cached data
        current_time = time.time()
        cache_key = None
        
        if len(klines) > 0:
            # Create a cache key based on the first and last timestamp + length
            first_timestamp = klines[0][0]
            last_timestamp = klines[-1][0]
            cache_key = f"{first_timestamp}_{last_timestamp}_{len(klines)}"
            
            # Check if we have cached data for this input
            if cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                # Check if cache entry is still valid (not expired)
                if current_time - cache_entry['time'] < self._cache_expiry:
                    logger.debug(f"Using cached data for {cache_key}")
                    return cache_entry['data']
        
        try:
            # Convert to DataFrame if cache miss
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert string values to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
                
            # Convert timestamps to datetime
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            # Ensure dataframe is sorted by time
            df = df.sort_values('open_time', ascending=True).reset_index(drop=True)
            
            # Store in cache if we have a valid key
            if cache_key:
                # Manage cache size - remove oldest entry if needed
                if len(self._cache) >= self._max_cache_entries:
                    oldest_key = min(self._cache.keys(), 
                                    key=lambda k: self._cache[k].get('time', 0))
                    del self._cache[oldest_key]
                    logger.debug(f"Cache full, removed oldest entry {oldest_key}")
                
                # Store in cache with timestamp
                self._cache[cache_key] = {
                    'data': df,
                    'time': current_time
                }
                logger.debug(f"Cached data for {cache_key}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in prepare_data: {e}")
            # Return empty DataFrame on error
            return pd.DataFrame()
    
    def set_risk_manager(self, risk_manager):
        """Set the risk manager for the strategy"""
        self.risk_manager = risk_manager
        logger.info(f"Risk manager set for {self.strategy_name} strategy")
    
    def get_signal(self, klines):
        """
        Main signal generation method that orchestrates all specialized signal methods
        with priority-based decision making and comprehensive validation.
        
        Signal Priority Order:
        1. V-shaped reversal signals (highest priority - extreme conditions)
        2. Squeeze breakout signals (high priority - volatility breakouts)
        3. Market condition specific signals (medium priority - trend following)
        4. Extreme market signals (lower priority - specialized conditions)
        5. Institutional order flow signals (integrates with other signals)
        
        Returns: 'BUY', 'SELL', or None
        """
        try:
            # Prepare data with indicators
            df = self.prepare_data(klines)
            if df.empty or len(df) < self.adx_period:
                logger.warning("Insufficient data for signal generation")
                return None
            
            # Add all technical indicators including order flow analysis
            df = self.add_indicators(df)
            
            # Validate critical data before signal generation
            if not self._validate_critical_data(df):
                logger.warning("Critical data validation failed - skipping signal generation")
                return None
            
            # Get latest market data
            latest = df.iloc[-1]
            current_time = latest['open_time']
            market_condition = latest['market_condition']
            current_price = latest['close']
            
            # Check for cool-off period after consecutive losses
            if self.in_cooloff_period(current_time):
                logger.info(f"In cool-off period. Skipping signal generation.")
                return None
            
            # Initialize signal candidates with their priority scores
            signal_candidates = []
            
            # === PRIORITY 1: V-shaped reversal signals (Emergency/Extreme conditions) ===
            v_reversal_signal = self.get_v_reversal_signal(df)
            if v_reversal_signal:
                signal_candidates.append({
                    'signal': v_reversal_signal,
                    'priority': 1,
                    'confidence': 0.9,
                    'source': 'v_reversal'
                })
            
            # === PRIORITY 2: Squeeze breakout signals (High volatility events) ===
            squeeze_signal = self.get_squeeze_breakout_signal(df)
            if squeeze_signal:
                signal_candidates.append({
                    'signal': squeeze_signal,
                    'priority': 2,
                    'confidence': 0.85,
                    'source': 'squeeze_breakout'
                })
            
            # === PRIORITY 3: Institutional order flow signals ===
            institutional_signal = self.get_institutional_order_flow_signal(df)
            if institutional_signal:
                signal_candidates.append({
                    'signal': institutional_signal,
                    'priority': 2,  # Same priority as squeeze signals
                    'confidence': 0.85,
                    'source': 'institutional_flow'
                })
            
            # === PRIORITY 3: Market condition specific signals ===
            condition_signal = None
            if market_condition == 'SIDEWAYS':
                condition_signal = self.get_sideways_signal(df)
            elif market_condition in ['BULLISH', 'EXTREME_BULLISH']:
                condition_signal = self.get_bullish_signal(df)
            elif market_condition in ['BEARISH', 'EXTREME_BEARISH']:
                condition_signal = self.get_bearish_signal(df)
            
            if condition_signal:
                signal_candidates.append({
                    'signal': condition_signal,
                    'priority': 3,
                    'confidence': 0.7,
                    'source': 'market_condition'
                })
            
            # === PRIORITY 4: Extreme market signals (Specialized conditions) ===
            extreme_signal = self.get_extreme_market_signal(df)
            if extreme_signal:
                signal_candidates.append({
                    'signal': extreme_signal,
                    'priority': 4,
                    'confidence': 0.6,
                    'source': 'extreme_market'
                })
            
            # === SIGNAL CONSOLIDATION AND VALIDATION ===
            if not signal_candidates:
                logger.debug("No signal candidates found")
                return None
            
            # Sort by priority (lower number = higher priority)
            signal_candidates.sort(key=lambda x: x['priority'])
            
            # Apply signal validation and conflict resolution
            final_signal = self._validate_and_consolidate_signals(signal_candidates, df)
            
            if final_signal:
                logger.info(f"Generated {final_signal['signal']} signal from {final_signal['source']} with priority {final_signal['priority']}")
                return final_signal['signal']
            
            logger.debug("No valid signals after consolidation and validation")
            return None
            
        except Exception as e:
            logger.error(f"Error in get_signal: {e}", exc_info=True)
            return None
    
    def get_institutional_order_flow_signal(self, df):
        """
        Generate trading signals based on institutional order flow imbalance
        """
        if len(df) < 5 or 'inst_signal' not in df.columns:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        
        # Return None if required data is missing
        required_columns = ['inst_signal', 'accumulation', 'distribution', 'delta', 'close']
        if not all(col in df.columns for col in required_columns):
            logger.warning(f"Missing required columns for institutional order flow signal")
            return None
        
        # Get last 5 bars for trend analysis
        recent_bars = df.iloc[-5:]
        
        # 1. Strong institutional buying
        strong_buying = (
            latest['inst_signal'] == 1 and
            latest['delta'] > 0 and
            recent_bars['inst_signal'].sum() >= 2 and
            latest['accumulation'] == 1
        )
        
        # 2. Strong institutional selling
        strong_selling = (
            latest['inst_signal'] == -1 and
            latest['delta'] < 0 and
            recent_bars['inst_signal'].sum() <= -2 and
            latest['distribution'] == 1
        )
        
        # 3. Liquidity grab signals (price near VWAP with institutional activity)
        liquidity_buy = (
            latest['liquidity_zone'] == 1 and
            latest['close'] < latest['vwap'] and
            latest['delta'] > 0
        )
        
        liquidity_sell = (
            latest['liquidity_zone'] == -1 and
            latest['close'] > latest['vwap'] and
            latest['delta'] < 0
        )
        
        # 4. Delta divergence (hidden institutional activity)
        bullish_divergence = (
            latest['delta_divergence'] > 30 and
            recent_bars['close'].pct_change().sum() < -0.01
        )
        
        bearish_divergence = (
            latest['delta_divergence'] < -30 and
            recent_bars['close'].pct_change().sum() > 0.01
        )
        
        # Generate signal based on institutional activity
        if (strong_buying or liquidity_buy or bullish_divergence) and not strong_selling:
            logger.info(f"Institutional BUY signal detected: strong_buying={strong_buying}, liquidity_buy={liquidity_buy}, bullish_divergence={bullish_divergence}")
            return "BUY"
        
        if (strong_selling or liquidity_sell or bearish_divergence) and not strong_buying:
            logger.info(f"Institutional SELL signal detected: strong_selling={strong_selling}, liquidity_sell={liquidity_sell}, bearish_divergence={bearish_divergence}")
            return "SELL"
        
        return None
    
    def _validate_and_consolidate_signals(self, signal_candidates, df):
        """
        Validate and consolidate multiple signals with conflict resolution
        
        Args:
            signal_candidates: List of signal dictionaries with priority, confidence, etc.
            df: DataFrame with market data and indicators
            
        Returns:
            Final validated signal dictionary or None
        """
        if not signal_candidates:
            return None
        
        try:
            latest = df.iloc[-1]
            market_condition = latest['market_condition']
            
            # Group signals by type
            buy_signals = [s for s in signal_candidates if s['signal'] == 'BUY']
            sell_signals = [s for s in signal_candidates if s['signal'] == 'SELL']
            
            # Check for institutional confirmation if available
            has_inst_confirmation = False
            inst_bias = 0
            
            if 'inst_signal' in latest:
                inst_bias = latest['inst_signal']
                
                # Strengthen signals that align with institutional activity
                if inst_bias > 0:  # Institutional buying
                    for signal in buy_signals:
                        signal['confidence'] += 0.1
                        signal['priority'] -= 0.5  # Improve priority (lower is better)
                    has_inst_confirmation = len(buy_signals) > 0
                    
                elif inst_bias < 0:  # Institutional selling
                    for signal in sell_signals:
                        signal['confidence'] += 0.1
                        signal['priority'] -= 0.5  # Improve priority
                    has_inst_confirmation = len(sell_signals) > 0
            
            # If no conflicting signals, return the highest priority signal
            if len(buy_signals) > 0 and len(sell_signals) == 0:
                buy_signals.sort(key=lambda x: x['priority'])
                logger.info(f"Selected BUY signal with priority {buy_signals[0]['priority']} from {buy_signals[0]['source']}" +
                           (", confirmed by institutional activity" if has_inst_confirmation and inst_bias > 0 else ""))
                return buy_signals[0]
            elif len(sell_signals) > 0 and len(buy_signals) == 0:
                sell_signals.sort(key=lambda x: x['priority'])
                logger.info(f"Selected SELL signal with priority {sell_signals[0]['priority']} from {sell_signals[0]['source']}" +
                           (", confirmed by institutional activity" if has_inst_confirmation and inst_bias < 0 else ""))
                return sell_signals[0]
            
            # Handle conflicting signals (both BUY and SELL present)
            if len(buy_signals) > 0 and len(sell_signals) > 0:
                # Sort by priority
                buy_signals.sort(key=lambda x: x['priority'])
                sell_signals.sort(key=lambda x: x['priority'])
                
                best_buy = buy_signals[0]
                best_sell = sell_signals[0]
                
                # If one has significantly better priority or confidence, choose it
                if best_buy['priority'] < best_sell['priority'] - 1:
                    logger.info(f"Selected BUY signal (priority {best_buy['priority']}) over SELL signal (priority {best_sell['priority']})")
                    return best_buy
                elif best_sell['priority'] < best_buy['priority'] - 1:
                    logger.info(f"Selected SELL signal (priority {best_sell['priority']}) over BUY signal (priority {best_buy['priority']})")
                    return best_sell
                
                # Otherwise, use institutional bias to break the tie
                if inst_bias > 0:
                    logger.info(f"Selected BUY signal over SELL signal based on institutional buying bias")
                    return best_buy
                elif inst_bias < 0:
                    logger.info(f"Selected SELL signal over BUY signal based on institutional selling bias")
                    return best_sell
                
                # If still tied, prefer the one that aligns with market condition
                if market_condition in ['BULLISH', 'EXTREME_BULLISH', 'accumulation']:
                    return best_buy
                elif market_condition in ['BEARISH', 'EXTREME_BEARISH', 'distribution']:
                    return best_sell
                else:
                    # Last resort: pick the one with highest confidence
                    return best_buy if best_buy['confidence'] >= best_sell['confidence'] else best_sell
            
            # Apply additional validation filters
            final_signal = buy_signals[0] if buy_signals else sell_signals[0]
            
            # Volume validation - ensure sufficient volume for the signal
            volume_ratio = latest.get('volume_ratio', 1.0)
            min_volume_for_priority = {1: 1.0, 2: 1.2, 3: 1.0, 4: 1.3, 5: 1.1, 6: 1.0}
            required_volume = min_volume_for_priority.get(final_signal['priority'], 1.1)
            
            if volume_ratio < required_volume:
                logger.warning(f"Signal rejected due to insufficient volume: {volume_ratio:.2f} < {required_volume:.2f}")
                return None
            
            # RSI extreme validation - avoid signals in very extreme RSI conditions unless high priority
            rsi = latest.get('rsi', 50)
            if final_signal['priority'] > 2:  # Lower priority signals
                if (final_signal['signal'] == 'BUY' and rsi > 85) or (final_signal['signal'] == 'SELL' and rsi < 15):
                    logger.warning(f"Signal rejected due to extreme RSI: {rsi:.1f}")
                    return None
            
            # Market sentiment validation
            try:
                sentiment = self.calculate_market_sentiment(df).iloc[-1]
                # Check if signal aligns with strong market sentiment
                if abs(sentiment) > 0.7:  # Strong sentiment
                    if (final_signal['signal'] == 'BUY' and sentiment < -0.5) or \
                       (final_signal['signal'] == 'SELL' and sentiment > 0.5):
                        # Signal against strong sentiment - reduce confidence or reject low priority signals
                        if final_signal['priority'] > 3:
                            logger.warning(f"Signal against strong sentiment rejected: {final_signal['signal']} vs sentiment {sentiment:.2f}")
                            return None
                        else:
                            # High priority signal - reduce confidence but allow
                            final_signal['confidence'] *= 0.8
                            logger.info(f"High priority signal against sentiment - confidence reduced to {final_signal['confidence']:.2f}")
            except Exception:
                pass  # Sentiment validation failed, continue without it
            
            logger.info(f"Signal validated and approved: {final_signal['signal']} from {final_signal['source']}")
            return final_signal
            
        except Exception as e:
            logger.error(f"Error in signal validation: {e}")
            # Return the first signal as fallback
            return signal_candidates[0] if signal_candidates else None

class DynamicStrategy(TradingStrategy):
    """
    Enhanced Dynamic RAYSOL Trading Strategy that adapts to market trends
    and different market conditions (bullish, bearish, and sideways).
    
    Features:
    - Dynamic position sizing based on volatility and account equity
    - Cool-off period after consecutive losses
    - Supertrend indicator for faster trend detection
    - VWAP for sideways markets
    - Volume-weighted RSI for better signals
    - Bollinger Band squeeze detection for breakouts
    - Fibonacci level integration for support/resistance
    - Enhanced momentum filtering and multi-indicator confirmation
    - Sophisticated reversal detection
    - Order Flow Imbalance for institutional activity detection
    """
    def __init__(self, 
                 trend_ema_fast=8,
                 trend_ema_slow=21,
                 volatility_lookback=20,
                 rsi_period=14,
                 rsi_overbought=70,
                 rsi_oversold=30,
                 volume_ma_period=20,
                 adx_period=14,
                 adx_threshold=25,
                 sideways_threshold=15,
                 # RAYSOL-specific parameters
                 volatility_multiplier=1.1,
                 trend_condition_multiplier=1.3,
                 # New parameters for enhanced features
                 supertrend_period=10,
                 supertrend_multiplier=3.0,
                 fibonacci_levels=[0.236, 0.382, 0.5, 0.618, 0.786],
                 squeeze_threshold=0.5,
                 cooloff_period=3,
                 max_consecutive_losses=2):
        
        super().__init__('DynamicStrategy')
        
        # Base parameters
        self.trend_ema_fast = trend_ema_fast
        self.trend_ema_slow = trend_ema_slow
        self.volatility_lookback = volatility_lookback
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_ma_period = volume_ma_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.sideways_threshold = sideways_threshold
        
        # RAYSOL-specific parameters
        self.volatility_multiplier = volatility_multiplier
        self.trend_condition_multiplier = trend_condition_multiplier
        
        # Enhanced feature parameters
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.fibonacci_levels = fibonacci_levels
        self.squeeze_threshold = squeeze_threshold
        self.cooloff_period = cooloff_period
        self.max_consecutive_losses = max_consecutive_losses
        
        # Initialize advanced components
        self.market_analyzer = AdvancedMarketAnalyzer()
        self.supertrend_indicator = SupertrendIndicator(
            period=self.supertrend_period, 
            multiplier=self.supertrend_multiplier
        )
        
        # Add order flow analyzer
        self.order_flow_analyzer = OrderFlowAnalyzer()
        
        # Initialize trade tracking
        self.consecutive_losses = 0
        self.last_trade_time = None
        self.last_loss_time = None
        
        # Initialize storage for levels
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        
        # State variables
        self.current_trend = None
        self.current_market_condition = None
        self.position_size_pct = 1.0  # Default position size percentage
        
        # Market phase tracking
        self.current_phase = 'UNKNOWN'
        self.phase_confidence = 0.0
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown_experienced': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Cached indicators to avoid recalculation
        self._last_kline_time = None
        self._cached_dataframe = None
        
        # Enhanced caching system
        self._cache = {}
        self._cache_expiry = 300  # 5 minutes
        self._max_cache_entries = 50
        
        logger.info("Enhanced RAYSOL Dynamic Strategy initialized with advanced features")
        
    def prepare_data(self, klines):
        """
        Convert raw klines to a DataFrame with OHLCV data
        Overrides base method to implement enhanced caching for performance
        """
        # Generate a cache key based on first and last kline timestamps
        cache_key = None
        if len(klines) > 0:
            cache_key = f"{klines[0][0]}_{klines[-1][0]}"
        
        # Check if we can use cached data
        if cache_key:
            current_time = int(datetime.now().timestamp())
            
            # Clean up expired cache entries periodically
            if random.random() < 0.1:  # 10% chance to clean on each call
                self._cleanup_expired_cache(current_time)
            
            # Look for cache entry
            if cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                cache_time = cache_entry.get('time', 0)
                
                # Check if cache is still valid (not expired)
                if current_time - cache_time < self._cache_expiry:
                    logger.debug(f"Using cached data for {cache_key}")
                    return cache_entry['data']
        
        # Fall back to simple cache check if complex caching fails
        if len(klines) > 0 and self._last_kline_time == klines[-1][0]:
            if self._cached_dataframe is not None:
                return self._cached_dataframe
            
        # Otherwise prepare data normally
        df = super().prepare_data(klines)
        
        # Cache the result
        if len(klines) > 0:
            # Simple caching (backward compatible)
            self._last_kline_time = klines[-1][0]
            self._cached_dataframe = df
            
            # Enhanced caching with expiry and size management
            if cache_key:
                # Manage cache size - remove oldest entry if needed
                if len(self._cache) >= self._max_cache_entries:
                    oldest_key = min(self._cache.keys(), 
                                    key=lambda k: self._cache[k].get('time', 0))
                    del self._cache[oldest_key]
                    logger.debug(f"Cache full, removed oldest entry {oldest_key}")
                
                # Store in cache with timestamp
                self._cache[cache_key] = {
                    'data': df,
                    'time': current_time
                }
                logger.debug(f"Cached data for {cache_key}")
            
        return df
    
    def _cleanup_expired_cache(self, current_time):
        """Clean up expired cache entries"""
        try:
            expired_keys = []
            for key, cache_entry in self._cache.items():
                cache_time = cache_entry.get('time', 0)
                if current_time - cache_time >= self._cache_expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self._cache[key]
                
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        except Exception as e:
            logger.warning(f"Cache cleanup failed: {e}")
    
    def add_indicators(self, df):
        """Add technical indicators to the DataFrame with enhanced features"""
        try:
            # Input validation
            if df is None or df.empty:
                logger.warning("DataFrame is None or empty")
                return pd.DataFrame()
            
            if len(df) < 20:
                logger.warning("Insufficient data for indicator calculation")
                return df
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required column '{col}' missing from DataFrame")
                    return df
            
            # Trend indicators
            df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=self.trend_ema_fast)
            df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=self.trend_ema_slow)
            
            # Add Supertrend indicator for faster trend detection
            df = self.supertrend_indicator.calculate(df)
            df['trend'] = np.where(df['supertrend_direction'] == 1, 'UPTREND', 'DOWNTREND')
            
            # Momentum indicators
            df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
            
            # Volume indicators
            df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=self.volume_ma_period)
            df['volume_ratio'] = df['volume'] / df['volume_ma'].where(df['volume_ma'] > 0, 1)
            df['volume_weighted_rsi'] = df['rsi'] * df['volume_ratio']
            
            # Volatility indicators
            df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 
                                                        window=self.volatility_lookback)
            df['atr_pct'] = (df['atr'] / df['close'].where(df['close'] > 0, 1)) * 100
            
            # ADX for trend strength
            adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 
                                                 window=self.adx_period)
            df['adx'] = adx_indicator.adx()
            df['di_plus'] = adx_indicator.adx_pos()
            df['di_minus'] = adx_indicator.adx_neg()
            
            # Bollinger Bands
            indicator_bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = indicator_bb.bollinger_hband()
            df['bb_middle'] = indicator_bb.bollinger_mavg()
            df['bb_lower'] = indicator_bb.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].where(df['bb_middle'] > 0, 1)
            df['bb_squeeze'] = df['bb_width'] < self.squeeze_threshold
            
            # MACD for trend confirmation
            macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()
            df['macd_crossover'] = np.where(
                (df['macd'].shift(1) < df['macd_signal'].shift(1)) & 
                (df['macd'] > df['macd_signal']), 1,
                np.where((df['macd'].shift(1) > df['macd_signal'].shift(1)) & 
                        (df['macd'] < df['macd_signal']), -1, 0))
            
            # Calculate VWAP
            df['vwap'] = self.calculate_vwap(df)
            
            # Calculate Fibonacci levels
            self.calculate_fibonacci_levels(df)
            
            # Market condition classification
            df['market_condition'] = self.classify_market_condition(df)
            
            # Reversal pattern detection
            df['potential_reversal'] = self.detect_reversal_patterns(df)
            
            # Advanced market analysis
            df['market_sentiment'] = self.calculate_market_sentiment(df)
            df['flash_crash'] = self.detect_flash_crash(df)
            df['pump_event'] = self.detect_pump_events(df)
            
            # Order Flow Imbalance analysis
            try:
                df_with_order_flow = self.order_flow_analyzer.calculate_order_flow_imbalance(df)
                order_flow_columns = ['delta', 'cumulative_delta', 'ofi', 'buy_imbalance', 'sell_imbalance', 
                                    'inst_score', 'accumulation', 'distribution', 'delta_divergence']
                
                for col in order_flow_columns:
                    if col in df_with_order_flow.columns:
                        df[col] = df_with_order_flow[col]
                    else:
                        if col in ['delta', 'cumulative_delta', 'ofi', 'inst_score', 'delta_divergence']:
                            df[col] = 0.0
                        elif col in ['buy_imbalance', 'sell_imbalance', 'accumulation', 'distribution']:
                            df[col] = 0
                
                # Generate institutional signals
                df['inst_signal'] = np.where(
                    (df['inst_score'] > 30) & (df['delta'] > 0), 1,
                    np.where((df['inst_score'] < -30) & (df['delta'] < 0), -1, 0)
                )
                
                # Identify liquidity zones
                df['liquidity_zone'] = np.where(
                    df['inst_score'].abs() > 50, 
                    np.sign(df['inst_score']), 0
                )
                
            except Exception as e:
                logger.warning(f"Order flow analysis failed, using defaults: {e}")
                for col in ['delta', 'cumulative_delta', 'ofi', 'inst_score', 'delta_divergence']:
                    df[col] = 0.0
                for col in ['buy_imbalance', 'sell_imbalance', 'accumulation', 'distribution', 'inst_signal', 'liquidity_zone']:
                    df[col] = 0
            
            # Position scoring
            df['position_score'] = self.calculate_position_score(df)
            
            # Handle NaN values
            df = self._handle_nan_values(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            # Return original DataFrame with minimal indicators on error
            if 'rsi' not in df.columns:
                df['rsi'] = 50.0
            if 'market_condition' not in df.columns:
                df['market_condition'] = 'SIDEWAYS'
            return df
        df['volume_ratio'] = df['volume'] / df['volume_ma'].where(df['volume_ma'] > 0, 1)  # Avoid division by zero
        
        # Volume-weighted RSI (using the volume_ratio calculated above)
        df['volume_weighted_rsi'] = df['rsi'] * df['volume_ratio']
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], 
                                                    df['close'], 
                                                    window=self.volatility_lookback)
        df['atr_pct'] = (df['atr'] / df['close'].where(df['close'] > 0, 1)) * 100  # Avoid division by zero
        
        # ADX for trend strength
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 
                                             window=self.adx_period)
        df['adx'] = adx_indicator.adx()
        df['di_plus'] = adx_indicator.adx_pos()
        df['di_minus'] = adx_indicator.adx_neg()
        
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(df['close'], 
                                                   window=20, 
                                                   window_dev=2)
        df['bb_upper'] = indicator_bb.bollinger_hband()
        df['bb_middle'] = indicator_bb.bollinger_mavg()
        df['bb_lower'] = indicator_bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].where(df['bb_middle'] > 0, 1)  # Avoid division by zero
        
        # Bollinger Band Squeeze detection
        df['bb_squeeze'] = df['bb_width'] < self.squeeze_threshold
        
        # MACD for additional trend confirmation
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_crossover'] = np.where(
            (df['macd'].shift(1) < df['macd_signal'].shift(1)) & 
            (df['macd'] > df['macd_signal']), 
            1, np.where(
                (df['macd'].shift(1) > df['macd_signal'].shift(1)) & 
                (df['macd'] < df['macd_signal']), 
                -1, 0
            )
        )
        
        # VWAP (Volume Weighted Average Price) - calculated per day
        df['vwap'] = self.calculate_vwap(df)
        
        # Calculate Fibonacci levels based on recent swing highs and lows
        self.calculate_fibonacci_levels(df)
        
        # Market condition classification with improved detection
        df['market_condition'] = self.classify_market_condition(df)
        
        # Reversal detection indicators
        df['potential_reversal'] = self.detect_reversal_patterns(df)
        
        # Advanced market analysis
        df['market_sentiment'] = self.calculate_market_sentiment(df)
        df['flash_crash'] = self.detect_flash_crash(df)
        df['pump_event'] = self.detect_pump_events(df)
        
        # Add Order Flow Imbalance analysis - CRITICAL FIX
        try:
            # Calculate order flow imbalance and merge with main dataframe
            df_with_order_flow = self.order_flow_analyzer.calculate_order_flow_imbalance(df)
            
            # Add the new order flow columns to the main dataframe
            order_flow_columns = ['delta', 'cumulative_delta', 'ofi', 'buy_imbalance', 'sell_imbalance', 
                                'inst_score', 'accumulation', 'distribution', 'delta_divergence']
            
            for col in order_flow_columns:
                if col in df_with_order_flow.columns:
                    df[col] = df_with_order_flow[col]
                else:
                    # Provide safe defaults if column is missing
                    if col in ['delta', 'cumulative_delta', 'ofi', 'inst_score', 'delta_divergence']:
                        df[col] = 0.0
                    elif col in ['buy_imbalance', 'sell_imbalance', 'accumulation', 'distribution']:
                        df[col] = 0
            
            # Generate institutional signals based on order flow
            df['inst_signal'] = np.where(
                (df['inst_score'] > 30) & (df['delta'] > 0), 1,  # Strong buying
                np.where((df['inst_score'] < -30) & (df['delta'] < 0), -1, 0)  # Strong selling
            )
            
            # Identify liquidity zones (areas where institutional activity is high)
            df['liquidity_zone'] = np.where(
                df['inst_score'].abs() > 50, 
                np.sign(df['inst_score']), 0
            )
            
            logger.debug("Order flow analysis successfully integrated")
            
        except Exception as e:
            logger.warning(f"Order flow analysis failed, using defaults: {e}")
            # Provide safe defaults for all order flow columns
            df['delta'] = 0.0
            df['cumulative_delta'] = 0.0
            df['ofi'] = 0.0
            df['buy_imbalance'] = 0
            df['sell_imbalance'] = 0
            df['inst_score'] = 0.0
            df['accumulation'] = 0
            df['distribution'] = 0
            df['delta_divergence'] = 0.0
            df['inst_signal'] = 0
            df['liquidity_zone'] = 0
        
        # Market phase detection
        phase, confidence = self.detect_market_phase(df)
        self.current_phase = phase
        self.phase_confidence = confidence
        
        # Position scoring for signal prioritization
        df['position_score'] = self.calculate_position_score(df)
        
        # Handle NaN values for robust trading
        df = self._handle_nan_values(df)
        
        return df
    
    def _handle_nan_values(self, df):
        """Handle NaN values in technical indicators for robust trading"""
        try:
            if df.empty:
                return df
                
            # Critical indicators that must not be NaN for signal generation
            critical_indicators = [
                'rsi', 'supertrend_direction', 'adx', 'close', 'volume',
                'atr', 'ema_fast', 'ema_slow', 'volume_ratio'
            ]
            
            # Handle NaN values with appropriate fallbacks
            for indicator in critical_indicators:
                if indicator in df.columns:
                    if indicator == 'rsi':
                        df[indicator] = df[indicator].fillna(50)  # Neutral RSI
                    elif indicator == 'supertrend_direction':
                        df[indicator] = df[indicator].fillna(1)  # Default bullish
                    elif indicator == 'adx':
                        df[indicator] = df[indicator].fillna(self.adx_threshold - 1)  # Below threshold
                    elif indicator == 'volume_ratio':
                        df[indicator] = df[indicator].fillna(1.0)  # Normal volume
                    elif indicator in ['atr', 'ema_fast', 'ema_slow']:
                        # Forward fill then backward fill for price-based indicators
                        df[indicator] = df[indicator].ffill().bfill()
                    else:
                        # For other indicators, use forward fill
                        df[indicator] = df[indicator].ffill()
            
            # Handle optional indicators with safe defaults
            optional_indicators = {
                'bb_upper': df['close'] * 1.02,  # 2% above close
                'bb_lower': df['close'] * 0.98,  # 2% below close  
                'bb_middle': df['close'],        # Current close
                'bb_width': 0.04,               # 4% width
                'macd': 0,                      # Neutral MACD
                'macd_signal': 0,               # Neutral signal
                'macd_diff': 0,                 # No difference
                'vwap': df['close'],            # Use close as VWAP fallback
                'volume_weighted_rsi': df['rsi'] if 'rsi' in df.columns else 50,
                'market_condition': 'SIDEWAYS', # Default market condition
                'potential_reversal': 0,        # No reversal
                'position_score': 0.5          # Neutral score
            }
            
            for indicator, fallback in optional_indicators.items():
                if indicator in df.columns:
                    if isinstance(fallback, (int, float)):
                        df[indicator] = df[indicator].fillna(fallback)
                    else:
                        # For Series fallbacks
                        df[indicator] = df[indicator].fillna(fallback)
            
            # Ensure no infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Final check - if any critical indicators still have NaN, log warning
            for indicator in critical_indicators:
                if indicator in df.columns and df[indicator].isna().any():
                    logger.warning(f"Critical indicator {indicator} still contains NaN values after cleaning")
                    
            return df
            
        except Exception as e:
            logger.error(f"Error handling NaN values: {e}")
            return df  # Return original dataframe if cleaning fails
    
    def _validate_critical_data(self, df):
        """Validate that critical data is available and valid for signal generation"""
        try:
            if df.empty or len(df) == 0:
                logger.warning("DataFrame is empty")
                return False
                
            latest = df.iloc[-1]
            
            # Check for critical price data
            required_price_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_price_columns:
                if col not in df.columns or pd.isna(latest[col]) or latest[col] <= 0:
                    logger.warning(f"Invalid or missing price data for {col}: {latest.get(col, 'N/A')}")
                    return False
            
            # Check for critical indicators
            critical_indicators = ['rsi', 'supertrend_direction', 'adx']
            for indicator in critical_indicators:
                if indicator not in df.columns or pd.isna(latest[indicator]):
                    logger.warning(f"Missing or invalid indicator data for {indicator}")
                    return False
            
            # Validate reasonable price ranges (sanity check)
            if latest['high'] < latest['low']:
                logger.warning("Invalid price data: high < low")
                return False
                
            if latest['close'] > latest['high'] or latest['close'] < latest['low']:
                logger.warning("Invalid price data: close outside high-low range")
                return False
            
            # Validate RSI is in valid range
            if not (0 <= latest['rsi'] <= 100):
                logger.warning(f"RSI out of valid range: {latest['rsi']}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating critical data: {e}")
            return False
            logger.error(f"Error validating critical data: {e}")
            return False
    
    def calculate_vwap(self, df):
        """Calculate VWAP (Volume Weighted Average Price)"""
        try:
            if df.empty or len(df) == 0:
                logger.warning("Empty dataframe for VWAP calculation")
                return pd.Series([], dtype=float)
            
            # Ensure we have required columns
            if not all(col in df.columns for col in ['close', 'volume']):
                logger.warning("Missing required columns for VWAP calculation")
                return df['close'].copy()
            
            # Calculate VWAP for entire dataset if no time column
            if 'open_time' not in df.columns:
                cum_vol_price = (df['close'] * df['volume']).cumsum()
                cum_vol = df['volume'].cumsum()
                return cum_vol_price / cum_vol.where(cum_vol > 0, 1)
            
            # Convert to datetime if needed
            if not hasattr(df['open_time'], 'dt'):
                try:
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
                except:
                    # Fallback to simple VWAP
                    cum_vol_price = (df['close'] * df['volume']).cumsum()
                    cum_vol = df['volume'].cumsum()
                    return cum_vol_price / cum_vol.where(cum_vol > 0, 1)
            
            # Get date component for daily VWAP
            df['date'] = df['open_time'].dt.date
            
            # Calculate VWAP for each day
            vwap = pd.Series(index=df.index, dtype=float)
            for date, group in df.groupby('date'):
                # Calculate cumulative sum of price * volume
                cum_vol_price = (group['close'] * group['volume']).cumsum()
                # Calculate cumulative sum of volume
                cum_vol = group['volume'].cumsum()
                # Calculate VWAP with division by zero protection
                daily_vwap = cum_vol_price / cum_vol.where(cum_vol > 0, 1)
                # Add to result series
                vwap.loc[group.index] = daily_vwap
                
            # Clean up temporary column
            if 'date' in df.columns:
                df.drop('date', axis=1, inplace=True)
                
            return vwap.fillna(df['close'])
            
        except Exception as e:
            logger.warning(f"VWAP calculation failed: {e}")
            # Return close price as fallback
            return df['close'].copy()
    
    def calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement/extension levels for support and resistance"""
        try:
            if df.empty or len(df) < 20:
                logger.debug("Insufficient data for Fibonacci calculation")
                self.fib_support_levels = []
                self.fib_resistance_levels = []
                return
            
            # Find recent swing high and low points
            window = min(100, len(df) - 1)
            price_data = df['close'].iloc[-window:]
            
            # Identify swing high and low
            swing_high = price_data.max()
            swing_low = price_data.min()
            
            # Check if there's sufficient price movement
            price_range = swing_high - swing_low
            if price_range <= 0 or price_range / swing_low < 0.01:  # Less than 1% range
                logger.debug("Insufficient price range for meaningful Fibonacci levels")
                self.fib_support_levels = []
                self.fib_resistance_levels = []
                return
            
            # Reset fibonacci levels
            self.fib_support_levels = []
            self.fib_resistance_levels = []
            
            # Get current price and trend
            latest = df.iloc[-1]
            current_price = latest['close']
            
            # Determine trend based on supertrend or price movement
            if 'supertrend_direction' in df.columns and not pd.isna(latest['supertrend_direction']):
                is_uptrend = latest['supertrend_direction'] == 1
            else:
                # Fallback: use recent price movement
                is_uptrend = current_price > price_data.iloc[-20:].mean()
            
            if is_uptrend:
                # In uptrend, calculate fib retracements from low to high for support
                for fib in self.fibonacci_levels:
                    level = swing_low + (swing_high - swing_low) * (1 - fib)
                    if level < current_price:
                        self.fib_support_levels.append(level)
                    else:
                        self.fib_resistance_levels.append(level)
                        
                # Add extension levels for resistance
                for ext in [1.272, 1.618, 2.0]:
                    level = swing_low + (swing_high - swing_low) * ext
                    self.fib_resistance_levels.append(level)
                    
            else:  # DOWNTREND
                # In downtrend, calculate fib retracements from high to low for resistance
                for fib in self.fibonacci_levels:
                    level = swing_high - (swing_high - swing_low) * fib
                    if level > current_price:
                        self.fib_resistance_levels.append(level)
                    else:
                        self.fib_support_levels.append(level)
                        
                # Add extension levels for support
                for ext in [1.272, 1.618, 2.0]:
                    level = swing_high - (swing_high - swing_low) * ext
                    self.fib_support_levels.append(level)
            
            # Sort the levels
            self.fib_support_levels.sort(reverse=True)  # Descending
            self.fib_resistance_levels.sort()  # Ascending
            
            # Keep only closest levels to current price
            self.fib_support_levels = self.fib_support_levels[:5]
            self.fib_resistance_levels = self.fib_resistance_levels[:5]
            
        except Exception as e:
            logger.warning(f"Fibonacci calculation failed: {e}")
            self.fib_support_levels = []
            self.fib_resistance_levels = []
        price_data = df['close'].iloc[-window:]
        
        # Identify swing high and low
        swing_high = price_data.max()
        swing_low = price_data.min()
        
        # Check if there's sufficient price movement to calculate meaningful levels
        price_range = swing_high - swing_low
        if price_range <= 0 or price_range / swing_low < 0.01:  # Less than 1% range
            logger.debug("Insufficient price range for meaningful Fibonacci levels")
            return
        
        # Reset fibonacci levels
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        
        # Calculate levels based on trend
        latest = df.iloc[-1]
        current_price = latest['close']
        current_trend = latest['trend']
        
        if current_trend == 'UPTREND':
            # In uptrend, calculate fib retracements from low to high for support
            for fib in self.fibonacci_levels:
                level = swing_low + (swing_high - swing_low) * (1 - fib)
                if level < current_price:
                    self.fib_support_levels.append(level)
                else:
                    self.fib_resistance_levels.append(level)
                    
            # Add extension levels for resistance
            for ext in [1.272, 1.618, 2.0]:
                level = swing_low + (swing_high - swing_low) * ext
                self.fib_resistance_levels.append(level)
                
        else:  # DOWNTREND
            # In downtrend, calculate fib retracements from high to low for resistance
            for fib in self.fibonacci_levels:
                level = swing_high - (swing_high - swing_low) * fib
                if level > current_price:
                    self.fib_resistance_levels.append(level)
                else:
                    self.fib_support_levels.append(level)
                    
            # Add extension levels for support
            for ext in [1.272, 1.618, 2.0]:
                level = swing_high - (swing_high - swing_low) * ext
                self.fib_support_levels.append(level)
        
        # Sort the levels
        self.fib_support_levels.sort(reverse=True)  # Descending
        self.fib_resistance_levels.sort()  # Ascending
    
    def detect_reversal_patterns(self, df):
        """
        Optimized reversal pattern detection using vectorized operations where possible
        """
        if len(df) < 5:
            return pd.Series(0, index=df.index)
            
        # Initialize result series
        reversal = pd.Series(0, index=df.index)
        
        # Vectorized calculations for pattern components
        is_bullish = df['close'] > df['open']
        is_bearish = df['close'] < df['open']
        
        # Shifted data for pattern comparison
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_is_bullish = is_bullish.shift(1)
        prev_is_bearish = is_bearish.shift(1)
        
        # RSI divergence data
        prev2_low = df['low'].shift(2)
        prev2_high = df['high'].shift(2)
        prev2_rsi = df['rsi'].shift(2)
        prev1_rsi = df['rsi'].shift(1)
        
        # Vectorized bullish engulfing pattern
        bullish_engulfing = (
            is_bullish & prev_is_bearish &
            (df['close'] > prev_open) & (df['open'] < prev_close)
        )
        
        # Vectorized bearish engulfing pattern  
        bearish_engulfing = (
            is_bearish & prev_is_bullish &
            (df['close'] < prev_open) & (df['open'] > prev_close)
        )
        
        # Vectorized RSI bullish divergence
        rsi_bull_div = (
            (prev2_low > df['low'].shift(1)) &  # Price making lower low
            (prev2_rsi < prev1_rsi) &           # RSI making higher low
            (df['supertrend_direction'] == 1)    # Confirmed by Supertrend
        )
        
        # Vectorized RSI bearish divergence
        rsi_bear_div = (
            (prev2_high < df['high'].shift(1)) & # Price making higher high
            (prev2_rsi > prev1_rsi) &            # RSI making lower high
            (df['supertrend_direction'] == -1)   # Confirmed by Supertrend
        )
        
        # Apply patterns to reversal series
        reversal[bullish_engulfing | rsi_bull_div] = 1
        reversal[bearish_engulfing | rsi_bear_div] = -1
        
        # Complex patterns that still need loops (hammer, shooting star)
        for i in range(4, len(df)):
            curr = df.iloc[i]
            
            # Hammer pattern (bullish) - complex geometry
            body_size = abs(curr['close'] - curr['open'])
            lower_shadow = min(curr['open'], curr['close']) - curr['low']
            upper_shadow = curr['high'] - max(curr['open'], curr['close'])
            
            if (lower_shadow > body_size * 2 and 
                lower_shadow > upper_shadow * 3 and 
                body_size > 0):
                reversal.iloc[i] = 1
                
            # Shooting star (bearish) - complex geometry
            elif (upper_shadow > body_size * 2 and 
                  upper_shadow > lower_shadow * 3 and 
                  body_size > 0):
                reversal.iloc[i] = -1
                
        return reversal
    
    def classify_market_condition(self, df):
        """
        Enhanced market condition classification with better state transitions and stability
        """
        conditions = []
        lookback_period = 10  # Use more historical data for smoother transitions
        current_condition = None
        condition_streak = 0  # Track how long we've been in a condition
        
        for i in range(len(df)):
            if i < self.adx_period:
                conditions.append('SIDEWAYS')  # Default for initial rows
                continue
                
            # Get relevant indicators
            adx = df['adx'].iloc[i]
            di_plus = df['di_plus'].iloc[i]
            di_minus = df['di_minus'].iloc[i]
            rsi = df['rsi'].iloc[i]
            bb_width = df['bb_width'].iloc[i]
            supertrend_dir = df['supertrend_direction'].iloc[i] if i >= self.supertrend_period else 0
            macd_crossover = df['macd_crossover'].iloc[i] if 'macd_crossover' in df else 0
            
            # Get average ADX for more stability
            lookback = min(lookback_period, i)
            avg_adx = df['adx'].iloc[i-lookback:i+1].mean() if i >= lookback else adx
            
            # Check for squeeze condition (low volatility, potential breakout)
            is_squeeze = bb_width < self.squeeze_threshold
            
            # Calculate the strength of each potential market condition
            bullish_strength = 0
            bearish_strength = 0
            sideways_strength = 0
            squeeze_strength = 0
            
            # ADX trending strength (0-100)
            trend_strength = min(100, adx * 2)  # Normalize to 0-100 scale
            
            # Directional bias (-100 to +100, negative = bearish, positive = bullish)
            if di_plus + di_minus > 0:  # Avoid division by zero
                directional_bias = 100 * (di_plus - di_minus) / (di_plus + di_minus)
            else:
                directional_bias = 0
                
            # Supertrend confirmation
            supertrend_bias = 100 if supertrend_dir > 0 else -100 if supertrend_dir < 0 else 0
            
            # RSI bias (normalized to -100 to +100)
            rsi_bias = (rsi - 50) * 2  # 0 = neutral, +100 = extremely bullish, -100 = extremely bearish
            
            # Combine for final bias score
            bias_score = (directional_bias + supertrend_bias + rsi_bias) / 3
            
            # Calculate condition strengths
            if bias_score > 30 and trend_strength > 50:  # Strong bullish
                bullish_strength = trend_strength
                
                # Determine if extreme bullish
                if bias_score > 60 and trend_strength > 70 and adx > self.adx_threshold * 1.5:
                    bullish_strength += 30  # Extra boost for extreme bullish
            
            if bias_score < -30 and trend_strength > 50:  # Strong bearish
                bearish_strength = trend_strength
                
                # Determine if extreme bearish
                if bias_score < -60 and trend_strength > 70 and adx > self.adx_threshold * 1.5:
                    bearish_strength += 30  # Extra boost for extreme bearish
            
            if trend_strength < 40:  # Low trend strength = sideways
                sideways_strength = 100 - trend_strength
                
            if is_squeeze:
                squeeze_strength = 100 if bb_width < self.squeeze_threshold * 0.7 else 70
            
            # Prevent rapid condition changes by requiring larger threshold to change states
            new_condition = None
            
            # Determine the new condition based on highest strength
            if max(bullish_strength, bearish_strength, sideways_strength, squeeze_strength) == bullish_strength:
                if bullish_strength > 80:
                    new_condition = 'EXTREME_BULLISH'
                else:
                    new_condition = 'BULLISH'
            elif max(bullish_strength, bearish_strength, sideways_strength, squeeze_strength) == bearish_strength:
                if bearish_strength > 80:
                    new_condition = 'EXTREME_BEARISH'
                else:
                    new_condition = 'BEARISH'
            elif max(bullish_strength, bearish_strength, sideways_strength, squeeze_strength) == squeeze_strength:
                new_condition = 'SQUEEZE'
            else:
                new_condition = 'SIDEWAYS'
            
            # Apply hysteresis to avoid rapid condition changes
            # Only change condition if new one is persistent or very strong
            if i > 0 and current_condition:
                previous_condition = conditions[i-1]
                
                # Keep current condition unless we have a strong signal to change
                # This prevents whipsaws between market states
                if previous_condition != new_condition:
                    # For a change to occur, the new condition needs to be significantly stronger
                    change_threshold = 20 if condition_streak < 3 else 10
                    
                    max_current_strength = 0
                    if previous_condition == 'BULLISH' or previous_condition == 'EXTREME_BULLISH':
                        max_current_strength = bullish_strength
                    elif previous_condition == 'BEARISH' or previous_condition == 'EXTREME_BEARISH':
                        max_current_strength = bearish_strength
                    elif previous_condition == 'SQUEEZE':
                        max_current_strength = squeeze_strength
                    else:  # SIDEWAYS
                        max_current_strength = sideways_strength
                    
                    max_new_strength = 0
                    if new_condition == 'BULLISH' or new_condition == 'EXTREME_BULLISH':
                        max_new_strength = bullish_strength
                    elif new_condition == 'BEARISH' or new_condition == 'EXTREME_BEARISH':
                        max_new_strength = bearish_strength
                    elif new_condition == 'SQUEEZE':
                        max_new_strength = squeeze_strength
                    else:  # SIDEWAYS
                        max_new_strength = sideways_strength
                    
                    # Only change if the new condition is significantly stronger
                    if max_new_strength > max_current_strength + change_threshold:
                        conditions.append(new_condition)
                        current_condition = new_condition
                        condition_streak = 1
                    else:
                        conditions.append(previous_condition)
                        current_condition = previous_condition
                        condition_streak += 1
                else:
                    conditions.append(previous_condition)
                    current_condition = previous_condition
                    condition_streak += 1
            else:
                conditions.append(new_condition)
                current_condition = new_condition
                condition_streak = 1
        
        return pd.Series(conditions, index=df.index)
    
    def calculate_dynamic_position_size(self, df, base_position=1.0):
        """
        Calculate dynamic position size based on volatility and market condition
        
        Args:
            df: DataFrame with indicators
            base_position: Base position size (1.0 = 100% of allowed position)
            
        Returns:
            float: Position size multiplier (e.g., 0.8 means 80% of base position)
        """
        try:
            if df.empty or len(df) < 20:
                logger.warning("Insufficient data for dynamic position sizing, using default")
                return base_position
                
            latest = df.iloc[-1]
            
            # Get ATR as volatility measure
            if 'atr_pct' not in df.columns:
                logger.warning("ATR percentage not found in dataframe, using default position size")
                return base_position
                
            atr_pct = latest['atr_pct']
            avg_atr_pct = df['atr_pct'].tail(20).mean()
            
            # Base position sizing on volatility relative to average
            if atr_pct > avg_atr_pct * 1.5:
                # High volatility - reduce position size
                volatility_factor = 0.7
            elif atr_pct < avg_atr_pct * 0.7:
                # Low volatility - increase position size slightly
                volatility_factor = 1.1
            else:
                # Normal volatility
                volatility_factor = 1.0
                
            # Adjust based on market condition
            market_condition = latest.get('market_condition', 'UNKNOWN')
            if market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
                # Extreme trend - reduce position size for safety
                condition_factor = 0.75
            elif market_condition in ['BULLISH', 'BEARISH']:
                # Clear trend - standard position
                condition_factor = 1.0
            elif market_condition == 'SQUEEZE':
                # Squeeze condition - smaller position for breakout
                condition_factor = 0.8
            else:  # SIDEWAYS or UNKNOWN
                # Sideways market - slightly smaller position
                condition_factor = 0.9
            
            # Calculate final position size - reduced base_position from 1.0 to 0.8 for less aggressive sizing
            position_size = base_position * volatility_factor * condition_factor
            
            # Apply additional adjustments based on trend consistency
            if 'trend_consistency' in df.columns:
                trend_consistency = latest.get('trend_consistency', 0.5)
                
                # If trend is very consistent, we can be more confident
                if trend_consistency > 0.8:
                    position_size *= 1.1
                elif trend_consistency < 0.3:
                    position_size *= 0.9
            
            # Clamp position size to reasonable values for real trading safety
            position_size = max(0.05, min(1.0, position_size))  # 5% to 100% maximum
            
            # Additional safety check for real money trading
            if position_size > 0.8:
                logger.warning(f"Large position size calculated: {position_size:.2f}, capping at 0.8 for safety")
                position_size = 0.8
            
            logger.debug(f"Dynamic position size: {position_size:.2f} (volatility: {volatility_factor:.2f}, " +
                         f"market condition: {market_condition}, condition factor: {condition_factor:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating dynamic position size: {e}")
            return base_position  # Return base position on error
            
    def in_cooloff_period(self, current_time):
        """Check if we're in a cool-off period after losses"""
        if self.consecutive_losses >= self.max_consecutive_losses and self.last_loss_time:
            try:
                # Convert from pandas timestamp if needed
                if hasattr(self.last_loss_time, 'to_pydatetime'):
                    last_loss_time = self.last_loss_time.to_pydatetime()
                else:
                    last_loss_time = self.last_loss_time
                    
                # Convert current_time from pandas timestamp if needed
                if hasattr(current_time, 'to_pydatetime'):
                    current_time = current_time.to_pydatetime()
                
                # Check if we're still in the cool-off period
                cooloff_end_time = last_loss_time + timedelta(minutes=self.cooloff_period)
                in_cooloff = current_time < cooloff_end_time
                
                # Log cooloff status
                if in_cooloff:
                    time_remaining = (cooloff_end_time - current_time).total_seconds() / 60
                    logger.info(f"In cool-off period. {time_remaining:.1f} minutes remaining.")
                else:
                    # Reset consecutive losses if cooloff period has ended
                    logger.info(f"Cool-off period ended. Resetting consecutive losses and cache.")
                    self.consecutive_losses = 0
                    self.last_loss_time = None
                    # Reset all cached data to ensure fresh start
                    self.reset_cache()
                
                return in_cooloff
            except Exception as e:
                logger.error(f"Error in cooloff period calculation: {e}")
                # Default to False if there's an error in comparison
                return False
            
        return False
    
    def get_v_reversal_signal(self, df):
        """
        Detect V-shaped reversals in extreme market conditions
        """
        if len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        potential_reversal = latest['potential_reversal']
        
        # Only look for reversals in extreme conditions
        if market_condition not in ['EXTREME_BEARISH', 'EXTREME_BULLISH']:
            return None
            
        # Return reversal signal if detected
        if potential_reversal == 1 and market_condition == 'EXTREME_BEARISH':
            logger.info("V-shaped bullish reversal detected in extreme bearish market")
            return 'BUY'
        elif potential_reversal == -1 and market_condition == 'EXTREME_BULLISH':
            logger.info("V-shaped bearish reversal detected in extreme bullish market")
            return 'SELL'
            
        return None
    
    def get_squeeze_breakout_signal(self, df):
        """
        Detect breakouts from low-volatility squeeze conditions
        """
        if len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        market_condition = latest['market_condition']
        
        # Only look for breakout if we're in or just exited a squeeze
        if market_condition != 'SQUEEZE' and prev['market_condition'] != 'SQUEEZE':
            return None
            
        # Volume spike indicates breakout
        if latest['volume_ratio'] > 1.5:
            # Direction of breakout
            if latest['close'] > latest['bb_upper']:
                return 'BUY'
            elif latest['close'] < latest['bb_lower']:
                return 'SELL'
                
        return None
    
    
    def get_sideways_signal(self, df):
        """Enhanced sideways market signal with VWAP integration"""
        latest = df.iloc[-1]
        
        # In sideways markets, use VWAP as a dynamic anchor point
        
        # Buy near lower Bollinger Band with VWAP confirmation
        if latest['close'] < latest['bb_lower'] * 1.01 and latest['close'] < latest['vwap']:
            return 'BUY'
            
        # Sell near upper Bollinger Band with VWAP confirmation
        elif latest['close'] > latest['bb_upper'] * 0.99 and latest['close'] > latest['vwap']:
            return 'SELL'
            
        # Volume-weighted RSI signals in sideways markets
        elif latest['volume_weighted_rsi'] < 30:
            return 'BUY'
        elif latest['volume_weighted_rsi'] > 70:
            return 'SELL'
            
        return None
    
    def get_bullish_signal(self, df):
        """
        Enhanced signal for bullish market with aggressive trend thresholds
        """
        if len(df) < 3:
            return None
            
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            market_condition = latest['market_condition']
            
            # Adjust RSI thresholds based on market condition
            rsi_oversold = 25 if market_condition == 'EXTREME_BULLISH' else 35
            
            # More aggressive oversold conditions for BUY signals in bullish markets
            if latest['rsi'] < rsi_oversold:
                return 'BUY'
                
            # BUY on MACD crossover with volume confirmation
            if (prev['macd'] < prev['macd_signal'] and 
                latest['macd'] > latest['macd_signal'] and 
                latest['volume_ratio'] > 1.2):
                return 'BUY'
                
            # BUY on Supertrend direction change
            if prev['supertrend_direction'] == -1 and latest['supertrend_direction'] == 1:
                return 'BUY'
                
            # Sell only on extreme overbought conditions in bullish markets
            if (latest['rsi'] > 80 and 
                latest['close'] > latest['bb_upper'] * 1.01 and
                latest['close'] > latest['vwap'] * 1.03):
                return 'SELL'
                
            return None
            
        except Exception as e:
            logger.error(f"Error in get_bullish_signal: {e}")
            return None
    
    def get_bearish_signal(self, df):
        """
        Enhanced signal for bearish market with aggressive trend thresholds
        """
        if len(df) < 3:
            return None
            
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            market_condition = latest['market_condition']
            
            # Adjust RSI thresholds based on market condition
            rsi_overbought = 75 if market_condition == 'EXTREME_BEARISH' else 65
            
            # More aggressive overbought conditions for SELL signals in bearish markets
            if latest['rsi'] > rsi_overbought:
                return 'SELL'
                
            # SELL on MACD crossover with volume confirmation
            if (prev['macd'] > prev['macd_signal'] and 
                latest['macd'] < latest['macd_signal'] and 
                latest['volume_ratio'] > 1.2):
                return 'SELL'
                
            # SELL on Supertrend direction change
            if prev['supertrend_direction'] == 1 and latest['supertrend_direction'] == -1:
                return 'SELL'
                
            # Buy only on extreme oversold conditions in bearish markets
            if (latest['rsi'] < 20 and 
                latest['close'] < latest['bb_lower'] * 0.99 and
                latest['close'] < latest['vwap'] * 0.97):
                return 'BUY'
                
            return None
            
        except Exception as e:
            logger.error(f"Error in get_bearish_signal: {e}")
            return None
            
    def update_trade_result(self, was_profitable):
        """
        Update consecutive losses counter for cool-off period calculation
        
        Args:
            was_profitable: Boolean indicating if the last trade was profitable
        """
        if was_profitable:
            # Reset consecutive losses on profitable trade
            self.consecutive_losses = 0
            self.last_loss_time = None
            logger.info("Profitable trade - reset consecutive losses counter")
        else:
            # Increment consecutive losses
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
            logger.info(f"Loss recorded - consecutive losses: {self.consecutive_losses}")
            
            # Check if we need to enter cool-off period
            if self.consecutive_losses >= self.max_consecutive_losses:
                # Reset cache to ensure we start fresh after cooloff
                self.reset_cache()
                logger.info(f"Entering cool-off period for {self.cooloff_period} minutes. All caches have been reset.")
    
    def get_extreme_market_signal(self, df):
        """
        Specialized signal generation for extreme market conditions
        """
        if len(df) < 3:
            return None
            
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        
        # Only process if we're in extreme market conditions
        if market_condition not in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
            return None
            
        # In extreme bullish market
        if market_condition == 'EXTREME_BULLISH':
            # Look for BUYING opportunities on small dips
            if (latest['close'] < latest['vwap'] and 
                latest['supertrend_direction'] == 1 and
                latest['rsi'] < 40):
                return 'BUY'
                
        # In extreme bearish market
        elif market_condition == 'EXTREME_BEARISH':
            # Look for SELLING opportunities on small rallies
            if (latest['close'] > latest['vwap'] and 
                latest['supertrend_direction'] == -1 and
                latest['rsi'] > 60):
                return 'SELL'
                
        return None
    
    def reset_cache(self):
        """Reset all caches to ensure fresh data after cool-off periods"""
        self._cache = {}
        self._cached_dataframe = None
        self._last_kline_time = None
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        logger.info("All strategy caches have been reset")
    
    def _cleanup_expired_cache(self, current_time):
        """Clean up expired cache entries to prevent memory leaks"""
        try:
            expired_keys = []
            for k, v in list(self._cache.items()):
                cache_time = v.get('time', 0)
                if current_time - cache_time > self._cache_expiry:
                    expired_keys.append(k)
            
            for k in expired_keys:
                del self._cache[k]
                
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
    
    def calculate_market_sentiment(self, df):
        """Advanced market sentiment calculation with multiple indicators"""
        try:
            # Price momentum component
            price_momentum = df['close'].pct_change(5).rolling(10).mean()
            
            # Volume sentiment
            volume_avg = df['volume'].rolling(20).mean()
            volume_sentiment = ((df['volume'] / volume_avg.where(volume_avg > 0, 1)) - 1).clip(-1, 1)
            
            # RSI sentiment (normalized to -1 to 1)
            rsi_sentiment = (df['rsi'] - 50) / 50
            
            # MACD sentiment
            macd_sentiment = np.where(df['macd'] > df['macd_signal'], 1, -1)
            
            # Bollinger Bands position sentiment
            bb_position = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).where((df['bb_upper'] - df['bb_lower']) > 0, 1)
            bb_sentiment = (bb_position - 0.5) * 2  # Normalize to -1 to 1
            
            # Supertrend sentiment
            supertrend_sentiment = np.where(df['supertrend_direction'] == 1, 1, -1)
            
            # Combine all sentiments with weights
            sentiment = (
                price_momentum * 0.25 +
                volume_sentiment * 0.15 +
                rsi_sentiment * 0.2 +
                macd_sentiment * 0.15 +
                bb_sentiment * 0.15 +
                supertrend_sentiment * 0.1
            ).fillna(0).clip(-1, 1)
            
            return sentiment
            
        except Exception as e:
            logger.warning(f"Market sentiment calculation failed: {e}")
            return pd.Series([0] * len(df), index=df.index)

    def detect_flash_crash(self, df, threshold=-0.05, volume_threshold=2.0):
        """Detect flash crash events (sudden large drops with high volume)"""
        try:
            # Calculate price drops
            returns = df['close'].pct_change(1)
            
            # Check for volume spikes
            volume_ratio = df['volume_ratio'].fillna(1)
            
            # Flash crash conditions
            flash_crashes = (
                (returns < threshold) &  # Large price drop
                (volume_ratio > volume_threshold)  # High volume
            )
            
            return flash_crashes
            
        except Exception as e:
            logger.warning(f"Flash crash detection failed: {e}")
            return pd.Series([False] * len(df), index=df.index)

    def detect_pump_events(self, df, threshold=0.05, volume_threshold=2.0):
        """Detect pump events (sudden large gains with high volume)"""
        try:
            # Calculate price gains
            returns = df['close'].pct_change(1)
            
            # Check for volume spikes
            volume_ratio = df['volume_ratio'].fillna(1)
            
            # Pump conditions
            pump_events = (
                (returns > threshold) &  # Large price gain
                (volume_ratio > volume_threshold)  # High volume
            )
            
            return pump_events
            
        except Exception as e:
            logger.warning(f"Pump event detection failed: {e}")
            return pd.Series([False] * len(df), index=df.index)

    def detect_market_phase(self, df):
        """Detect current market phase (accumulation, markup, distribution, markdown)"""
        try:
            if len(df) < 50:
                return 'UNKNOWN', 0.5
            
            # Price trend analysis
            if df['close'].iloc[-20] != 0:
                price_trend = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
            else:
                price_trend = 0
            
            # Volume analysis
            recent_volume = df['volume'].iloc[-10:].mean()
            historical_volume = df['volume'].iloc[-50:-10].mean()
            volume_trend = recent_volume / historical_volume if historical_volume > 0 else 1
            
            # Volatility analysis
            recent_volatility = df['atr_pct'].iloc[-10:].mean()
            historical_volatility = df['atr_pct'].iloc[-50:-10].mean()
            volatility_ratio = recent_volatility / historical_volatility if historical_volatility > 0 else 1
            
            # Phase detection logic
            if price_trend > 0.02 and volume_trend > 1.2:  # Rising prices with increasing volume
                if volatility_ratio < 1.2:  # Controlled volatility
                    return 'MARKUP', 0.8  # Strong uptrend
                else:
                    return 'DISTRIBUTION', 0.7  # Possible top formation
            elif price_trend < -0.02 and volume_trend > 1.2:  # Falling prices with increasing volume
                return 'MARKDOWN', 0.8  # Strong downtrend
            elif abs(price_trend) < 0.02 and volume_trend < 0.8:  # Sideways with low volume
                return 'ACCUMULATION', 0.6  # Potential bottom formation
            else:
                return 'TRANSITION', 0.5  # Unclear phase
            
        except Exception as e:
            logger.warning(f"Market phase detection failed: {e}")
            return 'UNKNOWN', 0.5

    def calculate_position_score(self, df):
        """Calculate a comprehensive position score for signal prioritization"""
        try:
            latest = df.iloc[-1]
            score = 0.0
            
            # Trend alignment score (30% weight)
            if latest.get('supertrend_direction', 0) == 1:  # Uptrend
                score += 0.3
            elif latest.get('supertrend_direction', 0) == -1:  # Downtrend
                score -= 0.3
            
            # RSI score (20% weight)
            rsi = latest.get('rsi', 50)
            if 30 <= rsi <= 70:  # Healthy RSI range
                score += 0.2
            elif rsi < 30:  # Oversold
                score += 0.1
            elif rsi > 70:  # Overbought
                score -= 0.1
            
            # Volume confirmation score (20% weight)
            volume_ratio = latest.get('volume_ratio', 1)
            if volume_ratio > 1.5:  # Strong volume
                score += 0.2
            elif volume_ratio > 1.0:  # Above average volume
                score += 0.1
            
            # Market sentiment score (15% weight)
            try:
                sentiment = self.calculate_market_sentiment(df).iloc[-1]
                score += sentiment * 0.15
            except Exception:
                pass
            
            # Volatility score (15% weight)
            atr_pct = latest.get('atr_pct', 2.0)
            if 1.0 <= atr_pct <= 3.0:  # Healthy volatility
                score += 0.15
            elif atr_pct > 5.0:  # Too volatile
                score -= 0.15
            
            return max(-1.0, min(1.0, score))  # Clamp between -1 and 1
            
        except Exception as e:
            logger.warning(f"Position score calculation failed: {e}")
            return 0.0

    def get_strategy_insights(self, df):
        """Provide comprehensive strategy insights and recommendations"""
        try:
            insights = {
                'market_phase': 'UNKNOWN',
                'market_sentiment': 0.0,
                'trend_strength': 'WEAK',
                'volatility_level': 'NORMAL',
                'volume_analysis': 'NORMAL',
                'risk_level': 'MEDIUM',
                'recommended_action': 'HOLD',
                'confidence_score': 0.5,
                'key_levels': {
                    'support': [],
                    'resistance': [],
                    'fibonacci': []
                },
                'warnings': [],
                'opportunities': []
            }
            
            if len(df) < 20:
                insights['warnings'].append("Insufficient data for comprehensive analysis")
                return insights
            
            latest = df.iloc[-1]
            
            # Market phase analysis
            phase, phase_confidence = self.detect_market_phase(df)
            insights['market_phase'] = phase
            
            # Market sentiment
            sentiment = self.calculate_market_sentiment(df).iloc[-1]
            insights['market_sentiment'] = float(sentiment)
            
            # Trend strength analysis
            adx = latest.get('adx', 20)
            if adx > 40:
                insights['trend_strength'] = 'VERY_STRONG'
            elif adx > 25:
                insights['trend_strength'] = 'STRONG'
            elif adx > 15:
                insights['trend_strength'] = 'MODERATE'
            else:
                insights['trend_strength'] = 'WEAK'
            
            # Volatility analysis
            atr_pct = latest.get('atr_pct', 2.0)
            if atr_pct > 5.0:
                insights['volatility_level'] = 'VERY_HIGH'
                insights['warnings'].append("Extremely high volatility detected")
            elif atr_pct > 3.0:
                insights['volatility_level'] = 'HIGH'
            elif atr_pct < 1.0:
                insights['volatility_level'] = 'LOW'
            
            # Volume analysis
            volume_ratio = latest.get('volume_ratio', 1.0)
            if volume_ratio > 2.0:
                insights['volume_analysis'] = 'VERY_HIGH'
                insights['opportunities'].append("High volume activity - potential breakout")
            elif volume_ratio > 1.5:
                insights['volume_analysis'] = 'HIGH'
            elif volume_ratio < 0.7:
                insights['volume_analysis'] = 'LOW'
                insights['warnings'].append("Low volume - weak price action")
            
            # Risk level assessment
            risk_factors = 0
            if atr_pct > 4.0:
                risk_factors += 1
            if abs(sentiment) > 0.8:
                risk_factors += 1
            if adx < 15:  # Weak trend
                risk_factors += 1
            
            if risk_factors >= 2:
                insights['risk_level'] = 'HIGH'
            elif risk_factors == 1:
                insights['risk_level'] = 'MEDIUM'
            else:
                insights['risk_level'] = 'LOW'
            
            # Generate recommendation
            position_score = self.calculate_position_score(df)
            if position_score > 0.3:
                insights['recommended_action'] = 'BUY'
                insights['confidence_score'] = min(0.9, 0.5 + position_score)
            elif position_score < -0.3:
                insights['recommended_action'] = 'SELL'
                insights['confidence_score'] = min(0.9, 0.5 + abs(position_score))
            else:
                insights['recommended_action'] = 'HOLD'
                insights['confidence_score'] = 0.4
            
            # Add key levels
            if self.fib_support_levels:
                insights['key_levels']['support'] = self.fib_support_levels[-3:]  # Last 3 levels
            if self.fib_resistance_levels:
                insights['key_levels']['resistance'] = self.fib_resistance_levels[-3:]
            
            # Detect specific opportunities
            if latest.get('bb_squeeze', False):
                insights['opportunities'].append("Bollinger Band squeeze - potential volatility expansion")
            
            if latest.get('potential_reversal', False):
                insights['opportunities'].append("Potential reversal pattern detected")
            
            # Flash crash or pump detection
            if self.detect_flash_crash(df.tail(5)).any():
                insights['warnings'].append("Recent flash crash detected - exercise caution")
            
            if self.detect_pump_events(df.tail(5)).any():
                insights['warnings'].append("Recent pump event detected - potential correction ahead")
            
            return insights
            
        except Exception as e:
            logger.error(f"Strategy insights generation failed: {e}")
            return {
                'market_phase': 'ERROR',
                'error': str(e),
                'confidence_score': 0.0
            }

# Update the factory function to include only RAYSOL strategy
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    from modules.config import (
        # RAYSOL parameters
        RAYSOL_TREND_EMA_FAST, RAYSOL_TREND_EMA_SLOW,
        RAYSOL_VOLATILITY_LOOKBACK, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
        RAYSOL_VOLUME_MA_PERIOD, RAYSOL_ADX_PERIOD, RAYSOL_ADX_THRESHOLD, RAYSOL_SIDEWAYS_THRESHOLD,
        RAYSOL_VOLATILITY_MULTIPLIER, RAYSOL_TREND_CONDITION_MULTIPLIER
    )
    
    strategies = {
        'DynamicStrategy': DynamicStrategy(
            trend_ema_fast=RAYSOL_TREND_EMA_FAST,
            trend_ema_slow=RAYSOL_TREND_EMA_SLOW,
            volatility_lookback=RAYSOL_VOLATILITY_LOOKBACK,
            rsi_period=RSI_PERIOD,
            rsi_overbought=RSI_OVERBOUGHT,
            rsi_oversold=RSI_OVERSOLD,
            volume_ma_period=RAYSOL_VOLUME_MA_PERIOD,
            adx_period=RAYSOL_ADX_PERIOD,
            adx_threshold=RAYSOL_ADX_THRESHOLD,
            sideways_threshold=RAYSOL_SIDEWAYS_THRESHOLD,
            # Pass RAYSOL specific parameters
            volatility_multiplier=RAYSOL_VOLATILITY_MULTIPLIER,
            trend_condition_multiplier=RAYSOL_TREND_CONDITION_MULTIPLIER
        )
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]
    
    logger.warning(f"Strategy {strategy_name} not found. Defaulting to base trading strategy.")
    return TradingStrategy(strategy_name)


def get_strategy_for_symbol(symbol, strategy_name=None):
    """Get the appropriate strategy based on the trading symbol"""
    # If a specific strategy is requested, use it
    if strategy_name:
        return get_strategy(strategy_name)
    
    # Default to SUIUSDT strategy for any symbol
    return DynamicStrategy()
    
    # Default to base strategy if needed
    # return TradingStrategy(symbol)