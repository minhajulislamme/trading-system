from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import ta
import ta.momentum
import ta.trend
import ta.volatility
import logging
import warnings

# Setup logging
logger = logging.getLogger(__name__)
warnings.simplefilter(action='ignore', category=FutureWarning)


class TradingStrategy:
    """Base trading strategy class"""
    
    def __init__(self, name="BaseStrategy"):
        self.name = name
        self.risk_manager = None
    
    @property
    def strategy_name(self):
        """Property to access strategy name (for compatibility)"""
        return self.name
        
    def set_risk_manager(self, risk_manager):
        """Set the risk manager for this strategy"""
        self.risk_manager = risk_manager
        
    def get_signal(self, klines):
        """Get trading signal from klines data. Override in subclasses."""
        return None
        
        
    def add_indicators(self, df):
        """Add technical indicators to dataframe. Override in subclasses."""
        return df


class SmartTrendCatcher(TradingStrategy):
    """
    EMA Crossover Strategy with Enhanced Signal Filtering (9/26 EMA):
    
    Core Strategy:
    - EMA crossover as primary signal generation (9 EMA vs 26 EMA)
    - Fast EMA (9) crosses above/below slow EMA (26) for entries
    - Enhanced volume and volatility filtering
    - MACD confirmation for better signal quality
    
    Signal Generation:
    - BUY: 9 EMA crosses above 26 EMA with confirmations
    - SELL: 9 EMA crosses below 26 EMA with confirmations
    - Additional filters to reduce false signals
    """
    
    def __init__(self, 
                 # EMA crossover parameters based on chart analysis
                 ema_slow=26,               # Slow EMA matching MACD slow (26 period)
                 ema_fast=9,                # Fast EMA for crossover (9 period)
                 
                 # MACD parameters matching the chart configuration
                 macd_fast=12,
                 macd_slow=26,
                 macd_signal=9,
                 macd_histogram_threshold=0.00001,  # Very sensitive threshold for MACD histogram
                 
                 # Enhanced false signal reduction (optimized for chart patterns)
                 volume_filter_enabled=True,
                 volume_period=20,
                 volume_multiplier=1.2,     # More lenient for trend continuation
                 volume_surge_multiplier=1.6,  # Lower threshold for volume surges
                 
                 atr_filter_enabled=True,
                 atr_period=14,
                 atr_threshold=0.5,         # Lower volatility requirement for more signals
                 atr_trend_factor=1.0,      # More responsive ATR trend confirmation
                 
                 # Bollinger Bands for volatility confirmation
                 bb_period=20,
                 bb_std=2.0,
                 bb_squeeze_threshold=0.08,  # Slightly lower for more breakout signals
                 
                 # Price action filters (more lenient for trending markets)
                 min_candle_body_pct=0.4,   # Lower minimum candle body size
                 max_wick_ratio=3.5,        # Allow slightly more wick ratio
                 
                 # Advanced filtering parameters
                 confluence_required=2,      # Number of confirmations needed
                 
                 # Dynamic position sizing (more aggressive for trending)
                 base_position_pct=0.35,     # Slightly larger base position
                 max_position_pct=0.6,      # Higher maximum position size
                 confidence_multiplier=1.4): # Confidence multiplier
        
        super().__init__("SmartTrendCatcher")
        
        # Enhanced parameter validation
        if ema_slow <= 0 or ema_fast <= 0:
            raise ValueError("EMA periods must be positive")
        if ema_fast >= ema_slow:
            raise ValueError("Fast EMA must be less than slow EMA")
        if macd_fast <= 0 or macd_slow <= 0 or macd_signal <= 0 or macd_fast >= macd_slow:
            raise ValueError("Invalid MACD parameters")
        if not (0 < base_position_pct <= max_position_pct <= 1.0):
            raise ValueError("Invalid position sizing parameters")
        if confluence_required < 0:
            raise ValueError("Confluence required must be at least 0")
        
        # Store enhanced parameters
        self.ema_slow = ema_slow
        self.ema_fast = ema_fast
        
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.macd_histogram_threshold = macd_histogram_threshold
        
        # Enhanced false signal reduction
        self.volume_filter_enabled = volume_filter_enabled
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        self.volume_surge_multiplier = volume_surge_multiplier
        
        self.atr_filter_enabled = atr_filter_enabled
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        self.atr_trend_factor = atr_trend_factor
        
        # Bollinger Bands parameters
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.bb_squeeze_threshold = bb_squeeze_threshold
        
        # Price action parameters
        self.min_candle_body_pct = min_candle_body_pct
        self.max_wick_ratio = max_wick_ratio
        
        # Advanced filtering
        self.confluence_required = confluence_required
        
        # Dynamic position sizing
        self.base_position_pct = base_position_pct
        self.max_position_pct = max_position_pct
        self.confidence_multiplier = confidence_multiplier
        
        # Trading state tracking
        self._last_confidence = confluence_required
        self._warning_count = 0
        
        logger.info(f"Enhanced {self.name} initialized with:")
        logger.info(f"  EMA Crossover: {ema_fast}/{ema_slow}")
        logger.info(f"  Confluence required: {confluence_required}")
        logger.info(f"  Position sizing: {base_position_pct:.1%}-{max_position_pct:.1%}")
    
    def validate_data_quality(self, df):
        """Comprehensive data quality validation"""
        issues = []
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                issues.append(f"Missing required column: {col}")
            elif df[col].isna().all():
                issues.append(f"Column {col} contains only NaN values")
            elif (df[col] <= 0).any() and col != 'volume':  # Volume can be zero
                issues.append(f"Invalid values in {col} (zero or negative prices)")
        
        # Check OHLC relationships
        if len(issues) == 0:  # Only check if basic data is valid
            high_low_issues = (df['high'] < df['low']).any()
            close_range_issues = ((df['close'] > df['high']) | (df['close'] < df['low'])).any()
            open_range_issues = ((df['open'] > df['high']) | (df['open'] < df['low'])).any()
            
            if high_low_issues:
                issues.append("High prices lower than low prices detected")
            if close_range_issues:
                issues.append("Close prices outside high-low range detected")
            if open_range_issues:
                issues.append("Open prices outside high-low range detected")
        
        return issues
    
    def add_indicators(self, df):
        """Add enhanced indicators with multi-layer filtering"""
        try:
            # Ensure sufficient data
            min_required = max(self.ema_slow, self.macd_slow, self.volume_period, 
                             self.atr_period, self.bb_period) + 20
            if len(df) < min_required:
                logger.warning(f"Insufficient data: need {min_required}, got {len(df)}")
                return df
            
            # Comprehensive data quality validation
            data_issues = self.validate_data_quality(df)
            if data_issues:
                logger.error(f"Data quality issues found: {', '.join(data_issues)}")
                return df
                    
            # Additional data cleaning with improved approach
            if df['close'].isna().any():
                logger.warning("Found NaN values in close prices, cleaning data")
                df['close'] = df['close'].interpolate(method='linear').bfill().ffill()
                
            # Check for zero or negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if (df[col] <= 0).any():
                    logger.warning(f"Found zero or negative values in {col}, using interpolation")
                    # Use interpolation instead of forward fill to avoid look-ahead bias
                    df[col] = df[col].replace(0, np.nan)
                    # Try linear interpolation first, then backward fill, then forward fill as last resort
                    df[col] = df[col].interpolate(method='linear').bfill().ffill()
                    
            # Ensure high >= low, close between high and low
            df['high'] = np.maximum(df['high'], np.maximum(df['open'], df['close']))
            df['low'] = np.minimum(df['low'], np.minimum(df['open'], df['close']))
            
            # 1. EMA Crossover Analysis (Primary Signal)
            df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=self.ema_slow)
            df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=self.ema_fast)
            # Use interpolation for better data integrity
            df['ema_slow'] = df['ema_slow'].interpolate(method='linear').bfill()
            df['ema_fast'] = df['ema_fast'].interpolate(method='linear').bfill()
            
            # EMA crossover signals
            df['ema_fast_above'] = df['ema_fast'] > df['ema_slow']
            df['ema_fast_below'] = df['ema_fast'] < df['ema_slow']
            
            # Crossover detection (primary signals)
            df['ema_bullish_cross'] = (df['ema_fast'] > df['ema_slow']) & (df['ema_fast'].shift(1) <= df['ema_slow'].shift(1))
            df['ema_bearish_cross'] = (df['ema_fast'] < df['ema_slow']) & (df['ema_fast'].shift(1) >= df['ema_slow'].shift(1))
            
            # EMA spread for signal strength (safe division)
            df['ema_spread'] = np.where(
                df['ema_slow'] > 0, 
                np.abs(df['ema_fast'] - df['ema_slow']) / df['ema_slow'],
                0
            )
            df['strong_crossover'] = df['ema_spread'] > 0.002  # At least 0.2% spread
            
            # 2. Enhanced MACD Analysis with better error handling
            try:
                macd = ta.trend.MACD(df['close'], window_slow=self.macd_slow, 
                                   window_fast=self.macd_fast, window_sign=self.macd_signal)
                df['macd'] = macd.macd()
                df['macd_signal_line'] = macd.macd_signal()
                df['macd_histogram'] = df['macd'] - df['macd_signal_line']
                
                # Handle NaN values in MACD indicators with better approach
                macd_cols = ['macd', 'macd_signal_line', 'macd_histogram']
                for col in macd_cols:
                    if df[col].isna().any():
                        # Use interpolation first
                        df[col] = df[col].interpolate(method='linear')
                        # For remaining NaN values, use small values instead of zeros
                        if df[col].isna().any():
                            # Use a small fraction of price volatility as fallback
                            price_std = df['close'].rolling(window=10, min_periods=1).std()
                            fallback_value = price_std * 0.001  # Small but realistic value
                            df[col] = df[col].fillna(fallback_value)
                            
            except Exception as e:
                logger.error(f"Error calculating MACD: {e}, using price volatility proxy")
                # Fallback to price volatility-based MACD proxy
                price_std = df['close'].rolling(window=10, min_periods=1).std()
                df['macd'] = price_std * 0.001
                df['macd_signal_line'] = price_std * 0.0005
                df['macd_histogram'] = df['macd'] - df['macd_signal_line']
            
            # MACD conditions with threshold
            df['macd_hist_positive'] = df['macd_histogram'] > self.macd_histogram_threshold
            df['macd_hist_negative'] = df['macd_histogram'] < -self.macd_histogram_threshold
            df['macd_hist_increasing'] = df['macd_histogram'] > df['macd_histogram'].shift(1)
            df['macd_hist_decreasing'] = df['macd_histogram'] < df['macd_histogram'].shift(1)
            df['macd_strong_momentum'] = np.abs(df['macd_histogram']) > self.macd_histogram_threshold * 3
            
            # 4. Enhanced Volume Analysis
            if self.volume_filter_enabled:
                df['volume_sma'] = df['volume'].rolling(window=self.volume_period).mean()
                df['volume_sma'] = df['volume_sma'].interpolate(method='linear').bfill()
                
                # Multiple volume conditions
                df['volume_above_avg'] = df['volume'] > (df['volume_sma'] * self.volume_multiplier)
                df['volume_surge'] = df['volume'] > (df['volume_sma'] * self.volume_surge_multiplier)
                df['volume_increasing'] = df['volume'] > df['volume'].shift(1)
                df['volume_momentum'] = df['volume'].rolling(3).mean() > df['volume_sma']
                
                # Volume confirmation score (optimized)
                df['volume_score'] = (
                    df['volume_above_avg'].astype(int) +
                    df['volume_surge'].astype(int) +
                    df['volume_increasing'].astype(int) +
                    df['volume_momentum'].astype(int)
                )
                
                # Clean up intermediate columns to save memory
                df.drop(['volume_above_avg', 'volume_surge', 'volume_increasing', 'volume_momentum'], 
                       axis=1, inplace=True)
            else:
                df['volume_score'] = 4  # Max score if disabled
            
            # 5. Enhanced Volatility Analysis
            if self.atr_filter_enabled:
                df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 
                                                           window=self.atr_period)
                df['atr'] = df['atr'].interpolate(method='linear').bfill()
                # Safe percentage calculation
                df['atr_pct'] = np.where(df['close'] > 0, (df['atr'] / df['close']) * 100, 0)
                
                # ATR conditions
                df['sufficient_volatility'] = df['atr_pct'] > self.atr_threshold
                df['atr_increasing'] = df['atr'] > df['atr'].shift(1)
                df['atr_trend'] = df['atr'].rolling(5).mean() > df['atr'].rolling(20).mean()
                
                # Volatility score (optimized)
                df['volatility_score'] = (
                    df['sufficient_volatility'].astype(int) +
                    df['atr_increasing'].astype(int) +
                    df['atr_trend'].astype(int)
                )
                
                # Clean up intermediate columns to save memory
                df.drop(['sufficient_volatility', 'atr_increasing', 'atr_trend'], 
                       axis=1, inplace=True)
            else:
                df['volatility_score'] = 3  # Max score if disabled
            
            # 6. Enhanced Bollinger Bands for Squeeze Detection
            try:
                # Ensure we have enough data for Bollinger Bands calculation
                if len(df) >= self.bb_period:
                    bb = ta.volatility.BollingerBands(df['close'], window=self.bb_period, window_dev=self.bb_std)
                    df['bb_upper'] = bb.bollinger_hband()
                    df['bb_lower'] = bb.bollinger_lband()
                    df['bb_middle'] = bb.bollinger_mavg()
                    
                    # Handle NaN values with better interpolation strategy
                    bb_cols = ['bb_upper', 'bb_lower', 'bb_middle']
                    for col in bb_cols:
                        if df[col].isna().any():
                            # Use interpolation first for better data continuity
                            df[col] = df[col].interpolate(method='linear')
                            # For remaining NaN values, use intelligent price-based fallbacks
                            if df[col].isna().any():
                                if col == 'bb_upper':
                                    # Use recent volatility for better upper band estimation
                                    recent_vol = df['close'].rolling(5, min_periods=1).std()
                                    df[col] = df[col].fillna(df['close'] + (recent_vol * 2))
                                elif col == 'bb_lower':
                                    # Use recent volatility for better lower band estimation
                                    recent_vol = df['close'].rolling(5, min_periods=1).std()
                                    df[col] = df[col].fillna(df['close'] - (recent_vol * 2))
                                elif col == 'bb_middle':
                                    # Use moving average for middle band
                                    df[col] = df[col].fillna(df['close'].rolling(self.bb_period, min_periods=1).mean())
                    
                    # Calculate bb_width with safe division
                    df['bb_width'] = np.where(
                        df['bb_middle'] > 0,
                        (df['bb_upper'] - df['bb_lower']) / df['bb_middle'],
                        0.1  # Default width if division would be invalid
                    )
                    
                else:
                    # Not enough data - use intelligent price-based fallbacks
                    logger.warning(f"Insufficient data for Bollinger Bands ({len(df)} < {self.bb_period}), using dynamic fallbacks")
                    recent_vol = df['close'].rolling(min(len(df), 5), min_periods=1).std()
                    df['bb_upper'] = df['close'] + (recent_vol * 2)
                    df['bb_lower'] = df['close'] - (recent_vol * 2)
                    df['bb_middle'] = df['close'].rolling(min(len(df), self.bb_period), min_periods=1).mean()
                    df['bb_width'] = np.where(
                        df['bb_middle'] > 0,
                        (df['bb_upper'] - df['bb_lower']) / df['bb_middle'],
                        0.04  # 4% default width
                    )
                    
            except Exception as e:
                logger.error(f"Error calculating Bollinger Bands: {e}, using dynamic fallbacks")
                recent_vol = df['close'].rolling(min(len(df), 5), min_periods=1).std()
                df['bb_upper'] = df['close'] + (recent_vol * 2)
                df['bb_lower'] = df['close'] - (recent_vol * 2)
                df['bb_middle'] = df['close'].rolling(min(len(df), self.bb_period), min_periods=1).mean()
                df['bb_width'] = np.where(
                    df['bb_middle'] > 0,
                    (df['bb_upper'] - df['bb_lower']) / df['bb_middle'],
                    0.04  # 4% default width
                )
            
            # Bollinger Band conditions (safe division)
            df['bb_squeeze'] = df['bb_width'] < self.bb_squeeze_threshold
            df['bb_expansion'] = df['bb_width'] > df['bb_width'].shift(1)
            df['bb_breakout_up'] = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
            df['bb_breakout_down'] = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
            
            # 7. Price Action Analysis with improved safety
            df['candle_body'] = np.abs(df['close'] - df['open'])
            df['candle_range'] = df['high'] - df['low']
            # Safe division for body percentage
            df['body_pct'] = np.where(
                df['candle_range'] > 0,
                df['candle_body'] / df['candle_range'],
                0.5  # Default to 50% if range is zero
            )
            df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
            
            # Price action conditions with better handling
            df['strong_candle'] = df['body_pct'] > self.min_candle_body_pct
            # Improved wick ratio calculation
            df['reasonable_wicks'] = np.where(
                df['candle_body'] > df['candle_body'].quantile(0.1),  # Not in bottom 10% of candle sizes
                (df['upper_wick'] / np.maximum(df['candle_body'], df['candle_body'].quantile(0.1)) < self.max_wick_ratio) & \
                (df['lower_wick'] / np.maximum(df['candle_body'], df['candle_body'].quantile(0.1)) < self.max_wick_ratio),
                True  # Consider very small candles as having reasonable wicks
            )
            df['bullish_candle'] = df['close'] > df['open']
            df['bearish_candle'] = df['close'] < df['open']
            
            # 8. EMA Crossover Signal Generation
            df['buy_signal'] = False
            df['sell_signal'] = False
            
            # Primary EMA crossover signals
            df.loc[df['ema_bullish_cross'], 'buy_signal'] = True
            df.loc[df['ema_bearish_cross'], 'sell_signal'] = True
            
            # Additional confirmations for signal quality
            df['buy_confirmation'] = 0
            df['sell_confirmation'] = 0
            
            # Add confirmations for buy signals
            df.loc[df['macd_hist_positive'], 'buy_confirmation'] += 1
            df.loc[df['volume_score'] >= 2, 'buy_confirmation'] += 1
            df.loc[df['volatility_score'] >= 2, 'buy_confirmation'] += 1
            df.loc[df['strong_candle'] & df['bullish_candle'], 'buy_confirmation'] += 1
            
            # Add confirmations for sell signals
            df.loc[df['macd_hist_negative'], 'sell_confirmation'] += 1
            df.loc[df['volume_score'] >= 2, 'sell_confirmation'] += 1
            df.loc[df['volatility_score'] >= 2, 'sell_confirmation'] += 1
            df.loc[df['strong_candle'] & df['bearish_candle'], 'sell_confirmation'] += 1
            
            # Final signals with minimum confirmations
            df['confirmed_buy'] = df['buy_signal'] & (df['buy_confirmation'] >= self.confluence_required)
            df['confirmed_sell'] = df['sell_signal'] & (df['sell_confirmation'] >= self.confluence_required)
            
            # Memory optimization - clean up intermediate columns we don't need for final signals
            columns_to_drop = [
                'macd_hist_positive', 'macd_hist_negative',
                'macd_hist_increasing', 'macd_hist_decreasing', 'macd_strong_momentum',
                'bb_squeeze', 'bb_expansion', 'bb_breakout_up', 'bb_breakout_down',
                'candle_body', 'candle_range', 'body_pct', 'upper_wick', 'lower_wick',
                'strong_candle', 'reasonable_wicks', 'bullish_candle', 'bearish_candle'
            ]
            # Only drop columns that exist to avoid KeyError
            existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
            if existing_columns_to_drop:
                df.drop(existing_columns_to_drop, axis=1, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding enhanced indicators: {e}")
            return df
    
    def get_signal(self, klines):
        """Generate EMA crossover signals with confirmations"""
        try:
            min_required = max(self.ema_slow, self.macd_slow, self.volume_period, 
                             self.atr_period, self.bb_period) + 20
            if not klines or len(klines) < min_required:
                # Show warning every 10th time to reduce log spam
                if self._warning_count % 10 == 0:
                    logger.warning(f"Insufficient data for EMA crossover signal generation (need {min_required}, have {len(klines) if klines else 0})")
                self._warning_count += 1
                return None
            
            # Convert and validate data
            df = pd.DataFrame(klines)
            if len(df.columns) != 12:
                logger.error(f"Invalid klines format: expected 12 columns, got {len(df.columns)}")
                return None
                
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                         'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            
            # Data cleaning with improved method
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    logger.warning(f"Cleaning NaN values in {col}")
                    # Use interpolation to avoid look-ahead bias
                    df[col] = df[col].interpolate(method='linear').bfill().ffill()
            
            # Final validation after cleaning
            if df[numeric_columns].isna().any().any():
                logger.error("Failed to clean price data after interpolation")
                return None
            
            # Add enhanced indicators
            df = self.add_indicators(df)
            
            if len(df) < 2:
                return None
            
            current_idx = len(df) - 1
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Validate required columns for EMA crossover
            required_columns = ['confirmed_buy', 'confirmed_sell', 'buy_confirmation', 'sell_confirmation', 
                              'ema_bullish_cross', 'ema_bearish_cross', 'ema_fast', 'ema_slow']
            
            for col in required_columns:
                if col not in df.columns or pd.isna(latest[col]):
                    logger.warning(f"Missing or invalid EMA crossover indicator: {col}")
                    return None
            
            # EMA Crossover Signal Generation
            signal = None
            
            # BUY Signal: EMA bullish crossover with confirmations
            if latest['confirmed_buy']:
                signal = 'BUY'
                confirmations = int(latest['buy_confirmation'])
                self._last_confidence = confirmations  # Store for position sizing
                
                logger.info(f"🟢 BUY Signal - EMA Crossover Confirmed")
                logger.info(f"   Fast EMA: {latest['ema_fast']:.6f}, Slow EMA: {latest['ema_slow']:.6f}")
                logger.info(f"   MACD: {latest['macd_histogram']:.6f}")
                logger.info(f"   Confirmations: {confirmations}, Volume Score: {latest.get('volume_score', 0)}")
            
            # SELL Signal: EMA bearish crossover with confirmations
            elif latest['confirmed_sell']:
                signal = 'SELL'
                confirmations = int(latest['sell_confirmation'])
                self._last_confidence = confirmations  # Store for position sizing
                
                logger.info(f"🔴 SELL Signal - EMA Crossover Confirmed")
                logger.info(f"   Fast EMA: {latest['ema_fast']:.6f}, Slow EMA: {latest['ema_slow']:.6f}")
                logger.info(f"   MACD: {latest['macd_histogram']:.6f}")
                logger.info(f"   Confirmations: {confirmations}, Volume Score: {latest.get('volume_score', 0)}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in enhanced signal generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_position_size_multiplier(self):
        """Get dynamic position size multiplier based on signal confidence"""
        try:
            confidence = getattr(self, '_last_confidence', self.confluence_required)
            
            # Start with base position percentage as multiplier
            multiplier = self.base_position_pct
            
            # Adjust based on confidence level
            if confidence >= self.confluence_required + 2:  # High confidence
                multiplier = min(multiplier * self.confidence_multiplier, self.max_position_pct)
            elif confidence >= self.confluence_required + 1:  # Medium-high confidence
                multiplier = min(multiplier * ((self.confidence_multiplier + 1) / 2), self.max_position_pct)
            # else: use base multiplier for minimum confidence
            
            # Ensure multiplier is within valid range
            multiplier = max(0.1, min(multiplier, self.max_position_pct))
            
            logger.info(f"Position size multiplier: {multiplier:.2f} (confidence: {confidence})")
            return multiplier
            
        except Exception as e:
            logger.error(f"Error calculating position size multiplier: {e}")
            return self.base_position_pct  # Safe default

# Factory function to get a strategy by name
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    # Import additional config values
    try:
        from modules.config import (
            FAST_EMA, SLOW_EMA, MACD_FAST, MACD_SLOW, MACD_SIGNAL,
            VOLUME_PERIOD, ATR_PERIOD, BB_PERIOD, VOLUME_MULTIPLIER,
            ATR_THRESHOLD, CONFLUENCE_REQUIRED
        )
    except ImportError:
        # Fallback values
        FAST_EMA = 8
        SLOW_EMA = 21
        MACD_FAST = 8
        MACD_SLOW = 17
        MACD_SIGNAL = 6
        VOLUME_PERIOD = 10
        ATR_PERIOD = 7
        BB_PERIOD = 10
        VOLUME_MULTIPLIER = 1.2
        ATR_THRESHOLD = 0.4
        CONFLUENCE_REQUIRED = 2
    
    strategies = {
        'SmartTrendCatcher': SmartTrendCatcher(
            # EMA crossover parameters from config
            ema_slow=SLOW_EMA,
            ema_fast=FAST_EMA,
            # MACD parameters from config
            macd_fast=MACD_FAST,
            macd_slow=MACD_SLOW,
            macd_signal=MACD_SIGNAL,
            # Volume and volatility periods from config
            volume_period=VOLUME_PERIOD,
            atr_period=ATR_PERIOD,
            bb_period=BB_PERIOD,
            # Signal filtering parameters from config
            volume_filter_enabled=True,
            volume_multiplier=VOLUME_MULTIPLIER,
            atr_filter_enabled=True,
            atr_threshold=ATR_THRESHOLD,
            # Confluence requirement from config
            confluence_required=CONFLUENCE_REQUIRED
        ),
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]
    
    logger.warning(f"Strategy {strategy_name} not found. Defaulting to SmartTrendCatcher.")
    return strategies['SmartTrendCatcher']


def get_strategy_for_symbol(symbol, strategy_name=None):
    """Get the appropriate strategy based on the trading symbol"""
    # If a specific strategy is requested, use it
    if strategy_name:
        return get_strategy(strategy_name)
    
    # Default to SmartTrendCatcher for any symbol
    return get_strategy('SmartTrendCatcher')