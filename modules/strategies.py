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
    Smart Trend Catcher Strategy with False Signal Reduction:
    
    Core Strategy:
    - Trend Filter: 50 EMA (Price > 50 EMA = Uptrend)
    - Pullback Detection: RSI levels for opportunity identification
    - Confirmation: MACD Histogram momentum + RSI recovery
    
    False Signal Reduction:
    - Volume Filter: Requires volume to be above average (configurable multiplier)
    - ATR Filter: Requires minimum volatility to avoid choppy sideways markets
    """
    
    def __init__(self, 
                 # Trend filter parameters
                 ema_trend=50,
                 
                 # RSI parameters for pullback detection
                 rsi_period=14,
                 rsi_pullback_low=30,    # Lower bound for uptrend pullback
                 rsi_pullback_high=50,   # Upper bound for uptrend pullback
                 rsi_recovery=50,        # RSI recovery level
                 
                 # MACD parameters for confirmation
                 macd_fast=12,
                 macd_slow=26,
                 macd_signal=9,
                 
                 # False signal reduction parameters
                 volume_filter_enabled=True,
                 volume_period=20,              # Period for volume SMA
                 volume_multiplier=1.2,         # Volume must be X times above average
                 
                 atr_filter_enabled=True,
                 atr_period=14,                 # Period for ATR calculation
                 atr_threshold=0.5):            # Minimum ATR as % of price
        
        super().__init__("SmartTrendCatcher")
        
        # Validate parameters
        if ema_trend <= 0:
            raise ValueError("EMA trend period must be positive")
        if rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        if not (0 <= rsi_pullback_low <= rsi_pullback_high <= rsi_recovery <= 100):
            raise ValueError("Invalid RSI levels: must be 0 <= pullback_low <= pullback_high <= recovery <= 100")
        if macd_fast <= 0 or macd_slow <= 0 or macd_signal <= 0 or macd_fast >= macd_slow:
            raise ValueError("Invalid MACD parameters")
        if volume_period <= 0:
            raise ValueError("Volume period must be positive")
        if volume_multiplier <= 0:
            raise ValueError("Volume multiplier must be positive")
        if atr_period <= 0:
            raise ValueError("ATR period must be positive")
        if not (0 <= atr_threshold <= 100):
            raise ValueError("ATR threshold must be between 0 and 100")
        
        # Store parameters
        self.ema_trend = ema_trend
        
        self.rsi_period = rsi_period
        self.rsi_pullback_low = rsi_pullback_low
        self.rsi_pullback_high = rsi_pullback_high
        self.rsi_recovery = rsi_recovery
        
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        # False signal reduction parameters
        self.volume_filter_enabled = volume_filter_enabled
        self.volume_period = volume_period
        self.volume_multiplier = volume_multiplier
        
        self.atr_filter_enabled = atr_filter_enabled
        self.atr_period = atr_period
        self.atr_threshold = atr_threshold
        
        logger.info(f"Initialized {self.name} with EMA({ema_trend}), RSI({rsi_period}), MACD({macd_fast},{macd_slow},{macd_signal})")
        logger.info(f"Volume Filter: {volume_filter_enabled} (period={volume_period}, multiplier={volume_multiplier})")
        logger.info(f"ATR Filter: {atr_filter_enabled} (period={atr_period}, threshold={atr_threshold}%)")
    
    def add_indicators(self, df):
        """Add EMA, RSI, MACD indicators plus volume and ATR filters for false signal reduction"""
        try:
            # Ensure we have enough data
            min_required = max(self.ema_trend, self.macd_slow, self.volume_period, self.atr_period) + 10
            if len(df) < min_required:
                logger.warning(f"Not enough data for indicators. Need at least {min_required} candles")
                return df
            
            # 1. Trend Filter (50 EMA)
            df['ema_trend'] = ta.trend.ema_indicator(df['close'], window=self.ema_trend)
            df['ema_trend'] = df['ema_trend'].bfill()
            
            # Trend direction
            df['uptrend'] = df['close'] > df['ema_trend']
            df['downtrend'] = df['close'] < df['ema_trend']
            
            # 2. RSI for Pullback Detection
            df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
            df['rsi'] = df['rsi'].bfill()
            
            # RSI conditions for pullback and recovery
            df['rsi_pullback_zone'] = (df['rsi'] >= self.rsi_pullback_low) & (df['rsi'] <= self.rsi_pullback_high)
            df['rsi_recovery_bull'] = df['rsi'] > self.rsi_recovery
            df['rsi_recovery_bear'] = df['rsi'] < (100 - self.rsi_recovery)  # For short signals
            
            # 3. MACD for Confirmation
            macd = ta.trend.MACD(df['close'], window_slow=self.macd_slow, window_fast=self.macd_fast, window_sign=self.macd_signal)
            df['macd'] = macd.macd()
            df['macd_signal_line'] = macd.macd_signal()
            df['macd_histogram'] = df['macd'] - df['macd_signal_line']
            
            # Handle NaN values in MACD
            df['macd'] = df['macd'].bfill()
            df['macd_signal_line'] = df['macd_signal_line'].bfill()
            df['macd_histogram'] = df['macd_histogram'].bfill()
            
            # MACD momentum conditions
            df['macd_hist_positive'] = df['macd_histogram'] > 0
            df['macd_hist_negative'] = df['macd_histogram'] < 0
            df['macd_hist_increasing'] = df['macd_histogram'] > df['macd_histogram'].shift(1)
            df['macd_hist_decreasing'] = df['macd_histogram'] < df['macd_histogram'].shift(1)
            
            # 4. Volume Filter for False Signal Reduction
            if self.volume_filter_enabled:
                # Calculate volume moving average
                df['volume_sma'] = df['volume'].rolling(window=self.volume_period).mean()
                df['volume_sma'] = df['volume_sma'].bfill()
                
                # Volume condition: current volume should be above average
                df['volume_above_avg'] = df['volume'] > (df['volume_sma'] * self.volume_multiplier)
                
                # Additional volume momentum indicator
                df['volume_increasing'] = df['volume'] > df['volume'].shift(1)
            else:
                # If volume filter is disabled, always pass volume check
                df['volume_above_avg'] = True
                df['volume_increasing'] = True
            
            # 5. ATR Filter for Volatility-based False Signal Reduction
            if self.atr_filter_enabled:
                # Calculate Average True Range for volatility measurement
                df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period)
                df['atr'] = df['atr'].bfill()
                
                # ATR as percentage of current price
                df['atr_pct'] = (df['atr'] / df['close']) * 100
                
                # Volatility condition: ATR should be above minimum threshold to avoid choppy markets
                df['sufficient_volatility'] = df['atr_pct'] > self.atr_threshold
                
                # Additional volatility momentum
                df['volatility_increasing'] = df['atr'] > df['atr'].shift(1)
            else:
                # If ATR filter is disabled, always pass volatility check
                df['sufficient_volatility'] = True
                df['volatility_increasing'] = True
            
            # 6. Combined Filter Conditions
            df['volume_filter_pass'] = df['volume_above_avg'] if self.volume_filter_enabled else True
            df['atr_filter_pass'] = df['sufficient_volatility'] if self.atr_filter_enabled else True
            df['noise_filter_pass'] = df['volume_filter_pass'] & df['atr_filter_pass']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding indicators: {e}")
            return df
    
    def get_signal(self, klines):
        """Generate Smart Trend Catcher signals"""
        try:
            min_required = max(self.ema_trend, self.macd_slow, self.volume_period, self.atr_period) + 10
            if not klines or len(klines) < min_required:
                logger.warning(f"Insufficient data for signal generation. Need at least {min_required} candles")
                return None
            
            # Convert klines to DataFrame
            df = pd.DataFrame(klines)
            if len(df.columns) != 12:
                logger.error(f"Invalid klines data format. Expected 12 columns, got {len(df.columns)}")
                return None
                
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            
            # Convert to numeric and handle NaN values
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    logger.warning(f"NaN values detected in {col} column, filling with previous values")
                    df[col] = df[col].ffill().bfill()
            
            # Check if we still have invalid data after cleaning
            if df[numeric_columns].isna().any().any():
                logger.error("Unable to clean NaN values from price data")
                return None
            
            # Add indicators
            df = self.add_indicators(df)
            
            if len(df) < 2:
                return None
            
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Validate that all required indicators are available and not NaN
            required_columns = ['uptrend', 'downtrend', 'rsi', 'rsi_pullback_zone', 'rsi_recovery_bull', 'rsi_recovery_bear', 
                              'macd_hist_positive', 'macd_hist_negative', 'macd_hist_increasing', 'macd_hist_decreasing',
                              'noise_filter_pass', 'volume_filter_pass', 'atr_filter_pass']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Required indicator {col} not found in DataFrame")
                    return None
                if pd.isna(latest[col]):
                    logger.warning(f"NaN value found in {col} indicator, skipping signal")
                    return None
            
            # Signal generation logic: Smart Trend Catcher with False Signal Reduction
            
            # First check if noise filters pass (volume and ATR filters)
            if not bool(latest['noise_filter_pass']):
                # Log why signal was filtered out
                filter_reasons = []
                if self.volume_filter_enabled and not bool(latest['volume_filter_pass']):
                    filter_reasons.append(f"Low volume ({latest['volume']:.0f} vs avg {latest.get('volume_sma', 0) * self.volume_multiplier:.0f})")
                if self.atr_filter_enabled and not bool(latest['atr_filter_pass']):
                    filter_reasons.append(f"Low volatility (ATR {latest.get('atr_pct', 0):.2f}% < {self.atr_threshold}%)")
                
                logger.debug(f"Signal filtered out due to: {', '.join(filter_reasons)}")
                return None
            
            # BUY Signal: Uptrend + (Pullback recovery OR MACD momentum confirmation) + Noise filters pass
            if bool(latest['uptrend']):  # Trend Filter: Price > 50 EMA
                
                # Check for pullback recovery: RSI was in pullback zone and now above recovery level
                rsi_pullback_recovery = (bool(previous['rsi_pullback_zone']) and bool(latest['rsi_recovery_bull']))
                
                # Check for MACD momentum: Histogram positive and increasing
                macd_momentum_bull = (bool(latest['macd_hist_positive']) and bool(latest['macd_hist_increasing']))
                
                # Entry condition: Either RSI recovery OR MACD momentum confirmation
                if rsi_pullback_recovery or macd_momentum_bull:
                    volume_info = f", Vol: {latest['volume']:.0f}" if self.volume_filter_enabled else ""
                    atr_info = f", ATR: {latest.get('atr_pct', 0):.2f}%" if self.atr_filter_enabled else ""
                    logger.info(f"BUY signal - Uptrend, RSI: {latest['rsi']:.2f}, MACD Hist: {latest['macd_histogram']:.6f}{volume_info}{atr_info} - Price: {latest['close']:.6f}")
                    return 'BUY'
            
            # SELL Signal: Downtrend + (Pullback recovery OR MACD momentum confirmation) + Noise filters pass
            elif bool(latest['downtrend']):  # Trend Filter: Price < 50 EMA
                
                # For downtrend, pullback would be RSI 50-70, recovery would be RSI < 50
                rsi_pullback_zone_bear = (latest['rsi'] >= self.rsi_recovery) & (latest['rsi'] <= (100 - self.rsi_pullback_low))
                rsi_pullback_recovery_bear = (previous['rsi'] >= self.rsi_recovery and bool(latest['rsi_recovery_bear']))
                
                # Check for MACD momentum: Histogram negative and decreasing
                macd_momentum_bear = (bool(latest['macd_hist_negative']) and bool(latest['macd_hist_decreasing']))
                
                # Entry condition: Either RSI recovery OR MACD momentum confirmation
                if rsi_pullback_recovery_bear or macd_momentum_bear:
                    volume_info = f", Vol: {latest['volume']:.0f}" if self.volume_filter_enabled else ""
                    atr_info = f", ATR: {latest.get('atr_pct', 0):.2f}%" if self.atr_filter_enabled else ""
                    logger.info(f"SELL signal - Downtrend, RSI: {latest['rsi']:.2f}, MACD Hist: {latest['macd_histogram']:.6f}{volume_info}{atr_info} - Price: {latest['close']:.6f}")
                    return 'SELL'
            
            return None
            
        except Exception as e:
            logger.error(f"Error in get_signal: {e}")
            return None

# Factory function to get a strategy by name
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    try:
        from modules.config import RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD
    except ImportError:
        # Default values if config import fails
        RSI_PERIOD = 14
        RSI_OVERBOUGHT = 70
        RSI_OVERSOLD = 30
    
    strategies = {
        'SmartTrendCatcher': SmartTrendCatcher(
            ema_trend=50,
            rsi_period=RSI_PERIOD,
            rsi_pullback_low=30,
            rsi_pullback_high=50,
            rsi_recovery=50,
            # False signal reduction parameters
            volume_filter_enabled=True,
            volume_period=20,
            volume_multiplier=1.2,
            atr_filter_enabled=True,
            atr_period=14,
            atr_threshold=0.5
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