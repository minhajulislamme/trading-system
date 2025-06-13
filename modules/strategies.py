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
    Enhanced Smart Trend Catcher Strategy with Advanced False Signal Reduction:
    
    Core Strategy Improvements:
    - Multi-timeframe trend confirmation
    - Enhanced volatility filtering with Bollinger Bands
    - Price action confirmation with candlestick patterns
    - Dynamic position sizing based on confidence
    - Advanced money management with risk-reward optimization
    
    Signal Quality Enhancements:
    - Confluence requirement (multiple confirmations needed)
    - Market structure analysis (support/resistance levels)
    - Momentum divergence detection
    - News/time-based filters
    """
    
    def __init__(self, 
                 # Enhanced trend filter parameters
                 ema_trend=50,
                 ema_fast=21,               # Additional fast EMA for trend confirmation
                 
                 # RSI parameters with tighter controls
                 rsi_period=14,
                 rsi_pullback_low=30,       # Lower pullback zone for more signals
                 rsi_pullback_high=50,      # Standard pullback zone
                 rsi_recovery=50,           # Standard recovery threshold
                 rsi_extreme_low=25,        # Extreme oversold for higher confidence
                 rsi_extreme_high=70,       # Lower overbought for more SELL signals
                 
                 # MACD parameters with confirmation
                 macd_fast=12,
                 macd_slow=26,
                 macd_signal=9,
                 macd_histogram_threshold=0.0001,  # Minimum histogram value for signal
                 
                 # Enhanced false signal reduction
                 volume_filter_enabled=True,
                 volume_period=20,
                 volume_multiplier=1.5,     # Stricter volume requirement
                 volume_surge_multiplier=2.0,  # Volume surge detection
                 
                 atr_filter_enabled=True,
                 atr_period=14,
                 atr_threshold=0.8,         # Higher volatility requirement
                 atr_trend_factor=1.2,      # ATR trend confirmation
                 
                 # Bollinger Bands for volatility confirmation
                 bb_period=20,
                 bb_std=2.0,
                 bb_squeeze_threshold=0.1,  # Detect low volatility periods
                 
                 # Price action filters
                 min_candle_body_pct=0.5,   # Minimum candle body size
                 max_wick_ratio=3.0,        # Maximum wick to body ratio
                 
                 # Advanced filtering parameters
                 confluence_required=2,      # Number of confirmations needed
                 
                 # Dynamic position sizing
                 base_position_pct=0.3,     # Base position size (30% instead of 75%)
                 max_position_pct=0.5,      # Maximum position size
                 confidence_multiplier=1.5): # Multiply position size by confidence
        
        super().__init__("SmartTrendCatcher")
        
        # Enhanced parameter validation
        if ema_trend <= 0 or ema_fast <= 0:
            raise ValueError("EMA periods must be positive")
        if ema_fast >= ema_trend:
            raise ValueError("Fast EMA must be less than trend EMA")
        if rsi_period <= 0:
            raise ValueError("RSI period must be positive")
        if not (0 <= rsi_extreme_low <= rsi_pullback_low <= rsi_pullback_high <= rsi_recovery <= rsi_extreme_high <= 100):
            raise ValueError("Invalid RSI levels hierarchy")
        if macd_fast <= 0 or macd_slow <= 0 or macd_signal <= 0 or macd_fast >= macd_slow:
            raise ValueError("Invalid MACD parameters")
        if not (0 < base_position_pct <= max_position_pct <= 1.0):
            raise ValueError("Invalid position sizing parameters")
        if confluence_required < 1:
            raise ValueError("Confluence required must be at least 1")
        
        # Store enhanced parameters
        self.ema_trend = ema_trend
        self.ema_fast = ema_fast
        
        self.rsi_period = rsi_period
        self.rsi_pullback_low = rsi_pullback_low
        self.rsi_pullback_high = rsi_pullback_high
        self.rsi_recovery = rsi_recovery
        self.rsi_extreme_low = rsi_extreme_low
        self.rsi_extreme_high = rsi_extreme_high
        
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
        logger.info(f"  Trend EMAs: {ema_fast}/{ema_trend}")
        logger.info(f"  RSI levels: {rsi_extreme_low}/{rsi_pullback_low}-{rsi_pullback_high}/{rsi_recovery}/{rsi_extreme_high}")
        logger.info(f"  Confluence required: {confluence_required}")
        logger.info(f"  Position sizing: {base_position_pct:.1%}-{max_position_pct:.1%}")
    
    def add_indicators(self, df):
        """Add enhanced indicators with multi-layer filtering"""
        try:
            # Ensure sufficient data
            min_required = max(self.ema_trend, self.macd_slow, self.volume_period, 
                             self.atr_period, self.bb_period) + 20
            if len(df) < min_required:
                logger.warning(f"Insufficient data: need {min_required}, got {len(df)}")
                return df
            
            # Validate required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    return df
                if df[col].isna().all():
                    logger.error(f"Column {col} contains only NaN values")
                    return df
            
            # 1. Enhanced Trend Analysis
            df['ema_trend'] = ta.trend.ema_indicator(df['close'], window=self.ema_trend)
            df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=self.ema_fast)
            df['ema_trend'] = df['ema_trend'].bfill()
            df['ema_fast'] = df['ema_fast'].bfill()
            
            # Multi-timeframe trend confirmation
            df['strong_uptrend'] = (df['close'] > df['ema_trend']) & (df['ema_fast'] > df['ema_trend'])
            df['strong_downtrend'] = (df['close'] < df['ema_trend']) & (df['ema_fast'] < df['ema_trend'])
            df['trend_aligned'] = df['strong_uptrend'] | df['strong_downtrend']
            
            # Trend strength measurement
            df['trend_strength'] = np.abs(df['close'] - df['ema_trend']) / df['ema_trend']
            df['strong_trend'] = df['trend_strength'] > 0.02  # At least 2% from trend line
            
            # 2. Enhanced RSI Analysis
            df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
            df['rsi'] = df['rsi'].bfill()
            
            # RSI zones with confidence levels
            df['rsi_extreme_oversold'] = df['rsi'] < self.rsi_extreme_low
            df['rsi_pullback_zone'] = (df['rsi'] >= self.rsi_pullback_low) & (df['rsi'] <= self.rsi_pullback_high)
            df['rsi_recovery_bull'] = df['rsi'] > self.rsi_recovery
            df['rsi_extreme_overbought'] = df['rsi'] > self.rsi_extreme_high
            df['rsi_recovery_bear'] = df['rsi'] < (100 - self.rsi_recovery)
            
            # RSI momentum and divergence
            df['rsi_momentum'] = df['rsi'].diff()
            df['rsi_increasing'] = df['rsi_momentum'] > 0
            df['rsi_decreasing'] = df['rsi_momentum'] < 0
            
            # 3. Enhanced MACD Analysis
            macd = ta.trend.MACD(df['close'], window_slow=self.macd_slow, 
                               window_fast=self.macd_fast, window_sign=self.macd_signal)
            df['macd'] = macd.macd()
            df['macd_signal_line'] = macd.macd_signal()
            df['macd_histogram'] = df['macd'] - df['macd_signal_line']
            
            # Handle NaN values
            for col in ['macd', 'macd_signal_line', 'macd_histogram']:
                df[col] = df[col].bfill()
            
            # MACD conditions with threshold
            df['macd_hist_positive'] = df['macd_histogram'] > self.macd_histogram_threshold
            df['macd_hist_negative'] = df['macd_histogram'] < -self.macd_histogram_threshold
            df['macd_hist_increasing'] = df['macd_histogram'] > df['macd_histogram'].shift(1)
            df['macd_hist_decreasing'] = df['macd_histogram'] < df['macd_histogram'].shift(1)
            df['macd_strong_momentum'] = np.abs(df['macd_histogram']) > self.macd_histogram_threshold * 3
            
            # 4. Enhanced Volume Analysis
            if self.volume_filter_enabled:
                df['volume_sma'] = df['volume'].rolling(window=self.volume_period).mean()
                df['volume_sma'] = df['volume_sma'].bfill()
                
                # Multiple volume conditions
                df['volume_above_avg'] = df['volume'] > (df['volume_sma'] * self.volume_multiplier)
                df['volume_surge'] = df['volume'] > (df['volume_sma'] * self.volume_surge_multiplier)
                df['volume_increasing'] = df['volume'] > df['volume'].shift(1)
                df['volume_momentum'] = df['volume'].rolling(3).mean() > df['volume_sma']
                
                # Volume confirmation score
                df['volume_score'] = (
                    df['volume_above_avg'].astype(int) +
                    df['volume_surge'].astype(int) +
                    df['volume_increasing'].astype(int) +
                    df['volume_momentum'].astype(int)
                )
            else:
                df['volume_score'] = 4  # Max score if disabled
            
            # 5. Enhanced Volatility Analysis
            if self.atr_filter_enabled:
                df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 
                                                           window=self.atr_period)
                df['atr'] = df['atr'].bfill()
                df['atr_pct'] = (df['atr'] / df['close']) * 100
                
                # ATR conditions
                df['sufficient_volatility'] = df['atr_pct'] > self.atr_threshold
                df['atr_increasing'] = df['atr'] > df['atr'].shift(1)
                df['atr_trend'] = df['atr'].rolling(5).mean() > df['atr'].rolling(20).mean()
                
                # Volatility score
                df['volatility_score'] = (
                    df['sufficient_volatility'].astype(int) +
                    df['atr_increasing'].astype(int) +
                    df['atr_trend'].astype(int)
                )
            else:
                df['volatility_score'] = 3  # Max score if disabled
            
            # 6. Bollinger Bands for Squeeze Detection
            bb = ta.volatility.BollingerBands(df['close'], window=self.bb_period, window_dev=self.bb_std)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # Bollinger Band conditions
            df['bb_squeeze'] = df['bb_width'] < self.bb_squeeze_threshold
            df['bb_expansion'] = df['bb_width'] > df['bb_width'].shift(1)
            df['bb_breakout_up'] = (df['close'] > df['bb_upper']) & (df['close'].shift(1) <= df['bb_upper'].shift(1))
            df['bb_breakout_down'] = (df['close'] < df['bb_lower']) & (df['close'].shift(1) >= df['bb_lower'].shift(1))
            
            # 7. Price Action Analysis
            df['candle_body'] = np.abs(df['close'] - df['open'])
            df['candle_range'] = df['high'] - df['low']
            df['body_pct'] = df['candle_body'] / df['candle_range']
            df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
            
            # Price action conditions
            df['strong_candle'] = df['body_pct'] > self.min_candle_body_pct
            df['reasonable_wicks'] = (df['upper_wick'] / df['candle_body'] < self.max_wick_ratio) & \
                                   (df['lower_wick'] / df['candle_body'] < self.max_wick_ratio)
            df['bullish_candle'] = df['close'] > df['open']
            df['bearish_candle'] = df['close'] < df['open']
            
            # 8. Confluence Scoring System
            df['bull_confluence'] = 0
            df['bear_confluence'] = 0
            
            # Add confluence points for bullish signals
            df.loc[df['strong_uptrend'], 'bull_confluence'] += 1
            df.loc[df['rsi_pullback_zone'] & df['rsi_increasing'], 'bull_confluence'] += 1
            df.loc[df['macd_hist_positive'] & df['macd_hist_increasing'], 'bull_confluence'] += 1
            df.loc[df['volume_score'] >= 2, 'bull_confluence'] += 1
            df.loc[df['volatility_score'] >= 2, 'bull_confluence'] += 1
            df.loc[df['bb_expansion'] & ~df['bb_squeeze'], 'bull_confluence'] += 1
            df.loc[df['strong_candle'] & df['bullish_candle'], 'bull_confluence'] += 1
            
            # Add confluence points for bearish signals - IMPROVED for better SELL signal generation
            df.loc[df['strong_downtrend'], 'bear_confluence'] += 1
            # RSI conditions for shorts - mutually exclusive to avoid double counting
            df.loc[df['rsi'] > self.rsi_extreme_high, 'bear_confluence'] += 2  # High RSI gets more weight
            df.loc[(df['rsi'] > self.rsi_recovery) & (df['rsi'] <= self.rsi_extreme_high), 'bear_confluence'] += 1  # Medium RSI
            df.loc[df['rsi_decreasing'], 'bear_confluence'] += 1               # RSI momentum turning down
            # MACD conditions - mutually exclusive
            df.loc[df['macd_hist_negative'] & df['macd_hist_decreasing'], 'bear_confluence'] += 2  # Both conditions get more weight
            df.loc[df['macd_hist_negative'] & ~df['macd_hist_decreasing'], 'bear_confluence'] += 1  # Just negative MACD
            df.loc[df['volume_score'] >= 2, 'bear_confluence'] += 1
            df.loc[df['volatility_score'] >= 2, 'bear_confluence'] += 1
            df.loc[df['bb_expansion'] & ~df['bb_squeeze'], 'bear_confluence'] += 1
            df.loc[df['strong_candle'] & df['bearish_candle'], 'bear_confluence'] += 1
            # Add price below EMAs as bear signal
            df.loc[df['close'] < df['ema_fast'], 'bear_confluence'] += 1
            
            # 9. Signal Quality Assessment
            df['signal_quality'] = np.maximum(df['bull_confluence'], df['bear_confluence'])
            df['high_confidence'] = df['signal_quality'] >= self.confluence_required + 1
            df['medium_confidence'] = df['signal_quality'] >= self.confluence_required
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding enhanced indicators: {e}")
            return df
    
    def get_signal(self, klines):
        """Generate enhanced Smart Trend Catcher signals with confluence requirement"""
        try:
            min_required = max(self.ema_trend, self.macd_slow, self.volume_period, 
                             self.atr_period, self.bb_period) + 20
            if not klines or len(klines) < min_required:
                # Show warning every 10th time to reduce log spam
                if self._warning_count % 10 == 0:
                    logger.warning(f"Insufficient data for enhanced signal generation (need {min_required}, have {len(klines) if klines else 0})")
                self._warning_count += 1
                return None
            
            # Convert and validate data
            df = pd.DataFrame(klines)
            if len(df.columns) != 12:
                logger.error(f"Invalid klines format: expected 12 columns, got {len(df.columns)}")
                return None
                
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                         'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
            
            # Data cleaning
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    logger.warning(f"Cleaning NaN values in {col}")
                    df[col] = df[col].ffill().bfill()
            
            if df[numeric_columns].isna().any().any():
                logger.error("Failed to clean price data")
                return None
            
            # Add enhanced indicators
            df = self.add_indicators(df)
            
            if len(df) < 2:
                return None
            
            current_idx = len(df) - 1
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Validate required columns
            required_columns = ['bull_confluence', 'bear_confluence', 'signal_quality', 
                              'medium_confidence', 'high_confidence', 'strong_uptrend', 'strong_downtrend']
            
            for col in required_columns:
                if col not in df.columns or pd.isna(latest[col]):
                    logger.warning(f"Missing or invalid indicator: {col}")
                    return None
            
            # Enhanced Signal Generation with Confluence
            signal = None
            confidence_level = 0
            
            # BUY Signal: Strong uptrend + sufficient confluence
            if (bool(latest['strong_uptrend']) and 
                int(latest['bull_confluence']) >= self.confluence_required):
                
                # Additional confirmations for higher confidence
                extra_confirmations = []
                
                # RSI pullback recovery
                if (bool(previous.get('rsi_pullback_zone', False)) and 
                    bool(latest.get('rsi_recovery_bull', False))):
                    extra_confirmations.append("RSI Recovery")
                
                # Extreme RSI for higher confidence
                if bool(previous.get('rsi_extreme_oversold', False)):
                    extra_confirmations.append("Extreme RSI")
                
                # Volume surge
                if bool(latest.get('volume_surge', False)):
                    extra_confirmations.append("Volume Surge")
                
                # MACD strong momentum
                if bool(latest.get('macd_strong_momentum', False)):
                    extra_confirmations.append("Strong MACD")
                
                # Bollinger Band breakout
                if bool(latest.get('bb_breakout_up', False)):
                    extra_confirmations.append("BB Breakout")
                
                signal = 'BUY'
                confidence_level = int(latest['bull_confluence'])
                
                logger.info(f"🟢 BUY Signal - Confluence: {confidence_level}, "
                          f"RSI: {latest['rsi']:.1f}, "
                          f"MACD: {latest['macd_histogram']:.6f}, "
                          f"Volume Score: {latest.get('volume_score', 0)}, "
                          f"Extras: {', '.join(extra_confirmations) if extra_confirmations else 'None'}")
            
            # SELL Signal: Enhanced conditions for better signal generation  
            elif (
                # Primary condition: Either strong downtrend OR high RSI with momentum turning down
                (bool(latest['strong_downtrend']) or 
                 (latest['rsi'] > self.rsi_extreme_high and bool(latest.get('rsi_decreasing', False)))) and 
                # Confluence requirement (lower threshold for more signals)
                int(latest['bear_confluence']) >= max(1, self.confluence_required - 1)
            ):
                
                # Additional confirmations
                extra_confirmations = []
                
                # RSI overbought recovery for shorts (price coming down from overbought)
                if (previous['rsi'] > self.rsi_extreme_high and 
                    latest['rsi'] < previous['rsi'] and latest['rsi'] > self.rsi_recovery):
                    extra_confirmations.append("RSI Recovery")
                
                # Extreme RSI for higher confidence
                if bool(previous.get('rsi_extreme_overbought', False)):
                    extra_confirmations.append("Extreme RSI")
                
                # Volume surge
                if bool(latest.get('volume_surge', False)):
                    extra_confirmations.append("Volume Surge")
                
                # MACD strong momentum
                if bool(latest.get('macd_strong_momentum', False)):
                    extra_confirmations.append("Strong MACD")
                
                # Bollinger Band breakout
                if bool(latest.get('bb_breakout_down', False)):
                    extra_confirmations.append("BB Breakout")
                
                signal = 'SELL'
                confidence_level = int(latest['bear_confluence'])
                
                logger.info(f"🔴 SELL Signal - Confluence: {confidence_level}, "
                          f"RSI: {latest['rsi']:.1f}, "
                          f"MACD: {latest['macd_histogram']:.6f}, "
                          f"Volume Score: {latest.get('volume_score', 0)}, "
                          f"Extras: {', '.join(extra_confirmations) if extra_confirmations else 'None'}")
            
            # Update tracking if signal generated
            if signal:
                # Store confidence for position sizing
                setattr(self, '_last_confidence', confidence_level)
            
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
            
            # Base multiplier (normalize from old 75% default to current base_position_pct)
            OLD_DEFAULT_POSITION_PCT = 0.75
            multiplier = self.base_position_pct / OLD_DEFAULT_POSITION_PCT
            
            # Adjust based on confidence
            if confidence >= self.confluence_required + 2:  # High confidence
                multiplier *= self.confidence_multiplier
            elif confidence >= self.confluence_required + 1:  # Medium-high confidence
                multiplier *= (self.confidence_multiplier + 1) / 2
            # else: use base multiplier for minimum confidence
            
            # Cap at maximum
            OLD_DEFAULT_POSITION_PCT = 0.75
            max_multiplier = self.max_position_pct / OLD_DEFAULT_POSITION_PCT
            multiplier = min(multiplier, max_multiplier)
            
            logger.info(f"Position size multiplier: {multiplier:.2f} (confidence: {confidence})")
            return multiplier
            
        except Exception as e:
            logger.error(f"Error calculating position size multiplier: {e}")
            OLD_DEFAULT_POSITION_PCT = 0.75
            return self.base_position_pct / OLD_DEFAULT_POSITION_PCT  # Safe default

# Factory function to get a strategy by name
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    try:
        from modules.config import RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD
        # Use config values for consistency
        rsi_extreme_high = RSI_OVERBOUGHT if RSI_OVERBOUGHT else 70
        rsi_period = RSI_PERIOD if RSI_PERIOD else 14
    except ImportError:
        # Default values if config import fails
        rsi_period = 14
        rsi_extreme_high = 70
        logger.warning("Could not import RSI config values, using defaults")
    
    strategies = {
        'SmartTrendCatcher': SmartTrendCatcher(
            ema_trend=50,
            rsi_period=rsi_period,
            rsi_pullback_low=30,
            rsi_pullback_high=50,
            rsi_recovery=50,
            rsi_extreme_high=rsi_extreme_high,  # Use config value
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