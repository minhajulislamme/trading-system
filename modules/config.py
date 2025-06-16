import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Testnet configuration
API_TESTNET = os.getenv('BINANCE_API_TESTNET', 'False').lower() == 'true'

# API URLs - Automatically determined based on testnet setting
if API_TESTNET:
    # Testnet URLs
    API_URL = 'https://testnet.binancefuture.com'
    WS_BASE_URL = 'wss://stream.binancefuture.com'
else:
    # Production URLs
    API_URL = os.getenv('BINANCE_API_URL', 'https://fapi.binance.com')
    WS_BASE_URL = 'wss://fstream.binance.com'

# API request settings
RECV_WINDOW = int(os.getenv('BINANCE_RECV_WINDOW', '10000'))

# Trading parameters
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'SOLUSDT')
TRADING_TYPE = 'FUTURES'  # Use futures trading
LEVERAGE = int(os.getenv('LEVERAGE', '10'))
MARGIN_TYPE = os.getenv('MARGIN_TYPE', 'ISOLATED')  # ISOLATED or CROSSED
STRATEGY = os.getenv('STRATEGY', 'SmartTrendCatcher')

# Position sizing - Enhanced risk management (aligned with SmartTrendCatcher)
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '50.0'))
FIXED_TRADE_PERCENTAGE = float(os.getenv('FIXED_TRADE_PERCENTAGE', '0.30'))  # 30% to match strategy base_position_pct
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '3'))  # Conservative for better risk management

# Margin safety settings - More conservative
MARGIN_SAFETY_FACTOR = float(os.getenv('MARGIN_SAFETY_FACTOR', '0.90'))  # Use at most 90% of available margin
MAX_POSITION_SIZE_PCT = float(os.getenv('MAX_POSITION_SIZE_PCT', '0.50'))  # Max 50% position size (matches strategy)
MIN_FREE_BALANCE_PCT = float(os.getenv('MIN_FREE_BALANCE_PCT', '0.10'))  # Keep at least 10% free

# Multi-instance configuration for running separate bot instances per trading pair
MULTI_INSTANCE_MODE = os.getenv('MULTI_INSTANCE_MODE', 'True').lower() == 'true'
MAX_POSITIONS_PER_SYMBOL = int(os.getenv('MAX_POSITIONS_PER_SYMBOL', '3'))  # Updated to match .env

# Auto-compounding settings - Enhanced with performance-based adjustments
AUTO_COMPOUND = os.getenv('AUTO_COMPOUND', 'True').lower() == 'true'
COMPOUND_REINVEST_PERCENT = float(os.getenv('COMPOUND_REINVEST_PERCENT', '0.75'))
COMPOUND_INTERVAL = os.getenv('COMPOUND_INTERVAL', 'DAILY')

# Dynamic compounding adjustments
COMPOUND_PERFORMANCE_WINDOW = int(os.getenv('COMPOUND_PERFORMANCE_WINDOW', '7'))  # Look back 7 days
COMPOUND_MIN_WIN_RATE = float(os.getenv('COMPOUND_MIN_WIN_RATE', '0.6'))  # Require 60% win rate
COMPOUND_MAX_DRAWDOWN = float(os.getenv('COMPOUND_MAX_DRAWDOWN', '0.15'))  # Pause if >15% drawdown
COMPOUND_SCALING_FACTOR = float(os.getenv('COMPOUND_SCALING_FACTOR', '0.5'))  # Reduce compounding if performance poor

# Technical indicator parameters - SmartTrendCatcher Strategy (EMA 9/26 + MACD Configuration)

# EMA parameters (matching chart configuration - EMA 9/26)
FAST_EMA = int(os.getenv('FAST_EMA', '9'))     # Fast EMA matching chart (was 8)
SLOW_EMA = int(os.getenv('SLOW_EMA', '26'))    # Slow EMA matching MACD slow period (was 21)

# MACD parameters (matching chart configuration - 12,26,9)
MACD_FAST = int(os.getenv('MACD_FAST', '12'))    # Matches chart MACD fast
MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))    # Matches chart MACD slow
MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9')) # Matches chart MACD signal
MACD_HISTOGRAM_THRESHOLD = float(os.getenv('MACD_HISTOGRAM_THRESHOLD', '0.00001'))  # Very sensitive threshold

# Volume filter parameters (optimized for chart patterns)
VOLUME_FILTER_ENABLED = os.getenv('VOLUME_FILTER_ENABLED', 'True').lower() == 'true'
VOLUME_PERIOD = int(os.getenv('VOLUME_PERIOD', '20'))  # Standard volume period
VOLUME_MULTIPLIER = float(os.getenv('VOLUME_MULTIPLIER', '1.2'))  # More lenient for trend continuation (was 1.5)
VOLUME_SURGE_MULTIPLIER = float(os.getenv('VOLUME_SURGE_MULTIPLIER', '1.6'))  # Lower threshold (was 2.0)

# ATR (Average True Range) parameters (optimized for chart volatility)
ATR_FILTER_ENABLED = os.getenv('ATR_FILTER_ENABLED', 'True').lower() == 'true'
ATR_PERIOD = int(os.getenv('ATR_PERIOD', '14'))  # Standard ATR period
ATR_THRESHOLD = float(os.getenv('ATR_THRESHOLD', '0.5'))  # Lower threshold for more signals (was 0.8)
ATR_TREND_FACTOR = float(os.getenv('ATR_TREND_FACTOR', '1.0'))  # More responsive (was 1.2)

# Bollinger Bands parameters (optimized for trending breakouts)
BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))  # Standard BB period
BB_STD = float(os.getenv('BB_STD', '2.0'))
BB_SQUEEZE_THRESHOLD = float(os.getenv('BB_SQUEEZE_THRESHOLD', '0.08'))  # Lower for more breakout signals (was 0.1)

# Price action parameters (optimized for trending markets)
MIN_CANDLE_BODY_PCT = float(os.getenv('MIN_CANDLE_BODY_PCT', '0.4'))  # Lower for more signals (was 0.5)
MAX_WICK_RATIO = float(os.getenv('MAX_WICK_RATIO', '3.5'))  # Allow more wick ratio (was 3.0)

# Confluence and confidence parameters (aligned with chart configuration)
CONFLUENCE_REQUIRED = int(os.getenv('CONFLUENCE_REQUIRED', '0'))  # Set to 0 as requested
BASE_POSITION_PCT = float(os.getenv('BASE_POSITION_PCT', '0.35'))  # Larger base position (was 0.30)
MAX_POSITION_PCT = float(os.getenv('MAX_POSITION_PCT', '0.60'))    # Higher max position (was 0.50)
CONFIDENCE_MULTIPLIER = float(os.getenv('CONFIDENCE_MULTIPLIER', '1.4'))  # More conservative (was 1.5)

TIMEFRAME = os.getenv('TIMEFRAME', '15m')

# Risk management - Enhanced stop loss settings
USE_STOP_LOSS = os.getenv('USE_STOP_LOSS', 'True').lower() == 'true'
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.025'))  # 2.5% stop loss (more conservative)
TRAILING_STOP = os.getenv('TRAILING_STOP', 'True').lower() == 'true'
TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '0.015'))  # 1.5% trailing stop

# Enhanced backtesting parameters
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2023-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '')
BACKTEST_INITIAL_BALANCE = float(os.getenv('BACKTEST_INITIAL_BALANCE', '50.0'))
BACKTEST_COMMISSION = float(os.getenv('BACKTEST_COMMISSION', '0.0004'))
BACKTEST_USE_AUTO_COMPOUND = os.getenv('BACKTEST_USE_AUTO_COMPOUND', 'True').lower() == 'true'  # Enabled for enhanced auto-compounding test

# Enhanced validation requirements - More realistic for 90-day backtests
BACKTEST_BEFORE_LIVE = os.getenv('BACKTEST_BEFORE_LIVE', 'True').lower() == 'true'
BACKTEST_MIN_PROFIT_PCT = float(os.getenv('BACKTEST_MIN_PROFIT_PCT', '10.0'))  # Reduced for longer periods
BACKTEST_MIN_WIN_RATE = float(os.getenv('BACKTEST_MIN_WIN_RATE', '40.0'))  # More realistic for 90 days
BACKTEST_MAX_DRAWDOWN = float(os.getenv('BACKTEST_MAX_DRAWDOWN', '30.0'))  # Allow higher DD for longer periods
BACKTEST_MIN_PROFIT_FACTOR = float(os.getenv('BACKTEST_MIN_PROFIT_FACTOR', '1.2'))  # More conservative
BACKTEST_PERIOD = os.getenv('BACKTEST_PERIOD', '90 days')  # Default to 90 days for comprehensive testing

# Logging and notifications
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
USE_TELEGRAM = os.getenv('USE_TELEGRAM', 'True').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
SEND_DAILY_REPORT = os.getenv('SEND_DAILY_REPORT', 'True').lower() == 'true'
DAILY_REPORT_TIME = os.getenv('DAILY_REPORT_TIME', '00:00')  # 24-hour format

# Other settings
RETRY_COUNT = int(os.getenv('RETRY_COUNT', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))  # seconds

# Additional SmartTrendCatcher specific parameters
VOLUME_SURGE_MULTIPLIER = float(os.getenv('VOLUME_SURGE_MULTIPLIER', '2.0'))
ATR_TREND_FACTOR = float(os.getenv('ATR_TREND_FACTOR', '1.2'))
MIN_CANDLE_BODY_PCT = float(os.getenv('MIN_CANDLE_BODY_PCT', '0.5'))
MAX_WICK_RATIO = float(os.getenv('MAX_WICK_RATIO', '3.0'))