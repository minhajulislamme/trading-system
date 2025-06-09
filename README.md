# 🤖 Advanced Cryptocurrency Trading system

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Binance API](https://img.shields.io/badge/Binance-Futures%20API-yellow.svg)](https://binance-docs.github.io/apidocs/futures/en/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Telegram](https://img.shields.io/badge/Telegram-system%20Integration-blue.svg)](https://telegram.org/)

_A sophisticated, AI-enhanced cryptocurrency trading system with advanced technical analysis and risk management_

</div>

## 🌟 Overview

This is a cutting-edge cryptocurrency trading system designed for Binance Futures trading with a focus on sophisticated technical analysis, institutional-grade order flow detection, and adaptive risk management. Built with modern Python architecture, it features real-time WebSocket data streaming, comprehensive backtesting, and intelligent position sizing.

### 🎯 Key Features

- **🔥 Advanced DynamicStrategy** - AI-enhanced trading with 50+ technical indicators
- **📊 Institutional Order Flow Analysis** - Detect market maker vs retail activity
- **🎢 Market Regime Detection** - Adaptive algorithms for different market conditions
- **⚡ Real-time WebSocket Streaming** - Sub-second market data processing
- **🛡️ Sophisticated Risk Management** - Dynamic position sizing and stop-loss optimization
- **📈 Multi-timeframe Analysis** - Cross-timeframe signal confirmation
- **🤖 ML-Enhanced Signals** - Machine learning for pattern recognition
- **📱 Telegram Integration** - Real-time notifications and alerts
- **🔄 Comprehensive Backtesting** - Historical performance validation
- **⚙️ Multi-instance Support** - Run multiple strategies simultaneously

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Binance Futures account
- API keys with futures trading permissions

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/minhajulislamme/tradingsystem.git
   cd tradingsystem
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```

4. **Run setup script**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

### Quick Configuration

Edit the `.env` file with your credentials:

```env
# Binance API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
BINANCE_API_TESTNET=True  # Set to False for live trading

# Trading Configuration
TRADING_SYMBOL=SUIUSDT
LEVERAGE=10
STRATEGY=DynamicStrategy
INITIAL_BALANCE=50.0

# Risk Management
RISK_PER_TRADE=0.02
MAX_DAILY_TRADES=5
STOP_LOSS_PCT=2.0
TAKE_PROFIT_PCT=4.0

# Telegram Notifications (Optional)
USE_TELEGRAM=True
TELEGRAM_system_TOKEN=your_system_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 🎮 Usage

### Start Trading system

```bash
# Start with default configuration
python main.py

# Start with specific symbol
python main.py --symbol BTCUSDT

# Start with custom strategy
python main.py --strategy DynamicStrategy

# Run backtest first
python main.py --backtest --days 30
```

### Using Shell Scripts

```bash
# Start system in background
./run_system.sh

# Check system status
./check_system_status.sh

# Stop system manually
./stop_system_manual.sh
```

### Command Line Options

```
Options:
  --symbol TEXT          Trading symbol (default: SUIUSDT)
  --strategy TEXT        Strategy to use (default: DynamicStrategy)
  --timeframe TEXT       Timeframe for analysis (default: 15m)
  --backtest             Run backtest before live trading
  --days INTEGER         Days for backtesting (default: 30)
  --testnet             Use Binance testnet
  --telegram             Enable Telegram notifications
  --verbose             Enable verbose logging
```

## 🧠 Trading Strategies

### DynamicStrategy (Primary)

Our flagship strategy featuring:

#### 📊 Technical Indicators

- **Trend Analysis**: EMA (8/21), Supertrend, VWAP
- **Momentum**: RSI, MACD, Stochastic RSI
- **Volatility**: Bollinger Bands, ATR, Keltner Channels
- **Volume**: Volume Profile, OBV, Money Flow Index
- **Support/Resistance**: Fibonacci levels, Pivot points

#### 🎯 Advanced Features

- **Order Flow Imbalance Detection**: Identify institutional vs retail activity
- **Market Condition Classification**: BULLISH, BEARISH, SIDEWAYS, EXTREME
- **Reversal Pattern Recognition**: Hammer, Shooting Star, Engulfing patterns
- **Flash Crash Detection**: Protect against sudden market moves
- **Market Phase Analysis**: Accumulation, Markup, Distribution, Markdown

#### ⚡ Signal Generation

- **Multi-priority System**: HIGH, MEDIUM, LOW priority signals
- **Cross-timeframe Confirmation**: 15m primary with higher timeframe filters
- **Dynamic Position Sizing**: Based on volatility and market conditions
- **Adaptive Risk Management**: Cool-off periods and drawdown protection

## 🛡️ Risk Management

### Position Sizing

```python
# Dynamic position sizing based on:
# - Account balance
# - Market volatility (ATR)
# - Recent performance
# - Market conditions

position_size = base_size * volatility_multiplier * performance_factor
```

### Risk Controls

- **Maximum risk per trade**: 2% of account balance
- **Daily loss limit**: 5% of account balance
- **Maximum open positions**: 3 per symbol
- **Cool-off periods**: After consecutive losses
- **Emergency stop**: Automatic shutdown on extreme losses

### Stop Loss & Take Profit

- **Dynamic stop loss**: Based on ATR and support/resistance levels
- **Trailing stops**: Automatically adjust as price moves favorably
- **Partial profit taking**: Scale out at multiple levels
- **Risk-reward optimization**: Minimum 1:2 risk-reward ratio

## 📊 Backtesting

### Run Comprehensive Backtest

```bash
# Backtest specific symbol
python backtest_all_coins.py --symbol BTCUSDT --days 90

# Backtest multiple timeframes
python main.py --backtest --timeframes 15m,1h,4h

# Generate performance report
python main.py --backtest --report --export-csv
```

### Performance Metrics

- **Total Return**: Absolute and percentage returns
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Average Trade Duration**: Time in market per trade

## 🔧 Configuration

### Trading Parameters

| Parameter          | Description          | Default         |
| ------------------ | -------------------- | --------------- |
| `TRADING_SYMBOL`   | Primary trading pair | SUIUSDT         |
| `LEVERAGE`         | Futures leverage     | 10              |
| `TIMEFRAME`        | Analysis timeframe   | 15m             |
| `STRATEGY`         | Trading strategy     | DynamicStrategy |
| `RISK_PER_TRADE`   | Risk per trade (%)   | 2.0             |
| `MAX_DAILY_TRADES` | Daily trade limit    | 5               |

### Strategy Parameters

#### DynamicStrategy Configuration

```python
# EMA Settings
RAYSOL_TREND_EMA_FAST = 8
RAYSOL_TREND_EMA_SLOW = 21

# Volatility Settings
RAYSOL_VOLATILITY_LOOKBACK = 20
RAYSOL_VOLATILITY_MULTIPLIER = 1.1

# Market Condition Detection
RAYSOL_ADX_PERIOD = 14
RAYSOL_ADX_THRESHOLD = 25
RAYSOL_SIDEWAYS_THRESHOLD = 15
```

## 📱 Telegram Integration

### Setup Telegram system

1. **Create a system**: Message @systemFather on Telegram
2. **Get system token**: Save the token from systemFather
3. **Get chat ID**: Message your system and visit `https://api.telegram.org/system<TOKEN>/getUpdates`
4. **Configure**: Add credentials to `.env` file

### Notification Types

- 🎯 **Trade Signals**: Entry and exit alerts
- 📊 **Performance Updates**: Daily/weekly summaries
- ⚠️ **Risk Alerts**: Stop loss and margin warnings
- 🤖 **System Status**: system start/stop notifications
- 📈 **Market Analysis**: Trend and sentiment updates

## 📁 Project Structure

```
tradingsystem/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── .env                   # Environment configuration
├── modules/               # Core trading modules
│   ├── binance_client.py  # Binance API client
│   ├── strategies.py      # Trading strategies
│   ├── risk_manager.py    # Risk management
│   ├── websocket_handler.py # Real-time data
│   ├── backtest.py        # Backtesting engine
│   └── config.py          # Configuration management
├── scripts/               # Utility scripts
│   ├── run_system.sh         # Start system script
│   ├── stop_system_manual.sh # Stop system script
│   └── check_system_status.sh # Status check
└── logs/                  # Trading logs
    ├── trading_bot.log    # Main log file
    └── trades.log         # Trade history
```

## 🔍 Monitoring & Logging

### Log Files

- **`logs/trading_bot.log`**: Main application logs
- **`logs/trades.log`**: Detailed trade history
- **`logs/performance.log`**: Performance metrics
- **`logs/errors.log`**: Error tracking

### Real-time Monitoring

```bash
# Monitor main log
tail -f logs/trading_bot.log

# Monitor trades
tail -f logs/trades.log

# Check system status
./check_system_status.sh
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_strategies.py

# Run with coverage
python -m pytest --cov=modules tests/
```

### Integration Tests

```bash
# Test Binance API connection
python tests/test_binance_client.py

# Test WebSocket connection
python tests/test_websocket.py

# Test strategy backtesting
python tests/test_backtest.py
```

## 🔒 Security Best Practices

### API Security

- **Never commit API keys** to version control
- **Use environment variables** for sensitive data
- **Enable IP whitelist** on Binance account
- **Use testnet first** before live trading
- **Regular key rotation** for enhanced security

### System Security

- **Run with limited permissions** (non-root user)
- **Use firewall rules** to restrict network access
- **Regular system updates** and security patches
- **Monitor log files** for suspicious activity

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** with proper testing
4. **Commit your changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation as needed
- Test on testnet before production
- Include detailed commit messages

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT**: This trading system is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. The authors are not responsible for any financial losses incurred through the use of this software. Always:

- **Test thoroughly** on testnet before live trading
- **Start with small amounts** you can afford to lose
- **Understand the risks** of leveraged trading
- **Monitor your positions** regularly
- **Have proper risk management** in place

## 📞 Support & Contact

- **Author**: [Minhajul Islam](https://github.com/minhajulislamme)
- **GitHub**: [https://github.com/minhajulislamme](https://github.com/minhajulislamme)
- **Issues**: [Report bugs and feature requests](https://github.com/minhajulislamme/tradingsystem/issues)

---

<div align="center">

**⭐ If you find this project useful, please give it a star! ⭐**

_Built with ❤️ by [Minhajul Islam](https://github.com/minhajulislamme)_

</div>
