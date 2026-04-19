 About The Project
This is an end-to-end Data Science & Deep Learning project that predicts whether an Indian stock price will go 📈 UP or 📉 DOWN the next trading day.
The system:

Automatically downloads historical stock data from Yahoo Finance
Engineers 35+ technical indicators as features
Trains two AI models side by side — XGBoost & LSTM
Backtests both strategies against Buy & Hold
Delivers a combined prediction signal with confidence %
Saves a full visual dashboard automatically

🖼️ Dashboard Preview
┌─────────────────────────────────────────────┐
│   TCS.NS · Stock Market Predictor Dashboard  │
│         XGBoost  vs  LSTM (PyTorch)          │
├──────────────────────────────────────────────┤
│  Price Chart + SMA + Bollinger Bands         │
├────────────────┬─────────────────────────────┤
│  RSI (14)      │  MACD                       │
├────────────────┼─────────────────────────────┤
│  XGBoost       │  LSTM                       │
│  Confusion     │  Confusion                  │
│  Matrix        │  Matrix                     │
├────────────────┼─────────────────────────────┤
│  Feature       │  LSTM Training              │
│  Importances   │  Loss Curve                 │
├────────────────┴─────────────────────────────┤
│  Backtest: Strategy vs Buy & Hold            │
└──────────────────────────────────────────────┘

⚙️ Tech Stack
CategoryLibraryLanguagePython 3.8+Data CollectionyfinanceData Processingpandas, numpyTechnical IndicatorstaMachine Learningxgboost, scikit-learnDeep LearningPyTorchVisualisationmatplotlib

📊 Features Engineered (35+)
CategoryIndicatorsTrendSMA 20, SMA 50, EMA 20, MACD, ADXMomentumRSI, Stochastic K/D, Williams %R, ROCVolatilityBollinger Bands, ATR, BB WidthVolumeOBV, VWAPPrice ActionCandle body, Upper/Lower shadows, Lag featuresRolling StatsRolling mean, Rolling std deviation

🤖 Models
XGBoost Classifier

Gradient boosting decision tree
300 estimators, max depth 6
Provides feature importance rankings
Fast training, highly interpretable

LSTM Neural Network (PyTorch)

3-layer Long Short-Term Memory network
Looks back at 60 days of history
Batch Normalization + Dropout regularization
Early stopping to prevent overfitting

Combined Signal
XGBoost  → 📈 UP  (UP: 68.3% | DOWN: 31.7%)
LSTM     → 📈 UP  (UP: 71.2% | DOWN: 28.8%)
─────────────────────────────────────────────
COMBINED → 📈 UP  (Avg confidence: 69.7%)

🔧 Configuration
All settings are at the top of stock_predictor.py:
pythonTICKER     = "TCS.NS"       # Change to any NSE stock
START_DATE = "2018-01-01"   # Training data start
END_DATE   = "2024-12-31"   # Training data end
LOOK_BACK  = 60             # Days LSTM looks back
TEST_SPLIT = 0.2            # 20% data for testing
EPOCHS     = 60             # Max training epochs

📦 Supported Stocks
Just change the TICKER variable to any of these:
CompanyTickerTCSTCS.NSReliance IndustriesRELIANCE.NSInfosysINFY.NSHDFC BankHDFCBANK.NSWiproWIPRO.NSTata MotorsTATAMOTORS.NSSBISBIN.NSBajaj FinanceBAJFINANCE.NSAsian PaintsASIANPAINT.NSMaruti SuzukiMARUTI.NS

.NS = National Stock Exchange (NSE). Always keep this suffix!


📁 Project Structure
indian-stock-predictor/
│
├── stock_predictor.py        ← Main script (run this)
├── requirements.txt          ← All dependencies
├── README.md                 ← This file
└── TCS.NS_predictor_dashboard.png  ← Generated after run

📈 Backtesting
The strategy simulates trading with ₹1,00,000 starting capital:

BUY when model predicts UP
SELL when model predicts DOWN
Compared against simple Buy & Hold benchmark


🌱 What I Learned

Real-world financial data collection and cleaning
Technical indicator engineering for stock markets
XGBoost classification model building and tuning
LSTM deep learning with PyTorch for time-series
Backtesting trading strategies programmatically
End-to-end data science project development
