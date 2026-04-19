"""
=============================================================
  INDIAN STOCK MARKET PREDICTOR — RELIANCE.NS
  Models: LSTM (PyTorch) + XGBoost (ML)
  NO TENSORFLOW — Works on all Python versions!
=============================================================
"""

# ─────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ta

print("✅ All libraries loaded successfully!")
print(f"   PyTorch version  : {torch.__version__}")
print(f"   XGBoost version  : {xgb.__version__}")
print(f"   Device           : {'GPU' if torch.cuda.is_available() else 'CPU'}")


# ─────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────
TICKER      = "TCS.NS"
START_DATE  = "2018-01-01"
END_DATE    = "2024-12-31"
LOOK_BACK   = 60       # days of history LSTM looks at
TEST_SPLIT  = 0.2      # 20% for testing
EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 0.001
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────
# 3. DATA FETCHING
# ─────────────────────────────────────────────
def fetch_data(ticker, start, end):
    print(f"\n📥 Downloading data for {ticker} ...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"   Rows fetched : {len(df)}")
    print(f"   Date range   : {df.index[0].date()} → {df.index[-1].date()}")
    return df


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def add_technical_indicators(df):
    close = df["Close"].squeeze()
    high  = df["High"].squeeze()
    low   = df["Low"].squeeze()
    vol   = df["Volume"].squeeze()

    # Trend
    df["SMA_20"]    = ta.trend.sma_indicator(close, window=20)
    df["SMA_50"]    = ta.trend.sma_indicator(close, window=50)
    df["EMA_20"]    = ta.trend.ema_indicator(close, window=20)
    df["MACD"]      = ta.trend.macd(close)
    df["MACD_sig"]  = ta.trend.macd_signal(close)
    df["MACD_diff"] = ta.trend.macd_diff(close)
    df["ADX"]       = ta.trend.adx(high, low, close, window=14)

    # Momentum
    df["RSI"]       = ta.momentum.rsi(close, window=14)
    df["Stoch_K"]   = ta.momentum.stoch(high, low, close)
    df["Stoch_D"]   = ta.momentum.stoch_signal(high, low, close)
    df["Williams_R"]= ta.momentum.williams_r(high, low, close)
    df["ROC"]       = ta.momentum.roc(close, window=12)

    # Volatility
    bb = ta.volatility.BollingerBands(close, window=20)
    df["BB_upper"]  = bb.bollinger_hband()
    df["BB_lower"]  = bb.bollinger_lband()
    df["BB_width"]  = bb.bollinger_wband()
    df["ATR"]       = ta.volatility.average_true_range(high, low, close)

    # Volume
    df["OBV"]       = ta.volume.on_balance_volume(close, vol)
    df["VWAP"]      = ta.volume.volume_weighted_average_price(high, low, close, vol)

    # Price action
    df["Candle_body"]  = (close - df["Open"]).abs()
    df["Upper_shadow"] = high - df[["Open","Close"]].max(axis=1)
    df["Lower_shadow"] = df[["Open","Close"]].min(axis=1) - low
    df["Price_change"] = close.pct_change()
    df["Vol_change"]   = vol.pct_change()
    df["High_Low_pct"] = (high - low) / low * 100

    # Lag features
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f"Lag_{lag}"] = close.shift(lag)

    # Rolling stats
    df["Rolling_std_10"]  = close.rolling(10).std()
    df["Rolling_mean_10"] = close.rolling(10).mean()

    # TARGET: 1 = UP tomorrow, 0 = DOWN
    df["Target"] = (close.shift(-1) > close).astype(int)

    df.dropna(inplace=True)
    print(f"   Features engineered : {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────
# 5. XGBOOST MODEL
# ─────────────────────────────────────────────
def train_xgboost(df):
    print("\n" + "="*50)
    print("🌲  XGBOOST MODEL")
    print("="*50)

    feature_cols = [c for c in df.columns
                    if c not in ["Target","Open","High","Low","Close","Volume"]]
    X = df[feature_cols].values
    y = df["Target"].values

    split = int(len(X) * (1 - TEST_SPLIT))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)

    print(f"\n   Test Accuracy : {acc*100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=['DOWN','UP'])}")

    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.nlargest(10)

    return model, y_test, y_pred, y_proba, top10, acc, feature_cols, split


# ─────────────────────────────────────────────
# 6. PYTORCH LSTM MODEL
# ─────────────────────────────────────────────

# ── 6a. Model architecture ────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.fc1        = nn.Linear(hidden_size, 64)
        self.relu       = nn.ReLU()
        self.dropout    = nn.Dropout(0.2)
        self.fc2        = nn.Linear(64, 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out     = out[:, -1, :]          # take last timestep
        out     = self.batch_norm(out)
        out     = self.fc1(out)
        out     = self.relu(out)
        out     = self.dropout(out)
        out     = self.fc2(out)
        out     = self.sigmoid(out)
        return out.squeeze()


# ── 6b. Prepare sequences ─────────────────────
def prepare_lstm_sequences(df, look_back):
    feature_cols = [c for c in df.columns
                    if c not in ["Target","Open","High","Low","Close","Volume"]]
    X_raw = df[feature_cols].values
    y_raw = df["Target"].values

    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    X_seq, y_seq = [], []
    for i in range(look_back, len(X_scaled)):
        X_seq.append(X_scaled[i - look_back:i])
        y_seq.append(y_raw[i])

    return np.array(X_seq), np.array(y_seq), scaler, feature_cols


# ── 6c. Train LSTM ────────────────────────────
def train_lstm(df):
    print("\n" + "="*50)
    print("🧠  LSTM DEEP LEARNING MODEL  (PyTorch)")
    print("="*50)

    X_seq, y_seq, scaler, feature_cols = prepare_lstm_sequences(df, LOOK_BACK)

    split    = int(len(X_seq) * (1 - TEST_SPLIT))
    X_train  = torch.FloatTensor(X_seq[:split]).to(DEVICE)
    X_test   = torch.FloatTensor(X_seq[split:]).to(DEVICE)
    y_train  = torch.FloatTensor(y_seq[:split]).to(DEVICE)
    y_test_t = torch.FloatTensor(y_seq[split:]).to(DEVICE)

    print(f"   Train samples : {len(X_train)}")
    print(f"   Test  samples : {len(X_test)}")
    print(f"   Input shape   : {X_train.shape[1:]}")

    train_ds     = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_size = X_train.shape[2]
    model      = LSTMModel(input_size=input_size).to(DEVICE)
    criterion  = nn.BCELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
                 optimizer, factor=0.5, patience=5)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 10

    print(f"\n   Training for up to {EPOCHS} epochs ...")
    for epoch in range(EPOCHS):
        # ── Train ──
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test)
            val_loss  = criterion(val_preds, y_test_t).item()

        train_losses.append(avg_train)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch [{epoch+1:3d}/{EPOCHS}]  "
                  f"Train Loss: {avg_train:.4f}  "
                  f"Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_lstm.pt")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"   ⏹  Early stopping at epoch {epoch+1}")
                break

    # Load best weights
    model.load_state_dict(torch.load("best_lstm.pt", map_location=DEVICE))

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_proba = model(X_test).cpu().numpy()
    y_pred = (y_proba > 0.5).astype(int)
    y_test_np = y_seq[split:]
    acc = accuracy_score(y_test_np, y_pred)

    print(f"\n   Test Accuracy : {acc*100:.2f}%")
    print(f"\n{classification_report(y_test_np, y_pred, target_names=['DOWN','UP'])}")

    history = {"loss": train_losses, "val_loss": val_losses}
    return model, y_test_np, y_pred, y_proba, history, acc, scaler, feature_cols


# ─────────────────────────────────────────────
# 7. BACKTESTING
# ─────────────────────────────────────────────
def backtest(df, predictions, split_idx, model_name):
    print(f"\n📈 Backtesting {model_name} ...")

    prices      = df["Close"].values.flatten()
    test_prices = prices[split_idx: split_idx + len(predictions)]

    capital         = 100_000
    initial_capital = capital
    position        = 0
    portfolio       = []
    buy_and_hold    = []
    bh_shares       = capital / test_prices[0]

    for i, pred in enumerate(predictions):
        price  = test_prices[i]
        bh_val = bh_shares * price

        if pred == 1 and position == 0:
            position = capital / price
            capital  = 0
        elif pred == 0 and position > 0:
            capital  = position * price
            position = 0

        current_val = capital + (position * price)
        portfolio.append(current_val)
        buy_and_hold.append(bh_val)

    if position > 0:
        capital = position * test_prices[-1]
    portfolio[-1] = capital

    total_return = (capital - initial_capital) / initial_capital * 100
    bh_return    = (buy_and_hold[-1] - initial_capital) / initial_capital * 100

    print(f"   {model_name} Return   : {total_return:+.2f}%")
    print(f"   Buy-&-Hold Return : {bh_return:+.2f}%")
    print(f"   Final Capital     : ₹{capital:,.0f}")

    return {
        "model_name"   : model_name,
        "portfolio"    : portfolio,
        "buy_and_hold" : buy_and_hold,
        "total_return" : total_return,
        "bh_return"    : bh_return,
        "test_prices"  : test_prices,
    }


# ─────────────────────────────────────────────
# 8. VISUALISATION
# ─────────────────────────────────────────────
def plot_all(df, xgb_results, lstm_results, bt_xgb, bt_lstm):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 26))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            hspace=0.55, wspace=0.35,
                            top=0.95, bottom=0.04,
                            left=0.07, right=0.96)

    ACCENT  = "#00d4aa"
    ACCENT2 = "#ff6b6b"
    ACCENT3 = "#ffd93d"
    GRID_C  = "#1e2433"
    TEXT_C  = "#c9d1d9"

    def style_ax(ax, title):
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors=TEXT_C, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)
        ax.yaxis.grid(True, color=GRID_C, linewidth=0.5)
        ax.set_title(title, color=ACCENT, fontsize=11,
                     fontweight="bold", pad=10)

    # ── Row 0: Price + indicators ──────────────
    ax0   = fig.add_subplot(gs[0, :])
    close = df["Close"].values.flatten()
    dates = df.index

    ax0.plot(dates, close, color=ACCENT, lw=1.5, label="Close Price")
    ax0.plot(dates, df["SMA_20"].values, color=ACCENT3, lw=1, alpha=0.7, label="SMA 20")
    ax0.plot(dates, df["SMA_50"].values, color=ACCENT2, lw=1, alpha=0.7, label="SMA 50")
    ax0.fill_between(dates, df["BB_upper"].values, df["BB_lower"].values,
                     alpha=0.08, color=ACCENT, label="Bollinger Bands")
    style_ax(ax0, f"{TICKER} — Price Action with Indicators")
    ax0.legend(facecolor="#161b22", edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)
    ax0.set_ylabel("Price (₹)", color=TEXT_C)

    # ── Row 1: RSI | MACD ──────────────────────
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(dates, df["RSI"].values, color=ACCENT3, lw=1.2)
    ax1.axhline(70, color=ACCENT2, lw=0.8, linestyle="--", alpha=0.7)
    ax1.axhline(30, color=ACCENT,  lw=0.8, linestyle="--", alpha=0.7)
    ax1.fill_between(dates, df["RSI"].values, 70,
                     where=(df["RSI"].values >= 70), color=ACCENT2, alpha=0.2)
    ax1.fill_between(dates, df["RSI"].values, 30,
                     where=(df["RSI"].values <= 30), color=ACCENT,  alpha=0.2)
    style_ax(ax1, "RSI (14)")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("RSI", color=TEXT_C)

    ax2       = fig.add_subplot(gs[1, 1])
    macd_diff = df["MACD_diff"].values
    colors_m  = [ACCENT if v >= 0 else ACCENT2 for v in macd_diff]
    ax2.bar(dates, macd_diff, color=colors_m, width=1.5, alpha=0.8)
    ax2.plot(dates, df["MACD"].values,     color=ACCENT3, lw=1, label="MACD")
    ax2.plot(dates, df["MACD_sig"].values, color=ACCENT2, lw=1, label="Signal")
    style_ax(ax2, "MACD")
    ax2.legend(facecolor="#161b22", edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

    # ── Row 2: Confusion matrices ───────────────
    xgb_model, xgb_y_test, xgb_y_pred, _, xgb_top10, xgb_acc, _, _ = xgb_results
    lstm_model, lstm_y_test, lstm_y_pred, _, _, lstm_acc, _, _        = lstm_results

    for col, (y_t, y_p, name, acc) in enumerate([
        (xgb_y_test,  xgb_y_pred,  "XGBoost", xgb_acc),
        (lstm_y_test, lstm_y_pred, "LSTM",    lstm_acc)
    ]):
        ax  = fig.add_subplot(gs[2, col])
        cm  = confusion_matrix(y_t, y_p)
        disp = ConfusionMatrixDisplay(cm, display_labels=["DOWN","UP"])
        disp.plot(ax=ax, colorbar=False, cmap="YlGn" if col == 0 else "PuBu")
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors=TEXT_C, labelsize=9)
        ax.set_title(f"{name} Confusion Matrix  |  Accuracy: {acc*100:.1f}%",
                     color=ACCENT, fontsize=11, fontweight="bold", pad=10)
        ax.xaxis.label.set_color(TEXT_C)
        ax.yaxis.label.set_color(TEXT_C)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_C)

    # ── Row 3: Feature importance | LSTM loss ───
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.barh(range(len(xgb_top10)), xgb_top10.values, color=ACCENT, alpha=0.85)
    ax5.set_yticks(range(len(xgb_top10)))
    ax5.set_yticklabels(xgb_top10.index, fontsize=8)
    style_ax(ax5, "XGBoost — Top 10 Feature Importances")
    ax5.set_xlabel("Importance", color=TEXT_C, fontsize=9)

    ax6  = fig.add_subplot(gs[3, 1])
    hist = lstm_results[4]
    ep   = range(1, len(hist["loss"]) + 1)
    ax6.plot(ep, hist["loss"],     color=ACCENT,  lw=1.5, label="Train Loss")
    ax6.plot(ep, hist["val_loss"], color=ACCENT2, lw=1.5, label="Val Loss")
    style_ax(ax6, "LSTM — Training Loss Curve")
    ax6.set_xlabel("Epoch", color=TEXT_C, fontsize=9)
    ax6.set_ylabel("Loss",  color=TEXT_C, fontsize=9)
    ax6.legend(facecolor="#161b22", edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=8)

    # ── Row 4: Backtest ─────────────────────────
    ax7 = fig.add_subplot(gs[4, :])
    ax7.plot(bt_xgb["portfolio"],
             color=ACCENT3, lw=1.8,
             label=f"XGBoost ({bt_xgb['total_return']:+.1f}%)")
    ax7.plot(bt_lstm["portfolio"],
             color=ACCENT, lw=1.8,
             label=f"LSTM ({bt_lstm['total_return']:+.1f}%)")
    ax7.plot(bt_xgb["buy_and_hold"],
             color=ACCENT2, lw=1.2, linestyle="--",
             label=f"Buy & Hold ({bt_xgb['bh_return']:+.1f}%)")
    ax7.axhline(100_000, color="#555", lw=0.8, linestyle=":")
    style_ax(ax7, "📊  Backtesting — Strategy Returns vs Buy & Hold  (₹1,00,000 initial capital)")
    ax7.set_xlabel("Trading Days (Test Period)", color=TEXT_C)
    ax7.set_ylabel("Portfolio Value (₹)", color=TEXT_C)
    ax7.legend(facecolor="#161b22", edgecolor=GRID_C, labelcolor=TEXT_C, fontsize=9)

    fig.suptitle(f"{TICKER}  ·  Stock Market Predictor Dashboard\n"
                 "XGBoost  vs  LSTM  (PyTorch)",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    out_path = f"{TICKER}_predictor_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\n✅ Dashboard saved → {out_path}")
    plt.close()


# ─────────────────────────────────────────────
# 9. PREDICT NEXT DAY
# ─────────────────────────────────────────────
def predict_next_day(df, xgb_model, lstm_model, feature_cols, scaler):
    print("\n" + "="*50)
    print("🔮  NEXT DAY PREDICTION")
    print("="*50)

    last_close = float(df["Close"].iloc[-1])
    print(f"   Last closing price : ₹{last_close:,.2f}")

    # XGBoost
    X_last    = df[feature_cols].iloc[-1].values.reshape(1, -1)
    xgb_pred  = xgb_model.predict(X_last)[0]
    xgb_proba = xgb_model.predict_proba(X_last)[0]

    # LSTM (PyTorch)
    X_scaled = scaler.transform(df[feature_cols].values)
    X_last_seq = torch.FloatTensor(
        X_scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, -1)
    ).to(DEVICE)

    lstm_model.eval()
    with torch.no_grad():
        lstm_proba_val = float(lstm_model(X_last_seq).cpu().numpy())
    lstm_pred = 1 if lstm_proba_val > 0.5 else 0

    direction = {1: "📈  UP", 0: "📉  DOWN"}

    print(f"\n   XGBoost → {direction[xgb_pred]}"
          f"  (UP: {xgb_proba[1]*100:.1f}%  |  DOWN: {xgb_proba[0]*100:.1f}%)")
    print(f"   LSTM    → {direction[lstm_pred]}"
          f"  (UP: {lstm_proba_val*100:.1f}%  |  DOWN: {(1-lstm_proba_val)*100:.1f}%)")

    avg_up   = (xgb_proba[1] + lstm_proba_val) / 2
    combined = "📈  UP" if avg_up > 0.5 else "📉  DOWN"
    print(f"\n   🤝 COMBINED SIGNAL → {combined}  (Avg confidence: {avg_up*100:.1f}%)")
    print(f"\n   ⚠️  Educational only — NOT financial advice.")


# ─────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    # Step 1 — Fetch
    raw_df = fetch_data(TICKER, START_DATE, END_DATE)

    # Step 2 — Features
    df = add_technical_indicators(raw_df.copy())

    # Step 3 — XGBoost
    xgb_out = train_xgboost(df)
    (xgb_model, xgb_y_test, xgb_y_pred,
     xgb_y_proba, xgb_top10, xgb_acc,
     feature_cols, xgb_split) = xgb_out

    # Step 4 — LSTM (PyTorch)
    lstm_out = train_lstm(df)
    (lstm_model, lstm_y_test, lstm_y_pred,
     lstm_y_proba, lstm_history, lstm_acc,
     scaler, lstm_feature_cols) = lstm_out

    # Step 5 — Backtest
    lstm_split = xgb_split + LOOK_BACK
    bt_xgb  = backtest(df, xgb_y_pred,  xgb_split,  "XGBoost")

    # Trim lstm predictions to match available prices
    prices      = df["Close"].values.flatten()
    available   = len(prices) - lstm_split
    lstm_y_pred_trimmed = lstm_y_pred[:available]
    bt_lstm = backtest(df, lstm_y_pred_trimmed, lstm_split, "LSTM")

    # Step 6 — Next day prediction
    predict_next_day(df, xgb_model, lstm_model, feature_cols, scaler)

    # Step 7 — Dashboard
    print("\n🎨 Generating dashboard ...")
    plot_all(df, xgb_out, lstm_out, bt_xgb, bt_lstm)

    print("\n🎉  Pipeline complete!")


if __name__ == "__main__":
    main()