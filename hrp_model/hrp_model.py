import numpy as np
import pandas as pd
import yfinance as yf
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import itertools

# Step 1: Data Download and Preprocessing
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            data = data['Adj Close']
        elif 'Close' in data.columns.levels[0]:
            data = data['Close']
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' columns found.")
    else:
        data = data['Adj Close'] if 'Adj Close' in data else data['Close']
    return data.dropna()

def compute_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

# Step 2: Correlation Distance
def correl_distance(corr):
    return np.sqrt(0.5 * (1 - corr))

# Step 3: Quasi-Diagonalization
def get_quasi_diag(linkage_matrix):
    num_items = linkage_matrix.shape[0] + 1
    sort_ix = pd.Series([int(linkage_matrix[-1, 0]), int(linkage_matrix[-1, 1])])
    while sort_ix.max() >= num_items:
        for i in list(sort_ix.index):
            if sort_ix[i] >= num_items:
                idx = int(sort_ix[i] - num_items)
                sort_ix.at[i] = int(linkage_matrix[idx, 0])
                sort_ix = pd.concat([sort_ix, pd.Series([int(linkage_matrix[idx, 1])])], ignore_index=True)
        sort_ix = sort_ix.sort_index().reset_index(drop=True)
    return sort_ix.astype(int).tolist()

# Step 4: Recursive Bisection
def get_cluster_variance(cov, cluster_items):
    cov_sub = cov.loc[cluster_items, cluster_items]
    inv_diag = 1 / np.diag(cov_sub)
    w = inv_diag / inv_diag.sum()
    return float(w @ cov_sub @ w)

def recursive_bisection(cov, ordered_tickers):
    w = pd.Series(1.0, index=ordered_tickers)
    clusters = [ordered_tickers]
    while clusters:
        next_clusters = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            split = len(cluster) // 2
            c1, c2 = cluster[:split], cluster[split:]
            v1 = get_cluster_variance(cov, c1)
            v2 = get_cluster_variance(cov, c2)
            alpha = 1 - v1 / (v1 + v2)
            w[c1] *= alpha
            w[c2] *= (1 - alpha)
            if len(c1) > 1: next_clusters.append(c1)
            if len(c2) > 1: next_clusters.append(c2)
        clusters = next_clusters
    return w / w.sum()

# Step 5: Performance Metrics
def compute_metrics(returns, weights):
    port = returns @ weights
    cum = (1 + port).cumprod()
    mean = port.mean()
    ann = mean * 252
    vol = port.std() * np.sqrt(252)
    sharpe = ann / vol if vol else np.nan
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    max_10d = port.rolling(10).sum().min()
    var95 = np.percentile(port, 5)
    cvar95 = port[port <= var95].mean()
    return {
        "Mean Return":        mean,
        "Annual Return":      ann,
        "Geometric Return":   cum.iloc[-1] ** (1/len(cum)) - 1,
        "Min Return":         port.min(),
        "Max Drawdown (%)":   max_dd * 100,
        "Max 10-Day DD (%)":  max_10d * 100,
        "Sharpe Ratio":       sharpe,
        "Volatility":         vol,
        "Skewness":           port.skew(),
        "Kurtosis":           port.kurt(),
        "VaR (%)":            var95 * 100,
        "CVaR (%)":           cvar95 * 100
    }, port, cum

# Single-period HRP using a chosen linkage (full-history)
def run_hrp_full(tickers, start, end,
                 rebalance_freq="1M", linkage_method="single"):
    prices  = get_data(tickers, start, end)
    returns = compute_returns(prices)
    cov     = returns.cov()
    corr    = returns.corr()
    dist    = correl_distance(corr)
    link    = sch.linkage(dist, method=linkage_method)

    fig, ax = plt.subplots(figsize=(8,4))
    sch.dendrogram(link, labels=returns.columns, ax=ax)
    ax.set_title(f"Dendrogram ({linkage_method})")

    order    = get_quasi_diag(link)
    ordered  = returns.columns[order].tolist()
    weights  = recursive_bisection(cov, ordered)
    metrics, port, cum = compute_metrics(returns, weights)

    spy = compute_returns(get_data(['SPY'], start, end))['SPY'].cumsum()

    return {
        "prices":     prices,
        "returns":    returns,
        "corr":       corr,
        "dist":       dist,
        "dendrogram": fig,
        "weights":    weights,
        "metrics":    metrics,
        "cumulative": cum,
        "benchmark":  spy
    }

# Backtest routine for calibration (rolling rebalances)
def backtest_and_evaluate(returns, lookback, rebalance_freq, linkage_method):
    dates = returns.index
    start = dates[0] + pd.Timedelta(days=lookback)
    rebals = pd.date_range(start=start, end=dates[-1], freq=rebalance_freq)
    rebals = [d for d in rebals if d in dates]
    if not rebals:
        rebals = [dates[lookback]]

    slices = []
    for i, d in enumerate(rebals):
        window = returns.loc[:d].tail(lookback)
        cov  = window.cov()
        corr = window.corr()
        dist = correl_distance(corr)
        link = sch.linkage(dist, method=linkage_method)
        order = get_quasi_diag(link)
        w     = recursive_bisection(cov, window.columns[order])
        w    /= w.sum()

        if i < len(rebals)-1:
            period = returns.loc[d:rebals[i+1]]
        else:
            period = returns.loc[d:]
        slices.append(period.dot(w))

    port = pd.concat(slices).dropna()
    m,_,_ = compute_metrics(port.to_frame(name='Strategy'), pd.Series(1.0, index=['Strategy']))
    # But we want strategy metrics directly:
    # We'll recompute correctly:
    mean = port.mean()*252
    vol  = port.std()*np.sqrt(252)
    sharpe = mean/vol if vol else np.nan
    cum = (1+port).cumprod()
    dd = cum/cum.cummax()-1
    maxdd = dd.min()
    return pd.DataFrame({
        'Strategy': {
            'Sharpe Ratio':      sharpe,
            'Max Drawdown (%)':  maxdd*100
        }
    })

# Calibration grid
def calibrate_hrp(returns, window_years):
    results = []
    for freq, linkage in itertools.product(['1M','3M'], ['single','average','ward']):
        lookback = window_years * 252
        mdf = backtest_and_evaluate(returns, lookback, freq, linkage)
        results.append({
            'Window (yrs)':       window_years,
            'Rebalance':          freq,
            'Linkage':            linkage,
            'Sharpe Ratio':       mdf.loc['Sharpe Ratio','Strategy'],
            'Max Drawdown (%)':   mdf.loc['Max Drawdown (%)','Strategy']
        })
    return pd.DataFrame(results)
