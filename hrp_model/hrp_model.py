import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from dateutil.relativedelta import relativedelta   # ← add this


def get_data(tickers, start_date, end_date):
    """
    Download adjusted close prices and compute daily returns, with fallback
    if 'Adj Close' is not present.

    Returns prices and returns DataFrames.
    """
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
    )
    # Handle multi-index or single-level columns
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        else:
            prices = data.droplevel(0, axis=1)
    else:
        prices = data['Close'] if 'Close' in data.columns else data
    returns = prices.pct_change().dropna()
    return prices, returns


def correlate_distance(corr):
    """
    Convert correlation matrix to distance matrix:
        d_{ij} = sqrt((1 - rho_{ij}) / 2)
    """
    return np.sqrt((1 - corr) / 2)


def get_correlation_matrix(returns):
    """
    Compute correlation matrix from returns DataFrame.
    Returns pandas DataFrame.
    """
    return returns.corr()


def get_distance_matrix(returns):
    """
    Compute distance matrix from returns DataFrame.
    Returns pandas DataFrame.
    """
    corr = returns.corr()
    dist = correlate_distance(corr)
    return pd.DataFrame(dist, index=corr.index, columns=corr.columns)


def get_quasi_diagonal(link):
    """
    Produce a quasi-diagonal ordering of clustered items given a linkage matrix.
    """
    n = link.shape[0] + 1
    def recurse(cid):
        if cid < n:
            return [cid]
        left, right = int(link[cid-n, 0]), int(link[cid-n, 1])
        return recurse(left) + recurse(right)
    return recurse(2*n - 2)


def get_cluster_var(cov, items):
    """Compute cluster variance via inverse-variance portfolio."""
    sub = cov.loc[items, items]
    inv = 1.0 / np.diag(sub.values)
    ivp = inv / inv.sum()
    return ivp @ sub.values @ ivp


def recursive_bisection(cov, sorted_idx):
    """
    Allocate weights recursively across a sorted index list.
    """
    w = pd.Series(1.0, index=sorted_idx)
    def split(items):
        if len(items) <= 1:
            return
        half = len(items) // 2
        left, right = items[:half], items[half:]
        var_l = get_cluster_var(cov, left)
        var_r = get_cluster_var(cov, right)
        alpha = 1 - var_l / (var_l + var_r)
        w[left] *= alpha
        w[right] *= (1 - alpha)
        split(left)
        split(right)
    split(list(sorted_idx))
    return w


def hrp_allocation(returns, method='single'):
    """Compute Hierarchical Risk Parity weights."""
    cov = returns.cov()
    corr = returns.corr()
    dist = correlate_distance(corr)
    cond = squareform(dist.values, checks=False)
    link = linkage(cond, method=method)
    idx = get_quasi_diagonal(link)
    tickers = returns.columns
    ordered = tickers[idx]
    raw = recursive_bisection(cov, ordered)
    return raw / raw.sum()


def equal_weight_returns(returns):
    """Compute equal-weight portfolio returns across columns."""
    return returns.mean(axis=1)


def benchmark_returns(returns, benchmark='SPY'):
    """Extract benchmark return series by ticker symbol."""
    if benchmark in returns.columns:
        return returns[benchmark]
    else:
        raise KeyError(f"Benchmark '{benchmark}' not in returns columns")


def compute_performance_metrics(port_ret, freq=252, dd_window=10, var_level=0.05):
    """Compute performance metrics for a return series."""
    r = port_ret.dropna()
    mean_r = r.mean()
    ann_ret = (1 + mean_r)**freq - 1
    geo_ret = r.add(1).prod()**(1/len(r)) - 1
    min_r = r.min()
    cum = (1 + r).cumprod()
    dd = cum / cum.cummax() - 1
    max_dd = dd.min()
    # short-term drawdowns
    short_dds = []
    for i in range(len(r) - dd_window + 1):
        w = (1 + r.iloc[i:i+dd_window]).cumprod()
        short_dds.append((w / w.cummax() - 1).min())
    max_short_dd = min(short_dds) if short_dds else np.nan
    vol_ann = r.std() * np.sqrt(freq)
    sharpe = ann_ret / vol_ann if vol_ann else np.nan
    skew = r.skew()
    kurt = r.kurtosis()
    var = -np.percentile(r, var_level*100)
    cvar = -r[r <= -var].mean() if any(r <= -var) else np.nan
    return pd.Series({
        'Mean Return': mean_r,
        'Annual Return': ann_ret,
        'Geometric Return': geo_ret,
        'Min Return': min_r,
        'Max Drawdown': max_dd,
        f'Max {dd_window}-Day Drawdown': max_short_dd,
        'Volatility': vol_ann,
        'Sharpe Ratio': sharpe,
        'Skewness': skew,
        'Kurtosis': kurt,
        f'VaR ({int((1-var_level)*100)}%)': var,
        f'CVaR ({int((1-var_level)*100)}%)': cvar,
    })

def walk_forward_validate(returns, method, window, rebalance,
                          train_years=10, test_years=1):
    """
    Perform a walk‐forward test:
      • Train on `train_years`, test on `test_years`, then roll forward by test_years.
    Returns a DataFrame indexed by test‐start date with out‐of‐sample Sharpe ratios.
    """
    results = []
    idx = returns.index
    # ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(idx):
        raise ValueError("returns.index must be datetime")
    start_date = idx[0]
    train_start = start_date
    train_end = train_start + relativedelta(years=train_years)

    while True:
        test_start = train_end
        test_end = test_start + relativedelta(years=test_years)
        if test_end > idx[-1]:
            break

        train_data = returns[(returns.index >= train_start) & (returns.index < test_start)]
        test_data  = returns[(returns.index >= test_start)  & (returns.index < test_end)]

        # skip if not enough data
        if len(train_data) < window or test_data.empty:
            break

        # compute HRP weights on the train slice
        w = hrp_allocation(train_data, method=method)

        # apply to test slice
        port_ret = (test_data * w).sum(axis=1)
        sharpe   = compute_performance_metrics(port_ret)['Sharpe Ratio']

        results.append({
            'Test Start': test_start,
            'Test End':   test_end,
            'Sharpe':     sharpe
        })

        # roll forward
        train_start = train_start + relativedelta(years=test_years)
        train_end   = train_start + relativedelta(years=train_years)

    return pd.DataFrame(results).set_index('Test Start')


def sensitivity_analysis(returns, method, window, rebalance,
                         window_tol=20, rebalance_tol=3):
    """
    Evaluate Sharpe sensitivity to (window ± window_tol) and (rebalance ± rebalance_tol).
    Returns a DataFrame indexed by window values, with rebalance values as columns.
    """
    # build parameter grids
    window_vals = [window - window_tol, window, window + window_tol]
    rebalance_vals = [rebalance - rebalance_tol, rebalance, rebalance + rebalance_tol]
    # filter out non-positive
    window_vals = [w for w in window_vals if w > 0]
    rebalance_vals = [r for r in rebalance_vals if r > 0]

    # container for results
    table = pd.DataFrame(index=window_vals, columns=rebalance_vals, dtype=float)

    for w_ in window_vals:
        for r_ in rebalance_vals:
            # dynamic backtest with params (w_, r_)
            wmat = pd.DataFrame(index=returns.index, columns=returns.columns)
            for i in range(int(w_), len(returns), int(r_)):
                train = returns.iloc[i-int(w_):i]
                wmat.iloc[i] = hrp_allocation(train, method=method)
            wmat.ffill(inplace=True)
            port_ret = (wmat.shift(1) * returns).sum(axis=1)

            # compute Sharpe
            sharpe = compute_performance_metrics(port_ret)['Sharpe Ratio']
            table.at[w_, r_] = sharpe

    table.index.name = 'Window'
    table.columns.name = 'Rebalance'
    return table


if __name__ == '__main__':
    # Example usage
    tickers = ['SPY','IWM','EFA','EEM','AGG','LQD','HYG','TLT',
               'GLD','VNQ','DBC','VT','XLE','XLK','UUP']
    prices, rets = get_data(tickers, '2008-07-01', '2025-03-01')
    w = hrp_allocation(rets)
    port = (rets * w).sum(axis=1)
    eq = equal_weight_returns(rets)
    bm = benchmark_returns(rets)
    metrics_hrp = compute_performance_metrics(port)
    metrics_eq = compute_performance_metrics(eq)
    metrics_bm = compute_performance_metrics(bm)
    print('HRP Metrics:\n', metrics_hrp)
    print('Equal-Weight Metrics:\n', metrics_eq)
    print('Benchmark Metrics:\n', metrics_bm)

    wf = walk_forward_validate(rets, method='complete', window=175, rebalance=13)
    print("\nWalk-forward Sharpe by period:\n", wf)

    sens = sensitivity_analysis(rets, method='complete',
                                window=175, rebalance=13,
                                window_tol=20, rebalance_tol=3)
    print("\nSensitivity matrix of Sharpe:\n", sens)
