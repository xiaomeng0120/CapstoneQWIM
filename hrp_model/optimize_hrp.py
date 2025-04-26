import optuna
from hrp_model import get_data, hrp_allocation, equal_weight_returns, benchmark_returns, compute_performance_metrics

# Load returns once
_, rets = get_data(TICKERS, START, END)

def objective(trial):
    method    = trial.suggest_categorical('method', METHODS)
    window    = trial.suggest_int('window', 60, 756)      # continuous between ~3m and 3y
    rebalance = trial.suggest_int('rebalance', 5, 126)    # weekly up to ~6m

    # run rolling‐window backtest exactly as before
    # compute hrp_ret, eq_ret, bm_ret …
    sharpe_hrp = compute_performance_metrics(hrp_ret)['Sharpe Ratio']

    # optional: enforce outperformance
    sharpe_eq  = compute_performance_metrics(eq_ret)['Sharpe Ratio']
    sharpe_bm  = compute_performance_metrics(bm_ret)['Sharpe Ratio']
    if sharpe_hrp <= max(sharpe_eq, sharpe_bm):
        return -1e6  # penalize any trial that doesn’t beat both benchmarks

    return sharpe_hrp

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    print('Best params:', study.best_params)
    print('Best Sharpe:', study.best_value)
