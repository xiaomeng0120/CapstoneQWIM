import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as sch
import optuna
from shiny import App, ui, render, reactive
import hrp_model

# ---- UI ----
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h2('Hierarchical Risk Parity Dashboard'),
        ui.input_text('tickers', 'Tickers (comma-separated)',
            value=','.join(['SPY','IWM','EFA','EEM','AGG','LQD','HYG','TLT',
                             'GLD','VNQ','DBC','VT','XLE','XLK','UUP'])
        ),
        ui.input_date_range('date_range', 'Date range',
            start='2008-07-01', end='2025-03-01'
        ),
        ui.input_select('link_method', 'Linkage Method',
            choices=['single','complete','average','ward'], selected='complete'
        ),
        ui.input_slider('window', 'Estimation Window (days)',
            min=60, max=756, value=175, step=1
        ),
        ui.input_slider('rebalance', 'Rebalance Frequency (days)',
            min=5, max=126, value=13, step=1
        ),
        ui.input_action_button('run', 'Run HRP'),
        ui.input_action_button('optimize', 'Bayesian Optimize HRP Setting')
    ),
    ui.h2('Results'),
    ui.navset_tab(
        ui.nav_panel('Asset Relationships', ui.output_plot('corr_heatmap'), ui.output_plot('dist_heatmap')),
        ui.nav_panel('Dendrogram', ui.output_plot('dendrogram')),
        ui.nav_panel('Weights', ui.output_table('weights_table'), ui.output_plot('weights_plot')),
        ui.nav_panel('Performance', ui.output_plot('cumret_plot'), ui.output_plot('drawdown_plot')),
        ui.nav_panel('Metrics', ui.output_table('metrics_table')),
        ui.nav_panel('Top Configuration',
                     ui.HTML(
                        '<table style="width:50%; margin-bottom:1em;">'
                        '<tr><th>Parameter</th><th>Value</th></tr>'
                        '<tr><td>method</td><td>complete</td></tr>'
                        '<tr><td>window</td><td>175</td></tr>'
                        '<tr><td>rebalance</td><td>13</td></tr>'
                        '<tr><td>Sharpe HRP</td><td>0.908908</td></tr>'
                        '</table>'
                    ),
                     ui.h5('Click Bayesian Optimize HRP Setting button to achieve the highest Sharpe ratio.'),
                     ui.p('This may take a few minutes.'),
                     ui.output_ui('opt_status'),
                     ui.output_table('opt_table', spinner=True)),
        ui.nav_panel(
            'Validation & Sensitivity',
            ui.p('This part may take a few time to show the results.'),
            ui.h4('Out-of-Sample Walk-Forward Sharpe'),
            ui.output_table('wf_table'),
            ui.h4('Parameter Sensitivity Heatmap (Sharpe)'),
            ui.output_plot('sens_plot')
        )
    )
)

# 
# ---- Server ----
def server(input, output, session):
    # Reactive returns
    @reactive.Calc
    @reactive.event(input.run, input.optimize)
    def rets():
        tickers = [t.strip() for t in input.tickers().split(',')]
        start, end = input.date_range()
        _, returns = hrp_model.get_data(tickers, start, end)
        return returns

    # Linkage
    @reactive.Calc
    @reactive.event(input.run)
    def link_matrix():
        corr = rets().corr()
        dist = hrp_model.correlate_distance(corr)
        return sch.linkage(squareform(dist.values, checks=False), method=input.link_method())

    # Backtest
    @reactive.Calc
    @reactive.event(input.run)
    def backtest():
        returns = rets()
        w = pd.DataFrame(np.nan,index=returns.index, columns=returns.columns, dtype=float)
        window, freq = input.window(), input.rebalance()
        for i in range(window, len(returns), freq):
            train = returns.iloc[i-window:i]
            w.iloc[i] = hrp_model.hrp_allocation(train, method=input.link_method())
        w.ffill(inplace=True)
        hrp_ret = (w.shift(1) * returns).sum(axis=1)
        eq_ret  = hrp_model.equal_weight_returns(returns)
        bm_ret  = hrp_model.benchmark_returns(returns)
        return hrp_ret, eq_ret, bm_ret

    # Status message for optimization
    opt_status_msg = reactive.Value('')

    # Heatmaps
    @output
    @render.plot
    @reactive.event(input.run)
    def corr_heatmap():
        corr = hrp_model.get_correlation_matrix(rets())
        fig, ax = plt.subplots()
        cax = ax.matshow(corr, vmin=-1, vmax=1)
        plt.colorbar(cax, ax=ax)
        labels = corr.columns
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', fontsize=6,
                        color='white' if abs(corr.iloc[i,j])>0.5 else 'black')
        ax.set_title('Correlation Matrix')
        plt.tight_layout()
        return fig

    @output
    @render.plot
    @reactive.event(input.run)
    def dist_heatmap():
        dist = hrp_model.get_distance_matrix(rets())
        fig, ax = plt.subplots()
        cax = ax.matshow(dist, vmin=0, vmax=1)
        plt.colorbar(cax, ax=ax)
        labels = dist.columns
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{dist.iloc[i,j]:.2f}", ha='center', va='center', fontsize=6,
                        color='white' if dist.iloc[i,j]>0.5 else 'black')
        ax.set_title('Distance Matrix')
        plt.tight_layout()
        return fig

    # Dendrogram
    @output
    @render.plot
    @reactive.event(input.run)
    def dendrogram():
        fig, ax = plt.subplots()
        sch.dendrogram(link_matrix(), labels=rets().columns.tolist(), ax=ax)
        ax.set_title('Asset Clustering Dendrogram')
        plt.tight_layout()
        return fig

    # Weights
    @output
    @render.table
    @reactive.event(input.run)
    def weights_table():
        w = hrp_model.hrp_allocation(rets(), method=input.link_method())
        return pd.DataFrame({'Ticker': w.index, 'Weight': w.values})

    @output
    @render.plot
    @reactive.event(input.run)
    def weights_plot():
        w = hrp_model.hrp_allocation(rets(), method=input.link_method())
        fig, ax = plt.subplots()
        w.sort_values().plot.barh(ax=ax)
        ax.set_xlabel('Weight')
        ax.set_title('HRP Portfolio Weights')
        plt.tight_layout()
        return fig

    # Performance
    @output
    @render.plot
    @reactive.event(input.run)
    def cumret_plot():
        hrp_ret, eq_ret, bm_ret = backtest()
        fig, ax = plt.subplots()
        (1+hrp_ret).cumprod().plot(ax=ax, label='HRP')
        (1+eq_ret).cumprod().plot(ax=ax, label='Equal Weight')
        (1+bm_ret).cumprod().plot(ax=ax, label='Benchmark')
        ax.legend(); ax.set_title('Cumulative Returns'); plt.tight_layout()
        return fig

    @output
    @render.plot
    @reactive.event(input.run)
    def drawdown_plot():
        hrp_ret, eq_ret, bm_ret = backtest()
        cum = lambda s: (1+s).cumprod()
        dd = lambda s: cum(s)/cum(s).cummax()-1
        fig, ax = plt.subplots()
        dd(hrp_ret).plot(ax=ax, label='HRP')
        dd(eq_ret).plot(ax=ax, label='Equal Weight')
        dd(bm_ret).plot(ax=ax, label='Benchmark')
        ax.legend(); ax.set_title('Drawdown'); plt.tight_layout()
        return fig

    # Metrics Table
    @output
    @render.table
    @reactive.event(input.run)
    def metrics_table():
        hrp_ret, eq_ret, bm_ret = backtest()
        met_hrp = hrp_model.compute_performance_metrics(hrp_ret)
        met_eq  = hrp_model.compute_performance_metrics(eq_ret)
        met_bm  = hrp_model.compute_performance_metrics(bm_ret)
        df = pd.DataFrame({'HRP': met_hrp, 'Equal Weight': met_eq, 'Benchmark': met_bm})
        df.reset_index(inplace=True); df.rename(columns={'index':'Metric'}, inplace=True)
        return df

    # Render optimization status
    @output
    @render.ui
    @reactive.event(input.optimize)
    def opt_status():
        return ui.p(opt_status_msg())

    # Bayesian Optimization with seed, constraint, callback
    @reactive.Calc
    @reactive.event(input.optimize)
    def optimize():
        total_trials = 50
        # configure deterministic sampler
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        
        def callback(study, trial):
            current = trial.number + 1
            best = study.best_value
            opt_status_msg.set(f'Trial {current}/{total_trials}: Best Sharpe {best:.2f}')
        
        def objective(trial):
            method    = trial.suggest_categorical('method', ['single','complete','average','ward'])
            window    = trial.suggest_int('window', 120, 200)
            rebalance = trial.suggest_int('rebalance', 5, 63)
            w = pd.DataFrame(np.nan,index=rets().index, columns=rets().columns,dtype=float)
            for i in range(window, len(rets()), rebalance):
                train = rets().iloc[i-window:i]
                w.iloc[i] = hrp_model.hrp_allocation(train, method=method)
            w.ffill(inplace=True)
            hrp_ret = (w.shift(1) * rets()).sum(axis=1)
            s_hrp = hrp_model.compute_performance_metrics(hrp_ret)['Sharpe Ratio']
            s_eq  = hrp_model.compute_performance_metrics(hrp_model.equal_weight_returns(rets()))['Sharpe Ratio']
            s_bm  = hrp_model.compute_performance_metrics(hrp_model.benchmark_returns(rets()))['Sharpe Ratio']
            return s_hrp if (s_hrp > s_eq and s_hrp > s_bm) else -1e3
        
        opt_status_msg.set('Starting Bayesian optimization...')
        study.optimize(objective, n_trials=total_trials, callbacks=[callback])
        params, value = study.best_params, study.best_value
        opt_status_msg.set(f'Done! Best Sharpe: {value:.2f}')
        df = pd.DataFrame({
            'Parameter': list(params.keys()) + ['Sharpe HRP'],
            'Value':     list(params.values()) + [value]
        })
        return df

    @output
    @render.table
    @reactive.event(input.optimize)
    def opt_table():
        return optimize()
    
    # Walk-forward validation table
    @output
    @render.table
    @reactive.event(input.run)
    def wf_table():
        returns   = rets()
        method    = input.link_method()
        window    = input.window()
        rebalance = input.rebalance()
        return hrp_model.walk_forward_validate(
            returns, method, window, rebalance,
            train_years=10, test_years=1
        )

    # Sensitivity heatmap plot
    @output
    @render.plot
    @reactive.event(input.run)
    def sens_plot():
        returns   = rets()
        method    = input.link_method()
        window    = input.window()
        rebalance = input.rebalance()
        sens = hrp_model.sensitivity_analysis(
            returns, method, window, rebalance,
            window_tol=20, rebalance_tol=3
        )
        fig, ax = plt.subplots()
        cax = ax.matshow(sens.values, aspect='auto', vmin=sens.values.min(), vmax=sens.values.max())
        fig.colorbar(cax, ax=ax)
        ax.set_xticks(range(len(sens.columns)))
        ax.set_xticklabels(sens.columns, rotation=90)
        ax.set_yticks(range(len(sens.index)))
        ax.set_yticklabels(sens.index)
        ax.set_xlabel('Rebalance (days)')
        ax.set_ylabel('Window (days)')
        ax.set_title('Sharpe Sensitivity Heatmap')
        plt.tight_layout()
        return fig

# ---- App ----
app = App(app_ui, server)
