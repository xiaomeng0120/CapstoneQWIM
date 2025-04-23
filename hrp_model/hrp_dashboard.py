import datetime
from shiny import App, ui, render, reactive
from hrp_model import run_hrp_full, calibrate_hrp
import pandas as pd
import matplotlib.pyplot as plt

app_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel("ETF Info",
            ui.input_text("tickers", "Tickers (comma-separated)",
                          value="SPY,IWM,EFA,EEM,AGG,LQD,HYG,TLT,GLD,VNQ,DBC,VT,XLE,XLK,UUP"),
            ui.input_date_range("daterange", "Select Date Range",
                                start=datetime.date(2008, 7, 1),
                                end=datetime.date(2025, 3, 1)),
            ui.output_table("info_prices"),
            ui.input_text("focus_ticker", "Plot Specific Ticker:", value="SPY"),
            ui.output_plot("price_plot"),
            ui.output_plot("log_return_plot")
        ),

        ui.nav_panel("Correlation & Distance",
            ui.layout_columns(
                ui.output_plot("corr_heatmap", height="800px", width="800px"),
                ui.output_plot("dist_heatmap", height="800px", width="800px")
            )
        ),

        ui.nav_panel("Clustering Dendrogram",
            ui.output_plot("dendro_plot")
        ),

        ui.nav_panel("Calibration",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_slider("calib_window", "Lookback Window (years)",
                                    min=1, max=5, value=2, step=1),
                    ui.input_select("chosen_rebalance", "Use Rebalance:",
                                    {"1M": "1M", "3M": "3M"},
                                    selected=None),
                    ui.input_select("chosen_linkage", "Use Linkage:",
                                    {"single": "single",
                                     "average": "average",
                                     "ward": "ward"},
                                    selected=None),
                    ui.input_action_button("run_calib", "Run Calibration")
                ),
                ui.output_table("calibration_table")
            )
        ),

        ui.nav_panel("HRP Weights & Performance",
            ui.layout_columns(
                ui.output_table("weights_table", width=5),
            ),
            ui.output_plot("cumulative_plot")
        ),

        ui.nav_panel("Risk Metrics",
            ui.output_table("metric_table")
        )
    )
)

def server(input, output, session):
    # reactive.Value to hold the best calibration row (a pd.Series)
    best_calib = reactive.Value(None)

    @reactive.calc
    def base_results():
        tickers = [t.strip().upper() for t in input.tickers().split(',')]
        start, end = [d.strftime("%Y-%m-%d") for d in input.daterange()]
        return run_hrp_full(tickers, start, end)

    @reactive.event(input.run_calib)
    def calibration_results():
        df = calibrate_hrp(base_results()["returns"],
                           window_years=input.calib_window())
        # pick the row with highest Sharpe
        best = df.sort_values("Sharpe Ratio", ascending=False).iloc[0]
        best_calib.set(best)
        return df

    @output
    @render.table
    def calibration_table():
        return calibration_results()

    def get_hrp_runner():
        base = base_results()
        # 1) read dropdowns; they return either None or a str
        reb = input.chosen_rebalance()
        lin = input.chosen_linkage()

        # 2) if still None, fall back to calibration
        best = best_calib.get()
        if reb is None and best is not None:
            reb = best["Rebalance"]
        if lin is None and best is not None:
            lin = best["Linkage"]

        # 3) build kwargs only when we have valid strings
        kwargs = {}
        if isinstance(reb, str):
            kwargs["rebalance_freq"] = reb
        if isinstance(lin, str):
            kwargs["linkage_method"] = lin

        return run_hrp_full(
            list(base["prices"].columns),
            base["prices"].index[0].strftime("%Y-%m-%d"),
            base["prices"].index[-1].strftime("%Y-%m-%d"),
            **kwargs
        )

    @output
    @render.table
    def info_prices():
        df = base_results()["prices"].copy()
        df = df.reset_index()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
        df.index.name = None
        return df.head()



    @output
    @render.plot
    def price_plot():
        fig, ax = plt.subplots()
        t = input.focus_ticker().strip().upper()
        base_results()["prices"][t].plot(ax=ax, title=f"{t} Close Price")
        return fig

    @output
    @render.plot
    def log_return_plot():
        fig, ax = plt.subplots()
        t = input.focus_ticker().strip().upper()
        base_results()["returns"][t].plot(ax=ax, title=f"{t} Log Returns")
        return fig

    @output
    @render.plot
    def corr_heatmap():
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(base_results()["corr"], annot=True, fmt=".2f",
                    cmap="coolwarm", square=True, ax=ax)
        ax.set_title("Correlation Matrix")
        return fig

    @output
    @render.plot
    def dist_heatmap():
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(base_results()["dist"], annot=True, fmt=".2f",
                    cmap="viridis", square=True, ax=ax)
        ax.set_title("Distance Matrix")
        return fig

    @output
    @render.plot
    def dendro_plot():
        return base_results()["dendrogram"]

    @output
    @render.table
    def weights_table():
        w = get_hrp_runner()["weights"]
        return pd.DataFrame({"Asset": w.index, "Weight": w.values}).round(4)

    @output
    @render.plot
    def cumulative_plot():
        hrp = get_hrp_runner()
        fig, ax = plt.subplots(figsize=(10,5))
        hrp["cumulative"].plot(ax=ax, label="HRP Portfolio", lw=2)
        hrp["benchmark"].plot(ax=ax, label="SPY Benchmark", lw=2, linestyle="--")
        ax.set_title("Cumulative Returns: HRP vs SPY")
        ax.legend()
        return fig

    @output
    @render.table
    def metric_table():
        m = get_hrp_runner()["metrics"]
        df = pd.DataFrame(m, index=["Value"]).T.round(4)
        df.index.name = "Metric"
        return df.reset_index()

app = App(app_ui, server)
