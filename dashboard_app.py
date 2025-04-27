from shiny import App, render, ui
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Load updated data ===
returns_df = pd.read_csv("cumulative_returns_.csv", index_col=0, parse_dates=True)
metrics_df = pd.read_csv("Final_Metrics_Table_.csv", index_col=0)
prices_df = pd.read_csv("prices_.csv", index_col=0, parse_dates=True)

returns_matrix = prices_df.pct_change().dropna()
available_metrics = list(metrics_df.index)

# === UI ===
app_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel(
            "📈 Cumulative Returns",
            ui.row(
                ui.column(6,
                    ui.input_date_range(
                        "date_range",
                        "Select Date Range",
                        start=returns_df.index.min(),
                        end=returns_df.index.max()
                    ),
                    ui.input_checkbox_group(
                        "model_choices",
                        "Select Portfolios/ETFs to Plot",
                        choices=list(returns_df.columns),
                        selected=list(returns_df.columns[:4])
                    ),
                    ui.input_slider(
                        "lambda_slider",
                        "RMT λ₊ Sensitivity (Live Overlay)",
                        min=0.8, max=2.0, value=1.0, step=0.1
                    )
                )
            ),
            ui.output_plot("returns_plot", height="500px")
        ),
        ui.nav_panel(
            "🔬 Eigenvalue Spectrum",
            ui.output_plot("spectrum_plot", height="500px")
        ),
        ui.nav_panel(
            "📊 Risk Metrics Table",
            ui.input_checkbox_group(
                "selected_metrics",
                "Select Metrics to View",
                choices=available_metrics,
                selected=[
                    "Geometric Return", "Sharpe Ratio",
                    "Max Drawdown (%)", "Volatility",
                    "VaR (%)", "CVaR (%)"
                ]
            ),
            ui.output_data_frame("metrics_table")
        )
    )
)

# === Server ===
def server(input, output, session):

    @output
    @render.plot
    def returns_plot():
        df = returns_df.copy()
        df = df.loc[input.date_range()[0]:input.date_range()[1]]
        selected = input.model_choices()

        plt.figure(figsize=(12, 6))
        for col in selected:
            if col in df.columns:
                plt.plot(df.index, df[col], label=col)

        lambda_val = input.lambda_slider()
        if "RMT λ₊ Live" not in df.columns:
            live_curve = df.mean(axis=1) * lambda_val / 2
            plt.plot(df.index, live_curve, label=f"RMT λ₊ Live (λ={lambda_val:.1f})", linestyle="--", color="black")

        plt.title("Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Growth of $1")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    @output
    @render.plot
    def spectrum_plot():
        T, N = returns_matrix.shape
        sample_cov = np.cov(returns_matrix.values, rowvar=False)
        eigenvals = np.sort(np.linalg.eigvalsh(sample_cov))

        q = T / N
        lambda_plus = (sample_cov.trace() / N) * (1 + 1 / np.sqrt(q)) ** 2 * input.lambda_slider()

        plt.figure(figsize=(10, 5))
        plt.plot(eigenvals, marker="o", label="Sorted Eigenvalues")
        plt.axhline(y=lambda_plus, color="red", linestyle="--", label=f"λ₊ (λ={input.lambda_slider():.2f})")
        plt.title("Eigenvalue Spectrum (Live RMT λ₊)")
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

    @output
    @render.data_frame
    def metrics_table():
        selected = input.selected_metrics()
        if selected:
            selected_ordered = [m for m in metrics_df.index if m in selected]
            df = metrics_df.loc[selected_ordered]
        else:
            df = metrics_df.copy()

        df.index.name = "Metric"
        return df

# === Run App ===
app = App(app_ui, server)