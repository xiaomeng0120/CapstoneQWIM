"""
QWIM Dashboard Main Application
===============================

This is the main entry point for the QWIM Dashboard application,
a Shiny dashboard for visualizing and analyzing multiple time series.

The dashboard is organized into four tabs:
- Inputs: For parameter selection and data filtering
- Data Analysis: For exploratory data analysis
- Results: For displaying final visualizations and insights
- Portfolios: For comparing and analyzing portfolio data

Modules
-------
inputs_module
    Handles data selection and parameter input
analysis_module
    Provides analytical tools and visualizations
results_module
    Displays final results and insights
portfolios_module
    Provides portfolio comparison and analysis tools

Functions
---------
.. autosummary::
   :toctree: generated/

   get_data
   get_series_names
   app_Server

Classes
-------
App
    Main Shiny application instance

Usage
-----
Run this script directly to start the dashboard:

.. code-block:: bash

   python main_App.py

Or import the app object to use in another script:

.. code-block:: python

   from src.dashboard.main_App import app
   app.run()

Author
------
QWIM Team

Version
-------
1.0.0 (2025-03-20)
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from shinywidgets import render_plotly, output_widget, render_widget

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Configure logging with file output
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist
log_file = log_dir / f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("QWIM-Dashboard")

# Add project root to path
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

import numpy as np
import polars as pl
import shinyswatch
from shiny import App, reactive, render, ui

# Import modules
from src.dashboard.modules.analysis_module import analysis_server, analysis_ui
from src.dashboard.modules.inputs_module import inputs_server, inputs_ui
from src.dashboard.modules.black_litterman import model1_ui, model1_server
#from src.dashboard.modules.model2 import model2_ui, model2_server
#from src.dashboard.modules.model3 import model3_ui, model3_server
from src.dashboard.modules.cvar_portfolio import model4_ui, model4_server

def get_data():
    """Load the time series data from CSV file.
    
    Reads the time series data from a predefined CSV file path and
    ensures the date column is correctly formatted.
    
    Returns
    -------
    pl.DataFrame
        The loaded time series data with properly formatted date column
        
    Raises
    ------
    FileNotFoundError
        If the data file doesn't exist at the expected location
        
    Notes
    -----
    The function expects the CSV file to have a 'date' column and
    at least one additional column containing time series data.
    
    See Also
    --------
    get_series_names : Extract series names from loaded data
    """
    data_path = project_dir / "data" / "raw" / "data_timeseries.csv"
    logger.info(f"Loading data from {data_path}")
    
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    # Load data using polars
    df = pl.read_csv(data_path)
    
    # Ensure date column is properly formatted
    df = df.with_columns(pl.col("date").str.to_date())
    
    logger.info(f"Successfully loaded data: {len(df)} rows, {len(df.columns)} columns")
    return df


def get_series_names(data):
    """Extract series names from the dataframe.
    
    Identifies all column names except 'date' as time series names.
    
    Parameters
    ----------
    data : pl.DataFrame
        The time series dataframe containing a 'date' column and series columns
    
    Returns
    -------
    list
        List of time series names (excluding the date column)
        
    Notes
    -----
    This function assumes that the 'date' column is the only non-time-series
    column in the dataframe. All other columns are treated as time series.
    """
    series_names = [col for col in data.columns if col != "date"]
    logger.info(f"Found {len(series_names)} time series: {', '.join(series_names)}")
    return series_names


# Define app version and build information
__version__ = "3.1.1"
__build_date__ = "2025-04-17"


# App UI
app_UI = ui.page_navbar(
    ui.nav_panel(
        "Factor Based Black Litterman Model Visualization", 
        ui.layout_sidebar(
            ui.sidebar(
                shinyswatch.theme_picker_ui("cosmo"),
                ui.p("Four Models Portfolio Comparison and Visualization for Mid-to-Long-Term Investment"),
                ui.hr(),
                ui.tags.h4("About"),
                ui.p("This dashboard visualizes the performance of four models for mid-to-long-term investment:"),
                ui.tags.ul(
                    ui.tags.li("Compare portfolio strategies: Evaluate how four models perform."),
                    ui.tags.li("Analyze key metrics: Explore returns, volatility, Sharpe ratio, and more."),
                    ui.tags.li("Interactive comparison: Compare models based on different assumptions.")
                ),
                ui.hr(),
                #ui.a("Capstone Final Report", href="https://docs.google.com/document/d/1W4b-2jRYdd6uWcNfW4JMaI5DPXPapZprtaeNpefX000/edit?usp=sharing", target="_blank"),
                ui.p("This dashboard produced by: Xiaomeng Ren"),
                ui.hr(),
                ui.p(f"Version: {__version__}"),
                ui.p(f"Build: {__build_date__}"),
            ),
            ui.navset_tab(
                ui.nav_panel("Inputs", inputs_ui("ID_inputs")),
                ui.nav_panel("Data Analysis", analysis_ui("ID_analysis")),
                ui.nav_panel("Black-Litterman Model", model1_ui("ID_model1")),
                ui.nav_panel("CVaR Portfolio Optimization", model4_ui("ID_model4")),
                #ui.nav_panel("Model 2", model2_ui("ID_model2")),
                #ui.nav_panel("Model 3", model3_ui("ID_model3")),

                ## add more or back more
                id="main_tabs",
            ),
        ),
    ),
    ui.nav_spacer(),
    ui.nav_menu(
        "Settings",
        ui.nav_panel(
            "About", 
            ui.div(
                ui.h2("About QWIM Dashboard"),
                ui.p("""This dashboard was created for time series visualization and analysis.
                       It uses Python packages like polars, plotly, and d3blocks for data processing 
                       and visualization."""),
                ui.h3("Features"),
                ui.tags.ul(
                    ui.tags.li("Interactive time series visualization"),
                    ui.tags.li("Statistical analysis of time series data"),
                    ui.tags.li("Correlation and heatmap analysis"),
                    ui.tags.li("Downloadable reports and visualizations"),
                ),
                ui.h3("Data"),
                ui.p("The dashboard uses synthetic time series data stored in CSV format."),
                ui.h3("Contact"),
                ui.p("For more information, please contact the administrator."),
                ui.hr(),
                ui.div(
                    ui.tags.small(f"Version {__version__} (Build {__build_date__})"),
                    class_="text-muted",
                ),
            ),
        ),
        ui.nav_panel(
            "Help",
            ui.div(
                ui.h2("Help"),
                ui.h3("Getting Started"),
                ui.p("1. Go to the 'Inputs' tab and select one or more time series."),
                ui.p("2. Choose date range and filtering options."),
                ui.p("3. Switch to 'Data Analysis' or 'Results' tabs to view visualizations."),
                ui.h3("Common Issues"),
                ui.tags.dl(
                    ui.tags.dt("No data displayed"),
                    ui.tags.dd("Make sure you've selected at least one time series in the Inputs tab."),
                    ui.tags.dt("Slow performance"),
                    ui.tags.dd("Try selecting a smaller date range or fewer time series."),
                    ui.tags.dt("Download not working"),
                    ui.tags.dd("Ensure your browser allows downloads from this application."),
                ),
                ui.h3("Keyboard Shortcuts"),
                ui.tags.table(
                    [
                        ui.tags.tr([ui.tags.th("Key"), ui.tags.th("Action")]),
                        ui.tags.tr([ui.tags.td("Ctrl+1"), ui.tags.td("Go to Inputs tab")]),
                        ui.tags.tr([ui.tags.td("Ctrl+2"), ui.tags.td("Go to Data Analysis tab")]),
                        ui.tags.tr([ui.tags.td("Ctrl+3"), ui.tags.td("Go to Results tab")]),
                        ui.tags.tr([ui.tags.td("Ctrl+H"), ui.tags.td("Show/hide help")]),
                    ],
                    class_="table table-striped table-sm",
                ),
            ),
        ),
        ui.nav_panel(
            "Settings",
            ui.div(
                ui.h2("Dashboard Settings"),
                ui.p("These settings affect the overall dashboard behavior."),
                ui.h3("Data Source"),
                ui.input_file(
                    "upload_data",
                    "Upload custom data file:",
                    multiple=False,
                    accept=[".csv"],
                ),
                ui.p("Note: Uploaded file must contain a 'date' column and at least one time series column."),
                ui.hr(),
                ui.h3("Theme"),
                ui.input_select(
                    "app_theme",
                    "Dashboard Theme:",
                    {
                        "cosmo": "Cosmo (Default)",
                        "flatly": "Flatly",
                        "darkly": "Darkly",
                        "journal": "Journal",
                        "lumen": "Lumen",
                    },
                    selected="cosmo",
                ),
                ui.input_action_button("apply_theme", "Apply Theme", class_="btn-primary mt-2"),
                ui.p("Note: Theme changes will be applied immediately."),
                ui.hr(),
                ui.h3("Performance"),
                ui.input_checkbox("use_caching", "Enable Data Caching", True),
                ui.p("Caching improves performance but may increase memory usage."),
                ui.input_slider(
                    "max_points",
                    "Maximum Data Points to Display:",
                    min=100,
                    max=10000,
                    value=5000,
                    step=100,
                ),
                ui.p("Lower values improve performance but reduce detail in plots."),
                ui.hr(),
                ui.h3("Data Export"),
                ui.download_button(
                    "download_all_data",
                    "Export All Data",
                    class_="btn-outline-primary",
                ),
                ui.p("Download the complete dataset in CSV format."),
            ),
        ),
        ui.nav_panel(
            "System Info", 
            ui.div(
                ui.h2("System Information"),
                ui.output_text_verbatim("system_info"),
                ui.hr(),
                ui.h3("Data Summary"),
                ui.output_text_verbatim("data_summary"),
                ui.hr(),
                ui.h3("Session Info"),
                ui.output_text_verbatim("session_info"),
            ),
        ),
    ),
    title="QWIM Dashboard",
    footer=ui.div(
        ui.row(
            ui.column(4, ui.p("QWIM Dashboard Footer")),
            ui.column(
                4,
                ui.p(
                    ui.a("Documentation", href="#", target="_blank"),
                    class_="text-center d-block",
                ),
            ),
            ui.column(4, ui.p(f"v{__version__}", class_="text-right d-block")),
        ),
        class_="container-fluid",
    ),
    inverse=True,
    theme=shinyswatch.theme.cosmo,
)


def app_Server(input, output, session):
    """Server function for the QWIM Dashboard application.
    
    This function initializes data, creates reactive values, and sets up module servers.
    It handles data loading, UI interactivity, theme changes, keyboard shortcuts,
    file uploads, and system information display.
    
    Parameters
    ----------
    input : shiny.Inputs
        Shiny input object containing all user interface inputs
    output : shiny.Outputs
        Shiny output object for rendering results back to the UI
    session : shiny.session.Session
        Shiny session object for managing client state
        
    Returns
    -------
    None
        This function does not return a value
        
    Notes
    -----
    This server function handles several types of functionality:
    
    - Data initialization and reactive value creation
    - Module server initialization with appropriate data sharing
    - UI customization including theme changes
    - File upload handling with validation
    - System information display
    - Session tracking and management
    
    The function uses reactive programming patterns to ensure UI updates
    and data processing occur efficiently when inputs change.
    
    See Also
    --------
    inputs_server : Server component for the Inputs tab
    analysis_server : Server component for the Data Analysis tab
    results_server : Server component for the Results tab
    portfolios_server : Server component for the Portfolios tab
    """
    # Include theme_picker_server server in the root of your server function
    shinyswatch.theme_picker_server()
    
    # Load data
    try:
        data = get_data()
        series_names = get_series_names(data)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {str(e)}")
        ui.notification_show(
            f"Error loading data: {str(e)}",
            type="error",
            duration=None,
        )
        data = pl.DataFrame({"date": [], "placeholder": []})
        series_names = []

    # Create reactive values to share data between modules
    data_r = reactive.Value(data)
    series_names_r = reactive.Value(series_names)
    
    # Define caching behavior based on settings
    @reactive.Effect
    def _setup_caching():
        """Configure caching based on user settings.
        
        This effect monitors changes to caching-related inputs and
        updates the application's caching behavior accordingly.
        """
        use_caching = input.use_caching()
        max_points = input.max_points()
        
        # We would implement actual caching configuration here
        logger.info(f"Cache settings updated: enabled={use_caching}, max_points={max_points}")
    
    # Add custom JavaScript for keyboard shortcuts
    ui.insert_ui(
        ui.tags.script("""
        $(document).ready(function() {
            Shiny.addCustomMessageHandler('setup-shortcuts', function(message) {
                $(document).keydown(function(event) {
                    // Ctrl+1: Go to Inputs tab
                    if (event.ctrlKey && event.which === 49) {
                        $('a[data-value="Inputs"]').tab('show');
                        event.preventDefault();
                    }
                    // Ctrl+2: Go to Analysis tab
                    else if (event.ctrlKey && event.which === 50) {
                        $('a[data-value="Data Analysis"]').tab('show');
                        event.preventDefault();
                    }
                    // Ctrl+3: Go to Results tab
                    else if (event.ctrlKey && event.which === 51) {
                        $('a[data-value="Results"]').tab('show');
                        event.preventDefault();
                    }
                    // Ctrl+H: Show/hide help
                    else if (event.ctrlKey && event.which === 72) {
                        $('a[data-value="Help"]').click();
                        event.preventDefault();
                    }
                });
            });
        });
        """),
        "head",
    )

    # Handle file uploads
    @reactive.effect
    @reactive.event(input.upload_data)
    def _handle_file_upload():
        """Handle file uploads and update data if valid.
        
        This function validates uploaded CSV files, updates the data
        if the file is valid, and notifies the user of the result.
        """
        file_info = input.upload_data()
        if file_info and len(file_info) > 0:
            file_path = file_info[0]["datapath"]
            logger.info(f"File uploaded: {file_path}")
            
            try:
                # Read the uploaded CSV
                new_data = pl.read_csv(file_path)
                
                # Validate required columns
                if "date" not in new_data.columns:
                    ui.notification_show(
                        "Uploaded file must contain a 'date' column",
                        type="error",
                        duration=5,
                    )
                    return
                
                # Ensure date column is properly formatted
                new_data = new_data.with_columns(pl.col("date").str.to_date())
                
                # Get new series names
                new_series_names = [col for col in new_data.columns if col != "date"]
                
                if len(new_series_names) == 0:
                    ui.notification_show(
                        "Uploaded file must contain at least one time series column",
                        type="error",
                        duration=5,
                    )
                    return
                
                # Update reactive values
                data_r.set(new_data)
                series_names_r.set(new_series_names)
                
                ui.notification_show(
                    f"Data updated successfully: {len(new_data)} rows, {len(new_series_names)} series",
                    type="success",
                    duration=3,
                )
                logger.info(f"Data updated from upload: {len(new_data)} rows, {len(new_series_names)} series")
                
            except Exception as e:
                logger.error(f"Error processing uploaded file: {str(e)}")
                ui.notification_show(
                    f"Error loading file: {str(e)}",
                    type="error",
                    duration=5,
                )
    
    # Handle theme changes
    @reactive.effect
    @reactive.event(input.apply_theme)
    def _handle_theme_change():
        """Apply theme changes through JavaScript.
        
        This function updates the dashboard's theme based on user selection
        and displays a notification confirming the change.
        """
        theme = input.app_theme()
        logger.info(f"Applying theme: {theme}")
        
        ui.notification_show(
            f"Theme changed to {theme}",
            type="message",
            duration=3,
        )
    
    # Add JavaScript for theme changing
    ui.insert_ui(
        ui.tags.script("""
        Shiny.addCustomMessageHandler('change-theme', function(message) {
            var theme = message.theme;
            var oldLink = document.getElementById('theme-stylesheet');
            var newLink = document.createElement('link');
            
            newLink.rel = 'stylesheet';
            newLink.id = 'theme-stylesheet';
            newLink.href = 'https://cdn.jsdelivr.net/npm/bootswatch@5.1.3/dist/' + theme + '/bootstrap.min.css';
            
            if (oldLink) {
                document.head.replaceChild(newLink, oldLink);
            } else {
                document.head.appendChild(newLink);
            }
        });
        """),
        "head",
    )
    
    # System information output
    @output
    @render.text
    def system_info():
        """Display system information.
        
        Gathers and formats information about the system environment,
        including Python version, platform details, and package versions.
        
        Returns
        -------
        str
            Formatted string containing system information
        """
        import platform

        info = [
            f"Python Version: {platform.python_version()}",
            f"Platform: {platform.platform()}",
            f"NumPy Version: {np.__version__}",
            f"Pandas Version: {pd.__version__}",
            f"Polars Version: {pl.__version__}",
            f"Dashboard Version: {__version__}",
            f"Build Date: {__build_date__}",
        ]
        return "\n".join(info)
    
    # Data summary output
    @output
    @render.text
    def data_summary():
        """Display summary of the data.
        
        Generates a summary of the currently loaded dataset, including
        row count, number of series, date range, and memory usage.
        
        Returns
        -------
        str
            Formatted string containing data summary information
        """
        current_data = data_r()
        
        if current_data is None or current_data.is_empty():
            return "No data available"
        
        series_count = len([col for col in current_data.columns if col != "date"])
        date_range = current_data.select("date").to_pandas()["date"]
        memory_usage = sys.getsizeof(current_data) / (1024 * 1024)  # Convert to MB
        
        summary = [
            f"Rows: {len(current_data)}",
            f"Series: {series_count}",
            f"Date Range: {min(date_range)} to {max(date_range)}",
            f"Memory Usage: {memory_usage:.2f} MB",
        ]
        return "\n".join(summary)
    
    # Session info output
    @output
    @render.text
    def session_info():
        """Display session information.
        
        Provides information about the current user session, including
        session ID, client address, start time, and configuration settings.
        
        Returns
        -------
        str
            Formatted string containing session information
        """
        info = [
            f"Session ID: {session.id}",
            f"Client Address: {session.client_address}",
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Cache Enabled: {input.use_caching()}",
            f"Max Points: {input.max_points()}",
        ]
        return "\n".join(info)
    
    # Download handler for exporting all data
    @render.download(filename="qwim_data_export.csv")
    def download_all_data():
        """Export all data as CSV.
        
        Creates a downloadable CSV file containing all currently loaded data.
        
        Returns
        -------
        str
            CSV-formatted string containing all data
            
        Notes
        -----
        If no data is available, returns an error message instead of empty CSV.
        """
        current_data = data_r()
        if current_data is None or current_data.is_empty():
            return "No data available for export"
        
        # Convert to CSV string
        csv_data = current_data.write_csv()
        
        # Log the download
        logger.info(f"Data export requested: {len(current_data)} rows")
        
        return csv_data
    
    # Portfolio data loading and validation
    portfolio_data_path = project_dir / "data" / "processed" / "sample_portfolio_values.csv"
    benchmark_data_path = project_dir / "data" / "processed" / "benchmark_portfolio_values.csv"
    weights_data_path = project_dir / "data" / "raw" / "sample_portfolio_weights_ETFs.csv"

    # Create reactive values specifically for portfolio data
    portfolio_data_r = reactive.Value(None)
    portfolio_settings_r = reactive.Value({
        "normalize": False,
        "show_difference": False,
        "date_range": [None, None]
    })

    # Check for missing files
    missing_files = []
    if not portfolio_data_path.exists():
        missing_files.append("sample_portfolio_values.csv")
    if not benchmark_data_path.exists():
        missing_files.append("benchmark_portfolio_values.csv")
    if not weights_data_path.exists():
        missing_files.append("sample_portfolio_weights_ETFs.csv")

    if missing_files:
        logger.warning(f"Missing portfolio data files: {', '.join(missing_files)}")
        ui.notification_show(
            f"Some portfolio data files are missing: {', '.join(missing_files)}. Portfolio functionality may be limited.",
            type="warning",
            duration=10,
        )
    else:
        # Initialize portfolio data if files exist
        try:
            import pandas as pd
            portfolio_df = pd.read_csv(portfolio_data_path)
            benchmark_df = pd.read_csv(benchmark_data_path)
            weights_df = pd.read_csv(weights_data_path)
            
            # Ensure date columns are datetime
            portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
            benchmark_df['Date'] = pd.to_datetime(benchmark_df['Date'])
            if 'Date' in weights_df.columns:
                weights_df['Date'] = pd.to_datetime(weights_df['Date'])
            elif 'date' in weights_df.columns:  # Check lowercase alternative
                weights_df['date'] = pd.to_datetime(weights_df['date'])
                weights_df.rename(columns={'date': 'Date'}, inplace=True)
            
            # Process data to ensure it's ready for plotting
            portfolio_data = {
                'portfolio': portfolio_df,
                'benchmark': benchmark_df,
                'weights': weights_df
            }
            
            # Validate portfolio data format
            expected_cols = ['Date', 'Value']
            for df_name in ['portfolio', 'benchmark']:
                df = portfolio_data[df_name]
                if not all(col in df.columns for col in expected_cols):
                    logger.warning(f"{df_name} data missing required columns: {expected_cols}")
                    # Add placeholder columns if missing
                    for col in expected_cols:
                        if col not in df.columns:
                            df[col] = None if col == 'Date' else 0
            
            portfolio_data_r.set(portfolio_data)
            logger.info("Portfolio data loaded successfully")
        except Exception as e:
            logger.error(f"Error loading portfolio data: {str(e)}")
            # Create empty portfolio data with correct structure
            portfolio_data_r.set({
                'portfolio': pd.DataFrame({'Date': [], 'Value': []}),
                'benchmark': pd.DataFrame({'Date': [], 'Value': []}),
                'weights': pd.DataFrame({'Date': [], 'Asset': [], 'Weight': []})
            })

    # Initialize module servers with error handling
    try:
        logger.info("Initializing inputs module...")
        inputs_data = inputs_server("ID_inputs", data_r, series_names_r)
        logger.info("Inputs module initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing inputs module: {str(e)}")
        inputs_data = {}  # Provide a fallback

    try:
        logger.info("Initializing analysis module...")
        analysis_server("ID_analysis", inputs_data, data_r, series_names_r)
        logger.info("Analysis module initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing analysis module: {str(e)}")

    try:
        logger.info("Initializing black-litterman module...")
        model1_server("ID_model1", data_r, series_names_r)
        logger.info("Results module initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing black-litterman module: {str(e)}")

    try:
        logger.info("Initializing CVaR Portfolio Optimization module...")
        model4_server("ID_model4", data_r, series_names_r)
        logger.info("Results module initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Model 4 module: {str(e)}")
    '''
    try:
        logger.info("Initializing Model 2 module...")
        #model2_server("ID_model2", data_r, series_names_r)
        logger.info("Model 2 module initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Model 2 module: {str(e)}")

    try:
        logger.info("Initializing Model 3 module...")
        model3_server("ID_model3", data_r, series_names_r)
        logger.info("Model 3 module initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Model 3 module: {str(e)}")

    '''

    current_tab = reactive.Value("Inputs")  # Default tab

    # Monitor main tab changes
    @reactive.effect
    def _track_selected_tab():
        # Get the currently selected tab
        selected_tab = input["main_tabs"]()
        if selected_tab:
            prev_tab = current_tab.get()
            current_tab.set(selected_tab)
            
            # Log tab changes
            logger.info(f"Tab changed from {prev_tab} to {selected_tab}")
            
            # When the portfolios tab is selected, ensure it's properly initialized
            if selected_tab == "Portfolios" and prev_tab != "Portfolios":
                # Trigger a refresh after a short delay to ensure plots render
                reactive.invalidate_later(0.5)
                # Update UI notification
                ui.notification_show(
                    "Loading portfolio data...", 
                    type="message",
                    duration=2
                )

    try:
        logger.info("Initializing portfolios module...")
        # Use only the parameters that the function actually accepts
        #portfolios_server(
        #    "ID_portfolios", 
        #    inputs_data=inputs_data, 
        #    data_r=data_r
        #)
        logger.info("Portfolios module initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing portfolios module: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    # Add a special handler for portfolio tab rendering
    @reactive.effect
    async def _ensure_portfolio_plots_render():
        """Force portfolio plots to render after a short delay.
        
        This function ensures that portfolio plots are properly rendered
        when the Portfolios tab is activated.
        """
        # Only run this effect when Portfolios tab is active
        if current_tab.get() != "Portfolios":
            return
        
        # Get the current portfolio data to trigger reactivity
        portfolio_data = portfolio_data_r.get()
        
        # Add a slight delay to allow DOM to update
        reactive.invalidate_later(0.5)
        
        # Use JavaScript to ensure plots are properly sized and rendered
        await session.send_custom_message("javascript", """
        $(document).ready(function() {
            console.log("Refreshing portfolio plots");
            // Trigger resize events to force plot redraws
            setTimeout(function() {
                $(window).trigger('resize');
                $('.shiny-plot-output').each(function() {
                    $(this).trigger('shown');
                });
                // Force redraw of any plotly outputs
                $('.plotly-graph-div').each(function() {
                    if (this.id && window.Plotly) {
                        var plotDiv = document.getElementById(this.id);
                        if (plotDiv && plotDiv.data) {
                            Plotly.redraw(plotDiv);
                        }
                    }
                });
            }, 500);
        });
        """)

    # Add debugging helpers for portfolio issues
    @reactive.effect
    def _monitor_portfolio_data_changes():
        """Monitor changes to portfolio data for debugging."""
        portfolio_data = portfolio_data_r.get()
        if portfolio_data:
            # Log info about the data without being too verbose
            data_summary = {}
            for key, df in portfolio_data.items():
                if isinstance(df, pd.DataFrame):
                    data_summary[key] = {
                        'rows': len(df),
                        'columns': list(df.columns)
                    }
                else:
                    data_summary[key] = f"Not a DataFrame: {type(df)}"
            
            logger.info(f"Portfolio data updated: {data_summary}")
        else:
            logger.warning("Portfolio data is None or empty")

    # Add special observers for portfolio-specific inputs
    for input_id in ['ID_comparison_date_range', 'ID_analysis_date_range', 'ID_weights_date_range']:
        @reactive.effect
        @reactive.event(lambda i=input_id: getattr(input, i)())
        def _update_portfolio_date_range(i=input_id):
            """Update portfolio settings when date range changes."""
            date_range = getattr(input, i)()
            if date_range and len(date_range) == 2:
                logger.info(f"Portfolio date range updated via {i}: {date_range}")
                # Update portfolio settings
                current_settings = portfolio_settings_r.get()
                current_settings['date_range'] = date_range
                portfolio_settings_r.set(current_settings)

    # Log session information when app starts
    logger.info(f"New session started: {session.id}")
    
    # Register callback for session end
    @session.on_ended
    def _log_session_end():
        """Log when the session ends.
        
        This function is called automatically when a user session ends,
        allowing for proper cleanup and logging.
        """
        logger.info(f"Session ended: {session.id}")


# Initialize the app
app = App(
    ui=app_UI,
    server=app_Server,
)


# Run the app
if __name__ == "__main__":
    # Get port from environment variable or use default
    port_App_Shiny = int(os.environ.get("SHINY_PORT", 8000))
    host_App_Shiny = os.environ.get("SHINY_HOST", "0.0.0.0")
    
    logger.info(f"Starting QWIM Dashboard application on {host_App_Shiny}:{port_App_Shiny}")
    app.run(
        host=host_App_Shiny,
        port=port_App_Shiny,
    )
else:
    logger.info("QWIM Dashboard module imported")