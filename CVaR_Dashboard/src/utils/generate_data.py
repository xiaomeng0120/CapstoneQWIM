"""
Time Series Data Generator Module
================================

This module generates synthetic time series data for seven different series
with various patterns and saves the result to a CSV file.

The module creates time series with different characteristics:
- Series AA: Upward trend with seasonal pattern
- Series BB: Cyclical pattern with random walks
- Series CC: Exponential growth with noise
- Series DD: Stationary with occasional jumps
- Series EE: Declining trend with seasonality
- Series FF: Highly volatile series
- Series GG: Combination of trend and cycles

Functions
---------
generate_monthly_timeseries
    Generates monthly time series data with various patterns
main
    Main execution function that generates and saves the data

Examples
--------
To generate and save time series data:

.. code-block:: python

    from src.utils.generate_data import main
    main()
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl


def generate_monthly_timeseries(
    start_date: str = "2002-01-01", 
    end_date: str = "2025-03-31",
) -> pl.DataFrame:
    """Generate monthly time series data for seven series.
    
    This function creates synthetic time series data for seven different series,
    each with unique patterns including trends, seasonality, cycles, and random components.
    
    Parameters
    ----------
    start_date : str, optional
        Start date in YYYY-MM-DD format, by default "2002-01-01"
    end_date : str, optional
        End date in YYYY-MM-DD format, by default "2025-03-31"
        
    Returns
    -------
    pl.DataFrame
        A polars DataFrame with the generated time series with columns:
        date, AA, BB, CC, DD, EE, FF, GG
        
    Notes
    -----
    The function uses a fixed random seed (42) to ensure reproducibility.
    Each series has specific characteristics:
    
    - AA: Upward trend with annual seasonality
    - BB: Cyclical pattern (3-year cycle) with random walk component
    - CC: Exponential growth with added noise
    - DD: Stationary series with a structural break (level shift)
    - EE: Declining trend with annual seasonality
    - FF: Highly volatile series with slight upward trend
    - GG: Complex pattern with multiple seasonal components
    
    Examples
    --------
    >>> df = generate_monthly_timeseries(start_date="2020-01-01", end_date="2020-12-31")
    >>> df.shape
    (12, 8)
    >>> list(df.columns)
    ['date', 'AA', 'BB', 'CC', 'DD', 'EE', 'FF', 'GG']
    """
    format_string = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, format_string)
    end_date = datetime.strptime(end_date, format_string)
    
    # Generate monthly dates
    dates = pl.date_range(
        start=start_date,
        end=end_date,
        interval="1mo",
        closed="left",
        eager=True,
    ).alias("dates")

    # Number of data points
    n = len(dates)
    
    # Generate time indices for trends and cycles
    t = np.arange(n)
    
    # Generate different time series with various patterns
    np.random.seed(seed=42)  # For reproducibility
    
    # Series AA: Upward trend with seasonal pattern
    aa = (
        100 + 0.5 * t + 
        15 * np.sin(2 * np.pi * t / 12) + 
        np.random.normal(loc=0, scale=5, size=n)
    )
    
    # Series BB: Cyclical pattern with random walks
    bb = (
        150 + 
        30 * np.sin(2 * np.pi * t / 36) + 
        np.cumsum(np.random.normal(loc=0, scale=1, size=n))
    )
    
    # Series CC: Exponential growth with noise
    cc = 50 * np.exp(0.005 * t) + np.random.normal(loc=0, scale=5, size=n)
    
    # Series DD: Stationary with occasional jumps
    dd = 200 + np.random.normal(loc=0, scale=10, size=n)
    dd[n//3:n//2] += 50  # Add a level shift in the middle
    
    # Series EE: Declining trend with seasonality
    ee = (
        300 - 0.3 * t + 
        20 * np.sin(2 * np.pi * t / 12) + 
        np.random.normal(loc=0, scale=8, size=n)
    )
    
    # Series FF: Highly volatile series
    ff = 120 + np.random.normal(loc=0, scale=25, size=n) + 0.2 * t
    
    # Series GG: Combination of trend and cycles
    gg = (
        80 + 0.3 * t + 
        20 * np.sin(2 * np.pi * t / 24) + 
        10 * np.sin(2 * np.pi * t / 6) + 
        np.random.normal(loc=0, scale=7, size=n)
    )
    
    # Create the DataFrame
    df = pl.DataFrame({
        "date": dates,
        "AA": aa.round(decimals=2),
        "BB": bb.round(decimals=2),
        "CC": cc.round(decimals=2),
        "DD": dd.round(decimals=2),
        "EE": ee.round(decimals=2),
        "FF": ff.round(decimals=2),
        "GG": gg.round(decimals=2),
    })
    
    return df


def main() -> None:
    """Generate time series data and save it to CSV.
    
    This function creates the necessary directories if they don't exist,
    generates synthetic time series data, and saves it to a CSV file
    in the project's data/raw directory.
    
    Returns
    -------
    None
    
    Notes
    -----
    The output file is saved at PROJECT_ROOT/data/raw/data_timeseries.csv.
    The function will create the directories if they don't exist.
    
    The time range for the generated data is from 2002-01-01 to 2025-03-31.
    
    Examples
    --------
    >>> from src.utils.generate_data import main
    >>> main()
    Generating time series data...
    Data saved to .../data/raw/data_timeseries.csv
    Generated 279 rows from 2002-01-01 to 2025-03-01
    """
    # Define project paths
    project_dir = Path(__file__).parent.parent.parent
    data_dir = project_dir / "data"
    raw_data_dir = data_dir / "raw"
    output_file = raw_data_dir / "data_timeseries.csv"
    
    # Generate the data
    print("Generating time series data...")
    df = generate_monthly_timeseries(
        start_date="2002-01-01",
        end_date="2025-03-31",
    )
    
    # Create directory if it doesn't exist
    raw_data_dir.mkdir(
        parents=True, 
        exist_ok=True,
    )
    
    # Save to CSV
    df.write_csv(file=output_file)
    
    print(f"Data saved to {output_file}")
    print(f"Generated {len(df)} rows from {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()