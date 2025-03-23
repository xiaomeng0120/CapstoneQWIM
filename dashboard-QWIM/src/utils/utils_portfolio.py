#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Utilities Module
==========================

This module provides utility functions for working with portfolios,
including loading data files, creating portfolios, and calculating
portfolio values over time.

Functions
---------
load_sample_etf_data
    Load ETF price data from a CSV file in the data/raw directory
load_portfolio_weights
    Load portfolio weights data from a CSV file in the data/raw directory
create_sample_portfolio
    Create a portfolio object from portfolio weights data
calculate_portfolio_values
    Calculate portfolio value time series given weights and price data
save_portfolio_values_to_csv
    Save portfolio values to a CSV file in the data/processed directory
create_benchmark_portfolio_values
    Create a benchmark portfolio with specified random variations
save_benchmark_portfolio_values_to_csv
    Save benchmark portfolio values to a CSV file in the data/processed directory
get_sample_portfolio
    Get a pre-configured sample portfolio for testing and demonstrations
visualize_portfolio_weights
    Visualize portfolio weights over time with matplotlib
debug_dataframe
    Print debug information about a DataFrame for troubleshooting

Examples
--------
>>> from utils_portfolio import get_sample_portfolio
>>> portfolio_obj, etf_data, values = get_sample_portfolio()  # doctest: +SKIP
>>> print(f"Portfolio has {portfolio_obj.get_num_components} components")  # doctest: +SKIP
Portfolio has 5 components
"""

import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import polars as pl

# Add the parent directory to sys.path to enable the portfolios import
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))

# Import local module - with fallback
try:
    from portfolios.portfolio import portfolio
except ImportError:
    print("Warning: Could not import portfolio. Make sure it's in your path.")
    print(f"Current sys.path: {sys.path}")


def debug_dataframe(df: pl.DataFrame, name: str) -> None:
    """Print debug information about a DataFrame.
    
    Parameters
    ----------
    df : pl.DataFrame
        The DataFrame to debug
    name : str
        Name of the DataFrame for identification
    """
    print(f"\n=== DEBUG INFO: {name} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print("First few rows:")
    print(df.head(3))
    print("Column types:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    print("=" * 40)


def ensure_path_exists(path: Union[str, Path]) -> Path:
    """Ensure a directory path exists, creating it if necessary.
    
    Parameters
    ----------
    path : str or Path
        Path to ensure exists
        
    Returns
    -------
    Path
        The resolved Path object
    """
    path_obj = Path(path).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_sample_etf_data(filepath: Optional[Union[str, Path]] = None) -> pl.DataFrame:
    """Load ETF price data from a CSV file.
    
    Parameters
    ----------
    filepath : str or Path, optional
        Path to the CSV file containing ETF data.
        If None, uses the default "data_ETFs.csv" file in the data/raw directory.
        
    Returns
    -------
    pl.DataFrame
        DataFrame containing ETF price data.
        
    Raises
    ------
    FileNotFoundError
        If the ETF data file cannot be found at the specified location.
        
    Notes
    -----
    The expected CSV format has a "Date" column and columns for each ETF's price.
    """
    try:
        # Define the root project directory for absolute paths - one level up
        project_root = Path(__file__).resolve().parents[2]
        
        if filepath is None:
            # Use default file in the data/raw directory with absolute path
            filepath = project_root / "data" / "raw" / "data_ETFs.csv"

        # Ensure the filepath is a Path object with resolved absolute path
        filepath = Path(filepath).resolve()

        # Load data with better error message
        if not filepath.exists():
            raise FileNotFoundError(
                f"ETF data file not found: {filepath}\n"
                f"Make sure the file exists in the data/raw directory."
            )

        # Read CSV with date formats explicitly specified
        try:
            # Try reading with automatically detected date format
            data = pl.read_csv(filepath)
        except Exception as e:
            print(f"Warning: Could not read CSV with automatic parsing: {e}")
            # Try reading with explicit date formats
            data = pl.read_csv(filepath, try_parse_dates=False)
        
        # Check if there's a date column but it's not named exactly 'Date'
        date_column_variants = ['date', 'Date', 'DATE', 'datetime', 'Datetime', 'time', 'Time']
        date_columns = [col for col in data.columns if col.lower() in [v.lower() for v in date_column_variants]]
        
        if not date_columns:
            raise ValueError(f"No date column found in {filepath}. Please ensure your CSV file has a 'Date' column.")
        elif 'Date' not in data.columns and date_columns:
            # Rename the first found date column to 'Date'
            old_name = date_columns[0]
            data = data.rename({old_name: 'Date'})
            print(f"Renamed column '{old_name}' to 'Date' for consistency.")
        
        # Ensure Date column is properly parsed - try multiple date formats
        if "Date" in data.columns:
            # Get the first few values to determine the format
            date_sample = data["Date"].head(5).to_list()
            print(f"Date sample values: {date_sample}")
            
            # Try to parse the date with various formats
            date_formats = [
                "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", 
                "%d-%m-%Y", "%m-%d-%Y", "%Y%m%d", "%d.%m.%Y"
            ]
            
            parsed = False
            for date_format in date_formats:
                try:
                    # Try to parse with this format
                    from datetime import datetime
                    # Check if the first value can be parsed
                    if isinstance(date_sample[0], str):
                        datetime.strptime(date_sample[0], date_format)
                        # If we got here, the format works
                        print(f"Using date format: {date_format}")
                        
                        # Apply the format to parse dates
                        data = data.with_columns(
                            pl.col("Date").str.strptime(pl.Date, date_format, strict=False)
                        )
                        parsed = True
                        break
                except (ValueError, TypeError):
                    # This format didn't work, try the next one
                    continue
            
            if not parsed and isinstance(date_sample[0], str):
                print(f"Warning: Could not parse 'Date' column as date with any standard format.")
                print("Common date formats are: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY")
                print("Continuing with string dates.")
        
        # Add a validation check to make sure the Date column exists
        if "Date" not in data.columns:
            raise ValueError(f"Required column 'Date' is missing from the ETF data: {filepath}")
        
        return data
        
    except Exception as e:
        print(f"Error in load_sample_etf_data: {e}")
        raise


def load_portfolio_weights(filepath: Optional[Union[str, Path]] = None) -> pl.DataFrame:
    """Load portfolio weights data from a CSV file.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to the CSV file containing portfolio weights.
        If None, uses the default "sample_portfolio_weights_ETFs.csv" file in the data/raw directory.

    Returns
    -------
    pl.DataFrame
        DataFrame containing portfolio weights data.

    Raises
    ------
    FileNotFoundError
        If the portfolio weights file cannot be found at the specified location.

    Notes
    -----
    The expected CSV format has a "Date" column and columns for each asset's weight.
    The weights in each row should sum to approximately 1.0.
    """
    try:
        # Define the root project directory for absolute paths
        project_root = Path(__file__).resolve().parents[2]
        
        if filepath is None:
            # Use default file in the data/raw directory with absolute path
            filepath = project_root / "data" / "raw" / "sample_portfolio_weights_ETFs.csv"

        # Ensure the filepath is a Path object with resolved absolute path
        filepath = Path(filepath).resolve()

        # Load data with better error message
        if not filepath.exists():
            # Try to list files in the directory to help debugging
            try:
                parent_dir = filepath.parent
                if parent_dir.exists():
                    files = list(parent_dir.glob("*.csv"))
                    file_list = "\n  ".join([f.name for f in files])
                    raise FileNotFoundError(
                        f"Portfolio weights file not found: {filepath}\n"
                        f"Available CSV files in {parent_dir}:\n  {file_list}"
                    )
            except Exception:
                pass
                
            raise FileNotFoundError(
                f"Portfolio weights file not found: {filepath}\n"
                f"Make sure the file exists in the data/raw directory."
            )

        # Read CSV and ensure proper column naming
        data = pl.read_csv(filepath)
        
        # Check if there's a date column but it's not named exactly 'Date'
        date_column_variants = ['date', 'Date', 'DATE', 'datetime', 'Datetime', 'time', 'Time']
        date_columns = [col for col in data.columns if col.lower() in [v.lower() for v in date_column_variants]]
        
        if not date_columns:
            raise ValueError(f"No date column found in {filepath}. Please ensure your CSV file has a 'Date' column.")
        elif 'Date' not in data.columns and date_columns:
            # Rename the first found date column to 'Date'
            data = data.rename({date_columns[0]: 'Date'})
            print(f"Renamed column '{date_columns[0]}' to 'Date' for consistency.")
        
        # Ensure Date column is properly parsed
        if "Date" in data.columns:
            try:
                # For polars >= 0.19.0
                if hasattr(pl.col("Date").str, "to_date"):
                    data = data.with_columns(pl.col("Date").str.to_date())
                # For older polars versions
                else:
                    data = data.with_columns(pl.col("Date").cast(pl.Date))
            except Exception as e:
                print(f"Warning: Could not parse 'Date' column as date: {e}")
                print("Continuing with string dates.")
        
        # Add a validation check to make sure the Date column exists
        if "Date" not in data.columns:
            raise ValueError(f"Required column 'Date' is missing from the portfolio weights data: {filepath}")
            
        return data
        
    except Exception as e:
        print(f"Error in load_portfolio_weights: {e}")
        raise


def create_sample_portfolio(weights_data: pl.DataFrame) -> portfolio:
    """Create a portfolio object from portfolio weights data.

    Parameters
    ----------
    weights_data : pl.DataFrame
        DataFrame containing portfolio weights data.

    Returns
    -------
    portfolio
        Portfolio object initialized with the provided weights data.

    Examples
    --------
    >>> from utils_portfolio import load_portfolio_weights, create_sample_portfolio
    >>> weights = load_portfolio_weights()
    >>> p = create_sample_portfolio(weights)
    >>> print(p.get_num_components)
    5
    """
    try:
        # Validate the weights data before creating the portfolio
        if "Date" not in weights_data.columns:
            raise ValueError("The weights data must contain a 'Date' column.")
        
        # Check if there are any non-Date columns that could be component weights
        component_columns = [col for col in weights_data.columns if col != "Date"]
        if not component_columns:
            raise ValueError("The weights data must contain columns for component weights in addition to 'Date'.")
        
        # Report potentially problematic columns
        for col in component_columns:
            if col.isdigit() or (col.startswith('-') and col[1:].isdigit()):
                print(f"Warning: Column '{col}' is numeric and might be an index rather than a component name.")

        # Clean up weights_data to ensure proper format
        cleaned_weights = weights_data.clone()
        
        # Make sure we have the portfolio class available
        if 'portfolio' not in globals():
            raise ImportError("The portfolio class could not be imported. Check your installation.")
        
        # Proceed with creating the portfolio - using the updated constructor order
        portfolio_obj = portfolio(
            name_portfolio="Sample Portfolio",  # Now the first parameter
            portfolio_weights=cleaned_weights
        )
        
        # Verify the portfolio was created correctly
        if not portfolio_obj.get_portfolio_components:
            raise ValueError("Created portfolio has no components. Check your weights data format.")
            
        return portfolio_obj
    
    except Exception as e:
        print(f"Error in create_sample_portfolio: {e}")
        raise


def calculate_portfolio_values(
    portfolio_obj: portfolio,
    price_data: pl.DataFrame,
    initial_value: float = 100.0,
) -> pl.DataFrame:
    """Calculate time series of portfolio values based on weights and price data.

    Parameters
    ----------
    portfolio_obj : portfolio
        Portfolio object containing weight information.
    price_data : pl.DataFrame
        DataFrame containing price data for the assets in the portfolio.
    initial_value : float, optional
        Initial portfolio value (default: 100.0).

    Returns
    -------
    pl.DataFrame
        DataFrame with columns for Date and Portfolio_Value.

    Raises
    ------
    ValueError
        If components in the portfolio are missing from price data.
        If no weight dates are found in the portfolio.

    Notes
    -----
    This function:
    1. Aligns portfolio weights with price data dates using date ranges
    2. For dates between DateOne and DateTwo in the weights DataFrame,
       applies the weights from DateTwo to all price dates in that range
    3. Calculates portfolio values by multiplying weights with prices
    4. Returns a time series of portfolio values
    """
    try:
        # Get portfolio weights and components
        weights_df = portfolio_obj.get_portfolio_weights
        components = portfolio_obj.get_portfolio_components

        # Debug info
        print(f"Portfolio components: {components}")
        print(f"Price data columns: {price_data.columns}")
        
        # Filter out numeric column names (likely index columns) from components
        valid_components = []
        for comp in components:
            if comp != "Date" and comp in price_data.columns:
                valid_components.append(comp)
            elif comp.isdigit() or (comp.startswith('-') and comp[1:].isdigit()):
                print(f"Warning: Ignoring numeric component '{comp}' which is likely an index column")
            else:
                print(f"Warning: Component '{comp}' not found in price data columns")
                
        if not valid_components:
            raise ValueError(
                "No valid components found that match between portfolio and price data.\n"
                "Check that your portfolio weights columns match ETF names in price data."
            )

        # Replace components with valid_components
        components = valid_components
        print(f"Using these components: {components}")

        # Sort weights by date
        weights_df = weights_df.sort("Date")
        
        # Sort price data by date
        price_data = price_data.sort("Date")
        
        # Filter price data to include only the components in the portfolio and the date
        price_cols = ["Date"] + components
        filtered_prices = price_data.select(price_cols)
        
        # Create result DataFrame with dates from filtered prices
        # FIX: Use list constructor instead of pl.lit for null values
        dates = filtered_prices["Date"].to_list()
        null_values = [None] * len(dates)
        
        result = pl.DataFrame({
            "Date": dates,
            "Portfolio_Value": null_values,
        })
        
        # Get weight dates as a sorted list
        weight_dates = weights_df["Date"].to_list()
        
        if not weight_dates:
            raise ValueError("No weight dates found in portfolio")
        
        # Initialize portfolio value
        current_portfolio_value = initial_value
        portfolio_values = []
        
        # Apply the weights according to date ranges
        for i, price_date in enumerate(filtered_prices["Date"]):
            # Find the applicable weight date (largest weight_date that's <= price_date)
            applicable_weight_date = None
            for weight_date in reversed(weight_dates):
                if weight_date <= price_date:
                    applicable_weight_date = weight_date
                    break
            
            # If no applicable weight date found, skip this price date
            if applicable_weight_date is None:
                portfolio_values.append(None)
                continue
            
            # Get the weights for this date
            weights_filter = weights_df.filter(pl.col("Date") == applicable_weight_date)
            if weights_filter.shape[0] == 0:
                portfolio_values.append(None)
                continue
                
            weight_row = weights_filter.row(0, named=True)
            
            if i == 0:
                # For the first valid date, set initial portfolio value
                portfolio_values.append(initial_value)
                continue
                
            # Get prices for current and previous days
            curr_prices = filtered_prices.filter(pl.col("Date") == price_date).row(0, named=True)
            prev_date = filtered_prices["Date"][i-1]
            prev_prices = filtered_prices.filter(pl.col("Date") == prev_date).row(0, named=True)
            
            # Calculate portfolio return for this day
            portfolio_return = 0.0
            for component in components:
                # Calculate return for this component
                try:
                    if prev_prices[component] > 0:
                        component_return = (curr_prices[component] / prev_prices[component]) - 1
                        # Apply weight to component return if the component exists in weights
                        if component in weight_row:
                            portfolio_return += component_return * weight_row[component]
                        else:
                            # Try with string conversion if numeric columns are involved
                            str_component = str(component)
                            if str_component in weight_row:
                                portfolio_return += component_return * weight_row[str_component]
                except (ValueError, TypeError, ZeroDivisionError) as e:
                    print(f"Error calculating return for {component}: {e}")
                    # Skip this component
                    continue
            
            # Update portfolio value
            current_portfolio_value *= (1 + portfolio_return)
            portfolio_values.append(current_portfolio_value)
        
        # Add portfolio values to result DataFrame - version-agnostic approach
        # FIX: Use with_columns that works across polars versions instead of with_column
        try:
            # Try with newer polars versions first
            result = result.with_columns(
                pl.Series("Portfolio_Value", portfolio_values)
            )
        except (AttributeError, TypeError):
            try:
                # Try with legacy polars versions
                result = result.with_column(
                    pl.Series("Portfolio_Value", portfolio_values)
                )
            except (AttributeError, TypeError):
                # Ultimate fallback: recreate the DataFrame
                result = pl.DataFrame({
                    "Date": dates,
                    "Portfolio_Value": portfolio_values,
                })
        
        # Filter out rows with null portfolio values
        result = result.filter(pl.col("Portfolio_Value").is_not_null())
        
        # Check if we have any results
        if result.shape[0] == 0:
            raise ValueError("No portfolio values could be calculated. Please check your data.")
            
        return result
    
    except Exception as e:
        print(f"Error in calculate_portfolio_values: {e}")
        import traceback
        traceback.print_exc()
        # Return an empty DataFrame instead of raising an exception
        return pl.DataFrame({"Date": [], "Portfolio_Value": []})


def save_portfolio_values_to_csv(
    portfolio_values: pl.DataFrame, 
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save portfolio values to a CSV file.

    Parameters
    ----------
    portfolio_values : pl.DataFrame
        DataFrame with portfolio values data.
    output_path : str or Path, optional
        Path where to save the CSV file. If None, saves to
        'data/processed/sample_portfolio_values.csv' by default.

    Returns
    -------
    Path
        Absolute Path where the CSV file was saved.

    Notes
    -----
    The output CSV file will have two columns:
    - Date: containing dates
    - Value: containing portfolio values
    """
    # Define the root project directory for absolute paths
    project_root = Path(__file__).resolve().parents[2]
    
    if output_path is None:
        # Use default path in the data/processed directory with absolute path
        output_path = project_root / "data" / "processed" / "sample_portfolio_values.csv"
    
    # Ensure the filepath is a Path object with resolved absolute path
    output_path = Path(output_path).resolve()
    
    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create a new DataFrame with the desired format
    output_portfolio_values = portfolio_values.select(
        pl.col("Date"),
        pl.col("Portfolio_Value").alias("Value"),
    )
    
    # Write to CSV
    output_portfolio_values.write_csv(output_path)
    
    return output_path


def create_benchmark_portfolio_values(portfolio_values: pl.DataFrame) -> pl.DataFrame:
    """Create a benchmark portfolio with time-based variations from the original.

    Parameters
    ----------
    portfolio_values : pl.DataFrame
        DataFrame with original portfolio values data.

    Returns
    -------
    pl.DataFrame
        DataFrame with benchmark portfolio values.

    Notes
    -----
    This function creates a benchmark with time-based pattern:
    - First 2 years: Subtract 0.11 * portfolio_value * random(0,1)
    - Next 3 years: Add 0.04 * portfolio_value * random(0,1)
    - Next 4 years: Subtract 0.08 * portfolio_value * random(0,1)
    - Next 2 years: Add 0.03 * portfolio_value * random(0,1)
    - Remaining years: Subtract 0.05 * portfolio_value * random(0,1)
    
    This creates a benchmark that has different performance characteristics
    over different time periods, simulating varying market conditions.

    Examples
    --------
    >>> import polars as pl
    >>> df = pl.DataFrame({"Date": ["2023-01-01", "2023-01-02"], "Portfolio_Value": [100.0, 105.0]})
    >>> benchmark = create_benchmark_portfolio_values(df)
    >>> print(benchmark.columns)
    ['Date', 'Value']
    >>> print(len(benchmark))
    2
    """
    # Extract dates and values
    dates = portfolio_values["Date"].to_list()
    values = portfolio_values["Portfolio_Value"].to_list()
    
    # Create benchmark values with time-based adjustments
    benchmark_values = []
    
    # Use fixed seed for reproducibility
    random.seed(42)
    
    # Convert dates to datetime objects if they are strings
    date_objects = []
    if dates and isinstance(dates[0], str):
        # Try common date formats
        date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
        
        # Find the correct format by trying each one
        from datetime import datetime
        format_found = False
        
        for date_format in date_formats:
            try:
                date_objects = [datetime.strptime(date, date_format).date() for date in dates]
                format_found = True
                break
            except ValueError:
                continue
        
        if not format_found:
            # If no format works, use numeric indices instead of dates for time periods
            print("Warning: Could not parse dates. Using sequential indices for time periods.")
            date_objects = []
    else:
        # Dates are already datetime objects
        date_objects = dates
    
    # If we have valid date objects, use them to determine time periods
    if date_objects:
        # Find the start date
        start_date = min(date_objects)
        
        # Define the time periods in years
        period_years = [2, 3, 4, 2]  # 2, 3, 4, 2, and remaining years
        
        # Calculate the end dates for each period
        from datetime import timedelta
        period_end_dates = []
        current_date = start_date
        
        for years in period_years:
            # Approximate years by adding 365.25 days per year
            days = int(years * 365.25)
            current_date = current_date + timedelta(days=days)
            period_end_dates.append(current_date)
        
        # Apply the adjustments based on which period each date falls into
        for i, (date, value) in enumerate(zip(date_objects, values)):
            if i == 0:
                # For the first date, keep the original value
                benchmark_values.append(value)
                continue
                
            # Determine which period this date falls into
            period = 4  # Default to the last period (index 4)
            for p, end_date in enumerate(period_end_dates):
                if date <= end_date:
                    period = p
                    break
            
            # Apply the appropriate adjustment based on the period
            rand_factor = random.random()  # Random number between 0 and 1
            
            if period == 0:
                # First 2 years: Subtract 0.11 * value * random
                adjustment = value * 0.11 * rand_factor
                benchmark_value = value - adjustment
            elif period == 1:
                # Next 3 years: Add 0.04 * value * random
                adjustment = value * 0.04 * rand_factor
                benchmark_value = value + adjustment
            elif period == 2:
                # Next 4 years: Subtract 0.08 * value * random
                adjustment = value * 0.08 * rand_factor
                benchmark_value = value - adjustment
            elif period == 3:
                # Next 2 years: Add 0.03 * value * random
                adjustment = value * 0.03 * rand_factor
                benchmark_value = value + adjustment
            else:
                # Remaining years: Subtract 0.05 * value * random
                adjustment = value * 0.05 * rand_factor
                benchmark_value = value - adjustment
            
            benchmark_values.append(benchmark_value)
    else:
        # If date parsing failed, use a simpler approach based on index position
        total_periods = len(values)
        
        # Calculate period lengths
        period_lengths = []
        remaining_length = total_periods
        
        # Define period fractions (approximate years out of the total)
        period_fractions = [2/11, 3/11, 4/11, 2/11]  # 2, 3, 4, 2 years out of total
        
        for fraction in period_fractions:
            length = max(1, int(total_periods * fraction))
            if length < remaining_length:
                period_lengths.append(length)
                remaining_length -= length
            else:
                period_lengths.append(remaining_length)
                remaining_length = 0
                break
        
        if remaining_length > 0:
            period_lengths.append(remaining_length)
        
        # Calculate the ending indices for each period
        period_end_indices = []
        current_index = 0
        
        for length in period_lengths:
            current_index += length
            period_end_indices.append(current_index)
        
        # Apply the adjustments based on which period each index falls into
        for i, value in enumerate(values):
            if i == 0:
                # For the first value, keep the original
                benchmark_values.append(value)
                continue
                
            # Determine which period this index falls into
            period = len(period_lengths)  # Default to beyond all defined periods
            for p, end_index in enumerate(period_end_indices):
                if i < end_index:
                    period = p
                    break
            
            # Apply the appropriate adjustment based on the period
            rand_factor = random.random()  # Random number between 0 and 1
            
            if period == 0:
                # First period: Subtract 0.11 * value * random
                adjustment = value * 0.11 * rand_factor
                benchmark_value = value - adjustment
            elif period == 1:
                # Second period: Add 0.04 * value * random
                adjustment = value * 0.04 * rand_factor
                benchmark_value = value + adjustment
            elif period == 2:
                # Third period: Subtract 0.08 * value * random
                adjustment = value * 0.08 * rand_factor
                benchmark_value = value - adjustment
            elif period == 3:
                # Fourth period: Add 0.03 * value * random
                adjustment = value * 0.03 * rand_factor
                benchmark_value = value + adjustment
            else:
                # Remaining period: Subtract 0.05 * value * random
                adjustment = value * 0.05 * rand_factor
                benchmark_value = value - adjustment
            
            benchmark_values.append(benchmark_value)
    
    # Create the benchmark dataframe
    benchmark_portfolio_values = pl.DataFrame({
        "Date": dates,
        "Value": benchmark_values,
    })
    
    return benchmark_portfolio_values


def save_benchmark_portfolio_values_to_csv(
    benchmark_portfolio_values: pl.DataFrame, 
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save benchmark portfolio values to a CSV file.

    Parameters
    ----------
    benchmark_portfolio_values : pl.DataFrame
        DataFrame with benchmark portfolio values data.
    output_path : str or Path, optional
        Path where to save the CSV file. If None, saves to
        'data/processed/benchmark_portfolio_values.csv' by default.

    Returns
    -------
    Path
        Absolute Path where the CSV file was saved.

    Notes
    -----
    The output CSV file will have two columns:
    - Date: containing dates
    - Value: containing benchmark portfolio values
    """
    # Define the root project directory for absolute paths
    project_root = Path(__file__).resolve().parents[2]
    
    if output_path is None:
        # Use default path in the data/processed directory with absolute path
        output_path = project_root / "data" / "processed" / "benchmark_portfolio_values.csv"
    
    # Ensure the filepath is a Path object with resolved absolute path
    output_path = Path(output_path).resolve()
    
    # Ensure the directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to CSV
    benchmark_portfolio_values.write_csv(output_path)
    
    return output_path


def get_sample_portfolio() -> Tuple[portfolio, pl.DataFrame, pl.DataFrame]:
    """Get a sample portfolio, raw data, and calculated portfolio values.

    This is a convenience function for testing and demonstrations.

    Returns
    -------
    tuple
        A tuple containing:
        - portfolio: The sample portfolio object
        - pl.DataFrame: The ETF price data
        - pl.DataFrame: The calculated portfolio values time series

    Examples
    --------
    >>> p, data, values = get_sample_portfolio()
    >>> print(f"Portfolio has {len(p.get_portfolio_components)} components")
    Portfolio has 5 components
    >>> print(f"Values span {len(values)} days")
    Values span 252 days
    """
    # Load data
    sample_data_etfs = load_sample_etf_data()
    sample_portfolio_data = load_portfolio_weights()
    
    # Create portfolio - using the create_sample_portfolio function which we fixed above
    sample_portfolio_obj = create_sample_portfolio(sample_portfolio_data)
    
    # Calculate portfolio values
    portfolio_values = calculate_portfolio_values(
        sample_portfolio_obj, sample_data_etfs,
    )
    
    return sample_portfolio_obj, sample_data_etfs, portfolio_values


def visualize_portfolio_weights(
    portfolio_obj: portfolio, 
    output_path: Optional[Union[str, Path]] = None,
) -> None:
    """Visualize portfolio weights over time.

    Parameters
    ----------
    portfolio_obj : portfolio
        Portfolio object containing weight information.
    output_path : str or Path, optional
        Path to save the visualization. If None, the plot is displayed but not saved.

    Notes
    -----
    This function requires matplotlib and seaborn to be installed.

    Examples
    --------
    >>> p, _, _ = get_sample_portfolio()
    >>> visualize_portfolio_weights(p)  # Displays a plot
    >>> # Save to a file
    >>> visualize_portfolio_weights(p, "portfolio_weights.png")
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Get weights data
        weights_df = portfolio_obj.get_portfolio_weights
        
        # Convert to pandas for easier plotting
        import pandas as pd
        weights_pd = weights_df.to_pandas().set_index("Date")
        
        # Create a stacked area plot
        plt.figure(figsize=(12, 6))
        weights_pd.plot.area(stacked=True, alpha=0.7, figsize=(12, 6))
        
        plt.title("Portfolio Weights Over Time", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Weight", fontsize=12)
        plt.legend(title="Components", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Add annotations for date transitions
        for date in weights_pd.index[1:]:
            plt.axvline(x=date, color='k', linestyle='--', alpha=0.3)
        
        if output_path:
            # Convert to absolute path
            output_path = Path(output_path).resolve()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
            
    except ImportError:
        print("Visualization requires matplotlib and seaborn to be installed.")


def suggest_component_matches(components, etf_columns):
    """Suggest possible matches between portfolio components and ETF columns.
    
    Parameters
    ----------
    components : list
        List of component names from portfolio
    etf_columns : list
        List of column names from ETF data
        
    Returns
    -------
    dict
        Dictionary mapping component names to potential ETF column matches
    """
    suggestions = {}
    
    # Convert all to lowercase for case-insensitive matching
    etf_lower = [col.lower() for col in etf_columns]
    
    for comp in components:
        if comp == "Date":
            continue
            
        comp_lower = comp.lower()
        
        # Check for exact match but case-insensitive
        matches = []
        for i, etf in enumerate(etf_lower):
            # Exact match
            if etf == comp_lower:
                matches.append(etf_columns[i])
                continue
                
            # Partial match: one is contained in the other
            if etf in comp_lower or comp_lower in etf:
                matches.append(etf_columns[i])
                continue
                
            # Numeric component might match numeric ETF
            if (comp.isdigit() and etf.isdigit()) or \
               (comp.startswith('-') and comp[1:].isdigit() and etf.startswith('-') and etf[1:].isdigit()):
                matches.append(etf_columns[i])
                
        if matches:
            suggestions[comp] = matches
    
    return suggestions


# Example usage when run as a script
if __name__ == "__main__":
    try:
        # Define and create required directories
        project_root = Path(__file__).resolve().parents[2]
        raw_dir = project_root / "data" / "raw"
        processed_dir = project_root / "data" / "processed"
        
        # Create directories if they don't exist
        for directory in [raw_dir, processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Print directory information
        print(f"Project root: {project_root}")
        print(f"Raw data directory: {raw_dir}")
        print(f"Processed data directory: {processed_dir}")
        
        try:
            # Check if sample files exist
            etf_file = raw_dir / "data_ETFs.csv"
            weights_file = raw_dir / "sample_portfolio_weights_ETFs.csv"
            
            if not etf_file.exists():
                print(f"Warning: ETF data file not found at {etf_file}")
                print("Please ensure this file exists before continuing.")
                
            if not weights_file.exists():
                print(f"Warning: Portfolio weights file not found at {weights_file}")
                print("Please ensure this file exists before continuing.")
            
            # Get sample portfolio and data
            sample_data_etfs = load_sample_etf_data()
            debug_dataframe(sample_data_etfs, "ETF Data")
            
            sample_portfolio_data = load_portfolio_weights()
            debug_dataframe(sample_portfolio_data, "Portfolio Weights")
            
            # Create portfolio
            sample_portfolio_obj = create_sample_portfolio(sample_portfolio_data)
            
            # Calculate portfolio values
            print("Calculating portfolio values...")
            portfolio_values = calculate_portfolio_values(
                sample_portfolio_obj, sample_data_etfs
            )
            
            # Check if we have portfolio values
            if (portfolio_values.shape[0] == 0):
                raise ValueError("No portfolio values were calculated. Please check your data.")
                
            debug_dataframe(portfolio_values, "Portfolio Values")
            
            # Display information
            print(f"Sample Portfolio Components: {sample_portfolio_obj.get_portfolio_components}")
            print(f"Number of ETFs: {sample_portfolio_obj.get_num_components}")
            print(f"\nSample Portfolio Weights (first 3 rows):")
            print(sample_portfolio_obj.get_portfolio_weights.head(3))
            
            print(f"\nETF Data (first 3 rows):")
            print(sample_data_etfs.head(3))
            
            print(f"\nPortfolio Values (first 3 rows):")
            print(portfolio_values.head(3))
            
            # Save portfolio values to CSV
            output_portfolio_values = portfolio_values.select(
                pl.col("Date"),
                pl.col("Portfolio_Value").alias("Value"),
            )
            
            # Save sample portfolio values to CSV
            output_path = processed_dir / "sample_portfolio_values.csv"
            output_portfolio_values.write_csv(output_path)
            print(f"\nPortfolio values saved to: {output_path}")
            
            # Create benchmark portfolio values
            benchmark_portfolio_values = create_benchmark_portfolio_values(portfolio_values)
            
            # Save benchmark portfolio values to CSV
            benchmark_path = processed_dir / "benchmark_portfolio_values.csv"
            benchmark_portfolio_values.write_csv(benchmark_path)
            print(f"Benchmark portfolio values saved to: {benchmark_path}")
            
            # Try to visualize weights
            try:
                vis_path = processed_dir / "portfolio_weights.png"
                visualize_portfolio_weights(sample_portfolio_obj, vis_path)
            except Exception as e:
                print(f"Could not visualize weights: {e}")
            
            # Show sample comparison
            print("\nComparison of Original vs Benchmark (first 3 rows):")
            comparison = pl.DataFrame({
                "Date": output_portfolio_values["Date"].head(3),
                "Original": output_portfolio_values["Value"].head(3),
                "Benchmark": benchmark_portfolio_values["Value"].head(3),
                "Difference": (
                    output_portfolio_values["Value"] - benchmark_portfolio_values["Value"]
                ).head(3),
            })
            print(comparison)
            
            print(f"\nPortfolio Performance Summary:")
            initial_value = portfolio_values["Portfolio_Value"].to_list()[0]
            final_value = portfolio_values["Portfolio_Value"].to_list()[-1]
            total_return = (final_value / initial_value - 1) * 100
            print(f"Starting Value: {initial_value:.2f}")
            print(f"Final Value: {final_value:.2f}")
            print(f"Total Return: {total_return:.2f}%")
            
            # Calculate annualized return if possible
            try:
                start_date = portfolio_values["Date"][0]
                end_date = portfolio_values["Date"][-1]
                
                # Check if dates are strings and try to convert them
                if isinstance(start_date, str) or isinstance(end_date, str):
                    from datetime import datetime
                    
                    # Try different date formats
                    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", 
                                    "%d-%m-%Y", "%m-%d-%Y", "%Y%m%d", "%d.%m.%Y"]
                    
                    for date_format in date_formats:
                        try:
                            if isinstance(start_date, str):
                                start_date = datetime.strptime(start_date, date_format).date()
                            if isinstance(end_date, str):
                                end_date = datetime.strptime(end_date, date_format).date()
                            break
                        except ValueError:
                            continue
                    
                    if isinstance(start_date, str) or isinstance(end_date, str):
                        raise ValueError(f"Could not parse date strings: {start_date}, {end_date}")
                
                # Calculate days between dates
                days = (end_date - start_date).days
                
                if days <= 0:
                    raise ValueError(f"Invalid date range: {start_date} to {end_date}")
                    
                years = days / 365.0
                annualized_return = ((final_value / initial_value) ** (1/max(years, 0.01)) - 1) * 100
                print(f"Annualized Return: {annualized_return:.2f}% (over {years:.2f} years)")
            except Exception as e:
                print(f"Could not calculate annualized return: {e}")
                # Add additional debugging info
                print(f"Date types - Start: {type(portfolio_values['Date'][0])}, End: {type(portfolio_values['Date'][-1])}")
                print(f"Date values - Start: {portfolio_values['Date'][0]}, End: {portfolio_values['Date'][-1]}")
            
            # In the main block after loading the data
            print("\nChecking portfolio components vs. ETF data columns...")
            components = sample_portfolio_obj.get_portfolio_components
            etf_columns = [col for col in sample_data_etfs.columns if col != "Date"]
            missing_components = [c for c in components if c not in etf_columns]
            if missing_components:
                print(f"WARNING: These components are missing in the ETF data: {missing_components}")
                print(f"Available ETF columns: {etf_columns}")
                
                # Suggest possible matches
                suggestions = suggest_component_matches(missing_components, etf_columns)
                if suggestions:
                    print("\nPossible column matches:")
                    for comp, matches in suggestions.items():
                        print(f"  '{comp}' might match: {matches}")
                
                # Offer to create a mapping file
                print("\nConsider creating a mapping file to rename components or ETFs to match each other.")
            else:
                print("All components are present in ETF data.")
            
        except ImportError as e:
            print(f"ImportError: {e}")
            print("This may be because the 'portfolio' class could not be imported.")
            print("Make sure your Python path includes the directory containing the portfolios module.")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()


# Update any examples in docstrings that create portfolio instances

# For example, if there's a function like this:
def create_custom_portfolio(components, date=None):
    """Create a custom portfolio with equal weights for the given components.
    
    Examples
    --------
    >>> create_custom_portfolio(["AAPL", "MSFT"])
    portfolio(name='Custom Portfolio', components=['AAPL', 'MSFT'], dates=1 rows)
    """
    # Update instantiation to match new constructor
    return portfolio(
        name_portfolio="Custom Portfolio",  # Now the first parameter
        names_components=components,
        date=date
    )

