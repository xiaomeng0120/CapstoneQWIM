"""
Unit tests for the data generation module.

This module provides unit tests for functions that generate synthetic time series data
for the QWIM Dashboard. Tests verify that generated data meets expected formats,
patterns, and statistical properties.

Tests
-----
* Data Structure Tests: Verify the structure and format of generated data
* Time Series Pattern Tests: Validate patterns in generated time series
* Statistical Property Tests: Test statistical properties of generated data
* Correlation Tests: Verify that correlations between series match expectations
* File Output Tests: Test the proper creation of output files

Notes
-----
These tests use fixed random seeds to ensure reproducible results when
generating synthetic data, allowing for consistent validation of properties.
"""

import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy import stats

# Add project root to path for imports
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Import components to test
from src.dashboard.scripts.generate_data import (
    add_anomalies,
    add_correlations,
    add_pattern_components,
    add_seasonality,
    add_trends,
    create_base_dataframe,
    generate_data_for_dashboard,
    generate_random_walk,
    save_data_to_csv,
)


@pytest.fixture
def base_dates():
    """Create a sequence of dates for testing.
    
    This fixture generates daily dates for a two-year period to use
    in data generation tests.
    
    Returns
    -------
    list
        List of date objects representing daily dates for two years
        
    Notes
    -----
    The date range starts from 2023-01-01 and goes through 2024-12-31.
    """
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date.date())
        current_date += timedelta(days=1)
    
    return dates


@pytest.fixture
def empty_dataframe(base_dates):
    """Create an empty dataframe with only date column.
    
    Parameters
    ----------
    base_dates : list
        List of dates from the base_dates fixture
        
    Returns
    -------
    pd.DataFrame
        DataFrame with a properly formatted date column
        
    Notes
    -----
    This fixture provides the starting point for data generation tests,
    mimicking the structure returned by create_base_dataframe().
    """
    return pd.DataFrame({"date": base_dates})


@pytest.fixture
def random_seed():
    """Provide a fixed random seed for reproducible tests.
    
    Returns
    -------
    int
        Seed value for random number generators
        
    Notes
    -----
    Using a fixed seed ensures test results are reproducible.
    """
    return 42


def test_create_base_dataframe():
    """Test that create_base_dataframe generates correct date structure.
    
    Verifies that the function creates a DataFrame with the expected date range
    and properly formatted date column.
    
    Notes
    -----
    The function should create daily dates spanning the requested period
    with dates sorted in ascending order.
    """
    # Test with a 1-year period
    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)
    
    # Call function
    result = create_base_dataframe(start_date, end_date)
    
    # Check result
    assert isinstance(result, pd.DataFrame)
    assert "date" in result.columns
    assert len(result) == 365  # Days in 2023
    assert result["date"].iloc[0] == start_date
    assert result["date"].iloc[-1] == end_date
    assert pd.api.types.is_datetime64_dtype(result["date"]) or pd.api.types.is_period_dtype(result["date"])


def test_generate_random_walk(random_seed):
    """Test that random walk generation produces expected patterns.
    
    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility
        
    Notes
    -----
    With a fixed seed, the random walk should have consistent statistical
    properties despite being stochastic in nature.
    """
    np.random.seed(random_seed)
    
    # Generate a random walk
    length = 1000
    drift = 0.1
    volatility = 1.0
    start_value = 100
    
    result = generate_random_walk(length, drift, volatility, start_value)
    
    # Check basic properties
    assert len(result) == length
    assert result[0] == start_value
    
    # With positive drift, we expect the final value to be higher on average
    # but we can't test this deterministically without a fixed seed
    assert isinstance(result, np.ndarray)
    
    # Test that volatility affects the standard deviation of step changes
    steps = np.diff(result)
    assert np.std(steps) > 0
    
    # Generate a second walk with higher volatility
    result2 = generate_random_walk(length, drift, volatility * 2, start_value)
    steps2 = np.diff(result2)
    
    # Higher volatility should generally lead to higher standard deviation in steps
    # This might not always be true for any random seed, but is true for our fixed seed
    assert np.std(steps2) > np.std(steps)


def test_add_trends(empty_dataframe):
    """Test that trend addition creates expected patterns in data.
    
    Parameters
    ----------
    empty_dataframe : pd.DataFrame
        DataFrame with date column from fixture
        
    Notes
    -----
    The function should add series with different trend patterns:
    upward linear, downward linear, and exponential.
    """
    df = empty_dataframe.copy()
    
    # Call function with predictable parameters
    result = add_trends(df, n_series=3, base_value=100, trend_strength=0.1)
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert "upward_trend" in result.columns
    assert "downward_trend" in result.columns
    assert "exp_trend" in result.columns
    
    # Check patterns over time
    # For upward trend, last value should be higher than first
    assert result["upward_trend"].iloc[-1] > result["upward_trend"].iloc[0]
    
    # For downward trend, last value should be lower than first
    assert result["downward_trend"].iloc[-1] < result["downward_trend"].iloc[0]
    
    # For exponential trend, differences should increase over time
    diffs = np.diff(result["exp_trend"])
    first_diffs = diffs[:10]
    last_diffs = diffs[-10:]
    assert np.mean(last_diffs) > np.mean(first_diffs)


def test_add_seasonality(empty_dataframe):
    """Test that seasonality addition creates cyclical patterns.
    
    Parameters
    ----------
    empty_dataframe : pd.DataFrame
        DataFrame with date column from fixture
        
    Notes
    -----
    The function should add series with recognizable seasonal patterns
    with different periods and amplitudes.
    """
    df = empty_dataframe.copy()
    
    # Call function
    result = add_seasonality(df, n_series=2, base_value=100, max_amplitude=10)
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert "seasonal_365d" in result.columns
    assert "seasonal_30d" in result.columns
    
    # For annual seasonality, compute autocorrelation at lag 365
    annual_series = result["seasonal_365d"].values
    # Skip first 365 days to allow for full cycle comparison
    if len(annual_series) > 730:  # need at least 2 years
        corr = np.corrcoef(annual_series[365:730], annual_series[730:1095])[0, 1]
        assert corr > 0.8  # Strong correlation at annual lag
    
    # For monthly seasonality, values ~30 days apart should be similar
    monthly_series = result["seasonal_30d"].values
    if len(monthly_series) > 60:
        corr = np.corrcoef(monthly_series[30:60], monthly_series[:30])[0, 1]
        assert corr > 0.8  # Strong correlation at monthly lag


def test_add_pattern_components(empty_dataframe, random_seed):
    """Test that pattern components create complex but predictable series.
    
    Parameters
    ----------
    empty_dataframe : pd.DataFrame
        DataFrame with date column from fixture
    random_seed : int
        Random seed for reproducibility
        
    Notes
    -----
    The function should combine trends, seasonality, and noise to create
    realistic looking time series with recognizable patterns.
    """
    np.random.seed(random_seed)
    df = empty_dataframe.copy()
    
    # Call function
    result = add_pattern_components(
        df, n_series=3, base_values=[100, 200, 300], noise_level=0.02
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert "pattern1" in result.columns
    assert "pattern2" in result.columns
    assert "pattern3" in result.columns
    
    # Basic range checks
    assert result["pattern1"].mean() > 90 and result["pattern1"].mean() < 110
    assert result["pattern2"].mean() > 190 and result["pattern2"].mean() < 210
    assert result["pattern3"].mean() > 290 and result["pattern3"].mean() < 310
    
    # Check that noise is present (values aren't perfectly smooth)
    for col in ["pattern1", "pattern2", "pattern3"]:
        # Calculate rolling mean and check that original values deviate from it
        rolling_mean = result[col].rolling(window=7, center=True).mean()
        # Skip first and last few points where rolling mean isn't defined
        diffs = result[col][7:-7] - rolling_mean[7:-7]
        assert np.std(diffs) > 0


def test_add_correlations(empty_dataframe, random_seed):
    """Test that correlation addition creates related series.
    
    Parameters
    ----------
    empty_dataframe : pd.DataFrame
        DataFrame with date column from fixture
    random_seed : int
        Random seed for reproducibility
        
    Notes
    -----
    The function should generate series with specified correlation strengths
    to existing series, creating a mix of correlated and anticorrelated series.
    """
    np.random.seed(random_seed)
    df = empty_dataframe.copy()
    
    # First add some base series
    df["base1"] = np.random.normal(100, 10, len(df))
    df["base2"] = np.random.normal(200, 20, len(df))
    
    # Call function
    result = add_correlations(
        df, 
        base_columns=["base1", "base2"],
        n_series=4, 
        correlation_range=(-0.8, 0.8)
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert "correlated_1" in result.columns
    assert "correlated_2" in result.columns
    assert "correlated_3" in result.columns
    assert "correlated_4" in result.columns
    
    # Calculate actual correlations
    corr_matrix = result.drop(columns=["date"]).corr()
    
    # Check correlations against base columns exist
    for i in range(1, 5):
        col_name = f"correlated_{i}"
        # At least one of the correlations should be significant
        max_corr = max(
            abs(corr_matrix.loc["base1", col_name]),
            abs(corr_matrix.loc["base2", col_name])
        )
        assert max_corr > 0.2  # Allowing for some randomness


def test_add_anomalies(empty_dataframe, random_seed):
    """Test that anomaly addition creates detectable outliers.
    
    Parameters
    ----------
    empty_dataframe : pd.DataFrame
        DataFrame with date column from fixture
    random_seed : int
        Random seed for reproducibility
        
    Notes
    -----
    The function should add point anomalies and anomalous periods to
    the time series that are detectable as statistical outliers.
    """
    np.random.seed(random_seed)
    df = empty_dataframe.copy()
    
    # Add a base series
    df["base"] = np.random.normal(100, 10, len(df))
    
    # Call function
    result = add_anomalies(
        df, 
        series_columns=["base"],
        point_anomaly_prob=0.01,
        period_anomaly_prob=0.005,
        anomaly_scale=5.0
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert "base" in result.columns
    
    # Calculate z-scores
    mean = result["base"].mean()
    std = result["base"].std()
    z_scores = np.abs((result["base"] - mean) / std)
    
    # Check for presence of outliers (z-score > 3)
    assert np.sum(z_scores > 3) > 0


def test_save_data_to_csv():
    """Test that data is correctly saved to CSV.
    
    Notes
    -----
    The function should properly save a DataFrame to CSV format
    with the expected structure and content.
    """
    # Create a simple test dataframe
    test_df = pd.DataFrame({
        "date": [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 3)],
        "series1": [10, 11, 12],
        "series2": [20, 21, 22],
    })
    
    # Use a temp file for testing
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = temp.name
    
    try:
        # Call function
        save_data_to_csv(test_df, temp_path)
        
        # Check file exists
        assert os.path.exists(temp_path)
        
        # Read back and verify content
        df_read = pd.read_csv(temp_path)
        
        assert "date" in df_read.columns
        assert "series1" in df_read.columns
        assert "series2" in df_read.columns
        assert len(df_read) == 3
        
        # Convert date back to datetime for comparison
        df_read["date"] = pd.to_datetime(df_read["date"]).dt.date
        
        # Check values match
        assert df_read["series1"].tolist() == [10, 11, 12]
        assert df_read["series2"].tolist() == [20, 21, 22]
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_generate_data_for_dashboard(random_seed):
    """Test the main data generation function's output.
    
    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility
        
    Notes
    -----
    This test verifies that the main generation function produces a dataset
    with the expected structure, number of series, and date range.
    """
    np.random.seed(random_seed)
    
    # Call function with specific parameters
    result = generate_data_for_dashboard(
        start_date=date(2023, 1, 1),
        end_date=date(2023, 3, 31),
        n_trend_series=2,
        n_seasonal_series=2,
        n_pattern_series=2,
        n_correlated_series=2,
        base_value=100,
        seed=random_seed,
    )
    
    # Check structure
    assert isinstance(result, pd.DataFrame)
    assert "date" in result.columns
    
    # Check date range
    assert result["date"].iloc[0] == date(2023, 1, 1)
    assert result["date"].iloc[-1] == date(2023, 3, 31)
    assert len(result) == 90  # Days in Jan-Mar 2023
    
    # Check number of series
    expected_cols = 1 + 2 + 2 + 2 + 2  # date + trends + seasonal + pattern + correlated
    assert len(result.columns) == expected_cols
    
    # Check that values are within a reasonable range
    for col in result.columns:
        if col != "date":  # Skip date column
            assert not result[col].isna().any()  # No NaN values
            
            # Basic range check - values shouldn't be extreme
            assert result[col].min() > -1000
            assert result[col].max() < 1000


def test_full_pipeline_with_saving():
    """Test the full data generation and saving pipeline.
    
    Notes
    -----
    This test verifies that the complete pipeline of generating
    synthetic data and saving it to a file works as expected.
    """
    # Use a temp file for testing
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
        temp_path = temp.name
    
    try:
        # Set parameters
        start_date = date(2023, 1, 1)
        end_date = date(2023, 1, 31)
        n_series = 5
        seed = 42
        
        # Generate data and save
        np.random.seed(seed)
        df = generate_data_for_dashboard(
            start_date=start_date,
            end_date=end_date,
            n_trend_series=n_series,
            n_seasonal_series=0,
            n_pattern_series=0,
            n_correlated_series=0,
            base_value=100,
            seed=seed,
        )
        save_data_to_csv(df, temp_path)
        
        # Check file exists
        assert os.path.exists(temp_path)
        
        # Read back and verify
        df_read = pd.read_csv(temp_path)
        
        assert len(df_read) == 31  # Days in January
        assert len(df_read.columns) == 1 + n_series  # date + n_series
        
        # Check date formatting in CSV
        date_col = pd.to_datetime(df_read["date"])
        assert date_col.iloc[0].date() == start_date
        assert date_col.iloc[-1].date() == end_date
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_statistical_properties_of_generated_data(random_seed):
    """Test statistical properties of the generated data.
    
    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility
        
    Notes
    -----
    This test verifies that the generated data has expected statistical
    properties such as distributions, stationarity, and autocorrelation.
    """
    np.random.seed(random_seed)
    
    # Generate test data
    df = generate_data_for_dashboard(
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31),
        n_trend_series=1,
        n_seasonal_series=1,
        n_pattern_series=1,
        n_correlated_series=1,
        base_value=100,
        seed=random_seed,
    )
    
    # Get columns excluding date
    data_columns = [col for col in df.columns if col != "date"]
    
    # Test for non-stationarity in trend series
    # Typically we'd use an ADF test, but simplified here
    trend_col = [col for col in data_columns if "trend" in col][0]
    first_quarter = df[trend_col].iloc[:90].mean()
    last_quarter = df[trend_col].iloc[-90:].mean()
    assert abs(first_quarter - last_quarter) > 5  # Significant difference indicates trend
    
    # Test for seasonality using autocorrelation
    seasonal_col = [col for col in data_columns if "season" in col][0]
    # For annual patterns, we'd need multiple years, so check shorter cycles
    seasonal_data = df[seasonal_col].values
    if len(seasonal_data) > 60:  # At least 2 months
        # Calculate autocorrelation at ~30-day lag
        acf_30 = np.corrcoef(seasonal_data[30:], seasonal_data[:-30])[0, 1]
        assert abs(acf_30) > 0.3  # Moderate correlation at seasonal lag
    
    # Check normality of noise in pattern series
    pattern_col = [col for col in data_columns if "pattern" in col][0]
    # Detrend by differencing
    diff = df[pattern_col].diff().dropna()
    # Shapiro-Wilk test for normality
    _, p_value = stats.shapiro(diff)
    # Residuals should be approximately normal for properly generated patterns
    # This might not always be true depending on the pattern type, so use a loose threshold
    assert p_value > 0.001  # Extremely low p-values would indicate non-normality


if __name__ == "__main__":
    pytest.main(["-v", __file__])