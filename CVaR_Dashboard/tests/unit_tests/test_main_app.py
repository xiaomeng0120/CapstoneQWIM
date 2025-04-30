"""
Unit tests for the QWIM Dashboard main application.

This module provides unit tests for the main Shiny application components
defined in main_App.py. Tests cover application initialization, data loading,
series extraction, server functions, and UI components.

Tests
-----
* Data Loading Tests: Verify data loading and processing functions
* UI Component Tests: Validate that UI components are correctly generated
* Server Function Tests: Test application server functionality
* Theme Management Tests: Verify theme application and updates
* System Info Tests: Test system information gathering and display

Notes
-----
Like other tests in this suite, these tests use unittest.mock to simulate
Shiny's reactive environment without requiring a running Shiny server.
"""

import datetime
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
from shiny import reactive

# Add project root to path for imports
project_dir = Path(__file__).parent.parent.parent
if str(project_dir) not in sys.path:
    sys.path.insert(0, str(project_dir))

# Import components to test
from src.dashboard.main_App import (
    app,
    app_Server,
    app_UI,
    get_data,
    get_series_names,
)


@pytest.fixture
def mock_csv_data():
    """Create mock CSV data for testing.
    
    This fixture generates a CSV string representing time series data
    that can be used to mock file reading operations.
    
    Returns
    -------
    str
        CSV content as a string with header and sample data rows
        
    Notes
    -----
    The CSV contains a date column and three series columns with sample data.
    """
    return (
        "date,series1,series2,series3\n"
        "2023-01-01,100,50,30\n"
        "2023-02-01,101,51,32\n"
        "2023-03-01,102,53,28\n"
        "2023-04-01,103,52,35\n"
    )


@pytest.fixture
def sample_polars_df():
    """Create a sample polars DataFrame for testing.
    
    This fixture generates a polars DataFrame with time series data
    that mimics the structure returned by get_data().
    
    Returns
    -------
    pl.DataFrame
        DataFrame with date column and three series columns
        
    Notes
    -----
    The DataFrame contains a proper date column (not string) and
    numeric values for each series, similar to what would be loaded
    from a CSV file.
    """
    data = {
        "date": [
            datetime.date(2023, 1, 1),
            datetime.date(2023, 2, 1),
            datetime.date(2023, 3, 1),
            datetime.date(2023, 4, 1),
        ],
        "series1": [100.0, 101.0, 102.0, 103.0],
        "series2": [50.0, 51.0, 53.0, 52.0],
        "series3": [30.0, 32.0, 28.0, 35.0],
    }
    
    return pl.DataFrame(data)


@pytest.fixture
def mock_session():
    """Create a mock Shiny session object for testing.
    
    Returns
    -------
    unittest.mock.MagicMock
        Mocked session with required methods and properties
        
    Notes
    -----
    The mock includes common session attributes needed for testing,
    including client information and reactive state.
    """
    session = MagicMock()
    session.id = "test-session-id"
    session.client_address = "127.0.0.1"
    
    # Mock on_ended decorator
    session.on_ended = lambda f: f
    
    return session


@pytest.fixture
def mock_input():
    """Create a mock Shiny input object for testing.
    
    Returns
    -------
    unittest.mock.MagicMock
        Mock object that simulates Shiny's input object
        
    Notes
    -----
    The mock input is configured to return appropriate values for
    relevant input IDs used in the main application.
    """
    input_values = {
        "use_caching": True,
        "max_points": 5000,
        "app_theme": "cosmo",
        "apply_theme": 0,
        "upload_data": None,
    }
    
    mock_input = MagicMock()
    
    # Make the mock input work like a callable dictionary
    def side_effect(key):
        return input_values.get(key)
    
    mock_input.side_effect = side_effect
    
    return mock_input


@pytest.fixture
def mock_output():
    """Create a mock Shiny output object for testing.
    
    Returns
    -------
    unittest.mock.MagicMock
        Mock object that simulates Shiny's output object
    """
    return MagicMock()


def test_get_data_loads_data_correctly(mock_csv_data):
    """Test that get_data loads and processes the CSV data correctly.
    
    Parameters
    ----------
    mock_csv_data : str
        Mock CSV data fixture
        
    Notes
    -----
    This test mocks the file reading operation to verify that get_data
    properly loads the data and formats the date column.
    """
    # Mock the file existence check
    with patch("pathlib.Path.exists", return_value=True), \
         patch("polars.read_csv", return_value=pl.read_csv(mock_csv_data.splitlines())):
        
        # Call the function
        result = get_data()
        
        # Check that the result is a polars DataFrame
        assert isinstance(result, pl.DataFrame)
        
        # Check that it has the expected columns
        assert "date" in result.columns
        assert "series1" in result.columns
        assert "series2" in result.columns
        assert "series3" in result.columns
        
        # Check that the date column is properly formatted as date type
        assert result.schema["date"] == pl.Date


def test_get_data_raises_error_when_file_not_found():
    """Test that get_data raises FileNotFoundError when data file is missing.
    
    Notes
    -----
    This test confirms that the function properly handles missing files
    by raising an appropriate exception rather than failing silently.
    """
    # Mock the file existence check to return False
    with patch("pathlib.Path.exists", return_value=False):
        # Check that the function raises FileNotFoundError
        with pytest.raises(FileNotFoundError):
            get_data()


def test_get_series_names_extracts_correct_columns(sample_polars_df):
    """Test that get_series_names correctly identifies series columns.
    
    Parameters
    ----------
    sample_polars_df : pl.DataFrame
        Sample DataFrame fixture
        
    Notes
    -----
    The function should return all column names except 'date' as series names.
    """
    # Call the function
    result = get_series_names(sample_polars_df)
    
    # Check that we got the expected columns
    assert isinstance(result, list)
    assert len(result) == 3
    assert "series1" in result
    assert "series2" in result
    assert "series3" in result
    assert "date" not in result


def test_app_object_exists():
    """Test that the app object is correctly instantiated.
    
    Notes
    -----
    This test simply verifies that the app object exists and appears to be
    a Shiny app instance with the expected server and UI components.
    """
    # Check that app exists
    assert app is not None
    
    # Check that it has expected properties (very basic check)
    assert hasattr(app, "ui")
    assert hasattr(app, "server")


def test_app_UI_contains_expected_components():
    """Test that the app_UI contains the expected structural components.
    
    Notes
    -----
    This test checks for the presence of key UI elements by examining
    the string representation of the UI object, without fully parsing
    the structure.
    """
    # Convert UI to string for simple parsing
    ui_str = str(app_UI)
    
    # Check for major components
    assert "nav_panel" in ui_str
    assert "navset_tab" in ui_str
    assert "sidebar" in ui_str
    assert "Inputs" in ui_str
    assert "Data Analysis" in ui_str
    assert "Results" in ui_str
    assert "QWIM Dashboard" in ui_str


def test_server_initializes_reactive_values(
    mock_input, mock_output, mock_session, sample_polars_df
):
    """Test that the server function initializes reactive values correctly.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
    sample_polars_df : pl.DataFrame
        Sample DataFrame fixture
        
    Notes
    -----
    This test verifies that the server function creates reactive values
    with the expected data when initialized successfully.
    """
    # Mock get_data and get_series_names
    with patch("src.dashboard.main_App.get_data", return_value=sample_polars_df), \
         patch(
             "src.dashboard.main_App.get_series_names", 
             return_value=["series1", "series2", "series3"]
         ), \
         patch("src.dashboard.main_App.reactive.Value") as mock_reactive_value, \
         patch("src.dashboard.main_App.inputs_server", return_value=MagicMock()), \
         patch("src.dashboard.main_App.analysis_server"), \
         patch("src.dashboard.main_App.results_server"), \
         patch("src.dashboard.main_App.shinyswatch.theme_picker_server"):
        
        # Call the server function
        app_Server(mock_input, mock_output, mock_session)
        
        # Check that reactive.Value was called twice (for data_r and series_names_r)
        assert mock_reactive_value.call_count == 2


def test_server_handles_file_not_found_gracefully(
    mock_input, mock_output, mock_session
):
    """Test that the server handles missing data files without crashing.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
        
    Notes
    -----
    This test verifies that the server function creates placeholder data
    and shows a notification when the data file is not found.
    """
    # Mock get_data to raise FileNotFoundError
    with patch(
            "src.dashboard.main_App.get_data", 
            side_effect=FileNotFoundError("Test file not found")
         ), \
         patch("src.dashboard.main_App.reactive.Value") as mock_reactive_value, \
         patch("src.dashboard.main_App.inputs_server", return_value=MagicMock()), \
         patch("src.dashboard.main_App.analysis_server"), \
         patch("src.dashboard.main_App.results_server"), \
         patch("src.dashboard.main_App.ui.notification_show") as mock_notification, \
         patch("src.dashboard.main_App.shinyswatch.theme_picker_server"):
        
        # Call the server function
        app_Server(mock_input, mock_output, mock_session)
        
        # Check that a notification was shown
        mock_notification.assert_called_once()
        
        # Check that reactive.Value was still called twice with placeholder data
        assert mock_reactive_value.call_count == 2


def test_system_info_output_contains_expected_data(mock_input, mock_output, mock_session):
    """Test that the system_info function returns expected information.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
        
    Notes
    -----
    This test verifies that the system_info function returns a string
    containing key system information.
    """
    # Import just the system_info function
    from src.dashboard.main_App import system_info
    
    # Mock platform and package versions
    with patch("platform.python_version", return_value="3.9.0"), \
         patch("platform.platform", return_value="Test Platform"), \
         patch.object(np, "__version__", "1.22.0"), \
         patch.object(pd, "__version__", "1.4.0"), \
         patch.object(pl, "__version__", "0.15.0"):
        
        # Call the function
        result = system_info()
        
        # Check that result is a string
        assert isinstance(result, str)
        
        # Check for expected content
        assert "Python Version: 3.9.0" in result
        assert "Platform: Test Platform" in result
        assert "NumPy Version: 1.22.0" in result
        assert "Pandas Version: 1.4.0" in result
        assert "Polars Version: 0.15.0" in result


def test_data_summary_output(mock_input, mock_output, mock_session, sample_polars_df):
    """Test that the data_summary function formats data information correctly.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
    sample_polars_df : pl.DataFrame
        Sample DataFrame fixture
        
    Notes
    -----
    This test verifies that the data_summary function returns a properly
    formatted string with information about the loaded data.
    """
    # Import data_summary function
    from src.dashboard.main_App import data_summary
    
    # Create a mock reactive value that returns our sample dataframe
    mock_data_r = MagicMock()
    mock_data_r.return_value = sample_polars_df
    
    # Mock sys.getsizeof to return a predictable size
    with patch("src.dashboard.main_App.data_r", mock_data_r), \
         patch("sys.getsizeof", return_value=10 * 1024 * 1024):  # 10MB
        
        # Call the function
        result = data_summary()
        
        # Check that result is a string
        assert isinstance(result, str)
        
        # Check for expected content
        assert "Rows: 4" in result  # Our sample has 4 rows
        assert "Series: 3" in result  # Our sample has 3 series
        assert "Date Range" in result
        assert "Memory Usage: 10.00 MB" in result


def test_handle_file_upload(mock_input, mock_output, mock_session, mock_csv_data):
    """Test that the file upload handler processes uploaded files correctly.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
    mock_csv_data : str
        Mock CSV data fixture
        
    Notes
    -----
    This test verifies that the file upload handler correctly processes
    a valid CSV file and updates the reactive data values.
    """
    # Import the _handle_file_upload function
    from src.dashboard.main_App import _handle_file_upload
    
    # Create mock file info
    file_info = [{
        "name": "test.csv",
        "datapath": "path/to/test.csv",
        "size": len(mock_csv_data),
        "type": "text/csv",
    }]
    
    # Setup mock input to return file info
    mock_input.side_effect = lambda key: file_info if key == "upload_data" else None
    
    # Create mock reactive values to update
    mock_data_r = MagicMock()
    mock_series_names_r = MagicMock()
    
    # Setup patches
    with patch("src.dashboard.main_App.data_r", mock_data_r), \
         patch("src.dashboard.main_App.series_names_r", mock_series_names_r), \
         patch("polars.read_csv", return_value=pl.read_csv(mock_csv_data.splitlines())), \
         patch("src.dashboard.main_App.ui.notification_show") as mock_notification:
        
        # Call the function
        _handle_file_upload()
        
        # Check that reactive values were updated
        mock_data_r.set.assert_called_once()
        mock_series_names_r.set.assert_called_once()
        
        # Check that a success notification was shown
        mock_notification.assert_called_once()
        args = mock_notification.call_args[0]
        assert "success" in str(args).lower()


def test_handle_file_upload_validates_required_columns(
    mock_input, mock_output, mock_session
):
    """Test that file upload validation rejects files without required columns.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
        
    Notes
    -----
    This test verifies that the file upload handler shows an error notification
    when an uploaded CSV lacks the required 'date' column.
    """
    # Import the _handle_file_upload function
    from src.dashboard.main_App import _handle_file_upload
    
    # Create mock file info
    file_info = [{
        "name": "invalid.csv",
        "datapath": "path/to/invalid.csv",
        "size": 100,
        "type": "text/csv",
    }]
    
    # Setup mock input to return file info
    mock_input.side_effect = lambda key: file_info if key == "upload_data" else None
    
    # Create invalid CSV data (missing date column)
    invalid_csv = "col1,col2,col3\n1,2,3\n4,5,6"
    
    # Create mock reactive values
    mock_data_r = MagicMock()
    mock_series_names_r = MagicMock()
    
    # Setup patches
    with patch("src.dashboard.main_App.data_r", mock_data_r), \
         patch("src.dashboard.main_App.series_names_r", mock_series_names_r), \
         patch("polars.read_csv", return_value=pl.read_csv(invalid_csv.splitlines())), \
         patch("src.dashboard.main_App.ui.notification_show") as mock_notification:
        
        # Call the function
        _handle_file_upload()
        
        # Check that an error notification was shown
        mock_notification.assert_called_once()
        args = mock_notification.call_args[0]
        assert "date" in str(args).lower()
        assert "error" in str(mock_notification.call_args[1]).lower()
        
        # Check that reactive values were NOT updated
        mock_data_r.set.assert_not_called()
        mock_series_names_r.set.assert_not_called()


def test_handle_theme_change(mock_input, mock_output, mock_session):
    """Test that the theme change handler processes theme changes correctly.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
        
    Notes
    -----
    This test verifies that the theme change handler shows a notification
    when the theme is changed.
    """
    # Import the _handle_theme_change function
    from src.dashboard.main_App import _handle_theme_change
    
    # Setup mock input to return theme choice
    mock_input.side_effect = lambda key: "darkly" if key == "app_theme" else None
    
    # Setup patches
    with patch("src.dashboard.main_App.ui.notification_show") as mock_notification:
        
        # Call the function
        _handle_theme_change()
        
        # Check that a notification was shown
        mock_notification.assert_called_once()
        args = mock_notification.call_args[0]
        assert "darkly" in str(args).lower()


def test_setup_caching_configures_based_on_settings(
    mock_input, mock_output, mock_session
):
    """Test that the caching setup function configures caching correctly.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
        
    Notes
    -----
    This test verifies that the caching setup function reads input values
    correctly. Since the actual caching implementation is minimal in the code,
    we just check that the function runs without errors.
    """
    # Import the _setup_caching function
    from src.dashboard.main_App import _setup_caching
    
    # Setup mock input to return caching settings
    mock_input.side_effect = lambda key: (
        True if key == "use_caching" else
        5000 if key == "max_points" else
        None
    )
    
    # Call the function without error
    _setup_caching()
    # Since the function mainly logs information, we simply verify it doesn't raise exceptions


def test_session_info_output(mock_input, mock_output, mock_session):
    """Test that the session_info function returns expected session data.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
        
    Notes
    -----
    This test verifies that the session_info function returns a string
    containing key session information.
    """
    # Import the session_info function
    from src.dashboard.main_App import session_info
    
    # Setup mock input and session values
    mock_input.side_effect = lambda key: (
        True if key == "use_caching" else
        5000 if key == "max_points" else
        None
    )
    
    mock_session.id = "test-session-id"
    mock_session.client_address = "127.0.0.1"
    
    # Setup the patches
    with patch("src.dashboard.main_App.session", mock_session), \
         patch("src.dashboard.main_App.input", mock_input), \
         patch("src.dashboard.main_App.datetime") as mock_datetime:
        
        # Mock the datetime.now() call
        mock_datetime.now.return_value = datetime.datetime(2023, 4, 1, 12, 0, 0)
        mock_datetime.datetime = datetime.datetime
        
        # Call the function
        result = session_info()
        
        # Check that result is a string
        assert isinstance(result, str)
        
        # Check for expected content
        assert "Session ID: test-session-id" in result
        assert "Client Address: 127.0.0.1" in result
        assert "Started: 2023-04-01" in result
        assert "Cache Enabled: True" in result
        assert "Max Points: 5000" in result


def test_download_all_data(mock_input, mock_output, mock_session, sample_polars_df):
    """Test that the data download handler generates CSV data correctly.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
    sample_polars_df : pl.DataFrame
        Sample DataFrame fixture
        
    Notes
    -----
    This test verifies that the download handler returns CSV data
    when valid data is available.
    """
    # Import the download_all_data function
    from src.dashboard.main_App import download_all_data
    
    # Create a mock reactive value that returns our sample dataframe
    mock_data_r = MagicMock()
    mock_data_r.return_value = sample_polars_df
    
    # Setup patches
    with patch("src.dashboard.main_App.data_r", mock_data_r):
        
        # Call the function
        result = download_all_data()
        
        # Check that result is a string (CSV data)
        assert isinstance(result, str)
        
        # Check that it contains expected columns
        assert "date,series1,series2,series3" in result


def test_download_all_data_handles_empty_data(mock_input, mock_output, mock_session):
    """Test that the data download handler handles empty data gracefully.
    
    Parameters
    ----------
    mock_input : unittest.mock.MagicMock
        Mock input object fixture
    mock_output : unittest.mock.MagicMock
        Mock output object fixture
    mock_session : unittest.mock.MagicMock
        Mock session object fixture
        
    Notes
    -----
    This test verifies that the download handler returns an error message
    when no data is available.
    """
    # Import the download_all_data function
    from src.dashboard.main_App import download_all_data
    
    # Create a mock reactive value that returns an empty dataframe
    mock_data_r = MagicMock()
    mock_data_r.return_value = pl.DataFrame()
    mock_data_r.return_value.is_empty = lambda: True
    
    # Setup patches
    with patch("src.dashboard.main_App.data_r", mock_data_r):
        
        # Call the function
        result = download_all_data()
        
        # Check that result is a string with error message
        assert isinstance(result, str)
        assert "No data available" in result


if __name__ == "__main__":
    pytest.main(["-v", __file__])