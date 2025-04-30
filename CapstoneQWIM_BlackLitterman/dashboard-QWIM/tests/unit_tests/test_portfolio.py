#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the portfolio class.

This module contains pytest-based tests for the portfolio class in the
portfolios.portfolio module, covering initialization, property access,
and other functionality.
"""

import pytest
import polars as pl
from datetime import datetime
import sys
import os
from typing import List

# Add the src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the portfolio class
from src.portfolios.portfolio import portfolio


# Fixtures
@pytest.fixture
def sample_components() -> List[str]:
    """Return a list of sample component names."""
    return ["AAPL", "MSFT", "GOOG", "AMZN"]


@pytest.fixture
def sample_weights_df() -> pl.DataFrame:
    """Return a sample portfolio weights DataFrame."""
    return pl.DataFrame({
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "AAPL": [0.4, 0.45, 0.5],
        "MSFT": [0.3, 0.25, 0.3],
        "GOOG": [0.3, 0.3, 0.2]
    })


# Tests for portfolio initialization
class TestPortfolioInitialization:
    """Tests for the portfolio initialization methods."""

    def test_init_with_components_and_name(self, sample_components):
        """Test initializing portfolio with component names and a portfolio name."""
        port = portfolio(names_components=sample_components, name_portfolio="Test Portfolio")
        
        # Check that properties were set correctly
        assert port.get_portfolio_name == "Test Portfolio"
        assert port.get_num_components == len(sample_components)
        assert port.get_portfolio_components == sample_components
        
        # Check that weights dataframe was created with equal weights
        weights_df = port.get_portfolio_weights
        assert "Date" in weights_df.columns
        for component in sample_components:
            assert component in weights_df.columns
            # Check equal weights (1.0 / num_components)
            assert weights_df[component][0] == pytest.approx(1.0 / len(sample_components))

    def test_init_with_components_and_date(self, sample_components):
        """Test initializing portfolio with component names and a specific date."""
        test_date = "2022-12-15"
        port = portfolio(
            names_components=sample_components, 
            date=test_date,
            name_portfolio="Dated Portfolio"
        )
        
        # Check that the date was set correctly
        weights_df = port.get_portfolio_weights
        assert weights_df["Date"][0] == test_date

    def test_init_with_dataframe(self, sample_weights_df):
        """Test initializing portfolio with an existing weights DataFrame."""
        port = portfolio(
            portfolio_weights=sample_weights_df,
            name_portfolio="DataFrame Portfolio"
        )
        
        # Check that properties were set correctly
        assert port.get_portfolio_name == "DataFrame Portfolio"
        assert port.get_num_components == 3  # AAPL, MSFT, GOOG
        assert set(port.get_portfolio_components) == {"AAPL", "MSFT", "GOOG"}
        
        # Check that the weights dataframe matches the input
        weights_df = port.get_portfolio_weights
        assert weights_df.shape == sample_weights_df.shape
        # Verify some specific values
        assert weights_df["AAPL"][0] == 0.4
        assert weights_df["MSFT"][2] == 0.3

    def test_init_with_datetime_object(self, sample_components):
        """Test initializing portfolio with a datetime object for the date."""
        test_date = datetime(2022, 12, 15)
        expected_date_str = "2022-12-15"
        
        port = portfolio(
            names_components=sample_components, 
            date=test_date,
            name_portfolio="Datetime Portfolio"
        )
        
        # Check that the date was converted to string correctly
        weights_df = port.get_portfolio_weights
        assert weights_df["Date"][0] == expected_date_str

    def test_init_requires_portfolio_name(self, sample_components):
        """Test that portfolio initialization requires a name."""
        # Should raise TypeError if name_portfolio is not provided
        with pytest.raises(TypeError):
            portfolio(names_components=sample_components)

    def test_init_requires_components_or_dataframe(self):
        """Test that portfolio initialization requires either components or dataframe."""
        # Should raise ValueError if neither is provided
        with pytest.raises(ValueError):
            portfolio(name_portfolio="Empty Portfolio")

    def test_init_validates_dataframe_has_date_column(self):
        """Test that portfolio initialization validates the dataframe has a Date column."""
        # Create DataFrame without a Date column
        invalid_df = pl.DataFrame({
            "AAPL": [0.6, 0.7],
            "MSFT": [0.4, 0.3]
        })
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            portfolio(portfolio_weights=invalid_df, name_portfolio="Invalid DF Portfolio")


# Tests for portfolio properties and methods
class TestPortfolioProperties:
    """Tests for the portfolio properties and methods."""

    def test_get_portfolio_weights(self, sample_weights_df):
        """Test the get_portfolio_weights property."""
        port = portfolio(
            portfolio_weights=sample_weights_df,
            name_portfolio="Weights Test Portfolio"
        )
        
        # Get the weights
        weights = port.get_portfolio_weights
        
        # Check that it's a clone (not the same object)
        assert weights is not sample_weights_df
        
        # But should have the same data
        assert weights.shape == sample_weights_df.shape
        assert weights.columns == sample_weights_df.columns

    def test_get_portfolio_components(self, sample_components):
        """Test the get_portfolio_components property."""
        port = portfolio(
            names_components=sample_components, 
            name_portfolio="Components Test Portfolio"
        )
        
        # Get the components
        components = port.get_portfolio_components
        
        # Check that it's a copy (not the same object)
        assert components is not sample_components
        
        # But should have the same data
        assert components == sample_components

    def test_get_num_components(self, sample_components):
        """Test the get_num_components property."""
        port = portfolio(
            names_components=sample_components, 
            name_portfolio="Count Test Portfolio"
        )
        
        # Check that the number of components is correct
        assert port.get_num_components == len(sample_components)

    def test_get_portfolio_name(self):
        """Test the get_portfolio_name property."""
        test_name = "Name Test Portfolio"
        port = portfolio(
            names_components=["A", "B"], 
            name_portfolio=test_name
        )
        
        # Check that the name is correct
        assert port.get_portfolio_name == test_name

    def test_set_portfolio_name(self):
        """Test the set_portfolio_name method."""
        original_name = "Original Name"
        new_name = "New Name"
        
        port = portfolio(
            names_components=["A", "B"], 
            name_portfolio=original_name
        )
        
        # Change the name
        port.set_portfolio_name(new_name)
        
        # Check that the name was updated
        assert port.get_portfolio_name == new_name


# Tests for string representation methods
class TestPortfolioStringRepresentation:
    """Tests for the portfolio string representation methods."""

    def test_str_representation(self, sample_components):
        """Test the __str__ method."""
        port = portfolio(
            names_components=sample_components, 
            name_portfolio="String Test Portfolio"
        )
        
        # Get the string representation
        str_rep = str(port)
        
        # Check that it contains key information
        assert "String Test Portfolio" in str_rep
        assert str(len(sample_components)) in str_rep
        
        # Should also contain a representation of the DataFrame
        assert "Date" in str_rep
        for component in sample_components:
            assert component in str_rep

    def test_repr_representation(self, sample_components):
        """Test the __repr__ method."""
        port = portfolio(
            names_components=sample_components, 
            name_portfolio="Repr Test Portfolio"
        )
        
        # Get the repr representation
        repr_rep = repr(port)
        
        # Check that it contains key information
        assert "name='Repr Test Portfolio'" in repr_rep
        assert "components=" in repr_rep
        assert "dates=" in repr_rep
        
        # Should include all component names
        for component in sample_components:
            assert component in repr_rep


# Tests for dataframe handling
class TestPortfolioDataFrameHandling:
    """Tests for portfolio dataframe handling."""

    def test_dataframe_column_order(self):
        """Test that the Date column is always first in the DataFrame."""
        # Create a DataFrame with Date not as the first column
        df = pl.DataFrame({
            "AAPL": [0.6, 0.7],
            "Date": ["2023-01-01", "2023-01-02"],
            "MSFT": [0.4, 0.3]
        })
        
        port = portfolio(portfolio_weights=df, name_portfolio="Column Order Test")
        
        # Check that Date is the first column in the returned DataFrame
        weights_df = port.get_portfolio_weights
        assert weights_df.columns[0] == "Date"
        
        # And other columns follow
        assert "AAPL" in weights_df.columns
        assert "MSFT" in weights_df.columns

    def test_dataframe_modification_isolation(self, sample_weights_df):
        """Test that modifying the returned DataFrame doesn't affect the internal one."""
        port = portfolio(
            portfolio_weights=sample_weights_df,
            name_portfolio="Isolation Test Portfolio"
        )
        
        # Get the weights DataFrame and modify it
        weights_df = port.get_portfolio_weights
        
        # Try to modify the DataFrame (add a column)
        # This will create a new DataFrame in Polars
        modified_df = weights_df.with_columns(pl.lit(1.0).alias("NEW_COL"))
        
        # Get the weights again
        weights_df2 = port.get_portfolio_weights
        
        # The internal DataFrame should not have the new column
        assert "NEW_COL" not in weights_df2.columns


if __name__ == "__main__":
    pytest.main(["-v", __file__])