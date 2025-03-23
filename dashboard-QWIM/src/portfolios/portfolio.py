"""
Portfolio Module
===============

This module provides functionality for managing financial portfolios with time-varying weights.

Classes
-------
portfolio
    A class representing a financial portfolio with weights of components over time.
"""

import polars as pl
from datetime import datetime
from typing import List, Optional, Union


class portfolio:
    """A class representing a financial portfolio with weights of components over time.

    The portfolio is represented as a DataFrame with dates and component weights.

    Attributes
    ----------
    _portfolio_weights : pl.DataFrame
        DataFrame containing dates and portfolio weights.
    _portfolio_components : List[str]
        List of strings containing names of the portfolio components.
    _num_components : int
        Number of portfolio components.
    _portfolio_name : str
        Name of the portfolio.

    Examples
    --------
    Creating a portfolio with equal weights:

    >>> import polars as pl
    >>> from portfolios.portfolio import portfolio
    >>> # Create portfolio with equal weights for three components
    >>> p1 = portfolio(name_portfolio="Tech Portfolio", names_components=["AAPL", "MSFT", "GOOG"])
    >>> p1.get_portfolio_components
    ['AAPL', 'MSFT', 'GOOG']
    >>> p1.get_num_components
    3
    >>> p1.get_portfolio_name
    'Tech Portfolio'

    Creating a portfolio from existing data:

    >>> # Create portfolio from existing data
    >>> df = pl.DataFrame({
    ...     "Date": ["2023-01-01", "2023-02-01"],
    ...     "AAPL": [0.5, 0.6],
    ...     "MSFT": [0.5, 0.4]
    ... })
    >>> p2 = portfolio(name_portfolio="Apple-Microsoft Mix", portfolio_weights=df)
    >>> p2.get_num_components
    2
    >>> p2.get_portfolio_name
    'Apple-Microsoft Mix'
    """

    def __init__(
        self,
        name_portfolio: str,  # Move to the beginning as it's required
        portfolio_weights: Optional[pl.DataFrame] = None,
        names_components: Optional[List[str]] = None,
        date: Optional[Union[str, datetime]] = None,
    ):
        """Initialize a portfolio object.

        This constructor supports two initialization patterns:
        1. From an existing DataFrame containing dates and component weights
        2. From a list of component names and an optional date, creating equal weights

        Parameters
        ----------
        name_portfolio : str
            Name of the portfolio. This is a required parameter.
        portfolio_weights : pl.DataFrame, optional
            DataFrame with dates and portfolio weights.
        names_components : List[str], optional
            Names of portfolio components.
        date : str or datetime, optional
            Date for the portfolio weights, defaults to current date if not provided.

        Raises
        ------
        ValueError
            If neither portfolio_weights nor names_components is provided.
            If portfolio_weights does not have a 'Date' column.

        Notes
        -----
        When both portfolio_weights and names_components are provided, 
        portfolio_weights takes precedence.
        """
        # Store the portfolio name
        self._portfolio_name = name_portfolio
        
        # Check initialization pattern
        if portfolio_weights is not None:
            self._initialize_from_dataframe(portfolio_weights)
        elif names_components is not None:
            self._initialize_from_components(names_components, date)
        else:
            raise ValueError("Either portfolio_weights or names_components must be provided")

    def _initialize_from_dataframe(self, portfolio_weights: pl.DataFrame) -> None:
        """Initialize portfolio from an existing DataFrame.

        Parameters
        ----------
        portfolio_weights : pl.DataFrame
            DataFrame with dates and portfolio weights.

        Raises
        ------
        ValueError
            If portfolio_weights does not have a 'Date' column.
        
        Warning
        -------
        This method assumes that all non-Date columns are portfolio components.
        Ensure your DataFrame doesn't contain any non-component columns besides 'Date'.
        """
        # Validate the dataframe
        if "Date" not in portfolio_weights.columns:
            raise ValueError("portfolio_weights must have a 'Date' column")

        # Store the dataframe
        self._portfolio_weights = portfolio_weights.clone()

        # Extract component names (all columns except 'Date')
        self._portfolio_components = [col for col in portfolio_weights.columns if col != "Date"]
        self._num_components = len(self._portfolio_components)

        # Validate that Date is the first column
        if portfolio_weights.columns[0] != "Date":
            # Reorder columns to ensure Date is first
            cols = ["Date"] + self._portfolio_components
            self._portfolio_weights = self._portfolio_weights.select(cols)

    def _initialize_from_components(
        self,
        names_components: List[str],
        date: Optional[Union[str, datetime]] = None,
    ) -> None:
        """Initialize portfolio with equal weights for given components.

        Parameters
        ----------
        names_components : List[str]
            Names of portfolio components.
        date : str or datetime, optional
            Date for the portfolio weights, defaults to current date if not provided.
        
        Notes
        -----
        Equal weights are calculated as:
        
        .. math::
           w_i = \\frac{1}{n}
           
        Where:
        
        - :math:`w_i` is the weight of component :math:`i`
        - :math:`n` is the number of components
        """
        # Set the date to current date if not provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        elif isinstance(date, datetime):
            date = date.strftime("%Y-%m-%d")

        # Calculate equal weights
        num_components = len(names_components)
        equal_weight = 1.0 / num_components

        # Create the dataframe with one row
        data = {"Date": [date]}
        for component in names_components:
            data[component] = [equal_weight]

        # Store the portfolio data
        self._portfolio_weights = pl.DataFrame(data)
        self._portfolio_components = names_components
        self._num_components = num_components

    @property
    def get_portfolio_weights(self) -> pl.DataFrame:
        """Get the portfolio weights DataFrame.

        Returns
        -------
        pl.DataFrame
            A copy of the portfolio weights DataFrame.

        Notes
        -----
        Returns a clone of the internal DataFrame to prevent unintended modifications.
        """
        return self._portfolio_weights.clone()

    @property
    def get_portfolio_components(self) -> List[str]:
        """Get the list of portfolio components.

        Returns
        -------
        List[str]
            A copy of the list of portfolio component names.
        """
        return self._portfolio_components.copy()

    @property
    def get_num_components(self) -> int:
        """Get the number of portfolio components.

        Returns
        -------
        int
            The number of components in the portfolio.
        """
        return self._num_components

    @property
    def get_portfolio_name(self) -> str:
        """Get the name of the portfolio.
        
        Returns
        -------
        str
            The name of the portfolio.
            
        See Also
        --------
        set_portfolio_name : Method to set a new portfolio name
        """
        return self._portfolio_name

    def set_portfolio_name(self, name: str) -> None:
        """Set a new name for the portfolio.

        Parameters
        ----------
        name : str
            The new portfolio name.
        """
        self._portfolio_name = name

    def __str__(self) -> str:
        """Return string representation of the portfolio.

        Returns
        -------
        str
            A string representation showing the portfolio name, number of components, 
            and the portfolio weights.
        """
        return (
            f"Portfolio '{self._portfolio_name}' with {self._num_components} components:"
            f"\n{self._portfolio_weights}"
        )

    def __repr__(self) -> str:
        """Return string representation for developers.

        Returns
        -------
        str
            A detailed string representation showing the portfolio name, components, 
            and number of dates.
        """
        return (
            f"portfolio(name='{self._portfolio_name}', "
            f"components={self._portfolio_components}, "
            f"dates={len(self._portfolio_weights)} rows)"
        )

