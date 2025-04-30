"""
Data Retrieval Module - daily price (ETFs data and ff5+MoM)
========================

This module retrieves:
1. historical daily price data for a selection of ETFs from Yahoo Finance
   and saves it to a CSV file in a wide-table format.
2. Fama-French 5+momentum daily data from Kenneth French Data Library
   and saves it to a CSV file in a wide-table format.

The script fetches daily closing prices for the following ETFs:
- SPY:       SPDR S&P 500 ETF
- IWM:       iShares Russell 2000 ETF
- EFA:       iShares MSCI EAFE ETF
- EEM:       iShares MSCI Emerging Markets ETF
- AGG:       iShares Core U.S. Aggregate Bond ETF
- LQD:       iShares iBoxx $ Investment Grade Corporate Bond ETF
- HYG:       iShares iBoxx $ High Yield Corporate Bond ETF
- TLT:       iShares 20+ Year Treasury Bond ETF
- GLD:       SPDR Gold Shares
- VNQ:       Vanguard Real Estate ETF
- DBC:       Invesco DB Commodity Index Tracking Fund
- VT:        Vanguard Total World Stock ETF
- XLE:       Energy Select Sector SPDR Fund
- XLK:       Technology Select Sector SPDR Fund
- UUP:       Invesco DB US Dollar Index Bullish Fund

Functions
---------
get_etf_data
    Retrieves historical daily price data from Yahoo Finance
get_ff5_mom_data
    Retrieves Fama-French 5+momentum daily data from Yahoo Finance
main
    Main execution function that retrieves data and saves to CSV

To Run This
--------
In teminal:
    python -m src.utils.get_data
"""
import os
import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def check_file_exists(file_path: Path):
    """check if file eixtst"""
    if not file_path.exists():
        raise FileNotFoundError(f"Cannot find file: {file_path}")
    print(f"file exists: {file_path}")


def get_etf_data(
    tickers: list,
    start_date: str,
    end_date: str,
    save_path: str
) -> pd.DataFrame:
    """Download historical ETF data from Yahoo Finance and save to CSV.
    
    Downloads the closing prices for specified ETF tickers and saves the data as CSV.
    
    Parameters
    ----------
    tickers : list
        List of ETF ticker symbols to retrieve.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    save_path : str
        Path to save the CSV file.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the ETF data with Date as the index.
    """
    try:
        print(f"Downloading ETF data for {len(tickers)} tickers from {start_date} to {end_date}...")
        
        # Download the data using yfinance
        etf_data = yf.download(tickers, start=start_date, end=end_date)["Close"]
        print(etf_data.head())
        print()
        
        # Save to CSV
        etf_data.to_csv(save_path)
        print(f"Data saved to {save_path}")
        
        return etf_data
    except Exception as e:
        print(f"Error downloading ETF data: {e}")
        raise


def get_ff5_mom_data(
    ff5_file: str,
    mom_file: str,
    start_date: str,
    end_date: str,
    save_path: str
) -> pd.DataFrame:
    """Download FF5 and MOM data, process, and save to CSV.
    
    Reads the FF5 and MOM data, processes them into a single DataFrame, and saves to CSV.
    
    Parameters
    ----------
    ff5_file : str
        Path to FF5 data CSV file.
    mom_file : str
        Path to MOM data CSV file.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.
    save_path : str
        Path to save the combined CSV file.
        
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with FF5 and MOM data.
    """
    try:
        # Load FF5 data
        ff_factors = pd.read_csv(ff5_file, skiprows=3, dtype={0: str})
        ff_factors = ff_factors[ff_factors.iloc[:, 0].str.match(r'^\d{8}$', na=False)]

        ff_factors = (
            ff_factors.rename(columns={ff_factors.columns[0]: "Date"})
            .assign(Date=lambda x: pd.to_datetime(x["Date"], format="%Y%m%d", errors='coerce'))
            .dropna(subset=["Date"])
            .set_index("Date")
        )

        numeric_cols = ff_factors.select_dtypes(include='number').columns
        ff_factors[numeric_cols] = ff_factors[numeric_cols].astype(float) / 100
        print(ff_factors.head())
        print()
        
        # Load MOM data
        mom_data = pd.read_csv(mom_file, skiprows=13, encoding="latin1")
        mom_data.columns = mom_data.columns.str.strip()
        mom_data.rename(columns={"Mom": "MOM"}, inplace=True)
        mom_data.rename(columns={mom_data.columns[0]: "Date"}, inplace=True)
        mom_data = mom_data[mom_data["Date"].astype(str).str.len() == 8].copy()
        mom_data["Date"] = pd.to_datetime(mom_data["Date"], format="%Y%m%d")
        mom_data.set_index("Date", inplace=True)
        mom_data.replace([-99.99, -999], pd.NA, inplace=True)
        mom_data["MOM"] = mom_data["MOM"].astype(float) / 100
        
        # Time Range
        ff_factors = ff_factors.loc[start_date:end_date]
        mom_data = mom_data.loc[start_date:end_date]

        ff_factors = ff_factors.merge(mom_data, left_index=True, right_index=True, how="left")
        print(ff_factors.head())
        
        # Save to CSV
        ff_factors.to_csv(save_path)
        print(f"Combined data saved to {save_path}")
        
        return ff_factors
    except FileNotFoundError as fnf_error:
        print(f"File not found error: {fnf_error}")
        raise
    except pd.errors.ParserError as parse_error:
        print(f"Error parsing data: {parse_error}")
        raise
    except Exception as e:
        print(f"Error downloading or processing FF5 and MOM data: {e}")
        raise


def main():
    """Main execution function to download and save ETF and FF5 + MOM data."""
    try:
        # Paths for saving data
        project_dir = Path(__file__).resolve().parent.parent.parent
        data_dir = project_dir / "data"
        raw_data_dir = data_dir / "raw"
        processed_data_dir = data_dir / "processed"

        # Create directories if not exist
        raw_data_dir.mkdir(parents=True, exist_ok=True)
        processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # ETF data download
        etf_tickers = [
            "SPY", "IWM", "EFA", "EEM", "AGG", "LQD", "HYG", "TLT", "GLD", "VNQ", "DBC", "VT", "XLE", "XLK", "UUP"
        ]
        start_date = "2008-07-01"
        end_date = "2025-03-01"
        etf_save_path = raw_data_dir / "etf_data.csv"
        
        # Download ETF data and save to CSV
        etf_data = get_etf_data(etf_tickers, start_date, end_date, etf_save_path)
        
        # FF5 and MOM data download
        ff5_file_path = raw_data_dir / "F-F_Research_Data_5_Factors_2x3_daily.CSV"
        mom_file_path = raw_data_dir / "F-F_Momentum_Factor_daily.CSV"
        ff5_mom_save_path = processed_data_dir / "ff5_mom_data.csv"
        
        # Download and combine FF5 and MOM data, then save to CSV
        ff5_mom_data = get_ff5_mom_data(ff5_file_path, mom_file_path, start_date, end_date, ff5_mom_save_path)
    
    except Exception as e:
        print(f"Errors occurred during data download and processing: {e}")
        raise

if __name__ == "__main__":
    main()
