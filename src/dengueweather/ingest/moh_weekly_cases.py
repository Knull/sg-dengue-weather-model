"""Download and parse MOH weekly infectious disease bulletin counts.

This module contains functions to load the weekly infectious disease bulletin
from data.gov.sg (a CSV) and optionally extend the series by scraping
weekly bulletins posted as PDFs on the MOH website.  The scraping logic
is not implemented here; you should add it as needed.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from ..logging_setup import setup_logging


def load_weekly_counts(csv_path: str, disease: str = "Dengue Fever") -> pd.DataFrame:
    """Load weekly infectious disease counts from the MOH CSV.

    Parameters
    ----------
    csv_path: str
        Path to the CSV downloaded from data.gov.sg.
    disease: str
        Name of the disease to filter on.  Defaults to ``"Dengue Fever"``.

    Returns
    -------
    DataFrame
        A DataFrame with columns ``['epi_week', 'year', 'value']`` for the
        selected disease.
    """
    setup_logging()
    df = pd.read_csv(csv_path)
    df = df[df["disease"] == disease]
    df = df.rename(columns={"week": "epi_week", "no. of cases": "value", "year": "year"})
    return df[["year", "epi_week", "value"]].reset_index(drop=True)


def append_pdf_weeks(df: pd.DataFrame, pdf_dir: str) -> pd.DataFrame:
    """Append weekly counts scraped from PDFs to an existing DataFrame.

    Parameters
    ----------
    df: DataFrame
        Base DataFrame containing existing week numbers and values.
    pdf_dir: str
        Directory containing PDF bulletins for 2023 onward.

    Returns
    -------
    DataFrame
        The concatenated DataFrame.  Currently just returns the input
        unchanged.
    """
    # TODO: Implement scraping of weekly PDFs using pdfplumber or similar.
    return df