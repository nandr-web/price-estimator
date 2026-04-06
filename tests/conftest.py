"""Shared test fixtures for the price estimator test suite."""

from pathlib import Path

import pandas as pd
import pytest

DATA_PATH = Path(__file__).parent.parent / "resources" / "aora_historical_quotes.csv"


@pytest.fixture(scope="session")
def raw_data():
    """Load the actual CSV once for all tests."""
    from price_estimator.data import load_data

    return load_data(DATA_PATH)


@pytest.fixture
def sample_data():
    """Small synthetic dataset for fast, isolated tests."""
    return pd.DataFrame(
        {
            "QuoteID": ["Q-TEST-1", "Q-TEST-2", "Q-TEST-3", "Q-TEST-4", "Q-TEST-5"],
            "Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
            "PartDescription": [
                "Sensor Housing - threaded",
                "Manifold Block - complex internal channels",
                "Mounting Bracket - standard",
                "Actuator Linkage",
                "Turbine Blade Housing - thin walls",
            ],
            "Material": [
                "Inconel 718",
                "Titanium Grade 5",
                "Aluminum 6061",
                "Aluminum 7075",
                "Stainless Steel 17-4 PH",
            ],
            "Process": [
                "Wire EDM",
                "5-Axis Milling",
                "CNC Turning",
                "Surface Grinding",
                "3-Axis Milling",
            ],
            "Quantity": [1, 10, 50, 5, 20],
            "LeadTimeWeeks": [4, 8, 6, 3, 10],
            "RushJob": ["Yes", "No", "No", "No", "Yes"],
            "Estimator": [
                "Sato-san",
                "Tanaka-san",
                "Suzuki-san",
                "Tanaka-san",
                "Sato-san",
            ],
            "TotalPrice_USD": [703.11, 5000.00, 2500.00, 480.94, 13097.46],
        }
    )


@pytest.fixture
def sample_df(sample_data):
    """Sample data processed through load_data-style type conversions."""
    import numpy as np

    df = sample_data.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["RushJob"] = df["RushJob"].map({"Yes": True, "No": False})
    df = df.replace("", np.nan)
    return df
