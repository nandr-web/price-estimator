"""Tests for data loading, validation, and profiling."""

import numpy as np
import pandas as pd
import pytest

from price_estimator.data import (
    EXPECTED_COLUMNS,
    VALID_ESTIMATORS,
    VALID_MATERIALS,
    VALID_PROCESSES,
    VALID_QUANTITIES,
    compute_unit_price,
    get_missing_report,
    load_data,
    validate,
)


class TestLoadData:
    """Tests for load_data()."""

    def test_load_csv_schema(self, raw_data):
        """CSV loads with all expected columns."""
        assert set(EXPECTED_COLUMNS).issubset(raw_data.columns)

    def test_row_count(self, raw_data):
        """CSV has the expected number of rows."""
        assert len(raw_data) == 510

    def test_date_dtype(self, raw_data):
        """Date column is converted to datetime."""
        assert pd.api.types.is_datetime64_any_dtype(raw_data["Date"])

    def test_rush_job_dtype(self, raw_data):
        """RushJob column is converted to boolean."""
        assert raw_data["RushJob"].dtype == bool

    def test_quantity_dtype(self, raw_data):
        """Quantity column is integer."""
        assert pd.api.types.is_integer_dtype(raw_data["Quantity"])

    def test_price_dtype(self, raw_data):
        """TotalPrice_USD column is float."""
        assert pd.api.types.is_float_dtype(raw_data["TotalPrice_USD"])

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError raised for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "nonexistent.csv")

    def test_missing_columns_raises(self, tmp_path):
        """ValueError raised if required columns are missing."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("col1,col2\n1,2\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            load_data(bad_csv)

    def test_no_empty_strings_in_any_column(self, raw_data):
        """No empty strings remain in any column after loading."""
        for col in raw_data.columns:
            if raw_data[col].dtype == object:
                assert not (raw_data[col] == "").any(), f"Empty string found in {col}"


class TestValidate:
    """Tests for validate()."""

    def test_real_data_no_critical_errors(self, raw_data):
        """Validation on real data should not raise."""
        warnings = validate(raw_data)
        # Should return warnings (missing values) but not raise
        assert isinstance(warnings, list)

    def test_real_data_reports_missing_values(self, raw_data):
        """Validation should report missing Material and Process."""
        warnings = validate(raw_data)
        has_material_warning = any("missing Material" in w for w in warnings)
        has_process_warning = any("missing Process" in w for w in warnings)
        assert has_material_warning
        assert has_process_warning

    def test_negative_price_raises(self, sample_df):
        """ValueError raised for negative prices."""
        sample_df.loc[sample_df.index[0], "TotalPrice_USD"] = -100.0
        with pytest.raises(ValueError, match="non-positive prices"):
            validate(sample_df)

    def test_invalid_quantity_raises(self, sample_df):
        """ValueError raised for quantity not in valid set."""
        sample_df.loc[sample_df.index[0], "Quantity"] = 7
        with pytest.raises(ValueError, match="Invalid Quantity"):
            validate(sample_df)

    def test_zero_price_raises(self, sample_df):
        """ValueError raised for zero price (boundary of le(0) check)."""
        sample_df.loc[sample_df.index[0], "TotalPrice_USD"] = 0.0
        with pytest.raises(ValueError, match="non-positive prices"):
            validate(sample_df)

    def test_nan_rush_job_raises(self, sample_df):
        """ValueError raised when RushJob contains NaN."""
        sample_df.loc[sample_df.index[0], "RushJob"] = np.nan
        with pytest.raises(ValueError, match="NaN values in RushJob"):
            validate(sample_df)

    def test_valid_quantities(self, raw_data):
        """All quantities in real data are from the valid set."""
        assert raw_data["Quantity"].isin(VALID_QUANTITIES).all()

    def test_valid_materials(self, raw_data):
        """All non-null materials in real data are from the valid set."""
        known = raw_data["Material"].dropna()
        assert known.isin(VALID_MATERIALS).all()

    def test_valid_processes(self, raw_data):
        """All non-null processes in real data are from the valid set."""
        known = raw_data["Process"].dropna()
        assert known.isin(VALID_PROCESSES).all()

    def test_valid_estimators(self, raw_data):
        """All estimators in real data are from the valid set."""
        assert raw_data["Estimator"].isin(VALID_ESTIMATORS).all()

    def test_no_duplicate_quote_ids(self, raw_data):
        """QuoteIDs should be unique."""
        assert not raw_data["QuoteID"].duplicated().any()

    def test_all_prices_positive(self, raw_data):
        """All prices in real data are positive."""
        assert (raw_data["TotalPrice_USD"] > 0).all()


class TestMissingReport:
    """Tests for get_missing_report()."""

    def test_report_structure(self, raw_data):
        """Report contains all expected keys."""
        report = get_missing_report(raw_data)
        assert "counts" in report
        assert "missing_material_ids" in report
        assert "missing_process_ids" in report
        assert "overlap_count" in report
        assert "overlap_ids" in report
        assert "by_estimator" in report
        assert "by_part_type" in report

    def test_missing_counts_match(self, raw_data):
        """Reported missing counts match actual NaN counts."""
        report = get_missing_report(raw_data)
        assert report["counts"].get("Material", 0) == raw_data["Material"].isna().sum()
        assert report["counts"].get("Process", 0) == raw_data["Process"].isna().sum()

    def test_missing_ids_are_valid(self, raw_data):
        """All reported missing IDs exist in the data."""
        report = get_missing_report(raw_data)
        all_ids = set(raw_data["QuoteID"])
        assert all(qid in all_ids for qid in report["missing_material_ids"])
        assert all(qid in all_ids for qid in report["missing_process_ids"])

    def test_overlap_is_subset(self, raw_data):
        """Overlap IDs are a subset of both missing sets."""
        report = get_missing_report(raw_data)
        overlap = set(report["overlap_ids"])
        assert overlap <= set(report["missing_material_ids"])
        assert overlap <= set(report["missing_process_ids"])

    def test_clean_data_empty_report(self, sample_df):
        """Report on clean data returns empty counts and lists."""
        report = get_missing_report(sample_df)
        assert report["counts"] == {}
        assert report["missing_material_ids"] == []
        assert report["missing_process_ids"] == []
        assert report["overlap_count"] == 0


class TestComputeUnitPrice:
    """Tests for compute_unit_price()."""

    def test_basic_computation(self, sample_df):
        """Unit price = total price / quantity with known values."""
        unit_prices = compute_unit_price(sample_df)
        # Q-TEST-1: $703.11 / qty 1 = $703.11
        assert unit_prices.iloc[0] == pytest.approx(703.11)
        # Q-TEST-2: $5000.00 / qty 10 = $500.00
        assert unit_prices.iloc[1] == pytest.approx(500.00)
        # Q-TEST-3: $2500.00 / qty 50 = $50.00
        assert unit_prices.iloc[2] == pytest.approx(50.00)

    def test_all_positive(self, raw_data):
        """All unit prices should be positive."""
        unit_prices = compute_unit_price(raw_data)
        assert (unit_prices > 0).all()

    def test_unit_price_less_than_total_for_qty_gt_1(self, raw_data):
        """For qty > 1, unit price should be less than total price."""
        mask = raw_data["Quantity"] > 1
        unit_prices = compute_unit_price(raw_data)
        assert (unit_prices[mask] < raw_data.loc[mask, "TotalPrice_USD"]).all()
