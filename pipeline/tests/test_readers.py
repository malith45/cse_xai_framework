"""Tests for src/readers against the real CSE Data Library files."""

import re

import pandas as pd
import pytest

from readers import (
    HeaderNotFound,
    SheetPeriod,
    UnsupportedFormat,
    find_header_row,
    is_empty_sheet,
    load_raw_sheets,
    normalise_column_name,
    normalise_columns,
    parse_sheet_name,
    read_sheet,
    read_workbook,
    sort_sheet_names,
    spilled_rows,
)
from readers.formats import list_sheet_names

from .conftest import PRICES_2011_2020, PRICES_2021_2025, PUBLIC_HOLDING, requires_data

PH_COLS = ["quarter", "company_name", "security_id", "public_float"]
PRICE_COLS = [
    "company_code", "main_type", "sub_type", "short_name", "trading_date",
    "price_high", "price_low", "close_price", "open_price", "trade_volume", "share_volume", "turnover",
]
BONUS_COLS = ["company_name", "allotment_date", "old_proportion", "new_proportion", "shares_listed", "date_listed"]
DOT_ID = re.compile(r"^[A-Z0-9]+\.[A-Za-z]\d{4}$")


# ------------------------------------------------------------------ column normalisation (pure)
class TestColumnNormalisation:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("COMPANY NAME", "company_name"),
            ("COMPANY_NAME", "company_name"),
            ("  Company Name ", "company_name"),
            ("NAME", "company_name"),
            ("SECURITY", "security_id"),
            ("SECURITY ID", "security_id"),
            ("PUBLIC FLOAT", "public_float"),
            ("PUBLIC_FLOAT", "public_float"),
            ("  MAIN TYPE  ", "main_type"),
            ("COMPANY ID", "company_code"),
            ("PRICE HIGH (Rs.)", "price_high"),
            ("TRADE VOLUME (No.) ", "trade_volume"),
            ("TURNOVER (Rs.)", "turnover"),
            ("EX PRICE", "ex_price"),
            ("EX_PRICE", "ex_price"),
            ("Rate & Remarks", "rate_remarks"),
            ("BETA VALUE\n BASED ON ASPI", "beta_value_based_on_aspi"),
            ("Date of P/Allotment", "allotment_date"),
            ("", ""),
            (None, ""),
            (float("nan"), ""),
        ],
    )
    def test_variants_map_to_canonical(self, raw, expected):
        assert normalise_column_name(raw) == expected

    def test_duplicates_and_blanks_made_unique(self):
        assert normalise_columns(["SECURITY ID", "NAME", None, "SECURITY ID", ""]) == [
            "security_id", "company_name", "unnamed_2", "security_id_2", "unnamed_4",
        ]


# ------------------------------------------------------------------ sheet-name parsing (pure)
class TestSheetNames:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("2012 Q4", SheetPeriod(2012, 4)),
            ("2016 Q2", SheetPeriod(2016, 2)),
            ("2018_Q1", SheetPeriod(2018, 1)),
            ("2009_Q1", SheetPeriod(2009, 1)),
            ("2012Q2", SheetPeriod(2012, 2)),
            ("2025Q4", SheetPeriod(2025, 4)),
            ("1-2025", SheetPeriod(2025, 1)),
            ("4-2021", SheetPeriod(2021, 4)),
            ("2025", SheetPeriod(2025, None)),
            ("2018 ", SheetPeriod(2018, None)),
            (" 2020", SheetPeriod(2020, None)),
            ("Daily Share Price Q1", SheetPeriod(None, 1)),
            ("Daily Share Prices Q3", SheetPeriod(None, 3)),
            ("2024_Q4", SheetPeriod(2024, 4)),
            ("Hign_low_01-Jan-2020_to_31-de", SheetPeriod(2020, None)),
            ("HighLow", None),
            ("Index", None),
            ("Sheet1", None),
            ("Compatibility Report", None),
            (None, None),
        ],
    )
    def test_parse(self, name, expected):
        assert parse_sheet_name(name) == expected

    def test_quarter_year_is_quarter_not_month(self):
        assert parse_sheet_name("1-2025").quarter == 1
        assert parse_sheet_name("4-2025").quarter == 4

    def test_sort_reverse_chronological_to_chronological(self):
        assert sort_sheet_names(["2012 Q4", "2012 Q3", "2012 Q2", "2012 Q1"]) == ["2012 Q1", "2012 Q2", "2012 Q3", "2012 Q4"]
        assert sort_sheet_names(["2025", "2024", "2023"]) == ["2023", "2024", "2025"]

    def test_sort_mixed_conventions_and_unparseable_last(self):
        names = ["Sheet1", "1-2025", "2024Q4", "2018_Q1", "2012 Q4", "Index"]
        assert sort_sheet_names(names) == ["2012 Q4", "2018_Q1", "2024Q4", "1-2025", "Sheet1", "Index"]

    def test_str(self):
        assert str(SheetPeriod(2012, 4)) == "2012Q4"
        assert str(SheetPeriod(2025)) == "2025"


# ------------------------------------------------------------------ header detection (synthetic)
class TestHeaderDetection:
    def _grid(self):
        return pd.DataFrame([
            [None, None, None],
            ["Some title text", None, None],
            [None, None, None],
            ["QUARTER", "COMPANY_NAME", "PUBLIC_FLOAT"],
            ["2025-01", "ASIA ASSET", "27%"],
        ])

    def test_finds_header_by_names_not_position(self):
        assert find_header_row(self._grid(), ["quarter", "company_name", "public_float"]) == 3

    def test_required_subset(self):
        assert find_header_row(self._grid(), ["quarter", "security_id"], required=["quarter"]) == 3

    def test_returns_none_when_absent(self):
        assert find_header_row(self._grid(), ["quarter", "security_id"]) is None

    def test_respects_max_depth(self):
        assert find_header_row(self._grid(), ["quarter"], max_depth=2) is None

    def test_empty_sheet_detection(self):
        assert is_empty_sheet(pd.DataFrame([["Title"], ["No records for the period ending Q4, 2025"]]))
        assert is_empty_sheet(pd.DataFrame([[None, None], [None, None]]))
        assert not is_empty_sheet(self._grid())


# ------------------------------------------------------------------ real files: formats
@requires_data
class TestFormats:
    def test_xls_via_xlrd(self):
        frames = load_raw_sheets(PRICES_2021_2025 / "2021 Data.xls", ["Daily Share Price Q1"])
        assert list(frames) == ["Daily Share Price Q1"]
        assert len(frames["Daily Share Price Q1"]) > 15000

    def test_xlsx_via_openpyxl(self):
        frames = load_raw_sheets(PUBLIC_HOLDING / "Public Holding 2025.xlsx")
        assert list(frames) == ["1-2025", "2-2025", "3-2025", "4-2025"]

    def test_csv(self):
        frames = load_raw_sheets(PRICES_2021_2025 / "2023 Data.csv")
        assert list(frames) == ["2023 Data"]
        assert len(frames["2023 Data"]) > 62000

    def test_list_sheet_names(self):
        assert list_sheet_names(PUBLIC_HOLDING / "Public Holding 2012.xlsx") == ["2012 Q4", "2012 Q3", "2012 Q2", "2012 Q1"]
        assert list_sheet_names(PRICES_2021_2025 / "2023 Data.csv") == ["2023 Data"]

    def test_unsupported_extension(self, tmp_path):
        bad = tmp_path / "x.txt"
        bad.write_text("hello")
        with pytest.raises(UnsupportedFormat):
            load_raw_sheets(bad)

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_raw_sheets(PUBLIC_HOLDING / "Public Holding 1999.xlsx")

    @pytest.mark.parametrize(
        "file,sheet",
        [
            ("2020 Data.xls", "Hign_low_01-Jan-2020_to_31-de"),   # .xls, header row 3, in 32… folder
            ("2021 Data.xls", "Daily Share Price Q1"),            # .xls, header row 4
            ("2022 Data.xls", "Daily Share Prices Q3"),           # .xls, header row 5, datetime dates
            ("2024 Data.xls", "2024_Q4"),                         # .xls, header row 3, whole-year sheet
            ("2025 Data.xlsx", "HighLow"),                        # .xlsx, header row 4
        ],
    )
    def test_every_excel_price_year_yields_the_same_canonical_columns(self, file, sheet):
        folder = PRICES_2011_2020 if file.startswith("2020") else PRICES_2021_2025
        df = read_sheet(folder / file, sheet, PRICE_COLS)
        assert list(df.columns) == PRICE_COLS
        assert len(df) > 10000
        assert df["company_code"].iloc[0] == "AAF"
        assert (df["company_code"].astype(str).str.strip() != "").all()  # separator rows dropped
        assert len(spilled_rows(df)) == 0

    def test_2023_csv_header_found_and_spilled_rows_exposed_not_hidden(self):
        # Real defect in the library: prices >= Rs 1,000 are written with an unquoted thousands
        # separator ("1,225.00"), so those rows have 13-16 fields. The reader must keep the spilled
        # cells in unnamed columns and report the rows, never silently truncate them.
        df = read_sheet(PRICES_2021_2025 / "2023 Data.csv", "2023 Data", PRICE_COLS)
        assert list(df.columns)[:12] == PRICE_COLS
        assert [c for c in df.columns[12:]] == ["unnamed_12", "unnamed_13", "unnamed_14", "unnamed_15"]
        assert len(df) > 62000
        assert df["company_code"].iloc[0] == "AAF"
        spilled = spilled_rows(df)
        assert 1100 <= len(spilled) <= 1200
        assert "BREW" in set(df.loc[spilled, "company_code"])
        clean = df.drop(index=spilled)
        assert (clean["company_code"].astype(str).str.strip() != "").all()
        assert len(spilled_rows(clean)) == 0


# ------------------------------------------------------------------ real files: header depths
@requires_data
class TestHeaderDepths:
    @pytest.mark.parametrize(
        "file,sheet,expected_depth",
        [
            ("Public Holding 2012.xlsx", "2012 Q4", 4),
            ("Public Holding 2015.xlsx", "2015 Q4", 2),
            ("Public Holding 2018.xlsx", "2018_Q1", 3),
            ("Public Holding 2025.xlsx", "1-2025", 2),
        ],
    )
    def test_header_found_at_varying_depths(self, file, sheet, expected_depth):
        raw = load_raw_sheets(PUBLIC_HOLDING / file, [sheet])[sheet]
        assert find_header_row(raw, PH_COLS) == expected_depth
        df = read_sheet(PUBLIC_HOLDING / file, sheet, PH_COLS, raw=raw)
        assert list(df.columns) == PH_COLS
        assert len(df) > 250
        assert df["security_id"].map(lambda s: bool(DOT_ID.match(str(s)))).all()
        assert df["quarter"].notna().all()

    def test_2012_and_2025_spell_headers_differently_but_normalise_identically(self):
        a = read_sheet(PUBLIC_HOLDING / "Public Holding 2012.xlsx", "2012 Q4", PH_COLS)
        b = read_sheet(PUBLIC_HOLDING / "Public Holding 2025.xlsx", "4-2025", PH_COLS)
        assert list(a.columns) == list(b.columns) == PH_COLS


# ------------------------------------------------------------------ real files: sheet naming + order
@requires_data
class TestWorkbookAndSheetNaming:
    @pytest.mark.parametrize(
        "file,expected_order",
        [
            ("Public Holding 2012.xlsx", ["2012 Q1", "2012 Q2", "2012 Q3", "2012 Q4"]),  # workbook lists Q4 first
            ("Public Holding 2018.xlsx", ["2018_Q1", "2018_Q2", "2018_Q3", "2018_Q4"]),
            ("Public Holding 2025.xlsx", ["1-2025", "2-2025", "3-2025", "4-2025"]),
        ],
    )
    def test_read_workbook_all_conventions_chronological(self, file, expected_order):
        frames = read_workbook(PUBLIC_HOLDING / file, PH_COLS)
        assert list(frames) == expected_order
        for name, df in frames.items():
            period = parse_sheet_name(name)
            assert period.year == int(file[-9:-5]) and period.quarter in (1, 2, 3, 4)
            assert list(df.columns) == PH_COLS and len(df) > 250

    def test_read_workbook_keeps_workbook_order_when_asked(self):
        frames = read_workbook(PUBLIC_HOLDING / "Public Holding 2012.xlsx", PH_COLS, chronological=False)
        assert list(frames) == ["2012 Q4", "2012 Q3", "2012 Q2", "2012 Q1"]

    def test_beta_file_mixed_conventions_sort_correctly(self, data_root):
        names = list_sheet_names(data_root / "29 Beta Value.xls")
        ordered = sort_sheet_names(names)
        periods = [parse_sheet_name(n) for n in ordered]
        assert all(p is not None and p.quarter for p in periods)
        assert periods == sorted(periods, key=lambda p: p.sort_key)
        assert ordered[0] == "2009_Q1" and ordered[-1] == "2025Q4"


# ------------------------------------------------------------------ real files: empty sheets
@requires_data
class TestEmptySheets:
    @pytest.mark.parametrize("sheet", ["2024", "2025"])
    def test_no_records_sheet_returns_empty_frame(self, data_root, sheet):
        df = read_sheet(data_root / "03Capitalization of Reserves (Bonus Issues).xls", sheet, BONUS_COLS)
        assert df.empty
        assert list(df.columns) == BONUS_COLS

    def test_non_empty_neighbour_sheet_still_parses(self, data_root):
        df = read_sheet(data_root / "03Capitalization of Reserves (Bonus Issues).xls", "2023", BONUS_COLS)
        assert len(df) == 2
        assert df["company_name"].str.contains("Tokyo Cement").all()

    def test_read_workbook_includes_empty_sheets(self, data_root):
        frames = read_workbook(
            data_root / "03Capitalization of Reserves (Bonus Issues).xls", BONUS_COLS, sheets=["2025", "2024", "2023"]
        )
        assert list(frames) == ["2023", "2024", "2025"]
        assert frames["2024"].empty and frames["2025"].empty and not frames["2023"].empty


# ------------------------------------------------------------------ real files: failure policy
@requires_data
class TestHeaderNotFound:
    def test_error_names_file_and_sheet(self, data_root):
        path = data_root / "01Dividends.xls"
        with pytest.raises(HeaderNotFound) as info:
            read_sheet(path, "2025", PH_COLS)  # dividends sheet has no public-holding columns
        msg = str(info.value)
        assert "01Dividends.xls" in msg and "'2025'" in msg
        assert "public_float" in msg
        assert info.value.file == "01Dividends.xls" and info.value.sheet == "2025"

    def test_required_subset_bridges_year_to_year_column_changes(self, data_root):
        path = data_root / "01Dividends.xls"
        cols = ["date_of_announcement", "security_id", "rate_of_dividend", "remarks", "date_of_ex", "date_of_payment", "cum_price", "ex_price"]
        new = read_sheet(path, "2025", cols)
        assert set(cols) <= set(new.columns)
        with pytest.raises(HeaderNotFound):
            read_sheet(path, "2010", cols)  # 2010 has "Rate & Remarks" instead
        old = read_sheet(path, "2010", cols, required=["date_of_announcement", "security_id", "date_of_ex"])
        assert {"date_of_announcement", "security_id", "rate_remarks", "date_of_ex", "ex_price"} <= set(old.columns)
        assert len(old) > 100
