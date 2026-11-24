import pandas as pd
import pytest
from pydantic import BaseModel, StrictInt

from ChildProject.tables import (
    IncorrectDtypeException,
    IndexColumn,
    IndexTable,
    MissingColumnsException,
    assert_columns_presence,
    assert_dataframe,
    is_boolean,
    read_csv_with_dtype,
)


# ---------------------------------------------------------------- exceptions / asserts
def test_missing_columns_exception_message():
    exc = MissingColumnsException("segments", {"segment_onset"})
    assert str(exc) == "dataframe segments misses the following required columns: segment_onset"


def test_assert_dataframe_accepts_dataframe():
    assert_dataframe("df", pd.DataFrame({"a": [1]}), not_empty=True)


def test_assert_dataframe_rejects_other_types():
    with pytest.raises(AssertionError, match="should be a dataframe"):
        assert_dataframe("df", [1, 2])


def test_assert_dataframe_rejects_empty_when_required():
    with pytest.raises(AssertionError, match="should not be empty"):
        assert_dataframe("df", pd.DataFrame(), not_empty=True)


def test_assert_dataframe_allows_empty_by_default():
    assert_dataframe("df", pd.DataFrame())


def test_assert_columns_presence_passes():
    assert_columns_presence("df", pd.DataFrame(columns=["a", "b"]), ["a"])


def test_assert_columns_presence_raises():
    with pytest.raises(MissingColumnsException, match="misses the following required columns: c"):
        assert_columns_presence("df", pd.DataFrame(columns=["a", "b"]), {"a", "c"})


# ---------------------------------------------------------------- read_csv_with_dtype
def test_read_csv_with_dtype(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("a,b\n1,x\n")
    df = read_csv_with_dtype(path, {"a": "Int64", "b": "string"})
    assert str(df["a"].dtype) == "Int64"
    assert str(df["b"].dtype) == "string"


def test_read_csv_with_dtype_incorrect_type(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("a\nnot_a_number\n")
    with pytest.raises(IncorrectDtypeException, match="Incorrect type found in"):
        read_csv_with_dtype(path, {"a": "Int64"})


# ---------------------------------------------------------------- is_boolean
@pytest.mark.parametrize("value,expected", [("NA", True), ("0", True), ("1", True), (1, True), ("2", False)])
def test_is_boolean(value, expected):
    assert is_boolean(value) is expected


def test_is_boolean_non_numeric_raises():
    with pytest.raises(ValueError):
        is_boolean("yes")


# ---------------------------------------------------------------- IndexColumn
def test_index_column_defaults():
    col = IndexColumn(name="x")
    assert (col.required, col.unique, col.generated, col.filename) == (False, False, False, False)
    assert col.choices is None and col.dtype is None


def test_index_column_str_and_repr():
    col = IndexColumn(name="x")
    assert str(col) == repr(col) == "IndexColumn(name = x)"


# ---------------------------------------------------------------- IndexTable
COLUMNS = [IndexColumn(name="id", unique=True, dtype="string"), IndexColumn(name="value", dtype="Int64")]


@pytest.fixture
def csv(tmp_path):
    def write(content):
        path = tmp_path / "table.csv"
        path.write_text(content)
        return path
    return write


def test_index_table_read_offsets_index_to_csv_lines(csv):
    table = IndexTable("t", csv("id,value\na,1\nb,2\n"), COLUMNS)
    df = table.read()
    assert df.index.tolist() == [2, 3]
    assert table.df is df


def test_index_table_read_keeps_NA_string(csv):
    df = IndexTable("t", csv("id,value\nNA,1\n"), COLUMNS).read()
    assert df.loc[2, "id"] == "NA"


@pytest.mark.parametrize("empty_marker", ["", "NaN", "null", "N/A"])
def test_index_table_read_na_markers(csv, empty_marker):
    df = IndexTable("t", csv(f"id,value\n{empty_marker},1\n"), COLUMNS).read()
    assert pd.isna(df.loc[2, "id"])


def test_index_table_read_enforces_dtypes(csv):
    df = IndexTable("t", csv("id,value\n1,1\n"), COLUMNS, enforce_dtypes=True).read()
    assert str(df["id"].dtype) == "string"
    assert str(df["value"].dtype) == "Int64"


def test_index_table_read_infers_dtypes_by_default(csv):
    df = IndexTable("t", csv("id,value\n1,1\n"), COLUMNS).read()
    assert df["id"].dtype.kind == "i"


def test_index_table_msg_prefixes_path(csv):
    path = csv("id\n")
    assert IndexTable("t", path).msg("hello") == f"{path}: hello"


def test_index_table_validate_clean_table(csv):
    table = IndexTable("t", csv("id,value\na,1\nb,2\n"), COLUMNS)
    table.read()
    assert table.validate() == ([], [])


def test_index_table_validate_duplicates_in_unique_column(csv):
    path = csv("id,value\na,1\na,2\nb,3\n")
    table = IndexTable("t", path, COLUMNS)
    table.read()
    errors, warnings = table.validate()
    assert errors == [f"{path}: Duplicated values when it should be unique for column id, values {{'a'}} on lines {{2, 3}} appear multiple times"]
    assert warnings == []


@pytest.mark.parametrize("header,suffix", [
    ("id,value,extra", "unknown column 'extra' in t"),
    ("id,value,extra,other", "unknown columns 'extra,other' in t"),
])
def test_index_table_validate_unknown_columns_warn(csv, header, suffix):
    n_extra = header.count(",") - 1
    path = csv(header + "\n" + ",".join(["a", "1"] + ["x"] * n_extra) + "\n")
    table = IndexTable("t", path, COLUMNS)
    table.read()
    errors, warnings = table.validate()
    assert errors == []
    assert warnings == [f"{path}: {suffix}, expected columns are: id,value"]


class _Row(BaseModel):
    value: StrictInt


class _Validator:
    """minimal validator with the same contract as the pandantic validators used by ChildProject"""

    def validate(self, df):
        for row in df.to_dict(orient="records"):
            _Row(**row)


def test_index_table_validate_reports_validator_errors(csv):
    path = csv("id,value\na,not_int\n")
    table = IndexTable("t", path, COLUMNS, validator=_Validator())
    table.read()
    errors, _ = table.validate()
    assert len(errors) == 1
    assert errors[0].startswith(f"{path}: \n1 validation error for _Row\nvalue")
