import pandas as pd
import os
import re
import datetime
import numpy as np
from typing import Union, Set, List


class MissingColumnsException(Exception):
    def __init__(self, name: str, missing: Set):
        missing = ",".join(list(missing))

        super().__init__(
            f"dataframe {name} misses the following required columns: {missing}"
        )
        
class IncorrectDtypeException(Exception):
    """Exception when an Unexpected DType is found in a pandas DataFrame
    """

def assert_dataframe(name: str, df: pd.DataFrame, not_empty: bool = False):
    assert isinstance(
        df, pd.DataFrame
    ), f"{name} should be a dataframe, but type is '{type(df)}' instead."

    if not_empty:
        assert len(df) > 0, f"{name} should not be empty."


def assert_columns_presence(name: str, df: pd.DataFrame, columns: Union[Set, List]):
    missing = set(columns) - set(df.columns)

    if len(missing):
        raise MissingColumnsException(name, missing)
        
def read_csv_with_dtype(file: str, dtypes: dict):
    try:
        df = pd.read_csv(file,dtype=dtypes)
    except ValueError:
        raise IncorrectDtypeException('Incorrect type found in {}, expected column types are:\n{}'.format(file,dtypes))
    return df


def is_boolean(x):
    return x == "NA" or int(x) in [0, 1]


class IndexColumn:
    def __init__(
        self,
        name="",
        description="",
        required=False,
        regex=None,
        filename=False,
        datetime=None,
        function=None,
        choices=None,
        dtype=None,
        unique=False,
        generated=False,
    ):
        self.name = name
        self.description = description
        self.required = required
        self.filename = filename
        self.regex = regex
        self.datetime = datetime
        self.function = function
        self.choices = choices
        self.unique = unique
        self.generated = generated
        self.dtype = dtype

    def __str__(self):
        return "IndexColumn(name = {})".format(self.name)

    def __repr__(self):
        return "IndexColumn(name = {})".format(self.name)


class IndexTable:
    def __init__(self, name, path=None, columns=[], enforce_dtypes: bool = False):
        self.name = name
        self.path = path
        self.columns = columns
        self.df = None
        self.enforce_dtypes = enforce_dtypes

    def read(self):
        pd_flags = {
            "keep_default_na": False,
            "na_values": [
                "-1.#IND",
                "1.#QNAN",
                "1.#IND",
                "-1.#QNAN",
                "#N/A N/A",
                "#N/A",
                "N/A",
                "n/a",
                "",
                "#NA",
                "NULL",
                "null",
                "NaN",
                "-NaN",
                "nan",
                "-nan",
                "",
            ],
            "parse_dates": False,
            "index_col": False,
        }

        if self.enforce_dtypes:
            dtype = {
                column.name: column.dtype for column in self.columns if column.dtype
            }
            self.df = pd.read_csv(self.path, dtype=dtype, **pd_flags)
        else:
            self.df = pd.read_csv(self.path, **pd_flags)

        self.df.index = self.df.index + 2
        return self.df

    def msg(self, text):
        return "{}: {}".format(os.path.normcase(self.path), text)

    def validate(self):
        errors, warnings = [], []

        columns = {c.name: c for c in self.columns}

        for rc in self.columns:
            if not rc.required:
                continue

            if rc.name not in self.df.columns:
                errors.append(
                    self.msg(
                        "{} table is missing column '{}'".format(self.name, rc.name)
                    )
                )
                continue

            null = self.df[self.df[rc.name].isnull()].index.values.tolist()
            if len(null) > 0:
                errors.append(
                    self.msg(
                        """{} table has undefined values
                    for column '{}' in lines: {}""".format(
                            self.name, rc.name, ",".join([str(n) for n in null])
                        )
                    )
                )

        unknown_columns = [c for c in self.df.columns if c not in columns.keys()]

        if len(unknown_columns) > 0:
            warnings.append(
                self.msg(
                    "unknown column{} '{}' in {}, expected columns are: {}".format(
                        "s" if len(unknown_columns) > 1 else "",
                        ",".join(unknown_columns),
                        self.name,
                        ",".join(columns.keys()),
                    )
                )
            )

        rows = self.df.to_dict(orient="index")
        for line_number in rows:
            row = rows[line_number]
            for column_name in row.keys():
                column_attr = columns.get(column_name)

                if column_attr is None:
                    continue

                if callable(column_attr.function):
                    try:
                        ok = column_attr.function(str(row[column_name])) == True
                    except:
                        ok = False

                    if not ok:
                        message = "'{}' does not pass callable test for column '{}' on line {}".format(
                            row[column_name], column_name, line_number
                        )
                        if column_attr.required and str(row[column_name]) != "NA":
                            errors.append(self.msg(message))
                        elif column_attr.required or str(row[column_name]) != "NA":
                            warnings.append(self.msg(message))

                elif (
                    column_attr.choices
                    and str(row[column_name]) not in column_attr.choices
                ):
                    message = "'{}' is not a permitted value for column '{}' on line {}, should be any of [{}]".format(
                        row[column_name],
                        column_name,
                        line_number,
                        ",".join(column_attr.choices),
                    )
                    if column_attr.required and str(row[column_name]) != "NA":
                        errors.append(self.msg(message))
                    elif column_attr.required or str(row[column_name]) != "NA":
                        warnings.append(self.msg(message))

                elif column_attr.datetime:
                    passed = False
                    for frmt in column_attr.datetime:
                        try:
                            dt = datetime.datetime.strptime(
                                row[column_name], frmt
                            )
                            passed = True
                            break
                        except:
                            pass
                    if not passed:
                        message = "'{}' is not a proper date/time for column '{}' (expected: {}) on line {}".format(
                            row[column_name],
                            column_name,
                            ' / '.join(column_attr.datetime),
                            line_number,
                        )
                        if column_attr.required and str(row[column_name]) != "NA":
                            errors.append(self.msg(message))
                        elif column_attr.required or str(row[column_name]) != "NA":
                            warnings.append(self.msg(message))
                elif column_attr.regex:
                    if not re.fullmatch(column_attr.regex, str(row[column_name])):
                        message = "'{}' does not match the format required for '{}' on line {}, expected '{}'".format(
                            row[column_name],
                            column_name,
                            line_number,
                            column_attr.regex,
                        )
                        if column_attr.required and str(row[column_name]) != "NA":
                            errors.append(self.msg(message))
                        elif column_attr.required or str(row[column_name]) != "NA":
                            warnings.append(self.msg(message))

        for c in self.columns:
            if not c.unique:
                continue

            grouped = self.df[self.df[c.name] != "NA"]
            grouped = grouped.assign(lineno=grouped.index)
            grouped = (
                grouped.groupby(c.name)["lineno"]
                .agg(
                    [
                        ("count", len),
                        (
                            "lines",
                            lambda lines: ",".join(
                                [str(line) for line in sorted(lines)]
                            ),
                        ),
                        ("first", np.min),
                    ]
                )
                .sort_values("first")
            )

            duplicates = grouped[grouped["count"] > 1]
            for col, row in duplicates.iterrows():
                errors.append(
                    self.msg(
                        "{} '{}' appears {} times in lines [{}], should appear once".format(
                            c.name, col, row["count"], row["lines"]
                        )
                    )
                )

        return errors, warnings
