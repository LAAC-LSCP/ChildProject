import pandas as pd
import os
import re
import datetime
import numpy as np

def read_dataframe(filename):
    extension = os.path.splitext(filename)[1]

    pd_flags = {
        'keep_default_na': False,
        'na_values': ['-1.#IND', '1.#QNAN', '1.#IND', '-1.#QNAN',
                    '#N/A N/A', '#N/A', 'N/A', 'n/a', '', '#NA',
                    'NULL', 'null', 'NaN', '-NaN', 'nan',
                    '-nan', ''],
        'parse_dates': False,
        'index_col': False
    }

    if extension == '.csv':
        df = pd.read_csv(filename, **pd_flags)
    elif extension == '.xls' or extension == '.xlsx':
        df = pd.read_excel(filename, **pd_flags)
    else:
        raise Exception('table format not supported ({})'.format(extension))

    df.index = df.index+2
    return df

class IndexColumn:
    def __init__(self, name = "", description = "", required = False, regex = None,
                 filename = False, datetime = None, unique = False, generated = False):
        self.name = name
        self.description = description
        self.required = required
        self.filename = filename
        self.regex = regex
        self.datetime = datetime
        self.unique = unique
        self.generated = generated

class IndexTable:
    def __init__(self, name, path = None, columns = []):
        self.name = name
        self.path = path
        self.columns = columns
        self.df = None
    
    def read(self, lookup_extensions = None):
        if lookup_extensions is None:
            self.df = read_dataframe(self.path)
            return self.df
        else:
            for extension in lookup_extensions:
                if os.path.exists(self.path + extension):
                    self.df = read_dataframe(self.path + extension)
                    return self.df

        raise Exception("could not find table '{}'".format(self.path))

    def validate(self):
        errors, warnings = [], []

        for rc in self.columns:
            if not rc.required:
                continue

            if rc.name not in self.df.columns:
                errors.append("{} table is missing column '{}'".format(self.name, rc.name))

            null = self.df[self.df[rc.name].isnull()].index.values.tolist()
            if len(null) > 0:
                errors.append(
                    """{} table has undefined values
                    for column '{}' in lines: {}""".format(self.name, rc.name, ','.join([str(n) for n in null])))

        unknown_columns = [
            c for c in self.df.columns
            if c not in [c.name for c in self.columns]
        ]

        if len(unknown_columns) > 0:
            warnings.append("unknown column{} '{}' in {}, exepected columns are: {}".format(
                's' if len(unknown_columns) > 1 else '',
                ','.join(unknown_columns),
                self.name,
                ','.join([c.name for c in self.columns])
            ))

        for index, row in self.df.iterrows():
            # make sure that recordings exist
            for column_name in self.df.columns:
                column_attr = next((c for c in self.columns if c.name == column_name), None)

                if column_attr is None:
                    continue

                if column_attr.datetime:
                    try:
                        dt = datetime.datetime.strptime(row[column_name], column_attr.datetime)
                    except:
                        if column_attr.required and str(row[column_name]) != 'NA':
                            errors.append("'{}' is not a proper date/time (expected {}) on line {}".format(row[column_name], column_attr.datetime, index))
                        else:
                            warnings.append("'{}' is not a proper date/time (expected {}) on line {}".format(row[column_name], column_attr.datetime, index))
                elif column_attr.regex:
                    if not re.fullmatch(column_attr.regex, str(row[column_name])):
                        warnings.append("'{} does not match the format required for '{}' on line {}, expected '{}'".format(row[column_name], column_name, index, column_attr.regex))

        for c in self.columns:
            if not c.unique:
                continue

            grouped = self.df[self.df[c.name] != 'NA']
            grouped['lineno'] = grouped.index
            grouped = grouped.groupby(c.name)['lineno']\
                .agg([
                    ('count', len),
                    ('lines', lambda lines: ",".join([str(line) for line in sorted(lines)])),
                    ('first', np.min)
                ])\
                .sort_values('first')

            duplicates = grouped[grouped['count'] > 1]
            for col, row in duplicates.iterrows():
                errors.append("{} '{}' appears {} times in lines [{}], should appear once".format(
                    c.name,
                    col,
                    row['count'],
                    row['lines']
                ))

        return errors, warnings