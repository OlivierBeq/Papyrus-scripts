# -*- coding: utf-8 -*-

import pandas as pd


def equalize_cell_size_in_row(row, cols=None, fill_mode='internal', fill_value: object = ''):
    """Equalize the number of values in each list-containing cell of a pandas dataframe.

    Slightly adapted from user nphaibk (https://stackoverflow.com/questions/45846765/efficient-way-to-unnest-explode-multiple-list-columns-in-a-pandas-dataframe)

    :param row: pandas row the function should be applied to
    :param cols: columns for which equalization must be performed
    :param fill_mode: 'internal' to repeat the only/last value of a cell as much as needed
                      'external' to repeat fill_value as much as needed
                      'trim' to remove unaligned values
    :param fill_value: value to repeat as much as needed to equalize cells
    :return: the row with each cell having the same number of values
    """
    if not cols:
        cols = row.index
    jcols = [j for j, v in enumerate(row.index) if v in cols]
    if len(jcols) < 1:
        jcols = range(len(row.index))
    Ls = [len(x) for x in row.values]
    if not Ls[:-1] == Ls[1:]:
        vals = [v if isinstance(v, list) else [v] for v in row.values]
        if fill_mode == 'external':
            vals = [[e] + [fill_value] * (max(Ls) - 1) if (not j in jcols) and (isinstance(row.values[j], list))
                    else e + [fill_value] * (max(Ls) - len(e))
                    for j, e in enumerate(vals)]
        elif fill_mode == 'internal':
            vals = [[e] + [e] * (max(Ls) - 1) if (not j in jcols) and (isinstance(row.values[j], list))
                    else e + [e[-1]] * (max(Ls) - len(e))
                    for j, e in enumerate(vals)]
        elif fill_mode == 'trim':
            vals = [e[0:min(Ls)] for e in vals]
        else:
            raise ValueError("fill_mode must be one of ['internal', 'external', 'trim']")
        row = pd.Series(vals, index=row.index.tolist())
    return row
