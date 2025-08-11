"""Interaction feature utilities.

This module defines helper functions to construct interaction features
for statistical or ml models. Interaction features
capture the combined effect of two variables that may not be
properly represented by their individual contributions. The
functions here allow you to create simple multiplicative terms,
pairwise products among many columns or cross-group interactions.

Functions
---------
add_multiplicative_interaction
    Multiply two columns together to form an interaction term.

add_pairwise_interactions
    Create pairwise multiplicative interactions among a list of
    numeric columns.

add_cross_interactions
    Compute interactions between each column in one group and each
    column in another group.

These helpers are intentionally generic; they do not enforce any
particular naming convention beyond the caller-provided prefixes.

Examples
--------
>>> import pandas as pd
>>> from dengueweather.model.interactions import add_pairwise_interactions
>>> df = pd.DataFrame({"rain": [1, 2], "temp": [3, 4], "wind": [5, 6]})
>>> out = add_pairwise_interactions(df, ["rain", "temp", "wind"], prefix="int")
>>> sorted(out.columns)
['int_rain_temp', 'int_rain_wind', 'int_temp_wind']
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Sequence

import pandas as pd


def add_multiplicative_interaction(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    out_col: str | None = None,
) -> pd.DataFrame:
    """Create a single multiplicative interaction term.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the columns to be multiplied.
    col1 : str
        Name of the first column.
    col2 : str
        Name of the second column.
    out_col : str, optional
        Name of the output interaction column.  If None, a name of
        the form ``f"{col1}__x__{col2}"`` is used.

    Returns
    -------
    DataFrame
        A new DataFrame with the additional interaction column.

    Notes
    -----
    The original DataFrame is not modified.  The returned DataFrame
    contains all original columns plus one new interaction column.
    Missing values propagate according to pandas multiplication rules.
    """
    df = df.copy()
    if out_col is None:
        out_col = f"{col1}__x__{col2}"
    df[out_col] = df[col1] * df[col2]
    return df


def add_pairwise_interactions(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    prefix: str = "int",
    include_self: bool = False,
    dedup: bool = True,
) -> pd.DataFrame:
    """Create multiplicative interactions among all pairs of columns.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame containing the specified columns.
    columns : list-like
        Names of columns on which to compute pairwise products.
    prefix : str, optional
        Prefix for the generated interaction column names.  Defaults to
        ``"int"``.  The generated name for columns ``a`` and ``b`` is
        ``f"{prefix}_{a}_{b}"``.
    include_self : bool, optional
        If True, include interactions of a column with itself
        (i.e. squared terms).  Defaults to False.
    dedup : bool, optional
        If True (default), interactions ``a*b`` and ``b*a`` are
        treated as duplicates and only one column is created.  If
        False, both ``a*b`` and ``b*a`` will be generated with
        distinct names.

    Returns
    -------
    DataFrame
        A new DataFrame with additional columns for each interaction.

    Notes
    -----
    Columns that do not exist in the input DataFrame are ignored.
    """
    df = df.copy()
    # Filter columns that exist
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df
    pairs: List[tuple[str, str]] = []
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if not include_self and c1 == c2:
                continue
            if dedup and j <= i:
                # only upper triangle (without diagonal)
                continue
            pairs.append((c1, c2))
    for c1, c2 in pairs:
        out_col = f"{prefix}_{c1}_{c2}"
        df[out_col] = df[c1] * df[c2]
    return df


def add_cross_interactions(
    df: pd.DataFrame,
    groups: Dict[str, Sequence[str]],
    *,
    prefix: str = "cross",
    dedup: bool = True,
) -> pd.DataFrame:
    """Generate cross-group interaction terms.

    Given a mapping from group names to lists of columns, this function
    computes multiplicative interactions between every pair of
    columns belonging to different groups.  Interactions within
    the same group are not created unless you explicitly include the
    same column in more than one group.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    groups : dict[str, sequence[str]]
        Mapping of group identifiers to column names.  Columns that do
        not exist in the DataFrame are silently ignored.  Groups with
        fewer than one valid column contribute no interactions.
    prefix : str, optional
        Prefix for the output column names.  Defaults to ``"cross"``.
        The generated name for ``col_a`` in group ``g1`` and
        ``col_b`` in group ``g2`` is ``f"{prefix}_{g1}_{col_a}__{g2}_{col_b}"``.
    dedup : bool, optional
        If True (default), the order of groups does not matter -
        interactions between ``g1`` and ``g2`` are generated only once
        (i.e. if ``(g1, g2)`` has been processed, ``(g2, g1)`` is
        skipped).  If False, both orders will be computed.

    Returns
    -------
    DataFrame
        DataFrame with the original columns plus new cross-group
        interaction columns.
    """
    df = df.copy()
    # Prepare list of valid columns for each group
    valid_groups: Dict[str, List[str]] = {
        g: [c for c in cols if c in df.columns] for g, cols in groups.items()
    }
    # Remove groups with fewer than one valid column
    valid_groups = {g: cols for g, cols in valid_groups.items() if cols}
    group_names = list(valid_groups.keys())
    for i, g1 in enumerate(group_names):
        cols1 = valid_groups[g1]
        for j, g2 in enumerate(group_names):
            if dedup and j <= i:
                continue  # avoid duplicate or self interactions
            cols2 = valid_groups[g2]
            for c1 in cols1:
                for c2 in cols2:
                    out_col = f"{prefix}_{g1}_{c1}__{g2}_{c2}"
                    df[out_col] = df[c1] * df[c2]
    return df