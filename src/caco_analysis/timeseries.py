"""Fetch monitoring time series from the CaCo MongoDB into pandas DataFrames.

Storage layout: one document per property per time bucket::

    {"name": "ECC_..._GeneralStatus",
     "date":  <datetime of bucket start>,
     "values": {"<offset_seconds>": value, ...}}

The pipeline is split into small pieces, each doing one job:

1. :func:`parse_time`        -- normalize user time input to ``datetime``
2. :func:`build_query`       -- turn (property, time range) into a MongoDB query
3. :func:`expand_document`   -- turn one document into ``(timestamp, value)`` rows
4. :func:`get_timeseries`    -- thin orchestrator: query -> expand -> DataFrame

Steps 1-3 are pure functions: no database access, easy to test and reuse.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Iterator
from typing import Any

import pandas as pd
import pymongo

from .connection import DEFAULT_DATABASE, CacoDB

TimeLike = datetime.datetime | str | None
Row = tuple[datetime.datetime, Any]


def parse_time(t: TimeLike) -> datetime.datetime | None:
    """Normalize a time input: strings become ``datetime``, None stays None.

    Accepts ISO 8601 strings; fields do not need to be zero-padded.

    Examples
    --------
    >>> parse_time("2026-10-11T18:00:00")
    datetime.datetime(2026, 10, 11, 18, 0)
    >>> parse_time("2026-8-1T08:05:09")  # unpadded fields work too
    datetime.datetime(2026, 8, 1, 8, 5, 9)
    >>> parse_time("2026-10-11")
    datetime.datetime(2026, 10, 11, 0, 0)
    >>> parse_time(None) is None
    True
    """
    if isinstance(t, str):
        return pd.Timestamp(t).to_pydatetime()
    return t


def build_query(
    property_name: str,
    tstart: TimeLike = None,
    tstop: TimeLike = None,
) -> dict:
    """Build the MongoDB query for one property, optionally bounded in time.

    Note: ``date`` is the *bucket start*, so the query only narrows down
    which buckets to read; exact second-level filtering happens in
    :func:`expand_document`.
    """
    query = {"name": property_name}

    date_filter = {}
    if tstart := parse_time(tstart):
        date_filter["$gte"] = tstart
    if tstop := parse_time(tstop):
        date_filter["$lte"] = tstop
    if date_filter:
        query["date"] = date_filter

    return query


def expand_document(
    doc: dict,
    tstart: datetime.datetime | None = None,
    tstop: datetime.datetime | None = None,
) -> Iterator[Row]:
    """Expand one document into ``(timestamp, value)`` rows, in time order.

    ``timestamp = doc["date"] + offset_seconds`` for each entry in
    ``doc["values"]``. Rows outside ``[tstart, tstop]`` are dropped
    (needed because the MongoDB query can only filter on bucket starts).
    """
    base_date = doc["date"]
    values = doc.get("values", {})

    for key in sorted(values.keys(), key=int):
        timestamp = base_date + datetime.timedelta(seconds=int(key))
        if tstart and timestamp < tstart:
            continue
        if tstop and timestamp > tstop:
            continue
        yield timestamp, values[key]


def rows_to_dataframe(rows: Iterable[Row]) -> pd.DataFrame:
    """Assemble ``(timestamp, value)`` rows into a sorted DataFrame."""
    df = pd.DataFrame(list(rows), columns=["timestamp", "value"])
    if not df.empty:
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def get_timeseries(
    db: CacoDB | pymongo.MongoClient,
    collection_name: str,
    property_name: str,
    tstart: TimeLike = None,
    tstop: TimeLike = None,
    database_name: str = DEFAULT_DATABASE,
) -> pd.DataFrame:
    """Get the time series of one property as a DataFrame.

    Parameters
    ----------
    db : CacoDB or pymongo.MongoClient
    collection_name : str
        e.g. ``"ECC_min"``.
    property_name : str
        e.g. ``"ECC_Monitoring_Shutter_GeneralStatus"``.
    tstart, tstop : datetime or ISO str, optional
        Time range filter (inclusive).
    database_name : str
        Defaults to ``"CACO"``.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp`` and ``value``, sorted by time.
    """
    client = db._client if isinstance(db, CacoDB) else db
    collection = client[database_name][collection_name]

    tstart = parse_time(tstart)
    tstop = parse_time(tstop)

    docs = collection.find(build_query(property_name, tstart, tstop)).sort("date", 1)
    rows = (row for doc in docs for row in expand_document(doc, tstart, tstop))
    return rows_to_dataframe(rows)
