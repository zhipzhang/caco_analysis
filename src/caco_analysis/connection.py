"""Connection interface to the LST-1 CaCo MongoDB.

The MongoDB stores monitoring data in a fixed layout:

- databases: ``CACO``, ``CALIBRATIONS``, ``SCANS``, ...
- the ``CACO`` database holds one collection per subsystem and time
  resolution, e.g. ``ECC_min``, ``ECC_hour``, ``CACO_day``
  (plus ``RUN_INFORMATION`` and ``STATE``)
- each document is ``{"name": property_name, "date": base_datetime,
  "values": {"<offset_seconds>": value, ...}}``
"""

from __future__ import annotations

import pymongo

DEFAULT_URI = "lst103:27018"
DEFAULT_DATABASE = "CACO"


class CacoDB:
    """Thin wrapper around :class:`pymongo.MongoClient` for the CaCo database.

    Parameters
    ----------
    uri : str, optional
        MongoDB URI or ``host:port``. Defaults to ``lst103:27018``.
    database : str, optional
        Default database name (``CACO``).

    Examples
    --------
    >>> with CacoDB() as db:
    ...     print(db.collections())
    ...     print(db.properties("ECC_min", pattern="Shutter"))
    """

    def __init__(self, uri: str = DEFAULT_URI, database: str = DEFAULT_DATABASE):
        self.uri = uri
        self.database = database
        self._client = pymongo.MongoClient(uri)

    def __enter__(self) -> "CacoDB":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def ping(self) -> bool:
        """Return True if the server answers a ping."""
        try:
            self._client.admin.command("ping")
            return True
        except pymongo.errors.PyMongoError:
            return False

    def databases(self) -> list[str]:
        """List available database names."""
        return self._client.list_database_names()

    def collections(self, database: str | None = None) -> list[str]:
        """List collection names in a database (default: ``self.database``)."""
        return sorted(self._client[database or self.database].list_collection_names())

    def properties(
        self,
        collection: str,
        database: str | None = None,
        pattern: str | None = None,
    ) -> list[str]:
        """List distinct property names stored in a collection.

        Parameters
        ----------
        collection : str
            e.g. ``"ECC_min"``.
        pattern : str, optional
            Case-insensitive substring filter, e.g. ``"Shutter"``.
        """
        names = self._client[database or self.database][collection].distinct("name")
        names = sorted(names)
        if pattern is not None:
            names = [n for n in names if pattern.lower() in n.lower()]
        return names

    def collection_name(self, subsystem: str, resolution: str = "min") -> str:
        """Build a collection name like ``ECC_min`` from subsystem + resolution."""
        name = f"{subsystem.upper()}_{resolution}"
        if name not in self.collections():
            raise ValueError(
                f"Collection {name!r} not found. Available: {self.collections()}"
            )
        return name

    def timeseries(
        self,
        collection: str,
        property_name: str,
        tstart=None,
        tstop=None,
    ):
        """Time series of one property as a DataFrame.

        Shortcut for :func:`caco_analysis.get_timeseries` using this
        connection and its default database::

            with CacoDB() as db:
                df = db.timeseries("ECC_min", "ECC_Monitoring_Shutter_GeneralStatus")
        """
        from .timeseries import get_timeseries

        return get_timeseries(
            self, collection, property_name,
            tstart=tstart, tstop=tstop, database_name=self.database,
        )
