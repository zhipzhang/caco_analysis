"""Analysis of the shutter ``GeneralStatus`` time series.

Concepts
--------
The ECC controller records ``ECC_Monitoring_Shutter_GeneralStatus``
**only when the value changes** — never periodically:

======= ========
value   meaning
======= ========
0       CLOSED
1       OPEN
2       TRANSIT
======= ========

Three rules follow from this recording scheme:

1. **No repeats** — the same value never appears twice in a row.
2. **Only legal steps** — a stable state (OPEN/CLOSED) can only be
   followed by TRANSIT, and TRANSIT only by a stable state.
3. **Canonical movements** — a full movement is always one of::

       OPENING = CLOSED -> TRANSIT -> OPEN
       CLOSING = OPEN   -> TRANSIT -> CLOSED

Segmentation model
------------------
The stable states (OPEN/CLOSED) act as *anchors* in the series.
A **movement** is the TRANSIT run between two consecutive anchors::

    OPEN -------- TRANSIT -------- CLOSED -------- TRANSIT -------- OPEN
    ^anchor       ^movement "closing"              ^movement "opening"

:class:`ShutterStatus` implements exactly this model, and its
:meth:`~ShutterStatus.anomalies` method checks the three rules above
(plus edge cases at the series boundaries).

Quick start
-----------
::

    from caco_analysis import CacoDB, ShutterStatus

    with CacoDB() as db:
        sh = ShutterStatus.from_db(db, tstart="2026-10-11", tstop="2026-10-12")

    sh.movements_df()   # one row per opening/closing, with durations
    sh.anomalies()      # everything that breaks the rules
    sh.summary()        # counts at a glance
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import IntEnum

import pandas as pd

from .connection import CacoDB

GENERAL_STATUS_COLLECTION = "ECC_min"
GENERAL_STATUS_PROPERTY = "ECC_Monitoring_Shutter_GeneralStatus"


class ShutterState(IntEnum):
    """The three recorded values of ``GeneralStatus``."""

    CLOSED = 0  #: shutter fully closed
    OPEN = 1  #: shutter fully open
    TRANSIT = 2  #: shutter moving between open and closed


#: The canonical movement patterns (rule 3).
OPENING = (ShutterState.CLOSED, ShutterState.TRANSIT, ShutterState.OPEN)
CLOSING = (ShutterState.OPEN, ShutterState.TRANSIT, ShutterState.CLOSED)

#: The legal single steps (rule 2): state -> set of states allowed next.
EXPECTED_TRANSITIONS: dict[ShutterState, set[ShutterState]] = {
    ShutterState.CLOSED: {ShutterState.TRANSIT},
    ShutterState.TRANSIT: {ShutterState.OPEN, ShutterState.CLOSED},
    ShutterState.OPEN: {ShutterState.TRANSIT},
}

#: States in which the shutter can rest; they anchor the movements.
STABLE_STATES = frozenset({ShutterState.OPEN, ShutterState.CLOSED})


@dataclass
class Movement:
    """The TRANSIT run between two stable states (anchors).

    ``start_state`` or ``end_state`` is ``None`` when the data series
    begins or ends in the middle of a movement.

    Examples
    --------
    >>> t0 = datetime.datetime(2026, 10, 11, 18, 0, 0)
    >>> m = Movement(start_time=t0,
    ...              end_time=t0 + datetime.timedelta(seconds=30),
    ...              start_state=ShutterState.CLOSED,
    ...              end_state=ShutterState.OPEN,
    ...              transit_times=[t0 + datetime.timedelta(seconds=5)])
    >>> m.kind
    'opening'
    >>> m.transit_duration
    datetime.timedelta(seconds=25)
    """

    start_time: datetime.datetime
    end_time: datetime.datetime | None
    start_state: ShutterState | None
    end_state: ShutterState | None
    transit_times: list[datetime.datetime]

    @property
    def duration(self) -> datetime.timedelta | None:
        """Anchor-to-anchor time (includes time spent at the start state)."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    @property
    def transit_duration(self) -> datetime.timedelta | None:
        """Time physically moving: first TRANSIT sample until the end anchor."""
        if not self.transit_times:
            return None
        stop = self.end_time if self.end_time is not None else self.transit_times[-1]
        return stop - self.transit_times[0]

    @property
    def states(self) -> tuple[ShutterState | None, ...]:
        """Full state sequence, e.g. ``(CLOSED, TRANSIT, OPEN)``."""
        middle = (ShutterState.TRANSIT,) * len(self.transit_times)
        return (self.start_state, *middle, self.end_state)

    @property
    def kind(self) -> str:
        """Classify the movement against the canonical patterns.

        ======== ================ ============================================
        start    end              kind
        ======== ================ ============================================
        CLOSED   OPEN             ``opening``
        OPEN     CLOSED           ``closing``
        X        X (same state)   ``bounce_back`` — returned where it started
        None     any              ``incomplete_start`` — series cut mid-movement
        any      None             ``stuck_in_transit`` — never reached a state
        ======== ================ ============================================
        """
        if self.start_state is None:
            return "incomplete_start"
        if self.end_state is None:
            return "stuck_in_transit"
        if (self.start_state, self.end_state) == (ShutterState.CLOSED, ShutterState.OPEN):
            return "opening"
        if (self.start_state, self.end_state) == (ShutterState.OPEN, ShutterState.CLOSED):
            return "closing"
        return "bounce_back"


class ShutterStatus:
    """Wrapper around a ``GeneralStatus`` time series for analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Columns ``timestamp`` and ``value``, as returned by
        :func:`caco_analysis.get_timeseries`. Rows are sorted by time.

    Examples
    --------
    A healthy evening: the shutter opens, then closes again.

    >>> df = pd.DataFrame({
    ...     "timestamp": pd.to_datetime([
    ...         "2026-10-11 18:00:00", "2026-10-11 18:00:10",
    ...         "2026-10-11 18:00:40", "2026-10-11 20:00:00",
    ...         "2026-10-11 20:00:30"]),
    ...     "value": [0, 2, 1, 2, 0],
    ... })
    >>> sh = ShutterStatus(df)
    >>> [m.kind for m in sh.movements()]
    ['opening', 'closing']
    >>> sh.anomalies().empty
    True
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.sort_values("timestamp").reset_index(drop=True)

    @classmethod
    def from_db(
        cls,
        db: CacoDB,
        tstart=None,
        tstop=None,
        collection: str = GENERAL_STATUS_COLLECTION,
        property_name: str = GENERAL_STATUS_PROPERTY,
    ) -> "ShutterStatus":
        """Fetch ``GeneralStatus`` from the database and wrap it.

        Examples
        --------
        ::

            with CacoDB() as db:
                sh = ShutterStatus.from_db(
                    db, tstart="2026-10-11T18:00:00", tstop="2026-10-11T23:00:00")
        """
        return cls(db.timeseries(collection, property_name, tstart=tstart, tstop=tstop))

    # ------------------------------------------------------------------
    # Basic views

    @property
    def states(self) -> list[ShutterState | None]:
        """One :class:`ShutterState` per row; ``None`` for values that are not 0/1/2.

        Examples
        --------
        >>> ShutterStatus(pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2026-10-11", "2026-10-12"]),
        ...     "value": [0, 2]})).states
        [<ShutterState.CLOSED: 0>, <ShutterState.TRANSIT: 2>]
        """
        states = []
        for value in self.df["value"]:
            try:
                states.append(ShutterState(int(value)))
            except (TypeError, ValueError):
                states.append(None)
        return states

    def transitions(self) -> pd.DataFrame:
        """Every recorded step ``from_state -> to_state`` with times and duration."""
        states = self.states
        ts = self.df["timestamp"]
        return pd.DataFrame([
            {
                "from_state": states[i - 1],
                "to_state": states[i],
                "start": ts[i - 1],
                "end": ts[i],
                "duration": ts[i] - ts[i - 1],
            }
            for i in range(1, len(states))
        ])

    def state_durations(self) -> pd.DataFrame:
        """How long each recorded state lasted (until the next change).

        The last row has ``end``/``duration`` unset — the state is still
        current (or the series was cut).
        """
        states = self.states
        ts = self.df["timestamp"]
        return pd.DataFrame([
            {
                "state": state,
                "start": ts[i],
                "end": ts[i + 1] if i + 1 < len(ts) else None,
                "duration": (ts[i + 1] - ts[i]) if i + 1 < len(ts) else None,
            }
            for i, state in enumerate(states)
        ])

    # ------------------------------------------------------------------
    # Movement segmentation

    def movements(self) -> list[Movement]:
        """Segment the series into movements (TRANSIT runs between anchors).

        Anchors are the stable states (OPEN/CLOSED). Between two
        consecutive anchors:

        - with TRANSIT samples in between -> a :class:`Movement`
        - without TRANSIT -> not a movement; the rule checks in
          :meth:`anomalies` report it (``repeated_value`` or
          ``invalid_transition``) instead.

        TRANSIT runs touching the series edges become movements with
        ``start_state``/``end_state`` set to ``None``.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     "timestamp": pd.to_datetime([
        ...         "2026-10-11 18:00:00", "2026-10-11 18:00:10",
        ...         "2026-10-11 18:00:40", "2026-10-11 20:00:00",
        ...         "2026-10-11 20:00:30"]),
        ...     "value": [0, 2, 1, 2, 0],
        ... })
        >>> sh = ShutterStatus(df)
        >>> [m.kind for m in sh.movements()]
        ['opening', 'closing']
        >>> sh.movements()[0].states
        (<ShutterState.CLOSED: 0>, <ShutterState.TRANSIT: 2>, <ShutterState.OPEN: 1>)
        """
        states = self.states
        ts = list(self.df["timestamp"])
        anchors = [i for i, s in enumerate(states) if s in STABLE_STATES]
        movements = []

        def transit_times_between(a: int, b: int) -> list[datetime.datetime]:
            return [ts[k] for k in range(a + 1, b) if states[k] is ShutterState.TRANSIT]

        # TRANSIT run before the first anchor: series starts mid-movement
        if anchors and anchors[0] > 0:
            first = anchors[0]
            movements.append(Movement(
                start_time=ts[0], end_time=ts[first],
                start_state=None, end_state=states[first],
                transit_times=[ts[k] for k in range(first)
                               if states[k] is ShutterState.TRANSIT],
            ))

        # TRANSIT run between each pair of consecutive anchors
        for a, b in zip(anchors, anchors[1:]):
            transit = transit_times_between(a, b)
            if not transit:
                continue  # no TRANSIT in between: not a movement
            movements.append(Movement(
                start_time=ts[a], end_time=ts[b],
                start_state=states[a], end_state=states[b],
                transit_times=transit,
            ))

        # TRANSIT run after the last anchor (or a series with no anchor at
        # all): the movement never completes
        rest = range(anchors[-1] + 1, len(ts)) if anchors else range(len(ts))
        transit = [ts[k] for k in rest if states[k] is ShutterState.TRANSIT]
        if transit:
            movements.append(Movement(
                start_time=ts[anchors[-1]] if anchors else ts[0],
                end_time=None,
                start_state=states[anchors[-1]] if anchors else None,
                end_state=None,
                transit_times=transit,
            ))
        return movements

    def movements_df(self) -> pd.DataFrame:
        """The movements as a DataFrame, one row per movement."""
        return pd.DataFrame([
            {
                "kind": m.kind,
                "start_time": m.start_time,
                "end_time": m.end_time,
                "duration": m.duration,
                "transit_duration": m.transit_duration,
                "start_state": m.start_state,
                "end_state": m.end_state,
            }
            for m in self.movements()
        ])

    # ------------------------------------------------------------------
    # Anomaly detection: one method per rule

    def anomalies(
        self,
        max_transit_duration: datetime.timedelta | None = None,
    ) -> pd.DataFrame:
        """Check all the rules and report every violation, sorted by time.

        Parameters
        ----------
        max_transit_duration : datetime.timedelta, optional
            If given, complete movements whose physical moving time
            exceeds this are reported as ``long_transit``.

        Returns
        -------
        pandas.DataFrame
            Columns ``timestamp``, ``kind``, ``detail``. Possible kinds:
            ``unknown_value``, ``repeated_value``, ``invalid_transition``,
            ``bounce_back``, ``incomplete_start``, ``stuck_in_transit``,
            ``long_transit``.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2026-10-11 18:00",
        ...                                  "2026-10-11 18:01",
        ...                                  "2026-10-11 18:02"]),
        ...     "value": [0, 0, 1],  # repeat, then CLOSED -> OPEN directly
        ... })
        >>> list(ShutterStatus(df).anomalies()["kind"])
        ['repeated_value', 'invalid_transition']
        """
        checks = [
            self._check_unknown_values(),
            self._check_repeated_values(),
            self._check_invalid_transitions(),
            self._check_movements(max_transit_duration),
        ]
        rows = [row for check in checks for row in check]
        out = pd.DataFrame(rows, columns=["timestamp", "kind", "detail"])
        if not out.empty:
            out.sort_values("timestamp", inplace=True)
            out.reset_index(drop=True, inplace=True)
        return out

    def _check_unknown_values(self) -> list[dict]:
        """Rule: every value must be a valid ShutterState (0/1/2)."""
        return [
            {"timestamp": ts, "kind": "unknown_value", "detail": f"value={value!r}"}
            for ts, value, state in zip(self.df["timestamp"], self.df["value"], self.states)
            if state is None
        ]

    def _check_repeated_values(self) -> list[dict]:
        """Rule 1: change-only recording never repeats a value.

        A repeat means the logger missed the change in between, or wrote
        the same value twice.
        """
        states = self.states
        ts = self.df["timestamp"]
        return [
            {"timestamp": ts[i], "kind": "repeated_value",
             "detail": f"{states[i].name} recorded again"}
            for i in range(1, len(states))
            if states[i] is not None and states[i] == states[i - 1]
        ]

    def _check_invalid_transitions(self) -> list[dict]:
        """Rule 2: every step must be in EXPECTED_TRANSITIONS.

        E.g. OPEN -> CLOSED without an intermediate TRANSIT means a
        sample was lost or the mechanics skipped a state.
        """
        states = self.states
        ts = self.df["timestamp"]
        rows = []
        for i in range(1, len(states)):
            a, b = states[i - 1], states[i]
            if a is None or b is None or a == b:
                continue  # unknown values / repeats have their own checks
            if b not in EXPECTED_TRANSITIONS[a]:
                rows.append({"timestamp": ts[i], "kind": "invalid_transition",
                             "detail": f"{a.name} -> {b.name}"})
        return rows

    def _check_movements(self, max_transit_duration) -> list[dict]:
        """Rule 3: every movement must be a canonical OPENING or CLOSING.

        Also reports the edge cases (series cut mid-movement) and,
        optionally, transits that take suspiciously long.
        """
        rows = []
        for m in self.movements():
            detail = " -> ".join(s.name if s is not None else "?" for s in m.states)
            if m.kind in ("bounce_back", "incomplete_start", "stuck_in_transit"):
                rows.append({"timestamp": m.start_time, "kind": m.kind, "detail": detail})
            elif (
                max_transit_duration is not None
                and m.transit_duration is not None
                and m.transit_duration > max_transit_duration
            ):
                rows.append({"timestamp": m.start_time, "kind": "long_transit",
                             "detail": f"{detail} took {m.transit_duration}"})
        return rows

    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Quick overview: time span, movement counts, anomaly count.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...     "timestamp": pd.to_datetime(["2026-10-11 18:00:00",
        ...                                  "2026-10-11 18:00:30",
        ...                                  "2026-10-11 18:01:00"]),
        ...     "value": [0, 2, 1],
        ... })
        >>> ShutterStatus(df).summary()["movements_by_kind"]
        {'opening': 1}
        """
        movements = self.movements_df()
        return {
            "n_changes": len(self.df),
            "first": self.df["timestamp"].iloc[0] if len(self.df) else None,
            "last": self.df["timestamp"].iloc[-1] if len(self.df) else None,
            "n_movements": len(movements),
            "movements_by_kind": movements["kind"].value_counts().to_dict(),
            "n_anomalies": len(self.anomalies()),
        }
