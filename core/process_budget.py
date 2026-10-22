"""
process_budget.py

Per-process abundance ledger for scripts/run_famiano.py.

Each solver step changes the cloud abundances through several separate
operations (jet-cloud cascade, thermonuclear reactions, radioactive decay,
baryon renormalisation, dilution and jet-composition mixing). The run loop
records the change each process made, as Delta Y aligned with
state.cloud.species, into a ProcessBudget. Whenever a history row is
written, the budget writes the change accumulated since the previous row,
one CSV row per process, and resets.

By construction the rows for one interval sum to
    Y(t_s) - Y(t_start_s)
for every species (a "solver residual" process absorbs the difference
between the implicit solve's actual update and dt * the per-process rates
at the post-solve abundances), so the file is a closed budget rather than a
reconstruction. scripts/plot_reaction_rates.py reads it.

CSV columns: step, t_start_s, t_s, category, process, <species...>
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np

# Category names used by run_famiano.py.
THERMONUCLEAR = "thermonuclear"
DECAY = "decay"
JET_CLOUD = "jet-cloud"
MIXING = "mixing"
SOLVER = "solver"


class ProcessBudget:
    """Accumulates Delta Y per (category, process) between history rows."""

    def __init__(self, species: Iterable[str], path: Path, t_start_s: float) -> None:
        self.species: List[str] = list(species)
        self.path = Path(path)
        self._acc: Dict[Tuple[str, str], np.ndarray] = {}
        self._t_start_s = float(t_start_s)
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["step", "t_start_s", "t_s", "category", "process"] + self.species)

    def add(self, category: str, process: str, delta_y: np.ndarray) -> None:
        key = (category, process)
        if key in self._acc:
            self._acc[key] += delta_y
        else:
            self._acc[key] = np.array(delta_y, dtype=float)

    def add_many(self, category: str, deltas: Mapping[str, np.ndarray], scale: float = 1.0) -> None:
        for process, delta_y in deltas.items():
            self.add(category, process, scale * np.asarray(delta_y, dtype=float))

    def write(self, step: int, t_s: float) -> None:
        """Write the change accumulated since the last write, then reset."""
        if float(t_s) == self._t_start_s:
            return
        for (category, process), dy in self._acc.items():
            if not np.any(dy):
                continue
            self._writer.writerow(
                [step, repr(self._t_start_s), repr(float(t_s)), category, process]
                + [repr(float(v)) for v in dy]
            )
        self._fh.flush()
        self._acc.clear()
        self._t_start_s = float(t_s)

    def close(self) -> None:
        self._fh.close()


def read_budget(path: Path) -> Tuple[List[str], List[dict]]:
    """Read a process_budget CSV into (species, rows); each row's 'dY' is an array."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        species = header[5:]
        rows = []
        for r in reader:
            rows.append({
                "step": int(r[0]),
                "t_start_s": float(r[1]),
                "t_s": float(r[2]),
                "category": r[3],
                "process": r[4],
                "dY": np.array(r[5:], dtype=float),
            })
    return species, rows
