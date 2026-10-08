"""Gain parameterisations -- how many free numbers a commissioning engineer sets.

The three TENSION zones are (UW, out-feeder nip, RW). The in-feeder is the velocity
master: it fixes the line speed v_feed and has no Kp/T_I in this problem at all.

Every structure maps a search vector x to the same (kp[3], ti[3]) the plant kernel
takes, so all of them are scored by one identical cost and differ ONLY in how many
degrees of freedom the search may use.

    shared-2D          (Kp, TI) for all three            <- the paper's baseline
    outfeeder-only-2D  UW and RW frozen at shared; nip free
    ends-only-2D       nip frozen at shared; UW and RW free but TIED to each other
    ends-tied-4D       (Kp,TI) for UW=RW, (Kp,TI) for nip
    full-6D            a free pair per zone

`ends-tied-4D` is not arbitrary. On all six retuning plants the unwinder and the
rewinder are the SAME machine -- J, f and R are identical to the last digit, while
the nip differs by up to 95x in inertia (P189: J = [44.90, 0.47, 44.90]). So the
physically motivated split is ends-versus-middle, not three independent loops, and
it costs an engineer four numbers instead of six.

The two 2-D restricted structures are the two halves of the question "if we freeze
some loops, which ones may we freeze?" -- one frees only the middle, one only the
ends. Both need a baseline to freeze the others AT, which is the shared-2D optimum
found under the same cost tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

KP_BOUNDS = (1.0, 500.0)        # log-uniform, as the authors sample it
TI_BOUNDS_S = (0.5, 30.0)       # uniform
UW, NIP, RW = 0, 1, 2


@dataclass(frozen=True)
class Structure:
    name: str
    dim: int
    #  (lo, hi, log_scale) per search coordinate
    bounds: tuple
    #  x[dim], baseline(kp3, ti3) -> (kp3, ti3)
    expand: Callable
    needs_baseline: bool = False
    blurb: str = ""

    def expand_batch(self, x: np.ndarray, baseline=None):
        """x[n, dim] -> kp[n,3], ti[n,3]."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        kp = np.empty((x.shape[0], 3))
        ti = np.empty((x.shape[0], 3))
        for i, row in enumerate(x):
            kp[i], ti[i] = self.expand(row, baseline)
        return kp, ti


def _kp_ti(*pairs):
    kp = np.array([p[0] for p in pairs], dtype=float)
    ti = np.array([p[1] for p in pairs], dtype=float)
    return kp, ti


_KP = (*KP_BOUNDS, True)
_TI = (*TI_BOUNDS_S, False)


def _shared(x, _b):
    return np.full(3, x[0]), np.full(3, x[1])


def _outfeeder_only(x, b):
    kp, ti = np.array(b[0], dtype=float).copy(), np.array(b[1], dtype=float).copy()
    kp[NIP], ti[NIP] = x[0], x[1]
    return kp, ti


def _ends_only(x, b):
    kp, ti = np.array(b[0], dtype=float).copy(), np.array(b[1], dtype=float).copy()
    kp[UW] = kp[RW] = x[0]
    ti[UW] = ti[RW] = x[1]
    return kp, ti


def _ends_tied(x, _b):
    return _kp_ti((x[0], x[2]), (x[1], x[3]), (x[0], x[2]))


def _full(x, _b):
    return _kp_ti((x[0], x[3]), (x[1], x[4]), (x[2], x[5]))


STRUCTURES = (
    Structure("shared-2D", 2, (_KP, _TI), _shared, False,
              "the paper: one (Kp*, TI) broadcast to all three tension zones"),
    Structure("outfeeder-only-2D", 2, (_KP, _TI), _outfeeder_only, True,
              "freeze UW and RW at the shared optimum, tune only the out-feeder"),
    Structure("ends-only-2D", 2, (_KP, _TI), _ends_only, True,
              "freeze the out-feeder at the shared optimum, tune UW and RW together"),
    Structure("ends-tied-4D", 4, (_KP, _KP, _TI, _TI), _ends_tied, False,
              "UW and RW share a pair (identical machines); the out-feeder gets its own"),
    Structure("full-6D", 6, (_KP, _KP, _KP, _TI, _TI, _TI), _full, False,
              "a free pair per zone -- the upper bound on what retuning can buy"),
)

BY_NAME = {s.name: s for s in STRUCTURES}
BASELINE = "shared-2D"


def gain_asymmetry(kp) -> float:
    """max/min of the three proportional gains -- 1.0 means a shared pair suffices."""
    kp = np.asarray(kp, dtype=float)
    return float(np.max(kp) / max(np.min(kp), 1e-12))
