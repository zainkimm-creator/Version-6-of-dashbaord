"""The digital twin: a plant rebuilt from an identified theta.

The estimator reports the paper's Eq. (6) ratio parameters, k_t,i = R_i^2/J_i
and k_f,i = f_i/J_i. Rebuilding a plant from them takes the roller geometry,
span lengths, line speed and set-point as known from commissioning -- they are
not identified -- and inverts the ratios:

    J_i = R_i^2 / k_t,i          f_i = k_f,i * J_i

This is the same inversion `retuning.identify_twin` performs internally. It is
repeated here because the twin the Plant tab derives from must be a PlantSpec,
not an R2RParameters.
"""

from __future__ import annotations

from typing import Mapping

from .payloads import PlantSpec
from .plant import params_from_spec


# The seven Eq. (6) parameters a theta-hat has to carry to be invertible.
THETA_KEYS = ("kt_UW", "kt_Nip", "kt_RW", "kf_UW", "kf_Nip", "kf_RW", "EA")


def plant_spec_from_theta(base: PlantSpec, estimates: Mapping[str, float]) -> PlantSpec:
    """Return the identified plant as a custom PlantSpec.

    `estimates` comes off the wire (`/plant/derive`'s `swap_with_estimates`),
    so a missing key is a client mistake, not a server fault. Naming the
    missing keys as a ValueError lets the route map it to a 400 instead of
    letting a bare KeyError escape as a 500.
    """

    missing = [key for key in THETA_KEYS if key not in estimates]
    if missing:
        raise ValueError(
            f"estimates is missing {', '.join(missing)}; theta-hat must carry "
            f"all seven of {', '.join(THETA_KEYS)}"
        )

    params, meta = params_from_spec(base)
    radii = tuple(float(r) for r in params.roller_radius_m)
    kt = [float(estimates[f"kt_{name}"]) for name in ("UW", "Nip", "RW")]
    kf = [float(estimates[f"kf_{name}"]) for name in ("UW", "Nip", "RW")]

    inertia = tuple(
        radius * radius / ratio if ratio > 0 else fallback
        for radius, ratio, fallback in zip(radii, kt, params.inertia_kg_m2)
    )
    friction = tuple(ratio * J for ratio, J in zip(kf, inertia))

    return PlantSpec(
        source="custom",
        preset_id=None,
        R=radii,
        L=tuple(float(value) for value in params.span_length_m),
        J=inertia,
        f=friction,
        EA=float(estimates["EA"]),
        v0=float(meta["v_ref_m_s"]),
        T_ref=float(meta["T_ref_N"]),
    )


def active_plant_spec(
    base: PlantSpec, estimates: Mapping[str, float] | None, swapped: bool
) -> PlantSpec:
    """The plant downstream analysis should treat as truth.

    With swap off, or with no estimate to swap in, that is the base plant.
    With swap on it is the identified twin -- which is the 'values are
    exchangeable' claim, made checkable.
    """

    if not swapped or not estimates:
        return base
    return plant_spec_from_theta(base, estimates)
