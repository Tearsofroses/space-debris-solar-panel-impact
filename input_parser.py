"""Input parsing and validation.

Every physical parameter is read from JSON/CSV files in inputs/ and data/.
Non-physical values (negative mass, unknown collision type, ...) are rejected here,
before any computation starts.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from physics_model import COLLISION_TYPES

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Constants:
    mu_earth: float              # G * M_earth, m^3/s^2
    earth_radius: float          # m


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    # satellite (rigid body: bus + solar panels)
    sat_mass: float              # M, kg
    sat_inertia: float           # I_z about the centre of mass, kg m^2
    panel_point: np.ndarray      # impact point relative to CoM (body frame), m
    panel_normal_deg: float      # direction of the outward panel normal, deg from +x
    # debris
    debris_mass: float           # m, kg
    debris_material: str
    debris_density: float        # kg/m^3
    speed: float                 # relative impact speed v0, m/s
    impact_angle_deg: float      # angle between debris path and the panel normal
    # collision
    collision_type: str          # "elastic" or "completely_inelastic"
    # orbit (only used to compare dV with the orbital speed)
    altitude: float              # m

    @property
    def panel_normal(self):
        a = np.radians(self.panel_normal_deg)
        return np.array([np.cos(a), np.sin(a)])

    @property
    def impact_angle(self):
        return np.radians(self.impact_angle_deg)

    @property
    def debris_diameter(self):
        """Equivalent sphere diameter of the fragment."""
        return (6.0 * self.debris_mass / (np.pi * self.debris_density)) ** (1.0 / 3.0)


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON ({exc})") from exc


def _require(section, key, where):
    if key not in section:
        raise KeyError(f"{where}: missing required key '{key}'")
    return section[key]


def _positive(value, label):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Non-physical input: {label} must be > 0 (got {value})")
    return value


def load_constants(path=DATA_DIR / "constants.json"):
    raw, where = _read_json(path), str(path)
    return Constants(
        mu_earth=_positive(_require(raw, "earth_gravitational_parameter_m3_s2", where),
                           "mu_earth"),
        earth_radius=_positive(_require(raw, "earth_radius_m", where), "earth_radius"),
    )


def load_materials(path=DATA_DIR / "materials.csv"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    for col in ("material", "density_kg_m3"):
        if col not in df.columns:
            raise KeyError(f"{path}: missing required column '{col}'")
    density = pd.to_numeric(df["density_kg_m3"], errors="coerce")
    if density.isna().any() or (density <= 0).any():
        raise ValueError(f"{path}: all material densities must be numbers > 0")
    return dict(zip(df["material"], density.astype(float)))


def validate(s):
    """Physical sanity checks on a fully built Scenario. Returns the scenario."""
    for label, value in [("satellite mass", s.sat_mass), ("moment of inertia", s.sat_inertia),
                         ("debris mass", s.debris_mass), ("impact speed", s.speed),
                         ("altitude", s.altitude)]:
        _positive(value, label)
    if s.collision_type not in COLLISION_TYPES:
        raise ValueError(f"collision_type must be one of {list(COLLISION_TYPES)} "
                         f"(got {s.collision_type!r})")
    if not 0.0 <= s.impact_angle_deg < 90.0:
        raise ValueError(f"Impact angle must be in [0, 90) deg so the fragment actually "
                         f"reaches the panel (got {s.impact_angle_deg})")
    if s.panel_point.shape != (2,) or not np.all(np.isfinite(s.panel_point)):
        raise ValueError("panel_impact_point_m must be a list of two finite numbers [x, y]")
    return s


def load_scenario(path, materials):
    raw = _read_json(path)
    where = str(path)
    sat = _require(raw, "satellite", where)
    deb = _require(raw, "debris", where)
    con = _require(raw, "contact", where)
    orb = _require(raw, "orbit", where)

    material = _require(deb, "material", where)
    if material not in materials:
        raise ValueError(f"{where}: unknown debris material '{material}'. "
                         f"Known: {sorted(materials)}")

    s = Scenario(
        name=raw.get("scenario", Path(path).stem),
        description=raw.get("description", ""),
        sat_mass=float(_require(sat, "mass_kg", where)),
        sat_inertia=float(_require(sat, "moment_of_inertia_kg_m2", where)),
        panel_point=np.asarray(_require(sat, "panel_impact_point_m", where), dtype=float),
        panel_normal_deg=float(_require(sat, "panel_normal_angle_deg", where)),
        debris_mass=float(_require(deb, "mass_kg", where)),
        debris_material=material,
        debris_density=float(materials[material]),
        speed=float(_require(deb, "relative_speed_m_s", where)),
        impact_angle_deg=float(_require(deb, "impact_angle_deg", where)),
        collision_type=str(_require(con, "collision_type", where)),
        altitude=float(_require(orb, "altitude_m", where)),
    )
    return validate(s)


def _vector(value):
    return np.asarray(value, dtype=float)


# Keys allowed in an "extreme_cases" override -> (Scenario field, type conversion).
_OVERRIDE_FIELDS = {
    "name": ("name", str),
    "debris_mass_kg": ("debris_mass", float),
    "relative_speed_m_s": ("speed", float),
    "impact_angle_deg": ("impact_angle_deg", float),
    "collision_type": ("collision_type", str),
    "panel_impact_point_m": ("panel_point", _vector),
    "satellite_mass_kg": ("sat_mass", float),
}


def override_scenario(base, overrides):
    """Return a validated copy of `base` with some fields changed."""
    changes = {}
    for key, value in overrides.items():
        if key not in _OVERRIDE_FIELDS:
            raise KeyError(f"Unknown override key '{key}'. Allowed: {sorted(_OVERRIDE_FIELDS)}")
        field, convert = _OVERRIDE_FIELDS[key]
        changes[field] = convert(value)
    return validate(replace(base, **changes))


def _check_range(section, lo_key, hi_key, n_key, where, positive=True):
    """lo < hi (both > 0 if they go through log10) and an integer count n >= 2."""
    lo, hi = float(_require(section, lo_key, where)), float(_require(section, hi_key, where))
    n = _require(section, n_key, where)
    if positive:
        _positive(lo, f"{where}: {lo_key}")
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"{where}: need {lo_key} < {hi_key} (got {lo}, {hi})")
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError(f"{where}: {n_key} must be an integer >= 2 (got {n!r})")
    return lo, hi


def _check_collision_types(section, where):
    kinds = _require(section, "collision_types", where)
    if not kinds or any(k not in COLLISION_TYPES for k in kinds):
        raise ValueError(f"{where}: collision_types must be a non-empty list from "
                         f"{list(COLLISION_TYPES)} (got {kinds!r})")


def load_parametric_config(path):
    """Read the parametric-study file and reject impossible sweep settings."""
    raw = _read_json(path)
    where = str(path)
    for key in ["reference_scenario", "mass_speed_grid", "impact_angle_sweep",
                "mass_ratio_sweep", "extreme_cases"]:
        _require(raw, key, where)

    grid = raw["mass_speed_grid"]
    _check_range(grid, "mass_min_kg", "mass_max_kg", "n_mass", f"{where} mass_speed_grid")
    _check_range(grid, "speed_min_m_s", "speed_max_m_s", "n_speed", f"{where} mass_speed_grid")

    ang = raw["impact_angle_sweep"]
    lo, hi = _check_range(ang, "angle_min_deg", "angle_max_deg", "n_angle",
                          f"{where} impact_angle_sweep", positive=False)
    if lo < 0.0 or hi >= 90.0:
        raise ValueError(f"{where} impact_angle_sweep: angles must be in [0, 90) deg "
                         f"(got {lo} to {hi})")
    _check_collision_types(ang, f"{where} impact_angle_sweep")

    ratio = raw["mass_ratio_sweep"]
    _check_range(ratio, "ratio_min", "ratio_max", "n_points", f"{where} mass_ratio_sweep")
    _check_collision_types(ratio, f"{where} mass_ratio_sweep")

    if not isinstance(raw["extreme_cases"], list):
        raise ValueError(f"{where}: extreme_cases must be a list")
    for case in raw["extreme_cases"]:
        _require(case, "name", f"{where} extreme_cases")
    return raw
