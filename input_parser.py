"""Input parsing and validation.

Every physical parameter is read from JSON/CSV files in inputs/ and data/.
Non-physical values (negative mass, e outside [0, 1], ...) are rejected here,
before any computation starts.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class Constants:
    mu_earth: float              # m^3/s^2
    earth_radius: float          # m
    emr_threshold: float         # J/kg, catastrophic breakup threshold
    vaporization_energy: float   # J/kg
    hypervelocity_threshold: float  # m/s


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
    # contact model
    restitution: float           # e, normal coefficient of restitution
    contact_duration: float      # t_c, s
    # orbit
    altitude: float              # m
    n_orbits: float
    # numerics
    method: str
    rtol: float
    t_max_factor: float
    n_output_impact: int
    n_output_orbit: int

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
    raw = _read_json(path)
    return Constants(
        mu_earth=_positive(raw["earth_gravitational_parameter_m3_s2"], "mu_earth"),
        earth_radius=_positive(raw["earth_radius_m"], "earth_radius"),
        emr_threshold=_positive(raw["catastrophic_emr_threshold_J_kg"], "EMR threshold"),
        vaporization_energy=_positive(raw["aluminium_vaporization_specific_energy_J_kg"],
                                      "vaporization energy"),
        hypervelocity_threshold=_positive(raw["hypervelocity_threshold_m_s"],
                                          "hypervelocity threshold"),
    )


def load_materials(path=DATA_DIR / "materials.csv"):
    df = pd.read_csv(path)
    if (df["density_kg_m3"] <= 0).any():
        raise ValueError(f"{path}: all material densities must be > 0")
    return dict(zip(df["material"], df["density_kg_m3"].astype(float)))


def validate(s):
    """Physical sanity checks on a fully built Scenario. Returns the scenario."""
    for label, value in [("satellite mass", s.sat_mass), ("moment of inertia", s.sat_inertia),
                         ("debris mass", s.debris_mass), ("impact speed", s.speed),
                         ("contact duration", s.contact_duration), ("altitude", s.altitude),
                         ("n_orbits", s.n_orbits), ("rtol", s.rtol),
                         ("impact_t_max_factor", s.t_max_factor)]:
        _positive(value, label)
    if not 0.0 <= s.restitution <= 1.0:
        raise ValueError(f"Non-physical input: coefficient of restitution must be in [0, 1] "
                         f"(got {s.restitution})")
    if not 0.0 <= s.impact_angle_deg < 90.0:
        raise ValueError(f"Impact angle must be in [0, 90) deg so the fragment actually "
                         f"reaches the panel (got {s.impact_angle_deg})")
    if s.panel_point.shape != (2,) or not np.all(np.isfinite(s.panel_point)):
        raise ValueError("panel_impact_point_m must be a list of two finite numbers [x, y]")
    if s.t_max_factor < 1.0:
        raise ValueError("impact_t_max_factor must be >= 1 to resolve the whole contact")
    if s.n_output_impact < 10 or s.n_output_orbit < 10:
        raise ValueError("n_output_impact and n_output_orbit must be >= 10")
    return s


def load_scenario(path, materials=None):
    raw = _read_json(path)
    materials = load_materials() if materials is None else materials
    where = str(path)
    sat = _require(raw, "satellite", where)
    deb = _require(raw, "debris", where)
    con = _require(raw, "contact", where)
    orb = _require(raw, "orbit", where)
    sim = raw.get("simulation", {})

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
        restitution=float(_require(con, "coefficient_of_restitution", where)),
        contact_duration=float(_require(con, "contact_duration_s", where)),
        altitude=float(_require(orb, "altitude_m", where)),
        n_orbits=float(orb.get("n_orbits", 3.0)),
        method=str(sim.get("method", "DOP853")),
        rtol=float(sim.get("rtol", 1e-10)),
        t_max_factor=float(sim.get("impact_t_max_factor", 6.0)),
        n_output_impact=int(sim.get("n_output_impact", 2000)),
        n_output_orbit=int(sim.get("n_output_orbit", 3000)),
    )
    return validate(s)


# Keys allowed in an "extreme_cases" override, mapped to Scenario fields.
_OVERRIDE_FIELDS = {
    "debris_mass_kg": "debris_mass",
    "relative_speed_m_s": "speed",
    "impact_angle_deg": "impact_angle_deg",
    "coefficient_of_restitution": "restitution",
    "panel_impact_point_m": "panel_point",
    "satellite_mass_kg": "sat_mass",
}


def override_scenario(base, overrides):
    """Return a validated copy of `base` with some fields changed."""
    changes = {}
    for key, value in overrides.items():
        if key == "name":
            changes["name"] = value
        elif key in _OVERRIDE_FIELDS:
            field = _OVERRIDE_FIELDS[key]
            changes[field] = np.asarray(value, dtype=float) if field == "panel_point" else float(value)
        else:
            raise KeyError(f"Unknown override key '{key}'. Allowed: {sorted(_OVERRIDE_FIELDS)}")
    return validate(replace(base, **changes))


def load_parametric_config(path):
    raw = _read_json(path)
    for key in ["reference_scenario", "mass_speed_grid", "impact_angle_sweep",
                "restitution_sweep", "mass_ratio_sweep", "monte_carlo", "extreme_cases"]:
        _require(raw, key, str(path))
    mc = raw["monte_carlo"]
    if mc["mass_min_kg"] <= 0 or mc["mass_max_kg"] <= mc["mass_min_kg"]:
        raise ValueError("monte_carlo: need 0 < mass_min_kg < mass_max_kg")
    if not 0.0 <= mc["restitution_min"] <= mc["restitution_max"] <= 1.0:
        raise ValueError("monte_carlo: need 0 <= restitution_min <= restitution_max <= 1")
    return raw
