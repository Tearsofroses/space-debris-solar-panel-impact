"""Output generation: figures (labelled axes with units, legends, grids) for every
scenario and for the parametric studies.

Each figure is drawn at the size it has on the page of the report, so the font sizes
below are the printed sizes. The caption in the report gives the title of the figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import physics_model as pm  # noqa: E402

# Categorical colours (fixed order, colour-blind checked) and neutral inks
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
MILLI, KILO = 1e3, 1e-3   # SI-prefix scale factors, for plotting only

# Printed width and height (inches), from the width of each figure in the report
CM = 1 / 2.54
SIZE_BUDGET = (7.92 * CM, 3.168 * CM)     # Fig. 5.1, one sub-figure
SIZE_MAP = (6.08 * CM, 3.715 * CM)        # Fig. 5.2
SIZE_SWEEP = (10.24 * CM, 3.925 * CM)     # Figs. 5.3 and 5.4

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 600, "font.size": 7,
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",   # matches the Times text
    "axes.labelcolor": INK, "axes.edgecolor": INK2, "axes.linewidth": 0.5,
    "axes.labelpad": 1.5, "axes.axisbelow": True, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "axes.grid": True, "grid.color": GRID, "grid.linestyle": "--", "grid.linewidth": 0.3,
    "lines.linewidth": 1.0, "legend.frameon": False, "axes.spines.top": False,
    "axes.spines.right": False, "legend.fontsize": 6.5, "legend.handlelength": 1.5,
    "legend.borderpad": 0.2, "legend.labelspacing": 0.2,
    "axes.prop_cycle": matplotlib.cycler(color=[C1, C2, C3, C4]),
})

LABEL_DV = r"$|\Delta\mathbf{V}|$"


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2, w_pad=0.8)
    fig.savefig(path)
    plt.close(fig)
    return path


def _floor(err):
    """Log axes cannot show exact zeros; clip to machine precision."""
    return np.maximum(np.asarray(err, dtype=float), np.finfo(float).eps * 1e-2)


# ----------------------------------------------------------------------------
# Per-scenario figure: momentum and energy budget before / after the impact
# ----------------------------------------------------------------------------
def plot_budget(summary, s, out):
    n = s.panel_normal
    vd0 = pm.debris_velocity(s.speed, s.impact_angle, n)
    vd1 = np.array([summary["debris_vx_after_m_s"], summary["debris_vy_after_m_s"]])
    dV = np.array([summary["sat_dv_x_m_s"], summary["sat_dv_y_m_s"]])
    m, M = s.debris_mass, s.sat_mass

    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE_BUDGET,
                                 gridspec_kw={"width_ratios": [1.5, 1]})
    # (a) linear momentum, projected on the panel normal n and the panel tangent t
    before, after = [], []
    for e in (n, pm.z_cross(n)):
        p_debris0, p_debris1, p_sat1 = m * np.dot(vd0, e), m * np.dot(vd1, e), M * np.dot(dV, e)
        before += [p_debris0, 0.0, p_debris0]
        after += [p_debris1, p_sat1, p_debris1 + p_sat1]
    x = np.arange(len(before))
    a1.bar(x - 0.2, before, 0.4, color=C1, label="Before")
    a1.bar(x + 0.2, after, 0.4, color=C2, label="After")
    a1.axhline(0, color=INK2, lw=0.5)
    a1.axvline(2.5, color=INK2, lw=0.5, ls=":")
    a1.set_xticks(x)
    a1.set_xticklabels(["Deb", "Sat", "Tot"] * 2, fontsize=6)
    a1.grid(False, axis="x")
    for middle, axis in ((1, "n"), (4, "t")):
        a1.annotate(rf"along $\hat{{\mathbf{{{axis}}}}}$", (middle, 0),
                    xycoords=("data", "axes fraction"), xytext=(0, -16),
                    textcoords="offset points", ha="center", va="baseline")
    a1.set_ylabel(r"Momentum (kg m s$^{-1}$)")
    a1.legend(loc="best")
    # (b) energy budget: a part too small to see gets no label
    k0 = summary["impact_energy_J"]
    parts = [("Debris\n" r"$K_{d1}$", summary["ke_debris_after_J"]),
             ("Satellite\n" r"$K_T$", summary["ke_sat_translation_J"]),
             ("Satellite\n" r"$K_R$", summary["ke_sat_rotation_J"]),
             ("Lost", summary["dissipated_energy_J"])]
    bar_text = {"ha": "center", "va": "center", "color": "white", "fontsize": 6}
    a2.bar(0, k0 * KILO, 0.9, color=C1)
    a2.text(0, 0.5 * k0 * KILO, "Debris\n" r"$K_0$", **bar_text)
    bottom = 0.0
    for (name, v), c in zip(parts, [C2, C3, C4, INK2]):
        a2.bar(1, v * KILO, 0.9, bottom=bottom * KILO, color=c)
        if v > 0.05 * k0:
            a2.text(1, (bottom + 0.5 * v) * KILO, f"{name}\n{v / k0:.4%}", **bar_text)
        bottom += v
    a2.set_xticks([0, 1], ["Before", "After"])
    a2.grid(False, axis="x")
    a2.set_yticks(k0 * KILO * np.array([0, 0.5, 1]))
    a2.set_ylim(0, 1.05 * k0 * KILO)
    a2.set_ylabel("Energy (kJ)")
    return _save(fig, out)


# ----------------------------------------------------------------------------
# Parametric-study figures
# ----------------------------------------------------------------------------
def plot_mass_speed_map(df, out):
    dv = df.pivot(index="impact_speed_m_s", columns="debris_mass_kg", values="sat_dv_mag_m_s")
    masses, speeds = dv.columns.to_numpy(), dv.index.to_numpy()
    fig, ax = plt.subplots(figsize=SIZE_MAP)
    mesh = ax.pcolormesh(masses, speeds * KILO, dv.values, cmap="Blues", shading="auto",
                         norm=LogNorm(vmin=dv.values.min(), vmax=dv.values.max()))
    bar = fig.colorbar(mesh, ax=ax, pad=0.03)
    bar.set_label(LABEL_DV + r" (m s$^{-1}$)")
    bar.set_ticks([1e-5, 1e-3, 1e-1, 1e1])
    bar.ax.minorticks_off()
    bar.ax.tick_params(length=2, width=0.5)
    bar.outline.set_linewidth(0.5)
    ax.set_xscale("log")
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
    ax.minorticks_off()
    ax.grid(False)
    ax.set_xlabel("Debris mass (kg)")
    ax.set_ylabel(r"Impact speed (km s$^{-1}$)")
    return _save(fig, out)


def _collision_label(kind):
    return kind.replace("_", " ")


def plot_angle_sweep(df, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE_SWEEP)
    for kind, g in df.groupby("collision_type"):          # colours from axes.prop_cycle
        label = _collision_label(kind)
        a1.plot(g["impact_angle_deg"], g["sat_dv_mag_m_s"] * MILLI, label=label)
        a2.plot(g["impact_angle_deg"], np.degrees(np.abs(g["sat_omega_rad_s"])), label=label)
    a1.set_ylabel(LABEL_DV + r" (mm s$^{-1}$)")
    a2.set_ylabel(r"$|\omega|$ ($^\circ$ s$^{-1}$)")
    for a in (a1, a2):
        a.set_xlabel(r"Impact angle $\theta$ ($^\circ$)")
        a.set_ylim(bottom=0)
        a.legend(loc="lower left")
    return _save(fig, out)


def plot_mass_ratio_sweep(df, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=SIZE_SWEEP)
    for kind, g in df.groupby("collision_type"):          # colours from axes.prop_cycle
        label = _collision_label(kind)
        a1.plot(g["mass_ratio_m_over_M"], g["sat_dv_over_v0"], label=label)
        a2.plot(g["mass_ratio_m_over_M"], g["debris_v_after_over_v0"], label=label)
    a2.axhline(0.0, color=INK2, lw=0.5)
    a1.set_yscale("log")
    a1.set_yticks([1e-6, 1e-4, 1e-2, 1])
    a1.set_ylabel(LABEL_DV + r"$/v_0$")
    a2.set_ylabel(r"$v_{d1}/v_0$ (forward $>0$)")
    for a in (a1, a2):
        a.axvline(1.0, color=INK2, lw=0.5, ls=":")
        a.set_xscale("log")
        a.set_xticks([1e-6, 1e-4, 1e-2, 1, 1e2])
        a.minorticks_off()
        a.set_xlabel(r"Mass ratio $m/M$")
    a1.legend(loc="lower right", frameon=True, edgecolor="none", framealpha=1)
    a2.legend(loc="upper left")
    return _save(fig, out)


def plot_conservation(df, out):
    labels = [f"{r.scenario}\n{r.quantity}" for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(5, max(2, 0.2 * len(df))))
    ax.barh(range(len(df)), _floor(df["rel_error"]), color=C1, height=0.7,
            label="|after - before| / before")
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.grid(False, axis="y")
    ax.set_xlabel("Relative residual (round-off level)")
    ax.legend(loc="lower right")
    return _save(fig, out)
