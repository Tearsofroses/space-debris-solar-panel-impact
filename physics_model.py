"""Physical model: a debris fragment hits a satellite's solar panel.

Only concepts from the lecture slides (Ch2-Ch3) are used:
    - impulse-momentum theorem, conservation of linear momentum   (Ch3, 1.3)
    - elastic and completely inelastic collisions                 (Ch3, 1.4)
    - torque, moment of inertia, L = I w, conservation of L       (Ch3, 2.3-2.6)
    - kinetic energy of translation and rotation                  (Ch3, 2.6)
    - gravitational force and centripetal force (circular orbit)  (Ch2, 2.1, 2.3)

Frame (2-D, orbital plane): x = radial, y = along-track, z = orbit normal (spin axis).

Bodies
    debris    : point mass m, velocity v_d (relative to the satellite before impact)
    satellite : rigid body of mass M and moment of inertia I about z, initially at rest.
                The impact point on the panel is at arm r from the centre of mass (CoM),
                and the panel surface has outward unit normal n.

Collision (frictionless contact: only a normal impulse J along n)
    Impulse-momentum theorem for each body, with the satellite also rotating:
        debris    : m  dv_d   = +J n
        satellite : M  dV     = -J n
                    I  domega = r x (-J n)            (torque about the CoM)
    Along n, the approach speed of the debris relative to the struck point is v_n and,
    after the impulse, it has changed by  J k,  where
        k = 1/m + 1/M + (r x n)^2 / I .
    Two cases, as in the slides:
        elastic              : kinetic energy is conserved -> the relative normal speed
                               reverses, J k = 2 v_n.
        completely inelastic : the fragment sticks to the panel -> the relative normal
                               speed becomes zero, J k = v_n.
    Hence  J = f v_n / k  with f = 2 (elastic) or f = 1 (completely inelastic).
    The energy lost is  0 (elastic)  or  v_n^2 / (2k)  (completely inelastic).
"""

import numpy as np

COLLISION_FACTORS = {"elastic": 2.0, "inelastic": 1.0}


# ----------------------------------------------------------------------------
# 2-D vector helpers (vectorised: work on shape (2,) or (2, N) arrays)
# ----------------------------------------------------------------------------
def dot(a, b):
    """a . b for 2-D vectors."""
    return a[0] * b[0] + a[1] * b[1]


def cross_z(a, b):
    """z-component of a x b for 2-D vectors."""
    return a[0] * b[1] - a[1] * b[0]


def tangent_of(n):
    """Unit tangent of the panel surface: t = z_hat x n."""
    return np.array([-n[1], n[0]])


# ----------------------------------------------------------------------------
# Collision mechanics (impulse-momentum solution)
# ----------------------------------------------------------------------------
def normal_compliance(m, M, I, rxn):
    """k = 1/m + 1/M + (r x n)^2 / I: change of relative normal speed per unit impulse."""
    return 1.0 / m + 1.0 / M + rxn ** 2 / I


def debris_velocity(speed, impact_angle, n):
    """Debris velocity for a hit at `impact_angle` from the panel normal (moving into the panel)."""
    t = tangent_of(n)
    cos_a, sin_a = np.cos(impact_angle), np.sin(impact_angle)
    return -speed * (np.multiply.outer(n, cos_a) + np.multiply.outer(t, sin_a))


def analytic_collision(m, M, I, arm, n, speed, impact_angle, collision):
    """Closed-form post-impact state. m, speed and impact_angle may be numpy arrays.

    `collision` is "elastic" or "inelastic" (completely inelastic).
    Satellite is at rest before the impact (velocities are relative to it).
    """
    if collision not in COLLISION_FACTORS:
        raise ValueError(f"collision must be one of {sorted(COLLISION_FACTORS)} "
                         f"(got {collision!r})")
    m, speed, impact_angle = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                                   for x in (m, speed, impact_angle)))
    f = COLLISION_FACTORS[collision]
    rxn = cross_z(arm, n)                        # lever arm of the impulse, r x n
    k = normal_compliance(m, M, I, rxn)
    v_n = speed * np.cos(impact_angle)           # normal approach speed
    J = f * v_n / k                              # normal impulse magnitude, N s

    Jn = np.multiply.outer(n, J)                 # impulse vector J n
    dv_sat = -Jn / M
    d_omega = -J * rxn / I
    vd0 = debris_velocity(speed, impact_angle, n)
    vd1 = vd0 + Jn / m

    ke0 = 0.5 * m * speed ** 2
    # energy lost = J v_n - J^2 k / 2 = f (2 - f) / 2 * v_n^2 / k: 0 elastic, v_n^2/(2k) inelastic
    e_diss = 0.5 * f * (2.0 - f) * v_n ** 2 / k
    dv_mag = np.hypot(*dv_sat)
    return {
        "impulse_Ns": J,
        "sat_dv_x_m_s": dv_sat[0],
        "sat_dv_y_m_s": dv_sat[1],
        "sat_dv_mag_m_s": dv_mag,
        "sat_omega_rad_s": d_omega,
        "debris_vx_after_m_s": vd1[0],
        "debris_vy_after_m_s": vd1[1],
        "impact_energy_J": ke0,
        "dissipated_energy_J": e_diss,
        "dissipated_fraction": np.divide(e_diss, ke0, out=np.zeros_like(ke0), where=ke0 > 0),
        "ke_debris_after_J": 0.5 * m * dot(vd1, vd1),
        "ke_sat_translation_J": 0.5 * M * dv_mag ** 2,
        "ke_sat_rotation_J": 0.5 * I * d_omega ** 2,
        # momentum transfer ratio: impulse given / normal momentum carried in
        "momentum_transfer_beta": J / (m * v_n),
    }


def conservation_check(r, m, M, I, arm, n, speed, impact_angle):
    """Before/after balance of P, L (about the satellite CoM) and energy for one impact.

    `r` is the analytic_collision result for the same inputs.
    Returns the relative residuals; all should be at round-off level.
    """
    vd0 = debris_velocity(speed, impact_angle, n)
    vd1 = np.array([r["debris_vx_after_m_s"], r["debris_vy_after_m_s"]])
    dV = np.array([r["sat_dv_x_m_s"], r["sat_dv_y_m_s"]])

    p_before = m * vd0
    p_after = m * vd1 + M * dV
    l_before = m * cross_z(arm, vd0)
    l_after = m * cross_z(arm, vd1) + I * r["sat_omega_rad_s"]
    ke_after = (r["ke_debris_after_J"] + r["ke_sat_translation_J"]
                + r["ke_sat_rotation_J"])
    energy_gap = r["impact_energy_J"] - ke_after - r["dissipated_energy_J"]

    p_scale = float(np.hypot(*p_before))
    l_scale = max(abs(float(l_before)), p_scale * float(np.hypot(*arm)))
    # a hit through the CoM (arm = 0) carries no angular momentum: L = 0 before and after
    l_error = abs(float(l_after - l_before)) / l_scale if l_scale > 0 else 0.0
    return {
        "momentum_rel_error": float(np.hypot(*(p_after - p_before))) / p_scale,
        "angular_momentum_rel_error": l_error,
        "energy_rel_error": abs(float(energy_gap)) / float(r["impact_energy_J"]),
        "momentum_before_kg_m_s": p_scale,
        "momentum_after_kg_m_s": float(np.hypot(*p_after)),
        "angular_momentum_before": float(l_before),
        "angular_momentum_after": float(l_after),
        "energy_before_J": float(r["impact_energy_J"]),
        "energy_after_J": float(ke_after + r["dissipated_energy_J"]),
    }


# ----------------------------------------------------------------------------
# Orbit: circular orbit speed from gravity = centripetal force
# ----------------------------------------------------------------------------
def orbital_speed(mu, radius):
    """G M m / r^2 = m v^2 / r  ->  v = sqrt(G M / r)."""
    return np.sqrt(mu / radius)
