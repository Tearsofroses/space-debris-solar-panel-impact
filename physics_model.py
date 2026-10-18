"""Physical model: a debris fragment hits a satellite's solar panel.

Frame (2-D, orbital plane, LVLH frame centred on the satellite's original slot):
    x = radial (away from Earth), y = along-track (flight direction),
    z = orbit normal (rotation axis of the satellite in this model).

Bodies
    debris    : point mass m, velocity v_d (relative to the satellite before impact)
    satellite : rigid body of mass M and moment of inertia I about z. The impact
                point on the panel is at arm r_p from the centre of mass (CoM),
                and the panel surface has outward unit normal n.

Collision (impulse-momentum, frictionless contact, coefficient of restitution e)
    1/m_eff = 1/m + 1/M + (r_p x n)_z^2 / I
    J       = (1 + e) m_eff v_n          (v_n = normal approach speed)
    debris : m  dv_d   = +J n
    sat    : M  dV     = -J n
             I  domega = (r_p x (-J n))_z
    The total linear momentum m v_d + M V is the same before and after.
    The energy lost is  E_diss = 1/2 m_eff v_n^2 (1 - e^2).

Resolved contact (for the time-resolved numerical simulation)
    Linear spring-dashpot (Kelvin-Voigt) along n with penetration delta:
        f = k delta + c d(delta)/dt   (while delta > 0)
    with k = m_eff (pi / t_c)^2 and damping ratio zeta(e) chosen so the bounce
    reproduces the target coefficient of restitution e.

Post-impact motion: Clohessy-Wiltshire (Hill) equations for the satellite's
drift away from its original orbital slot.
"""

import numpy as np


# ----------------------------------------------------------------------------
# 2-D vector helpers (vectorised: work on shape (2,) or (2, N) arrays)
# ----------------------------------------------------------------------------
def cross_z(a, b):
    """z-component of a x b for 2-D vectors."""
    return a[0] * b[1] - a[1] * b[0]


def omega_cross(omega, r):
    """omega z_hat x r for 2-D r."""
    return np.array([-omega * r[1], omega * r[0]])


def rotate(vec, phi):
    """Rotate the body-frame vector `vec` by angle(s) phi."""
    c, s = np.cos(phi), np.sin(phi)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def tangent_of(n):
    """Unit tangent of the panel surface: t = z_hat x n."""
    return np.array([-n[1], n[0]])


# ----------------------------------------------------------------------------
# Collision mechanics (analytic impulse-momentum solution)
# ----------------------------------------------------------------------------
def effective_mass(m, M, I, arm, n):
    """Reduced mass felt by the debris along n, including the satellite's rotation."""
    return 1.0 / (1.0 / m + 1.0 / M + cross_z(arm, n) ** 2 / I)


def debris_velocity(speed, impact_angle, n):
    """Debris velocity for a hit at `impact_angle` from the panel normal (moving into the panel)."""
    t = tangent_of(n)
    cos_a, sin_a = np.cos(impact_angle), np.sin(impact_angle)
    return np.array([-speed * (cos_a * n[0] + sin_a * t[0]),
                     -speed * (cos_a * n[1] + sin_a * t[1])])


def analytic_collision(m, M, I, arm, n, speed, impact_angle, e):
    """Closed-form post-impact state. All scalar inputs may be numpy arrays (broadcast).

    Satellite is at rest before the impact (velocities are relative to it).
    """
    m, speed, impact_angle, e = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                                      for x in (m, speed, impact_angle, e)))
    m_eff = effective_mass(m, M, I, arm, n)
    v_n = speed * np.cos(impact_angle)           # normal approach speed
    J = (1.0 + e) * m_eff * v_n                  # normal impulse magnitude, N s

    dv_sat_x, dv_sat_y = -J * n[0] / M, -J * n[1] / M
    d_omega = -J * cross_z(arm, n) / I
    vd0 = debris_velocity(speed, impact_angle, n)
    vd1 = vd0 + np.array([J * n[0], J * n[1]]) / m

    ke0 = 0.5 * m * speed ** 2
    e_diss = 0.5 * m_eff * v_n ** 2 * (1.0 - e ** 2)
    return {
        "impulse_Ns": J,
        "effective_mass_kg": m_eff,
        "sat_dv_x_m_s": dv_sat_x,
        "sat_dv_y_m_s": dv_sat_y,
        "sat_dv_mag_m_s": np.hypot(dv_sat_x, dv_sat_y),
        "sat_omega_rad_s": d_omega,
        "debris_vx_after_m_s": vd1[0],
        "debris_vy_after_m_s": vd1[1],
        "impact_energy_J": ke0,
        "dissipated_energy_J": e_diss,
        "dissipated_fraction": np.divide(e_diss, ke0, out=np.zeros_like(ke0), where=ke0 > 0),
        # momentum transfer efficiency: impulse given / normal momentum carried in
        "momentum_transfer_beta": J / (m * v_n),
        "emr_J_kg": ke0 / M,
    }


# ----------------------------------------------------------------------------
# Resolved contact model (spring-dashpot)
# ----------------------------------------------------------------------------
def damping_ratio(e):
    """Damping ratio of a linear spring-dashpot that bounces with restitution e.

    e = exp(-zeta pi / sqrt(1 - zeta^2))  ->  zeta = -ln e / sqrt(pi^2 + ln^2 e).
    e = 0 gives critical damping (zeta = 1): the fragment stays embedded.
    """
    if e <= 0.0:
        return 1.0
    ln_e = np.log(e)
    return -ln_e / np.sqrt(np.pi ** 2 + ln_e ** 2)


def contact_parameters(m_eff, contact_duration, e):
    """Spring stiffness k (N/m) and damping c (N s/m) of the contact."""
    omega_n = np.pi / contact_duration   # undamped half-period equals t_c
    k = m_eff * omega_n ** 2
    c = 2.0 * damping_ratio(e) * np.sqrt(k * m_eff)
    return k, c


# State vector of the impact ODE (vectorised over columns when shape is (11, N)):
#   0-1  debris position   x_d, y_d          6-7  satellite velocity V_x, V_y
#   2-3  debris velocity   vx_d, vy_d        8    satellite attitude phi
#   4-5  satellite CoM     X, Y              9    satellite spin omega
#                                            10   energy dissipated in the contact
def contact_geometry(y, p):
    """Penetration depth, its rate, contact arm and normal for state(s) y."""
    r_d, v_d = y[0:2], y[2:4]
    R, V, phi, omega = y[4:6], y[6:8], y[8], y[9]
    n = rotate(p["normal_body"], phi)
    p0 = R + rotate(p["arm_body"], phi)          # material point first struck
    delta = -((r_d[0] - p0[0]) * n[0] + (r_d[1] - p0[1]) * n[1])
    contact_pt = r_d + delta * n                 # foot of debris on the panel plane
    arm = contact_pt - R
    v_c = V + omega_cross(omega, arm)            # panel velocity at contact point
    delta_dot = (v_c[0] - v_d[0]) * n[0] + (v_c[1] - v_d[1]) * n[1]
    return delta, delta_dot, arm, n


def contact_force(delta, delta_dot, p):
    """Normal contact force magnitude f (pushes debris along +n)."""
    return p["k"] * delta + p["c"] * delta_dot


def impact_rhs(t, y, p, in_contact):
    """Equations of motion during (in_contact=True) or after the contact."""
    dydt = np.zeros_like(y)
    dydt[0:2] = y[2:4]
    dydt[4:6] = y[6:8]
    dydt[8] = y[9]
    if in_contact:
        delta, delta_dot, arm, n = contact_geometry(y, p)
        f = contact_force(delta, delta_dot, p)
        dydt[2:4] = f * n / p["m"]                          # Newton II, debris
        dydt[6:8] = -f * n / p["M"]                         # Newton III, satellite
        dydt[9] = cross_z(arm, -f * n) / p["I"]             # torque about CoM
        dydt[10] = p["c"] * delta_dot ** 2                  # dashpot power
    return dydt


def system_momentum(y, p):
    """Total linear momentum P (2,) and angular momentum L_z about the origin."""
    r_d, v_d, R, V, omega = y[0:2], y[2:4], y[4:6], y[6:8], y[9]
    P = p["m"] * v_d + p["M"] * V
    L = p["m"] * cross_z(r_d, v_d) + p["M"] * cross_z(R, V) + p["I"] * omega
    return P, L


def system_energy(y, p, in_contact):
    """Kinetic, contact (spring) potential and dissipated energy."""
    v_d, V, omega = y[2:4], y[6:8], y[9]
    ke = (0.5 * p["m"] * (v_d[0] ** 2 + v_d[1] ** 2)
          + 0.5 * p["M"] * (V[0] ** 2 + V[1] ** 2)
          + 0.5 * p["I"] * omega ** 2)
    delta = contact_geometry(y, p)[0]
    pe = np.where(in_contact, 0.5 * p["k"] * np.maximum(delta, 0.0) ** 2, 0.0)
    return ke, pe, y[10]


# ----------------------------------------------------------------------------
# Orbit (circular reference orbit + Clohessy-Wiltshire relative motion)
# ----------------------------------------------------------------------------
def mean_motion(mu, a):
    return np.sqrt(mu / a ** 3)


def cw_rhs(t, y, n):
    """Clohessy-Wiltshire equations (in-plane): x radial, y along-track."""
    x, _, vx, vy = y
    return np.array([vx, vy, 3.0 * n ** 2 * x + 2.0 * n * vy, -2.0 * n * vx])


def cw_analytic(t, x0, y0, vx0, vy0, n):
    """Closed-form CW solution, used to check the numerical integration."""
    nt = n * t
    s, c = np.sin(nt), np.cos(nt)
    x = (4 - 3 * c) * x0 + s / n * vx0 + 2 / n * (1 - c) * vy0
    y = 6 * (s - nt) * x0 + y0 - 2 / n * (1 - c) * vx0 + (4 * s - 3 * nt) / n * vy0
    vx = 3 * n * s * x0 + c * vx0 + 2 * s * vy0
    vy = -6 * n * (1 - c) * x0 - 2 * s * vx0 + (4 * c - 3) * vy0
    return x, y, vx, vy


def cw_energy(M, n, x, vx, vy):
    """CW Jacobi integral split into 'kinetic' and effective tidal 'potential' energy.

    d/dt [1/2 (vx^2 + vy^2) - 3/2 n^2 x^2] = 0 for the CW equations.
    """
    ke = 0.5 * M * (vx ** 2 + vy ** 2)
    pe = -1.5 * M * n ** 2 * x ** 2
    return ke, pe


def orbit_change(mu, a, dv_radial, dv_along):
    """Exact (vis-viva) and linearised change of the semi-major axis and period."""
    v_circ = np.sqrt(mu / a)
    v_new_sq = dv_radial ** 2 + (v_circ + dv_along) ** 2
    a_new = 1.0 / (2.0 / a - v_new_sq / mu)
    period = 2 * np.pi * np.sqrt(a ** 3 / mu)
    period_new = 2 * np.pi * np.sqrt(a_new ** 3 / mu)
    da_linear = 2.0 * a * dv_along / v_circ
    return {
        "orbital_speed_m_s": v_circ,
        "period_s": period,
        "delta_a_m": a_new - a,
        "delta_a_linear_m": da_linear,
        "delta_period_s": period_new - period,
        "along_track_drift_per_orbit_m": -3.0 * np.pi * da_linear,
    }


# ----------------------------------------------------------------------------
# Limits of applicability
# ----------------------------------------------------------------------------
def validity_checks(m, M, speed, constants):
    """Flags telling whether the rigid-body momentum model is believable."""
    ke = 0.5 * m * speed ** 2
    emr = ke / M
    specific = 0.5 * speed ** 2      # impact energy per kg of debris
    return {
        "emr_J_kg": emr,
        "catastrophic_breakup": bool(emr >= constants.emr_threshold),
        "specific_energy_over_vaporization": specific / constants.vaporization_energy,
        "hypervelocity": bool(speed >= constants.hypervelocity_threshold),
    }
