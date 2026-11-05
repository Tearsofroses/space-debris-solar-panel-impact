"""Physical model: a debris fragment hits a satellite's solar panel.

Only concepts from the lecture slides (Ch2-Ch3) are used:
    - centre of mass, P = M v_CM                                  (Ch3, 1.1, 1.3)
    - impulse-momentum theorem, conservation of linear momentum   (Ch3, 1.3)
    - elastic and completely inelastic collisions, 1-D and 2-D    (Ch3, 1.4)
    - torque, moment of inertia, parallel-axis theorem            (Ch3, 2.3-2.4)
    - L = r x p (particle), L = I w (rigid body), conservation    (Ch3, 2.6.2)
    - kinetic energy K = 1/2 M v_CM^2 + 1/2 I_CM w^2              (Ch3, 2.5-2.7)
    - gravitational force as the centripetal force (circular orbit) (Ch2, 2.1, 2.3)

After the impact (post-impact orbit, see the end of this file)
    The struck body (satellite, plus the fragment if it stuck) moves under Newton's gravity
        M_b dv/dt = -G M_E M_b r / |r|^3          (Ch2: gravitational force)
    and spins freely at the constant rate w (no external torque about its CoM).
    The total energy E = K + U and the angular momentum L_z are tracked at each time step.

Frame (2-D, orbital plane): x = radial, y = along-track, z = orbit normal (spin axis).

Bodies
    debris    : point mass m, velocity v_d (relative to the satellite before impact)
    satellite : rigid body of mass M and moment of inertia I about z through its CoM,
                initially at rest, CoM at the origin. The impact point on the panel is at
                arm r from the CoM, and the panel surface has outward unit normal n.

The collision is instantaneous, so gravity does not act during it: the total linear
momentum and the total angular momentum (about the satellite CoM) are both conserved.

Elastic (kinetic energy conserved; frictionless contact, so only a normal impulse J n)
    Impulse-momentum theorem for each body, with the satellite also rotating:
        debris    : m  dv_d   = +J n
        satellite : M  dV     = -J n
                    I  domega = r x (-J n)            (torque about the CoM)
    One unit of impulse changes the relative normal speed of the debris and the struck
    point by  k = 1/m + 1/M + (r x n)^2 / I.  Kinetic energy is conserved when the relative
    normal speed reverses (as in the 1-D elastic collision), so  J = 2 v_n / k.

Completely inelastic (the fragment sticks to the panel: one rigid body afterwards)
    Same method as the slides' "collision of extended bodies in 2-D":
        centre of mass      r_c  = m r / (m + M)
        linear momentum     m v_d = (m + M) v_CM
        moment of inertia   I_c  = I + M |r_c|^2 + m |r - r_c|^2     (parallel-axis theorem)
        angular momentum    m (r - r_c) x v_d = I_c w                (about the new CoM)
    Energy lost  = 1/2 m v_d^2 - 1/2 (m + M) v_CM^2 - 1/2 I_c w^2.
"""

import numpy as np

COLLISION_TYPES = ("elastic", "completely_inelastic")


# ----------------------------------------------------------------------------
# 2-D vector helpers (vectorised: work on shape (2,) or (2, N) arrays)
# ----------------------------------------------------------------------------
def dot(a, b):
    """a . b for 2-D vectors."""
    return a[0] * b[0] + a[1] * b[1]


def cross_z(a, b):
    """z-component of a x b for 2-D vectors."""
    return a[0] * b[1] - a[1] * b[0]


def z_cross(a):
    """z_hat x a: the direction of w x a for a spin w about z (v = w x r)."""
    return np.array([-a[1], a[0]])


# ----------------------------------------------------------------------------
# Collision mechanics
# ----------------------------------------------------------------------------
def debris_velocity(speed, impact_angle, n):
    """Debris velocity for a hit at `impact_angle` from the panel normal (moving into the panel)."""
    t = z_cross(n)
    cos_a, sin_a = np.cos(impact_angle), np.sin(impact_angle)
    return -speed * (np.multiply.outer(n, cos_a) + np.multiply.outer(t, sin_a))


def _elastic(m, M, I, arm, n, vd0, v_n):
    """Frictionless elastic collision: normal impulse J = 2 v_n / k.

    Returns (dV, w, vd1, energy lost = 0).
    """
    rxn = cross_z(arm, n)                        # lever arm of the impulse, r x n
    k = 1.0 / m + 1.0 / M + rxn ** 2 / I
    J = 2.0 * v_n / k
    Jn = np.multiply.outer(n, J)                 # impulse on the debris, J n
    return -Jn / M, -J * rxn / I, vd0 + Jn / m, np.zeros_like(J)


def _completely_inelastic(m, M, I, arm, vd0):
    """Fragment sticks: P and L conserved for the combined rigid body.

    `arm` has shape (2, 1, ...) so that it broadcasts against the arrays m and vd0.
    Returns (dV, w, vd1, energy lost).
    """
    r_c = arm * m / (m + M)                      # new CoM, measured from the satellite CoM
    r_sat, r_deb = -r_c, arm - r_c               # positions from the new CoM
    v_cm = m * vd0 / (m + M)
    I_c = I + M * dot(r_sat, r_sat) + m * dot(r_deb, r_deb)
    w = m * cross_z(r_deb, vd0) / I_c
    k_after = 0.5 * (m + M) * dot(v_cm, v_cm) + 0.5 * I_c * w ** 2
    e_diss = 0.5 * m * dot(vd0, vd0) - k_after
    # every point of the combined body moves with v = v_CM + w x (position from the new CoM)
    return v_cm + w * z_cross(r_sat), w, v_cm + w * z_cross(r_deb), e_diss


def analytic_collision(m, M, I, arm, n, speed, impact_angle, collision):
    """Closed-form post-impact state. m, speed and impact_angle may be numpy arrays.

    `collision` is "elastic" or "completely_inelastic".
    Satellite is at rest before the impact (velocities are relative to it).
    """
    if collision not in COLLISION_TYPES:
        raise ValueError(f"collision must be one of {list(COLLISION_TYPES)} "
                         f"(got {collision!r})")
    m, speed, impact_angle = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                                   for x in (m, speed, impact_angle)))
    arm = np.asarray(arm, dtype=float)
    vd0 = debris_velocity(speed, impact_angle, n)
    if collision == "elastic":
        v_n = speed * np.cos(impact_angle)       # normal approach speed
        dV, w, vd1, e_diss = _elastic(m, M, I, arm, n, vd0, v_n)
    else:
        arm_b = arm.reshape((2,) + (1,) * m.ndim)
        dV, w, vd1, e_diss = _completely_inelastic(m, M, I, arm_b, vd0)

    ke0 = 0.5 * m * speed ** 2
    dv_mag = np.hypot(*dV)
    return {
        "impulse_Ns": M * dv_mag,                # |impulse on the satellite| = M |dV|
        "sat_dv_x_m_s": dV[0],
        "sat_dv_y_m_s": dV[1],
        "sat_dv_mag_m_s": dv_mag,
        "sat_omega_rad_s": w,
        "debris_vx_after_m_s": vd1[0],
        "debris_vy_after_m_s": vd1[1],
        "impact_energy_J": ke0,
        # kinetic energy turned into heat, crater damage, sound (0 if elastic)
        "dissipated_energy_J": e_diss,
        "dissipated_fraction": np.divide(e_diss, ke0, out=np.zeros_like(ke0), where=ke0 > 0),
        "ke_debris_after_J": 0.5 * m * dot(vd1, vd1),
        "ke_sat_translation_J": 0.5 * M * dv_mag ** 2,
        "ke_sat_rotation_J": 0.5 * I * w ** 2,
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


def orbital_period(mu, radius):
    """T = 2 pi r / v for a circular orbit."""
    return 2.0 * np.pi * radius / orbital_speed(mu, radius)


def gravity_rhs(t, y, mu):
    """Equations of motion for solve_ivp. State y = [x, y, vx, vy, phi, omega].

    dr/dt = v,  dv/dt = -mu r / |r|^3  (Newton's law of gravitation, per unit mass),
    dphi/dt = omega,  domega/dt = 0  (torque-free spin about the CoM).
    Works on a single state (6,) or on many states (6, N).
    """
    x, yy, vx, vy, _, omega = y
    r3 = np.hypot(x, yy) ** 3
    return np.array([vx, vy, -mu * x / r3, -mu * yy / r3, omega, np.zeros_like(omega)])


def mechanical_energy(state, mass, inertia, mu):
    """K (translation + rotation), U = -G M_E M_b / r and E = K + U for states (6, N)."""
    x, yy, vx, vy, _, omega = state
    kinetic = 0.5 * mass * (vx ** 2 + vy ** 2) + 0.5 * inertia * omega ** 2
    potential = -mu * mass / np.hypot(x, yy)
    return kinetic, potential, kinetic + potential


def angular_momentum_z(state, mass, inertia):
    """L_z about Earth's centre: orbital M_b (r x v) plus spin I w."""
    x, yy, vx, vy, _, omega = state
    return mass * cross_z((x, yy), (vx, vy)) + inertia * omega
