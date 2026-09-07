import numpy as np

def characterize_discovered_planet(P_days, depth_fraction, dur_days, r_star_solar=1.0, m_star_solar=1.0, t_star_k=5778.0):
    G = 6.67430e-11
    M_sun = 1.98847e30
    R_sun = 6.957e8
    R_earth = 6.371e6
    M_earth = 5.9722e24
    AU = 1.495978707e11

    P_sec = P_days * 86400.0
    a_m = ((G * (m_star_solar * M_sun) * (P_sec**2)) / (4 * (np.pi**2))) ** (1.0 / 3.0)
    a_AU = a_m / AU

    k = np.sqrt(max(depth_fraction, 1e-6))
    r_star_m = r_star_solar * R_sun
    R_p_m = k * r_star_m
    R_p_earth = R_p_m / R_earth

    # Chen & Kipping (2017) Kütle Tahmini
    if R_p_earth <= 1.23:
        M_p_earth = R_p_earth ** 3.68
    elif R_p_earth <= 14.3:
        M_p_earth = 0.97 * (R_p_earth ** 1.70)
    else:
        M_p_earth = 317.8 * (R_p_earth / 11.2) ** 0.5

    T_eq = t_star_k * np.sqrt(r_star_m / (2 * a_m)) * ((1 - 0.3)**0.25)
    
    return {
        "Period_days": P_days,
        "SemiMajorAxis_AU": a_AU,
        "Radius_Earth": R_p_earth,
        "Mass_Earth": M_p_earth,
        "T_eq_K": T_eq
    }
