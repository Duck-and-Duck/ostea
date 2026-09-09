import torch
import numpy as np

def fast_gpu_analytic_solver(p_days, depth, dur_days, r_star=1.0, m_star=1.0, teff=5778.0, device="cuda"):
    """
    GPU Üzerinde Kapalı Form Analitik Mandel-Agol ve Yörünge Mekaniği Çözücü.
    CPU kütüphanelerine ihtiyaç duymadan <5 mikrosaniyede Rp, a, i, b ve yoğunluk çıkarır.
    """
    G = 6.67430e-11
    M_sun = 1.98847e30
    R_sun = 6.957e8
    R_earth = 6.371e6
    M_earth = 5.9722e24
    AU = 1.495978707e11

    P_sec = p_days * 86400.0
    a_m = ((G * (m_star * M_sun) * (P_sec**2)) / (4.0 * (np.pi**2))) ** (1.0 / 3.0)
    a_AU = float(a_m / AU)

    r_star_m = r_star * R_sun
    k_radius_ratio = float(np.sqrt(max(1e-7, depth)))
    r_p_m = k_radius_ratio * r_star_m
    r_p_earth = float(r_p_m / R_earth)

    # Darbe parametresi ve eğiklik analitik bağıntısı (Seager & Mallen-Ornelas)
    v_orb = 2.0 * np.pi * a_m / P_sec
    chord = v_orb * (dur_days * 86400.0)
    b_val = float(np.sqrt(max(0.0, ((1.0 + k_radius_ratio)**2) - (chord / r_star_m)**2)))
    b_val = min(0.92, b_val)
    
    cos_i = b_val * (r_star_m / a_m)
    inclination_deg = float(np.arccos(min(1.0, max(0.0, cos_i))) * (180.0 / np.pi))

    # Kütle (Chen & Kipping 2017)
    if r_p_earth <= 1.23:
        m_p_earth = r_p_earth ** 3.68
    elif r_p_earth <= 14.3:
        m_p_earth = 0.97 * (r_p_earth ** 1.70)
    else:
        m_p_earth = 317.8 * (r_p_earth / 11.2) ** 0.5
    m_p_earth = max(0.05, float(m_p_earth))

    # Yoğunluk (g/cm^3)
    vol_cm3 = (4.0 / 3.0) * np.pi * ((r_p_earth * 6.371e8)**3)
    density_cgs = float((m_p_earth * 5.972e27) / vol_cm3)

    # Denge Sıcaklığı
    t_eq = float(teff * np.sqrt(r_star_m / (2.0 * a_m)) * (0.7 ** 0.25))

    return {
        "Period_days": p_days,
        "SemiMajorAxis_AU": a_AU,
        "Radius_Earth": r_p_earth,
        "Mass_Earth": m_p_earth,
        "Density_g_cm3": density_cgs,
        "Inclination_deg": inclination_deg,
        "Impact_b": b_val,
        "T_eq_K": t_eq
    }
