import numpy as np

def characterize_discovered_planet(
    P_days, depth_fraction, dur_days, impact_b=0.2,
    r_star_solar=1.0, m_star_solar=1.0, t_star_k=5778.0,
    sigma_depth=None, sigma_r_star=0.03, sigma_m_star=0.04, sigma_t_star=60.0,
    v_mag=11.0
):
    G = 6.67430e-11
    M_sun = 1.98847e30
    R_sun = 6.957e8
    R_earth = 6.371e6
    M_earth = 5.9722e24
    AU = 1.495978707e11

    if sigma_depth is None:
        sigma_depth = max(1e-6, depth_fraction * 0.04)

    # 1. YÖRÜNGESEL YARI-BÜYÜK EKSEN
    P_sec = P_days * 86400.0
    a_m = ((G * (m_star_solar * M_sun) * (P_sec**2)) / (4 * (np.pi**2))) ** (1.0 / 3.0)
    a_AU = a_m / AU
    sigma_a_AU = a_AU * (1.0 / 3.0) * (sigma_m_star / max(0.1, m_star_solar))

    # Yörünge Eğikliği (i = arccos(b * R* / a))
    r_star_m = r_star_solar * R_sun
    cos_i = min(1.0, max(0.0, (impact_b * r_star_m) / a_m))
    inclination_deg = float(np.arccos(cos_i) * (180.0 / np.pi))

    # 2. GEZEGEN YARIÇAPI (Rp = sqrt(depth) * R*)
    k = np.sqrt(max(depth_fraction, 1e-7))
    R_p_m = k * r_star_m
    R_p_earth = R_p_m / R_earth
    rel_err_rp = np.sqrt((0.5 * sigma_depth / max(1e-7, depth_fraction))**2 + (sigma_r_star / max(0.1, r_star_solar))**2)
    sigma_R_p_earth = R_p_earth * rel_err_rp

    # 3. KÜTLE TAHMİNİ (Chen & Kipping 2017)
    if R_p_earth <= 1.23:
        M_p_earth = R_p_earth ** 3.68
        sigma_M_p_earth = M_p_earth * 3.68 * (sigma_R_p_earth / max(0.1, R_p_earth))
    elif R_p_earth <= 14.3:
        M_p_earth = 0.97 * (R_p_earth ** 1.70)
        sigma_M_p_earth = M_p_earth * 1.70 * (sigma_R_p_earth / max(0.1, R_p_earth))
    else:
        M_p_earth = 317.8 * (R_p_earth / 11.2) ** 0.5
        sigma_M_p_earth = M_p_earth * 0.5 * (sigma_R_p_earth / max(0.1, R_p_earth))

    # 4. ORTALAMA YOĞUNLUK VE YÜZEY YERÇEKİMİ
    vol_cm3 = (4.0 / 3.0) * np.pi * ((R_p_earth * 6.371e8) ** 3)
    mass_g = M_p_earth * 5.972e27
    density_cgs = mass_g / vol_cm3

    g_ms2 = (G * (M_p_earth * M_earth)) / (R_p_m**2)
    log_g = float(np.log10(g_ms2 * 100.0))

    # 5. TERMODİNAMİK DENGE SICAKLIĞI (Teq)
    T_eq = float(t_star_k * np.sqrt(r_star_m / (2.0 * a_m)) * (0.7 ** 0.25))
    sigma_T_eq = T_eq * np.sqrt((sigma_t_star / max(100.0, t_star_k))**2 + 0.25 * (sigma_r_star / max(0.1, r_star_solar))**2 + 0.25 * (sigma_a_AU / max(0.01, a_AU))**2)

    # 6. YAŞANABİLİR BÖLGE (Kopparapu et al. 2013, 2014)
    L_star = (r_star_solar**2) * ((t_star_k / 5778.0)**4)
    insolation_earth = L_star / (a_AU**2)
    
    t_diff = t_star_k - 5780.0
    s_inner = 1.0512 + 1.3242e-4 * t_diff + 1.5418e-8 * (t_diff**2)
    s_outer = 0.3438 + 5.8942e-5 * t_diff + 1.1731e-8 * (t_diff**2)

    if s_outer <= insolation_earth <= s_inner:
        hz_status = "OPTIMAL YAŞANABİLİR BÖLGE (Sıvı Su Mümkün)"
    elif insolation_earth > s_inner:
        hz_status = "SICAK BÖLGE (Sera Etkisi)"
    else:
        hz_status = "SOĞUK BÖLGE (Buzul Rejimi)"

    # 7. JWST TSM SKORU
    tsm_scale = 1.15 if R_p_earth < 1.5 else (1.26 if R_p_earth < 2.75 else 1.0)
    tsm = float(tsm_scale * ((R_p_earth**3) * T_eq) / (max(0.1, M_p_earth) * (r_star_solar**2)) * (10 ** (-v_mag / 5.0)))

    # 8. RV YARI-GENLİĞİ
    k_rv_ms = float(28.435 * ((M_p_earth / 317.8) / ((m_star_solar)**(2.0/3.0))) * ((P_days / 365.25)**(-1.0/3.0)))

    return {
        "Period_days": P_days,
        "SemiMajorAxis_AU": a_AU,
        "SemiMajorAxis_err": sigma_a_AU,
        "Inclination_deg": inclination_deg,
        "Impact_b": impact_b,
        "Radius_Earth": R_p_earth,
        "Radius_Earth_err": sigma_R_p_earth,
        "Mass_Earth": M_p_earth,
        "Mass_Earth_err": sigma_M_p_earth,
        "Density_g_cm3": density_cgs,
        "Surface_log_g": log_g,
        "T_eq_K": T_eq,
        "T_eq_err": sigma_T_eq,
        "Insolation_Earth": insolation_earth,
        "Habitable_Zone_Status": hz_status,
        "JWST_TSM_Score": tsm,
        "RV_SemiAmplitude_m_s": k_rv_ms
    }
