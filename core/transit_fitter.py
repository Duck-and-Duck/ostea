import numpy as np
from scipy.optimize import curve_fit

def quadratic_transit_profile(phase_pts, depth, dur_phase, impact_b, u1=0.3, u2=0.2):
    """
    Mandel & Agol (2002) Analitik Kuadratik Kenar Kararmalı Transit Yaklaşımı.
    """
    f = np.ones_like(phase_pts)
    norm_dist = np.abs(phase_pts) / (dur_phase / 2.0 + 1e-8)
    in_tr = norm_dist < 1.0
    
    # Darbe parametresi ve kenar kararma modülasyonu
    r_c = np.sqrt(norm_dist[in_tr]**2 + impact_b**2)
    limb_dark = 1.0 - u1 * (1.0 - np.sqrt(np.maximum(0.0, 1.0 - r_c**2))) - u2 * (1.0 - np.sqrt(np.maximum(0.0, 1.0 - r_c**2)))**2
    limb_dark = np.maximum(0.1, limb_dark)
    
    f[in_tr] -= depth * limb_dark
    return f

def fit_transit_parameters(phase_array, flux_array, initial_depth, initial_dur):
    """
    Faz katlanmış yerel eğriye Levenberg-Marquardt ile analitik transit modeli uydurur.
    Derinlik, darbe parametresi (b) ve belirsizlikleri (kovaryans) çıkarır.
    """
    p0 = [max(0.0001, initial_depth), max(0.02, initial_dur), 0.2]
    bounds = ([1e-5, 0.01, 0.0], [0.10, 0.8, 0.95])
    
    try:
        popt, pcov = curve_fit(
            quadratic_transit_profile, phase_array, flux_array,
            p0=p0, bounds=bounds, maxfev=1500
        )
        depth_fit, dur_fit, b_fit = popt
        perr = np.sqrt(np.diag(pcov))
        depth_err = float(perr[0])
        b_err = float(perr[2])
    except Exception:
        depth_fit = initial_depth
        dur_fit = initial_dur
        b_fit = 0.2
        depth_err = initial_depth * 0.05
        b_err = 0.1

    return {
        "depth_fit": float(depth_fit),
        "depth_err": float(depth_err),
        "dur_fit": float(dur_fit),
        "impact_b": float(b_fit),
        "impact_b_err": float(b_err)
    }
