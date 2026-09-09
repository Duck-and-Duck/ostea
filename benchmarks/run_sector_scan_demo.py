# ==============================================================================
#   1,000 YILDIZLIK TESS SEKTÖR TARAMA VE RESMİ ADAY KATALOĞU DEMOSU
# ==============================================================================
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.sector_scanner import TESSSectorScanner

print("="*95)
print("  OSTE-MoE: 1,000 YILDIZLIK TESS SEKTÖRÜ OTONOM TARAMA VE ADAY KATALOĞU DEMOSU")
print("="*95)

time_pts = np.linspace(0, 27.4, 6000)
sector_stars = []
np.random.seed(2026)

# 1,000 Yıldız Simülasyonu: 900 Sakin Yıldız, 100 Gezegenli Yıldız (Bazıları çok gezegenli)
for i in range(1000):
    flux = 1.0 + np.random.normal(0, 0.00015, len(time_pts))
    params = {"r_s": float(np.random.uniform(0.4, 1.2)), "m_s": float(np.random.uniform(0.4, 1.2)), "teff": float(np.random.uniform(3400.0, 6200.0))}

    if i < 900:
        # Sakin Yıldız
        pass
    else:
        # Gezegenli Yıldız
        p = float(np.random.uniform(2.0, 14.0))
        dur = float(0.10 * (p/5.0)**(1.0/3.0))
        t0 = float(np.random.uniform(0.5, 2.5))
        d = float(10 ** np.random.uniform(np.log10(0.0008), np.log10(0.0120)))
        ph = ((time_pts - t0 + 0.5 * p) % p) - 0.5 * p
        flux[np.abs(ph) < (dur / 2.0)] -= d

    sector_stars.append({
        "tic_id": f"TIC_{10000000 + i}",
        "time": time_pts,
        "flux": flux,
        "params": params
    })

scanner = TESSSectorScanner(device="cuda")
csv_path = os.path.join(os.path.dirname(__file__), "..", "predicted_toi_catalog.csv")
catalog = scanner.scan_sector(sector_stars, output_csv=csv_path)

if catalog:
    print("\n--> [BİLİM İNSANININ ÖNÜNE KONAN İLK 3 TAHMİN EDİLMİŞ ADAY]:")
    for c in catalog[:3]:
        print(f"  * Aday: {c['Candidate_ID']} | P: {c['Period_Days']} g | Rp: {c['Radius_Earth']} R_Dunya | Teq: {c['Teq_K']} K | Atmosfer: {c['Has_Atmosphere']} (H2O: {c['log_H2O']}, CO2: {c['log_CO2']})")
print("="*95)
