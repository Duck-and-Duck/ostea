# ==============================================================================
#   OSTE-MoE UÇTAN UCA HİYERARŞİK MoE RESMİ DOĞRULAMA GAUNTLET'İ
#   (NASA Hedefleri + BEB Tuzakları + SBI Atmosfer Çözümlemesi)
# ==============================================================================
import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import OSTE_MoE_Pipeline

print("="*90)
print("  OSTE-MoE: UÇTAN UCA HİYERARŞİK MoE TÜM KATMANLAR RESMİ AUDIT RAPORU")
print("  (L1 Anomaly Gate -> L1.5 Centroid -> L2 Folding -> L3 AstroNet -> L4 SBI)")
print("="*90)

pipeline = OSTE_MoE_Pipeline()

time_pts = np.linspace(0, 27.4, 6000)

def create_prf_stamp(center_r, center_c, flux_total, grid_size=11, sigma_prf=1.2):
    r, c = np.indices((grid_size, grid_size))
    prf = np.exp(-((r - center_r)**2 + (c - center_c)**2) / (2.0 * sigma_prf**2))
    prf /= np.sum(prf)
    return prf * flux_total

# 4 Kanonik Test Senaryosu
CANONICAL_TARGETS = [
    {
        "name": "WASP-18 b (Ultra-Sıcak Jüpiter)",
        "p": 0.941452, "t0": 0.45, "dur": 0.090, "depth": 0.0093,
        "is_beb": False, "is_quiet": False, "exp": "PLANET",
        "star": {"r_s": 1.25, "m_s": 1.22, "teff": 6400.0}
    },
    {
        "name": "TOI-270 b (Süper-Dünya)",
        "p": 3.359857, "t0": 1.20, "dur": 0.070, "depth": 0.0011,
        "is_beb": False, "is_quiet": False, "exp": "PLANET",
        "star": {"r_s": 0.38, "m_s": 0.40, "teff": 3386.0}
    },
    {
        "name": "Sinsi Arka Plan İkilisi (BEB Tuzağı)",
        "p": 2.500000, "t0": 0.80, "dur": 0.080, "depth": 0.0080,
        "is_beb": True, "is_quiet": False, "exp": "BINARY",
        "star": {"r_s": 1.0, "m_s": 1.0, "teff": 5778.0}
    },
    {
        "name": "Sessiz Durağan Yıldız (Gürültü Referansı)",
        "p": 3.500000, "t0": 1.00, "dur": 0.100, "depth": 0.0000,
        "is_beb": False, "is_quiet": True, "exp": "NON_PLANET",
        "star": {"r_s": 1.0, "m_s": 1.0, "teff": 5778.0}
    }
]

hits = 0

for idx, tgt in enumerate(CANONICAL_TARGETS, 1):
    ph = ((time_pts - tgt["t0"] + 0.5 * tgt["p"]) % tgt["p"]) - 0.5 * tgt["p"]
    in_tr = np.abs(ph) < (tgt["dur"] / 2.0)
    
    flux = 1.0 + np.random.normal(0, 0.00015, len(time_pts))
    target_flux = 10000.0
    img_oot = create_prf_stamp(5.0, 5.0, target_flux) + np.random.normal(0, 0.18, (11, 11))

    if tgt["is_quiet"]:
        img_in = img_oot.copy() + np.random.normal(0, 0.18, (11, 11))
    elif tgt["is_beb"]:
        flux[in_tr] -= tgt["depth"]
        # Arka plan ikilisi 2 piksel ofsette
        bg_oot = create_prf_stamp(7.0, 4.0, 600.0)
        bg_in = create_prf_stamp(7.0, 4.0, 600.0 * 0.4)
        img_oot += bg_oot
        img_in = create_prf_stamp(5.0, 5.0, target_flux) + bg_in + np.random.normal(0, 0.18, (11, 11))
    else:
        flux[in_tr] -= tgt["depth"] * (1.0 - 0.2 * (2.0 * ph[in_tr] / tgt["dur"])**2)
        img_in = create_prf_stamp(5.0, 5.0, target_flux * (1.0 - tgt["depth"])) + np.random.normal(0, 0.18, (11, 11))

    report = pipeline.process_candidate(
        time_pts, flux, tgt["p"], tgt["t0"], tgt["dur"],
        img_oot=img_oot, img_in=img_in, star_params=tgt["star"]
    )

    is_correct = (report["decision"] == tgt["exp"])
    if is_correct: hits += 1
    status = "[✓ BAŞARILI]" if is_correct else "[✗ HATA]"

    print(f"\n[{idx}/4] {tgt['name']}")
    print(f"  * Karar: {report['decision']:<10} (Beklenen: {tgt['exp']}) | Süre: {report['latency_ms']:.2f} ms {status}")

    if report["decision"] == "PLANET":
        p_prop = report["physical_properties"]
        atmo = report["atmosphere"]
        print(f"  * [FİZİKSEL REJİM]  : Yarıçap = {p_prop['Radius_Earth']:.2f} R_Dünya | Yarı-Büyük Eksen = {p_prop['SemiMajorAxis_AU']:.4f} AU")
        print(f"  * [KATMAN 4 SBI]   : Teq = {atmo['T_eq_K']:.1f} K | log(H2O) = {atmo['log_H2O']:.2f} | log(CO2) = {atmo['log_CO2']:.2f}")
    elif report["decision"] == "BINARY":
        print(f"  * [KATMAN 1.5 ELEME]: {report['reason']} (Açısal Kayma: {report.get('centroid_offset_arcsec', 0.0):.1f}'')")

print("\n" + "="*90)
print(f"--> UÇTAN UCA HİYERARŞİK MoE DENETİM SKORU: {hits} / 4 (%{hits/4.0*100:.1f})")
print("=================================================================================")
if hits == 4:
    print("[RESMİ AKREDİTASYON]: TÜM KATMANLAR ENTEGRE ÇALIŞMAKTA VE GITHUB İÇİN HAZIRDIR.")
print("="*90)
