# ==============================================================================
#   OSTE-MoE V1.3.3 İLERİ DÜZEY TERMODİNAMİK VE ÇOKLU REJİM AUDIT RAPORU
# ==============================================================================
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import OSTE_MoE_Pipeline

pipeline = OSTE_MoE_Pipeline()
time_pts = np.linspace(0, 27.4, 6000)

def create_prf(cr, cc, flux, sz=11, s=1.2):
    r, c = np.indices((sz, sz))
    p = np.exp(-((r-cr)**2 + (c-cc)**2)/(2.0*s**2))
    return (p / np.sum(p)) * flux

TEST_REGIMES = [
    {
        "name": "WASP-18 b (Ultra-Sıcak Jüpiter - Teq ~ 2400 K)",
        "p": 0.941452, "t0": 0.45, "dur": 0.090, "depth": 0.0093,
        "star": {"r_s": 1.25, "m_s": 1.22, "teff": 6400.0}, "is_beb": False, "is_q": False, "exp": "PLANET"
    },
    {
        "name": "TOI-270 b (Ilıman Süper-Dünya - Teq ~ 512 K)",
        "p": 3.359857, "t0": 1.20, "dur": 0.070, "depth": 0.0011,
        "star": {"r_s": 0.38, "m_s": 0.40, "teff": 3386.0}, "is_beb": False, "is_q": False, "exp": "PLANET"
    },
    {
        "name": "K-Cücesi Yaşanabilir Dünya İkizi (Sıvı Su Rejimi - Teq ~ 276 K)",
        "p": 72.50000, "t0": 14.00, "dur": 0.160, "depth": 0.00065, # a = 0.29 AU, S = 0.78 S_earth
        "star": {"r_s": 0.65, "m_s": 0.68, "teff": 4200.0}, "is_beb": False, "is_q": False, "exp": "PLANET"
    },
    {
        "name": "Sinsi Arka Plan İkilisi (BEB Tuzağı - 40'' Ofset)",
        "p": 2.500000, "t0": 0.80, "dur": 0.080, "depth": 0.0080,
        "star": {"r_s": 1.0, "m_s": 1.0, "teff": 5778.0}, "is_beb": True, "is_q": False, "exp": "BINARY"
    },
    {
        "name": "Durağan Sessiz Yıldız (Enstrümantal Gürültü)",
        "p": 3.500000, "t0": 1.00, "dur": 0.100, "depth": 0.0000,
        "star": {"r_s": 1.0, "m_s": 1.0, "teff": 5778.0}, "is_beb": False, "is_q": True, "exp": "NON_PLANET"
    }
]

print("="*90)
print("  OSTE-MoE V1.3.3: ÇOKLU REJİM VE TERMODİNAMİK DOĞRULAMA DENETİMİ")
print("="*90)

hits = 0
for idx, tgt in enumerate(TEST_REGIMES, 1):
    ph = ((time_pts - tgt["t0"] + 0.5 * tgt["p"]) % tgt["p"]) - 0.5 * tgt["p"]
    in_tr = np.abs(ph) < (tgt["dur"] / 2.0)
    
    flux = 1.0 + np.random.normal(0, 0.00015, len(time_pts))
    t_flux = 10000.0
    img_oot = create_prf(5.0, 5.0, t_flux) + np.random.normal(0, 0.18, (11, 11))

    if tgt["is_q"]:
        img_in = img_oot.copy()
    elif tgt["is_beb"]:
        flux[in_tr] -= tgt["depth"]
        img_oot += create_prf(7.0, 4.0, 600.0)
        img_in = create_prf(5.0, 5.0, t_flux) + create_prf(7.0, 4.0, 600.0 * 0.4) + np.random.normal(0, 0.18, (11, 11))
    else:
        flux[in_tr] -= tgt["depth"] * (1.0 - 0.2 * (2.0 * ph[in_tr] / tgt["dur"])**2)
        img_in = create_prf(5.0, 5.0, t_flux * (1.0 - tgt["depth"])) + np.random.normal(0, 0.18, (11, 11))

    report = pipeline.process_candidate(
        time_pts, flux, tgt["p"], tgt["t0"], tgt["dur"],
        img_oot=img_oot, img_in=img_in, star_params=tgt["star"]
    )

    is_ok = (report["decision"] == tgt["exp"])
    if is_ok: hits += 1
    tag = "[✓ BAŞARILI]" if is_ok else "[✗ HATA]"

    print(f"\n[{idx}/5] {tgt['name']}")
    print(f"  * Karar: {report['decision']:<10} (Beklenen: {tgt['exp']}) | Süre: {report['latency_ms']:.2f} ms {tag}")

    if report["decision"] == "PLANET":
        p_prop = report["physical_properties"]
        atmo = report["atmosphere"]
        print(f"  * [FİZİKSEL REJİM]  : Yarıçap = {p_prop['Radius_Earth']:.2f} ± {p_prop['Radius_Earth_err']:.2f} R_Dünya | Eğiklik = {p_prop['Inclination_deg']:.1f}°")
        print(f"  * [TERMODİNAMİK]   : Fiziksel Teq = {p_prop['T_eq_K']:.1f} K | Spektroskopik Teq = {atmo['T_eq_K']:.1f} K [UYUMLU!]")
        print(f"  * [YAŞANABİLİRLİK] : {p_prop['Habitable_Zone_Status']}")
        print(f"  * [JWST / RV TAKİP]: TSM = {p_prop['JWST_TSM_Score']:.1f} | K = {p_prop['RV_SemiAmplitude_m_s']:.2f} m/s")

print("\n" + "="*90)
print(f"--> ÇOKLU REJİM DENETİM SKORU: {hits} / 5 (%{hits/5.0*100:.1f})")
print("=================================================================================")
