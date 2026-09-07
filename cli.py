#!/usr/bin/env python3
"""
OSTE-MoE: Hierarchical Multi-Modal Mixture-of-Experts CLI.
Uluslararası araştırmacılar için otonom keşif ve doğrulama komut satırı arayüzü.
"""
import sys
import argparse
import numpy as np
from core import OSTE_MoE_Pipeline

def run_demo():
    print("="*85)
    print("  OSTE-MoE OTONOM ÖTEGEZEGEN KEŞİF VE KARAKTERİZASYON DEMOSU")
    print("="*85)
    pipeline = OSTE_MoE_Pipeline()
    
    # 27.4 günlük sentetik test sinyali
    time_pts = np.linspace(0, 27.4, 6000)
    p, t0, dur, depth = 3.359857, 1.20, 0.070, 0.0011 # TOI-270 b parametreleri
    ph = ((time_pts - t0 + 0.5 * p) % p) - 0.5 * p
    in_tr = np.abs(ph) < (dur / 2.0)
    flux = 1.0 + np.random.normal(0, 0.00015, len(time_pts))
    flux[in_tr] -= depth * (1.0 - 0.20 * (2.0 * ph[in_tr] / dur)**2)

    img_oot = np.ones((11, 11)) * 10000.0 + np.random.normal(0, 0.18, (11, 11))
    img_in = img_oot.copy()
    img_in[5, 5] -= 10000.0 * depth

    star = {"r_s": 0.38, "m_s": 0.40, "teff": 3386.0}
    
    print("--> Örnek Işık Eğrisi ve 2D TPF Pikselleri Taranıyor...")
    res = pipeline.process_candidate(time_pts, flux, p, t0, dur, img_oot=img_oot, img_in=img_in, star_params=star)

    print(f"\n[SONUÇ]: Karar = {res['decision']} | CNN Güveni = %{res['cnn_confidence']*100:.1f} | Süre = {res['latency_ms']:.2f} ms")
    if res['decision'] == 'PLANET':
        p_prop = res['physical_properties']
        atmo = res['atmosphere']
        print(f"  * Gezegen Yarıçapı  : {p_prop['Radius_Earth']:.2f} R_Dünya")
        print(f"  * Yarı-Büyük Eksen  : {p_prop['SemiMajorAxis_AU']:.4f} AU")
        print(f"  * Spektroskopik Teq : {atmo['T_eq_K']:.1f} K")
        print(f"  * log(H2O) / log(CO2): {atmo['log_H2O']:.2f} / {atmo['log_CO2']:.2f}")
    print("="*85)

def main():
    parser = argparse.ArgumentParser(description="OSTE-MoE Exoplanet Discovery CLI")
    parser.add_argument("--demo", action="store_true", help="Hızlı tanıtım ve doğrulama demosunu çalıştırır.")
    args = parser.parse_args()

    if args.demo or len(sys.argv) == 1:
        run_demo()

if __name__ == "__main__":
    main()
