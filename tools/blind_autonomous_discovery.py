#!/usr/bin/env python3
"""
==============================================================================
   OSTE-MoE BLIND AUTONOMOUS DISCOVERY ENGINE (ZERO-PRIOR SPACE MINER)
==============================================================================
* Modele kesinlikle hazır periyot, epoch veya transit süresi verilmez.
* Ham 1D akı üzerinden periyot ve transit geometrisi otonom olarak aranır.
* Katman 1.5 (Astrometri) ve Katman 3 (AstroNet-HQ) ile aday tescillenir.
* Doğrulanan gezegenler Katman 4 SBI Atmosfer motoruyla çözülür.
* Karar verildikten SONRA NASA arşiviyle bağımsız çapraz denetim yapılır.
==============================================================================
"""

import os
import sys
import time
import json
import warnings
import numpy as np
import torch
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import AstroNetHQ, AstrometricCentroidExpert, gpu_fast_fold, AtmosphericInversionExpert, characterize_discovered_planet

print("="*95)
print("  OSTE-MoE: SIFIR ÖN BİLGİLİ KÖR OTONOM KEŞİF VE KARAKTERİZASYON MOTORU")
print(f"--> PLATFORM: {device.upper()} (NVIDIA Tensor Cores Aktif)")
print("--> KURAL   : Periyot ve transit parametreleri KATALOGDAN ALINMAZ; ham veriden aranır!")
print("="*95)

VAULT_DIR = os.path.join(os.path.dirname(__file__), "..", "_DATA_VAULT_")

# Katman 3 ve Katman 1.5 Hazırlığı
weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "astronet_hq.pt")
model_cnn = AstroNetHQ().to(device).half()
if os.path.exists(weights_path):
    model_cnn.load_state_dict(torch.load(weights_path, map_location=device))
model_cnn.eval()

centroid_expert = AstrometricCentroidExpert()
atmo_expert = AtmosphericInversionExpert(device=device)

def blind_frequency_search(time_arr, flux_arr):
    """
    Sıfır ön bilgi ile kör frekans taraması (Kovács et al. 2002).
    0.4 gün ile 16.0 gün arasında 10.000 deneme periyoduyla transit arar.
    """
    t0_s = time.perf_counter()
    bls = BoxLeastSquares(time_arr, flux_arr)
    periods = np.exp(np.linspace(np.log(0.4), np.log(16.0), 10000))
    durations = np.linspace(0.04, 0.20, 8)
    periodogram = bls.power(periods, durations)

    best_idx = np.argmax(periodogram.power)
    p_cand = float(periodogram.period[best_idx])
    t0_cand = float(periodogram.transit_time[best_idx])
    dur_cand = float(periodogram.duration[best_idx])
    p_power = float(periodogram.power[best_idx])
    std_pow = np.std(periodogram.power)
    snr_cand = p_power / (std_pow if std_pow > 0 else 1e-7)

    # 0.5x ve 2.0x Rezonans Harmonik Çözücü
    final_p, final_t0, final_dur = p_cand, t0_cand, dur_cand
    for factor in [0.5, 2.0]:
        test_p = p_cand * factor
        if 0.4 <= test_p <= 16.0:
            sub = bls.power(np.array([test_p]), [dur_cand])
            if len(sub.power) > 0 and float(sub.power[0]) >= 0.85 * p_power:
                final_p = test_p
                final_t0 = float(sub.transit_time[0])
                break

    lat_search_ms = (time.perf_counter() - t0_s) * 1000.0
    return final_p, final_t0, final_dur, snr_cand, lat_search_ms

# KASADAKİ TÜM DOSYALARI LİSTELE
vault_files = [f for f in os.listdir(VAULT_DIR) if f.endswith(".npz")]
if len(vault_files) == 0:
    print("[!] HATA: _DATA_VAULT_ klasöründe veri bulunamadı! Önce smart_data_loader.py çalıştırılmalı.")
    sys.exit(1)

print(f"--> Kasadaki {len(vault_files)} Gerçek Sistem Kör Arama Moduna Alınıyor...\n")

discovery_records = []
t_grand_start = time.perf_counter()

for idx, fname in enumerate(vault_files, 1):
    fpath = os.path.join(VAULT_DIR, fname)
    data = np.load(fpath)

    # HAM VERİLERİ ÇEK (Asla hedef bilgisine bakılmaz!)
    t_raw = data["time"]
    f_raw = data["flux"]
    img_oot = data["img_oot"]
    img_in = data["img_in"]
    target_info = json.loads(str(data["target_info"])) # Sadece en son doğruluk kıyası için!

    # 1. KÖR TRANSİT VE PERİYOT ARAMASI (Yapay Zekâ Sinyali Kendisi Bulur!)
    p_found, t0_found, dur_found, snr_search, lat_bls = blind_frequency_search(t_raw, f_raw)

    # 2. KATMAN 1.5: ASTROMETRİK CENTROID KONTROLÜ
    cen_res = centroid_expert.evaluate_tpf_centroid(img_oot, img_in)
    passed_astrometry = cen_res["is_on_target"]

    # 3. KATMAN 2 & 3: GPU KATLAMA VE ASTRONET-HQ VETTING
    t_gpu = torch.tensor(t_raw, dtype=torch.float32, device=device)
    f_gpu = torch.tensor(f_raw, dtype=torch.float32, device=device)
    g, l, d_meas = gpu_fast_fold(t_gpu, f_gpu, p_found, t0_found, dur_found, device=device)

    dummy_bg = g.half().repeat(256, 1, 1)
    dummy_bl = l.half().repeat(256, 1, 1)
    with torch.no_grad():
        prob_ai = torch.sigmoid(model_cnn(dummy_bg, dummy_bl))[0].item()

    # OTONOM KARAR MEKANİZMASI
    if not passed_astrometry:
        decision = "BINARY"
        reason = f"BEB_OFFSET ({cen_res['offset_arcsec']:.1f}'')"
    elif d_meas >= 0.028:
        decision = "BINARY"
        reason = "DEEP_ECLIPSE"
    elif prob_ai >= 0.35 and 0.00020 <= d_meas < 0.025:
        decision = "PLANET"
        reason = "TRANSIT_VERIFIED"
    else:
        decision = "NON_PLANET"
        reason = "NOISE_OR_NO_TRANSIT"

    # 4. KARAKTERİZASYON VE ATMOSFER (Eğer Gezegen Onaylandıysa)
    char_info, atmo_info = None, None
    if decision == "PLANET":
        char_info = characterize_discovered_planet(p_found, d_meas, dur_found)
        atmo_info = atmo_expert.retrieve_atmosphere(d_meas)

    # 5. ADLİ TIP: GERÇEKLE BAĞIMSIZ KARŞILAŞTIRMA (Ground-Truth Check)
    p_true = target_info.get("p", None)
    expected_type = target_info.get("type", "UNKNOWN")
    target_name = target_info.get("name", fname)

    p_error_pct = 999.0
    period_hit = False
    if p_true is not None and p_true > 0:
        p_error_pct = abs(p_found - p_true) / p_true * 100.0
        # Tam isabet veya 0.5x / 2.0x harmonik toleransı
        period_hit = (p_error_pct < 1.5) or (abs(p_found - 0.5*p_true)/(0.5*p_true)*100.0 < 1.5) or (abs(p_found - 2.0*p_true)/(2.0*p_true)*100.0 < 1.5)

    discovery_records.append({
        "name": target_name,
        "tic": target_info.get("tic", ""),
        "expected": expected_type,
        "decision": decision,
        "reason": reason,
        "p_found": p_found,
        "p_true": p_true,
        "p_err": p_error_pct,
        "period_hit": period_hit,
        "depth_ppm": d_meas * 1e6,
        "cnn_prob": prob_ai,
        "search_ms": lat_bls,
        "char": char_info,
        "atmo": atmo_info
    })

    status_tag = "[✓ KÖR KEŞİF BAŞARILI]" if (decision == "PLANET" and period_hit) or (decision != "PLANET" and "PLANET" not in expected_type) else "[!]"
    print(f"[{idx:02d}/{len(vault_files):02d}] Hedef: {target_name:<16} | Karar: {decision:<10} | P_bul={p_found:7.4f}g (P_ger={p_true if p_true else 0.0:7.4f}g) {status_tag}")

tot_grand_time = time.perf_counter() - t_grand_start

# =========================================================================
# RESMİ BİLİMSEL KEŞİF AUDIT RAPORU
# =========================================================================
print("\n" + "="*95)
print("             OSTE-MoE RESMİ KÖR OTONOM KEŞİF DENETİM RAPORU                       ")
print("="*95)
print(f"{'HEDEF ADI':<16} | {'KARAR':<10} | {'P_BULUNAN':<10} | {'P_GERÇEK':<10} | {'HATA':<8} | {'CNN GÜVEN':<9} | {'DURUM'}")
print("-" * 95)

confirmed_planets = 0
correct_rejections = 0
total_targets = len(discovery_records)

for r in discovery_records:
    p_found_str = f"{r['p_found']:.4f}g"
    p_true_str = f"{r['p_true']:.4f}g" if r['p_true'] else "N/A"
    err_str = f"%{r['p_err']:.2f}" if r['p_true'] else "N/A"
    
    is_success = False
    if r['decision'] == "PLANET" and r['period_hit']:
        confirmed_planets += 1
        is_success = True
        status_label = "[✓ GEZEGEN TAM BULUNDU]"
    elif r['decision'] != "PLANET" and ("BINARY" in r['expected'] or "NULL" in r['expected'] or "STABLE" in r['expected']):
        correct_rejections += 1
        is_success = True
        status_label = "[✓ TUZAK/GÜRÜLTÜ ELENDİ]"
    else:
        status_label = "[✗ TUTMADI]"

    print(f"{r['name']:<16} | {r['decision']:<10} | {p_found_str:<10} | {p_true_str:<10} | {err_str:<8} | %{r['cnn_prob']*100:6.1f}   | {status_label}")

success_rate = ((confirmed_planets + correct_rejections) / total_targets) * 100.0

print("-" * 95)
print(f"--> Toplam Taranan Hedef Sayısı : {total_targets}")
print(f"--> Körlemesine Doğrulanan Gezegen: {confirmed_planets}")
print(f"--> Doğru Elenen Tuzak / Gürültü : {correct_rejections}")
print(f"--> GENEL KÖR KEŞİF BAŞARISI     : %{success_rate:.1f}")
print(f"--> Toplam İşleme Süresi         : {tot_grand_time:.2f} Saniye (~{tot_grand_time/total_targets:.2f} sn/Sistem)")
print("="*95)

# ONAYLANAN BİR GEZEGENİN ÖRNEK FİZİKSEL VE ATMOSFERİK KARNESİ
planet_samples = [r for r in discovery_records if r['decision'] == "PLANET" and r['char'] is not None]
if len(planet_samples) > 0:
    top_p = planet_samples[0]
    print("\n" + "="*95)
    print(f"  ÖRNEK KÖR KEŞİF DOSYASI: {top_p['name']} ({top_p['tic']})")
    print("="*95)
    print(f"  * Otonom Bulunan Yörünge: P = {top_p['p_found']:.5f} Gün (Katalog Hatası: %{top_p['p_err']:.3f})")
    print(f"  * Gezegen Yarıçapı      : {top_p['char']['Radius_Earth']:.2f} R_Dünya | Yarı-Büyük Eksen: {top_p['char']['SemiMajorAxis_AU']:.4f} AU")
    print(f"  * Termodinamik Rejim    : Denge Sıcaklığı (Teq) = {top_p['char']['T_eq_K']:.1f} K")
    if top_p['atmo'] is not None:
        print(f"  * Katman 4 SBI Atmosfer : log(H2O) = {top_p['atmo']['log_H2O']:.2f} | log(CO2) = {top_p['atmo']['log_CO2']:.2f}")
    print("="*95)
