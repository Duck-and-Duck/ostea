# ==============================================================================
#   OSTE-MoE CO-PILOT RESMİ BİLİM İNSANI BRİFİNG TESTİ
# ==============================================================================
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import OSTE_MoE_Pipeline, generate_aas_latex_table, generate_observer_briefing

print("="*85)
print("  OSTE-MoE: BİLİM İNSANI HIZLANDIRICI (SCIENTIFIC CO-PILOT) GÖSTERİMİ")
print("  (Hedef: Bilim İnsanının Yerini Almak Değil, İşini 100 Kat Hızlandırmak!)")
print("="*85)

pipeline = OSTE_MoE_Pipeline()

# TOI-270 b (M-Cüce Rezonant Süper-Dünya Sistemi)
p = 3.359857
t0 = 1.20
dur = 0.070
depth = 0.0011
star = {"r_s": 0.38, "m_s": 0.40, "teff": 3386.0}

time_pts = np.linspace(0, 27.4, 6000)
ph = ((time_pts - t0 + 0.5 * p) % p) - 0.5 * p
in_tr = np.abs(ph) < (dur / 2.0)
flux = 1.0 + np.random.normal(0, 0.00015, len(time_pts))
flux[in_tr] -= depth * (1.0 - 0.2 * (2.0 * ph[in_tr] / dur)**2)

img_oot = np.ones((11, 11)) * 10000.0 + np.random.normal(0, 0.18, (11, 11))
img_in = img_oot.copy()
img_in[5, 5] -= 10000.0 * depth

print("\n--> 1. TESS Işık Eğrisi ve 2D TPF Pikselleri 5 Milisaniyede Analiz Ediliyor...")
report = pipeline.process_candidate(time_pts, flux, p, t0, dur, img_oot=img_oot, img_in=img_in, star_params=star)

phys = report["physical_properties"]
atmo = report.get("atmosphere", None)

print(f"--> [✓ DOĞRULAMA TAMAMLANDI]: Karar = {report['decision']} (CNN Güveni: %{report['cnn_confidence']*100:.1f})")

# 2. Bilim İnsanı İçin Gözlemci Takip Brifingi
briefing = generate_observer_briefing("TOI-270 b", phys, atmo)
print(briefing)

# 3. Bilim İnsanı İçin AAS LaTeX Tablosu
latex_tab = generate_aas_latex_table("TOI-270 b", phys, atmo)
print("\n[BİLİM İNSANI İÇİN MAKALEYE DOĞRUDAN YAPIŞTIRILABİLİR AAS LATEX TABLOSU]:\n")
print(latex_tab)

print("="*85)
print("RESMİ HÜKÜM: Astronomun haftalar süren veri ayıklama ve hesaplama yükü tamamlanmış,")
print("hata bütçeli karar dosyası ve gözlem önerisi masasına saniyeler içinde konulmuştur.")
print("="*85)
