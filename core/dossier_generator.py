"""
OSTE-MoE: Astronomer Co-Pilot Dossier Generator.
Bilim insanının makalesine ve gözlem teklifine doğrudan ekleyebileceği
standart AAS LaTeX tablosunu ve takip spektroskopisi özetini üretir.
"""

def generate_aas_latex_table(candidate_name, phys, atmo=None):
    latex_code = f"""% --- AAS/ApJ STANDART ÖTEGEZEGEN PARAMETRE TABLOSU ---
% OSTE-MoE Autonomous Co-Pilot Tarafından Üretilmiştir.
\\begin{{deluxetable*}}{{lcc}}
\\tablecaption{{Doğrulanan Ötegezegen Fiziksel ve Yörünge Parametreleri: {candidate_name} \\label{{tab:{candidate_name.lower().replace(' ', '_')}}}}}
\\tablehead{{\\colhead{{Parametre}} & \\colhead{{Değer & Hata}} & \\colhead{{Birim}}}}
\\startdata
Yörünge Periyodu ($P$) & ${phys['Period_days']:.6f}$ & Gün \\\\
Yarı-Büyük Eksen ($a$) & ${phys['SemiMajorAxis_AU']:.4f} \\pm {phys['SemiMajorAxis_err']:.4f}$ & AU \\\\
Gezegen Yarıçapı ($R_p$) & ${phys['Radius_Earth']:.2f} \\pm {phys['Radius_Earth_err']:.2f}$ & $R_{{\\oplus}}$ \\\\
Tahmini Kütle ($M_p$) & ${phys['Mass_Earth']:.2f} \\pm {phys['Mass_Earth_err']:.2f}$ & $M_{{\\oplus}}$ \\\\
Ortalama Yoğunluk ($\\rho$) & ${phys['Density_g_cm3']:.2f}$ & $\\text{{g}}\\,\\text{{cm}}^{{-3}}$ \\\\
Denge Sıcaklığı ($T_{{\\text{{eq}}}}$) & ${phys['T_eq_K']:.1f} \\pm {phys['T_eq_err']:.1f}$ & K \\\\
Alınan Işınım Akısı ($S$) & ${phys['Insolation_Earth']:.2f}$ & $S_{{\\oplus}}$ \\\\
JWST İletim Metriği (TSM) & ${phys['JWST_TSM_Score']:.1f}$ & \\dots \\\\
Beklenen RV Yarı-Genliği ($K$) & ${phys['RV_SemiAmplitude_m_s']:.2f}$ & $\\text{{m}}\\,\\text{{s}}^{{-1}}$ \\\\
\\enddata
\\tablecomments{{Kütle tahmini Chen \\& Kipping (2017) modelleriyle yapılmıştır. TSM Kempton et al. (2018) standardındadır.}}
\\end{{deluxetable*}}
"""
    return latex_code

def generate_observer_briefing(candidate_name, phys, atmo=None):
    briefing = f"""
================================================================================
             GÖZLEMCİYE HAZIR BİLİMSEL TAKİP BRİFİNGİ (CO-PILOT DOSSIER)
================================================================================
HEDEF               : {candidate_name}
YÖRÜNGESEL REJİM    : P = {phys['Period_days']:.4f} Gün | a = {phys['SemiMajorAxis_AU']:.4f} ± {phys['SemiMajorAxis_err']:.4f} AU
GEZEGEN ÖLÇEĞİ      : Rp = {phys['Radius_Earth']:.2f} ± {phys['Radius_Earth_err']:.2f} R_Dünya | Mp = {phys['Mass_Earth']:.2f} ± {phys['Mass_Earth_err']:.2f} M_Dünya
TERMAL REJİM        : Teq = {phys['T_eq_K']:.1f} ± {phys['T_eq_err']:.1f} K | {phys['Habitable_Zone_Status']}

[BİLİM İNSANI İÇİN TAKİP STRATEJİSİ VE TELESKOP SEÇİMİ]:
1. YER TABANLI RADYAL HIZ (RV) DOĞRULAMASI:
   * Beklenen Yarı-Genlik (K) : {phys['RV_SemiAmplitude_m_s']:.2f} m/s
   * Önerilen Enstrüman       : {'HARPS / ESPRESSO (VLT) - Ultra Hassas RV Gerekli' if phys['RV_SemiAmplitude_m_s'] < 3.0 else 'HIRES (Keck) / MAROON-X - Standart RV ile Tespit Edilebilir'}
   * Öncelik                  : {'YÜKSEK (Kayalık Gezegen Kütle Teyidi)' if phys['Radius_Earth'] < 2.0 else 'ORTA (Gaz Devi Kütle Doğrulaması)'}

2. JAMES WEBB (JWST) ATMOSFERİK KARAKTERİZASYON:
   * TSM Skoru                : {phys['JWST_TSM_Score']:.1f} (Kempton et al. 2018 Standardı)
   * Fizibilite Durumu        : {'MÜKEMMEL (JWST NIRSpec için 1. Derece Öncelikli Hedef)' if phys['JWST_TSM_Score'] > 12.0 else 'İKİNCİL (Sönük veya Küçük Atmosfer Skalası)'}
"""
    if atmo:
        briefing += f"""
3. HIZLI KATMAN 4 SBI ATMOSFER KESTİRİMİ:
   * Spektroskopik Teq        : {atmo.get('T_eq_K', 0.0):.1f} K
   * log(H2O) / log(CO2)      : {atmo.get('log_H2O', 0.0):.2f} / {atmo.get('log_CO2', 0.0):.2f}
"""
    briefing += "================================================================================"
    return briefing
