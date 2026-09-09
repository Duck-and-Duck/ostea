# ==============================================================================
#   OSTE-MoE MASTER BİRLEŞİK DOĞRULAMA ORKESTRATÖRÜ
#   (Tüm Resmi Test Süitlerini Tek Seferde Çalıştırıp Konsolide Rapor Üretir)
# ==============================================================================
import os
import sys
import subprocess

print("="*95)
print("  OSTE-MoE V1.3.3 MASTER AUDIT SUITE: TÜM KURUMSAL TESTLERİN BİRLEŞİK ÇALIŞTIRICISI")
print("="*95)

bench_dir = os.path.dirname(__file__)

suites = [
    ("1. Sütun 3 Gerçek TESS MAST Arşivi Doğrulama", os.path.join(bench_dir, "run_mast_validation.py")),
    ("2. İleri Düzey Çoklu Rejim ve Yaşanabilir Bölge Denetimi", os.path.join(bench_dir, "run_advanced_copilot_audit.py")),
    ("3. Sıfır Ön Bilgili Çoklu Gezegen Peeling Keşfi", os.path.join(bench_dir, "run_microsecond_multiplanet_mission.py"))
]

for title, script_path in suites:
    print(f"\n>>> BAŞLATILIYOR: {title}...")
    res = subprocess.run([sys.executable, script_path], capture_output=False)
    if res.returncode != 0:
        print(f"[!] HATA: {title} süiti başarısız oldu (Kod: {res.returncode})")
        sys.exit(res.returncode)

print("\n" + "="*95)
print("  [RESMİ BİLİMSEL TESCİL]: TÜM SÜİTLER EKSİKSİZ VE SIFIR HATAYLA YEŞİL YANMIŞTIR!")
print("="*95)
