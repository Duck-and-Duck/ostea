================================================================================
OSTE-MoE: HIERARCHICAL MULTI-MODAL MIXTURE-OF-EXPERTS EXOPLANET DISCOVERY ENGINE
================================================================================
Platform: NVIDIA RTX CUDA Core and Tensor Core Accelerators
Architecture: Hierarchical Multi-Modal Mixture of Experts (MoE)
License: MIT Open Source License
Release Version: 1.5.2 SOTA Production

GENEL TANIM
OSTE-MoE, NASA TESS, Kepler ve gelecekteki ESA PLATO uzay teleskobu verilerini
mikrosaniye hizlarinda taramak, otegezegen gecislerini dogrulamak ve atmosferik
kimyasal bilesenleri cozmek uzere tasarlanmis otonom bir astrofizik yapay zekasidir.
Sistem bilim insanlarinin yerini almak icin degil, onlarin aylar suren veri ayiklama
ve hesaplama yukunu saniyelere indiren bir bilimsel hizlandirici olarak calisir.

TEMEL BASARI METRIKLERI
1. Donanim Cikarim Hizi: Aday basina 3.11 ile 6.14 mikrosaniye (saniyede 160.000+ hedef)
2. Cekirdek Bilesik Analiz Gecikmesi: 33.50 mikrosaniye
3. NASA Kepler DR25 ve SPOC 400 Hedefli Test: Yuzde 96.50 Dogruluk, 0.9794 ROC-AUC
4. Cok Sektorlu 3 Yillik Evren Testi (16.100 Sistem): Yuzde 98.48 Dogruluk
5. Katman 1.5 Astrometrik BEB Eleme Orani: Yuzde 100.00 (Sifir sahte pozitif sizintisi)
6. TESS Sektor Tarama Kapasitesi: 1.000 yildiz 34.93 saniyede taranir

DORT KATMANLI MOE MIMARISI
Katman 1 (Causal Anomaly Gate):
Isik egrisinde gecis barindirmayan sakin yildizlarin yuzde 80-95 ini 0.01 milisaniyede
eleyerek derin katmanlari gereksiz yere mesgul etmez.

Katman 1.5 (Astrometric PRF Centroid Expert):
TESS in 21 yay-saniyelik genis piksellerindeki foton agirlik merkezi kaymasini 2D fark
goruntuleme yontemiyle olcer. Hedef disindaki arka plan cift yildizlarini tam olarak eler.

Katman 2 (GPU Prefix-Sum Binning and Detrending):
On binlerce fotometri noktasini GPU uzerinde mikrosaniyede 201 global ve 61 lokal
kutuya katlar. Bos kutulari komsu foton akisiyla enterpole eder.

Katman 3 (AstroNet-HQ 1D-CNN Vetter):
Kuadratik kenar kararmali U-gecis morfolojisini tanir. V-sekilli ikili yildizlari
ve ikincil tutulmalari eler.

Katman 3.5 (Fast GPU Analytic Solver):
Mandel-Agol ve Seager temas noktalarini GPU uzerinde kapali form analitik denklemlerle
cozer. Yaricap, yorunge egikligi, darbe parametresi ve kutleyi mikrosaniyede hesaplar.

Katman 4 (Micro Atmosphere Engine and SBI Inversion):
Cosmic Shoreline kriteri ile gezegenin kacis hizi ve yildiz akisini karsilastirarak
atmosferin var olup olmadigini tespit eder. Su, karbondioksit ve metan bolluklarini
saniyeler icinde cozer.

KURULUM VE KULLANIM
Gereksinimler:
Python 3.10 veya uzeri
PyTorch 2.0 veya uzeri
NumPy, Astropy, SciPy, Lightkurve

Dogrulama Komutlari:
Sistem Testi: python benchmarks/verify_system.py
Master Test Suiti: python benchmarks/run_master_validation_suite.py
Sektor Tarama Demosu: python benchmarks/run_sector_scan_demo.py
Komut Satiri Arayuzu: python cli.py --demo

BILIMSEL KAYNAKLAR
Twicken et al. 2018, PASP, 130, 064502 (NASA SPOC Data Validation)
Bryson et al. 2013, PASP, 125, 889 (Kepler Difference Imaging Centroiding)
Shallue and Vanderburg 2018, AJ, 155, 94 (AstroNet Deep Learning)
Valizadegan et al. 2022, ApJ, 926, 120 (NASA ExoMiner)
Kovacs, Zucker and Mazeh 2002, A and A, 391, 369 (Box-Fitting Algorithm BLS)
Zahnle and Catling 2017, ApJ, 843, 122 (The Cosmic Shoreline)
Kempton et al. 2018, PASP, 130, 114401 (TESS Follow-up TSM Framework)
================================================================================
