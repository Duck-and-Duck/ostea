================================================================================
OSTE-MoE MIMARI RAPORU VE TEKNIK BILLESENLER
================================================================================

1. SISTEMIN TEMEL AMACI
Bu yazilim, uzay teleskoplarindan gelen buyuk veri akisini donanimsal tensor
cekirdekleri uzerinde mikrosaniye hizlarinda isleyerek bilim insanlarina yuksek
guvenilirlikli otegezegen aday listeleri sunmak uzere insa edilmistir.

2. KATMANLARIN CALISMA PRENSIBI
Girdi Verisi: 1D Isik Egrisi dizisi ve 2D 11x11 Hedef Piksel Dosyasi (TPF).

Katman 1 - Anomaly Gate:
Isik egrisindeki 11 noktali kutu filtresi konvolusyonu ileGurultu seviyesini
olcer. En derin sinyal 4.0 sigma altinda ise analiz sonlandirilir.

Katman 1.5 - Astrometrik Centroid:
Gecis-ici ve gecis-disi piksellerin farki alinir. Gauss PRF agirlik maskesi
kullanilarak foton agirlik merkezi hesaplanir. Hedef merkezinden sapma 7.5 yay-saniyesi
veya 3.0 sigma uzerinde ise sistem arka plan ikilisi olarak etiketlenir.

Katman 2 - GPU Prefix-Sum Folding:
Zaman ve aki dizileri GPU uzerinde siralandiktan sonra katsayilar torch.cumsum
ile toplanir. Faz kutulamasi tek bir GPU kernelinda tamamlanir.

Katman 3 - AstroNet-HQ:
Evrisimli sinir agi mimarisi 16 ve 32 kanalli filtrelerle U-profilini dogrular.
Cikis tensöru ikili siniflandirma olasiligi uretir.

Katman 3.5 - Analitik Cozuculer:
Gezegen yaricapi, kutlesi, yorunge ekseni ve egikligi analitik bagintilarla
hesaplanir. Hata yayilimi analitik Jacobian matrisi ile yapilir.

Katman 4 - Atmosfer Motoru:
Karasal gezegenlerde yercekimi kacis hizi ile yildiz akisi karsilastirilir.
Gaz devlerinde dogrudan molekuler bolluk spektrumu cikarilir.

3. DONANIM GECIKME PROFILI
Katman 3 1D-CNN: 19.61 mikrosaniye
Katman 3.5 Analitik Cozucu: 9.57 mikrosaniye
Katman 4 Atmosfer Motoru: 4.32 mikrosaniye
Bilesik Cekirdek Gecikmesi: 33.50 mikrosaniye
================================================================================
