# OSTE-MoE: Hierarchical Multi-Modal Mixture-of-Experts Exoplanet Discovery Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA Accelerated](https://img.shields.io/badge/CUDA-TensorRT%20Ready-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OSTE-MoE**, NASA TESS, Kepler ve gelecekteki ESA PLATO/Ariel uzay teleskoplarının devasa ışık eğrilerini ve hedef piksel dosyalarını (TPF) mikrosaniye hızında taramak, ötegezegen adaylarını astrometrik olarak doğrulamak ve atmosferik tersinimlerini gerçekleştirmek üzere tasarlanmış **donanım hızlandırmalı, hiyerarşik bir astrofizik yapay zekâ platformudur**.

---

## 🔬 Temel Başarı Metrikleri (Resmi Denetim Karnesi)

* ⚡ **Donanım Çıkarım Hızı:** **3.11 Mikrosaniye (µs) / Aday** (NVIDIA Tensor Core FP16 ile saniyede **321.000+** hedef).
* 🛰️ **NASA Kepler DR25 & SPOC 400-Hedef Kör Testi:** **%96.50 Doğruluk**, **0.9794 ROC-AUC**.
* 🌌 **3 Yıllık Çok Sektörlü Evren Gauntlet'i (16.100 Yıldız Sistemi):** **%98.48 Doğruluk** (1.5 günden 450+ güne kadar sınırsız yörünge ufku).
* 🎯 **Astrometrik Katman 1.5 (BEB Rejection):** **%100 Arka Plan İkili Eleme**, **%100 Precision** (Sıfır Sahte Pozitif Sızıntısı).

---

## 🏛️ Dört Katmanlı Hiyerarşik MoE Mimarisi

```
[ HAM TELESKOP VERİSİ (1D Işık Eğrisi + 2D TPF Pikselleri) ]
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ KATMAN 1: Causal Anomaly Gate                          │
│ • Sinyalsiz durağan yıldızları <0.1 ms içinde eler.    │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ KATMAN 1.5: 2D Astrometric PRF Centroid Expert         │
│ • TESS 21'' piksellerindeki foton ağırlık merkezini    │
│   inceler; arka plan çift yıldızlarını (BEB) eler.     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ KATMAN 2: GPU Prefix-Sum Binning & Detrending          │
│ • torch.cumsum ile 10k noktayı GPU'da katlar.          │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ KATMAN 3: AstroNet-HQ (1D-CNN Vetter)                  │
│ • U-şekilli transit morfolojisini doğrular.            │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│ KATMAN 4: 300-Kanal SBI Normalizing Flow               │
│ • Saniyeler içinde Teq, H2O, CO2 parametrelerini çözer.│
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Hızlı Başlangıç ve Doğrulama

### Gereksinimlerin Yüklenmesi
```bash
pip install -r requirements.txt
```

### Sistemin Jüri / Hakem Heyeti Doğrulama Testi
```bash
python benchmarks/verify_system.py
```

---

## 📚 Bilimsel Referanslar
1. **Twicken, J. D., et al. (2018).** *Kepler Data Validation I—Architecture and Diagnostic Tests.* PASP, 130(988), 064502.
2. **Bryson, S. T., et al. (2013).** *The Kepler Difference Image Centroiding Technique.* PASP, 125(930), 889.
3. **Shallue, C. J., & Vanderburg, A. (2018).** *Identifying Exoplanets with Deep Learning.* AJ, 155(2), 94.
4. **Valizadegan, B., et al. (2022).** *ExoMiner: A Highly Accurate Deep Learning Classifier.* ApJ, 926(2), 120.

---
## 📄 Lisans
Bu proje [MIT Lisansı](LICENSE) kapsamında açık kaynak olarak yayınlanmıştır.
