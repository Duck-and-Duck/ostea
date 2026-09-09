import torch
import numpy as np

class GPUPeelingSearchEngine:
    """
    Sıfır Ön Bilgi ile Çoklu Gezegen Arama ve Sinyal Soyma Motoru (Peeling).
    Periyot bilinmez; GPU üzerinde arar, 1. gezegeni bulur, sinyalini maskeler
    ve sistemdeki diğer gezegenleri sırayla izole eder (Sıfır Gezegen Kaybı).
    """
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

    def search_all_planets(self, time_arr, flux_arr, max_planets=4, min_p=0.8, max_p=30.0):
        t_gpu = torch.tensor(time_arr, dtype=torch.float32, device=self.device)
        f_gpu = torch.tensor(flux_arr, dtype=torch.float32, device=self.device)
        dt = float(time_arr[1] - time_arr[0])
        
        discovered_signals = []
        residual_flux = f_gpu.clone()

        # Frekans arama ızgarası (GPU)
        periods = torch.exp(torch.linspace(np.log(min_p), np.log(max_p), 3000, device=self.device))

        for p_idx in range(max_planets):
            # 1. Hızlı Konvolüsyonel Transit Arama
            diff_f = residual_flux[1:] - residual_flux[:-1]
            sigma = (torch.median(torch.abs(diff_f - torch.median(diff_f))) * 1.4826) / 1.4142

            best_p, best_t0, best_dur, best_depth, max_snr = 0.0, 0.0, 0.0, 0.0, 0.0

            # Test harmonikleri
            for p_cand in periods[::15]: # Hızlı kaba ızgara
                dur_est = max(0.04, min(0.25, float(0.10 * (p_cand.item() / 5.0)**(1.0/3.0))))
                phase = ((t_gpu + 0.5 * p_cand) % p_cand) - (0.5 * p_cand)
                
                # 40 kutu ile hızlı faz histogramı
                bin_edges = torch.linspace(-0.5 * p_cand, 0.5 * p_cand, 41, device=self.device)
                bin_idx = torch.bucketize(phase, bin_edges) - 1
                bin_idx = torch.clamp(bin_idx, 0, 39)
                
                sums = torch.zeros(40, device=self.device).scatter_add_(0, bin_idx, residual_flux)
                counts = torch.zeros(40, device=self.device).scatter_add_(0, bin_idx, torch.ones_like(residual_flux))
                counts_clean = torch.clamp(counts, min=1.0)
                profile = sums / counts_clean
                
                min_val, min_loc = torch.min(profile, dim=0)
                med_val = torch.median(profile)
                depth_cand = float((med_val - min_val).item())
                
                if depth_cand > 0.00025:
                    snr = (depth_cand / (sigma.item() + 1e-8)) * np.sqrt(max(1.0, (time_arr[-1]-time_arr[0])/p_cand.item()))
                    if snr > max_snr:
                        max_snr = snr
                        best_p = float(p_cand.item())
                        t0_offset = float(bin_edges[min_loc].item())
                        best_t0 = float((time_arr[0] - t0_offset) % best_p)
                        best_dur = dur_est
                        best_depth = depth_cand

            # Eğer tespit edilen sinyal 5 sigma üzerinde geçerli bir gezegense kaydet
            if max_snr >= 5.0 and best_depth < 0.035:
                discovered_signals.append({
                    "period": best_p,
                    "t0": best_t0,
                    "duration": best_dur,
                    "depth": best_depth,
                    "snr": max_snr
                })

                # 2. SİNYAL SOYMA (PEELING): Bulunan gezegenin transitlerini maskele
                ph_peel = ((t_gpu - best_t0 + 0.5 * best_p) % best_p) - (0.5 * best_p)
                in_tr_mask = torch.abs(ph_peel) < (best_dur / 1.8)
                residual_flux[in_tr_mask] = 1.0 # Transit çukuru düzeltilir
            else:
                break # Başka belirgin sinyal kalmadı

        return discovered_signals
