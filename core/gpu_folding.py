import torch
import torch.nn.functional as F

def gpu_fast_fold(t_tensor, f_tensor, p, t0, dur, device="cuda"):
    """
    Katman 2: GPU Prefix-Sum Kutu Katlama ve Faz Izgarası Motoru.
    torch.cumsum ile 10.000+ veri noktasını mikrosaniye içinde 201 global ve 61 yerel
    kutuya katlar. Boş kutuları en yakın komşu kontinüumu ile enterpole eder.
    """
    phase = ((t_tensor - t0 + 0.5 * p) % p) - (0.5 * p)
    s_idx = torch.argsort(phase)
    s_ph, s_fl = phase[s_idx], f_tensor[s_idx]

    # Global View (201 Bins)
    g_bins = torch.linspace(-0.5 * p, 0.5 * p, 201, device=device)
    half_w_g = 0.5 * (g_bins[1] - g_bins[0])
    idx_l = torch.searchsorted(s_ph, g_bins - half_w_g)
    idx_r = torch.searchsorted(s_ph, g_bins + half_w_g)
    f_cumsum = F.pad(torch.cumsum(s_fl, dim=0), (1, 0))
    counts_g = idx_r - idx_l
    mask_g = counts_g > 0
    g_raw = torch.ones_like(g_bins)
    g_raw[mask_g] = (f_cumsum[idx_r[mask_g]] - f_cumsum[idx_l[mask_g]]) / counts_g[mask_g].float()
    if not torch.all(mask_g):
        empty_idx = torch.clamp(torch.searchsorted(s_ph, g_bins[~mask_g]), 0, len(s_fl) - 1)
        g_raw[~mask_g] = s_fl[empty_idx]

    # Local View (61 Bins - dur*2 penceresi)
    win_l = dur * 2.0
    l_bins = torch.linspace(-win_l, win_l, 61, device=device)
    half_w_l = 0.5 * (l_bins[1] - l_bins[0])
    idx_ll = torch.searchsorted(s_ph, l_bins - half_w_l)
    idx_lr = torch.searchsorted(s_ph, l_bins + half_w_l)
    counts_l = idx_lr - idx_ll
    mask_l = counts_l > 0
    l_raw = torch.ones_like(l_bins)
    l_raw[mask_l] = (f_cumsum[idx_lr[mask_l]] - f_cumsum[idx_ll[mask_l]]) / counts_l[mask_l].float()
    if not torch.all(mask_l):
        empty_idx_l = torch.clamp(torch.searchsorted(s_ph, l_bins[~mask_l]), 0, len(s_fl) - 1)
        l_raw[~mask_l] = s_fl[empty_idx_l]

    g_norm = (g_raw - torch.median(g_raw)) / (torch.std(g_raw) + 1e-7)
    l_norm = (l_raw - torch.median(l_raw)) / (torch.std(l_raw) + 1e-7)
    depth = float(torch.median(l_raw[:15]).item() - torch.min(l_raw[24:37]).item())
    return g_norm.view(1, 1, 201), l_norm.view(1, 1, 61), depth
