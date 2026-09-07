import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.model import AstroNetHQ

device = "cuda" if torch.cuda.is_available() else "cpu"
print("="*85)
print("  OSTE-MoE SÜTUN 4: C++ LIBTORCH VE TENSORRT / ONNX DERLEME HATTI (FIXED)")
print(f"--> PLATFORM: {device.upper()} | ONNX UYUMLU FRAKSİYONEL HAVUZLAMA")
print("="*85)

weights_in = os.path.join(os.path.dirname(__file__), "..", "weights", "astronet_hq.pt")
weights_dir = os.path.join(os.path.dirname(__file__), "..", "weights")

base_model = AstroNetHQ().to(device)
if os.path.exists(weights_in):
    base_model.load_state_dict(torch.load(weights_in, map_location=device))
base_model.eval()

dummy_g = torch.randn(1, 1, 201, device=device)
dummy_l = torch.randn(1, 1, 61, device=device)

# 1. C++ LibTorch Traced Export (.pt)
traced_model = torch.jit.trace(base_model, (dummy_g, dummy_l))
frozen_traced = torch.jit.freeze(traced_model)
libtorch_path = os.path.join(weights_dir, "astronet_hq_libtorch.pt")
frozen_traced.save(libtorch_path)
print(f"--> [✓ C++ LIBTORCH]: Model C++ için donduruldu -> {libtorch_path}")

# 2. ONNX Uyumlu Sarmalayıcı (ONNX 1D Adaptive Pool Desteği)
class AstroNetHQ_ONNX(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.orig = original_model

    def forward(self, g, l):
        # Global kol (Pool öncesine kadar)
        x_g = self.orig.global_conv[:8](g) # [B, 32, 50]
        # ONNX'in desteklediği 2D Adaptive Pool formatına genişlet
        x_g_2d = x_g.unsqueeze(-1) # [B, 32, 50, 1]
        x_g_pool = F.adaptive_avg_pool2d(x_g_2d, (15, 1)).squeeze(-1) # [B, 32, 15]
        feat_g = x_g_pool.flatten(1) # [B, 480]

        # Local kol (Pool öncesine kadar)
        x_l = self.orig.local_conv[:10](l) # [B, 32, 30]
        x_l_2d = x_l.unsqueeze(-1) # [B, 32, 30, 1]
        x_l_pool = F.adaptive_avg_pool2d(x_l_2d, (15, 1)).squeeze(-1) # [B, 32, 15]
        feat_l = x_l_pool.flatten(1) # [B, 480]

        feat_cat = torch.cat([feat_g, feat_l], dim=1) # [B, 960]
        return self.orig.fc(feat_cat)

onnx_model = AstroNetHQ_ONNX(base_model).to(device).eval()

# Çıktı tutarlılığı kontrolü
with torch.no_grad():
    y_orig = base_model(dummy_g, dummy_l)
    y_onnx = onnx_model(dummy_g, dummy_l)
    diff = torch.max(torch.abs(y_orig - y_onnx)).item()

if diff < 1e-5:
    print(f"--> [✓ MODEL TUTARLILIĞI]: Orijinal ve ONNX sarmalayıcı farkı = {diff:.2e} (Tam Uyumlu)")

onnx_path = os.path.join(weights_dir, "astronet_hq.onnx")
torch.onnx.export(
    onnx_model,
    (dummy_g, dummy_l),
    onnx_path,
    input_names=["global_view", "local_view"],
    output_names=["transit_probability"],
    dynamic_axes={
        "global_view": {0: "batch_size"},
        "local_view": {0: "batch_size"},
        "transit_probability": {0: "batch_size"}
    },
    opset_version=14,
    do_constant_folding=True
)
print(f"--> [✓ NVIDIA TENSORRT ONNX]: ONNX grafiği başarıyla üretildi -> {onnx_path}")
print("="*85)
