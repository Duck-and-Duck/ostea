# ==============================================================================
#  OSTE-MoE: HIERARCHICAL MULTI-MODAL MIXTURE-OF-EXPERTS EXOPLANET DISCOVERY ENGINE
# ==============================================================================
# Version: 1.0.0 SOTA Enterprise | Platform: CUDA TensorRT Ready | License: MIT
# Developed for High-Cadence Optical Transit Photometry & Astrometric Vetting
# Target Missions: NASA TESS, Kepler, ESA PLATO, Ariel
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class AstroNetHQ(nn.Module):
    """
    Katman 3: 1D-CNN AstroNet-HQ Morfolojik Doğrulama Modeli.
    NASA SPOC ve Google AstroNet standardında kuadratik kenar kararmalı U-transit
    morfolojisini tanır, V-şekilli çift yıldızları ve ikincil tutulmaları eler.
    """
    def __init__(self):
        super(AstroNetHQ, self).__init__()
        self.global_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(15),
            nn.Flatten()
        )
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(15),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 15 + 32 * 15, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.25),
            nn.Linear(64, 1)
        )

    def forward(self, g, l):
        return self.fc(torch.cat([self.global_conv(g), self.local_conv(l)], dim=1))
