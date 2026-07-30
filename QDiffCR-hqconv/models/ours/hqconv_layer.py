"""
HQConv Quanvolutional Layer for QDiffCR.

Implements NesyaLab's HQConv_ansatz (the best-performing variant from
"Quantum-Hybrid-Diffusion-Models-for-EO") adapted for the DiffCR architecture.

Placement: encoder stage 2 (16×16 spatial at 64×64 input resolution)
Qubits: 8 (2×2 patch × 2 channels)
Ansatz: HQConv (block_A intra-group CRZ+CRX + block_B inter-group CRZ+CRX)
Layers: 3
Params: 28 per layer × 3 = 84 trainable quantum parameters

Reference: https://github.com/NesyaLab/Quantum-Hybrid-Diffusion-Models-for-EO
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


class HQConvQuanvLayer(nn.Module):
    """Quanvolutional layer using the HQConv ansatz on 2×2 spatial patches.

    Projects input to quantum_channels (2 channels), extracts 2×2 patches
    (4 pixels × 2 channels = 8 qubits), runs each patch through HQConv_ansatz,
    folds back, projects to original channels, adds as residual.

    For a 16×16 feature map with stride 2:
      - 64 patches, quantum_channels/2 groups = 64 circuit calls per sample
    """

    def __init__(self, channels, n_qubits=8, n_layers=3, quantum_channels=2):
        super().__init__()
        assert n_qubits == 8, "HQConv ansatz requires exactly 8 qubits (2×2 patch × 2 channels)"
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.quantum_channels = quantum_channels  # must be 2 (for 8 qubits = 4 pixels × 2 ch)

        # Project input channels to quantum_channels (2)
        self.pre_conv = nn.Conv2d(channels, quantum_channels, kernel_size=1)
        # Project back to original channels
        self.post_conv = nn.Conv2d(quantum_channels, channels, kernel_size=1)

        # HQConv circuit: 8 qubits, 28 params per layer
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # Data encoding: AngleEmbedding on all 8 qubits
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")

            # HQConv ansatz: n_layers repetitions
            for layer in range(n_layers):
                layer_w = weights[layer]  # [28] params for this layer

                # block_A on qubits 0-3 (first group of 4)
                # 4 pairs of CRZ + CRX in ring pattern
                for i in range(3, -1, -1):
                    idx = (3 - i) * 2
                    qml.CRZ(2 * np.pi * layer_w[idx], wires=[(i + 1) % 4, i])
                    qml.CRX(2 * np.pi * layer_w[idx + 1], wires=[(i + 1) % 4, i])

                # block_A on qubits 4-7 (second group of 4)
                for i in range(3, -1, -1):
                    idx = 8 + (3 - i) * 2
                    qml.CRZ(2 * np.pi * layer_w[idx], wires=[4 + (i + 1) % 4, 4 + i])
                    qml.CRX(2 * np.pi * layer_w[idx + 1], wires=[4 + (i + 1) % 4, 4 + i])

                # block_B: connect group 0 to group 1 (inter-group entanglement)
                # CRZ + CRX from qubit 0 to qubit 4
                qml.CRZ(2 * np.pi * layer_w[16], wires=[0, 4])
                qml.CRX(2 * np.pi * layer_w[17], wires=[0, 4])

                # block_B: connect group 1 to group 0
                # CRZ + CRX from qubit 4 to qubit 0
                qml.CRZ(2 * np.pi * layer_w[18], wires=[4, 0])
                qml.CRX(2 * np.pi * layer_w[19], wires=[4, 0])

                # Additional inter-group connections for richer entanglement
                # (following NesyaLab's pattern of connecting across groups)
                qml.CRZ(2 * np.pi * layer_w[20], wires=[1, 5])
                qml.CRX(2 * np.pi * layer_w[21], wires=[1, 5])
                qml.CRZ(2 * np.pi * layer_w[22], wires=[2, 6])
                qml.CRX(2 * np.pi * layer_w[23], wires=[2, 6])
                qml.CRZ(2 * np.pi * layer_w[24], wires=[3, 7])
                qml.CRX(2 * np.pi * layer_w[25], wires=[3, 7])

                # Final two params for reverse inter-group
                qml.CRZ(2 * np.pi * layer_w[26], wires=[5, 1])
                qml.CRX(2 * np.pi * layer_w[27], wires=[7, 3])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # Weight shape: (n_layers, 28) — 28 params per HQConv layer
        weight_shapes = {"weights": (n_layers, 28)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        residual = x
        device = x.device
        B, C, H, W = x.shape

        # Project to 2 channels: [B, C, H, W] -> [B, 2, H, W]
        z = self.pre_conv(x)

        # Extract 2×2 patches stride 2: [B, 2*4, n_patches] = [B, 8, n_patches]
        patches = F.unfold(z, kernel_size=2, stride=2)
        n_patches = patches.shape[2]

        # Reshape to [B * n_patches, 8] — each row is one 2×2 patch across 2 channels
        patches = patches.permute(0, 2, 1).reshape(-1, 8)

        # Scale and run quantum circuit
        patches = torch.tanh(patches) * torch.pi
        patches = patches.cpu()
        out = self.quantum_layer(patches)
        out = out.to(device)

        # Fold back: [B*n_patches, 8] -> [B, 2, H, W]
        out = out.view(B, n_patches, 8).permute(0, 2, 1)  # [B, 8, n_patches]
        out = F.fold(out, output_size=(H, W), kernel_size=2, stride=2)

        # Project back to full channel count
        out = self.post_conv(out)
        return residual + out
