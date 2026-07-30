import torch
import torch.nn as nn
import pennylane as qml


class QuantumBottleneckLayer(nn.Module):
    """Quantum bottleneck layer that acts as a global feature modulator.

    Pools spatial features -> projects to n_qubits dims -> quantum circuit -> projects back.
    Added as a residual to the input tensor.
    """

    def __init__(self, channels, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pre_linear = nn.Linear(channels, n_qubits)
        self.post_linear = nn.Linear(n_qubits, channels)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        residual = x
        device = x.device
        z = self.pool(x).flatten(1)
        z = self.pre_linear(z)
        z = torch.tanh(z) * torch.pi
        z = z.cpu()
        z = self.quantum_layer(z)
        z = z.to(device)
        z = self.post_linear(z)
        return residual + z.unsqueeze(-1).unsqueeze(-1)


class ClassicalBottleneckLayer(nn.Module):
    """Parameter-matched CLASSICAL control for QuantumBottleneckLayer.

    Identical structure (global pool -> pre_linear -> [block] -> post_linear -> residual),
    but the quantum circuit is replaced by a small classical MLP with >= as many
    parameters as the quantum circuit. Isolates whether the quantum circuit itself
    contributes, vs. just adding trainable parameters at the bottleneck.
    """

    def __init__(self, channels, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pre_linear = nn.Linear(channels, n_qubits)
        self.post_linear = nn.Linear(n_qubits, channels)

        hidden = max(n_qubits, n_layers * n_qubits)
        self.block = nn.Sequential(
            nn.Linear(n_qubits, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_qubits),
            nn.Tanh(),
        )

    def forward(self, x):
        residual = x
        z = self.pool(x).flatten(1)
        z = self.pre_linear(z)
        z = torch.tanh(z) * torch.pi  # identical input scaling to the quantum variant
        z = self.block(z)
        z = self.post_linear(z)
        return residual + z.unsqueeze(-1).unsqueeze(-1)


class QuantumMatchedLayer(nn.Module):
    """Parameter-matched quantum bottleneck using StronglyEntanglingLayers.

    Same pool -> project -> circuit -> project -> residual structure, but with:
      - 5 qubits (Hilbert space 2^5 = 32)
      - 5 layers of StronglyEntanglingLayers (3 rotations per qubit + all-to-all CNOT)
      - 75 trainable quantum params (vs control's 76 classical params)

    This directly tests whether the quantum circuit provides an advantage over the
    classical control at matched parameter count.
    """

    def __init__(self, channels, n_qubits=5, n_layers=5):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.pre_linear = nn.Linear(channels, n_qubits)
        self.post_linear = nn.Linear(n_qubits, channels)

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        # StronglyEntanglingLayers weight shape: (n_layers, n_qubits, 3)
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.quantum_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        residual = x
        device = x.device
        z = self.pool(x).flatten(1)
        z = self.pre_linear(z)
        z = torch.tanh(z) * torch.pi
        z = z.cpu()
        z = self.quantum_layer(z)
        z = z.to(device)
        z = self.post_linear(z)
        return residual + z.unsqueeze(-1).unsqueeze(-1)
