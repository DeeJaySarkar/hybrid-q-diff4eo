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

        dev = qml.device("lightning.gpu", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="adjoint")
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
