"""
Smoke test for quantum-enhanced DiffCR pipeline.
Verifies:
1. QuantumBottleneckLayer forward/backward pass
2. Quantum UNet forward/backward pass
3. Gradient flow through quantum circuit
"""
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn


def test_quantum_layer():
    print("=" * 60)
    print("TEST 1: QuantumBottleneckLayer")
    print("=" * 60)
    from models.ours.quantum_layer import QuantumBottleneckLayer

    layer = QuantumBottleneckLayer(channels=512, n_qubits=4, n_layers=2)
    x = torch.randn(2, 512, 16, 16, requires_grad=True)
    out = layer(x)

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")

    # Check gradients flow
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "No gradient on input!"
    
    # Check quantum layer params have gradients
    q_params_with_grad = sum(1 for p in layer.quantum_layer.parameters() if p.grad is not None)
    total_q_params = sum(1 for _ in layer.quantum_layer.parameters())
    print(f"  Quantum params with gradients: {q_params_with_grad}/{total_q_params}")
    assert q_params_with_grad > 0, "No gradients on quantum parameters!"

    print("  PASSED\n")


def test_quantum_unet():
    print("=" * 60)
    print("TEST 2: Quantum UNet forward/backward")
    print("=" * 60)
    from models.ours.nafnet_double_encoder_splitcaCond_splitcaUnet_quantum import UNet

    net = UNet(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
        quantum={"enabled": True, "n_qubits": 4, "n_layers": 2},
    )

    # Input: 3 cond images (3ch each) + 1 noisy image (3ch) = 12ch
    inp = torch.randn(1, 12, 256, 256)
    gammas = torch.rand(1)

    out = net(inp, gammas)
    assert out.shape == (1, 3, 256, 256), f"Output shape: {out.shape}"
    print(f"  Input shape:  {inp.shape}")
    print(f"  Output shape: {out.shape}")

    # Backward pass
    loss = out.sum()
    loss.backward()

    # Count total params and quantum-specific params
    total_params = sum(p.numel() for p in net.parameters())
    quantum_params = sum(p.numel() for p in net.quantum_bottleneck.parameters())
    print(f"  Total params:   {total_params:,}")
    print(f"  Quantum module: {quantum_params:,} (linear + circuit)")
    print(f"  Overhead:       {quantum_params/total_params*100:.2f}%")
    print("  PASSED\n")


def test_quantum_disabled():
    print("=" * 60)
    print("TEST 3: Quantum disabled (classical fallback)")
    print("=" * 60)
    from models.ours.nafnet_double_encoder_splitcaCond_splitcaUnet_quantum import UNet

    net = UNet(
        img_channel=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
        quantum={"enabled": False},
    )

    inp = torch.randn(1, 12, 256, 256)
    gammas = torch.rand(1)
    out = net(inp, gammas)

    assert out.shape == (1, 3, 256, 256)
    assert net.quantum_bottleneck is None
    print(f"  quantum_bottleneck is None: True")
    print(f"  Output shape: {out.shape}")
    print("  PASSED\n")


if __name__ == "__main__":
    print("\nQuantum-Enhanced DiffCR Pipeline Test")
    print("PennyLane backend: lightning.gpu (CUDA-accelerated simulator)")
    print()

    test_quantum_layer()
    test_quantum_unet()
    test_quantum_disabled()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nTo train: python run.py -c config/ours_sigmoid_w32_quantum.json -p train")
