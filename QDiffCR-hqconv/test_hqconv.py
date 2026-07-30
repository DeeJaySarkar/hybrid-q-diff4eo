"""
Smoke test for the HQConv quanvolutional variant.
Verifies forward/backward pass and gradient flow through the 8-qubit HQConv circuit.
"""
import sys
sys.path.insert(0, '.')
import torch

def test_hqconv_layer():
    print("=" * 60)
    print("TEST 1: HQConvQuanvLayer standalone")
    print("=" * 60)
    from models.ours.hqconv_layer import HQConvQuanvLayer
    layer = HQConvQuanvLayer(channels=128, n_qubits=8, n_layers=3, quantum_channels=2)
    x = torch.randn(1, 128, 16, 16, requires_grad=True)
    out = layer(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape}"
    loss = out.sum(); loss.backward()
    q_grad = list(layer.quantum_layer.parameters())[0].grad is not None
    print(f"  Input:  {tuple(x.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Circuit params: {sum(p.numel() for p in layer.quantum_layer.parameters())} (expect 84)")
    print(f"  Quantum grad: {q_grad}")
    assert q_grad, "No gradient on quantum params!"
    print("  PASSED\n")

def test_unet():
    print("=" * 60)
    print("TEST 2: Full UNet with HQConv at encoder stage 2")
    print("=" * 60)
    from models.ours.nafnet_double_encoder_splitcaCond_splitcaUnet_quantum import UNet
    net = UNet(img_channel=3, width=32, middle_blk_num=1,
               enc_blk_nums=[1,1,1,1], dec_blk_nums=[1,1,1,1],
               quantum={'enabled': True, 'mode': 'hqconv_encoder', 'encoder_stage': 2, 'n_layers': 3})
    inp = torch.randn(1, 12, 64, 64)
    out = net(inp, torch.rand(1))
    assert out.shape == (1, 3, 64, 64)
    out.sum().backward()
    total = sum(p.numel() for p in net.parameters())
    hqconv = sum(p.numel() for p in net.hqconv_layer.parameters())
    print(f"  Input:  {tuple(inp.shape)}")
    print(f"  Output: {tuple(out.shape)}")
    print(f"  Total params: {total:,}")
    print(f"  HQConv module: {hqconv} (circuit: 84)")
    print(f"  Placement: encoder stage {net.hqconv_stage} (16×16 spatial at 64×64 input)")
    print(f"  Bottleneck quantum: {net.quantum_bottleneck}")
    print("  PASSED\n")

if __name__ == "__main__":
    print("\nQDiffCR-HQConv Pipeline Test")
    print("8-qubit HQConv ansatz, encoder stage 2, default.qubit + backprop\n")
    test_hqconv_layer()
    test_unet()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
    print("\nTo train: python run.py -c config/ours_sigmoid_w32_hqconv.json -p train")
