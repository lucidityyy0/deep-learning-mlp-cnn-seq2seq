import numpy as np
import torch


def cross_correlate2d(X: np.ndarray, K: np.ndarray) -> np.ndarray:
    kH, kW = K.shape
    H, W = X.shape
    out = np.zeros((H - kH + 1, W - kW + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = (X[i:i + kH, j:j + kW] * K).sum()
    return out


def max_pool2d(X: np.ndarray, k: int = 2, stride: int = 2) -> np.ndarray:
    H, W = X.shape
    out = np.zeros((H // stride, W // stride))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = X[i * stride:i * stride + k, j * stride:j * stride + k].max()
    return out


def conv_output_size(W, P, K, S):
    return int((W + 2 * P - K) / S) + 1


def pool_output_size(W, K, S):
    return int((W - K) / S) + 1


def conv_params(K, C_in, C_out):
    return (K * K * C_in + 1) * C_out


def conv1x1_params(C_in, C_out):
    return (C_in + 1) * C_out


def demo_manual_conv():
    print("=== Manual Convolution Verification ===")
    np.random.seed(0)
    X = np.random.randn(6, 6)
    K = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]], dtype=float)

    manual_out = cross_correlate2d(X, K)

    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    K_t = torch.tensor(K, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    import torch.nn.functional as F
    torch_out = F.conv2d(X_t, K_t).squeeze().numpy()

    max_diff = np.abs(manual_out - torch_out).max()
    print(f"Input shape:   {X.shape}")
    print(f"Kernel shape:  {K.shape}")
    print(f"Output shape:  {manual_out.shape}  (expected {conv_output_size(6,0,3,1)}x{conv_output_size(6,0,3,1)})")
    print(f"Max diff vs PyTorch: {max_diff:.2e}  ({'PASS' if max_diff < 1e-5 else 'FAIL'})")

    pool_out = max_pool2d(manual_out, k=2, stride=2)
    print(f"After 2x2 MaxPool: {pool_out.shape}")

    print("\n=== Parameter Count Examples ===")
    print(f"Conv1 (1->6, k=5):  {conv_params(5, 1, 6):6d} params")
    print(f"Conv2 (6->16, k=5): {conv_params(5, 6, 16):6d} params")
    print(f"1x1 conv (6->16):   {conv1x1_params(6, 16):6d} params")

    print("\n=== LeNet-5 Output Size Walk-through ===")
    w = 28
    w = conv_output_size(w, 2, 5, 1); print(f"After Conv1(pad=2): {w}x{w}")
    w = pool_output_size(w, 2, 2); print(f"After Pool1:        {w}x{w}")
    w = conv_output_size(w, 0, 5, 1); print(f"After Conv2:        {w}x{w}")
    w = pool_output_size(w, 2, 2); print(f"After Pool2:        {w}x{w}")
    print(f"Flatten: {w}x{w}x16 = {w*w*16}")


if __name__ == "__main__":
    demo_manual_conv()
