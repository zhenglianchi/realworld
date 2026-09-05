import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    """Standard sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def delta_sigmoid_full(x):
    """
    Complete mapping including negative x:
    Δ(x) = max(0, σ(10 max(0, x) - 5))
    Weight = 10.0, bias = -5.0
    x ∈ [-0.5, 1.2] → Δ ∈ [0, ≈1]
    Symmetric: Δ(0) ≈ 0, Δ(0.5) = 0.5, Δ(1) ≈ 1
    """
    x_pos = np.maximum(0, x)
    return np.maximum(0, sigmoid(10.0 * x_pos - 5.0))

def delta_sigmoid_no_truncate(x):
    """For comparison: without truncation, showing the full sigmoid curve."""
    return sigmoid(10.0 * x - 5.0)

# Create plot
x = np.linspace(-0.5, 1.2, 300)
y_full = delta_sigmoid_full(x)
y_raw = delta_sigmoid_no_truncate(x)

plt.figure(figsize=(9, 5), dpi=120)
plt.plot(x, y_full, 'b-', linewidth=2.5, label=r'$\Delta(x^+) = \max\left(0,\ \sigma(10 x^+ - 5)\right)$')
plt.plot(x, y_raw, 'r--', linewidth=1, alpha=0.6, label='without truncation')
plt.grid(True, alpha=0.3)
plt.xlabel('$x = d - \\tau$ (input)', fontsize=12)
plt.ylabel('$\\Delta$ (adjustment factor)', fontsize=12)
plt.title('Sigmoid Smooth Mapping: slope = 10.0, bias = -5.0 (symmetric)', fontsize=11)

# Mark key points
key_points = [(0, delta_sigmoid_full(0)), (0.5, delta_sigmoid_full(0.5)), (1.0, delta_sigmoid_full(1.0))]
for xp, dp in key_points:
    plt.scatter(xp, dp, color='blue', s=60, zorder=5)
    plt.text(xp + 0.02, dp + 0.01, f'$({xp:.1f}, {dp:.3f})$', fontsize=10)

plt.xlim(-0.5, 1.2)
plt.ylim(-0.05, 1.05)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('./sigmoid_mapping_plot.png', dpi=150, bbox_inches='tight')
plt.show()

# Print key values
print('Key values (with truncation, slope = 10.0, bias = -5.0):')
print(f'  x = -0.5: Δ = {delta_sigmoid_full(-0.5):.6f}')
print(f'  x =  0.0: Δ = {delta_sigmoid_full(0.0):.6f} (≈ 0)')
print(f'  x =  0.5: Δ = {delta_sigmoid_full(0.5):.6f} (= 0.5 exactly)')
print(f'  x =  1.0: Δ = {delta_sigmoid_full(1.0):.6f} (≈ 1)')
