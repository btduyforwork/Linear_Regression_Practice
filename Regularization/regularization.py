import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# -----------------------------
# 1. Tạo dữ liệu cho phần LOSS SURFACE
# -----------------------------
np.random.seed(42)

n = 100
x1 = np.random.uniform(-2, 2, n)
x2 = 9 * x1   # phụ thuộc tuyến tính hoàn hảo

true_alpha = 4.0
y = true_alpha * x1 + np.random.normal(0, 0.3, n)

# -----------------------------
# 2. Hệ số regularization
# -----------------------------
lambda_reg = 1.0   # bạn có thể thử 0.1, 1.0, 5.0

# -----------------------------
# 3. Tạo lưới w1, w2
# -----------------------------
w1_vals = np.linspace(-100, 40, 320)
w2_vals = np.linspace(-10, 10, 220)
W1, W2 = np.meshgrid(w1_vals, w2_vals)

Loss = np.zeros_like(W1)

# -----------------------------
# 4. Tính Loss = MSE + lambda*(w1^2 + w2^2)
# -----------------------------
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w1 = W1[i, j]
        w2 = W2[i, j]
        y_hat = w1 * x1 + w2 * x2
        mse = np.mean((y - y_hat) ** 2)
        Loss[i, j] = mse

# -----------------------------
# 5. Chọn vị trí cắt lớp
# -----------------------------
w2_fixed = 6
w2_idx = np.argmin(np.abs(w2_vals - w2_fixed))

w1_fixed = 4.0
w1_idx = np.argmin(np.abs(w1_vals - w1_fixed))

# -----------------------------
# 6. Lấy 2 lát cắt từ ma trận Loss
# -----------------------------
loss_vs_w1 = Loss[w2_idx, :]
loss_vs_w2 = Loss[:, w1_idx]

# -----------------------------
# 7. Đồng bộ màu giữa surface và contour
# -----------------------------
cmap_name = "viridis"
vmin = Loss.min()
vmax = Loss.max()
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# -----------------------------
# 8. Tính minimum lý thuyết theo w1 khi giữ w2 cố định
#    Loss(w1) = MSE + lambda*w1^2 + const
# -----------------------------
A = np.mean(x1**2)
B = np.mean(x1 * y)

# Vì x2 = 9x1 nên:
# y_hat = (w1 + 9w2)*x1
# MSE = mean((y - (w1 + 9w2)x1)^2)
# d/dw1 [MSE + lambda*w1^2] = 0
w1_star_reg = (2 * B - 18 * A * w2_fixed) / (2 * A + 2 * lambda_reg)

# -----------------------------
# 9. Vẽ surface + contour + 2 slices
# -----------------------------
fig = plt.figure(figsize=(15, 10))

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
surf = ax1.plot_surface(
    W1, W2, Loss,
    cmap=cmap_name,
    norm=norm,
    linewidth=0,
    antialiased=True
)
ax1.set_title(f"Loss surface with L2 regularization (lambda={lambda_reg})")
ax1.set_xlabel("w1")
ax1.set_ylabel("w2")
ax1.set_zlabel("Loss")
ax1.view_init(elev=22, azim=-35)
fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.1, label="Loss")

ax2 = fig.add_subplot(2, 2, 2)
cont = ax2.contourf(
    W1, W2, Loss,
    levels=50,
    cmap=cmap_name,
    norm=norm
)
fig.colorbar(cont, ax=ax2, label="Loss")
ax2.axhline(w2_vals[w2_idx], color="red", linestyle="--", label=f"w2 = {w2_vals[w2_idx]:.2f}")
ax2.axvline(w1_vals[w1_idx], color="white", linestyle="--", label=f"w1 = {w1_vals[w1_idx]:.2f}")
ax2.set_title("Contour with slices")
ax2.set_xlabel("w1")
ax2.set_ylabel("w2")
ax2.legend()

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(w1_vals, loss_vs_w1, linewidth=2)
ax3.set_title(f"Slice: Loss vs w1 (w2 = {w2_vals[w2_idx]:.2f})")
ax3.set_xlabel("w1")
ax3.set_ylabel("Loss")
ax3.grid(alpha=0.25)

# đánh dấu minimum lý thuyết có regularization
ax3.axvline(
    w1_star_reg,
    color="red",
    linestyle="--",
    label=f"regularized min near w1={w1_star_reg:.2f}"
)
ax3.legend()

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(w2_vals, loss_vs_w2, linewidth=2)
ax4.set_title(f"Slice: Loss vs w2 (w1 = {w1_vals[w1_idx]:.2f})")
ax4.set_xlabel("w2")
ax4.set_ylabel("Loss")
ax4.grid(alpha=0.25)

plt.tight_layout()
plt.show()

# ============================================================
# 10. Hình minh họa scatter 3D + các plane
#     Lưu ý: phần này là hình dữ liệu và plane, không trực tiếp "vẽ regularization"
# ============================================================

x2_vis = 9 * x1 + np.random.normal(0, 0.8, n)
y_vis = true_alpha * x1 + np.random.normal(0, 0.45, n)

fig2 = plt.figure(figsize=(11, 8))
ax = fig2.add_subplot(111, projection='3d')

ax.scatter(x1, x2_vis, y_vis, s=28, alpha=0.85, label="Observed data (slightly dispersed)")

# Grid để vẽ plane
x1_plane = np.linspace(x1.min() - 0.3, x1.max() + 0.3, 25)
x2_plane = np.linspace(x2_vis.min() - 1.5, x2_vis.max() + 1.5, 25)
X1_plane, X2_plane = np.meshgrid(x1_plane, x2_plane)

# Một vài mặt phẳng minh họa
plane_params = [
    (4.0, 0.0),
    (-5.0, 1.0),
    (13.0, -1.0),
    (22.0, -2.0),
    (-14.0, 2.0),
]

for (w1_p, w2_p) in plane_params:
    Y_plane = w1_p * X1_plane + w2_p * X2_plane
    ax.plot_surface(
        X1_plane,
        X2_plane,
        Y_plane,
        alpha=0.10,
        linewidth=0,
        antialiased=True
    )

# Đường fit hiệu dụng màu đỏ
x1_line = np.linspace(x1.min(), x1.max(), 300)
x2_line = 9 * x1_line
y_line = true_alpha * x1_line

ax.plot(
    x1_line,
    x2_line,
    y_line,
    color="red",
    linewidth=3.5,
    label="Effective 2D fitted line"
)

sort_idx = np.argsort(x1)
ax.plot(
    x1[sort_idx],
    x2_vis[sort_idx],
    y_vis[sort_idx],
    color="black",
    linewidth=1.0,
    alpha=0.55,
    label="Noisy data trajectory"
)

ax.set_title("Many planes + slightly dispersed data")
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.view_init(elev=22, azim=-55)
ax.legend()

plt.tight_layout()
plt.show()