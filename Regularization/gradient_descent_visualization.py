import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Tạo dữ liệu: x1 và x2 độc lập
# -----------------------------
np.random.seed(42)

n = 200
x1 = np.random.uniform(-2, 2, n)
x2 = np.random.uniform(-2, 2, n)   # độc lập với x1

# Ground truth
true_w1 = 2.5
true_w2 = -1.7
true_b = 0.8

noise = np.random.normal(0, 2.0, n)
y = true_w1 * x1 + true_w2 * x2 + true_b + noise

# -----------------------------
# 1.3. Fit một mặt phẳng 3D tốt nhất:
#      y ≈ a*x1 + b*x2 + c
# -----------------------------
X_plane_fit = np.column_stack([x1, x2, np.ones_like(x1)])
a_fit, b_fit, c_fit = np.linalg.lstsq(X_plane_fit, y, rcond=None)[0]

# tạo grid để vẽ plane phủ hết data
pad_x1 = 0.3
pad_x2 = 0.3

x1_plane = np.linspace(x1.min() - pad_x1, x1.max() + pad_x1, 30)
x2_plane = np.linspace(x2.min() - pad_x2, x2.max() + pad_x2, 30)
X1_plane, X2_plane = np.meshgrid(x1_plane, x2_plane)
Y_plane = a_fit * X1_plane + b_fit * X2_plane + c_fit

# -----------------------------
# 1.5. Vẽ scatter plot của dữ liệu + plane mờ
# -----------------------------
fig_data = plt.figure(figsize=(8, 6))
ax_data = fig_data.add_subplot(111, projection='3d')

ax_data.scatter(x1, x2, y, s=25, alpha=0.8, label="Data points")

ax_data.plot_surface(
    X1_plane,
    X2_plane,
    Y_plane,
    alpha=0.25,          # độ mờ của plane
    edgecolor='none'
)

ax_data.set_title("3D scatter plot of training data with fitted plane", fontsize=14)
ax_data.set_xlabel("x1")
ax_data.set_ylabel("x2")
ax_data.set_zlabel("y")
ax_data.view_init(elev=22, azim=-55)
ax_data.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 2. Tạo lưới giá trị w1, w2
# -----------------------------
w1_vals = np.linspace(-6, 6, 120)
w2_vals = np.linspace(-6, 6, 120)
W1, W2 = np.meshgrid(w1_vals, w2_vals)

Loss = np.zeros_like(W1)

# Giữ bias cố định để vẽ surface theo riêng w1, w2
b_fixed = true_b

# -----------------------------
# 3. Tính MSE tại từng điểm (w1, w2)
# -----------------------------
for i in range(W1.shape[0]):
    for j in range(W1.shape[1]):
        w1 = W1[i, j]
        w2 = W2[i, j]
        y_hat = w1 * x1 + w2 * x2 + b_fixed
        Loss[i, j] = np.mean((y - y_hat) ** 2)

# -----------------------------
# 4. Hàm tính loss và gradient
# -----------------------------
def compute_loss(w1, w2):
    y_hat = w1 * x1 + w2 * x2 + b_fixed
    return np.mean((y - y_hat) ** 2)

def compute_gradients(w1, w2):
    y_hat = w1 * x1 + w2 * x2 + b_fixed
    error = y_hat - y
    dL_dw1 = 2 * np.mean(error * x1)
    dL_dw2 = 2 * np.mean(error * x2)
    return dL_dw1, dL_dw2

# -----------------------------
# 5. Chạy gradient descent để lấy quỹ đạo đi xuống
# -----------------------------
lr = 0.2
num_steps = 20

# điểm bắt đầu
w1_start, w2_start = -5.0, 5.0

path_w1 = [w1_start]
path_w2 = [w2_start]
path_loss = [compute_loss(w1_start, w2_start)]

w1_curr, w2_curr = w1_start, w2_start

for _ in range(num_steps):
    grad_w1, grad_w2 = compute_gradients(w1_curr, w2_curr)
    w1_curr = w1_curr - lr * grad_w1
    w2_curr = w2_curr - lr * grad_w2

    path_w1.append(w1_curr)
    path_w2.append(w2_curr)
    path_loss.append(compute_loss(w1_curr, w2_curr))

# -----------------------------
# 6. Vẽ 3D surface
# -----------------------------
fig = plt.figure(figsize=(8, 10))

ax1 = fig.add_subplot(2, 1, 1, projection='3d')
ax1.plot_surface(W1, W2, Loss, cmap='viridis', edgecolor='none', alpha=0.9)
ax1.plot(path_w1, path_w2, path_loss, color='red', marker='o', markersize=4, linewidth=2)
ax1.set_title("Loss surface when x1 and x2 are independent", fontsize=16)
ax1.set_xlabel("w1")
ax1.set_ylabel("w2")
ax1.set_zlabel("Loss")
ax1.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.show()

# -----------------------------
# 7. Vẽ contour plot từ trên xuống + quỹ đạo gradient descent
# -----------------------------
plt.figure(figsize=(8, 7))

contour = plt.contourf(W1, W2, Loss, levels=40, cmap='viridis')
plt.colorbar(contour, label='Loss')

lines = plt.contour(W1, W2, Loss, levels=15, colors='white', linewidths=0.5, alpha=0.6)
plt.clabel(lines, inline=True, fontsize=8, fmt="%.1f")

# vẽ quỹ đạo gradient descent
plt.plot(path_w1, path_w2, color='red', marker='o', linewidth=2, markersize=4, label='Gradient descent path')

# đánh dấu điểm bắt đầu và điểm cuối
plt.scatter(path_w1[0], path_w2[0], color='orange', s=80, label='Start')
plt.scatter(path_w1[-1], path_w2[-1], color='cyan', s=80, label='End')

# vẽ mũi tên cho vài bước đầu để thấy hướng đi
for i in range(len(path_w1) - 1):
    plt.arrow(
        path_w1[i], path_w2[i],
        path_w1[i+1] - path_w1[i],
        path_w2[i+1] - path_w2[i],
        shape='full',
        lw=0,
        length_includes_head=True,
        head_width=0.10,
        color='red',
        alpha=0.75
    )

plt.title("Top-down contour with gradient descent path", fontsize=14)
plt.xlabel("w1")
plt.ylabel("w2")
plt.legend()
plt.show()