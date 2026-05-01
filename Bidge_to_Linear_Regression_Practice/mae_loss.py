import matplotlib.pyplot as plt

def predict(x, w, b):
    return w * x + b

# Compute Gradient for MAE
def compute_gradient_mae(y_hat, y, x):
    error = y_hat - y

    if error > 0:
        sign = 1
    elif error < 0:
        sign = -1
    else:
        sign = 0

    dw = x * sign
    db = sign
    return (dw, db)

# Update Parameter
def update_paramters(w, b, lr, dw, db):
    w_new = w - lr * dw
    b_new = b - lr * db
    return (w_new, b_new)

# MAE Loss Function
def compute_loss_mae(y_hat, y):
    return abs(y_hat - y)

def plot_chart(w, b, x_values, y_values):
    line_x = [min(x_values) - 1, max(x_values) + 1]
    line_y = [predict(x, w, b) for x in line_x]

    plt.figure(figsize=(6, 4))
    plt.scatter(x_values, y_values, color="blue")
    plt.plot(line_x, line_y, color="red")

    equation = f"y = {w:.2f}x + {b:.2f}"
    plt.text(line_x[0] + 0.2, line_y[0] + 1, equation, color="red")

    plt.xlim(min(x_values) - 0.5, max(x_values) + 0.5)
    plt.ylim(min(y_values) - 5, max(y_values) + 5)
    plt.show()

def mini_batch_training_mae(m=8):
    # Initialization
    b = 0.04
    w = -0.34
    lr = 0.01

    # How long
    epoch_max = 30

    # Training Data
    X_values = [1, 2, 3, 4, 5, 6, 7, 8]
    y_values = [2, 3, 4, 5, 6, 7, 8, 40]

    N = len(X_values)
    losses = []

    # Implementation
    for _ in range(epoch_max):
        for i in range(0, N, m):
            dw, db = [], []
            loss_total = 0

            batch_end = min(i + m, N)
            batch_size = batch_end - i

            for j in range(i, batch_end):
                X = X_values[j]
                y = y_values[j]

                y_hat = predict(X, w, b)
                loss_total += compute_loss_mae(y_hat, y)

                grad_w, grad_b = compute_gradient_mae(y_hat, y, X)
                dw.append(grad_w)
                db.append(grad_b)

            losses.append(loss_total / batch_size)
            combined_dw = sum(dw) / batch_size
            combined_db = sum(db) / batch_size
            w, b = update_paramters(w, b, lr, combined_dw, combined_db)

    plot_chart(w, b, X_values, y_values)

    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.title("MAE Loss")
    plt.xlabel("Batch step")
    plt.ylabel("Loss")
    plt.show()

    return w, b
def main() -> None:
    # simple_linear_regression()
    # mini_batch_training(2)
    mini_batch_training_mae()


if __name__ == "__main__":
    main()
