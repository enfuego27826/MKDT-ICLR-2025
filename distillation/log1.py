import re
import matplotlib.pyplot as plt

def parse_log_with_lr(log_path):
    with open(log_path, "r") as f:
        lines = f.readlines()

    epochs = []
    losses = []
    lrs = []

    loss_pattern = re.compile(r"Epoch (\d+): Loss = ([0-9.]+)")

    for i in range(0, len(lines) - 1, 2):
        loss_line = lines[i]
        lr_line = lines[i + 1]

        match = loss_pattern.search(loss_line)
        if match:
            epoch = int(match.group(1))
            loss = float(match.group(2))
            lr = float(lr_line.strip().split()[-1])  # FIXED

            epochs.append(epoch)
            losses.append(loss)
            lrs.append(lr)

    return epochs, losses, lrs


def plot_loss_and_lr(log_path, save_path=None):
    epochs, losses, lrs = parse_log_with_lr(log_path)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot loss on left y-axis
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    ax1.plot(epochs, losses, label="Loss", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    # Plot learning rate on right y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Learning Rate", color="tab:red")
    ax2.plot(epochs, lrs, label="LR", color="tab:red", linestyle="--")
    ax2.tick_params(axis='y', labelcolor="tab:red")

    plt.title("Barlow Twins SSL: Loss and Learning Rate")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# === Usage ===
log_file_path = "/home/anurag/log.txt"
plot_loss_and_lr(log_file_path)
