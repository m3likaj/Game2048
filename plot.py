import pandas as pd
import matplotlib.pyplot as plt
# Load the CSV file
csv_file = "2048-main/training_metrics_7.csv"
metrics = pd.read_csv(csv_file)

version = "PRI 7"

# Plot the score over episodes
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Plot Score
axs[0, 0].plot(metrics["Episode"], metrics["Score"], label="Score", color="blue")
axs[0, 0].set_title(f"Score Over Episodes {version}")
axs[0, 0].set_xlabel("Episode")
axs[0, 0].set_ylabel("Score")

# Plot Invalid Moves
axs[0, 1].plot(metrics["Episode"], metrics["Invalid Moves"], label="Invalid Moves", color="red")
axs[0, 1].set_title(f"Invalid Moves Over Episodes {version}")
axs[0, 1].set_xlabel("Episode")
axs[0, 1].set_ylabel("Invalid Moves")

# Plot Highest Tile
axs[1, 0].plot(metrics["Episode"], metrics["Highest Tile"], label="Highest Tile", color="green")
axs[1, 0].set_title(f"Highest Tile Over Episodes {version}")
axs[1, 0].set_xlabel("Episode")
axs[1, 0].set_ylabel("Highest Tile")

# Plot Total Reward
axs[1, 1].plot(metrics["Episode"], metrics["Total Reward"], label="Total Reward", color="purple")
axs[1, 1].set_title(f"Total Reward Over Episodes {version}")
axs[1, 1].set_xlabel("Episode")
axs[1, 1].set_ylabel("Total Reward")

# Adjust layout
plt.tight_layout()
plt.show()


