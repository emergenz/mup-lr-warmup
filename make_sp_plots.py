import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Login to wandb
wandb.login()

# Define the wandb project details
project = 'mup_transformer_warmup'
entity = 'aidos-labs'

# Names of the runs with their numerical values for warmup steps
run_names = {
    "h56rs6la": 0,
    "3s15aown": 1e6,
    "2m7q35c0": 5e6,
    "pr9oo4n6": 10e6,
    "mwocweu7": 20e6,
    "4xriu42v": 40e6,
    "d5vn89dc": 60e6
}

# Collect data
data = []

for run_name, label in run_names.items():
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_name}')
    history = run.history(keys=["Validation Loss", "Validation PPL", "Step"])
    history['run_name'] = label
    data.append(history)

# Concatenate all data
df = pd.concat(data)

# Calculate the absolute difference to 10e6 for each run
differences = {run_name: abs(label - 10e6) for run_name, label in run_names.items()}

# Sort run_names based on the differences
sorted_run_names = sorted(run_names, key=lambda x: differences[x])

# Define a sequential color palette
palette = sns.color_palette("viridis", len(run_names))

# Create a mapping from run_name to color based on sorted order
colors = {run_names[run_name]: palette[i] for i, run_name in enumerate(sorted_run_names)}

# Plot for Validation Loss
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Step', y='Validation Loss', hue='run_name', palette=colors)
plt.title('Validation Loss (SP); 46 Million Parameters; 100 Million Tokens')
plt.xlabel('Tokens Seen')
plt.ylabel('Validation Loss')
plt.legend(title='LR Warmup (Tokens Seen)')
plt.savefig('sp_validation_loss.png')

# Plot for Validation PPL
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Step', y='Validation PPL', hue='run_name', palette=colors)
plt.title('Validation PPL (SP); 46 Million Parameters; 100 Million Tokens')
plt.xlabel('Tokens Seen')
plt.ylabel('Validation PPL')
plt.legend(title='LR Warmup (Tokens Seen)')
plt.savefig('sp_validation_ppl.png')
