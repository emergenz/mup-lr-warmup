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

# Names of the runs with their numerical values for warmup steps for SP and muP
sp_run_names = {
    "h56rs6la": 0,
    "3s15aown": 1e6,
    "2m7q35c0": 5e6,
    "pr9oo4n6": 10e6,
    "mwocweu7": 20e6,
    "4xriu42v": 40e6,
    "d5vn89dc": 60e6
}

mup_run_names = {
    "8fb6mfx3": 0,
    "v2r80mbm": 1e6,
    "h8rngn55": 5e6,
    "2mvt5tww": 10e6,
    "hirf1fqo": 20e6,
    "bkq4vee0": 40e6,
    "j3jvxyz5": 60e6
}

# Collect data for SP
sp_data = []
for run_name, label in sp_run_names.items():
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_name}')
    history = run.history(keys=["Validation Loss", "Validation PPL", "Step"])
    history['run_name'] = label
    history['regime'] = 'SP'
    sp_data.append(history)

# Collect data for muP
mup_data = []
for run_name, label in mup_run_names.items():
    api = wandb.Api()
    run = api.run(f'{entity}/{project}/{run_name}')
    history = run.history(keys=["Validation Loss", "Validation PPL", "Step"])
    history['run_name'] = label
    history['regime'] = 'muP'
    mup_data.append(history)

# Concatenate all data
df = pd.concat(sp_data + mup_data)

# Calculate the absolute difference to 10e6 for each run
differences = {run_name: abs(label - 10e6) for run_name, label in {**sp_run_names, **mup_run_names}.items()}

# Sort run_names based on the differences
sorted_run_names = sorted({**sp_run_names, **mup_run_names}, key=lambda x: differences[x])

# Define a color palette for regimes and run names
regime_palette = {"SP": "blue", "muP": "orange"}
palette = sns.color_palette("viridis", len(sp_run_names) + len(mup_run_names))

# Create a mapping from run_name to color based on sorted order
run_name_colors = {run_name: palette[i] for i, run_name in enumerate(sorted_run_names)}

# Plot for Validation Loss
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Step', y='Validation Loss', hue='regime', style='run_name', palette=regime_palette)
for run_name, color in run_name_colors.items():
    sns.lineplot(data=df[df['run_name'] == run_name], x='Step', y='Validation Loss', color=color, legend=False)
plt.title('Validation Loss Comparison (SP vs muP); 46 Million Parameters; 100 Million Tokens')
plt.xlabel('Tokens Seen')
plt.ylabel('Validation Loss')
plt.legend(title='Regime / LR Warmup (Tokens Seen)')
plt.savefig('comparison_validation_loss.png')

# Plot for Validation PPL
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x='Step', y='Validation PPL', hue='regime', style='run_name', palette=regime_palette)
for run_name, color in run_name_colors.items():
    sns.lineplot(data=df[df['run_name'] == run_name], x='Step', y='Validation PPL', color=color, legend=False)
plt.title('Validation PPL Comparison (SP vs muP); 46 Million Parameters; 100 Million Tokens')
plt.xlabel('Tokens Seen')
plt.ylabel('Validation PPL')
plt.legend(title='Regime / LR Warmup (Tokens Seen)')
plt.savefig('comparison_validation_ppl.png')
