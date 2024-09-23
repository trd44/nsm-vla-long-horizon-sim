import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = {
    'Hole': {
        'Bi-Model': {
            'Tadapt(x10^4)': 29.0,
            'Tadapt(x10^4) error': 14.4,
            'SRpost-training': 0.81,
            'SRpost-training error': 0.37
        },
        'HyGOAL': {
            'Tadapt(x10^4)': 39.8,
            'Tadapt(x10^4) error': 13.7,
            'SRpost-training': 0.41,
            'SRpost-training error': 0.50
        },
        'RapidLearn': {
            'Tadapt(x10^4)': 47.8,
            'Tadapt(x10^4) error': 6.96,
            'SRpost-training': 0.18,
            'SRpost-training error': 0.35
        },
        'LTL&GO': {
            'Tadapt(x10^4)': 41.6,
            'Tadapt(x10^4) error': 16.2,
            'SRpost-training': 0.21,
            'SRpost-training error': 0.34
        },
        'SAC': {
            'Tadapt(x10^4)': 50.0,
            'Tadapt(x10^4) error': 0.0,
            'SRpost-training': 0.035,
            'SRpost-training error': 0.00
        }
    },
    'Elevated': {
        'Bi-Model': {
            'Tadapt(x10^4)': 10.0,
            'Tadapt(x10^4) error': 5.96,
            'SRpost-training': 1.0,
            'SRpost-training error': 0.0
        },
        'HyGOAL': {
            'Tadapt(x10^4)': 16.6,
            'Tadapt(x10^4) error': 13.0,
            'SRpost-training': 0.91,
            'SRpost-training error': 0.30
        },
        'RapidLearn': {
            'Tadapt(x10^4)': 21.4,
            'Tadapt(x10^4) error': 12.4,
            'SRpost-training': 0.84,
            'SRpost-training error': 0.30
        },
        'LTL&GO': {
            'Tadapt(x10^4)': 16.2,
            'Tadapt(x10^4) error': 18.0,
            'SRpost-training': 0.76,
            'SRpost-training error': 0.37
        },
        'SAC': {
            'Tadapt(x10^4)': 50.0,
            'Tadapt(x10^4) error': 0.0,
            'SRpost-training': 0.035,
            'SRpost-training error': 0.00
        }
    },
    'Obstacle': {
        'Bi-Model': {
            'Tadapt(x10^4)': 38.4,
            'Tadapt(x10^4) error': 10.9,
            'SRpost-training': 0.71,
            'SRpost-training error': 0.43
        },
        'HyGOAL': {
            'Tadapt(x10^4)': 39.0,
            'Tadapt(x10^4) error': 16.0,
            'SRpost-training': 0.38,
            'SRpost-training error': 0.48
        },
        'RapidLearn': {
            'Tadapt(x10^4)': 36.2,
            'Tadapt(x10^4) error': 13.6,
            'SRpost-training': 0.66,
            'SRpost-training error': 0.40
        },
        'LTL&GO': {
            'Tadapt(x10^4)': 50.0,
            'Tadapt(x10^4) error': 0.0,
            'SRpost-training': 0.04,
            'SRpost-training error': 0.10
        },
        'SAC': {
            'Tadapt(x10^4)': 50.0,
            'Tadapt(x10^4) error': 0.0,
            'SRpost-training': 0.035,
            'SRpost-training error': 0.00
        }
    },
    'Locked Door': {
        'Bi-Model': {
            'Tadapt(x10^4)': 20.8,
            'Tadapt(x10^4) error': 12.5,
            'SRpost-training': 0.93,
            'SRpost-training error': 0.24
        },
        'HyGOAL': {
            'Tadapt(x10^4)': 13.8,
            'Tadapt(x10^4) error': 5.00,
            'SRpost-training': 0.90,
            'SRpost-training error': 0.28
        },
        'RapidLearn': {
            'Tadapt(x10^4)': 29.4,
            'Tadapt(x10^4) error': 15.8,
            'SRpost-training': 0.72,
            'SRpost-training error': 0.38
        },
        'LTL&GO': {
            'Tadapt(x10^4)': 39.6,
            'Tadapt(x10^4) error': 17.3,
            'SRpost-training': 0.21,
            'SRpost-training error': 0.30
        },
        'SAC': {
            'Tadapt(x10^4)': 50.0,
            'Tadapt(x10^4) error': 0.0,
            'SRpost-training': 0.035,
            'SRpost-training error': 0.00
        }
    },
    'Light Off': {
        'Bi-Model': {
            'Tadapt(x10^4)': 28.8,
            'Tadapt(x10^4) error': 16.1,
            'SRpost-training': 0.85,
            'SRpost-training error': 0.31
        },
        'HyGOAL': {
            'Tadapt(x10^4)': 26.4,
            'Tadapt(x10^4) error': 14.8,
            'SRpost-training': 0.80,
            'SRpost-training error': 0.37
        },
        'RapidLearn': {
            'Tadapt(x10^4)': 31.0,
            'Tadapt(x10^4) error': 11.6,
            'SRpost-training': 0.73,
            'SRpost-training error': 0.32
        },
        'LTL&GO': {
            'Tadapt(x10^4)': 41.6,
            'Tadapt(x10^4) error': 17.8,
            'SRpost-training': 0.23,
            'SRpost-training error': 0.38
        },
        'SAC': {
            'Tadapt(x10^4)': 50.0,
            'Tadapt(x10^4) error': 0.0,
            'SRpost-training': 0.035,
            'SRpost-training error': 0.00
        }
    }
}

data_ablation = {
    'Hole ablation': {
        'Bi-Model': {
            'Tadapt(x10^4)': 29.0,
            'Tadapt(x10^4) error': 14.4,
            'SRpost-training': 0.81,
            'SRpost-training error': 0.37
        },
        'ICM Only': {
            'Tadapt(x10^4)': 41.6,
            'Tadapt(x10^4) error': 13.6,
            'SRpost-training': 0.54,
            'SRpost-training error': 0.45
        },
        'PRM Only': {
            'Tadapt(x10^4)': 31.8,
            'Tadapt(x10^4) error': 12.8,
            'SRpost-training': 0.80,
            'SRpost-training error': 0.39
        }
    },
    'Elevated ablation': {
        'Bi-Model': {
            'Tadapt(x10^4)': 10.0,
            'Tadapt(x10^4) error': 5.96,
            'SRpost-training': 1.0,
            'SRpost-training error': 0.0
        },
        'ICM Only': {
            'Tadapt(x10^4)': 19.4,
            'Tadapt(x10^4) error': 13.0,
            'SRpost-training': 0.95,
            'SRpost-training error': 0.14
        },
        'PRM Only': {
            'Tadapt(x10^4)': 12.8,
            'Tadapt(x10^4) error': 14.0,
            'SRpost-training': 0.93,
            'SRpost-training error': 0.22
        }
    },
    'Obstacle ablation': {
        'Bi-Model': {
            'Tadapt(x10^4)': 38.4,
            'Tadapt(x10^4) error': 10.9,
            'SRpost-training': 0.71,
            'SRpost-training error': 0.43
        },
        'ICM Only': {
            'Tadapt(x10^4)': 35.4,
            'Tadapt(x10^4) error': 11.8,
            'SRpost-training': 0.79,
            'SRpost-training error': 0.40
        },
        'PRM Only': {
            'Tadapt(x10^4)': 39.6,
            'Tadapt(x10^4) error': 11.9,
            'SRpost-training': 0.60,
            'SRpost-training error': 0.47
        }
    },
    'Locked Door ablation': {
        'Bi-Model': {
            'Tadapt(x10^4)': 20.8,
            'Tadapt(x10^4) error': 12.5,
            'SRpost-training': 0.93,
            'SRpost-training error': 0.24
        },
        'ICM Only': {
            'Tadapt(x10^4)': 26.4,
            'Tadapt(x10^4) error': 17.2,
            'SRpost-training': 0.71,
            'SRpost-training error': 0.48
        },
        'PRM Only': {
            'Tadapt(x10^4)': 27.4,
            'Tadapt(x10^4) error': 19.5,
            'SRpost-training': 0.63,
            'SRpost-training error': 0.48
        }
    },
    'Light Off ablation': {
        'Bi-Model': {
            'Tadapt(x10^4)': 28.8,
            'Tadapt(x10^4) error': 16.1,
            'SRpost-training': 0.85,
            'SRpost-training error': 0.31
        },
        'ICM Only': {
            'Tadapt(x10^4)': 33.0,
            'Tadapt(x10^4) error': 16.5,
            'SRpost-training': 0.74,
            'SRpost-training error': 0.41
        },
        'PRM Only': {
            'Tadapt(x10^4)': 27.6,
            'Tadapt(x10^4) error': 13.8,
            'SRpost-training': 0.97,
            'SRpost-training error': 0.08
        }
    }
}
categories = ['Tadapt(x10^4)', 'SRpost-training']
algorithms = ['Bi-Model', 'HyGOAL', 'RapidLearn', 'LTL&GO', 'SAC']
ablation_algorithms = ['Bi-Model', 'ICM Only', 'PRM Only']
# Create a function to plot a 1 x 2 seaborn bar chart with error bars. Each bar represents a different algorithm. One chart for each category e.g. Tadapt and SRpost-training.
def plot(category='Tadapt(x10^4)'):
    sns.set(style='darkgrid')  # Set the style to seaborn
    fig, axs = plt.subplots(1, 10, figsize=(200, 5), sharey=True) # 10 subplots. One row for each category

    for j, novelty in enumerate(data.keys()):
        for _, algorithm in enumerate(algorithms):
            ax = axs[j]
            if category == 'Tadapt(x10^4)':
                ax.bar(x=algorithm, height=data[novelty][algorithm][category], yerr=data[novelty][algorithm][f'{category} error'], label=algorithm)
                #ax.set_ylim(0, 60)
            elif category == 'SRpost-training':
                ax.bar(x=algorithm, height=data[novelty][algorithm][category], yerr=data[novelty][algorithm][f'{category} error'], label=algorithm)
                #ax.set_ylim(0, 1.2)
            ax.set(xticklabels=[])  # Remove the x-label
        
    for j, novelty in enumerate(data_ablation.keys()):
        for _, algorithm in enumerate(ablation_algorithms):
            ax = axs[j+len(data.keys())]
            if category == 'Tadapt(x10^4)':
                ax.bar(x=algorithm, height=data_ablation[novelty][algorithm][category], yerr=data_ablation[novelty][algorithm][f'{category} error'], label=algorithm)
                #ax.set_ylim(0, 60)
            elif category == 'SRpost-training':
                ax.bar(x=algorithm, height=data_ablation[novelty][algorithm][category], yerr=data_ablation[novelty][algorithm][f'{category} error'], label=algorithm)
                #ax.set_ylim(0, 1.2)
            ax.set(xticklabels=[])  # Remove the x-label
        #ax.tick_params(axis='y', labelsize=30)  # Increase the y tick size
        fig.tight_layout()
    plt.savefig(f'{category}.png')  # Save the figure with the provided novelty_name
plot(category='Tadapt(x10^4)')
plot(category='SRpost-training')
# for novelty_name, data in data.items():
#     plot(data, novelty_name)
# for novelty_name, data in data_ablation.items():
#     plot(data, novelty_name+' ablation', True)