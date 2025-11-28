from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def time_to_seconds(time_str: str) -> float:
    """
    Converts a time string in "MM:SS.SS" format to the total number of seconds.

    Args:
        time_str (str): A string representing the time in "MM:SS.SS" format.

    Returns:
        float: The total time in seconds.
    """
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = float(parts[1])
    return minutes * 60 + seconds


def time_to_minutes(sec: int = None, time_str: str = None) -> float:
    """
    Converts time values into minutes.

    Args:
        sec (int, optional): Time in seconds. If provided, it will be converted
            directly to minutes.
        time_str (str, optional): Time string representation (e.g., 'HH:MM:SS').
            If provided, it will be parsed and converted to minutes.

    Returns:
        float: The equivalent time in minutes.

    Raises:
        ValueError: If neither 'sec' nor 'time_str' is provided.
    """
    if sec is not None:
        return sec / 60
    elif time_str is not None:
        return time_to_seconds(time_str) / 60
    else:
        raise ValueError("Either 'sec' or 'time_str' must be provided.")


def plot_workflowtime(data: Dict[str, Dict[str, float]], species2genomes: Dict[str, int], output: Path) -> None:
    """
    Plots workflow execution time as stacked bars with genome count overlay.

    Args:
        data: Dictionary mapping species to their execution times.
              Format: {species: {'load': float, 'annotation': float,
                                 'detection': float, 'projection': float}}
        species2genomes: Dictionary mapping species names to genome counts.
        output: Path to save the output plot.
    """
    # Create figure with primary and secondary y-axes
    _, ax1 = plt.subplots(figsize=(10, 6))

    # Extract data in consistent order
    species_list = list(species2genomes.keys())
    load = [data[sp]['load'] for sp in species_list]
    annotation = [data[sp]['annotation'] for sp in species_list]
    detection = [data[sp]['detection'] for sp in species_list]
    projection = [data[sp]['projection'] for sp in species_list]
    genomes = list(species2genomes.values())

    # Bar positions
    x = np.arange(len(species_list))
    width = 0.6

    # Stacked bars
    ax1.bar(x, load, width, label='Load', color='#8884d8')
    ax1.bar(x, annotation, width, bottom=load, label='Annotation', color='#82ca9d')
    ax1.bar(x, detection, width,
            bottom=np.array(load) + np.array(annotation),
            label='Detection', color='#ffc658')
    ax1.bar(x, projection, width,
            bottom=np.array(load) + np.array(annotation) + np.array(detection),
            label='Projection', color='#ff7c7c')

    # Primary y-axis (time)
    ax1.set_xlabel('Species', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Execution time (minutes)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(species_list, rotation=45, ha='right', style='italic')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_axisbelow(True)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Secondary y-axis (number of genomes)
    ax2 = ax1.twinx()
    ax2.plot(x, genomes, 'o-', color='#2c3e50', linewidth=2.5,
             markersize=8, label='Number of genomes', zorder=5)
    ax2.set_ylabel('Number of genomes', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2c3e50')

    # Combine legends
    bars_legend = ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.legend(loc='upper center', frameon=True, fancybox=True, shadow=True)

    # Add legend manually to show both
    ax1.add_artist(bars_legend)

    plt.title('Workflow Execution Time by Species', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output/'workflow_performance.png', dpi=300, bbox_inches='tight')
