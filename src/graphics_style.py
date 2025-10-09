# graphics_style.py
import matplotlib.pyplot as plt
import seaborn as sns

def apply_graphics_style():
    # Base Matplotlib settings
    plt.style.use('default')  # Reset to default before applying custom style
    plt.rcParams.update({
        'font.family': 'Arial',  # Sans-serif font
        'font.size': 12,         # Readable base size
        'axes.titlesize': 14,    # Slightly larger titles
        'axes.labelsize': 12,    # Clear axis labels
        'xtick.labelsize': 10,   # Smaller tick labels
        'ytick.labelsize': 10,
        'axes.titleweight': 'bold',  # Bold titles
        'axes.linewidth': 0.8,       # Thin axis lines
        'axes.edgecolor': '#333333', # Dark grey edges
        'axes.facecolor': '#FFFFFF', # White background
        'figure.facecolor': '#FFFFFF',
        'grid.color': '#D3D3D3',     # Light grey gridlines
        'grid.linestyle': '--',      # Dashed gridlines
        'grid.alpha': 0.5,           # Subtle grid
    })

    # Seaborn-specific settings
    sns.set_style("white", {
        'axes.grid': True,
        'grid.color': '#D3D3D3',
        'grid.linestyle': '--',
        'axes.spines.top': False,    # Remove top spine
        'axes.spines.right': False,  # Remove right spine
    })

    # Custom color palette
    graphics_colors = [
        '#005566',  # Dark teal
        '#A3C1AD',  # Light green-blue
        '#4D4D4D',  # Dark grey
        '#8B0000',  # Dark red (used for thresholds, but available for groups)
        '#8C5B79',  # Muted purple
        '#FFB347',  # Soft orange
        '#6A9BC3',  # Medium blue
        '#A9A9A9',  # Light grey
    ]
    sns.set_palette(graphics_colors)

    # Additional tweaks for specific plot types
    plt.rcParams['lines.linewidth'] = 2  # Thicker lines for survival curves
    plt.rcParams['legend.frameon'] = False  # No legend border
    plt.rcParams['legend.loc'] = 'upper right'  # Default legend position

# Apply style globally when imported
apply_graphics_style()