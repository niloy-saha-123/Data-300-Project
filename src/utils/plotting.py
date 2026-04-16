import matplotlib.pyplot as plt
import seaborn as sns

def set_shared_style():
    """
    Sets the global matplotlib and seaborn styles for consistent 
    report-quality figures across all notebooks.
    """
    # Use the mandated seaborn style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Set the figure DPI to 150 for high-quality exports
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['savefig.dpi'] = 150
    
    # Define a shared color palette (Starbucks-inspired greens/neutrals as an example)
    custom_palette = ["#00704A", "#27251F", "#D4E9E2", "#CBA258", "#F2F0EB"]
    sns.set_palette(sns.color_palette(custom_palette))
    
    # Adjust font sizes for readability in presentations
    plt.rcParams.update({
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

# Execute the function so importing this module automatically applies the style
set_shared_style()