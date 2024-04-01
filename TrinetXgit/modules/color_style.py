import matplotlib.pyplot as plt
import warnings
from lifelines.utils import ApproximationWarning

def color_style():
    plt.style.use('seaborn')
    warnings.simplefilter("ignore", category=ApproximationWarning)

    color_palette_lib = {
        'vibrant_cyan': '#43aecc',
        'neutral_grey': '#78787c',
        'deep_blue': '#323e68',
        'darker_grey': '#5b5b60',
        'darker_blue': '#4a5572',
        'lighter_cyan': '#6bc7e5',
        'muted_greenish_blue': '#526f7b',
        'soft_purple': '#867c9a',
        'muted_burgundy_red': '#8B3A3A'  # The added deep muted red
    }

    color_palette = [
    
        '#323e68',  # Original deep blue
        '#8B3A3A',   # Deep muted red (added)
        '#43aecc',  # Original vibrant cyan
        '#78787c',  # Original neutral grey
        
        '#5b5b60',  # Darker grey
        '#4a5572',  # Darker blue
        '#6bc7e5',  # Lighter cyan
        '#526f7b',  # Muted greenish-blue
        '#867c9a',  # Soft purple
        
    ]