import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from datetime import timedelta

def plot_solar_intensity(ax, df, title="Solar Intensity Over Time"):
    """Plot solar intensity trends."""
    if "intensity" not in df.columns:
        logging.error("Column 'intensity' not found for intensity plot.")
        return
    
    ax.plot(df['timestamp'], df['intensity'], 
            label="Solar Intensity", color='orange', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity")
    ax.grid(True)
    ax.legend()

def plot_kp_index(ax, df, title="Geomagnetic Activity (Kp Index)"):
    """Plot Kp index trends."""
    if "kp_index" not in df.columns:
        logging.error("Column 'kp_index' not found for Kp index plot.")
        return
    
    ax.plot(df['timestamp'], df['kp_index'], 
            label="Kp Index", color='blue', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Kp Index")
    ax.grid(True)
    ax.legend()

def plot_event_distribution(ax, df, title="Event Distribution"):
    """Plot event distribution if available."""
    if "event" not in df.columns:
        logging.error("Column 'event' not found for event distribution plot.")
        return
    
    events = df['event'].value_counts()
    ax.bar(['No Event', 'Event'], 
           [events.get(0, 0), events.get(1, 0)],
           color=['green', 'red'])
    ax.set_title(title)
    ax.set_ylabel("Count")
    ax.grid(True, axis='y')

def plot_location_heatmap(ax, df, title="Solar Activity Locations"):
    """Plot heatmap of solar activity locations."""
    if "location" not in df.columns:
        logging.error("Column 'location' not found for location heatmap.")
        return
    
    # Extract coordinates from location strings (e.g., 'N30E45' -> (30, 45))
    def extract_coords(loc):
        try:
            ns = int(loc[1:3]) * (1 if loc[0] == 'N' else -1)
            ew = int(loc[4:6]) * (1 if loc[3] == 'E' else -1)
            return ns, ew
        except:
            return None
    
    # Create heatmap data
    valid_locations = df['location'].apply(extract_coords).dropna()
    if len(valid_locations) > 0:
        ns_coords, ew_coords = zip(*valid_locations)
        ax.hist2d(ew_coords, ns_coords, bins=10, cmap='YlOrRd')
        ax.set_title(title)
        ax.set_xlabel("East-West Position")
        ax.set_ylabel("North-South Position")
    else:
        logging.warning("No valid location data for heatmap.")

def visualize_data(df):
    """
    Create comprehensive visualization of solar activity data.
    
    Parameters:
    df (pandas.DataFrame): Data containing time-series solar activity observations.
    """
    if df.empty:
        logging.error("Input DataFrame is empty in visualize_data.")
        return

    # Ensure timestamp column exists
    if "timestamp" not in df.columns:
        logging.error("Required column 'timestamp' not found.")
        return

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Solar Intensity Time Series
    ax1 = fig.add_subplot(gs[0, :])
    plot_solar_intensity(ax1, df)

    # Plot 2: Kp Index Time Series
    ax2 = fig.add_subplot(gs[1, :])
    plot_kp_index(ax2, df)

    # Plot 3: Event Distribution
    ax3 = fig.add_subplot(gs[2, 0])
    plot_event_distribution(ax3, df)

    # Plot 4: Location Heatmap
    ax4 = fig.add_subplot(gs[2, 1])
    plot_location_heatmap(ax4, df)

    # Add overall title
    plt.suptitle("Solar Activity Analysis Dashboard", size=16, y=1.02)
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    
    logging.info("Generated comprehensive solar activity visualization.")

    # Save the plot if needed
    try:
        plt.savefig('solar_activity_dashboard.png', bbox_inches='tight', dpi=300)
        logging.info("Saved visualization to solar_activity_dashboard.png")
    except Exception as e:
        logging.error(f"Error saving visualization: {e}") 