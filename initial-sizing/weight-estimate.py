import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Inputs
MTOW = 18.7              # Maximum Takeoff Weight (kg)
ENGINE_WEIGHT = 4.38     # Engine weight (kg)
FUEL_FLOW = 0.00615      # Fuel flow rate (kg/s)

INIT_STRUCT_WEIGHT = 4.0 # Default structural weight (kg)

def calculate_available_mass(struct_wt):
    return MTOW - ENGINE_WEIGHT - struct_wt

# Plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75, top=0.9, hspace=0.3)
ax_slider = plt.axes([0.85, 0.15, 0.04, 0.7])

# Structural weight slider
struct_slider = Slider(
    ax=ax_slider,
    label='Structural\nWeight (kg)',
    valmin=2.0,
    valmax=15.0,
    valinit=INIT_STRUCT_WEIGHT,
    orientation='vertical'
)

def update(val):
    struct_wt = struct_slider.val
    available_mass = calculate_available_mass(struct_wt)
    
    ax1.clear()
    ax2.clear()
    
    if available_mass <= 0:
        ax1.text(0.5, 0.5, "MTOW Exceeded by Structure/Engine", ha='center', va='center', color='red', fontsize=12)
        ax2.text(0.5, 0.5, "MTOW Exceeded", ha='center', va='center', color='red', fontsize=12)
    else:
        fuel_weights = np.linspace(0, available_mass, 100)
        payload_weights = available_mass - fuel_weights
        endurance_mins = (fuel_weights / FUEL_FLOW) / 60.0
        
        # Plot 1: Payload vs Fuel
        ax1.plot(fuel_weights, payload_weights, linewidth=2, color='blue')
        ax1.fill_between(fuel_weights, payload_weights, alpha=0.1, color='blue')
        ax1.set_title(f"Payload vs. Fuel Mass\n(MTOW: {MTOW}kg | Engine Weight: {ENGINE_WEIGHT}kg | Available Mass for Payload+Fuel = {available_mass:.2f} kg)")
        ax1.set_ylabel("Payload Mass (kg)")
        ax1.set_xlabel("Fuel Mass (kg)")
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_xlim([0, available_mass])
        ax1.set_ylim([0, max(payload_weights) if max(payload_weights) > 0 else 1])
        
        # Plot 2: Endurance vs Fuel
        ax2.plot(fuel_weights, endurance_mins, linewidth=2, color='green')
        ax2.fill_between(fuel_weights, endurance_mins, alpha=0.1, color='green')
        ax2.set_title(f"Endurance vs. Fuel Mass")
        ax2.set_xlabel("Fuel Mass (kg)")
        ax2.set_ylabel("Endurance (minutes)")
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xlim([0, available_mass])
        ax2.set_ylim([0, max(endurance_mins) if max(endurance_mins) > 0 else 1])

    fig.canvas.draw_idle()

struct_slider.on_changed(update)
update(INIT_STRUCT_WEIGHT)
plt.show()