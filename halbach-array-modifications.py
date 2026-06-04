import numpy as np
import halbachFields
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os
import pickle

"""
Based on homogeneityOptimization.py (Modified to isolate forward field calculations, RG 2025)

Created on Mon Sep 24 16:17:41 2018

@author: to_reilly
"""

def graph_fields(coordinateAxis, maskedField, shimmedField):
    '''
    Tom O'Reilly's original code for graphing resulting simulated fields, 
    included as a function with field parameters passed in.
    Modified version by Robert Griffin.

    Visualize various slices of the simulated magnetic field.
    
    coordinateAxis: magnetic field magnitude after applying the cylindrical mask.
    shimmedField : Full simulated magnetic field (in Tesla) before masking.

    Displays figures, no output
    '''    
    subFull, Fullax = plt.subplots(1, 3, figsize=(20, 5))
    Fullax[0].plot(coordinateAxis * 1e2,
                    maskedField[int(np.floor(np.size(maskedField, 0) / 2)),
                    int(np.floor(np.size(maskedField, 1) / 2)), :]+ 0.005)
    Fullax[1].plot(coordinateAxis * 1e2,
                    maskedField[int(np.floor(np.size(maskedField, 0) / 2)), :,
                    int(np.floor(np.size(maskedField, 2) / 2))])
    Fullax[2].plot(coordinateAxis * 1e2,
                    maskedField[:, int(np.floor(np.size(maskedField, 1) / 2)),
                    int(np.floor(np.size(maskedField, 2) / 2))])

    Fullax[0].set_xlabel('X axis (cm)')
    Fullax[0].set_ylabel('Field strength (Tesla)')
    Fullax[1].set_xlabel('Z axis (cm)')
    Fullax[1].set_ylabel('Field strength (Tesla)')
    Fullax[2].set_xlabel('Y axis (cm)')
    Fullax[2].set_ylabel('Field strength (Tesla)')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 3)
    im1 = ax[0].imshow(maskedField[:, :, int(np.floor(np.size(maskedField, 2) / 2))])
    im2 = ax[1].imshow(maskedField[:, int(np.floor(np.size(maskedField, 1) / 2)), :])
    im3 = ax[2].imshow(maskedField[int(np.floor(np.size(maskedField, 0) / 2)), :, :])
    ax[0].set_xlabel('Z (mm)')
    ax[0].set_ylabel('Y (mm)')
    ax[1].set_xlabel('X (mm)')
    ax[1].set_ylabel('Y (mm)')
    ax[2].set_xlabel('X (mm)')
    ax[2].set_ylabel('Z (mm)')

    cbar1 = fig.colorbar(im1, ax=ax[0])
    cbar2 = fig.colorbar(im2, ax=ax[1])
    cbar3 = fig.colorbar(im3, ax=ax[2])

    figu, axe = plt.subplots(1, 3)
    i1 = axe[0].imshow(shimmedField[:, :, int(np.floor(np.size(shimmedField, 2) / 2))])
    i2 = axe[1].imshow(shimmedField[:, int(np.floor(np.size(shimmedField, 1) / 2)), :])
    i3 = axe[2].imshow(shimmedField[int(np.floor(np.size(shimmedField, 0) / 2)), :, :])
    axe[0].set_xlabel('Z (mm)')
    axe[0].set_ylabel('Y (mm)')
    axe[1].set_xlabel('X (mm)')
    axe[1].set_ylabel('Y (mm)')
    axe[2].set_xlabel('X (mm)')
    axe[2].set_ylabel('Z (mm)')

    cb1 = figu.colorbar(i1, ax=axe[0])
    cb2 = figu.colorbar(i2, ax=axe[1])
    cb3 = figu.colorbar(i3, ax=axe[2])

    plt.show()
Simulating the Halbach Array MRI (with **one** set of input parameters):
def halbach_array_simulation(numRings, innerRingRadii, innerNumMagnets, ringSep, graph):
    '''
    Tom O'Reilly's original code for simulating fields, 
    included as a function with field parameters passed in.
    Modified version by Robert Griffin.

    Run a single Halbach array MRI field simulation using the given design parameters.

    This function builds the magnet array geometry (ring positions, radii, magnet counts), 
    constructs a cylindrical mask defining the field-of-view,
    computes the 3D magnetic field using halbachFields.createHalbach(),
    applies the mask and computes mean field magnitude and homogeneity, and
    can optionally visualize the field distribution

    numRings: Number of magnet rings (symmetric about x = 0)
    innerRingRadii radii (in meters) for the inner Halbach rings
    innerNumMagnets: Number of magnets in each ring
    ringSep Spacial separation between adjacent rings (in mm)
    graph: Plots field slices and cross-sections if true

    Outputs:
    field_details:
    - 'Shimmed mean' (mT)
    - 'Shimmed homogeneity (mt)'
    - 'Shimmed homogeneity (ppm)'
    maskedField: Masked field magnitude used to compute homogeneity
    shimmedField: Full unmasked magnetic field (Bx aligned component)
    coordinateAxis: Axis coordinates (in meters) for the field volume
    '''
    ##########################################    Define Magnet Array    ##########################################
    magnetLength = (numRings - 1) * ringSep * 1e-3
    # After using the gap size to calculate the total length of the magnet, determine the position of the center of each magnet ring (evenly spaced)
    ringPositions = np.linspace(-magnetLength / 2, magnetLength / 2, numRings)

    resolution = 1                  # voxel size in mm
    DSV = 80                        # Size of the unmasked simulation region in mm
    # Make sure that the total simulated field fits into the main memory of your computer 
    # (reduce DSV or increase voxel size to reduce memory usage) or this will code will run extremely slow
    
    CSVname = "3D_Field.csv"        # Name for the CSV file storing 3D field values resulting from the magnet simulation
    #graph = True                    # Toggle for plotting code

    single_ring = False             # This parameter allows for the addition of a larger magnet ring that is concentric with the original ring

    if single_ring:
        outerRingRadii = innerRingRadii
        outerNumMagnets = innerNumMagnets
    else:
        outerRingRadii = innerRingRadii + 21 * 1e-3
        outerNumMagnets = innerNumMagnets + 7

    ##########################################    Create Cylindrical mask    ##########################################
    # Field properties of the simulated magnet are calculated over a cynlindrical volume around its center
    eval_Radius = 20                # Radius of the field evaluation region
    eval_half_Thickness = 10             # Thickness of the field evaluation region

    simDimensions = (DSV * 1e-3, DSV * 1e-3, DSV * 1e-3)

    coordinateAxis = np.linspace(-simDimensions[0] / 2, simDimensions[0] / 2,
                                     int(1e3 * simDimensions[0] / resolution + 1))
    coords = np.meshgrid(coordinateAxis, coordinateAxis, coordinateAxis)

    mask = np.zeros(np.shape(coords[0]))    # Initialize an empty mask
    mask[np.square(coords[0]) + np.square(coords[1]) <= (eval_Radius * 1e-3) ** 2] = 1  # Mark all voxels within the cylindrical mask radius
    mask[np.absolute(coords[2]) >= (eval_half_Thickness * 1e-3)+.001] = 0        # Unmark voxels outside cylindrical mask thickness

    ##########################################        Magnet Simulation       ##########################################
    ringPositionsSymmetery = ringPositions[ringPositions >= 0]      # Remove negative ring positions. Because this script assumes a symmetric magnet,
    # negative ring positions will have a corresponding non-negative duplicate that can be used instead

    shimmedField = np.zeros(np.shape(mask)  + (3,))
    for positionIdx, position in enumerate(ringPositionsSymmetery):     #Loop over every non-negative ring position
        if position == 0:
            rings = (0,)    #If a ring is positioned at x = 0, simulate only that ring
        else:
            rings = (-position, position)   # If a ring is positioned away from x = 0, simulate both that ring and the version reflected over the yz plane at x = 0

        # Call createHalbach() to simulate the field of a manget ring based on: 
        #       the number of magnets in the ring(s) (numMagnets), 
        #       the x position of the ring(s) (rings), 
        #       the radius of the ring(s) (radius), 
        #       the size of the permanent magnets (magnetSize), 
        #       the resolution of the simulation (resolution), 
        #       and the size of the total simulation area (simDimensions)
        shimmedField += halbachFields.createHalbach(numMagnets=innerNumMagnets[positionIdx], rings=rings,
                                                    radius=innerRingRadii[positionIdx], magnetSize=0.012,
                                                    resolution=1e3 / resolution, simDimensions=simDimensions)
        if not single_ring:     # If there are multiple rings at the same x position, simulate the field of the outer ring
            # using new values for numMagnets and radius
            shimmedField += halbachFields.createHalbach(numMagnets=outerNumMagnets[positionIdx], rings=rings,
                                                    radius=outerRingRadii[positionIdx], magnetSize=0.012,
                                                    resolution=1e3 / resolution, simDimensions=simDimensions)

    # Save 3D field values to a CSV file
    Nx, Ny, Nz, _ = shimmedField.shape
    with open(CSVname, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z", "Bx", "By", "Bz"])

        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    Bx, By, Bz = shimmedField[i, j, k]
                    writer.writerow([k, j, i, Bz, By, Bx])

    shimmedField = shimmedField[..., 0]

    #Apply the Cylindrical mask to the simulated magnetic field
    mask[mask == 0] = np.nan
    maskedField = np.abs(np.multiply(shimmedField, mask))

    #Graph
    if graph:
        graph_fields(coordinateAxis, maskedField, shimmedField)

    #Print an overview of the magnet's field characteristics
    print("Shimmed mean: %.2f mT" % (1e3 * np.nanmean(maskedField)))
    print("Shimmed homogeneity: %.4f mT" % (1e3 * (np.nanmax(maskedField) - np.nanmin(maskedField))))
    print("Shimmed homogeneity: %i ppm" % (
                1e6 * ((np.nanmax(maskedField) - np.nanmin(maskedField)) / np.nanmean(maskedField))))

    field_details = {
        "Shimmed mean" : 1e3 * np.nanmean(maskedField),
        "Shimmed homogeneity (mt)" : 1e3 * (np.nanmax(maskedField) - np.nanmin(maskedField)),
        "Shimmed homogeneity (ppm)" : 1e6 * ((np.nanmax(maskedField) - np.nanmin(maskedField)) / np.nanmean(maskedField))
    }

    #return field_details
    return field_details, maskedField, shimmedField, coordinateAxis
Testing the Simulation with Original Parameters:
result = halbach_array_simulation(4, np.array([148, 151]) * 1e-3, np.array([50, 51]), 15, True)
Saving & Loading Results (Just to save time, takes ~15 minutes to execute all cells otherwise): 
def save_results(results_list, field_data_list, filename_base):
    """Save simulation results to files"""
    # Save results as CSV for easy viewing
    results_df = pd.DataFrame([{
        'config_name': r.get('config_name', r.get('name', 'Unknown')),
        'numRings': r['numRings'],
        'ringSep': r['ringSep'],
        'mean_field_mT': r['field_details']['Shimmed mean'],
        'homogeneity_mT': r['field_details']['Shimmed homogeneity (mt)'],
        'homogeneity_ppm': r['field_details']['Shimmed homogeneity (ppm)']
    } for r in results_list])
    results_df.to_csv(f"{filename_base}_results.csv", index=False)
    
    # Save full results including field data as pickle for complete reconstruction
    with open(f"{filename_base}_full.pkl", 'wb') as f:
        pickle.dump({'results': results_list, 'fields': field_data_list}, f)
    
    print(f"Results saved to {filename_base}_results.csv and {filename_base}_full.pkl")

def load_results(filename_base):
    """Load simulation results from files"""
    pkl_file = f"{filename_base}_full.pkl"
    if os.path.exists(pkl_file):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded results from {pkl_file}")
        return data['results'], data['fields']
    return None, None
## Plotting & Simulation (using **List** of Parameters) Functions
---
Running Halbach Simulation for each list of parameters in input:
def run_parameter_sweep(halbach_simulation_func, param_configs, show_progress=True, save_fields=True):
    """
    Execute a batch of Halbach array simulations over multiple parameter sets.

    halbach_simulation_func: Simulation function to call for each parameter configuration
    param_configs:
    - 'numRings'
    - 'innerRingRadii'
    - 'innerNumMagnets'
    - 'ringSep'
    and optionally 'config_name'.
    show_progress: prints progress updates and computed metrics during the sweep if true
    save_fields: stores masked field data and coordinate axis for each simulation if true

    outputs:
    results_list: Summary of parameters and computed field metrics for each configuration
    field_data_list: (maskedField, coordinateAxis, field_details) tuple if save_fields
    """
    results_list = []
    field_data_list = [] if save_fields else None
    
    for i, config in enumerate(param_configs):
        # Handle both 'name' and 'config_name' keys
        config_name = config.get('config_name', config.get('name', f'Config_{i+1}'))
        
        if show_progress:
            print(f"\nRunning simulation {i+1}/{len(param_configs)}: {config_name}")
        
        # Run simulation with graph=False to suppress individual plots
        result = halbach_simulation_func(
            numRings=config['numRings'],
            innerRingRadii=config['innerRingRadii'],
            innerNumMagnets=config['innerNumMagnets'],
            ringSep=config['ringSep'],
            graph=False
        )
        
        # Unpack results
        if len(result) == 4:
            field_details, maskedField, shimmedField, coordinateAxis = result
            
            # Save field data if requested
            if save_fields:
                field_data_list.append((maskedField, coordinateAxis, field_details))
        else:
            field_details = result
        
        # Calculate innerRingRadii_mm if not provided
        innerRingRadii_mm = config.get('innerRingRadii_mm', config['innerRingRadii'] * 1e3)
        
        # Store results with parameters
        result_entry = {
            'numRings': config['numRings'],
            'innerRingRadii': config['innerRingRadii'],
            'innerRingRadii_mm': innerRingRadii_mm,
            'innerNumMagnets': config['innerNumMagnets'],
            'ringSep': config['ringSep'],
            'config_name': config_name,
            'field_details': field_details
        }
        
        results_list.append(result_entry)
        
        if show_progress:
            print(f"  Mean: {field_details['Field mean']:.2f} mT, "
                  f"Homogeneity: {field_details['Field homogeneity (ppm)']:.0f} ppm")
    
    print(f"\n{'='*60}")
    print(f"Parameter sweep complete! {len(results_list)} simulations finished.")
    print(f"{'='*60}")
    
    # Find best configuration
    best_idx = np.argmin([r['field_details']['Field homogeneity (ppm)'] 
                          for r in results_list])
    best_config = results_list[best_idx]
    print(f"\nBest configuration: {best_config['config_name']}")
    print(f"  Homogeneity: {best_config['field_details']['Field homogeneity (ppm)']:.0f} ppm")
    print(f"  Mean field: {best_config['field_details']['Field mean']:.2f} mT")
    
    if save_fields:
        return results_list, field_data_list
    else:
        return results_list
Visualizing results:
def plot_field_metrics_histogram(all_results):
    """
    Creates a grouped bar chart showing both Shimmed mean (mT) and 
    Shimmed homogeneity (ppm) for each configuration.
    
    all_results: list of result dictionaries containing field details
        
    outputs:
    fig: figure containing grouped bar chart
    """
    # Extract values and names
    homogeneity_ppm = []
    shimmed_mean_mT = []
    names = []
    
    for r in all_results:
        # Check if it's from run_parameter_sweep (nested format)
        if 'field_details' in r:
            homogeneity_ppm.append(r['field_details']['Shimmed homogeneity (ppm)'])
            shimmed_mean_mT.append(r['field_details']['Shimmed mean'])
            names.append(r.get('config_name', r.get('name', 'Unknown')))
        # Or from run_simulation (direct format)
        else:
            homogeneity_ppm.append(r['homogeneity_ppm'])
            shimmed_mean_mT.append(r.get('shimmed_mean_mT', r.get('mean_field_mT', 0)))
            names.append(r['name'])
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    x_pos = np.arange(len(names))
    width = 0.35  # Width of bars
    
    # Create first set of bars (Field mean) on left y-axis
    bars1 = ax1.bar(x_pos - width/2, shimmed_mean_mT, width, 
                     label='Field Mean (mT)', 
                     color='dodgerblue', edgecolor='black', alpha=0.7)
    
    # Color the best (highest) mean bar in green, worst in red
    best_mean_idx = shimmed_mean_mT.index(max(shimmed_mean_mT))
    worst_mean_idx = shimmed_mean_mT.index(min(shimmed_mean_mT))
    bars1[best_mean_idx].set_color('green')
    bars1[worst_mean_idx].set_color('red')
    
    # Set up first y-axis
    ax1.set_ylabel('Field Mean (mT)', fontsize=12, fontweight='bold', color='dodgerblue')
    ax1.tick_params(axis='y', labelcolor='dodgerblue')
    ax1.set_xlabel('Configuration', fontsize=12, fontweight='bold')
    
    # Create second y-axis for homogeneity
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + width/2, homogeneity_ppm, width, 
                     label='Field Homogeneity (ppm)', 
                     color='grey', edgecolor='black', alpha=0.7)
    
    # Color the best (lowest) homogeneity bar in green, worst in red
    best_homog_idx = homogeneity_ppm.index(min(homogeneity_ppm))
    worst_homog_idx = homogeneity_ppm.index(max(homogeneity_ppm))
    bars2[best_homog_idx].set_color('green')
    bars2[worst_homog_idx].set_color('red')
    
    # Set up second y-axis
    ax2.set_ylabel('Field Homogeneity (ppm)', fontsize=12, fontweight='bold', color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    # Set x-axis
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=14)
    ax1.set_title('Homogeneity and Mean Field Comparison', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, shimmed_mean_mT):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height, 
                f'{val:.1f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='darkblue')
    
    for bar, val in zip(bars2, homogeneity_ppm):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height, 
                f'{val:.0f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='darkred')
    
    # Create proxy artists for legend with correct colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='dodgerblue', edgecolor='black', alpha=0.7, label='Field Mean (mT)'),
        Patch(facecolor='grey', edgecolor='black', alpha=0.7, label='Field Homogeneity (ppm)')
    ]
    
    # Add legend with proxy artists
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=11, 
               framealpha=0.95, edgecolor='black')
    
    # Adjust y-axis limits to create space at top for legend and bar labels
    y1_max = max(shimmed_mean_mT)
    y2_max = max(homogeneity_ppm)
    ax1.set_ylim(0, y1_max * 1.25)  # Add 25% space at top
    ax2.set_ylim(0, y2_max * 1.25)  # Add 25% space at top
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)  # Add space at top for legend
    return fig
def param_homogeneity_trendline(all_results, param, param_label, degree, param_extractor=None):
    """
    Plot homogeneity versus a single design parameter with regression trendline.

    all_results: results from `run_parameter_sweep()`
    param_name: Display name of current parameter 
    param_extractor: Function mapping a result dict; numeric parameter value
    order: Order for regression line (1 for linear, 2 for quadratic)

    outputs:
    fig: Figure containing the scatter plots and regression curves.
    """
    # Create a line plot with trendline showing gap vs homogeneity
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract parameter values - handle arrays
    params_raw = [r[param] for r in all_results]
    
    # If param_extractor is provided, use it; otherwise use mean for arrays
    if param_extractor is None:
        params = np.array([np.mean(p) if isinstance(p, np.ndarray) else p 
                          for p in params_raw])  # Convert to mm if array
    else:
        params = np.array([param_extractor(p) for p in params_raw])
    
    homogeneity = np.array([r['field_details']['Shimmed homogeneity (ppm)'] for r in all_results])

    # Plot actual data
    ax.plot(params, homogeneity, marker='o', linewidth=2, markersize=8, 
            color='steelblue', label='Measured', zorder=3)

    # Fit polynomial trendline using degree given:
    z = np.polyfit(params, homogeneity, degree)
    p = np.poly1d(z)

    # Generate smooth line for trendline
    params_smooth = np.linspace(params.min(), params.max(), 100)
    homogeneity_trend = p(params_smooth)

    # Plotting the trendline
    ax.plot(params_smooth, homogeneity_trend, '--', linewidth=2, 
            color='red', alpha=0.7, label='Trendline', zorder=2)

    # Calculate R²
    homogeneity_pred = p(params)
    ss_res = np.sum((homogeneity - homogeneity_pred) ** 2)
    ss_tot = np.sum((homogeneity - np.mean(homogeneity)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Format equation based on degree
    if degree == 1:
        equation_text = f'y = {z[0]:.2f}x + {z[1]:.2f}'
    else:
        equation_text = f'y = {z[0]:.2f}x² + {z[1]:.2f}x + {z[2]:.2f}'

    ax.text(0.04, 0.10, equation_text, transform=ax.transAxes,
            fontsize=16, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel(param_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Homogeneity (ppm)', fontsize=12, fontweight='bold')
    ax.set_title(f'Effect of {param_label} on Field Homogeneity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, h in zip(params, homogeneity):
        ax.text(i, h+200, f'{h:.0f}', ha='center', va='bottom', fontsize=12)

    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()
    print(f'\tR² = {r_squared:.4f}')
def param_field_strength_trendline(all_results, param, param_label, degree, param_extractor=None):
    """
    Plot field strength versus a single design parameter with regression trendline.

    all_results: results from `run_parameter_sweep()`
    param_name: Display name of current parameter 
    param_extractor: Function mapping a result dict; numeric parameter value
    order: Order for regression line (1 for linear, 2 for quadratic)

    outputs:
    fig: Figure containing the scatter plots and regression curves.
    """

    # Create a line plot with trendline showing gap vs field_strength
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract parameter values - handle arrays
    params_raw = [r[param] for r in all_results]
    
    # If param_extractor is provided, use it; otherwise use mean for arrays
    if param_extractor is None:
        params = np.array([np.mean(p) if isinstance(p, np.ndarray) else p 
                          for p in params_raw])
    else:
        params = np.array([param_extractor(p) for p in params_raw])
    
    # Convert field strength to mT
    field_strength = np.array([r['field_details']['Shimmed mean'] for r in all_results])

    # Plot actual data
    ax.plot(params, field_strength, marker='o', linewidth=2, markersize=8, 
            color='steelblue', label='Measured', zorder=3)

    # Fit polynomial trendline using degree given:
    z = np.polyfit(params, field_strength, degree)
    p = np.poly1d(z)

    # Generate smooth line for trendline
    params_smooth = np.linspace(params.min(), params.max(), 100)
    field_strength_trend = p(params_smooth)

    # Plotting the trendline
    ax.plot(params_smooth, field_strength_trend, '--', linewidth=2, 
            color='red', alpha=0.7, label='Trendline', zorder=2)

    # Calculate R²
    field_strength_pred = p(params)
    ss_res = np.sum((field_strength - field_strength_pred) ** 2)
    ss_tot = np.sum((field_strength - np.mean(field_strength)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Format equation based on degree
    if degree == 1:
        equation_text = f'y = {z[0]:.4f}x + {z[1]:.4f}'
    else:
        equation_text = f'y = {z[0]:.4f}x² + {z[1]:.4f}x + {z[2]:.4f}'

    ax.text(0.04, 0.10, equation_text, transform=ax.transAxes,
            fontsize=16, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel(param_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Field Strength (mT)', fontsize=12, fontweight='bold')
    ax.set_title(f'Effect of {param_label} on Field Strength', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add value labels with adaptive offset
    y_range = field_strength.max() - field_strength.min()
    offset = y_range * 0.02  # 2% of the range
    
    for i, h in zip(params, field_strength):
        ax.text(i, h + offset, f'{h:.2f}', ha='center', va='bottom', fontsize=10)

    ax.legend(fontsize=11, loc='best')
    plt.tight_layout()
    plt.show()
    print(f'\tR² = {r_squared:.4f}')
## **Altering Parameters:** Number of Rings
---
### First, keeping the array values **constant:**
numrings_constant_configs = []

for n in range(2, 20+1, 2):
    numrings_constant_configs.append({
        'name': f'{n}-ring (constant)',
        'numRings': n,
        'innerRingRadii': np.array([150] * int(n/2)) * 1e-3,
        'innerNumMagnets': np.array([51] * int(n/2)),
        'ringSep': 20
    })
# NumRings constant sweep
print("\nRunning numRings (constant radii) test sweep...")
numrings_const_results, numrings_const_field_data = load_results('numrings_constant_sweep')
if numrings_const_results is None:
    print("No saved results found. Running simulations...")
    numrings_const_results, numrings_const_field_data = run_parameter_sweep(
        halbach_array_simulation, 
        numrings_constant_configs,
        show_progress=True,
        save_fields=True
    )
    save_results(numrings_const_results, numrings_const_field_data, 'numrings_constant_sweep')
else:
    print("Using previously-saved results.")
param_homogeneity_trendline(numrings_const_results, 'numRings', 'Number of Rings (Constant)', 1)
param_homogeneity_trendline(numrings_const_results, 'numRings', 'Number of Rings (Constant)', 2)

param_field_strength_trendline(numrings_const_results, 'numRings', 'Number of Rings (Constant)', 1)
param_field_strength_trendline(numrings_const_results, 'numRings', 'Number of Rings (Constant)', 2)

plot_field_metrics_histogram(numrings_const_results)
plt.show()
## **Altering Parameters:** Radii Size
---
radii_test_configs = [
    {
        'name': 'radii (140-143)',
        'numRings': 4,
        'innerRingRadii': np.array([140, 143]) * 1e-3,
        'innerNumMagnets': np.array([48, 49]),
        'ringSep': 20
    },
    {
        'name': 'radii (142-145)',
        'numRings': 4,
        'innerRingRadii': np.array([142, 145]) * 1e-3,
        'innerNumMagnets': np.array([49, 50]),
        'ringSep': 20
    },
    {
        'name': 'radii (144-147)',
        'numRings': 4,
        'innerRingRadii': np.array([144, 147]) * 1e-3,
        'innerNumMagnets': np.array([49, 50]),
        'ringSep': 20
    },
    {
        'name': 'radii (146-149)',
        'numRings': 4,
        'innerRingRadii': np.array([146, 149]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 20
    },
    {
        'name': 'radii (148-151)',  # Baseline
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 20
    },
    {
        'name': 'radii (150-153)',
        'numRings': 4,
        'innerRingRadii': np.array([150, 153]) * 1e-3,
        'innerNumMagnets': np.array([51, 52]),
        'ringSep': 20
    },
    {
        'name': 'radii (152-155)',
        'numRings': 4,
        'innerRingRadii': np.array([152, 155]) * 1e-3,
        'innerNumMagnets': np.array([52, 53]),
        'ringSep': 20
    },
    {
        'name': 'radii (154-157)',
        'numRings': 4,
        'innerRingRadii': np.array([154, 157]) * 1e-3,
        'innerNumMagnets': np.array([53, 54]),
        'ringSep': 20
    },
    {
        'name': 'radii (156-159)',
        'numRings': 4,
        'innerRingRadii': np.array([156, 159]) * 1e-3,
        'innerNumMagnets': np.array([53, 54]),
        'ringSep': 20
    }
]
print("Running radii test sweep...")
radii_results, radii_field_data = load_results('radii_sweep')
if radii_results is None:
    print("No saved results found. Running simulations...")
    radii_results, radii_field_data = run_parameter_sweep(
        halbach_array_simulation, 
        radii_test_configs,
        show_progress=True,
        save_fields=True
    )
    save_results(radii_results, radii_field_data, 'radii_sweep')
else:
    print("Using previously-saved results.")
param_homogeneity_trendline(radii_results, 'innerRingRadii', 'Mean Ring Radius (m)', 1)
param_homogeneity_trendline(radii_results, 'innerRingRadii', 'Mean Ring Radius (m)', 2)

param_field_strength_trendline(radii_results, 'innerRingRadii', 'Mean Ring Radius (m)', 1)
param_field_strength_trendline(radii_results, 'innerRingRadii', 'Mean Ring Radius (m)', 2)

plot_field_metrics_histogram(radii_results)
plt.show()
## **Altering Parameters:** Gap Size
---
gap_test_configs = [
    # Baseline configuration with varying gap sizes
    {
        'name': 'gap 12mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 12  # Minimum reasonable/practical value
    },
    {
        'name': 'gap 13mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 13
    },
    {
        'name': 'gap 14mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 14
    },
    {
        'name': 'gap 15mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 15
    },
    {
        'name': 'gap 16mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 16
    },
    {
        'name': 'gap 17mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 17
    },
    {
        'name': 'gap 18mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 18
    },
    {
        'name': 'gap 19mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 19
    },
    {
        'name': 'gap 20mm', 
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 20 # Maximum reasonable/practical value
    },
    {
        'name': 'gap 21mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 21
    },
    {
        'name': 'gap 22mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 22
    },
    {
        'name': 'gap 23mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 23
    },
    {
        'name': 'gap 24mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 24
    },
    {
        'name': 'gap 25mm',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 25
    }
]
print("\nRunning gap test sweep...")
gap_results, gap_field_data = load_results('gap_sweep')

if gap_results is None:
    print("No saved results found. Running simulations...")
    gap_results, gap_field_data = run_parameter_sweep(
        halbach_array_simulation, 
        gap_test_configs,
        show_progress=True,
        save_fields=True
    )
    save_results(gap_results, gap_field_data, 'gap_sweep')
else:
    print("Using previously-saved results.")
param_homogeneity_trendline(gap_results, 'ringSep', 'Ring Separation (mm)', 1)
param_homogeneity_trendline(gap_results, 'ringSep', 'Ring Separation (mm)', 2)

param_field_strength_trendline(gap_results, 'ringSep', 'Ring Separation (mm)', 1)
param_field_strength_trendline(gap_results, 'ringSep', 'Ring Separation (mm)', 2)

plot_field_metrics_histogram(gap_results)
plt.show()
<!-- ### Second, defining the array values **linearly:** -->
## **Analysis of Parameter Combinations**
---
**NOTE:** These parameter lists are **not** automatically/procedurally generated. These are just different permuatations of values we manually inputted.
# Running multiple simulations with different parameters, and SAVING THE RESULTS (for comparison later)
different_param_passes = [
    # Baseline/original
    {
        'name': 'baseline 4-ring',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 20
    },
    {
        'name': 'small_gap 4-ring',
        'numRings': 4,
        'innerRingRadii': np.array([148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51]),
        'ringSep': 13
    },
    {
        'name': 'compact 6-ring',
        'numRings': 6,
        'innerRingRadii': np.array([145, 148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51, 52]),
        'ringSep': 20
    },
    {
        'name': 'spaced 6-ring',
        'numRings': 6,
        'innerRingRadii': np.array([145, 148, 151]) * 1e-3,
        'innerNumMagnets': np.array([50, 51, 52]),
        'ringSep': 25
    },
    {
        'name': 'wide_radii 6-ring',
        'numRings': 6,
        'innerRingRadii': np.array([140, 150, 160]) * 1e-3,
        'innerNumMagnets': np.array([48, 52, 56]),
        'ringSep': 22
    },
    {
        'name': 'optimal 8-ring',
        'numRings': 8,
        'innerRingRadii': np.array([140, 145, 150, 155]) * 1e-3,
        'innerNumMagnets': np.array([48, 50, 52, 54]),
        'ringSep': 25
    },
    {
        'name': 'compact 8-ring',
        'numRings': 8,
        'innerRingRadii': np.array([140, 145, 150, 155]) * 1e-3,
        'innerNumMagnets': np.array([48, 50, 52, 54]),
        'ringSep': 20
    },
    {
        'name': 'spaced 8-ring',
        'numRings': 8,
        'innerRingRadii': np.array([140, 145, 150, 155]) * 1e-3,
        'innerNumMagnets': np.array([48, 50, 52, 54]),
        'ringSep': 18 
    },
    {
        'name': 'linear_radii 6-ring',
        'numRings': 6,
        'innerRingRadii': np.array([142, 146, 150, 154, 158, 162]) * 1e-3,
        'innerNumMagnets': np.array([49, 50, 51, 52, 53, 54]),
        'ringSep': 22
    },
    {
        'name': 'constant_radii 6-ring',
        'numRings': 6,
        'innerRingRadii': np.array([150, 150, 150]) * 1e-3,
        'innerNumMagnets': np.array([51, 51, 51]),
        'ringSep': 22
    }
]
Running the simulation with each list of parameters:\
*Using saved results if they exist, just to save some time...*
def plot_parameter_pairs(results_list):
    """
    Creates two separate figures showing parameter relationships:
    Figure 1: Colored by Shimmed homogeneity (ppm), Figure 2: Colored by Shimmed mean (mT)
    
    Each figure contains three scatter plots showing
    * Number of Rings vs Ring Separation
    * Number of Rings vs Mean Inner Ring Radius
    * Ring Separation vs Mean Inner Ring Radius

    all_results: simulation results including 'numRings', 'innerRingRadii_mm', and 'ringSep'

    outputs:
    fig: Figure containing the parameter-pair scatterplot matrix
    """
    num_rings = []
    mean_radii = []
    ring_sep = []
    homogeneity_ppm = []
    shimmed_mean_mT = []
    
    for result in results_list:
        num_rings.append(result['numRings'])
        mean_radii.append(np.mean(result['innerRingRadii']) * 1e3)
        ring_sep.append(result['ringSep'])
        homogeneity_ppm.append(result['field_details']['Shimmed homogeneity (ppm)'])
        shimmed_mean_mT.append(result['field_details']['Shimmed mean'])
    
    # Convert to numpy arrays
    num_rings = np.array(num_rings)
    mean_radii = np.array(mean_radii)
    ring_sep = np.array(ring_sep)
    homogeneity_ppm = np.array(homogeneity_ppm)
    shimmed_mean_mT = np.array(shimmed_mean_mT)
    
    # ========== FIGURE 1: Homogeneity (ppm) ==========
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Color mapping based on homogeneity (lower is better)
    norm1 = plt.Normalize(vmin=homogeneity_ppm.min(), vmax=homogeneity_ppm.max())
    cmap1 = matplotlib.cm.get_cmap('RdYlGn_r')  # Red (bad) to Green (good), reversed
    
    # Plot 1: numRings vs ringSep
    scatter1_1 = axes1[0].scatter(num_rings, ring_sep, 
                                   c=homogeneity_ppm, s=150, 
                                   cmap=cmap1, norm=norm1,
                                   edgecolors='black', linewidth=1.5, alpha=0.8)
    axes1[0].set_xlabel('Number of Rings', fontsize=12, fontweight='bold')
    axes1[0].set_ylabel('Ring Separation (mm)', fontsize=12, fontweight='bold')
    axes1[0].set_title('Homogeneity vs Ring Configuration', fontsize=13, fontweight='bold')
    axes1[0].grid(True, alpha=0.3)
    
    for i, (x, y, h) in enumerate(zip(num_rings, ring_sep, homogeneity_ppm)):
        axes1[0].annotate(f'{h:.0f}', (x, y), 
                          textcoords="offset points", xytext=(0,10), 
                          ha='center', fontsize=9, fontweight='bold')
    
    # Plot 2: numRings vs mean innerRingRadii
    scatter1_2 = axes1[1].scatter(num_rings, mean_radii, 
                                   c=homogeneity_ppm, s=150, 
                                   cmap=cmap1, norm=norm1,
                                   edgecolors='black', linewidth=1.5, alpha=0.8)
    axes1[1].set_xlabel('Number of Rings', fontsize=12, fontweight='bold')
    axes1[1].set_ylabel('Mean Inner Ring Radius (mm)', fontsize=12, fontweight='bold')
    axes1[1].set_title('Homogeneity vs Ring Count & Radius', fontsize=13, fontweight='bold')
    axes1[1].grid(True, alpha=0.3)
    
    for i, (x, y, h) in enumerate(zip(num_rings, mean_radii, homogeneity_ppm)):
        axes1[1].annotate(f'{h:.0f}', (x, y), 
                          textcoords="offset points", xytext=(0,10), 
                          ha='center', fontsize=9, fontweight='bold')
    
    # Plot 3: ringSep vs mean innerRingRadii
    scatter1_3 = axes1[2].scatter(ring_sep, mean_radii, 
                                   c=homogeneity_ppm, s=150, 
                                   cmap=cmap1, norm=norm1,
                                   edgecolors='black', linewidth=1.5, alpha=0.8)
    axes1[2].set_xlabel('Ring Separation (mm)', fontsize=12, fontweight='bold')
    axes1[2].set_ylabel('Mean Inner Ring Radius (mm)', fontsize=12, fontweight='bold')
    axes1[2].set_title('Homogeneity vs Separation & Radius', fontsize=13, fontweight='bold')
    axes1[2].grid(True, alpha=0.3)
    
    for i, (x, y, h) in enumerate(zip(ring_sep, mean_radii, homogeneity_ppm)):
        axes1[2].annotate(f'{h:.0f}', (x, y), 
                          textcoords="offset points", xytext=(0,10), 
                          ha='center', fontsize=9, fontweight='bold')
    
    # Add colorbar for Figure 1
    fig1.subplots_adjust(right=0.92)
    cbar_ax1 = fig1.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar1 = fig1.colorbar(scatter1_3, cax=cbar_ax1)
    cbar1.set_label('Field Homogeneity (ppm)', rotation=270, labelpad=25, 
                    fontsize=12, fontweight='bold')
    
    plt.suptitle('Parameter Effects on Field Homogeneity', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    
    # ========== FIGURE 2: Shimmed Mean (mT) ==========
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Color mapping based on shimmed mean (higher is better)
    norm2 = plt.Normalize(vmin=shimmed_mean_mT.min(), vmax=shimmed_mean_mT.max())
    cmap2 = matplotlib.cm.get_cmap('RdYlGn')  # Red (low) to Green (high)
    
    # Plot 1: numRings vs ringSep
    scatter2_1 = axes2[0].scatter(num_rings, ring_sep, 
                                   c=shimmed_mean_mT, s=150, 
                                   cmap=cmap2, norm=norm2,
                                   edgecolors='black', linewidth=1.5, alpha=0.8)
    axes2[0].set_xlabel('Number of Rings', fontsize=12, fontweight='bold')
    axes2[0].set_ylabel('Ring Separation (mm)', fontsize=12, fontweight='bold')
    axes2[0].set_title('Mean Field vs Ring Configuration', fontsize=13, fontweight='bold')
    axes2[0].grid(True, alpha=0.3)
    
    for i, (x, y, m) in enumerate(zip(num_rings, ring_sep, shimmed_mean_mT)):
        axes2[0].annotate(f'{m:.1f}', (x, y), 
                          textcoords="offset points", xytext=(0,10), 
                          ha='center', fontsize=9, fontweight='bold')
    
    # Plot 2: numRings vs mean innerRingRadii
    scatter2_2 = axes2[1].scatter(num_rings, mean_radii, 
                                   c=shimmed_mean_mT, s=150, 
                                   cmap=cmap2, norm=norm2,
                                   edgecolors='black', linewidth=1.5, alpha=0.8)
    axes2[1].set_xlabel('Number of Rings', fontsize=12, fontweight='bold')
    axes2[1].set_ylabel('Mean Inner Ring Radius (mm)', fontsize=12, fontweight='bold')
    axes2[1].set_title('Mean Field vs Ring Count & Radius', fontsize=13, fontweight='bold')
    axes2[1].grid(True, alpha=0.3)
    
    for i, (x, y, m) in enumerate(zip(num_rings, mean_radii, shimmed_mean_mT)):
        axes2[1].annotate(f'{m:.1f}', (x, y), 
                          textcoords="offset points", xytext=(0,10), 
                          ha='center', fontsize=9, fontweight='bold')
    
    # Plot 3: ringSep vs mean innerRingRadii
    scatter2_3 = axes2[2].scatter(ring_sep, mean_radii, 
                                   c=shimmed_mean_mT, s=150, 
                                   cmap=cmap2, norm=norm2,
                                   edgecolors='black', linewidth=1.5, alpha=0.8)
    axes2[2].set_xlabel('Ring Separation (mm)', fontsize=12, fontweight='bold')
    axes2[2].set_ylabel('Mean Inner Ring Radius (mm)', fontsize=12, fontweight='bold')
    axes2[2].set_title('Mean Field vs Separation & Radius', fontsize=13, fontweight='bold')
    axes2[2].grid(True, alpha=0.3)
    
    for i, (x, y, m) in enumerate(zip(ring_sep, mean_radii, shimmed_mean_mT)):
        axes2[2].annotate(f'{m:.1f}', (x, y), 
                          textcoords="offset points", xytext=(0,10), 
                          ha='center', fontsize=9, fontweight='bold')
    
    # Add colorbar for Figure 2
    fig2.subplots_adjust(right=0.92)
    cbar_ax2 = fig2.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar2 = fig2.colorbar(scatter2_3, cax=cbar_ax2)
    cbar2.set_label('Field Mean (mT)', rotation=270, labelpad=25, 
                    fontsize=12, fontweight='bold')
    
    plt.suptitle('Parameter Effects on Mean Field Strength', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    
    return fig1, fig2
print("Running main parameter sweep...")
results_list, field_data_list = load_results('main_params_sweep')

if results_list is None:
    print("No saved results found. Running simulations...")
    results_list, field_data_list = run_parameter_sweep(
        halbach_array_simulation, 
        different_param_passes,
        show_progress=True,
        save_fields=True
    )
    save_results(results_list, field_data_list, 'main_params_sweep')
else:
    print("Using previously-saved results.")
# Create pair-plots
fig_params = plot_parameter_pairs(results_list)
plt.show()
fig = plot_field_metrics_histogram(results_list)
plt.show()
optimal_result = halbach_array_simulation(8, np.array([140, 145, 150, 155]) * 1e-3, np.array([48, 50, 52, 54]), 25, True)
def plot_field_strength_vs_homogeneity(all_results):
    """
    Plot mean magnetic field strength versus homogeneity for each configuration.

    Each simulation appears as a single point. This visualization highlights
    parameter combinations that maximize field strength while minimizing homogeneity.

    all_results: Simulation result entries containing `field_details`.

    outputs:
    fig: Scatter plot showing strength–homogeneity trade-off.
    """
    # Extract data - handle both formats
    field_strength = []
    homogeneity_ppm = []
    names = []
    
    for r in all_results:
        # Check if it's from run_parameter_sweep
        if 'field_details' in r:
            field_strength.append(r['field_details']['Shimmed mean'])
            homogeneity_ppm.append(r['field_details']['Shimmed homogeneity (ppm)'])
            names.append(r.get('config_name', r.get('name', 'Unknown')))
        # Or from run_simulation (one simulation's format for output isn't a dictionary, just one list)
        else:
            field_strength.append(r['mean_field_mT'])
            homogeneity_ppm.append(r['homogeneity_ppm'])
            names.append(r['name'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Find best configuration (lowest homogeneity)
    best_idx = homogeneity_ppm.index(min(homogeneity_ppm))
    
    # Create scatter plot with different colors
    colors = ['green' if i == best_idx else 'steelblue' for i in range(len(names))]
    scatter = ax.scatter(field_strength, homogeneity_ppm, 
                        s=200, c=colors, alpha=0.6, 
                        edgecolors='black', linewidth=2)
    
    # Add labels for each point
    for i, (x, y, name) in enumerate(zip(field_strength, homogeneity_ppm, names)):
        # Offset the text slightly to avoid overlapping with points
        offset_x = 10
        offset_y = 10 + (i % 5) * 8
        
        # Adjust offset for best configuration
        if i == best_idx:
            ax.annotate(name, (x, y), 
                       xytext=(0.05, 25), 
                       textcoords='offset points',
                       fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor='lightgreen', 
                                edgecolor='green', 
                                linewidth=2, alpha=0.8),
                       arrowprops=dict(arrowstyle='->', 
                                      connectionstyle='arc3,rad=0',
                                      color='green', linewidth=2))
        else:
            ax.annotate(name, (x, y), 
                       xytext=(offset_x, offset_y), 
                       textcoords='offset points',
                       fontsize=16,
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                alpha=0.7),
                       arrowprops=dict(arrowstyle='->', 
                                      connectionstyle='arc3,rad=0.2',
                                      alpha=0.5))
    
    # Labels and formatting
    ax.set_xlabel('Field Strength (mT)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Homogeneity (ppm)', fontsize=18, fontweight='bold')
    ax.set_title('Field Strength vs Homogeneity Trade-off', 
                fontsize=20, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference lines for the best configuration
    ax.axhline(y=min(homogeneity_ppm), color='green', 
              linestyle='--', alpha=0.3, linewidth=2, 
              label=f'Best Homogeneity: {min(homogeneity_ppm):.0f} ppm')
    ax.axvline(x=field_strength[best_idx], color='green', 
              linestyle='--', alpha=0.3, linewidth=2,
              label=f'Best Field: {field_strength[best_idx]:.2f} mT')
    
    # Add legend
    ax.legend(loc='upper right', fontsize=16, framealpha=0.9)
    
    plt.tight_layout()
    return fig
fig1 = plot_field_strength_vs_homogeneity(results_list)
plt.show()
