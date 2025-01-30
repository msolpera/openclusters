from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
from galpy.potential import (
    MWPotential2014, 
    NFWPotential, 
    MiyamotoNagaiPotential,
    PowerSphericalPotentialwCutoff, 
    plotRotcurve
)

def load_and_filter_data(asteca_file, vphi_file):
    """Load and filter the data files."""
    asteca = ascii.read(asteca_file)
    data = ascii.read(vphi_file)
    
    # Filter data
    msk_rv = data['Vphi_final'] != '--'
    asteca = asteca[msk_rv]
    
    # Extract and process v_phi values
    v_phi = np.array([abs(float(i)) for i in data['Vphi_final'][msk_rv]])
    r = data['Rgc'][msk_rv]
    
    return asteca, v_phi, r

def calculate_velocity_statistics(r, v_phi, n_bins=20):
    """Calculate velocity statistics in radial bins."""
    r_bins = plt.hist(r, bins=n_bins)[1][4:14]  # Get bin edges
    plt.close()  # Close the temporary histogram
    
    velocities = []
    errors = []
    
    for i in range(len(r_bins)):
        if i == len(r_bins) - 1:
            mask = r >= r_bins[i]
        else:
            mask = (r < r_bins[i+1]) & (r >= r_bins[i])
            
        v_phi_filtered = abs(v_phi[mask])
        velocities.append(np.median(v_phi_filtered))
        errors.append(np.std(v_phi_filtered))
        
    return r_bins, np.array(velocities), np.array(errors)

def create_potentials(ro=8.34, vo=240.):
    """Create galactic potentials."""
    return {
        'Miyamoto-Nagai': MiyamotoNagaiPotential(
            a=3./8., b=0.28/8., normalize=.6, ro=ro, vo=vo
        ),
        'NFW': NFWPotential(
            a=16/8., normalize=.35, ro=ro, vo=vo
        ),
        'PowerSpherical': PowerSphericalPotentialwCutoff(
            alpha=1.8, rc=1.9/8., normalize=0.05, ro=ro, vo=vo
        )
    }

def get_rotation_curves(potentials, r_range=(0.1, 16), n_points=1001):
    """Calculate rotation curves for different potentials."""
    curves = {}
    
    # Add MWPotential2014
    curves['MW2014'] = plotRotcurve(
        MWPotential2014, Rrange=r_range, grid=n_points, 
        overplot=True
    )[0].get_xydata()
    
    # Add other potentials
    for name, potential in potentials.items():
        curves[name] = plotRotcurve(
            potential, Rrange=r_range, grid=n_points, 
            overplot=True
        )[0].get_xydata()
        
    return curves

def plot_rotation_curves(r_bins, velocities, velocity_errors, curves):
    """Create the final plot with all rotation curves."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot observed data
    ax.errorbar(r_bins, velocities, velocity_errors, 
                capsize=2, label='Este trabajo', zorder=10)
    ax.scatter(r_bins, velocities, s=6)
    
    # Plot rotation curves
    styles = {
        'MW2014': {'color': 'orange', 'ls': '-', 
                   'label': r'$\mathrm{MWPotential2014}$'},
        'Miyamoto-Nagai': {'color': 'purple', 'ls': '--', 
                          'label': r'$\mathrm{MiyamotoNagaiPotential}$'},
        'PowerSpherical': {'color': 'lightblue', 'ls': '--', 
                          'label': r'$\mathrm{PowerSphericalPotentialwCutoff}$'},
        'NFW': {'color': 'gray', 'ls': '--', 
                'label': r'$\mathrm{NFWPotential}$'}
    }
    
    for name, curve in curves.items():
        ax.plot(curve[:, 0], curve[:, 1], lw=2, **styles[name])
    
    # Configure plot
    ax.set_xlim(0.1, 16)
    ax.set_ylim(25, 350)
    ax.set_xlabel(r'$R_{GC}[kpc]$')
    ax.set_ylabel(r'$V_{\phi} [kms^{-1}]$')
    ax.legend(loc='upper left')
    
    return fig, ax

def main():
    # Set plotting style
    plt.style.use(['science', 'no-latex'])
    
    # Load and process data
    asteca, v_phi, r = load_and_filter_data(
        '../asteca_output_final_actualizado.dat',
        'vphi_rv.dat'
    )
    
    # Calculate velocity statistics
    r_bins, velocities, errors = calculate_velocity_statistics(r, v_phi)
    
    # Create potentials and get rotation curves
    potentials = create_potentials()
    curves = get_rotation_curves(potentials)
    
    # Create plot
    fig, ax = plot_rotation_curves(r_bins, velocities, errors, curves)
    
    # Save and show plot
    plt.savefig('potentials_std.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
