import numpy as np
from galpy import potential
from galpy.orbit import Orbit
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import CartesianDifferential
import matplotlib.pyplot as plt
from astropy.io import ascii
from galpy.potential import (MWPotential2014, SpiralArmsPotential, DehnenBarPotential,
                           MiyamotoNagaiPotential, NFWPotential, SoftenedNeedleBarPotential, 
                           PlummerPotential)
from galpy.util import conversion

def setup_potentials():
    """Set up galactic potentials for orbit calculations."""
    # Define components of the custom potential
    disk_pot = MiyamotoNagaiPotential(
        amp=6e10/conversion.mass_in_msol(240., 8.),
        a=3.*u.kpc,
        b=0.28*u.kpc
    )
    halo_pot = NFWPotential(a=2., normalize=0.35, ro=8., vo=240.)
    bar_pot = SoftenedNeedleBarPotential(
        amp=1e10/conversion.mass_in_msol(240., 8.),
        a=3.5*u.kpc,
        c=1.*u.kpc,
        omegab=1.85
    )
    nuclear_pot = PlummerPotential(
        amp=2e9/conversion.mass_in_msol(240., 8.),
        b=250.*u.pc
    )
    spiral_pot = SpiralArmsPotential(
        N=2, amp=0.15, phi_ref=25.*u.deg,
        alpha=15.*u.deg, omega=2./3.,
        ro=8., vo=240.
    )
    
    return [disk_pot, halo_pot, nuclear_pot, bar_pot, spiral_pot], MWPotential2014

def create_orbit(cluster_data, velocity_data, cluster_name):
    """Create an orbit object for a given cluster."""
    msk = cluster_data['NAME'] == cluster_name
    
    # Set up solar position and velocity
    s_xyz = SkyCoord(-8.34, 0., 0.027, unit='kpc', representation_type='cartesian')
    v_sun = [11.1, 240+12.24, 7.25] * (u.km / u.s)
    
    # Get cluster parameters
    gc = SkyCoord(l=cluster_data['GLON'][msk]*u.degree, 
                  b=cluster_data['GLAT'][msk]*u.degree, 
                  frame='galactic')
    
    dist_pc = 10**(cluster_data['dm_median'][msk][0]/5)*10
    pmra = cluster_data['pmRA'][msk][0]
    pmde = cluster_data['pmDE'][msk][0]
    vr = float(velocity_data['rv_final'][velocity_data['name'] == cluster_name][0])
    
    # Create SkyCoord object
    coord = SkyCoord(
        ra=gc.fk5.ra.value[0]*u.deg,
        dec=gc.fk5.dec.value[0]*u.deg,
        distance=dist_pc*u.pc,
        pm_ra_cosdec=pmra*u.mas/u.yr,
        pm_dec=pmde*u.mas/u.yr,
        radial_velocity=vr*u.km/u.s,
        galcen_distance=-s_xyz.x,
        z_sun=s_xyz.z,
        galcen_v_sun=v_sun
    )
    
    return Orbit(coord, vo=240)

def calculate_integration_time(cluster_data, cluster_name):
    """Calculate integration time based on cluster age."""
    age = 10**cluster_data['a_median'][cluster_data['NAME'] == cluster_name][0]*u.yr.to(u.Gyr)
    return max(2, age)

def plot_orbits(fig, ax, orbit1, orbit2, t, cluster_name, plot_pos):
    """Plot orbits in x-y and r-z planes."""
    n, m = plot_pos
    
    # Calculate orbital positions
    x1, y1 = orbit1.x(t), orbit1.y(t)
    r1, z1 = orbit1.r(t), orbit1.z(t)
    x2, y2 = orbit2.x(t), orbit2.y(t)
    r2, z2 = orbit2.r(t), orbit2.z(t)
    
    # Plot x-y projection
    ax[n,m].plot(x1, y1, label='Bar+Spiral Arms')
    ax[n,m].plot(x2, y2, label='MWPotential2014')
    ax[n,m].scatter([orbit1.x()], [orbit1.y()], marker="o", color='white', edgecolor='k', zorder=1)
    ax[n,m].legend()
    ax[n,m].set_title(cluster_name.upper())
    ax[n,m].set_xlabel('x [kpc]')
    ax[n,m].set_ylabel('y [kpc]')
    
    # Plot r-z projection
    ax[n,m+1].plot(r1, z1, color='orange', label='Bar+Spiral Arms')
    ax[n,m+1].plot(r2, z2, color='orange', label='MWPotential2014')
    ax[n,m+1].scatter([orbit1.r()], [orbit1.z()], marker="o", color='white', edgecolor='k', zorder=1)
    ax[n,m+1].legend()
    ax[n,m+1].set_title(cluster_name.upper())
    ax[n,m+1].set_xlabel('r [kpc]')
    ax[n,m+1].set_ylabel('z [kpc]')

def main():
    # Read data
    data = ascii.read('../final_table_goodage_actualizado.dat')
    vel = ascii.read('./RV_pyup_gaia_final.dat')
    
    # Define cluster list
    clusters = ['ngc_2682', 'ngc_2516', 'ngc_2506', 'ngc_2420', 
               'ngc_3680', 'trumpler_19', 'trumpler_20', 'haffner_22']
    
    # Setup plot
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    
    # Setup potentials
    custom_pot, mw_pot = setup_potentials()

    position = [(i,j) for i in range(4) for j in range(4)]
    # Process each cluster
    for i, cluster in enumerate(clusters):
        
        # Create orbit objects
        orbit = create_orbit(data, vel, cluster)
        orbit2 = create_orbit(data, vel, cluster)
        
        # Calculate integration time
        t_final = calculate_integration_time(data, cluster)
        t = np.linspace(0, -t_final, 10000)*u.Gyr
        
        # Integrate orbits
        orbit.integrate(t, custom_pot)
        orbit2.integrate(t, mw_pot)
        
        # Plot results
        plot_orbits(fig, ax, orbit, orbit2, t, cluster, positions)
    
    plt.tight_layout()
    plt.savefig('orbits_clusters_zaltos_spiral.png', dpi=300)

if __name__ == "__main__":
    main()
