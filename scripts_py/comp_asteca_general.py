import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii

def load_data(filename, name_col='NAME', delimiter=' '):
    """Load data from file and convert names to lowercase if specified"""
    data = ascii.read(filename, delimiter=delimiter)
    if name_col in data.colnames:
        data[name_col] = [i.lower() for i in data[name_col]]
    return data

def calculate_distance(dm):
    """Calculate distance in parsecs from distance modulus"""
    return 10 ** (dm/5) * 10

def process_data(asteca_data, comparison_data, comparison_cols, comparison_cols_e, asteca_cols, asteca_cols_e):
    results = {
        'comparison': [],
        'asteca': [],
        'asteca_errors': [],
        'comparison_errors': [],
        'n_stars': []
    }

    for cluster in asteca_data['NAME']:
        try:
            comp_mask = comparison_data[comparison_cols['NAME']] == cluster
            if not any(comp_mask):
                continue
            # Get comparison values
            comp_values = [comparison_data[col][comp_mask][0] for col in comparison_cols]
            if any(val == '---' for val in comp_values):
                continue
            
            # Get ASteCA values
            asteca_mask = asteca_data['NAME'] == cluster
            asteca_values = [asteca_data[col][asteca_mask][0] for col in asteca_cols]
            asteca_lower = [asteca_data[col][asteca_mask][0] for col in asteca_cols_e[0]]
            asteca_upper = [asteca_data[col][asteca_mask][0] for col in asteca_cols_e[1]]
            
            results['comparison'].append(comp_values)
            results['asteca'].append(asteca_values)
            results['asteca_errors'].append([
                [v - l for v, l in zip(asteca_values, asteca_lower)],
                [u - v for v, u in zip(asteca_values, asteca_upper)]
            ])
            results['n_stars'].append(asteca_data['N_cl'][asteca_mask][0])
            
            # Add comparison errors if available
            for error in comparison_cols_e:
                if len(error)>1:
                    results['comparison_errors'].append([
                    [v - l for v, l in zip(comp_values, error[0])],
                    [u - v for v, u in zip(asteca_values, error[-1])]
                    ])
                else:
                    results['comparison_errors'].append.append([
                    [v - l, v + l] for v, l in zip(comp_values, error)
                    ])
            
        except IndexError:
            continue
            
    return {k: np.array(v) for k, v in results.items()}

def main():
    # Load data
    # All files must have a column named 'NAME' as the header.
    asteca = load_data('../final_table_goodage_actualizado.dat')
    cg = load_data('./cg2020.dat')
    hunt = load_data('./hunt_2023.csv', delimiter=';')
    cavallo = load_data('./cavallo23.csv')
    dias = load_data('./dias.dat')

    asteca['dm_median'] = calculate_distance(asteca['dm_median'])
    cavallo['dMod_50'] = calculate_distance(cavallo['dMod_50'])


    asteca_cols = ['dm_median','a_median','Av_median']
    comparison_data = [cg, hunt, cavallo, dias]
    comparison_cols = [['D_pc','Age','A'], ['dist50', 'age50', 'av50'], ['dMod_50', 'logAge_50', 'Av_50'],\
                       ['dist', 'age', 'av']]
    comparison_cols_errors = [[],[['dist16', 'dist84'], ['age16', 'age84'],['av16', 'av84']], \
                              [['dMod_16', 'dMod84'], ['logAge_16','logAge_84'], ['Av_16','Av_84']], ['e_dist','e_age', 'e_av']]
    asteca_cols_e = [['dm_16th','dm_84th'] ,['a_16th','a_84th'],['Av_16th','Av_84th']]


    plt.style.use(['science', 'no-latex'])
    fig, axs = plt.subplots(4, 3, figsize=(10, 12),layout='constrained')
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)


    for fila, comparison_data in enumerate(comparison_data):
        data = process_data(asteca, comparison_data, comparison_cols_errors, asteca_cols, asteca_cols_e)
        for i in range(3):
            axs[fila,i].scatter(
            data['comparison'][:, i], 
            data['asteca'][:, i],
            s=16,
            edgecolor='none',
            c=np.log(data['n_stars']),
            cmap='plasma',
            norm=plt.Normalize(vmin=min(np.log(data['n_stars'])), vmax=max(np.log(data['n_stars'])))
            )   
            axs[fila,i].errorbar(
                data['comparison'][:, i],
                data['asteca'][:, i],
                xerr=data['comparison_errors'],
                yerr=data['asteca_errors'],
                ls='none',
                c='gray',
                alpha=0.7,
                zorder=-1
            )

            # Add reference line 
            axs[fila,i].axline((0, 0), slope=1, color='black', ls='--')


            # Calculate and add statistics
            rmse = np.sqrt(np.mean((data['comparison'][:, i] - data['asteca'][:, i])**2))
            r2 = np.corrcoef(data['comparison'][:, i], data['asteca'][:, i])[0, 1]**2
            stats_text = f'$RMSE = {rmse:.2f}$\n$R^2 = {r2:.2f}$'
            axs[fila,i].text(0.1, 0.8, stats_text, fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
                    transform=axs[fila,i].transAxes)
    
    plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()