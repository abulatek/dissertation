import numpy as np
import pyspeckit
from pyspeckit.spectrum.models.lte_molecule import get_molecular_parameters
from pyspeckit.spectrum.models import lte_molecule
from astropy import log, constants, units as u
from astropy.table import QTable
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from astropy.stats import mad_std
from pyspeckit.spectrum.models.lte_molecule import nupper_of_kkms
import pandas
pandas.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
log.setLevel('ERROR')
from tqdm.auto import tqdm
# from spectral_cube import SpectralCube
# Alternative way to read in spectra (with spectral-cube): 
# https://spectral-cube.readthedocs.io/en/latest/api/spectral_cube.lower_dimensional_structures.OneDSpectrum.html

def plot_full_model(zone, moltbl_fn, figdir, specfns, outdir, write_models=False, plot_noise_range=True, save_figs=False, show_figs=True):
    spectra = []
    c_km_s = (constants.c).to(u.km/u.s)
    # Read in the spectra
    for fn in specfns:
        spec = pyspeckit.Spectrum(fn)
        spectra.append(spec)
    # Reorder spectra from lowest to highest frequency
    lowestfreqs = [np.min(spec.xarr).value for spec in spectra]
    correct_order = np.argsort(lowestfreqs)
    spectra_inorder = [spectra[i] for i in correct_order]
    # Read in the molecule table
    tbl = pandas.read_csv(moltbl_fn)
    tbl = tbl[tbl['Index'].notnull()] # Only consider molecules we've found in catalogs
    # Extract info from table, and make sure the provided zone is valid
    supported_zones = ['core','frown','nw']
    if type(zone) is not str:
        raise TypeError("The parameter zone must be a string")
    elif zone in supported_zones:
        tbl, inlocs, molnames, specieses, specieses_ltx, catalogs, moltags, v_cens, v_disps, temps, N_tots = extract_from_tbl(zone, tbl)
    elif zone not in supported_zones:
        raise ValueError(f"The provided zone is not supported (must be in {supported_zones})")
    # Initialize dict to build into an exportable table later
    tbl_cols = ['mn_tbl', 'spc_tbl', 'spc_ltx_tbl', 'cat_tbl', 'mt_tbl', 'freq_tbl', 'freq_rs_tbl', 'EU_K_tbl', 'aij_tbl', 'qn_tbl', 'peak_tbl',
                'peak_val', 'intint_tbl', 'intint_val', 'nupper_deg_tbl', 'in_aces_tbl']
    tbl_dict = {key:[] for key in tbl_cols}
    # Gather and plot data and models
    freq_centers_arr = []
    sigma_arr = []
    for spec in tqdm(spectra_inorder): # If you want the code to run faster, only run a subset of spws through the code by changing what part of spectra_inorder you use
        freq_labels, name_labels = [], []
        spec.xarr.convert_to_unit(u.GHz)
        # noise = spec.data.std()
        noise = mad_std(spec.data)
        mod = np.zeros(len(spec.xarr))
        # Loop through each molecule
        for (inloc, molname, species, species_ltx, catalog, moltag, v_cen, v_disp, temp, N_tot) in zip(inlocs, molnames, specieses, specieses_ltx, catalogs, moltags, v_cens, v_disps, temps, N_tots):
            if catalog in ['JPL', 'CDMS']:
                try:
                    freqs, aij, deg, EU, partfunc, tbl = get_molecular_parameters(molecule_name=None, molecule_tag=moltag, catalog=catalog,
                                                                                  parse_name_locally=False, return_table=True,
                                                                                  fmin=spec.xarr.min(),
                                                                                  fmax=spec.xarr.max()*(1+70/3e5),
                                                                                  fallback_to_getmolecule=True) # The 70 km/s is to ensure redshifted lines get plotted
                except TypeError as te:
                    continue
                except:
                    continue
            else:
                raise ValueError("The provided catalog is not supported (must be either CDMS or JPL)") 
            freqs_to_keep = (freqs > spec.xarr.min()) & (freqs < spec.xarr.max()) # Get boolean array of indices within bound for that cube
            freqs = freqs[freqs_to_keep] 
            aij = aij[freqs_to_keep]
            deg = deg[freqs_to_keep]
            EU = EU[freqs_to_keep]
            tbl = tbl[freqs_to_keep]
            # Plot lines of that molecule
            if any(loc in inloc for loc in ['Y', 'SA', 'ST', 'RS']) and len(freqs) > 0:
                mod_new = lte_molecule.generate_model(spec.xarr.value*spec.xarr.unit, float(v_cen)*u.km/u.s, float(v_disp)*u.km/u.s, 
                                                      float(temp), float(N_tot), freqs, aij, deg, EU, partfunc)
                # Check if model is producing all NaNs
                if not np.isnan(mod_new).all():
                    # if inloc == 'Y': # Only add this line to the model if the molecule is not self-absorbed or subthermally excited or due to residual signal
                    mod = np.add(mod, mod_new)
                else:
                    print(f"This molecule is producing all NaNs: {species}")

                tbl.add_column(aij, name='A_ij', index=2)
                for line in tbl:
                    freq = (line['FREQ']*u.MHz).to(u.GHz) # Extract the frequency of the line
                    lgint = line['LGINT']
                    lines = tbl[(tbl['FREQ'] > (freq-0.01*u.GHz)) & (tbl['FREQ'] < (freq+0.01*u.GHz))] # Sample lines nearby, within 0.02 GHz (for reducing HFS line crowding)
                    if (lgint >= np.max(lines['LGINT'])) or (species in ['CH3CN', 'CH3CCH']): # Only label if all lines of that species within 0.02 GHz have lower or equal LGINT, or if it's a ladder molecule
                        freq_rs = freq*(1-((float(v_cen)*u.km/u.s)/c_km_s)) # Calculate redshifted frequency
                        val = mod_new[np.argmin(np.abs(spec.xarr.to(u.GHz) - freq_rs))] # Find the model value closest to the redshifted frequency
                        if abs(val > noise/10.): # If the model has signal above 1/10 of the noise, count that line in labeling (should be for detections, SA ULs, and ST)
                            freq_labels.append(freq_rs) # Label at the redshifted (observed) frequency
                            # print(f'{species_ltx}, v_cen = {v_cen}, new freq: {freq_rs}')
                            if 'Y' in inloc:
                                name_labels.append(species_ltx)
                            elif 'SA' in inloc:
                                name_labels.append(species_ltx+' (SA)')
                            elif 'ST' in inloc:
                                name_labels.append(species_ltx+' (ST)')
                            elif 'RS' in inloc:
                                name_labels.append(species_ltx+' (ext.)')
                            if 'Y' in inloc: # Only put the line in the table if it's a detection
                                peak = spec.data[np.argmin(np.abs(spec.xarr.to(u.GHz) - freq_rs))]*spec.unit # Find data value closest to the redshifted frequency
                                # if peak >= 2.*noise*spec.unit: # Further, only put the line in the table if it's not an "UL" line (like in the rotational diagrams)
                                tbl_dict['mn_tbl'].append(molname) # Name of molecule
                                tbl_dict['spc_tbl'].append(species) # Species
                                tbl_dict['spc_ltx_tbl'].append(species_ltx) # Species in LaTeX format
                                tbl_dict['cat_tbl'].append(catalog) # Catalog
                                tbl_dict['mt_tbl'].append(moltag) # Index
                                tbl_dict['freq_tbl'].append(freq) # Rest frequency
                                tbl_dict['freq_rs_tbl'].append(freq_rs) # Redshifted frequency
                                EL_K = (((line['ELO']*(1./u.cm)*constants.h*constants.c).to(u.erg))/constants.k_B).decompose()
                                EU_K = EL_K + (constants.h*freq)/constants.k_B
                                tbl_dict['EU_K_tbl'].append(EU_K) # Upper state energy in K
                                tbl_dict['aij_tbl'].append(line['A_ij']) # A_ij
                                if catalog == 'CDMS':
                                    if not isinstance(line['vu'], np.ma.core.MaskedConstant):
                                        # qn = f"{line['Ju']}({line['Ku']},{line['vu']}) – {line['Jl']}({line['Kl']},{line['vl']})"
                                        qn = f"${line['Ju']}_{{{line['Ku']},{line['vu']}}} – {line['Jl']}_{{{line['Kl']},{line['vl']}}}$" # To match JPL format
                                    elif not isinstance(line['Ku'], np.ma.core.MaskedConstant):
                                        # qn = f"{line['Ju']}({line['Ku']}) – {line['Jl']}({line['Kl']})"
                                        qn = f"${line['Ju']}_{line['Ku']} – {line['Jl']}_{line['Kl']}$" # To match JPL format
                                    else: 
                                        qn = f"${line['Ju']} – {line['Jl']}$"
                                elif catalog == 'JPL':
                                    qn1u = line["QN'1"]
                                    qn1l = line['QN"1']
                                    qn2u = line["QN'2"]
                                    qn2l = line['QN"2']
                                    qn3u = line["QN'3"]
                                    qn3l = line['QN"3']
                                    qn4u = line["QN'4"]
                                    qn4l = line['QN"4']
                                    if line['QNFMT'] in [1404, 814]:
                                        # 1404: asymmetric rotor with vibration, Q = 14, DR = 3 order = N, K−1, K+1, v, (J), (F)
                                        # 814: linear — case a (2S+1 even), Q = 8, DR = 2, order = J+1/2, omega+1/2, lambda, (F1), (F2), (F)
                                        qn = f"${qn1u}_{{{qn2u}, {qn3u}, {qn4u}}} – {qn1l}_{{{qn2l}, {qn3l}, {qn4l}}}$"
                                    else:
                                        raise NotImplementedError(f"This quantum number format ({line['QNFMT']}) has no parsing code")
                                tbl_dict['qn_tbl'].append(qn) # Quantum numbers
                                tbl_dict['peak_tbl'].append(f"${peak.value} \\pm {noise}$") # Peak intensity
                                tbl_dict['peak_val'].append(peak) # For calculating brights
                                ## Section for integrated intensity
                                try:
                                    spec_vel = pyspeckit.Spectrum(xarr=spec.xarr.copy(), data=spec.data, unit=spec.unit)
                                    spec_vel.xarr.convert_to_unit(u.km/u.s, refX=freq, velocity_convention='radio') # This converts ALL spectra into velocity space...
                                    spec_vel_cutout = spec_vel.slice(float(v_cen)*(u.km/u.s) - 4.5*(u.km/u.s), float(v_cen)*(u.km/u.s) + 4.5*(u.km/u.s))
                                    # Calculate integrated intensity (sum slice and multiply by channel width)
                                    slice_sum = spec_vel_cutout.data.sum()*spec_vel_cutout.unit
                                    channel_width = np.abs(spec_vel_cutout.xarr.cdelt())
                                    integrated_intensity = slice_sum*channel_width
                                    spec_vel.xarr.convert_to_unit(u.GHz, refX=freq, velocity_convention='radio') # Need to convert ALL spectra back
                                    # Get the error on that too
                                    intint_err = noise*spec.unit*channel_width*np.sqrt(len(spec_vel_cutout))
                                    tbl_dict['intint_tbl'].append(f"${integrated_intensity.value} \\pm {intint_err.value}$") # Integrated intensity and error
                                    tbl_dict['intint_val'].append(integrated_intensity.value)
                                except ValueError: # Ignore lines that are "weird", I'll remove them from the table
                                    tbl_dict['intint_tbl'].append('ERROR')
                                    continue
                                ## Section for N_upper/deg
                                N_upper = nupper_of_kkms(integrated_intensity, freq, 10**line['A_ij'])
                                # And error on that
                                N_upper_err = nupper_of_kkms(intint_err, freq, 10**line['A_ij'])
                                tbl_dict['nupper_deg_tbl'].append(f"${N_upper/line['GUP']} \\pm {N_upper_err/line['GUP']}$") # N_upper/deg and err
                                if (85.9656 < freq.value < 86.4344) or (86.6656 < freq.value < 87.1344) or (89.1592 < freq.value < 89.2178) or (87.8959 < freq.value < 87.9545) or (97.6625 < freq.value < 99.5375) or (99.5625 < freq.value < 101.438): # From Table 1 in ACES overview paper
                                    tbl_dict['in_aces_tbl'].append(True)
                                else:
                                    tbl_dict['in_aces_tbl'].append(False)
    
        if zone == 'core':
            # Add labels to individual lines affected by LSS
            minfreq = spec.xarr.to(u.GHz).min()
            maxfreq = spec.xarr.to(u.GHz).max()
            if minfreq <= 94.4*u.GHz <= maxfreq:
                freq_labels.append(94.4*u.GHz)
                name_labels.append(r'$\mathrm{^{13}CH_3OH}$ (RS, 16 km s$^{-1}$)')
            if minfreq <= 104.025*u.GHz <= maxfreq:
                freq_labels.append(104.025*u.GHz)
                name_labels.append(r'$\mathrm{SO_2}$ (RS, 13 km s$^{-1}$)')
            if minfreq <= 106.9*u.GHz <= maxfreq:
                freq_labels.append(106.9*u.GHz)
                name_labels.append(r'$\mathrm{HOCO^{+}}$ (RS, 40 km s$^{-1}$)')
    
            # Add labels to feature caused by edge-of-bandpass issue
            if minfreq <= 271.470841*u.GHz <= maxfreq:
                if str(minfreq) == '269.60466047609003 GHz': # This frequency appears in two spws, so only label relevant one
                    freq_labels.append(271.470841*u.GHz)
                    name_labels.append('EOB feat.')
        
        # Make and write model
        mod = pyspeckit.Spectrum(xarr=spec.xarr.copy(), data=mod)
        if write_models:
            mod_fn = f"/model_{spec.xarr.min().value}_{spec.xarr.max().value}.fits"
            mod.write(outdir+mod_fn, type='fits')

        # Divide the spectrum into two for better visibility
        spec_midpt = ((spec.xarr.to(u.GHz).min().value+spec.xarr.to(u.GHz).max().value)/2)*u.GHz # Calculate midpoint of spectrum
        spec.xarr.convert_to_unit(u.GHz) # Just making sure the conversion worked because of error we got in plotting
        mod.xarr.convert_to_unit(u.GHz) # Just making sure the conversion worked because of error we got in plotting
        spec_LH = spec.slice(spec.xarr.to(u.GHz).min(), spec_midpt)
        spec_UH = spec.slice(spec_midpt, spec.xarr.to(u.GHz).max())
        mod_LH = mod.slice(mod.xarr.to(u.GHz).min(), spec_midpt)
        mod_UH = mod.slice(spec_midpt, mod.xarr.to(u.GHz).max())
        
        freq_center_LH, sigma_LH = plot_spectrum(zone, spec_LH, mod_LH, name_labels, freq_labels, figdir, plot_noise_range, save_figs, show_figs)
        freq_center_UH, sigma_UH = plot_spectrum(zone, spec_UH, mod_UH, name_labels, freq_labels, figdir, plot_noise_range, save_figs, show_figs)

        freq_centers_arr.append(freq_center_LH.value)
        freq_centers_arr.append(freq_center_UH.value)
        sigma_arr.append(sigma_LH.value)
        sigma_arr.append(sigma_UH.value)
    
    line_tbl = QTable(tbl_dict) # Can't figure out how to rename columns in Astropy tables; can do it in pandas though
    line_tbl.sort('freq_rs_tbl')
    
    return line_tbl, freq_centers_arr*freq_center_LH.unit, sigma_arr*sigma_LH.unit

def extract_from_tbl(zone, tbl):
    if zone == 'core':
        tbl = tbl[(tbl['core'] != 'UL') & (tbl['core'] != '-') & (tbl['core'] != 'N/A') & tbl['core'].notnull()] # Only exclude ULs, irrelevant vel. components, and blanks
        # tbl = tbl[(tbl['core'] == 'Y') | (tbl['core'] == 'SA') | (tbl['core'] == 'ST') | (tbl['core'] == 'RS')] # This is a version that must be constantly updated
        # Export handy columns about the parameters
        inlocs = tbl['core'].to_list()
        moltags = tbl['Index'].astype('int').to_list()
        molnames = tbl['Name'].to_list()
        specieses = tbl['Species'].to_list()
        specieses_ltx = tbl['Species_latex'].to_list()
        catalogs = tbl['Catalog'].to_list()
        # Core-specific
        v_cens = tbl['core v_c, km/s'].to_list()
        v_disps = tbl['core v_d, km/s'].to_list()
        temps = tbl['core T_rot, K'].to_list()
        N_tots = tbl['core N_tot, cm^-2'].to_list()
    elif zone == 'frown' or 'nw':
        tbl = tbl[(tbl['frown'] != 'UL') & (tbl['frown'] != '-') & (tbl['frown'] != 'N/A') & tbl['frown'].notnull()] # Only exclude ULs, irrelevant vel. components, and blanks
        # tbl = tbl[(tbl['frown'] == 'Y') | (tbl['frown'] == 'SA') | (tbl['frown'] == 'ST') | (tbl['frown'] == 'RS')] # This is a version that must be constantly updated
        # Export handy columns about the parameters
        inlocs = tbl['frown'].to_list()
        moltags = tbl['Index'].astype('int').to_list()
        molnames = tbl['Name'].to_list()
        specieses = tbl['Species'].to_list()
        specieses_ltx = tbl['Species_latex'].to_list()
        catalogs = tbl['Catalog'].to_list()
        # Frown-specific
        v_cens = tbl['frown v_c, km/s'].to_list()
        v_disps = tbl['frown v_d, km/s'].to_list()
        temps = tbl['frown T_rot, K'].to_list()
        N_tots = tbl['frown N_tot, cm^-2'].to_list()
    return tbl, inlocs, molnames, specieses, specieses_ltx, catalogs,  moltags, v_cens, v_disps, temps, N_tots

def plot_spectrum(zone, spec, mod, name_labels, freq_labels, figdir, plot_noise_range, save_figs, show_figs):
    # Make new spectrum that is the maximum of both data and model (for line ID plotting)
    maxspec = np.maximum(spec.data, mod.data)
    maxspec = pyspeckit.Spectrum(xarr=spec.xarr, data=np.ma.getdata(maxspec.data))
    # Plot spectrum and model, and plot a "ghost" maxspec (necessary for line IDs)
    fig = plt.figure(figsize=(30, 6)) # 30, 5
    gs = fig.add_gridspec(2, hspace = 0, height_ratios = [2, 1])
    axs = gs.subplots(sharex = True, sharey = False)
    spec.plotter(axis=axs[0], figure=fig)
    if zone == 'core':
        mod.plotter(axis=axs[0], clear=False, color='blue', drawstyle='default')
    elif zone == 'frown':
        mod.plotter(axis=axs[0], clear=False, color='red', drawstyle='default')
    elif zone == 'nw':
        mod.plotter(axis=axs[0], clear=False, color='green', drawstyle='default')
    maxspec.plotter(axis=axs[0], clear=False, linewidth=0)
    # Plot residuals
    resid = spec - mod
    resid = pyspeckit.Spectrum(xarr=spec.xarr, data=np.ma.getdata(resid.data))
    resid.plotter(axis=axs[1], clear=False)
    axs[1].axhline(linestyle='--', linewidth=0.5, color='grey', alpha=.75)
    # Adjust limits
    axs[0].set_ylim(np.min([spec.plotter.ymin,mod.plotter.ymin]), np.max(([spec.plotter.ymax,mod.plotter.ymax])))
    # Add labels
    axs[1].set_xlabel(f"Frequency [{(spec.xarr.to(u.GHz)).unit}]", fontsize = 16)
    axs[1].xaxis.set_major_locator(MultipleLocator(0.1))
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.01))
    axs[0].set_ylabel(f"Brightness temperature [K]", fontsize = 16)
    axs[1].set_ylabel(f"Residual [K]", fontsize = 16)
    axs[0].tick_params(axis = 'both', which = 'major', labelsize = 14)
    axs[1].tick_params(axis = 'both', which = 'major', labelsize = 14)
    # Make a DIY legend
    code_artist = mpatches.Rectangle([0, 0], 0, 0, fill='grey', color='grey', alpha=.25)
    code_label = r"2$\sigma$ error"
    axs[0].legend(handles = [code_artist], labels = [code_label], loc="lower right", fontsize = 12)
    # Add line IDs
    maxspec.plotter.line_ids(name_labels, freq_labels*u.GHz, velocity_offset=0.*u.km/u.s, plot_kwargs={'color':'grey', 'linewidth':0.5}) # This used to be diff for core/frown when we had one velocity offset per region
    if plot_noise_range:
        # Plot a grey rectangle from -2sigma to 2sigma about 0, where sigma is the mad_std from the spectrum
        sigma = mad_std(spec.data)*u.K 
        freq_center = (spec.xarr.to(u.GHz).min() + spec.xarr.to(u.GHz).min())/2.
        axs[0].fill_between(x=axs[0].get_xlim(), y1=2*sigma.value, y2=-2*sigma.value, color='grey', alpha=.25)
        axs[1].fill_between(x=axs[1].get_xlim(), y1=2*sigma.value, y2=-2*sigma.value, color='grey', alpha=.25)
    if save_figs:
        if zone == 'core':
            # print(f'data_model_c_{np.min(spec.xarr.to(u.GHz)).value:07.3f}_{np.max(spec.xarr.to(u.GHz)).value:07.3f}')
            plt.savefig(figdir+'/'+f'data_model_c_{np.min(spec.xarr.to(u.GHz)).value:07.3f}_{np.max(spec.xarr.to(u.GHz)).value:07.3f}.png', bbox_inches='tight')
            plt.savefig(figdir+'/'+f'data_model_c_{np.min(spec.xarr.to(u.GHz)).value:07.3f}_{np.max(spec.xarr.to(u.GHz)).value:07.3f}.pdf', bbox_inches='tight')
        elif zone == 'frown':
            # print(f'data_model_f_{np.min(spec.xarr.to(u.GHz)).value:07.3f}_{np.max(spec.xarr.to(u.GHz)).value:07.3f}')
            plt.savefig(figdir+'/'+f'data_model_f_{np.min(spec.xarr.to(u.GHz)).value:07.3f}_{np.max(spec.xarr.to(u.GHz)).value:07.3f}.png', bbox_inches='tight')
            plt.savefig(figdir+'/'+f'data_model_f_{np.min(spec.xarr.to(u.GHz)).value:07.3f}_{np.max(spec.xarr.to(u.GHz)).value:07.3f}.pdf', bbox_inches='tight')
        elif zone == 'nw':
            # print(f'data_model_f_{np.min(spec.xarr.to(u.GHz)).value:07.3f}_{np.max(spec.xarr.to(u.GHz)).value:07.3f}')
            plt.savefig(figdir+'/'+f'data_model_n_{np.min(spec.xarr.to(u.GHz)).value:07.3f}_{np.max(spec.xarr.to(u.GHz)).value:07.3f}.png', bbox_inches='tight')
            plt.savefig(figdir+'/'+f'data_model_n_{np.min(spec.xarr.to(u.GHz)).value:07.3f}_{np.max(spec.xarr.to(u.GHz)).value:07.3f}.pdf', bbox_inches='tight')
    if show_figs:
        plt.show()
    return freq_center, sigma