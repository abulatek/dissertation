from spectral_cube import SpectralCube
from astropy.io import fits
from reproject import reproject_interp
from astroquery.splatalogue.utils import minimize_table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import pyspeckit
from pyspeckit.spectrum.models.lte_molecule import get_molecular_parameters
from pyspeckit.spectrum.models import lte_molecule
from pyspeckit.spectrum.models.lte_molecule import nupper_of_kkms
from astropy import coordinates, constants, units as u
from astropy.stats import mad_std
from tqdm.auto import tqdm

def fetch_cubes(cubefns, catalog, mol_name_pyspeckit=None, mol_tag_pyspeckit=None, parse_loc=False, 
                ret_tbl=False, mute=False):
    # Get only the cubes that have our line of interest in them
    cubes = []
    tbls = []
    fns = []
    if not mute:
        fn_all = tqdm(cubefns)
    else:
        fn_all = cubefns
    for fn in fn_all:
        molcube = SpectralCube.read(fn, format='fits', use_dask=False)
        # print(molcube)
        if catalog in ['JPL', 'CDMS']:
            try:
                if ret_tbl:
                    freqs, aij, deg, EU, partfunc, tbl = get_molecular_parameters(molecule_name=mol_name_pyspeckit,
                                                                                  molecule_tag=mol_tag_pyspeckit,
                                                                                  catalog=catalog, 
                                                                                  parse_name_locally=parse_loc, 
                                                                                  return_table=ret_tbl,
                                                                                  fmin=molcube.spectral_axis.min(), 
                                                                                  fmax=molcube.spectral_axis.max(),
                                                                                  fallback_to_getmolecule=True)
                else: 
                    freqs, aij, deg, EU, partfunc = get_molecular_parameters(molecule_name=mol_name_pyspeckit,
                                                                             molecule_tag=mol_tag_pyspeckit, 
                                                                             catalog=catalog, 
                                                                             parse_name_locally=parse_loc, 
                                                                             return_table=ret_tbl,
                                                                             fmin=molcube.spectral_axis.min(), 
                                                                             fmax=molcube.spectral_axis.max(),
                                                                             fallback_to_getmolecule=True)
            except TypeError as te:
                # print(f"ERROR: {te}")
                continue  
            except:
                continue
        else:
            raise ValueError("The provided catalog is not supported (must be either CDMS or JPL)")
        freqs_to_keep = (freqs > molcube.spectral_axis.min()) & (freqs < molcube.spectral_axis.max()) # Get boolean array of indices within bound for that cube
        freqs = freqs[freqs_to_keep] 
        aij = aij[freqs_to_keep]
        deg = deg[freqs_to_keep]
        EU = EU[freqs_to_keep]
        tbl = tbl[freqs_to_keep]
        if len(freqs) > 0:
            cubes.append(molcube)
            fns.append(fn)
            # print(molcube.beam)
            if ret_tbl:
                tbls.append(tbl)
    # Print out which cubes have lines in them
    if not mute:
        print("These cubes have lines from this molecule:")
        for fn_with_line in fns:
            print(fn_with_line)
    # Reorder selected cubes from lowest to highest frequency
    lowestfreqs = [np.min(molcube.spectral_axis).value for molcube in cubes]
    correct_order = np.argsort(lowestfreqs)
    cubes_inorder = [cubes[i] for i in correct_order]
    # Return cubes (and tables if requested)
    if ret_tbl: 
        # Reorder tables from lowest to highest frequency
        lowestfreqs = [np.min(tbl['FREQ']) for tbl in tbls]
        correct_order = np.argsort(lowestfreqs)
        tbls_inorder = [tbls[i] for i in correct_order]
        if not mute: 
            print([tbl for tbl in tbls_inorder])
            print("\n*** NOTE: Make sure you are capturing the cubes and tables into different variables!")
        return cubes_inorder, tbls_inorder
    else:
        return cubes_inorder

def get_cubes_from_mask(region_fn, region_ind, cubes, plot_region=False, return_mask=False, mute=False):
    # Set region with diffuse emission
    diffuse_regions = fits.open(region_fn)
    diffuse_region_mask = diffuse_regions[0].data == region_ind
    # Plot which region you are using (optional)
    if plot_region:
        plt.imshow(diffuse_region_mask, origin="lower")
    cubes_masked = []
    if not mute:
        cube_list = tqdm(cubes)
    else:
        cube_list = cubes
    for molcube in cube_list:
        array, footprint = reproject_interp((diffuse_region_mask, diffuse_regions[0].header), 
                                             molcube.wcs.celestial, shape_out=(molcube.shape[1],
                                                                               molcube.shape[2]))
        mask = array == 1
        # Get subcube from the mask
        molcube_masked = molcube.with_mask(mask)
        molcube_masked_cut = molcube_masked.subcube_from_mask(mask)
        cubes_masked.append(molcube_masked_cut)
    if not return_mask:
        return cubes_masked
    if return_mask:
        return cubes_masked, mask

def model_and_plot(cubes, temp, N_tot, v_cen, v_disp, catalog, fig_width=10, fig_height=10, nrows=5, ncols=5, 
                   mol_name_pyspeckit=None, mol_tag_pyspeckit=None, parse_loc=True, ret_tbl=True, 
                   line_by_line=False, name_for_plot=None, print_diag=False, extr_type=None, crd=None, 
                   just_data=False, EU_cutoff_K=100, aij_cutoff=-5, LGINT_cutoff=-5, calc_N_uppers=True, 
                   intint_halfwidth_c_kms=4.5, intint_halfwidth_f_kms=9.0, show_int_range=True, show_2_sigma=False,
                   return_freqs=True, chisq=False, mute=False):
    # Plot all lines of the molecule and overplot model spectrum, given an input T, N_tot, and velocity params
    # Assumes input cubes are continuum-subtracted
    if not mute:
        fig = plt.figure(figsize = (fig_width, fig_height))
    # Formatting for plot title
    if extr_type == 'coord':
        extr_type_plot = 'core'
    elif extr_type == 'reg':
        extr_type_plot = 'frown'
    if not mute:
        plt.suptitle(f"{name_for_plot}, {extr_type_plot}, $T_{{rot}} =$ {temp:0.1f} K, $N_{{tot}} =$ {N_tot:0.5g} cm$^{{-2}}$, $v_{{cen}} =$ {v_cen.value} km s$^{{-1}}$, $v_{{disp}} =$ {v_disp.value} km s$^{{-1}}$", fontsize = 16)
    ind=0 # Need this for subplots
    if calc_N_uppers:
        EUs = []
        log_N_upper_gs = []
        log_N_upper_g_errs = []
        ULs = []
    if return_freqs:
        mom0_freqs = []
    # Loop through cubes and extract lines
    if chisq:
        dataminusmodsq = []
        dataminuszeromodsq = []
    if not mute:
        cube_list = tqdm(cubes)
    else:
        cube_list = cubes
    for molcube in cube_list:
        if catalog in ['JPL', 'CDMS']:
            try:
                if ret_tbl:
                    freqs, aij, deg, EU, partfunc, tbl = get_molecular_parameters(molecule_name=mol_name_pyspeckit,
                                                                                  molecule_tag=mol_tag_pyspeckit,
                                                                                  catalog=catalog, 
                                                                                  parse_name_locally=parse_loc, 
                                                                                  return_table=ret_tbl,
                                                                                  fmin=molcube.spectral_axis.min(), 
                                                                                  fmax=molcube.spectral_axis.max(),
                                                                                  fallback_to_getmolecule=True)
                else: 
                    freqs, aij, deg, EU, partfunc = get_molecular_parameters(molecule_name=mol_name_pyspeckit,
                                                                             molecule_tag=mol_tag_pyspeckit, 
                                                                             catalog=catalog, 
                                                                             parse_name_locally=parse_loc, 
                                                                             return_table=ret_tbl,
                                                                             fmin=molcube.spectral_axis.min(), 
                                                                             fmax=molcube.spectral_axis.max(),
                                                                             fallback_to_getmolecule=True)
            except TypeError as te:
                # print(f"ERROR: {te}")
                continue  
            except:
                continue
        else:
            raise ValueError("The provided catalog is not supported (must be either CDMS or JPL)")
        freqs_to_keep = (freqs > molcube.spectral_axis.min()) & (freqs < molcube.spectral_axis.max()) # Get boolean array of indices within bound for that cube
        freqs = freqs[freqs_to_keep] 
        aij = aij[freqs_to_keep]
        deg = deg[freqs_to_keep]
        EU = EU[freqs_to_keep]
        tbl = tbl[freqs_to_keep]
        if print_diag: print("Got molecular parameters")

        # # Do line extraction on a spw-by-spw basis (mainly used for chi-sq calculation--but not anymore)
        # if not line_by_line:
        #     # Do a temporary extraction to get frequency boundaries
        #     if extr_type == "coord":
        #         # Extract spectrum from provided coordinate
        #         x,y = map(int, molcube.wcs.celestial.world_to_pixel(crd))
        #         molcube_temp = molcube[:, y, x]
        #     elif extr_type == "reg":
        #         # Extract spectrum from the subcube made from the mask
        #         molcube_temp = molcube.mean(axis=(1,2), how='slice')
        #     else:
        #         raise ValueError("Invalid extraction type specification")
        #     # Make a temporary model spectrum
        #     mod_temp = lte_molecule.generate_model(molcube_temp.spectral_axis, v_cen, v_disp, temp, N_tot, 
        #                                            freqs, aij, deg, EU, partfunc)
        #     # If there is signal in the model spectrum, get the boundaries around the lines
        #     if sum(mod_temp > 0.01*mod_temp.max()) > 0:       
        #         ax = plt.subplot(nrows, ncols, ind+1)
        #         minfreq = molcube.spectral_axis[mod_temp > 0.01*mod_temp.max()].min().to(u.GHz) - 0.01*u.GHz
        #         maxfreq = molcube.spectral_axis[mod_temp > 0.01*mod_temp.max()].max().to(u.GHz) + 0.01*u.GHz
        #         if print_diag: print("Calculated spectral line boundaries")
        #     else:
        #         continue
        #     # Extract the spectrum
        #     molcube_slice = molcube.spectral_slab(minfreq, maxfreq)
        #     if extr_type == "coord":
        #         # Extract spectrum from provided coordinate
        #         x,y = map(int, molcube.wcs.celestial.world_to_pixel(crd))
        #         data_sp = molcube_slice[:, y, x]
        #     elif extr_type == "reg":
        #         # Extract spectrum from the subcube made from the mask
        #         data_sp = molcube_slice.mean(axis=(1,2), how='slice')
        #     else:
        #         raise ValueError("Invalid extraction type specification")
        #     # Continuum-subtract the spectrum
        #     # data_sp_contsub = data_sp - np.median(data_sp)
        #     data_sp_contsub = data_sp # Assumes contsub is already done
        #     if print_diag: print("extracted continuum-subtracted spectrum")
        #     # Make the model spectrum
        #     mod = lte_molecule.generate_model(molcube_slice.spectral_axis, v_cen, v_disp, temp, N_tot, 
        #                                       freqs, aij, deg, EU, partfunc)
        #     if print_diag: print("made model")
        #     if not mute:
        #         # Plot the data and model spectra
        #         ax.plot(molcube_slice.spectral_axis.to(u.GHz), data_sp_contsub, color='k')
        #         if not just_data:
        #             if extr_type == "reg":
        #                 ax.plot(molcube_slice.spectral_axis.to(u.GHz), mod, color='red')
        #             elif extr_type == "coord":
        #                 ax.plot(molcube_slice.spectral_axis.to(u.GHz), mod, color='blue')
        #         ax.set_xlabel(f"Frequency [{(molcube_slice.spectral_axis.to(u.GHz)).unit}]")
        #         ax.set_ylabel(f"Brightness temperature [{data_sp_contsub.unit}]")
        #         EU_K = ((EU*u.erg)/constants.k_B).decompose()
        #         ax.text(0.05, 0.9, f"$E_U = ${EU_K[-1]:0.1f}", transform=ax.transAxes)
        #         if ret_tbl:
        #             ax.text(0.05, 0.8, f"{tbl['Ju'][-1]}({tbl['Ku'][-1]}) – {tbl['Jl'][-1]}({tbl['Kl'][-1]})", 
        #                     transform=ax.transAxes)
        #     # Show the local 2-sigma mark on each axis if requested (for upper-limit measurements)
        #     noise_level = mad_std(data_sp_contsub.data)*u.K
        #     if show_2_sigma:
        #         ax.hlines(y=2*noise_level.value, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], linewidth=1, 
        #                   color='r')
        #     ind+=1
        
        # Do line extraction on a line-by-line basis
        if line_by_line:
            for freq, this_EU, this_aij, this_deg, this_tbl in zip(freqs, EU, aij, deg, tbl):
                this_EU_K = ((this_EU*u.erg)/constants.k_B).decompose()
                # print(freq, this_EU_K, this_aij)
                this_LGINT = this_tbl['LGINT']
                if this_EU_K < EU_cutoff_K*u.K and this_LGINT > LGINT_cutoff: # this_aij > aij_cutoff 
                    # Set center at vcen, then do bandwidth of 50 km/s
                    freq_to_vel = u.doppler_radio(freq)
                    minfreq = (v_cen - 50*u.km/u.s).to(u.GHz, equivalencies=freq_to_vel)
                    maxfreq = (v_cen + 50*u.km/u.s).to(u.GHz, equivalencies=freq_to_vel)
                    if print_diag: print("calculated spectral line boundaries")
                    # Extract the spectrum
                    molcube_slice = molcube.spectral_slab(minfreq, maxfreq)
                    if extr_type == 'coord':
                        # Extract spectrum from provided coordinate
                        x,y = map(int, molcube.wcs.celestial.world_to_pixel(crd))
                        data_sp = molcube_slice[:, y, x]
                    elif extr_type == 'reg':
                        # Extract spectrum from the subcube made from the mask
                        data_sp = molcube_slice.mean(axis=(1,2), how='slice')
                    else:
                        raise ValueError("Invalid extraction type specification")
                    # Continuum-subtract the spectrum
                    # data_sp_contsub = data_sp - np.median(data_sp)
                    data_sp_contsub = data_sp # Assumes contsub is already done
                    # Calculate noise level in spectrum
                    noise_level = mad_std(data_sp_contsub.data)*u.K
                    if print_diag: print("extracted continuum-subtracted spectrum")
                    # Make the model spectrum
                    mod = lte_molecule.generate_model(molcube_slice.spectral_axis, v_cen, v_disp, temp, N_tot,
                                                      freqs, aij, deg, EU, partfunc)
                    if print_diag: print("made model")

                    if extr_type == 'coord':
                        intint_halfwidth_kms = intint_halfwidth_c_kms
                    elif extr_type == 'reg':
                        intint_halfwidth_kms = intint_halfwidth_f_kms
                    
                    # Record EUs and N_uppers if requested
                    if calc_N_uppers:
                        # print(freq)
                        lines = tbl[(tbl['FREQ'] > (freq-0.01*u.GHz)) & (tbl['FREQ'] < (freq+0.01*u.GHz))] # Sample lines around this one (for dealing with unsplit HFS)
                        if this_LGINT >= np.max(lines['LGINT']): 
                            spec = pyspeckit.Spectrum(data=data_sp_contsub, 
                                                      xarr=molcube_slice.spectral_axis.to(u.GHz), unit=u.K)
                            spec.xarr.convert_to_unit(u.km/u.s, refX = freq, velocity_convention = 'radio')
                            spec_slice = spec.slice(v_cen - intint_halfwidth_kms*u.km/u.s, v_cen + intint_halfwidth_kms*u.km/u.s)
                            # Calculate integrated intensity (sum slice and multiply by channel width)
                            try:
                                # Determine if each point is an upper limit or not using peak intensity and local noise level
                                c_km_s = (constants.c).to(u.km/u.s)
                                freq_rs = freq*(1-(v_cen/c_km_s)) # Calculate redshifted frequency
                                peak = spec.data[np.argmin(np.abs(spec.xarr.to(u.GHz) - freq_rs))]*spec.unit # Find data value closest to the redshifted frequency
                                UL = False if peak >= 2.*noise_level else True
                                ULs.append(UL)
                                # Put in E_U
                                EUs.append(this_EU_K)
                                if not UL:
                                    print("real line:", freq_rs)
                                    # Treat the line as real
                                    spec_slice_sum = spec_slice.data.sum()*u.K
                                    channel_width = np.abs(spec_slice.xarr.cdelt())
                                    mom0 = spec_slice_sum*channel_width
                                    # Calculate upper state column density from integrated intensity, convert to logscale
                                    N_upper = nupper_of_kkms(mom0, freq, 10**this_aij)
                                    print(N_upper)
                                    log_N_upper_g = np.log10(N_upper.value/this_deg)
                                    log_N_upper_gs.append(log_N_upper_g)
                                    # Propagate error on N_upper measurement
                                    noise_map_int = noise_level*channel_width*np.sqrt(len(spec_slice))
                                    N_upper_err = nupper_of_kkms(noise_map_int, freq, 10**this_aij)
                                    log_N_upper_g_err = np.abs(N_upper_err/(N_upper*np.log(10.)))
                                    log_N_upper_g_errs.append(log_N_upper_g_err)
                                elif UL:
                                    print("non-detection:", freq_rs)
                                    # Treat the line as not real and use the noise level to set a 2-sigma upper limit on N_upper
                                    channel_width = np.abs(spec_slice.xarr.cdelt())
                                    mom0_lim = 2*np.sqrt(len(spec_slice))*noise_level*channel_width
                                    N_upper = nupper_of_kkms(mom0_lim, freq, 10**this_aij)
                                    log_N_upper_g = np.log10(N_upper.value/this_deg)
                                    log_N_upper_gs.append(log_N_upper_g)
                                    # Need to put in an error too, even though the error does not get plotted on UL points
                                    log_N_upper_g_errs.append(np.nan)
                            except ValueError: # This means it's a "bad line" (there's no signal in the range I'm integrating)
                                print("bad line:", freq_rs)
                                log_N_upper_gs.append(np.nan)
                                EUs.append(this_EU_K)
                                log_N_upper_g_errs.append(np.nan)
                                ULs.append(True) # This is not true but it doesn't matter because they're not being plotted
 
                    if chisq:
                        freq_axis = molcube.spectral_axis.to(u.GHz)
                        if freq_axis.min() < 115.265*u.GHz < freq_axis.max(): # Do not use area of spectrum affected by CO
                            # This is inefficient but the safest way I can think of to do it
                            belowCO = molcube.spectral_slab(np.min(molcube.spectral_axis).to(u.GHz), 115.19*u.GHz)
                            aboveCO = molcube.spectral_slab(115.32*u.GHz, np.max(molcube.spectral_axis).to(u.GHz))
                            if extr_type == 'coord':
                                # Extract spectra from provided coordinate
                                x,y = map(int, molcube.wcs.celestial.world_to_pixel(crd))
                                data_sp_belowCO = belowCO[:, y, x]
                                data_sp_aboveCO = aboveCO[:, y, x]
                            elif extr_type == 'reg':
                                # Extract spectra from the subcube made from the mask
                                data_sp_belowCO = belowCO.mean(axis=(1,2), how='slice')
                                data_sp_aboveCO = aboveCO.mean(axis=(1,2), how='slice')
                            merged = np.concatenate([data_sp_belowCO.value, data_sp_aboveCO.value])
                            variance = (mad_std(merged)*u.K)**2
                        else:
                            if extr_type == 'coord':
                                # Extract spectra from provided coordinate
                                x,y = map(int, molcube.wcs.celestial.world_to_pixel(crd))
                                data_sp_cube = molcube[:, y, x]
                            elif extr_type == 'reg':
                                # Extract spectra from the subcube made from the mask
                                data_sp_cube = molcube.mean(axis=(1,2), how='slice')
                            variance = (mad_std(data_sp_cube))**2
                        dataminuszeromodsq += (((np.abs(data_sp_contsub.value))**2)/variance.value).tolist() # Model should be better than a flat line at 0, like turning column density all the way down (to test if my calculation is correct)
                        dataminusmodsq += (((data_sp_contsub.value - mod)**2)/variance.value).tolist() # Keep track of data minus model squared
                    if return_freqs:
                        mom0_freqs.append(freq)

                    # Sanity check
                    if chisq:
                        temp_xa = np.sum((((data_sp_contsub.value - mod)**2)/variance.value).tolist())
                        temp_xb = np.sum((((np.abs(data_sp_contsub.value))**2)/variance.value).tolist())
                        if temp_xa > temp_xb:
                           print(f"Model ({temp_xa}) is better than zero model ({temp_xb}) for range {minfreq} to {maxfreq}") 

                    # Plot the data and model spectra; need the plotting to be last so it will ignore "weird spectra"
                    if not mute:
                        ax = plt.subplot(nrows, ncols, ind+1)
                        ax.plot(molcube_slice.spectral_axis.to(u.GHz), data_sp_contsub, color='k')
                        if not just_data:
                            if extr_type == 'reg':
                                ax.plot(molcube_slice.spectral_axis.to(u.GHz), mod, color='red')
                            elif extr_type == 'coord':
                                ax.plot(molcube_slice.spectral_axis.to(u.GHz), mod, color='blue')
                        ax.ticklabel_format(useOffset=False, style='plain')
                        fig.supxlabel(f"Frequency [{(molcube_slice.spectral_axis.to(u.GHz)).unit}]", y=0.05)
                        fig.supylabel(f"Brightness temperature [{data_sp_contsub.unit}]", x=0.01)
                        
                    # Add labels
                    if not mute:
                        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)] * 3
                        labels = [f"$E_U =${this_EU_K:0.2f}", f"log$_{{10}}(A_{{i,j}}) =${str(round(this_aij, 2)).replace('-','−')}", f"LGINT ={str(round(this_LGINT, 2)).replace('-','−')}"]
                        ax.legend(handles, labels, loc='lower left', handlelength=0, handletextpad=0)
                    # ax.text(0.05, 0.85, f"$E_U =${this_EU_K:0.2f}", transform=ax.transAxes)
                    # ax.text(0.05, 0.75, f"$log_{{10}}(A_{{i,j}}) =${str(round(this_aij, 2)).replace('-','−')}", 
                    #         transform=ax.transAxes)
                    # if ret_tbl:
                    #     ax.text(0.05, 0.75, f"{tbl['Ju'][-1]}({tbl['Ku'][-1]}) – {tbl['Jl'][-1]}({tbl['Kl'][-1]})", 
                    #             transform=ax.transAxes)
                    
                    # Show the local (+/- 50 km/s) 2-sigma mark on each axis if requested (for upper-limit measurements), using previously calculated noise level
                    if show_2_sigma:
                        ax.hlines(y=2*noise_level.value, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], 
                                  linewidth=1, color='r', linestyle='--')
                    if show_int_range:
                        freq_to_vel_p = u.doppler_radio(freq)
                        minfreq_p = (v_cen - intint_halfwidth_kms*u.km/u.s).to(u.GHz, equivalencies=freq_to_vel_p)
                        maxfreq_p = (v_cen + intint_halfwidth_kms*u.km/u.s).to(u.GHz, equivalencies=freq_to_vel_p)
                        ax.vlines(x=[minfreq_p.value, maxfreq_p.value], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
                                  linewidth=1, color='g', linestyle='--')
                    
                    ind+=1
    # Save the spectrum grid figures
    if not mute:
        plt.tight_layout()
        name_for_file = name_for_plot.replace('$','').replace('^','').replace('{','').replace('}','').replace('_','')
        plt.savefig('/blue/adamginsburg/abulatek/brick/first_results/molecule_table/spectra_grids/grid_'+name_for_file+'_'+extr_type+'.pdf',
                    facecolor = 'w', edgecolor = 'w', bbox_inches = 'tight')
        plt.savefig('/blue/adamginsburg/abulatek/brick/first_results/molecule_table/spectra_grids/grid_'+name_for_file+'_'+extr_type+'.png', 
                    dpi = 250, bbox_inches = 'tight')
    # Export the chi-squared value for the spectrum if requested 
    if chisq:
        chisq_val = sum(dataminusmodsq)
        chisq_val_0model = sum(dataminuszeromodsq)
        # print(f"Chi-squared (this model): {chisq_val}")
        # print(f"Chi-squared (model = 0): {chisq_val_0model}")
        return chisq_val
    # Export the upper state column density if requested
    if calc_N_uppers and not return_freqs:
        return EUs, log_N_upper_gs, log_N_upper_g_errs, ULs
    # Export the frequencies for moment maps if requested
    if return_freqs and not calc_N_uppers:
        return mom0_freqs
    # If you need both, export both
    if return_freqs and calc_N_uppers:
        return mom0_freqs, EUs, log_N_upper_gs, log_N_upper_g_errs, ULs

def plot_mom0s(cubes, freqs, v_cen, fig_width, fig_height, nrows, ncols, name_for_plot=None, reg=None, 
               y=0.98, v_half=4.5):
    freqs_vals = [freq.value for freq in freqs]
    freqs = np.unique(np.array(freqs_vals))*u.MHz
    fig = plt.figure(figsize = (fig_width, fig_height))
    plt.suptitle(f"{name_for_plot}, $v_{{cen}} =$ {v_cen.value} km s$^{{-1}}$", fontsize = 16, y=y)
    ind=0 # Need this for subplots
    # Loop through cubes and extract lines
    for molcube in tqdm(cubes):
        for freq in freqs:
            if molcube.spectral_axis.min() <= freq <= molcube.spectral_axis.max():
                ax = plt.subplot(nrows, ncols, ind+1)
                # Make moment map of line
                # molcube = molcube - molcube.median(axis=0, iterate_rays=True) # Continuum subtraction
                molcube_slice = molcube.with_spectral_unit(unit=u.km/u.s, velocity_convention='radio', 
                                                           rest_value=freq).spectral_slab(v_cen - v_half*u.km/u.s, v_cen + v_half*u.km/u.s)
                mom0 = molcube_slice.moment0()
                # print(mom0.shape)
                ax.imshow(mom0.value, origin='lower', cmap='inferno')
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                if reg is not None:
                    ax.contour(reg, levels = [0, 1], linewidths=0.75, colors = ['c'])
                ax.text(0.05, 0.92, f"{freq.to(u.GHz):0.2f}", transform=ax.transAxes)
                ind+=1
    # Save the moment 0 map grid figures
    plt.tight_layout()
    name_for_file = name_for_plot.replace('$','').replace('^','').replace('{','').replace('}','').replace('_','')
    plt.savefig('/blue/adamginsburg/abulatek/brick/first_results/molecule_table/moment0_map_grids/mgrid_'+name_for_file+'_'+str(v_cen.value)+'.pdf', 
                facecolor = 'w', edgecolor = 'w', bbox_inches = 'tight')
    plt.savefig('/blue/adamginsburg/abulatek/brick/first_results/molecule_table/moment0_map_grids/mgrid_'+name_for_file+'_'+str(v_cen.value)+'.png', 
                dpi = 250, bbox_inches = 'tight')
    
def list_mol_tags(catalog):
    if catalog == 'JPL':
        from astroquery.linelists.jplspec import JPLSpec as QueryTool
    elif catalog == 'CDMS':
        from astroquery.linelists.cdms import CDMS as QueryTool
    else:
        raise ValueError("Invalid catalog specification")
    speciestab = QueryTool.get_species_table()
    speciestab.pprint(max_lines=-1)
    return speciestab

# def get_molecular_parameters_plus(catalog, fmin, fmax, mol_name_pyspeckit=None, mol_tag_pyspeckit=None):
#     # This is unnecessary because Adam pushed changes to get_molecular_parameters that fixed it; no need for hacking
#     molecule_name = mol_name_pyspeckit
#     molecule_tag = mol_tag_pyspeckit
#     if catalog == 'JPL':
#         from astroquery.jplspec import JPLSpec as QueryTool
#     elif catalog == 'CDMS':
#         from astroquery.linelists.cdms import CDMS as QueryTool
#     else:
#         raise ValueError("Invalid catalog specification")
#     speciestab = QueryTool.get_species_table()
#     if 'NAME' in speciestab.colnames:
#         molcol = 'NAME'
#     elif 'molecule' in speciestab.colnames:
#         molcol = 'molecule'
#     else:
#         raise ValueError(f"Did not find NAME or molecule in table columns: {speciestab.colnames}")
#     if molecule_tag is not None:
#         tagcol = 'tag' if 'tag' in speciestab.colnames else 'TAG'
#         match = speciestab[tagcol] == molecule_tag
#         molecule_name = speciestab[match][molcol][0]
#         if catalog == 'CDMS':
#             molsearchname = f'{molecule_tag:06d} {molecule_name}'
#         else:
#             molsearchname = f'{molecule_tag} {molecule_name}'
#         parse_names_locally = False
#         if molecule_name is not None:
#             print(f"WARNING: molecule_tag overrides molecule_name.  New molecule_name={molecule_name}.  Searchname = {molsearchname}")
#         else:
#             print(f"molecule_name={molecule_name} for tag molecule_tag={molecule_tag}.  Searchname = {molsearchname}")
#     else:
#         molsearchname = molecule_name
#         match = speciestab[molcol] == molecule_name
#         if match.sum() == 0:
#             # retry using partial string matching
#             match = np.core.defchararray.find(speciestab[molcol], molecule_name) != -1
#     if match.sum() != 1:
#         raise ValueError(f"Too many or too few matches ({match.sum()}) to {molecule_name}")
#     jpltable = speciestab[match]
#     jpltbl = QueryTool.query_lines(fmin, fmax, molecule=molsearchname,
#                                    parse_name_locally=parse_name_locally)
#     return jpltbl