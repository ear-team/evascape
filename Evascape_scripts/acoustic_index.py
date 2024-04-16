# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:48:05 2023

@author: ear-field
"""

import pandas as pd
from maad import sound, util, features, rois
from skimage.morphology import closing
from toolbox import waveread
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# indexes from Morrison et al 2021 
    # ADI (richness) : 
    # AEI (evenness) :
    # H (heterogeneity) :
    # Bi (amplitude) :
        
# chosen indices :
    # ACI : maad.features.acoustic_complexity_index(Sxx)
    # H
# roi index :
    
#setup
samprate = 44100

# load database dataframe
#temp_dir = Path('C:/Users/ecoac-field/OneDrive/Documents/Articles-Recherches/Reconstructor/Samples/temp')
#database_path = temp_dir / 'reconstructed_files/database_data.csv'
# database_dir = Path("./Database_20240315_Desync")  
# database_path = database_dir / "database_data.csv"
# database_df = pd.read_csv(database_path, sep=';', index_col = 0)

# indices_path = database_dir / 'indices.csv'
# indices_df = pd.read_csv(indices_path, sep=';', index_col = 0)

# average_path = database_dir / 'average.csv'
# average_df = pd.read_csv(average_path, sep=';', index_col = 0)


#Indices computing
def compute_all_indices(database_df, samprate = 44100):
    indices_df = database_df.copy()
    file_count = 1
    for file in indices_df.index:
        path = Path(indices_df.fullfilename[file])
        vector = waveread(path)
        vector[vector == 0] = 10**(-8)
        
        #Spectrogram variables
        Sxx_power,tn,fn,ext = sound.spectrogram(x = vector, fs = samprate, mode='amplitude')
        Sxx_power[Sxx_power == 0] = 10**(-25)
        
        #ROI indices
        Sxx_noNoise= sound.median_equalizer(Sxx_power)
        Sxx_dB_noNoise = util.power2dB(Sxx_noNoise)
        #Sxx_dB_noNoise[Sxx_dB_noNoise == 0] = 10**(-10)
        # nROI, aROI = features.region_of_interest_index(Sxx_dB_noNoise, 
        #                                                             tn, fn, 
        #                                                             smooth_param1 = 1,
        #                                                             mask_mode='absolute', 
        #                                                             mask_param1 = 10, 
        #                                                             mask_param2 = 3, 
        #                                                             remove_rain = False,
        #                                                             max_ratio_xy = 10,
        #                                                             display=False)
        nROI, aROI = region_of_interest_index2(Sxx_power, tn, fn,   
                                        seed_level=13, 
                                        low_level=6, 
                                        fusion_rois=(0.05, 100), # (seconds, hertz) ,
                                        remove_rois_fmin_lim = (50,20000),
                                        remove_rois_fmax_lim = (20000),
                                        remove_rain = True,
                                        min_event_duration=0.025, 
                                        max_event_duration=None, 
                                        min_freq_bw=50, 
                                        max_freq_bw=None, 
                                        max_xy_ratio = 10,
                                        verbose=False)
        indices_df.loc[file, 'nROI'] = nROI
        indices_df.loc[file, 'aROI'] = aROI
        
        #Classic indices (from Alcocer et al. 2022)
        _, _ , ACI = features.acoustic_complexity_index(Sxx_power)
        indices_df.loc[file, 'ACI'] = ACI
          
        Hf, Ht_per_bin = features.frequency_entropy(Sxx_power)
        Ht = features.temporal_entropy (vector)
        indices_df.loc[file, 'H'] = Hf * Ht
        
        NDSI, ratioBA, antroPh, bioPh  = features.soundscape_index(Sxx_power,fn)
        indices_df.loc[file, 'NDSI'] = NDSI
        
        ADI  = features.acoustic_diversity_index(Sxx_power,fn,fmax=10000, dB_threshold = -30)
        indices_df.loc[file, 'ADI'] = ADI
        
        #BI
        BI = features.bioacoustics_index(Sxx_power,fn)
        indices_df.loc[file, 'BI'] = BI
        
        print(f'{file}\t{file_count}/{len(indices_df)}')
        file_count += 1
              
    return indices_df
    

def index_plot(indices_df, index, info_df = None,  width = 25, height = 10):

    if info_df is None:
            info_dic = {'channel2' : ['ambient_sound', 'quiet', 'rain_pw01', 'rain_pw02', 'rain_pw03',
                            'tettigonia_veridissima', 'wind_pw01', 'wind_pw02', 'wind_pw03'],
                'color': ['mediumorchid', 'grey', 'deepskyblue', 'royalblue', 'darkblue', 'green', 'lightcoral', 'red', 'brown'],
                'label' : ['ambient sound', 'quiet', 'light rain', 'medium rain', 'strong rain',
                                    'tettigonia veridissima', 'light wind', 'medium wind', 'strong wind']}
            info_df = pd.DataFrame(info_dic).set_index('channel2')

    #plot
    ch2_list = indices_df.channel2.unique()

    fig, axs = plt.subplots(nrows = 1, ncols = 2, sharey='row', figsize=(width,height))
    index_df =  indices_df.loc[:,['richness','abundance','channel2'] + [index]]
    
    supx_list = ['richness','abundance']
    ch2_list = indices_df.channel2.unique()
    corrcoeff_df = pd.DataFrame(index = ch2_list, columns = supx_list)
    pvalue_df = pd.DataFrame(index = ch2_list, columns = supx_list)
    for j, supx in enumerate(supx_list): 
        xvar_list = indices_df[supx].unique()
        df_len = len(xvar_list) * len(ch2_list)
        condition_list = [supx, 'channel2']
        supx_df = pd.DataFrame(index = list(range(df_len)),columns = condition_list)
        i=0
        for x_var in xvar_list :
            for channel2 in ch2_list:
                sub_df = index_df.loc[(index_df.richness == x_var) & 
                                      (index_df.channel2 == channel2)]
                mean = np.mean(sub_df[index])
                error = np.std(sub_df[index],ddof=1)/np.sqrt(np.size(sub_df[index]))
                supx_df.loc[i,condition_list] = [x_var, channel2]
                supx_df.loc[i, index + '_mean'] = mean
                supx_df.loc[i, index + '_err'] = error
                i += 1
        mean_name = supx_df.columns[-2]
        err_name = supx_df.columns[-1]
        mean_df = supx_df.drop(err_name, axis = 1)
        mean_df = mean_df.pivot(index = supx, columns = 'channel2', values = mean_name)
        averagemean_df = np.mean(mean_df, axis = 1)
        mean_columns = list(mean_df.columns)
        err_df = supx_df.drop(mean_name, axis = 1)
        err_df = err_df.pivot(index = supx, columns = 'channel2', values = err_name)
        err_df['mean'] = 0.0
        
        mean_df['index'] = list(range(len(mean_df)))
        mean_df[supx] = mean_df.index
        mean_df = mean_df.set_index('index')

        averagemean_df.plot(ax = axs[j], legend = 0, linewidth = 4,
                            color = 'black', linestyle = 'dashed', label = 'average')
        mean_df.plot(x = supx, y = mean_columns, fontsize=20,
                     yerr = err_df, ax = axs[j], legend = 0,
                     linewidth = 2.5, color = info_df.color)
        axs[j].set_xlabel(supx,fontsize = 25)
        axs[j].autoscale()
        
        #correlation coefficient and p-value
        for channel2 in ch2_list:
            coeff, pvalue = pearsonr(xvar_list, mean_df[channel2])
            corrcoeff_df.loc[channel2, supx] = coeff
            pvalue_df.loc[channel2, supx] = pvalue
            if (coeff > 0 and coeff < 0.7) or (coeff < 0 and coeff < -0.7):
                print(f'{index}, {supx} : {channel2} correlation is weak')
            if pvalue > 0.05:
                print(f'{index}, {supx} : {channel2} p-value is not significant')
    
    corr_col = ['rich_coeff', 'rich_pvalue', 'ab_coeff', 'ab_pvalue']
    correlation_df = pd.DataFrame(index = ch2_list, columns = corr_col)
    correlation_df[['rich_coeff','ab_coeff']] = corrcoeff_df
    correlation_df[['rich_pvalue','ab_pvalue']] = pvalue_df
    
    handles, labels = axs[0].get_legend_handles_labels()
    labels = ['average'] + info_df.label.tolist()
    fig.legend(handles, labels, fontsize = 20, bbox_to_anchor=(1.1, 0.65))
    fig.suptitle(index, fontsize = 40, x = 1, y=0.8)
    
    return fig, correlation_df



def average_all_indices(indices_df, index_list):
    richness_list = indices_df.richness.unique()
    abundance_list = indices_df.abundance.unique()
    ch2_list = indices_df.channel2.unique()
    df_len = len(richness_list) * len(abundance_list) * len(ch2_list)
    condition_list = ['richness','abundance','channel2']
    average_df = pd.DataFrame(index = list(range(df_len)),columns = condition_list)
    for index in index_list:
        index_df = indices_df.loc[:,condition_list + [index]]
        i=0
        for richness in richness_list:
            for abundance in abundance_list: 
                for channel2 in ch2_list:
                    sub_df = index_df.loc[(index_df.richness == richness) & 
                                          (index_df.abundance == abundance) &
                                          (index_df.channel2 == channel2)]
                    mean = np.mean(sub_df[index])
                    error = np.std(sub_df[index],ddof=1)/np.sqrt(np.size(sub_df[index]))
                    average_df.loc[i,condition_list] = [richness, abundance, channel2]
                    average_df.loc[i, index + '_mean'] = mean
                    average_df.loc[i, index + '_err'] = error
                    i += 1
    return(average_df)

# index_list = ['nROI','aROI','ACI','H', 'NDSI', 'ADI', 'BI']
# average_df = average_all_indices(indices_df, index_list)
# average_path = database_dir / 'average.csv'
# average_df.to_csv(average_path, sep=';')        


#### Plot : Index score against variable 1 (richness or abundance), variable 2 is line (abundance or richness)
#### Subplot : Index nature against channel2 
def multiplot(average_df,
              x_var = 'richness', line_var = 'abundance', supx_var = 'channel2',
              index_list = ['nROI','aROI','ACI','H', 'NDSI', 'ADI', 'BI']):

    average_df['total_abundance'] = average_df.richness * average_df.abundance
    supx_list = list(average_df[supx_var].unique())
    if x_var == 'total_abundance':
        condition_list = [x_var, supx_var]
    else :
        condition_list = [x_var, line_var, supx_var]
    #comb_list = pd.DataFrame(product(index_list, supx_list), columns = ['index',supx_var])
    
    fig, axs = plt.subplots(nrows = len(index_list), ncols = len(supx_list), #sharex = True, sharey = True,
                             sharey='row', figsize=(30,25))
    #min_x = min(average_df[x_var])
    #max_x = max(average_df[x_var])
    for i, index in enumerate(index_list):
        index_column = 2 + (i+1)*2 - 1
        index_df = pd.concat([average_df.loc[:,condition_list], 
                              average_df.iloc[:,[index_column, index_column + 1],]
                              ], axis = 1)
        mean_name = index_df.columns[-2]
        err_name = index_df.columns[-1]
        
        for j, supx in enumerate(supx_list):
            supx_df = index_df.loc[index_df[supx_var] == supx].drop([supx_var], axis = 1)
            
            if x_var == 'total_abundance':
                mean_df = pd.DataFrame(columns = [x_var, mean_name])
                err_df = pd.DataFrame(columns = [x_var, err_name])
                index = 0
                for abundance in supx_df.total_abundance.unique():
                    toaverage_df = supx_df.loc[supx_df.total_abundance == abundance]
                    mean = np.mean(toaverage_df[mean_name])
                    mean_df.loc[index] = [abundance, mean]
                    std_mean = np.sqrt(sum(std**2 for std in toaverage_df[err_name])/len(toaverage_df))
                    err_df.loc[index] = [abundance, std_mean]
                    index += 1
                mean_df = mean_df.sort_values(by = x_var)
                err_df = err_df.sort_values(by = x_var).set_index(x_var)
                mean_df.plot(x = x_var, y = mean_name, xticks = list(mean_df[x_var]),
                             yerr = err_df, ax = axs[i,j], legend = 0) #, capsize=5

            else:    
                mean_df = supx_df.drop(err_name, axis = 1)
                mean_df = mean_df.pivot(index = x_var, columns = line_var, values = mean_name)
                averagemean_df = np.mean(mean_df, axis = 1)
                mean_columns = list(mean_df.columns)
                err_df = supx_df.drop(mean_name, axis = 1)
                err_df = err_df.pivot(index = x_var, columns = line_var, values = err_name)
                err_df['mean'] = 0.0
                
                mean_df['index'] = list(range(len(mean_df)))
                mean_df[x_var] = mean_df.index
                mean_df = mean_df.set_index('index')

                averagemean_df.plot(ax = axs[i,j], legend = 0, 
                                    color = 'black', linestyle = 'dashed', zorder = 3.5)
                mean_df.plot(x = x_var, y = mean_columns, xticks = list(mean_df[x_var]),
                             yerr = err_df, ax = axs[i,j], legend = 0, zorder = 4.5) #, capsize=5
    
    #axis and title
    for ax, col in zip(axs[0], supx_list):
        ax.set_title(col, size=20)
    for ax, row in zip(axs[:,0], index_list):
        ax.set_ylabel(row, size=20) #, rotation=0
    fig.supxlabel, fig.supylabel = supx_var, 'index'
    fig.suptitle = f'Index Scores against {x_var}'
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout()
    
    return fig

# multiplot(average_df, 
#           x_var = 'abundance', line_var = 'richness', supx_var = 'channel2',
#           index_list = ['nROI','aROI','ACI','H', 'NDSI', 'ADI', 'BI'])

# multiplot(average_df, 
#           x_var = 'richness', line_var = 'abundance', supx_var = 'channel2',
#           index_list = ['nROI','aROI','ACI','H', 'NDSI', 'ADI', 'BI'])

# multiplot(average_df, 
#           x_var = 'total_abundance', line_var = 'richness', supx_var = 'channel2',
#           index_list = ['nROI','aROI','ACI','H', 'NDSI', 'ADI', 'BI'])


#### New index function to compute ROI indices

def region_of_interest_index2(Sxx_power, tn, fn, 
                            seed_level=16, 
                            low_level=6, 
                            fusion_rois=(0.05, 100),
                            remove_rois_fmin_lim = 50,
                            remove_rois_fmax_lim = None,
                            remove_rain = True,
                            min_event_duration=None, 
                            max_event_duration=None, 
                            min_freq_bw=None, 
                            max_freq_bw=None, 
                            max_xy_ratio=None,
                            verbose=False,
                            **kwargs):

    """
    Calculates region of interest (ROI) indices based on a spectrogram.

    Parameters:
    - Sxx_power (numpy.ndarray): spectrogram with power values (Sxx_amplitude²).
    - tn (numpy.ndarray): Time axis values.
    - fn (numpy.ndarray): Frequency axis values.
    - seed_level (int): Parameter for binarization of the spectrogram (default: 16).
    - low_level (int): Parameter for binarization of the spectrogram (default: 6).
    - fusion_rois (tuple): Fusion parameters for ROIs (default: (0.05, 100)).
    - remove_rois_fmin_lim (float or tuple): Frequency threshold(s) to remove ROIs (default: None).
    - remove_rois_fmax_lim (float): Frequency threshold to remove ROIs (default: None).
    - remove_rain (bool): Whether to remove rain noise (default: True).
    - min_event_duration (float): Minimum time duration of an event in seconds (default: None).
    - max_event_duration (float): Maximum time duration of an event in seconds (default: None).
    - min_freq_bw (float): Minimum frequency bandwidth in Hz (default: None).
    - max_freq_bw (float): Maximum frequency bandwidth in Hz (default: None).
    - **kwargs: Additional arguments.

    Returns:
    - nROI (int): Number of ROIs.
    - aROI (float): Percentage of ROIs area over the total area.

    Note:
    - This function requires the following packages: skimage, numpy, pandas, matplotlib, maad.

    """

    FUSION_ROIS = fusion_rois
    REMOVE_ROIS_FMIN_LIM = remove_rois_fmin_lim
    REMOVE_ROIS_FMAX_LIM = remove_rois_fmax_lim
    REMOVE_RAIN = remove_rain
    MIN_EVENT_DUR = min_event_duration # Minimum time duration of an event (in s)
    MAX_EVENT_DUR = max_event_duration
    MIN_FREQ_BW = min_freq_bw # Minimum frequency bandwidth (in Hz)
    MAX_FREQ_BW = max_freq_bw
    if (MIN_EVENT_DUR is not None) and (MIN_FREQ_BW is not None):  
        MIN_ROI_AREA = MIN_EVENT_DUR * MIN_FREQ_BW 
    else :
        MIN_ROI_AREA = None
    if (MAX_EVENT_DUR is not None) and (MAX_FREQ_BW is not None):  
        MAX_ROI_AREA = MAX_EVENT_DUR * MAX_FREQ_BW 
    else :
        MAX_ROI_AREA = None
    
    MAX_XY_RATIO = max_xy_ratio
    
    BIN_H = seed_level
    BIN_L = low_level

    """*********************** Convert into dB********* ********************"""
    #
    Sxx_dB = util.power2dB(Sxx_power, db_range=96) + 96

    """*********************** Remove stationnary noise ********************"""       
    #### Use median_equalizer function as it is fast reliable
    if REMOVE_RAIN: 
        # remove single vertical lines
        Sxx_dB_without_rain, _ = sound.remove_background_along_axis(Sxx_dB.T,
                                                            mode='mean',
                                                            N=1,
                                                            display=False)
        Sxx_dB = Sxx_dB_without_rain.T

    # ==================================================================================> TO COMMENT
    # remove single horizontal lines
    Sxx_dB_noNoise, _ = sound.remove_background_along_axis(Sxx_dB,
                                                    mode='median',
                                                    N=1,
                                                    display=False)
    # ==================================================================================> END TO COMMENT (change)

    Sxx_dB_noNoise[Sxx_dB_noNoise<=0] = 0
    
    """**************************** Find ROIS ******************************"""  
    # Time resolution (in s)
    DELTA_T = tn[1]-tn[0]
    # Frequency resolution (in Hz)
    DELTA_F = fn[1]-fn[0]

    # snr estimation to threshold the spectrogram
    _,bgn,snr,_,_,_ = sound.spectral_snr(util.dB2power(Sxx_dB_noNoise))
    if verbose :
        print('BGN {}dB / SNR {}dB'.format(bgn,snr))
        

    # binarization of the spectrogram to select part of the spectrogram with 
    # acoustic activity
    im_mask = rois.create_mask(
        Sxx_dB_noNoise,  
        mode_bin = 'absolute', 
        bin_h=BIN_H, 
        bin_l=BIN_L,
        # bin_h=snr+BIN_H,
        # bin_l=snr+BIN_L,
        # bin_h=util.add_dB(BIN_H,snr),
        # bin_l=util.add_dB(BIN_L,snr),
        )    
    
    if verbose :
        print('bin_h {}dB / bin_l {}dB'.format(util.add_dB(BIN_H,snr),util.add_dB(BIN_L,snr)))
        
    """**************************** Fusion ROIS ******************************"""  
    if type(FUSION_ROIS) is tuple :
        Ny_elements = round(FUSION_ROIS[0] / DELTA_T)
        Nx_elements = round(FUSION_ROIS[1] / DELTA_F)
        im_mask = closing(im_mask, footprint=np.ones([Nx_elements,Ny_elements]))

    # get the mask with rois (im_rois) and the bounding box for each rois (rois_bbox) 
    # and an unique index for each rois => in the pandas dataframe rois
    im_rois, df_rois  = rois.select_rois(
        im_mask,
        min_roi=MIN_ROI_AREA, 
        max_roi=MAX_ROI_AREA)
    
    """**************************** add a column ratio_xy ****************************"""  
    # add ratio x/y
    df_rois['ratio_xy'] = (df_rois.max_y -df_rois.min_y) / (df_rois.max_x -df_rois.min_x) 

    """************ remove some ROIs based on duration and bandwidth *****************"""  
    # remove min and max duration 
    df_rois['duration'] = (df_rois.max_x -df_rois.min_x) * DELTA_T 
    if MIN_EVENT_DUR is not None :
        df_rois = df_rois[df_rois['duration'] >= MIN_EVENT_DUR]
    if MAX_EVENT_DUR is not None :    
        df_rois = df_rois[df_rois['duration'] <= MAX_EVENT_DUR]
    df_rois.drop(columns=['duration'])

    # remove min and max frequency bandwidth 
    df_rois['bw'] = (df_rois.max_y -df_rois.min_y) * DELTA_F 
    if MIN_FREQ_BW is not None :
        df_rois = df_rois[df_rois['bw'] >= MIN_FREQ_BW]
    if MAX_FREQ_BW is not None :    
        df_rois = df_rois[df_rois['bw'] <= MAX_FREQ_BW]
    df_rois.drop(columns=['bw'])
    

    """**************************** Remove some ROIS ******************************"""  
    if len(df_rois) >0 :
        if REMOVE_ROIS_FMIN_LIM is not None:
            low_frequency_threshold_in_pixels=None
            high_frequency_threshold_in_pixels=None

            if isinstance(REMOVE_ROIS_FMIN_LIM, (float, int)) :
                low_frequency_threshold_in_pixels = max(1, round(REMOVE_ROIS_FMIN_LIM / DELTA_F))
            elif isinstance(REMOVE_ROIS_FMIN_LIM, (tuple, list, np.ndarray)) and len(REMOVE_ROIS_FMIN_LIM) == 2 :
                low_frequency_threshold_in_pixels = max(1, round(REMOVE_ROIS_FMIN_LIM[0] / DELTA_F))
                high_frequency_threshold_in_pixels = min(im_rois.shape[1]-1, round(REMOVE_ROIS_FMIN_LIM[1] / DELTA_F))
            else:
                raise ValueError ('REMOVE_ROIS_FMAX_LIM should be {None, a single value or a tuple of 2 values')

            # retrieve the list of labels that match the condition
            list_labelID = df_rois[df_rois['min_y']<=low_frequency_threshold_in_pixels]['labelID']
            # set to 0 all the pixel that match the labelID that we want to remove
            for labelID in list_labelID.astype(int).tolist() :
                im_rois[im_rois==labelID] = 0
            # delete the rois corresponding to the labelID that we removed in im_mask
            df_rois = df_rois[~df_rois['labelID'].isin(list_labelID)]

            if high_frequency_threshold_in_pixels is not None :
                # retrieve the list of labels that match the condition  
                list_labelID = df_rois[df_rois['min_y']>=high_frequency_threshold_in_pixels]['labelID']
                # set to 0 all the pixel that match the labelID that we want to remove
                for labelID in list_labelID.astype(int).tolist() :
                    im_rois[im_rois==labelID] = 0
                # delete the rois corresponding to the labelID that we removed in im_mask
                df_rois = df_rois[~df_rois['labelID'].isin(list_labelID)]

        if REMOVE_ROIS_FMAX_LIM is not None:
            if isinstance(REMOVE_ROIS_FMAX_LIM, (float, int)) :
                high_frequency_threshold_in_pixels = min(im_rois.shape[1]-1, round(REMOVE_ROIS_FMAX_LIM / DELTA_F))
            else:
                raise ValueError ('REMOVE_ROIS_FMAX_LIM should be {None, or single value')

            # retrieve the list of labels that match the condition  
            list_labelID = df_rois[df_rois['max_y']>=high_frequency_threshold_in_pixels]['labelID']
            # set to 0 all the pixel that match the labelID that we want to remove
            for labelID in list_labelID.astype(int).tolist() :
                im_rois[im_rois==labelID] = 0
            # delete the rois corresponding to the labelID that we removed in im_mask
            df_rois = df_rois[~df_rois['labelID'].isin(list_labelID)]
        
        if MAX_XY_RATIO is not None:
            df_rois = df_rois[df_rois['ratio_xy'] < MAX_XY_RATIO]  

    """**************************** Index calculation ******************************""" 

    # Convert boolean (True or False) into integer (0 or 1)
    im_mask_filtered = im_rois>0 * 1
    # number of ROIs / min
    nROI = len(df_rois) / (tn[-1] / 60) 
    # ROIs coverage in %
    aROI = im_mask_filtered.sum() / (im_mask_filtered.shape[0]*im_mask_filtered.shape[1]) *100 

    if verbose :
        print('===> nROI : {:.0f}#/min | aROI : {:.2f}%'.format(round(nROI), aROI))

    return nROI, aROI