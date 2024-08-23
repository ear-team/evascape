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
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t
import seaborn as sns

#setup
SAMPLING_RATE = 44100


#### Indices computing

def compute_all_indices(database_df, root_dir, samprate=SAMPLING_RATE):
    indices_df = database_df.copy()
    file_count = 1
    for file in indices_df.index:
        path = Path(root_dir) / Path(indices_df.fullfilename[file])
        vector = waveread(path)
        vector[vector == 0] = 10**(-8)
        
        #Spectrogram variables
        Sxx_power,tn,fn,ext = sound.spectrogram(x = vector, fs = samprate, mode='amplitude')
        Sxx_power[Sxx_power == 0] = 10**(-25)
        
        #ROI indices
        nROI, aROI = region_of_interest_index2(Sxx_power, tn, fn,   
                                        seed_level=12, 
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

####  Plot indices 

def plot_indices(
            indices_df, 
            index, 
            width      =25, 
            height     =12, 
            **kwargs):

    default_info_dic = {'channel2' : [
                            'no_background',
                            'ambient_sound',  
                            'aircraft_pw01',
                            'aircraft_pw02',
                            'rain_pw01', 
                            'rain_pw02', 
                            'rain_pw03',
                            'tettigonia_veridissima', 
                            'wind_pw01', 
                            'wind_pw02', 
                            'wind_pw03'],
                        'color': [
                            'grey', 
                            'black', 
                            'violet',
                            'darkviolet',
                            'deepskyblue', 
                            'royalblue', 
                            'darkblue', 
                            'green', 
                            'lightcoral', 
                            'red', 
                            'brown'],
                        'label' : [
                            'no background', 
                            'ambient sound', 
                            'light aircraft noise',
                            'strong aircraft noise',
                            'light rain', 
                            'medium rain', 
                            'strong rain',
                            'tettigonia viridissima', 
                            'light wind', 
                            'medium wind', 
                            'strong wind']
                                            }
    
    # plot box plots of the INDICE for each channel2
    sns.set_theme(context='talk', style='whitegrid')

    info_dic = kwargs.pop('info_dic', default_info_dic)

    # set INDICE
    INDICE = index   

    # create a copy of the dataframe indices_df
    df = indices_df.copy()

    # sort the column channel2 following info_dic['channel2']
    df['channel2'] = pd.Categorical(df['channel2'], info_dic['channel2'])

    # create a figure with 2 subplots
    fig, axs = plt.subplots(nrows = 1, ncols = 2, sharey='row', figsize=(width,height), gridspec_kw={'width_ratios': [8, 5]})

    # filter the dataframe to keep only the rows where the richness or the abundance is equal to 1
    df = df[(df['abundance'] == 1)|(df['richness'] == 1)]

    # bonferroni correction
    alpha = 0.05 / (len(['richness','abundance']) * len(df['channel2'].unique()))

    for jj, GT in enumerate(['richness','abundance']): 
        print("----- {} {} -----".format(INDICE, GT))

        # filter the dataframe to keep only the rows where the GT is equal to
        if GT == 'richness':
            subdf = df[(df['abundance']==1)]
        else :
            subdf = df[(df['richness']==1)]

        # print the spearman correlation coefficient between the mean of the index and the GT
        for CH2 in subdf['channel2'].unique():
            r, p = spearmanr(subdf[subdf["channel2"]==CH2][INDICE], subdf[subdf["channel2"]==CH2][GT])
            # bonferroni correction
            if (p < alpha):
                print("{:.2f}*".format(r))
            else:
                print("{:.2f}".format(r))

        # # calculate the mean and the standard deviation of the index for each channel2 in the dataframe subdf
        # mean_df = subdf[['channel2']+[GT]+[INDICE]].groupby(['channel2',GT]).apply(np.mean, axis=0)

        # # calculate the standard error of the mean
        # # from https://www.statology.org/standard-error-of-mean-python/
        # n = subdf[['channel2']+[GT]+[INDICE]].groupby(['channel2',GT]).count()
        # # print("count : {}".format(n))
        # error_df = subdf[['channel2']+[GT]+[INDICE]].groupby(['channel2',GT]).apply(np.std, axis=0, ddof=1) / np.sqrt(n)

        # # calculate the average mean of the index for each GT
        # average_mean_df = mean_df.reset_index(level='channel2', drop=True).groupby([GT]).apply(np.mean, axis=0)

        # # merge mean and margin_error dataframes
        # mean_df = pd.merge(mean_df, error_df, on = ['channel2',GT], suffixes = ('_mean', '_stdv'))

        # # reset the index
        # mean_df = mean_df.reset_index()
        # average_mean_df = average_mean_df.reset_index()

        # # rename the last column of the dataframe averagemean_df to INDICE
        # average_mean_df.columns = [GT,INDICE]
        
        # Create the box plot
        sns.boxplot(
            data=subdf,
            x=GT,
            y=INDICE,
            hue='channel2',
            ax=axs[jj],
            fill=True, 
            legend='brief',
            gap=.1,
            notch=True, 
            palette = info_dic['color'],
            saturation=1,
            linewidth=0,
            )

        # # Loop through each hue category
        for container, color in zip(axs[jj].containers, info_dic['color']):

            for flier in container.fliers:
                flier.set(marker='o', color=color, alpha=0.5)
                flier.set_color(color)

            # Set whisker caps (endpoints)
            for line in container.whiskers:
                line.set_color(color)  # color for whisker caps and lines
                line.set_linewidth(1)  # thickness of whisker caps and lines
        
        axs[jj].get_legend().remove()            
        axs[jj].set_xlabel(GT,fontsize = 30)
        axs[jj].set_ylabel(axs[jj].get_ylabel(), fontsize = 30)
        axs[jj].tick_params(labelsize=20)  # Set size for all tick labels
        axs[jj].autoscale()
            
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, info_dic['label'], fontsize = 20, bbox_to_anchor=(1.07, 0.65))
    fig.suptitle(index, fontsize = 100, x = 0.99, y=0.8)
    
    return fig



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