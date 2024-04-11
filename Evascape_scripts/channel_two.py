# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:38:39 2023

@author: ear-field
"""

import numpy as np
import pandas as pd
import bambird as bb
import maad as maad
import random
import soundfile
from pathlib import Path
import scipy.signal as sig
import matplotlib.pyplot as plt
from toolbox import waveread


###
#sort geophony
###

def psd(vector, samprate = 44100): # Estimates the power spectral density "vec_p" from the signal's vector "vec_x"
    welch_window = sig.get_window(window="boxcar", Nx=512)
    frequ_array, psd_array = sig.welch(vector, samprate, welch_window)
    return frequ_array, psd_array

def psd_plot(frequ_array, psd_array, f_min = None, f_max = None): 
    if f_min == None:
        f_min = min(frequ_array)
    else :
        f_min = float(frequ_array[np.where(abs(frequ_array - f_min) == min(abs(frequ_array - f_min)))])
    if f_max == None:
        f_max = max(frequ_array)
    else :
        f_max = float(frequ_array[np.where(abs(frequ_array - f_max) == min(abs(frequ_array - f_max)))])
    i_min = int(np.where(frequ_array == f_min)[0])
    i_max = int(np.where(frequ_array == f_max)[0])
    plt.plot(frequ_array[i_min:i_max], psd_array[i_min:i_max])
    plt.show()

def psd_factor(vector, winbot=600, wintop=1200, samprate=44100): 
    # 1- Estimates the power spectral density "vec_p" from the signal's vector "vec_x"
    frequ_array, psd_array = psd(vector)
    # 2- Extracts "vec_a" from "vec_p" : 600-1200 Hz window
    start = (np.abs(frequ_array - winbot)).argmin()
    end = (np.abs(frequ_array - wintop)).argmin()
    vec_a =  psd_array[start:end]
    # 3- Estimate "mean_a" = rain intensity
    return np.mean(vec_a)

def geophony_psd(record_directory, psd_window_df):
    grab_df = bb.grab_audio_to_df(record_directory, 'wav')
    geophtype_list = list(psd_window_df.index)
    geoph_df = pd.DataFrame(columns = grab_df.columns)
    for geophtype in geophtype_list:
        geophtype_df = grab_df.loc[grab_df.categories == geophtype]
        geoph_df = pd.concat([geoph_df,geophtype_df])
    geoph_df['psd_factor'] = ['nan' for file in geoph_df.index]
    for file in geoph_df.index:
        soundtype = geoph_df.categories[file]
        soundtype_winbot = psd_window_df.loc[soundtype,'winbot']
        soundtype_wintop = psd_window_df.loc[soundtype,'wintop']
        waveform = waveread(geoph_df.fullfilename[file])
        psd_fact = psd_factor(waveform, winbot = soundtype_winbot, wintop = soundtype_wintop)
        geoph_df.psd_factor.loc[file] = psd_fact
        print(file, psd_fact)
    return geoph_df

def background_rms(record_directory):
    grab_df = bb.grab_audio_to_df(record_directory, 'wav')
    background_df = grab_df.copy()
    background_df['rms'] = ['nan' for file in background_df.index]
    for file in background_df.index:
        waveform = waveread(background_df.fullfilename[file])
        rms = maad.util.rms(waveform)
        background_df.rms.loc[file] = rms
        print(file, rms)
    return background_df

def geophony_sorting(psd_df):
    geophtype_list = list(psd_df.categories.unique())
    sorted_df = psd_df.copy()
    for geophtype in geophtype_list:
        print(geophtype)
        geophtype_df = sorted_df.loc[sorted_df.categories == geophtype]
        psd_array = np.array(geophtype_df.psd_factor)
        #interquartile range
        Q1 = np.percentile(psd_array, 25, interpolation = 'midpoint')
        Q3 = np.percentile(psd_array, 75, interpolation = 'midpoint') # first category max
        print('Q3 = ',Q3)
        IQR = Q3 - Q1
        box_max = Q3 + 1.5*IQR # third category min
        print('box_max = ',box_max)
        for file in geophtype_df.index :
            if geophtype_df.psd_factor.loc[file] < Q3:
                new_soundtype = geophtype + '_pw01'
            elif geophtype_df.psd_factor.loc[file] >= box_max:
                new_soundtype = geophtype + '_pw03'
            else:
                new_soundtype = geophtype + '_pw02'
            sorted_df.categories.loc[file] = new_soundtype
    return sorted_df

def aircraft_sorting(background_df, category = 'aircraft', bot_max = 0.045, top_max = 0.074):
    sorted_df = background_df.copy()
    sorted_df['categories'] = [category for file in sorted_df.index]
    for file in sorted_df.index :
        rms = sorted_df.rms.loc[file]
        if rms < bot_max:
            power = '_pw01'
        elif rms >= bot_max and rms < top_max:
            power = '_pw02'
        else:
            power = '_pw03'
        sorted_df.categories.loc[file] = category + power
    outlier = category + '_pw03'
    classified_df = sorted_df[sorted_df.categories != outlier ]    
    return classified_df



def dataframe_selection(sorted_df, percent_select = 0.25):
    categories = list(sorted_df.categories.unique())
    selection_df = pd.DataFrame(columns = sorted_df.columns)
    for category in categories:
        category_df = sorted_df[sorted_df.categories == category]
        file_nb = int(round(percent_select * len(category_df)))
        selection = random.sample(list(category_df.index), k = file_nb)
        selection_df = pd.concat([selection_df,category_df.loc[selection]])
    return selection_df

###########
# Analyse and normalize channel_2
###########

def channel2_analyze(sorted_df, max_dB = 100): #max_dB = max(volume_df.dBSPL)  
    soundtype_list = sorted_df.categories.unique()
    new_columns = ["mean_dB", "dBmax_ratio", "psd_factor", "rms_power", "rms_ratio", "ratio_mean", "ratio_std"]
    analyze_df = sorted_df.copy(deep=True)
    analyze_df = analyze_df.reindex(columns = [*analyze_df.columns.tolist(), *new_columns])
    for soundtype in soundtype_list :
        soundtype_df = analyze_df.loc[analyze_df.categories == soundtype]
        for file in soundtype_df.index:
            waveform = waveread(soundtype_df.fullfilename[file])
            leq = maad.spl.wav2leq(waveform, f = 44100, gain = 42)
            mean_dB = maad.util.mean_dB(leq)
            soundtype_df.mean_dB[file] = mean_dB
            dB_var = max_dB - mean_dB
            if dB_var < 0:
                print(file, "is louder than max_dB")
            soundtype_df.dBmax_ratio[file] = 1 / 10**(dB_var/20) #lier à l'enregistrement ou au soundtype ?  
            soundtype_df.rms_power[file] = maad.util.rms(waveform)
            print (file)
            soundtype_df.psd_factor[file] = psd_factor(waveform)
        rms_mean = np.mean(soundtype_df.rms_power)
        soundtype_df.rms_ratio = soundtype_df.rms_power / rms_mean        
        soundtype_df.ratio_mean = np.mean(soundtype_df.rms_ratio) 
        soundtype_df.ratio_std = np.std(soundtype_df.rms_ratio)
        analyze_df.update(soundtype_df)
    return analyze_df

def channel2_normalize(analysis_df, save_directory, samprate = 44100):
    save_df = analysis_df.copy()
    for file in analysis_df.index:
        raw_waveform = waveread(analysis_df.fullfilename[file])
        norm_waveform = raw_waveform / analysis_df.rms_power[file]
        weighted_waveform = norm_waveform * analysis_df.dBmax_ratio[file]
        save_name = file[:-4] + '_norm.wav'
        save_df.loc[file, 'norm_filename'] = save_name
        save_path = Path(f"{save_directory}/{analysis_df.categories[file]}/{save_name}")
        save_df.loc[file, 'norm_fullfilename'] = save_path
        save_path.parent.mkdir(exist_ok=True, parents=True) #creates parent folders if necessary
        soundfile.write(save_path, weighted_waveform, samprate)
        print(file)
    return save_df

