# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 14:33:44 2022

@author: Elie grinfeder
"""

import numpy as np
import pandas as pd
import bambird as bb
from pathlib import Path
from scipy.stats import shapiro, skew, kurtosis
import soundfile
from toolbox import waveread, flatsound, addin

## ROIs identification and extraction

def audio_extension(source_dir, save_path, min_duration = 60, sr = 44100):
    recording_df = bb.grab_audio_to_df(source_dir, 'wav')
    extension_df = recording_df.copy()
    len_min = min_duration*sr
    for recording in recording_df.index:
        vector_original = waveread(recording_df.fullfilename[recording])
        len_vector = len(vector_original)
        
        if len_vector < len_min :
            vector_empty = flatsound(val = 0, d = min_duration, sr = sr)
            vector_extended = addin(vector_empty, vector_original, time_code=0, ramp_duration = 0, sr = sr)
            file_name = recording[:-4] + "_ext.wav"
            extension_df.filename[recording] = file_name
            file_path = save_path / extension_df.categories[recording] / file_name
            extension_df.fullfilename[recording] = file_path
            file_path.parent.mkdir(exist_ok=True, parents=True) #creates parent folders if necessary
            soundfile.write(extension_df.fullfilename[recording], vector_extended, sr)
        else:
            file_path = save_path / extension_df.categories[recording] / extension_df.filename[recording]
            extension_df.fullfilename[recording] = file_path
            file_path.parent.mkdir(exist_ok=True, parents=True) #creates parent folders if necessary
            soundfile.write(extension_df.fullfilename[recording], vector_original, sr)
    return(extension_df)
        

def roi_extraction(recording_df, roi_dir, config_path): 
    
    params = bb.config.load_config(config_path) 
    
    #first check if directory already exists. If not create one inside it
    
    roi_df, csv_roi = bb.multicpu_extract_rois(recording_df,  #extract ROIs from recordings and save them in ROI_dir
                                           params = params['PARAMS_EXTRACT'],
                                           save_path = roi_dir)
    feature_df, csv_features = bb.multicpu_compute_features(roi_df, #calculate features in ROIs
                                        params      = params['PARAMS_FEATURES'],
                                        save_path   = roi_dir)
    cluster_df, csv_cluster = bb.find_cluster(feature_df, 
                                   params=bb.config.PARAMS['PARAMS_CLUSTER'],
                                   save_path = roi_dir)
    return cluster_df

#display spectrogram + ROIs
def display_all_rois(cluster_df, species_name = None):
    if species_name != None :
        species_df = cluster_df.loc[cluster_df.categories == species_name]
    else :
        species_df = cluster_df.copy()
    bird_list = species_df.filename.unique()
    for bird in bird_list :
        file_cluster = species_df.loc[species_df.filename == bird]
        bb.overlay_rois(file_cluster)
        
def extract_dominant_clusters(cluster_dataframe): #recoder pour prendre eb
    recording_list = cluster_dataframe.filename.unique()
    sorted_dataframe = pd.DataFrame(columns = cluster_dataframe.columns)
    for recording in recording_list:
        recording_cluster = cluster_dataframe.loc[cluster_dataframe.filename == recording]
        cluster_id = int(recording_cluster.auto_label.mode())
        dominant_cluster = recording_cluster.loc[recording_cluster.auto_label == cluster_id]
        sorted_dataframe = sorted_dataframe.append(dominant_cluster)
    return sorted_dataframe          

def delete_outlier(data):
    q75,q25 = np.percentile(data,[75,25])
    intr_qr = q75-q25
    min_value = q25-(1.5*intr_qr)
    max_value = q75+(1.5*intr_qr)
    new_data = [value for value in data if min_value < value < max_value]
    return new_data

def temporal_analysis(cluster_df) : #returns all intersing time (ist) & duration + mean & std
    global_df = cluster_df.copy()
    global_df['duration'] = global_df.max_t - global_df.min_t
    species_list = global_df.categories.unique()
    for species in species_list : #ist calculation for one species
        species_cluster = global_df.loc[global_df.categories == species]
        recording_list = species_cluster.filename.unique()
        
        for recording in recording_list:
            recording_cluster = global_df.loc[global_df.filename == recording].sort_values(by=["min_t"])
            
            #sort cluster by min_t
            for i in range(len(recording_cluster) - 1) : # ist calculation for one recording
                end_roi1 = recording_cluster.max_t.iloc[i] # time of first song ending
                start_roi2 = recording_cluster.min_t.iloc[i+1] # time of second song beginning
                intersing_time = start_roi2 - end_roi1
                
                #check if intersingtime value is too short or negative 
                if intersing_time < 0:
                    print ("WARNING : Negative ist was spotted. Please check song clustering.")
                    #print(f'species : {species}/trecording : {recording}')
                    global_df.loc[recording_cluster.index[i],['ist']] = np.nan() # ist is ignored
                elif intersing_time < 0.03:
                    print ("WARNING : Very small intersing time interval (<0.03 s) was spotted. Please check song clustering.")
                    #print(f'species : {species}/trecording : {recording}')
                    global_df.loc[recording_cluster.index[i],['ist']] = np.nan() # ist is ignored
                else:
                    global_df.loc[recording_cluster.index[i],['ist']] = intersing_time  # ist is recorded in df

        # temporal analysis
        ##ist_list = delete_outlier(ist_list)
        analysis_df = global_df[(~global_df.ist.isna()) & (global_df.categories == species)]
        file_list = (global_df.categories == species)
        global_df.loc[file_list,['ist_mean']] = np.mean(analysis_df.ist) #calculate ist mean
        global_df.loc[file_list,['ist_std']] = np.std(analysis_df.ist) #calculate ist std
        global_df.loc[file_list,["duration_mean"]] = np.mean(analysis_df.duration) #calculate ist mean
        global_df.loc[file_list,["duration_std"]]  = np.std(analysis_df.duration)
        global_df.loc[file_list,["ist_skewness"]] = skew(analysis_df.ist)
        global_df.loc[file_list,["ist_kurtosis"]] = kurtosis(analysis_df.ist)
        
        #Shapiro-Wilk test
        stat, p_value = shapiro(analysis_df.ist)
        global_df.loc[file_list,['shapiro_stat']] = stat 
        global_df.loc[file_list,['shapiro_pval']] = p_value 
        print(f'{species} : stat = {stat}\tp-value = {p_value}')
    
    temporal_df = global_df[['fullfilename_ts', 'categories', 
                             'min_f', 'min_t', 'max_f', 'max_t',
                             'fullfilename', 'filename',
                             'duration', 'duration_mean', 'duration_std', 
                             'ist', 'ist_mean', 'ist_std', 
                             'ist_skewness', 'ist_kurtosis',
                             'shapiro_stat','shapiro_pval']]

    return temporal_df


def temporal_summary(temporal_analysis_df):
    selection_df = temporal_analysis_df[['categories', 'duration_mean', 'duration_std', 
                                      'ist_mean', 'ist_std']].set_index('categories')
    summary_df = selection_df.drop_duplicates()
    return summary_df

# Intersing amplitude (ISA) weight register 
def amplitude_analysis(temporal_analysis_df): 
    global_df = temporal_analysis_df.copy()
    
    recording_list = global_df.filename.unique()   
    for recording in recording_list:
        file_list = global_df.loc[global_df.filename == recording].index
        
        # compute and store peak amplitude for all songs in one recording
        for file in file_list : 
            waveform = waveread(global_df.fullfilename_ts[file])
            peak_amp = max(waveform)
            global_df.loc[file, 'peak_amp'] = peak_amp
            #print(f"filename : {file}\t\tpeak_amp : {peak_amp}") #\n our retour à la ligne
            
        # calculate amplitude ratios for all songs in the recording with respect to highest peak song
        recording_df = global_df.loc[global_df.filename == recording]
        max_peak = max(recording_df.peak_amp) 
        global_df.loc[global_df.filename == recording, 'peak_ratio'] = recording_df.peak_amp/max_peak
        print(recording, "done")

    return global_df

# Normalize songs regarding interspecies volume difference
def normalize_allsongs(amplitude_analysis_df, species_volume_df, save_directory,samprate = 44100):
    global_df = amplitude_analysis_df.copy()

    for file in global_df.index:
        raw_song = waveread(global_df.fullfilename_ts[file])
        norm_song = raw_song/max(raw_song)
        intersing_ratio = float(global_df.peak_ratio[file])
        intersing_weighted_song = norm_song * intersing_ratio
        species_name = global_df.categories[file]
        interspecies_ratio = float(species_volume_df.pressure_ratio[species_name])
        interspecies_weighted_song = intersing_weighted_song * interspecies_ratio
        global_df.loc[file, 'species_dBSPL1m'] = species_volume_df.dBSPL[species_name]
        global_df.peak_amp[file] = max(interspecies_weighted_song)
        
        save_name = file[:-4] + '_norm.wav'
        save_path = Path(f"{save_directory}/{global_df.categories[file]}/{global_df.filename[file][:-4]}/{save_name}")
        global_df.loc[file, 'song_filename'] = save_name
        global_df.loc[file, 'song_fullfilename'] = save_path
        
        save_path.parent.mkdir(exist_ok=True, parents=True) #creates parent folders if necessary
        soundfile.write(save_path, interspecies_weighted_song, samprate)
        print (file)
    new_colnames = {"filename" : "source_filename" , 
                      'fullfilename' : 'source_fullfilename'}
    global_df = global_df.set_index('song_filename').drop(columns='fullfilename_ts').rename(columns = new_colnames)
    return global_df





    

