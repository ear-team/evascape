# -*- coding: utf-8 -*-
"""
Created on Thu May  4 14:52:46 2023

@author: ear-field
"""

#import packages
import pandas as pd
import maad as maad
import random
import os
import numpy as np
import bambird.config as cfg
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import scipy.stats

#import reconstructor scripts
#from toolbox import reverb, waveread, addin, show_spectrogram, bracket_ramp
from channel_one import audio_extension, roi_extraction, temporal_analysis, amplitude_analysis, normalize_allsongs
from channel_two import geophony_psd, geophony_sorting, channel2_analyze, channel2_normalize, background_rms, aircraft_sorting, dataframe_selection
from assemblage import database, database_overlap, database_all_isti, database_isti_average_analysis, database_isti_detailed_analysis, ist_comparison
from acoustic_index import compute_all_indices, average_all_indices, index_plot


# SETUP
###########

#Temporal directory
temp_dir = ''

# Evascape sounds directory
sound_dir = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\reconstructor_scripts\Evascape_20240405\Evascape_soundfiles")

# load sound dataframes

normch1_df = normsong_df = pd.read_csv(sound_dir / 'bird_channel.csv', sep=';', index_col = 0)
normch2_df = pd.read_csv(sound_dir / 'background_channel.csv', sep=';', index_col = 0)

# updating bird channel song paths
for sound in normch1_df.index:
    old_path = Path(normch1_df.song_fullfilename.loc[sound])
    path_parts = old_path.parts
    cut_index = path_parts.index('temp') + 2
    cut_tuple = tuple(path_parts[cut_index:])
    cut_path = Path(*cut_tuple)
    normch1_df.song_fullfilename.loc[sound] = sound_dir / 'bird_channel' / cut_path
normch1_df.to_csv(sound_dir / 'bird_channel.csv', sep=';')

# updating background channel song paths
for sound in normch2_df.index:
    old_path = Path(normch2_df.norm_fullfilename.loc[sound])
    path_parts = old_path.parts
    cut_index = path_parts.index('temp') + 2
    cut_tuple = tuple(path_parts[cut_index:])
    cut_path = Path(*cut_tuple)
    normch2_df.norm_fullfilename.loc[sound] = sound_dir / 'background_channel' / cut_path
normch2_df.to_csv(sound_dir / 'background_channel.csv', sep=';')


# CHANNEL ONE
###########


#check for files that are to short and extend them to CHUNK_DURATION(cf config bambird) if necessary:
save_path = temp_dir / "extended_files"
extension_df = audio_extension(source_dir = record_dir, save_path = save_path, min_duration = 60, sr = 44100)
extension_df.to_csv(temp_dir / 'extended_files.csv', sep=';')

# ROI extraction
CONFIG_FILE = "C:/Users/ecoac-field/OneDrive/Documents/Articles-Recherches/Reconstructor/reconstructor_scripts/reconstructor/config_bambird.yaml"
params = cfg.load_config(CONFIG_FILE) 

#recording_df = bb.grab_audio_to_df(source_dir, 'wav') #lists all recordings in a given directory, considers folders as sound categories
cluster_df = roi_extraction(recording_df = extension_df, roi_dir = song_dir, config_path = CONFIG_FILE)

# # Display spectrogram
# display_file = "MNHN-SO-2016-12457_extrl.wav" #"MNHN-SO-2016-9920_extr.wav", "MNHN-SO-2016-12457_extrl.wav"
# file_cluster = cluster_df.loc[cluster_df.filename == display_file]
# bb.overlay_rois(file_cluster)

# Temporal analysis = intersing time computation
temporal_analysis_df = temporal_analysis(cluster_df)
temporal_analysis_df.to_csv(temp_dir / 'temporal_analysis.csv', sep=';')

# Amplitude analysis = intersing amplitude ratio computation
amplitude_analysis_df = amplitude_analysis(temporal_analysis_df)
amplitude_analysis_df.to_csv(temp_dir / 'amplitude_analysis.csv', sep=';')


# import JS species dBSPL at 1m values
dBSPL = [90,86, 78, 80, 75, 88, 87, 100]
pressure = maad.spl.dBSPL2pressure(dBSPL)
pressure_ratio = pressure/max(pressure)
volume_dic = {'species' : ["erirub", "fricoe", "perate","phycol", "regreg", "sylatr", "turmer", "turphi"],
              'dBSPL' : dBSPL,
              'pressure' : pressure,
              'pressure_ratio' : pressure_ratio}
species_volume_df = pd.DataFrame(volume_dic).set_index('species')
species_volume_df.to_csv(temp_dir / 'species_volume.csv', sep=';')

# Song normalization
save_directory = temp_dir / 'norm_song'
normsong_df = normalize_allsongs(amplitude_analysis_df, species_volume_df, save_directory)
normsong_df.to_csv(temp_dir / 'normsong.csv', sep=';')


# CHANNEL TWO
##########

# original channel two recording path (geophony, ambient sound and insects)
record_directory = "C:/Users/ecoac-field/OneDrive/Documents/Articles-Recherches/Reconstructor/Samples/channel_2"

# setup power spectral density window
psd_window = {'soundtype' : ["rain", "wind", "wind_and_rain"],
              'winbot' : [600,0,0],
              'wintop' : [1200,500,500]}
psd_window_df = pd.DataFrame(psd_window).set_index('soundtype')

# computation of powerspectral density
psd_df = geophony_psd(record_directory, psd_window_df)
psd_df.to_csv(temp_dir / 'geophony_psd.csv', sep=';')

# # display psd
# a = np.array(psd_df.psd_factor.loc[psd_df.categories == 'wind_and_rain'])
# hist, bins, _ = plt.hist(x=a, bins=100, color='#0504aa',
#                             alpha=0.7, rwidth=0.85)
# plt.xscale('log')
# plt.show()
# "x=np.log(a.astype(np.float32))"

# geophony sorting based on power spectral density
sorted_geoph_df = geophony_sorting(psd_df)
sorted_geoph_df.to_csv(temp_dir / 'sorted_geophony.csv', sep=';')

# compute rms ratio between recordings and peak ratio between recordings and maxdB
ch2_analysis_df = channel2_analyze(record_directory, sorted_geoph_df)
ch2_analysis_df.to_csv(temp_dir / 'channel2_data.csv', sep=';')

# normalize channel two recordings
save_directory = temp_dir / 'norm_ch2'
normch2_df = channel2_normalize(ch2_analysis_df, save_directory)
normch2_df.to_csv(temp_dir / 'normch2.csv', sep=';')

# aircraft sorting, analysis and normalization
record_directory = Path("F:/aircraft_july_2019")
background_df = background_rms(record_directory)
background_df.to_csv(temp_dir / 'aircraft_rms.csv', sep=';')


analysis_array = np.array(background_df.rms).reshape(-1,1)
n = len(analysis_array)
gm = GaussianMixture(n_components=2, random_state=0).fit(analysis_array)
mu1 = gm.means_[0]
mu2 = gm.means_[1]
sigma1 = np.sqrt(gm.covariances_[0])
sigma2 = np.sqrt(gm.covariances_[1])
w1 = gm.weights_[0]
w2 = gm.weights_[1]
n1 = int(n * gm.weights_[0])
n2 = int(n * gm.weights_[1])

def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

intersect = solve(mu1,mu2,sigma1,sigma2)


sorted_df = aircraft_sorting(background_df, 
                             category = 'aircraft', 
                             bot_max = 0.045, 
                             top_max = 0.074) #aircraft are sorted along rms power instead of psd_factor
sorted_df.to_csv(temp_dir / 'aircraft_sorting.csv', sep=';')
random.seed(444)
selection_df = dataframe_selection(sorted_df, percent_select = 0.10) #extract 10% of recordings from each power category
del_list = ['S4A03544_20190703_050000.wav', 'S4A03544_20190703_191500.wav', 
            'S4A03544_20190708_234500.wav', 'S4A03544_20190711_211500.wav',
            'S4A03544_20190713_230000.wav', 'S4A03544_20190715_214500.wav',
            'S4A03544_20190723_024500.wav', 'S4A03544_20190725_130000.wav',
            'S4A03544_20190727_213000.wav', 'S4A03544_20190731_120000.wav',
            'S4A03895_20190701_033000.wav', 'S4A03895_20190702_221500.wav',
            'S4A03895_20190707_183000.wav', 'S4A03895_20190711_050000.wav',
            'S4A03895_20190712_194500.wav', 'S4A03895_20190714_211500.wav',
            'S4A03895_20190715_221500.wav', 'S4A03895_20190715_223000.wav',
            'S4A03895_20190715_224500.wav', 'S4A03895_20190718_184500.wav',
            'S4A03895_20190719_044500.wav', 'S4A03895_20190719_220000.wav',
            'S4A03895_20190725_054500.wav', 'S4A03895_20190731_213000.wav',
            'S4A03536_20190702_220000.wav', 'S4A03536_20190702_224500.wav',
            'S4A03536_20190711_184500.wav', 'S4A03536_20190713_223000.wav',
            'S4A03536_20190721_141500.wav', 'S4A03536_20190722_073000.wav',
            'S4A03536_20190722_210000.wav', 'S4A03536_20190726_133000.wav',
            'S4A03536_20190729_011500.wav', 'S4A03536_20190730_094500.wav',
            'S4A03536_20190730_204500.wav', 'S4A03544_20190709_220000.wav',
            'S4A04430_20190702_220000.wav', 'S4A04430_20190708_043000.wav',
            'S4A04430_20190711_220000.wav', 'S4A04430_20190715_221500.wav',
            'S4A04430_20190717_224500.wav', 'S4A04430_20190719_124500.wav',
            'S4A04430_20190721_054500.wav', 'S4A04430_20190721_174500.wav',
            'S4A04430_20190722_070000.wav', 'S4A04430_20190725_161500.wav',
            'S4A04430_20190729_221500.wav', 'S4A04430_20190731_234500.wav']
selection_df = selection_df.loc[~selection_df.filename.isin(del_list)]
selection_df.to_csv(temp_dir / 'aircraft_selection.csv', sep=';')


analysis_df = channel2_analyze(selection_df, max_dB = 100)
analysis_df.to_csv(temp_dir / 'aircraft_analysis.csv', sep=';')

aircraft_df = channel2_normalize(analysis_df, temp_dir / 'norm_ch2', samprate = 44100)
aircraft_df.to_csv(temp_dir / 'aircraft_normalized.csv', sep=';')

#mix geophony and aircraft
geobio_df_path = temp_dir / 'normch2.csv'
geobio_df = pd.read_csv(geobio_df_path, sep=';', index_col = 0)
geobio_df = geobio_df.loc[~geobio_df.categories.isin(['aircraft_pw01','aircraft_pw02','aircraft_pw03'])]
normch2_df = pd.concat([geobio_df,aircraft_df])
normch2_df.to_csv(temp_dir / 'normch2.csv', sep=';')


# Overlap
##########

#check if there is a significant effects of biodiversity type (richness/abundance) on overlap amount
rand_rich_overlap, rand_ab_overlap = database_overlap(normch1_df, normch2_df,  d_min = 10, d_max = 80,
                                     channel1_list = ["erirub", "fricoe", "perate","phycol", "regreg", "sylatr", "turmer", "turphi"],
                                     impulse_response = None, random_behavior = True, 
                                     all_combinations = False, sample_size = 80, duration = 60, samprate = 44100)

f, p = scipy.stats.f_oneway(rand_rich_overlap, rand_ab_overlap) # p-value >0.05, effect of biodiversity type is not significant

desync_rich_overlap, desync_ab_overlap = database_overlap(normch1_df, normch2_df, d_min = 0, d_max = 80,
                                         channel1_list = ["erirub", "fricoe", "perate","phycol", "regreg", "sylatr", "turmer", "turphi"],
                                         impulse_response = None, random_behavior = False, 
                                         all_combinations = False, sample_size = 80, duration = 60, samprate = 44100)

f, p = scipy.stats.f_oneway(desync_rich_overlap, desync_ab_overlap)  # p-value >0.05, efefct of biodiversity type is not not significant

#check if there is a significant effects of simulation type (random/desync) on overlap amount
rand_overlap = rand_rich_overlap + rand_ab_overlap
desync_overlap = desync_rich_overlap + desync_ab_overlap
f, p = scipy.stats.f_oneway(rand_overlap, desync_overlap ) 
# f = 56.30994901555806
# p = 6.258609679819664e-13
# p-value < 0.05, effect of simulation type on overlap amount is significant



# Generate Database
########

random.seed(444)
database_dir = Path("D:/Database_20240315_Desync")               
database_df = database(normch1_df, normch2_df, save_dir = database_dir,
                         d_min = 10, d_max = 80,
                         richness_list = [1, 2, 3, 4 ,5 , 6, 7, 8], 
                         abundance_lvls = [1, 2, 3, 4 ,5],  
                         channel1_list = ["erirub", "fricoe", "perate","phycol", 
                                          "regreg", "sylatr", "turmer", "turphi"],
                         channel2_list = ['quiet','ambient_sound', 'rain_pw01', 
                                          'rain_pw02','rain_pw03','wind_pw01',
                                          'wind_pw02','wind_pw03', 'aircraft_pw01',
                                          'aircraft_pw02', 'tettigonia_veridissima'],
                         impulse_response = None, random_behavior = False, anonymous_ID = False,
                         all_combinations = False, sample_size = 10, duration = 100, samprate = 44100)
database_df.to_csv(database_dir / 'database_data.csv', sep=';')


# INDEX calculation + ploting

#Indice calcultation
index_df = compute_all_indices(database_df)
index_df = index_df.rename(columns = {'ROI_nb' : 'nROI', 'ROI_cov' : 'aROI', 'BIO' : 'BI'})
index_path = database_dir / 'index.csv'
index_df.to_csv(index_path, sep=';')

#Average all indices
index_list = ['nROI','aROI','ACI','H', 'NDSI', 'ADI', 'BI']
average_df = average_all_indices(index_df, index_list)
average_path = database_dir / 'average.csv'
average_df.to_csv(average_path, sep=';') 

#ISTI calculation
all_isti_df = database_all_isti(database_dir)
all_isti_path = database_dir / 'all_isti.csv'
all_isti_df.to_csv(all_isti_path, sep=';') 

average_isti_df = database_isti_average_analysis(all_isti_df)
average_isti_path = database_dir / 'average_isti.csv'
average_isti_df.to_csv(average_isti_path, sep=';') 

detailed_isti_df = database_isti_detailed_analysis(all_isti_df)
detailed_isti_path = database_dir / 'detailed_isti.csv'
detailed_isti_df.to_csv(detailed_isti_path, sep=';') 

# Comparison between observed and simulated ISTI
observed_ISTI_path = temp_dir / 'temporal_analysis.csv'
observed_ISTI_df = pd.read_csv(observed_ISTI_path, sep=';', index_col = 0)
database_dir = Path("D:/Database_20240315_Desync") 
simulated_ISTI_path = database_dir / 'all_isti.csv'
simulated_ISTI_df = pd.read_csv(simulated_ISTI_path, sep=';', index_col = 0)

isti_solo_df = ist_comparison(observed_ISTI_df, simulated_ISTI_df, solo_simulation = True)
isti_solo_path = database_dir / 'observed_vs_solo_isti.csv'
isti_solo_df.to_csv(isti_solo_path, sep=';') 
isti_multi_df = ist_comparison(observed_ISTI_df, simulated_ISTI_df, solo_simulation = False)
isti_multi_path = database_dir / 'observed_vs_multi_isti.csv'
isti_multi_df.to_csv(isti_multi_path, sep=';')

#Plot Dataframe
index_list = ['nROI','aROI','ACI','H', 'NDSI', 'ADI', 'BI']
for index in index_list:
    fig, correlation_df = index_plot(index_df, index)
    fig_path = database_dir / f'Figures/{index}_fig.png'
    fig_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(fig_path)
    correlation_path = database_dir / f'Figures/{index}_correlation.csv'
    correlation_df.to_csv(correlation_path, sep=';') 



#Psychoacoustic Database
##########################

random.seed(333)
global_dir = Path(r"D:/Psychoac_20240329_5S")
if not os.path.exists(global_dir):
    os.mkdir(global_dir)

condition_dic = {   'Database_ID' :    ['A','B','C','D','E','F','G','H','I','J','K'],
                    'Focus' :          ['richness','richness','richness','richness','richness','abundance','abundance','abundance','abundance','abundance','richness'], #vermillion
                    'Behavior' :       [True,False,True,True,True, True, False,True,True,True,True], 
                    'Propagation' :    [True,True,False,True,True,True,True,False,True,True,True], 
                    'Background' :     ['ambient','ambient','ambient','geophony','aircraft','ambient','ambient','ambient','geophony','aircraft','no_background']
                }

condition_df = pd.DataFrame(condition_dic).T.T.set_index('Database_ID')
condition_df.to_csv(global_dir / 'experimental_conditions.csv', sep=';')

for condition in condition_df.index :

#for condition in condition_df.index :
    
    # Focus on richness or abundance
    if condition_df.Focus[condition] == 'richness':
        richness_cond = [1, 2, 3, 4 ,5]
        abundance_cond = [1]
    else:
        richness_cond = [1]
        abundance_cond = [1, 2, 3, 4 ,5]
        
    # Propagation
    if condition_df.Propagation[condition] == True:
        d_min, d_max = 10, 80
    else :
       d_min = d_max = 45 # ?
       
    #Background
    if condition_df.Background[condition] == 'ambient' :
        background_cond = ['ambient_sound']
    elif condition_df.Background[condition] == 'no_background' :
        background_cond = ['quiet']
    elif condition_df.Background[condition] == 'geophony':
        background_cond = ['rain_pw02']
    elif condition_df.Background[condition] == 'aircraft':
        background_cond = ['aircraft_pw02']
    
    random.seed(333) 
    database_dir = global_dir / f"Cond_{condition}"               
    test_database = database(normch1_df = normch1_df, 
                             normch2_df = normch2_df, 
                             save_dir = database_dir,
                             d_min = d_min, 
                             d_max = d_max,
                             richness_list = richness_cond, 
                             abundance_lvls = abundance_cond,  
                             channel1_list = ["erirub", "fricoe", "perate","phycol", "regreg", "sylatr", "turmer", "turphi"],
                             channel2_list = background_cond,
                             impulse_response = None, 
                             random_behavior = not(condition_df.Behavior[condition]), 
                             anonymous_ID = True,
                             database_label = f'evascape_{condition}',
                             all_combinations = False, 
                             sample_size = 8, 
                             duration = 5,
                             samprate = 44100)
    test_database.to_csv(database_dir / f'database_{condition}.csv', sep=';')

#ISTI calculation
database_dir = Path(r"D:/Psychoac_20240315_OK")
for condition in os.listdir(database_dir) :
    condition_path = os.path.join(database_dir, condition)
    all_isti_df = database_all_isti(condition_path)
    all_isti_path = Path(condition_path) / 'all_isti.csv'
    all_isti_df.to_csv(all_isti_path, sep=';') 