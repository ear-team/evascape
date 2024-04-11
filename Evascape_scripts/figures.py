# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 10:18:06 2023

@author: ear-field
"""

import numpy as np
import pandas as pd
import maad as maad
import random
import matplotlib.pyplot as plt
import soundfile
from pathlib import Path

from toolbox import waveread, addin, flatsound, show_spectrogram, bracket_ramp, rms_normalize, one_normalize, short
from bird_behavior import empty_abundance_dataframe, bird_behavior, timeline_chart
from assemblage import singing_session

temp_dir = Path("C:/Users/ecoac-field/OneDrive/Documents/Articles-Recherches/Reconstructor/Samples/temp")
fig_dir = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\Figures\fig_20240222")

normch1_path = temp_dir / 'normsong.csv'
normch1_df = pd.read_csv(normch1_path, sep=';', index_col = 0)

normch2_path = temp_dir / 'normch2.csv'
normch2_df = pd.read_csv(normch2_path, sep=';', index_col = 0)

samprate = 44100
duration = 60

#BIOPHONY DESCRIPTION
########
species_names = ['fricoe','phycol','sylatr','turphi']
abundance_df = empty_abundance_dataframe(normch1_df)
abundance_df.loc[species_names] = 1

#BEHAVIOR
#########
random.seed(333)
behavior_df = bird_behavior(normch1_df, abundance_df, duration)
timeline_fig = timeline_chart(behavior_df, y_size = 0.5)
timeline_path = fig_dir / 'timeline_fig'
timeline_fig.savefig(timeline_path)


#SINGING SESSION
########
bird_list = list(behavior_df.bird_filename.unique())
species_list =[] 
for bird_filename in bird_list:
    species = behavior_df.categories[behavior_df.bird_filename == bird_filename].unique()[0]
    species_list += [species]
bird_df = pd.DataFrame(bird_list, index = species_list)

#fricoe
fricoe_session = singing_session(behavior_df, bird_df.loc['fricoe'][0], duration = 60, samprate = 44100, flims = [0, 15000])
fricoe_spectrogram = show_spectrogram(fricoe_session, vmin = -70, vmax = -20, samprate = samprate, )
save_path = fig_dir / 'fricoe_session.wav'
soundfile.write(save_path, fricoe_session, samprate)
spectro_path = fig_dir / 'fricoe_session'
fricoe_spectrogram.savefig(spectro_path)

#phycol
phycol_session = singing_session(behavior_df, bird_df.loc['phycol'][0], duration = 60, samprate = 44100, flims = [0, 15000])
phycol_spectrogram = show_spectrogram(phycol_session, vmin = -70, vmax = -20, samprate = samprate)
save_path = fig_dir / 'phycol_session.wav'
soundfile.write(save_path, phycol_session, samprate)
spectro_path = fig_dir / 'phycol_session'
phycol_spectrogram.savefig(spectro_path)

#sylatr
sylatr_session = singing_session(behavior_df, bird_df.loc['sylatr'][0], duration = 60, samprate = 44100, flims = [0, 15000])
sylatr_spectrogram = show_spectrogram(sylatr_session, vmin = -70, vmax = -20, samprate = samprate)
save_path = fig_dir / 'sylatr_session.wav'
soundfile.write(save_path, sylatr_session, samprate)
spectro_path = fig_dir / 'sylatr_session'
sylatr_spectrogram.savefig(spectro_path)

#turphi
turphi_session = singing_session(behavior_df, bird_df.loc['turphi'][0], duration = 60, samprate = 44100, flims = [0, 15000])
turphi_spectrogram = show_spectrogram(turphi_session, vmin = -70, vmax = -20, samprate = samprate)
save_path = fig_dir / 'turphi_session.wav'
soundfile.write(save_path, turphi_session, samprate)
spectro_path = fig_dir / 'turphi_session'
turphi_spectrogram.savefig(spectro_path)


#PROPAGATED SINGING SESSION
#########

#fricoe[40m]
fricoe_propagated =  maad.spl.apply_attenuation(fricoe_session, samprate, r = 60)
fricoe_propagated_spectro = show_spectrogram(fricoe_propagated, vmin = -70, vmax = -20, samprate = samprate)
save_path = fig_dir / 'fricoe_propagated.wav'
soundfile.write(save_path, fricoe_propagated, samprate)

#phycol [20m]
phycol_propagated =  maad.spl.apply_attenuation(phycol_session, samprate, r = 50)
phycol_propagated_spectro = show_spectrogram(phycol_propagated, vmin = -70, vmax = -20, samprate = samprate)
save_path = fig_dir / 'phycol_propagated.wav'
soundfile.write(save_path, phycol_propagated, samprate)

#sylatr [10m]
sylatr_propagated =  maad.spl.apply_attenuation(sylatr_session, samprate, r = 30)
sylatr_propagated_spectro = show_spectrogram(sylatr_propagated, vmin = -70, vmax = -20, samprate = samprate)
save_path = fig_dir / 'sylatr_propagated.wav'
soundfile.write(save_path, sylatr_propagated, samprate)

#turphi [60m]
turphi_propagated =  maad.spl.apply_attenuation(turphi_session, samprate, r = 100)
turphi_propagated_spectro = show_spectrogram(turphi_propagated, vmin = -70, vmax = -20, samprate = samprate)
save_path = fig_dir / 'turphi_propagated.wav'
soundfile.write(save_path, turphi_propagated, samprate)


#BIOPHONY ASSEMBLAGE
#######
chorus_list  = [fricoe_propagated, phycol_propagated, sylatr_propagated, turphi_propagated]

chorus = flatsound(val = 0, d=60, sr=samprate)
for bird in chorus_list:
    chorus = addin(chorus, bird, time_code=0, ramp_duration = 0, sr= samprate)

chorus_spectro = show_spectrogram(chorus, vmin = -70, vmax = -20, samprate = samprate)
save_path = fig_dir / 'chorus.wav'
soundfile.write(save_path, chorus, samprate)


#CHANNEL_2
#######
#S4A03536_20190720_030000_norm.wav
channel2_path = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\Samples\temp\norm_ch2\ambient_sound\S4A03536_20190720_030000_norm.wav")
channel2 = waveread(channel2_path)
channel2_spectro = show_spectrogram(channel2, vmin = -70, vmax = -20, samprate = samprate)


#FINAL ASSEMBLAGE
######

final_assemblage = addin(base_sound = channel2, 
                       added_sound = chorus, 
                       time_code = 0, 
                       ramp_duration = 0, sr = samprate)
final_assemblage = bracket_ramp(final_assemblage)
final_assemblage_spectro = show_spectrogram(final_assemblage, vmin = -70, vmax = -20, samprate = samprate)
save_path = fig_dir / 'final_assemblage.wav'
soundfile.write(save_path, final_assemblage, samprate)

norm_assemblage = one_normalize(final_assemblage)
norm_assemblage_spectro = show_spectrogram(norm_assemblage, vmin = -50, vmax = -20, samprate = samprate)
save_path = fig_dir / 'norm_assemblage_far2.wav'
soundfile.write(save_path, norm_assemblage, samprate)

assemblage_sample = short(norm_assemblage, start = 25, end = 35)    
save_path = fig_dir / 'norm_assemblage_sample.wav'  
soundfile.write(save_path, assemblage_sample, samprate)  


# TIMELINE VS SPECTROGRAM
######   

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize=(30,15))

#timeline
plt.sca(axs[0])
plt.figure(timeline_fig)
axs[0].set_xlabel('Time [s]', size = 20)

#spectrogram
fig_spectro, axs_spectro = plt.subplots(nrows = 2, ncols = 1)


axs_spectro[0].imshow(fricoe_spectrogram)

axs_spectro[1].imshow(phycol_spectrogram)

axs_spectro[2].imshow(sylatr_spectrogram)

axs_spectro[3].imshow(turphi_spectrogram)

plt.figure(fig_spectro)
axs[1].set_title('singing session spectrograms', size=20)
#axs[i1].set_xlabel('frequency [Hz]', size = 20)

fricoe_spectrogram = show_spectrogram(fricoe_session, vmin = -70, vmax = -20, samprate = samprate)

spectrogram1 = fricoe_spectrogram
spectrogram2 = phycol_spectrogram

fig, axs = plt.subplots(nrows = 2, ncols = 1)
axs[0].imshow(spectrogram1)
axs[1].imshow(spectrogram2)
plt.show()

N = 4096
vmin = None
vmax = None


fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize=(30,15))

fricoe_Sxx_power,tn,fn,ext = maad.sound.spectrogram(fricoe_session, 44100, nperseg=N, noverlap=N//2, flims = [0, 15000], mode = 'amplitude')
fig_kwargs = {'vmax': vmax,
                  'vmin':vmin,
                  'extent':ext,
                  'figsize':(4,13),
                  'title':'Power spectrogram density (PSD)',
                  'xlabel':'Time [sec]',
                  'ylabel':'Frequency [Hz]',
                  }
fricoe_Sxx_dB = maad.util.power2dB(fricoe_Sxx_power) # convert into dB
maad.util.plot2d(fricoe_Sxx_dB,**fig_kwargs, ax = axs[0])

phycol_Sxx_power,tn,fn,ext = maad.sound.spectrogram(phycol_session, 44100, nperseg=N, noverlap=N//2, flims = [0, 15000], mode = 'amplitude')
fig_kwargs = {'vmax': vmax,
                  'vmin':vmin,
                  'extent':ext,
                  'figsize':(4,13),
                  'title':'Power spectrogram density (PSD)',
                  'xlabel':'Time [sec]',
                  'ylabel':'Frequency [Hz]',
                  }
phycol_Sxx_dB = maad.util.power2dB(phycol_Sxx_power) # convert into dB
maad.util.plot2d(phycol_Sxx_dB,**fig_kwargs, ax = axs[1])

# Comparaison ISTI
observed_ISTI_path = temp_dir / 'temporal_analysis.csv'
observed_ISTI_df = pd.read_csv(observed_ISTI_path, sep=';', index_col = 0)
database_dir = Path("D:/Database_20240315_Desync") 
simulated_ISTI_path = database_dir / 'all_isti.csv'
simulated_ISTI_df = pd.read_csv(simulated_ISTI_path, sep=';', index_col = 0)

species_list = list(observed_ISTI_df.categories.unique())
species_list.sort()

#fig, ax = plt.subplots(nrows = 1, ncols = len(species_list * 2), figsize=(15,4), sharey='row')
toplot_list = []
label_list = []
for species in species_list:
    
    # observed ISTI bloxplot
    obs_species_df = observed_ISTI_df.loc[observed_ISTI_df.categories == species]
    ISTI_list = list(obs_species_df.ist)
    ISTI_list = [x for x in ISTI_list if str(x) != 'nan']
    toplot_list += [ISTI_list]
    label_list += [f'{species}_obs']
    
    # simulated ISTI boxplot
    sim_species_df = simulated_ISTI_df.loc[simulated_ISTI_df.species == species]
    toplot_list += [list(sim_species_df.isti)]
    label_list += [f'{species}_sim']

plt.boxplot(toplot_list, labels = label_list)
plt.xticks(rotation = 90)

#COMPARAISON AVEC ANNOTATION JB
#####

# On compare avec un enregistrement qu'il a annoté : S4A03536_20190624_103000.wav
jurarec_path = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\annotation_JB\DATA\S4A03536_20190624_103000.wav")
jurarec_sig = one_normalize(bracket_ramp(waveread(jurarec_path), fade_duration=0.10, sr=samprate))
show_spectrogram(jurarec_sig, vmin = -50, vmax = -20)
save_path = fig_dir + '/jura_recording.wav'
soundfile.write(save_path, jurarec_sig, samprate)

jurarec_sample = short(jurarec_sig, start = 25, end = 35)    
save_path = fig_dir + '/jura_recording_sample.wav'  
soundfile.write(save_path, jurarec_sample, samprate)  

# species : fricoe, erirub, sylatr, phycol
# distances : turphi (30) ; fricoe (10 - 30 - 50) ; sylatr (20)

abundance_df = empty_abundance_dataframe(normch1_df)
species_list = ['fricoe','turphi','sylatr','phycol']
abundance_df.loc[species_list] = 1


# # behavior_df = bird_behavior(normch1_df, abundance_df, duration)
# # timeline_fig = timeline_chart(behavior_df, y_size = 0.5)

# comparison_assemblage, comparison_df = assemblage(normch1_df, normch2_df, abundance_df, 
#                                                    d_min = 10, d_max = 60, impulse_response = None, 
#                                                    channel2 = 'ambient_sound', random_behavior = False,
#                                                    duration = 60, samprate = 44100)
# save_path = fig_dir + '/comparison_assemblage.wav'
# soundfile.write(save_path, comparison_assemblage, samprate)
# csv_path = fig_dir + '/comparison_assemblage.csv'
# comparison_df.to_csv(csv_path, sep=';')

# BAD exemple : no behavior, quiet background, same distance for all birds, 
fricoe_path = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\Samples\birds_audacity_extract\Fringilla coelebs\MNHN-SO-2016-5474_full.wav")
turphi_path = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\Samples\birds_audacity_extract\Turdus philomelos\MNHN-SO-2016-13891_extr.wav")
sylatr_path = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\Samples\birds_audacity_extract\Sylvia atricapilla\MNHN-SO-2016-12453_full.wav")
phycol_path = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\Samples\birds_audacity_extract\Phylloscopus collybita\MNHN-SO-2016-9911_full.wav")

fricoe_sig = rms_normalize(waveread(fricoe_path))
turphi_sig = rms_normalize(waveread(turphi_path))
sylatr_sig = rms_normalize(waveread(sylatr_path))
phycol_sig = rms_normalize(waveread(phycol_path))

bird_names = [fricoe_sig,turphi_sig,sylatr_sig,phycol_sig]
chorus = flatsound(val = 0, d=60, sr=samprate)
for bird in bird_names:
    chorus = addin(chorus, bird, time_code=0, ramp_duration = 0, sr= samprate)
basic_sig = one_normalize(bracket_ramp(chorus, fade_duration=0.10, sr=samprate))
show_spectrogram(basic_sig, vmin = -50, vmax = -20)
save_path = fig_dir + '/basic_reconstruction.wav'
soundfile.write(save_path, basic_sig, samprate)

basic_sample = short(basic_sig, start = 25, end = 35)    
save_path = fig_dir + '/basic_reconstruction_sample.wav'  
soundfile.write(save_path, basic_sample, samprate)          
                    



# PSYCHOAC

# background power spectrum
background_list = ['ambient_sound', 'rain_pw02', 'aircraft_pw02']
title_list = ['(a) ambient sound','(b) rain','(c) aircraft noise']
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize=(15,4), sharey='row')
# plt.rc('xtick', labelsize=5) 
# plt.rc('ytick', labelsize=5) 

for i, background in enumerate(background_list):
    background_df = normch2_df.loc[normch2_df.categories == background]
    add_power_spectro = np.array([0 for i in range(512)])
    for sound_file in background_df.index:
        signal = waveread(background_df.fullfilename[sound_file])
        Sxx_power,tn, fn, ext = maad.sound.spectrogram (signal, samprate, mode="amplitude")
        power_spectro = maad.sound.avg_amplitude_spectro(Sxx_power)
        add_power_spectro = np.add(add_power_spectro, power_spectro)
    avg_power_spectro = add_power_spectro/len(background_df)
    plt.sca(axs[i])
    plt.xscale('log')
    plt.plot(fn, avg_power_spectro)
    axs[i].set_title(title_list[i], size=20)
    axs[i].set_xlabel('frequency [Hz]', size = 15)
axs[0].set_ylabel('amplitude [Pa]', size=15)

