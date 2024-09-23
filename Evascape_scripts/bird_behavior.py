# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 11:16:02 2023

@author: Elie grinfeder
"""

import numpy as np
import pandas as pd
import random
import maad as maad
import matplotlib.pyplot as plt

####################
# BEHAVIOR ALGORITHM
####################

def bird_selection(normsong_df, abundance_df):
    bird_list = []
    for species in abundance_df.index : 
        abundance = abundance_df.abundance[species]
        if abundance >= 1 :
            species_bird_list = list(normsong_df.source_filename[normsong_df.categories == species].unique())
            chosen_list = random.sample(species_bird_list, k = abundance)
            bird_list += chosen_list
    return bird_list

def repartition_inclusion(repartition_array, rep_index, rep_time, bird_name, song_start, song_end):
    if song_start < min(rep_time):
        song_start = min(rep_time)
    if song_end < max(rep_time):
        song_end = max(rep_time)
    
    bird_i = rep_index.index(bird_name)
    min_i = rep_time.index(song_start)
    max_i =rep_time.index(song_end)
    repartition_array[bird_i,min_i:max_i] = True
    return repartition_array
    
def repartition_overlap(repartition_array, rep_index, rep_time, bird_name, song_start, song_end):
    if song_start < min(rep_time):
        song_start = min(rep_time)
    if song_end < max(rep_time):
        song_end = max(rep_time)
    
    bird_i = rep_index.index(bird_name)
    min_i = rep_time.index(song_start)
    max_i =rep_time.index(song_end)
    compare_array = np.delete(repartition_array, obj = bird_i, axis=0)
    compare_index = np.delete(rep_index, obj = bird_i, axis=0)
    overlap_birds = []
    for comp_bird in range(len(compare_array)):
        if True in compare_array[comp_bird, min_i:max_i]:
            comp_name = compare_index[comp_bird]
            overlap_birds += [comp_name]
    return overlap_birds 

def empty_abundance_dataframe(normsong_df):
    species_list = normsong_df.categories.unique()
    abundance_df = pd.DataFrame(0, columns= ['abundance'], index = species_list)
    return abundance_df

def bird_distances(bird_list, d_min, d_max, d_step = 10): 
    distance_range = np.arange(d_min, d_max, d_step)
    distance_list = np.random.choice(distance_range, size = len(bird_list))
    return distance_list

def random_ist(normsong_df, song_name, norm_distrib = True):
    species = normsong_df.categories.loc[song_name]
    species_ist_df = normsong_df.copy()
    species_ist_df = species_ist_df.ist.loc[species_ist_df.categories == species].dropna()
    
    if norm_distrib == True:
        rand_ist = round(np.random.normal(loc = normsong_df.ist_mean.loc[song_name], 
                                         scale = normsong_df.ist_std.loc[song_name]),
                           ndigits = 2)
        min_ist = min(species_ist_df)
        max_ist = max(species_ist_df)
        while rand_ist < min_ist or rand_ist > max_ist :
            rand_ist = round(np.random.normal(loc = normsong_df.ist_mean.loc[song_name], 
                                             scale = normsong_df.ist_std.loc[song_name]),
                               ndigits = 2)
    else :
        rand_ist = random.choice(species_ist_df)

    return rand_ist

def trial_and_error(song_name, song_len, bird, lastsong_df, normsong_df):
    # produce a random intersing time duration from species normal distribution
    rand_ist = random_ist(normsong_df, song_name, norm_distrib = True)
    
    song_start = round(lastsong_df.max_t.loc[bird] + rand_ist, ndigits = 2)
    song_end = round(song_start + song_len, ndigits = 2)
    
    return song_start, song_end
    
def desync(song_name, song_len, max_ist, overlap_birds, focal_bird, lastsong_df, normsong_df, samprate):
    if len(overlap_birds) > 1:
        max_dB = 0
        focal_distance = lastsong_df.distance.loc[focal_bird]
        for bird in overlap_birds:
            
            overlap_distance = lastsong_df.distance.loc[bird]
            relative_distance = abs(focal_distance - overlap_distance)
            
            bird_amp = normsong_df.peak_amp.loc[song_name]
            bird_dB = maad.spl.power2dBSPL(bird_amp, gain = 42)
            att_dB, df_att = maad.spl.attenuation_dB(f = 8000, r = relative_distance, r0 = 1)
            propag_dB = bird_dB - att_dB[0]
            
            #propag_amp = maad.spl.apply_attenuation([amp_bird,amp_bird], samprate, r = relative_distance)
            
            if propag_dB > max_dB:
                max_dB = propag_dB
                overlap_bird = bird
    else :
        overlap_bird = overlap_birds[0]
    
    # calculate next singing time based on overlap_bird predicted next song timing
    overlap_tmax = lastsong_df.max_t[overlap_bird]
    overlap_mean_ist = normsong_df.ist_mean.loc[normsong_df.source_filename == overlap_bird].unique()[0]
    nextoverlap_tmin = overlap_tmax + overlap_mean_ist
    midpoint_t = overlap_tmax + (nextoverlap_tmin - overlap_tmax)/2

    song_start = round(midpoint_t - (song_len/2), ndigits = 2)
    song_end = round(midpoint_t + (song_len/2), ndigits = 2)
    last_start = lastsong_df.max_t[focal_bird]
    ist = song_start - last_start
    
    if ist > max_ist or song_start < last_start:
        rand_ist = random_ist(normsong_df, song_name, norm_distrib = False)
        song_start = round(lastsong_df.max_t.loc[focal_bird] + rand_ist, ndigits = 2)
        song_end = round(song_start + song_len, ndigits = 2)   
    
    return song_start, song_end

def bird_behavior(normsong_df, abundance_df, d_min, d_max,
                  recording_duration = 60,  #behavior algorithm taken from Suzuki et al. 2012
                  all_random_ist = False, # does not take Suzuki's algorithm and makes all intersing intervals random instead
                  duplicate = False,
                  samprate = 44100): # allows for a single song recording to be present multiple times if there is not enough songs to fill recording_duration
    availablesong_df = normsong_df.copy(deep=True)
    bird_list = bird_selection(normsong_df, abundance_df)
    song_list = []
    lastsong_df = pd.DataFrame(0, index = bird_list, 
                               columns = ['min_t','max_t','nextmin','distance'])
    
    #bird distance:
    if d_min == d_max:
        lastsong_df.loc[:,'distance'] = d_min
    else:
        lastsong_df.distance = bird_distances(bird_list, d_min = d_min, d_max = d_max, d_step = 1)
    
    if recording_duration < 60 : #if recording is shorter than 60s, avoid border effects. For example, if recording_duration = 10s, computes behavior for 60s but only give out behavior between 25 and 35 s 
        behavior_duration = 60
    else :
        behavior_duration = recording_duration
    
    #repartition array
    rep_index = list(bird_list)
    rep_time = np.round(np.arange(0.00, behavior_duration, 0.01),2).tolist()
    max_time = max(rep_time)
    repartition_array = np.full (shape = (len(rep_index),len(rep_time)), 
                                 fill_value = False) 
    
    # min/max ist estimation
    ist_estimation_df = normsong_df.copy(deep=True)
    min_ist = (normsong_df.ist_mean - normsong_df.ist_std).astype(float).round(2)
    # print(min_ist)
    # if min_ist < 0: #check pour chaque ist 
    #     min_ist = 0.5
    ist_estimation_df['min_ist'] = min_ist
    max_ist = (normsong_df.ist_mean + normsong_df.ist_std).astype(float).round(2)
    ist_estimation_df['max_ist'] = max_ist
    
    #initiate first song for all species
    for bird in bird_list :         
        #choose new song    
        birdsongs_list = availablesong_df[availablesong_df.source_filename == bird].index
        song_name = random.choice(birdsongs_list)
        #setting starting time
        species = normsong_df.categories.loc[normsong_df.source_filename == bird].unique()[0]
        max_ist = ist_estimation_df.max_ist[song_name]
        song_len = availablesong_df.duration.loc[song_name]
        song_start = lastsong_df.min_t.loc[bird] = round(random.uniform(0, max_ist), ndigits = 2)
        song_end = lastsong_df.max_t.loc[bird] = round(song_start + song_len, ndigits = 2)
        #predicting next song start min
        min_ist = ist_estimation_df.min_ist[song_name]
        nextmin = lastsong_df.nextmin.loc[bird] = round(song_end + min_ist, 2)
        # saving infos + deleting song from availablesong_df
        distance = lastsong_df.distance.loc[bird]
        repartition_array = repartition_inclusion(repartition_array, rep_index, rep_time, bird, song_start, song_end)
        song_list += [[song_name, normsong_df.song_fullfilename[song_name], bird, species, song_start, song_end, song_len, distance]]
        availablesong_df = availablesong_df.drop(song_name)
    
    # behavior loop
    while min(lastsong_df.nextmin) < max_time :
        # identify next bird
        bird = lastsong_df.index[lastsong_df.nextmin == min(lastsong_df.nextmin)]
        if len(bird) > 1:
            bird = random.choice(bird)
        else:
            bird = bird[0]
        species = normsong_df.categories.loc[normsong_df.source_filename == bird].unique()[0]
        # choose new song
        birdsongs_list = availablesong_df[availablesong_df.source_filename == bird].index
        
        if len(birdsongs_list) == 0: # if there is not enough songs in the recording to fill duration,
            if duplicate == True : # duplicate the songs so it can fill behavior_duration
                availablesong_df = availablesong_df.append(normsong_df.loc[normsong_df.source_filename == bird]) #put the delete songs back in the catalog
                birdsongs_list = availablesong_df[availablesong_df.source_filename == bird].index #ressets the songlist
            else: # no duplicate songs are produced, creates and edge effect
                song_start = max_time # the bird cannot sing anymore
        else:
            song_name = random.choice(birdsongs_list)
            song_len = availablesong_df.duration.loc[song_name]
            
            # overlap decision
            if all_random_ist == True or sum(abundance_df.abundance) == 1:
                rand_ist = random_ist(normsong_df, song_name, norm_distrib = False)
                song_start = round(lastsong_df.max_t.loc[bird] + rand_ist, ndigits = 2)
                song_end = round(song_start + song_len, ndigits = 2)
                
            else :
                overlap_birds = repartition_overlap(repartition_array, rep_index, rep_time, #check if the song is overlaping with any other song
                                                 bird_name = bird, song_start = lastsong_df.min_t.loc[bird], 
                                                 song_end = lastsong_df.max_t.loc[bird]) 
                if len(overlap_birds) > 0 : #if there is overlap
                    max_ist = ist_estimation_df.max_ist[song_name]
                    song_start, song_end = desync(song_name = song_name, 
                                                   song_len = song_len, 
                                                   max_ist = max_ist,
                                                   overlap_birds = overlap_birds, 
                                                   focal_bird = bird, 
                                                   lastsong_df = lastsong_df, 
                                                   normsong_df = normsong_df, 
                                                   samprate = samprate
                                                   )
                else : 
                    rand_ist = random_ist(normsong_df, song_name, norm_distrib = False)
                    song_start = round(lastsong_df.max_t.loc[bird] + rand_ist, ndigits = 2)
                    song_end = round(song_start + song_len, ndigits = 2)
                    
            
        #check if song is out of time bounds + saving infos + predicts next earliest possible song from this bird 
        if song_start >= max_time: # if the generated song starts after the end of the recording
            lastsong_df.nextmin[bird] = max_time # the song is not recorded + the bird cannot sing anymore
        elif song_end > max_time : # if the generated song ends after the end of the recording
            song_end = max_time #cuts the song end so it stops at the end of the recording
            lastsong_df.nextmin[bird] = nextmin = max_time
            # saving infos + deleting song from availablesong_df
            distance = lastsong_df.distance.loc[bird]
            repartition_array = repartition_inclusion(repartition_array, rep_index, rep_time, bird, song_start, song_end)
            song_list += [[song_name, normsong_df.song_fullfilename[song_name],bird, species, song_start, song_end, song_len, distance]]
            lastsong_df.loc[bird,['min_t','max_t','nextmin']] = [song_start, song_end, nextmin]
            availablesong_df = availablesong_df.drop(song_name)
        else :
            min_ist = ist_estimation_df.min_ist[song_name]
            nextmin = round(song_end + min_ist, 2)
            # saving infos + deleting song from availablesong_df
            distance = lastsong_df.distance.loc[bird]
            repartition_array = repartition_inclusion(repartition_array, rep_index, rep_time, bird, song_start, song_end)
            song_list += [[song_name, normsong_df.song_fullfilename[song_name],bird, species, song_start, song_end, song_len, distance]]
            lastsong_df.loc[bird,['min_t','max_t','nextmin']] = [song_start, song_end, nextmin]
            availablesong_df = availablesong_df.drop(song_name)
    allsongs_df = pd.DataFrame(song_list, columns = ['song_filename', 'song_fullfilename', 'bird_filename', 'categories', 'min_t', 'max_t', 'song_len','distance'])
    
    if recording_duration < 60 :
        t_min = 30 - (recording_duration/2)
        t_max = 30 + (recording_duration/2)
        allsongs_df = allsongs_df[(allsongs_df.max_t > t_min)][(allsongs_df.min_t < t_max)]
        allsongs_df.min_t = allsongs_df.min_t - t_min
        allsongs_df.max_t = allsongs_df.max_t - t_min
    
    return allsongs_df


def silence_ratio(file_df, duration):
    
    # sort df by min_t and retrieve geophony
    song_df = file_df.copy(deep=True).sort_values(by = ['min_t'])
    if not isinstance(file_df.bird_filename.iloc[-1], str):
        song_df = song_df.iloc[:-1]
    
    # collect silences between songs
    silence_list = []
    t_mark = t_min = t_max = 0
    for song in song_df.index: 
        if t_min < 0 :
            t_min = 0
        else:
            t_min = song_df.min_t.loc[song]
        
        if t_max > duration:
            t_max = duration
        else:
            t_max = song_df.max_t.loc[song]
        
        if t_min > t_mark:
            silence_list += [t_min - t_mark]
        if t_max > t_mark:
            t_mark = t_max
    if t_mark < duration :
        silence_list += [duration - t_mark]
    
    # calculate silence ratio
    silence_sum = sum(silence_list)
    silence_ratio = silence_sum / duration
    
    return silence_ratio

def check_songlen(song_df, duration, songlen_min = 0.5):
    
    check_df = song_df.copy()
    check_df.min_t.loc[check_df.min_t < 0] = 0
    check_df.max_t.loc[check_df.max_t > duration] = duration
    songlen_list = list(check_df.max_t - check_df.min_t)
    is_enough = all(songlen > songlen_min for songlen in songlen_list)
    
    return is_enough

def overlap_amount(file_df): #only for two individuals

    # sort df by min_t and retrieve geophony
    song_df = file_df.copy(deep=True).sort_values(by = ['min_t'])
    if not isinstance(file_df.bird_filename.iloc[-1], str):
        song_df = song_df.iloc[:-1]
    
    bird_list = list(song_df.bird_filename.unique())
    focus_df = song_df.loc[song_df.bird_filename == bird_list[0]]
    compare_df = song_df.loc[song_df.bird_filename == bird_list[1]]
    
    overlap_list = []
    for focus_song in focus_df.index: 
        focus_mint = song_df.min_t.loc[focus_song]
        focus_maxt =  song_df.max_t.loc[focus_song]
        for compare_song in compare_df.index :
            
            compare_mint = song_df.min_t.loc[compare_song]
            compare_maxt =  song_df.max_t.loc[compare_song]
            
            if compare_mint <= focus_mint <= compare_maxt:
                mark_mint = focus_mint
            elif focus_mint <= compare_mint <= focus_maxt:
                mark_mint = compare_mint
            else :
                mark_mint = 0
                
            if compare_mint <= focus_maxt <= compare_maxt:
                mark_maxt = focus_maxt
            elif focus_mint <= compare_maxt <= focus_maxt:
                mark_maxt = compare_maxt
            else:
                mark_maxt = 0
            
            if (mark_maxt - mark_mint) <0 :
                print(focus_mint, focus_maxt, compare_mint, compare_maxt)
            overlap_list += [mark_maxt - mark_mint]
            
    # calculate overlap ratio
    overlap_sum = sum(overlap_list)
    
    return overlap_sum

def overlap_ratio(file_df, duration = 60): #only for two individuals

    overlap_t = overlap_amount(file_df)
    overlap_ratio = overlap_t / duration
    
    return overlap_ratio
                
# path = Path(r"D:\Psychoac_20240301_desync\Cond_F\evascape_F_N02.csv")       
# file_df = pd.read_csv(path, sep=';', index_col = 0)    

# abundance_test = empty_abundance_dataframe(normch1_df)
# abundance_test.loc['erirub','phycol','sylatr'] = 2
# test_df = bird_behavior(normsong_df = normch1_df, abundance_df = abundance_test, recording_duration = 60)


#plots results from bird_behavior() with a timeline-type chartimport colorsys

# import seaborn as sns
# from PIL import Image, ImageDraw

okabe_ito = {'erirub' : [213/255,94/255,0/255,1], #vermillion
             'fricoe' : [230/255,159/255,0/255,1], #orange
             'perate' : [0/255,114/255,178/255,1], #royal blue
             'phycol' : [0/255,158/255,115/255,1], #bluish green
             'regreg' : [240/255,228/255,66/255,1], #yellow
             'sylatr' : [204/255,121/255,167/255,1], #reddish purple
             'turmer' : [0/255,0/255,0/255,1], #black
             'turphi' : [86/255,180/255,233/255,1]} #sky blue
color_df = pd.DataFrame(okabe_ito, columns = ['R','G','B','A'])

def timeline_chart(file_df, color_palet = okabe_ito, y_size = 0.5, duration = 60): #change so it works with no color palet
    
    # retrieve geophony 
    behavior_df = file_df.copy(deep=True)
    if not isinstance(file_df.bird_filename.iloc[-1], str):
        behavior_df = behavior_df.iloc[:-1]
    
    species_list = list(behavior_df.categories.unique())
    bird_list = list(behavior_df.bird_filename.unique())
    
    # if color_palet == None:
    #     color_rgba = []
    #     for species in species_list:
    #         color_rgb = list(np.random.uniform(low = 0, high = 1, 
    #                                         size = 3))
    #         color_rgba += [color_rgb + [1]]
    #     color_df = pd.DataFrame(color_rgba, index = species_list, columns = ['R','G','B','A'])
    # else:
    #     color_df = color_palet.loc[species_list]
    
    fig, ax = plt.subplots()
    for song in behavior_df.index:
        species = behavior_df.categories[song]
        x_ranges = [( behavior_df.min_t[song], behavior_df.song_len[song])]
        y_start = bird_list.index(behavior_df.bird_filename[song]) + 1
        plt.broken_barh(x_ranges, (y_start-(y_size/2), y_size),facecolors = color_palet[species])
    ax.set_xlabel('Time[s]')
    ax.set_ylabel('bird ID')
    plt.title('bird behavior timeline', fontsize=20)
    
    y_labels = []
    species_count = [1 for species in species_list] 
    counter_df = pd.DataFrame(species_count, index = species_list)
    for bird in bird_list:
        species = behavior_df.categories[behavior_df.bird_filename == bird].unique()[0]
        display_name = species + '_0' + str(counter_df.loc[species][0])
        y_labels += [display_name]
        counter_df.loc[species] += 1
        
    plt.yticks(ticks=list(range(1,len(bird_list)+1)),
           labels=y_labels, fontsize=10)
    plt.axvline(x=0, color = 'grey', linestyle = 'dashed')
    plt.axvline(x=duration, color = 'grey', linestyle = 'dashed')

    return fig

# temp_dir = "C:/Users/ecoac-field/OneDrive/Documents/Articles-Recherches/Reconstructor/Samples/temp"
# behavior_path = temp_dir / 'reconstructed_files/evascape_quiet_rich7_ab1_n2.csv'
# behavior_df = pd.read_csv(behavior_path, sep=';', index_col = 0)
# timeline_chart(behavior_df)
