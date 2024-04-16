# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:22:33 2023

@author: Elie grinfeder
"""

############
# ASSEMBLAGE
############

import numpy as np
import pandas as pd
from pathlib import Path
from maad import spl
import soundfile
import scipy
import random
import time
import os
from toolbox import addin, flatsound, waveread, bracket_ramp
from bird_behavior import bird_behavior, check_songlen, overlap_amount
from pathlib import Path
from itertools import product, combinations

def database_presets(channel1_list, channel2_list, richness_list, abundance_lvls, 
                     sample_size, all_combinations = True):
    database_col = ['filename', 'fullfilename', 'sample_id', 'channel2', 'richness', 'abundance'] + channel1_list
    sample_list = list(range(1,sample_size + 1))
    empty_array = np.zeros([1,len(channel1_list)])[0]
    empty_df = pd.DataFrame(empty_array, index = channel1_list, 
                            columns = ['abundance']).astype('int')
    
    if all_combinations == True:
        
        variables_prod = pd.DataFrame(product(abundance_lvls, channel2_list, sample_list),columns = ['abundance', 'channel2', 'sample_id'])
        comb_list = []
        for richness in richness_list:
            comb_list += list(combinations(channel1_list, richness))
        combination_nb = len(comb_list)
            
        database_list = []
        sample_max = sample_size * combination_nb
        sample_id = 1
        for i in variables_prod.index: 
            for j in range(combination_nb):
                species_list = comb_list[j]
                richness = len(species_list)
                abundance = variables_prod.abundance[i]
                distribution_df = empty_df.copy()
                distribution_df.loc[species_list,] = abundance
                database_list += [[richness] + [abundance] + [variables_prod.channel2[i]] 
                                  + [sample_id] + list(distribution_df.abundance)]
                if sample_id == sample_max:
                    sample_id = 1 
                else :
                    sample_id += 1
        list_col = ['richness', 'abundance', 'channel2','sample_id'] + channel1_list
        database_df = pd.DataFrame(database_list, columns = list_col).reindex(columns=database_col)
            
    else :
        variables_prod = list(product(richness_list, abundance_lvls, channel2_list, sample_list))
        database_df = pd.DataFrame(variables_prod,
                                   columns = ['richness', 'abundance', 'channel2','sample_id']).reindex(columns=database_col)
        database_df[channel1_list] = np.zeros([len(database_df),len(channel1_list)]).astype('int')
 
        for channel2 in channel2_list:
            for abundance in abundance_lvls:
                for richness in richness_list:               
                    index_list = database_df[(database_df.abundance == abundance) & (database_df.richness == richness) & (database_df.channel2 == channel2)].index
                    remain_list = [list(species) for species in combinations(channel1_list, richness)]
                    for i in range(sample_size):
                        if remain_list == []:
                            remain_list = [list(species) for species in combinations(channel1_list, richness)]
                        sample = random.choice(remain_list)  
                        index = index_list[i]
                        database_df.loc[index, sample] = abundance
                        remain_list.remove(sample) 
    
    return database_df


def singing_session(song_df, bird_filename, duration = 60, samprate = 44100):
    birdsong_df = song_df.loc[song_df.bird_filename == bird_filename]
    session_vector = flatsound(val = 0, d = duration, sr = samprate)
    for song in birdsong_df.index :
        song_vector = waveread(Path(birdsong_df.song_fullfilename[song]))
        session_vector = addin(base_sound = session_vector, 
                               added_sound = song_vector, 
                               time_code = birdsong_df.min_t[song], 
                               ramp_duration = 0.10, sr = samprate) #check fade duration = config 
    return session_vector


def assemblage(normch1_df, normch2_df, abundance_df, 
               d_min, d_max, impulse_response = None, 
               channel2 = 'ambient_sound', random_behavior = False,
               duration = 60, samprate = 44100):
    
    #channel 1 
    tot_abundance = sum(abundance_df.abundance)
    bird_nb = 0
    songlen_good = False
    while (bird_nb < tot_abundance) or not(songlen_good):
        song_df = bird_behavior(normsong_df = normch1_df, 
                                abundance_df = abundance_df, 
                                d_min = d_min, 
                                d_max = d_max,
                                recording_duration = duration,  #behavior algorithm taken from Suzuki et al. 2012
                                all_random_ist = random_behavior, # does not take Suzuki's algorithm and makes all intersing intervals random instead
                                duplicate = False,
                                samprate = samprate)
        bird_nb = len(song_df.bird_filename.unique())
        songlen_good = check_songlen(song_df, duration, songlen_min = 0.5)
        if (bird_nb < tot_abundance) or not(songlen_good): 
            print('recast')
        
    
    channel1_vector = flatsound(val = 0, d = duration, sr = samprate)
    
    for song in song_df.index:
        song_vector = waveread(Path(song_df.song_fullfilename[song]))
        bird_distance = int(song_df.distance[song])
        if bird_distance != 0:
            song_vector = spl.apply_attenuation(song_vector, samprate, r = bird_distance)
        channel1_vector = addin(base_sound = channel1_vector, 
                               added_sound = song_vector, 
                               time_code = song_df.min_t[song], 
                               ramp_duration = 0.10, sr = samprate)
    
    if isinstance(impulse_response, np.ndarray):
        channel1_vector = reverb(signal = channel1_vector, 
                                 impulse_response = impulse_response)
    
    #channel 2
    if channel2 == 'quiet':
        channel2_vector = flatsound(val = 0, d= duration, sr = samprate)
        soundtype_df = normch2_df.loc[normch2_df.categories == 'ambient_sound']
        channel2_filename = random.choice(soundtype_df.index)
        channel2_fullfilename = normch2_df.norm_fullfilename[channel2_filename]
        channel2_vector = waveread(Path(channel2_fullfilename))
        dB_var = 50
        quiet_factor = 1 / 10**(dB_var/20)
        channel2_vector = channel2_vector * quiet_factor
    else :
        soundtype_df = normch2_df.loc[normch2_df.categories == channel2]
        channel2_filename = random.choice(soundtype_df.index)
        channel2_fullfilename = normch2_df.norm_fullfilename[channel2_filename]
        channel2_vector = waveread(Path(channel2_fullfilename))
        
    stop = int(np.round(duration * samprate, 2)) 
    channel2_vector = channel2_vector[:stop]
    
    channel2_dic = {'song_filename' : channel2_filename,
                    'song_fullfilename' : channel2_fullfilename, 
                    'species' : channel2}
    song_df.loc[len(song_df)] = channel2_dic
    
    #final assemblage
    final_vector = addin(base_sound = channel2_vector, 
                           added_sound = channel1_vector, 
                           time_code = 0, 
                           ramp_duration = 0, sr = samprate)
    if max(final_vector) > 1: # ok ?
        final_vector = final_vector / max(final_vector)
 
    tosave_vector = bracket_ramp(final_vector - np.mean(final_vector),
                                 fade_duration = 0.10) #delete DC offset + add a ramp
    
    # tosave_vector = bracket_ramp(final_vector,
    #                              fade_duration = 0.10)
    
    return tosave_vector, song_df 
    

def database(normch1_df, normch2_df, save_dir, d_min, d_max,
             richness_list = [1, 2, 3, 4 ,5 , 6, 7, 8], 
             abundance_lvls = [1, 2, 3, 4 ,5],  
             channel1_list = ["erirub", "fricoe", "perate","phycol", "regreg", "sylatr", "turmer", "turphi"],
             channel2_list = ['quiet','ambient_sound','rain_pw02','wind_pw02', 'tettigonia_veridissima'],
             impulse_response = None, random_behavior = False, anonymous_ID = False, database_label = 'evascape',
             all_combinations = False, sample_size = 1, duration = 60, samprate = 44100):
    
    global_start = time.time()
                        
    #database generation
    database_df = database_presets(channel1_list = channel1_list, channel2_list = channel2_list, 
                                   richness_list = richness_list, abundance_lvls = abundance_lvls,
                                   sample_size = sample_size, all_combinations = all_combinations)    
    
    availablesongs_df = normch1_df.copy()
    
    reset_df = pd.DataFrame([False for species in channel1_list], index = channel1_list, columns = ['is_reset'])
    for file in database_df.index:
        start = time.time()
        abundance_list = list(database_df.loc[file,channel1_list])
        abundance_df = pd.DataFrame(abundance_list, index = channel1_list, columns = ['abundance'])
        channel2 = database_df.channel2.loc[file]
        
        # check if there are enough available birds
        species_tocheck = abundance_df.index[abundance_df.abundance > 0]
        for species in species_tocheck:
            abundance = abundance_df.abundance[species]
            available_birds = availablesongs_df[availablesongs_df.categories == species].source_filename.unique()
            available_nb = len(available_birds)
            diff = available_nb - abundance
            #print(f'species : {species} \t available_nb = {available_nb}\t abundance = {abundance}\t diff = {diff} ')
            if available_nb == 0 or reset_df.is_reset[species] == True :
                added_df = normch1_df[normch1_df.categories == species]
                availablesongs_df = pd.concat([availablesongs_df,added_df])
                reset_df.is_reset[species] == False
            elif  diff < 0: #add birds from the species to availablesongs_df if needed
                all_birds = normch1_df[normch1_df.categories == species].source_filename.unique()
                notavailable_birds = [i for i in all_birds if i not in available_birds]
                #print(f'available birds : {available_birds}/t available_nb = {available_nb} /t diff = {diff} /t notavailable_birds : {notavailable_birds}')
                added_birds = random.sample(notavailable_birds, k = abs(diff))
                added_df = normch1_df[normch1_df.source_filename.isin(added_birds)]
                availablesongs_df = pd.concat([availablesongs_df,added_df])
                reset_df.is_reset[species] == True
        
        # soundscape reconstruction
        reconstructed_vector, file_df = assemblage(normch1_df = availablesongs_df, 
                                                   normch2_df = normch2_df, 
                                                   abundance_df = abundance_df,  
                                                   d_min = d_min, 
                                                   d_max = d_max,
                                                   impulse_response = impulse_response,
                                                   channel2 = channel2,
                                                   random_behavior = random_behavior,
                                                   duration = duration, 
                                                   samprate = samprate)

        #remove used birds
        used_birds = file_df.bird_filename.unique()
        all_birds = normch1_df.source_filename.unique()
        available_birds = [i for i in all_birds if i not in used_birds]
        availablesongs_df = availablesongs_df[availablesongs_df.source_filename.isin(available_birds)]
        
        # saving soundscape into wavefile
        richness, abundance, sample_id = database_df.richness.loc[file], database_df.abundance.loc[file], database_df.sample_id.loc[file]
        
        if anonymous_ID == True :
            num_size = len(str(len(database_df)))
            row_num = database_df.index.get_loc(file) + 1
            ID = f"%0{num_size}d" % (row_num,)
            save_name = f'{database_label}_N{ID}'
        else :
            save_name = f'{database_label}_{channel2}_rich{richness}_ab{abundance}_n{sample_id}'
        save_path = Path(f'{save_dir}/{save_name}.wav')
        save_path.parent.mkdir(exist_ok=True, parents=True)
        soundfile.write(save_path, reconstructed_vector, samprate)
        database_df.filename.loc[file] = save_name
        database_df.fullfilename.loc[file] = save_path
        
        #saving soundscape metadata as csv file
        csv_path = f'{save_dir}/{save_name}.csv'
        file_df.to_csv(csv_path, sep=';')
        
        stop = time.time()
        n = stop - start
        time_format = time.strftime("%H:%M:%S", time.gmtime(n))
        print(f'{save_name}\t{file + 1}/{len(database_df)}\t time = {time_format}')
                    
    database_df = database_df.set_index('filename')
    
    global_stop = time.time()
    
    n = global_stop - global_start
    time_format = time.strftime("%H:%M:%S", time.gmtime(n))
    print('global time = ',time_format)

    return database_df


#Overlap

def database_overlap(normch1_df, normch2_df, d_min, d_max,
             channel1_list = ["erirub", "fricoe", "perate","phycol", "regreg", "sylatr", "turmer", "turphi"],
             impulse_response = None, random_behavior = False, 
             all_combinations = False, sample_size = 8, duration = 60, samprate = 44100):

                        
    #database generation
    richness_df = database_presets(channel1_list = channel1_list, channel2_list = ['ambient_sound'], 
                                   richness_list = [2], abundance_lvls = [1],
                                   sample_size = sample_size, all_combinations = all_combinations)    
    richness_df['focus'] = 'richness'
    abundance_df = database_presets(channel1_list = channel1_list, channel2_list = ['ambient_sound'], 
                                   richness_list = [1], abundance_lvls = [2],
                                   sample_size = sample_size, all_combinations = all_combinations)  
    abundance_df['focus'] = 'abundance'
    database_df = pd.concat([richness_df, abundance_df])
    database_df.index = list(range(len(database_df)))

    availablesongs_df = normch1_df.copy()
    
    richness_overlap_list = []
    abundance_overlap_list = []
    
    reset_df = pd.DataFrame([False for species in channel1_list], index = channel1_list, columns = ['is_reset'])
    for file in database_df.index:

        abundance_list = list(database_df.loc[file,channel1_list])
        abundance_df = pd.DataFrame(abundance_list, index = channel1_list, columns = ['abundance'])
        
        # check if there are enough available birds
        species_tocheck = abundance_df.index[abundance_df.abundance > 0]
        for species in species_tocheck:
            abundance = abundance_df.abundance[species]
            available_birds = availablesongs_df[availablesongs_df.categories == species].source_filename.unique()
            available_nb = len(available_birds)
            diff = available_nb - abundance
            #print(f'species : {species} \t available_nb = {available_nb}\t abundance = {abundance}\t diff = {diff} ')
            if available_nb == 0 or reset_df.is_reset[species] == True :
                added_df = normch1_df[normch1_df.categories == species]
                availablesongs_df = pd.concat([availablesongs_df,added_df])
                reset_df.is_reset[species] == False
            elif  diff < 0: #add birds from the species to availablesongs_df if needed
                all_birds = normch1_df[normch1_df.categories == species].source_filename.unique()
                notavailable_birds = [i for i in all_birds if i not in available_birds]
                #print(f'available birds : {available_birds}/t available_nb = {available_nb} /t diff = {diff} /t notavailable_birds : {notavailable_birds}')
                added_birds = random.sample(notavailable_birds, k = abs(diff))
                added_df = normch1_df[normch1_df.source_filename.isin(added_birds)]
                availablesongs_df = pd.concat([availablesongs_df,added_df])
                reset_df.is_reset[species] == True
        
        # soundscape reconstruction
        file_df = bird_behavior(normsong_df = normch1_df, 
                                abundance_df = abundance_df, 
                                d_min = d_min, 
                                d_max = d_max,
                                recording_duration = duration,  #behavior algorithm taken from Suzuki et al. 2012
                                all_random_ist = random_behavior, # does not take Suzuki's algorithm and makes all intersing intervals random instead
                                duplicate = False,
                                samprate = samprate)
        
        overlap = overlap_amount(file_df)
        focus = database_df.focus.loc[file]
        if focus == 'richness':
            richness_overlap_list += [overlap]
        else:
            abundance_overlap_list += [overlap]

  
        #remove used birds
        used_birds = file_df.bird_filename.unique()
        all_birds = normch1_df.source_filename.unique()
        available_birds = [i for i in all_birds if i not in used_birds]
        availablesongs_df = availablesongs_df[availablesongs_df.source_filename.isin(available_birds)]

    return richness_overlap_list, abundance_overlap_list

#IST

def database_all_isti(database_dir) : #returns all intersing time (ist) & duration + mean & std

    isti_list = []
    file_df_col_nb = 8
    
    for file in os.listdir(database_dir):
        if file.endswith(".csv"):
            full_path = os.path.join(database_dir, file)
            file_df = pd.read_csv(full_path, sep=';', index_col = 0)

            if len(file_df.columns) == file_df_col_nb :
                species_list = list(file_df.categories.unique())
                richness = len(species_list)
                
                for species in species_list :
                    species_df = file_df.loc[file_df.categories == species]
                    bird_list = list(species_df.bird_filename.unique())
                    abundance = len(bird_list)
                    
                    for bird in bird_list:
                        bird_df = species_df.loc[species_df.bird_filename == bird]
                        
                        for song in range(len(bird_df)-1):
                            ist_start = bird_df.max_t.iloc[song]
                            ist_end = bird_df.min_t.iloc[song + 1]
                            isti = ist_end - ist_start
                            isti_list += [[file,richness,abundance,species,bird,isti]]
                print(file)
                            
    column_list = ['file_name','richness','abundance','species','bird','isti']
    isti_df = pd.DataFrame(isti_list, columns = column_list)
    
    return isti_df

def database_isti_average_analysis(all_isti_df):
    
    species_list = list(all_isti_df.species.unique())
    species_list.sort()
    column_list = ['mean','std','skewness','kurtosis']
    average_isti_df = pd.DataFrame(np.nan, index = species_list, columns = column_list)
    
    for species in species_list:
        species_df = all_isti_df.loc[all_isti_df.species == species]
        mean = np.mean(species_df.isti)
        std = np.std(species_df.isti)
        skewness = scipy.stats.skew(species_df.isti)
        kurtosis = scipy.stats.kurtosis(species_df.isti)
        average_isti_df.loc[species] = [mean,std,skewness,kurtosis]
    
    return average_isti_df

def database_isti_detailed_analysis(all_isti_df):
    ab_isti_df = all_isti_df.copy()
    ab_isti_df['total_abundance'] = ab_isti_df.richness * ab_isti_df.abundance
    abundance_list = list(ab_isti_df.total_abundance.unique())
    abundance_list.sort()
    species_list = list(all_isti_df.species.unique())
    species_list.sort()
    
    detailed_isti_df = pd.DataFrame(index = abundance_list)
    detailed_isti_df.index.name = 'total_abundance'
    
    for abundance in abundance_list:
        abundance_df = ab_isti_df[ab_isti_df.total_abundance == abundance]
        
        for species in species_list:
            species_df = abundance_df.loc[abundance_df.species == species]
            detailed_isti_df.loc[abundance,f'{species}_mean'] = np.mean(species_df.isti)
            detailed_isti_df.loc[abundance,f'{species}_std'] = np.std(species_df.isti)
    
    return detailed_isti_df


def database_ist_summary(temporal_analysis_df):
    selection_df = temporal_analysis_df[['categories', 'duration_mean', 'duration_std', 
                                      'ist_mean', 'ist_std']].set_index('categories')
    summary_df = selection_df.drop_duplicates()
    return summary_df

import statsmodels.api as sm 
from statsmodels.formula.api import ols 


def ist_comparison(obs_isti_df, sim_isti_df, solo_simulation = False):

    obs_comp_df = obs_isti_df.copy() 
    sim_comp_df = sim_isti_df.copy()    
     
    # observed ISTI distribution
    obs_comp_df = obs_isti_df[['ist','categories']]
    obs_comp_df.columns  = ['isti','species']
    obs_comp_df['origin'] = 'observed'
    
    
    # simulated ISTI distribution
 
    if solo_simulation :
        richness = np.array(sim_comp_df.richness == 1)
        abundance = np.array(sim_comp_df.abundance == 1)
    else :
        richness = np.array(sim_comp_df.richness > 1)
        abundance = np.array(sim_comp_df.abundance > 1)
 
    sim_comp_df = sim_comp_df.loc[richness & abundance] 
    sim_comp_df = sim_comp_df[['isti','species']]
    sim_comp_df['origin'] = 'simulated'
    
    # two_way ANOVA
    comparison_df = pd.concat([obs_comp_df,sim_comp_df])
    comparison_df = comparison_df.dropna(subset = ['isti'])
    formula = 'isti ~ C(species) + C(origin) + C(species):C(origin)'
    model = ols(formula, data=comparison_df).fit() 
    results = sm.stats.anova_lm(model, type=2) 
    
    #Verify residuals normality
    residuals = model.resid
    stat, p_value = scipy.stats.shapiro(residuals)
    
    return results
        