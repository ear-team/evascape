# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:34:04 2022

@author: Elie grinfeder
"""

### SOUNDSCAPE TOOLBOX ###
### list of of various utilitary functions to build the reconstructor

import numpy as np
import librosa
import soundfile
import maad
from pathlib import Path


#general settings
duration = 3 #sound duration in seconds
samprate = 44100
time = np.linspace(0, duration, samprate*duration)

#Utilitary sound
def flatsound(val = 0, d=duration, sr=samprate):
    return(np.array([val for i in range(int(np.round(sr*d)))]))

def noisesound(d=duration, sr=samprate):
    return(np.random.uniform(low = -1, high = 1, size = d * sr))

#Pure sound
def puresound(amp, freq, phase=0, t=time): 
    return(amp * np.sin(2 * np.pi * freq * t + phase))

#Amplitude Modulated sound
def amsound(carrier, rate, amp, freq, phase=0, t=time): #0< rate <1 ; amp<1
    mod_sound = puresound(amp, freq, phase, t)
    return(carrier * (1 + rate * mod_sound))

#Frequency modulated sound
def fmsound(car_amp, car_freq, fm_amp, fm_freq,  
            car_phase=0, fm_phase=0, t=time):
    mod = puresound(fm_amp, fm_freq)
    return(puresound(car_amp, car_freq + mod))
           
#Phase modulated sound
def pmsound(car_amp, car_freq, pm_amp, pm_freq,  
            car_phase=0, pm_phase=0, t=time):
    mod = puresound(pm_amp, pm_freq)
    return(puresound(car_amp, car_freq, mod))

#Harmonic series
def harmsound(fund,harm_nb, harm_amp=1) :
    hs = 0
    for i in range(harm_nb) :
        hs += puresound(harm_amp, (i+1)*fund) #harm_amp/(i+1)
    return(hs)

#Clic/burst
def clicsound(amp=0.5,clic_duration=0.001,d=3.0, sr=samprate):
    if clic_duration>duration:
        print("Error : clic is longer than overall sound")
    else:
        clic = flatsound(val = 0.0, d=duration, sr=samprate)
        t1 = sr
        t2 = sr + int(clic_duration*sr)
        clic[t1:t2]= amp
        return clic

#Ramp
def sineramp(sound, start = True, fade_duration=0.25, sr=samprate):
    fade_array = np.array([1.0 for i in range(len(sound))])
    fade_len = int(np.absolute(sr*fade_duration))
    fade_sine = puresound(amp = 0.5, freq = 1/(fade_duration), phase = np.pi/2, 
                          t = np.linspace(0, fade_duration, fade_len*2)) + 0.5
    if start == True :
        fade_array[:fade_len] = fade_sine[fade_len:]
    else :
        fade_array[len(fade_array) - fade_len:] = fade_sine[:fade_len]
    ramped_sound = np.multiply(sound , fade_array)
    return ramped_sound

#Ramp on both side
def bracket_ramp(sound, fade_duration=0.10, sr=samprate):
    startramp_sound = sineramp(sound, start=True, fade_duration= fade_duration, sr = samprate)
    endramped_sound = sineramp(startramp_sound, start=False, fade_duration = fade_duration, sr = samprate)
    return endramped_sound

def short(sound, start, end, ramp = True, sr = samprate):
    start = int(np.round(start * sr, 2))
    end = int(np.round(end * sr, 2))
    short_signal = sound[start:end]
    if ramp == True:
        short_signal = bracket_ramp(short_signal, fade_duration=0.10, sr=samprate)
    return short_signal

#Add a sound to another
# créer un flatsound de base de la taille du son le plus long ?
def addin(base_sound, added_sound, time_code=0, ramp_duration = 0.25, sr= samprate): 

    if ramp_duration == 0 :
        ramped_sound = added_sound
    else:
        ramped_sound = bracket_ramp(added_sound)    

    if time_code < 0 :
        new_start = int(np.round(abs(time_code) * sr, 2))
        ramped_sound = ramped_sound[new_start:]
        start = 0

    else:
        start = int(np.round(time_code * sr, 2))
    
    end = int(start + len(ramped_sound))
        
    
    
    added_string = np.array([0.0 for i in range(len(base_sound))])
    
    if end > len(base_sound):
        sound_stop = len(base_sound) - start
        ramped_sound = ramped_sound[:sound_stop]
        added_string[start:] = ramped_sound
    else:
        added_string[start:end] = ramped_sound
    
    final_sound = np.add(base_sound, added_string)
    return final_sound

#read wave file
def waveread(file_name, samprate=44100):
    y, s = librosa.load(file_name, sr=samprate, mono = True)   
    return y

#peak-normalize signal
def peak_normalize(song_array, peak_value, dBSPL = False , save_path = None, samprate = 44100): 
    if dBSPL == True:
        # convert dBSPL in wave amplitude
        peak_wave = maad.util.dBSPL2wav(peak_value)
        norm_song = song_array/np.amax(np.absolute(song_array))*peak_wave
    else: # peak_value is given in wave amplitude
        norm_song = song_array/np.amax(np.absolute(song_array))*peak_value
    if save_path != None:
        soundfile.write(save_path, norm_song, samprate=samprate)
    return norm_song

def one_normalize(song_array):
    max_array = max(song_array)
    return song_array / max_array

def rms_normalize(song_array):
    rms_power =  maad.util.rms(song_array)
    return (song_array / rms_power)


#apply reverberation

# ir_path = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\Reverb\trollers-gill\mono\dales_site1_1way_mono.wav")
# alt_ir = waveread(ir_path)

# path = Path(r"C:\Users\ecoac-field\OneDrive\Documents\Articles-Recherches\Reconstructor\Reverb\simulation_goestchel\Risoux200Hz\Time.npy")
# goestchel_ir = np.load(path)

# ir_path = "C:/Users/ecoac-field/OneDrive/Documents/Articles-Recherches/Reconstructor/Reverb/koli-national-park-summer/mono/koli_summer_site2_4way_mono.wav"
# koli_ir = waveread(ir_path)


# def reverb(signal, impulse_response = koli_ir, normalize_ir = False, samprate = 44100):
#     if normalize_ir == True :
#         impulse_response = one_normalize(impulse_response)   
#     return np.convolve(signal, impulse_response)

def show_spectrogram(sound, min_freq = 0, max_freq = 20000, vmin = None, vmax = None, samprate = 44100):
    N = 4096
    Sxx_power,tn,fn,ext = maad.sound.spectrogram(sound, 
                                                 samprate, 
                                                 nperseg=N, 
                                                 noverlap=N//2, 
                                                 flims = [min_freq, max_freq],
                                                 mode = 'amplitude')
    Sxx_dB = maad.util.power2dB(Sxx_power) # convert into dB
    fig_kwargs = {'vmax': vmax,
                      'vmin':vmin,
                      'extent':ext,
                      'figsize':(4,13),
                      'title':'Power spectrogram density (PSD)',
                      'xlabel':'Time [sec]',
                      'ylabel':'Frequency [Hz]',
                      }
    ax, fig = maad.util.plot2d(Sxx_dB,**fig_kwargs)
    return fig