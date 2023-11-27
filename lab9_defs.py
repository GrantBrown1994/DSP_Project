import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math as m
import os
cwd = os.getcwd()
np.set_printoptions(threshold=np.inf)
from scipy.io import wavfile
from scipy.io.wavfile import write
import pyaudio
import wave
import sys

def compute_fir_filter_output(x_n):
    w_n = np.zeros(len(x_n) + 1)
    for i in range(0, len(x_n)):
        if i == 0:
            w_n[i] = x_n[i]
        elif i == len(x_n) + 1:
            w_n[i] = -0.9*x_n[i-1]
        else:
            w_n[i] = x_n[i] - 0.9*x_n[i-1]
    return w_n

def open_wav(title):
    samplerate, data = wavfile.read(title)
    return data,samplerate

def echoFIR(x_n,td,fs,echo_amp):
    samp_delay = int(fs*td)
    bk = np.zeros(samp_delay,dtype=float)
    bk[0] = 1
    bk[samp_delay - 1] = echo_amp
    echo = np.convolve(x_n,bk)
    return echo

def new_wav(data,rate, title):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(title, rate, scaled)

def play_wav(title):
        # open the file for reading.
        # length of data to read.
        chunk = 1024
        wf = wave.open(title, 'rb')

        # create an audio object
        p = pyaudio.PyAudio()

        # open stream based on the wave object which has been input.
        stream = p.open(format =
                        p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True)

        # read data (based on the chunk size)
        data = wf.readframes(chunk)

        # play stream (looping from beginning of file to the end)
        while data:
            # writing to the stream is what *actually* plays the sound.
            stream.write(data)
            data = wf.readframes(chunk)


        # cleanup stuff.
        wf.close()
        stream.close()    
        p.terminate()


if __name__ == "__main__":
    pass