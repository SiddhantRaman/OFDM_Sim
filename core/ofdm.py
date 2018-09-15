# ofdm.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
print("OFDM SIMULATION CODE")

###############################################################################################################
# This function plots Fig1 with the OFDM based orthogonal Frequency Divided Spectrum
# Fig2 plots the effect of frequency drift on OFDM symbol in the system
# input: Bandwidth(1000), Maximum Noise in Frequency allowed(0.3), Number of carriers in which the whole Bandwidth will be divided(11)
# f : calculating the x axis to compute sinc function values on x axis different points
# iMin & iMax : Frequency Bin indices around 0 frequency
# c : sinc(6/500 * (f - 500*i/6 + fnoise)) : ith Frequency Bin starting from iMin + random noise
# C : Combined OFDM symbol
################################################################################################################
def ofdm_noise_plot(Bandwidth=1000, fnoiseMax=0.3, NoOfCarriers=11):
    iMin = -(NoOfCarriers - 1)/2
    iMax = (NoOfCarriers-1)/2
    f = np.linspace(-Bandwidth, Bandwidth, NoOfCarriers*50)

    fig1 = plt.figure(figsize = (10,10))
    plt.ylim((-0.5, 1.5))
    plt.xlim((-Bandwidth, Bandwidth))

    C = np.zeros(np.size(f))
    for i in np.linspace(iMin, iMax, NoOfCarriers):
        c = np.sinc((NoOfCarriers+1)/Bandwidth * (f - i*Bandwidth/(NoOfCarriers+1)))
        C = C + c
        plt.plot(f, c, '-b')
    plt.stem(Bandwidth/(NoOfCarriers+1) * np.linspace(iMin, iMax, NoOfCarriers), np.ones(NoOfCarriers), '-r')
    plt.plot(f, C, '-g', label='OFDM Symbol')
    plt.plot(f, np.zeros(np.size(f)), '.k')
    plt.legend(loc='upper center', shadow=True)
    #plt.show()

    fnoise = fnoiseMax * 2*(np.random.random_sample((1, NoOfCarriers)) - 0.5)
    #print(fnoise)
    fig2 = plt.figure(figsize = (10,10))
    plt.ylim = ((-0.5, 1.5))
    Cnoise = np.zeros(np.size(f))
    for i in np.linspace(iMin, iMax, NoOfCarriers):
        c = np.sinc((NoOfCarriers+1)/Bandwidth * (f - i*Bandwidth/(NoOfCarriers+1) + Bandwidth/(NoOfCarriers+1)*fnoise[0, int(i + iMax)]))
        Cnoise = Cnoise + c
        plt.plot(f, c, '-b')
    myx = Bandwidth/(NoOfCarriers+1) * (np.linspace(iMin, iMax, NoOfCarriers) - fnoise)
    print(myx)
    plt.stem(np.transpose(myx), np.ones(NoOfCarriers), '-r')
    plt.plot(f, Cnoise, '-g', label='OFDM Symbol')
    plt.plot(f, np.zeros(np.size(f)), '-k')
    plt.legend(loc='upper center', shadow=True)
    plt.show()

ofdm_noise_plot(1000, 0.3, 11)
