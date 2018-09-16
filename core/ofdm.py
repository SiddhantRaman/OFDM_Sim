# ofdm.py

import numpy as np
from scipy import signal
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
    f = np.linspace(-Bandwidth, Bandwidth, NoOfCarriers*5000)

    fig1 = plt.figure(figsize = (100,5))
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
    fig2 = plt.figure(figsize = (100,5))
    plt.ylim = ((-0.5, 1.5))
    Cnoise = np.zeros(np.size(f))
    for i in np.linspace(iMin, iMax, NoOfCarriers):
        c = np.sinc((NoOfCarriers+1)/Bandwidth * (f - i*Bandwidth/(NoOfCarriers+1) + Bandwidth/(NoOfCarriers+1)*fnoise[0, int(i + iMax)]))
        Cnoise = Cnoise + c
        plt.plot(f, c, '-b')
    myx = Bandwidth/(NoOfCarriers+1) * (np.linspace(iMin, iMax, NoOfCarriers) - fnoise)
    print(myx)
    plt.stem(np.transpose(myx), np.ones(NoOfCarriers), '-r')
    plt.plot(f, Cnoise, '-g', label='OFDM Symbol with freq noise')
    plt.plot(f, np.zeros(np.size(f)), '-k')
    plt.legend(loc='upper center', shadow=True)
    plt.show()

##############################################################################
# ofdm_symbol() func generates a time-domain OFDM symbol with the CP appended
# input params : Bandwidth, Number of Carriers, Guard Size, CP Size
# return : NumberOfCarrier + CPSize sized timedomain np.array using BPSK
# numpy fft module is used to generate ifft and fft of the freq domain symbol
##############################################################################
def ofdm_symbol(Bandwidth=1000, NoOfCarriers=64, GuardSize=7, CPSize=16):
    Data1 = np.random.randint(2, size=int(NoOfCarriers/2 - GuardSize))
    Data1[Data1 == 0] = -1
    Data2 = np.random.randint(2, size=int(NoOfCarriers/2 - GuardSize -1))
    Data2[Data2 == 0] = -1

    Data = np.concatenate((np.zeros(GuardSize), Data1, np.zeros(1))) 
    Data = np.concatenate((Data, Data2, np.zeros(GuardSize)))
    
    FftData = Data[int(NoOfCarriers/2) : ]
    FftData = np.concatenate((FftData, Data[:int(NoOfCarriers/2)]))
    
    TimeDomainData = np.fft.ifft(FftData, NoOfCarriers)
    #print(TimeDomainData.size)
    #h = np.fft.fft(TimeDomainData, NoOfCarriers)
    #fig1 = plt.figure()
    #plt.plot(500/np.pi * np.linspace(-np.pi, np.pi, NoOfCarriers), abs(h), '-g')
    TimeDomainSymbol = np.concatenate((TimeDomainData[NoOfCarriers-CPSize: ], TimeDomainData))

    #h1 = np.fft.fft(TimeDomainSymbol, 4096)
    #plt.plot(500/np.pi * np.linspace(-np.pi, np.pi, 4096), abs(h1), '-r')
    
    # Lets try Welch function to estimate PSD of our OFDM symbol
    #f, Pxx_den = signal.welch(TimeDomainSymbol, Bandwidth)
    #fig2 = plt.figure()
    #plt.semilogy(f, Pxx_den)
    #plt.xlabel('Frequency [Hz]')
    #plt.ylabel('PSD [V**2/Hz]')

    #plt.show()
    return TimeDomainSymbol

###################################################################################
# ofdm_symbol_boundary() func implements estimation of OFDM sysmbol boundary with
#    the help of Cyclic Prefix which are appended at the beginning of each OFDM
#    symbol.
# Core algorithm uses Cross-Correlation between 2 windows of size CPSize. Whereever
#    Corr value is maximum that mean both windows are the same which means we are
#    at the OFDM symbol boundary
# input params: Bandwidth, Number of Carriers, GuardSize, CPSize
# return : Symbol Boundary Offset
###################################################################################
def ofdm_symbol_boundary(Bandwidth=1000, NoOfCarriers=64, GuardSize=7, CPSize=16):
    Symbol1 = ofdm_symbol(Bandwidth, NoOfCarriers, GuardSize, CPSize)
    Symbol2 = ofdm_symbol(Bandwidth, NoOfCarriers, GuardSize, CPSize)
    Symbol3 = ofdm_symbol(Bandwidth, NoOfCarriers, GuardSize, CPSize)
    DataStream = np.concatenate((Symbol1, Symbol2, Symbol3))
    #print(DataStream)
    f, Pxx_den = signal.welch(DataStream, Bandwidth)
    fig1 = plt.figure()
    plt.plot(f, 10*np.log10(Pxx_den))

    #DataStream will not start with a Symbol perfectly
    #So slice the DataStream to start in middle of first symbol
    DataStream = DataStream[20:]

    #Start seraching for the next Symbol Boundary
    CorrMax = 0
    CorrMaxInd = 0
    for i in np.linspace(0, NoOfCarriers+CPSize, NoOfCarriers + CPSize + 1):
        Window1 = DataStream[int(i) : int(i + CPSize-1)]
        Window2 = DataStream[int(i + NoOfCarriers) : int(i + NoOfCarriers + CPSize - 1)]
        Corr = np.inner(Window1, np.conj(Window2))
        if np.abs(Corr) > np.abs(CorrMax) :
            CorrMax = Corr
            CorrMaxInd = int(i)
    print(CorrMax, CorrMaxInd)
    #Theoritically CorrMaxInd = 60
    fig2 = plt.figure()
    plt.stem(np.linspace(0, CPSize-1, CPSize), DataStream[CorrMaxInd:CorrMaxInd+CPSize], '-g')
    plt.stem(np.linspace(0, CPSize-1, CPSize), DataStream[CorrMaxInd+NoOfCarriers : CorrMaxInd+NoOfCarriers+CPSize], '.r')
    plt.show()
    return CorrMaxInd


#ofdm_symbol_boundary(1000, 64, 7, 16)
#ofdm_symbol(1000, 64, 7, 16)
ofdm_noise_plot(1000, 0.25, 11)
