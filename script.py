
# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
from numpy import pi
from numpy.fft import fft, ifft, fftshift

import scipy
from scipy import signal

import matplotlib.pyplot as plt 

import sounddevice as sd

# ==============================================================================
# Debug Plotting Functions
# ==============================================================================

PLOT_ON = True

"""
Takes the Fourier transform of samples, shifts them, and plots them
"""
def plot_fft(x,title='FFT',logplot=False,normalize=False):
    X = fftshift(fft(x))
    plot_freq(X,title,logplot,normalize)

"""
Plots samples from a Fourier transform
"""
def plot_freq(X,title='FFT',logplot=False,normalize=False):
    # checks if plotting has been disabled
    global PLOT_ON
    if not PLOT_ON:
        return

    # normalizes the fft
    if normalize:
        X /= len(X)

    # calculates xaxis and yaxis for plot
    freq = np.arange(len(X)) / len(X) * 2 - 1
    resp = np.abs(X)
    
    # plots FFT
    if not logplot:
        plt.plot(freq,resp)
    else:
        plt.semilogy(freq,resp)
    plt.title(title)
    plt.xlabel('Normalized Freq')
    plt.ylabel('Freq Response')
    plt.grid()
    axes = plt.gca()
    axes.set_xlim([-1,1])
    plt.show()

# ==============================================================================



# ==============================================================================
# Components
# ==============================================================================

"""
Modulates selected channel down to baseband, lowpass filters so you can safely
decimate, and decimates signal from 2.048e6 Hz to 256e3 Hz.

Outlined in Project 1 Section 2.
"""
def channel_select(x,w0):
    # modulate to baseband
    n = np.arange(len(x))
    baseband = x * np.exp(-1j*w0*n)
    #plot_fft(baseband,'Modulated Signal to Baseband')

    # low-pass filter
    lpf = signal.remez(512, [0, 100e3, 128e3, 1.024e6], [1, 0], Hz=2.048e6)
    #plot_fft(lpf,'Pre-Decimation Low-Pass Filter')

    filtered  = signal.convolve(baseband,lpf)
    #plot_fft(filtered,'Low-Pass Filtered Baseband Signal')

    decimated = filtered[0::8]
    #plot_fft(decimated,'Decimated Filtered Baseband Signal (2.048 MHz -> 256 kHz)')

    return decimated

# ==============================================================================

"""
Produces the derivative of the phase of the input signal. Implemented via a
complex discriminator.

x :: input signal
M :: order of differentiator (i.e. length = M+1)

Outlined in Project 1 Section 3 Part A. Also see Figure 4.
"""
def freq_discriminator(x, M):
    #plot_fft(x,'Limiter Input')
    y1 = limiter(x)
    #plot_fft(y1,'Limiter Output')

    top = differentiator_top(y1,M)
    bot = differentiator_bot(y1,M)

    y2 = top*bot
    #print('y2:',y2)

    m = np.imag(y2)
    return m

"""
Makes magnitude of all samples unity.

x :: input signal

Outlined in Project 1 Section 3 Part A. Also see Figure 4.
"""
def limiter(x):
    mag = np.abs(x)
    y = x/mag
    return y

"""
Implmentation of the discrete time differentiator, or the top path in the
frequency discriminator.

x :: input signal
M :: order of differentiator (i.e. length = M+1)

Outlined in Project 1 Section 3 Part A. Also see Figure 4.
"""
def differentiator_top(x,M):
    ddt = signal.remez(M+1, [0, 128e3], [1], Hz=256e3, type='differentiator')
    #plot_fft(ddt,'Differentiator Frequency Response')

    filtered = signal.convolve(x,ddt)
    return filtered[0:-M]

"""
Implmentation of conjugate and non-integer delay, or the bot path in frequency
discriminator.

x :: input signal
M :: order of differentiator (i.e. length = M+1)

Outlined in Project 1 Section 3 Part A. Also see Figure 4.
"""
def differentiator_bot(x,M):
    conj = np.conj(x)

    freqs = np.arange(0,2*pi,2*pi/len(x))
    delay = np.exp(-1j*freqs*M/2)

    return ifft(fft(conj)*delay)

# ==============================================================================

"""
Implements the continuous deemphasis filter,
H_d(s) = 1/(1+s*tau_d)
H_d(jW) = 1/(1+j*W*tau_d)
as a discrete filter using the bilinear transform.

Transformed discrete filter is,
H_d(z) = (1+z^-1)/((1+tan(T/(2*tau_d))) + (1-tan(T/(2*tau_d)))z^-1)

x :: input signal
T :: sampling period (i.e. inverse of sampling frequency or T=1/fs)
tau_d :: proivded deemphesis period (tau_d = 75e-6 seconds)

Outlined in Project 1 Section 3 Part B.
"""
def deemphasis_filter(x,T,tau_d):
    numer = np.array([1,1])
    denom = np.array([
        1+np.tan(T/(2*tau_d)),
        1-np.tan(T/(2*tau_d))
    ])
    #print('numer:',numer)
    #print('denom:',denom)
    y = signal.lfilter(numer,denom,x)
    return y

# ==============================================================================

"""
Implements mono select lowpass filter and final decimation.

x :: input signal

Outlined in Project 1 Section 3 Parts C and D.
"""
def mono_select(x):
    lpf = signal.remez(512, [0, 15e3, 18e3, 128e3], [1, 0], Hz=256e3)
    #plot_fft(lpf,'Mono Select Low-Pass Filter')

    filtered  = signal.convolve(x,lpf)
    #plot_fft(filtered,'Low-Pass Filtered Mono Select')

    decimated = filtered[0::4]
    #plot_fft(decimated,'Decimated Mono Select Signal (256 kHz -> 64 kHz)')

    return decimated

# ==============================================================================



# ==============================================================================
# Processing Function
# ==============================================================================

def demod(infile, w0):
    # load data
    samples = scipy.fromfile(open(infile), dtype=scipy.complex64)

    # select how much data to process
    x = samples
    #x = samples[0:len(samples)//20] # only use 0.5 sec of data for quick debug
    #plot_fft(x,'Input Samples')

    # Project 1 Section 2
    #plot_fft(x,'Channel Select Input')
    y = channel_select(x,w0)
    #plot_fft(y,'Channel Select Output')

    # Project 1 Section 3 Part A
    #plot_fft(y,'Frequency Discriminator Input')
    m = freq_discriminator(y,511)
    #plot_fft(m,'Frequency Discriminator Output')

    # Project 1 Section 3 Part B
    #plot_fft(m,'Deemphasis Filter Input')
    md = deemphasis_filter(m, 1/256e3, 75e-6)
    #plot_fft(md,'Deemphasis Filter Output')

    # Project 1 Section 3 Parts C and D
    #plot_fft(md,'Mono Select Input')
    a = mono_select(md)
    #plot_fft(a,'Mono Select Output')

    sd.play(a,64e3)
    sd.wait()

# ==============================================================================



# ==============================================================================
# Main Function
# ==============================================================================

if __name__ == '__main__':
    demod('blind_test.raw', -1.2916)

# ==============================================================================
