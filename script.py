
# ==============================================================================
# Imports
# ==============================================================================
import numpy as np
from numpy import pi
from numpy.fft import fft, fftshift

import scipy
from scipy import signal

import matplotlib.pyplot as plt 

import sounddevice as sd

# ==============================================================================
# Debug Plotting Functions
# ==============================================================================

PLOT_ON = True
SAVE_FIG = True
FIG_COUNT = 0
PLAY_AUDIO = False

"""
Takes the Fourier transform of samples, shifts them, and plots them
"""
def plot_fft(x,title='FFT',phaseplot=False,dbplot=True,normalize=False):
    X = fftshift(fft(x,n=4096))
    plot_freq(X,title,phaseplot,dbplot,normalize)

"""
Plots samples from a Fourier transform
"""
def plot_freq(X,title='FFT',phaseplot=False,dbplot=True,normalize=False):
    # checks if plotting has been disabled
    global PLOT_ON, SAVE_FIG, FIG_COUNT
    if not PLOT_ON:
        return

    # normalizes the fft
    if normalize:
        X /= len(X)

    # calculates xaxis for plot
    freq = np.arange(len(X)) / len(X) * 2 - 1
    
    # plots FFT
    if phaseplot:
        resp = np.angle(X)
        norm = resp/pi

        plt.plot(freq,norm)
        plt.ylabel('Normalized Phase')
    elif dbplot:
        resp = np.abs(X)
        norm = 20*np.log10(resp)

        plt.plot(freq,norm)
        plt.ylabel('Magnitude (dB)')
    else:
        resp = np.abs(X)

        plt.plot(freq,resp)
        plt.ylabel('Magnitude')

    plt.title(title)
    plt.xlabel('Normalized Freq')

    plt.grid()
    axes = plt.gca()
    axes.set_xlim([-1,1])

    if SAVE_FIG:
        fname = 'fig%02d.png' % FIG_COUNT
        plt.savefig(fname)
        print('Saved %s' % fname)
        FIG_COUNT += 1
        plt.gcf().clear()
    else:
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
    plot_fft(baseband,'Modulated Signal to Baseband')

    # low-pass filter
    lpf = signal.remez(512, [0, 100e3, 128e3, 1.024e6], [1, 0], Hz=2.048e6)
    plot_fft(lpf,'Pre-Decimation Low-Pass Filter')

    filtered  = signal.convolve(baseband,lpf)
    plot_fft(filtered,'Low-Pass Filtered Baseband Signal')

    decimated = filtered[0::8]
    plot_fft(decimated,'Decimated Filtered Baseband Signal (2.048 MHz -> 256 kHz)')

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
    plot_fft(x,'Limiter Input')
    plot_fft(x,'Limiter Input (Phase)',phaseplot=True)
    y1 = limiter(x)
    plot_fft(y1,'Limiter Output')
    plot_fft(y1,'Limiter Output (Phase)',phaseplot=True)

    top = differentiator_top(y1,M)
    bot = differentiator_bot(y1,M)

    y2 = top*bot

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
    plot_fft(ddt,'Differentiator Frequency Response',dbplot=False)

    filtered = signal.convolve(x,ddt)
    return filtered#[0:-M]

"""
Implmentation of conjugate and non-integer delay, or the bot path in frequency
discriminator.

x :: input signal
M :: order of differentiator (i.e. length = M+1)

Outlined in Project 1 Section 3 Part A. Also see Figure 4.
"""
def differentiator_bot(x,M):
    conj = np.conj(x)

    n = np.arange(0,M+1)
    c = pi*(n-M/2)
    h = np.sin(c)/c

    conv = signal.convolve(conj,h)
    return conv

"""
Question:
freqs = np.arange(0,2*pi,2*pi/len(x))
delay = np.exp(-1j*freqs*M/2)
return ifft(fft(conj)*delay)
"""

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
    #print('numer:',numer,'denom:',denom)

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
    plot_fft(lpf,'Mono Select Low-Pass Filter')

    filtered  = signal.convolve(x,lpf)
    plot_fft(filtered,'Low-Pass Filtered Mono Select')

    decimated = filtered[0::4]
    plot_fft(decimated,'Decimated Mono Select Signal (256 kHz -> 64 kHz)')

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
    plot_fft(x,'Input Samples')

    # Project 1 Section 2
    plot_fft(x,'Channel Select Input')
    y = channel_select(x,w0)
    plot_fft(y,'Channel Select Output')

    # Project 1 Section 3 Part A
    plot_fft(y,'Frequency Discriminator Input')
    m = freq_discriminator(y,511)
    plot_fft(m,'Frequency Discriminator Output')

    # Project 1 Section 3 Part B
    plot_fft(m,'Deemphasis Filter Input')
    md = deemphasis_filter(m, 1/256e3, 75e-6)
    plot_fft(md,'Deemphasis Filter Output')

    # Project 1 Section 3 Parts C and D
    plot_fft(md,'Mono Select Input')
    a = mono_select(md)
    plot_fft(a,'Mono Select Output')

    plot_fft(a,'Final Audio')

    global PLAY_AUDIO
    if PLAY_AUDIO:
        sd.play(a,64e3)
        sd.wait()

# ==============================================================================



# ==============================================================================
# Main Function
# ==============================================================================

if __name__ == '__main__':
    demod('blind_test.raw', -1.2916)

# ==============================================================================
