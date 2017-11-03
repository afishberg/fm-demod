
import numpy as np
from numpy import pi
import scipy
from scipy import signal

from scipy.fftpack import fft, ifft
from numpy.fft import fftshift
import matplotlib.pyplot as plt

# ==============================================================================
# DEBUG
# ==============================================================================

"""
Generates 5 seconds worth of sampling of a 4Hz sin wave at 16 samples/sec
Shows plot of sin samples
Shows plot of FFT of sin
"""
def test_fft():
    fs = 16
    dur = 5
    hz = 4
    x = np.sin(hz*2*pi*np.arange(0,dur,1/fs))

    plt.plot(np.arange(len(x)),x)
    plt.show()

    X = fft(x)
    Xs = fftshift(X)
    plot_fft(Xs,'Test FFT',True)

"""
Plots a provided FFT
"""
def plot_fft(X,title='FFT',normalize=False):
    # normalizes the fft
    if normalize:
        X /= len(X)

    # calculates xaxis and yaxis for plot
    freq = np.arange(len(X)) / len(X) * 2 - 1
    resp = np.absolute(X)
    
    # plots FFT
    plt.plot(freq,resp)
    plt.title(title)
    plt.xlabel('Normalized Freq')
    plt.ylabel('Freq Response')
    plt.grid()
    axes = plt.gca()
    #axes.set_xlim([-1,1])
    plt.show()

def plot_freq_response(x):
    plot_fft(fftshift(fft(x)))

"""
Code I found on the internet for testing generated remez filter
"""
def plot_filter_response(h):
    freq, response = signal.freqz(h)
    ampl = np.abs(response)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.semilogy(freq/(2*pi), ampl, 'b-')  # freq in Hz
    plt.show()

"""
Generates a remez filter and then shows freq response and my fft function
"""
def test_filter():
    lpf = signal.remez(256, [0, 0.1, 0.125, .5], [1, 0])
    plot_filter_response(lpf)

    X = fftshift(fft(lpf))
    plot_fft(X,'lpf')

# ==============================================================================



# ==============================================================================
# Components
# ==============================================================================

def dt_channel_select(x,w0):
    # modulate to baseband
    n = np.arange(len(x))
    baseband = x * np.exp(-1j*w0*n)

    #plot_freq_response(baseband)

    # low-pass filter
    #lpf = signal.remez(1024, [0, 0.048, 0.05, .5], [1, 0])
    lpf = signal.remez(512, [0, 100e3, 128e3, 1.024e6], [1, 0], Hz=2.048e6)
    #plot_freq_response(lpf)

    filtered  = signal.convolve(baseband,lpf)
    decimated = filtered[0::8]

    return decimated

# ==============================================================================

def freq_discriminator(x):
    limited = limiter(x)

    top = differentiator_top(limited)
    bot = differentiator_bot(limited)

    y2 = top*bot
    # TODO Shouldn't this be only Im() at this point? Currently has Re() and Im().
    #print(y2)
    im = np.imag(y2)
    return im

def limiter(x):
    mag = np.absolute(x)
    y = x/mag
    return y

# TODO make ensure TOP/BOT use same M
def differentiator_top(x):
    ddt = signal.remez(512, [0, 128e3], [1], Hz=256e3, type='differentiator')
    #plot_freq_response(ddt)
    filtered = signal.convolve(x,ddt)
    return filtered[0:-511]

def differentiator_bot(x):
    conj = np.conj(x)

    freqs = np.arange(0,2*pi,2*pi/len(x))
    delay = np.exp(-1j*freqs*511)
    return conj*delay

# ==============================================================================

def deemphasis_filter(x):
    None

# ==============================================================================

def mono_select(x):
    None

# ==============================================================================

# ==============================================================================

def demod(infile, w0):
    # load data
    samples = scipy.fromfile(open(infile), dtype=scipy.complex64)

    x = samples[0:len(samples)//20] # only use 0.5 sec of data for quick dbg
    #plot_freq_response(x)

    y = dt_channel_select(x,w0)

    m = freq_discriminator(y)

if __name__ == '__main__':
    demod('blind_test.raw', -1.2916)

    # tests
    #test_fft()
    #test_filter()
