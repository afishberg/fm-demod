
import numpy as np
from numpy import pi
import scipy
from scipy import signal

from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt

# ==============================================================================
# DEBUG
# ==============================================================================

"""
Generates 5 seconds worth of sampling of a 4Hz sin wave at 16 samples/sec
Shows plot of sin samples
Shows plot of FFT of sin

Question: Why does it always show 0.5 peaks at -pi/2 and pi/2 instead of -pi/4
and pi/4?
"""
def test_fft():
    fs = 16
    dur = 5
    hz = 4
    x = np.sin(hz*2*pi*np.arange(0,dur,1/fs))

    plt.plot(np.arange(len(x)),x)
    plt.show()

    X = np.fft.fft(x)
    plot_fft(X,'Test FFT',True)

"""
Plots a provided FFT

Question: Doesn't seem to work? I must misunderstand what FFT returns?
"""
def plot_fft(X,title='FFT',normalize=False):
    # normalizes the fft
    if normalize:
        X /= len(X)

    # calculates xaxis and yaxis for plot
    freq = np.angle(X)/pi
    resp = np.absolute(X)

    # orders values properly for plot
    seq = freq.argsort()
    freq = freq[seq]
    resp = resp[seq]

    # plots FFT
    plt.plot(freq,resp)
    plt.title(title)
    plt.xlabel('Normalized Freq')
    plt.ylabel('Freq Response')
    plt.grid()
    axes = plt.gca()
    axes.set_xlim([-1,1])
    plt.show()

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

Question: Shouldn't these be the same?
Question: Is my fft wrapping?
"""
def test_filter():
    lpf = signal.remez(256, [0, 0.1, 0.125, .5], [1, 0])
    plot_filter_response(lpf)

    X = fft(lpf)
    plot_fft(X,'lpf')

# ==============================================================================



# ==============================================================================
# Components
# ==============================================================================

def dt_channel_select(x,w0):
    # modulate to baseband
    n = np.arange(len(x))
    baseband = x * np.exp(-1j*w0*n)

    # low-pass filter
    #lpf = signal.remez(1024, [0,1/9,1/8,1],[1,0])
    #lpf = signal.remez(72, [0, 0.1, 0.2, 0.4, 0.45, 0.5], [0, 1, 0])
    lpf = signal.remez(256, [0, 0.1, 0.125, .5], [1, 0])
    plot_filter_response(lpf)

    X = fft(lpf)
    plot_fft(X,'lpf')

# ==============================================================================

def demod(infile, w0):
    # load data
    samples = scipy.fromfile(open(infile), dtype=scipy.complex64)
    samples = samples[0:len(samples)//20] # only use 0.5 sec of data for quick dbg

    dt_channel_select(samples,w0)

if __name__ == '__main__':
    # demod('blind_test.raw', -1.2916)

    # tests
    test_fft()
    #test_filter()
