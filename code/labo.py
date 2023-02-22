import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from zplane import zplane


def prob1():
    print('allooo')
    K = 1
    z1 = 0.8j
    z2 = -0.8j
    p1 = 0.95*np.exp((np.pi*1j)/8)
    p2 = 0.95*np.exp((-np.pi*1j)/8)

    num = np.poly([z1, z2])
    denum = np.poly([p1, p2])

    zplane(num, denum)

    w, amp = signal.freqz(num, denum)

    fig, ax1 = plt.subplots()
    ax1.set_title('Amplitude et phase de la réponse en fréquence')
    ax1.plot(w, 20 * np.log10(abs(amp)), 'r')
    ax1.set_xlabel('Fréquence (rad/éch.)')
    ax1.set_ylabel('Amplitude (dB)', color='r')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(amp))
    ax2.plot(w, angles, 'b')
    ax2.set_ylabel('Angle (rad)', color='b')
    plt.show()

    impulsion = np.zeros(500)
    impulsion[int(len(impulsion)/2)] = 1
    plt.stem(impulsion)
    plt.show()

    h = signal.lfilter(num, denum, impulsion)
    plt.plot(h)
    plt.show()

    h_inv = signal.lfilter(denum, num, h)
    plt.stem(h_inv)
    plt.show()

if __name__ == '__main__':
    prob1()