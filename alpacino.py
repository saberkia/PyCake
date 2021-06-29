""" On behalf of Al Pacino """

from PyEMD import EEMD, EMD, BEMD
import numpy as np
import matplotlib.pyplot as plt


def calc_emd(signal):
    """ Empirical Mode Decomposition.
    Method of decomposing signal into Intrinsic Mode Functions (IMFs)

    Parameters
    ----------
    signal : numpy array
        input signal

    Returns
    -------
    numpy array
        the IMFs calculated by EMD.
    """
    emd_transform = EMD()
    IMFs = emd_transform(signal)
    return IMFs


def calc_eemd(signal):
    """ Ensemble Empirical Mode Decomposition.

        Ensemble empirical mode decomposition (EEMD) [Wu2009]_
    is noise-assisted technique, which is meant to be more robust
    than simple Empirical Mode Decomposition (EMD). The robustness is
    checked by performing many decompositions on signals slightly
    perturbed from their initial position. In the grand average over
    all IMF results the noise will cancel each other out and the result
    is pure decomposition.

        Parameters
        ----------
        signal : numpy array
            input signal

        Returns
        -------
        numpy array
            the IMFs calculated by Ensemble EEMD.
        """
    eemd_transform = EEMD()
    IMFs = eemd_transform(signal)
    return IMFs


def calc_bemd(signal):
    """ Bidimensional Empirical Mode Decomposition.

        Method decomposition 2D arrays like gray-scale images into 2D representations of
    Intrinsic Mode Functions (IMFs).

        Parameters
        ----------
        signal : int or float
            input signal

        Returns
        -------
        numpy array
            the IMFs calculated by BEMD.
        """
    bemd_transform = BEMD()
    IMFs = bemd_transform(signal)
    return IMFs


if __name__ == "__main__":
    N = 800
    tMin, tMax = 0, 2 * np.pi
    T = np.linspace(tMin, tMax, N)
    S = np.sin(30 * T * (1 + 0.2 * T)) + np.sin(4 * T + 0.5) #+ 0.1*np.sin(20 * T)
    IMFs = calc_emd(S)
    imfNo = IMFs.shape[0]
    c = 1
    r = int(np.ceil((imfNo + 1) / c))
    print(imfNo)

    plt.ioff()
    plt.subplot(r, c, 1)
    plt.plot(T, S, 'r')
    plt.xlim((tMin, tMax))
    plt.title("Original signal")

    for num in range(imfNo):
        plt.subplot(r, c, int(num + 2))
        plt.plot(T, IMFs[num], 'g')
        plt.xlim((tMin, tMax))
        plt.ylabel("IMF " + str(num + 1))

    plt.tight_layout()
    plt.show()
