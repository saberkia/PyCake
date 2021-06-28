from PyEMD import EMD
import numpy as np
import matplotlib.pyplot as plt

N = 200
tMin, tMax = 0, 2*np.pi
T = np.linspace(tMin, tMax, N)
S = np.sin(20*T*(1+0.2*T)) + T**2 + np.sin(13*T)
emd = EMD()
IMFs = emd(S)
imfNo = IMFs.shape[0]
c = 1
r = np.ceil((imfNo+1)/c)
print(imfNo)

plt.ioff()
plt.subplot(r, c, 1)
plt.plot(T, S, 'r')
plt.xlim((tMin, tMax))
plt.title("Original signal")

for num in range(imfNo):
    plt.subplot(r, c, num+2)
    plt.plot(T, IMFs[num], 'g')
    plt.xlim((tMin, tMax))
    plt.ylabel("Imf " + str(num+1))

plt.tight_layout()
plt.show()