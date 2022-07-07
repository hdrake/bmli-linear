import numpy as np
import scipy

import os, sys

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

N = 1.e-3 # buoyancy frequency
f = 5.e-5 # Coriolis parameter
θ = 2.e-3 # slope angle

g = 9.81 # gravitation acceleration
α = 2.e-4 # thermal expansion coeff

Hbfz = 300. # thickness of bottom frontal zone
M = 4.0e-5 # strength of bottom front
δz = 25. # interface smoothing scale

κ0 = 5.e-5 # background diffusivity
κ1 = 5.e-3 # bottom enhancement of diffusivity
h = 250. # decay scale of mixing
σ = 1 # Prandtl number

H = 1500. # domain height

def calc_Mc(N, θ):
    return N*np.sqrt(np.sin(θ))

def calc_Nbfz(N, M, θ):
    return np.sqrt(N**2 - M**2/np.tan(θ))
    
def calc_m_from_M(M, Mc):
    return M/(Mc-M)

def calc_M_from_m(m, Mc):
    return Mc*(m/(1. +m))

def calc_M_from_Ri(Ri, N, f, θ):
    return np.sqrt(
        (-f**2*np.tan(θ)**(-1) + np.sqrt(f**4*np.tan(θ)**(-2) + 4*Ri*f**2*N**2))
        / (2*Ri)
    )

def calc_Ri(N, M, θ, f):
    Bzv = N**2 - M**2*np.tan(θ)**(-1)
    Bxh = M**2
    Vzv = Bxh/f
    return Bzv/(Vzv**2)

def calc_Ld(N, H, f, Ri):
    return (N*H/f) * np.sqrt(1. + 1./Ri)

def calc_Lmax(N, H, f, Ri):
    return 3.9*calc_Ld(N, H, f, Ri)

def calc_td(f, Ri):
    return np.sqrt(54. / 5.) * np.sqrt( (1. + Ri) ) / np.abs(f)

def Houter(h, k1, k0): return h*np.log(k1/k0)

def sigmoid(z, z0=Hbfz, δz=δz):
    return (scipy.special.erf(np.sqrt(np.pi)/2. * -(z-z0)/ δz ) + 1.)/2.

def sigmoid_int(z, z0=Hbfz, δz=δz):
    return ( (z-z0)*(scipy.special.erf(np.sqrt(np.pi)/2. * -(z-z0)/ δz ) + 1.) - 2/np.pi*δz*np.exp(-np.pi*(z-z0)**2/(2*δz)**2) )/2.

def bottom_frontal_zone_basic(z, M, Hbfz=Hbfz, θ=θ, f=f, δz=δz):
    B = np.zeros_like(z)
    U = np.zeros_like(z)
    V = np.zeros_like(z)
    
    B = -M**2/np.sin(θ) * sigmoid_int(z, z0=Hbfz, δz=δz)
    V =  M**2 / (f * np.cos(θ)) * sigmoid_int(z, z0=Hbfz, δz=δz)
    V -= np.min(V)
    
    return B, U, V
