#Decay vs No Decay analysis: SNO with varying theta12 and marginalising
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import gamma
from scipy.optimize import minimize 
import scipy.optimize
from scipy.optimize import Bounds
from scipy import special
import time
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Computer Modern Roman']})


########################
## Unit conversions
ergtoMeV = 6.24*10**5
cmtoMeV = 0.5*10**(11)
MeVsqtocmsq=(197.3*1.e-13)**2
kmtoMeV=1/(200*1.e-15*100*1.e-5)
MeVtoeV=1.e6

#####################
### Neutrino oscillation parameters
#th12=33.56*(np.pi)/180.0
#Th12Central=33.56*(np.pi)/180.0
#SigTh12=0.088
th13=8.46*(np.pi)/180.0
th23=41.6*(np.pi)/180.0
dmsq=7.5 * 10**(-5)* 10**(-12) ## in MeV^2


#####################
### Other parameters
GF=1.16*10**(-11)  ## in MeV^2
Rsun=6957000  ## in km
NA=6*10**(23)
me=0.511 #MeV
dSun=1.5*1.e8 # in km


################
### functions returning mixing angles within the Sun

def Ne(r):
    return 245*NA*np.exp(-10.54*r / (1.0*Rsun))  #### in cm^-3

def A(r):
    return np.sqrt(2)* GF* Ne(r) * (1/cmtoMeV )**3

def DeltaM(r, En):
    return np.sqrt( ( np.cos(2*th12)*dmsq/(2* En) - A(r) )**2  + ( np.sin(2*th12)*dmsq/(2* En)  )**2    )

def DeltaMtest(r, En, th, dm):
    thtest12=th
    dmtest12=dm
    return np.sqrt( ( np.cos(2*thtest12)*dmtest12/(2* En) - A(r) )**2  + ( np.sin(2*thtest12)*dmtest12/(2* En)  )**2    )

### Cos(2 ThetaM)
def Cos2thM( r, En):
    return  ( np.cos(2*th12)*dmsq/(2* En) - A(r)) / (DeltaM(r, En) )

def Cos2thMtest( r, En,th, dm):
    thtest12=th
    dmtest12=dm
    return  ( np.cos(2*thtest12)*dmtest12/(2* En) - A(r)) / (DeltaMtest(r, En, thtest12,dmtest12 ) )

### Cos(ThetaM)^2
def CosthMsq( r, En ):
    return 0.5*(1 + Cos2thM( r, En ) )

def CosthMsqtest( r, En ,th, dm):
    thtest12=th
    dmtest12=dm
    return 0.5*(1 + Cos2thMtest( r, En,thtest12 , dmtest12) )

### Sin(ThetaM)^2
def SinthMsq( r, En ):
    return 0.5*(1 - Cos2thM( r, En ) )

def SinthMsqtest( r, En ,th, dm):
    thtest12=th
    dmtest12=dm
    return 0.5*(1 - Cos2thMtest( r, En,  thtest12, dmtest12 ) )


############ Differential decay probabilities : A PL + B PR ################

def dGamdE2HCpp(A, B, E3, E2, m3, m2):
    return (1.0 /(16* np.pi * E3**2 * (1 - m3**2/(2* E3**2))) *((A**2* m2**4 * m3**4)/(
   32* E2**3 * E3**3) + (A**2 * m2**4* m3**2)/(16 * E2**3 *E3) - (A**2 * m2**4 * m3**2)/(
   16* E2**2* E3**2) - (A**2 *m2**2 * m3**4)/(16* E2**2* E3**2) + (
   3 *A**2* m2**2* m3**4)/(16* E2* E3**3) + (A**2 *E2* m3**4)/(4* E3**3) + (
   A**2* m2**2 *m3**2)/(2 *E2 *E3) + (A**2 *E2* m3**2)/E3 - (A**2 *m2**2 *m3**2)/(
   4* E3**2) - (A**2* m3**4)/(4* E3**2) + (A *B* m2**5* m3**5)/(64* E2**4 *E3**4) + (
   3 *A *B *m2**5* m3**3)/(32 *E2**4* E3**2) + (A* B *m2**5 * m3)/(8* E2**4) - (
   A *B *m2**5* m3**3)/(32 *E2**3* E3**3) - (A *B *m2**3 *m3**5)/(32 *E2**3 *E3**3) - (
   A* B *m2**5 *m3)/(8 *E2**3 *E3) - (A *B *m2**3 *m3**3)/(8 *E2**3 *E3) + (
   3* A *B *m2**3 *m3**5)/(32 *E2**2 *E3**4) + (9 *A *B *m2**3 *m3**3)/(
   16 *E2**2 *E3**2) + (3 *A *B *m2**3 *m3)/(4 *E2**2) - (A *B *m2**3 *m3**3)/(
   8 *E2 *E3**3) - (A *B *m2 *m3**5)/(8 *E2 *E3**3) - (A *B *m2**3 *m3)/(
   2 *E2 *E3) - (A *B *m2 *m3**3)/(2 *E2 *E3) + (A *B *m2 *m3**5)/(8 *E3**4) + (
   3* A *B *m2 *m3**3)/(4 *E3**2) + 2 *A *B *m2 *m3 + (B**2 *m2**4 *m3**4)/(
   32 *E2**3 *E3**3) + (3* B**2 *m2**4 *m3**2)/(16 *E2**3 *E3) + (B**2 *E3 *m2**4)/(
   4 *E2**3) - (B**2 *m2**4 *m3**2)/(16 *E2**2 *E3**2) - (B**2 *m2**2 *m3**4)/(
   16 *E2**2 *E3**2) - (B**2 *m2**4)/(4 *E2**2) - (B**2 *m2**2 *m3**2)/(4 *E2**2) + (
   B**2 *m2**2* m3**4)/(16* E2* E3**3) + (B**2 *m2**2 *m3**2)/(2 *E2 *E3) + (
   B**2 *E3 *m2**2)/E2))

def  dGamdE2HCmm(A, B, E3, E2, m3, m2):
    return  ( (1/(16*np.pi*E3**2*(1 - m3**2/(2*E3**2))))*((A**2*m2**4*m3**4)/(32*E2**3*E3**3) + 
      (3*A**2*m2**4*m3**2)/(16*E2**3*E3) + (A**2*E3*m2**4)/(4*E2**3) - (A**2*
      m2**4*m3**2)/(16*E2**2*E3**2) - 
      (A**2*m2**2*m3**4)/(16*E2**2*E3**2) - (A**2*m2**4)/(4*E2**2) - (A**2*
      m2**2*m3**2)/(4*E2**2) + 
      (A**2*m2**2*m3**4)/(16*E2*E3**3) + (A**2*m2**2*m3**2)/(2*E2*E3) + (A**2*
      E3*m2**2)/E2 + 
      (A*B*m2**5*m3**5)/(64*E2**4*E3**4) + (3*A*B*m2**5*m3**3)/(32*E2**4*
      E3**2) + 
      (A*B*m2**5*m3)/(8*E2**4) - (A*B*m2**5*m3**3)/(32*E2**3*E3**3) - (A*B*
      m2**3*m3**5)/(32*E2**3*E3**3) - 
      (A*B*m2**5*m3)/(8*E2**3*E3) - (A*B*m2**3*m3**3)/(8*E2**3*E3) + 
      (3*A*B*m2**3*m3**5)/(32*E2**2*E3**4) + (9*A*B*m2**3*m3**3)/(16*E2**2*
      E3**2) + 
      (3*A*B*m2**3*m3)/(4*E2**2) - (A*B*m2**3*m3**3)/(8*E2*E3**3) - (A*B*
      m2*m3**5)/(8*E2*E3**3) - 
      (A*B*m2**3*m3)/(2*E2*E3) - (A*B*m2*m3**3)/(2*E2*E3) + (A*B*m2*
      m3**5)/(8*E3**4) + 
      (3*A*B*m2*m3**3)/(4*E3**2) + 
   2*A*B*m2*m3 + (B**2*m2**4*m3**4)/(32*E2**3*E3**3) + 
      (B**2*m2**4*m3**2)/(16*E2**3*E3) - (B**2*m2**4*m3**2)/(16*E2**2*
      E3**2) - 
      (B**2*m2**2*m3**4)/(16*E2**2*E3**2) + (3*B**2*m2**2*m3**4)/(16*E2*
      E3**3) + (B**2*E2*m3**4)/(4*E3**3) + 
      (B**2*m2**2*m3**2)/(2*E2*E3) + (B**2*E2*m3**2)/
    E3 - (B**2*m2**2*m3**2)/(4*E3**2) - 
      (B**2*m3**4)/(4*E3**2))  )

def  dGamdE2HFmp(A, B, E3, E2, m3, m2):
    return  ( (1/(16*np.pi*
     E3**2*(1 - m3**2/(2*E3**2))))*(-((A**2*m2**4*m3**4)/(32*E2**3*E3**3)) - 
      (3*A**2*m2**4*m3**2)/(16*E2**3*E3) - (A**2*E3*m2**4)/(4*E2**3) + (A**2*
      m2**4*m3**2)/(16*E2**2*E3**2) + 
      (A**2*m2**2*m3**4)/(16*E2**2*E3**2) + (A**2*m2**4)/(4*E2**2) + (A**2*
      m2**2*m3**2)/(4*E2**2) - 
      (3*A**2*m2**2*m3**4)/(16*E2*E3**3) - (A**2*E2*m3**4)/(4*E3**3) - (A**2*
      m2**2*m3**2)/(E2*E3) - 
      (A**2*E3*m2**2)/E2 - (A**2*E2*m3**2)/
    E3 + (A**2*m2**2*m3**2)/(4*E3**2) + (A**2*m3**4)/(4*E3**2) + 
      A**2*m2**2 + 
   A**2*m3**2 - (A*B*m2**5*m3**5)/(64*E2**4*E3**4) - (3*A*B*m2**5*m3**3)/(32*
      E2**4*E3**2) - 
      (A*B*m2**5*m3)/(8*E2**4) + (A*B*m2**5*m3**3)/(32*E2**3*E3**3) + (A*B*
      m2**3*m3**5)/(32*E2**3*E3**3) + 
      (A*B*m2**5*m3)/(8*E2**3*E3) + (A*B*m2**3*m3**3)/(8*E2**3*E3) - 
      (3*A*B*m2**3*m3**5)/(32*E2**2*E3**4) - (9*A*B*m2**3*m3**3)/(16*E2**2*
      E3**2) - 
      (3*A*B*m2**3*m3)/(4*E2**2) + (A*B*m2**3*m3**3)/(8*E2*E3**3) + (A*B*
      m2*m3**5)/(8*E2*E3**3) + 
      (A*B*m2**3*m3)/(2*E2*E3) + (A*B*m2*m3**3)/(2*E2*E3) - (A*B*m2*
      m3**5)/(8*E3**4) - 
      (3*A*B*m2*m3**3)/(4*E3**2) - (B**2*m2**4*m3**4)/(32*E2**3*
      E3**3) - (B**2*m2**4*m3**2)/(16*E2**3*E3) + 
      (B**2*m2**4*m3**2)/(16*E2**2*E3**2) + (B**2*m2**2*m3**4)/(16*E2**2*
      E3**2) - 
      (B**2*m2**2*m3**4)/(16*E2*E3**3))  )


def  dGamdE2HFpm(A, B, E3, E2, m3, m2):
    return  (  (1/(16*np.pi*
     E3**2*(1 - m3**2/(2*E3**2))))*(-((A**2*m2**4*m3**4)/(32*E2**3*E3**3)) - 
      (A**2*m2**4*m3**2)/(16*E2**3*E3) + (A**2*m2**4*m3**2)/(16*E2**2*
      E3**2) + 
      (A**2*m2**2*m3**4)/(16*E2**2*E3**2) - (A**2*m2**2*m3**4)/(16*E2*
      E3**3) - 
      (A*B*m2**5*m3**5)/(64*E2**4*E3**4) - (3*A*B*m2**5*m3**3)/(32*E2**4*
      E3**2) - 
      (A*B*m2**5*m3)/(8*E2**4) + (A*B*m2**5*m3**3)/(32*E2**3*E3**3) + (A*B*
      m2**3*m3**5)/(32*E2**3*E3**3) + 
      (A*B*m2**5*m3)/(8*E2**3*E3) + (A*B*m2**3*m3**3)/(8*E2**3*E3) - 
      (3*A*B*m2**3*m3**5)/(32*E2**2*E3**4) - (9*A*B*m2**3*m3**3)/(16*E2**2*
      E3**2) - 
      (3*A*B*m2**3*m3)/(4*E2**2) + (A*B*m2**3*m3**3)/(8*E2*E3**3) + (A*B*
      m2*m3**5)/(8*E2*E3**3) + 
      (A*B*m2**3*m3)/(2*E2*E3) + (A*B*m2*m3**3)/(2*E2*E3) - (A*B*m2*
      m3**5)/(8*E3**4) - 
      (3*A*B*m2*m3**3)/(4*E3**2) - (B**2*m2**4*m3**4)/(32*E2**3*E3**3) - 
      (3*B**2*m2**4*m3**2)/(16*E2**3*E3) - (B**2*E3*m2**4)/(4*E2**3) + (B**2*
      m2**4*m3**2)/(16*E2**2*E3**2) + 
      (B**2*m2**2*m3**4)/(16*E2**2*E3**2) + (B**2*m2**4)/(4*E2**2) + (B**2*
      m2**2*m3**2)/(4*E2**2) - 
      (3*B**2*m2**2*m3**4)/(16*E2*E3**3) - (B**2*E2*m3**4)/(4*E3**3) - (B**2*
      m2**2*m3**2)/(E2*E3) - 
      (B**2*E3*m2**2)/E2 - (B**2*E2*m3**2)/
    E3 + (B**2*m2**2*m3**2)/(4*E3**2) + (B**2*m3**4)/(4*E3**2) + 
      B**2*m2**2 + B**2*m3**2))



def  GammaHCpp(A, B, E3,  m3, m2):
    return (  (1/(16*np.pi*E3**2*(1 - m3**2/(2*E3**2))))*
   ((1/(16*E3**3))*m2*
    np.log(E3)*(A**2*m2*m3**2*(8*E3**2 + 3*m3**2) - 
      2*A*B*m3*(4*E3**2 + m3**2)*
             (m2**2 + m3**2) + 
      B**2*m2*(4*E3**2 + m3**2)**2) - (1/(16*E3**3))*m2*np.log((E3*m2**2)/m3**2)*
        (A**2*m2*m3**2*(8*E3**2 + 3*m3**2) - 
      2*A*B*m3*(4*E3**2 + m3**2)*(m2**2 + m3**2) + 
           B**2*m2*(4*E3**2 + m3**2)**2) - (1/(192*E3**7*m2*m3**2))*(m2**2 - 
      m3**2)*
        (3*A**2*E3**2*
       m2*(32*E3**6 - 8*E3**4*m3**2 - 2*E3**2*m3**4 + m3**6)*(m2**2 + m3**2) +  A*B*m3*(384*E3**8*m2**2 + 288*E3**6*m2**2*m3**2 - 
         4*E3**4*m3**2*(m2**4 - 29*m2**2*m3**2 + m3**4) + 
                3*E3**2*m3**4*(m2**4 + 6*m2**2*m3**2 + m3**4) + 
         m3**6*(m2**4 + m2**2*m3**2 + m3**4)) +  3*B**2*E3**2*m2*
       m3**2*(-(8*E3**4) + 2*E3**2*m3**2 + m3**4)*(m2**2 + m3**2))) )

def  GammaHCmm(A, B, E3,  m3, m2):
    return (  (1/(16*np.pi*E3**2*(1 - m3**2/(2*E3**2))))*
   ((1/(16*E3**3))*m2*
    np.log(E3)*(A**2*m2*(4*E3**2 + m3**2)**2 - 2*A*B*m3*(4*E3**2 + m3**2)*
             (m2**2 + m3**2) + 
      B**2*m2*m3**2*(8*E3**2 + 3*m3**2)) - (1/(16*E3**3))*m2*
    np.log((E3*m2**2)/m3**2)*
        (A**2*m2*(4*E3**2 + m3**2)**2 - 
      2*A*B*m3*(4*E3**2 + m3**2)*(m2**2 + m3**2) +  B**2*m2*m3**2*(8*E3**2 + 3*m3**2)) - (1/(192*E3**7*m2*m3**2))*(m2**2 - 
      m3**2)*
        (3*A**2*E3**2*m2*
       m3**2*(-(8*E3**4) + 2*E3**2*m3**2 + m3**4)*(m2**2 + m3**2) + A*B*m3*(384*E3**8*m2**2 + 288*E3**6*m2**2*m3**2 - 
         4*E3**4*m3**2*(m2**4 - 29*m2**2*m3**2 + m3**4) + 
                3*E3**2*m3**4*(m2**4 + 6*m2**2*m3**2 + m3**4) + 
         m3**6*(m2**4 + m2**2*m3**2 + m3**4)) + 3*B**2*E3**2*
       m2*(32*E3**6 - 8*E3**4*m3**2 - 2*E3**2*m3**4 + m3**6)*(m2**2 + m3**2))) )


def  GammaHFmp(A, B, E3,  m3, m2):
    return (  (1/(16*np.pi*E3**2*(1 - m3**2/(2*E3**2))))*((1/(192*E3**7*m2))*
      (-(12*E3**4*m2**2*
        np.log(E3)*(A**2*m2*(16*E3**4 + 16*E3**2*m3**2 + 3*m3**4) - 
                 2*A*B*m3*(4*E3**2 + m3**2)*(m2**2 + m3**2) + 
          B**2*m2*m3**4)) + 
         12*E3**4*m2**2*
      np.log((E3*m2**2)/
        m3**2)*(A**2*m2*(16*E3**4 + 16*E3**2*m3**2 + 3*m3**4) - 
              2*A*B*m3*(4*E3**2 + m3**2)*(m2**2 + m3**2) + 
        B**2*m2*m3**4) + 
         (1/m3**2)*(m2**2 - 
        m3**2)*(3*A**2*E3**2*
         m2*(-(32*E3**6) - 16*E3**4*m3**2 + 2*E3**2*m3**4 + m3**6)*
                (m2**2 + m3**2) + 
        A*B*m3**3*(288*E3**6*m2**2 - 
           4*E3**4*(m2**4 - 29*m2**2*m3**2 + m3**4) + 
                   3*E3**2*m3**2*(m2**4 + 6*m2**2*m3**2 + m3**4) + 
           m3**4*(m2**4 + m2**2*m3**2 + m3**4)) - 
              3*B**2*E3**2*m2*m3**4*(2*E3**2 - m3**2)*(m2**2 + m3**2)))) )


def  GammaHFpm(A, B, E3,  m3, m2):
    return (   (1/(16*np.pi*E3**2*(1 - m3**2/(2*E3**2))))*((1/(192*E3**7*m2))*
      (-(12*E3**4*m2**2*
        np.log(E3)*(A**2*m2*m3**4 - 
          2*A*B*m3*(4*E3**2 + m3**2)*(m2**2 + m3**2) + 
                 B**2*m2*(16*E3**4 + 16*E3**2*m3**2 + 3*m3**4))) + 
     12*E3**4*m2**2*np.log((E3*m2**2)/m3**2)*
           (A**2*m2*m3**4 - 2*A*B*m3*(4*E3**2 + m3**2)*(m2**2 + m3**2) + 
              B**2*m2*(16*E3**4 + 16*E3**2*m3**2 + 3*m3**4)) + (1/
        m3**2)*(m2**2 - m3**2)*
           (-(3*A**2*E3**2*m2*m3**4*(2*E3**2 - m3**2)*(m2**2 + m3**2)) + A*B*m3**3*(288*E3**6*m2**2 - 
           4*E3**4*(m2**4 - 29*m2**2*m3**2 + m3**4) + 
                   3*E3**2*m3**2*(m2**4 + 6*m2**2*m3**2 + m3**4) + 
           m3**4*(m2**4 + m2**2*m3**2 + m3**4)) + 3*B**2*E3**2*
         m2*(-(32*E3**6) - 16*E3**4*m3**2 + 2*E3**2*m3**4 + m3**6)*(m2**2 + 
           m3**2))))   )

def  GammaTot(A, B, E3,  m3, m2):
    return 0.5 * (GammaHCpp(A, B, E3,  m3, m2)+ GammaHCmm(A, B, E3,  m3, m2)+GammaHFpm(A, B, E3,  m3, m2)+GammaHFmp(A, B, E3,  m3, m2))

                  
def WHCpp(A, B, E3, E2, m3, m2):
    return (dGamdE2HCpp(A, B, E3, E2, m3, m2)/GammaTot(A, B, E3,  m3, m2))


def WHCmm(A, B, E3, E2, m3, m2):
    return (dGamdE2HCmm(A, B, E3, E2, m3, m2)/GammaTot(A, B, E3,  m3, m2))


def WHFpm(A, B, E3, E2, m3, m2):
    return (dGamdE2HFpm(A, B, E3, E2, m3, m2)/GammaTot(A, B, E3,  m3, m2))
            

def WHFmp(A, B, E3, E2, m3, m2):
    return  (dGamdE2HFmp(A, B, E3, E2, m3, m2)/GammaTot(A, B, E3,  m3, m2))
             

#exit()

######### Probability at the Earth
def Pee(En):
    return ((CosthMsq( 0 , En ) * np.cos(th12)**2 + SinthMsq( 0 , En ) * np.sin(th12)**2) * np.cos(th13)**4)

def Peetest(En ,th, dm):
    thtest12=th
    dmtest12=dm
    return ((CosthMsqtest( 0 , En ,thtest12, dmtest12) * np.cos(thtest12)**2 + SinthMsqtest( 0 , En, thtest12, dmtest12 ) * np.sin(thtest12)**2) * np.cos(th13)**4)


#def PeeInvDec(f , En):  #f in eV^2, and En in MeV , hence MeVtoeV**2
  #  return ((CosthMsq( 0 , En ) * np.cos(th12)**2 + np.exp(-  dSun * kmtoMeV * f / ( En * MeVtoeV**2) ) *SinthMsq( 0 , En ) * np.sin(th12)**2 ) * np.cos(th13)**4)


EnuArr=np.logspace(np.log10(0.1),1.15,30)





###ProbeeSol=Pee(EnuArr)
##ProbeeSoltest1=Peetest(EnuArr, 33.56*(np.pi)/180.0, 1 * 10**(-5)* 10**(-12))
##ProbeeSoltest5=Peetest(EnuArr, 33.56*(np.pi)/180.0, 5* 10**(-5)* 10**(-12))
##ProbeeSoltest10=Peetest(EnuArr, 33.56*(np.pi)/180.0, 10 * 10**(-5)* 10**(-12))
##ProbeeSoltest50=Peetest(EnuArr, 33.56*(np.pi)/180.0, 50 * 10**(-5)* 10**(-12))
##ProbeeSoltest0p1=Peetest(EnuArr, 33.56*(np.pi)/180.0, 0.1 * 10**(-5)* 10**(-12))
###plt.semilogx(EnuArr,ProbeeSol)
##plt.semilogx(EnuArr,  ProbeeSoltest0p1,color='r',linewidth=2.5,label=r'$\Delta m^2=0.1\times10^{-5}$')
##plt.semilogx(EnuArr,  ProbeeSoltest1,color='g',linewidth=2.5,label=r'$\Delta m^2=1\times10^{-5}$')
##plt.semilogx(EnuArr,  ProbeeSoltest5,color='b',linewidth=2.5, label=r'$\Delta m^2=5\times10^{-5}$')
##plt.semilogx(EnuArr,  ProbeeSoltest10,color='m',linewidth=2.5, label=r'$\Delta m^2=10\times10^{-5}$')
##plt.semilogx(EnuArr,  ProbeeSoltest50,color='y',linewidth=2.5,label=r'$\Delta m^2=50\times10^{-5}$')
##plt.xlabel(r'$E(MeV)$',fontsize=20)
##plt.ylabel(r'$P_{ee}$',fontsize=20)
##plt.legend(loc='upper right', fontsize=12)
##plt.xlim(0.1,15)
##plt.ylim(0,1)
###plt.show()
##plt.savefig('PeeVariationWithDmSq.pdf', bbox_inches='tight')
##exit()


#ProbeeSol=Pee(EnuArr)
#ProbeeSoltest=Peetest(EnuArr, 33.56*(np.pi)/180.0, 7.5 * 10**(-5)* 10**(-12))
#print(ProbeeSol)
#print(ProbeeSoltest)
#exit()





###########
## Reading the 8B flux ###########
B8fluxArr=[]
B8data=np.array(np.loadtxt('Sun-nuclear-B8.dat'))
B8flux=interp1d( 1.e-6* B8data[::,0] , 1.e6* B8data[::,-1], kind='linear' )

#print(1.e-6* B8data[len(B8data)-1,0] )
#exit()

###########
## SNO Cross section ###########
SnoCSArr=[]
SnoCSdata=np.array(np.loadtxt('SNO_NueDcs.dat',skiprows=3,usecols=(0,3))) ## cs in cm^2
SnoCS=interp1d(SnoCSdata[::,0],SnoCSdata[::,1],kind='linear')

Emax=14.0 #MeV
Emin=7.0
DelEnbin=0.5
Nbins=int((Emax-Emin)/DelEnbin)
Entab=np.linspace(Emin,Emax,Nbins+1)
EthrSNO=1.446

NtarSno=3.01*10**31
DeltaT=1*3.15*1.e7  ## 1 yr to s
Epluslist=np.linspace(Emin, Emax, num=10)

def sigma(Etrue):
    return -0.462 + 0.5470*np.sqrt(Etrue) + 0.008722 * Etrue

def ResSno(Teff,Etrue): ### Ee_eff = Teff + m_e
    return np.sqrt(1.0/(2*np.pi)) *  (1.0/sigma(Etrue)) *np.exp(-0.5*((Teff +me+EthrSNO - Etrue)/(sigma(Etrue)))**2 )

def SpecVis(A, B, Ed, mdtest,thtest12):
    xsq= (mptest/mdtest)**2
    Eup=xsq * Ed
    EmaxB8=1.e-6* B8data[len(B8data)-1,0]
    if Eup>EmaxB8:
        Eup=EmaxB8
    Edlist = np.linspace(Ed, Eup , num=10)


    P1visIntegrand= np.array([ SinthMsqtest( 0 , Ep, thtest12, dmsq ) * WHCmm(A, B, Ep, Ed, mptest, mdtest) *
                               ( 1 -   np.exp(-  dSun * kmtoMeV * GammaTot(A, B, Ep,  mptest, mdtest)  ) ) * B8flux(Ep) for Ep in Edlist])
    P1vis=np.trapz(P1visIntegrand, Edlist)
    #print(Eup)
    #print(P1vis)

    return (np.cos(th13)**4   * np.cos(thtest12)**2 * P1vis + (CosthMsqtest( 0 , Ed, thtest12, dmsq ) * np.cos(thtest12)**2 +
             np.exp(-dSun * kmtoMeV * GammaTot(A, B, Ed,  mptest, mdtest) ) *SinthMsqtest( 0 , Ed, thtest12, dmsq  ) * np.sin(thtest12)**2 )* np.cos(th13)**4 * B8flux(Ed) )


# Central value data for SNO
def NdetSno(g, Teff, mdtest, thtest12):
    integral=np.array([NtarSno*DeltaT* ResSno(Teff, Epl) * SnoCS(Epl)* SpecVis(g ,0, Epl, mdtest, thtest12) for Epl in Epluslist])
    return np.trapz(integral,Epluslist)

def NEventsSno(g, Ebinarr, mdtest, thtest12):
    NEventsarr=np.array([NdetSno(g, Edet, mdtest, thtest12) for Edet in Ebinarr])
    return np.trapz(NEventsarr,Ebinarr)

### Central SNO data
NEventslist=[132.68985161287836, 131.41087531953355, 125.19162786750536, 115.01637950131297, 101.89925084852553, 86.93804774135691, 71.28407559152294, 56.038841011763, 42.129525518684375, 30.20832042144115, 20.603424635434898, 13.331413404240102, 8.162804414535442, 4.718557690300552]

##gval0=1.e-12
##mptest=1.e-6
##md0=1.e-12
##NEventslist=[]
##for nbin in range(Nbins):
##    Ebinlist=np.linspace(Emin+nbin*DelEnbin, Emin+(nbin+1)*DelEnbin,num=5)
##    NEventslist.append( NEventsSno(gval0, Ebinlist, md0,th12))
##print(NEventslist)
##exit()

########## Vary the coupling and the mass of daughter

def  ChiSNOThetaMarg(inpMarg, gval, mdval):
    #print(inpMarg)
    Testg=10**(gval)
    Testmd= mdval # in MeV
    Testth12=inpMarg[0]*(np.pi)/180.0 ## in radians
    NEventslisttestMarg=[]
    for nbin in range(Nbins):
        Ebinlist=np.linspace(Emin+nbin*DelEnbin, Emin+(nbin+1)*DelEnbin,num=5)
        NEventslisttestMarg.append( NEventsSno(Testg,Ebinlist, Testmd, Testth12))

    #print( NEventslisttestMarg)    
    chisqMarg= np.array([((n1-n2)**2)/(1.0*n2) for n1, n2 in zip(NEventslist, NEventslisttestMarg)])
    return sum(chisqMarg)


def  ChiSNO(inp):
    # Test data for SNO
    Testg=10**(inp[0])
    Testmd= inp[1] # in MeV
    Testth12= inp[2] ## in radians
    NEventslisttest=[]
    for nbin in range(Nbins):
        Ebinlist=np.linspace(Emin+nbin*DelEnbin, Emin+(nbin+1)*DelEnbin,num=5)
        NEventslisttest.append( NEventsSno( Testg, Ebinlist, Testmd, Testth12))
    #chisq= np.array([((n1-n2)**2)/(1.0*n2) for n1, n2 in zip(NEventslist, NEventslisttest)])
    chisq= np.array([((n1-n2)**2)/(1.0*n2)+(np.sin(Testth12)-SinsqTh12Central)**2/ (1.0*SigssqTh12**2)  for n1, n2 in zip(NEventslist, NEventslisttest)])
    return sum(chisq)


#inp_arr=(-6,0.05)  ### (log10 g, mdval*10^-6)


glist=np.linspace(-5.0,1.0,7) 
deltalist=np.append(np.linspace(0.1,0.9,9) , [0.99] )
Sinsqthlist=np.linspace(0.1,0.45,8) 
thlist=np.array([np.arcsin(np.sqrt(ssq)) for ssq in Sinsqthlist]) ### in radians
SinsqTh12Central=0.306
SigssqTh12=0.05


with open('chi_combined_SNO_Run1.txt', 'a') as f:
     for g in glist:
        for delta in deltalist:
            for th in thlist:
                mptest=np.sqrt(dmsq/(1-delta**2) )  ## in MeV
                md=np.sqrt(dmsq*delta**2/(1-delta**2))  ## in MeV
                Gamm=g**2 * dmsq * (1 + delta**2)/ (32* np.pi *mptest)  ## in MeV       
                inp_arr=(g, md,th)
                print(g, delta,  np.sin(th)**2,   ChiSNO(inp_arr)  )
                f.write('%0.5f , %0.5f, %0.5f , %0.5f\n' %(g, delta, np.sin(th)**2 ,  ChiSNO(inp_arr)  ))
                
exit()
    
                      





#exit()



##### Visible decay plot
##mptest=1.e-6
##mdtest=1.e-7
##FluxB8Dec=np.array([SpecVis(1.e-4,0, EnuArrVal) for EnuArrVal in EnuArr ])
##FluxB8NoDec=np.array([Pee(EnuArrVal)*B8flux( EnuArrVal) for EnuArrVal in EnuArr ])
##fig,ax=plt.subplots(1,1)
##fig,ax=plt.subplots(1,1)
##ax.loglog(EnuArr, FluxB8NoDec,color='r',label='No decay')
##ax.loglog(EnuArr, FluxB8Dec, color='b',label='Decay')
##ax.set_xlabel(r'$E(MeV)$',fontsize=20)
##ax.set_ylabel(r'$\Phi_{B8} (MeV^{-1} cm^{-2} s^{-1})$',fontsize=20)
##ax.set_xlim(0.1,20)
##ax.set_ylim(0.1,1.e6)
##plt.legend(loc='lower right', fontsize=12)
##ax.tick_params('both', length=6, width=1, which='major', labelsize=15, direction='in')
##ax.tick_params('both', length=4, width=1, which='minor', direction='in')
##plt.savefig('B8_VisibleDec_g_1e-4.pdf', bbox_inches='tight')
##plt.show()
#exit()




#######Plotting the solar probability ########
###########
##plt.semilogx(EnuArr, ProbeeSol)
##plt.xlabel(r'$E(MeV)$',fontsize=20)
##plt.ylabel(r'$P_{ee}$',fontsize=20)
##plt.xlim(0.1,10)
##plt.ylim(0,1)
##plt.savefig('Pee_Solar_NoDecay.pdf', bbox_inches='tight')
##

##### Printing the B8 flux
##############
##fig,ax=plt.subplots(1,1)
##ax.loglog(EnuArr, B8flux(EnuArr),color='r',label=r'$\Phi_{B8}$')
##ax.set_xlabel(r'$E(MeV)$',fontsize=20)
##ax.set_ylabel(r'$\Phi_{B8} (MeV^{-1} cm^{-2} s^{-1})$',fontsize=20)
##ax.set_xlim(0.1,20)
##ax.set_ylim(0.1,1.e6)
##plt.legend(loc='lower right', fontsize=12)
##ax.tick_params('both', length=6, width=1, which='major', labelsize=15, direction='in')
##ax.tick_params('both', length=4, width=1, which='minor', direction='in')
##plt.savefig('B8_NoDecay.pdf', bbox_inches='tight')
#plt.show()

##### Printing the SNO flux
######
##fig,ax=plt.subplots(1,1)
##ax.loglog(EnuArr, SnoCS(EnuArr),color='r',label=r'$SNO_{\nu_e d}$')
##ax.set_xlabel(r'$E(MeV)$',fontsize=20)
##ax.set_ylabel(r'$\sigma_{CC} (cm^{2})$',fontsize=20)
##ax.set_xlim(2,40)
##plt.legend(loc='lower right', fontsize=12)
##ax.tick_params('both', length=6, width=1, which='major', labelsize=15, direction='in')
##ax.tick_params('both', length=4, width=1, which='minor', direction='in')
##plt.savefig('SNO_CC_cs.pdf', bbox_inches='tight')
#plt.show()

############ SNO Events plot
############
##histEventSno=[]
##Enhist=[]
##k1=0
##for i in range(Nbins):
##    histEventSno.append(NEventslist[i])
##    Enhist.append(Entab[i])
##    histEventSno.append(NEventslist[i])
##    Enhist.append(Entab[i+1])    
##fig, ax=plt.subplots(1,1)
##ax.plot( Enhist, histEventSno, color='r',linewidth=2.5,label=r'$\nu_{e}+d$')
##plt.xlabel(r'$T_{\rm eff}(MeV)$',fontsize=20)
##plt.ylabel(r'$N_{\nu_e}$',fontsize=20)
##plt.title("SNO, 1 year ")
##plt.legend(loc='upper right', fontsize=12)
##plt.savefig('CC_SNO_Nue.pdf', bbox_inches='tight')
###plt.show()

