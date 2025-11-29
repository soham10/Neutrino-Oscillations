#Decay vs No Decay analysis: SK with theta12 varying and getting marginalised over
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
th12=33.56*(np.pi)/180.0
SigTh12=0.088
th13=8.46*(np.pi)/180.0
th23=41.6*(np.pi)/180.0
dmsq=7.5 * 10**(-5)* 10**(-12) ## in MeV^2


#####################
### Other parameters
GF=1.16*10**(-11)  ## in MeV^2
Rsun=6957000  ## in km
NA=6*10**(23)
me=0.511 #MeV
sinsqthW=0.23
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


############ Differential decay probabilities: Operator: A PL + B PR ################

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
      E3**2) - SigTh12
      (A*B*m2**5*m3)/(8*E2**4) + (A*B*m2**5*m3**3)/(32*E2**3*E3**3) + (A*B* # type: ignore
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

##def PeeInvDec(f , En):  #f in eV^2, and En in MeV , hence MeVtoeV**2
##    return ((CosthMsq( 0 , En ) * np.cos(th12)**2 + np.exp(-  dSun * kmtoMeV * f / ( En * MeVtoeV**2) ) *SinthMsq( 0 , En ) * np.sin(th12)**2 ) * np.cos(th13)**4)
##



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
B8data=np.array(np.loadtxt('lambda.csv', delimiter=','))
B8flux=interp1d( 1.e-6* B8data[::,0] , 1.e6* B8data[::,-1], kind='linear' ) # (MeV^{-1} cm^{-2} s^{-1})



###########
## SK Cross section ###########
def dSigmadTeNUe_SK(Te, Enu):
    ep= sinsqthW   #g2
    em=0.5+ sinsqthW  #g1
    return (2/np.pi)*me*GF**2* ( em**2 + ep**2 * (1- Te/Enu)**2 - em* ep* me* Te/ (Enu)**2)*MeVsqtocmsq  ## in cm^2

def dSigmadTeNUmu_SK(Te, Enu):
    g1=-0.5  + sinsqthW
    g2= sinsqthW
    return (2/np.pi)*me*GF**2* ( g1**2 + g2**2 * (1- Te/Enu)**2 - g1* g2* me* Te/ (Enu)**2)*MeVsqtocmsq  ## in cm^2    

Emax=15.0 #MeV
#Emin=8

Temin=6.5
DelEnbin=0.5
Temax= 2*Emax**2/(me+2*Emax)
Nbins=int((Temax-Temin)/DelEnbin)
#Entab=np.linspace(Emin,Emax,Nbins+1)


Telist=np.linspace(Temin,Temax, num=Nbins)


NtarSK=2*1.5*10**33
DeltaT=1.4*3.15*1.e7  ## 504 days = 1.4yr to s

def sigma(Teff,Etrue):
##    if Teff<6.5:
##        return  0.354/np.sqrt(Etrue)
##    else:
##        return 0.0644 + 0.403/np.sqrt(Etrue)
    return -0.084 + 0.376* np.sqrt(Etrue) + 0.040*Etrue 

def Pvisible(A, B, Ed, mdtest, thtest12):
    xsq= (mptest/mdtest)**2
    Eup=xsq * Ed
    EmaxB8=1.e-6* B8data[len(B8data)-1,0]
    if Eup>EmaxB8:
        Eup=EmaxB8
    Edlist = np.linspace(Ed, Eup , num=10)
    P1visIntegrand= np.array([ SinthMsqtest( 0 , Ep ,thtest12, dmsq) * WHCmm(A, B, Ep, Ed, mptest, mdtest) *
                               ( 1 -   np.exp(-  dSun * kmtoMeV * GammaTot(A, B, Ep,  mptest, mdtest)  ) ) * B8flux(Ep) for Ep in Edlist])
    return(simps(P1visIntegrand, Edlist))

def ResSK(Teff,Etrue): 
    return np.sqrt(1.0/(2*np.pi)) *  (1.0/sigma(Teff,Etrue)) *np.exp(-0.5*((Teff - Etrue)/(sigma(Teff,Etrue)))**2 )
    #return 1.0

def SpecVisNue(A, B, Ed, mdtest, thtest12):
    return (np.cos(th13)**4   * np.cos(thtest12)**2 * Pvisible(A, B, Ed,mdtest,thtest12) +(CosthMsqtest( 0 , Ed ,thtest12, dmsq)* np.cos(thtest12)**2 +np.exp(-dSun * kmtoMeV * GammaTot(A, B, Ed,  mptest, mdtest) ) *SinthMsqtest( 0 , Ed ,thtest12, dmsq) * np.sin(thtest12)**2 ) * np.cos(th13)**4 * B8flux(Ed) )


def SpecVisNux(A, B, Ed, mdtest, thtest12):
    return (np.cos(th13)**4   * np.sin(thtest12)**2 * Pvisible(A, B, Ed, mdtest, thtest12) + (CosthMsqtest( 0 , Ed ,thtest12, dmsq)* np.sin(thtest12)**2+ np.exp(-dSun * kmtoMeV * GammaTot(A, B, Ed,  mptest, mdtest) ) *SinthMsqtest( 0 , Ed ,thtest12, dmsq) * np.cos(thtest12)**2 ) * np.cos(th13)**4 * B8flux(Ed) )


def EnuMin(Te):
    return (Te/2) * (1. + np.sqrt(1.+2.*me/Te))


def NdetSK(g, mdtest, thtest12):   
    integrandTe=[]
    for Te in Telist:
        Ealpha_min=min(EnuMin(Te),Emax)
        Enulist=np.linspace(Ealpha_min, Emax, num=10)
        integrandEnu=np.array([(NtarSK*DeltaT*( dSigmadTeNUe_SK(Te, Enu)*  SpecVisNue(g, 0, Enu, mdtest, thtest12)+
                        dSigmadTeNUmu_SK(Te, Enu)* SpecVisNux(g, 0, Enu, mdtest, thtest12)))   for Enu in Enulist])
        integrandTe.append(np.trapz(integrandEnu,Enulist) + 1.e-10)  ## to compensate for 0 events
    #print(integrandTe)
    return integrandTe

##def NEventsSK(g, Ebinarr):
##    NEventsarr=np.array([NdetSK(g, TErec) for TErec in Ebinarr])
##    return np.trapz(NEventsarr,Ebinarr)

#print(NdetSK(5))
#exit()


### Central SK data
NEventslist=[1196.730650292426, 1017.1358241542039, 846.0593185017079, 686.6049823027431, 541.4876559364409, 412.8502848240704, 302.2761645177103, 210.57189545636152, 137.73729697973886, 83.02230482454634, 44.807901478750104, 20.69777598579962, 7.621340100743071, 2.0317169360530034, 0.35023731292330224, 1.e-10]
#print(NEventslist)






def  ChiSK(inp):
    # Test data for SK
    Testg=10**(inp[0])
    Testmd= inp[1]# in MeV
    Testth12= inp[2] ## in radians
    NEventslisttest= NdetSK(Testg,Testmd, Testth12)
    chisq= np.array([((n1-n2)**2)/(1.0*n2) + (np.sin(Testth12)-SinsqTh12Central)**2/ (1.0*SigssqTh12**2)   for n1, n2 in zip(NEventslist, NEventslisttest)])
    return sum(chisq)



#mptest=1.e-6
#glist=np.concatenate((np.linspace(-6,-5.1,5) , np.linspace(-5.0,-4,40) )) 
#glist=np.linspace(-5.0,-4.0,40) 
#mdlist=np.linspace(0.1,0.95,40) 
#thlist=np.concatenate((np.linspace(1,25,4) , np.linspace(26,76,45), np.linspace(77,89,4)))


glist=np.linspace(-5.0,1.0,7) 
deltalist=np.append(np.linspace(0.1,0.9,9) , [0.99] )
Sinsqthlist=np.linspace(0.1,0.45,8) 
thlist=np.array([np.arcsin(np.sqrt(ssq)) for ssq in Sinsqthlist]) ### in radians
SinsqTh12Central=0.306
SigssqTh12=0.05

## DO THIS FOR A SERIES OF Sin^2 theta12

#delta=0.1
#mptest=np.sqrt(dmsq/(1-delta**2) )  ## in MeV
#md=np.sqrt(dmsq*delta**2/(1-delta**2)) 
#print(NdetSK(10**-5, md ,np.arcsin(np.sqrt(0.306)) ))
#exit()

with open('chi_combined_SK_Run1.txt', 'a') as f:
    for g in glist:
        for delta in deltalist:
            for th in thlist:        		
                mptest=np.sqrt(dmsq/(1-delta**2) )  ## in MeV
                md=np.sqrt(dmsq*delta**2/(1-delta**2))  ## in MeV
                Gamm=g**2 * dmsq * (1 + delta**2)/ (32* np.pi *mptest)  ## in MeV      
                inp_arr=(g, md, th)
                print(g, delta,  np.sin(th)**2,  ChiSK(inp_arr))
                f.write('%0.5f , %0.5f, %0.5f , %0.5f\n' %(g, delta, np.sin(th)**2 , ChiSK(inp_arr) ))

exit()





















































### SK Events data
#mptest=1.e-6
#mdtest=0.1*1.e-6
#gval0=1.e-10
#gval1= 1.e-4 
#gval2= 1.e-5
NEventslistNoDec=NdetSK(gval0)
NEventslistDec1=NdetSK(gval1)
#NEventslistDec2=[]
#print(Temin)
for nbin in range(Nbins):
    #print(nbin)
    TEbinlist=np.linspace(Temin+nbin*DelEnbin, Temin+(nbin+1)*DelEnbin , num=5)
    #NEventslistNoDec.append(NEventsSK(gval0, TEbinlist)/(504*0.5))  ### to reproduce per day per 0.5 MeV bin?? Is this true? 
    #NEventslistDec1.append(NEventsSK(gval1, TEbinlist)/(504*0.5))
    #NEventslistDec2.append(NEventsSK(gval2, TEbinlist)/(504*0.5))

    
#print( NEventslist )
#exit()
############ SK Events plot
############
histEventSKNoDec=[]
histEventSKDec1=[]
#histEventSKDec2=[]
Enhist=[]
k1=0
for i in range(Nbins-1):
    histEventSKNoDec.append(NEventslistNoDec[i]/(504*0.5))
    histEventSKDec1.append(NEventslistDec1[i]/(504*0.5))
    #histEventSKDec2.append(NEventslistDec2[i])
    Enhist.append(Telist[i])
    histEventSKNoDec.append(NEventslistNoDec[i]/(504*0.5))
    histEventSKDec1.append(NEventslistDec1[i]/(504*0.5))
    #histEventSKDec2.append(NEventslistDec2[i])
    Enhist.append(Telist[i+1])
fig, ax=plt.subplots(1,1)
ax.plot( Enhist, histEventSKNoDec, color='r',linewidth=2.5,label=r'$\nu_{\alpha}+e. g=0$')
#ax.plot( Enhist, histEventSKDec2, color='g',linewidth=2.5,label=r'$g=10^{-5}$')
ax.plot( Enhist, histEventSKDec1, color='b',linewidth=2.5,label=r'$g=10^{-4}$')
ax.set_yscale('log')
ax.set_xlim(7,15)
plt.xlabel(r'$T_{e}(MeV)$',fontsize=20)
plt.ylabel(r'$N_{\nu_e}/{\rm day/0.5MeV}$',fontsize=20)
plt.title("SK,decay, --, 504 days, md=0.1mp ")
plt.legend(loc='upper right', fontsize=12)
#plt.savefig('SK_Nue_VisDecay_md01mp.pdf', bbox_inches='tight')
plt.show()    

exit()

    
#print(NEventslist)


exit()





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

