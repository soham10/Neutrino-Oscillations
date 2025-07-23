"""
Program to read in polarization modulation curve data, plot the data,
fit a constant value, and fit and plot a modulation curve of the form
A + B*cos^2(phi-phi0).

Data are assumed to be in csv format.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# calculate weighted average
def wavg(value, err):
  a = np.sum(value/err)
  b =  np.sum(1/err)
  return (a/b)

# define function used to fit modulation curve
def func_cos2t(phi, a, b, phi0):
  # print('In func_cos2t a, b, phi0 = ', a, b, phi0)
  return a + b*np.cos((phi-phi0)*np.pi/180)**2


# read in the modulation curve data
(angle, rate, rate_err) = np.loadtxt('input.txt', delimiter=',', unpack=True)
print('Read in ', len(angle), ' data points.')

# plot the modulation curve
plt.ion() # interactive plotting
plt.figure('Rotation curve') # make a plot window
plt.clf() # clear the plot window
# plot the data with errorbars
plt.errorbar(angle, rate, rate_err, fmt='o')
plt.ylim(ymin=0) # set the bottom of the y-axis to zero
plt.title('MXS Rotation curve')
plt.xlabel('Rotation angle (degrees)')
plt.ylabel('Count rate (events/second)')
plt.show()

# fit with a constant
print('Fit data with a constant:')
avg = wavg(rate, rate_err)
# find the chi-squared
chisq0 = sum(((rate-avg)/rate_err)**2)
print( 'Average (c/s) = %.3f' % (avg))
dof0 = len(angle)-1  # Degrees of Freedom
print('Chisq/Dof = %.1f/%d' % (chisq0, dof0))
print()

# fit to a + b*cos^2(phi-phi0)
print('Fit data with A + B*cos^2(phi-phi0) :')
#pinit = [avg, 0.0, 0.0] # an ok initial guess
pinit = [1.0, 1.0, 1.0] # the default initial guess
#pinit = [0.0, -0.05, -30.0] # a confusing initial guess
# use the scipy curve_fit routine using Levenberg-Marquardt algorithm to find the best fit
popt, pcov = curve_fit(func_cos2t, angle, rate, p0=pinit, sigma=rate_err, method='lm')
# translate fit results to user friendly variable
a, a_err = popt[0], np.sqrt(pcov[0,0])
b, b_err = popt[1], np.sqrt(pcov[1,1])
phi0, phi0_err = popt[2], np.sqrt(pcov[2,2])
# print the results
print('A = ', a, ' +/- ', a_err)
print('B = ', b, ' +/- ', b_err)
print('phi_0 = ', phi0, ' +/- ', phi0_err)
print('Modulation = ', b/(2*a+b))
# find the chi-squared
chisq = sum(((rate-func_cos2t(angle, a, b, phi0))/rate_err)**2)
dof = len(angle)-3
print('Chisq/Dof = ', chisq, '/', dof)
# plot the fit
pang = np.linspace(0.0, 180.0, 180)
plt.plot(pang, func_cos2t(pang, a, b, phi0), 'b--')


#find errors on parameters B and phi0
# find chi-squared over a grid in modulation and B (keep A fixed)
bt = b+np.linspace(-0.1*b, +0.1*b, 100) # trial values for B
phi0t = phi0+np.linspace(-5, +5, 100) # trial values for phi0
dchisq = np.zeros((len(bt), len(phi0t))) # array to hold delta chi-squared
for j in range(len(bt)): # loop over B values
  for i in range(len(phi0t)): # loop over phi0 values
    # find chi-squared at a, trial b, trial phi0
    dchisq[i,j] = sum(((rate-func_cos2t(angle, a, bt[j], phi0t[i]))/rate_err)**2) - chisq

plt.figure('Fitted parameter ranges')
plt.clf()
# make the contour plot and save the contour information in cs
# 1-sigma, 2-sigma, 3-sigma for two parameters of interest
cs = plt.contour(bt, phi0t, dchisq, levels=[2.30,6.17,11.8]) 
plt.clabel(cs, inline=1, fontsize=10) # label the contours
plt.plot(b, phi0, '+r') # plot the best fitted point
plt.xlabel('B') # label the axes
plt.ylabel('phi_0 (degree)')
plt.pause(10)

dchi2 = 2.30 # use 1-sigma errors for two parameters of interest
bl, bu = max(bt), min(bt) # lower and upper bounds on b
for i in range(len(bt)): # loop over trial values of b
  t = dchisq[i,:] # pick out slice with this trial value of b
  if sum(t <= dchi2) > 0: # check if any points have low enough delta chi-squared
    bl = min([bl, min(bt[np.where(t <= dchi2)])]) # update lower bound
    bu = max([bu, max(bt[np.where(t <= dchi2)])]) # update upper bound
#print(bl, b, bu) # for debugging, nicer printing later
phi0l, phi0u = max(phi0t), min(phi0t) # lower and upper bounds on phi0
for j in range(len(phi0t)): # loop over trial values of phi0
  t = dchisq[:,j] # pick out slice with this trial value of phi0
  if sum(t <= dchi2) > 0: # check if any points have low enough delta chi-squared
    phi0l = min([phi0l, min(phi0t[np.where(t <= dchi2)])]) # update lower bound on phi0
    phi0u = max([phi0u, max(phi0t[np.where(t <= dchi2)])]) # update upper bound on phi0
#print(phi0l, phi0, phi0u)

# print fit results with errors
#print 'Average (c/s) = %.2f +/- %.2f' % (avg, avgerr)
print('B (c/s) = %.4f +%.4f -%.4f' % (b, bu-b, b-bl))
print('phi_0 (deg) = %.1f +%.1f -%.1f' % (phi0, phi0u-phi0, phi0-phi0l))
print('Chisq/Dof = %.1f/%d' % (chisq, dof))

# if they seem to disagree, increase number of points in grid
plt.plot([bl,bl], [min(phi0t),max(phi0t)], 'r:')
plt.plot([bu,bu], [min(phi0t),max(phi0t)], 'r:')
plt.plot([min(bt),max(bt)], [phi0l,phi0l], 'r:')
plt.plot([min(bt),max(bt)], [phi0u,phi0u], 'r:')
