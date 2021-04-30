import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit 


data = pd.read_excel('Compoundpendulumdata.xlsx',
                     names=('pivotdistance', 't1up', 't2rev'), 
                        usecols=(0,1,2), 
                        nrows=22)
print(data)

fine_d = np.arange(0,1,0.01)
fm = 1                              #     fixed mass /kg
m = 1.4                             # moveable mass /kg
kd = 0.9939                         # distance between knife edges /m

def curve1(x, a, b, c):
    return a*(x**2) + b*x +c

popt, pcov = curve_fit(curve1, data.pivotdistance, data.t1up)
a, b, c = popt
a = popt[0]
b = popt[1]
c = popt[2]
erra = np.sqrt(float(pcov[0][0]))
errb = np.sqrt(float(pcov[1][1]))
errc = np.sqrt(float(pcov[2][2]))

data_fit1 = curve1(fine_d, a, b, c)

def curve2(x, e, f, g):
    return e*(x**2) + f*x +g

popt, pcov = curve_fit(curve2, data.pivotdistance, data.t2rev)
e, f, g = popt
e = popt[0]
f = popt[1]
g = popt[2]
erre = np.sqrt(float(pcov[0][0]))
errf = np.sqrt(float(pcov[1][1]))
errg = np.sqrt(float(pcov[2][2]))
data_fit2 = curve2(fine_d, e, f, g)

plt.figure()
fig = plt.figure(figsize=(10,8))
plt.scatter(data.pivotdistance, data.t1up, label = 'Upright', marker = 'x', color = 'r')
plt.scatter(data.pivotdistance, data.t2rev, label = 'Reversed', marker = 'x', color = 'b')
plt.plot(fine_d, data_fit1, label='fitted curve', color = 'r')
plt.plot(fine_d, data_fit2, label='fitted curve', color = 'b')
plt.ylim([96.0, 105.0])


plt.rcParams.update({'font.size': 20})

plt.xlabel('Mass distance from pivot / m')
plt.ylabel('Time for 50 oscillations / s')
plt.show
fig.savefig('RPGraph.png')
print('The equation for t1up is {:.2f} x^2+ {:.2f} x+ {:.2f} and the equation for t2rev is {:.2f} x^2+ {:.2f} + {:.2f}'.format(a,b,c,e,f,g))
