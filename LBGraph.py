import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit 

data = pd.read_excel('Balldata.xlsx',
                     names=('distance', 'time'), 
                        usecols=(0,1), 
                        nrows=9)
print(data)

Mass = 0.0636 #kg
cd = 0.5 
p = 1.2 #density of air kg/m^3
A = np.pi*(0.02496/2)**2 #cross sectional area

k = cd*p*A*1/2 # drag coefficiant

fine_t = np.arange(0.25,0.65,0.01)

def line(t, g):
    return ((Mass*0.5*g)/k)*(1-(np.exp((-k*t)/Mass)))*t

popt, pcov = curve_fit(line, data.time , data.distance)
g = popt [0]

errg = np.sqrt(float(pcov[0][0]))


data_fit = line(fine_t, g)


fig = plt.figure(figsize=(10,8))

plt.scatter(data.time, data.distance,)
plt.plot(fine_t, data_fit)
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.show()
fig.savefig('LBGraph.png')