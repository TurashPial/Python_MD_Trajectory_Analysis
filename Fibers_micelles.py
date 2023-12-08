#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import MDAnalysis as mda
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from MDAnalysis.analysis import rdf
import pandas as pd


# In[2]:


def radical_distribution(Universe, ref, sel, dimension="xyz", b=0, e=-1, com=False, relative_rho=True):
    ''' Universe: MDA.universe
        ref: reference point or axis(two points)
        sel: MDA.atomgroups
        b: starting time
        e: ending time
        dimension: "xy" for cylindrical distribution, "xyz" for spherical
        com: calculate distribution using center of mass instead of every atom
        relative_rho: if normalize with respect to bulk concentration
        '''
   
    #calculate the rdf around a point (xyz) or around an axis (xy)
    
    #first determine if com, if true, calculate the COM of each residue
    if com == True:
        ref_com = np.array()
        
    #for 2D, only support vertical Z axis (x=c1, y=c2)
    distri = []
    if dimension == "xyz":
        for ts in Universe.trajectory[b:e]:
            for atom in sel:
                distance = np.linalg.norm(atom.position - ref)
                distri.append(distance/10)
        if relative_rho == True:
            rho0 = len(sel)/(Universe.dimensions[0]*Universe.dimensions[1]*Universe.dimensions[2]/1000)
        else:
            rho0 = 1/(6.022*10**23)*10**24
        
    if dimension == "xy":
        def lineseg_dist(p, a, b):
            #def a function to calculate the distance from a point to a line
            # normalized tangent vector
            d = np.divide(b - a, np.linalg.norm(b - a))
            # signed parallel distance components
            s = np.dot(a - p, d)
            t = np.dot(p - b, d)
            # clamped parallel distance
            h = np.maximum.reduce([s, t, 0])
            # perpendicular distance component
            c = np.cross(p - a, d)
        
            return np.hypot(h, np.linalg.norm(c)) 
            
        for ts in Universe.trajectory[b:e]:
            for atom in sel:
                distance = lineseg_dist(atom.position, ref[0], ref[1])
                distri.append(distance/10)
        rho0 = len(sel)/(Universe.dimensions[0]*Universe.dimensions[1]/100)
        if relative_rho == False:
            rho0 = rho0 / (len(sel)/(Universe.dimensions[0]*Universe.dimensions[1]*Universe.dimensions[2]/1000))/(6.022*10**23)*10**24

    hist, edges = np.histogram(
    distri,
    bins=200,
    range=[0,5.5],
    density=False)
    
    # rho0 = rho* volume of box / area of the cross section of the box
    
    gr = []
    for i in range(len(hist)):
        if hist[i] == 0 or i == 0: 
            gr.append(0)
        else:
            if dimension == 'xy':
                gr.append(float(hist[i])/((2*np.pi*0.0265*i*0.0265)*rho0*len(Universe.trajectory[b:e])))
            else:
                gr.append(float(hist[i])/((4*np.pi*((0.0265*i)**2)*0.0265)*rho0*len(Universe.trajectory[b:e])))
    
    return gr, edges


# ## Analysis on Fiber

# In[3]:


os.chdir("/home/leon/Documents/Research/DMREF/all_atom/fibers/filler_P/.")

u = mda.Universe("T298.gro", "T298_center.xtc")
NA = u.select_atoms("name NA")
CL = u.select_atoms("name CL")
GLN = u.select_atoms("resname GLUP")
GLU = u.select_atoms("resname GLUe")
GLU_all = u.select_atoms("resname GLUP GLUe")



hist1, edges1 = radical_distribution(u, ref=np.array([[60,56,-10],[60,56,100]]), sel=NA, dimension='xy', b=1, e=-2, relative_rho=False)
hist2, edges2 = radical_distribution(u, ref=np.array([[60,56,-10],[60,56,100]]), sel=CL, dimension='xy', b=1, e=-2, relative_rho=False)


# In[23]:


hist1_new = savgol_filter(hist1, 100, 6)
hist2_new = savgol_filter(hist2, 100, 6)

#plt.plot(edges1[:-1]/10, hist1, label='$Na^+$')
#plt.plot(edges1[:-1]/10, hist2, label='$Cl^-$')
plt.plot(edges1[:-1]/10, hist1_new, label='$Na^+$', linewidth=2)
plt.plot(edges1[:-1]/10, hist2_new, label='$Cl^-$', linewidth=2)

plt.xlabel('Distance (nm)', size=15)
plt.ylabel('Concentration (mol/L)', size=15)
plt.tick_params(labelsize=13)
plt.legend()


# In[5]:


W = u.select_atoms("name OW")
w_gr, edges1 = radical_distribution(u, ref=np.array([[60,56,-10],[60,56,100]]), sel=W, dimension='xy', b=1, e=-2, relative_rho=False)


# In[72]:


epsilon = np.zeros(len(w_gr))
w_max = max(w_gr)
for i in range(len(w_gr)):
    epsilon[i] = (1-w_gr[i]/w_max)*2 + w_gr[i]/w_max * 80
plt.plot(edges1[:-1],epsilon)
plt.plot(edges1[:-1],w_gr)
plt.show()


# In[143]:


debye_len = []
e0 = 8.85418782*(10**-12)
kBT = 4.114 * (10**-21)
e = 1.60217663 * (10**-19) 
for i in range(len(epsilon)):
    if hist1[i] + hist2[i] == 0:
        debye_len.append(0)
    else:
        concNA = hist1[i] * 6.022*10**23 *1000
        concCL = hist2[i] * 6.022*10**23 *1000
        lambdaD = np.sqrt((epsilon[i]*e0*kBT)/((concNA+concCL)*(e**2)))
        debye_len.append(lambdaD)
for i in range(50):
    debye_len[i] = 0

debye_len = np.array(debye_len) * (10**9)

debye_new = savgol_filter(debye_len, 15, 3)
for i in range(55):
    debye_len[i] = None
    debye_new[i] = None

plt.plot(edges1[:-1],debye_len)
plt.plot(edges1[:-1],debye_new)
plt.show()


# In[209]:


GLN = u.select_atoms("resname GLUP")
GLU = u.select_atoms("resname GLUe")
GLU_all = u.select_atoms("resname GLUP GLUe")
hist3, edges3 = radical_distribution(u, ref=np.array([[60,56,-10],[60,56,100]]), sel=GLN, dimension='xy', b=1, e=-2)
hist4, edges4 = radical_distribution(u, ref=np.array([[60,56,-10],[60,56,100]]), sel=GLU, dimension='xy', b=1, e=-2)


# In[186]:


fig, ax1 = plt.subplots()
ax1.plot(edges1[:-1]/10, hist3, linewidth=2, label='$E_1$')
ax1.plot(edges1[:-1]/10, hist4, linewidth=2, label='$E_2$')
ax1.axvline(x=edgesa[68]/10, linestyle="--", color='grey', linewidth=1)
ax1.axvline(x=edgesa[85]/10, linestyle="--", color='grey', linewidth=1)
ax1.tick_params(labelsize=12)
plt.xlabel('Distance from fiber center (nm)', size=15)
plt.ylabel('Radial distribution of GLUs', size=15)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(edges1[:-1]/10, debye_new, color='k', linewidth=2, label='Debye length')
ax2.tick_params(labelsize=12)
ax2.set_ylabel('Debye length (nm)', size=15)
ax2.legend(loc='upper right')


print ('The debye length at E1 = %f nm' %(debye_len[np.argmax(hist3)]))
print ('The debye length at E2 = %f nm' %(debye_len[np.argmax(hist4)]))


# In[10]:


fig, ax3 = plt.subplots()
ax3.plot(edges1[:-1], hist1)
ax3.plot(edges1[:-1], hist2)

ax4 = ax3.twinx()
ax4.plot(edges1[:-1], debye_len, color='k')


# In[11]:


fig, ax5 = plt.subplots()
ax5.plot(edges1[:-1], epsilon)

ax4 = ax5.twinx()
ax4.plot(edges1[:-1], debye_len, color='k')


# In[25]:


# Save all data so no need to rerun 
import csv

with open('rdf_fiber.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(hist1) # Na+
    writer.writerow(hist2) # Cl-
    writer.writerow(w_gr)  # water
    writer.writerow(hist3) # E1
    writer.writerow(hist4) # E2

print (os.getcwd())


# In[6]:


GLN_O = u.select_atoms("resname GLUP and name OE1 OE2")
GLU_O = u.select_atoms("resname GLUe and name OE1 OE2")
EO_1, edges1 = radical_distribution(u, ref=np.array([[60,56,-10],[60,56,100]]), sel=GLN_O, dimension='xy', b=1, e=-2, relative_rho=False)
EO_2, edges1 = radical_distribution(u, ref=np.array([[60,56,-10],[60,56,100]]), sel=GLU_O, dimension='xy', b=1, e=-2, relative_rho=False)



# In[8]:


EO_1_new = savgol_filter(EO_1, 20, 6)
EO_2_new = savgol_filter(EO_2, 20, 6)

fig, ax1 = plt.subplots()
ax1.plot(edges1[:-1]/10, EO_1, linewidth=2, label='$E_1$')
ax1.plot(edges1[:-1]/10, EO_2, linewidth=2, label='$E_2$')
ax1.plot(edges1[:-1]/10, EO_1_new, linewidth=2, label='$E_1$')
ax1.plot(edges1[:-1]/10, EO_2_new, linewidth=2, label='$E_2$')


# In[237]:


fig, ax1 = plt.subplots()
ax1.plot(edges1[:-1]/10, EO_1_new, linewidth=2, label='$E_1$')
ax1.plot(edges1[:-1]/10, EO_2_new, linewidth=2, label='$E_2$')
ax1.axvline(x=edgesa[69]/10, linestyle="--", color='grey', linewidth=1)
ax1.axvline(x=edgesa[90]/10, linestyle="--", color='grey', linewidth=1)
ax1.tick_params(labelsize=12)
plt.xlabel('Distance from fiber center (nm)', size=15)
plt.ylabel('Radial distribution of GLUs', size=15)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(edges1[:-1]/10, debye_new, color='k', linewidth=2, label='Debye length')
ax2.tick_params(labelsize=12)
ax2.set_ylabel('Debye length (nm)', size=15)
ax2.legend(loc='upper right')


print ('The debye length at E1 = %.2f nm' %(debye_len[np.argmax(EO_1)]))
print ('The debye length at E2 = %.2f nm' %(debye_len[np.argmax(EO_2)]))


# In[42]:


fig2, axa = plt.subplots()
axa.plot(edges1[:-1]/10, EO_1_new, linewidth=2, label='$E_1$')
axa.plot(edges1[:-1]/10, EO_2_new, linewidth=2, label='$E_2$')
axa.legend(loc='upper left')

axb = axa.twinx()
axb.plot(edges1[:-1]/10, hist1_new, color='k', linewidth=2, label='Na+')
axb.plot(edges1[:-1]/10, hist2_new, color='yellow', linewidth=2, label='Cl-')
axb.vlines(x=edges1[np.argmax(hist1_new)-1]/10, ymin=0, ymax=0.4, colors='k', ls='--', lw=2)
axb.tick_params(labelsize=12)
axb.set_ylabel('Ion concentration (M)', size=15)
axb.legend(loc='upper right')

print("The highest concentration of Na+ is at %.3f nm away from fiber center \
      " % (edges1[np.argmax(hist1_new)-1]/10))


# ## Analysis on Micelle

# In[52]:


os.chdir("/home/leon/Documents/Research/DMREF/all_atom/fibers/filler_P_0.2/micelle/.")

u2 = mda.Universe("T298_center.gro", "T298_center.xtc")
NA2 = u2.select_atoms("name NA")
CL2 = u2.select_atoms("name CL")
GLN2 = u2.select_atoms("resname GLUP GLU")
GLU2 = u2.select_atoms("resname GLUe")


# In[53]:


Na_2, edgesa = radical_distribution(u2, ref=np.array([53.4, 53.4, 57.9]), sel=NA2, dimension='xyz', b=1, e=-2, relative_rho=False)
Cl_2, edgesa = radical_distribution(u2, ref=np.array([53.4, 53.4, 57.9]), sel=CL2, dimension='xyz', b=1, e=-2, relative_rho=False)


# In[54]:


plt.plot(edgesa[:-1]/10, Na_2, label='$Na^+$')
plt.plot(edgesa[:-1]/10, Cl_2, label='$Cl^-$')
plt.xlabel('Distance (nm)')
plt.ylabel('Concentration (mol/L)')
plt.legend()


# In[55]:


W2 = u2.select_atoms("name OW")
w_gr2, edgesa = radical_distribution(u2, ref=np.array([53.4, 53.4, 57.9]), sel=W2, dimension='xyz', b=1, e=-2, relative_rho=False)


# In[57]:


epsilon2 = np.zeros(len(w_gr2))
w_max2 = max(w_gr2)
for i in range(len(w_gr2)):
    epsilon2[i] = (1-w_gr2[i]/w_max2)*2 + w_gr2[i]/w_max2 * 80
plt.plot(edgesa[:-1],epsilon2)
#plt.plot(edgesa[:-1],epsilon)
plt.show()


# In[58]:


debye_len2 = []
e0 = 8.85418782*(10**-12)
kBT = 4.114 * (10**-21)
e = 1.60217663 * (10**-19) 
for i in range(len(epsilon2)):
    if Na_2[i] + Cl_2[i] == 0:
        debye_len2.append(0)
    else:
        concNA = Na_2[i] * 6.022*10**23 *1000
        concCL = Cl_2[i] * 6.022*10**23 *1000
        lambdaD = np.sqrt((epsilon2[i]*e0*kBT)/((concNA+concCL)*(e**2)))
        debye_len2.append(lambdaD)

a = np.argmax(debye_len2)
for i in range(a-3):
    debye_len2[i] = 0

debye_len2 = np.array(debye_len2) * (10**9)

debye_new2 = savgol_filter(debye_len2, 10, 3)
for i in range(a):
    debye_len2[i] = None
    debye_new2[i] = None
   
plt.plot(edgesa[:-1],debye_len2)
plt.plot(edgesa[:-1],debye_new2)
plt.show()


# In[59]:


E1_2, edgesa = radical_distribution(u2, ref=np.array([53.4, 53.4, 57.9]), sel=GLN2, dimension='xyz', b=1, e=-2)
E2_2, edgesa = radical_distribution(u2, ref=np.array([53.4, 53.4, 57.9]), sel=GLU2, dimension='xyz', b=1, e=-2)


# In[60]:


fig, ax1 = plt.subplots()
ax1.plot(edgesa[:-1], E1_2)
ax1.plot(edgesa[:-1], E2_2)
ax1.axvline(x=edgesa[84], linestyle="--", color='grey', linewidth=1)
ax1.axvline(x=edgesa[94], linestyle="--", color='grey', linewidth=1)

ax2 = ax1.twinx()
ax2.plot(edgesa[:-1], debye_len2)

print ('The debye length at E1 = %f nm' %(debye_len2[84]* 10**9))
print ('The debye length at E2 = %f nm' %(debye_len2[np.argmax(E2_2)]* 10**9))


# In[61]:


Egr_1 = u2.select_atoms("resname GLUP GLU and name OE1 OE2")
Egr_2 = u2.select_atoms("resname GLUe and name OE1 OE2")
EO1_2, edgesa = radical_distribution(u2, ref=np.array([53.4, 53.4, 57.9]), sel=Egr_1, dimension='xyz', b=1, e=-2)
EO2_2, edgesa = radical_distribution(u2, ref=np.array([53.4, 53.4, 57.9]), sel=Egr_2, dimension='xyz', b=1, e=-2)


# In[62]:


EO1_2_smooth = savgol_filter(EO1_2, 20, 6)
EO2_2_smooth = savgol_filter(EO2_2, 20, 6)


# In[71]:


fig, ax1 = plt.subplots()
ax1.plot(edgesa[:-1]/10, EO1_2_smooth, linewidth=2, label='$E_1$')
ax1.plot(edgesa[:-1]/10, EO2_2_smooth, linewidth=2, label='$E_2$')
ax1.axvline(x=edgesa[88]/10, linestyle="--", color='grey', linewidth=1)
ax1.axvline(x=edgesa[100]/10, linestyle="--", color='grey', linewidth=1)
ax1.tick_params(labelsize=12)
plt.xlabel('Distance from micelle center (nm)', size=15)
plt.ylabel('Radial distribution of GLUs', size=15)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(edgesa[:-1]/10, debye_new2, linewidth=2, label='Debye length', color='k')
ax2.tick_params(labelsize=12)
ax2.set_ylabel('Debye length (nm)', size=15)
ax2.legend(loc='upper right')

print ('The debye length at E1 = %.2f nm' %(debye_len2[np.argmax(EO1_2)]))
print ('The debye length at E2 = %.2f nm' %(debye_len2[np.argmax(EO2_2)]))


# In[64]:


# Save data again

with open('rdf_micelle.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ')
    writer.writerow(Na_2) # Na+
    writer.writerow(Cl_2) # Cl-
    writer.writerow(w_gr2)  # water
    writer.writerow(E1_2) # E1
    writer.writerow(E2_2) # E2

print (os.getcwd())


# In[65]:


Na2_smooth = savgol_filter(Na_2, 40, 6)
Cl2_smooth = savgol_filter(Cl_2, 40, 6)


# In[67]:


plt.plot(edges1[:-5]/10, (np.array(hist1_new)+np.array(hist2_new))[:-4], label='$Fiber$', linewidth=2)
#plt.plot(edges1[:-1]/10, hist2, label='$Cl^-$')
plt.plot(edges1[:-5]/10, (np.array(Na2_smooth)+np.array(Cl2_smooth))[:-4], label='$Micelle$', linewidth=2)
#plt.plot(edges1[:-1]/10, Cl_2, label='$Cl^-_2$')
plt.xlabel('Distance from fiber center (nm)', size=15)
plt.ylabel('Ion Concentration (mol/L)', size=15)
plt.legend()
plt.tick_params(labelsize=12)
plt.show()


# In[68]:


print(max((np.array(hist1_new)+np.array(hist2_new))))
print(max((np.array(Na2_smooth)+np.array(Cl2_smooth))))


# In[69]:


x1,y1,y2 = np.loadtxt('/home/leon/Documents/Research/DMREF/all_atom/fibers/filler_P_0.2/sasa.xvg', comments=['#','@'], unpack=True)
x2,y3,y4 = np.loadtxt('/home/leon/Documents/Research/DMREF/all_atom/fibers/filler_P/sasa.xvg', comments=['#','@'], unpack=True)


# In[70]:


plt.plot(x1[200:]/1000,y3[200:]/144, label='Fiber', linewidth=2)
plt.plot(x1[200:]/1000,y1[200:]/144, label='Micelle', linewidth=2)
plt.legend()
plt.xlabel('Time (ns)',size=15)
plt.ylabel('Average Surface Area $(nm^2)$',size=15)
plt.tick_params(labelsize=12)
plt.show()


# # Analysis of C16

# In[41]:


os.chdir("/home/leon/Documents/Research/DMREF/all_atom/fibers/filler_C16")

u3 = mda.Universe("T298_center.gro", "T298_center.xtc")
NA3 = u3.select_atoms("name NA")
CL3 = u3.select_atoms("name CL and resname CL")
GLN3 = u3.select_atoms("resname GLUP GLU")
GLU3 = u3.select_atoms("resname GLUe")


# In[42]:


Na_3, edgesa = radical_distribution(u3, ref=np.array([[52, 52, -10],[52, 52, 100]]), sel=NA3, dimension='xy', b=1, e=-2, relative_rho=False)
Cl_3, edgesa = radical_distribution(u3, ref=np.array([[52, 52, -10],[52, 52, 100]]), sel=CL3, dimension='xy', b=1, e=-2, relative_rho=False)


# In[43]:


plt.plot(edgesa[:-1]/10, Na_3, label='$Na^+$')
plt.plot(edgesa[:-1]/10, Cl_3, label='$Cl^-$')
plt.xlabel('Distance (nm)')
plt.ylabel('Concentration (mol/L)')
plt.legend()


# In[44]:


os.chdir("/home/leon/Documents/Research/DMREF/all_atom/fibers/filler_C16_P")

u4 = mda.Universe("T298_center.gro", "T298_center.xtc")
NA4 = u4.select_atoms("name NA")
CL4 = u4.select_atoms("name CL and resname CL")
GLN4 = u4.select_atoms("resname GLUP GLU")
GLU4 = u4.select_atoms("resname GLUe")


# In[45]:


Na_4, edges4 = radical_distribution(u4, ref=np.array([[68, 68, -10],[68, 68, 100]]), sel=NA4, dimension='xy', b=1, e=-2, relative_rho=False)
Cl_4, edges4 = radical_distribution(u4, ref=np.array([[68, 68, -10],[68, 68, 100]]), sel=CL4, dimension='xy', b=1, e=-2, relative_rho=False)


# In[46]:


plt.plot(edges4[:-1]/10, Na_4, label='$Na^+$')
plt.plot(edges4[:-1]/10, Cl_4, label='$Cl^-$')
plt.xlabel('Distance (nm)')
plt.ylabel('Concentration (mol/L)')
plt.legend()


# In[59]:


Ion3 = savgol_filter(np.add(Na_3,Cl_3), 50, 4)
Ion4 = savgol_filter(np.add(Na_4,Cl_4), 50, 4)


# In[63]:


plt.plot(edges4[:-1]/10, Ion3, label='$C_{16}VVE^-E^-$',linewidth=2)
plt.plot(edges4[:-1]/10, Ion4, label='$C_{16}VVEE^-$',linewidth=2, color='r')
plt.xlabel('Distance (nm)', size=15)
plt.ylabel('Concentration (mol/L)', size=15)
plt.legend()
plt.tick_params(labelsize=12)
plt.show()


# In[7]:


os.chdir("/home/leon/Documents/Research/DMREF/all_atom/fibers/filler_C16_P/.")

df1 = pd.read_csv('z1.csv')
time = df1['Time (ps)']/1000
df1['Time (ns)'] = time
df1 = df1.drop(columns=['Time (ps)'])

df2 = pd.read_csv('z.csv')
time = df2['Time (ps)']/40000 + 5
df2['Time (ns)'] = time
df2 = df2.drop(columns=['Time (ps)'])
z_df = pd.concat([df1,df2],axis=0)

charge_density = 144/z_df[' Box-Z']
z_df['Linear Charge Density'] = charge_density
z_df


# In[8]:





# In[10]:


os.chdir("/home/leon/Documents/Research/DMREF/all_atom/fibers/filler_C16/.")

df1 = pd.read_csv('z1.csv')
time = df1['Time (ps)']/1000
df1['Time (ns)'] = time
df1 = df1.drop(columns=['Time (ps)'])

df2 = pd.read_csv('z.csv')
time = df2['Time (ps)']/40000 + 5
df2['Time (ns)'] = time
df2 = df2.drop(columns=['Time (ps)'])
z_df_2 = pd.concat([df1,df2],axis=0)

charge_density = 288/z_df_2[' Box-Z']
z_df_2['Linear Charge Density'] = charge_density
z_df_2


# In[45]:


plt.plot(z_df['Time (ns)'][0:4502], z_df['Linear Charge Density'][0:4502],label='$C_{16}-VVE^0E^{-1}$',linewidth=2,\
        color = 'red')
plt.plot(z_df_2['Time (ns)'], z_df_2['Linear Charge Density'],label='$C_{16}-VVE^{-1}E^{-1}$',linewidth=2, \
         color = 'royalblue')

plt.xlabel('Time (ns)', size=15)
plt.xticks([0,2.5,5,7.5,10],[0, 2.5, 5, 105, 205])
plt.ylabel('Linear Charge Density ($e$/nm)',size=15)
plt.legend(frameon=True)
plt.show()


# In[ ]:




