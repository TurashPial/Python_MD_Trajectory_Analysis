#!/usr/bin/env python
# coding: utf-8

import scipy.io as sio
import numpy.matlib as nm
import numpy as np
import matplotlib.pyplot as plt



cmp = np.arange(25).reshape(25, 1)
tth = np.linspace(1, 49, 49)
cmp



pss_pre = np.loadtxt("pss_pre.txt", dtype='f')
print(pss_pre)


pss = np.loadtxt("pss.txt", dtype='f')
print(pss)



pss_pret=pss_pre.T;
psst=pss.T;
print(pss_pret.shape)
print(psst.shape)


plt.subplot(121)
plt.plot(tth.T,pss_pre);
plt.subplot(122)
plt.plot(tth.T,pss);
plt.subplots_adjust( right=2)



p_all=np.concatenate((paat,psst),axis=0)
print(p_all.shape)



from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances

d1 = pairwise_distances(p_all, metric='manhattan')
d2 = pairwise_distances(p_all, metric='euclidean')
d3 = pairwise_distances(p_all, metric='cosine')

# L1, Manhattan
#plt.subplot(131)
#plt.imshow(d1)
#plt.title('Manhatton');

# Euclidean
#plt.subplot(132)
#plt.imshow(d2)
#plt.title('Euclidean')

# And Cosine
#plt.subplot(133)
plt.imshow(d3)
plt.colorbar
plt.show()

#plt.title('Cosine')

#plt.tight_layout()
#plt.subplots_adjust( right=2)

plt.savefig('pss_simi.png', dpi=300)

paa_pre = np.loadtxt("paa_pre.txt", dtype='f')
print(paa_pre)


paa = np.loadtxt("paa.txt", dtype='f')
print(paa)



paa_pret=paa.T;
paat=paa.T;
print(paa_pret.shape)
print(paat.shape)


plt.subplot(121)
plt.plot(tth.T,paa_pre);
plt.subplot(122)
plt.plot(tth.T,paa);
plt.subplots_adjust( right=2)



p_all=np.concatenate((paat,psst),axis=0)
print(p_all.shape)



from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances

d1 = pairwise_distances(p_all, metric='manhattan')
d2 = pairwise_distances(p_all, metric='euclidean')
d3 = pairwise_distances(p_all, metric='cosine')

# L1, Manhattan
#plt.subplot(131)
#plt.imshow(d1)
#plt.title('Manhatton');

# Euclidean
#plt.subplot(132)
#plt.imshow(d2)
#plt.title('Euclidean')

# And Cosine
#plt.subplot(133)
plt.imshow(d3)
plt.colorbar
plt.show()

#plt.title('Cosine')

#plt.tight_layout()
#plt.subplots_adjust( right=2)

plt.savefig('paa_simi.png', dpi=300)


