# -*- coding: utf-8 -*-
# on imite la fonction de Ravik (annexe B, page 123 de sa thèse)
# on affiche nos prédictions contre les prédictions de Ravik pour comparer. Si possible avec les données expérimentales.

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

mat_contents=sio.loadmat('kennel_set20.mat')
ravik_results=mat_contents['MITB']
ravik=pd.DataFrame(data=ravik_results,columns=['DTsup','WHF','phifc','phisc','phie','Npp','Nppb','Dd','f'])

plt.figure(figsize=[6,6])
ax=plt.subplot(111)
plt.plot(ravik['DTsup'],ravik['WHF'],label='WHF',color='orange')
plt.plot(ravik['DTsup'],ravik['phifc'],label='phifc',color='blue')
plt.plot(ravik['DTsup'],ravik['phisc'],label='phisc',color='green')
plt.plot(ravik['DTsup'],ravik['phie'],label='phie',color='brown')
#plt.plot(ravik['DTsup'],ravik['hfc'],label='hfc',color='brown')
plt.title('Ravik predictions')
#ax.set_ylim(0,4.5e6)
ax.set_xlim(0,20)
plt.legend()

plt.grid()
plt.show()
