#script pour nettoyer les fichiers de données
import numpy as np
import pandas as pd

df=pd.DataFrame(data=np.loadtxt('A6.dat'),columns=['numero','position','pression','temp','vitesse_entree','puisance de chauffe','flux','titre de sortie','nb de bulles traitees','freq interception','t vapeur moyen','alpha','vitessemesuree','ai','diamSauter','Dg','densité de centres'])

df=df[['position','alpha','diamSauter','vitessemesuree']]
df['position']*=-1
#suppression des 4 dernières observations
df=df[:-4]
#inversion des lignes pour que ça soit plus joli
df=df.iloc[::-1]
#écriture
df.to_csv('clean_exp.dat',index=False,sep=" ")
