#script pour nettoyer les fichiers de donnees profil_z
import numpy as np
import pandas as pd

df=pd.DataFrame(data=np.loadtxt('profil_z_3484.dat'),columns=['distance','X','Y','Z','diameter_field2','temp_field1','temp_field2','velocity_field1','velocity_field2','alpha','mass_trans_field1','XD_field2','BK','COAL','COMP','NUCL','W1','MT','SMBR','ALPB','CONDNUM','GAMA','GAMANUC','epsilon'])

df=df[['X','alpha','diameter_field2','velocity_field1','velocity_field2']]
#écriture
df.to_csv('clean_profile.dat',index=False,sep=" ")
