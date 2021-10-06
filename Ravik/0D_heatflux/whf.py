# -*- coding: utf-8 -*-
# on imite la fonction de Ravik (annexe B, page 123 de sa thèse)

import numpy as np
from scipy import optimize
from pyXSteam.XSteam import XSteam
import matplotlib.pyplot as plt

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)   # m/kg/sec/°C/bar/W  

def HibikiIshii(rhof,rhog,hfg,p,DTsup,Tsat,theta,sigma):
    #corrélation Hibiki
    p=p*1E5 #conversion de bar en Pa
    #theta est en radians
    Tg=Tsat+DTsup #Température du gaz approchée par la température wall (Tsat+DTsup)
    #températures en kelvin
    Tg=Tg+273.15
    Tsat=Tsat+273.15
    rhoplus=np.log10((rhof-rhog)/rhog)
    frhoplus=-0.01064+0.48246*rhoplus-0.22712*rhoplus**2+0.05468*rhoplus**3
    Rc=(2*sigma*(1+(rhog/rhof)))/(p*(np.exp(hfg*(Tg-Tsat)/(462*Tg*Tsat))-1))
    return (4.72E5)*(1-np.exp(-(theta**2)/(8*(0.722)**2)))*(np.exp(frhoplus*(2.5E-6)/Rc)-1)

def heat_partitioning(p,Twall,Tbulk,vel,Dh,theta):
    #fonction dans Appendix B de Ravik.
    #calcul des propriétés fluides avec pyXsteam 
    #Tsat
    Tsat=steamTable.tsat_p(p)
    #rhof = masse vol. fluide
    rhof=steamTable.rho_pt(p,Tbulk)
    #muf = visc dyn. fluide
    muf=steamTable.my_pt(p,Tbulk) 
    #rhog = masse vol. gaz à saturation
    rhog=steamTable.rhoV_p(p)   
    #cpf = capacité massique isobare liquide à bulk temperature
    cpf=steamTable.Cp_pt(p,Tbulk)*1000
    #kf = conductivité thermique liquide à bulk temperature
    kf=steamTable.tc_pt(p,Tbulk)
    #hfg = enthalpie de changement d'état fluide - gaz
    hfg=(steamTable.hV_p(p)-steamTable.hL_p(p))*1000
    #sigma : tension de surface fluide-gaz
    sigma=steamTable.st_p(p)

    DTsub=Tsat-Tbulk
    DTsup=Twall-Tsat
    Re=rhof*vel*Dh/muf
    Pr=muf*cpf/kf
    Jasub=rhof*cpf*DTsub/(rhog*hfg)
    Jasup=rhof*cpf*DTsup/(rhog*hfg)
    print(Jasup**0.9)
    etaf=kf/(rhof*cpf)

    #Darcy friction factor. Résolution itérative de l'équation de Colebrook
    #eps est la rugosité de la conduite. 
    eps=0 #0 d'après le mail de Ravik
    def colebrook(f):
        return 2*np.log10(2.51/(Re*np.sqrt(f))+eps/(3.7*Dh))+1./np.sqrt(f)
    fric=optimize.root_scalar(colebrook,bracket=[0.001,1],method='brentq').root
    if fric==1 or fric==0.001:
        print("friction coef aux bornes")
            
    #Gnielinski correlation
    NuD=((fric/8)*(Re-1000)*Pr)/(1+12.7*np.sqrt(fric/8)*(Pr**(2./3.)-1))
    hfc=NuD*kf/Dh

    #boiling closures eq 2.2
    Dd=18.9*1E-6*((rhof-rhog)/rhog)**(0.27)*(Jasup**(0.75))*((1+Jasub)**(-0.3))*vel**(-0.26)
    twait=(0.0061*Jasub**(0.6317))/DTsup
    chi=max(0,0.05*DTsub/DTsup)
    
    #on répète le code dans le mail de Ravik
    c1=1.243/np.sqrt(Pr)
    c2=1.954*chi
    c2=-1*min(abs(c2),0.5*c1)
    K=(c1+c2)*Jasup*np.sqrt(etaf)
    tgrowth=(0.25*Dd/K)**2
    #attention dans l'annexe B freq et fric sont notés par la même lettre f
    freq=1./(twait+tgrowth)
    #calcul de N0 non marqué dans l'annexe, cf. texte eq. 4.30
    N0=freq*tgrowth*3.1415*((0.5*Dd)**2)
    Npp=HibikiIshii(rhof,rhog,hfg,p,DTsup,Tsat,theta,sigma)

    #Static interaction of nucleation sites
    if(N0*Npp<np.exp(-1)):
        Nppb=Npp
    elif (N0*Npp<np.exp(1)):
        Nppb=(0.2689*N0*Npp+0.2690)/N0
    else:
        Nppb=(np.log(N0*Npp)-np.log(np.log(N0*Npp)))/N0

    #inception diameter and microlayer thickness

    Ca=(muf*K)/(sigma*np.sqrt(tgrowth))
    Ca0=2.16*1E-4*(DTsup**1.216)
    rappD=max(0.1237*Ca**(-.373)*np.sin(theta),1)
    
    Dinception=rappD*Dd
    Dml=Dinception/2.
    deltaml=4E-6*np.sqrt(Ca/Ca0)
    
    #compute heat flux partitions
    
    phiml=rhof*hfg*freq*Nppb*(deltaml*(Dml**2)*(3.1415/12.)*(2-(rappD**2+rappD)))
    phiinception=1.33*3.1415*(Dinception/2.)**(3)*rhog*hfg*freq*Nppb
    phie=phiml+phiinception
    #pour Dlo : voir premier paragraphe section 2.3
    Dlo=1.2*Dd
    Asl=(Dlo+Dd)/(2*np.sqrt(Nppb))
    tstar=(kf**2)/(hfc*3.1415*etaf)

    Ssl=min(1,Asl*Nppb*tstar*freq)
    phisc=2*hfc*Ssl*(DTsup+DTsub)
    phifc=(1-Ssl)*hfc*(DTsup+DTsub)

    #print ("\nWall heat flux :")
    return phifc+phisc+phie

def chaleur(p,DTsup,DTsub,vel,Dh,theta):
    #fonction initiale à appeler. calcule la température de saturation d'abord.
    Tsat=steamTable.tsat_p(p)
    Twall=Tsat+DTsup #en degrés
    Tbulk=Tsat-DTsub
    return heat_partitioning(p,Twall,Tbulk,vel,Dh,theta)



#définition du cas simulé

pressure=4 #bar
velocity=1.22 #m/s
subcooling=55.5 #K
hdiameter=1.956E-2 #m
contactangledegres=40 #degrés
contactanglerad=contactangledegres*3.1415/180. #rad


print("-------------")
print("pressure : "+str(pressure))
print("velocity : "+str(velocity))
print("Subcooling : "+str(subcooling))
print("HDiameter : "+str(hdiameter))
print("Contact angle : "+str(contactangledegres))


#appel à la fonction et plot

X=np.linspace(0.2,30,200)
Y=np.zeros(200)
for i in np.arange(200):
    Y[i]=chaleur(pressure,X[i],subcooling,velocity,hdiameter,contactanglerad)

plt.figure(figsize=[5,4])
ax=plt.subplot(111)
plt.plot(X,Y)
ax.set_xlabel('Wall superheat [K]')
ax.set_ylabel('Wall heat flux [W/m2]')
ax.set_ylim([0,3.5E6])
plt.grid()
plt.show()

print('ok')
