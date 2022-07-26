# -*- coding: utf-8 -*-
# on imite la fonction de Ravik (annexe B, page 123 de sa thèse)
# on affiche nos prédictions contre les prédictions de Ravik pour comparer. Si possible avec les données expérimentales.

import numpy as np
from scipy import optimize
from pyXSteam.XSteam import XSteam
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd
import copy

chosen_case=0 #variable globale qui permet de savoir quel cas a été choisi.

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)   # m/kg/sec/°C/bar/W

ParamsRavik=np.array([40,18.9E-6,0.27,0.75,-0.3,-0.26,6.1E-3,0.6317,0.1237,-0.373])
acceptable_cases={3,4,5,6,14,15,16,18,19,20,21,22,23} #cas expérimentaux acceptés

#hardcoded
exp_conditions=[
    [3,2,3.66,27.8,19.56],
    [4,2,0.3,27.8,19.56],
    [5,2,1.22,27.8,19.56],
    [6,4,1.22,55.5,19.56],   
    [8,4,1.22,83.3,19.56], 
    #[9,4,1.22,83.3,19.56], pas confiance car dans la thèse de Kennel ce n'est pas cohérent.
    [14,2,1.22,27.8,19.56],
    [15,4,0.3,55.5,19.56],
    [16,4,0.3,27.8,19.56],
    [18,4,1.22,11.1,19.56],
    [19,4,3.66,11.1,19.56],
    [20,6,0.3,27.8,19.56],
    [21,6,1.22,27.8,19.56],
    [22,4,3.35,55.5,10.67],
    [23,4,1.22,55.5,10.67],
    [24,4,11.16,55.5,10.67],
]

exp_conditions=pd.DataFrame(exp_conditions,columns=['case nr','p','v','DTsub','diam'])


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
    if((p*(np.exp(hfg*(Tg-Tsat)/(462*Tg*Tsat))-1))<1E-3):
        print("p :" +str(p)+" hfg : " +str(hfg)+" tg :" +str(Tg)+ " Tsat :" +str(Tsat))
    return (4.72E5)*(1-np.exp(-(theta**2)/(8*(0.722)**2)))*(np.exp(frhoplus*(2.5E-6)/Rc)-1)

def heat_partitioning(p,Twall,Tbulk,vel,Dh,param):
    #param est les paramètres à calibrer. Il contient : 0: angle de contact (degrés)
    #fonction dans Appendix B de Ravik.
    angle=param[0]*3.1415/180. #conversion en radians
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
    etaf=kf/(rhof*cpf)

    #Darcy friction factor. Résolution itérative de l'équation de Colebrook
    #eps est la rugosité de la conduite. 
    eps=0 #0 d'après le mail de Ravik
    def colebrook(f):
        return 2*np.log10(2.51/(Re*np.sqrt(f)))+1./np.sqrt(f)
    fric=optimize.root_scalar(colebrook,bracket=[0.001,1],method='brentq').root
    if fric==1 or fric==0.001:
        print("friction coef aux bornes")
            
    #Gnielinski correlation
    NuD=((fric/8)*(Re-1000)*Pr)/(1+12.7*np.sqrt(fric/8)*(Pr**(2./3.)-1))
    hfc=NuD*kf/Dh
#    hfc=9273 pour cas 6

    #boiling closures eq 2.2
    Dd=param[1]*((rhof-rhog)/rhog)**(param[2])*(Jasup**(param[3]))*((1+Jasub)**(param[4]))*vel**(param[5])
    
    twait=(param[6]*Jasub**(param[7]))/DTsup
    chi=max(0,0.05*DTsub/DTsup)
    
    #on répète le code dans le mail de Ravik
    c1=1.243/np.sqrt(Pr)
    c2=1.954*chi
    c3=-1*min(abs(c2),0.5*c1)
    K=(c1+c3)*Jasup*np.sqrt(etaf)
    tgrowth=(0.25*Dd/K)**2
    #attention dans l'annexe B freq et fric sont notés par la même lettre f
    freq=1./(twait+tgrowth)
    #calcul de N0 non marqué dans l'annexe, cf. texte eq. 4.30
    N0=freq*tgrowth*3.1415*((0.5*Dd)**2)
    Npp=HibikiIshii(rhof,rhog,hfg,p,DTsup,Tsat,angle,sigma)

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


    ###tester juste si angle est toujours en radian.
    rappD=max(param[8]*Ca**(param[9])*np.sin(angle),1)
    
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
    twait=0.0061*Jasub**(0.63)/(DTsup)
    tstar=(kf**2)/((hfc**2)*3.1415*etaf)
    tstar=min(tstar,twait)
    
    Ssl=min(1,Asl*Nppb*tstar*freq)
    phisc=2*hfc*Ssl*(DTsup+DTsub)
    phifc=(1-Ssl)*hfc*(DTsup+DTsub)

    #print ("\nWall heat flux :")
    return phifc+phisc+phie
    return [phifc,phisc,phie]
    return hfc


    return Nppb
    return Npp
    return freq
    return Dd


def chaleur(p,DTsup,DTsub,vel,Dh,param):
    #fonction à appeler avant heat_partitioning. calcule la température de saturation d'abord.
    Tsat=steamTable.tsat_p(p)
    Twall=Tsat+DTsup #en degrés
    Tbulk=Tsat-DTsub
    Dh=Dh/1000. #conversion mm en m
    return heat_partitioning(p,Twall,Tbulk,vel,Dh,param)

def initialize_case(case):
    #à appeler en premier.
    #met la variable globale "chosen_case" à la valeur voulue. Test pour savoir si elle existe.
    global chosen_case
    if case in acceptable_cases:
        chosen_case=case
        print('case number'+str(case)+' loaded')
    else:
        print("Erreur : le cas souhaité n'est pas disponible")
    return 0    

def exp_datab(i):
    #renvoie les données expérimentales du cas i.
    casestr=str(i)
    kennel=np.loadtxt('/home/catB/nl255551/Documents/Code/Ravik/courbes_ravik/Kennel'+casestr+'.csv',delimiter=",")
    return kennel.tolist()

def exp_case(chosencase):
    #renvoie les conditions expérimentales du cas chosen_case. Si le cas choisi n'appartient pas à ce qui est demandé, ça plantera.
    return (exp_conditions.loc[exp_conditions['case nr']==chosencase]).to_numpy()
    
def run_model(DTsup,param,chosencase):
    #on récupère les conditions expérimentales prescrites
    exp_cond=pd.DataFrame(data=exp_case(chosencase),columns=['case nr','p','v','DTsub','diam'])
    #print(exp_cond)
    #print("paramètres :")
    #print(param)
    #fait tourner le modèle pour les conditions expérimentales prescrites.
    return chaleur(exp_cond['p'].iloc[0],DTsup,exp_cond['DTsub'].iloc[0],exp_cond['v'].iloc[0],exp_cond['diam'].iloc[0],param)

def plot_model(chosencase):
    X=np.linspace(0.001,20,100)
    Y1=np.empty([100])
    Y2=np.empty([100])
    Y3=np.empty([100])
    params=ParamsRavik
    for i in np.arange(100):
        Y1[i]=run_model(X[i],ParamsRavik,chosencase)
        Y2[i]=run_model(X[i],ParamsRavik,chosencase)
        Y3[i]=run_model(X[i],ParamsRavik,chosencase)

    plt.figure(figsize=[6,6])
    ax=plt.subplot(111)
    plt.plot(X,Y1,label='whf',color='orange')

    plt.plot(X,Y1,label='phifc',color='blue')
    plt.plot(X,Y2,label='phisc',color='green')
    plt.plot(X,Y3,label='phiie',color='brown')
    
    ax.set_xlim(0,20)
    plt.title('my predictions')
    plt.legend()
    plt.grid()
    plt.show()

#chosen_case=20 #variable globale qui permet de savoir quel cas a été choisi.
#plot_model(16)


##########
####################################
# initialize_case(caselol)         #
# print(caselol)                   #
# print(chosen_case)               #
# print(run_model(14,ParamsRavik,16)) #
####################################



    
