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

chosen_case=0 #variable qui permet de savoir quel cas a été choisi.

steamTable = XSteam(XSteam.UNIT_SYSTEM_MKS)   # m/kg/sec/°C/bar/W

acceptable_cases={3,4,5,6,14,15,16,18,19,20,21,22,23} #cas expérimentaux acceptés


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

def heat_partitioning(p,Twall,Tbulk,vel,Dh,param):
    #param est les paramètres à calibrer. Il contient : 0: angle de contact (degrés)
    #fonction dans Appendix B de Ravik.
    angle=param[0]*3.1415/180.
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
        return 2*np.log10(2.51/(Re*np.sqrt(f))+eps/(3.7*Dh))+1./np.sqrt(f)
    fric=optimize.root_scalar(colebrook,bracket=[0.001,1],method='brentq').root
    if fric==1 or fric==0.001:
        print("friction coef aux bornes")
            
    #Gnielinski correlation
    NuD=((fric/8)*(Re-1000)*Pr)/(1+12.7*np.sqrt(fric/8)*(Pr**(2./3.)-1))
    hfc=NuD*kf/Dh

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
        print("cas 1")
    elif (N0*Npp<np.exp(1)):
        Nppb=(0.2689*N0*Npp+0.2690)/N0
        print("cas 2")
    else:
        Nppb=(np.log(N0*Npp)-np.log(np.log(N0*Npp)))/N0
        print("cas 3")

    #inception diameter and microlayer thickness

    Ca=(muf*K)/(sigma*np.sqrt(tgrowth))
    Ca0=2.16*1E-4*(DTsup**1.216)
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
    tstar=(kf**2)/(hfc*3.1415*etaf)
    print("Asl = "+str(Asl))
    print("Nppb = "+str(Nppb))
    print("tstar = "+str(tstar))
    print("freq = "+str(freq))
    
    Ssl=min(1,Asl*Nppb*tstar*freq)
    print(Ssl)
    phisc=2*hfc*Ssl*(DTsup+DTsub)
    phifc=(1-Ssl)*hfc*(DTsup+DTsub)

    #print ("\nWall heat flux :")
    return [phifc,phisc,phie]

def chaleur(p,DTsup,DTsub,vel,Dh,param):
    #fonction à appeler avant heat_partitioning. calcule la température de saturation d'abord.
    Tsat=steamTable.tsat_p(p)
    Twall=Tsat+DTsup #en degrés
    Tbulk=Tsat-DTsub
    return heat_partitioning(p,Twall,Tbulk,vel,Dh,param)


#définition du cas simulé

ParamsRavik=np.array([40,18.9E-6,0.27,0.75,-0.3,-0.26,6.1E-3,0.6317,0.1237,-0.373])

def least_squares(par,data):
    #data est sous la forme d'un dataframe. pressure, superheat, subcooling, velocity, hdiameter. La dernière case est la valeur mesurée de flux.
    ssr=0
    for x in range(data.shape[0]):
        ssr+=(chaleur(data['p'][x],data['DTsup'][x],data['DTsub'][x],data['v'][x],data['diam'][x],par)-data['Flux'][x])**2
    return ssr

kennel_data=[
    [4,15,27.8,0.3,19.56E-3,2E5],
    [4,20,27.8,0.3,19.56E-3,7E5],
]

kennel_data=pd.DataFrame(kennel_data,columns=['p','DTsup','DTsub','v','diam','Flux'])
#print(least_squares(ParamsRavik,kennel_data))

def var_param(nominal,indice,valeur):
    #renvoie une liste de tableaux où on a fait varier le paramètre numéro indice de valeur%.
    parammoins=np.array(nominal)
    paramplus=np.array(nominal)
    parammoins[indice]*=1-(float(valeur)/float(100))
    paramplus[indice]*=1+(float(valeur)/float(100))
    return [nominal,parammoins,paramplus]

#fonction pour plot plusieurs prédictions du modèle.
def plot_multiple_pred(tab_list):
    #df_list est une liste de tableaux. Chaque tableau correspond à une valeur de paramètres.
    #déjà trouver l'élement de différence entre les tableaux.
    X=np.linspace(0.2,30,200)
    Y=np.zeros(200)
    indice=0
    ecart=0
    plt.figure(figsize=[5,4])
    ax=plt.subplot(111)
    for i in np.arange(len(tab_list[0])):
        if tab_list[0][i]!=tab_list[1][i]:
            indice=i
            ecart=(1-tab_list[1][i]/tab_list[0][i])*100.
    for i in np.arange(len(tab_list)):
        for j in np.arange(200):
            Y[j]=chaleur(pressure,X[j],subcooling,velocity,hdiameter,tab_list[i])
        plt.plot(X,Y,label='Model '+str(i))
    plt.scatter('DTsup','Flux',data=kennel_data,color='red',label='Kennel data')
    plt.title('Prédictions avec variations du paramètre '+str(indice)+', écart '+str(ecart)+'%')
    ax.set_xlabel('Wall superheat [K]')
    ax.set_ylabel('Wall heat flux [W/m2]')
    ax.set_ylim([0,1.2E6])
    plt.legend()
    plt.grid()
    plt.show()

#construction db.
#tableau 5.2 de Ravik. Conditions expérimentales des expériences de Kennel.
exp_cases=[
    [3,2,3.66,27.8,19.56],
    [4,2,0.3,27.8,19.56],
    [5,2,1.22,27.8,19.56],
    [6,4,1.22,55.5,19.56],
    [14,2,1.22,27.8,19.56],
    [15,4,0.3,55.5,19.56],
    [16,4,0.3,27.8,19.56],
    [18,4,1.22,11.1,19.56],
    [19,4,3.66,11.1,19.56],
    [20,6,0.3,27.8,19.56],
    [21,6,1.22,27.8,19.56],
    [22,4,3.35,55.5,10.67],
    [23,4,1.22,55.5,10.67]
]

exp_cases=pd.DataFrame(exp_cases,columns=['case nr','p','v','DTsub','diam'])

    
P1=np.array([40,18.9E-6,0.27,0.75,-0.3,-0.26,6.1E-3,0.6317,0.1237,-0.373])
P2=np.array([20,18.9E-6,0.27,0.75,-0.3,-0.26,6.1E-3,0.6317,0.1237,-0.373])
P3=np.array([60,18.9E-6,0.27,0.75,-0.3,-0.26,6.1E-3,0.6317,0.1237,-0.373])

tab_list=[P1,P2,P3]

#plot_multiple_pred(var_param(ParamsRavik,9,50))

#on range les données expérimentales de la manière suivante : un dataframe, avec la première colonne le numéro de cas, et en deuxième colonne la liste qui comprend les pts expé.


#importation des .csv en tant que numpy array
kennel3=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel3.csv',delimiter=",")
kennel4=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel4.csv',delimiter=",")
kennel5=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel5.csv',delimiter=",")
kennel6=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel6.csv',delimiter=",")
kennel14=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel14.csv',delimiter=",")
kennel15=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel15.csv',delimiter=",")
kennel16=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel16.csv',delimiter=",")
kennel18=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel18.csv',delimiter=",")
kennel20=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel20.csv',delimiter=",")
kennel21=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel21.csv',delimiter=",")
kennel22=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel22.csv',delimiter=",")
kennel23=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel23.csv',delimiter=",")

exp_data=[
    [3,kennel3],
    [4,kennel4],
    [5,kennel5],
    [6,kennel6],
    [14,kennel14],
    [15,kennel15],
    [16,kennel16],
    [18,kennel18],
    [20,kennel20],
    [21,kennel21],
    [22,kennel22],
    [23,kennel23]
]

exp_data=pd.DataFrame(exp_data,columns=['case nr','points'])
Ravikdata=np.loadtxt('/home/nl255551/Bureau/articles_ravik/courbes_ravik/Kennel20_ravik.csv',delimiter=",")



def pred_capabilityL2(indice,param):
    #renvoie l'erreur L2 moyenne entre les prédictions faites avec le paramètre param, et les mesures faites au cas i.
    #sélection de la ligne correspondant au cas
    selected_data=exp_data[exp_data['case nr']==indice]
    selected_case=exp_cases[exp_cases['case nr']==indice]
    res=0
    for i in range(len(selected_data['points'].iloc[0])):
        res+=(chaleur(selected_case['p'].iloc[0],selected_data['points'].iloc[0][i][0],selected_case['DTsub'].iloc[0],selected_case['v'].iloc[0],selected_case['diam'].iloc[0],param)-selected_data['points'].iloc[0][i][1])**2
    return res/float(len(selected_data['points'].iloc[0]))

def pred_capability01(indice,param):
    #renvoie l'erreur 0-1 moyenne entre les prédictions faites avec le paramètre param, et les mesures faites au cas i.
    #sélection de la ligne correspondant au cas
    selected_data=exp_data[exp_data['case nr']==indice]
    selected_case=exp_cases[exp_cases['case nr']==indice]
    res=0
    for i in range(len(selected_data['points'].iloc[0])):
        res+=int(crossed(selected_data['points'].iloc[0][i],1,selected_case,param))
    return res/float(len(selected_data['points'].iloc[0]))



def print_predictions(indice,param):
    #affiche les prédictions des paramètres param sur le cas numéro indice.
    selected_data=exp_data[exp_data['case nr']==indice]
    selected_case=exp_cases[exp_cases['case nr']==indice]
    data_points=np.array(selected_data['points'].iloc[0])
    X=np.linspace(0.2,30,200)
    Yphifc=np.zeros(200)
    Yphisc=np.zeros(200)
    Yphie=np.zeros(200)
    YRavik=np.zeros(200)
    for i in np.arange(200):
        a=chaleur(selected_case['p'].iloc[0],X[i],selected_case['DTsub'].iloc[0],selected_case['v'].iloc[0],selected_case['diam'].iloc[0],param)
        print("DTSup = "+str(X[i]))
        YRavik[i]=sum(chaleur(selected_case['p'].iloc[0],X[i],selected_case['DTsub'].iloc[0],selected_case['v'].iloc[0],selected_case['diam'].iloc[0],ParamsRavik))
        Yphifc[i]=a[0]
        Yphisc[i]=a[1]
        Yphie[i]=a[2]
    ax=plt.subplot(111)
    plt.plot(X,YRavik,label='My prediction',color='black')
    plt.plot(X,Yphifc,label='Forced convection',color='tab:blue',linewidth=4)
    plt.plot(X,Yphisc,label='Sliding conduction',color='tab:green')
    plt.plot(X,Yphie,label='Evaporation',color='tab:brown')
    plt.plot(Ravikdata[:,0],Ravikdata[:,1],label='MITB prediction', color='tab:orange')
    plt.scatter(data_points[:,0],data_points[:,1],color='tab:red',label='Exp data')
    ax.set_xlabel('Wall superheat [K]')
    ax.set_ylabel('Wall heat flux [W/m2]')
    ax.set_ylim([0,data_points[-1,1]])
    plt.title('Kennel case : '+str(indice))
    plt.legend()
    plt.grid()
    plt.show()

#on se place dans le framework Scipy optimize pour optimiser. On chercher à n'optimiser que sur les 6 premiers paramètres et non sur tout. En fait c'est pas logique de chercher à optimiser disons une calibration qui se rapporte à la pression si nos données expérimentales ne varient pas en fonction de la pression. Si on calibre que sur 1 cas Kennel, il y a 1 pression, 1 vitesse, 1 subcooling. Donc on peut faire varier seulement le coef en Ja_sup, l'angle de contact et le coef multiplicateur de la corrélation du diamètre.
#coefs numéro 0,1,3.



#fonctions pour tester l'erreur 0-1
def crossed(obs,deltax,selected_case,param):
    #l'erreur est de deltax de chaque côté (donc 2deltax en tout)
    #l'argument selected_case fait écho à la fonction pred_capability, il est utilisé de la même manière.
    before=chaleur(selected_case['p'].iloc[0],obs[0]-deltax,selected_case['DTsub'].iloc[0],selected_case['v'].iloc[0],selected_case['diam'].iloc[0],param)-obs[1]
    after=chaleur(selected_case['p'].iloc[0],obs[0]+deltax,selected_case['DTsub'].iloc[0],selected_case['v'].iloc[0],selected_case['diam'].iloc[0],param)-obs[1]
    #si ils sont du même signe : return false car les lignes ne se sont pas croisées.
    return (before*after<=0)
    

caselol=int(input('case ?'))
lb=np.array([0.5*ParamsRavik[0],0.5*ParamsRavik[1],0.6])
ub=np.array([1.5*ParamsRavik[0],1.5*ParamsRavik[1],0.9])

def performanceL2(X):
    paramscourant=copy.deepcopy(ParamsRavik)
    paramscourant[0]=X[0]
    paramscourant[1]=X[1]
    paramscourant[3]=X[2]
    return pred_capabilityL2(caselol,paramscourant)

def performance01(X):
    paramscourant=copy.deepcopy(ParamsRavik)
    paramscourant[0]=X[0]
    paramscourant[1]=X[1]
    paramscourant[3]=X[2]
    return -pred_capability01(caselol,paramscourant)



X0=np.array([ParamsRavik[0],ParamsRavik[1],ParamsRavik[3]])
bounds=opt.Bounds(lb,ub,keep_feasible=True)



print_predictions(caselol,ParamsRavik)

    
