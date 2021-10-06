// Tests pour l'adaptative DoE. Dans ce fichier, on utilise tout le framework réalité/modèle.
//Code utilisé pour approcher une fonction (my_function) par un processsus gaussien dans le framework OLM. On utilise l'algorithme Bayesian Optimisation pour trouver les points d'acquisition. On teste différentes fonctions d'acquisition : Expected Improvement, et mon critère.


#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <random>
#include <functional>
#include <set>
#include "densities.h"

using namespace std;
using namespace Eigen;
int neval=0;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);
const int dimx=1;
const int dimtheta=1;
const int dimgp=dimx+dimtheta;
VectorXd hpars_noedm(3);


VectorXd randpert(int const n){
  //renvoie une permutation aléatoire de {0,1,...,n-1}.. Mélange de Fisher-Yates.
  VectorXd result(n);
  for (int i=0;i<n;i++){
    result(i)=i;
  }
  for (int i=n-1;i>0;i--){
    int j=int(floor(distU(generator)*(i+1)));
    double a=result(i);
    result(i)=result(j);
    result(j)=a;
  }
  return result;
}

vector<VectorXd> InitGridRect(VectorXd const &lb_t,VectorXd const &ub_t, int const n){
  // Construction du grid initial de thetas. On le fait une bonne fois pour toutes. Il faut construire au préalable les vecteurs lb_t et ub_t car on ne peut pas les déclarer de manière globale.
  // n correspond au nombre de points total.
  int dim_theta=1;
  int npoints=n*dim_theta;
  vector<VectorXd> grid;
  VectorXd theta_courant(1);
  VectorXd ind_courant(1);
  for(int i=0;i<npoints;i++){
    theta_courant(0)=lb_t(0)+(i+0.5)*(ub_t(0)-lb_t(0))/double(n);
    grid.push_back(theta_courant);
  }
  return grid;  
}

vector<VectorXd> InitGridUnif(VectorXd const &lb_t,VectorXd const &ub_t, int const n){
  // Construction du grid initial de thetas. On le fait une bonne fois pour toutes. Il faut construire au préalable les vecteurs lb_t et ub_t car on ne peut pas les déclarer de manière globale.
  // n correspond au nombre de points par dimension.
  int dim_theta=lb_t.size();
  int npoints=n*dim_theta;
  vector<VectorXd> grid;
  VectorXd theta_courant(dim_theta);
  VectorXd ind_courant(dim_theta);
  for(int i=0;i<npoints;i++){
    for (int j=0;j<dim_theta;j++){
      theta_courant(j)=lb_t(j)+distU(generator)*(ub_t(j)-lb_t(j));
    }
    grid.push_back(theta_courant);
  }
  return grid;  
}

vector<VectorXd> InitGridLHS(VectorXd const &lb_t,VectorXd const &ub_t, int const npoints){
  // Construction du grid initial de thetas. On le fait une bonne fois pour toutes. Il faut construire au préalable les vecteurs lb_t et ub_t car on ne peut pas les déclarer de manière globale.
  //   // n correspond au nombre de points TOTAL
  int dim_theta=lb_t.size();
  // division de chaque dimension en npoints : on génère dim_theta permutations de {0,npoints-1}.
  std::vector<VectorXd> perm(dim_theta);
  for (int i=0;i<dim_theta;i++){
    perm[i]=randpert(npoints);
  }
  // calcul des coordonnées de chaque point par un LHS.
  vector<VectorXd> grid;
  VectorXd theta_courant(dim_theta);
  for(int i=0;i<npoints;i++){
    for (int j=0;j<dim_theta;j++){
      theta_courant(j)=lb_t(j)+(ub_t(j)-lb_t(j))*(perm[j](i)+distU(generator))/double(npoints);
    }
    grid.push_back(theta_courant);    
  }
  return grid;
}

double truth_function(VectorXd const &x){
  //fonction vérité
  return x(0);
};

double my_model(VectorXd const &x, VectorXd const &theta){
  //modèle
  return x(0)*sin(2.0*theta(0)*x(0))+(x(0)+0.15)*(1-theta(0));
};

/* Evaluate Kernel of the Stochastic process for the two points x and y, given the parameters in par:
	- par(0) is the variance,
	- par(1) is the correlation length
*/
double Kernel_GP(Eigen::VectorXd const &x, Eigen::VectorXd const &y, const Eigen::VectorXd &par){
  // lcorx / lcort / sobs / sx / st
  double sqdistx=(x-y).head(dimx).squaredNorm();
  double sqdistt=(x-y).tail(dimtheta).squaredNorm();
  double kx=pow(par(3),2)*exp(-0.5*sqdistx/par(0));
  double kt=pow(par(4),2)*exp(-0.5*sqdistt/par(1));
	return kx*kt; /* squared exponential kernel */ //on multiplie les deux kernels
}

double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &xp, Eigen::VectorXd const &hpar){
  //Fonction Kernel sans bruit. hpar(0) = sig_edm, hpar(1) = sig_exp, hpar(2) = lcor
  double d=abs(x(0)-xp(0));
  return pow(hpar(0),2)*(1+((2.24*d)/hpar(2))+1.66*pow(d/hpar(2),2))*exp(-(2.24*d)/hpar(2)); /*Matern 5/2*/
  return pow(hpar(0),2)*exp(-(d)/hpar(2)); /*Matern 1/2*/
  return pow(hpar(0),2)*(1+(1.732*d)/hpar(2))*exp(-(1.732*d)/hpar(2)); /*Matern 3/2*/ // marche bien
  return hpar(0)*hpar(0)*exp(-pow((x(0)-xp(0))/hpar(2),2)*.5); /* squared exponential kernel */
  
  return hpar(0)*hpar(0)*exp(-pow((x(0)-xp(0))/hpar(2),2)*.5); /* squared exponential kernel */
};

double PriorMean(VectorXd const &x, VectorXd const &hpars){
  return 0;
}

double logprior_hpars(VectorXd const &hpars){
  return 0;//-2*(log(hpars(1)))-2*(log(hpars(0)));
}

double logprior_pars(VectorXd const &pars){
 return 0;
}

double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
	/* This is the function you optimize for defining the GP */
	GP* proc = (GP*) data;											//Pointer to the GP
	Eigen::VectorXd p(x.size());									//Parameters to be optimized
	for(int i=0; i<(int) x.size(); i++) p(i) = x[i];				//Setting the proposed value of the parameters
	double value = proc->SetGP(p);									//Evaluate the function
	if (!grad.empty()) {											//Cannot compute gradient : stop!
		std::cout << "Asking for gradient, I stop !" << std::endl; exit(1);
	}
	neval++;														//increment the number of evaluation count
	return value;
};


double MaxValGP(GP &gp){
  //renvoie la valeur maximale d'un GP observée jusqu'à présent (utile pour l'expected improvement)
  double max_courant=gp.FDat(0);
  for (int i=1;i<gp.NDat();i++){
    if(gp.FDat(i)>max_courant){
      max_courant=gp.FDat(i);
    }
  }
  return max_courant;
}
double EI(GP &gp, VectorXd &xprime){
  //calcule le critère d'expected improvement au point x.
  double fmax(MaxValGP(gp));
  VectorXd Pred=gp.Eval(xprime); //prédiction du gp. case 1 : moyenne, case 2 : variance.
  double Z=(Pred(0)-fmax)/sqrt(Pred(1));
  return (Pred(0)-fmax)*0.5*(1+erf(Z/sqrt(2)))+sqrt(Pred(1))*exp(-0.5*pow(Z,2));
}

double mycriteria(GP &gp, VectorXd &xprime){
  //calcule mon critère.
  VectorXd Pred=gp.Eval(xprime); //prédiction du gp. case 1 : moyenne, case 2 : variance.
  double m(1./Pred(1));
  Eigen::LDLT<Eigen::MatrixXd> ldlt=gp.GetLDLT();
  //calcule le vecteur k à xprime
  VectorXd kp(gp.NDat());
  for (int i=0;i<kp.size();i++){kp(i)=gp.Kij(gp.XDat(i),xprime);}
  VectorXd alpha=ldlt.solve(kp); //=K^-1*kp
  //calcul de l'intégrale. On créé un grid de thetastar.
  VectorXd lb_t(1);
  VectorXd ub_t(1);
  lb_t(0)=0;
  ub_t(0)=10;
  const int gridsize=40; //20 points pour calculer l'intégrale
  vector<VectorXd> Grid=InitGridRect(lb_t,ub_t,gridsize);
  double critere(0);
  double weight(0);
  for (const auto& xstar :Grid){
    VectorXd ks(gp.NDat());
    for (int i=0;i<ks.size();i++){ks(i)=gp.Kij(gp.XDat(i),xstar);}
    double g(gp.EvalMean(xstar));
    //double g(my_function(xstar)); si on veut la réponse
    critere+=pow(gp.Kij(xprime,xstar)-ks.transpose()*alpha,2)*g;
    weight+=g;
  }
  critere*=m/weight;
  return critere; 
}
double test_gp(GP &gp,VectorXd lb_t, VectorXd ub_t, VectorXd lb_x, VectorXd ub_x){
  //calcul de la norme L2 du GP par rapport au modèle, pour vérifier sa validité.
  int dx=100;
  int dt=100;
  vector<VectorXd> gridX=InitGridUnif(lb_x,ub_x,dx);
  vector<VectorXd> gridT=InitGridUnif(lb_t,ub_t,dt);
  double L2=0;
  for(int i=0;i<dx;i++){
    for(int j=0;j<dt;j++){
      VectorXd X=gridX[i];
      VectorXd T=gridT[j];
      VectorXd I(dimgp);
      I.head(dimx)=X;
      I.tail(dimtheta)=T;
      L2+=pow(gp.EvalMean(I)-my_model(X,T),2);
    }
  }
  L2/=(dx*dt);
  return L2;
}

double EI_prob(GP &gp, VectorXd thetaprime, VectorXd hpars_typical, MCMC_opti &mcmc, vector<DATA> &yobs, default_random_engine &generator, double max_courant_logpost, double & buffer_ss){
  //calcule l'EI de la log-probabilité d'amélioration, au point thetaprime.
  //on créée les données (obs-gp(theta)) qu'on met dans un vector <DATA> de taille nobs
  int dimx=yobs[0].GetX().size(); int dimtheta=thetaprime.size(); int dimgp=dimx+dimtheta;
  vector<DATA> data2;
  int nsimus=500; //on demande 20 tirages du GP pour être convergés en proba.
  //stockage des Xobs pour pouvoir y sampler le gp
  vector<VectorXd> vI(yobs.size());
  for (int i=0;i<yobs.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=yobs[i].GetX();
    I.tail(dimtheta)=thetaprime;
    vI[i]=I;
  }
  //on calcule les hpars optimaux à la prédiction moyenne et on les garde pour tous les samples
  vector<DATA> datamoy;
  for (int i=0;i<yobs.size();i++){
    VectorXd I(dimgp);
    I=vI[i];
    double Pred=gp.EvalMean(I);
    DATA dat; dat.SetX(vI[i].head(dimx)); dat.SetValue(yobs[i].Value()-Pred);
    datamoy.push_back(dat);  
  }
  VectorXd hparsopti=mcmc.Opti_hpars(datamoy,hpars_typical);
  //calcul de la ll aux params moyens pour mettre dans le buffer.
  buffer_ss=mcmc.loglikelihood(&datamoy,hparsopti);
  MatrixXd Samples=gp.SampleGP(vI,nsimus,generator); //Samples de taille (vI.size(),nsimus)
  double moy_crit=0;
  for (int ie=0;ie<nsimus;ie++){
    //construction des données
    for (int i=0;i<yobs.size();i++){
      //considérons juste la préd moyenne.
      DATA dat; dat.SetX(vI[i].head(dimx)); dat.SetValue(yobs[i].Value()-Samples(i,ie));
      data2.push_back(dat);  
    }
    //optimisation des hpars avec données quelconques.
    //calcul du critère SS
    double SS = mcmc.loglikelihood(&data2,hparsopti);
    data2.clear();
    if(SS>=max_courant_logpost){moy_crit+=SS-max_courant_logpost;}
  }
  moy_crit/=nsimus;
  //cout << "moy_crit : " << moy_crit << endl;

  //cout << "theta : " << thetaprime.transpose() << ", SS : " << SS << endl;
  //on a calculé SS. Maintenant il faut calculer sa prob. d'improvement.
  return moy_crit;
}

double EI_prob_noedm(GP &gp, VectorXd thetaprime, VectorXd hpars_typical, MCMC_opti &mcmc, vector<DATA> &yobs, default_random_engine &generator, double max_courant_logpost,double & buffer_ss){
  //calcule l'EI de la log-probabilité d'amélioration, au point thetaprime.
  //on créée les données (obs-gp(theta)) qu'on met dans un vector <DATA> de taille nobs
  int dimx=yobs[0].GetX().size(); int dimtheta=thetaprime.size(); int dimgp=dimx+dimtheta;
  vector<DATA> data2;
  int nsimus=500; //on demande 20 tirages du GP pour être convergés en proba.
  //stockage des Xobs pour pouvoir y sampler le gp
  vector<VectorXd> vI(yobs.size());
  for (int i=0;i<yobs.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=yobs[i].GetX();
    I.tail(dimtheta)=thetaprime;
    vI[i]=I;
  }
  vector<DATA> datamoy;
  for (int i=0;i<yobs.size();i++){
    VectorXd I(dimgp);
    I=vI[i];
    double Pred=gp.EvalMean(I);
    DATA dat; dat.SetX(vI[i].head(dimx)); dat.SetValue(yobs[i].Value()-Pred);
    datamoy.push_back(dat);  
  }
  buffer_ss=mcmc.loglikelihood(&datamoy,hpars_noedm);
  MatrixXd Samples=gp.SampleGP(vI,nsimus,generator); //Samples de taille (vI.size(),nsimus)
  double moy_crit=0;
  for (int ie=0;ie<nsimus;ie++){
    //construction des données
    for (int i=0;i<yobs.size();i++){
      DATA dat; dat.SetX(vI[i].head(dimx)); dat.SetValue(yobs[i].Value()-Samples(i,ie));
      data2.push_back(dat);  
    }
    //optimisation des hpars avec données quelconques.
    //calcul du critère SS
    double SS = mcmc.loglikelihood(&data2,hpars_noedm);
    data2.clear();
    //cout << "SS : " << SS <<",mcl : " << max_courant_logpost<< endl;
    if(SS>=max_courant_logpost){moy_crit+=SS-max_courant_logpost;}
  }
  moy_crit/=nsimus;
  //cout << "moy_crit : " << moy_crit << endl; 
  //cout << "theta : " << thetaprime.transpose() << ", SS : " << SS << endl;
  //on a calculé SS. Maintenant il faut calculer sa prob. d'improvement.
  return moy_crit;
}

double optfunc__ei(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /*Renvoie l'EI*/
  GP* gp2 = static_cast<GP*>(data); // cast du null pointer en type désiré
  VectorXd X(1);
  X(0)=x[0];
  return EI(*gp2,X);
}

double optfunc__crit(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /*Renvoie l'EI*/
  GP* gp2 = static_cast<GP*>(data); // cast du null pointer en type désiré
  VectorXd X(1);
  X(0)=x[0];
  return mycriteria(*gp2,X);
}

void printGP(GP &gp, const char* file_name,VectorXd lb_x, VectorXd ub_x){
  //affiche le GP dans un fichier
  int ndim=1;
  FILE* out = fopen(file_name,"w");
  fprintf(out,"#Observation du pg. ndim premieres colonnes : coordonnes, avant-derniere colonne : moyenne pg calibre, derniere colonne : 1 sd du pg calibre \n");
  for (unsigned is = 0; is < 500; is++)
    {
      VectorXd x(ndim);
      for (unsigned id = 0; id < ndim; id++){
	x(id) =lb_x(0)+(ub_x(0)-lb_x(0))*double(is)/500;
	fprintf(out,"%e ",x(id));
      }
      VectorXd eval = gp.Eval(x);
      fprintf(out,"%e %e %e %e %e\n",eval(0),eval(0)+sqrt(eval(1)),eval(0)-sqrt(eval(1)),EI(gp,x),mycriteria(gp,x));
    }
  fclose(out);
}

VectorXd f_theta(VectorXd const &theta, vector<DATA> const &yobs){
  //renvoie le modèle évalué au paramètre theta, sur tous les x. On ne s'embête pas avec le choix d'un x (type Damblin).
  VectorXd ftheta(yobs.size());
  for (int i=0;i<ftheta.size();i++){
    ftheta(i)=my_model(yobs[i].GetX(),theta);
  }
  return ftheta;
}

const double Big = -1.e16;


int main( int argc, char **argv){
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distU(0,1);
  std::normal_distribution<double> distN(0,1);



  // Inputs:
  int npts_init = 3; //nombre de theta pour la préévaluation du gp
  const int ndim = 1;
  const int nobs=10;

  //Bornes de x
  VectorXd lb_x(dimx); lb_x(0)=0;
  VectorXd ub_x(dimx); ub_x(0)=1;
  //bornes de theta
  VectorXd lb_t(dimtheta); lb_t(0)=-0.5;
  VectorXd ub_t(dimtheta); ub_t(0)=1.5;
  //bornes des hpars DE Z
  const int dim_hparsz=3;
  std::vector<double> lb_hpars(dim_hparsz); lb_hpars[0]=0.005;lb_hpars[1]=0.005;lb_hpars[2]=0.1; //-5 avant
  std::vector<double> ub_hpars(dim_hparsz); ub_hpars[0]=2;ub_hpars[1]=2;ub_hpars[2]=2; //bornes sur l'optimisation des hpars. edm exp lcor
  VectorXd hpars_typical(dim_hparsz); hpars_typical(0)=0.5;hpars_typical(1)=0.5;hpars_typical(2)=1;
  //hpars représentatifs de no edm
  hpars_noedm(0)=0; hpars_noedm(1)=1;hpars_noedm(2)=0.1;

  //bornes des hpars DE F
  const int np = 5; //nombre de hpars pour le GPOLM. 
	MatrixXd Bounds(2,np);
	Bounds(0,0) = 0.1; Bounds(1,0) = 0.5; //lcor x
	Bounds(0,1) = 0.1; Bounds(1,1) = 0.5; //lcor theta
	Bounds(0,2) = 1e-8; Bounds(1,2) = 1e-7; //sigma exp (très faible)
  Bounds(0,3) = 0.05; Bounds(1,3) = 1; //sigma x
  Bounds(0,4) = 0.05; Bounds(1,4) = 1; //sigma theta
  VectorXd hpars_typical_f(np); for (int i=0;i<np;i++){hpars_typical_f(i)=Bounds(0,i);}

  //génération des observations
  vector<DATA> yobs;
  VectorXd yobsXd(nobs); //on stocke aussi la valeur des observations dans un vectorXd, pratique.
  for (int i=0;i<nobs;i++){
    DATA dat;
    VectorXd x(1);
    x=lb_x+(ub_x-lb_x)*distU(generator);
    double f=truth_function(x);
    dat.SetX(x); dat.SetValue(f);
    yobs.push_back(dat);
    yobsXd(i)=f;
  }

  //création d'une instance de densité car c'est plus pratique
  DoE doe_null(lb_t,ub_t,2);// gridsize peu importe
  doe_null.Fill(my_model,&yobs);
  Density MainDensity(doe_null);
  MainDensity.SetModel(my_model);
  MainDensity.SetKernel(Kernel);
  MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(PriorMean);
  MCMC test(MainDensity,1);
  MCMC_opti mcmc(test,1);

  //Construction du DoE initial, un LHS.
  vector<VectorXd> gridLHSINIT=InitGridLHS(lb_t,ub_t,npts_init);

  //Réalisation du modèle sur le LHS initial.

  FILE* out = fopen("results/observations.gnu","w");
  fprintf(out,"#Fichier du LHS initial. ndim premieres colonnes : coordonnes, derniere colonne : réalisation du modèle\n");
  vector<DATA> data_GP;
  double ll_max1=-1E300;
  double ll_max2=ll_max1;
  for(const auto& theta:gridLHSINIT){
    for(int j=0;j<nobs;j++){
      VectorXd X=yobs[j].GetX();
      VectorXd I(dimgp);
      I.head(dimx)=X;
      I.tail(dimtheta)=theta;
      for(unsigned id=0; id<dimgp; id++){fprintf(out,"%e ",I(id));
      DATA dat; dat.SetX(I); dat.SetValue(my_model(I.head(dimx),I.tail(dimtheta)));
      fprintf(out,"%e \n",dat.Value());
      data_GP.push_back(dat);
    }
  }
  fclose(out);
  //pour chaque theta, on calcule la loglikelihood. on sauvegarde la meilleure pour servir de référence.
  for (auto theta:gridLHSINIT){
    VectorXd hpars_opti=mcmc.Opti_hpars(theta,hpars_typical);
    double l=mcmc.loglikelihood_theta(&yobs,hpars_opti,theta);
    double l2=mcmc.loglikelihood_theta(&yobs,hpars_noedm,theta);
    cout << "l :" << l << endl;
    cout << "l2 : " << l2 << endl;
    if(l>ll_max1){ll_max1=l;}
    if(l2>ll_max2){ll_max2=l2;}
  }

  const int nlhs=80; //taille du LHS sur lequel on cherche les points theta

  //tirage de tous les points possibles d'évaluation du modèle
  vector<DATA> data_GP1=data_GP;
  vector<DATA> data_GP2=data_GP;
  vector<VectorXd> lhs=InitGridLHS(lb_t,ub_t,nlhs); // 200 évaluations possibles
  vector<VectorXd> lhs1=lhs; //va servir à l'ei avec edm
  vector<VectorXd> lhs2=lhs; //va servir à l'ei sans edm
  
  GP gp1(Kernel_GP);
  gp1.SetData(data_GP1);
  VectorXd hpars_gp=hpars_typical_f;
  gp1.SetGP(hpars_gp); //attention on vit dans l'espace x x theta.
  gp1.OptimizeGP(myoptfunc_gp, &Bounds, &hpars_gp, np);

  //on fera 10 calculs supplémentaires.
  vector<VectorXd> visited_theta1;
  int nobs_max=10;
  double buffer_ss=0;
  //nettoyage du fichier L2.gnu
  out=fopen("results/L2.gnu","w");
  fclose(out);
  
  for (int i=0;i<nobs_max;i++){
    //boucle sur les observations possibles pour savoir où le critère est le plus élevé
    VectorXd theta_candidate(dimtheta);
    double crit_max=-1E300;
    string filename="results/step"+to_string(i)+".gnu";
    out=fopen(filename.c_str(),"w");
    for (const auto &theta:lhs1){
      double e=EI_prob(gp1,theta,hpars_typical,mcmc,yobs,generator,ll_max1,buffer_ss);
      fprintf(out,"%e %e %e\n",theta(0),e,buffer_ss);
      //cout << e << endl;
      if(e>crit_max){
        theta_candidate=theta;
        crit_max=e;
      }
    }
    fclose(out);
    cout << "espérance d'amélioration : "<< crit_max << endl;
    VectorXd hpars_opti=mcmc.Opti_hpars(theta_candidate,hpars_typical);
    cout << "hpars optimaux : " << hpars_opti.transpose() << endl;
    double l=mcmc.loglikelihood_theta(&yobs,hpars_opti,theta_candidate); //calcul avec f.
    cout << "l :" << l << endl;
    if(l>ll_max1){ll_max1=l;}
    std::vector<VectorXd>::iterator position = std::find(lhs1.begin(), lhs1.end(), theta_candidate);
    //on nettoie le lhs
    lhs1.erase(position);
    cout << "theta_candidate : " << theta_candidate.transpose() << endl;
    cout << "ll_max1 :" << ll_max1 << endl;
    visited_theta1.push_back(theta_candidate);
    //on réalise l'évaluation à theta_candidate
    for (int ie=0;ie<yobs.size();ie++){
      VectorXd X=yobs[ie].GetX();
      VectorXd I(dimgp);
      I.head(dimx)=X;
      I.tail(dimtheta)=theta_candidate;
      DATA dat; dat.SetX(I); dat.SetValue(my_model(X,theta_candidate));
      data_GP1.push_back(dat);
    }
    //on calcule la vraisemblance de theta_candidate pour servir de référence.
    gp1.SetData(data_GP1);
    gp1.SetGP(hpars_gp);
    //on optimise les hpars du gp
    gp1.OptimizeGP(myoptfunc_gp, &Bounds, &hpars_gp, np);
	  cout << "Neval " << neval << endl;
  	cout  << "par (Guess) : " << hpars_gp.transpose() << endl;
  	hpars_gp = gp1.GetPar();
  	cout  << "par (Optim) (lcor,var,sigmaexp) : " << hpars_gp.transpose() << endl;
    double d=test_gp(gp1,lb_t,ub_t,lb_x,ub_x); //norme L2
    out=fopen("results/L2.gnu","a");
    fprintf(out,"%e\n",d);
    fclose(out);
  }
  //affichage des points visités
  out=fopen("results/visited_points_ei.gnu","w");
  for (int i=0;i<npts_init;i++){
    fprintf(out, "%e\n",gridLHSINIT[i](0));
  }
  for (int i=0;i<visited_theta1.size();i++){
    fprintf(out, "%e\n",visited_theta1[i](0));
  }
  fclose(out);

  /* EI 2 avec pas d'erreur de modèle*/
  vector<VectorXd> visited_theta2;
  VectorXd hpars_gp2=hpars_typical_f;
  GP gp2(Kernel_GP);
  gp2.SetData(data_GP2);
  gp2.SetGP(hpars_gp2); //attention on vit dans l'espace x x theta.
  gp2.OptimizeGP(myoptfunc_gp, &Bounds, &hpars_gp2, np);
  vector<VectorXd> visited_theta_einoedm;
  //nettoyage du fichier L2noedm.gnu
  out=fopen("results/L2noedm.gnu","w");
  fclose(out);
    for (int i=0;i<nobs_max;i++){
    string filename="results/step"+to_string(i)+"noedm.gnu";
    out=fopen(filename.c_str(),"w");
    //boucle sur les observations possibles pour savoir où le critère est le plus élevé
    VectorXd theta_candidate(dimtheta);
    double crit_max=-1E300;
    for (const auto &theta:lhs2){
      double e=EI_prob_noedm(gp2,theta,hpars_typical,mcmc,yobs,generator,ll_max2,buffer_ss);
      fprintf(out,"%e %e %e\n",theta(0),e,buffer_ss);
      if(e>crit_max){
        theta_candidate=theta;
        crit_max=e;
      }
    }
    fclose(out);
    cout << "espérance d'amélioration : "<< crit_max << endl;
    double l=mcmc.loglikelihood_theta(&yobs,hpars_noedm,theta_candidate);
    if(l>ll_max2){ll_max2=l;}
    cout << "l2 : " << l << endl;
    std::vector<VectorXd>::iterator position = std::find(lhs2.begin(), lhs2.end(), theta_candidate);
    //on nettoie le lhs
    lhs2.erase(position);
    cout << "theta_candidate : " << theta_candidate.transpose() << endl;
    cout << "ll_max2 :" << ll_max2 << endl;
    visited_theta2.push_back(theta_candidate);
    //on réalise l'évaluation à theta_candidate et on rajoute dans le GP.
    for (int ie=0;ie<yobs.size();ie++){
      VectorXd X=yobs[ie].GetX();
      VectorXd I(dimgp);
      I.head(dimx)=X;
      I.tail(dimtheta)=theta_candidate;
      DATA dat; dat.SetX(I); dat.SetValue(my_model(X,theta_candidate));
      data_GP2.push_back(dat);
    }
    gp2.SetData(data_GP2);
    gp2.SetGP(hpars_gp2);
    //on optimise les hpars du gp
    gp2.OptimizeGP(myoptfunc_gp, &Bounds, &hpars_gp2, np);
	  cout << "Neval " << neval << endl;
  	cout  << "par (Guess) : " << hpars_gp2.transpose() << endl;
  	hpars_gp2 = gp2.GetPar();
  	cout  << "par (Optim) (lcor,var,sigmaexp) : " << hpars_gp2.transpose() << endl;
    out=fopen("results/L2noedm.gnu","a");
    double d=test_gp(gp2,lb_t,ub_t,lb_x,ub_x);
    fprintf(out,"%e\n",d);
    fclose(out);
  }
  //affichage des points visités
  out=fopen("results/visited_points_ei.gnu","w");
  for (int i=0;i<npts_init;i++){
    fprintf(out, "%e %e\n",gridLHSINIT[i](0),gridLHSINIT[i](0));
  }
  for (int i=0;i<visited_theta1.size();i++){
    fprintf(out, "%e %e\n",visited_theta1[i](0),visited_theta2[i](0));
  }
  fclose(out);


  //affichage des critères sur le LHS
  out=fopen("results/EIprob.gnu","w");
  for (int i=0;i<lhs.size();i++){
    double e=EI_prob(gp1,lhs1[i],hpars_typical,mcmc,yobs,generator,ll_max1,buffer_ss);
    double f=EI_prob_noedm(gp2,lhs2[i],hpars_typical,mcmc,yobs,generator,ll_max2,buffer_ss);
    fprintf(out,"%e %e %e\n",lhs1[i](0),e,f);
  }
  fclose(out);

  //affichage des GPs
  VectorXd theta_aff(1);
  theta_aff(0)=0.9;
  vector<VectorXd> lhs_x=InitGridLHS(lb_x,ub_x,200);
  out=fopen("results/gp.gnu","w");
  for (int i=0;i<lhs_x.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=lhs_x[i];
    I.tail(dimtheta)=theta_aff;
    VectorXd Pred=gp1.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    VectorXd Pred2=gp2.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    fprintf(out,"%e %e %e\n",I(0),Pred(0),Pred2(0));
  }
  fclose(out);
  //affichage en -0.1
  theta_aff(0)=-0.1;
  out=fopen("results/gp2.gnu","w");
  for (int i=0;i<lhs_x.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=lhs_x[i];
    I.tail(dimtheta)=theta_aff;
    VectorXd Pred=gp1.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    VectorXd Pred2=gp2.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    fprintf(out,"%e %e %e\n",I(0),Pred(0),Pred2(0));
  }
  fclose(out);
  //affichage en 0.5
  theta_aff(0)=0.5;
  out=fopen("results/gp3.gnu","w");
  for (int i=0;i<lhs_x.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=lhs_x[i];
    I.tail(dimtheta)=theta_aff;
    VectorXd Pred=gp1.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    VectorXd Pred2=gp2.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    fprintf(out,"%e %e %e\n",I(0),Pred(0),Pred2(0));
  }
  fclose(out);





  exit(0);
 }
}
