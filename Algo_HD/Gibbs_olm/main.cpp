// Dans ce fichier : on met en oeuvre différents algorithmes d'optimisation pour KOH. Voir lequel est le meilleur.
// On met en place une quadrature pour évaluer de manière précise l'intégrale KOH.
// On regarde maintenant la sensibilité aux observations.
// On essaye avec un hpar supplémentaire : moyenne de model bias constante


#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <random>
#include <functional>
#include "pcb++.h"
#include "cub++.h"
#include "vstoch++.h"
#include "mstoch++.h"
#include "sampler.h"
#include <ctime>
#include "densities.h"
#include "gibbs.h"
#include <Python.h>


using namespace std;
using namespace Eigen;

int neval=1;
//std::default_random_engine generator;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);
vector<DATA> data;
vector<VectorXd> Grid;

int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
int dim_theta=1;


double my_function(VectorXd const &x){
  return x(0);
};

double gaussprob(double x,double mu, double sigma){
  //renvoie la probabilité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}

double my_model(VectorXd const &x, VectorXd const &theta){
  //taille totale des paramètres dans le modèle : 10.
  //création d'une liste pour les arguments
  return x(0)*sin(2.0*theta(0)*x(0))+(x(0)+0.15)*(1-theta(0)); 
}

double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &xp, Eigen::VectorXd const &hpar){
  //Fonction Kernel sans bruit. hpar(0) = sig_edm, hpar(1) = sig_exp, hpar(2) = lcor

  double d=abs(x(0)-xp(0));
  return hpar(0)*hpar(0)*exp(-pow((d)/hpar(2),2)*.5); /* squared exponential kernel */
  return pow(hpar(0),2)*(1+((2.24*d)/hpar(2))+1.66*pow(d/hpar(2),2))*exp(-(2.24*d)/hpar(2)); /*Matern 5/2*/
  return pow(hpar(0),2)*exp(-(d)/hpar(2)); /*Matern 1/2*/
  return pow(hpar(0),2)*(1+(1.732*d)/hpar(2))*exp(-(1.732*d)/hpar(2)); /*Matern 3/2*/ // marche bien
  return hpar(0)*hpar(0)*exp(-pow((x(0)-xp(0))/hpar(2),2)*.5); /* squared exponential kernel */
  
  
  
};

double PriorMean(VectorXd const &x, VectorXd const &hpars){
  return 0;
}

double logprior_hpars(VectorXd const &hpars){
  if(hpars(2)<=0){return -999;}
  double alpha_ig=5.5;
  double beta_ig=0.3;
  return log(pow(beta_ig,alpha_ig)*pow(hpars(2),-alpha_ig-1)*exp(-beta_ig/hpars(2))/tgamma(alpha_ig));//-log(hpars(0));
}

double logprior_pars(VectorXd const &pars){
  //prior uniforme sur les pramètres
  return 0;
}

void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
}

void PrintVector(vector<VectorXd> &X, const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e\n",X[i](0));
  }
  fclose(out);
}

int Choose_hpars(VectorXd theta, vector<MCMC_par> const & mcmc_v, default_random_engine &generator){
  // tire au hasard un kde duquel choisir notre prochaine valeur de paramètres selon le critère de vraisemblance.
  uniform_real_distribution<double> distU(0,1);
  vector<double> vrais;
  double sum_of_elements(0);
  for (int i=0;i<mcmc_v.size();i++){
    vrais.push_back(exp(mcmc_v[i].loglikelihood_theta_fast(theta)+logprior_hpars(mcmc_v[i].GetHpars())));
    sum_of_elements+=vrais[i];
  }
  double running_sum(0);
  double threshold=sum_of_elements*distU(generator);
  for (int i=0;i<mcmc_v.size();i++){
    running_sum+=vrais[i];
    if(running_sum >= threshold){
      return i;
    }
  }
}

int Choose_interval(VectorXd theta,vector<VectorXd> const &theta_fences){
  // détermine dans quel intervalle tombe le theta en argument.
  // ne marche qu'en dimension 1.
  if(theta(0)<=theta_fences[0](0)){
    return -1;
  }
  for (int i=1;i<theta_fences.size();i++){
    if(theta(0)>= theta_fences[i-1](0) && theta(0)<= theta_fences[i](0)){
      return i-1;
    }
  }
  return -1;
}

VectorXd Class(vector<vector<VectorXd>> const &v){
  int n=v.size();
  int m=v[0].size();
  VectorXd result=VectorXd::Zero(n);
  for (int i=0;i<n;i++){
    for (int j=0;j<m;j++){
      if(v[i][j](0)<=0.5){
        //on compte les thetas inf à 0.5
        ++result(i);
      }
    }
    result(i)/=m*1.0;
    result(i)*=100;
  }
  return result;
}

const double Big = -1.e16;


int main(int argc, char **argv){
  if(argc != 3){
    cout << "Usage :\n";
    cout << "\t Number of obs, seed_obs\n";
    exit(0);
  }
  int nd  = atoi(argv[1]);
  uint32_t seed_obs=atoi(argv[2]);//
  generator.seed(seed_obs);


  // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(1); lb_t(0)=-1.5;
  VectorXd ub_t(1); ub_t(0)=1.5; 

  std::vector<double> lb_hpars(3); lb_hpars[0]=exp(-5);lb_hpars[1]=exp(-6);lb_hpars[2]=exp(-3); //-5 avant
  std::vector<double> ub_hpars(3); ub_hpars[0]=exp(0);ub_hpars[1]=exp(-1);ub_hpars[2]=2; //bornes sur l'optimisation des hpars. edm exp lcor
  
  //génération des observations
  double xp_noise=0.01;
   for(int d=0; d<nd; d++){
   VectorXd x(1);
   x(0) = (double) d/(double)nd;
   double f = my_function(x) + distN(generator)*xp_noise;
   DATA dat; dat.SetX(x); dat.SetValue(f);
   data.push_back(dat);
  }

 
  generator.seed(12789533); //seed pour les mcmc.
  /*initialisation du python*/
  vector<VectorXd> v(100);
  for (int i=0;i<100;i++){
    VectorXd X(3);
    X << distU(generator),distU(generator),distU(generator);
    v[i]=X;
  }
  
  //construction des pts de prédiction
  vector<VectorXd> X(150); for (int i=0;i<X.size();i++){VectorXd x(1); x(0)=double(i)/double(X.size()); X[i]=x;}
  
  DoE doe_init(lb_t,ub_t,140);//,generator); gridsize
  doe_init.Fill(my_model,&data);
  doe_init.Fill_Predictions(X);
  
  
  //configuration de l'instance de base de densité
  Density MainDensity(doe_init);
  MainDensity.SetModel(my_model);
  MainDensity.SetKernel(Kernel);
  MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(PriorMean);

  
  
  VectorXd t_init2(4);
  t_init2(0)=1;
  t_init2(1)=0.07;
  t_init2(2)=0.01;
  t_init2(3)=0.20;
  MatrixXd COV=MatrixXd::Zero(4,4);
  COV(0,0)=pow(0.30,2);
  COV(1,1)=pow(0.03,2);
  COV(2,2)=pow(0.005,2);
  COV(3,3)=pow(0.03,2);
  MCMC mcmc(MainDensity,2000000);
  mcmc.Run(t_init2,COV,generator);
  mcmc.SelectSamples(2000);
  mcmc.WriteSelectedSamples("results/mcmc_samp.gnu");

  VectorXd t_init(1);
  t_init(0)=0.5;
  MatrixXd COVright=MatrixXd::Zero(1,1);
  COVright(0,0)=pow(0.22,2);
  
  vector<VectorXd> hpars_v;
  vector<MCMC_par> mcmc_v;
  vector<KDE> kde_v;

  //remplissage du vecteur d'hyperparamètres
  VectorXd hpars1(3);
  hpars1 << 9.45E-2,8E-3,0.35;
  VectorXd hpars2(3);
  hpars2 << 5.73E-2,8E-3,0.20;
  VectorXd hpars3(3);
  hpars3 << 0.2,8E-3,0.30;
  /*VectorXd hpars4(3);
  hpars4 << 0.1,8E-3,0.30;
  VectorXd hpars5(3);
  hpars5 << 0.1,8E-3,0.37;*/
  hpars_v.push_back(hpars1);
  hpars_v.push_back(hpars2);
  hpars_v.push_back(hpars3);
  int n_hpars=hpars_v.size();
  
  for(int i=0;i<n_hpars;i++){
    string filename="results/samp"+to_string(i)+".gnu";
    MCMC_par mcmc2(MainDensity,hpars_v[i],2000000);
    mcmc2.Run(t_init,COVright,generator);
    mcmc2.SelectSamples(2000);
    mcmc2.WriteSelectedSamples(filename.c_str());
    vector<VectorXd> samples=mcmc2.GetSelectedSamples();
    KDE kde(samples);
    mcmc_v.push_back(mcmc2);
    kde_v.push_back(kde);
  }
  

  //fitter un KDE sur un échantillon de densités, tirer des échantillons et voir ce que ça donne.

  //maintenant faire un gibbs sampling.
  {
    clock_t c_start = std::clock();
    vector<VectorXd> sel;
    VectorXd theta_courant(1);
    theta_courant(0)=0.2;
    for (int i=0;i<2000000;i++){
      int inext=Choose_hpars(theta_courant,mcmc_v,generator);
      theta_courant=kde_v[inext].Sample(generator);
      if(i%(2000000/2000)==0 &&i>(2000000/2000)){
        sel.push_back(theta_courant);
      }
    }
    PrintVector(sel,"results/selgibbs.gnu");
    clock_t c_end = std::clock();
    double time=(c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "Temps pour Gibbs : " << time << " s\n";
  }
  cout << "Méthode avec Diracs" << endl;

  int n_intervals=40;
  //maintenant : partie avec hypothèses de Diracs. 
  //séparation de l'espace des thetas en n intervalles réguliers. ne marche qu'en dimension 1.
  vector<VectorXd> theta_fences(n_intervals+1); //on inclut les bords
  for (int i=0;i<n_intervals+1;i++){
    VectorXd theta(1);
    theta(0)=lb_t(0)+(ub_t(0)-lb_t(0))*i/double (n_intervals);
    theta_fences[i]=theta;
  }
  //choix d'une valeur d'hyperparamètres par intervalle, puis mcmcs et KDE.
  vector<VectorXd> hparsopt(n_intervals);
  vector<MCMC_par> mcmcopt; //obligé de ne pas initialiser car je n'ai pas défini de constructeur vide.
  vector<vector<VectorXd>> samplesopt(n_intervals);
  vector<KDE> kdeopt;

  for (int i=0;i<n_intervals;i++){
    string filename="results/sampopt"+to_string(i)+".gnu";
    VectorXd theta_courant(1);
    theta_courant(0)=theta_fences[i](0)+(theta_fences[i+1](0)-theta_fences[i](0))*distU(generator);
    VectorXd hpars=MainDensity.Opti_hpars(theta_courant,hpars1);
    hparsopt[i]=hpars;
    
    MCMC_par mcmc2(MainDensity,hparsopt[i],2000000);
    mcmc2.Run(t_init,COVright,generator);
    mcmc2.SelectSamples(2000);
    mcmc2.WriteSelectedSamples(filename.c_str());
    vector<VectorXd> samples=mcmc2.GetSelectedSamples();
    samplesopt[i]=samples;
    KDE kde(samples);
    mcmcopt.push_back(mcmc2);
    kdeopt.push_back(kde);
  }
  //Gibbs sampling.
  {
    vector<int> freq(hparsopt.size(),0);
    int naccept(0);
    clock_t c_start = std::clock();
    vector<VectorXd> sel;
    VectorXd theta_courant(1);
    theta_courant(0)=0.2;
    for (int i=0;i<2000000;i++){
      int inext=Choose_interval(theta_courant,theta_fences);
      VectorXd theta_proposal=kdeopt[inext].Sample(generator);
      int iprop=Choose_interval(theta_proposal,theta_fences);
      if(iprop>-1){theta_courant=theta_proposal; naccept++; freq[iprop]++;}
      if(i%(2000000/2000)==0 &&i>(2000000/2000)){
        sel.push_back(theta_courant);
      }
    }
    PrintVector(sel,"results/selgibbsopt.gnu");
    clock_t c_end = std::clock();
    double time=(c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "Temps pour Gibbs Opt: " << time << " s\n";
    cout << "accept rate: " << naccept/2000000. << endl;
    //affichage des fréquences :
    for (int i=0;i<freq.size();i++){
      cout << "(" << i << "," << 100.*freq[i]/(naccept*1.0) << ") ";
    }
    cout << endl;
  }

 //on refait le Gibbs OLM avec la discrétisation fine.
    {
    vector<int> freq(hparsopt.size(),0);
    clock_t c_start = std::clock();
    vector<VectorXd> sel;
    VectorXd theta_courant(1);
    theta_courant(0)=0.2;
    for (int i=0;i<2000000;i++){
      int inext=Choose_hpars(theta_courant,mcmcopt,generator);
      theta_courant=kdeopt[inext].Sample(generator);
      freq[inext]++;
      if(i%(2000000/2000)==0 &&i>(2000000/2000)){
        sel.push_back(theta_courant);
      }
    }
    PrintVector(sel,"results/selgibbs2.gnu");
    clock_t c_end = std::clock();
    double time=(c_end-c_start) / CLOCKS_PER_SEC;
    std::cout << "Temps pour Gibbs2 : " << time << " s\n";
    for (int i=0;i<freq.size();i++){
      cout << "(" << i << "," << 100.*freq[i]/(2000000) << ") ";
    }
    cout << endl;
  }
  //on printe le vecteur de diagnostic des MCMCs individuelles
  {
    FILE* out=fopen("results/diag.gnu","w");
    VectorXd diag=Class(samplesopt);
    for (int i=0;i<diag.size();i++){
      fprintf(out,"%d %e\n",i,diag(i));
    }
    fclose(out);
  }
  //affichage des tests de qualité pour les KDE
  for (int i=0;i<samplesopt.size();i++){
    cout << "kde num" << i << " :" << endl;
    kdeopt[i].FidTest(generator);
  }

  exit(0); 
};
