// Dans ce fichier : on met en oeuvre différents algorithmes d'optimisation pour KOH. Voir lequel est le meilleur.
// On met en place une quadrature pour évaluer de manière précise l'intégrale KOH.
// On regarde maintenant la sensibilité aux observations.


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
#include "halton.hpp"
#include "densities.h"


using namespace std;
using namespace Eigen;

int neval=1;
//std::default_random_engine generator;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);
vector<DATA> data;
vector<VectorXd> Grid;


int gridstep=140; //Finesse du grid en theta
int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
int dim_theta=1;
int alpha_ig=5; // paramètres du prior sur lcor
double beta_ig=0.4;


double my_function(VectorXd const &x){
  return x(0);
};

double my_model(VectorXd const &x,VectorXd const &theta){
  return x(0)*sin(2.0*theta(0)*x(0))+(x(0)+0.15)*(1-theta(0));
}

double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &xp, Eigen::VectorXd const &hpar){
  //Fonction Kernel sans bruit. hpar(0) = sig_edm, hpar(1) = sig_exp, hpar(2) = lcor
   return hpar(0)*hpar(0)*exp(-pow((x(0)-xp(0))/hpar(2),2)*.5 ); /* squared exponential kernel */
};

double PriorMean(VectorXd const &x, VectorXd const &hpars){
  return 0;
}

double logprior_hpars(VectorXd const &hpars){
  double alpha_ig=5;
  double beta_ig=0.4;
  return log(pow(beta_ig,alpha_ig)*pow(hpars(2),-alpha_ig-1)*exp(-beta_ig/hpars(2))/tgamma(alpha_ig));
}

VectorXd sample_hpars(std::default_random_engine &generator)
{
  //tirage de hpars selon la prior.
  double alpha_ig=5;
  double beta_ig=0.4;
  std::vector<double> lb_hpars(3); lb_hpars[0]=exp(-5);lb_hpars[1]=exp(-6);lb_hpars[2]=exp(-3); //-5 avant
  std::vector<double> ub_hpars(3); ub_hpars[0]=exp(0);ub_hpars[1]=exp(-1);ub_hpars[2]=2; //bornes sur l'optimisation des hpars. edm exp lcor
  
  VectorXd hpars(3);
  hpars(0)=lb_hpars[0]+(ub_hpars[0]-lb_hpars[0])*distU(generator);
  hpars(1)=lb_hpars[1]+(ub_hpars[1]-lb_hpars[1])*distU(generator);
  gamma_distribution<double> dist(alpha_ig,1./beta_ig);
  hpars(2)=1./dist(generator);
  return hpars;

}

double logprior_pars(VectorXd const &pars){
  //prior uniforme sur les pramètres
  return 1;
}

void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
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

  // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(1); lb_t(0)=-0.5;
  VectorXd ub_t(1); ub_t(0)=1.5; 

  std::vector<double> lb_hpars(3); lb_hpars[0]=exp(-5);lb_hpars[1]=exp(-6);lb_hpars[2]=exp(-3); //-5 avant
  std::vector<double> ub_hpars(3); ub_hpars[0]=exp(0);ub_hpars[1]=exp(-1);ub_hpars[2]=2; //bornes sur l'optimisation des hpars. edm exp lcor
  
  //construction du DoE initial en LHS
  DoE doe_init(lb_t,ub_t,140);//,generator);
  
  //construction des observations
  double xp_noise=0.01;
  vector<DATA> data;
  for(unsigned d=0; d<nd; d++){
   VectorXd x(1);
   x(0) = (double) d/(double)nd;
   double f = my_function(x) + distN(generator)*xp_noise;
   DATA dat; dat.SetX(x); dat.SetValue(f);
   data.push_back(dat);
  }

  //configuration de l'instance de base de densité
  Density MainDensity(doe_init);
  MainDensity.SetObs(data);
  MainDensity.SetModel(my_model);
  MainDensity.SetKernel(Kernel);
  MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(PriorMean);

  int samp_size=50;
  vector<VectorXd> X(samp_size); for (int i=0;i<samp_size;i++){VectorXd x(1); x(0)=double(i)/double(samp_size); X[i]=x;}
  cout << "au revoir" << endl;
  /* Partie MCMC */
  {
  int nchain=2000000;
  VectorXd t_init(4);
  t_init(0)=0;
  t_init(1)=0.07;
  t_init(2)=0.1;
  t_init(3)=0.07;
  
  MatrixXd COV=MatrixXd::Zero(4,4);
  COV(0,0)=pow(0.2,2);
  COV(1,1)=pow(0.05,2);
  COV(2,2)=pow(0.001,2);
  COV(3,3)=pow(0.1,2);
  MCMC mcmc(MainDensity,nchain);
  mcmc.Run(t_init,COV,generator);
  mcmc.SelectSamples(2000);
  cout << "map :" << mcmc.MAP().transpose() << endl;
  cout << "mean :" << mcmc.Mean().transpose() << endl;
  cout << "cov : " << mcmc.Cov() << endl;

  MatrixXd Predictions=mcmc.VarPred(X);
  for (int i=0;i<X.size();i++)
  {
    cout << " X : " << X[i] << ", varpred : " << Predictions(i,i) << endl;
  }
  }

  cout << "bonjour" << endl;

  /*Partie KOH*/
  DensityKOH DKOH(MainDensity);
  DKOH.Build();
  cout << "hpars koh :" << DKOH.GetHpars().transpose() << endl;
  cout << DKOH.MAP().transpose() << endl;
  cout << DKOH.Mean().transpose() << endl;
  cout << DKOH.Cov() << endl;
  cout << DKOH.Entropy() << endl;
  DKOH.WritePost("resultsdensity/pkoh.gnu");

  VectorXd temp=DKOH.DrawSample(X,generator);
  PrintVector(X,temp,"resultsdensity/predkoh1.gnu");
  temp=DKOH.DrawSample(X,generator);
  PrintVector(X,temp,"resultsdensity/predkoh2.gnu");
  temp=DKOH.DrawSample(X,generator);
  PrintVector(X,temp,"resultsdensity/predkoh3.gnu");


  DensityOpt Dopt(MainDensity);

  Dopt.Build();
  cout << Dopt.MAP().transpose() << endl;
  cout << Dopt.Mean().transpose() << endl;
  cout << Dopt.Cov() << endl;
  cout << Dopt.Entropy() << endl;
  Dopt.WritePost("resultsdensity/popt.gnu");
  Dopt.WriteHpars("resultsdensity/hopt.gnu");
 
  temp=Dopt.DrawSample(X,generator);
  PrintVector(X,temp,"resultsdensity/predopt1.gnu");
  temp=Dopt.DrawSample(X,generator);
  PrintVector(X,temp,"resultsdensity/predopt2.gnu");
  temp=Dopt.DrawSample(X,generator);
  PrintVector(X,temp,"resultsdensity/predopt3.gnu");

  DensityCV DCV(DKOH);
  DCV.Build();
  DCV.WritePost("resultsdensity/pcv.gnu");

  temp=DCV.DrawSample(X,generator);
  PrintVector(X,temp,"resultsdensity/predcv1.gnu");
  temp=DCV.DrawSample(X,generator);
  PrintVector(X,temp,"resultsdensity/predcv2.gnu");
  temp=DCV.DrawSample(X,generator);
  PrintVector(X,temp,"resultsdensity/predcv3.gnu");


  exit(0);
  DensityBayes DBayes(MainDensity);
  DBayes.SetSampleHpars(sample_hpars);
  DBayes.Build();
  cout << DBayes.MAP().transpose() << endl;
  cout << DBayes.Mean().transpose() << endl;
  cout << DBayes.Cov() << endl;
  cout << DBayes.Entropy() << endl;
  DBayes.WritePost("resultsdensity/pbayes.gnu");



  exit(0);
  
};
