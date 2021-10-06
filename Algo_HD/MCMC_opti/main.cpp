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

//paramètres pour python
PyObject *pFunc, *pArgs;
PyObject *pParamsRavik; //liste comprenant les valeurs nominales des paramètres de Ravik.

double gaussprob(double x,double mu, double sigma){
  //renvoie la probabilité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}

double my_function(VectorXd const &x){
  return x(0);
};


double my_model(VectorXd const &x, VectorXd const &theta){
  //taille totale des paramètres dans le modèle : 10.
  //création d'une liste pour les arguments
  return x(0)*sin(2.0*theta(0)*x(0))+(x(0)+0.15)*(1-theta(0));
}

double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &xp, Eigen::VectorXd const &hpar){
  //Fonction Kernel sans bruit. hpar(0) = sig_edm, hpar(1) = sig_exp, hpar(2) = lcor
  double d=abs(x(0)-xp(0));
  return hpar(0)*hpar(0)*exp(-pow(d/hpar(2),2)*.5); /* squared exponential kernel */

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
  return 0;
}

void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
}
MatrixXd Gamma(std::vector<Eigen::VectorXd> &vectorx,VectorXd hpars) {
  // Renvoie la matrice de corrélation avec  bruit
  int nd=vectorx.size();
  Eigen::MatrixXd A(nd,nd);
  for(int i=0; i<nd; i++){
    for(int j=i; j<nd; j++){
      A(i,j) = Kernel(vectorx[i],vectorx[j], hpars);
      if(i!=j){
	A(j,i) = A(i,j);
      }else{
	A(i,j) += hpars(1)*hpars(1);					//Noise correlation
      }
    }
  }
  return A;
}

MatrixXd Testou(std::vector<Eigen::VectorXd> &vectorx,VectorXd hpars){ //new
  VectorXd Hparscourant=hpars; //valeur des hpars
  MatrixXd Kstar1(vectorx.size(),vectorx.size());
  MatrixXd Kstar2(vectorx.size(),vectorx.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<vectorx.size();j++){
      Kstar1(i,j)=Kernel(vectorx[i],vectorx[j],Hparscourant);
      Kstar2(j,i)=Kernel(vectorx[j],vectorx[i],Hparscourant);
    }
  }
  MatrixXd Kprior(vectorx.size(),vectorx.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<vectorx.size();j++){
      Kprior(i,j)=Kernel(vectorx[i],vectorx[j],Hparscourant);
    }
  }
  MatrixXd G=Gamma(vectorx,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  MatrixXd varred=Kstar1*ldlt.solve(Kstar2);
  return Kprior-varred;
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
  int cas=20;

  // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(1); lb_t(0)=-0.5;
  VectorXd ub_t(1); ub_t(0)=1.5; 

  std::vector<double> lb_hpars(3); lb_hpars[0]=exp(-5);lb_hpars[1]=exp(-6);lb_hpars[2]=exp(-3); //-5 avant
  std::vector<double> ub_hpars(3); ub_hpars[0]=exp(0);ub_hpars[1]=exp(-1);ub_hpars[2]=2; //bornes sur l'optimisation des hpars. edm exp lcor
  generator.seed(seed_obs);
  //construction des observations et écriture dans un fichier
  FILE* out = fopen("results/observations.gnu","w");
  double xp_noise=0.01;
  vector<DATA> data;
  for(unsigned d=0; d<nd; d++){
   VectorXd x(1);
   x(0) = (double) d/(double)nd;
   fprintf(out,"%e ",x(0));
   double f = my_function(x) + distN(generator)*xp_noise;
   DATA dat; dat.SetX(x); dat.SetValue(f);
   fprintf(out,"%e \n",dat.Value());
   data.push_back(dat);
  }
  fclose(out);


  
  //Points sur lesquels on va tracer des réalisations de notre modèle
  int samp_size=80; //80 avant
  vector<VectorXd> X(samp_size); for (int i=0;i<samp_size;i++){VectorXd x(1); x(0)=0.01+1*double(i)/double(samp_size); X[i]=x;}  
  //construction du DoE initial en LHS (generator) ou Grid Sampling(sans generator)
   
  
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



  /*Partie Opti à faire fonctionner d'abord.*/
  
  DensityOpt Dopt(MainDensity);
  Dopt.Build();
  cout << "MAP : " << Dopt.MAP().transpose() << endl;
  cout << "Mean : " << Dopt.Mean().transpose() << endl;
  cout << Dopt.Cov() << endl;
  Dopt.WritePost("results/popt.gnu");
  Dopt.WriteHpars("results/hopt.gnu");
  VectorXd temp=Dopt.DrawSample(generator);
  PrintVector(X,temp,"results/predopt1.gnu");
  temp=Dopt.DrawSample(generator);
  PrintVector(X,temp,"results/predopt2.gnu");
  temp=Dopt.DrawSample(generator);
  PrintVector(X,temp,"results/predopt3.gnu");
  cout << "Pred Opti over." << endl;

  

  /*Partie KOH*/
  //il faut récupérer la constante de normalisation
  cout << "max opti : " << Dopt.GetNormCst() << endl;
  DensityKOH DKOH(MainDensity,Dopt.GetNormCst());
  DKOH.Build();
  cout << "hpars koh :" << DKOH.GetHpars().transpose() << endl;
  cout << "MAP : " << DKOH.MAP().transpose() << endl;
  cout << "Mean : " << DKOH.Mean().transpose() << endl;
  cout << DKOH.Cov() << endl;
  DKOH.WritePost("results/pkoh.gnu");
  //phase de test. On vérifie les valeurs du critère KOH pour l'hypothèse "haute EDM faible EXP"
  //appel à la fonction

  temp=DKOH.DrawSample(generator);
  PrintVector(X,temp,"results/predkoh1.gnu");
  temp=DKOH.DrawSample(generator);
  PrintVector(X,temp,"results/predkoh2.gnu");
  temp=DKOH.DrawSample(generator);
  PrintVector(X,temp,"results/predkoh3.gnu");
  cout << "Pred KOH over." << endl;


  DensitySimple DSimple(MainDensity,Dopt.GetNormCst());
  DSimple.Build();
  cout << "hpars Simple :" << DSimple.GetHpars().transpose() << endl;
  cout << "MAP : " << DSimple.MAP().transpose() << endl;
  cout << "Mean : " << DSimple.Mean().transpose() << endl;
  cout << DSimple.Cov() << endl;
  DSimple.WritePost("results/psimp.gnu");
  cout << "Pred Simple over." << endl;

  
  
  DKOH.WritePredictions("results/predkoh.gnu");
  Dopt.WritePredictions("results/predopt.gnu");
  DKOH.WritePredictionsFZ("results/predkohFZ.gnu");
  Dopt.WritePredictionsFZ("results/predoptFZ.gnu");
  DSimple.WritePredictions("results/predsimp.gnu");
  DSimple.WritePredictionsFZ("results/predsimpFZ.gnu");
  {
    VectorXd hparskoh=DKOH.GetHpars();
    FILE* out=fopen("results/summary.gnu","w");
    fprintf(out,"Hpars KOH :\n");
    fprintf(out,"%e %e %e",hparskoh(0),hparskoh(1),hparskoh(2));
    fclose(out);
  }
  /* Partie MCMC */
  {
    int nchain=2000000;///370s pour 50000steps 1500000 la best value pour 3000 samples
    VectorXd t_init(4);
    t_init(0)=1;
    t_init(1)=0.07;
    t_init(2)=0.01;
    t_init(3)=0.20;

    cout << "t_init : " << t_init.transpose() << endl;
    MatrixXd COV=MatrixXd::Zero(4,4);
    COV(0,0)=pow(0.13,2);
    COV(1,1)=pow(0.03,2);
    COV(2,2)=pow(0.005,2);
    COV(3,3)=pow(0.03,2);
    cout << "COV :" << COV << endl;

    clock_t c_start = std::clock();

    cout << "running mcmc1..." << endl;
    MCMC mcmc(MainDensity,nchain);
    generator.seed(119);
    mcmc.Run(t_init,COV,generator);
    mcmc.SelectSamples(2000);
    mcmc.WriteAllSamples("results/mcmc_allsamples.gnu");
    mcmc.WriteSelectedSamples("results/mcmc_selectedsamples.gnu");
    mcmc.WritePredictions("results/predmcmc.gnu");
    mcmc.WritePredictionsFZ("results/predmcmcFZ.gnu");
    cout << "calcul de la corrélation mcmc..." << endl;
    mcmc.Autocorrelation_diagnosis(1000);
    clock_t c_end = std::clock();
    cout << "temps d'exécution mcmc1:" << (c_end-c_start) / CLOCKS_PER_SEC << endl;
    cout << "running mcmc2..." << endl;

    //MCMC avec opti.
    MCMC_opti mcmc2(mcmc,1000);//optimisation tous les 1000 steps
    mcmc2.Run(t_init,COV,generator);
    mcmc2.SelectSamples(2000);
    mcmc2.WriteAllSamples("results/mcmc2_allsamples.gnu");
    mcmc2.WriteSelectedSamples("results/mcmc2_selectedsamples.gnu");
    mcmc2.WritePredictions("results/predmcmc2.gnu");
    mcmc2.WritePredictionsFZ("results/predmcmc2FZ.gnu");
    mcmc2.Autocorrelation_diagnosis(1000);

    c_start = std::clock();
    cout << "temps d'exécution mcmc2:" << (c_start-c_end) / CLOCKS_PER_SEC << endl;

    
    //MCMC avec opti.
    MCMC_opti mcmc3(mcmc,100);//optimisation tous les 100 steps
    mcmc3.Run(t_init,COV,generator);
    mcmc3.SelectSamples(2000);
    mcmc3.WriteAllSamples("results/mcmc3_allsamples.gnu");
    mcmc3.WriteSelectedSamples("results/mcmc3_selectedsamples.gnu");
    mcmc3.WritePredictions("results/predmcmc3.gnu");
    mcmc3.WritePredictionsFZ("results/predmcmc3FZ.gnu");
    mcmc3.Autocorrelation_diagnosis(1000);

    c_end = std::clock();
    cout << "temps d'exécution mcmc3:" << -(c_start-c_end) / CLOCKS_PER_SEC << endl;

  }





  exit(0);

  //test de variance prédictive

  {
  int samp_size(5);
  vector<VectorXd> X(samp_size); for (int i=0;i<samp_size;i++){VectorXd x(1); x(0)=1+25*double(i)/double(samp_size); X[i]=x;}
  VectorXd hpars(3);
  hpars(0)=1;
  hpars(1)=0.001;
  hpars(2)=20;
  cout << Testou(X,hpars) << endl;

  }
  exit(0);  
};
