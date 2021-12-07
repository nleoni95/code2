// Dans ce fichier : on met en oeuvre différents algorithmes d'optimisation pour KOH. Voir lequel est le meilleur.
// On reproduit l'exemple 1 de l'article avec notre nouveau fichier pour être solide.


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
  if(hpars(2)<=0){return -999;}
  double alpha_ig=5.5;
  double beta_ig=0.3;
  return log(pow(beta_ig,alpha_ig)*pow(hpars(2),-alpha_ig-1)*exp(-beta_ig/hpars(2))/tgamma(alpha_ig));//-log(hpars(0));
}

VectorXd sample_hpars(std::default_random_engine &generator)
{
  //tirage de hpars selon la prior.
  double alpha_ig=5.5;
  double beta_ig=0.3;
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
  return 0;
}

void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
}

VectorXd moy_mcmc(const vector<VectorXd> &thetas, const vector<double> &lp){
  //Calcul d'une moyenne. Attention, formalisme très tuned à notre code : lp contient des logprobabilités non normalisées, thetas est de dimension 4.
  //On va tout d'abord normaliser ces probas.
  int ns=lp.size();
  vector<double> p(ns);
  double weight=0;
  for (int i=0;i<ns;i++){
    p[i]=exp(lp[i]);
    weight+=p[i];
  }
  for (int i=0;i<ns;i++){
    p[i]=p[i]/weight;
  }
  VectorXd Mean=VectorXd::Zero(4);
  for (int i=0;i<ns;i++){
    Mean+=thetas[i]*p[i];
  }
  return Mean;
}

MatrixXd cov_mcmc(const vector<VectorXd> &thetas, const vector<double> &lp){
  //Calcul d'une variance. Attention formalisme très tuné à notre code (voir fonction précédente)
  //je sais que le calcul n'est pas bon car on pondère par les probabilités au lieu de moyenner. C'est 
  int ns=lp.size();
  vector<double> p(ns);
  double weight=0;
  for (int i=0;i<ns;i++){
    p[i]=exp(lp[i]);
    weight+=p[i];
  }
  for (int i=0;i<ns;i++){
    p[i]=p[i]/weight;
  }
  VectorXd Mean=moy_mcmc(thetas,lp);
  MatrixXd COV=MatrixXd::Zero(4,4);
  for (int i=0;i<ns;i++){
    COV+=(thetas[i]-Mean)*(thetas[i]-Mean).transpose()*p[i];
  }
  return COV;
}

double reconstructed_prior(double x, double moy, double var){
  //évaluation d'une gaussienne de moyenne moy et de variance var
  return (1./(sqrt(2*M_PI*var))*exp(-0.5*pow(x-moy,2)/var));
}

double post_bayes_rec(double x, double p1, double moyleft, double varleft, double moyright, double varright){
  //évaluation de la post bayes
  double left=p1*reconstructed_prior(x,moyleft,varleft);
  double right=(1-p1)*reconstructed_prior(x,moyright,varright);
  return left+right;
}

double post_fmp_rec(double x, double p1, double moyleft, double varleft, double moyright, double varright){
    //évaluation de la post fmp
  double left=p1*reconstructed_prior(x,moyleft,varleft);
  double right=(1-p1)*reconstructed_prior(x,moyright,varright);
  return left+right;
}

double post_koh_rec(double x, double moy, double var){
    //évaluation de la post fmp
  return reconstructed_prior(x,moy,var);
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
  
  vector<VectorXd> X(150); for (int i=0;i<X.size();i++){VectorXd x(1); x(0)=double(i)/double(X.size()); X[i]=x;}
  //construction du DoE initial en grid
  DoE doe_init(lb_t,ub_t,140);//,generator);
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

  int samp_size=100;
  

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
  DensityKOH DKOH(MainDensity,Dopt.GetNormCst());
  DKOH.Build();
  cout << "hpars koh :" << DKOH.GetHpars().transpose() << endl;
  cout << "MAP : " << DKOH.MAP().transpose() << endl;
  cout << "Mean : " << DKOH.Mean().transpose() << endl;
  cout << DKOH.Cov() << endl;
  DKOH.WritePost("results/pkoh.gnu");
  
  DKOH.WritePredictions("results/predkoh.gnu");
  Dopt.WritePredictions("results/predopt.gnu");
  DKOH.WritePredictionsFZ("results/predkohFZ.gnu");
  Dopt.WritePredictionsFZ("results/predoptFZ.gnu");
  
  DensityCV DCV(DKOH);
  DCV.Build();
  DCV.WritePost("results/pcv.gnu");
  DCV.WritePredictions("results/predcv.gnu");
  
  DensitySimple DSimple(MainDensity,Dopt.GetNormCst());
  DSimple.Build();
  cout << "hpars Simple :" << DSimple.GetHpars().transpose() << endl;
  cout << "MAP : " << DSimple.MAP().transpose() << endl;
  cout << "Mean : " << DSimple.Mean().transpose() << endl;
  cout << DSimple.Cov() << endl;
  DSimple.WritePost("results/psimp.gnu");
  DSimple.WritePredictions("results/predsimp.gnu");
  DSimple.WritePredictionsFZ("results/predsimpFZ.gnu");

  /* Partie MCMC */
  {
  int nchain=2000000;
  VectorXd t_init(4);
  t_init(0)=1;
  t_init(1)=0.07;
  t_init(2)=0.01;
  t_init(3)=0.20;
  
  MatrixXd COV=MatrixXd::Zero(4,4);
  COV(0,0)=pow(0.13,2);
  COV(1,1)=pow(0.03,2);
  COV(2,2)=pow(0.005,2);
  COV(3,3)=pow(0.03,2);
  MCMC mcmc(MainDensity,nchain);
  mcmc.Run(t_init,COV,generator);
  mcmc.SelectSamples(6000);
  cout << "map :" << mcmc.MAP().transpose() << endl;
  cout << "mean :" << mcmc.Mean().transpose() << endl;
  cout << "cov : " << mcmc.Cov() << endl;
  mcmc.WriteSelectedSamples("results/mcmcselectedsamples.gnu");
  mcmc.WriteAllSamples("results/mcmcallsamples.gnu");
  mcmc.WritePredictions("results/predmcmc.gnu");
  mcmc.WritePredictionsFZ("results/predmcmcFZ.gnu");
  out=fopen("results/summary.gnu","w");
  //Ecriture du fichier de résultats
  fprintf(out,"Hpars KOH : %e %e %e\n",DKOH.GetHpars()(0),DKOH.GetHpars()(1),DKOH.GetHpars()(2));
  fprintf(out,"Hpars CV : %e %e %e\n",DCV.GetHpars()(0),DCV.GetHpars()(1),DCV.GetHpars()(2));
  fprintf(out,"Hpars Simp : %e\n",DSimple.GetHpars()(1));
  fprintf(out,"MCMC accept rate : %e\n",mcmc.GetAccRate());
  //traitement MCMC
 
  {
    //tri des samples en deux bins. critère : 0.5
    vector<VectorXd> thetaselect=mcmc.GetSelectedSamples();
    vector<double> fselect=mcmc.GetSelectedValues();
    vector<VectorXd> thetaselectleft;
    vector<double> fselectleft;
    vector<VectorXd> thetaselectright;
    vector<double> fselectright;

    for (int i=0;i<thetaselect.size();i++){
      if (thetaselect[i](0)<0.5){
        thetaselectleft.push_back(thetaselect[i]);
        fselectleft.push_back(fselect[i]);
      }
      else{
        thetaselectright.push_back(thetaselect[i]);
        fselectright.push_back(fselect[i]);
      }
    }
    cout << "nombre d'échantillons à gauche : " << thetaselectleft.size() << " et à droite : " << thetaselectright.size() << endl;
    //Find le max à gauche et le max à droite en utilisant les selected samples.
    VectorXd t_maxleft(4);
    VectorXd t_maxright(4);
    double f_maxleft=0;
    double f_maxright=0;
    for (int i=0;i<thetaselect.size();i++){
      if(fselect[i]>f_maxleft && thetaselect[i](0)<0.5){
        f_maxleft=fselect[i];
        t_maxleft=thetaselect[i];
      }
      else if (fselect[i]>f_maxright && thetaselect[i](0)>0.5){
        f_maxright=fselect[i];
        t_maxright=thetaselect[i];
      }
    }

    cout << " max à gauche : " << t_maxleft.transpose() << "logp : " << f_maxleft << endl; //
    cout << " max à droite : " << t_maxright.transpose() << "logp : " << f_maxright << endl;
    MatrixXd COVLEFT=cov_mcmc(thetaselectleft,fselectleft);
    MatrixXd COVRIGHT=cov_mcmc(thetaselectright,fselectright);
  
    //cout << "matrice cov à gauche : \n" << COVLEFT << endl;
    //cout << "matrice cov à droite : \n" << COVRIGHT << endl;

    double detleft=pow(COVLEFT.llt().matrixL().determinant(),2);
    double detright=pow(COVRIGHT.llt().matrixL().determinant(),2);
    double ratio=sqrt(detleft/detright);

    f_maxleft=exp(f_maxleft);
    f_maxright=exp(f_maxright);
    double pi1=(double) thetaselectleft.size()/((double) thetaselectleft.size()+thetaselectright.size());
    cout << "pi1 : " << pi1 << endl;
    //Calcul du critère KOH :
    MatrixXd COVLEFT_hpars=COVLEFT.block(1,1,3,3);
    VectorXd COVLEFT_cross=COVLEFT.col(0).tail(3); //les 3 derniers éléments de la première colonne
    MatrixXd COVRIGHT_hpars=COVRIGHT.block(1,1,3,3);
    VectorXd COVRIGHT_cross=COVRIGHT.col(0).tail(3); //les 3 derniers éléments de la première colonne
    double critkohleft=pi1/(COVLEFT_hpars-COVLEFT_cross*COVLEFT_cross.transpose()/COVLEFT(0,0)).llt().matrixL().determinant();//c'est bien la racine du déterminant.
    double critkohright=(1-pi1)/(COVRIGHT_hpars-COVRIGHT_cross*COVRIGHT_cross.transpose()/COVRIGHT(0,0)).llt().matrixL().determinant();//c'est bien la racine du déterminant.
    cout << "crit koh left : " << critkohleft << ", right : " << critkohright << endl;
    bool choice_koh_left=(critkohleft>critkohright); //trouver si KOH choisit left ou right
    VectorXd t_maxkoh(4);
    double var_maxkoh(0);
    if(choice_koh_left){t_maxkoh=t_maxleft; var_maxkoh=COVLEFT(0,0)-COVLEFT_cross.transpose()*COVLEFT_hpars.llt().solve(MatrixXd::Identity(3,3))*COVLEFT_cross;}

    else{t_maxkoh=t_maxright; var_maxkoh=COVRIGHT(0,0)-COVRIGHT_cross.transpose()*COVRIGHT_hpars.llt().solve(MatrixXd::Identity(3,3))*COVRIGHT_cross;}

    double varbayesleft=COVLEFT(0,0);
    double varbayesright=COVRIGHT(0,0);
    double varfmpleft=varbayesleft;
    double varfmpright=varbayesright;
    double pi1fmp=pi1*(sqrt(COVLEFT(0,0)/detleft))/(pi1*(sqrt(COVLEFT(0,0)/detleft))+(1-pi1)*(sqrt(COVRIGHT(0,0)/detright)));
    cout << "pi1fmp : " << pi1fmp << endl;
    cout << "varright : " << varbayesright << ", varkohright : " << var_maxkoh << endl;
    cout << "COVLEFT : " << COVLEFT << endl;
    cout << "COVRIGHT : " << COVRIGHT << endl;

    FILE* out =fopen("results/recons.gnu","w");

    for (int i=0;i<140;i++){
      double theta=-0.5+2*((double) i)/140.;
      double p_bayes=post_bayes_rec(theta,pi1,t_maxleft(0),varbayesleft,t_maxright(0),varbayesright);
      double p_fmp=post_fmp_rec(theta,pi1fmp,t_maxleft(0),varfmpleft,t_maxright(0),varfmpright);
      double p_koh=post_koh_rec(theta,t_maxkoh(0),var_maxkoh);
      fprintf(out, "%e %e %e %e\n",theta,p_fmp,p_koh,p_bayes);    
    }
    fclose(out);

  }



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
  }
}
