//sur cet exemple, on fait le MCMC amélioré où les hpars du point candidats sont utilisés pour l'étape. On veut montrer que la postérieure obtenue est meilleure que la méthode Opti. Pourquoi pas faire sur l'exemple 1 de l'article ? comme ça on le montre et aussi on voit l'échec KOH.
// On fait une phase de construction des surrogates des hyperparamètres, puis un KOH, un MCMC avec opti et sans opti, et un MCMC full.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <functional>
#include "pcb++.h"
#include "cub++.h"
#include "vstoch++.h"
#include "mstoch++.h"
#include "sampler.h"
#include <ctime>
#include "densities.h"


using namespace std;
using namespace Eigen;
//pour stocker les valeurs calculées d'hyperparamètres optimaux. On a un vectorxd qui est le theta, et une map (int vectorxd) qui permet de retrouver l'hyperparamètre optimal (vectorxd) du cas i (int)

int neval=1;
//std::default_random_engine generator;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);
vector<DATA> data;
vector<VectorXd> Grid;




double gaussprob(double x,double mu, double sigma){
  //renvoie la probabilité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(2),2)); //3/2
}

double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2)))*exp(-d/hpar(2)); //3/2
}


double kernel(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double d=abs(x(0)-y(0));
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(1),2)); //3/2
}

VectorXd XtoLX(VectorXd const &X){
  //pour faire la chaîne en log sigma.
  VectorXd LX(3);
  LX(0)=X(0);
  LX(1)=log(X(1));
  LX(2)=X(2);
  return LX;
}

VectorXd LXtoX(VectorXd const &LX){
  //pour faire la chaîne en log sigma.
  VectorXd X(3);
  X(0)=LX(0);
  X(1)=exp(LX(1));
  X(2)=LX(2);
  return X;
}

void Run_Burn_Phase_MCMC(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento,function<double(VectorXd const &)> const & compute_score,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //phase de burn.
  int dim_mcmc=COV_init.cols();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  VectorXd Xinit=Xcurrento;
  MatrixXd COV=COV_init;
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(dim_mcmc);
  MatrixXd acc_var=MatrixXd::Zero(dim_mcmc,dim_mcmc);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(dim_mcmc))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
}

tuple<vector<VectorXd>,vector<double>> Run_MCMC(int nsteps,VectorXd const & Xinit,MatrixXd const & COV_init,function<double(VectorXd const &)> const & compute_score,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //MCMC générique. renvoie un tuple <samples,hparsofsamples,allsamples>
  int dim_mcmc=Xinit.size();
  vector<VectorXd> allsamples;
  vector<double> scores_of_samples;
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  VectorXd Xinit0=Xinit;
  Run_Burn_Phase_MCMC(nsteps*0.1,COV,Xinit0,compute_score,in_bounds,generator);
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(Xinit0);
  VectorXd Xcurrent=Xinit0;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    allsamples.push_back(Xcurrent);
    scores_of_samples.push_back(fcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;

  cout << allsamples[0].transpose() << endl;
  cout << allsamples[nsteps/2].transpose() << endl;
  auto tp=make_tuple(allsamples,scores_of_samples);
  return tp;
}

void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
}


void WriteObs(string filename,vector<AUGDATA> &augdata){
  ofstream ofile(filename);
  AUGDATA a=augdata[0];
  VectorXd x=a.GetX();
  VectorXd y=a.Value();
  for(int i=0;i<x.size();i++){
    ofile << x(i) << " " << y(i) << endl;
  }
  ofile.close();
}

tuple<VectorXd,MatrixXd> GaussFit(vector<VectorXd> const & samples){
  //fit d'une gaussienne multivariée sur un sample.
  int d=samples[0].size();
  VectorXd mean=VectorXd::Zero(d);
  MatrixXd SecondMoment=MatrixXd::Zero(d,d);
  for_each(samples.begin(),samples.end(),[&SecondMoment,&mean](VectorXd const &x)mutable{
    mean+=x;
    SecondMoment+=x*x.transpose();
  });
  mean/=samples.size();
  MatrixXd Var=SecondMoment/samples.size()-mean*mean.transpose();
  auto tp=make_tuple(mean,Var);
  return tp;
}

tuple<double,double> GaussFit(vector<double> const & samples){
  //fit d'une gaussienne multivariée sur un sample.
  double mean=0;
  double SecondMoment=0;
  for(int i=0;i<samples.size();i++){
    double x=samples[i];
    mean+=x;
    SecondMoment+=pow(x,2);
  }
  mean/=samples.size();
  double Var=SecondMoment/samples.size()-pow(mean,2);
  auto tp=make_tuple(mean,Var);
  return tp;
}

MatrixXd QQplot(vector<VectorXd> const & samples, default_random_engine & generator){
  //calcul d'un QQ plot. On rend une matrice avec samples(0).size() colonnes et autant de lignes que du nombre de quantiles choisi.
  //si j'ai bien compris, la première colonne est les quantiles de la loi normale. chaque colonne ensuite correspond aux quantiles du premier paramètre, du premier hpar, etc.
  //on met le tout dans un vector car je ne sais faire les QQplot qu'une dimension à la fois.
  int nquantiles=50; //on choisit de calculer 50 quantiles
  normal_distribution<double> distN(0,1);
  int ndim=samples[0].size();
  MatrixXd res(nquantiles,ndim+1);
  //tirage d'un échantillon de loi normale 1D
  vector<double> sample_normal(samples.size());
  transform(sample_normal.begin(),sample_normal.end(),sample_normal.begin(),[&generator,&distN](double d){
    return distN(generator);
  });
  sort(sample_normal.begin(),sample_normal.end());
  VectorXd quant_normal(nquantiles);
  for(int i=0;i<nquantiles;i++){
    double q=(i+0.5)/(1.0*nquantiles); // on ne prend ni le quantile 0 ni le quantile 100
    int n=q*sample_normal.size();
    quant_normal(i)=sample_normal[n];
  }
  res.col(0)=quant_normal;
  for(int j=0;j<ndim;j++){
    //création du sample réduit
    vector<double> sample_1D(samples.size()); for(int i=0;i<samples.size();i++){sample_1D[i]=samples[i](j);}
    //on centre, on réduit et on trie
    auto tpg=GaussFit(sample_1D);
    double m=get<0>(tpg);
    double s=sqrt(get<1>(tpg));
    transform(sample_1D.begin(),sample_1D.end(),sample_1D.begin(),[m,s](double x){
      double r=(x-m)/s;
      return r;
    });
    sort(sample_1D.begin(),sample_1D.end());
    VectorXd quant_1D(nquantiles);
    for(int i=0;i<nquantiles;i++){
      double q=(i+0.5)/(1.0*nquantiles); // on ne prend ni le quantile 0 ni le quantile 100
      int n=q*sample_1D.size();
      quant_1D(i)=sample_1D[n];
    }
    //on met les deux vecteurs de quantiles dans une même matrice. quantiles théoriques d'abord.
    MatrixXd M(nquantiles,2); M.col(0)=quant_normal; M.col(1)=quant_1D;
    res.col(j+1)=quant_1D;
  }
  return res;
}

void WriteObs(string filename,VectorXd const &x,VectorXd const &y){
  ofstream ofile(filename);
  for(int i=0;i<x.size();i++){
    ofile << x(i) << " " << y(i) << endl;
  }
  ofile.close();
}

void WriteObs(string filename,VectorXd const &x,vector<VectorXd> const &y_vect){
  ofstream ofile(filename);
  for(int i=0;i<x.size();i++){
    ofile << x(i) << " ";
      for(VectorXd const &y:y_vect){
        ofile << y(i) << " ";
      }
     ofile << endl;
  }
  ofile.close();
}



const double Big = -1.e16;


int main(int argc, char **argv){
  
  double noise=sqrt(0.1); //sigma noise
  int nombre_steps_mcmc=5e5;
  int nombre_samples_collected=5e4; //on garde la moitié des samples de la mcmc
  int dim_theta=1;
  int dim_hpars=2;
  
    // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(dim_theta); lb_t(0)=-2; //0.2 avant
  VectorXd ub_t(dim_theta); ub_t(0)=2;

  VectorXd lb_hpars(dim_hpars); lb_hpars(0)=0;lb_hpars(1)=1e-2;
  VectorXd ub_hpars(dim_hpars); ub_hpars(0)=3;ub_hpars(1)=20;

  VectorXd X_init(dim_theta+dim_hpars);
  X_init(0)=-1;
  //X_init(1)=1e-2;
  //X_init(2)=1;
  X_init.tail(dim_hpars)=0.5*(lb_hpars+ub_hpars);
  MatrixXd COV_init=MatrixXd::Identity(dim_theta+dim_hpars,dim_theta+dim_hpars);
  COV_init(0,0)=2e-2; //pour KOH separate : 1e-2 partout fonctionne bien.
  COV_init(1,1)=1e-1;
  COV_init(2,2)=1.4;

  //true function
  auto true_fct_scalar=[](double x){
    double x0=x*2*M_PI;
    //true underlying process
    return sin(x0)*exp(x0/10);
  };

  auto lambda_model=[](VectorXd const & Xprofile, VectorXd const & theta){
      //le vecteur Xprofile contient tous les x scalaires. Il faut renvoyer une prédiction de même taille que Xprofile.
      VectorXd res(Xprofile.size());
      for(int i=0;i<res.size();i++){
        double x=Xprofile(i)*2*M_PI;
        res(i)=sin(x)*exp(x/10)-abs(theta(0)+1)*(sin(theta(0)*x)+cos(theta(0)*x));
      }
      return res;
    };

  //remplissage des observations


  auto logprior_pars=[](VectorXd const &p){
      return 0;
    };

    auto logprior_hpars=[](VectorXd const &h){
      return 0;
    };
    auto lambda_priormean=[](VectorXd const &X,VectorXd const &h){
      return VectorXd::Zero(X.size());
    };


DoE doe_init(lb_t,ub_t,200,10); // DoE Halton
 string fname_data=foldname+"obs"+endname;
 WriteObs(fname_data,augdata);


  //phase KOH
  Density Dens(doe_init);
  Dens.SetModel(lambda_model);
  Dens.SetKernel(kernel); //n'utilisons pas les dérivées pour ce cas.
  Dens.SetHparsBounds(lb_hpars,ub_hpars);
  Dens.SetLogPriorHpars(logprior_hpars);
  Dens.SetLogPriorPars(logprior_pars);
  Dens.SetPriorMean(lambda_priormean);

  Dens.SetDataExp(augdata);
  Dens.SetXprofile(augdata[0].GetX()); 
  //hpars koh

  auto scoring_fct_koh=[](){

  };


  //construction des hyperparamètres optimaux
  //faire un grid et des évaluations manuelles des hpars optimaux, non ?


  //phase Full Bayes


  //phase Opti
  // construction des hyperparamètres optimaux





  generator.seed(42);
  string foldname="results/init/";
  auto tp=run_analysis(foldname,initgrid,yobs_init);
  double t=get<0>(tp)(0);
  double s=get<0>(tp)(1);
  double tstd=get<1>(tp)(0,0);
  double sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;

  exit(0);

}

