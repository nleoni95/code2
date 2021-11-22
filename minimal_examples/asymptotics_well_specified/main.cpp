//exemple très simple où l'on cherche sur des problèmes simples à calculer la postérieure p(theta,psi) pour voir si elle est gaussienne.
//exemple de Tuo2015

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

tuple<vector<VectorXd>,vector<double>> Run_MCMC_adapt(int nsteps, VectorXd const & Xinit, MatrixXd const & COV_init,function<double(VectorXd const &)> const & compute_score,function<bool(VectorXd)> const & in_bounds,double lambda, double gamma, default_random_engine & generator){
  //algorithme andrieu global AM with global adaptive scaling (algorithme 4)
  //paramètres initiaux conseillés :     double lambda=pow(2.38,2)/dim_mcmc; double gamma=0.01;
  cout << "running mcmc adapt with " << nsteps << " steps, adaptative algorithm, gamma = "<< gamma << endl;
  double alphastar=0.234; //valeur conseillée dans l'article. taux d'acceptation optimal.
  vector<VectorXd> allsamples;
  vector<double> scores_of_samples;
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  int dim_mcmc=Xinit.size();
   auto draw_prop=[dim_mcmc](double lambda,MatrixXd COV,default_random_engine & generator, normal_distribution<double> & distN ){
    //tire une proposal de matrice de cov sqrt(lambda)*COV*sqrt(lambda)
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    MatrixXd sqrtCOV=COV.llt().matrixL();
    VectorXd s=sqrt(lambda)*sqrtCOV*Step;
    return s;
  };

  auto update_params=[dim_mcmc,gamma,alphastar](VectorXd & mu, MatrixXd & COV,double & lambda,double alpha,VectorXd Xcurrent){
    //update les paramètres de l'algo MCMC.
    lambda*=exp(gamma*(alpha-alphastar));
    COV=COV+gamma*((Xcurrent-mu)*(Xcurrent-mu).transpose()-COV);
    COV+=1e-10*MatrixXd::Identity(dim_mcmc,dim_mcmc);
    mu=mu+gamma*(Xcurrent-mu);
  };

  MatrixXd COV=COV_init;
  cout << "cov : " <<COV << endl;
  VectorXd mu=Xinit;
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  double lambdan=lambda;
  auto begin=chrono::steady_clock::now();
  double alpha=0;
  for(int i=0;i<nsteps;i++){
    VectorXd Xcandidate=Xcurrent+draw_prop(lambdan,COV,generator,distN);
    //cout << "candidate : " << Xcandidate.transpose() << endl;
    if(in_bounds(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      alpha=min(exp(fcandidate-fcurrent),1.);
      double c=distU(generator);
      if(alpha>=c){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(isnan(alpha)){alpha=1e-20;};
    update_params(mu,COV,lambdan,alpha,Xcurrent);
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
  COV_init(0,0)=pow(0.8,2); //pour KOH separate : 1e-2 partout fonctionne bien.
  COV_init(1,1)=pow(0.5,2);
  COV_init(2,2)=pow(0.5,2);

  //fonctions pour obtenir des observations.

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

  //grid initial dans [0,1]
  int nobs=20; 
  VectorXd initgrid(nobs);
  for(int i=0;i<nobs;i++){
    double x=i*1.0/nobs;
    initgrid(i)=x;
  }
  //FD: fixed domain, ID: increasing domain, RD: repeated domain
  //grid2FD
  VectorXd grid2FD(nobs*2);
  for(int i=0;i<nobs*2;i++){
    double x=i*1.0/(nobs*2);
    grid2FD(i)=x;
  }
  //grid3FD
  VectorXd grid3FD(nobs*4);
  for(int i=0;i<nobs*4;i++){
    double x=i*1.0/(nobs*4);
    grid3FD(i)=x;
  }  
  //grid4FD
  VectorXd grid4FD(nobs*8);
  for(int i=0;i<nobs*8;i++){
    double x=i*1.0/(nobs*8);
    grid4FD(i)=x;
  }
  //grid2ID. On va jusque dans [0,2]
  VectorXd grid2ID(nobs*2);
  for(int i=0;i<nobs*2;i++){
    double x=i*2.0/(nobs*2);
    grid2ID(i)=x;
  }
  //grid3ID. On va jusque dans [0,4]
  VectorXd grid3ID(nobs*4);
  for(int i=0;i<nobs*4;i++){
    double x=i*4.0/(nobs*4);
    grid3ID(i)=x;
  }  
  //grid4ID. On va jusque dans [0,8]
  VectorXd grid4ID(nobs*8);
  for(int i=0;i<nobs*8;i++){
    double x=i*8.0/(nobs*8);
    grid4ID(i)=x;
  }

  //on tire les observations sur le grid increasing max, déjà.
  int seed=42; //seed pour les observations
  default_random_engine generator(seed);
  VectorXd yobs_4ID(nobs*8);
  for(int i=0;i<8*nobs;i++){
    double y=true_fct_scalar(grid4ID(i))+noise*distN(generator);
    yobs_4ID(i)=y;
  }
  //on les range ensuite dans les grids 2ID puis init
  VectorXd yobs_3ID(nobs*4);
  VectorXd yobs_2ID(nobs*2);
  VectorXd yobs_init(nobs);
  for(int i=0;i<nobs;i++){
    yobs_init(i)=yobs_4ID(i);
    yobs_2ID(i)=yobs_4ID(i);
    yobs_2ID(i+nobs)=yobs_4ID(i+nobs);    
    yobs_3ID(i)=yobs_4ID(i);
    yobs_3ID(i+nobs)=yobs_4ID(i+nobs);
    yobs_3ID(i+2*nobs)=yobs_4ID(i+2*nobs);
    yobs_3ID(i+3*nobs)=yobs_4ID(i+3*nobs);
  }
  //on les range dans les grids denses et on tire le reste.
  VectorXd yobs_2FD(2*nobs);
  VectorXd yobs_3FD(4*nobs);
  VectorXd yobs_4FD(8*nobs);
  for(int i=0;i<8*nobs;i++){
    if(i%8==0){
      //observations provenant du grid init
      yobs_2FD(i/4)=yobs_init(i/8);
      yobs_3FD(i/2)=yobs_init(i/8);
      yobs_4FD(i)=yobs_init(i/8);
    }
    else if (i%8==4){
      //tirer l'observation et la mettre dans les 2 grids
      double y=true_fct_scalar(grid4FD(i))+noise*distN(generator);
      yobs_2FD(i/4)=y;
      yobs_3FD(i/2)=y;
      yobs_4FD(i)=y;
    }
    else if (i%2==0){
      double y=true_fct_scalar(grid4FD(i))+noise*distN(generator);
      yobs_3FD(i/2)=y;
      yobs_4FD(i)=y;
    }
    else{
      //obs du grid fin
      double y=true_fct_scalar(grid4FD(i))+noise*distN(generator);
      yobs_4FD(i)=y;
    }
  }

  //tirage des observations du RD.
  vector<VectorXd> yobs_RD8(8);
  yobs_RD8[0]=yobs_init;
  for(int i=0;i<7;i++){
    VectorXd yobs_1RD(nobs);
      for(int j=0;j<nobs;j++){
        double y=true_fct_scalar(initgrid(j))+noise*distN(generator);
        yobs_1RD(j)=y;
      }
    yobs_RD8[i+1]=yobs_1RD;  
  }

auto run_analysis=[noise,&generator,nombre_steps_mcmc,nombre_samples_collected,lb_t,ub_t,lb_hpars,ub_hpars,dim_theta,dim_hpars,lambda_model,X_init,COV_init](string foldname,VectorXd const &Xobs,VectorXd const &Yobs){
    //fonction qui trouve un échantillon de la postérieure à partir du grid, des observations, et écrit tout dans un dossier.
    int nobs=Xobs.size();
    string endname=".gnu";
    int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
    

    int samp_size=80; //80 avant
    VectorXd X_predictions(samp_size); for (int i=0;i<samp_size;i++){X_predictions(i)=Xobs(nobs-1)*double(i)/double(samp_size);}  

    //pour la MCMC

    int nautocor=500;
    int nsel=nombre_samples_collected; //c'est la même variable avec 2 noms


  



    auto logprior_pars=[](VectorXd const &p){
      return 0;
    };

    auto logprior_hpars=[](VectorXd const &h){
      return 0;
    };
    auto lambda_priormean=[](VectorXd const &X,VectorXd const &h){
      return VectorXd::Zero(X.size());
    };

    vector<AUGDATA> augdata; //remplir les données ici.

    AUGDATA a; a.SetX(Xobs); a.SetValue(Yobs);
    augdata.push_back(a);
    //calcul des samples bayes et de l'approximation gaussienne/
    DoE doe_init(lb_t,ub_t,200,10); // DoE Halton
    string fname_data=foldname+"obs"+endname;
    WriteObs(fname_data,augdata);

    Density Dens(doe_init);
    Dens.SetModel(lambda_model);
    Dens.SetKernel(kernel); //n'utilisons pas les dérivées pour ce cas.
    Dens.SetHparsBounds(lb_hpars,ub_hpars);
    Dens.SetLogPriorHpars(logprior_hpars);
    Dens.SetLogPriorPars(logprior_pars);
    Dens.SetPriorMean(lambda_priormean);



    Dens.SetDataExp(augdata);
    Dens.SetXprofile(augdata[0].GetX()); 


    //phase Full Bayes
    
      //on a écrit une fonction de MCMC. Il faut juste lui donner la fonction de scoring.
      // c'est là-dedans qu'on va sommer sur la dimension de vecteur des observations ? 
      auto scoring_function=[Dens,dim_theta,dim_hpars](VectorXd const & X){
        VectorXd theta(dim_theta); theta=X.head(dim_theta);
        VectorXd hpars(dim_hpars); hpars=X.tail(dim_hpars);
        return Dens.loglikelihood_theta(theta,hpars); //priors uniformes
      };

      auto in_bounds=[lb_hpars,ub_hpars,lb_t,ub_t,dim_theta,dim_hpars](VectorXd const &X){
        VectorXd theta=X.head(dim_theta);
        VectorXd hpars=X.tail(dim_hpars);
        for(int i=0;i<dim_theta;i++){
          if(theta(i)>ub_t(i) || theta(i)<lb_t(i)){
            return false;
          }
        }
        for(int i=0;i<dim_hpars;i++){
          if(hpars(i)>ub_hpars(i) || hpars(i)<lb_hpars(i)){
            return false;
          }
        }
        return true;
      };
      double lambda=pow(2.38,2)/2;
      double gamma=0.01;
      auto res=Run_MCMC(nombre_steps_mcmc,X_init,COV_init,scoring_function,in_bounds,generator); //tuple qui contient tous les samples et 
      //auto res=Run_MCMC_adapt(nombre_steps_mcmc,X_init,COV_init,scoring_function,in_bounds,lambda,gamma,generator); //tuple qui contient tous les samples et toutes les données.
      //sélection des samples 
      tuple<vector<VectorXd>,vector<double>> selected_res;
      vector<VectorXd> selected_samples;
      vector<double> selected_ll;
      //sélection uniforme
      /*
      for(int i=0;i<nsel;i++){
        int id=i*get<0>(res).size()/nsel;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      */
      //on prend la seconde partie des samples de la chaîne.
      for(int i=0;i<nsel;i++){
        int id=get<0>(res).size()-1-i;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      
      vector<VectorXd> sel_thetas(nsel);
      vector<VectorXd> sel_hpars(nsel);  
      for(int i=0;i<nsel;i++){
        VectorXd X=selected_samples[i];
        sel_thetas[i]=X.head(dim_theta);
        sel_hpars[i]=X.tail(dim_hpars);
      }
      //on met dans la densité
      Dens.SetNewAllSamples(get<0>(res));
      Dens.SetNewSamples(sel_thetas);
      Dens.SetNewHparsOfSamples(sel_hpars);
      //vérification du mélange de la MCMC

      string fname=foldname+"autocor"+endname;
      Dens.Autocor_diagnosis(nautocor,fname);

      // écriture des samples et aussi quelques prédictions pour test visuel
      Dens.WriteSamples(foldname+"samples"+endname);
      Dens.WritePredictions(X_predictions,foldname+"preds"+endname);

      //fit d'une gaussienne
      auto tp_gauss=GaussFit(selected_samples);
      cout << "gaussfit : " << get<0>(tp_gauss).transpose() << endl; //moyenne
      cout << get<1>(tp_gauss) << endl; //covariance matrix

      //écriture dans un fichier
      fname=foldname+"gaussfit"+endname;
      ofstream ofile(fname);
      ofile << " moyenne gaussienne : " << endl << get<0>(tp_gauss).transpose() << endl;
      ofile << "stds : " << sqrt(get<1>(tp_gauss)(0,0)) << " " << sqrt(get<1>(tp_gauss)(1,1)) << " "<< sqrt(get<1>(tp_gauss)(2,2)) << endl;
      ofile << "matrice de variance : " << endl << get<1>(tp_gauss) << endl;

      ofile.close();

      //calcul des quantiles normalisés et affichage dans un fichier
      MatrixXd Quant=QQplot(selected_samples,generator);
      fname=foldname+"qqplot"+endname;
      ofile.open(fname);
      for(int i=0;i<Quant.rows();i++){
        for(int j=0;j<Quant.cols();j++){
          ofile << Quant(i,j) << " ";
        }
        ofile << endl;
      }
      ofile.close();

      fname=foldname+"obs"+endname;
      //WriteObs(fname,Xobs,Yobs);
    
    return make_tuple(get<0>(tp_gauss),get<1>(tp_gauss)); //renvoie les params du fit gaussien sur les samples.
  };
auto run_analysis_MCMCadapt=[noise,&generator,nombre_steps_mcmc,nombre_samples_collected,lb_t,ub_t,lb_hpars,ub_hpars,dim_theta,dim_hpars,lambda_model,X_init,COV_init](string foldname,VectorXd const &Xobs,VectorXd const &Yobs){
    //fonction qui trouve un échantillon de la postérieure à partir du grid, des observations, et écrit tout dans un dossier.
    int nobs=Xobs.size();
    string endname=".gnu";
    int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
    

    int samp_size=80; //80 avant
    VectorXd X_predictions(samp_size); for (int i=0;i<samp_size;i++){X_predictions(i)=Xobs(nobs-1)*double(i)/double(samp_size);}  

    //pour la MCMC

    int nautocor=500;
    int nsel=nombre_samples_collected; //c'est la même variable avec 2 noms

 
  


    auto logprior_pars=[](VectorXd const &p){
      return 0;
    };

    auto logprior_hpars=[](VectorXd const &h){
      return 0;
    };
    auto lambda_priormean=[](VectorXd const &X,VectorXd const &h){
      return VectorXd::Zero(X.size());
    };

    vector<AUGDATA> augdata; //remplir les données ici.

    AUGDATA a; a.SetX(Xobs); a.SetValue(Yobs);
    augdata.push_back(a);
    //calcul des samples bayes et de l'approximation gaussienne/
    DoE doe_init(lb_t,ub_t,200,10); // DoE Halton
    string fname_data=foldname+"obs"+endname;
    WriteObs(fname_data,augdata);

    Density Dens(doe_init);
    Dens.SetModel(lambda_model);
    Dens.SetKernel(kernel); //n'utilisons pas les dérivées pour ce cas.
    Dens.SetHparsBounds(lb_hpars,ub_hpars);
    Dens.SetLogPriorHpars(logprior_hpars);
    Dens.SetLogPriorPars(logprior_pars);
    Dens.SetPriorMean(lambda_priormean);



    Dens.SetDataExp(augdata);
    Dens.SetXprofile(augdata[0].GetX()); 


    //phase Full Bayes
    
      //on a écrit une fonction de MCMC. Il faut juste lui donner la fonction de scoring.
      // c'est là-dedans qu'on va sommer sur la dimension de vecteur des observations ? 
      auto scoring_function=[Dens,dim_theta,dim_hpars](VectorXd const & X){
        VectorXd theta(dim_theta); theta=X.head(dim_theta);
        VectorXd hpars(dim_hpars); hpars=X.tail(dim_hpars);
        return Dens.loglikelihood_theta(theta,hpars); //priors uniformes
      };

      auto in_bounds=[lb_hpars,ub_hpars,lb_t,ub_t,dim_theta,dim_hpars](VectorXd const &X){
        VectorXd theta=X.head(dim_theta);
        VectorXd hpars=X.tail(dim_hpars);
        for(int i=0;i<dim_theta;i++){
          if(theta(i)>ub_t(i) || theta(i)<lb_t(i)){
            return false;
          }
        }
        for(int i=0;i<dim_hpars;i++){
          if(hpars(i)>ub_hpars(i) || hpars(i)<lb_hpars(i)){
            return false;
          }
        }
        return true;
      };
      double lambda=pow(2.38,2)/2;
      double gamma=0.01;
      //auto res=Run_MCMC(nombre_steps_mcmc,X_init,COV_init,scoring_function,in_bounds,generator); //tuple qui contient tous les samples et 
      auto res=Run_MCMC_adapt(nombre_steps_mcmc,X_init,COV_init,scoring_function,in_bounds,lambda,gamma,generator); //tuple qui contient tous les samples et toutes les données.
      //sélection des samples 
      tuple<vector<VectorXd>,vector<double>> selected_res;
      vector<VectorXd> selected_samples;
      vector<double> selected_ll;
      //sélection uniforme
      /*
      for(int i=0;i<nsel;i++){
        int id=i*get<0>(res).size()/nsel;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      */
      //on prend la seconde partie des samples de la chaîne.
      for(int i=0;i<nsel;i++){
        int id=get<0>(res).size()-1-i;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      
      vector<VectorXd> sel_thetas(nsel);
      vector<VectorXd> sel_hpars(nsel);  
      for(int i=0;i<nsel;i++){
        VectorXd X=selected_samples[i];
        sel_thetas[i]=X.head(dim_theta);
        sel_hpars[i]=X.tail(dim_hpars);
      }
      //on met dans la densité
      Dens.SetNewAllSamples(get<0>(res));
      Dens.SetNewSamples(sel_thetas);
      Dens.SetNewHparsOfSamples(sel_hpars);
      //vérification du mélange de la MCMC

      string fname=foldname+"autocor"+endname;
      Dens.Autocor_diagnosis(nautocor,fname);

      // écriture des samples et aussi quelques prédictions pour test visuel
      Dens.WriteSamples(foldname+"samples"+endname);
      Dens.WritePredictions(X_predictions,foldname+"preds"+endname);

      //fit d'une gaussienne
      auto tp_gauss=GaussFit(selected_samples);
      cout << "gaussfit : " << get<0>(tp_gauss).transpose() << endl; //moyenne
      cout << get<1>(tp_gauss) << endl; //covariance matrix

      //écriture dans un fichier
      fname=foldname+"gaussfit"+endname;
      ofstream ofile(fname);
      ofile << " moyenne gaussienne : " << endl << get<0>(tp_gauss).transpose() << endl;
      ofile << "stds : " << sqrt(get<1>(tp_gauss)(0,0)) << " " << sqrt(get<1>(tp_gauss)(1,1)) << " "<< sqrt(get<1>(tp_gauss)(2,2)) << endl; 
      ofile << "matrice de variance : " << endl << get<1>(tp_gauss) << endl;


      ofile.close();

      //calcul des quantiles normalisés et affichage dans un fichier
      MatrixXd Quant=QQplot(selected_samples,generator);
      fname=foldname+"qqplot"+endname;
      ofile.open(fname);
      for(int i=0;i<Quant.rows();i++){
        for(int j=0;j<Quant.cols();j++){
          ofile << Quant(i,j) << " ";
        }
        ofile << endl;
      }
      ofile.close();

      fname=foldname+"obs"+endname;
      //WriteObs(fname,Xobs,Yobs);
    
    return make_tuple(get<0>(tp_gauss),get<1>(tp_gauss)); //renvoie les params du fit gaussien sur les samples.
  }; 
auto run_analysis_RD=[noise,&generator,nombre_steps_mcmc,nombre_samples_collected,lb_t,ub_t,lb_hpars,ub_hpars,dim_theta,dim_hpars,lambda_model,X_init,COV_init](string foldname,VectorXd const &Xobs,vector<VectorXd> const &Yobs_vect){
    //run_analysis mais avec plusieurs vecteurs d'observations.
    //fonction qui trouve un échantillon de la postérieure à partir du grid, des observations, et écrit tout dans un dossier.
    int nobs=Xobs.size();
    string endname=".gnu";
    int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.


    int samp_size=80; //80 avant
    VectorXd X_predictions(samp_size); for (int i=0;i<samp_size;i++){X_predictions(i)=Xobs(nobs-1)*double(i)/double(samp_size);}  

    //pour la MCMC

    int nautocor=500;
    int nsel=nombre_samples_collected; //c'est la même variable avec 2 noms


  
    auto logprior_pars=[](VectorXd const &p){
      return 0;
    };

    auto logprior_hpars=[](VectorXd const &h){
      return 0;
    };
    auto lambda_priormean=[](VectorXd const &X,VectorXd const &h){
      return VectorXd::Zero(X.size());
    };

    vector<Density> Dens_v;
    for(int i=0;i<Yobs_vect.size();i++){
      vector<AUGDATA> augdata; //remplir les données ici.
      AUGDATA a; a.SetX(Xobs); a.SetValue(Yobs_vect[i]);
      augdata.push_back(a);
      //calcul des samples bayes et de l'approximation gaussienne/
      DoE doe_init(lb_t,ub_t,200,10); // DoE Halton
      string fname_data=foldname+"obs"+endname;

      Density Dens(doe_init);
      Dens.SetModel(lambda_model);
      Dens.SetKernel(kernel); //n'utilisons pas les dérivées pour ce cas.
      Dens.SetHparsBounds(lb_hpars,ub_hpars);
      Dens.SetLogPriorHpars(logprior_hpars);
      Dens.SetLogPriorPars(logprior_pars);
      Dens.SetPriorMean(lambda_priormean);
      Dens.SetDataExp(augdata);
      Dens.SetXprofile(augdata[0].GetX()); 
      Dens_v.push_back(Dens);
    }


    //phase Full Bayes
    
      //on a écrit une fonction de MCMC. Il faut juste lui donner la fonction de scoring.
      // c'est là-dedans qu'on va sommer sur la dimension de vecteur des observations ? 
      auto scoring_function=[Dens_v,dim_theta,dim_hpars](VectorXd const & X){
        VectorXd theta(dim_theta); theta=X.head(dim_theta);
        VectorXd hpars(dim_hpars); hpars=X.tail(dim_hpars);
        double res=0;
        //cout << "res : ";
        for(int i=0;i<Dens_v.size();i++){
          //cout << d << "+";
          res+=Dens_v[i].loglikelihood_theta(theta,hpars);
        }
        //cout << endl << "res = " << res << endl;
        return res;
      };

      auto in_bounds=[lb_hpars,ub_hpars,lb_t,ub_t,dim_theta,dim_hpars](VectorXd const &X){
        VectorXd theta=X.head(dim_theta);
        VectorXd hpars=X.tail(dim_hpars);
        for(int i=0;i<dim_theta;i++){
          if(theta(i)>ub_t(i) || theta(i)<lb_t(i)){
            return false;
          }
        }
        for(int i=0;i<dim_hpars;i++){
          if(hpars(i)>ub_hpars(i) || hpars(i)<lb_hpars(i)){
            return false;
          }
        }
        return true;
      };
      //auto res=Run_MCMC(nombre_steps_mcmc,X_init,COV_init,scoring_function,in_bounds,generator); //tuple qui contient tous les samples et 
      auto res=Run_MCMC(nombre_steps_mcmc,X_init,COV_init,scoring_function,in_bounds,generator); //tuple qui contient tous les samples et toutes les données.
      //sélection des samples 
      tuple<vector<VectorXd>,vector<double>> selected_res;
      vector<VectorXd> selected_samples;
      vector<double> selected_ll;
      //sélection uniforme
      /*
      for(int i=0;i<nsel;i++){
        int id=i*get<0>(res).size()/nsel;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      */
      //on prend la seconde partie des samples de la chaîne.
      for(int i=0;i<nsel;i++){
        int id=get<0>(res).size()-1-i;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      
      vector<VectorXd> sel_thetas(nsel);
      vector<VectorXd> sel_hpars(nsel);  
      for(int i=0;i<nsel;i++){
        VectorXd X=selected_samples[i];
        sel_thetas[i]=X.head(dim_theta);
        sel_hpars[i]=X.tail(dim_hpars);
      }
      //on met dans la densité
      Density Dens=Dens_v[0];
      Dens.SetNewAllSamples(get<0>(res));
      Dens.SetNewSamples(sel_thetas);
      Dens.SetNewHparsOfSamples(sel_hpars);
      //vérification du mélange de la MCMC

      string fname=foldname+"autocor"+endname;
      Dens.Autocor_diagnosis(nautocor,fname);

      // écriture des samples et aussi quelques prédictions pour test visuel
      Dens.WriteSamples(foldname+"samples"+endname);
      Dens.WritePredictions(X_predictions,foldname+"preds"+endname);

      //fit d'une gaussienne
      auto tp_gauss=GaussFit(selected_samples);
      cout << "gaussfit : " << get<0>(tp_gauss).transpose() << endl; //moyenne
      cout << get<1>(tp_gauss) << endl; //covariance matrix

      //écriture dans un fichier
      fname=foldname+"gaussfit"+endname;
      ofstream ofile(fname);
      ofile << " moyenne gaussienne : " << endl << get<0>(tp_gauss).transpose() << endl;
      ofile << "stds : " << sqrt(get<1>(tp_gauss)(0,0)) << " " << sqrt(get<1>(tp_gauss)(1,1)) << " "<< sqrt(get<1>(tp_gauss)(2,2)) << endl;
      ofile << "matrice de variance : " << endl << get<1>(tp_gauss) << endl;

      ofile.close();

      //calcul des quantiles normalisés et affichage dans un fichier
      MatrixXd Quant=QQplot(selected_samples,generator);
      fname=foldname+"qqplot"+endname;
      ofile.open(fname);
      for(int i=0;i<Quant.rows();i++){
        for(int j=0;j<Quant.cols();j++){
          ofile << Quant(i,j) << " ";
        }
        ofile << endl;
      }
      ofile.close();

      fname=foldname+"obs"+endname;
      WriteObs(fname,Xobs,Yobs_vect);
    
    return make_tuple(get<0>(tp_gauss),get<1>(tp_gauss)); //renvoie les params du fit gaussien sur les samples.
  };
auto run_analysis_light=[noise,&generator,nombre_steps_mcmc,nombre_samples_collected,lb_t,ub_t,lb_hpars,ub_hpars,dim_theta,dim_hpars,lambda_model,X_init,COV_init](string foldname,VectorXd const &Xobs,VectorXd const &Yobs){
    //fonction qui trouve un échantillon de la postérieure à partir du grid, des observations, et écrit tout dans un dossier.
    int nobs=Xobs.size();
    string endname=".gnu";
    int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
    

    int samp_size=80; //80 avant
    VectorXd X_predictions(samp_size); for (int i=0;i<samp_size;i++){X_predictions(i)=Xobs(nobs-1)*double(i)/double(samp_size);}  

    //pour la MCMC

    int nautocor=500;
    int nsel=nombre_samples_collected; //c'est la même variable avec 2 noms


  

 
    auto logprior_pars=[](VectorXd const &p){
      return 0;
    };

    auto logprior_hpars=[](VectorXd const &h){
      return 0;
    };
    auto lambda_priormean=[](VectorXd const &X,VectorXd const &h){
      return VectorXd::Zero(X.size());
    };

    vector<AUGDATA> augdata; //remplir les données ici.

    AUGDATA a; a.SetX(Xobs); a.SetValue(Yobs);
    augdata.push_back(a);
    //calcul des samples bayes et de l'approximation gaussienne/
    DoE doe_init(lb_t,ub_t,200,10); // DoE Halton
    string fname_data=foldname+"obs"+endname;
    WriteObs(fname_data,augdata);

    Density Dens(doe_init);
    Dens.SetModel(lambda_model);
    Dens.SetKernel(kernel); //n'utilisons pas les dérivées pour ce cas.
    Dens.SetHparsBounds(lb_hpars,ub_hpars);
    Dens.SetLogPriorHpars(logprior_hpars);
    Dens.SetLogPriorPars(logprior_pars);
    Dens.SetPriorMean(lambda_priormean);



    Dens.SetDataExp(augdata);
    Dens.SetXprofile(augdata[0].GetX()); 


    //phase Full Bayes
    
      //on a écrit une fonction de MCMC. Il faut juste lui donner la fonction de scoring.
      // c'est là-dedans qu'on va sommer sur la dimension de vecteur des observations ? 
      auto scoring_function=[Dens,dim_theta,dim_hpars](VectorXd const & X){
        VectorXd theta(dim_theta); theta=X.head(dim_theta);
        VectorXd hpars(dim_hpars); hpars=X.tail(dim_hpars);
        return Dens.loglikelihood_theta(theta,hpars); //priors uniformes
      };

      auto in_bounds=[lb_hpars,ub_hpars,lb_t,ub_t,dim_theta,dim_hpars](VectorXd const &X){
        VectorXd theta=X.head(dim_theta);
        VectorXd hpars=X.tail(dim_hpars);
        for(int i=0;i<dim_theta;i++){
          if(theta(i)>ub_t(i) || theta(i)<lb_t(i)){
            return false;
          }
        }
        for(int i=0;i<dim_hpars;i++){
          if(hpars(i)>ub_hpars(i) || hpars(i)<lb_hpars(i)){
            return false;
          }
        }
        return true;
      };
      double lambda=pow(2.38,2)/2;
      double gamma=0.01;
      auto res=Run_MCMC(nombre_steps_mcmc,X_init,COV_init,scoring_function,in_bounds,generator); //tuple qui contient tous les samples et 
      //auto res=Run_MCMC_adapt(nombre_steps_mcmc,X_init,COV_init,scoring_function,in_bounds,lambda,gamma,generator); //tuple qui contient tous les samples et toutes les données.
      //sélection des samples 
      tuple<vector<VectorXd>,vector<double>> selected_res;
      vector<VectorXd> selected_samples;
      vector<double> selected_ll;
      //sélection uniforme
      /*
      for(int i=0;i<nsel;i++){
        int id=i*get<0>(res).size()/nsel;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      */
      //on prend la seconde partie des samples de la chaîne.
      for(int i=0;i<nsel;i++){
        int id=get<0>(res).size()-1-i;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      
      vector<VectorXd> sel_thetas(nsel);
      vector<VectorXd> sel_hpars(nsel);  
      for(int i=0;i<nsel;i++){
        VectorXd X=selected_samples[i];
        sel_thetas[i]=X.head(dim_theta);
        sel_hpars[i]=X.tail(dim_hpars);
      }
      //on met dans la densité
      Dens.SetNewAllSamples(get<0>(res));
      Dens.SetNewSamples(sel_thetas);
      Dens.SetNewHparsOfSamples(sel_hpars);
      //vérification du mélange de la MCMC

      string fname=foldname+"autocor"+endname;
      Dens.Autocor_diagnosis(nautocor,fname);

      // écriture des samples et aussi quelques prédictions pour test visuel
      Dens.WriteSamples(foldname+"samples"+endname);
      //fit d'une gaussienne
      auto tp_gauss=GaussFit(selected_samples);
      cout << "gaussfit : " << get<0>(tp_gauss).transpose() << endl; //moyenne
      cout << get<1>(tp_gauss) << endl; //covariance matrix

      //écriture dans un fichier
      fname=foldname+"gaussfit"+endname;
      ofstream ofile(fname);
      ofile << " moyenne gaussienne : " << endl << get<0>(tp_gauss).transpose() << endl;
      ofile << "stds : " << sqrt(get<1>(tp_gauss)(0,0)) << " " << sqrt(get<1>(tp_gauss)(1,1)) << " "<< sqrt(get<1>(tp_gauss)(2,2)) << endl;
      ofile << "matrice de variance : " << endl << get<1>(tp_gauss) << endl;

      ofile.close();

      //calcul des quantiles normalisés et affichage dans un fichier
      MatrixXd Quant=QQplot(selected_samples,generator);
      fname=foldname+"qqplot"+endname;
      ofile.open(fname);
      for(int i=0;i<Quant.rows();i++){
        for(int j=0;j<Quant.cols();j++){
          ofile << Quant(i,j) << " ";
        }
        ofile << endl;
      }
      ofile.close();

      fname=foldname+"obs"+endname;
      //WriteObs(fname,Xobs,Yobs);
    
    return make_tuple(get<0>(tp_gauss),get<1>(tp_gauss)); //renvoie les params du fit gaussien sur les samples.
  };
auto draw_pts=[&generator](vector<int> pts_depart, int taille_finale){
    //pour retirer un certain nombre de points d'une liste d'indices.
    uniform_real_distribution<double> distU(0,1);
    vector<int> pts_arrivee=pts_depart;
    while(pts_arrivee.size()>taille_finale){
      int ind_remove=pts_arrivee.size()*distU(generator);
      pts_arrivee.erase(pts_arrivee.begin()+ind_remove);
    };
  return pts_arrivee;
  };

auto create_obsFD_total=[true_fct_scalar,noise,&generator,draw_pts](int nobs){
    //on remplit le grix au max
    VectorXd gridmaxFD(nobs*45);
    VectorXd yobsmaxFD(nobs*45);
    for(int i=0;i<nobs*45;i++){
      double x=i*1.0/(nobs*45);
      gridmaxFD(i)=x;
      yobsmaxFD(i)=true_fct_scalar(gridmaxFD(i))+noise*distN(generator);
    }
    vector<int> ind45(nobs*45); for(int i=0;i<ind45.size();i++){ind45[i]=i;}
    vector<int> ind30=draw_pts(ind45,nobs*30);
    VectorXd grid6FD(nobs*30);
    VectorXd yobs6FD(nobs*30);
    //mettre les nouveaux pts
    for(int i=0;i<nobs*30;i++){grid6FD(i)=gridmaxFD(ind30[i]); yobs6FD(i)=yobsmaxFD(ind30[i]);}
    //tirer les points à enlever
    for(int i=0;i<nobs*30;i++){ind30[i]=i;}
    vector<int> ind15=draw_pts(ind30,nobs*15);
    VectorXd grid5FD(nobs*15);
    VectorXd yobs5FD(nobs*15);
    //mettre les nouveaux pts
    for(int i=0;i<nobs*15;i++){grid5FD(i)=grid6FD(ind15[i]); yobs5FD(i)=yobs6FD(ind15[i]);}
    for(int i=0;i<nobs*15;i++){ind15[i]=i;}
    vector<int> ind=draw_pts(ind15,nobs);
    VectorXd grid4FD(nobs*5);
    VectorXd yobs4FD(nobs*5);
    for(int i=0;i<nobs*5;i++){grid4FD(i)=grid5FD(ind[i]); yobs4FD(i)=yobs5FD(ind[i]);}    
    return make_tuple(grid4FD,yobs4FD,grid5FD,yobs5FD,grid6FD,yobs6FD,gridmaxFD,yobsmaxFD);
  };
auto create_obsID_total=[true_fct_scalar,noise,&generator](int nobs){
    //on remplit le grix au max
    VectorXd gridmaxID(nobs*45);
    VectorXd yobsmaxID(nobs*45);
    for(int i=0;i<nobs*45;i++){
      double x=i*45.0/(nobs*45);
      gridmaxID(i)=x;
      yobsmaxID(i)=true_fct_scalar(gridmaxID(i))+noise*distN(generator);
    }
    VectorXd grid6ID(nobs*30);
    VectorXd yobs6ID(nobs*30);
    //mettre les nouveaux pts
    for(int i=0;i<nobs*30;i++){grid6ID(i)=gridmaxID(i); yobs6ID(i)=yobsmaxID(i);}
    //tirer les points à enlever
    VectorXd grid5ID(nobs*15);
    VectorXd yobs5ID(nobs*15);
    //mettre les nouveaux pts
    for(int i=0;i<nobs*15;i++){grid5ID(i)=grid6ID(i); yobs5ID(i)=yobs6ID(i);}
    VectorXd grid4ID(nobs*5);
    VectorXd yobs4ID(nobs*5);
    for(int i=0;i<nobs*5;i++){grid4ID(i)=grid5ID(i); yobs4ID(i)=yobs5ID(i);}    
    return make_tuple(grid4ID,yobs4ID,grid5ID,yobs5ID,grid6ID,yobs6ID,gridmaxID,yobsmaxID);
  };
auto create_obsRD_total=[true_fct_scalar,noise,&generator](int nobs){
    //on remplit le grix au max
    VectorXd grid(nobs);
    vector<VectorXd> yobsmaxRD(45);
    vector<VectorXd> yobs6RD(30);
    vector<VectorXd> yobs5RD(15);
    vector<VectorXd> yobs4RD(5);
    for(int i=0;i<nobs;i++){
      double x=i*1.0/(nobs);
      grid(i)=x;
    }
    for(int j=0;j<45;j++){
      VectorXd y(nobs);
      for(int i=0;i<nobs;i++){
        y(i)=true_fct_scalar(grid(i))+noise*distN(generator);
      }
      yobsmaxRD[j]=y;
    }
    for(int j=0;j<30;j++){yobs6RD[j]=yobsmaxRD[j];}
    for(int j=0;j<15;j++){yobs5RD[j]=yobsmaxRD[j];}
    for(int j=0;j<5;j++){yobs4RD[j]=yobsmaxRD[j];}
    return make_tuple(grid,yobs4RD,yobs5RD,yobs6RD,yobsmaxRD);
  };
auto runanal=[run_analysis_light,&generator](VectorXd const & grid, VectorXd const& obs,string foldname){
    auto tp=run_analysis_light(foldname,grid,obs);
    double t=get<0>(tp)(0);
    double s=get<0>(tp)(1);
    double tstd=get<1>(tp)(0,0);
    double sstd=get<1>(tp)(1,1);
    cout << " t : " << t << " " << sqrt(tstd) << endl;
    cout << " s : " << s << " " << sqrt(sstd) << endl;
  };
auto runanal_RD=[run_analysis_RD,&generator](VectorXd const & grid, vector<VectorXd> const& obs,string foldname){
    auto tp=run_analysis_RD(foldname,grid,obs);
    double t=get<0>(tp)(0);
    double s=get<0>(tp)(1);
    double tstd=get<1>(tp)(0,0);
    double sstd=get<1>(tp)(1,1);
    cout << " t : " << t << " " << sqrt(tstd) << endl;
    cout << " s : " << s << " " << sqrt(sstd) << endl;
  };
  

  
  generator.seed(56974364);

   auto tup=create_obsFD_total(nobs);
  string fname="results/4FD/";
  runanal(get<0>(tup),get<1>(tup),fname);
  

  cout << "FD5" << endl;
  fname="results/5FD/";
  runanal(get<2>(tup),get<3>(tup),fname);

  cout << "FD6" << endl;
  fname="results/6FD/";
  runanal(get<4>(tup),get<5>(tup),fname);

  cout << "FD7" << endl;
  fname="results/7FD/";
  runanal(get<6>(tup),get<7>(tup),fname);
  

  auto tupID=create_obsID_total(nobs);
  cout << "ID4" << endl;
  fname="results/4ID/";
  runanal(get<0>(tupID),get<1>(tupID),fname);

  cout << "ID5" << endl;
  fname="results/5ID/";
  runanal(get<2>(tupID),get<3>(tupID),fname);

  cout << "ID6" << endl;
  fname="results/6ID/";
  runanal(get<4>(tupID),get<5>(tupID),fname);

  cout << "ID7" << endl;
  fname="results/7ID/";
  runanal(get<6>(tupID),get<7>(tupID),fname);
  /*

  auto tupRD=create_obsRD_total(nobs);
  cout << "RD4" << endl;
  fname="results/4RD/";
  runanal_RD(get<0>(tupRD),get<1>(tupRD),fname);

  cout << "RD5" << endl;
  fname="results/5RD/";
  runanal_RD(get<0>(tupRD),get<2>(tupRD),fname);

  cout << "RD6" << endl;
  fname="results/6RD/";
  runanal_RD(get<0>(tupRD),get<3>(tupRD),fname);

  cout << "RD7" << endl;
  fname="results/7RD/";
  runanal_RD(get<0>(tupRD),get<4>(tupRD),fname);
  */

  exit(0);


  string foldname="results/init/";
  auto tp=run_analysis(foldname,initgrid,yobs_init);
  double t=get<0>(tp)(0);
  double s=get<0>(tp)(1);
  double tstd=get<1>(tp)(0,0);
  double sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;

  foldname="results/2ID/";
  tp=run_analysis(foldname,grid2ID,yobs_2ID);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;
/*
  foldname="results/3ID/";
  tp=run_analysis_MCMCadapt(foldname,grid3ID,yobs_3ID);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;


  foldname="results/4ID/";
  tp=run_analysis_MCMCadapt(foldname,grid4ID,yobs_4ID);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;
  
*/
  foldname="results/2FD/";
  tp=run_analysis(foldname,grid2FD,yobs_2FD);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;
/*

  foldname="results/3FD/";
  tp=run_analysis_MCMCadapt(foldname,grid3FD,yobs_3FD);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;


  foldname="results/4FD/";
  tp=run_analysis_MCMCadapt(foldname,grid4FD,yobs_4FD);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;
  

  vector<VectorXd> yobs_RD2(2);
  vector<VectorXd> yobs_RD4(4);
  for(int i=0;i<2;i++){
    yobs_RD2[i]=yobs_RD8[i];
    yobs_RD4[i]=yobs_RD8[i];
    yobs_RD4[i+2]=yobs_RD8[i+2];
  }

  foldname="results/2RD/";
  tp=run_analysis_RD(foldname,initgrid,yobs_RD2);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;


  foldname="results/3RD/";
  tp=run_analysis_RD(foldname,initgrid,yobs_RD4);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;


  foldname="results/4RD/";
  tp=run_analysis_RD(foldname,initgrid,yobs_RD8);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;
  
*/
  exit(0);

}

