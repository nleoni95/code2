// Dans ce fichier : un petit exemple illustratif de la fausse confiance de KOH.
// travaillons à bruit expérimental fixé. 


#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <random>
#include <functional>
#include <iterator>
#include <chrono>
#include "densities.h"
#include "pcb++.h"
#include "cub++.h"
#include "vstoch++.h"
#include "mstoch++.h"


using namespace std;
using namespace Eigen;

typedef map<int,VectorXd> map_doe; //key : int, value : vectorXd
typedef map<int,vector<VectorXd>> map_results; //key : int, value : vecteur de VectorXd
typedef map<string,VectorXd> map_exp; //contient les valeurs expérimentales

int neval=1;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);

double const flux_nominal=128790;

int dim_theta=5;

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(1),2));
  //return pow(hpar(0),2)*(1+(d/hpar(2))+0.33*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D1Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  return 2*Kernel_Z_SE(x,y,hpar)/hpar(0);
}

double D2Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  return 0;
}

double D3Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(d,2)*Kernel_Z_SE(x,y,hpar)/pow(hpar(2),3);
}

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2))+0.33*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}
double D1Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à sigma
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return 2*hpar(0)*(1+(d/hpar(2))+0.33*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D2Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  return 0;
}

double D3Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à lcor
  double d=abs(x(0)-y(0));
  double X=d/hpar(2);
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*exp(-X)*0.33*(X+pow(X,2))*X/hpar(2);
}

double Kernel_GP_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double cor=0;
  cor+=pow((x(0)-y(0))/hpar(1),2); //phi
  cor+=pow((x(1)-y(1))/hpar(3),2); //BK
  cor+=pow((x(2)-y(2))/hpar(4),2); //COAL
  cor+=pow((x(3)-y(3))/hpar(5),2); //NUCL
  cor+=pow((x(4)-y(4))/hpar(6),2); //MT
  return pow(hpar(0),2)*exp(-0.5*cor);
}

double Kernel_GP_Matern12(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double cor=0;
  cor+=0.5*abs(x(0)-y(0))/hpar(1); //phi
  cor+=0.5*abs(x(1)-y(1))/hpar(3); //BK
  cor+=0.5*abs(x(2)-y(2))/hpar(4); //COAL
  cor+=0.5*abs(x(3)-y(3))/hpar(5); //NUCL
  cor+=0.5*abs(x(4)-y(4))/hpar(6); //MT
  return pow(hpar(0),2)*exp(-cor);
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor X
  double cor=pow(hpar(0),2);
  cor*=(1+abs(x(0)-y(0))/hpar(1))*exp(-abs(x(0)-y(0))/hpar(1)); //phi
  return cor;
}

double logprior_hpars(VectorXd const &hpars){
  //edm, exp, lcor
  return 0;
}

double logprior_pars(VectorXd const &pars){
  //prior uniforme sur les pramètres
  return 0;
}

//myoptfunc_gp est définie dans densities.cpp


void PrintVector(VectorXd const &X, VectorXd const &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X(i),values(i));
  }
  fclose(out);
}

int main(int argc, char **argv){
  generator.seed(9);

  auto calib=[](int seed){
  /*Calibration avec seedobs qui vaut seed*/
  

    generator.seed(seed);
    //on les range dans un vecteur<AUGDATA>.
    vector<AUGDATA> data_exp(1);

    VectorXd Xpredictions(80);
    for(int i=0;i<80;i++){Xpredictions(i)=7*(i*1.0)/80;}
    vector<int> indexes={10,20,30,40,50,60,70};

    //remplissage ligne droite. marche bien.
    {
      AUGDATA dat; 
      VectorXd X_obs(indexes.size());
      VectorXd Yobs(X_obs.size());
      for(int i=0;i<X_obs.size();i++){X_obs(i)=Xpredictions(indexes[i]);}
      for(int i=0;i<Yobs.size();i++){Yobs(i)=0.5+distU(generator);}
      //Yobs << 1.3,0.8,1.ons05,0.9,0.91,1.12,1.25;
      dat.SetX(X_obs); dat.SetValue(Yobs);
      data_exp[0]=dat;
    }

    //PrintVector(data_exp[0].GetX(),data_exp[0].Value(),"results/obs.gnu");

    int dim_theta=1;
    int dim_hpars=2;

    //bornes des paramètres de f et des hpars de z.
    
    VectorXd lb_t(dim_theta);
    VectorXd ub_t(dim_theta);
    lb_t << -1;
    ub_t << 1;

    //hpars z : sedm, sobs, lcor.
    VectorXd lb_hpars(dim_hpars);
    VectorXd ub_hpars(dim_hpars);
    lb_hpars << 1e-2,1e-2;
    ub_hpars << 1,1;

    VectorXd hpars_z_guess(dim_hpars);
    hpars_z_guess =0.5*(lb_hpars+ub_hpars);

    int size_xexpe=data_exp[0].Value().size();
    //lambda priormean
    auto lambda_priormean=[size_xexpe](VectorXd const & X, VectorXd const & hpars){
      return VectorXd::Zero(X.size());
    };

    auto my_funct=[](double x, VectorXd const &theta){
      return 1+theta(0)*sin(x);
    };

    auto lambda_model=[&my_funct](VectorXd const &X, VectorXd const &theta){
      //renvoie le vecteur de toutes les prédictions du modèle en tous les points de X.
      VectorXd res(X.size());
      for(int i=0;i<X.size();i++){
        res(i)=my_funct(X(i),theta);
      }
      return res;
    };

    double noise=1e-2;

    // pour la MCMC
    MatrixXd COV_init=pow(0.7,2)*MatrixXd::Identity(1,1);
    //cout << "COV_init : " << endl << COV_init << endl;
    VectorXd Xinit(1);
    Xinit << 0.5;

    int nombre_steps_mcmc=1e5;
    int nombre_samples_collected=100;
    int nautocor=1000;

    //construction du grid
    int npts_init=50;
    DoE doe_init(lb_t,ub_t,npts_init,1);
    //afichage du grid lhs
    //doe_init.WriteGrid("results/save/grid.gnu");

    //instance de base de densité pour alpha
    Density MainDensity(doe_init);
    MainDensity.SetNoise(noise);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetKernel(Kernel_Z_SE);
    MainDensity.SetKernelDerivatives(D1Kernel_Z_SE,D2Kernel_Z_SE,D3Kernel_Z_SE);
    MainDensity.SetModel(lambda_model);
    MainDensity.SetPriorMean(lambda_priormean);
    MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
    MainDensity.SetDataExp(data_exp);
    MainDensity.SetXprofile(data_exp[0].GetX());



   
   
  //phase KOH
  
  
    
      VectorXd hpars_koh=MainDensity.HparsKOH(hpars_z_guess,50);
      //cout << "hpars koh opt : " << hpars_koh.transpose() << endl;
      MainDensity.Run_MCMC_fixed_hpars(nombre_steps_mcmc,nombre_samples_collected,Xinit,COV_init,hpars_koh,generator);
      //diagnostic
      //MainDensity.Autocor_diagnosis(nautocor,"results/diag/autocorkoh.gnu");
      //écriture des samples
      //MainDensity.WriteSamples("results/save/sampkoh.gnu");
      //prédictions
      //MainDensity.WritePredictions(Xpredictions,"results/preds/predskoh.gnu");
    

    //phase OPTI
  
    
      MatrixXd Bounds_hpars_gp(2,3);
      Bounds_hpars_gp(0,0)=1E-2; Bounds_hpars_gp(1,0)=1e1; //std
      Bounds_hpars_gp(0,1)=1E-2; Bounds_hpars_gp(1,1)=5; //lcor
      Bounds_hpars_gp(0,2)=1E-2; Bounds_hpars_gp(1,2)=1e1; //noise
      VectorXd hpars_gp_guess=0.5*(Bounds_hpars_gp.row(0)+Bounds_hpars_gp.row(1)).transpose();
      DensityOpt DensOpt(MainDensity);
      DensOpt.Compute_optimal_hpars(1);
      DensOpt.BuildHGPs(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess,2);
      DensOpt.opti_allgps(hpars_gp_guess);
      DensOpt.Run_MCMC_opti_hGPs(nombre_steps_mcmc,nombre_samples_collected,Xinit,COV_init,generator);
      //DensOpt.Autocor_diagnosis(nautocor,"results/diag/autocoropt.gnu");
      //écriture des samples
      //DensOpt.WriteSamples("results/save/sampopt.gnu");
      //prédictions

      double varkoh=MainDensity.Var();
      double varopt=DensOpt.Var();
      double mapkoh=MainDensity.map()(0);
      double mapopt=DensOpt.map()(0);
      int counterkohf=MainDensity.WritePredictionsF(Xpredictions,"results/preds/predskohF.gnu");
      int counteroptf=DensOpt.WritePredictionsF(Xpredictions,"results/preds/predsoptF.gnu");
      int counterkohfz=MainDensity.WritePredictions(Xpredictions,"results/preds/predskoh.gnu");
      int counteroptfz=DensOpt.WritePredictions(Xpredictions,"results/preds/predsopt.gnu");
      cout << " varkoh : " << varkoh << endl;
      cout << " varopt : " << varopt << endl;
      cout << " mapkoh : " << mapkoh << endl;
      cout << " mapopt : " << mapopt << endl;
      cout << " countkohf : " << counterkohf << endl;
      cout << " countoptf : " << counteroptf << endl;
      cout << " countkohfz : " << counterkohfz << endl;
      cout << " countoptfz : " << counteroptfz << endl;
    
      vector<double> v(8);
      v[0]=varkoh; v[1]=varopt;v[2]=mapkoh;v[3]=mapopt;v[4]=counterkohf;v[5]=counteroptf;v[6]=counterkohfz;v[7]=counteroptfz;
      return v;
  };
  int nrepet=100;
  vector<double> k(8);
  for(double d:k){
    d=0;
  }
  for(int i=0;i<nrepet;i++){
    int seedobs=distU(generator)*1e6;
    vector<double> v=calib(seedobs);
    for(int i=0;i<8;i++){
      k[i]+=v[i];
    }
  }
  for(int i=0;i<8;i++){
    k[i]/=nrepet;
  }

  cout << " varkoh : " << k[0] << endl;
  cout << " varopt : " << k[1] << endl;
  cout << " mapkoh : " << k[2] << endl;
  cout << " mapopt : " << k[3] << endl;
  cout << " countkohf : " << k[4] << endl;
  cout << " countoptf : " << k[5] << endl;
  cout << " countkohfz : " << k[6] << endl;
  cout << " countoptfz : " << k[7] << endl;
  
  exit(0);
};