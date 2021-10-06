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
#include <chrono>
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
int dim_theta=3;

//paramètres pour python
PyObject *pFunc, *pArgs;
PyObject *pParamsRavik; //liste comprenant les valeurs nominales des paramètres de Ravik.

double gaussprob(double x,double mu, double sigma){
  //renvoie la probabilité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}

double my_model(VectorXd const &x, VectorXd const &theta){
  //taille totale des paramètres dans le modèle : 10.
  // x de dimension 1.
  //création d'une liste pour les arguments
  PyList_SetItem(pParamsRavik,0,PyFloat_FromDouble(theta(0))); //angle. VN 40deg
  PyList_SetItem(pParamsRavik,1,PyFloat_FromDouble(theta(1))); //coef multiplicateur. VN 18.9E6
  PyList_SetItem(pParamsRavik,3,PyFloat_FromDouble(theta(2))); //param de DTsup. VN 0.75
  pArgs=PyTuple_New(2);
  PyTuple_SetItem(pArgs,0,PyFloat_FromDouble(x(0)));
  PyTuple_SetItem(pArgs,1,PyList_AsTuple(pParamsRavik));
  return PyFloat_AsDouble(PyObject_CallObject(pFunc, pArgs));
}

double Kernel_Z_Matern_52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2))+0.33*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D1Kernel_Z_Matern(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à sigma
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return 2*hpar(0)*(1+(d/hpar(2))+0.33*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D2Kernel_Z_Matern(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  return 0;
}

double D3Kernel_Z_Matern(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à lcor
  double d=abs(x(0)-y(0));
  double X=d/hpar(2);
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*exp(-X)*0.33*(X+pow(X,2))*X/hpar(2);
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double cor=pow(hpar(0),2);
  cor*=(1+abs(x(0)-y(0))/hpar(1))*exp(-abs(x(0)-y(0))/hpar(1)); //phi
  cor*=(1+abs(x(1)-y(1))/hpar(3))*exp(-abs(x(1)-y(1))/hpar(3)); //BK
  cor*=(1+abs(x(2)-y(2))/hpar(4))*exp(-abs(x(2)-y(2))/hpar(4)); //COAL
  return cor;
}

VectorXd RtoGP(VectorXd const & X){
  //transforme theta dans l'espace réel en [0,1]. transformation affine.
  //bornes de l'intervalle ici.
  VectorXd lb_t(3); lb_t(0)=10;lb_t(1)=0.5*18.9E-6;lb_t(2)=0.3; //0.2 avant
  VectorXd ub_t(3); ub_t(0)=80;ub_t(1)=1.5*18.9E-6;ub_t(2)=0.9;
  if(X(0)<lb_t(0) || X(0)>ub_t(0)){cerr << "erreur de dimension rtogp 0" << endl;}
  if(X(1)<lb_t(1) || X(1)>ub_t(1)){cerr << "erreur de dimension rtogp 1" << endl;}
  if(X(2)<lb_t(2) || X(2)>ub_t(2)){cerr << "erreur de dimension rtogp 2" << endl;}
  if(X.size()!=3){cerr << "erreur de dimension rtogp" << endl;}
  VectorXd Xgp(3);
  for(int i=0;i<3;i++){
    Xgp(i)=(X(i)-lb_t(i))/(ub_t(i)-lb_t(i));
  }
  return Xgp;
}

VectorXd GPtoR(VectorXd const & X){
  //transforme theta dans l'espace gp vers l'espace réel. transformation affine.
  //bornes de l'intervalle ici.
  VectorXd lb_t(3); lb_t(0)=10;lb_t(1)=0.5*18.9E-6;lb_t(2)=0.3; //0.2 avant
  VectorXd ub_t(3); ub_t(0)=80;ub_t(1)=1.5*18.9E-6;ub_t(2)=0.9;
  if(X(0)<0 || X(0)>1){cerr << "erreur de dimension rtogp 0" << endl;}
  if(X(1)<0 || X(1)>1){cerr << "erreur de dimension rtogp 1" << endl;}
  if(X(2)<0 || X(2)>1){cerr << "erreur de dimension rtogp 2" << endl;}
  VectorXd Xr(3);
  for(int i=0;i<3;i++){
    Xr(i)=lb_t(i)+(ub_t(i)-lb_t(i))*X(i);
  }
  return Xr;
}

double PriorMean(VectorXd const &x, VectorXd const &hpars){
  return 0;
}

double logprior_hpars(VectorXd const &hpars){
  //edm, exp, lcor
  return 0;//-2*log(hpars(0));
}

double logprior_pars(VectorXd const &pars){
  //prior gaussien sur les paramètres.
  VectorXd parsreal=GPtoR(pars);
  double moy0=40;
  double moy1=18.9e-6;
  double moy2=0.6;
  double sig0=2*5;
  double sig1=18.9e-6/10.; //double sig1=2*1e-6;
  double sig2=0.6/10.;
  //return log(gaussprob(parsreal(0),moy0,sig0)*gaussprob(parsreal(1),moy1,sig1)*gaussprob(parsreal(2),moy2,sig2));
  return 0;
}

double logprior_pars2(VectorXd const &pars){
  return 0;
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
  list<int> all_cases={19,20};
  int cas=20;

  //lb_hpars[3]=-0.5e5;
  //ub_hpars[3]=0;

  /*initialisation du python*/

  Py_Initialize(); 
  PyRun_SimpleString("import sys"); // déjà initialisé par Py_Initialize ?
  PyRun_SimpleString("import os");
  PyRun_SimpleString("sys.path.append(os.getcwd())");
  PyObject *pName, *pModule, *pValue;
  //https://medium.com/datadriveninvestor/how-to-quickly-embed-python-in-your-c-application-23c19694813
  pName = PyUnicode_FromString((char*)"model"); //nom du fichier sans .py
  pModule = PyImport_Import(pName);
  //PyErr_Print(); pratique !!
  pFunc= PyObject_GetAttrString(pModule, (char*)"initialize_case");//nom de la fonction
  pArgs = PyTuple_Pack(1, PyLong_FromLong(cas));//premier argument : nombre d'arguments de la fonction python, ensuite les arguments.
  pValue = PyObject_CallObject(pFunc, pArgs); //pvalue est alors l'objet de retour. //appel à initialize_case

  pFunc = PyObject_GetAttrString(pModule, (char*)"exp_datab");//nom de la fonction
  pArgs = PyTuple_New(0); //tuple vide
  pValue = PyObject_CallObject(pFunc, pArgs); //pvalue est alors l'objet de retour.

    //initialisation des paramètres de Ravik
  pParamsRavik=PyList_New(10);
  PyList_SetItem(pParamsRavik,0,PyFloat_FromDouble(40)); //angle de contact
  PyList_SetItem(pParamsRavik,1,PyFloat_FromDouble(18.9e-6)); //paramètres de corrélation Dd
  PyList_SetItem(pParamsRavik,2,PyFloat_FromDouble(0.27));
  PyList_SetItem(pParamsRavik,3,PyFloat_FromDouble(0.75));
  PyList_SetItem(pParamsRavik,4,PyFloat_FromDouble(-0.3));
  PyList_SetItem(pParamsRavik,5,PyFloat_FromDouble(-0.26));
  PyList_SetItem(pParamsRavik,6,PyFloat_FromDouble(6.1E-3)); //corrélation twait
  PyList_SetItem(pParamsRavik,7,PyFloat_FromDouble(0.6317));
  PyList_SetItem(pParamsRavik,8,PyFloat_FromDouble(0.1237)); //corrélation rappD
  PyList_SetItem(pParamsRavik,9,PyFloat_FromDouble(-0.373));

  pFunc = PyObject_GetAttrString(pModule, (char*)"run_model");//nom de la fonction
  //récupération des observations. Attention au déréférencement du pointeur et au nombre de données.
  if(PyList_Check(pValue)!=1){cerr << "erreur : la fonction exp_datab n'a pas renvoyé une liste" << endl;}
  cout << "Cas expérimental : " << cas <<", Nombre d'observations : "<< PyList_Size(pValue) << endl;

  vector<DATA> data(PyList_Size(pValue));
  for (int i=0;i<PyList_Size(pValue);i++){
    //remplissage des observations
    DATA dat;
    VectorXd x(1);
    x(0)=PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(pValue,i),0));
    dat.SetX(x);
    dat.SetValue(PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(pValue,i),1)));
    data[i]=dat;
  }

  //écriture des observations
  ofstream ofile("results/obs.gnu");
  for(auto const &d:data){
    ofile << d.GetX()(0) << " " << d.Value() << endl;
  }
  ofile.close();

  //on réécrit en vector augdata.
  vector<AUGDATA> data_exp(1);
  {
    VectorXd X_points(data.size());
    VectorXd Values(data.size());
    for(int i=0;i<data.size();i++){
      X_points(i)=data[i].GetX()(0);
      Values(i)=data[i].Value();
    }
    AUGDATA dat; dat.SetX(X_points); dat.SetValue(Values); data_exp[0]=dat;
  }


   // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(dim_theta); lb_t(0)=0;lb_t(1)=0;lb_t(2)=0; //0.2 avant
  VectorXd ub_t(dim_theta); ub_t(0)=1;ub_t(1)=1;ub_t(2)=1;

  VectorXd lb_hpars(3); lb_hpars(0)=1e3;lb_hpars(1)=1e3;lb_hpars(2)=1; //-5 avant
  VectorXd ub_hpars(3); ub_hpars(0)=9e5;ub_hpars(1)=3e5;ub_hpars(2)=25; //bornes sur l'optimisation des hpars. edm exp lcor

  
  //Points sur lesquels on va faire les prédictions du modèle. car sinon juste en 6 points pas top. Mais commençons par les points expérimentaux.
  int samp_size=80; //80 avant
  VectorXd X_predictions(samp_size); for (int i=0;i<samp_size;i++){X_predictions(i)=0.01+25*double(i)/double(samp_size);}  

  default_random_engine generator(123456);

  //pour la MCMC
  int nombre_steps_mcmc=1e5;
  int nombre_samples_collected=200;
  int nautocor=2000;

  VectorXd X_init_mcmc=lb_t+0.5*(ub_t-lb_t);
  MatrixXd COV_init=MatrixXd::Identity(3,3);
  COV_init(0,0)=pow(0.1,2);
  COV_init(1,1)=pow(0.1,2);
  COV_init(2,2)=pow(0.1,2);

  auto lambda_model=[](VectorXd const &X, VectorXd const & theta){
    //renvoie toutes les prédictions du modèle aux points donnés par X.
    VectorXd pred(X.size());
    for(int i=0;i<X.size();i++){
      VectorXd x(1);
      x(0)=X(i);
      pred(i)=my_model(x,GPtoR(theta));
    }
    return pred;
  };

  auto lambda_priormean=[](VectorXd const &X, VectorXd const &hpars){
    return VectorXd::Zero(X.size());
  };

  //juste un test. 500 évaluations de modèle ? 

  //construction du DoE initial en LHS (generator) ou Grid Sampling(sans generator)
  DoE doe_init(lb_t,ub_t,500,1);//doe halton. 500 points en dimension 3 (3 paramètres incertains).
  doe_init.WriteGrid("results/save/grid.gnu");
  
  
  //configuration de l'instance de base de densité
  Density MainDensity(doe_init);
  MainDensity.SetModel(lambda_model);
  MainDensity.SetKernel(Kernel_Z_Matern_52);
  MainDensity.SetKernelDerivatives(D1Kernel_Z_Matern,D2Kernel_Z_Matern,D3Kernel_Z_Matern);
  MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(lambda_priormean);
  MainDensity.SetDataExp(data_exp);
  MainDensity.SetXprofile(data_exp[0].GetX());
  VectorXd hpars_koh(3);
  hpars_koh <<1e5,1e5,11;
  

  cout << "début koh : " << endl;
  {

  hpars_koh=MainDensity.HparsKOH(hpars_koh); //optimisation
  cout << "hpars_koh :" << hpars_koh.transpose() << endl;
  
  MainDensity.Run_MCMC_fixed_hpars(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,hpars_koh,generator);
  MainDensity.Autocor_diagnosis(nautocor,"results/diag/autocorkoh.gnu");
  MainDensity.WriteMCMCSamples("results/diag/allsampkoh.gnu");
  MainDensity.WriteSamples("results/save/sampalphakoh.gnu");
  MainDensity.WritePredictionsF(X_predictions,"results/preds/predkohF.gnu");
  MainDensity.WritePredictions(X_predictions,"results/preds/predkoh.gnu");

  }



/*{
    int nhpars_gp=5; //3 paramètres. + sedm et sobs.
    MatrixXd Bounds_hpars_gp(2,nhpars_gp);
    Bounds_hpars_gp(0,0)=1E-3; Bounds_hpars_gp(1,0)=1e2; //variance
    Bounds_hpars_gp(0,1)=1E-2;Bounds_hpars_gp(1,1)=2; //lcor
    Bounds_hpars_gp(0,3)=1E-2;Bounds_hpars_gp(1,3)=2; //lcor
    Bounds_hpars_gp(0,4)=1E-2;Bounds_hpars_gp(1,4)=2; //lcor
    Bounds_hpars_gp(0,2)=1E-3; Bounds_hpars_gp(1,2)=2E-3; //sigma obs
    VectorXd hpars_guess_gp=0.5*(Bounds_hpars_gp.row(0)+Bounds_hpars_gp.row(1)).transpose();

    cout << "début Opt :" << endl;
    DensityOpt DensOpt(MainDensity);
    DensOpt.Compute_optimal_hpars();
    DensOpt.BuildHGPs(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_guess_gp,3);
    DensOpt.opti_allgps(hpars_guess_gp);
    DensOpt.Test_hGPs();
    VectorXd v=DensOpt.HparsOpt(X_init_mcmc,hpars_koh);
    cout << " v : " << v.transpose() << endl;
    //exit(0);

    DensOpt.Run_MCMC_opti_hGPs(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,generator);
    DensOpt.Autocor_diagnosis(nautocor,"results/diag/autocoropt.gnu");
    DensOpt.WriteMCMCSamples("results/diag/allsampopt.gnu");

    DensOpt.WritePredictionsF(X_predictions,"results/preds/predoptF.gnu");
    DensOpt.WritePredictions(X_predictions,"results/preds/predopt.gnu"); 
}
*/
  

  exit(0);

 
};
