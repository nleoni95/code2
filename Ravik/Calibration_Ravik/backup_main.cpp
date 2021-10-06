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
int alpha_ig=5; // paramètres du prior sur lcor
double beta_ig=0.4;

//paramètres pour python
PyObject *pFunc, *pArgs;
PyObject *pParamsRavik; //liste comprenant les valeurs nominales des paramètres de Ravik.

double my_model(VectorXd const &x, VectorXd const &theta){
  //taille totale des paramètres dans le modèle : 10.
  //création d'une liste pour les arguments
  PyList_SetItem(pParamsRavik,0,PyFloat_FromDouble(theta(0))); //angle. VN 40deg
  PyList_SetItem(pParamsRavik,1,PyFloat_FromDouble(theta(1))); //coef multiplicateur. VN 18.9E6
  PyList_SetItem(pParamsRavik,3,PyFloat_FromDouble(theta(2))); //param de DTsup. VN 0.75
  pArgs=PyTuple_New(2);
  PyTuple_SetItem(pArgs,0,PyFloat_FromDouble(x(0)));
  PyTuple_SetItem(pArgs,1,PyList_AsTuple(pParamsRavik));
  return PyFloat_AsDouble(PyObject_CallObject(pFunc, pArgs));
}

double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &xp, Eigen::VectorXd const &hpar){
  //Fonction Kernel sans bruit. hpar(0) = sig_edm, hpar(1) = sig_exp, hpar(2) = lcor
   return hpar(0)*hpar(0)*exp(-pow((x(0)-xp(0))/hpar(2),2)*.5 ); /* squared exponential kernel */
};

double PriorMean(VectorXd const &x, VectorXd const &hpars){
  return 0;
}

double logprior_hpars(VectorXd const &hpars){
  //edm, exp, lcor
  //return log(pow(beta_ig,alpha_ig)*pow(hpars(2),-alpha_ig-1)*exp(-beta_ig/hpars(2))/tgamma(alpha_ig));
  return -2*(log(hpars(0))+log(hpars(1)))-log(hpars(2));
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
  int cas=4;

  // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(dim_theta); lb_t(0)=20;lb_t(1)=0.5*18.9E-6;lb_t(2)=0.6;
  VectorXd ub_t(dim_theta); ub_t(0)=60;ub_t(1)=1.5*18.9E-6;ub_t(2)=0.9;

  std::vector<double> lb_hpars(3); lb_hpars[0]=20;lb_hpars[1]=20;lb_hpars[2]=15; //-5 avant
  std::vector<double> ub_hpars(3); ub_hpars[0]=100000;ub_hpars[1]=500000;ub_hpars[2]=20; //bornes sur l'optimisation des hpars. edm exp lcor
  
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
  
  //construction du DoE initial en LHS (generator) ou Grid Sampling(sans generator)
  DoE doe_init(lb_t,ub_t,10);//,generator);
  doe_init.Fill(my_model,&data);

  //Points sur lesquels on va tracer des réalisations de notre modèle
  int samp_size=80;
  vector<VectorXd> X(samp_size); for (int i=0;i<samp_size;i++){VectorXd x(1); x(0)=1+25*double(i)/double(samp_size); X[i]=x;}  
  
  
  //configuration de l'instance de base de densité
  Density MainDensity(doe_init);
  MainDensity.SetModel(my_model);
  MainDensity.SetKernel(Kernel);
  MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(PriorMean);

  /*MCMC*/
    {
    int nchain=50000;///370s pour 50000steps
    VectorXd t_init(6);
    t_init(0)=42; t_init(1)=1.606e-05; t_init(2)=0.800;
    t_init(3)=50000; t_init(4)=50000; t_init(5)=17;
   
    cout << "t_init : " << t_init.transpose() << endl;
    MatrixXd COV=MatrixXd::Zero(6,6);
    COV(0,0)=pow(1,2);
    COV(1,1)=pow(5e-7,2);
    COV(2,2)=pow(1e-2,2);
    COV(3,3)=pow(300,2);
    COV(4,4)=pow(300,2);
    COV(5,5)=pow(0.1,2);
    cout << "COV :" << COV << endl;
    MCMC mcmc(MainDensity,nchain);
    mcmc.Run(t_init,COV,generator);
    mcmc.SelectSamples(2000);
    cout << "map :" << mcmc.MAP().transpose() << endl;
    cout << "mean :" << mcmc.Mean().transpose() << endl;
    cout << "cov : " << mcmc.Cov() << endl;
    cout << "écriture des samples" << endl;
    mcmc.WriteAllSamples("results/mcmc_allsamples.gnu");
    mcmc.WriteSelectedSamples("results/mcmc_selectedsamples.gnu");
    mcmc.PrintOOB();
  }
exit(0);

  /*Partie Opti à faire fonctionner d'abord.*/
  DensityOpt Dopt(MainDensity);
  Dopt.Build();
  cout << "MAP : " << Dopt.MAP().transpose() << endl;
  cout << "Mean : " << Dopt.Mean().transpose() << endl;
  cout << Dopt.Cov() << endl;
  Dopt.WritePost("results/popt.gnu");
  Dopt.WriteHpars("results/hopt.gnu");
  Dopt.WritePostHpars("results/hoptmarg");
  Dopt.WriteMarginals("results/poptmarg");
  VectorXd temp=Dopt.DrawSample(X,generator);
  PrintVector(X,temp,"results/predopt1.gnu");
  temp=Dopt.DrawSample(X,generator);
  PrintVector(X,temp,"results/predopt2.gnu");
  temp=Dopt.DrawSample(X,generator);
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
  DKOH.WriteMarginals("results/pkohmarg");

  temp=DKOH.DrawSample(X,generator);
  PrintVector(X,temp,"results/predkoh1.gnu");
  temp=DKOH.DrawSample(X,generator);
  PrintVector(X,temp,"results/predkoh2.gnu");
  temp=DKOH.DrawSample(X,generator);
  PrintVector(X,temp,"results/predkoh3.gnu");
  cout << "Pred KOH over." << endl;


  DensitySimple DSimple(MainDensity,Dopt.GetNormCst());
  DSimple.Build();
  cout << "hpars Simple :" << DSimple.GetHpars().transpose() << endl;
  cout << "MAP : " << DSimple.MAP().transpose() << endl;
  cout << "Mean : " << DSimple.Mean().transpose() << endl;
  cout << DSimple.Cov() << endl;
  DSimple.WritePost("results/psimp.gnu");
  DSimple.WriteMarginals("results/psimpmarg");

  temp=DSimple.DrawSample(X,generator);
  PrintVector(X,temp,"results/predsimp1.gnu");
  temp=DSimple.DrawSample(X,generator);
  PrintVector(X,temp,"results/predsimp2.gnu");
  temp=DSimple.DrawSample(X,generator);
  PrintVector(X,temp,"results/predsimp3.gnu");
  cout << "Pred Simple over." << endl;

  DensityCV DCV(DKOH);
  DCV.Build();
  DCV.WritePost("results/pcv.gnu");
  temp=DCV.DrawSample(X,generator);
  PrintVector(X,temp,"results/predcv1.gnu");
  temp=DCV.DrawSample(X,generator);
  PrintVector(X,temp,"results/predcv2.gnu");
  temp=DCV.DrawSample(X,generator);
  PrintVector(X,temp,"results/predcv3.gnu");
  /* Partie MCMC */





  exit(0);
  
};
