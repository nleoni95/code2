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
  //edm, exp, lcor
  if(hpars(2)<=0){return -999;}
  double alpha_ig=10;
  double beta_ig=140;
  return log(pow(beta_ig,alpha_ig)*pow(hpars(2),-alpha_ig-1)*exp(-beta_ig/hpars(2))/tgamma(alpha_ig));
  return 0;//-2*(log(hpars(1)))-2*(log(hpars(0)));
}

double logprior_pars(VectorXd const &pars){
  //prior uniforme sur les pramètres
  double moy0=40;
  double moy1=18.9e-6;
  double moy2=0.6;
  double sig0=2*5;
  double sig1=18.9e-6/10.; //double sig1=2*1e-6;
  double sig2=0.6/10.;
  return log(gaussprob(pars(0),moy0,sig0)*gaussprob(pars(1),moy1,sig1)*gaussprob(pars(2),moy2,sig2));
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
  VectorXd lb_t(dim_theta); lb_t(0)=10;lb_t(1)=0.5*18.9E-6;lb_t(2)=0.3; //0.2 avant
  VectorXd ub_t(dim_theta); ub_t(0)=80;ub_t(1)=1.5*18.9E-6;ub_t(2)=0.9;

  std::vector<double> lb_hpars(3); lb_hpars[0]=1e4;lb_hpars[1]=1e4;lb_hpars[2]=10; //-5 avant
  std::vector<double> ub_hpars(3); ub_hpars[0]=3e5;ub_hpars[1]=3e5;ub_hpars[2]=25; //bornes sur l'optimisation des hpars. edm exp lcor
  
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
  
  //Points sur lesquels on va tracer des réalisations de notre modèle
  int samp_size=80; //80 avant
  vector<VectorXd> X(samp_size); for (int i=0;i<samp_size;i++){VectorXd x(1); x(0)=0.01+25*double(i)/double(samp_size); X[i]=x;}  
  //construction du DoE initial en LHS (generator) ou Grid Sampling(sans generator)
   
  
  DoE doe_init(lb_t,ub_t,20);//,generator); gridsize
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
  Dopt.WriteMeanPreds("results/meanpredsopt.gnu");
  cout << "MAP : " << Dopt.MAP().transpose() << endl;
  cout << "Mean : " << Dopt.Mean().transpose() << endl;
  cout << Dopt.Cov() << endl;
  Dopt.WritePost("results/popt.gnu");
  Dopt.WriteHpars("results/hopt.gnu");
  Dopt.WritePostHpars("results/hoptmarg");
  Dopt.WriteMarginals("results/poptmarg");
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
  DKOH.WriteMeanPreds("results/meanpredskoh.gnu");
  cout << "hpars koh :" << DKOH.GetHpars().transpose() << endl;
  cout << "MAP : " << DKOH.MAP().transpose() << endl;
  cout << "Mean : " << DKOH.Mean().transpose() << endl;
  cout << DKOH.Cov() << endl;
  DKOH.WritePost("results/pkoh.gnu");
  DKOH.WriteMarginals("results/pkohmarg");
  //phase de test. On vérifie les valeurs du critère KOH pour l'hypothèse "haute EDM faible EXP"
  //appel à la fonction
  {
    FILE* out=fopen("critkoh.gnu","w");
    for (int i=0;i<60;i++){
      vector<double> x(3);
      x[0]=100000;x[1]=10000;x[2]=0.01+25*((double)i/60.);
      vector<double> grad(3);
      double crit=DKOH.optfunc(x,grad,&DKOH);
      fprintf(out,"%e %e\n",x[2],crit);
    }
    vector<double> x(3);
    vector<double> grad(3);
    for (int i=0;i<3;i++){x[i]=DKOH.GetHpars()(i);}
    double crit=DKOH.optfunc(x,grad,&DKOH);
    fprintf(out,"%e %e %e %e",x[0],x[1],x[2],crit);
    fclose(out);
  }
  


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
  DSimple.WriteMarginals("results/psimpmarg");

  temp=DSimple.DrawSample(generator);
  PrintVector(X,temp,"results/predsimp1.gnu");
  temp=DSimple.DrawSample(generator);
  PrintVector(X,temp,"results/predsimp2.gnu");
  temp=DSimple.DrawSample(generator);
  PrintVector(X,temp,"results/predsimp3.gnu");
  cout << "Pred Simple over." << endl;

  DensityCV DCV(DKOH);
  DCV.Build();
  DCV.WritePost("results/pcv.gnu");
  temp=DCV.DrawSample(generator);
  PrintVector(X,temp,"results/predcv1.gnu");
  temp=DCV.DrawSample(generator);
  PrintVector(X,temp,"results/predcv2.gnu");
  temp=DCV.DrawSample(generator);
  PrintVector(X,temp,"results/predcv3.gnu");
  /* Partie MCMC */
  DKOH.WritePredictions("results/predkoh.gnu");
  Dopt.WritePredictions("results/predopt.gnu");
  DKOH.WritePredictionsFZ("results/predkohFZ.gnu");
  Dopt.WritePredictionsFZ("results/predoptFZ.gnu");
  DCV.WritePredictions("results/predcv.gnu");
  DSimple.WritePredictions("results/predsimp.gnu");
  DSimple.WritePredictionsFZ("results/predsimpFZ.gnu");
  {
    VectorXd hparskoh=DKOH.GetHpars();
    VectorXd hparscv=DCV.GetHpars();
    FILE* out=fopen("results/summary.gnu","w");
    fprintf(out,"Hpars KOH :\n");
    fprintf(out,"%e %e %e",hparskoh(0),hparskoh(1),hparskoh(2));
    fprintf(out,"Hpars CV :\n");
    fprintf(out,"%e %e %e",hparscv(0),hparscv(1),hparscv(2));
    fclose(out);
  }
  
  {
    int nchain=2000000;///370s pour 50000steps 1500000 la best value pour 3000 samples
    VectorXd t_init(6);
    t_init(0)=42; t_init(1)=1.606e-05; t_init(2)=0.800;
    t_init(3)=150000; t_init(4)=150000; t_init(5)=17;
   
    cout << "t_init : " << t_init.transpose() << endl;
    MatrixXd COV=MatrixXd::Zero(6,6);
    COV(0,0)=pow(3,2);
    COV(1,1)=pow(1.5e-6,2);
    COV(2,2)=pow(5e-2,2);
    COV(3,3)=pow(50000,2);
    COV(4,4)=pow(40000,2);
    COV(5,5)=pow(1.9,2);
    cout << "COV :" << COV << endl;
    MCMC mcmc(MainDensity,nchain);
    generator.seed(119);
    mcmc.Run(t_init,COV,generator);
    mcmc.SelectSamples(3000);
    cout << "map :" << mcmc.MAP().transpose() << endl;
    cout << "mean :" << mcmc.Mean().transpose() << endl;
    cout << "cov : " << mcmc.Cov() << endl;
    cout << "écriture des samples" << endl;
    mcmc.WriteAllSamples("results/mcmc_allsamples.gnu");
    mcmc.WriteSelectedSamples("results/mcmc_selectedsamples.gnu");
    mcmc.PrintOOB();
    clock_t c_start = std::clock();
    cout << "écriture des prédictions mcmc..." << endl;
    mcmc.WritePredictions("results/predmcmc.gnu");
    mcmc.WritePredictionsFZ("results/predmcmcFZ.gnu");
    clock_t c_end = std::clock();
    cout << "temps d'exécution pred mcmc:" << (c_end-c_start) / CLOCKS_PER_SEC << endl;
    cout << "calcul de la corrélation mcmc..." << endl;
    mcmc.Autocorrelation_diagnosis(1000);
    clock_t c_end2 = std::clock();
    cout << "temps d'exécution calcul autocor:" << (c_end2-c_end) / CLOCKS_PER_SEC << endl;
    
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
