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
//pour stocker les valeurs calculées d'hyperparamètres optimaux. On a un vectorxd qui est le theta, et une map (int vectorxd) qui permet de retrouver l'hyperparamètre optimal (vectorxd) du cas i (int)

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
PyObject *pFunc_model;
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
  PyObject *pArgs=PyTuple_New(2);
  PyTuple_SetItem(pArgs,0,PyFloat_FromDouble(x(0)));
  PyTuple_SetItem(pArgs,1,PyList_AsTuple(pParamsRavik));
  return PyFloat_AsDouble(PyObject_CallObject(pFunc_model, pArgs));
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

double Kernel_Z_Linear(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau linéaire
  return pow(hpar(0),2)+pow(hpar(2),2)*(x(0)-hpar(3))*(y(0)-hpar(3));
  //hpar 0 : ordonnée en x=0. Equivalent à cst prior mean.
  // hpars 2 : pente de la fonction
  // hpars 3 : point en x où l'incertitude sera nulle.
}

double Kernel_Z_Quad(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau linéaire
  double d=(x(0)-hpar(3))*(y(0)-hpar(3));
  return pow(hpar(0),2)+pow(hpar(2),2)*d+pow(hpar(4),2)*pow(d,2);
  //hpar 0 : ordonnée en x=0. Equivalent à cst prior mean.
  // hpars 2 : pente de la fonction
  // hpars 3 : point en x où l'incertitude sera nulle.
  // hpars 4 : coeff du second degré.
}

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2))+(1./3)*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D1Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à sigma
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return 2*hpar(0)*(1+(d/hpar(2))+(1./3)*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D2Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  return 0;
}

double D3Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à lcor
  double d=abs(x(0)-y(0));
  double X=d/hpar(2);
  return pow(hpar(0),2)*exp(-X)*pow(X,2)*(d+hpar(2))/(3*pow(hpar(2),2));
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
  VectorXd lb_t(3); lb_t(0)=10;lb_t(1)=0.5*18.9E-6;lb_t(2)=0.5; //0.2 avant
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

double logprior_hpars_linearkernel(VectorXd const &hpars){
  //ordonnée à l'origine, exp, pente fct, abscisse initiale.
  return -2*(log((1+abs(hpars(0)))*hpars(1)*hpars(2)));
}

double logprior_hpars_quadkernel(VectorXd const &hpars){
  //ordonnée à l'origine, exp, pente fct, abscisse initiale.
  return -2*(log((1+abs(hpars(0)))*hpars(1)*hpars(2)));
}


double logprior_hpars(VectorXd const &hpars){
  //edm, exp, lcor
  return -2*(log(hpars(0)));
}

double logprior_pars(VectorXd const &pars){
  //prior gaussien sur les paramètres.
  return 0;
  VectorXd parsreal=GPtoR(pars);
  double moy0=40;
  double moy1=18.9e-6;
  double moy2=0.6;
  double sig0=2*5;
  double sig1=18.9e-6/10.; //double sig1=2*1e-6;
  double sig2=0.6/10.;
  //return log(gaussprob(parsreal(0),moy0,sig0)*gaussprob(parsreal(1),moy1,sig1)*gaussprob(parsreal(2),moy2,sig2));
  
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

vector<DATA> GetObs(PyObject *pFunc_exp,int cas){
  //setup la variable globale au cas.
  //renvoie les observations du cas numéro cas.
  PyObject *pArgs = PyTuple_Pack(1, PyLong_FromLong(cas));//
  PyObject *pValue = PyObject_CallObject(pFunc_exp, pArgs); //appel fct obs
  if(PyList_Check(pValue)!=1){cerr << "erreur : la fonction exp_datab n'a pas renvoyé une liste" << endl;}
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
  cout << "number of obs : " << data.size() << endl;
  return data;
}

vector<AUGDATA> Conversion(vector<DATA> const &data){
  //conversion de vector<DATA> en vector<AUGDATA>. Je ne rajoute pas le point(0,0) finalement.
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
  return data_exp;
}

void WriteObs(string filename,vector<DATA> &data){
  ofstream ofile(filename);
  for(auto const &d:data){
    ofile << d.GetX()(0) << " " << d.Value() << endl;
  }
  ofile.close();
}

/*
vector<VectorXd> read_hparskoh(string const &filename,int dim_hpars_koh){
  //lecture d'hpars koh
  vector<VectorXd> res;
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      VectorXd hpars(dim_hpars_koh);
      for(int i=0;i<hpars.size();i++){
        hpars(i)=stod(words[i+1]);
      }
      res.push_back(hpars);
    }
    cout << "number of samples loaded : " << res.size() << endl;
  }
  else{
    cerr << "empty file" << endl;
  }
}
*/

vector<VectorXd> read_hparskoh(string const &filename, list<int> & cases){
  //lecture d'un fichier d'hpars koh, qui est écrit de la forme n V, avec n le numéro du cas et V le vecteur d'hyperparamètres.
  map<int,VectorXd> m;
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      int key=stoi(words[0]);
      VectorXd hpars(3);
      for(int i=0;i<hpars.size();i++){
        hpars(i)=stod(words[i+1]);
      }
      m.insert(make_pair(key,hpars));
    }
    cout << "number of samples in the file : " << m.size() << endl;
  }
  vector<VectorXd> v;
  for(const int &i:cases){
    v.push_back(m[i]);
  }
  cout << "number of samples loaded: " << v.size() << endl;
  return v;
}

void write_hparskoh(string const &filename, list<int> & cases,vector<VectorXd> &hpars){
  ofstream ofile(filename);
  int c=0;
  for (int i:cases){
    ofile << i << " ";
    for(int j=0;j<hpars[c].size();j++){
      ofile << hpars[c](j) << " ";
    }
    ofile << endl;
    c++;
  }
  ofile.close();
}






const double Big = -1.e16;


int main(int argc, char **argv){
  if(argc != 3){
    cout << "Usage :\n";
    cout << "\t Number of obs, seed_obs\n";
    exit(0);
  }
  cout << "omygod" << endl;
  int nd  = atoi(argv[1]);
  uint32_t seed_obs=atoi(argv[2]);//
  int cas=23;

   
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
  PyObject *pFunc_init = PyObject_GetAttrString(pModule, (char*)"initialize_case");//nom de la fonction
  PyObject *pArgs = PyTuple_Pack(1, PyLong_FromLong(cas)); //tuple vide
  PyObject *pFunc_exp = PyObject_GetAttrString(pModule, (char*)"exp_datab");//nom de la fonction

  pValue = PyObject_CallObject(pFunc_exp, pArgs); //pvalue est alors l'objet de retour.

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

  pFunc_model = PyObject_GetAttrString(pModule, (char*)"run_model");//nom de la fonction. on est obligés de setup le chosen_case.
  //récupération des observations. Attention au déréférencement du pointeur et au nombre de données.
  //if(PyList_Check(pValue)!=1){cerr << "erreur : la fonction exp_datab n'a pas renvoyé une liste" << endl;}

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


   // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(dim_theta); lb_t(0)=0;lb_t(1)=0;lb_t(2)=0; //0.2 avant
  VectorXd ub_t(dim_theta); ub_t(0)=1;ub_t(1)=1;ub_t(2)=1;

  //VectorXd lb_hpars(3); lb_hpars(0)=1e3;lb_hpars(1)=1e3;lb_hpars(2)=1; //-5 avant
  //VectorXd ub_hpars(3); ub_hpars(0)=1e7;ub_hpars(1)=1e5;ub_hpars(2)=100; //bornes sur l'optimisation des hpars. edm exp lcor


  VectorXd lb_hpars(4); lb_hpars(0)=-1e5;lb_hpars(1)=9e4;lb_hpars(2)=0.1;lb_hpars(3)=-5;
  VectorXd ub_hpars(4); ub_hpars(0)=1e5;ub_hpars(1)=1e5;ub_hpars(2)=1e4;ub_hpars(3)=5; //bornes sur l'optimisation des hpars. edm exp lcor

  //VectorXd lb_hpars(5); lb_hpars(0)=1e3;lb_hpars(1)=1e3;lb_hpars(2)=1;lb_hpars(3)=-5e5;lb_hpars(4)=0.; //-5 avant
  //VectorXd ub_hpars(5); ub_hpars(0)=5e5;ub_hpars(1)=3e5;ub_hpars(2)=25;ub_hpars(3)=5e5;ub_hpars(4)=5e4; //bornes sur l'optimisation des hpars. edm exp lcor

  
  //Points sur lesquels on va faire les prédictions du modèle. car sinon juste en 6 points pas top. Mais commençons par les points expérimentaux.
  int samp_size=80; //80 avant
  VectorXd X_predictions(samp_size); for (int i=0;i<samp_size;i++){X_predictions(i)=0.01+30*double(i)/double(samp_size);}  

  default_random_engine generator(123456);

  //pour la MCMC
  int nombre_steps_mcmc=1e5;
  int nombre_samples_collected=200;
  int nautocor=2000;

  VectorXd X_init_mcmc(3);
  X_init_mcmc << 0.3,0.15,0.8;
  MatrixXd COV_init=MatrixXd::Identity(3,3);
  COV_init(0,0)=pow(5e-2,2); //pour KOH separate : 1e-2 partout fonctionne bien.
  COV_init(1,1)=pow(5e-2,2);
  COV_init(2,2)=pow(3e-2,2);

  auto lambda_model_23=[pFunc_init](VectorXd const &X, VectorXd const & theta) mutable{
    //renvoie toutes les prédictions du modèle aux points donnés par X.

  PyObject *pArgs = PyTuple_Pack(1, PyLong_FromLong(23)); //
  PyObject_CallObject(pFunc_init, pArgs); 
    VectorXd pred(X.size());
    for(int i=0;i<X.size();i++){
      VectorXd x(1);
      x(0)=X(i);
      pred(i)=my_model(x,GPtoR(theta));
    }
    return pred;
  };



  auto lambda_priormean=[](VectorXd const &X, VectorXd const &hpars){
    //zero
    VectorXd b=VectorXd::Zero(X.size()); return b;
   
    VectorXd c(X.size());
    for(int i=0;i<c.size();i++){
      c(i)=hpars(3)*X(i);
      }
    return c;


    VectorXd a(X.size());
    for(int i=0;i<a.size();i++){
      a(i)=hpars(3)*exp(hpars(4)*X(i));
    }
    return a;
  };

  //cout << data_exp_20[0].Value() << endl;
  //cout << data_exp_23[0].Value() << endl;
  
  //juste un test. 500 évaluations de modèle ? 

  //construction du DoE initial en LHS (generator) ou Grid Sampling(sans generator)
  DoE doe_init(lb_t,ub_t,200,10);//doe halton. 500 points en dimension 3 (3 paramètres incertains).
  doe_init.WriteGrid("results/save/grid.gnu");
  
  
  list<int> cases={3,4,5,6,14,15,16,18,20,21,22,23};
  //list<int> cases={14};
  vector<Density> Dens_v;
  int time_opt_opti=10; // 10 secondes par optimisation opti
  int time_opt_koh_loc=600; // 10 minutes par optimisation KOH locale
  int time_opt_koh_glo=7200; // 2h pour optimisation KOH globale

  for(int i:cases){
    auto lambda_model=[pFunc_init,i](VectorXd const &X, VectorXd const & theta) mutable{
    //renvoie toutes les prédictions du modèle aux points donnés par X.
      PyObject *pArgs = PyTuple_Pack(1, PyLong_FromLong(i)); //
      PyObject_CallObject(pFunc_init, pArgs); 
      VectorXd pred(X.size());
      for(int i=0;i<X.size();i++){
        VectorXd x(1);
        x(0)=X(i);
        pred(i)=my_model(x,GPtoR(theta));
      }
      return pred;
    };
    auto data=GetObs(pFunc_exp,i);
    auto data_exp=Conversion(data);
    string filename="results/obs"+to_string(i)+".gnu";
    WriteObs(filename,data);

    Density MainDensity(doe_init);
    MainDensity.SetModel(lambda_model);
    MainDensity.SetKernel(Kernel_Z_Linear);
    MainDensity.SetKernelDerivatives(D1Kernel_Z_Matern52,D2Kernel_Z_Matern52,D3Kernel_Z_Matern52);
    MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
    MainDensity.SetLogPriorHpars(logprior_hpars_linearkernel);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetPriorMean(lambda_priormean);
    MainDensity.SetDataExp(data_exp);
    MainDensity.SetXprofile(data_exp[0].GetX());
    Dens_v.push_back(MainDensity);

  }
  VectorXd hpars_koh(5); //inclusion de la prior mean en exponentielle.
  hpars_koh =0.5*(lb_hpars+ub_hpars);
  //hpars_koh <<1e7,1e5,15;

  //courses d'algorithmes
  /*
  {
    Density MainDensity=Dens_v[0]; // sur l'expérience numéro 3 je pense.
    DensityOpt denso(MainDensity);
    double max_time =2;
    //il me faut une liste de thetas. choisis at random disons.
    default_random_engine generator;
    generator.seed(6);
    int nthetas=50;
    vector<VectorXd> thetas;
    double llmean=0;
    for(int i=0;i<nthetas;i++){
      VectorXd t(3);
      t << distU(generator),distU(generator),distU(generator);
      thetas.push_back(t);
    }
    for(int i=0;i<nthetas;i++){
      VectorXd h=denso.HparsOpt(thetas[i],hpars_koh,max_time);
      llmean+=denso.loglikelihood_theta(thetas[i],h)+denso.EvaluateLogPHpars(h);
    }
    llmean/=50;
    cout << "llmean : " << llmean << " pour time : " << max_time << endl;
    exit(0);
  }
  */


  //configuration de l'instance de base de densité


  vector<VectorXd> hpars_guess_v(cases.size());
  for(int i=0;i<cases.size();i++){hpars_guess_v[i]=hpars_koh;}

  cout << "début calibration : " << endl;
  {
    Densities dens(Dens_v);
    dens.SetDimPars(3);
    dens.SetLogPriorPars(logprior_pars);
    //OPTI

    int nhpars_gp=5; //3 paramètres. + sedm et sobs.
    MatrixXd Bounds_hpars_gp(2,nhpars_gp);
    Bounds_hpars_gp(0,0)=1E-3; Bounds_hpars_gp(1,0)=1e2; //variance
    Bounds_hpars_gp(0,1)=1E-2;Bounds_hpars_gp(1,1)=2; //lcor
    Bounds_hpars_gp(0,3)=1E-2;Bounds_hpars_gp(1,3)=2; //lcor
    Bounds_hpars_gp(0,4)=1E-2;Bounds_hpars_gp(1,4)=2; //lcor
    Bounds_hpars_gp(0,2)=1E-3; Bounds_hpars_gp(1,2)=2E-3; //sigma obs
    VectorXd hpars_guess_gp=0.5*(Bounds_hpars_gp.row(0)+Bounds_hpars_gp.row(1)).transpose();
    hpars_guess_gp << 0.7,0.12,0.002,0.12,0.12; //test à la zob

    vector<VectorXd> hgp_guess_v(cases.size());
    for(int i=0;i<cases.size();i++){hgp_guess_v[i]=hpars_guess_gp;}

    double lambda=pow(2.38,2)/3;
    double gamma=0.01;

 

    //KOH
    
    {
    //auto hkohsep=dens.HparsKOH_separate(hpars_guess_v,time_opt_koh_loc);
    //auto hkohpooled=dens.HparsKOH_pooled(hpars_guess_v,time_opt_koh_loc);
    //auto hkohsep=read_hparskoh("results/hparskoh_separate.gnu",cases);
    //auto hkohpooled=read_hparskoh("results/hparskoh_pooled.gnu",cases);
/*
    cout << "hpars koh separate : " << endl;
    for(int i=0;i<cases.size();i++){
      cout << hkohsep[i].transpose() << endl;
    }
    */

    //write hpars koh separate
    /*
    ofstream ofile("results/hparskoh_separate.gnu");
    int c=0;
    for (int i:cases){
      ofile << i << " ";
      for(int j=0;j<hkohsep[c].size();j++){
        ofile << hkohsep[c](j) << " ";
      }
      ofile << endl;
      c++;
    }
    ofile.close();
    */


    //hpars_guess_v=dens.HparsKOH_pooled(hpars_guess_v,time_opt_koh_glo);
    /*
    cout << "hpars koh pooled : " << endl;
    for(int i=0;i<cases.size();i++){
      cout << hkohpooled[i].transpose() << endl;
    }
    */
    
    
    //write hpars koh pooled
    /*
    ofstream ofile2("results/hparskoh_pooled.gnu");
    c=0;
    for (int i:cases){
      ofile2 << i << " ";
      for(int j=0;j<hkohpooled[c].size();j++){
        ofile2 << hkohpooled[c](j) << " ";
      }
      ofile2 << endl;
      c++;
    }
    ofile2.close();
    */
    
    
    
    //MCMCx pooled et separate
    //dens.Run_MCMC_fixed_hpars(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,hkohsep,generator);
    /*
    dens.Run_MCMC_adapt(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,hkohsep,lambda,gamma,generator);
    dens.Autocor_diagnosis(nautocor,"results/diag/autocorkohsep.gnu");
    dens.WriteSamples("results/save/sampleskohsep.gnu");
    dens.WriteAllSamples("results/save/allsampleskohsep.gnu");
    dens.WritePredictionsF(X_predictions,"results/preds/predkohFsep");
    dens.WritePredictions(X_predictions,"results/preds/predkohsep");

    dens.Run_MCMC_adapt(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,hkohpooled,lambda,gamma,generator);
    dens.Autocor_diagnosis(nautocor,"results/diag/autocorkohpooled.gnu");
    dens.WriteSamples("results/save/sampleskohpooled.gnu");
    dens.WriteAllSamples("results/save/allsampleskohpooled.gnu");
    dens.WritePredictionsF(X_predictions,"results/preds/predkohFpooled");
    dens.WritePredictions(X_predictions,"results/preds/predkohpooled");
    
    exit(0);
    */

    //opt. 
    vector<VectorXd> grid_theta=doe_init.GetGrid();
    DensitiesOpt denso(dens);
    vector<vector<VectorXd>> hpars;
    //auto hpars=denso.compute_optimal_hpars(time_opt_opti);
    
    string filename_read="results/hparsopt.gnu";
    //vector<VectorXd> gride;
    string filename="results/hparsopt.gnu";
    denso.read_optimal_hpars(hpars,grid_theta,filename_read,cases.size(),4);
    //denso.write_optimal_hpars(hpars,grid_theta,filename);
    denso.update_hGPs_with_hpars(grid_theta,hpars,Kernel_GP_Matern32,Bounds_hpars_gp,hpars_guess_gp);
    denso.opti_allgps(hgp_guess_v);

    /*
    //phase de multiples burns pour tester.
    string f1,f2,f3,f4,f5;
    f1="results/burn9.gnu";
    f2="results/burn7.gnu";
    f3="results/burn8.gnu";
    f4="results/burn4.gnu";
    f5="results/burn5.gnu"; 
    //première chaine
    COV_init/=4;
    denso.Burn_phase_test(2e4,COV_init,X_init_mcmc,generator,f1);
    exit(0);
    COV_init(1,1)=pow(5e-2,2);
    //deuxième chaine en réduisant le pas en t2
    auto M=denso.Burn_phase_test(2e4,COV_init,X_init_mcmc,generator,f2);
    //troisième chaine en utilisant la matrice déduite.
    denso.Burn_phase_test(2e4,M,X_init_mcmc,generator,f3);
    exit(0);
    denso.Burn_phase_test(1e4,COV_init,X_init_mcmc,generator,f4);
    COV_init*=2;
    denso.Burn_phase_test(1e4,COV_init,X_init_mcmc,generator,f5);
    exit(0);
    */



    //denso.Test_hGPs(600,time_opt_opti);
    double lambda=pow(2.38,2)/3;
    double gamma=0.01;
    //denso.Run_MCMC_opti_hGPs(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,generator);
    denso.Run_MCMC_opti_adapt(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,lambda,gamma,generator);

    denso.WriteSamples("results/save/samplesopt.gnu");
    denso.Autocor_diagnosis(nautocor,"results/diag/autocoropt.gnu");

    denso.WritePredictionsF(X_predictions,"results/preds/predoptF");
    denso.WritePredictions(X_predictions,"results/preds/predopt");
    }
  }
  


/*
{
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

 
}

