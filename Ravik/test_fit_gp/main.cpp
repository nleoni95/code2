// Dans ce fichier : j'essaye de fitter simplement un GP sur des données exponentielles. Voir si le GP y arrive, quels coefficients il me donne. Ensuite, essayer sur une différence d'exponentielles. Sinon : noyau polynômial non ?

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

double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2)))*exp(-d/hpar(2)); //3/2
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

void WriteObs(string filename,vector<DATA> &data){
  ofstream ofile(filename);
  for(auto const &d:data){
    ofile << d.GetX()(0) << " " << d.Value() << endl;
  }
  ofile.close();
}

double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
	/* This is the function you optimize for defining the GP. inclure un prior si on le souhaite. */
	GP* proc = (GP*) data;											//Pointer to the GP
	Eigen::VectorXd p(x.size());									//Parameters to be optimized
	for(int i=0; i<(int) x.size(); i++) p(i) = x[i];				//Setting the proposed value of the parameters
	double value = proc->SetGP(p);									//Evaluate the function
	if (!grad.empty()) {											//Cannot compute gradient : stop!
		std::cout << "Asking for gradient, I stop !" << std::endl; exit(1);
	}
	return value;
};




const double Big = -1.e16;


int main(int argc, char **argv){
  if(argc != 3){
    cout << "Usage :\n";
    cout << "\t Number of obs, seed_obs\n";
    exit(0);
  }
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


   // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(dim_theta); lb_t(0)=0;lb_t(1)=0;lb_t(2)=0; //0.2 avant
  VectorXd ub_t(dim_theta); ub_t(0)=1;ub_t(1)=1;ub_t(2)=1;

  VectorXd lb_hpars(3); lb_hpars(0)=1e3;lb_hpars(1)=1e3;lb_hpars(2)=90; //-5 avant
  VectorXd ub_hpars(3); ub_hpars(0)=1e5;ub_hpars(1)=1e5;ub_hpars(2)=180; //bornes sur l'optimisation des hpars. edm exp lcor

  //Points sur lesquels on va faire les prédictions du modèle. car sinon juste en 6 points pas top. Mais commençons par les points expérimentaux.
  int samp_size=80; //80 avant
  VectorXd X_predictions(samp_size); for (int i=0;i<samp_size;i++){X_predictions(i)=0.01+30*double(i)/double(samp_size);}  
  vector<VectorXd> xpred(X_predictions.size());
  for(int i=0;i<X_predictions.size();i++){
    VectorXd X(1); X(0)=X_predictions(i); xpred[i]=X;
  }

  default_random_engine generator(123456);


  
  //list<int> cases={3,4,5,6,14,15,16,18,20,21,22,23};
  list<int> cases={3};

  for(int i:cases){
    data=GetObs(pFunc_exp,i);
    string filename="results/obs"+to_string(i)+".gnu";
    WriteObs(filename,data);
  }

  //fit d'un GP aux données.
  //Fonctionnement gp olm : si hpars est de taille 2, le bruit est automatiquement mis à 10^-8. Si hpars est de taille différente de 2, le bruit est le 3ème coefficient (en std, pas en variance).
  {
    auto Kernel_GP=[](VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
      double d=abs(x(0)-y(0));
      return pow(x(0)*y(0)+pow(hpar(0),2),hpar(1));
      //return pow(hpar(0),2)*(1+(d/hpar(1))+(1./3)*pow(d/hpar(1),2))*exp(-d/hpar(1)); //5/2
    };

    MatrixXd Bounds_hpars_gp(2,3);
    Bounds_hpars_gp(0,0)=1E1; Bounds_hpars_gp(1,0)=1E3; //stdz
    Bounds_hpars_gp(0,1)=0.1;Bounds_hpars_gp(1,1)=10; //lcor
    Bounds_hpars_gp(0,2)=1E4;Bounds_hpars_gp(1,2)=1E5; //noise std
    VectorXd hpars_guess_gp(3);
    hpars_guess_gp=0.5*(Bounds_hpars_gp.row(0).transpose()+Bounds_hpars_gp.row(1).transpose());
    GP gp(Kernel_GP);
    gp.SetData(data);
    gp.SetGP(hpars_guess_gp);
    cout << "optimisation du gp " <<endl;
    cout  << "par before opt : " << hpars_guess_gp.transpose() << endl;
    gp.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_guess_gp,3);
    hpars_guess_gp=gp.GetPar();
    cout  << "par after opt : " << hpars_guess_gp.transpose() << endl;

    //prédictions

    MatrixXd M=gp.SampleGPDirect(xpred,3,generator); //taille (xpred.size,samples.size)
    MatrixXd P(2,xpred.size());
    for(int i=0;i<xpred.size();i++){
      VectorXd p=gp.Eval(xpred[i]);
      P(0,i)=p(0); //mean pred
      P(1,i)=p(1); //variance.
    }
    //écriture
    string filename="results/preds.gnu";
    ofstream ofile(filename);
    for(int i=0;i<xpred.size();i++){
      ofile << xpred[i](0) << " " << P(0,i) << " " << 3*sqrt(P(1,i)) << " ";
      for(int j=0;j<M.cols();j++){
        ofile << M(i,j) << " ";
      }
      ofile << endl;
    } 
    ofile.close();
  }


  exit(0);

 
}

