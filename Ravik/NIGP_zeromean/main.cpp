// Dans ce fichier : on met en oeuvre différents algorithmes d'optimisation pour KOH. Voir lequel est le meilleur.
// On met en place une quadrature pour évaluer de manière précise l'intégrale KOH.
// On regarde maintenant la sensibilité aux observations.
// On essaye avec un hpar supplémentaire : moyenne de model bias constante
//travaillons à erreur input connue (non optimisée) et dérivées estimées à l'avance.
//commentaires olm et pietro : faire l'invt du jury le plus vite possible,
// chapter 3 : carrés blancs sur points noirs, + figure 10 + DKL numérique. (seulement si ça montre un gain net)
// réorganisation debora : plutôt sans EDM avant le reste ( à faire une fois que MG aura lu de totue façon)
// savoir exactement quels contours sont tracés sur les postérieure param.
// valeurs propres. Regarder la trace ou le déterminant ? hmm. de toute façon olm pas dispo.


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

//dans l'ordre : p,v,DTsub,diam,Tsat,rhof,muf,rhog,cpf,kf,hfg,sigma,fric.
//               0,1,  2,   3,    4,   5,  6,   7,  8,  9, 10,  11,   12
map<int,vector<double>> map_expe={
  {3,{2.000000e+00,3.660000e+00,2.780000e+01,1.956000e-02,1.202115e+02,9.637275e+02,3.059678e-04,1.129006e+00,4.207407e+03,6.743324e-01,2.201557e+06,5.492552e-02,1.527583e-02}},
  {4,{2.000000e+00,3.000000e-01,2.780000e+01,1.956000e-02,1.202115e+02,9.637275e+02,3.059678e-04,1.129006e+00,4.207407e+03,6.743324e-01,2.201557e+06,5.492552e-02,2.639084e-02}},
  {5,{2.000000e+00,1.220000e+00,2.780000e+01,1.956000e-02,1.202115e+02,9.637275e+02,3.059678e-04,1.129006e+00,4.207407e+03,6.743324e-01,2.201557e+06,5.492552e-02,1.910965e-02}},
  {6,{4.000000e+00,1.220000e+00,5.550000e+01,1.956000e-02,1.436125e+02,9.667158e+02,3.214146e-04,2.162668e+00,4.202403e+03,6.721438e-01,2.133333e+06,5.009629e-02,1.930011e-02}},
  {8,{4.000000e+00,1.220000e+00,8.330000e+01,1.956000e-02,1.436125e+02,9.831798e+02,4.642762e-04,2.162668e+00,4.182234e+03,6.512592e-01,2.133333e+06,5.009629e-02,2.084200e-02}},
  {14,{2.000000e+00,1.220000e+00,2.780000e+01,1.956000e-02,1.202115e+02,9.637275e+02,3.059678e-04,1.129006e+00,4.207407e+03,6.743324e-01,2.201557e+06,5.492552e-02,1.910965e-02}},
  {15,{4.000000e+00,3.000000e-01,5.550000e+01,1.956000e-02,1.436125e+02,9.667158e+02,3.214146e-04,2.162668e+00,4.202403e+03,6.721438e-01,2.133333e+06,5.009629e-02,2.669458e-02}},
  {16,{4.000000e+00,3.000000e-01,2.780000e+01,1.956000e-02,1.436125e+02,9.465546e+02,2.411237e-04,2.162668e+00,4.238815e+03,6.829080e-01,2.133333e+06,5.009629e-02,2.500814e-02}},
  {18,{4.000000e+00,1.220000e+00,1.110000e+01,1.956000e-02,1.436125e+02,9.327430e+02,2.085730e-04,2.162668e+00,4.269601e+03,6.849851e-01,2.133333e+06,5.009629e-02,1.774809e-02}},
  {20,{6.000000e+00,3.000000e-01,2.780000e+01,1.956000e-02,1.588324e+02,9.341209e+02,2.111695e-04,3.168816e+00,4.266018e+03,6.850530e-01,2.085638e+06,4.684359e-02,2.430147e-02}},
  {21,{6.000000e+00,1.220000e+00,2.780000e+01,1.956000e-02,1.588324e+02,9.341209e+02,2.111695e-04,3.168816e+00,4.266018e+03,6.850530e-01,2.085638e+06,4.684359e-02,1.778827e-02}},
  {22,{4.000000e+00,3.350000e+00,5.550000e+01,1.067000e-02,1.436125e+02,9.667158e+02,3.214146e-04,2.162668e+00,4.202403e+03,6.721438e-01,2.133333e+06,5.009629e-02,1.772091e-02}},
  {23,{4.000000e+00,1.220000e+00,5.550000e+01,1.067000e-02,1.436125e+02,9.667158e+02,3.214146e-04,2.162668e+00,4.202403e+03,6.721438e-01,2.133333e+06,5.009629e-02,2.207772e-02}}
  };

// MODELE RAVIK.

double ravik_model_physique(double DTsup,VectorXd const & params,int case_nr){
  //params contient en 0 : l'angle de contact, en 1 et 2 les 2 coefs de la corrélation du diamètre.
  //ne pas oublier de convertir params dans le domaine correct.
  vector<double> conds=map_expe.at(case_nr);
  double Tsat=conds[4]; double DTsub=conds[2]; double Twall=Tsat+DTsup; double Tbulk=Tsat-DTsub;
  double angle=params(0)*M_PI/180; double vel=conds[1]; double p=conds[0]; double Dh=conds[3];
  double rhof=conds[5]; double muf=conds[6]; double rhog=conds[7]; double cpf=conds[8]; double kf=conds[9]; double hfg=conds[10]; double sigma=conds[11];
  double Re=rhof*vel*Dh/muf;
  double Pr=muf*cpf/kf;
  double Jasub=rhof*cpf*DTsub/(rhog*hfg);
  double Jasup=rhof*cpf*DTsup/(rhog*hfg);
  double etaf=kf/(rhof*cpf);
  double fric=conds[12];
  double NuD=((fric/8)*(Re-1000)*Pr)/(1+12.7*sqrt(fric/8)*(pow(Pr,2./3.)-1));
  double hfc=NuD*kf/Dh;
  double Dd=params(1)*pow(((rhof-rhog)/rhog),0.27)*pow(Jasup,params(2))*pow(1+Jasub,-0.3)*pow(vel,-0.26);
  double twait=6.1e-3*pow(Jasub,0.6317)/DTsup;
  double chi=max(0.,0.05*DTsub/DTsup);
  double c1=1.243/sqrt(Pr);
  double c2=1.954*chi;
  double c3=-1*min(abs(c2),0.5*c1);
  double K=(c1+c3)*Jasup*sqrt(etaf);
  double tgrowth=pow(0.25*Dd/K,2);
  double freq=1./(twait+tgrowth);
  double N0=freq*tgrowth*3.1415*pow(0.5*Dd,2);
  //ishiihibiki
  double pp=p*1e5; double Tg=Tsat+DTsup+273.15; double TTsat=Tsat+273.15; double rhoplus=log10((rhof-rhog)/rhog);
  double frhoplus=-0.01064+0.48246*rhoplus-0.22712*pow(rhoplus,2)+0.05468*pow(rhoplus,3);
  double Rc=(2*sigma*(1+(rhog/rhof)))/(pp*(exp(hfg*(Tg-TTsat)/(462*Tg*TTsat))-1));
  double Npp=(4.72E5)*(1-exp(-(pow(angle,2))/(8*(pow(0.722,2)))))*(exp(frhoplus*(2.5E-6)/Rc)-1);
  double Nppb;
  if(N0*Npp<exp(-1)){
    Nppb=Npp;
  }
  else if(N0*Npp<exp(1)){
    Nppb=(0.2689*N0*Npp+0.2690)/N0;
  }
  else{
    Nppb=(log(N0*Npp)-log(log(N0*Npp)))/N0;
  }
  double Ca=(muf*K)/(sigma*sqrt(tgrowth));
  double Ca0=2.16*1E-4*(pow(DTsup,1.216));
  double rappD=max(0.1237*pow(Ca,-0.373)*sin(angle),1.);

  double Dinception=rappD*Dd;
  double Dml=Dinception/2.;
  double deltaml=4E-6*sqrt(Ca/Ca0);

  double phiml=rhof*hfg*freq*Nppb*(deltaml*(pow(Dml,2))*(3.1415/12.)*(2-(pow(rappD,2)+rappD)));
  double phiinception=1.33*3.1415*pow(Dinception/2.,3)*rhog*hfg*freq*Nppb;
  double phie=phiml+phiinception;
  double Dlo=1.2*Dd;
  double Asl=(Dlo+Dd)/(2*sqrt(Nppb));
  double tstar=(pow(kf,2))/((pow(hfc,2))*3.1415*etaf); tstar=min(tstar,twait);
  double Ssl=min(1.,Asl*Nppb*tstar*freq);
  double phisc=2*hfc*Ssl*(DTsup+DTsub);
  double phifc=(1-Ssl)*hfc*(DTsup+DTsub);



  return phisc+phifc+phie;
}



double gaussprob(double x,double mu, double sigma){
  //renvoie la probabilité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}

double my_model(VectorXd const &x, VectorXd const &theta,int kase, PyObject *pFunc_model, PyObject *pParamsRavik){
  //taille totale des paramètres dans le modèle : 10.
  // x de dimension 1.
  //création d'une liste pour les arguments
  PyList_SetItem(pParamsRavik,0,PyFloat_FromDouble(theta(0))); //angle. VN 40deg
  PyList_SetItem(pParamsRavik,1,PyFloat_FromDouble(theta(1))); //coef multiplicateur. VN 18.9E6
  PyList_SetItem(pParamsRavik,3,PyFloat_FromDouble(theta(2))); //param de DTsup. VN 0.75
  PyObject *pArgs=PyTuple_New(3);
  PyTuple_SetItem(pArgs,0,PyFloat_FromDouble(x(0)));
  PyTuple_SetItem(pArgs,1,PyList_AsTuple(pParamsRavik));
  PyTuple_SetItem(pArgs,2,PyLong_FromLong(kase));
  return PyFloat_AsDouble(PyObject_CallObject(pFunc_model, pArgs));
}

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(1),2)); //3/2
}

double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2)))*exp(-d/hpar(1)); //3/2
}


double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  double l=abs(hpar(1)); //pour se prémunir des mauvais apprentissages OPT
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/l)+(1./3)*pow(d/l,2))*exp(-d/l); //5/2
}
double D1Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à sigma
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return 2*hpar(0)*(1+(d/hpar(1))+(1./3)*pow(d/hpar(1),2))*exp(-d/hpar(1)); //5/2
}

double D2Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à lcor
  double d=abs(x(0)-y(0));
  double X=d/hpar(1);
  return pow(hpar(0),2)*exp(-X)*pow(X,2)*(d+hpar(1))/(3*pow(hpar(1),2));
}

double Kernel_GP_Matern12(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double cor=pow(hpar(0),2);
  cor*=exp(-abs(x(0)-y(0))/hpar(1)); //phi
  cor*=exp(-abs(x(1)-y(1))/hpar(3)); //BK
  cor*=exp(-abs(x(2)-y(2))/hpar(4)); //COAL
  return cor;
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double cor=pow(hpar(0),2);
  cor*=(1+abs(x(0)-y(0))/hpar(1))*exp(-abs(x(0)-y(0))/hpar(1)); //phi
  cor*=(1+abs(x(1)-y(1))/hpar(3))*exp(-abs(x(1)-y(1))/hpar(3)); //BK
  cor*=(1+abs(x(2)-y(2))/hpar(4))*exp(-abs(x(2)-y(2))/hpar(4)); //COAL
  return cor;
}

double Kernel_GPX_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor X, 3:lcor theta1, 4:lcor theta2, 5:lcortheta3
  double cor=pow(hpar(0),2);
  cor*=(1+abs(x(0)-y(0))/hpar(1))*exp(-abs(x(0)-y(0))/hpar(1)); //phi
  cor*=(1+abs(x(1)-y(1))/hpar(3))*exp(-abs(x(1)-y(1))/hpar(3)); //BK
  cor*=(1+abs(x(2)-y(2))/hpar(4))*exp(-abs(x(2)-y(2))/hpar(4)); //COAL
  cor*=(1+abs(x(3)-y(3))/hpar(5))*exp(-abs(x(3)-y(3))/hpar(5)); //COAL
  return cor;
}

double Kernel_GPX_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor X, 3:lcor theta1, 4:lcor theta2, 5:lcortheta3
  double cor=pow(hpar(0),2);
  double m=pow((x(0)-y(0))/hpar(1),2)+pow((x(1)-y(1))/hpar(3),2)+pow((x(2)-y(2))/hpar(4),2)+pow((x(3)-y(3))/hpar(5),2);
  return cor*exp(-0.5*m);
}


VectorXd RtoGP(VectorXd const & X){
  //transforme theta dans l'espace réel en [0,1]. transformation affine.
  int dim=X.size();
  //bornes de l'intervalle ici. il faut que la moyenne somme à 2*valeur nominale pour que 0.5 soit la moyenne.
  VectorXd lb_t(dim); lb_t(0)=20; lb_t(1)=0.5*18.9E-6; lb_t(2)=0.5*0.75; //0.2 avant
  VectorXd ub_t(dim); ub_t(0)=60; ub_t(1)=1.5*18.9E-6; ub_t(2)=1.5*0.75;
  for(int i=0;i<dim;i++){
    //if(X(i)<lb_t(i) || X(i)>ub_t(i)){cerr << "erreur de dimension rtogp " << i <<endl;}
  }
  VectorXd Xgp(dim);
  for(int i=0;i<dim;i++){
    Xgp(i)=(X(i)-lb_t(i))/(ub_t(i)-lb_t(i));
  }
  return Xgp;
}

VectorXd GPtoR(VectorXd const & X){
  //transforme theta dans l'espace gp vers l'espace réel. transformation affine.
  //bornes de l'intervalle ici.
  int dim=X.size();
  //bornes de l'intervalle ici. il faut que la moyenne somme à 2*valeur nominale pour que 0.5 soit la moyenne.
  VectorXd lb_t(dim); lb_t(0)=20; lb_t(1)=0.5*18.9E-6; lb_t(2)=0.5*0.75; //0.2 avant
  VectorXd ub_t(dim); ub_t(0)=60; ub_t(1)=1.5*18.9E-6; ub_t(2)=1.5*0.75;
  for(int i=0;i<dim;i++){
    //if(X(i)<0 || X(i)>1){cerr << "erreur de dimension gptor " << i <<endl;}
  }
  VectorXd Xr(dim);
  for(int i=0;i<dim;i++){
    Xr(i)=lb_t(i)+(ub_t(i)-lb_t(i))*X(i);
  }
  return Xr;
}


double logprior_hpars(VectorXd const &hpars){
  return 0;
}

double lognorm(double x, double mean, double std){
  return -0.5*pow((x-mean)/std,2);
}

double logprior_pars(VectorXd const &pars){
  //prior gaussien sur les paramètres. ils seront dans l'espace (0,1.)
  double d=0;
  for (int i=0;i<3;i++){
    d+=lognorm(pars(i),0.5,0.3);
  }
  return d;  
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
  cout << "case: " << cas <<"number of obs loaded : " << data.size() << endl;
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

vector<VectorXd> read_samples(string const &filename){
  //lecture d'un fichier de samples theta.
  int dim=3; //dimension des paramètres
  vector<VectorXd> v;
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      int key=stoi(words[0]);
      VectorXd theta(3);
      for(int i=0;i<theta.size();i++){
        theta(i)=stod(words[i]);
      }
      v.push_back(theta);
    }
  }
  cout << "number of samples loaded: " << v.size() << endl;
  return v;
}

pair<vector<VectorXd>,vector<VectorXd>> read_optimalhpars(string const &filename, int dim_theta,int dim_hpars){
  //lecture d'un fichier d'hpars optimaux., qui est écrit de la forme n V, avec n le numéro du cas et V le vecteur d'hyperparamètres.
  vector<VectorXd> thetas;
  vector<VectorXd> hparsv;
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      int key=stoi(words[0]);
      VectorXd theta(dim_theta);
      for(int i=0;i<theta.size();i++){
        theta(i)=stod(words[i]);
      }
      VectorXd hpars(dim_hpars);
      for(int i=0;i<hpars.size();i++){
        hpars(i)=stod(words[i+dim_theta]);
      }
      thetas.push_back(theta);
      hparsv.push_back(hpars);
    }
    cout << "number of samples in the file : " << thetas.size() << endl;
  }
  auto p=make_pair(thetas,hparsv);
  return p;
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

void writeVector(string const &filename, vector<VectorXd> const &v){
  ofstream ofile(filename);
  int size=v[0].size();
  for(int i=0;i<v.size();i++){
    for(int j=0;j<size;j++){
      ofile << v[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

vector<VectorXd> readVector(string const &filename){
  vector<VectorXd> v;
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      int key=stoi(words[0]);
      VectorXd hpars(words.size());
      for(int i=0;i<hpars.size();i++){
        hpars(i)=stod(words[i]);
      }
      v.push_back(hpars);
    }
  }
  return v;
}

void writeVectors(string const &filename, vector<VectorXd> &v1,vector<VectorXd> &v2){
  if(!v1.size()==v2.size()){cerr << "erreur de dimension dans writeVectors." << v1.size() << " " << v2.size() << endl;}
  ofstream ofile(filename);
  int size1=v1[0].size();
  int size2=v2[0].size();
  for(int i=0;i<v1.size();i++){
    for(int j=0;j<size1;j++){
      ofile << v1[i](j) << " ";
    }
    for(int j=0;j<size2;j++){
      ofile << v2[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

void Run_Burn_Phase_MCMC_adapt(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento,function<double(vector<VectorXd>, VectorXd const &)> const & compute_score, function<vector<VectorXd>(VectorXd)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //phase de burn. Inspirée de Andrieu algo 6. On explore la densité pendant toute la phase de burn, puis ensuite on fixe la covariance. Sachant que pour une chaîne de dimension p ,chaque étape fait p+1 évaluations de la vraisemblance.
  int dim=Xcurrento.size();
  VectorXd lambda=VectorXd::Ones(dim); //attention l'initialisation, il faut les mettre à quelles valeurs ? 
  double gamma=1; //initialisation ? c'est le decay de la chaîne.
  double alphastarstar=.44;

  auto eval_alphas=[compute_score,dim,get_hpars](vector<VectorXd> & optimal_hpars,VectorXd Xcurrent,VectorXd Step, double fcurrent,double & fcandidate){
    //renvoie tous les alphas résultants des m_dim_pars évaluations de la log_vraisemblance.
    VectorXd alphas(dim+1);
    optimal_hpars=get_hpars(Xcurrent+Step);
    fcandidate=compute_score(optimal_hpars,Xcurrent+Step);
    alphas(dim)=min(1.,exp(fcandidate-fcurrent));
    //toutes les évaluations supplémentaires de la log-vraisemblance
    for(int i=0;i<dim;i++){
      VectorXd X=Xcurrent; X(i)+=Step(i);
      vector<VectorXd> hpars=get_hpars(X);
      double alpha=min(1.,exp(compute_score(hpars,X)-fcurrent));
      if(isnan(alpha)) {alpha=1e-20;} //cas où alpha est trop faible
      alphas(i)=alpha;
    }
    return alphas;
  };

  auto draw_prop=[dim](VectorXd lambda,MatrixXd COV,default_random_engine & generator, normal_distribution<double> & distN ){
    //tire une proposal de matrice de cov sqrt(lambda)*COV*sqrt(lambda)
    VectorXd Step(dim); for(int j=0;j<dim;j++){Step(j)=distN(generator);}
    MatrixXd sqrtCOV=COV.llt().matrixL();
    VectorXd sqrtlambda(lambda.size());
    sqrtlambda.array()=lambda.array().sqrt();
    VectorXd s=sqrtlambda.asDiagonal()*sqrtCOV*Step;
    return s;
  };

  auto update_params=[gamma,alphastarstar,dim](VectorXd & mu, MatrixXd & COV,VectorXd & lambda,VectorXd alpha,VectorXd Xcurrent){
    //update les paramètres de l'algo MCMC.
    for(int i=0;i<dim;i++){
      lambda(i)*=exp(gamma*(alpha(i)-alphastarstar));
    }
    COV=COV+gamma*((Xcurrent-mu)*(Xcurrent-mu).transpose()-COV);
    COV+=1e-10*MatrixXd::Identity(dim,dim);
    mu=mu+gamma*(Xcurrent-mu);
  };

  int dim_mcmc=dim;
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  VectorXd Xinit=Xcurrento;
  vector<VectorXd> hparsinit=get_hpars(Xinit);
  MatrixXd COV=COV_init;
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(hparsinit,Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  VectorXd mu=Xinit;
  vector<VectorXd> hparscurrent=hparsinit;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(dim);
  MatrixXd acc_var=MatrixXd::Zero(dim,dim);
  VectorXd alphas(dim+1);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    if(nburn>100){gamma=1/10;}
    if(nburn>500){gamma=1/20;}
    if(nburn>2000){gamma=1/50;}
    //decay du gamma.
    VectorXd Step=draw_prop(lambda,COV,generator,distN);
    if(in_bounds(Xcurrent+Step)){
      vector<VectorXd> hparscandidate;
      double fcandidate;
      alphas=eval_alphas(hparscandidate,Xcurrent,Step,fcurrent,fcandidate); //on évalue hparscandidate en même temps.
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent+=Step;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    update_params(mu,COV,lambda,alphas,Xcurrent);
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(dim_mcmc))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2));
  MatrixXd CovAdapt=COV*lambda.asDiagonal();
  auto end=chrono::steady_clock::now();
  cout << "burn phase adapt over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovAdapt << endl;
  cout << "covprop old : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovAdapt;
}



void Run_Burn_Phase_MCMC(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento,function<double(vector<VectorXd>, VectorXd const &)> const & compute_score, function<vector<VectorXd>(VectorXd)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //phase de burn.
  int dim_mcmc=COV_init.cols();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  VectorXd Xinit=Xcurrento;
  vector<VectorXd> hparsinit=get_hpars(Xinit);
  MatrixXd COV=COV_init;
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(hparsinit,Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  vector<VectorXd> hparscurrent=hparsinit;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(dim_mcmc);
  MatrixXd acc_var=MatrixXd::Zero(dim_mcmc,dim_mcmc);
  auto begin=chrono::steady_clock::now();
  cout << "fcurrent : " << fcurrent << endl;
  for(int i=0;i<nburn;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds(Xcandidate)){
      vector<VectorXd> hparscandidate=get_hpars(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
    //cout << fcandidate;
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
        //cout << " +";
      }
      //cout << endl;
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

tuple<vector<VectorXd>,vector<vector<VectorXd>>,vector<VectorXd>,vector<double>,vector<vector<VectorXd>>> Run_MCMC(int nsteps,int nsamples, VectorXd const & Xinit,MatrixXd const & COV_init,function<double(vector<VectorXd>, VectorXd const &)> const & compute_score, function<vector<VectorXd>(VectorXd)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //MCMC générique. renvoie un tuple <samples,hparsofsamples,allsamples>
  cout << "starting MCMC with " << nsteps << " samples." << endl;
  int dim_mcmc=Xinit.size();
  vector<VectorXd> samples;
  vector<vector<VectorXd>> hparsofsamples;
  vector<VectorXd> allsamples;
  vector<vector<VectorXd>> allhpars;
  vector<double> scores_of_samples;
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  VectorXd Xinit0=Xinit;
  Run_Burn_Phase_MCMC(nsteps*0.1,COV,Xinit0,compute_score,get_hpars,in_bounds,generator);
  //scaling
  cout << "no scaling" << endl;
  //COV=scale_covmatrix(COV,Xinit0,compute_score,get_hpars,in_bounds,0,generator,"results/scaling.gnu");
  MatrixXd sqrtCOV=COV.llt().matrixL();
  vector<VectorXd> hparsinit=get_hpars(Xinit0);
  double finit=compute_score(hparsinit,Xinit0);
  VectorXd Xcurrent=Xinit0;
  double fcurrent=finit;
  vector<VectorXd> hparscurrent=hparsinit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds(Xcandidate)){
      vector<VectorXd> hparscandidate=get_hpars(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      samples.push_back(Xcurrent);
      hparsofsamples.push_back(hparscurrent);
      scores_of_samples.push_back(fcurrent);
    }
    allsamples.push_back(Xcurrent);
    allhpars.push_back(hparscurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << samples.size() << endl;
  cout << samples[0].transpose() << endl;
  cout << samples[1].transpose() << endl;
  cout << samples[2].transpose() << endl;
  cout << samples[3].transpose() << endl;
  cout << samples[4].transpose() << endl;
  auto tp=make_tuple(samples,hparsofsamples,allsamples,scores_of_samples,allhpars);
  return tp;
}

tuple<vector<VectorXd>,vector<vector<VectorXd>>,vector<VectorXd>,vector<double>,vector<vector<VectorXd>>> Run_MCMC_noburn(int nsteps,int nsamples, VectorXd const & Xinit,MatrixXd const & COV_init,function<double(vector<VectorXd>, VectorXd const &)> const & compute_score, function<vector<VectorXd>(VectorXd)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //MCMC générique. renvoie un tuple <samples,hparsofsamples,allsamples>
  cout << "starting MCMC with " << nsteps << " samples." << endl;
  int dim_mcmc=Xinit.size();
  vector<VectorXd> samples;
  vector<vector<VectorXd>> hparsofsamples;
  vector<VectorXd> allsamples;
  vector<vector<VectorXd>> allhpars;
  vector<double> scores_of_samples;
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  VectorXd Xinit0=Xinit;
  //scaling
  cout << "no scaling" << endl;
  //COV=scale_covmatrix(COV,Xinit0,compute_score,get_hpars,in_bounds,0,generator,"results/scaling.gnu");
  MatrixXd sqrtCOV=COV.llt().matrixL();
  vector<VectorXd> hparsinit=get_hpars(Xinit0);
  double finit=compute_score(hparsinit,Xinit0);
  VectorXd Xcurrent=Xinit0;
  double fcurrent=finit;
  vector<VectorXd> hparscurrent=hparsinit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds(Xcandidate)){
      vector<VectorXd> hparscandidate=get_hpars(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      samples.push_back(Xcurrent);
      hparsofsamples.push_back(hparscurrent);
      scores_of_samples.push_back(fcurrent);
    }
    allsamples.push_back(Xcurrent);
    allhpars.push_back(hparscurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << samples.size() << endl;
  cout << samples[0].transpose() << endl;
  cout << samples[1].transpose() << endl;
  cout << samples[2].transpose() << endl;
  cout << samples[3].transpose() << endl;
  cout << samples[4].transpose() << endl;
  auto tp=make_tuple(samples,hparsofsamples,allsamples,scores_of_samples,allhpars);
  return tp;
}

void compute_Rubin(vector<vector<VectorXd>> & v){
      auto begin=chrono::steady_clock::now();
  //calculer le critère de Rubin pour la variance inter chaînes.
  //centrer les échantillons
  int m=v.size(); //nombre de chaînes
  int n=v[0].size(); //taille des chaînes
  int d=v[0][0].size(); //dimension des chaînes
  vector<VectorXd> means;
  for(int i=0;i<m;i++){
    VectorXd mean=VectorXd::Zero(d);
    for(int j=0;j<n;j++){
      mean+=v[i][j];
    }
    mean/=n;
    means.push_back(mean);
  }
  for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
      v[i][j]-=means[i];
    }
  }
  VectorXd totalmean=VectorXd::Zero(d);
  for(int i=0;i<m;i++){
    totalmean+=means[i];
  }
  totalmean/=m;
  VectorXd B=VectorXd::Zero(d); //between-sequence variance
  for(int i=0;i<m;i++){
    B.array()+=(means[i]-totalmean).array().square();
  }
  B*=(1.0*n)/(m-1.0);
  VectorXd W=VectorXd::Zero(d);
  for(int i=0;i<m;i++){
    VectorXd sjs=VectorXd::Zero(d);
    for(int j=0;j<n;j++){
      sjs.array()+=(v[i][j]-means[i]).array().square();
    }
    sjs/=n-1.0;
    W+=sjs;
  }
  W/=1.0*m;
  cout << "B : " << B.transpose() << endl;
  cout << "W : " << W.transpose() << endl;
  VectorXd var=W*(n-1.0)/(n*1.0)+B*(1.0/n);
  VectorXd R=(var.array()/W.array()).sqrt();
  cout << "R : " << R.transpose() << endl;

    auto end=chrono::steady_clock::now();
  cout << "Rubin criterion over.  " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s " << endl;
}


vector<vector<VectorXd>> revert_vector(vector<vector<VectorXd>> const &v){
  //inverse les deux vecteurs.
  int s0=v.size();
  int s1=v[0].size();
  vector<vector<VectorXd>> res;
  for(int i=0;i<s1;i++){
    vector<VectorXd> tmp;
    for(int j=0;j<s0;j++){
      tmp.push_back(v[j][i]);
    }
    res.push_back(tmp);
  }
  return res;
}

int optroutine(nlopt::vfunc optfunc,void *data_ptr, vector<double> &x, vector<double> const & lb_hpars, vector<double> const & ub_hpars, double max_time){
    //routine d'optimisation sans gradient

  //paramètres d'optimisation
  double ftol_large=1e-5;
  double xtol_large=1e-3;
  double ftol_fin=1e-15;
  double xtol_fin=1e-7;
  // 1 optimiseur local et un global.
  nlopt::opt local_opt(nlopt::LD_LBFGS,x.size());
  local_opt.set_max_objective(optfunc, data_ptr); 
  local_opt.set_ftol_rel(ftol_large);
  local_opt.set_xtol_rel(xtol_large);
  local_opt.set_lower_bounds(lb_hpars);
  local_opt.set_upper_bounds(ub_hpars);

  nlopt::opt opt(nlopt::GD_MLSL_LDS, x.size());
  opt.set_max_objective(optfunc, data_ptr);
  opt.set_lower_bounds(lb_hpars);
  opt.set_upper_bounds(ub_hpars);
  //pas de contrainte de temps.
  opt.set_maxtime(max_time); //20 secondes au max.
  opt.set_local_optimizer(local_opt);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  //on relance une opti locale à partir du max. trouvé.
  local_opt.set_ftol_rel(ftol_fin);
  local_opt.set_xtol_rel(xtol_fin);
  fin=local_opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  return fin;
}

double optfuncKOH_pooled(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* fonction à optimiser pour trouver les hpars koh.*/
  auto ptp=(tuple<const vector<MatrixXd> *,vector<Density>*>*) data; //cast
  auto tp=*ptp;
  const vector<MatrixXd> *Residus_v=get<0>(tp);
  vector<Density> *d=get<1>(tp);
  //transformer x en vector de vectorXd pour avoir tous les hpars
  vector<VectorXd> hpars_v;
  int c=0;
  for(int i=0;i<d->size();i++){
    VectorXd h(d->at(i).GetBoundsHpars().first.size());
    for(int j=0;j<h.size();j++){
      h(j)=x[c];
      c++;
    }
    hpars_v.push_back(h);
  }
  //il faut que toutes les densités aient le même grid en theta.
  vector<LDLT<MatrixXd>> ldlt_v;
  //vecteur de tous les LDLTS.
  for(int i=0;i<d->size();i++){
    auto xlocs=*(d->at(i).GetXconverted());
    MatrixXd M=d->at(i).Gamma(xlocs,hpars_v[i])+d->at(i).Get_IncX();
    auto m=d->at(i).IncX(xlocs);
    LDLT<MatrixXd> L(M);
    ldlt_v.push_back(L);
  }

  vector<double> prob(Residus_v->at(0).cols());
  for(int i=0;i<prob.size();i++){
    double g=0;
    for(int j=0;j<d->size();j++){
      VectorXd priormean=d->at(j).EvaluatePMean(d->at(j).GetXprofile(),hpars_v[j]);
      double ll=d->at(j).loglikelihood_fast(Residus_v->at(j).col(i)-priormean,ldlt_v[j]);
      g+=ll;
    }
    prob[i]=g;
  }
  double logvstyp=-200;
  //normalisation et passage à l'exponentielle.
  //le max doit être le même sur toutes les itérations... d'où la nécessité de le définir à l'extérieur !!!
  for(int i=0;i<prob.size();i++){
    //passage à l'exponentielle.
    //on suppose que le logprior des paramètres est le même pour tous, et correspond à celui de la première densité.
    double l=prob[i];
    VectorXd theta=d->at(0).GetGrid()->at(i);
    double logprior=d->at(0).EvaluateLogPPars(theta);
    double f=exp(l+logprior-logvstyp);
    if(isinf(f)){cerr << "erreur myoptfunc_koh : infini. Valeur de la fonction : " << l+logprior << endl;}
    prob[i]=f;
  }

  //calcul de l'intégrale. suppose un grid régulier. modifier si on veut un prior pour les paramètres.
  //multiplication des priors pour chaques hyperparamètres !
  double res=accumulate(prob.begin(),prob.end(),0.0); res/=prob.size();
  return res;
};


vector<VectorXd> HparsKOH_pooled(vector<Density>& vDens,VectorXd const & hpars_guess, double max_time) {
  //optimisation KOH poolée entre toutes les densités. Il faut faire quoi alors ? On est obligé de calculer toutes les valeurs y-ftheta avant de passer à l'optfct. Sinon ça prend trop de temps.
  auto begin=chrono::steady_clock::now();
  int dim=vDens.size();
  int dim_hpars=hpars_guess.size();
  vector<MatrixXd> residus_v(dim);
  for(int i=0;i<dim;i++){
    VectorXd expvalues=vDens[i].GetExpData()->at(0).Value();
    VectorXd xvalues=vDens[i].GetXprofile();
    MatrixXd Residustheta(expvalues.size(),vDens[i].GetGrid()->size());
    for(int j=0;j<Residustheta.cols();j++){
      VectorXd theta=vDens[i].GetGrid()->at(j);
      Residustheta.col(j)=expvalues-vDens[i].EvaluateModel(xvalues,theta);
    }
    residus_v[i]=Residustheta;
  }
  auto tp=make_tuple(&residus_v,&vDens);
  //création des bornes des hpars et du guess. tout est fait en vector pour pouvoir gérer les tailles sans s'embêter.
  vector<double> lb_hpars,ub_hpars,guess;
  for(int i=0;i<dim;i++){
    auto p=vDens[i].GetBoundsHpars();
    for(int j=0;j<hpars_guess.size();j++){
      lb_hpars.push_back(p.first(j));
      ub_hpars.push_back(p.second(j));
      guess.push_back(hpars_guess(j));
    }
  }

  int fin=optroutine(optfuncKOH_pooled,&tp,guess,lb_hpars,ub_hpars,max_time);
  cout << "fin de l'opt koh pooled : message " << fin << endl;
  //il faut repasser guess en vector<vectorXd>.
  vector<VectorXd> ret;//=hpars_guess;
  int c=0;
  for(int i=0;i<dim;i++){
    VectorXd v(dim_hpars);
    for(int j=0;j<dim_hpars;j++){
      v(j)=guess[c];
      c++;
    }
    ret.push_back(v);
  }
  auto end=chrono::steady_clock::now();
  cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s." << endl;
  return ret;
}


vector<VectorXd> HparsKOH_separate(vector<Density>& vDens,VectorXd const & hpars_guess, double max_time) {
  //optimisation KOH pour chaque densité séparément. Attention, le max_time est divisé entre chaque optimisation.
  //un seul hpars_guess. on pourra changer ça si besoin. 
  auto begin=chrono::steady_clock::now();
  int dim=vDens.size();
  double indiv_time=max_time/dim;
  vector<VectorXd> hpars_koh;
  for(int i=0;i<dim;i++){
    VectorXd h=vDens[i].HparsKOH(hpars_guess,indiv_time);
    hpars_koh.push_back(h);
  }
  return hpars_koh;
}

double optfunc_opti(const std::vector<double> &x, std::vector<double> &grad, void *data){
  //pour chercher le maximum d'une densité liée à 1 configuration expérimentale.
  auto d=(DensityOpt*) data; //cast
  VectorXd X(x.size()); for(int j=0;j<x.size();j++){X(j)=x[j];}
  VectorXd p=d->EvaluateHparOpt(X);
  double l=d->loglikelihood_theta_incx(X,p);
  return l;
};

void drawPrior(default_random_engine &generator,string filename){
  //tire des échantillons de la prior, MVN. 50000 points comme les autres. les affiche dans un fichier.
  int npts=50000;
  VectorXd mean=0.5*VectorXd::Ones(3);
  MatrixXd cov=pow(0.3,2)*MatrixXd::Identity(3,3);
  MatrixXd sqrtCOV=cov.llt().matrixL();
  vector<VectorXd> v(npts);
  for (int i=0;i<npts;i++){
    VectorXd N(3);
    for (int j=0;j<3;j++){N(j)=distN(generator);}
    VectorXd x=mean+sqrtCOV*N;
    v[i]=x;
  }
  writeVector(filename,v);

}


double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
	/* This is the function you optimize for defining the GP */
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
  cout << "OUICBON" << endl;

  int nd  = atoi(argv[1]);
  uint32_t seed_obs=atoi(argv[2]);//
  //int cas=6;
  default_random_engine generator(123456);

  /*Paramètres de la simulation*/
  //pour la MCMC
  int nombre_steps_mcmc=5e5;
  int nombre_samples_collected=1000; //1 sample tous les 500. 
  int nautocor=2000;

  int time_opt_opti=10; // 10 secondes par optimisation opti
  int time_opt_koh_loc=600; // 10 minutes par optimisation KOH locale
  int time_opt_koh_glo=7200; // 2h pour optimisation KOH globale
  double inputerr=0.37; //sigma=0.1K d'incertitude en input.
  // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(dim_theta); for(int i=0;i<dim_theta;i++){lb_t(i)=-0.4;}
  VectorXd ub_t(dim_theta); for(int i=0;i<dim_theta;i++){ub_t(i)=1.4;}
  VectorXd lb_hpars(2); lb_hpars(0)=1e4;lb_hpars(1)=1;
  VectorXd ub_hpars(2); ub_hpars(0)=4e6;ub_hpars(1)=25; //bornes sur l'optimisation des hpars. edm exp lcor
  //VectorXd lb_hpars(2); lb_hpars(0)=1e4;lb_hpars(1)=0.1;
  //VectorXd ub_hpars(2); ub_hpars(0)=1e6;ub_hpars(1)=25; //bornes sur l'optimisation des hpars. edm exp lcor
  VectorXd lb_mi(3); lb_mi(0)=20; lb_mi(1)=0.5*18.9E-6; lb_mi(2)=0.5*0.75; //0.2 avant
  VectorXd ub_mi(3); ub_mi(0)=60; ub_mi(1)=1.5*18.9E-6; ub_mi(2)=1.5*0.75;
  VectorXd lb_t_support(dim_theta); for(int i=0;i<dim_theta;i++){lb_t_support(i)=-lb_mi(i)/(ub_mi(i)-lb_mi(i));}
  VectorXd ub_t_support=VectorXd::Zero(dim_theta); ub_t_support(0)=(90-lb_mi(0))/(ub_mi(0)-lb_mi(0));
  cout << "lb_t et ub_t support :" << lb_t_support.transpose() << ub_t_support.transpose() << endl;
 
  VectorXd hpars_z_guess=0.5*(lb_hpars+ub_hpars);

  //bornes pour hGPs
  MatrixXd Bounds_hpars_gp(2,5);
  Bounds_hpars_gp(0,0)=1E-4; Bounds_hpars_gp(1,0)=1e4; //variance
  Bounds_hpars_gp(0,2)=1E-4; Bounds_hpars_gp(1,2)=1e4; //sigma obs
  list<int> l={1,3,4};
  for (int i:l){
    Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=5; //lcors.
  }
  VectorXd hpars_gp_guess(5);
  for (int i=0;i<5;i++){
    hpars_gp_guess(i)=0.5*(Bounds_hpars_gp(1,i)+Bounds_hpars_gp(0,i));
  }
  hpars_gp_guess(0)=1; //var edm
  hpars_gp_guess(2)=1e-3; //var obs




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
  PyObject *pFunc_exp = PyObject_GetAttrString(pModule, (char*)"exp_datab");//nom de la fonction
    //initialisation des paramètres de Ravik
  PyObject * pParamsRavik=PyList_New(10);
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

  PyObject * pFunc_model = PyObject_GetAttrString(pModule, (char*)"run_model");//nom de la fonction. on est obligés de setup le chosen_case.
  //récupération des observations. Attention au déréférencement du pointeur et au nombre de données.
  //if(PyList_Check(pValue)!=1){cerr << "erreur : la fonction exp_datab n'a pas renvoyé une liste" << endl;}

  //construction du vecteur de Density. On construira le vecteur de DensOpt par dessus.
  //vector<int> cases={3,6,16,20};
  //vector<int> cases={6,14,15,16,18,20};
  
  


  vector<int> cases={3,4,5,6,8,14,15,16,18,20,21,22,23}; // total
  //vector<int> cases={3,4,5,6,15,16,18,20,21}; // cases cool sans ceux qui plantent .
  //vector<int> cases={3};
  //vector<int> cases={4,5,6,15,18,20,21,22}; //tous les bons
  //vector<int> cases={3,6,14}; //que quelque cas cette nuit
  //hyperparamètres optimaux pour les GPs des surrogates.
 



  int vsize=cases.size();
  //Points de prédiction, et aussi de construction de surrogate du modèle.
  int samp_size=60; //80 avant
  vector<VectorXd> X_predictions(samp_size);
  VectorXd XPREDS(samp_size);
   for (int i=0;i<samp_size;i++){
    VectorXd x(1);
    x << 0.01+35*double(i)/double(samp_size);
    X_predictions[i]=x;
    XPREDS(i)=x(0);
  }  

  //construction d'un surrogate model de Ravik. En fait l'appel constant à un truc Python nous nique le temps de calcul. Il faut le faire sous forme vecteur.
  // pas besoin de lire les observations. Seulement faire les évaluations de modèle. On veut un surrogate sur l'ensemble des XPREDS.
  
  auto lambda_priormean=[](VectorXd const &X, VectorXd const &hpars){
    VectorXd b=VectorXd::Zero(X.size());
    return b;
  };
  
  //construction du DoE initial en LHS (generator) ou Grid Sampling(sans generator)
  DoE doe_init(lb_t,ub_t,300,10);//doe halton. 300 points en dimension 3 (3 paramètres incertains).
  doe_init.WriteGrid("results/save/grid.gnu");

  vector<Density> vDens;
  for(int i=0;i<cases.size();i++){

    auto lambda_modeltrue=[&pFunc_model,&pParamsRavik,&cases,i](VectorXd const &X, VectorXd const & theta) mutable{
      //renvoie toutes les prédictions du modèle aux points donnés par X.
        VectorXd pred(X.size());
        for(int j=0;j<X.size();j++){
          VectorXd x(1);
          x(0)=X(j);
          pred(j)=my_model(x,GPtoR(theta),cases[i],pFunc_model,pParamsRavik);
        }
        return pred;
      };

      auto lambda_modelwish=[&pFunc_model,&pParamsRavik,&cases,i](VectorXd const &X, VectorXd const & theta) mutable{
      //renvoie toutes les prédictions du modèle aux points donnés par X.
        VectorXd pred(X.size());
        for(int j=0;j<X.size();j++){

          pred(j)=ravik_model_physique(X(j),GPtoR(theta),cases[i]);
        }
        return pred;
      };

    auto data=GetObs(pFunc_exp,cases[i]);
    auto data_exp=Conversion(data);
    vector<VectorXd> XOBS(data_exp[0].GetX().size());
    for (int j=0;j<XOBS.size();j++){
      VectorXd x(1);
      x << data_exp[0].GetX()(j);
      XOBS[j]=x;
    }  
    //test: affichage des observations et des prédictions nominales du modèle.
    VectorXd Xinit=0.5*VectorXd::Ones(3);
    cout << lambda_modeltrue(data_exp[0].GetX(),Xinit).transpose() << endl;
    cout << lambda_modelwish(data_exp[0].GetX(),Xinit).transpose() << endl;
    cout << endl;

    string filename="results/obs"+to_string(cases[i])+".gnu";
    string filename_fit="results/fit"+to_string(cases[i])+".gnu";
    Density MainDensity(doe_init);
    MainDensity.SetModel(lambda_modelwish);
    MainDensity.SetKernel(Kernel_Z_Matern52);
    MainDensity.SetKernelDerivatives(D1Kernel_Z_Matern52,D2Kernel_Z_Matern52,D1Kernel_Z_Matern52);
    MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetPriorMean(lambda_priormean);
    MainDensity.SetDataExp(data_exp);
    MainDensity.SetXprofile(data_exp[0].GetX());
    MainDensity.Compute_derivatives_f(XOBS,X_predictions,20,filename_fit);
    //inputerr à appeler après le calcul des dérivées.
    MainDensity.Set_Inputerr(inputerr);
    MainDensity.WriteObsWithUncertainty(filename);
    vDens.push_back(MainDensity);
  }

  //écriture des prior predictions
  {
    vector<VectorXd> prior_sample;
    VectorXd mean=0.5*VectorXd::Ones(3);
    MatrixXd cov=pow(0.3,2)*MatrixXd::Identity(3,3);
    MatrixXd sqrtCOV=cov.llt().matrixL();
    for(int i=0;i<nombre_samples_collected;i++){
      VectorXd N(lb_t.size());
      for (int j=0;j<N.size();j++){N(j)=distN(generator);}
      VectorXd x=mean+sqrtCOV*N;
      prior_sample.push_back(x);
    }
    for(int i;i<vsize;i++){
      string fname="results/fulldensities/prior/predsF"+to_string(cases[i])+".gnu";
      vDens[i].WritePriorPredictionsF(XPREDS,fname,prior_sample);
    }


  }

  VectorXd X_init_mcmc=0.5*VectorXd::Ones(dim_theta);
  MatrixXd COV_init=MatrixXd::Identity(dim_theta,dim_theta);
  COV_init(0,0)=pow(0.04,2);
  COV_init(1,1)=pow(0.17,2);
  COV_init(2,2)=pow(0.07,2);
  cout << "COV_init : " << endl << COV_init << endl;


//début des calibrations.

    //Phase Opti
  vector<DensityOpt> vDopt;
  //construction du vecteur de densityopt avec DoE fixe.
  
  for(int i=0;i<vsize;i++){
    DensityOpt Dopt(vDens[i]);
    string fname="results/hparsopt"+to_string(cases[i])+".gnu";
    //Dopt.Compute_optimal_hpars(2,fname);
    //Dopt.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
    auto p=read_optimalhpars(fname,3,2);
    Dopt.update_hGPs_noPCA(p.first,p.second,Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
    Dopt.opti_allgps(hpars_gp_guess);
    vDopt.push_back(Dopt);
  }
  

  //MCMC opt avec toutes les densités en même temps. on retente. 
  MatrixXd COV_init_from_FMP=MatrixXd::Zero(3+vsize*lb_hpars.size(),3+vsize*lb_hpars.size());
  VectorXd X_init_from_FMP=VectorXd(3+vsize*lb_hpars.size());

  {

    auto in_bounds=[&vDens,&lb_t_support,&ub_t_support](VectorXd const & X){
      bool in=true;
      if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
        in=false;}
      return in;
    };
    auto get_hpars_opti=[&vDopt,&vsize,&lb_hpars,&ub_hpars](VectorXd const & X){
      vector<VectorXd> p(vsize);
      for(int i=0;i<vsize;i++){
        VectorXd init=vDopt[i].EvaluateHparOpt(X);
        init(0)=max(init(0),lb_hpars(0));
        init(1)=max(init(1),lb_hpars(1));
        init(0)=min(init(0),ub_hpars(0));
        init(1)=min(init(1),ub_hpars(1));
        p[i]=vDopt[i].HparsOpt(X,init,5e-6);
      }
      return p;
    };
    auto compute_score_opti=[&vDopt,&vsize](vector<VectorXd> p, VectorXd const &X){
      double res=0;
      //cout << "comp score : " << endl;
      for(int i=0;i<vsize;i++){
        double d=vDopt[i].loglikelihood_theta_incx(X,p[i]);
        //cout << i << " : " << d << endl;
        res+=d;
      }
      res+=vDopt[0].EvaluateLogPPars(X);
      return res;
    };


    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_opti,get_hpars_opti,in_bounds,generator);
      auto samples_opti=get<0>(res);
      vector<VectorXd> allsamples_opti=get<2>(res);
      vector<vector<VectorXd>> vhpars_opti_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_opti=revert_vector(vhpars_opti_noninverted);
      vector<vector<VectorXd>> allhpars_opti_inverted=get<4>(res);// dans le cas 1D ici, le outer vector est de taille nsteps, et le inner vector de taille 1 (voir fct get_hpars)

      vector<vector<VectorXd>> allhpars_opti=revert_vector(allhpars_opti_inverted);

      //Set des samples dans chaque densité
      for(int i=0;i<vsize;i++){
        vDopt[i].SetNewSamples(samples_opti);
        vDopt[i].SetNewHparsOfSamples(vhpars_opti[i]);
      }
      vDopt[0].SetNewAllSamples(allsamples_opti);
      //diagnostic
      vDopt[0].Autocor_diagnosis(5000,"results/fulldensities/opt/autocor.gnu");
      //écriture des samples. Il faut une fonction dédiée. 
      for(int i=0;i<vsize;i++){
        string fname="results/fulldensities/opt/samp"+to_string(cases[i])+".gnu";
        string fnameall="results/fulldensities/opt/allsamp"+to_string(cases[i])+".gnu";
        string fnameallh="results/fulldensities/opt/allhpars"+to_string(cases[i])+".gnu";
        string fnamepred="results/fulldensities/opt/preds"+to_string(cases[i])+".gnu";
        string fnamepredF="results/fulldensities/opt/predsF"+to_string(cases[i])+".gnu";
        string fnameprior="results/fulldensities/prior/predsF"+to_string(cases[i])+".gnu";

      writeVector(fnameallh,allhpars_opti[i]);
        vDopt[i].WriteSamples(fname);
        vDopt[i].WriteMCMCSamples(fnameall);
        vDopt[i].WritePredictionsF(XPREDS,fnamepredF);
        vDopt[i].WritePredictions(XPREDS,fnamepred);
      }
      //calcul de la matrice de covariance complète.
      //on va faire un petit truc sympa (idée olm : calculer une matrice de covariance a posteriori sur les samples de FMP, puis utiliser ça comme starting point de la densité Bayes.)
      //création d'un grand vector<VectorXd> avec plein d'échantillons FMP.
      vector<VectorXd> grossamples_fmp;
      int t=3; int h=2;
      for(int k=0;k<allsamples_opti.size();k++){
        VectorXd theta=allsamples_opti[k];
        VectorXd grossample(theta.size()+vsize*lb_hpars.size());
        for(int j=0;j<t;j++){
          grossample(j)=theta(j);
        }
        for(int j=0;j<vsize;j++){
          VectorXd hp=allhpars_opti[j][k];
          grossample(3+2*j)=hp(0);
          grossample(3+2*j+1)=hp(1);
        }
        grossamples_fmp.push_back(grossample);
      }
      cout << "grossamples : " << endl;
      cout << grossamples_fmp[0].transpose() << endl;
      cout << grossamples_fmp[1].transpose() << endl;
      cout << grossamples_fmp[2].transpose() << endl;
      cout << grossamples_fmp[3].transpose() << endl;
      //calcul matrice de covariance de grossamples.
      VectorXd sampmean=VectorXd::Zero(grossamples_fmp[0].size());
      for(int j=0;j<grossamples_fmp.size();j++){
        sampmean+=grossamples_fmp[j];
      }
      sampmean/=grossamples_fmp.size();
      for(int j=0;j<grossamples_fmp.size();j++){
        VectorXd x=grossamples_fmp[j]-sampmean;
        COV_init_from_FMP+=x*x.transpose();
      }
      COV_init_from_FMP/=grossamples_fmp.size();
      cout << "COV_init_from_FMP : " << endl;
      cout << COV_init_from_FMP << endl;
      X_init_from_FMP=sampmean;
      cout << "X_init_from_FMP : " << endl;
      cout << X_init_from_FMP.transpose() << endl;
      for(int j=3;j<X_init_from_FMP.size();j++){
        if (X_init_from_FMP(j)<=1.0){
          X_init_from_FMP(j)=2; //pour les longueurs de corrélation mal prédites
        }
      }
      cout << "X_init_from_FMP update: " << endl;
      cout << X_init_from_FMP.transpose() << endl;
      //écriture de la matrice de corrélation dans un fichier pour la réutiliser trkl.

      vector<VectorXd> xsave;
      vector<VectorXd> Msave;
      xsave.push_back(X_init_from_FMP);
      for(int i=0;i<COV_init_from_FMP.cols(); i++){
        Msave.push_back(COV_init_from_FMP.col(i));
      }
      string f1="results/densities/Xinit.gnu";
      string f2="results/middensities/Minit.gnu";
      //writeVector(f1,xsave);
      //writeVector(f2,Msave);
  }

  exit(0);


//MCMC kohpooled toutes les expériences

     // calcul hpars ko pooled
    auto hparskoh_pooled=HparsKOH_pooled(vDens,hpars_z_guess,3300);
    cout << "hparskoh pooled:" << hparskoh_pooled[0].transpose() << endl;

  //cout << hparskoh_separate[1].transpose() << endl;
  {
    auto in_bounds=[&vDens,&lb_t_support,&ub_t_support](VectorXd const & X){
      bool in=true;
      if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
        in=false;}
      return in;
    };
    auto get_hpars_kohs=[&hparskoh_pooled](VectorXd const & X){
      return hparskoh_pooled;
    };

    auto compute_score_kohs=[&vDens,&vsize](vector<VectorXd> p, VectorXd const &X){
      double res=0;
      for(int i=0;i<vsize;i++){
        double d=vDens[i].loglikelihood_theta_incx(X,p[i]);
        res+=d;
      }
      res+=vDens[0].EvaluateLogPPars(X);
      return res;
    };
    auto res=Run_MCMC(3*nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_kohs,get_hpars_kohs,in_bounds,generator);
      auto samples_koh=get<0>(res);
      vector<VectorXd> allsamples_koh=get<2>(res);
      vector<vector<VectorXd>> vhpars_koh_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_koh=revert_vector(vhpars_koh_noninverted);

      //Set des samples dans chaque densité
      for(int i=0;i<vsize;i++){
        vDens[i].SetNewSamples(samples_koh);
        vDens[i].SetNewHparsOfSamples(vhpars_koh[i]);
      }
      vDens[0].SetNewAllSamples(allsamples_koh);
      //diagnostic
      vDens[0].Autocor_diagnosis(5000,"results/fulldensities/kohpooled/autocorkohs.gnu");
      //écriture des samples. Il faut une fonction dédiée. 
      for(int i=0;i<vsize;i++){

        
        string fname="results/fulldensities/kohpooled/samp"+to_string(cases[i])+".gnu";
        string fnameall="results/fulldensities/kohpooled/allsamp"+to_string(cases[i])+".gnu";
        string fnamepred="results/fulldensities/kohpooled/preds"+to_string(cases[i])+".gnu";
        string fnamepredF="results/fulldensities/kohpooled/predsF"+to_string(cases[i])+".gnu";
        vDens[i].WriteSamples(fname);
        vDens[i].WriteMCMCSamples(fnameall);

        vDens[i].WritePredictionsF(XPREDS,fnamepredF);
        vDens[i].WritePredictions(XPREDS,fnamepred);
      }
      //prédictions
  }


exit(0);

//constitution du sample multifb, puis prédictions sur chaque densité. on a 10 chaînes de 5e6 steps, on prend 100 points par chaîne. Donc 1 tous les 50000. 
//extraire intelligemment pour rester dans la mémoire d'un terminal.
{
  int nrepet=10;
  vector<VectorXd> sample_t;
  vector<VectorXd> sample_h;
  for(int i=0;i<nrepet;i++){
    string fname="results/middensities/multifb/allsamp"+to_string(i)+".gnu";
    vector<VectorXd> s=readVector(fname);
    for (int j=0;j<100;j++){
      int index=j*50000;
      sample_t.push_back(s[index].head(3));
      VectorXd h=VectorXd::Zero(26);
      VectorXd hfile=s[index].tail(18);
      for(int i=0;i<vsize;i++){
        if(i<4){
          h(2*i)=hfile(2*i);
          h(2*i+1)=hfile(2*i+1);
        }
        else if (i>5 && i<11){
          h(2*i)=hfile(2*i-4);
          h(2*i+1)=hfile(2*i-4+1);
        }
      }
      sample_h.push_back(h);
    }
  }
  //il faut transformer le vecteur hpars dans la forme choisie
  //iles hpars de l'exp i sont à la place 2*i et 2*i+1.
  vector<vector<VectorXd>> hpars;
  for(int i=0;i<vsize;i++){
    vector<VectorXd> h;
    for(int j=0;j<sample_h.size();j++){
        VectorXd H(2); H << sample_h[j](2*i), sample_h[j](2*i+1);
        h.push_back(H);
    }
    hpars.push_back(h);
  }

  cout << "taille du sample : " << sample_t.size();
  for(int i=0;i<vsize;i++){
    if(i==4 || i==5 || i==11 || i==12){}
    else{
      vDens[i].SetNewSamples(sample_t);
      vDens[i].SetNewHparsOfSamples(hpars[i]);
      string fname="results/middensities/multifb/samp"+to_string(cases[i])+".gnu";
      string fnamepredF="results/middensities/multifb/predsF"+to_string(cases[i])+".gnu";
      string fnamepred="results/middensities/multifb/preds"+to_string(cases[i])+".gnu";
      vDens[i].WriteSamples(fname);
      vDens[i].WritePredictionsF(XPREDS,fnamepredF);
      vDens[i].WritePredictions(XPREDS,fnamepred);
    }
  }
  for(int i=0;i<vsize;i++){
    if(i==4 || i==5 || i==11 || i==12){
        vDens[i].SetNewSamples(sample_t);
        vDens[i].SetNewHparsOfSamples(hpars[i]);
        string fname="results/middensities/multifb/samp"+to_string(cases[i])+".gnu";
        string fnamepredF="results/middensities/multifb/predsF"+to_string(cases[i])+".gnu";
        vDens[i].WriteSamples(fname);
        vDens[i].WritePredictionsF(XPREDS,fnamepredF);
    }
  }
}

exit(0);

//multiples MCMC phase Bayes sur la densité complète 

    {
      int nrepet=10; //on fait 1 MCMC pour l'instant      
      string f1="results/middensities/Xinit.gnu";
      string f2="results/middensities/Minit.gnu";
      vector<VectorXd> xsave=readVector(f1);
      vector<VectorXd> Msave=readVector(f2);
      MatrixXd COV_fromFMP(Msave.size(),Msave.size());
      for(int i=0;i<COV_fromFMP.cols();i++){
        COV_fromFMP.col(i)=Msave[i];
      }
      //tirage de multiples points de départ
      vector<VectorXd> starting_points; starting_points.push_back(xsave[0]);
      for(int i=0;i<nrepet-1;i++){
        VectorXd S(xsave[0].size());
        for(int j=0;j<3;j++){
          S(j)=lb_t(j)+(ub_t(j)-lb_t(j))*distU(generator);
        }
        for(int j=3;j<S.size();j++){
          int u; if(j%2){ //j impair donc edm
          u=0;}
          else{u=1;}
          S(j)=lb_hpars(u)+(ub_hpars(u)-lb_hpars(u))*distU(generator);
        }
        starting_points.push_back(S);
      }

      cout << "starting points : " << endl;
      for(int i=0;i<starting_points.size();i++){
        cout << starting_points[i].transpose() << endl;
      }
      cout << "cov matrix : " << endl;
      cout << COV_fromFMP << endl;
      

      for(int luck=0;luck<nrepet;luck++ ){
        //initial MCMC. remplissage des états initiaux.
        int mcmc_fb_size=3+vsize*lb_hpars.size();
        
        VectorXd X_init_mcmc_fb(mcmc_fb_size);
        MatrixXd COV_init_fb=MatrixXd::Zero(mcmc_fb_size,mcmc_fb_size);
        X_init_mcmc_fb.head(3)=X_init_mcmc;
        COV_init_fb(0,0)=pow(3.0/72,2);
        COV_init_fb(1,1)=pow(5e-7/18.9e-6,2);
        COV_init_fb(2,2)=pow(0.01/0.75,2);
        int m=lb_hpars.size();
        for(int i=0;i<vsize;i++){
          for(int j=0;j<ub_hpars.size();j++){
            X_init_mcmc_fb(3+i*m+j)=0.5*(lb_hpars(j)+ub_hpars(j));
          }
          COV_init_fb(3+i*m,3+i*m)=pow(2e4,2);
          COV_init_fb(3+i*m+1,3+i*m+1)=pow(0.5,2);
          //COV_init_fb(3+i*m+2,3+i*m+2)=pow(0.1,2);
          //COV_init_fb(3+i*m+3,3+i*m+3)=pow(0.2,2);
        }
        cout << "XinitMCMC : " << X_init_mcmc_fb.transpose() << endl;
        cout << "COV init FB : " << endl << COV_init_fb << endl;
        auto in_bounds_fb=[&vDens,&vsize,&lb_hpars,&ub_hpars,&lb_t_support,&ub_t_support](VectorXd const & X){
          VectorXd theta=X.head(3);
          vector<VectorXd> hp;
          for(int i=0;i<vsize;i++){
            VectorXd h(lb_hpars.size());
            for(int j=0;j<h.size();j++){
              h(j)=X(3+j+i*h.size());
            }
            hp.push_back(h);
          }
          bool in=true;
          for(int i=0;i<vsize;i++){
            for(int j=0;j<lb_hpars.size();j++){
              if (hp[i](j)<lb_hpars(j) || hp[i](j)>ub_hpars(j)){
              in=false;
              }
            }
          }
          if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
            in=false;}
          return in;
        };

        auto get_hpars_fb=[](VectorXd const & X){
          //renvoyer un vecteur sans intérêt.
          vector<VectorXd> v(1);
          return v;
        };
        auto compute_score_fb=[&vsize,&lb_hpars,&vDens](vector<VectorXd> p, VectorXd const &X){
          //il faut décomposer X en thetas/ hpars. 3 est la dimension des paramètres
          VectorXd theta=X.head(3);
          vector<VectorXd> hp;
          for(int i=0;i<vsize;i++){
            VectorXd h(lb_hpars.size());
            for(int j=0;j<h.size();j++){
              h(j)=X(3+j+i*h.size());
            }
            hp.push_back(h);
          }
          double res=0;
          for(int i=0;i<vsize;i++){
            double d=vDens[i].loglikelihood_theta_incx(theta,hp[i]);
            //cout << i << " : " << d << endl;
            res+=d;
          }
          res+=logprior_pars(theta);
          return res;
        };
        auto res=Run_MCMC(10*nombre_steps_mcmc,nombre_samples_collected,starting_points[luck],COV_fromFMP,compute_score_fb,get_hpars_fb,in_bounds_fb,generator);
          vector<VectorXd> allsamples_fb_gros=get<2>(res);
          //set des samples dans la première densité pour calculer la corrélation.
          vDens[0].SetNewAllSamples(allsamples_fb_gros);
          //diagnostic d'autocor doit être bien plus gros.
          auto begin=chrono::steady_clock::now();
          //vDens[0].Autocor_diagnosis(20000,"results/middensities/multifb/autocor"+to_string(luck)+".gnu");
          auto end=chrono::steady_clock::now();
          cout << "temps pour calcul autocor bayes:. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" <<  endl;
          string fnameall="results/middensities/multifb/allsamp"+to_string(luck)+".gnu";
          vDens[0].WriteMCMCSamples(fnameall);
    }

  }
//profitons-en pour le calcul du critère de Rubin
{
  int nrepet=10;
  vector<vector<VectorXd>> v;
  for(int i=0;i<nrepet;i++){
    string fname="results/middensities/multifb/allsamp"+to_string(i)+".gnu";
    vector<VectorXd> s=readVector(fname);
    v.push_back(s);
  }
  compute_Rubin(v);
}





exit(0);



//MCMC phase Bayes. sur la densité complète. oskour. 

  {
    string f1="results/fulldensities/Xinit.gnu";
    string f2="results/fulldensities/Minit.gnu";
    vector<VectorXd> xsave=readVector(f1);
    vector<VectorXd> Msave=readVector(f2);
    MatrixXd COV_fromFMP(Msave.size(),Msave.size());
    for(int i=0;i<COV_fromFMP.cols();i++){
      COV_fromFMP.col(i)=Msave[i];
    }
    //initial MCMC. remplissage des états initiaux.
    int mcmc_fb_size=3+vsize*lb_hpars.size();
    
    VectorXd X_init_mcmc_fb(mcmc_fb_size);
    MatrixXd COV_init_fb=MatrixXd::Zero(mcmc_fb_size,mcmc_fb_size);
    X_init_mcmc_fb.head(3)=X_init_mcmc;
    COV_init_fb(0,0)=pow(3.0/72,2);
    COV_init_fb(1,1)=pow(5e-7/18.9e-6,2);
    COV_init_fb(2,2)=pow(0.01/0.75,2);
    int m=lb_hpars.size();
    for(int i=0;i<vsize;i++){
      for(int j=0;j<ub_hpars.size();j++){
        X_init_mcmc_fb(3+i*m+j)=0.5*(lb_hpars(j)+ub_hpars(j));
      }
      COV_init_fb(3+i*m,3+i*m)=pow(2e4,2);
      COV_init_fb(3+i*m+1,3+i*m+1)=pow(0.5,2);
      //COV_init_fb(3+i*m+2,3+i*m+2)=pow(0.1,2);
      //COV_init_fb(3+i*m+3,3+i*m+3)=pow(0.2,2);
    }

    COV_fromFMP/=4;
    //COV_fromFMP(7,7)/=2;
    //COV_fromFMP(8,8)/=2;
    //COV_fromFMP(12,12)/=2;
    //COV_fromFMP(13,13)/=2;
    //COV_fromFMP(14,14)/=2;
    //COV_fromFMP(16,16)/=2;
    //COV_fromFMP(27,27)/=4;
    //COV_fromFMP(28,28)/=4;


    cout << "XinitMCMC : " << xsave[0].transpose() << endl;
    cout << "COV init FB : " << endl << COV_fromFMP << endl;
    auto in_bounds_fb=[&vDens,&vsize,&lb_hpars,&ub_hpars,&lb_t_support,&ub_t_support](VectorXd const & X){
      VectorXd theta=X.head(3);
      vector<VectorXd> hp;
      for(int i=0;i<vsize;i++){
        VectorXd h(lb_hpars.size());
        for(int j=0;j<h.size();j++){
          h(j)=X(3+j+i*h.size());
        }
        hp.push_back(h);
      }
      bool in=true;
      for(int i=0;i<vsize;i++){
        for(int j=0;j<lb_hpars.size();j++){
          if (hp[i](j)<lb_hpars(j) || hp[i](j)>ub_hpars(j)){
          in=false;
          }
        }
      }
      if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
        in=false;}
      return in;
    };

    auto get_hpars_fb=[](VectorXd const & X){
      //renvoyer un vecteur sans intérêt.
      vector<VectorXd> v(1);
      return v;
    };
    auto compute_score_fb=[&vsize,&lb_hpars,&vDens](vector<VectorXd> p, VectorXd const &X){
      //il faut décomposer X en thetas/ hpars. 3 est la dimension des paramètres
      VectorXd theta=X.head(3);
      vector<VectorXd> hp;
      for(int i=0;i<vsize;i++){
        VectorXd h(lb_hpars.size());
        for(int j=0;j<h.size();j++){
          h(j)=X(3+j+i*h.size());
        }
        hp.push_back(h);
      }
      double res=0;
      for(int i=0;i<vsize;i++){
        double d=vDens[i].loglikelihood_theta_incx(theta,hp[i]);
        //cout << i << " : " << d << endl;
        res+=d;
      }
      res+=logprior_pars(theta);
      return res;
    };
    generator.seed(42);
    auto res=Run_MCMC(10*nombre_steps_mcmc,nombre_samples_collected,xsave[0],COV_fromFMP,compute_score_fb,get_hpars_fb,in_bounds_fb,generator);
      auto samples_fb_gros=get<0>(res);
      vector<VectorXd> allsamples_fb_gros=get<2>(res);
      vector<VectorXd> samples_fb_theta;
      vector<vector<VectorXd>> samples_fb_hpars_wrongorder;
      //décomposer le tout en paramètres et hpars.
      for(int i=0;i<samples_fb_gros.size();i++){
        samples_fb_theta.push_back(samples_fb_gros[i].head(3));
        vector<VectorXd> hpars_stepi;
        for(int j=0;j<vsize;j++){
          VectorXd h(lb_hpars.size());
          for(int k=0;k<h.size();k++){
            h(k)=samples_fb_gros[i](3+k+j*h.size());
          }
          hpars_stepi.push_back(h);
        }
        samples_fb_hpars_wrongorder.push_back(hpars_stepi);
      }
      auto samples_fb_hpars=revert_vector(samples_fb_hpars_wrongorder);
      //set des samples dans la première densité pour calculer la corrélation.
      vDens[0].SetNewAllSamples(allsamples_fb_gros);
      //diagnostic d'autocor doit être bien plus gros.
      auto begin=chrono::steady_clock::now();
      vDens[0].Autocor_diagnosis(140000,"results/fulldensities/fb/autocor.gnu");
      auto end=chrono::steady_clock::now();
  cout << "temps pour calcul autocor bayes:. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" <<  endl;
      string fnameall="results/fulldensities/fb/allsamp.gnu";
      vDens[0].WriteMCMCSamples(fnameall);
      //Set des samples dans chaque densité + prédictions
      for(int i=0;i<vsize;i++){


        vDens[i].SetNewSamples(samples_fb_theta);
        vDens[i].SetNewHparsOfSamples(samples_fb_hpars[i]);
        string fname="results/fulldensities/fb/samp"+to_string(cases[i])+".gnu";
        string fnamepred="results/fulldensities/fb/preds"+to_string(cases[i])+".gnu";
        string fnamepredF="results/fulldensities/fb/predsF"+to_string(cases[i])+".gnu";
        vDens[i].WriteSamples(fname);

        vDens[i].WritePredictionsF(XPREDS,fnamepredF);
        vDens[i].WritePredictions(XPREDS,fnamepred);
      }


  }
exit(0);









//MCMC kohseparate toutes les expériences

     // calcul hpars ko separate
    auto hparskoh_separate=HparsKOH_separate(vDens,hpars_z_guess,20*cases.size());
    cout << "hparskoh sep:" << hparskoh_separate[0].transpose() << endl;

  //cout << hparskoh_separate[1].transpose() << endl;
  {
    auto in_bounds=[&vDens,&lb_t_support,&ub_t_support](VectorXd const & X){
      bool in=true;
      if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
        in=false;}
      return in;
    };
    auto get_hpars_kohs=[&hparskoh_separate](VectorXd const & X){
      return hparskoh_separate;
    };

    auto compute_score_kohs=[&vDens,&vsize](vector<VectorXd> p, VectorXd const &X){
      double res=0;
      for(int i=0;i<vsize;i++){
        double d=vDens[i].loglikelihood_theta_incx(X,p[i]);
        res+=d;
      }
      res+=vDens[0].EvaluateLogPPars(X);
      return res;
    };
    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_kohs,get_hpars_kohs,in_bounds,generator);
      auto samples_koh=get<0>(res);
      vector<VectorXd> allsamples_koh=get<2>(res);
      vector<vector<VectorXd>> vhpars_koh_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_koh=revert_vector(vhpars_koh_noninverted);

      //Set des samples dans chaque densité
      for(int i=0;i<vsize;i++){
        vDens[i].SetNewSamples(samples_koh);
        vDens[i].SetNewHparsOfSamples(vhpars_koh[i]);
      }
      vDens[0].SetNewAllSamples(allsamples_koh);
      //diagnostic
      vDens[0].Autocor_diagnosis(nautocor,"results/middensities/koh/autocorkohs.gnu");
      //écriture des samples. Il faut une fonction dédiée. 
      for(int i=0;i<vsize;i++){

        
        string fname="results/middensities/koh/samp"+to_string(cases[i])+".gnu";
        string fnameall="results/middensities/koh/allsamp"+to_string(cases[i])+".gnu";
        string fnamepred="results/middensities/koh/preds"+to_string(cases[i])+".gnu";
        string fnamepredF="results/middensities/koh/predsF"+to_string(cases[i])+".gnu";
        vDens[i].WriteSamples(fname);
        vDens[i].WriteMCMCSamples(fnameall);

        vDens[i].WritePredictionsF(XPREDS,fnamepredF);
        vDens[i].WritePredictions(XPREDS,fnamepred);
      }
      //prédictions
  }


exit(0);




exit(0);



       //MCMC phase Bayes sur les densités séparées. Pour chaque densités : que 2 hpars. La variance et la correlation length.
  {
    //initial MCMC. remplissage des états initiaux.
    int mcmc_fb_size=5;
    
    VectorXd X_init_mcmc_fb(mcmc_fb_size);
    MatrixXd COV_init_fb=MatrixXd::Zero(mcmc_fb_size,mcmc_fb_size);
    X_init_mcmc_fb.head(3)=X_init_mcmc;
    X_init_mcmc_fb(3)=1e6; X_init_mcmc_fb(4)=5;
    COV_init_fb.topLeftCorner(3,3)=COV_init;
    int m=lb_hpars.size();
    COV_init_fb(3,3)=pow(2e4,2);
    COV_init_fb(4,4)=pow(0.5,2);
    cout << "covinitfb" << endl << COV_init_fb << endl;
    cout << "Xinitfb" << endl << X_init_mcmc_fb.transpose() << endl;

for(int i=0;i<vsize;i++){
    auto in_bounds_fb=[&vDens,&vsize,&lb_hpars,&lb_t_support,&ub_t_support](VectorXd const & X){
      VectorXd hp=X.tail(2);
      VectorXd theta=X.head(3);
      bool in=vDens[0].in_bounds_hpars(hp);
      if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
        in=false;}
      return in;
    };

    auto get_hpars_fb=[](VectorXd const & X){
      //renvoyer un vecteur sans intérêt.
      vector<VectorXd> v(1);
      return v;
    };
    auto compute_score_fb=[&vsize,&lb_hpars,&vDens,&i](vector<VectorXd> p, VectorXd const &X){
      //il faut décomposer X en thetas/ hpars. 3 est la dimension des paramètres
      //logpriorhpars=0 I think
      VectorXd theta=X.head(3);
      VectorXd hp=X.tail(2);
      double d=vDens[i].loglikelihood_theta_incx(theta,hp);
      double lp=vDens[i].EvaluateLogPHpars(hp)+vDens[i].EvaluateLogPPars(theta);
      return d+lp;
    }; 
    cout << "début mcmc densité bayes expérience" << cases[i] << endl;
      
    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc_fb,COV_init_fb,compute_score_fb,get_hpars_fb,in_bounds_fb,generator);

      auto samples_fb_gros=get<0>(res);
      vector<VectorXd> allsamples_fb_gros=get<2>(res);
      vector<VectorXd> samples_fb_theta;
      vector<VectorXd> samples_fb_hpars;
      //décomposer le tout en paramètres et hpars.
      for(int j=0;j<samples_fb_gros.size();j++){
        VectorXd X=samples_fb_gros[j];
        samples_fb_theta.push_back(X.head(3));
        samples_fb_hpars.push_back(X.tail(2));
      }
      //set des samples dans la première densité pour calculer la corrélation.
      vDens[i].SetNewAllSamples(allsamples_fb_gros);
      //diagnostic
      vDens[i].Autocor_diagnosis(5000,"results/separatedensities/fb/autocor"+to_string(cases[i])+".gnu");

      vDens[i].SetNewSamples(samples_fb_theta);
      vDens[i].SetNewHparsOfSamples(samples_fb_hpars);

      //écriture des samples. Il faut une fonction dédiée.
      string fname="results/separatedensities/fb/samp"+to_string(cases[i])+".gnu";
      string fnameall="results/separatedensities/fb/allsamp"+to_string(cases[i])+".gnu";
        
      string fnamepred="results/separatedensities/fb/preds"+to_string(cases[i])+".gnu";
      string fnamepredF="results/separatedensities/fb/predsF"+to_string(cases[i])+".gnu";
      vDens[i].WriteSamples(fname);
      vDens[i].WriteMCMCSamples(fnameall); 

            //prédictions
      vDens[i].WritePredictionsF(XPREDS,fnamepredF);
      vDens[i].WritePredictions(XPREDS,fnamepred);

    }
  }


exit(0);



   //MCMC opt sur les densités séparées.


  //construction du vecteur de densityopt avec algorithme adaptatif. construction adaptative des surrogates
  /*
  for(int i=0;i<vsize;i++){
    cout << "FMP construction hGPs case " << cases[i] << endl;
    DensityOpt Dopt(vDens[i]);
    string fname2="results/hparsopt"+to_string(cases[i])+".gnu";
    vector<VectorXd> allthetas; //stockage de tous les thetas utilisés pour la construction, ainsi que leur hpars respectifs.
    vector<VectorXd> allhpars;
    DoE doe_light(lb_t,ub_t,100,2); // DoE light de 100 points.
    auto thetasinit=doe_light.GetGrid();
    Dopt.SetNewDoE(doe_light);
    auto hpinit=Dopt.Compute_optimal_hpars(2,fname2);
    Dopt.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
    Dopt.opti_allgps(hpars_gp_guess);

    allthetas.insert(allthetas.end(),thetasinit.begin(),thetasinit.end());
    allhpars.insert(allhpars.end(),hpinit.begin(),hpinit.end());

    //Début MCMC de sélection de points
      auto in_bounds=[&Dopt](VectorXd const & X){
          return Dopt.in_bounds_pars(X);
      };
      auto get_hpars_opti=[&Dopt](VectorXd const & X){
        vector<VectorXd> p(1);
        p[0]=Dopt.EvaluateHparOpt(X);
        return p;
      };
      auto compute_score_opti=[&Dopt](vector<VectorXd> p, VectorXd const &X){
        double d=Dopt.loglikelihood_theta_incx(X,p[0])+Dopt.EvaluateLogPPars(X);
        return d;
      };
      auto add_npoints=[&Dopt,&nombre_steps_mcmc,&X_init_mcmc,&COV_init,&compute_score_opti,&get_hpars_opti,&in_bounds,&Bounds_hpars_gp,&hpars_gp_guess,&allthetas,&allhpars](int npoints,default_random_engine &generator, int & npts_total, int nsamples_mcmc){
          npts_total+=npoints;
          auto begin=chrono::steady_clock::now();
          auto res=Run_MCMC(nombre_steps_mcmc,nsamples_mcmc,X_init_mcmc,COV_init,compute_score_opti,get_hpars_opti,in_bounds,generator);
          auto begin_hgps=chrono::steady_clock::now();
          vector<VectorXd> samples_opti=get<0>(res);
          vector<VectorXd> selected_thetas;
          vector<double> weights(samples_opti.size());
          for(int j=0;j<weights.size();j++){
            double a=Dopt.EstimatePredError(samples_opti[j]);
            weights[j]=a;
          }
          for(int i=0;i<npoints;i++){
            std::discrete_distribution<int> distribution(weights.begin(), weights.end());
            int drawn = distribution(generator);
            weights[drawn]=0;
            selected_thetas.push_back(samples_opti[drawn]);
          }
          cout << "rajout de " << selected_thetas.size() << " points :" << endl;
          auto selected_hpars=Dopt.update_hGPs_noPCA(selected_thetas,Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess,2);
          Dopt.opti_allgps(hpars_gp_guess);
          auto end_hgps=chrono::steady_clock::now();
          //on store tout
          allthetas.insert(allthetas.end(),selected_thetas.begin(),selected_thetas.end());
          allhpars.insert(allhpars.end(),selected_hpars.begin(),selected_hpars.end());
        };

      int npts_total=doe_light.GetGrid().size();
      add_npoints(200,generator,npts_total,2000);
      add_npoints(200,generator,npts_total,2000);
      //écriture de tous les paramètres et hyperparamètres utilisés pour la construction
      writeVectors(fname2,allthetas,allhpars);
      vDopt.push_back(Dopt);
    
  }
  */
  

    //calibration opti densités séparées
  {
    for(int i=0;i<vsize;i++){
      auto in_bounds=[&vDens,&lb_t_support,&ub_t_support](VectorXd const & X){
              bool in=true;
      if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
        in=false;}
      return in;
      };
      auto get_hpars_opti=[&vDopt,&i,&lb_hpars,&ub_hpars](VectorXd const & X){
        vector<VectorXd> p(1);
        //on fait une optimisation locale en partant du point deviné par le GP. ça permet de ne pas être hors des clous.
        VectorXd init=vDopt[i].EvaluateHparOpt(X);
        init(0)=max(init(0),lb_hpars(0));
        init(1)=max(init(1),lb_hpars(1));
        init(0)=min(init(0),ub_hpars(0));
        init(1)=min(init(1),ub_hpars(1));
        p[0]=vDopt[i].HparsOpt(X,init,1e-4);
        return p;
      };
      auto compute_score_opti=[&vDopt,&i](vector<VectorXd> const &p, VectorXd const &X){

        double d=vDopt[i].loglikelihood_theta_incx(X,p[0])+vDopt[i].EvaluateLogPPars(X);
        return d;
      };
      cout << "début mcmc densité fmp expérience" << cases[i] << endl;
      auto res=Run_MCMC(3*nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_opti,get_hpars_opti,in_bounds,generator);
      auto samples_opti=get<0>(res);
      vector<VectorXd> allsamples_opti=get<2>(res);
      vector<vector<VectorXd>> allhpars_opti_inverted=get<4>(res);// dans le cas 1D ici, le outer vector est de taille nsteps, et le inner vector de taille 1 (voir fct get_hpars) 
      vector<vector<VectorXd>> vhpars_opti_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_opti=revert_vector(vhpars_opti_noninverted);
      vector<VectorXd> allhpars_opti=revert_vector(allhpars_opti_inverted)[0];

      //Set des samples dans chaque densité  
      vDopt[i].SetNewSamples(samples_opti);
      vDopt[i].SetNewHparsOfSamples(vhpars_opti[0]);
      vDopt[i].SetNewAllSamples(allsamples_opti);
      //diagnostic
      vDopt[i].Autocor_diagnosis(nautocor,"results/separatedensities/opt/autocor"+to_string(cases[i])+".gnu");
      //écriture des samples. Il faut une fonction dédiée.
      string fname="results/separatedensities/opt/samp"+to_string(cases[i])+".gnu";
      string fnameall="results/separatedensities/opt/allsamp"+to_string(cases[i])+".gnu";
      string fnameallh="results/separatedensities/opt/allhpars"+to_string(cases[i])+".gnu";
      string fnamepred="results/separatedensities/opt/preds"+to_string(cases[i])+".gnu";
      string fnamepredF="results/separatedensities/opt/predsF"+to_string(cases[i])+".gnu";
      vDopt[i].WriteSamples(fname);
      vDopt[i].WriteMCMCSamples(fnameall);
      writeVector(fnameallh,allhpars_opti);

      //prédictions

      vDopt[i].WritePredictionsF(XPREDS,fnamepredF);
      vDopt[i].WritePredictions(XPREDS,fnamepred);
    }
  }


 //MCMC koh separate sur les densités séparées


//calibration koh separate
  {
    for(int i=0;i<vsize;i++){

    auto in_bounds=[&vDens,&ub_t_support,&lb_t_support](VectorXd const & X){
              bool in=true;
      if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
        in=false;}
      return in;
    };

    auto get_hpars_kohs=[&hparskoh_separate,&i](VectorXd const & X){
      vector<VectorXd> h(1);
      h[0]=hparskoh_separate[i];
      return h;
    };
    auto compute_score_kohs=[&vDens,&i](vector<VectorXd> p, VectorXd const &X){
      double res=vDens[i].loglikelihood_theta_incx(X,p[0])+vDens[i].EvaluateLogPPars(X);
      return res;
    };

    cout << "début mcmc densité koh separate expérience" << cases[i] << endl;
    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_kohs,get_hpars_kohs,in_bounds,generator);
      auto samples_opti=get<0>(res);
      vector<VectorXd> allsamples_opti=get<2>(res);
      vector<vector<VectorXd>> vhpars_opti_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_opti=revert_vector(vhpars_opti_noninverted);

      //Set des samples dans chaque densité
      
        vDens[i].SetNewSamples(samples_opti);
        vDens[i].SetNewHparsOfSamples(vhpars_opti[0]);
      
      vDens[i].SetNewAllSamples(allsamples_opti);
      //diagnostic
      vDens[i].Autocor_diagnosis(nautocor,"results/separatedensities/kohs/autocor"+to_string(cases[i])+".gnu");
      //écriture des samples. Il faut une fonction dédiée.
        string fname="results/separatedensities/kohs/samp"+to_string(cases[i])+".gnu";
        string fnameall="results/separatedensities/kohs/allsamp"+to_string(cases[i])+".gnu";
        string fnamepred="results/separatedensities/kohs/preds"+to_string(cases[i])+".gnu";
        string fnamepredF="results/separatedensities/kohs/predsF"+to_string(cases[i])+".gnu";
        vDens[i].WriteSamples(fname);
        vDens[i].WriteMCMCSamples(fnameall);

                    //prédictions
  
        vDens[i].WritePredictionsF(XPREDS,fnamepredF);
        vDens[i].WritePredictions(XPREDS,fnamepred);
        //prédictions
  }
  }





  exit(0);





  

exit(0);


exit(0);




 




   
   //Phase Opti

/*

  //construction du vecteur de densityopt.
  vector<DensityOpt> vDopt;
  for(int i=0;i<vsize;i++){
    DensityOpt Dopt(vDens[i]);
    string fname="results/hparsopt"+to_string(cases[i])+".gnu";
    //Dopt.Compute_optimal_hpars(2,fname);
    //Dopt.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
    auto p=read_optimalhpars(fname,3,2);
    Dopt.update_hGPs_noPCA(p.first,p.second,Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
    Dopt.opti_allgps(hpars_gp_guess);
    vDopt.push_back(Dopt);
  }
*/




  //lecture des samples python et prédictions en les utilisant.
  {
    string fname="results/samplespython/samplesopt.gnu";
    auto samples=read_samples(fname);
    //évaluation des hpars optimaux pour chacun.
    for(int i=0;i<vsize;i++){
      vDens[i].SetNewSamples(samples);
      //vDopt[i].SetNewHparsOfSamples(hparsopt);
      //prédictions. on rajoute un w pour "wish"
      //string fname="results/save/sampoptw"+to_string(cases[i])+".gnu";
      //string fnamepred="results/preds/predsoptw"+to_string(cases[i])+".gnu";
      string fnamepredF="results/predsfinal/opt/predsF"+to_string(cases[i])+".gnu";
      //vDopt[i].WriteSamples(fname);
      vDens[i].WritePredictionsF(XPREDS,fnamepredF);
      //vDopt[i].WritePredictions(XPREDS,fnamepred);
    }
  }

    //lecture des samples python et prédictions en les utilisant. version KOH
  {
    string fname="results/samplespython/sampleskoh.gnu";
    auto samples=read_samples(fname);
    //évaluation des hpars optimaux pour chacun.
    for(int i=0;i<vsize;i++){
      vDens[i].SetNewSamples(samples);
      //vDopt[i].SetNewHparsOfSamples(hparsopt);
      //prédictions. on rajoute un w pour "wish"
      //string fname="results/save/sampoptw"+to_string(cases[i])+".gnu";
      //string fnamepred="results/preds/predsoptw"+to_string(cases[i])+".gnu";
      string fnamepredF="results/predsfinal/koh/predsF"+to_string(cases[i])+".gnu";
      //vDopt[i].WriteSamples(fname);
      vDens[i].WritePredictionsF(XPREDS,fnamepredF);
      //vDopt[i].WritePredictions(XPREDS,fnamepred);
    }
  }


exit(0);  

 
   //MCMC opt sur les densités séparées.
/*
  {
    for(int i=0;i<vsize;i++){
   auto in_bounds=[&vDens,&lb_t_support,&ub_t_support](VectorXd const & X){
      bool in=true;
      if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
        in=false;}
      return in;
    };
      auto get_hpars_opti=[vDopt,i](VectorXd const & X){
        vector<VectorXd> p(1);
        p[0]=vDopt[i].EvaluateHparOpt(X);
        return p;
      };
      auto compute_score_opti=[vDopt,i](vector<VectorXd> p, VectorXd const &X){
        double d=vDopt[i].loglikelihood_theta_incx(X,p[0]);
        if (isnan(-d)){ cerr << "nan ll. cas numéro " << i << endl;
            cerr << "theta: " << X.transpose() << endl;
            cerr << "hpars: " << p[i].transpose() << endl;
        }
        return d;
      };
      cout << "début mcmc densité fmp expérience" << cases[i] << endl;
      auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_opti,get_hpars_opti,in_bounds,generator);
      auto samples_opti=get<0>(res);
      vector<VectorXd> allsamples_opti=get<2>(res);
      vector<vector<VectorXd>> vhpars_opti_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_opti=revert_vector(vhpars_opti_noninverted);

      //Set des samples dans chaque densité  
      vDopt[i].SetNewSamples(samples_opti);
      vDopt[i].SetNewHparsOfSamples(vhpars_opti[0]);
      vDopt[0].SetNewAllSamples(allsamples_opti);
      //diagnostic
      vDopt[0].Autocor_diagnosis(nautocor,"results/separatedensities/opt/autocor"+to_string(cases[i])+".gnu");
      //écriture des samples. Il faut une fonction dédiée.
      string fname="results/separatedensities/opt/samp"+to_string(cases[i])+".gnu";
      string fnamepred="results/separatedensities/opt/preds"+to_string(cases[i])+".gnu";
      string fnamepredF="results/separatedensities/opt/predsF"+to_string(cases[i])+".gnu";
      vDopt[i].WriteSamples(fname);
      vDopt[i].WritePredictionsF(XPREDS,fnamepredF);
      vDopt[i].WritePredictions(XPREDS,fnamepred);  
      //prédictions
    }
  }

*/




  //lecture des samples python et prédictions en les utilisant.
  {
    string fname="../NIGP/results/samplespython/samplesopt.gnu";
    auto samples=read_samples(fname);
    //évaluation des hpars optimaux pour chacun.
    for(int i=0;i<vsize;i++){
      vDens[i].SetNewSamples(samples);
      //vDopt[i].SetNewHparsOfSamples(hparsopt);
      //prédictions. on rajoute un w pour "wish"
      //string fname="results/save/sampoptw"+to_string(cases[i])+".gnu";
      //string fnamepred="results/preds/predsoptw"+to_string(cases[i])+".gnu";
      string fnamepredF="../NIGP/results/predsfinal/opt/predsF"+to_string(cases[i])+".gnu";
      //vDopt[i].WriteSamples(fname);
      vDens[i].WritePredictionsF(XPREDS,fnamepredF);
      //vDopt[i].WritePredictions(XPREDS,fnamepred);
    }
  }

    //lecture des samples python et prédictions en les utilisant. version KOH
  {
    string fname="../NIGP/results/samplespython/sampleskoh.gnu";
    auto samples=read_samples(fname);
    //évaluation des hpars optimaux pour chacun.
    for(int i=0;i<vsize;i++){
      vDens[i].SetNewSamples(samples);
      //vDopt[i].SetNewHparsOfSamples(hparsopt);
      //prédictions. on rajoute un w pour "wish"
      //string fname="results/save/sampoptw"+to_string(cases[i])+".gnu";
      //string fnamepred="results/preds/predsoptw"+to_string(cases[i])+".gnu";
      string fnamepredF="../NIGP/results/predsfinal/koh/predsF"+to_string(cases[i])+".gnu";
      //vDopt[i].WriteSamples(fname);
      vDens[i].WritePredictionsF(XPREDS,fnamepredF);
      //vDopt[i].WritePredictions(XPREDS,fnamepred);
    }
  }


exit(0);










    //phase KOH pooled
      //auto hparskoh_pooled=HparsKOH_pooled(vDens,hpars_z_guess,1200); //20 minutes
  cout << "hparskoh pooled:" << hparskoh_pooled[0].transpose() << endl;
  //cout << hparskoh_pooled[1].transpose() << endl;
  //mcmc koh pooled toutes les expériences à la fois
/*
  {
   auto in_bounds=[&vDens,&lb_t_support,&ub_t_support](VectorXd const & X){
      bool in=true;
      if(X(0)<lb_t_support(0) || X(0)>ub_t_support(0) || X(1)<lb_t_support(1) || X(2)<lb_t_support(2) ){
        in=false;}
      return in;
    };
    auto get_hpars_kohp=[hparskoh_pooled](VectorXd const & X){
      return hparskoh_pooled;
    };
    auto compute_score_kohp=[vDens,vsize](vector<VectorXd> p, VectorXd const &X){
      double res=0;
      for(int i=0;i<vsize;i++){
        double d=vDens[i].loglikelihood_theta_incx(X,p[i]);
        res+=d;
        if (isnan(d)){ cerr << "nan ll. cas numéro " << i << endl;}
      }
      res+=logprior_pars(X);
      return res;
    };
    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_kohp,get_hpars_kohp,in_bounds,generator);
      auto samples_koh=get<0>(res);
      vector<VectorXd> allsamples_koh=get<2>(res);
      vector<vector<VectorXd>> vhpars_koh_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_koh=revert_vector(vhpars_koh_noninverted);

      //Set des samples dans chaque densité
      for(int i=0;i<vsize;i++){
        vDens[i].SetNewSamples(samples_koh);
        vDens[i].SetNewHparsOfSamples(vhpars_koh[i]);
      }
      vDens[0].SetNewAllSamples(allsamples_koh);
      //diagnostic
      vDens[0].Autocor_diagnosis(nautocor,"results/diag/autocorkohp.gnu");
      //écriture des samples. Il faut une fonction dédiée. 
      for(int i=0;i<vsize;i++){
        string fname="results/save/sampkohp"+to_string(cases[i])+".gnu";
        string fnamepred="results/preds/predskohp"+to_string(cases[i])+".gnu";
        string fnamepredF="results/preds/predskohpF"+to_string(cases[i])+".gnu";
        vDens[i].WriteSamples(fname);
        vDens[i].WritePredictionsF(XPREDS,fnamepredF);
        vDens[i].WritePredictions(XPREDS,fnamepred);
      }
      //prédictions
  }
 */
 //MCMC koh pooled sur les densités séparées

  {
    for(int i=0;i<vsize;i++){

    auto in_bounds=[vDens](VectorXd const & X){
        return vDens[0].in_bounds_pars(X);
    };

    auto get_hpars_kohs=[hparskoh_pooled,i](VectorXd const & X){
      vector<VectorXd> h(1);
      h[0]=hparskoh_pooled[i];
      return h;
    };
    auto compute_score_kohs=[vDens,i](vector<VectorXd> p, VectorXd const &X){
      double res=vDens[i].loglikelihood_theta_incx(X,p[0]);
      return res;
    };

    cout << "début mcmc densité koh pooled expérience" << cases[i] << endl;
    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_kohs,get_hpars_kohs,in_bounds,generator);
      auto samples_opti=get<0>(res);
      vector<VectorXd> allsamples_opti=get<2>(res);
      vector<vector<VectorXd>> vhpars_opti_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_opti=revert_vector(vhpars_opti_noninverted);

      //Set des samples dans chaque densité
      
        vDens[i].SetNewSamples(samples_opti);
        vDens[i].SetNewHparsOfSamples(vhpars_opti[0]);
      
      vDens[0].SetNewAllSamples(allsamples_opti);
      //diagnostic
      vDens[0].Autocor_diagnosis(nautocor,"results/separatedensities/kohp/autocor"+to_string(cases[i])+".gnu");
      //écriture des samples. Il faut une fonction dédiée.
        string fname="results/separatedensities/kohp/samp"+to_string(cases[i])+".gnu";
        string fnamepred="results/separatedensities/kohp/preds"+to_string(cases[i])+".gnu";
        string fnamepredF="results/separatedensities/kohp/predsF"+to_string(cases[i])+".gnu";
        vDens[i].WriteSamples(fname);
        vDens[i].WritePredictionsF(XPREDS,fnamepredF);
        vDens[i].WritePredictions(XPREDS,fnamepred);
      
      //prédictions
  }
  }

  

exit(0);






    //MCMC opt avec toutes les densités en même temps
/*
  {
    //trouver les valeurs max de chaque densité et les utiliser comme seuil, pas moins de /20.
    vector<double> valmax;
    for(int i=0;i<vsize;i++){
      vector<double> lb_tv(lb_t.size()); for(int j=0;j<lb_t.size();j++){lb_tv[j]=lb_t(j);}
      vector<double> ub_tv(ub_t.size());for(int j=0;j<ub_t.size();j++){ub_tv[j]=ub_t(j);}
      vector<double> xguess(lb_tv.size()); for(int j=0;j<ub_t.size();j++){xguess[j]=0.5*(lb_t(j)+ub_t(j));}xguess[2]=0.1;
      DensityOpt* ptr=&vDopt[i];
      optroutine(optfunc_opti,ptr,xguess,lb_tv,ub_tv,120);
      valmax.push_back(optfunc_opti(xguess,xguess,ptr));
    }

    cout << "valmax: "; for (int i=0;i<vsize;i++){cout << valmax[i] << " ";} cout << endl;


    auto in_bounds=[vDens](VectorXd const & X){
        return vDens[0].in_bounds_pars(X);
    };
    auto get_hpars_opti=[vDopt,vsize](VectorXd const & X){
      vector<VectorXd> p(vsize);
      for(int i=0;i<vsize;i++){
        p[i]=vDopt[i].EvaluateHparOpt(X);
      }
      return p;
    };
    auto compute_score_opti=[vDopt,vsize,valmax](vector<VectorXd> p, VectorXd const &X){
      double res=0;
      //cout << "comp score : " << endl;
      for(int i=0;i<vsize;i++){
        double d=vDopt[i].loglikelihood_theta_incx(X,p[i]);
        if (d<valmax[i]-4){
          d=valmax[i]-4;
        }
        //cout << i << " : " << d << endl;
        res+=d;
      }
      //cout << endl << "res : " << res << endl;
      //res+=logprior_pars(X);
      //cout << "res : " << res << endl;
      return res;
    };

    //temporaire : affichage de la ll sur un grid en 2D pour voir ce qui merde autant. à theta(2) fixé.

    VectorXd lb_t2(2); for(int i=0;i<2;i++){lb_t2(i)=0;}
    VectorXd ub_t2(2); for(int i=0;i<2;i++){ub_t2(i)=1;}
    DoE doe_test(lb_t2,ub_t2,200,10);
    ofstream ofile("results/grid.gnu");
    auto vec=doe_test.GetGrid();
    for(auto t:vec){
      VectorXd theta(3); theta << t(0),t(1),0.5;
      double l=compute_score_opti(get_hpars_opti(theta),theta);
      ofile << t(0) << " " << t(1) << " " <<l << endl;
    }

   
   



    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_opti,get_hpars_opti,in_bounds,generator);
      auto samples_opti=get<0>(res);
      vector<VectorXd> allsamples_opti=get<2>(res);
      vector<vector<VectorXd>> vhpars_opti_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_opti=revert_vector(vhpars_opti_noninverted);

      //Set des samples dans chaque densité
      for(int i=0;i<vsize;i++){
        vDopt[i].SetNewSamples(samples_opti);
        vDopt[i].SetNewHparsOfSamples(vhpars_opti[i]);
      }
      vDopt[0].SetNewAllSamples(allsamples_opti);
      //diagnostic
      vDopt[0].Autocor_diagnosis(nautocor,"results/diag/autocoropt.gnu");
      //écriture des samples. Il faut une fonction dédiée. 
      for(int i=0;i<vsize;i++){
        string fname="results/save/sampopt"+to_string(cases[i])+".gnu";
        string fnamepred="results/preds/predsopt"+to_string(cases[i])+".gnu";
        string fnamepredF="results/preds/predsoptF"+to_string(cases[i])+".gnu";
        vDopt[i].WriteSamples(fname);
        vDopt[i].WritePredictionsF(XPREDS,fnamepredF);
        vDopt[i].WritePredictions(XPREDS,fnamepred);
      }
      //prédictions
  }

*/
 
 exit(0);





  




 






  exit(0);
 
}

