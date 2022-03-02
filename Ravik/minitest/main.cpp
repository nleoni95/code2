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
  cout << "expérience : " << case_nr << endl;
  cout << "Jasub : " << Jasub << endl;
  cout << "deltarhosurrho : " << (rhof-rhog)/rhog << endl;


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
    VectorXd obs(1); obs << 0 ;
    VectorXd Xinit=0.5*VectorXd::Ones(3);
    cout << lambda_modelwish(obs,Xinit).transpose() << endl;
  }
  exit(0);
 
}

