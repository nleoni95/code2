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
// dans ce fichier en particulier : on travaille avec une version modifiée de lambda_model qui permet de faire varier tous les paramètres de la corrélation de diamètre de bulle. On fait un OAT pour voir ce qui est le plus influent, et éventuellement ça enrichira nos paramètres.
// l'OAT peut être fait sur plusieurs cas représentatifs ? le 16 je sais qu'il marche bien, un de ceux du début et de la fin ? 


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
int dim_theta=6;

//paramètres pour python
PyObject *pFunc_model;
PyObject *pParamsRavik; //liste comprenant les valeurs nominales des paramètres de Ravik.

double gaussprob(double x,double mu, double sigma){
  //renvoie la probabilité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}

double my_model(VectorXd const &x, VectorXd const &theta, int kase){
  //taille totale des paramètres dans le modèle : 10.
  // x de dimension 1.
  //création d'une liste pour les arguments
  PyList_SetItem(pParamsRavik,0,PyFloat_FromDouble(theta(0))); //angle. VN 40deg
  PyList_SetItem(pParamsRavik,1,PyFloat_FromDouble(theta(1))); //coef multiplicateur. VN 18.9E6
  PyList_SetItem(pParamsRavik,2,PyFloat_FromDouble(theta(2))); //
  PyList_SetItem(pParamsRavik,3,PyFloat_FromDouble(theta(3))); //
  PyList_SetItem(pParamsRavik,4,PyFloat_FromDouble(theta(4))); //
  PyList_SetItem(pParamsRavik,5,PyFloat_FromDouble(theta(5))); //param de DTsup. VN 0.75
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
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(1))+(1./3)*pow(d/hpar(1),2))*exp(-d/hpar(1)); //5/2
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

VectorXd RtoGP(VectorXd const & X){
  //transforme theta dans l'espace réel en [0,1]. transformation affine.
  int dim=X.size();
  //bornes de l'intervalle ici. il faut que la moyenne somme à 2*valeur nominale pour que 0.5 soit la moyenne.
  VectorXd lb_t(dim); lb_t(0)=0.2*40; lb_t(1)=0.5*18.9E-6; lb_t(2)=0.5*0.27; lb_t(3)=0.5*0.75; lb_t(4)=0.5*-0.3; lb_t(5)=0.5*-0.26; //0.2 avant
  VectorXd ub_t(dim); ub_t(0)=1.8*40; ub_t(1)=1.5*18.9E-6; ub_t(2)=1.5*0.27; ub_t(3)=1.5*0.75; ub_t(4)=1.5*-0.3; ub_t(5)=1.5*-0.26;
  for(int i=0;i<dim;i++){
    if(X(i)<lb_t(i) || X(i)>ub_t(i)){cerr << "erreur de dimension rtogp " << i <<endl;}
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
  VectorXd lb_t(dim); lb_t(0)=0.2*40; lb_t(1)=0.5*18.9E-6; lb_t(2)=0.5*0.27; lb_t(3)=0.5*0.75; lb_t(4)=0.5*-0.3; lb_t(5)=0.5*-0.26; //0.2 avant
  VectorXd ub_t(dim); ub_t(0)=1.8*40; ub_t(1)=1.5*18.9E-6; ub_t(2)=1.5*0.27; ub_t(3)=1.5*0.75; ub_t(4)=1.5*-0.3; ub_t(5)=1.5*-0.26;
  for(int i=0;i<dim;i++){
    if(X(i)<lb_t(i) || X(i)>ub_t(i)){cerr << "erreur de dimension gptor " << i <<endl;}
  }
  VectorXd Xr(dim);
  for(int i=0;i<dim;i++){
    Xr(i)=lb_t(i)+(ub_t(i)-lb_t(i))*X(i);
  }
  return Xr;
}


double logprior_hpars(VectorXd const &hpars){
  return 0;
  return -2*log(hpars(0));


  double alpha_ig=19;
  double beta_ig=200;
  return log(pow(beta_ig,alpha_ig)*pow(hpars(2),-alpha_ig-1)*exp(-beta_ig/hpars(2))/tgamma(alpha_ig));
}

double logprior_pars(VectorXd const &pars){
  //prior gaussien sur les paramètres.
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
  for(int i=0;i<nburn;i++){
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

tuple<vector<VectorXd>,vector<vector<VectorXd>>,vector<VectorXd>,vector<double>> Run_MCMC(int nsteps,int nsamples, VectorXd const & Xinit,MatrixXd const & COV_init,function<double(vector<VectorXd>, VectorXd const &)> const & compute_score, function<vector<VectorXd>(VectorXd)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //MCMC générique. renvoie un tuple <samples,hparsofsamples,allsamples>
  int dim_mcmc=Xinit.size();
  vector<VectorXd> samples;
  vector<vector<VectorXd>> hparsofsamples;
  vector<VectorXd> allsamples;
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
  auto tp=make_tuple(samples,hparsofsamples,allsamples,scores_of_samples);
  return tp;
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
    MatrixXd M=d->at(i).Gamma(xlocs,hpars_v[i])+d->at(i).IncX(xlocs);
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
  for(int i=0;i<d->size();i++){
  res*=exp(d->at(i).EvaluateLogPHpars(hpars_v[i]));}
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
  int nombre_steps_mcmc=1e5;
  int nombre_samples_collected=500;
  int nautocor=2000;

  int time_opt_opti=10; // 10 secondes par optimisation opti
  int time_opt_koh_loc=600; // 10 minutes par optimisation KOH locale
  int time_opt_koh_glo=7200; // 2h pour optimisation KOH globale
  double inputerr=0.5; //sigma=0.5K d'incertitude en input.
  // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(dim_theta); for(int i=0;i<dim_theta;i++){lb_t(i)=0;}
  VectorXd ub_t(dim_theta); for(int i=0;i<dim_theta;i++){ub_t(i)=0;}
  VectorXd lb_hpars(4); lb_hpars(0)=1e4;lb_hpars(1)=5;lb_hpars(2)=0.1,lb_hpars(3)=0.1;
  VectorXd ub_hpars(4); ub_hpars(0)=2e6;ub_hpars(1)=25;ub_hpars(2)=15,ub_hpars(3)=20; //bornes sur l'optimisation des hpars. edm exp lcor
  //VectorXd lb_hpars(2); lb_hpars(0)=1e4;lb_hpars(1)=0.1;
  //VectorXd ub_hpars(2); ub_hpars(0)=1e6;ub_hpars(1)=25; //bornes sur l'optimisation des hpars. edm exp lcor
 
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

  //Points de prédiction
  int samp_size=80; //80 avant
  vector<VectorXd> X_predictions(samp_size);
  VectorXd XPREDS(samp_size);
   for (int i=0;i<samp_size;i++){
    VectorXd x(1);
    x << 0.01+30*double(i)/double(samp_size);
    X_predictions[i]=x;
    XPREDS(i)=x(0);
    }


  //simplement définir une fonction qui trace un calcul. Et ensuite une boucle d'appel. A faire aujourd'hui obligatoirement. 

      auto lambda_model=[pFunc_init](VectorXd const &X, VectorXd const & theta, int i) {
    //renvoie toutes les prédictions du modèle aux points donnés par X.
      VectorXd pred(X.size());
      for(int j=0;j<X.size();j++){
        VectorXd x(1);
        x(0)=X(j);
        pred(j)=my_model(x,GPtoR(theta),i);
      }
      return pred;
    };

  auto plot_model=[lambda_model,XPREDS](VectorXd const & theta, int caset, string const & filename){

    VectorXd res=lambda_model(XPREDS,theta,caset);
    ofstream ofile(filename);
    for(int i=0;i<res.size();i++){
      ofile << XPREDS(i) << " " << res(i) << endl;
    }
    ofile.close();
  };

  int chosen_case=15;
  for(int i=0;i<dim_theta;i++){
    VectorXd X=0.5*VectorXd::Ones(dim_theta);
    string fnamep="results/c"+to_string(i)+"+"+to_string(chosen_case)+".gnu";
    string fnamem="results/c"+to_string(i)+"-"+to_string(chosen_case)+".gnu";
    X(i)=0;
    plot_model(X,chosen_case,fnamem);
    X(i)=1;
    plot_model(X,chosen_case,fnamep);
  }


  exit(0);




  VectorXd X_init_mcmc(3);
  X_init_mcmc << 0.3,0.15,0.8;
  MatrixXd COV_init=pow(0.1,2)*MatrixXd::Identity(3,3);
  COV_init/=9;
  cout << "COV_init : " << endl << COV_init << endl;


  auto lambda_priormean=[](VectorXd const &X, VectorXd const &hpars){
    VectorXd b=VectorXd::Zero(X.size());
    return b;
    double t=hpars(3);
    for(int i=0;i<X.size();i++){
      double x=X(i);
      if(x<t){
        b(i)=0;
      }
      else{
        b(i)=pow(x-t,hpars(2));
      }
    }  
    return b;
  };

  auto lambda_priormean2=[](VectorXd const &X, VectorXd const &hpars){
    VectorXd b=VectorXd::Zero(X.size());
    double t=hpars(3);
    for(int i=0;i<X.size();i++){
      double x=X(i);
      if(x<t){
        b(i)=0;
      }
      else{
        b(i)=hpars(4)*pow(x-t,hpars(2));
      }
    }  
    return b;
  };

  //cout << data_exp_20[0].Value() << endl;
  //cout << data_exp_23[0].Value() << endl;
  
  //juste un test. 500 évaluations de modèle ? 
  //construction du DoE initial en LHS (generator) ou Grid Sampling(sans generator)
  DoE doe_init(lb_t,ub_t,200,10);//doe halton. 500 points en dimension 3 (3 paramètres incertains).
  doe_init.WriteGrid("results/save/grid.gnu");

  //construction du vecteur de Density. On construira le vecteur de DensOpt par dessus.
  vector<int> cases={6,14,15,16,18,20};
  //vector<int> cases={3,4,5,6,14,15,16,20,21,22,23}; // le cas 18 est puni.
  //vector<int> cases={20};
  int vsize=cases.size();
  vector<Density> vDens;

  //il faut construire un vecteur des fonctions modèle, avec copie, sinon pas de possibilité de réutiliser.
  //vector<function<VectorXd(VectorXd const &, VectorXd const &)>> vModels;

  for(int i:cases){
    //attention parce qu'on créé l'objet qui va disparaitre ensuite. Est-ce que cette lambda est bien copiée ? Hope so . Il faudrait vérifier par acquis de conscience. par exemple il capture i mais qui n'existe plus après....
    auto lambda_model=[pFunc_init,i](VectorXd const &X, VectorXd const & theta) mutable{
    //renvoie toutes les prédictions du modèle aux points donnés par X.
      VectorXd pred(X.size());
      for(int j=0;j<X.size();j++){
        VectorXd x(1);
        x(0)=X(j);
        pred(j)=my_model(x,GPtoR(theta),i);
      }
      return pred;
    };
    auto data=GetObs(pFunc_exp,i);
    auto data_exp=Conversion(data);
    vector<VectorXd> XOBS(data_exp[0].GetX().size());
    for (int j=0;j<XOBS.size();j++){
      VectorXd x(1);
      x << data_exp[0].GetX()(j);
      XOBS[j]=x;
    }  
    string filename="results/obs"+to_string(i)+".gnu";
    string filename_fit="results/fit"+to_string(i)+".gnu";
    //WriteObs(filename,data);

    Density MainDensity(doe_init);
    MainDensity.SetModel(lambda_model);
    MainDensity.SetKernel(Kernel_Z_Matern52);
    MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetPriorMean(lambda_priormean);
    MainDensity.SetDataExp(data_exp);
    MainDensity.SetXprofile(data_exp[0].GetX());
    MainDensity.Set_Inputerr(inputerr);
    MainDensity.Compute_derivatives_f(XOBS,X_predictions,20,filename_fit);
    MainDensity.WriteObsWithUncertainty(filename);
    vDens.push_back(MainDensity);
  }

    //MCMC phase KOH pooled
      auto hparskoh_pooled=HparsKOH_pooled(vDens,hpars_z_guess,300);
  cout << "hparskoh pooled:" << hparskoh_pooled[0].transpose() << endl;
  //cout << hparskoh_pooled[1].transpose() << endl;

  {
    auto in_bounds=[vDens](VectorXd const & X){
        return vDens[0].in_bounds_pars(X);
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



  exit(0);


  //MCMC phase KOH separate
    auto hparskoh_separate=HparsKOH_separate(vDens,hpars_z_guess,300);
    cout << "hparskoh sep:" << hparskoh_separate[0].transpose() << endl;
  //cout << hparskoh_separate[1].transpose() << endl;


  {
    auto in_bounds=[vDens](VectorXd const & X){
        return vDens[0].in_bounds_pars(X);
    };
    auto get_hpars_kohs=[hparskoh_separate](VectorXd const & X){
      return hparskoh_separate;
    };
    auto compute_score_kohs=[vDens,vsize](vector<VectorXd> p, VectorXd const &X){
      double res=0;
      for(int i=0;i<vsize;i++){
        double d=vDens[i].loglikelihood_theta_incx(X,p[i]);
        res+=d;
        if (isnan(d)){ cerr << "nan ll. cas numéro " << i << endl;}
      }
      res+=logprior_pars(X);
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
      vDens[0].Autocor_diagnosis(nautocor,"results/diag/autocorkohs.gnu");
      //écriture des samples. Il faut une fonction dédiée. 
      for(int i=0;i<vsize;i++){
        string fname="results/save/sampkohs"+to_string(cases[i])+".gnu";
        string fnamepred="results/preds/predskohs"+to_string(cases[i])+".gnu";
        string fnamepredF="results/preds/predskohsF"+to_string(cases[i])+".gnu";
        vDens[i].WriteSamples(fname);
        vDens[i].WritePredictionsF(XPREDS,fnamepredF);
        vDens[i].WritePredictions(XPREDS,fnamepred);
      }
      //prédictions
  }


  exit(0);
  //MCMC phase Opti

  //test du temps nécessaire pour les optimisations.


  //construction du vecteur de densityopt.
  vector<DensityOpt> vDopt;
  for(int i=0;i<vsize;i++){
    DensityOpt Dopt(vDens[i]);
    Dopt.Compute_optimal_hpars(0.5);
    Dopt.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
    Dopt.opti_allgps(hpars_gp_guess);
    vDopt.push_back(Dopt);
  }


  {
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
    auto compute_score_opti=[vDopt,vsize](vector<VectorXd> p, VectorXd const &X){
      double res=0;
      //cout << "comp score : " << endl;
      for(int i=0;i<vsize;i++){
        double d=vDopt[i].loglikelihood_theta_incx(X,p[i]);
        if (isnan(-d)){ cerr << "nan ll. cas numéro " << i << endl;
          cerr << "theta: " << X.transpose() << endl;
          cerr << "hpars: " << p[i].transpose() << endl;
        }
        //cout << i << " : " << d << endl;
        res+=d;
      }
      //res+=logprior_pars(X);
      //cout << "res : " << res << endl;
      return res;
    };
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

  




  //MCMC phase Bayes. ça promet.


  {
    //initial MCMC. remplissage des états initiaux.
    int mcmc_fb_size=3+vsize*lb_hpars.size();
    
    VectorXd X_init_mcmc_fb(mcmc_fb_size);
    MatrixXd COV_init_fb=MatrixXd::Zero(mcmc_fb_size,mcmc_fb_size);
    X_init_mcmc_fb.head(3)=X_init_mcmc;
    COV_init_fb.topLeftCorner(3,3)=COV_init;
    int m=lb_hpars.size();
    for(int i=0;i<vsize;i++){
      for(int j=0;j<ub_hpars.size();j++){
        X_init_mcmc_fb(3+i*m+j)=0.5*(lb_hpars(j)+ub_hpars(j));
      }
      COV_init_fb(3+i*m,3+i*m)=pow(5e4,2);
      COV_init_fb(3+i*m+1,3+i*m+1)=pow(0.5,2);
      //COV_init_fb(3+i*m+2,3+i*m+2)=pow(0.1,2);
      //COV_init_fb(3+i*m+3,3+i*m+3)=pow(0.2,2);
    }

    auto in_bounds_fb=[vDens,vsize,lb_hpars](VectorXd const & X){
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
      in = in && vDens[0].in_bounds_pars(theta);
      for(int i=0;i<vsize;i++){
        in=in && vDens[0].in_bounds_hpars(hp[i]);
      }
      return in;
    };

    auto get_hpars_fb=[](VectorXd const & X){
      //renvoyer un vecteur sans intérêt.
      vector<VectorXd> v;
      return v;
    };
    auto compute_score_fb=[vsize,lb_hpars,vDens](vector<VectorXd> p, VectorXd const &X){
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
        if (isnan(d)){ cerr << "nan ll. cas numéro " << i << endl;}
        //cout << i << " : " << d << endl;
        res+=d;
        double e=logprior_hpars(hp[i]);
        if (isnan(e)){ cerr << "nan lp. cas numéro " << i << endl;}

        res+=e;
      }
      res+=logprior_pars(theta);
      return res;
    };
    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc_fb,COV_init_fb,compute_score_fb,get_hpars_fb,in_bounds_fb,generator);
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
      //diagnostic
      vDens[0].Autocor_diagnosis(nautocor,"results/diag/autocorfb.gnu");

      //Set des samples dans chaque densité + prédictions
      for(int i=0;i<vsize;i++){
        vDens[i].SetNewSamples(samples_fb_theta);
        vDens[i].SetNewHparsOfSamples(samples_fb_hpars[i]);
        string fname="results/save/sampfb"+to_string(cases[i])+".gnu";
        string fnamepred="results/preds/predsfb"+to_string(cases[i])+".gnu";
        string fnamepredF="results/preds/predsfbF"+to_string(cases[i])+".gnu";
        vDens[i].WriteSamples(fname);
        vDens[i].WritePredictionsF(XPREDS,fnamepredF);
        vDens[i].WritePredictions(XPREDS,fnamepred);
      }


  }


  exit(0);
 
}

