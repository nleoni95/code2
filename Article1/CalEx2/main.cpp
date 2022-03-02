// Dans ce fichier : on reproduit l'exemple 1 de l'article. But : montrer la normalité bimodale de la postérieure.



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

int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
int dim_theta=1;

//soft clustering en tant que gaussian mixture, pour deux clusters
//on met la moyenne du second cluster à 1.13 pour l'aider
vector<VectorXd> GMM(vector<VectorXd> const & x,default_random_engine & generator){
  int n=x.size();
  int s=x[0].size();
  int nclusters=2;
  vector<VectorXd> p(n); for(int i=0;i<n;i++){p[i]=VectorXd::Zero(nclusters);}
  //on renvoie la probabilité d'appartenir au cluster 0.
  vector<VectorXd> mu(nclusters); for(int j=0;j<nclusters;j++){ mu[j]=x[n*distU(generator)];}; 
  VectorXd xbar=VectorXd::Zero(s);
  for(int i=0;i<n;i++){
    xbar+=x[i];
  }
  xbar/=n;
  double s2=0;
  for(int i=0;i<n;i++){
    s2+=(xbar-x[i]).squaredNorm();
  }
  s2/=n;
  vector<double> pi(nclusters); pi[0]=1;pi[1]=1;
  vector<MatrixXd> S(nclusters); S[0]=s2/nclusters*MatrixXd::Identity(s,s); S[1]=s2/nclusters*MatrixXd::Identity(s,s);
  int c=0;
  while(c<25){
    mu[0](0)=-0.025;
    mu[1](0)=1.02;
    cout << "iteration " << c << endl;
    cout << "cluster 1 : poids " << endl;
    cout << pi[0] << endl;
    cout << "cluster 1 : mean " << endl;
    cout << mu[0].transpose() << endl;
    cout << "cluster 1 : cov " << endl;
    cout << S[0] << endl;

    cout << "cluster 2 : poids " << endl;
    cout << pi[1] << endl;
    cout << "cluster 2 : mean " << endl;
    cout << mu[1].transpose() << endl;
    cout << "cluster 2 : cov " << endl;
    cout << S[1] << endl << endl;
    //calcul des décomp LDLT des matrices de covariance des clusters
    vector<LDLT<MatrixXd>> ldlt(nclusters);
    for(int j=0;j<nclusters;j++){
      LDLT<MatrixXd> l(S[j]);
      ldlt[j]=l;
    }
    //expectation
    for(int i=0;i<n;i++){
      double s=0;
      for(int j=0;j<nclusters;j++){
        VectorXd Xc=x[i]-mu[j];
        VectorXd Alpha=ldlt[j].solve(Xc);
        p[i](j)=pi[j]*exp(-0.5*Xc.dot(Alpha)-0.5*(ldlt[j].vectorD().array().log()).sum());
        s+=p[i](j);
      }
      for(int j=0;j<nclusters;j++){
        p[i](j)/=s;
      }
    }
    //maximisation
    
    for(int j=0;j<nclusters;j++){
      pi[j]=0; mu[j]=VectorXd::Zero(s);
      for(int i=0;i<n;i++){
        pi[j]+=p[i](j);
        mu[j]+=p[i](j)*x[i];
      }
      mu[j]/=pi[j];
      S[j]=MatrixXd::Zero(s,s);
      for(int i=0;i<n;i++){
        S[j]+=p[i](j)*(x[i]-mu[j])*(x[i]-mu[j]).transpose();
      }
      S[j]/=pi[j];
    }
    
    c++;
  }
  return p;
}


// MODELE RAVIK.

double computer_model(const double &x, const VectorXd &t){
  return x*sin(2*t(0)*x)+(x+0.15)*(1-t(0));
}

double gaussprob(double x,double mu, double sigma){
  //renvoie la probabilité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}

double Kernel_Z_lin(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //linear
  return pow(hpar(2),2)+pow(hpar(1),2)*x(0)*y(0); //3/2
}

double Kernel_Z_quad(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //linear
  return pow(hpar(1),2)*x(0)*y(0)+pow(hpar(2),2)*pow(x(0)*y(0),2); //3/2
}

double Kernel_Z_SE_lin(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential * linear
  double d=abs(x(0)-y(0));
  return pow(hpar(1),2)*exp(-0.5*pow(d/hpar(2),2))*x(0)*y(0); //3/2
}

double Kernel_Z_SE_pluslin(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential + linear
  double d=abs(x(0)-y(0));
  return pow(hpar(1),2)*exp(-0.5*pow(d/hpar(2),2))+pow(hpar(3),2)*x(0)*y(0); //3/2
}

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  return pow(hpar(1),2)*exp(-0.5*pow(d/hpar(2),2)); //3/2
}

double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2)))*exp(-d/hpar(1)); //3/2
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


/* natural log of the gamma function
   gammln as implemented in the
 * first edition of Numerical Recipes in C */
double gammln(double xx)
{
    double x, tmp, ser;
    const static double cof[6]={76.18009172947146,    -86.50532032941677,
                                24.01409824083091,    -1.231739572450155,
                                0.1208650973866179e-2,-0.5395239384953e-5};
    int j;

    x=xx-1.0;
    tmp=x+5.5;
    tmp -= (x+0.5)*log(tmp);
    ser=1.000000000190015;
    for (j=0;j<=5;j++) {
        x += 1.0;
        ser += cof[j]/x;
    }
    return -tmp+log(2.5066282746310005*ser);
}
double loginvgammaPdf(double x, double a, double b){                                                           
    return a*log(b) - gammln(a) + (a+1)*log(1/x) - b/x;                                       
}    

double lognorm(double x, double mean, double std){
  return -0.5*pow((x-mean)/std,2);
}

double logprior_hpars(VectorXd const &hpars){
  //seulement sur la longueur de corrélation pour que ça soit plus normal.
  return 0;
  //return lognorm(hpars(2),3,1);
  //+lognorm(hpars(1),0.25,0.1)+lognorm(hpars(0),0.1,0.05);
}

double logprior_hpars_invgamma(VectorXd const &hpars){
  double alpha=50;
  double beta=500;
  return loginvgammaPdf(hpars(1),alpha,beta);
}





double logprior_pars(VectorXd const &pars){
  return 0; 
}

double logprior_parsnormgrand(VectorXd const &pars){
  //prior gaussien large sur les paramètres. ils seront dans l'espace (0,1.)
  double d=0;
  for (int i=0;i<3;i++){
    d+=lognorm(pars(i),0.5,0.5);
  }
  return d;  
}

double logprior_parsunif(VectorXd const &pars){
  return 0;
}

void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
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

void Run_Burn_Phase_MCMC(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento,function<double(vector<VectorXd>, VectorXd const &)> const & compute_score, function<vector<VectorXd>(VectorXd const &)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //phase de burn.
  int dim_mcmc=COV_init.cols();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  VectorXd Xinit=Xcurrento;
  MatrixXd COV=COV_init;
  MatrixXd sqrtCOV=COV.llt().matrixL();
  vector<VectorXd> hparscurrent=get_hpars(Xinit);
  double finit=compute_score(hparscurrent,Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;  
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

tuple<vector<VectorXd>,vector<vector<VectorXd>>,vector<VectorXd>,vector<double>,vector<vector<VectorXd>>> Run_MCMC(int nsteps,int nsamples, VectorXd const & Xinit,MatrixXd const & COV_init,function<double(vector<VectorXd>, VectorXd const &)> const & compute_score, function<vector<VectorXd>(VectorXd const &)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
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
  vector<VectorXd> hparsstart=get_hpars(Xinit0);
  double finit=compute_score(hparsstart,Xinit0);
  VectorXd Xcurrent=Xinit0;
  double fcurrent=finit;
  vector<VectorXd> hparscurrent=hparsstart;
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


double optfunc_opti(const std::vector<double> &x, std::vector<double> &grad, void *data){
  //pour chercher le maximum d'une densité liée à 1 configuration expérimentale.
  auto d=(DensityOpt*) data; //cast
  VectorXd X(x.size()); for(int j=0;j<x.size();j++){X(j)=x[j];}
  VectorXd p=d->EvaluateHparOpt(X);
  double l=d->loglikelihood_theta(X,p);
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
  int nombre_samples_collected=5000; //1 sample tous les 100. 
  int nautocor=2000;

  int time_opt_opti=10; // 10 secondes par optimisation opti
  int time_opt_koh_loc=600; // 10 minutes par optimisation KOH locale
  int time_opt_koh_glo=7200; // 2h pour optimisation KOH globale
  double inputerr=0.6; //sigma=0.1K d'incertitude en input.
  // Bornesup des paramètres et hyperparamètres
  // lower bound : theta vaut zero, chacun des paramètres vaut la moitié de son coefficient.
  //upper bound : theta vaut 90, chacun des paramètres vaut 2* sa valeur

  VectorXd lb_t=-0.5*VectorXd::Ones(1);
  VectorXd ub_t=1.5*VectorXd::Ones(1);

  cout << "lb_t : " << lb_t.transpose() << endl;
  cout << "ub_t : " << ub_t.transpose() << endl;

  VectorXd lb_hpars(3); lb_hpars(0)=1e-5;lb_hpars(1)=1e-5;lb_hpars(2)=0.001;//lb_hpars(2)=0;
  VectorXd ub_hpars(3); ub_hpars(0)=1;ub_hpars(1)=1;ub_hpars(2)=5;//ub_hpars(2)=1e6;
 
  VectorXd hpars_z_guess=0.5*(lb_hpars+ub_hpars);
  //Points de prédiction.
  int samp_size=80; //80 avant
  vector<VectorXd> X_predictions(samp_size);
  VectorXd XPREDS(samp_size);
   for (int i=0;i<samp_size;i++){
    VectorXd x(1);
    x << 0+1*double(i)/double(samp_size);
    X_predictions[i]=x;
    XPREDS(i)=x(0);
  }  

  //création des observations sous la forme d'un vector<DATA>.
  double noise=0.1;
  int ndata=8;
  vector<DATA> data;
  for(int i=0;i<ndata;i++){
    DATA d;
    VectorXd x(1); x << 0.0625+double(i)/double(ndata);
    double y=x(0)+noise*distN(generator);
    d.SetX(x); d.SetValue(y);
    data.push_back(d);
  }
  string expname="results/obs.gnu";
  WriteObs(expname,data);

  auto lambda_priormean=[](VectorXd const &X, VectorXd const &hpars){
    VectorXd b=VectorXd::Zero(X.size());
    return b;
  };
  
  //construction du DoE initial en LHS (generator) ou Grid Sampling(sans generator)
  DoE doe_init(lb_t,ub_t,100,10);//doe halton. 300 points en dimension 3 (3 paramètres incertains).
  doe_init.WriteGrid("results/save/grid.gnu");



  auto lambda_model=[](VectorXd const &X, VectorXd const & theta) mutable{
    VectorXd pred(X.size());
    for(int j=0;j<X.size();j++){
      pred(j)=computer_model(X(j),theta);
    }
    return pred;
  };

  auto data_exp=Conversion(data);
  vector<VectorXd> XOBS(data_exp[0].GetX().size());
  for (int j=0;j<XOBS.size();j++){
    VectorXd x(1);
    x << data_exp[0].GetX()(j);
    XOBS[j]=x;
  }  
  //test: affichage des observations et des prédictions nominales du modèle.
  VectorXd Xinit=0.5*VectorXd::Ones(3);
  cout << lambda_model(data_exp[0].GetX(),Xinit).transpose() << endl;
  cout << endl;

  
  string filename_fit="results/fit.gnu";
  Density MainDensity(doe_init);
  MainDensity.SetModel(lambda_model);
  MainDensity.SetKernel(Kernel_Z_SE);
  //MainDensity.SetKernelDerivatives(D1Kernel_Z_Matern52,D2Kernel_Z_Matern52,D1Kernel_Z_Matern52);
  MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(lambda_priormean);
  MainDensity.SetDataExp(data_exp);
  MainDensity.SetXprofile(data_exp[0].GetX());


  VectorXd X_init_mcmc=0.5*VectorXd::Ones(dim_theta);
  MatrixXd COV_init=MatrixXd::Identity(dim_theta,dim_theta);
  COV_init(0,0)=pow(0.04,2);
  cout << "COV_init : " << endl << COV_init << endl;

//début des calibrations.



 //MCMC koh separate sur les densités séparées

  // calcul hpars ko separate
  vector<VectorXd> hparskoh_separate(1);
  hparskoh_separate[0]=MainDensity.HparsKOH(hpars_z_guess,10);


  cout << "hparskoh sep:" << hparskoh_separate[0].transpose() << endl;

//calibration koh separate
  {
    auto in_bounds=[&MainDensity](VectorXd const & X){
      return MainDensity.in_bounds_pars(X);
    };

    auto get_hpars_kohs=[&hparskoh_separate](VectorXd const & X){
      vector<VectorXd> h(1);
      h[0]=hparskoh_separate[0];
      return h;
    };
    auto compute_score_kohs=[&MainDensity](vector<VectorXd> p, VectorXd const &X){
      double res=MainDensity.loglikelihood_theta(X,p[0])+MainDensity.EvaluateLogPPars(X);
      return res;
    };

    cout << "début mcmc densité koh separate expérience" <<  endl;
    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_kohs,get_hpars_kohs,in_bounds,generator);
      auto samples_opti=get<0>(res);
      vector<VectorXd> allsamples_opti=get<2>(res);
      vector<vector<VectorXd>> vhpars_opti_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
      auto vhpars_opti=revert_vector(vhpars_opti_noninverted);

      //Set des samples dans chaque densité
      
      MainDensity.SetNewSamples(samples_opti);
      MainDensity.SetNewHparsOfSamples(vhpars_opti[0]);
      
      MainDensity.SetNewAllSamples(allsamples_opti);
      //diagnostic
      MainDensity.Autocor_diagnosis(nautocor,"results/separatedensities/kohs/autocor.gnu");
      //écriture des samples. Il faut une fonction dédiée.
      string fname="results/separatedensities/kohs/samp.gnu";
      string fnameall="results/separatedensities/kohs/allsamp.gnu";
      string fnamepred="results/separatedensities/kohs/preds.gnu";
      string fnamepredF="results/separatedensities/kohs/predsF.gnu";
      string fnamesF="results/separatedensities/kohs/sampsF.gnu";
      string fnamesZ="results/separatedensities/kohs/sampsZ.gnu";
      MainDensity.WriteSamples(fname);
      MainDensity.WriteMCMCSamples(fnameall);

                  //prédictions

      MainDensity.WritePredictionsF(XPREDS,fnamepredF);
      MainDensity.WritePredictions(XPREDS,fnamepred);
      MainDensity.WriteSamplesFandZ(XPREDS,fnamesF,fnamesZ);
      //prédictions
      //écriture des observations avec erreur expérimentale KOH
      string filename="results/obs.gnu";
  
  }


//Phase Opti
  //surrogates sur 200 points et assez rapides.
  DensityOpt Dopt(MainDensity);


   //MCMC opt sur les densités séparées.calibration opti densités séparées
  {
    auto in_bounds=[&Dopt](VectorXd const & X){
    return Dopt.in_bounds_pars(X);
    };
    auto get_hpars_opti=[Dopt,&hpars_z_guess](VectorXd const & X){
      vector<VectorXd> p(1);
      p[0]=Dopt.HparsOpt(X,hpars_z_guess,1e-4);
      return p;
    };
    auto compute_score_opti=[&Dopt](vector<VectorXd> const &p, VectorXd const &X){
      double d=Dopt.loglikelihood_theta(X,p[0])+Dopt.EvaluateLogPPars(X);
      return d;
    };


    //je fais 10000 optimisations. Juste pour voir.
    //tirer un grid de bi

    /*
    int bi=1e4;
    DoE doe_test(lb_t,ub_t,bi,1);
    vector<VectorXd> ttest=doe_test.GetGrid();
    auto begin=chrono::steady_clock::now();
    double rez=0;
    for(int u=0;u<bi;u++){
      auto V=get_hpars_opti(ttest[u]);
      rez+=compute_score_opti(V,ttest[u]);
    }
    auto end=chrono::steady_clock::now();
    cout << "temps pour 10000 itérations " << " time : " << chrono::duration_cast<chrono::milliseconds>(end-begin).count() << " ms" << "score :" << rez/bi <<endl;
    exit(0);
    */
    cout << "début mcmc densité fmp expérience" << endl;
    auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc,COV_init,compute_score_opti,get_hpars_opti,in_bounds,generator);
    auto samples_opti=get<0>(res);
    vector<VectorXd> allsamples_opti=get<2>(res);
    vector<vector<VectorXd>> allhpars_opti_inverted=get<4>(res);// dans le cas 1D ici, le outer vector est de taille nsteps, et le inner vector de taille 1 (voir fct get_hpars) 
    vector<vector<VectorXd>> vhpars_opti_noninverted=get<1>(res); //là le premier vecteur c'est les steps, et le second vecteur les densités. Il faut inverser ça...
    auto vhpars_opti=revert_vector(vhpars_opti_noninverted);
    vector<VectorXd> allhpars_opti=revert_vector(allhpars_opti_inverted)[0];

    //Set des samples dans chaque densité  
    Dopt.SetNewSamples(samples_opti);
    Dopt.SetNewHparsOfSamples(vhpars_opti[0]);
    Dopt.SetNewAllSamples(allsamples_opti);
    //diagnostic
    Dopt.Autocor_diagnosis(nautocor,"results/separatedensities/opt/autocor.gnu");
    //écriture des samples. Il faut une fonction dédiée.
    string fname="results/separatedensities/opt/samp.gnu";
    string fnameall="results/separatedensities/opt/allsamp.gnu";
    string fnameallh="results/separatedensities/opt/allhpars.gnu";
    string fnamepred="results/separatedensities/opt/preds.gnu";
    string fnamepredF="results/separatedensities/opt/predsF.gnu";
    string fnamesF="results/separatedensities/opt/sampsF.gnu";
    string fnamesZ="results/separatedensities/opt/sampsZ.gnu";
    Dopt.WriteSamples(fname);
    Dopt.WriteMCMCSamples(fnameall);
    writeVector(fnameallh,allhpars_opti);

    //prédictions

    Dopt.WritePredictionsF(XPREDS,fnamepredF);
    Dopt.WritePredictions(XPREDS,fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS,fnamesF,fnamesZ);
    
  }






  //MCMC phase Bayes sur les densités séparées. Pour chaque densités : que 2 hpars. La variance et la correlation length.
  //Bayes : il faut faire l'exporation sur log sigma. sinon trop dur à explorer.
  {
    //initial MCMC. remplissage des états initiaux.
    int dim_hpars=3;
    int mcmc_fb_size=1+dim_hpars;

    
    VectorXd X_init_mcmc_fb(mcmc_fb_size);
    MatrixXd COV_init_fb=MatrixXd::Zero(mcmc_fb_size,mcmc_fb_size);
    X_init_mcmc_fb.head(1)=X_init_mcmc;
    X_init_mcmc_fb(1)=0.5; X_init_mcmc_fb(2)=0.5;X_init_mcmc_fb(3)=0.5;//X_init_mcmc_fb(5)=1e4; //échelle log pour sigma
    COV_init_fb.topLeftCorner(1,1)=COV_init;
    int m=lb_hpars.size();
    COV_init_fb(1,1)=pow(0.01,2); 
    COV_init_fb(2,2)=pow(0.01,2);
    COV_init_fb(3,3)=pow(0.1,2);
    //COV_init_fb(5,5)=pow(1e3,2);
    cout << "covinitfb" << endl << COV_init_fb << endl;
    cout << "Xinitfb" << endl << X_init_mcmc_fb.transpose() << endl;


    auto in_bounds_fb=[&MainDensity,dim_hpars](VectorXd const & X){
      VectorXd hp=X.tail(dim_hpars);;
      VectorXd theta=X.head(1);
      return MainDensity.in_bounds_pars(X) && MainDensity.in_bounds_hpars(hp) ;
    };

    auto get_hpars_fb=[](VectorXd const & X){
      //renvoyer un vecteur sans intérêt.
      vector<VectorXd> v(1);
      return v;
    };
    auto compute_score_fb=[&lb_hpars,&MainDensity,dim_hpars](vector<VectorXd> const & p, VectorXd const &X){
      //il faut décomposer X en thetas/ hpars. 3 est la dimension des paramètres
      //logpriorhpars=0 I think
      VectorXd theta=X.head(1);
      VectorXd hp=X.tail(dim_hpars); 
      double d=MainDensity.loglikelihood_theta(theta,hp);
      double lp=MainDensity.EvaluateLogPHpars(hp)+MainDensity.EvaluateLogPPars(theta);
      return d+lp;
    }; 
    cout << "début mcmc densité bayes expérience" << endl;
      
    auto res=Run_MCMC(5*nombre_steps_mcmc,nombre_samples_collected,X_init_mcmc_fb,COV_init_fb,compute_score_fb,get_hpars_fb,in_bounds_fb,generator);

    auto samples_fb_gros=get<0>(res);
    vector<VectorXd> allsamples_fb_gros=get<2>(res);
    vector<VectorXd> samples_fb_theta;
    vector<VectorXd> samples_fb_hpars;
    //décomposer le tout en paramètres et hpars.
    for(int j=0;j<samples_fb_gros.size();j++){
      VectorXd X=samples_fb_gros[j];
      samples_fb_theta.push_back(X.head(1));
      VectorXd hp=X.tail(dim_hpars); 
      samples_fb_hpars.push_back(hp);
    }
    //set des samples dans la première densité pour calculer la corrélation.
    MainDensity.SetNewAllSamples(allsamples_fb_gros);
    //diagnostic
    MainDensity.Autocor_diagnosis(nautocor,"results/separatedensities/fb/autocor.gnu");

    MainDensity.SetNewSamples(samples_fb_theta);
    MainDensity.SetNewHparsOfSamples(samples_fb_hpars);

    //écriture des samples. Il faut une fonction dédiée.
    string fname="results/separatedensities/fb/samp.gnu";
    string fnameall="results/separatedensities/fb/allsamp.gnu";
      
    string fnamepred="results/separatedensities/fb/preds.gnu";
    string fnamepredF="results/separatedensities/fb/predsF.gnu";
    string fnamesF="results/separatedensities/fb/sampsF.gnu";
    string fnamesZ="results/separatedensities/fb/sampsZ.gnu";
    MainDensity.WriteSamples(fname);
    MainDensity.WriteMCMCSamples(fnameall); 

          //prédictions
    MainDensity.WritePredictionsF(XPREDS,fnamepredF);
    MainDensity.WritePredictions(XPREDS,fnamepred);
    MainDensity.WriteSamplesFandZ(XPREDS,fnamesF,fnamesZ);

    //calcul des clusters.
    {
    vector<VectorXd> weights=GMM(samples_fb_gros,generator);
    //on cherche combien de points sont dans chaque cluster
    int c1=0; int c2=0;
    for(int i=0;i<weights.size();i++){
      if(weights[i](0)>weights[i](1)){c1++;}
      else{c2++;}
    }
    cout << "nombre de points theta+hpars dans cluster 1 : " << c1 << " et 2 : " << c2 << endl;
    }
    /*
    {
    //calcul des clusters sur theta uniquement.
    vector<VectorXd> weights=GMM(samples_fb_theta,generator);
    //on cherche combien de points sont dans chaque cluster
    int c1=0; int c2=0;
    for(int i=0;i<weights.size();i++){
      if(weights[i](0)>weights[i](1)){c1++;}
      else{c2++;}
    }
    cout << "nombre de points theta dans cluster 1 : " << c1 << " et 2 : " << c2 << endl;
    }
    */
}

  exit(0);
 
}

