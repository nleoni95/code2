// Dans ce fichier : on lit les données du calcul et on les post-traite à notre souhait.
// On lit également les données expérimentales.
// On peut faire quelques plots des données de calcul
// On créée ensuite un GP qu'on sauvegarde dans un fichier tiers qui sera lu plus tard.
// Il faut également récupérer le point du DoE correspondant.


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


double MatrixevalLLT(MatrixXd const &M,default_random_engine & generator){
  //évaluation d'une méthode de sampling
  auto ps=[](MatrixXd const &A, MatrixXd const &B)->double{
    return (A.transpose()*B).trace();
  };
  int dim=M.cols();
  normal_distribution<double> distN(0,1);
  MatrixXd sqrtCOV=M.llt().matrixL();
  int nsamples=10000;
  vector<VectorXd> samples(nsamples);
  VectorXd acc_means=VectorXd::Zero(dim);
  MatrixXd acc_var=MatrixXd::Zero(dim,dim);
  for(int i=0;i<nsamples;i++){
    VectorXd Seed(dim);
    for(int j=0;j<dim;j++){Seed(j)=distN(generator);}
    VectorXd s=sqrtCOV*Seed;
    samples[i]=s;
    acc_means+=s;
    acc_var+=s*s.transpose();
  }
  acc_means/=nsamples;
  acc_var/=nsamples;
  MatrixXd cov_empirique=acc_var-acc_means*acc_means.transpose();
  double norme=sqrt(ps(M-cov_empirique,M-cov_empirique));
  double r=sqrt(ps(M,M));
  return norme/r;
}

double MatrixevalLDLT(MatrixXd const &M,default_random_engine & generator){
  //évaluation d'une méthode de sampling
  auto ps=[](MatrixXd const &A, MatrixXd const &B)->double{
    return (A.transpose()*B).trace();
  };
  int dim=M.cols();
  normal_distribution<double> distN(0,1);
  Eigen::LDLT<Eigen::MatrixXd> eig(M);
  MatrixXd sqrtD=eig.vectorD().cwiseSqrt().asDiagonal();
  auto tra=eig.transpositionsP().transpose();
  MatrixXd L=eig.matrixL();
  L=tra*L;
  MatrixXd sqrtCOV=L*sqrtD;
  //MatrixXd sqrtCOV(dim,dim);
  int nsamples=10000;
  vector<VectorXd> samples(nsamples);
  VectorXd acc_means=VectorXd::Zero(dim);
  MatrixXd acc_var=MatrixXd::Zero(dim,dim);
  for(int i=0;i<nsamples;i++){
    VectorXd Seed(dim);
    for(int j=0;j<dim;j++){Seed(j)=distN(generator);}
    VectorXd s=sqrtCOV*Seed;
    samples[i]=s;
    acc_means+=s;
    acc_var+=s*s.transpose();
  }
  acc_means/=nsamples;
  acc_var/=nsamples;
  MatrixXd cov_empirique=acc_var-acc_means*acc_means.transpose();
  double norme=sqrt(ps(M-cov_empirique,M-cov_empirique));
  double r=sqrt(ps(M,M));
  return norme/r;
}


double MatrixevalSVD(MatrixXd const &M,default_random_engine & generator){
  //évaluation d'une méthode de sampling
  auto ps=[](MatrixXd const &A, MatrixXd const &B)->double{
    return (A.transpose()*B).trace();
  };
  int dim=M.cols();
  normal_distribution<double> distN(0,1);
  JacobiSVD<MatrixXd> Jac(M,ComputeFullU);
  MatrixXd sqrtD=Jac.singularValues().cwiseSqrt().asDiagonal();
  MatrixXd sqrtCOV=Jac.matrixU()*sqrtD;
  int nsamples=10000;
  vector<VectorXd> samples(nsamples);
  VectorXd acc_means=VectorXd::Zero(dim);
  MatrixXd acc_var=MatrixXd::Zero(dim,dim);
  for(int i=0;i<nsamples;i++){
    VectorXd Seed(dim);
    for(int j=0;j<dim;j++){Seed(j)=distN(generator);}
    VectorXd s=sqrtCOV*Seed;
    samples[i]=s;
    acc_means+=s;
    acc_var+=s*s.transpose();
  }
  acc_means/=nsamples;
  acc_var/=nsamples;
  MatrixXd cov_empirique=acc_var-acc_means*acc_means.transpose();
  double norme=sqrt(ps(M-cov_empirique,M-cov_empirique));
  double r=sqrt(ps(M,M));
  return norme/r;
}


const double Big = -1.e16;


int main(int argc, char **argv){

  MatrixXd M(8,8);
  M.row(0).transpose() <<0.0510023  ,-0.0150041  , 0.0056043 , 0.00181503 ,-0.00496842,-4.78516e-06 ,3.66199e-08 ,0.000231717;
  M.row(1).transpose() <<-0.0150041 ,  0.0582006 ,  0.0310074 ,-0.00315353 , 0.00399994 ,3.16365e-07, 4.52593e-08 ,0.000226092;
  M.row(2).transpose() << 0.0056043 ,  0.0310074  , 0.0521516, -0.00237247, 0.000100514,-3.55867e-06 ,3.41654e-08, 0.000111734;
  M.row(3).transpose() <<0.00181503, -0.00315353 ,-0.00237247 ,  0.0545894 , 0.00020513, 1.41502e-06,-1.60478e-08 ,9.90913e-05;
  M.row(4).transpose() <<-0.00496842  ,0.00399994, 0.000100514,  0.00020513 ,   0.054611 ,2.55054e-06, -4.8382e-09, 4.02982e-05;
  M.row(5).transpose() <<-4.78516e-06 ,3.16365e-07,-3.55867e-06 ,1.41502e-06 ,2.55054e-06 ,8.34605e-09 ,2.02118e-11 ,2.60287e-07;
  M.row(6).transpose() <<3.66199e-08 ,4.52593e-08 ,3.41654e-08 ,-1.60478e-08, -4.8382e-09, 2.02118e-11, 7.33497e-11 ,7.05013e-09;
  M.row(7).transpose() <<0.000231717 ,0.000226092, 0.000111734, 9.90913e-05 ,4.02982e-05, 2.60287e-07, 7.05013e-09 ,6.07592e-05;
  cout << M << endl;


  MatrixXd M2(8,8);
  M2.row(0).transpose() <<0.000863777,-0.000568582 ,-0.000331836 , 0.000143077, -0.000368382 , -6.5137e-08 , 3.17902e-10  ,2.83376e-06;
  M2.row(1).transpose() <<-0.000568582  , 0.00237477 ,  0.00152079 ,-0.000297683,  0.000326012 ,-7.25043e-09 , 1.89461e-09,  8.64014e-06;
  M2.row(2).transpose() <<-0.000331836  , 0.00152079  , 0.00180511, -0.000211058,  9.31631e-05 ,-7.26675e-08 , 1.14965e-09 , 3.25652e-06;
  M2.row(3).transpose() <<0.000143077, -0.000297683, -0.000211058 , 0.000736823 , 7.07329e-06 , 1.73028e-08 ,-3.50882e-10 , 8.31735e-07;
  M2.row(4).transpose() <<-0.000368382 , 0.000326012 , 9.31631e-05 , 7.07329e-06,  0.000713031 , 5.62551e-08 ,-9.44434e-11  ,1.32144e-07;
  M2.row(5).transpose() <<-6.5137e-08 ,-7.25043e-09 ,-7.26675e-08 , 1.73028e-08,  5.62551e-08 , 2.21236e-10 ,-2.62361e-13  ,9.64332e-11;
  M2.row(6).transpose() <<3.17902e-10 , 1.89461e-09  ,1.14965e-09 ,-3.50882e-10, -9.44434e-11 ,-2.62361e-13,  2.28786e-14  ,1.96107e-11;
  M2.row(7).transpose() << 2.83376e-06 , 8.64014e-06 , 3.25652e-06 , 8.31735e-07,  1.32144e-07 , 9.64332e-11  ,1.96107e-11   ,1.4167e-07;

  cout << M2 << endl;

  default_random_engine generator(16);
  int nrepet=100;
  auto compute_err=[&generator,nrepet](MatrixXd const &M){
    double ellt=0;
    double eldlt=0;
    double esvd=0;
    for(int i=0;i<nrepet;i++){
      ellt+=MatrixevalLLT(M,generator);
      eldlt+=MatrixevalLDLT(M,generator);
      esvd+=MatrixevalSVD(M,generator);
    }
    ellt/=nrepet;eldlt/=nrepet;esvd/=nrepet;
    cout << "erreur llt : " << ellt << ", ldlt : " << eldlt << ", svd : " << esvd << endl;
  };

  compute_err(M);
  compute_err(M2);


  MatrixXd nugget=1e-6*MatrixXd::Identity(8,8);
  M+=nugget;
  M2+=nugget;

  compute_err(M);
  compute_err(M2);

  exit(0);
};