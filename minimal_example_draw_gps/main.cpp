// Tirage de trajectoires de processus gaussiens.


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


int neval=1;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);








double Kernel_GP_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le terme correctif Z
  //squared exponential
  double d=abs(x(0)-y(0));
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(1),2));
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double cor=pow(hpar(0),2);
  cor*=(1+abs(x(0)-y(0))/hpar(1))*exp(-abs(x(0)-y(0))/hpar(1)); //phi
  return cor;
}

double Kernel_GP_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(1))+0.33*pow(d/hpar(1),2))*exp(-d/hpar(1)); //5/2
}

const double Big = -1.e16;


int main(int argc, char **argv){
  generator.seed(123478);

  vector<VectorXd> Target;
  int n=200;
  double xmax=4;
  for(int i=0;i<n;i++){
    VectorXd x(1);
    x(0)=xmax*i/(1.0*(n-1));
    Target.push_back(x);
  }

  VectorXd hpars_gp(2);
  hpars_gp << 1,0.5;
  GP gp(Kernel_GP_Matern52);
  gp.SetGP(hpars_gp);
  MatrixXd M=gp.SampleGP(Target,10,generator); //size (targetsize, ns).
  string filename="results/samplesMatern52.gnu";
  ofstream ofile(filename);
  for(int i=0;i<M.rows();i++){
    ofile << Target[i](0) << " ";
    for(int j=0;j<M.cols();j++){
      ofile << M(i,j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
  exit(0);



};