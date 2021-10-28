// on met au clair cette histoire de fixed domain asymptotics. Je tire une trajectoire d'un p-g et j'affiche la fonction de vraisemblance.


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

double Kernel_GP_Matern12(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double cor=pow(hpar(0),2);
  cor*=exp(-abs(x(0)-y(0))/hpar(1)); //phi
  return cor;
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

MatrixXd Gamma(vector<VectorXd> const & locs, Eigen::VectorXd const &hpar,function<double(VectorXd const &, VectorXd const &, VectorXd const &)> kernel) {
  // Renvoie la matrice de corrélation avec  bruit
  int nd=locs.size();
  double noise=0;
  Eigen::MatrixXd A(nd,nd);
  for(int i=0; i<nd; i++){
    for(int j=i; j<nd; j++){
      A(i,j) = kernel(locs[i],locs[j], hpar);
      if(i!=j){
	A(j,i) = A(i,j);
      }else{
	A(i,j) += pow(noise,2);		
      }
    }
  }
  return A;
}

double loglikelihood(VectorXd &yobs,VectorXd &hpars,vector<VectorXd> const & locs,function<double(VectorXd const &, VectorXd const &, VectorXd const &)> kernel){
  //calcul de logvraisemblance
  LDLT<MatrixXd> ldlt(Gamma(locs,hpars,kernel));
  VectorXd Alpha=ldlt.solve(yobs);
  int nd=yobs.size();
  return -0.5*yobs.dot(Alpha)-0.5*(ldlt.vectorD().array().log()).sum()-0.5*nd*log(2*3.1415);
}

//évaluation de la ll des observations : construction de gamma, inversion et calcul du fit.



const double Big = -1.e16;


int main(int argc, char **argv){
  generator.seed(123478);
  cout << "salut" << endl;

  vector<VectorXd> Target; //les observations
  int n=400;
  double xmax=5; //x de 0 à 5
  for(int i=0;i<n;i++){
    VectorXd x(1);
    x(0)=xmax*i/(1.0*(n-1));
    Target.push_back(x);
  }
  //on va mettre un kernel simple, sans erreur d'observation.
  //véritables hpars
  auto kernel=Kernel_GP_Matern12;
  VectorXd hpars_gp(2);
  hpars_gp << 2,4;
  GP gp(kernel);
  gp.SetGP(hpars_gp);
  MatrixXd M=gp.SampleGP(Target,1,generator); //size (targetsize, ns). donc (n,1) ici.
  
  VectorXd yobs=M.col(0);
  //écriture des observations
  string filename="results/observations.gnu";
  ofstream ofile(filename);
  for(int i=0;i<M.rows();i++){
    ofile << Target[i](0) << " " << yobs(i);
    ofile << endl;
  }
  ofile.close();

  //construction du grid pour la ll
  VectorXd lb_hpars(2);
  VectorXd ub_hpars(2);
  lb_hpars <<1,1;
  ub_hpars <<20,100;

  //évaluation de la ll en chaque point du grid et écriture.
  filename="results/ll.gnu";
  ofile.open(filename);
  int fgrid=200; //finesse du grid
  for(int i=0;i<fgrid;i++){
    for(int j=0;j<fgrid;j++){
      VectorXd hpars(2);
      double sigma= lb_hpars[0]*pow(ub_hpars[0]/lb_hpars[0],(1.0*i)/fgrid);
      double l= lb_hpars[1]*pow(ub_hpars[1]/lb_hpars[1],(1.0*j)/fgrid);
      hpars << sigma,l;
      double ll=loglikelihood(yobs,hpars,Target,kernel);
      ofile << sigma << " "<< l << " "<< ll << endl;
    }
  }
  ofile.close();

  //calcul de la ll le long de la ligne sigma2/l = cst.
  filename="results/ll_linear.gnu";
  ofile.open(filename);
  fgrid=200; //raffinement
  for(int i=0;i<fgrid;i++){
      VectorXd hpars(2);
      double sigma= lb_hpars[0]*pow(ub_hpars[0]/lb_hpars[0],(1.0*i)/fgrid);
      double l= pow(sigma,2);
      hpars << sigma,l;
      double ll=loglikelihood(yobs,hpars,Target,kernel);
      ofile << sigma << " "<< l << " "<< ll << endl;
  }
  ofile.close();

  exit(0);



};