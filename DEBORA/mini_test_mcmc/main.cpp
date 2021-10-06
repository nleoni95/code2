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


double gaussprob(double x,double mu, double sigma){
  //densité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}

double gaussprob(double x,double sigma){
  //densité gaussienne centrée
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x)/sigma,2));
}

double Run_MCMC(int nsteps, double sproposal, double strue, default_random_engine & generator){
  //mcmc avec des gaussiennes 1D.
  normal_distribution<double> distN(0,1);
  auto compute_score=[strue](double x) -> double{
    return log(gaussprob(x,strue));
  };
  double xinit=strue;
  double finit=compute_score(xinit);
  int naccept=0;
  double xcurrent=xinit;
  double fcurrent=finit;
  for(int i=0;i<nsteps;i++){
    double xcandidate=xcurrent+sproposal*distN(generator);
    double fcandidate=compute_score(xcandidate);
    if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
      naccept++;
      xcurrent=xcandidate;
      fcurrent=fcandidate;
    }
  }
  double acc_rate=(double)(naccept)/(double)(nsteps);
  return acc_rate;
}



const double Big = -1.e16;

int main(int argc, char **argv){
  generator.seed(16);
  double strue=1;
  int nsteps=1e5;
  vector<double> rapports;
  vector<double> acc_rates;

  //plusieurs MCMCs en ayant un ratio sprop/strue allant de 0.1 à 10.
  int ndisc=100;
  for(int i=0;i<=ndisc;i++){
    double log10rapport=-1+2*(i*1.0)/ndisc;
    double sprop=pow(10,log10rapport)*strue;
    double acc_rate=Run_MCMC(nsteps,sprop,strue,generator);
    cout << "acc_rate : " << acc_rate << endl;
    rapports.push_back(sprop/strue);
    acc_rates.push_back(acc_rate);
  }

  ofstream ofile("results/mcmc.gnu");
  for(int i=0;i<rapports.size();i++){
    ofile << log(rapports[i]) << " " << acc_rates[i] << endl;
  }
  ofile.close();

  exit(0);  
};