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

std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0, 1);
std::normal_distribution<double> distN(0, 1);
vector<DATA> data;
vector<VectorXd> Grid;


//soft clustering en tant que gaussian mixture, pour deux clusters
//on met la moyenne du second cluster à 1.13 pour l'aider
vector<VectorXd> GMM(vector<VectorXd> const &x, default_random_engine &generator)
{
  int n = x.size();
  int s = x[0].size();
  int nclusters = 2;
  vector<VectorXd> p(n);
  for (int i = 0; i < n; i++)
  {
    p[i] = VectorXd::Zero(nclusters);
  }
  //on renvoie la probabilité d'appartenir au cluster 0.
  vector<VectorXd> mu(nclusters);
  for (int j = 0; j < nclusters; j++)
  {
    mu[j] = x[n * distU(generator)];
  };
  VectorXd xbar = VectorXd::Zero(s);
  for (int i = 0; i < n; i++)
  {
    xbar += x[i];
  }
  xbar /= n;
  double s2 = 0;
  for (int i = 0; i < n; i++)
  {
    s2 += (xbar - x[i]).squaredNorm();
  }
  s2 /= n;
  vector<double> pi(nclusters);
  pi[0] = 1;
  pi[1] = 1;
  vector<MatrixXd> S(nclusters);
  S[0] = s2 / nclusters * MatrixXd::Identity(s, s);
  S[1] = s2 / nclusters * MatrixXd::Identity(s, s);
  int c = 0;
  while (c < 25)
  {
    mu[0](0) = -0.025;
    mu[1](0) = 1.02;
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
    cout << S[1] << endl
         << endl;
    //calcul des décomp LDLT des matrices de covariance des clusters
    vector<LDLT<MatrixXd>> ldlt(nclusters);
    for (int j = 0; j < nclusters; j++)
    {
      LDLT<MatrixXd> l(S[j]);
      ldlt[j] = l;
    }
    //expectation
    for (int i = 0; i < n; i++)
    {
      double s = 0;
      for (int j = 0; j < nclusters; j++)
      {
        VectorXd Xc = x[i] - mu[j];
        VectorXd Alpha = ldlt[j].solve(Xc);
        p[i](j) = pi[j] * exp(-0.5 * Xc.dot(Alpha) - 0.5 * (ldlt[j].vectorD().array().log()).sum());
        s += p[i](j);
      }
      for (int j = 0; j < nclusters; j++)
      {
        p[i](j) /= s;
      }
    }
    //maximisation

    for (int j = 0; j < nclusters; j++)
    {
      pi[j] = 0;
      mu[j] = VectorXd::Zero(s);
      for (int i = 0; i < n; i++)
      {
        pi[j] += p[i](j);
        mu[j] += p[i](j) * x[i];
      }
      mu[j] /= pi[j];
      S[j] = MatrixXd::Zero(s, s);
      for (int i = 0; i < n; i++)
      {
        S[j] += p[i](j) * (x[i] - mu[j]) * (x[i] - mu[j]).transpose();
      }
      S[j] /= pi[j];
    }

    c++;
  }
  return p;
}

// MODELE RAVIK.

double computer_model(const double &x, const VectorXd &t)
{
  return x * sin(2 * t(0) * x) + (x + 0.15) * (1 - t(0));
}

double gaussprob(double x, double mu, double sigma)
{
  //renvoie la probabilité gaussienne
  return 1. / (sqrt(2 * 3.14 * pow(sigma, 2))) * exp(-0.5 * pow((x - mu) / sigma, 2));
}

double Kernel_Z_lin(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //linear
  return pow(hpar(2), 2) + pow(hpar(1), 2) * x(0) * y(0); //3/2
}

double Kernel_Z_quad(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //linear
  return pow(hpar(1), 2) * x(0) * y(0) + pow(hpar(2), 2) * pow(x(0) * y(0), 2); //3/2
}

double Kernel_Z_SE_lin(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //squared exponential * linear
  double d = abs(x(0) - y(0));
  return pow(hpar(1), 2) * exp(-0.5 * pow(d / hpar(2), 2)) * x(0) * y(0); //3/2
}

double Kernel_Z_SE_pluslin(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //squared exponential + linear
  double d = abs(x(0) - y(0));
  return pow(hpar(1), 2) * exp(-0.5 * pow(d / hpar(2), 2)) + pow(hpar(3), 2) * x(0) * y(0); //3/2
}

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //squared exponential
  double d = abs(x(0) - y(0));
  return pow(hpar(1), 2) * exp(-0.5 * pow(d / hpar(2), 2)); //3/2
}

double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //squared exponential
  double d = abs(x(0) - y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0), 2) * (1 + (d / hpar(2))) * exp(-d / hpar(1)); //3/2
}

double Kernel_GP_Matern12(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double cor = pow(hpar(0), 2);
  cor *= exp(-abs(x(0) - y(0)) / hpar(1)); //phi
  cor *= exp(-abs(x(1) - y(1)) / hpar(3)); //BK
  cor *= exp(-abs(x(2) - y(2)) / hpar(4)); //COAL
  return cor;
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double cor = pow(hpar(0), 2);
  cor *= (1 + abs(x(0) - y(0)) / hpar(1)) * exp(-abs(x(0) - y(0)) / hpar(1)); //phi
  cor *= (1 + abs(x(1) - y(1)) / hpar(3)) * exp(-abs(x(1) - y(1)) / hpar(3)); //BK
  cor *= (1 + abs(x(2) - y(2)) / hpar(4)) * exp(-abs(x(2) - y(2)) / hpar(4)); //COAL
  return cor;
}

double Kernel_GPX_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor X, 3:lcor theta1, 4:lcor theta2, 5:lcortheta3
  double cor = pow(hpar(0), 2);
  cor *= (1 + abs(x(0) - y(0)) / hpar(1)) * exp(-abs(x(0) - y(0)) / hpar(1)); //phi
  cor *= (1 + abs(x(1) - y(1)) / hpar(3)) * exp(-abs(x(1) - y(1)) / hpar(3)); //BK
  cor *= (1 + abs(x(2) - y(2)) / hpar(4)) * exp(-abs(x(2) - y(2)) / hpar(4)); //COAL
  cor *= (1 + abs(x(3) - y(3)) / hpar(5)) * exp(-abs(x(3) - y(3)) / hpar(5)); //COAL
  return cor;
}

double Kernel_GPX_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor X, 3:lcor theta1, 4:lcor theta2, 5:lcortheta3
  double cor = pow(hpar(0), 2);
  double m = pow((x(0) - y(0)) / hpar(1), 2) + pow((x(1) - y(1)) / hpar(3), 2) + pow((x(2) - y(2)) / hpar(4), 2) + pow((x(3) - y(3)) / hpar(5), 2);
  return cor * exp(-0.5 * m);
}

/* natural log of the gamma function
   gammln as implemented in the
 * first edition of Numerical Recipes in C */
double gammln(double xx)
{
  double x, tmp, ser;
  const static double cof[6] = {76.18009172947146, -86.50532032941677,
                                24.01409824083091, -1.231739572450155,
                                0.1208650973866179e-2, -0.5395239384953e-5};
  int j;

  x = xx - 1.0;
  tmp = x + 5.5;
  tmp -= (x + 0.5) * log(tmp);
  ser = 1.000000000190015;
  for (j = 0; j <= 5; j++)
  {
    x += 1.0;
    ser += cof[j] / x;
  }
  return -tmp + log(2.5066282746310005 * ser);
}
double loginvgammaPdf(double x, double a, double b)
{
  return a * log(b) - gammln(a) + (a + 1) * log(1 / x) - b / x;
}

double lognorm(double x, double mean, double std)
{
  return -0.5 * pow((x - mean) / std, 2);
}

double logprior_hpars(VectorXd const &hpars)
{
  //seulement sur la longueur de corrélation pour que ça soit plus normal.
  return 0;
  //return lognorm(hpars(2),3,1);
  //+lognorm(hpars(1),0.25,0.1)+lognorm(hpars(0),0.1,0.05);
}

double logprior_hpars_invgamma(VectorXd const &hpars)
{
  double alpha = 50;
  double beta = 500;
  return loginvgammaPdf(hpars(1), alpha, beta);
}

double logprior_pars(VectorXd const &pars)
{
  return 0;
}

double logprior_parsnormgrand(VectorXd const &pars)
{
  //prior gaussien large sur les paramètres. ils seront dans l'espace (0,1.)
  double d = 0;
  for (int i = 0; i < 3; i++)
  {
    d += lognorm(pars(i), 0.5, 0.5);
  }
  return d;
}

double logprior_parsunif(VectorXd const &pars)
{
  return 0;
}

void WriteObs( vector<VectorXd> &X,VectorXd &obs,string filename)
{
  ofstream ofile(filename);
  for (int i = 0; i<X.size();i++)
  {
    for (int j=0;j<X[i].size();j++)
    {
      ofile << X[i](j) << " ";
    }
    ofile << obs(i) << endl;
  }
  ofile.close();
}

vector<VectorXd> read_hparskoh(string const &filename, list<int> &cases)
{
  //lecture d'un fichier d'hpars koh, qui est écrit de la forme n V, avec n le numéro du cas et V le vecteur d'hyperparamètres.
  map<int, VectorXd> m;
  ifstream ifile(filename);
  if (ifile)
  {
    string line;
    while (getline(ifile, line))
    {
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>());
      int key = stoi(words[0]);
      VectorXd hpars(3);
      for (int i = 0; i < hpars.size(); i++)
      {
        hpars(i) = stod(words[i + 1]);
      }
      m.insert(make_pair(key, hpars));
    }
    cout << "number of samples in the file : " << m.size() << endl;
  }
  vector<VectorXd> v;
  for (const int &i : cases)
  {
    v.push_back(m[i]);
  }
  cout << "number of samples loaded: " << v.size() << endl;
  return v;
}

vector<VectorXd> read_samples(string const &filename)
{
  //lecture d'un fichier de samples theta.
  int dim = 3; //dimension des paramètres
  vector<VectorXd> v;
  ifstream ifile(filename);
  if (ifile)
  {
    string line;
    while (getline(ifile, line))
    {
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>());
      int key = stoi(words[0]);
      VectorXd theta(3);
      for (int i = 0; i < theta.size(); i++)
      {
        theta(i) = stod(words[i]);
      }
      v.push_back(theta);
    }
  }
  cout << "number of samples loaded: " << v.size() << endl;
  return v;
}

pair<vector<VectorXd>, vector<VectorXd>> read_optimalhpars(string const &filename, int dim_theta, int dim_hpars)
{
  //lecture d'un fichier d'hpars optimaux., qui est écrit de la forme n V, avec n le numéro du cas et V le vecteur d'hyperparamètres.
  vector<VectorXd> thetas;
  vector<VectorXd> hparsv;
  ifstream ifile(filename);
  if (ifile)
  {
    string line;
    while (getline(ifile, line))
    {
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>());
      int key = stoi(words[0]);
      VectorXd theta(dim_theta);
      for (int i = 0; i < theta.size(); i++)
      {
        theta(i) = stod(words[i]);
      }
      VectorXd hpars(dim_hpars);
      for (int i = 0; i < hpars.size(); i++)
      {
        hpars(i) = stod(words[i + dim_theta]);
      }
      thetas.push_back(theta);
      hparsv.push_back(hpars);
    }
    cout << "number of samples in the file : " << thetas.size() << endl;
  }
  auto p = make_pair(thetas, hparsv);
  return p;
}

void write_hparskoh(string const &filename, list<int> &cases, vector<VectorXd> &hpars)
{
  ofstream ofile(filename);
  int c = 0;
  for (int i : cases)
  {
    ofile << i << " ";
    for (int j = 0; j < hpars[c].size(); j++)
    {
      ofile << hpars[c](j) << " ";
    }
    ofile << endl;
    c++;
  }
  ofile.close();
}

vector<vector<VectorXd>> revert_vector(vector<vector<VectorXd>> const &v)
{
  //inverts the inner and outer vector.
  int s0 = v.size();
  int s1 = v[0].size();
  vector<vector<VectorXd>> res;
  for (int i = 0; i < s1; i++)
  {
    vector<VectorXd> tmp;
    for (int j = 0; j < s0; j++)
    {
      tmp.push_back(v[j][i]);
    }
    res.push_back(tmp);
  }
  return res;
}

double optfunc_opti(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  //pour chercher le maximum d'une densité liée à 1 configuration expérimentale.
  auto d = (DensityOpt *)data; //cast
  VectorXd X(x.size());
  for (int j = 0; j < x.size(); j++)
  {
    X(j) = x[j];
  }
  VectorXd p = d->EvaluateHparOpt(X);
  double l = d->loglikelihood_theta(X, p);
  return l;
};

int main(int argc, char **argv)
{
  default_random_engine generator(123456);

  /*Paramètres de simulation*/
  //pour la MCMC
  int nombre_steps_mcmc = 5e5;
  int nombre_samples_collected = 5000; //1 sample tous les 100.
  int nautocor = 2000;

  int time_opt_opti = 10;      // 10 secondes par optimisation opti
  int time_opt_koh_loc = 600;  // 10 minutes par optimisation KOH locale
  int time_opt_koh_glo = 7200; // 2h pour optimisation KOH globale
  double inputerr = 0.6;       //sigma=0.1K d'incertitude en input.

  // Bornes sup des paramètres et hyperparamètres
  int dim_theta = 1;
  int dim_hpars = 3;

  VectorXd lb_t = -0.5 * VectorXd::Ones(1);
  VectorXd ub_t = 1.5 * VectorXd::Ones(1);

  VectorXd lb_hpars(3);
  lb_hpars(0) = 1e-5;
  lb_hpars(1) = 1e-5;
  lb_hpars(2) = 0.001;
  VectorXd ub_hpars(3);
  ub_hpars(0) = 1;
  ub_hpars(1) = 1;
  ub_hpars(2) = 5; 

  VectorXd hpars_z_guess = 0.5 * (lb_hpars + ub_hpars);


  //création des observations.
  double noise = 0.1;
  int ndata = 8;
  vector<VectorXd> Xlocs(ndata);
  VectorXd data(ndata);
  for (int i = 0; i < ndata; i++)
  {
    VectorXd x(1);
    x << 0.0625 + double(i) / double(ndata);
    double y = x(0) + noise * distN(generator);
    Xlocs[i] = x;
    data(i) = y;
  }
  string expfilename = "results/obs.gnu";
  WriteObs(Xlocs,data,expfilename);
  //création des points de prédiction
  int samp_size = 80;
  vector<VectorXd> XPREDS(samp_size);
  for (int i = 0; i < samp_size; i++)
  {
    VectorXd x(1);
    x << 0 + 1 * double(i) / double(samp_size);
    XPREDS[i] = x;
  }


  //construction du DoE initial
  DoE doe_init(lb_t, ub_t, 100, 10); //doe halton de 100 points
  doe_init.WriteGrid("results/grid.gnu");

  auto lambda_priormean = [](vector<VectorXd> const &X, VectorXd const &hpars)
  {
    VectorXd b = VectorXd::Zero(X.size());
    return b;
  };
  auto lambda_model = [](vector<VectorXd> const &X, VectorXd const &theta) mutable
  {
    VectorXd pred(X.size());
    for (int j = 0; j < X.size(); j++)
    {
      pred(j) = computer_model(X[j](0), theta);
    }
    return pred;
  };

  Density MainDensity(doe_init);
  MainDensity.SetModel(lambda_model);
  MainDensity.SetKernel(Kernel_Z_SE);
  MainDensity.SetHparsBounds(lb_hpars, ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(lambda_priormean);
  MainDensity.SetObservations(Xlocs, data);

  VectorXd X_init_mcmc = 0.5 * VectorXd::Ones(dim_theta);
  MatrixXd COV_init = MatrixXd::Identity(dim_theta, dim_theta);
  COV_init(0, 0) = pow(0.04, 2);
  cout << "COV_init : " << endl
       << COV_init << endl;

  //début des calibrations.

  //MCMC koh separate sur les densités séparées

  // calcul hpars koh separate
  vector<VectorXd> hparskoh_separate(1);
  hparskoh_separate[0] = MainDensity.HparsKOH(hpars_z_guess, 10);

  cout << "hparskoh sep:" << hparskoh_separate[0].transpose() << endl;

  //calibration koh separate
  {
    auto in_bounds = [&MainDensity](VectorXd const &X)
    {
      return MainDensity.in_bounds_pars(X);
    };

    auto get_hpars_kohs = [&hparskoh_separate](VectorXd const &X)
    {
      vector<VectorXd> h(1);
      h[0] = hparskoh_separate[0];
      return h;
    };
    auto compute_score_kohs = [&MainDensity](vector<VectorXd> p, VectorXd const &X)
    {
      double res = MainDensity.loglikelihood_theta(X, p[0]) + MainDensity.EvaluateLogPPars(X);
      return res;
    };

    cout << "début mcmc densité koh separate expérience" << endl;
    auto visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_kohs, get_hpars_kohs, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for(int i=0;i<samples.size();i++){
      samples[i]=visited_steps[i*(visited_steps.size()/samples.size())];
      hparsofsamples[i]=get_hpars_kohs(samples[i])[0];
    }
    //Set des samples dans chaque densité

    MainDensity.SetNewSamples(samples);
    MainDensity.SetNewHparsOfSamples(hparsofsamples);
    MainDensity.SetNewAllSamples(visited_steps);

    //diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/separatedensities/kohs/autocor.gnu");
    //écriture des samples. Il faut une fonction dédiée.
    string fname = "results/separatedensities/kohs/samp.gnu";
    string fnameall = "results/separatedensities/kohs/allsamp.gnu";
    string fnamepred = "results/separatedensities/kohs/preds.gnu";
    string fnamepredF = "results/separatedensities/kohs/predsF.gnu";
    string fnamesF = "results/separatedensities/kohs/sampsF.gnu";
    string fnamesZ = "results/separatedensities/kohs/sampsZ.gnu";
    MainDensity.WriteSamples(fname);
    //MainDensity.WriteMCMCSamples(fnameall);
    //prédictions
    MainDensity.WritePredictions(XPREDS, fnamepred);
    MainDensity.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);
    //prédictions
    //écriture des observations avec erreur expérimentale KOH
    string filename = "results/obs.gnu";
    exit(0);
  }

  //Phase Opti
  //surrogates sur 200 points et assez rapides.
  DensityOpt Dopt(MainDensity);

  //MCMC opt sur les densités séparées.calibration opti densités séparées
  {
    auto in_bounds = [&Dopt](VectorXd const &X)
    {
      return Dopt.in_bounds_pars(X);
    };
    auto get_hpars_opti = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      vector<VectorXd> p(1);
      p[0] = Dopt.HparsOpt(X, hpars_z_guess, 1e-4);
      return p;
    };
    auto compute_score_opti = [&Dopt](vector<VectorXd> const &p, VectorXd const &X)
    {
      double d = Dopt.loglikelihood_theta(X, p[0]) + Dopt.EvaluateLogPPars(X);
      return d;
    };
    cout << "début mcmc densité fmp" << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for(int i=0;i<samples.size();i++){
      samples[i]=visited_steps[i*(visited_steps.size()/samples.size())];
      hparsofsamples[i]=get_hpars_opti(samples[i])[0];
    }
    //Set des samples dans chaque densité
    Dopt.SetNewSamples(samples);
    Dopt.SetNewHparsOfSamples(hparsofsamples);
    Dopt.SetNewAllSamples(visited_steps);
    //diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/separatedensities/opt/autocor.gnu");
    //écriture des samples. Il faut une fonction dédiée.
    string fname = "results/separatedensities/opt/samp.gnu";
    string fnameall = "results/separatedensities/opt/allsamp.gnu";
    string fnameallh = "results/separatedensities/opt/allhpars.gnu";
    string fnamepred = "results/separatedensities/opt/preds.gnu";
    string fnamepredF = "results/separatedensities/opt/predsF.gnu";
    string fnamesF = "results/separatedensities/opt/sampsF.gnu";
    string fnamesZ = "results/separatedensities/opt/sampsZ.gnu";
    Dopt.WriteSamples(fname);
    //Dopt.WriteMCMCSamples(fnameall);
    //WriteVector(fnameallh,allhpars_opti);

    //prédictions
    Dopt.WritePredictions(XPREDS, fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);
  }

  //MCMC phase Bayes sur les densités séparées. Pour chaque densités : que 2 hpars. La variance et la correlation length.
  {
    //initial MCMC. remplissage des états initiaux.

    VectorXd X_init_mcmc_fb(dim_theta + dim_hpars);
    MatrixXd COV_init_fb = MatrixXd::Zero(dim_theta + dim_hpars, dim_theta + dim_hpars);
    X_init_mcmc_fb.head(1) = X_init_mcmc;
    X_init_mcmc_fb(1) = 0.5;
    X_init_mcmc_fb(2) = 0.5;
    X_init_mcmc_fb(3) = 0.5; //X_init_mcmc_fb(5)=1e4; //échelle log pour sigma
    COV_init_fb.topLeftCorner(1, 1) = COV_init;
    int m = lb_hpars.size();
    COV_init_fb(1, 1) = pow(0.01, 2);
    COV_init_fb(2, 2) = pow(0.01, 2);
    COV_init_fb(3, 3) = pow(0.1, 2);
    cout << "covinitfb" << endl
         << COV_init_fb << endl;
    cout << "Xinitfb" << endl
         << X_init_mcmc_fb.transpose() << endl;

    auto in_bounds_fb = [&MainDensity, dim_hpars](VectorXd const &X)
    {
      VectorXd hp = X.tail(dim_hpars);
      ;
      VectorXd theta = X.head(1);
      return MainDensity.in_bounds_pars(X) && MainDensity.in_bounds_hpars(hp);
    };

    auto get_hpars_fb = [](VectorXd const &X)
    {
      //renvoyer un vecteur sans intérêt.
      vector<VectorXd> v(1);
      return v;
    };
    auto compute_score_fb = [&lb_hpars, &MainDensity, dim_hpars](vector<VectorXd> const &p, VectorXd const &X)
    {
      //il faut décomposer X en thetas/ hpars. 3 est la dimension des paramètres
      //logpriorhpars=0 I think
      VectorXd theta = X.head(1);
      VectorXd hp = X.tail(dim_hpars);
      double d = MainDensity.loglikelihood_theta(theta, hp);
      double lp = MainDensity.EvaluateLogPHpars(hp) + MainDensity.EvaluateLogPPars(theta);
      return d + lp;
    };
    cout << "début mcmc densité bayes expérience" << endl;

    vector<VectorXd> visited_steps = Run_MCMC(5 * nombre_steps_mcmc, X_init_mcmc_fb, COV_init_fb, compute_score_fb, get_hpars_fb, in_bounds_fb, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for(int i=0;i<samples.size();i++){
      samples[i]=visited_steps[i*(visited_steps.size()/samples.size())].head(dim_theta);
      hparsofsamples[i]=visited_steps[i*(visited_steps.size()/samples.size())].tail(dim_hpars);
    }
    //set des samples dans la première densité pour calculer la corrélation.
    //diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/separatedensities/fb/autocor.gnu");

    MainDensity.SetNewSamples(samples);
    MainDensity.SetNewHparsOfSamples(hparsofsamples);

    //écriture des samples. Il faut une fonction dédiée.
    string fname = "results/separatedensities/fb/samp.gnu";
    string fnameall = "results/separatedensities/fb/allsamp.gnu";

    string fnamepred = "results/separatedensities/fb/preds.gnu";
    string fnamepredF = "results/separatedensities/fb/predsF.gnu";
    string fnamesF = "results/separatedensities/fb/sampsF.gnu";
    string fnamesZ = "results/separatedensities/fb/sampsZ.gnu";
    MainDensity.WriteSamples(fname);
    //MainDensity.WriteMCMCSamples(fnameall);

    //prédictions
    MainDensity.WritePredictions(XPREDS, fnamepred);
    MainDensity.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);
  }

  exit(0);
}
