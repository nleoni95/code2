// Dans ce fichier : on fait un exemple de construction de l'algorithme surrogates opti.
// On commence par le toy problem de l'article 1.
// je veux comparer Opti et Opti fait avec des surrogates.
// je veux aussi trouver une mesure propre de la qualité d'un surrogate. Normalisation qqpart ? norme a priori ? ou plutôt a posteriori ?
// je veux mettre en place les méthodes de construction alternatives (choix progressif parmi les samples a posteriori, et construction QMC).

// il me faut à la fin de belles courbes d'erreur qui convergent. Et pourquoi pas les figures de points sélectionnés progressivement.
// premier test de convergence pour ma mesure d'erreur : que ça tende bien vers 0 lorsque le nombre de points augmente. faire des grids avec peu de pts, bcp de points... 

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

double computer_model(const double &x, const VectorXd &t)
{
  return x * sin(2 * t(0) * x) + (x + 0.15) * (1 - t(0));
}

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //squared exponential
  double d = abs(x(0) - y(0));
  return pow(hpar(1), 2) * exp(-0.5 * pow(d / hpar(2), 2)); //3/2
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double cor = pow(hpar(0), 2);
  cor *= (1 + abs(x(0) - y(0)) / hpar(1)) * exp(-abs(x(0) - y(0)) / hpar(1)); //phi
  return cor;
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
  return 0;
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

void WriteObs(vector<VectorXd> &X, VectorXd &obs, string filename)
{
  ofstream ofile(filename);
  for (int i = 0; i < X.size(); i++)
  {
    for (int j = 0; j < X[i].size(); j++)
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

VectorXd evaluate_surrogate(vector<VectorXd> const & thetas_ref, vector<VectorXd> const & hpars_ref,function<pair<VectorXd,VectorXd>(VectorXd const &)> const &get_hpars_and_var){
  //évaluation de la qualité d'un surrogate par la fonction get_hpars qu'il propose. On pourra en profiter pour vérifier que pour la densité fmp e référence ça fait bien zéro.
  //prenons l'erreur moyenne L2 relative ? Non. On va faire la moyenne + variance. averaged absolute individual standardized error. Et la moyenne a posteriori de cette quantité.
  //calcul de l'erreur dans chaque dimension.
  if(!thetas_ref.size()==hpars_ref.size()){cerr << "erreur : différentes tailles evaluate_surrogate !" << endl; exit(0);}
  VectorXd err=VectorXd::Zero(hpars_ref[0].size());
  for(int i=0;i<thetas_ref.size();i++)
  {
    auto p=get_hpars_and_var(thetas_ref[i]);
    VectorXd std=p.second.array().sqrt();
    VectorXd v=(hpars_ref[i]-p.first).cwiseQuotient(std).array().abs();
    err+=v;
  }
  return err/thetas_ref.size();
}


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
  WriteObs(Xlocs, data, expfilename);
  //création des points de prédiction
  int samp_size = 80;
  vector<VectorXd> XPREDS(samp_size);
  for (int i = 0; i < samp_size; i++)
  {
    VectorXd x(1);
    x << 0 + 1 * double(i) / double(samp_size);
    XPREDS[i] = x;
  }

  //building initial DoE
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
  MainDensity.SetOutputerr(true, 0.1, 0);
  //MainDensity.SetInputerr(false,0.0,0,VectorXd::Zero(Xlocs.size()));

  VectorXd X_init_mcmc = 0.5 * VectorXd::Ones(dim_theta);
  MatrixXd COV_init = MatrixXd::Identity(dim_theta, dim_theta);
  COV_init(0, 0) = pow(0.04, 2);
  cout << "COV_init : " << endl
       << COV_init << endl;

  MatrixXd Bounds_hpars_HGPs(2, 3); //3 hpars HGPs. std, noise, 1 longueur de corrélation.
  Bounds_hpars_HGPs(0, 0) = 1e-8;      //std
  Bounds_hpars_HGPs(1, 0) = 10;
  Bounds_hpars_HGPs(0, 1) = 1e-8; //lcor
  Bounds_hpars_HGPs(1, 1) = 10;
  Bounds_hpars_HGPs(0, 2) = 1e-8; //stdnoise
  Bounds_hpars_HGPs(1, 2) = 1;
  VectorXd Hpars_guess_HGPs = 0.5 * (Bounds_hpars_HGPs.row(0) + Bounds_hpars_HGPs.row(1)).transpose();

  //créer un DoE de test.
  vector<VectorXd> thetas_test;
  vector<VectorXd> hpars_test;
  {
    DensityOpt Dopt(MainDensity);
    int ngrid = 100;
    double timeopt = 1e-2;
    DoE doe1(lb_t, ub_t, ngrid, 1);
    thetas_test=doe1.GetGrid();
    for (int i = 0; i < ngrid; i++)
    {
      VectorXd hpar1 = Dopt.HparsOpt(thetas_test[i], hpars_z_guess, timeopt);
      hpars_test.push_back(hpar1);
    }
  }
  //écriture du test set.
  string fname="results/hgps/testset.gnu";
  WriteVectors(thetas_test,hpars_test,fname);

  //Calibration phase
  /*FMP calibration*/
  {
    DensityOpt Dopt(MainDensity);
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
    cout << "Beginning FMP calibration..." << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples[i] = get_hpars_opti(samples[i])[0];
    }
    //diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/fmp/autocor.gnu");

    //write samples
    string fnamesamp = "results/fmp/samp.gnu";
    string fnameallsamp = "results/fmp/allsteps.gnu";
    WriteVectors(samples, hparsofsamples, fnamesamp);
    WriteVector(visited_steps, fnameallsamp);

    //predictions
    Dopt.SetNewSamples(samples);
    Dopt.SetNewHparsOfSamples(hparsofsamples);
    string fnamepred = "results/fmp/preds.gnu";
    string fnamesF = "results/fmp/sampsF.gnu";
    string fnamesZ = "results/fmp/sampsZ.gnu";
    Dopt.WritePredictions(XPREDS, fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);
  }

  /*FMP calibration avec surrogate de 100 points*/
  {
    DensityOpt Dopt(MainDensity);
    int ngrid = 100;
    double timeopt = 1e-2;
    DoE doe1(lb_t, ub_t, ngrid, 1);
    auto thetas1 = doe1.GetGrid();
    vector<VectorXd> hpars1;
    for (int i = 0; i < ngrid; i++)
    {
      VectorXd hpar1 = Dopt.HparsOpt(thetas1[i], hpars_z_guess, timeopt);
      hpars1.push_back(hpar1);
    }
    //construction des surrogates des hpars optimaux.
    Dopt.BuildHGPs(thetas1, hpars1, Kernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, Hpars_guess_HGPs);

    //MCMC
    auto in_bounds = [&Dopt](VectorXd const &X)
    {
      return Dopt.in_bounds_pars(X);
    };
    auto get_hpars_opti = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      vector<VectorXd> p(1);
      p[0] = Dopt.EvaluateHparOpt(X);
      return p;
    };
    auto compute_score_opti = [&Dopt](vector<VectorXd> const &p, VectorXd const &X)
    {
      double d = Dopt.loglikelihood_theta(X, p[0]) + Dopt.EvaluateLogPPars(X);
      return d;
    };
    cout << "Beginning FMP calibration with QMC surrogate..." << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples[i] = get_hpars_opti(samples[i])[0];
    }
    //diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/fmpqmc/autocor.gnu");

    //write samples
    string fnamesamp = "results/fmpqmc/samp.gnu";
    string fnameallsamp = "results/fmpqmc/allsteps.gnu";
    WriteVectors(samples, hparsofsamples, fnamesamp);
    WriteVector(visited_steps, fnameallsamp);

    //predictions
    Dopt.SetNewSamples(samples);
    Dopt.SetNewHparsOfSamples(hparsofsamples);
    string fnamepred = "results/fmpqmc/preds.gnu";
    string fnamesF = "results/fmpqmc/sampsF.gnu";
    string fnamesZ = "results/fmpqmc/sampsZ.gnu";
    Dopt.WritePredictions(XPREDS, fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);

    //test. Juste afficher la variance de prédiction avec les quantités pour voir si ça marche bien. 
    auto get_hpars_and_var = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v = Dopt.EvaluateVarHparOpt(X);
      return make_pair(h,v);
    };

    
    auto v=evaluate_surrogate(thetas_test,hpars_test,get_hpars_and_var);

    cout << "final score : " << v.transpose() << endl;

    //je calcule toutes les prédictions du GP sur les points de test.
    vector<VectorXd> preds_test;
    vector<VectorXd> stds_test;
    for(int i=0;i<thetas_test.size();i++){
      auto p=get_hpars_and_var(thetas_test[i]);
      preds_test.push_back(p.first);
      VectorXd v = p.second.array().sqrt();
      stds_test.push_back(v);
    }
    string fnname="results/hgps/100.gnu";
    WriteVectors(thetas_test,preds_test,stds_test,fnname);

  }

  /*FMP calibration avec surrogate de 300 points*/
  {
    DensityOpt Dopt(MainDensity);
    int ngrid = 300;
    double timeopt = 1e-2;
    DoE doe1(lb_t, ub_t, ngrid, 1);
    auto thetas1 = doe1.GetGrid();
    vector<VectorXd> hpars1;
    for (int i = 0; i < ngrid; i++)
    {
      VectorXd hpar1 = Dopt.HparsOpt(thetas1[i], hpars_z_guess, timeopt);
      hpars1.push_back(hpar1);
    }
    //construction des surrogates des hpars optimaux.
    Dopt.BuildHGPs(thetas1, hpars1, Kernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, Hpars_guess_HGPs);

    //MCMC
    auto in_bounds = [&Dopt](VectorXd const &X)
    {
      return Dopt.in_bounds_pars(X);
    };
    auto get_hpars_opti = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      vector<VectorXd> p(1);
      p[0] = Dopt.EvaluateHparOpt(X);
      return p;
    };
    auto compute_score_opti = [&Dopt](vector<VectorXd> const &p, VectorXd const &X)
    {
      double d = Dopt.loglikelihood_theta(X, p[0]) + Dopt.EvaluateLogPPars(X);
      return d;
    };
    cout << "Beginning FMP calibration with QMC surrogate..." << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples[i] = get_hpars_opti(samples[i])[0];
    }
    //diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/fmpqmc/autocor.gnu");

    //write samples
    string fnamesamp = "results/fmpqmc/samp.gnu";
    string fnameallsamp = "results/fmpqmc/allsteps.gnu";
    WriteVectors(samples, hparsofsamples, fnamesamp);
    WriteVector(visited_steps, fnameallsamp);

    //predictions
    Dopt.SetNewSamples(samples);
    Dopt.SetNewHparsOfSamples(hparsofsamples);
    string fnamepred = "results/fmpqmc/preds.gnu";
    string fnamesF = "results/fmpqmc/sampsF.gnu";
    string fnamesZ = "results/fmpqmc/sampsZ.gnu";
    Dopt.WritePredictions(XPREDS, fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);

    //test. Juste afficher la variance de prédiction avec les quantités pour voir si ça marche bien. 
    auto get_hpars_and_var = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      VectorXd h= Dopt.EvaluateHparOpt(X);
      VectorXd v= Dopt.EvaluateVarHparOpt(X);
      return make_pair(h,v);
    };

    
    auto v=evaluate_surrogate(thetas_test,hpars_test,get_hpars_and_var);

    cout << "final score 300 pts : " << v.transpose() << endl;

    //je calcule toutes les prédictions du GP sur les points de test.
    vector<VectorXd> preds_test;
    vector<VectorXd> stds_test;
    for(int i=0;i<thetas_test.size();i++){
      auto p=get_hpars_and_var(thetas_test[i]);
      preds_test.push_back(p.first);
      VectorXd v = p.second.array().sqrt();
      stds_test.push_back(v);
    }
    string fnname="results/hgps/300.gnu";
    WriteVectors(thetas_test,preds_test,stds_test,fnname);
  }

  /*FMP calibration avec le test set*/
  {
    DensityOpt Dopt(MainDensity);

    //construction des surrogates des hpars optimaux.
    Dopt.BuildHGPs(thetas_test, hpars_test, Kernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, Hpars_guess_HGPs);

    //MCMC
    auto in_bounds = [&Dopt](VectorXd const &X)
    {
      return Dopt.in_bounds_pars(X);
    };
    auto get_hpars_opti = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      vector<VectorXd> p(1);
      p[0] = Dopt.EvaluateHparOpt(X);
      return p;
    };
    auto compute_score_opti = [&Dopt](vector<VectorXd> const &p, VectorXd const &X)
    {
      double d = Dopt.loglikelihood_theta(X, p[0]) + Dopt.EvaluateLogPPars(X);
      return d;
    };
    cout << "Beginning FMP calibration with QMC surrogate..." << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples[i] = get_hpars_opti(samples[i])[0];
    }
    //diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/fmpqmc/autocor.gnu");

    //write samples
    string fnamesamp = "results/fmpqmc/samp.gnu";
    string fnameallsamp = "results/fmpqmc/allsteps.gnu";
    WriteVectors(samples, hparsofsamples, fnamesamp);
    WriteVector(visited_steps, fnameallsamp);

    //predictions
    Dopt.SetNewSamples(samples);
    Dopt.SetNewHparsOfSamples(hparsofsamples);
    string fnamepred = "results/fmpqmc/preds.gnu";
    string fnamesF = "results/fmpqmc/sampsF.gnu";
    string fnamesZ = "results/fmpqmc/sampsZ.gnu";
    Dopt.WritePredictions(XPREDS, fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);

    //test. Juste afficher la variance de prédiction avec les quantités pour voir si ça marche bien. 
    auto get_hpars_and_var = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      VectorXd h= Dopt.EvaluateHparOpt(X);
      VectorXd v= Dopt.EvaluateVarHparOpt(X);
      return make_pair(h,v);
    };

    
    auto v=evaluate_surrogate(thetas_test,hpars_test,get_hpars_and_var);

    cout << "final score best hgps : " << v.transpose() << endl;

    //je calcule toutes les prédictions du GP sur les points de test.
    vector<VectorXd> preds_test;
    vector<VectorXd> stds_test;
    for(int i=0;i<thetas_test.size();i++){
      auto p=get_hpars_and_var(thetas_test[i]);
      preds_test.push_back(p.first);
      VectorXd v = p.second.array().sqrt();
      stds_test.push_back(v);
    }
    string fnname="results/hgps/best.gnu";
    WriteVectors(thetas_test,preds_test,stds_test,fnname);
  }

  exit(0);
}
