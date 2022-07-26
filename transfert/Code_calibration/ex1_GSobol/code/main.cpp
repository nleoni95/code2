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
#include <ctime>
#include "densities.h"

using namespace std;
using namespace Eigen;

std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0, 1);
std::normal_distribution<double> distN(0, 1);
vector<DATA> data;
vector<VectorXd> Grid;

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // échelle log pour les hpars.
  // l'indice 0 contient sigma, les longueur de corrélation ensuite.
  double d1 = abs(x(0) - y(0)) / exp(hpar(1));
  double d2 = abs(x(1) - y(1)) / exp(hpar(2));
  double d3 = abs(x(2) - y(2)) / exp(hpar(3));
  double d4 = abs(x(3) - y(3)) / exp(hpar(4));
  double d5 = abs(x(4) - y(4)) / exp(hpar(5));
  double d6 = abs(x(5) - y(5)) / exp(hpar(6));
  double cor = pow(exp(hpar(0)), 2);
  cor *= (1 + sqrt(5) * d1 + 5 * pow(d1, 2) / 3) * exp(-sqrt(5) * d1); // x1
  cor *= (1 + sqrt(5) * d2 + 5 * pow(d2, 2) / 3) * exp(-sqrt(5) * d2); // x1
  cor *= (1 + sqrt(5) * d3 + 5 * pow(d3, 2) / 3) * exp(-sqrt(5) * d3); // x1
  cor *= (1 + sqrt(5) * d4 + 5 * pow(d4, 2) / 3) * exp(-sqrt(5) * d4); // x1
  cor *= (1 + sqrt(5) * d5 + 5 * pow(d5, 2) / 3) * exp(-sqrt(5) * d5); // x1
  cor *= (1 + sqrt(5) * d6 + 5 * pow(d6, 2) / 3) * exp(-sqrt(5) * d6); // x1
  return cor;
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // noyau pour le GP. 0:sigma intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double cor = pow(hpar(0), 2);
  cor *= (1 + abs(x(0) - y(0)) / hpar(1)) * exp(-abs(x(0) - y(0)) / hpar(1)); // x1
  cor *= (1 + abs(x(1) - y(1)) / hpar(3)) * exp(-abs(x(1) - y(1)) / hpar(3)); // x2
  cor *= (1 + abs(x(2) - y(2)) / hpar(4)) * exp(-abs(x(2) - y(2)) / hpar(4)); // x3
  cor *= (1 + abs(x(3) - y(3)) / hpar(5)) * exp(-abs(x(3) - y(3)) / hpar(5)); // x4
  cor *= (1 + abs(x(4) - y(4)) / hpar(6)) * exp(-abs(x(4) - y(4)) / hpar(6)); // x5
  cor *= (1 + abs(x(5) - y(5)) / hpar(7)) * exp(-abs(x(5) - y(5)) / hpar(7)); // x6
  return cor;
}

double DKernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar, int p)
{
  // dérivée du noyau GP par rapport aux hyperparamètres.
  //  noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  if (p == 0)
  { // dérivée par rapport à sigma edm
    return 2 * Kernel_GP_Matern32(x, y, hpar) / hpar(0);
  }
  else if (p == 2)
  { // dérivée par rapport à sigma obs
    if (x == y)
    {
      return 2 * hpar(2);
    }
    else
    {
      return 0;
    }
  }
  else if (p <= 7)
  { // dérivée par rapport à une longueur de corrélation
    double cor = pow(hpar(0), 2);
    VectorXd d(6);
    d(0) = abs(x(0) - y(0)) / hpar(1);
    for (int i = 1; i < 6; i++)
    {
      d(i) = abs(x(i) - y(i)) / hpar(i + 2);
    }
    if (p == 1)
    {
      cor *= pow(d(0), 2) * exp(-d(0)) / hpar(1);
    }
    else
    {
      cor *= (1 + d(0)) * exp(-d(0));
    }
    for (int i = 3; i < 8; i++)
    {
      if (p == i)
      {
        cor *= pow(d(i - 2), 2) * exp(-d(i - 2)) / hpar(i);
      }
      else
      {
        cor *= (1 + d(i - 2)) * exp(-d(i - 2));
      }
    }
    return cor;
  }
  else
  {
    cerr << "erreur calcul dérivée. indice demandé : " << p << ". Maximum : 7" << endl;
    exit(0);
  }
}

double gammln(double xx)
{
  //fonction gamma
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
  // log densité inverse gamma
  return a * log(b) - gammln(a) + (a + 1) * log(1 / x) - b / x;
}

double logprior_hpars(VectorXd const &hpars)
{
  // invgamma pour les longueur de corrélation et sigma.
  double a = 1;
  double b = 1;
  double res = 0;
  for (int i = 1; i < hpars.size(); i++)
  {
    res += loginvgammaPdf(exp(hpars(i)), a, b);
  }
  double asigma = 10;
  double bsigma = 1;
  res += loginvgammaPdf(exp(hpars(0)), asigma, bsigma);
  return res;
}

double logprior_pars(VectorXd const &pars)
{
  // prior uniforme pour les paramètres
  return 0;
}

void WriteObs(vector<VectorXd> &X, VectorXd &obs, string filename)
{
  //écriture des obesrvations
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

double evaluate_hgp_surrogate(vector<VectorXd> const &thetas_ref, vector<VectorXd> const &hpars_ref, vector<double> const &scores_ref, function<pair<VectorXd, VectorXd>(VectorXd const &)> const &get_meanpred, function<double(VectorXd const &, VectorXd const &)> const &get_score)
{
  // calcul de la posterior-averaged log-likelihood error à partir d'un sample de référence
  vector<VectorXd> hpars_calcules;
  for (int i = 0; i < thetas_ref.size(); i++)
  {
    VectorXd hpcalc = get_meanpred(thetas_ref[i]).first;
    hpars_calcules.push_back(hpcalc);
  }
  double numerr_loglik = 0;
  double denomerr_loglik = 0;
  for (int i = 0; i < scores_ref.size(); i++)
  {
    double score_calcule = get_score(thetas_ref[i], hpars_calcules[i]);
    numerr_loglik += pow(scores_ref[i] - score_calcule, 2);
    denomerr_loglik += pow(scores_ref[i], 2);
  }
  double err_loglik = numerr_loglik / denomerr_loglik;
  return err_loglik;
}

double evaluate_ll_surrogate(vector<VectorXd> const &thetas_ref, vector<double> const &scores_ref, GP const &ll_gp)
{
  // calcul de la posterior-averaged log-likelihood error à partir d'un sample de référence
  double numerr_loglik = 0;
  double denomerr_loglik = 0;
  for (int i = 0; i < scores_ref.size(); i++)
  {
    double score_calcule = ll_gp.EvalMean(thetas_ref[i]);
    numerr_loglik += pow(scores_ref[i] - score_calcule, 2);
    denomerr_loglik += pow(scores_ref[i], 2);
  }
  double errrel = numerr_loglik / denomerr_loglik;
  return errrel;
}

double gsobol(VectorXd const &X, VectorXd const &theta)
{
  // fct gsobol en dimension 6
  double res = 1;
  for (int i = 0; i < 6; i++)
  {
    res *= (abs(4 * X(i) - 2) + theta(i)) / (1 + theta(i));
  }
  return res;
}

void Nodups(std::vector<VectorXd> &v)
{
  // supprimer les doublons dans un vector<VectorXd>
  auto end = v.end();
  for (auto it = v.begin(); it != end; ++it)
  {
    end = std::remove(it + 1, end, *it);
  }
  v.erase(end, v.end());
}

double optfunc_MAP(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  // fonction à optimiser pour trouver le MAP d'une densité.
  std::function<double(Eigen::VectorXd const &)> *p = static_cast<std::function<double(Eigen::VectorXd const &)> *>(data);
  auto f = *p;
  VectorXd arg = VtoVXD(x);
  return f(arg);
}

VectorXd find_MAP(std::function<double(Eigen::VectorXd const &)> f, Eigen::VectorXd const &lb_t, Eigen::VectorXd const &ub_t, double tolerance)
{
  // trouver le MAP d'une densité.
  VectorXd MAP(6);
  MAP << 0.55, 0.75, 0.3, 0.4, 0.6, 0.9;
  vector<double> x = VXDtoV(MAP);
  vector<double> lb = VXDtoV(lb_t);
  vector<double> ub = VXDtoV(ub_t);
  nlopt::opt local_opt(nlopt::LN_SBPLX, x.size());
  local_opt.set_max_objective(optfunc_MAP, &f);
  local_opt.set_ftol_rel(tolerance);
  local_opt.set_lower_bounds(lb);
  local_opt.set_upper_bounds(ub);
  double msup;
  local_opt.optimize(x, msup);
  MAP = VtoVXD(x);
  return MAP;
}

void Compute_map_mean_cov(vector<VectorXd> const &visited_steps, function<double(VectorXd const &)> score, VectorXd lb_t, VectorXd ub_t, string fnameMAP, double corlength)
{
  VectorXd map = find_MAP(score, lb_t, ub_t, 1e-4);
  VectorXd mean = VectorXd::Zero(visited_steps[0].size());
  for (VectorXd const &t : visited_steps)
  {
    mean += t;
  }
  mean /= visited_steps.size();
  MatrixXd var = MatrixXd::Zero(visited_steps[0].size(), visited_steps[0].size());
  for (VectorXd const &t : visited_steps)
  {
    var += t * t.transpose();
  }
  var = var / visited_steps.size() - mean * mean.transpose();
  ofstream omap(fnameMAP);
  omap << "#max cor length : " << corlength << endl;
  omap << "#map :" << endl;
  for (int i = 0; i < map.size(); i++)
  {
    omap << map(i) << " ";
  }
  omap << endl
       << "#mean :" << endl;
  for (int i = 0; i < map.size(); i++)
  {
    omap << mean(i) << " ";
  }
  omap << endl
       << "#cov :" << endl;
  for (int i = 0; i < var.rows(); i++)
  {
    for (int j = 0; j < var.cols(); j++)
    {
      omap << var(i, j) << " ";
    }
    omap << endl;
  }
}

int main(int argc, char **argv)
{
  default_random_engine generator(16031995);
  int dim_theta = 6;
  int dim_hpars = 7;
  // Bornes inf et sup des paramètres et hyperparamètres
  VectorXd lb_t = VectorXd::Zero(dim_theta); // paramètres dans [0,1]
  VectorXd ub_t = VectorXd::Ones(dim_theta); // paramètres dans [0,1]
  /*vecteur d'hyperparamètres : l'indice 0 contient sigma,
  les longueur de corrélation ensuite.
  on donne les bornes en échelle réelle, puis tout est passé en log.*/
  VectorXd lb_hpars(dim_hpars);
  lb_hpars(0) = 1e-3;
  lb_hpars(1) = 0.01;
  lb_hpars(2) = 0.01;
  lb_hpars(3) = 0.01;
  lb_hpars(4) = 0.01;
  lb_hpars(5) = 0.01;
  lb_hpars(6) = 0.01;
  VectorXd ub_hpars(dim_hpars);
  ub_hpars(0) = 5;
  ub_hpars(1) = 3;
  ub_hpars(2) = 3;
  ub_hpars(3) = 3;
  ub_hpars(4) = 3;
  ub_hpars(5) = 3;
  ub_hpars(6) = 3;
  // conversion en log.
  lb_hpars.array() = lb_hpars.array().log();
  ub_hpars.array() = ub_hpars.array().log();
  VectorXd hpars_z_guess = 0.5 * (lb_hpars + ub_hpars);

  // tirage des observations.
  VectorXd theta_true(dim_theta); // valeur de theta utilisée pour générer les observations
  theta_true << 0.55, 0.75, 0.3, 0.4, 0.6, 0.9;
  double noise = 5e-2;
  int ndata = 20;
  // Les observations sont placées selon un QMC.
  DoE doe_obs(lb_t, ub_t, ndata, 1); // DoE créé pour la variable x. On utilise le fait que x et theta sont chacune dans [0,1]^6.
  vector<VectorXd> Xlocs = doe_obs.GetGrid();
  VectorXd data(ndata);
  for (int i = 0; i < ndata; i++)
  {
    double y = gsobol(Xlocs[i], theta_true) + noise * distN(generator);
    data(i) = y;
  }
  string expfilename = "results/obs.gnu";
  WriteObs(Xlocs, data, expfilename);

  /* Définition de fonctions lambda pour le modèle f et la moyenne a priori de z.*/
  auto lambda_model = [](vector<VectorXd> const &X, VectorXd const &theta)
  {
    // prédictions du modèle.
    VectorXd pred(X.size());
    for (int j = 0; j < X.size(); j++)
    {
      pred(j) = gsobol(X[j], theta);
    }
    return pred;
  };

  auto lambda_priormean = [](vector<VectorXd> const &X, VectorXd const &hpars)
  {
    VectorXd b = VectorXd::Zero(X.size());
    return b;
  };

  DoE doe_init(lb_t, ub_t, 100, 1);
  Density MainDensity(doe_init);
  MainDensity.SetFModel(lambda_model);
  MainDensity.SetZKernel(Kernel_Z_Matern52);
  MainDensity.SetZPriorMean(lambda_priormean);
  MainDensity.SetHparsBounds(lb_hpars, ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetObservations(Xlocs, data);
  MainDensity.SetFixedOutputerr(log(noise)); // outputerr connue à sa véritable valeur

  /*Paramètres MCMC*/
  VectorXd X_init_mcmc = 0.5 * VectorXd::Ones(dim_theta);
  MatrixXd COV_init = MatrixXd::Identity(dim_theta, dim_theta);
  for (int i = 0; i < dim_theta; i++)
  {
    COV_init(i, i) = pow(0.05, 2);
  }

  /*Bornes pour les hyperparamètres des méthodes HP et LL.*/
  MatrixXd Bounds_hpars_HGPs(2, dim_hpars + 1); // 5 hpars HGPs. std, noise, 3 longueurs de corrélation. à changer en fct de theta.
  Bounds_hpars_HGPs(0, 0) = 0.1;                // std
  Bounds_hpars_HGPs(1, 0) = 3;
  for (int i = 1; i < dim_hpars + 1; i++)
  {
    Bounds_hpars_HGPs(0, i) = 1e-1; // lcors, sauf i=1 qui est std noise (corrigé juste après)
    Bounds_hpars_HGPs(1, i) = 3;
  }
  Bounds_hpars_HGPs(0, 2) = 1e-5; // stdnoise
  Bounds_hpars_HGPs(1, 2) = 1e-1;
  VectorXd Hpars_guess_HGPs = 0.5 * (Bounds_hpars_HGPs.row(0) + Bounds_hpars_HGPs.row(1)).transpose();

  MatrixXd Bounds_hpars_llgp(2, dim_hpars + 1); // 5 hpars llgp. std, noise, 3 longueurs de corrélation. à changer en fct de theta.
  Bounds_hpars_llgp(0, 0) = 2;                  // std
  Bounds_hpars_llgp(1, 0) = 50;
  for (int i = 1; i < dim_hpars + 1; i++)
  {
    Bounds_hpars_llgp(0, i) = 1e-1; // lcors, sauf i=1 qui est std noise (corrigé juste après)
    Bounds_hpars_llgp(1, i) = 4;
  }
  Bounds_hpars_llgp(0, 2) = 1e-3; // stdnoise
  Bounds_hpars_llgp(1, 2) = 1e-1;
  VectorXd Hpars_guess_llgp = 0.5 * (Bounds_hpars_llgp.row(0) + Bounds_hpars_llgp.row(1)).transpose();

  vector<VectorXd> guess_hpars_hgps; // on y met les meilleurs hpars, trouvés à partir d'un calcul fin.
  for (int i = 0; i < dim_hpars; i++)
  {
    guess_hpars_hgps.push_back(Hpars_guess_HGPs);
  }

  /* paramètres MCMC et des optimisations */
  int nombre_steps_mcmc = 1e5;
  int nombre_samples_collected = 500; // 1 sample tous les 200.
  int nautocor = 500;

  double ftol_rel_fmp = 1e-7; // ne surtout pas changer pour 20 obs.
  double ftol_rel_hgps = 1e-4;
  double ftol_rel_gp_llsurrogate = 1e-4;

  /* Lecture de l'échantillon de référence pour le calcul de la posterior-averaged log-likelihood error.*/
  vector<VectorXd> thetas_post;
  vector<VectorXd> hpars_post;
  vector<double> scores_post;
  {
    string fname_post = "reference_sample.gnu";
    thetas_post = ReadVector(fname_post); // on prend l'ensemble des points du sample.
    DensityOpt Dopt(MainDensity);
    for (int i = 0; i < thetas_post.size(); i++)
    {
      VectorXd hpars = Dopt.HparsOpt(thetas_post[i], hpars_z_guess, ftol_rel_fmp);
      hpars_post.push_back(hpars);
      scores_post.push_back(Dopt.loglikelihood_theta(thetas_post[i], hpars));
    }
  }

  // Calibration phase.

  /* HP-AS calibration*/
  {
    cout << "Begin HP-AS calibration" << endl;
    string foldname = "hpas";
    int ndraws = 100;                       // nombre de tirage des hGPs pour calcul des poids de resampling
    int npts_init = 100;                    // nombre de points du DoE initial
    int nmax = 400;                         // nombre de points maximum dans le training set des HGPs.
    int npts_per_iter = 100;                // nombre de points rajoutés dans le training set, par itération (n_s dans le papier 2)
    int nsamples_mcmc = 20 * npts_per_iter; // nombre de points candidats (n_c dans le papier 2)
    int nsteps_mcmc = 50 * nsamples_mcmc;   // longueur des chaînes pour l'adaptive sampling
    DoE doe_init(lb_t, ub_t, npts_init, generator);
    DensityOpt Dopt(MainDensity);
    vector<VectorXd> thetas_training;
    vector<VectorXd> hpars_training;
    auto tinit = doe_init.GetGrid();
    for (const auto theta : doe_init.GetGrid())
    {
      //construction du training set initial
      VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, ftol_rel_fmp);
      thetas_training.push_back(theta);
      hpars_training.push_back(hpars);
    }
    auto get_score = [&Dopt](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dopt.loglikelihood_theta(theta, hpars)+logprior_pars(theta);
    };
    auto get_hpars_and_var = [&Dopt](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v(1);
      return make_pair(h, v);
    };
    auto add_points = [&Dopt, nsteps_mcmc, &X_init_mcmc, &COV_init, &generator, hpars_z_guess, &thetas_training, &hpars_training, ftol_rel_fmp, ndraws, nsamples_mcmc, npts_per_iter](string foldname)
    {
      // tirage de points candidats par MCMC, weighted resampling, et ajout des points au training set.
      auto get_hpars_opti = [&Dopt, &hpars_z_guess](VectorXd const &X)
      {
        vector<VectorXd> p(1);
        p[0] = Dopt.EvaluateHparOpt(X);
        return p;
      };
      auto compute_score_opti = [&Dopt](vector<VectorXd> h, VectorXd const &X)
      {
        double ll1 = Dopt.loglikelihood_theta(X, h[0]);
        return ll1+logprior_pars(X);
      };
      auto in_bounds = [&Dopt](VectorXd const &X)
      {
        return Dopt.in_bounds_pars(X);
      };
      vector<VectorXd> allsteps = Run_MCMC(nsteps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
      vector<VectorXd> candidate_thetas;
      for (int i = 0; i < nsamples_mcmc; i++)
      {
        //thinning de l'échantillon pour constituer les points candidats
        candidate_thetas.push_back(allsteps[i * (allsteps.size() / nsamples_mcmc)]);
      }
      Nodups(candidate_thetas); //enlever les potentiels doublons dans les points candidats.
      vector<VectorXd> selected_thetas(npts_per_iter);
      /*Weighted resampling*/
      vector<double> weights(candidate_thetas.size());
      vector<vector<VectorXd>> draws = Dopt.SampleHparsOpt(candidate_thetas, ndraws, generator);
      for (int j = 0; j < weights.size(); j++)
      {
        //calcul des poids
        vector<double> scores;
        for (int k = 0; k < ndraws; k++)
        {
          scores.push_back(Dopt.loglikelihood_theta(candidate_thetas[j], draws[j][k]));
        }
        double mean = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        double sq_sum = std::inner_product(scores.begin(), scores.end(), scores.begin(), 0.0);
        double var = sq_sum / scores.size() - mean * mean;
        weights[j] = var;
      }
      for (int i = 0; i < npts_per_iter; i++)
      {
        //tirage des points parmi les candidats, et rajout dans le training set
        std::discrete_distribution<int> distribution(weights.begin(), weights.end());
        int drawn = distribution(generator);
        weights[drawn] = 0;
        selected_thetas[i] = candidate_thetas[drawn];
        VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, ftol_rel_fmp);
        thetas_training.push_back(selected_thetas[i]);
        hpars_training.push_back(hpars);
      }
      cout.clear();
    };

    auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
    {
      // évaluation de la posterior-averaged log-likelihood error.
      double score = evaluate_hgp_surrogate(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
      ofile << thetas_training.size() << " " << score << endl;
    };
    ofstream ofile("results/" + foldname + "/score.gnu");
    vector<VectorXd> hpars_opt_hgps;
    /* Construction des HGPs avec le training set initial*/
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      auto t1 = chrono::steady_clock::now();
      Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32, DKernel_GP_Matern32);
      Dopt.OptimizeHGPs(Bounds_hpars_HGPs, guess_hpars_hgps, ftol_rel_hgps);
      hpars_opt_hgps = Dopt.GetHparsHGPs();
      write_performance_hgps(ofile);
    
    while (thetas_training.size() < nmax)
    {
      /* Ajout itératif de points dans le training set */
      add_points(foldname);
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32, DKernel_GP_Matern32);
      Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, ftol_rel_hgps);
      hpars_opt_hgps = Dopt.GetHparsHGPs();
      write_performance_hgps(ofile);
    }
    /* Calibration */
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
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      // thinning de l'échantillon MCMC
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples[i] = get_hpars_opti(samples[i])[0];
    }
    ofstream fsamp("results/" + foldname + "/samples.gnu");
    WriteVectors(samples, hparsofsamples, fsamp); //écriture de l'échantillon et des hpars correspondants.
    double corlength = Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");
    auto score = [&get_hpars_opti, &compute_score_opti](VectorXd const &X)
    {
      return compute_score_opti(get_hpars_opti(X), X);
    };
    string fnameMAP = "results/" + foldname + "/map.gnu";
    Compute_map_mean_cov(visited_steps, score, lb_t, ub_t, fnameMAP, corlength); //calcul du MAP, postmean, postcov
  }

  /*HP-LHS calibration*/
  {
    cout << "Begin HP-LHS calibration" << endl;
    string foldname = "hplhs";
    // paramètres de l'algorithme
    int nmax = 400;
    // Construction d'un DoE initial.
    DoE doe_init(lb_t, ub_t, nmax, generator);
    DensityOpt Dopt(MainDensity);
    vector<VectorXd> thetas_training;
    vector<VectorXd> hpars_training;
    auto tinit = doe_init.GetGrid();
    for (const auto theta : doe_init.GetGrid())
    {
      VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, ftol_rel_fmp);
      thetas_training.push_back(theta);
      hpars_training.push_back(hpars);
    }
    // lambda fcts pour l'évaluation de la qualité des surrogates.
    auto get_score = [&Dopt](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dopt.loglikelihood_theta(theta, hpars)+logprior_pars(theta);
    };

    auto get_hpars_and_var = [&Dopt](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v(1);
      return make_pair(h, v);
    };
    auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      double score = evaluate_hgp_surrogate(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
      ofile << thetas_training.size() << " " << score << endl;
    };

    // début de l'algorithme
    ofstream ofile("results/" + foldname + "/score.gnu");
    // construction des hGPs
    cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
    Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32, DKernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, guess_hpars_hgps, ftol_rel_hgps);
    write_performance_hgps(ofile);
    // calibration
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
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples[i] = get_hpars_opti(samples[i])[0];
    }
    ofstream fnamesamp("results/" + foldname + "/samples.gnu");
    WriteVectors(samples, hparsofsamples, fnamesamp);
    double corlength = Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");
    auto score = [&get_hpars_opti, &compute_score_opti](VectorXd const &X)
    {
      return compute_score_opti(get_hpars_opti(X), X);
    };
    string fnameMAP = "results/" + foldname + "/map.gnu";
    Compute_map_mean_cov(visited_steps, score, lb_t, ub_t, fnameMAP, corlength);
  }

  /*LL-LHS calibration*/
  {
    cout << "begin LL-LHS calibration" << endl;
    string foldname = "lllhs";
    GP ll_surrogate(Kernel_GP_Matern32);
    ll_surrogate.SetDKernel(DKernel_GP_Matern32);
    int nmax = 400; // nombre de points pour construire le surrogate LL.
    DoE doe_init(lb_t, ub_t, nmax, generator);
    DensityOpt Dopt(MainDensity);
    vector<VectorXd> thetas_training;
    vector<VectorXd> hpars_training;
    auto tinit = doe_init.GetGrid();
    /* Création du training set*/
    for (const auto theta : tinit)
    {
      VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, ftol_rel_fmp);
      thetas_training.push_back(theta);
      hpars_training.push_back(hpars);
    }
    auto write_performance_gp_ll = [&thetas_post, &scores_post, &thetas_training](const GP &gp, ofstream &ofile)
    {
      //évaluer la posterior-averaged log-likelihood error
      double score = evaluate_ll_surrogate(thetas_post, scores_post, gp);
      ofile << thetas_training.size() << " " << score << endl;
    };
    ofstream ofile("results/" + foldname + "/score.gnu");
    vector<DATA> data_ll;
    for (int i = 0; i < thetas_training.size(); i++)
    {
      // mise en forme du training set comme vector<DATA> pour coincider avec la classe GP.
      DATA d;
      d.SetX(thetas_training[i]);
      d.SetValue(Dopt.loglikelihood_theta(thetas_training[i], hpars_training[i])+logprior_pars(thetas_training[i]));
      data_ll.push_back(d);
    }
    ll_surrogate.SetData(data_ll);
    OptimizeGPBis(ll_surrogate, Hpars_guess_llgp, Bounds_hpars_llgp.row(0), Bounds_hpars_llgp.row(1), ftol_rel_gp_llsurrogate);
    write_performance_gp_ll(ll_surrogate, ofile);
    /* calibration */
    auto in_bounds = [&Dopt](VectorXd const &X)
    {
      return Dopt.in_bounds_pars(X);
    };
    auto get_hpars_opti = [](VectorXd const &X)
    {
      // non utilisé dans la technique LL
      vector<VectorXd> p(1);
      return p;
    };
    auto compute_score_opti = [&ll_surrogate](vector<VectorXd> const &p, VectorXd const &X)
    {
      // calcul de la logpost par le surrogate.
      double d = ll_surrogate.EvalMean(X);
      return d;
    };
    cout << "Begin mcmc" << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      // thinning de l'échantillon MCMC
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
    }
    ofstream fnamesamp("results/" + foldname + "/samples.gnu");
    WriteVector(samples, fnamesamp); // écriture de l'échantillon
    double corlength = Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");
    /* Calcul du MAP, postmean, postcov.*/
    auto score = [&get_hpars_opti, &compute_score_opti](VectorXd const &X)
    {
      return compute_score_opti(get_hpars_opti(X), X);
    };
    string fnameMAP = "results/" + foldname + "/map.gnu";
    Compute_map_mean_cov(visited_steps, score, lb_t, ub_t, fnameMAP, corlength);
  }

  /*LL-AS calibration*/
  {
    cout << "Begin LL-AS calibration" << endl;
    string foldname = "llas";
    GP ll_surrogate(Kernel_GP_Matern32);
    ll_surrogate.SetDKernel(DKernel_GP_Matern32);
    int npts_init = 100;
    int nmax = 400;
    int npts_per_iter = 100;
    int nsamples_mcmc = 20 * npts_per_iter;
    int nsteps_mcmc = 50 * nsamples_mcmc;
    DoE doe_init(lb_t, ub_t, npts_init, generator);
    DensityOpt Dopt(MainDensity);
    vector<VectorXd> thetas_training;
    vector<VectorXd> hpars_training;
    for (const auto theta : doe_init.GetGrid())
    {
      VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, ftol_rel_fmp);
      thetas_training.push_back(theta);
      hpars_training.push_back(hpars);
    }
    auto add_points = [&Dopt, &generator, &lb_t, &ub_t, &hpars_z_guess, &ftol_rel_fmp, nsteps_mcmc, &ll_surrogate, &X_init_mcmc, COV_init, nsamples_mcmc, npts_per_iter](vector<VectorXd> &thetas_training, vector<VectorXd> &hpars_training, string foldname)
    {
      auto get_hpars_opti = [](VectorXd const &X)
      {
        vector<VectorXd> p(1);
        return p;
      };
      auto compute_score_opti = [&ll_surrogate](vector<VectorXd> h, VectorXd const &X)
      {
        double ll1 = ll_surrogate.EvalMean(X);
        return ll1;
      };
      auto in_bounds = [&Dopt](VectorXd const &X)
      {
        return Dopt.in_bounds_pars(X);
      };
      vector<VectorXd> allsteps = Run_MCMC(nsteps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
      vector<VectorXd> candidate_thetas;
      for (int i = 0; i < nsamples_mcmc; i++)
      {
        candidate_thetas.push_back(allsteps[i * (allsteps.size() / nsamples_mcmc)]);
      }
      Nodups(candidate_thetas);
      vector<VectorXd> selected_thetas(npts_per_iter);
      vector<double> weights(candidate_thetas.size());
      auto hparsurr = ll_surrogate.GetPar();
      cout.setstate(ios_base::failbit);
      auto begin = chrono::steady_clock::now();
      for (int j = 0; j < weights.size(); j++)
      {
        weights[j] = ll_surrogate.Eval(candidate_thetas[j])(1);
      }
      for (int i = 0; i < npts_per_iter; i++)
      {
        std::discrete_distribution<int> distribution(weights.begin(), weights.end());
        int drawn = distribution(generator);
        weights[drawn] = 0;
        selected_thetas[i] = candidate_thetas[drawn];
        VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, ftol_rel_fmp);
        thetas_training.push_back(selected_thetas[i]);
        hpars_training.push_back(hpars);
      }
      cout.clear();
    };

    auto write_performance_gp_ll = [&thetas_post, &scores_post, &thetas_training](const GP &gp, ofstream &ofile)
    {
      //évaluer la posterior-averaged log-likelihood error
      double score = evaluate_ll_surrogate(thetas_post, scores_post, gp);
      ofile << thetas_training.size() << " " << score << endl;
    };
    ofstream ofile("results/" + foldname + "/score.gnu");
    vector<DATA> data_ll;
    vector<VectorXd> hpars_gp_llsurr;

    for (int i = 0; i < thetas_training.size(); i++)
    {
      DATA d;
      d.SetX(thetas_training[i]);
      d.SetValue(Dopt.loglikelihood_theta(thetas_training[i], hpars_training[i])+logprior_pars(thetas_training[i]));
      data_ll.push_back(d);
    }
    /* Construction avec le training set initial */
    ll_surrogate.SetData(data_ll);
    OptimizeGPBis(ll_surrogate, Hpars_guess_llgp, Bounds_hpars_llgp.row(0), Bounds_hpars_llgp.row(1), ftol_rel_gp_llsurrogate);
    cout << "hparsmax pour surrogate ll : " << ll_surrogate.GetPar().transpose() << endl;
    hpars_gp_llsurr.clear();
    hpars_gp_llsurr.push_back(ll_surrogate.GetPar());
    write_performance_gp_ll(ll_surrogate, ofile);
    /* Ajout itératif de points au training set*/
    while (thetas_training.size() < nmax)
    {
      int previoussize = thetas_training.size();
      add_points(thetas_training, hpars_training, foldname);
      for (int i = previoussize; i < thetas_training.size(); i++)
      {
        DATA d;
        d.SetX(thetas_training[i]);
        d.SetValue(Dopt.loglikelihood_theta(thetas_training[i], hpars_training[i])+logprior_pars(thetas_training[i]));
        data_ll.push_back(d);
      }
      ll_surrogate.SetData(data_ll);
      OptimizeGPBis(ll_surrogate, hpars_gp_llsurr[0], Bounds_hpars_llgp.row(0), Bounds_hpars_llgp.row(1), ftol_rel_gp_llsurrogate);
      cout << "hparsmax ll pour surrogate ll : " << ll_surrogate.GetPar().transpose() << endl;
      hpars_gp_llsurr.clear();
      hpars_gp_llsurr.push_back(ll_surrogate.GetPar());
      write_performance_gp_ll(ll_surrogate, ofile);
    }
    /* Calibration */
    auto in_bounds = [&Dopt](VectorXd const &X)
    {
      return Dopt.in_bounds_pars(X);
    };
    auto get_hpars_opti = [](VectorXd const &X)
    {
      vector<VectorXd> p(1);
      return p;
    };
    auto compute_score_opti = [&ll_surrogate](vector<VectorXd> const &p, VectorXd const &X)
    {
      double d = ll_surrogate.EvalMean(X);
      return d;
    };
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
    }
    ofstream fnamesamp("results/" + foldname + "/samples.gnu");
    WriteVector(samples, fnamesamp);
    double corlength = Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");
    auto score = [&get_hpars_opti, &compute_score_opti](VectorXd const &X)
    {
      return compute_score_opti(get_hpars_opti(X), X);
    };
    string fnameMAP = "results/" + foldname + "/map.gnu";
    Compute_map_mean_cov(visited_steps, score, lb_t, ub_t, fnameMAP, corlength);
  }

  /* Full FMP calibration*/
  {
    string foldname = "fmp";
    DensityOpt Dopt(MainDensity);
    auto in_bounds = [&Dopt](VectorXd const &X)
    {
      return Dopt.in_bounds_pars(X);
    };
    auto get_hpars_opti = [Dopt, &hpars_z_guess, &ftol_rel_fmp](VectorXd const &X)
    {
      vector<VectorXd> p(1);
      p[0] = Dopt.HparsOpt(X, hpars_z_guess, ftol_rel_fmp);
      return p;
    };
    auto compute_score_opti = [&Dopt](vector<VectorXd> const &p, VectorXd const &X)
    {
      double d = Dopt.loglikelihood_theta(X, p[0])+logprior_pars(X);
      return d;
    };
    cout << "Beginning FMP calibration..." << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
    }
    ofstream fnamesamp("results/" + foldname + "/samples.gnu");
    WriteVector(samples, fnamesamp);
    double corlength = Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");

    // calcul MAP
    auto score = [&get_hpars_opti, &compute_score_opti](VectorXd const &X)
    {
      return compute_score_opti(get_hpars_opti(X), X);
    };
    string fnameMAP = "results/" + foldname + "/map.gnu";
    Compute_map_mean_cov(visited_steps, score, lb_t, ub_t, fnameMAP, corlength);
  }

  exit(0);
}
