// Dans ce fichier : on fait un exemple de construction de l'algorithme surrogates opti.
// On commence par le toy problem de l'article 1.
// je veux comparer Opti et Opti fait avec des surrogates.
// je veux aussi trouver une mesure propre de la qualité d'un surrogate. Normalisation qqpart ? norme a priori ? ou plutôt a posteriori ?
// je veux mettre en place les méthodes de construction alternatives (choix progressif parmi les samples a posteriori, et construction QMC).

// il me faut à la fin de belles courbes d'erreur qui convergent. Et pourquoi pas les figures de points sélectionnés progressivement.
// premier test de convergence pour ma mesure d'erreur : que ça tende bien vers 0 lorsque le nombre de points augmente. faire des grids avec peu de pts, bcp de points...

// je n'ai pas réussi à faire un bon exemple. Je passe sur DEBORA à la place.
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

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // squared exponential. x et y sont de dimension 3.
  //échelle log pour les hpars.
  // sigma en position 0, les lcors ensuite.
  double d1 = abs(x(0) - y(0)) / exp(hpar(1));
  double d2 = abs(x(1) - y(1)) / exp(hpar(2));
  double d3 = abs(x(2) - y(2)) / exp(hpar(3));
  double d4 = abs(x(3) - y(3)) / exp(hpar(4));
  double d5 = abs(x(4) - y(4)) / exp(hpar(5));
  double d6 = abs(x(5) - y(5)) / exp(hpar(6));
  return pow(exp(hpar(0)), 2) * exp(-0.5 * (pow(d1, 2) + pow(d2, 2) + pow(d3, 2) + pow(d4, 2) + pow(d5, 2) + pow(d6, 2)));
}

double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // squared exponential. x et y sont de dimension 3.
  //échelle log pour les hpars.
  // sigma en position 0, les lcors ensuite.
  double d1 = abs(x(0) - y(0)) / exp(hpar(1));
  double d2 = abs(x(1) - y(1)) / exp(hpar(2));
  double d3 = abs(x(2) - y(2)) / exp(hpar(3));
  double d4 = abs(x(3) - y(3)) / exp(hpar(4));
  double d5 = abs(x(4) - y(4)) / exp(hpar(5));
  double d6 = abs(x(5) - y(5)) / exp(hpar(6));
  double cor = pow(exp(hpar(0)), 2);
  cor *= (1 + sqrt(3) * d1) * exp(-sqrt(3) * d1); // x1
  cor *= (1 + sqrt(3) * d2) * exp(-sqrt(3) * d2); // x1
  cor *= (1 + sqrt(3) * d3) * exp(-sqrt(3) * d3); // x1
  cor *= (1 + sqrt(3) * d4) * exp(-sqrt(3) * d4); // x1
  cor *= (1 + sqrt(3) * d5) * exp(-sqrt(3) * d5); // x1
  cor *= (1 + sqrt(3) * d6) * exp(-sqrt(3) * d6); // x1
  return cor;
}

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // squared exponential. x et y sont de dimension 3.
  //échelle log pour les hpars.
  // sigma en position 0, les lcors ensuite.
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
  // noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
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
  else
    if(p <= 7)
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


double Kernel_GP_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double d1 = abs(x(0) - y(0)) / (hpar(1));
  double d2 = abs(x(1) - y(1)) / (hpar(3));
  double d3 = abs(x(2) - y(2)) / (hpar(4));
  double d4 = abs(x(3) - y(3)) / (hpar(5));
  double d5 = abs(x(4) - y(4)) / (hpar(6));
  double d6 = abs(x(5) - y(5)) / (hpar(7));
  return pow((hpar(0)), 2) * exp(-0.5 * (pow(d1, 2) + pow(d2, 2) + pow(d3, 2) + pow(d4, 2) + pow(d5, 2) + pow(d6, 2)));
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
  // on met loginvgamma pour toutes les longueurs de corrélation. Mais quelle moyenne et quelle variance ? J'essaye déjà avec des paramètres communs à toutes les lcor.
  // seul le paramètre 0 n'est pas une lcor.
  double a = 1; // voir wikipédia pour le choix de cee paramètres.
  double b = 1;
  double res = 0;
  for (int i = 1; i < hpars.size(); i++)
  {
    res += loginvgammaPdf(exp(hpars(i)), a, b);
  }
  return res;
}

double logprior_pars(VectorXd const &pars)
{
  return 0;
}

double logprior_parsnormgrand(VectorXd const &pars)
{
  // prior gaussien large sur les paramètres. ils seront dans l'espace (0,1.)
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
  // lecture d'un fichier d'hpars koh, qui est écrit de la forme n V, avec n le numéro du cas et V le vecteur d'hyperparamètres.
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
  // lecture d'un fichier d'hpars optimaux., qui est écrit de la forme n V, avec n le numéro du cas et V le vecteur d'hyperparamètres.
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
  // inverts the inner and outer vector.
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

VectorXd evaluate_hgp_surrogate(vector<VectorXd> const &thetas_ref, vector<VectorXd> const &hpars_ref, vector<double> const &scores_ref, function<pair<VectorXd, VectorXd>(VectorXd const &)> const &get_meanpred, function<double(VectorXd const &, VectorXd const &)> const &get_score)
{
  // toutes les erreurs au carré.
  if (!(thetas_ref.size() == hpars_ref.size()))
  {
    cerr << "erreur : différentes tailles evaluate_surrogate !" << endl;
    exit(0);
  }
  vector<VectorXd> hpars_calcules;
  VectorXd errnums = VectorXd::Zero(hpars_ref[0].size());
  VectorXd errdenoms = VectorXd::Zero(hpars_ref[0].size());
  for (int i = 0; i < thetas_ref.size(); i++)
  {
    VectorXd hpcalc = get_meanpred(thetas_ref[i]).first;
    hpars_calcules.push_back(hpcalc);
    VectorXd hpref = hpars_ref[i];
    VectorXd n = (hpref - hpcalc).array().square();
    errnums += n;
    VectorXd d = hpref.array().square();
    errdenoms += d;
  }
  VectorXd err_hpars = errnums.cwiseQuotient(errdenoms);
  double numerr_loglik = 0;
  double denomerr_loglik = 0;
  for (int i = 0; i < scores_ref.size(); i++)
  {
    double score_calcule = get_score(thetas_ref[i], hpars_calcules[i]);
    numerr_loglik += pow(scores_ref[i] - score_calcule, 2);
    denomerr_loglik += pow(scores_ref[i], 2);
  }
  double err_loglik = numerr_loglik / denomerr_loglik;
  VectorXd errtot(err_hpars.size() + 1);
  errtot.head(err_hpars.size()) = err_hpars; // droit de faire ça ?
  errtot(err_hpars.size()) = err_loglik;
  return errtot;
}

pair<double, double> evaluate_ll_surrogate(vector<VectorXd> const &thetas_ref, vector<double> const &scores_ref, GP const &ll_gp)
{
  // On va juste faire l'erreur L2 relative.
  // dernière composante : l'erreur L2 sur la fct likelihood elle-même. Plus interprétable.
  double numerr_loglik = 0;
  double denomerr_loglik = 0;
  for (int i = 0; i < scores_ref.size(); i++)
  {
    double score_calcule = ll_gp.EvalMean(thetas_ref[i]);
    numerr_loglik += pow(scores_ref[i] - score_calcule, 2);
    denomerr_loglik += pow(scores_ref[i], 2);
  }
  double errrel = numerr_loglik / denomerr_loglik;
  double errabs = numerr_loglik / scores_ref.size();
  return make_pair(errrel, errabs);
}

double gsobol(VectorXd const &X, VectorXd const &theta)
{
  // fct gsobol en dimension 6 !!
  double res = 1;
  for (int i = 0; i < 6; i++)
  {
    res *= (abs(4 * X(i) - 2) + theta(i)) / (1 + theta(i));
  }
  return res;
}

void Nodups(std::vector<VectorXd> &v)
{
  auto end = v.end();
  for (auto it = v.begin(); it != end; ++it)
  {
    end = std::remove(it + 1, end, *it);
  }
  v.erase(end, v.end());
}

int main(int argc, char **argv)
{
  default_random_engine generator(123456);

  /*Paramètres de simulation*/

  // Bornes sup des paramètres et hyperparamètres
  int dim_theta = 6;
  int dim_hpars = 7;

  VectorXd lb_t = VectorXd::Zero(dim_theta);
  VectorXd ub_t = VectorXd::Ones(dim_theta);
  VectorXd lb_hpars(dim_hpars);
  lb_hpars(0) = 1e-7; // pas 0 car échelle log
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

  // création des observations.
  VectorXd theta_true(dim_theta); // valeur de theta utilisée pour générer les observations
  theta_true << 0.55, 0.75, 0.3, 0.4, 0.6, 0.9;
  double noise = 5e-2;
  int ndata = 50;
  // grid des observations
  DoE doe_obs(lb_t, ub_t, ndata, 600);
  vector<VectorXd> Xlocs = doe_obs.GetGrid();
  VectorXd data(ndata);
  for (int i = 0; i < ndata; i++)
  {
    double y = gsobol(Xlocs[i], theta_true) + noise * distN(generator);
    data(i) = y;
  }
  string expfilename = "results/obs.gnu";
  WriteObs(Xlocs, data, expfilename);
  // création des points de prédiction
  int pred_size = 80;
  DoE doe_pred(lb_t, ub_t, pred_size, 900);
  vector<VectorXd> XPREDS = doe_pred.GetGrid();

  // building initial DoE
  DoE doe_init(lb_t, ub_t, 100, 1); // doe halton de 100 points
  doe_init.WriteGrid("results/grid.gnu");

  auto lambda_priormean = [](vector<VectorXd> const &X, VectorXd const &hpars)
  {
    VectorXd b = VectorXd::Zero(X.size());
    return b;
  };
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

  Density MainDensity(doe_init);
  MainDensity.SetModel(lambda_model);
  MainDensity.SetKernel(Kernel_Z_Matern52);
  MainDensity.SetHparsBounds(lb_hpars, ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars); // invgamma
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(lambda_priormean);
  MainDensity.SetObservations(Xlocs, data);
  MainDensity.SetOutputerr(false, noise, 2); // outputerr connue à sa véritable valeur
  // MainDensity.SetInputerr(false,0.0,0,VectorXd::Zero(Xlocs.size()));

  // créer un DoE de test.
  vector<VectorXd> thetas_test;
  vector<VectorXd> hpars_test;
  {
    DensityOpt Dopt(MainDensity);
    int ngrid = 100;
    double timeopt = 1e-1;
    DoE doe1(lb_t, ub_t, ngrid, 1);
    thetas_test = doe1.GetGrid();
    for (int i = 0; i < ngrid; i++)
    {
      VectorXd hpar1 = Dopt.HparsOpt(thetas_test[i], hpars_z_guess, timeopt);
      hpars_test.push_back(hpar1);
    }
  }
  //écriture du test set.
  string fname = "results/hgps/testset.gnu";
  WriteVectors(thetas_test, hpars_test, fname);

  VectorXd X_init_mcmc = 0.5 * VectorXd::Ones(dim_theta);
  MatrixXd COV_init = MatrixXd::Identity(dim_theta, dim_theta);
  for (int i = 0; i < dim_theta; i++)
  {
    COV_init(i, i) = pow(0.05, 2);
  }
  cout << "COV_init : " << endl
       << COV_init << endl;

  MatrixXd Bounds_hpars_HGPs(2, dim_hpars + 1); // 5 hpars HGPs. std, noise, 3 longueurs de corrélation. à changer en fct de theta.
  Bounds_hpars_HGPs(0, 0) = 0.1;                // std
  Bounds_hpars_HGPs(1, 0) = 3;
  for (int i = 1; i < dim_hpars + 1; i++)
  {
    Bounds_hpars_HGPs(0, i) = 1e-1; // lcors, sauf i=1 qui est std noise (corrigé juste après)
    Bounds_hpars_HGPs(1, i) = 5;
  }
  Bounds_hpars_HGPs(0, 2) = 1e-5; // stdnoise
  Bounds_hpars_HGPs(1, 2) = 1e-1;
  VectorXd Hpars_guess_HGPs = 0.5 * (Bounds_hpars_HGPs.row(0) + Bounds_hpars_HGPs.row(1)).transpose();

  MatrixXd Bounds_hpars_llgp(2, dim_hpars + 1); // 5 hpars llgp. std, noise, 3 longueurs de corrélation. à changer en fct de theta.
  Bounds_hpars_llgp(0, 0) = 5;                  // std
  Bounds_hpars_llgp(1, 0) = 50;
  for (int i = 1; i < dim_hpars + 1; i++)
  {
    Bounds_hpars_llgp(0, i) = 1e-8; // lcors, sauf i=1 qui est std noise (corrigé juste après)
    Bounds_hpars_llgp(1, i) = 20;
  }
  Bounds_hpars_llgp(0, 2) = 1e-8; // stdnoise
  Bounds_hpars_llgp(1, 2) = 1e-5;
  VectorXd Hpars_guess_llgp = 0.5 * (Bounds_hpars_llgp.row(0) + Bounds_hpars_llgp.row(1)).transpose();

  vector<VectorXd> best_hpars_hgps; //on y met les meilleurs hpars, troucés à partir d'un calcul fin.
  {
    VectorXd h1(8),h2(8),h3(8),h4(8),h5(8),h6(8),h7(8);
    h1 << 0.65,0.22,9e-6,0.64,0.25,0.22,0.29,0.35;
    h2 << 0.42,0.21,2e-8,0.43,0.19,0.26,0.29,0.29;
    h3 << 0.48,0.21,1e-5,0.43,0.28,0.28,0.56,0.59;
    h4 << 0.44,0.27,4.5e-7,1.11,0.23,0.28,0.49,0.34;
    h5 << 0.49,0.21,1.7e-7,0.44,0.25,0.23,0.38,0.35;
    h6 << 0.53,0.23,1.2e-8,0.59,0.29,0.32,0.40,0.33;
    h7 << 0.46,0.23,1e-5,0.71,0.25,0.27,0.40,0.39;
    best_hpars_hgps.push_back(h1);
    best_hpars_hgps.push_back(h2);
    best_hpars_hgps.push_back(h3);
    best_hpars_hgps.push_back(h4);
    best_hpars_hgps.push_back(h5);
    best_hpars_hgps.push_back(h6);
    best_hpars_hgps.push_back(h7);
  }


  // paramètres MCMC
  int nombre_steps_mcmc = 1e5;
  int nombre_samples_collected = 500; // 1 sample tous les 200.
  int nautocor = 500;

  double ftol_rel_fmp = 1e-6; // 0.05 marche
  double ftol_rel_hgps = 1e-4;
  double ftol_rel_gp_llsurrogate = 1e-4;

  // Calibration phase. cool beans
  /*récupération des échantllons post FMP précis et calcul des scores. Nécessaire pour évaluer l'erreur avec la fonction bofplusscores.*/

  /*expensive FMP calibration*/
  {
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
      double d = Dopt.loglikelihood_theta(X, p[0]);
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
    // diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/fmp/autocor.gnu");

    // write samples
    string fnamesamp = "results/fmp/samp.gnu";
    string fnameallsamp = "results/fmp/allsteps.gnu";
    WriteVectors(samples, hparsofsamples, fnamesamp);
    WriteVector(visited_steps, fnameallsamp);

    // predictions
    Dopt.SetNewSamples(samples);
    Dopt.SetNewHparsOfSamples(hparsofsamples);
    string fnamepred = "results/fmp/preds.gnu";
    string fnamesF = "results/fmp/sampsF.gnu";
    string fnamesZ = "results/fmp/sampsZ.gnu";
    Dopt.WritePredictions(XPREDS, fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);
  }


  /*Faire une slice en theta, calculer les vrais hpars optimaux, et le score. On comparera ensuite les valeurs obtenues pour chaque hGP.*/
  vector<VectorXd> thetas_slice;
  vector<VectorXd> hpars_slice;
  vector<double> scores_slice;
  {
    int nslice = 100;
    for (int i = 0; i < nslice; i++)
    {
      VectorXd theta = theta_true;
      theta(0) = (1.0 * i) / nslice;
      thetas_slice.push_back(theta);
    }
    DensityOpt Dopt(MainDensity);
    for (int i = 0; i < thetas_slice.size(); i++)
    {
      VectorXd hpars = Dopt.HparsOpt(thetas_slice[i], hpars_z_guess, ftol_rel_fmp);
      hpars_slice.push_back(hpars);
      scores_slice.push_back(Dopt.loglikelihood_theta(thetas_slice[i], hpars));
    }
    string fname = "results/slice.gnu";
    WriteVectors(thetas_slice, hpars_slice, scores_slice, fname);
  }

  /*récupération des thetapost pour le calcul d'erreur.*/
  vector<VectorXd> thetas_post;
  vector<VectorXd> hpars_post;
  vector<double> scores_post;

  {
    string fname_post = "results/fmp/samp.gnu";
    thetas_post = ReadVector(fname_post); // on prend l'ensemble des points du sample.
    DensityOpt Dopt(MainDensity);
    auto t1 = chrono::steady_clock::now();
    for (int i = 0; i < thetas_post.size(); i++)
    {
      VectorXd hpars = Dopt.HparsOpt(thetas_post[i], hpars_z_guess, ftol_rel_fmp);
      hpars_post.push_back(hpars);
      scores_post.push_back(Dopt.loglikelihood_theta(thetas_post[i], hpars));
    }
  }


  //entraîner des hGPs sur les thetas post et hpars post, juste pour connaitre les hpars hgps optimaux.
/*
  {
    vector<VectorXd> hpars_opt_hgps;
    for (int i = 0; i < hpars_z_guess.size(); i++)
    {
      hpars_opt_hgps.push_back(Hpars_guess_HGPs);
    }
    DensityOpt Dopt(MainDensity);
    Dopt.BuildHGPs(thetas_post, hpars_post, Kernel_GP_Matern32,DKernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, ftol_rel_hgps);
    best_hpars_hgps=Dopt.GetHparsHGPs();
  }
  */


    /*20 répétitions de hparssurrogate avec adaptive sampling, poids calculés par variance de la fct de vraisemblance. dossier as30local*/
  {
    int nrepet = 20;
    for (int k = 0; k < nrepet; k++)
    {
      string foldname = "as30_local/s" + to_string(k);
      // paramètres de l'algorithme
      int ndraws = 100; // nombre de répétitions pour estimer la variance lors du calcul des poids.
      int npts_init = 20;
      int nmax = 800;
      int npts_per_iter = 30;
      int nsteps_mcmc = 7.5e4;
      int nsamples_mcmc = 600;
      VectorXd times = VectorXd::Zero(5);
      // Construction d'un DoE initial. Le même que pour QMC.
      DoE doe_init(lb_t, ub_t, npts_init, generator);
      DensityOpt Dopt(MainDensity);
      vector<VectorXd> thetas_training;
      vector<VectorXd> hpars_training;
      auto tinit = doe_init.GetGrid();

      auto t1 = chrono::steady_clock::now();
      for (const auto theta : doe_init.GetGrid())
      {
        VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, ftol_rel_fmp);
        thetas_training.push_back(theta);
        hpars_training.push_back(hpars);
      }
      auto t2 = chrono::steady_clock::now();
      times(2) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
      // lambda fcts pour l'évaluation de la qualité des surrogates.
      auto get_score = [&Dopt](VectorXd const &theta, VectorXd const &hpars)
      {
        return Dopt.loglikelihood_theta(theta, hpars);
      };
    auto get_hpars_and_var = [&Dopt](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v(1);
      return make_pair(h, v);
    };
      // lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
      auto add_points = [&Dopt, nsteps_mcmc, &X_init_mcmc, &COV_init, &generator, hpars_z_guess, &thetas_training, &hpars_training, ftol_rel_fmp, ndraws, nsamples_mcmc, npts_per_iter](string foldname, VectorXd &times)
      {
        auto get_hpars_opti = [&Dopt, &hpars_z_guess](VectorXd const &X)
        {
          vector<VectorXd> p(1);
          p[0] = Dopt.EvaluateHparOpt(X);
          return p;
        };
        auto compute_score_opti = [&Dopt](vector<VectorXd> h, VectorXd const &X)
        {
          double ll1 = Dopt.loglikelihood_theta(X, h[0]);
          return ll1;
        };
        auto in_bounds = [&Dopt](VectorXd const &X)
        {
          return Dopt.in_bounds_pars(X);
        };
        auto t1 = chrono::steady_clock::now();
        vector<VectorXd> allsteps = Run_MCMC(nsteps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
        vector<VectorXd> candidate_thetas;
        for (int i = 0; i < nsamples_mcmc; i++)
        {
          candidate_thetas.push_back(allsteps[i * (allsteps.size() / nsamples_mcmc)]);
        }
        Nodups(candidate_thetas);
        auto t2 = chrono::steady_clock::now();
        times(0) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        vector<VectorXd> selected_thetas(npts_per_iter);
        vector<double> weights(candidate_thetas.size());
        // tirage sans remise pondéré par les poids.
        auto hpars_opt_hgps = Dopt.GetHparsHGPs();
        auto begin = chrono::steady_clock::now();
        vector<vector<VectorXd>> draws = Dopt.SampleHparsOpt(candidate_thetas, ndraws, generator); // dimensions (d'extérieur à intérieur) : candidate_thetas.size(), ndraws, dim_hpars/
        for (int j = 0; j < weights.size(); j++)
        {
          // calcul des poids.
          // calcul des logvraisemblances pour chacun des draws.
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
        auto t3 = chrono::steady_clock::now();
        times(1) = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();
        for (int i = 0; i < npts_per_iter; i++)
        {
          std::discrete_distribution<int> distribution(weights.begin(), weights.end());
          int drawn = distribution(generator);

          weights[drawn] = 0;
          // réalisation de l'optimisation au points tiré
          selected_thetas[i] = candidate_thetas[drawn];
          VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, ftol_rel_fmp);
          thetas_training.push_back(selected_thetas[i]);
          hpars_training.push_back(hpars);
        }
        cout.clear();
        // on remet cout.
        auto end = chrono::steady_clock::now();
        times(2) = chrono::duration_cast<chrono::milliseconds>(end - t3).count();
        cout << "time for resampling step : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds." << endl;
      };

      auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
      {
        //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
        VectorXd score = evaluate_hgp_surrogate(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
        ofile << thetas_training.size() << " ";
        for (int i = 0; i < score.size(); i++)
        {
          ofile << score(i) << " ";
        }
        ofile << endl;
      };

      // début de l'algorithme
      ofstream ofile("results/" + foldname + "/score.gnu");
      ofstream timefile("results/" + foldname + "/time.gnu");
      timefile << "#Columns : 1. npts 2. MCMC, 3. Resampling, 4. Optimisation des nouveaux points, 5. Mise à jour des GPs, 5. Coût total (1+2+3+4)" << endl;
      vector<VectorXd> hpars_opt_hgps;
      for (int i = 0; i < hpars_z_guess.size(); i++)
      {
        hpars_opt_hgps.push_back(Hpars_guess_HGPs);
      }
      {
        // première passe d'initialisation
        // build GPs
        cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
        auto t1 = chrono::steady_clock::now();
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32,DKernel_GP_Matern32);
        Dopt.SetHGPs(best_hpars_hgps);
        auto t2 = chrono::steady_clock::now();
        times(3) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        hpars_opt_hgps = Dopt.GetHparsHGPs();
        //évaluation de leur score
        write_performance_hgps(ofile);
        //écriture du temps pour cette étape.
        times(4) += times(0) + times(1) + times(2) + times(3);
        timefile << thetas_training.size() << " ";
        for (int o = 0; o < times.size(); o++)
        {
          timefile << times(o) << " ";
        }
        timefile << endl;
      }
      // boucle principale
      while (thetas_training.size() < nmax)
      {
        // ajout de nouveaux points
        add_points(foldname, times);
        // construction hGPs
        cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
        auto t1 = chrono::steady_clock::now();
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32,DKernel_GP_Matern32);
        Dopt.SetHGPs(best_hpars_hgps);
        auto t2 = chrono::steady_clock::now();
        times(3) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        hpars_opt_hgps = Dopt.GetHparsHGPs();
        cout << "hpars de hgps : " << endl;
        for (int i = 0; i < hpars_opt_hgps.size(); i++)
        {
          cout << hpars_opt_hgps[i].transpose() << endl;
        }
        //évaluation de leur score
        write_performance_hgps(ofile);
        //écriture du temps pour cette étape.
        times(4) += times(0) + times(1) + times(2) + times(3);
        timefile << thetas_training.size() << " ";
        for (int o = 0; o < times.size(); o++)
        {
          timefile << times(o) << " ";
        }
        timefile << endl;
      }
      //écriture des points d'apprentissage dans un fichier
      string o1("results/" + foldname + "/constr.gnu");
      WriteVectors(thetas_training, hpars_training, o1);
      if (k == nrepet - 1)
      {
        // si on est à la dernière répétition : calibration, et affichage des calculs sur la slice.
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
        string fnamesamp = "results/" + foldname + "/samp.gnu";
        WriteVectors(samples, hparsofsamples, fnamesamp);
        Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");

        // prédictions sur les slices
        vector<VectorXd> hpars_slice_pred;
        vector<double> scores_slice_pred;
        for (int i = 0; i < thetas_slice.size(); i++)
        {
          VectorXd hp = get_hpars_and_var(thetas_slice[i]).first;
          double score = get_score(thetas_slice[i], hp);
          hpars_slice_pred.push_back(hp);
          scores_slice_pred.push_back(score);
        }
        string fslice = "results/" + foldname + "/slice.gnu";
        WriteVectors(thetas_slice, hpars_slice_pred, scores_slice_pred, fslice);
      }
    }
  }


  /*20 répétitions de hparssurrogate avec LHS. Calibration et slice avec la dernière loop. dossier lhs*/
  {
    int nrepet = 20;
    for (int k = 0; k < nrepet; k++)
    {
      cout << "computing LHS number " << k << endl;
      string foldname = "lhs/s" + to_string(k);
      // paramètres de l'algorithme
      int npts_init = 20;
      int nmax = 800;
      int npts_per_iter = 30;
      // Construction d'un DoE initial. Le même que pour lhs.
      DoE doe_init(lb_t, ub_t, npts_init, generator); // la totalité des points
      DensityOpt Dopt(MainDensity);
      vector<VectorXd> thetas_training; // contient les points courant
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
        return Dopt.loglikelihood_theta(theta, hpars);
      };

      auto get_hpars_and_var = [&Dopt](VectorXd const &X)
      {
        VectorXd h = Dopt.EvaluateHparOpt(X);
        VectorXd v(1);
        return make_pair(h, v);
      };

      // lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
      auto add_points = [&thetas_training, &hpars_training, &Dopt, &generator, &lb_t, &ub_t, &hpars_z_guess, &ftol_rel_fmp, npts_per_iter]()
      {
        // retirer un LHS de taille incrémentée, et y mettre les nouveaux points
        auto hpars_opt_hgps = Dopt.GetHparsHGPs();
        int newsize = thetas_training.size() + npts_per_iter;
        DoE doenew(lb_t, ub_t, newsize, generator);
        thetas_training = doenew.GetGrid();
        hpars_training.clear();
        for (auto const &t : thetas_training)
        {
          VectorXd hp = Dopt.HparsOpt(t, hpars_z_guess, ftol_rel_fmp);
          hpars_training.push_back(hp);
        }
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32,DKernel_GP_Matern32);
        Dopt.SetHGPs(hpars_opt_hgps);
      };

      auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
      {
        //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
        VectorXd score = evaluate_hgp_surrogate(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
        ofile << thetas_training.size() << " ";
        for (int i = 0; i < score.size(); i++)
        {
          ofile << score(i) << " ";
        }
        ofile << endl;
      };

      // début de l'algorithme
      ofstream ofstream("results/" + foldname + "/score.gnu");
      vector<VectorXd> hpars_opt_hgps;
      for (int i = 0; i < hpars_z_guess.size(); i++)
      {
        hpars_opt_hgps.push_back(Hpars_guess_HGPs);
      }
      // initialisation : maj et évaluation.
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32,DKernel_GP_Matern32);
      Dopt.OptimizeHGPs(Bounds_hpars_HGPs, best_hpars_hgps, ftol_rel_hgps);
      hpars_opt_hgps = Dopt.GetHparsHGPs();
      write_performance_hgps(ofstream);
      // ajout de 50 en 50
      while (thetas_training.size() < nmax)
      {
        // ajout de nouveaux points
        add_points();
        // construction hGPs
        cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32,DKernel_GP_Matern32);
        Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, ftol_rel_hgps);
        hpars_opt_hgps = Dopt.GetHparsHGPs();
        //évaluation de leur score
        write_performance_hgps(ofstream);
        //écriture des points d'apprentissage dans un fichier
        string o1("results/" + foldname + "/constr.gnu");
        WriteVectors(thetas_training, hpars_training, o1);
      }
      ofstream.close();
      if (k == nrepet - 1)
      {
        // si on est à la dernière répétition : calibration, et affichage des calculs sur la slice.
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
        string fnamesamp = "results/" + foldname + "/samp.gnu";
        WriteVectors(samples, hparsofsamples, fnamesamp);
        Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");

        // prédictions sur les slices
        vector<VectorXd> hpars_slice_pred;
        vector<double> scores_slice_pred;
        for (int i = 0; i < thetas_slice.size(); i++)
        {
          VectorXd hp = get_hpars_and_var(thetas_slice[i]).first;
          double score = get_score(thetas_slice[i], hp);
          hpars_slice_pred.push_back(hp);
          scores_slice_pred.push_back(score);
        }
        string fslice = "results/" + foldname + "/slice.gnu";
        WriteVectors(thetas_slice, hpars_slice_pred, scores_slice_pred, fslice);
      }
    }
  }
  


  


  /*étude du temps de calcul minimal pour une optimisation FMP*/
  {
    // on fait pour 50 training points, 100 training points, et 400 training points.
    DensityOpt Dopt(MainDensity);
    DoE doe50(lb_t, ub_t, 50, generator);
    DoE doe100(lb_t, ub_t, 100, generator);
    DoE doe400(lb_t, ub_t, 400, generator);
    vector<double> times = {1e-2, 5e-2, 1e-1, 5e-1, 1, 2, 5}; // en secondes
    vector<VectorXd> hpars50;
    vector<VectorXd> hpars100;
    vector<VectorXd> hpars400;
    vector<VectorXd> thetas50 = doe50.GetGrid();
    vector<VectorXd> thetas100 = doe100.GetGrid();
    vector<VectorXd> thetas400 = doe400.GetGrid();
    cout << "o" << endl;
    for (int i = 0; i < thetas50.size(); i++)
    {
      hpars50.push_back(Dopt.HparsOpt(thetas50[i], hpars_z_guess, ftol_rel_fmp));
    }
    for (int i = 0; i < thetas100.size(); i++)
    {
      hpars100.push_back(Dopt.HparsOpt(thetas100[i], hpars_z_guess, ftol_rel_fmp));
    }
    for (int i = 0; i < thetas400.size(); i++)
    {
      hpars400.push_back(Dopt.HparsOpt(thetas400[i], hpars_z_guess, ftol_rel_fmp));
    }
    cout << "o" << endl;
    // avec bon point de départ.
    vector<VectorXd> hpars_opt_hgps;
    for (int i = 0; i < hpars_z_guess.size(); i++)
    {
      hpars_opt_hgps.push_back(Hpars_guess_HGPs);
    }
    //étude à 50
    Dopt.BuildHGPs(thetas50, hpars50, Kernel_GP_Matern32,DKernel_GP_Matern32);
    cout << endl
         << "étude à npts = 50" << endl
         << endl;
    for (double t : times)
    {
      cout << "time : " << t << "s." << endl;
      Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, t);
      hpars_opt_hgps = Dopt.GetHparsHGPs();
    }
    //étude à 100.
    for (int i = 0; i < hpars_z_guess.size(); i++)
    {
      hpars_opt_hgps.push_back(Hpars_guess_HGPs);
    }
    Dopt.BuildHGPs(thetas100, hpars100, Kernel_GP_Matern32,DKernel_GP_Matern32);
    cout << endl
         << "étude à npts = 100" << endl
         << endl;
    for (double t : times)
    {
      cout << "time : " << t << "s." << endl;
      Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, t);
      hpars_opt_hgps = Dopt.GetHparsHGPs();
    }
    //étude à 400.
    for (int i = 0; i < hpars_z_guess.size(); i++)
    {
      hpars_opt_hgps.push_back(Hpars_guess_HGPs);
    }
    Dopt.BuildHGPs(thetas400, hpars400, Kernel_GP_Matern32,DKernel_GP_Matern32);
    cout << endl
         << "étude à npts = 400" << endl
         << endl;
    for (double t : times)
    {
      cout << "time : " << t << "s." << endl;
      Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, t);
      hpars_opt_hgps = Dopt.GetHparsHGPs();
    }
  }

  exit(0);

  /*10 répétitions de surrogate direct logvs avec adaptive sampling. Calibration et slice avec la dernière loop. dossier ll_as */
  {
    int nrepet = 1;
    for (int k = 0; k < nrepet; k++)
    {
      cout << "surrogate logvs avec adaptive sampling. number  " << k << endl;
      string foldname = "ll_as/s" + to_string(k);
      // surrogate de la log_vraisemblance
      GP ll_surrogate(Kernel_GP_Matern32);
      // paramètres de l'algorithme
      int nmax = 400;
      int npts_per_iter = 20;
      int npts_init = 20;
      int nsteps_mcmc = 1e4;
      int nsamples_mcmc = 400;
      VectorXd times = VectorXd::Zero(5);
      // Construction d'un DoE initial. Le même que pour lhs.
      DoE doe_init(lb_t, ub_t, npts_init, generator); // la totalité des points
      DensityOpt Dopt(MainDensity);
      vector<VectorXd> thetas_training; // contient les points courant
      vector<VectorXd> hpars_training;
      auto t1 = chrono::steady_clock::now();
      for (const auto theta : doe_init.GetGrid())
      {
        VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, ftol_rel_fmp);
        thetas_training.push_back(theta);
        hpars_training.push_back(hpars);
      }
      auto t2 = chrono::steady_clock::now();
      times(2) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
      // lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
      auto add_points = [&Dopt, &generator, &lb_t, &ub_t, &hpars_z_guess, &ftol_rel_fmp, nsteps_mcmc, &ll_surrogate, &X_init_mcmc, COV_init, nsamples_mcmc, npts_per_iter](vector<VectorXd> &thetas_training, vector<VectorXd> &hpars_training, string foldname, VectorXd &times)
      {
        // impérativement : il faut modifier thetas_training et hpars_training. La reconstruction du gp surrogate likelihood se fait à l'extérieur de la fonction.
        auto get_hpars_opti = [](VectorXd const &X)
        {
          // inutile pour le surrogate ll.
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
        auto t1 = chrono::steady_clock::now();
        vector<VectorXd> allsteps = Run_MCMC(nsteps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
        vector<VectorXd> candidate_thetas;
        for (int i = 0; i < nsamples_mcmc; i++)
        {
          candidate_thetas.push_back(allsteps[i * (allsteps.size() / nsamples_mcmc)]);
        }
        Nodups(candidate_thetas);
        auto t2 = chrono::steady_clock::now();
        times(0) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        vector<VectorXd> selected_thetas(npts_per_iter);
        vector<double> weights(candidate_thetas.size());
        // tirage sans remise pondéré par les poids.
        auto hparsurr = ll_surrogate.GetPar();
        // muter cout le temps de construction des hgps
        cout.setstate(ios_base::failbit);
        auto begin = chrono::steady_clock::now();
        /* Tirage des poids sans MàJ du surrogate à chaque fois*/
        for (int j = 0; j < weights.size(); j++)
        {
          weights[j] = ll_surrogate.Eval(candidate_thetas[j])(1);
        }
        auto t3 = chrono::steady_clock::now();
        times(1) = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();
        // affichage du candidate set et des poids associés. Affichage également du selected set et des poids.
        // int indice = thetas_training.size();
        // string name1 = "results/" + foldname + "/cand" + to_string(indice) + ".gnu";
        // string name2 = "results/" + foldname + "/sel" + to_string(indice) + ".gnu";
        // ofstream o2(name2);
        // WriteVectors(candidate_thetas, weights, name1);
        for (int i = 0; i < npts_per_iter; i++)
        {
          std::discrete_distribution<int> distribution(weights.begin(), weights.end());
          int drawn = distribution(generator);
          // affichage du theta sélectionné et de son poids
          // VectorXd tsel = candidate_thetas[drawn];
          // for (int i = 0; i < tsel.size(); i++)
          //{
          //  o2 << tsel(i) << " ";
          //}
          // o2 << weights[drawn] << endl;
          // inclusion du theta sélectionné
          weights[drawn] = 0;
          // réalisation de l'optimisation au points tiré
          selected_thetas[i] = candidate_thetas[drawn];
          VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, ftol_rel_fmp);
          thetas_training.push_back(selected_thetas[i]);
          hpars_training.push_back(hpars);
        }
        cout.clear();
        auto end = chrono::steady_clock::now();
        times(2) = chrono::duration_cast<chrono::milliseconds>(end - t3).count();
      };

      auto write_performance_gp_ll = [&thetas_post, &scores_post, &thetas_training](const GP &gp, ofstream &ofile)
      {
        //évaluer la performance du surrogate de log-vraisemblance.
        auto score = evaluate_ll_surrogate(thetas_post, scores_post, gp); // à changer.
        ofile << thetas_training.size() << " " << score.first << " " << score.second << endl;
      };

      // début de l'algorithme
      ofstream ofile("results/" + foldname + "/score.gnu");
      ofstream timefile("results/" + foldname + "/time.gnu");
      timefile << "#Columns : 1. npts  2. MCMC, 3. Resampling, 4. Optimisation des nouveaux points, 5. Mise à jour des GPs, 6. Coût total (1+2+3+4)" << endl;
      // passe d'initialisation.
      vector<DATA> data_ll;
      {
        auto t1 = chrono::steady_clock::now();
        for (int i = 0; i < thetas_training.size(); i++)
        {
          DATA d;
          d.SetX(thetas_training[i]);
          d.SetValue(Dopt.loglikelihood_theta(thetas_training[i], hpars_training[i]));
          data_ll.push_back(d);
        }
        // MaJ du GP, et évaluation.
        auto t2 = chrono::steady_clock::now();
        times(2) += chrono::duration_cast<chrono::milliseconds>(t2 - t1).count(); // rajouter le temps de calcul des logvs.
        ll_surrogate.SetData(data_ll);
        OptimizeGPBis(ll_surrogate, Hpars_guess_llgp, Bounds_hpars_llgp.row(0), Bounds_hpars_llgp.row(1), ftol_rel_gp_llsurrogate);
        auto t3 = chrono::steady_clock::now();
        times(3) = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count(); //
        //écriture du temps pour cette étape.
        times(4) += times(0) + times(1) + times(2) + times(3);
        timefile << thetas_training.size() << " ";
        for (int o = 0; o < times.size(); o++)
        {
          timefile << times(o) << " ";
        }
        timefile << endl;
        cout << "hparsmax ll pour surrogate ll : " << ll_surrogate.GetPar().transpose() << endl;
        write_performance_gp_ll(ll_surrogate, ofile);
      }
      // on rajoute des points de 20 en 20 jusqu'à 400
      while (thetas_training.size() < nmax)
      {
        int previoussize = thetas_training.size();
        // ajout de nouveaux points
        add_points(thetas_training, hpars_training, foldname, times);
        // rajouter les data.
        auto t1 = chrono::steady_clock::now();
        for (int i = previoussize; i < thetas_training.size(); i++)
        {
          DATA d;
          d.SetX(thetas_training[i]);
          d.SetValue(Dopt.loglikelihood_theta(thetas_training[i], hpars_training[i]));
          data_ll.push_back(d);
        }
        auto t2 = chrono::steady_clock::now();
        times(2) += chrono::duration_cast<chrono::milliseconds>(t2 - t1).count(); // rajouter le temps de calcul des logvs.

        // MàJ du surrogate
        ll_surrogate.SetData(data_ll);
        OptimizeGPBis(ll_surrogate, Hpars_guess_llgp, Bounds_hpars_llgp.row(0), Bounds_hpars_llgp.row(1), ftol_rel_gp_llsurrogate);
        auto t3 = chrono::steady_clock::now();
        times(3) = chrono::duration_cast<chrono::milliseconds>(t3 - t2).count();
        cout << "hparsmax ll pour surrogate ll : " << ll_surrogate.GetPar().transpose() << endl;
        //écriture du temps pour cette étape.
        times(4) += times(0) + times(1) + times(2) + times(3);
        timefile << thetas_training.size() << " ";
        for (int o = 0; o < times.size(); o++)
        {
          timefile << times(o) << " ";
        }
        timefile << endl;
        //évaluation du nouveau score
        write_performance_gp_ll(ll_surrogate, ofile);
        //écriture des points d'apprentissage dans un fichier
        string o1("results/" + foldname + "/constr.gnu");
        WriteVectors(thetas_training, hpars_training, o1);
      }
      if (k == nrepet - 1)
      {
        // si on est à la dernière répétition : calibration, et affichage des calculs sur la slice.
        auto in_bounds = [&Dopt](VectorXd const &X)
        {
          return Dopt.in_bounds_pars(X);
        };
        auto get_hpars_opti = [](VectorXd const &X)
        {
          // inutile pour surrogate direct logvs.
          vector<VectorXd> p(1);
          return p;
        };
        auto compute_score_opti = [&ll_surrogate](vector<VectorXd> const &p, VectorXd const &X)
        {
          double d = ll_surrogate.EvalMean(X);
          return d;
        };
        cout << "start mcmc" << endl;
        vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
        vector<VectorXd> samples(nombre_samples_collected);
        for (int i = 0; i < samples.size(); i++)
        {
          samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
        }
        string fnamesamp = "results/" + foldname + "/samp.gnu";
        WriteVector(samples, fnamesamp);
        Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");

        // prédictions sur les slices
        vector<double> scores_slice_pred;
        for (int i = 0; i < thetas_slice.size(); i++)
        {
          double score = ll_surrogate.EvalMean(thetas_slice[i]);
          scores_slice_pred.push_back(score);
        }
        string fslice = "results/" + foldname + "/slice.gnu";
        WriteVectors(thetas_slice, scores_slice_pred, fslice);
      }
    }
  }



  /*10 répétitions de surrogate direct logvs avec LHS. Calibration et slice avec la dernière loop. dossier ll_lhs */
  {
    int nrepet = 10;
    for (int k = 0; k < nrepet; k++)
    {
      cout << "surrogate logvs avec LHS. number  " << k << endl;
      string foldname = "ll_lhs/s" + to_string(k);
      // surrogate de la log_vraisemblance
      GP ll_surrogate(Kernel_GP_Matern32);
      // paramètres de l'algorithme
      int nmax = 400;
      int npts_per_iter = 20;
      int npts_init = 20;
      // Construction d'un DoE initial. Le même que pour lhs.
      DoE doe_init(lb_t, ub_t, npts_init, generator); // la totalité des points
      DensityOpt Dopt(MainDensity);
      vector<VectorXd> thetas_training; // contient les points courant
      vector<VectorXd> hpars_training;
      auto tinit = doe_init.GetGrid();
      for (const auto theta : doe_init.GetGrid())
      {
        VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, ftol_rel_fmp);
        thetas_training.push_back(theta);
        hpars_training.push_back(hpars);
      }

      // lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
      auto add_points = [&Dopt, &generator, &lb_t, &ub_t, &hpars_z_guess, &ftol_rel_fmp, npts_per_iter](vector<VectorXd> &thetas_training, vector<VectorXd> &hpars_training)
      {
        // retirer un LHS de taille incrémentée, et y mettre les nouveaux points
        int newsize = thetas_training.size() + npts_per_iter;
        DoE doenew(lb_t, ub_t, newsize, generator);
        thetas_training = doenew.GetGrid();
        hpars_training.clear();
        for (auto const &t : thetas_training)
        {
          VectorXd hp = Dopt.HparsOpt(t, hpars_z_guess, ftol_rel_fmp);
          hpars_training.push_back(hp);
        }
      };

      auto write_performance_gp_ll = [&thetas_post, &scores_post, &thetas_training](const GP &gp, ofstream &ofile)
      {
        //évaluer la performance du surrogate de log-vraisemblance.
        auto score = evaluate_ll_surrogate(thetas_post, scores_post, gp); // à changer.
        ofile << thetas_training.size() << " " << score.first << " " << score.second << endl;
      };

      // début de l'algorithme
      ofstream ofstream("results/" + foldname + "/score.gnu");
      // initialisation.
      vector<DATA> data_ll;
      for (int i = 0; i < thetas_training.size(); i++)
      {
        DATA d;
        d.SetX(thetas_training[i]);
        d.SetValue(Dopt.loglikelihood_theta(thetas_training[i], hpars_training[i]));
        data_ll.push_back(d);
      }
      // MaJ du GP, et évaluation.
      ll_surrogate.SetData(data_ll);
      OptimizeGPBis(ll_surrogate, Hpars_guess_llgp, Bounds_hpars_llgp.row(0), Bounds_hpars_llgp.row(1), ftol_rel_gp_llsurrogate);
      cout << "hparsmax ll pour surrogate ll : " << ll_surrogate.GetPar().transpose() << endl;
      write_performance_gp_ll(ll_surrogate, ofstream);
      // ajout de 20 en 20
      while (thetas_training.size() < nmax)
      {
        // ajout de nouveaux points
        add_points(thetas_training, hpars_training);

        // construction du vecteur data
        data_ll.clear();
        for (int i = 0; i < thetas_training.size(); i++)
        {
          DATA d;
          d.SetX(thetas_training[i]);
          d.SetValue(Dopt.loglikelihood_theta(thetas_training[i], hpars_training[i]));
          data_ll.push_back(d);
        }
        // MàJ du surrogate
        ll_surrogate.SetData(data_ll);
        OptimizeGPBis(ll_surrogate, Hpars_guess_llgp, Bounds_hpars_llgp.row(0), Bounds_hpars_llgp.row(1), ftol_rel_gp_llsurrogate);
        cout << "hparsmax ll pour surrogate ll : " << ll_surrogate.GetPar().transpose() << endl;

        //évaluation du nouveau score
        write_performance_gp_ll(ll_surrogate, ofstream);

        //écriture des points d'apprentissage dans un fichier
        string o1("results/" + foldname + "/constr.gnu");
        WriteVectors(thetas_training, hpars_training, o1);
      }
      ofstream.close();
      if (k == nrepet - 1)
      {
        // si on est à la dernière répétition : calibration, et affichage des calculs sur la slice.
        auto in_bounds = [&Dopt](VectorXd const &X)
        {
          return Dopt.in_bounds_pars(X);
        };
        auto get_hpars_opti = [](VectorXd const &X)
        {
          // inutile pour surrogate direct logvs.
          vector<VectorXd> p(1);
          return p;
        };
        auto compute_score_opti = [&ll_surrogate](vector<VectorXd> const &p, VectorXd const &X)
        {
          double d = ll_surrogate.EvalMean(X);
          return d;
        };
        cout << "start mcmc" << endl;
        vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
        vector<VectorXd> samples(nombre_samples_collected);
        for (int i = 0; i < samples.size(); i++)
        {
          samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
        }
        string fnamesamp = "results/" + foldname + "/samp.gnu";
        WriteVector(samples, fnamesamp);
        Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");

        // prédictions sur les slices
        vector<double> scores_slice_pred;
        for (int i = 0; i < thetas_slice.size(); i++)
        {
          double score = ll_surrogate.EvalMean(thetas_slice[i]);
          scores_slice_pred.push_back(score);
        }
        string fslice = "results/" + foldname + "/slice.gnu";
        WriteVectors(thetas_slice, scores_slice_pred, fslice);
      }
    }
  }

  exit(0);



  /*Loop de l'algorithme de sélection automatique de points, sans resampling step. On draw le nouveau set directement de la MCMC.*/
  {
    int nrepet = 10;
    for (int k = 0; k < nrepet; k++)
    {
      cout << "starting dro s" << k << endl;
      string foldname = "dros" + to_string(k);
      // paramètres de l'algorithme
      int npts_init = 10;
      int nsteps_mcmc = 1e5;
      int nsamples_mcmc = 10;
      // Construction d'un DoE initial. Le même que pour QMC.
      DoE doe_init(lb_t, ub_t, npts_init, 1);
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
        return Dopt.loglikelihood_theta(theta, hpars);
      };

    auto get_hpars_and_var = [&Dopt](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v(1);
      return make_pair(h, v);
    };

      // lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
      auto add_points = [&Dopt, nsteps_mcmc, nsamples_mcmc, &X_init_mcmc, &COV_init, &generator, hpars_z_guess, &thetas_training, &hpars_training, ftol_rel_fmp]()
      {
        auto get_hpars_opti = [&Dopt, &hpars_z_guess](VectorXd const &X)
        {
          vector<VectorXd> p(1);
          p[0] = Dopt.EvaluateHparOpt(X);
          return p;
        };
        auto compute_score_opti = [&Dopt](vector<VectorXd> h, VectorXd const &X)
        {
          double ll1 = Dopt.loglikelihood_theta(X, h[0]);
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
        vector<VectorXd> selected_thetas(nsamples_mcmc);
        auto hpars_opt_hgps = Dopt.GetHparsHGPs();
        // muter cout le temps de construction des hgps
        for (int i = 0; i < candidate_thetas.size(); i++)
        {
          selected_thetas[i] = candidate_thetas[i];
          VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, ftol_rel_fmp);
          thetas_training.push_back(selected_thetas[i]);
          hpars_training.push_back(hpars);
        }
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32,DKernel_GP_Matern32);
        Dopt.SetHGPs(hpars_opt_hgps);
      };

      auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
      {
        //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
        VectorXd score = evaluate_hgp_surrogate(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
        ofile << thetas_training.size() << " ";
        for (int i = 0; i < score.size(); i++)
        {
          ofile << score(i) << " ";
        }
        ofile << endl;
      };

      // début de l'algorithme
      ofstream ofstream("results/" + foldname + "/score.gnu");
      vector<VectorXd> hpars_opt_hgps;
      for (int i = 0; i < hpars_z_guess.size(); i++)
      {
        hpars_opt_hgps.push_back(Hpars_guess_HGPs);
      }
      while (thetas_training.size() <= 100)
      {
        // construction hGPs
        cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32,DKernel_GP_Matern32);
        Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, ftol_rel_hgps);
        hpars_opt_hgps = Dopt.GetHparsHGPs();
        cout << "hpars de hgps : " << endl;
        for (int i = 0; i < hpars_opt_hgps.size(); i++)
        {
          cout << hpars_opt_hgps[i].transpose() << endl;
        }

        //évaluation de leur score
        write_performance_hgps(ofstream);

        // ajout de nouveaux points
        add_points();

        //écriture des points d'apprentissage dans un fichier
        string o1("results/" + foldname + "/constr.gnu");
        WriteVectors(thetas_training, hpars_training, o1);
      }
      ofstream.close();
    }
  }

  exit(0);

  /*QMC de 100 points*/
  {
    string foldname = "qmc100";
    DensityOpt Dopt(MainDensity);
    int ngrid = 100;
    DoE doe1(lb_t, ub_t, ngrid, 1);
    auto thetas1 = doe1.GetGrid();
    vector<VectorXd> hpars1;
    for (int i = 0; i < ngrid; i++)
    {
      VectorXd hpar1 = Dopt.HparsOpt(thetas1[i], hpars_z_guess, ftol_rel_fmp);
      hpars1.push_back(hpar1);
    }
    // afficher les points de construction
    string fconst = "results/" + foldname + "/constr.gnu";
    WriteVectors(thetas1, hpars1, fconst);
    // construction des surrogates des hpars optimaux.
    Dopt.BuildHGPs(thetas1, hpars1, Kernel_GP_Matern32,DKernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, Hpars_guess_HGPs, ftol_rel_hgps);

    // MCMC
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
    cout << "Beginning FMP calibration with QMC surrogate size 10..." << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples[i] = get_hpars_opti(samples[i])[0];
    }
    // diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");

    // write samples
    string fnamesamp = "results/" + foldname + "/samp.gnu";
    string fnameallsamp = "results/" + foldname + "/allsteps.gnu";
    WriteVectors(samples, hparsofsamples, fnamesamp);
    WriteVector(visited_steps, fnameallsamp);

    // predictions
    Dopt.SetNewSamples(samples);
    Dopt.SetNewHparsOfSamples(hparsofsamples);
    string fnamepred = "results/" + foldname + "/preds.gnu";
    string fnamesF = "results/" + foldname + "/sampsF.gnu";
    string fnamesZ = "results/" + foldname + "/sampsZ.gnu";
    Dopt.WritePredictions(XPREDS, fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);

    // test. Juste afficher la variance de prédiction avec les quantités pour voir si ça marche bien.
    auto get_hpars_and_var = [&Dopt](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v(1);
      return make_pair(h, v);
    };

    auto get_score = [&Dopt](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dopt.loglikelihood_theta(theta, hpars);
    };

    auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas1, &Dopt, &hpars1, &get_score, &scores_post](ofstream &ofile)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd score = evaluate_hgp_surrogate(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
      ofile << thetas1.size() << " ";
      for (int i = 0; i < score.size(); i++)
      {
        ofile << score(i) << " ";
      }
      ofile << endl;
    };

    string fscore = "results/" + foldname + "/score.gnu";
    ofstream ofile(fscore);
    write_performance_hgps(ofile);
    ofile.close();
    // prédictions sur les slices.
    vector<VectorXd> hpars_slice_pred;
    vector<double> scores_slice_pred;
    for (int i = 0; i < thetas_slice.size(); i++)
    {
      VectorXd hp = get_hpars_and_var(thetas_slice[i]).first;
      double score = get_score(thetas_slice[i], hp);
      hpars_slice_pred.push_back(hp);
      scores_slice_pred.push_back(score);
    }
    string fslice = "results/" + foldname + "/slice.gnu";
    WriteVectors(thetas_slice, hpars_slice_pred, scores_slice_pred, fslice);
  }

  

  exit(0);
  /*Analyse de la variabilité en fonction des observations.
  On fait chaque méthode avec seulement 100 points de training. On affiche l'erreur, et le temps total.
  Structure pour afficher tout ça ?
  */
 /*
  {
    // lambda functions pour faire des méthodes facilement.
    auto run_hpas = [&generator,&MainDensity,&lb_t,&ub_t,&X_init_mcmc,&COV_init,&hpars_z_guess](int seedmethod, ofstream &file_score)
    {
      // paramètres de l'algorithme
      int ndraws = 100; // nombre de répétitions pour estimer la variance lors du calcul des poids.
      int npts_init = 20;
      int nmax = 100;
      int npts_per_iter = 10;
      int nsteps_mcmc = 1e4;
      int nsamples_mcmc = 200;
      // Construction d'un DoE initial. Le même que pour QMC.
      DoE doe_init(lb_t, ub_t, npts_init, generator);
      generator.seed(seedmethod); // on laisse le LHS initial varier.
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
        return Dopt.loglikelihood_theta(theta, hpars);
      };
      auto get_hpars_and_var = [&Dopt](VectorXd const &X)
      {
        VectorXd h = Dopt.EvaluateHparOpt(X);
        VectorXd v = Dopt.EvaluateVarHparOpt(X);
        return make_pair(h, v);
      };
      // lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
      auto add_points = [&Dopt, nsteps_mcmc, &X_init_mcmc, &COV_init, &generator, hpars_z_guess, &thetas_training, &hpars_training, ftol_rel_fmp, ndraws, nsamples_mcmc, npts_per_iter]()
      {
        auto get_hpars_opti = [&Dopt, &hpars_z_guess](VectorXd const &X)
        {
          vector<VectorXd> p(1);
          p[0] = Dopt.EvaluateHparOpt(X);
          return p;
        };
        auto compute_score_opti = [&Dopt](vector<VectorXd> h, VectorXd const &X)
        {
          double ll1 = Dopt.loglikelihood_theta(X, h[0]);
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
        vector<double> weights(candidate_thetas.size());
        // tirage sans remise pondéré par les poids.
        auto hpars_opt_hgps = Dopt.GetHparsHGPs();
        auto begin = chrono::steady_clock::now();
        vector<vector<VectorXd>> draws = Dopt.SampleHparsOpt(candidate_thetas, ndraws, generator); // dimensions (d'extérieur à intérieur) : candidate_thetas.size(), ndraws, dim_hpars/
        for (int j = 0; j < weights.size(); j++)
        {
          // calcul des poids.
          // calcul des logvraisemblances pour chacun des draws.
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
          std::discrete_distribution<int> distribution(weights.begin(), weights.end());
          int drawn = distribution(generator);
          weights[drawn] = 0;
          selected_thetas[i] = candidate_thetas[drawn];
          VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, ftol_rel_fmp);
          thetas_training.push_back(selected_thetas[i]);
          hpars_training.push_back(hpars);
        }
        cout.clear();
        // on remet cout.
        auto end = chrono::steady_clock::now();
        cout << "time for resampling step : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds." << endl;
      };

      // début de l'algorithme
      vector<VectorXd> hpars_opt_hgps;
      for (int i = 0; i < hpars_z_guess.size(); i++)
      {
        hpars_opt_hgps.push_back(Hpars_guess_HGPs);
      }

      auto begin = chrono::steady_clock::now();
      {
        // première passe d'initialisation
        // build GPs
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
        Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, ftol_rel_hgps);
        hpars_opt_hgps = Dopt.GetHparsHGPs();
        //écriture du temps pour cette étape.
      }
      // boucle principale
      while (thetas_training.size() < nmax)
      {
        // ajout de nouveaux points
        add_points();
        // construction hGPs
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
        Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, ftol_rel_hgps);
        auto t2 = chrono::steady_clock::now();
        times(3) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
        hpars_opt_hgps = Dopt.GetHparsHGPs();
        cout << "hpars de hgps : " << endl;
        for (int i = 0; i < hpars_opt_hgps.size(); i++)
        {
          cout << hpars_opt_hgps[i].transpose() << endl;
        }
      }

      auto end = chrono::steady_clock::now();
      double time = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
      //écriture de la performance à 100 points
      VectorXd score = evaluate_hgp_surrogate(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
      file_score << seedmethod << " " << score(7) << " " << time << endl;
    };

    auto run_hplhs = [&generator,&MainDensity,&lb_t,&ub_t,&X_init_mcmc,&COV_init,&hpars_z_guess](int seedmethod, ofstream &file_score)
    {
      cout << "computing LHS number " << seedmethod << endl;
      // paramètres de l'algorithme
      int npts_init = 100;
      // Construction d'un DoE initial. Le même que pour lhs.
      generator.seed(seedmethod);
      DoE doe_init(lb_t, ub_t, npts_init, generator); // la totalité des points
      DensityOpt Dopt(MainDensity);
      vector<VectorXd> thetas_training; // contient les points courant
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
        return Dopt.loglikelihood_theta(theta, hpars);
      };

      auto get_hpars_and_var = [&Dopt](VectorXd const &X)
      {
        VectorXd h = Dopt.EvaluateHparOpt(X);
        VectorXd v = Dopt.EvaluateVarHparOpt(X);
        return make_pair(h, v);
      };

      auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
      {
        //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.

        ofile << thetas_training.size() << " ";
        for (int i = 0; i < score.size(); i++)
        {
          ofile << score(i) << " ";
        }
        ofile << endl;
      };
      vector<VectorXd> hpars_opt_hgps;
      for (int i = 0; i < hpars_z_guess.size(); i++)
      {
        hpars_opt_hgps.push_back(Hpars_guess_HGPs);
      }
      // initialisation : maj et évaluation.
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
      Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, ftol_rel_hgps);
      VectorXd score = evaluate_hgp_surrogate(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
      // fait 100 points
      file_score << seedmethod << " " << score(7) << " " << time << endl;
    }

    auto run_fmp_expensive = [&MainDensity,&lb_t,&ub_t,&generator,&X_init_mcmc,&COV_init,&hpars_z_guess](string fnamesamp)
    { // FMP expensive. fnamesamp est là où on stocke les samples pour le calcul d'erreur.
      int ftol_rel_fmp=5e-2;
      int nombre_steps_mcmc=5e4;
      int nombre_samples_collected=500;
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
        double d = Dopt.loglikelihood_theta(X, p[0]);
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
      // diagnostic
      Selfcor_diagnosis(visited_steps, nautocor, 1, "resultsfake");

      // write samples
      WriteVectors(samples, hparsofsamples, fnamesamp);
    };
  
  //il reste à tirer les observations, assurer le theta_post et afficher dans les fichiers.
  }
  */

  /*QMC de 10 points*/
  {
    string foldname = "qmc10";
    DensityOpt Dopt(MainDensity);
    int ngrid = 10;
    DoE doe1(lb_t, ub_t, ngrid, 1);
    auto thetas1 = doe1.GetGrid();
    vector<VectorXd> hpars1;
    for (int i = 0; i < ngrid; i++)
    {
      VectorXd hpar1 = Dopt.HparsOpt(thetas1[i], hpars_z_guess, ftol_rel_fmp);
      hpars1.push_back(hpar1);
    }
    // afficher les points de construction
    string fconst = "results/" + foldname + "/constr.gnu";
    WriteVectors(thetas1, hpars1, fconst);
    // construction des surrogates des hpars optimaux.
    Dopt.BuildHGPs(thetas1, hpars1, Kernel_GP_Matern32,DKernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, Hpars_guess_HGPs, ftol_rel_hgps);

    // MCMC
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
    cout << "Beginning FMP calibration with QMC surrogate size 10..." << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples[i] = get_hpars_opti(samples[i])[0];
    }
    // diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");

    // write samples
    string fnamesamp = "results/" + foldname + "/samp.gnu";
    string fnameallsamp = "results/" + foldname + "/allsteps.gnu";
    WriteVectors(samples, hparsofsamples, fnamesamp);
    WriteVector(visited_steps, fnameallsamp);

    // predictions
    Dopt.SetNewSamples(samples);
    Dopt.SetNewHparsOfSamples(hparsofsamples);
    string fnamepred = "results/" + foldname + "/preds.gnu";
    string fnamesF = "results/" + foldname + "/sampsF.gnu";
    string fnamesZ = "results/" + foldname + "/sampsZ.gnu";
    Dopt.WritePredictions(XPREDS, fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);

    // test. Juste afficher la variance de prédiction avec les quantités pour voir si ça marche bien.
    auto get_hpars_and_var = [&Dopt](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v(1);
      return make_pair(h, v);
    };

    auto get_score = [&Dopt](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dopt.loglikelihood_theta(theta, hpars);
    };

    auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas1, &Dopt, &hpars1, &get_score, &scores_post](ofstream &ofile)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd score = evaluate_hgp_surrogate(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
      ofile << thetas1.size() << " ";
      for (int i = 0; i < score.size(); i++)
      {
        ofile << score(i) << " ";
      }
      ofile << endl;
    };

    string fscore = "results/" + foldname + "/score.gnu";
    ofstream ofile(fscore);
    write_performance_hgps(ofile);
    ofile.close();
  }

  exit(0);

}
