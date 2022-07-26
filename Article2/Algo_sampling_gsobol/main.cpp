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
  return pow(exp(hpar(0)), 2) * exp(-0.5 * (pow(d1, 2) + pow(d2, 2) + pow(d3, 2)));
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double cor = pow(hpar(0), 2);
  cor *= (1 + abs(x(0) - y(0)) / hpar(1)) * exp(-abs(x(0) - y(0)) / hpar(1)); // x1
  cor *= (1 + abs(x(1) - y(1)) / hpar(3)) * exp(-abs(x(1) - y(1)) / hpar(3)); // x2
  cor *= (1 + abs(x(2) - y(2)) / hpar(4)) * exp(-abs(x(2) - y(2)) / hpar(4)); // x3
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

VectorXd evaluate_surrogate(vector<VectorXd> const &thetas_ref, vector<VectorXd> const &hpars_ref, function<pair<VectorXd, VectorXd>(VectorXd const &)> const &get_hpars_and_var)
{
  //évaluation de la qualité d'un surrogate par la fonction get_hpars qu'il propose. On pourra en profiter pour vérifier que pour la densité fmp e référence ça fait bien zéro.
  // prenons l'erreur moyenne L2 relative ? Non. On va faire la moyenne + variance. averaged absolute individual standardized error. Et la moyenne a posteriori de cette quantité.
  // calcul de l'erreur dans chaque dimension.
  if (!(thetas_ref.size() == hpars_ref.size()))
  {
    cerr << "erreur : différentes tailles evaluate_surrogate !" << endl;
    exit(0);
  }
  VectorXd err = VectorXd::Zero(hpars_ref[0].size());
  for (int i = 0; i < thetas_ref.size(); i++)
  {
    auto p = get_hpars_and_var(thetas_ref[i]);
    VectorXd std = p.second.array().sqrt();
    VectorXd v = (hpars_ref[i] - p.first).cwiseQuotient(std).array().abs();
    err += v;
  }
  return err / thetas_ref.size();
}

VectorXd evaluate_surrogate_bof(vector<VectorXd> const &thetas_ref, vector<VectorXd> const &hpars_ref, function<pair<VectorXd, VectorXd>(VectorXd const &)> const &get_hpars_and_var)
{
  //évaluation de la qualité d'un surrogate par la fonction get_hpars qu'il propose. On pourra en profiter pour vérifier que pour la densité fmp e référence ça fait bien zéro.
  // On va juste faire l'erreur L2 relative. relou mais avec la variance OLM ça marche pas trop bien.
  if (!(thetas_ref.size() == hpars_ref.size()))
  {
    cerr << "erreur : différentes tailles evaluate_surrogate !" << endl;
    exit(0);
  }
  VectorXd err = VectorXd::Zero(hpars_ref[0].size());
  for (int i = 0; i < thetas_ref.size(); i++)
  {
    auto p = get_hpars_and_var(thetas_ref[i]);
    VectorXd val = p.first.array();
    VectorXd v = (hpars_ref[i] - p.first).cwiseQuotient(val).array().abs();
    err += v;
  }
  cout << "score moyen : " << (err / thetas_ref.size()).array().mean() << endl;
  return err / thetas_ref.size();
}

VectorXd evaluate_surrogate_bofplusscores(vector<VectorXd> const &thetas_ref, vector<VectorXd> const &hpars_ref, vector<double> const &scores_ref, function<pair<VectorXd, VectorXd>(VectorXd const &)> const &get_meanpred, function<double(VectorXd const &, VectorXd const &)> const &get_score)
{
  // On va juste faire l'erreur L2 relative.
  // dernière composante : l'erreur L2 sur la fct likelihood elle-même. Plus interprétable.
  if (!(thetas_ref.size() == hpars_ref.size()))
  {
    cerr << "erreur : différentes tailles evaluate_surrogate !" << endl;
    exit(0);
  }
  vector<VectorXd> hpars_calcules;
  VectorXd err = VectorXd::Zero(hpars_ref[0].size());
  for (int i = 0; i < thetas_ref.size(); i++)
  {
    auto p = get_meanpred(thetas_ref[i]); // ne récupérer que le premier argument
    hpars_calcules.push_back(p.first);
    VectorXd val = hpars_ref[i];
    VectorXd v = (hpars_ref[i] - p.first).cwiseQuotient(val).array().square();
    err += v;
  }
  VectorXd err_hpars = (err / thetas_ref.size()).array().sqrt(); // erreur des hpars
  double err_loglik = 0;
  for (int i = 0; i < scores_ref.size(); i++)
  {
    double score_calcule = get_score(thetas_ref[i], hpars_calcules[i]);
    err_loglik += pow(scores_ref[i] - score_calcule, 2);
  }
  err_loglik = sqrt(err_loglik / scores_ref.size());
  VectorXd errtot(err_hpars.size() + 1);
  errtot.head(err_hpars.size()) = err_hpars; // droit de faire ça ?
  errtot(err_hpars.size()) = err_loglik;
  return errtot;
}

double gsobol(VectorXd const &X, VectorXd const &theta)
{
  // fct gsobol
  double res = 1;
  for (int i = 0; i < 3; i++)
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
  int dim_theta = 3;
  int dim_hpars = 4;

  VectorXd lb_t = VectorXd::Zero(dim_theta);
  VectorXd ub_t = VectorXd::Ones(dim_theta);
  VectorXd lb_hpars(dim_hpars);
  lb_hpars(0) = 1e-7; // pas 0 car échelle log
  lb_hpars(1) = 0.01;
  lb_hpars(2) = 0.01;
  lb_hpars(3) = 0.01;
  VectorXd ub_hpars(dim_hpars);
  ub_hpars(0) = 5;
  ub_hpars(1) = 3;
  ub_hpars(2) = 3;
  ub_hpars(3) = 3;
  // conversion en log.
  lb_hpars.array() = lb_hpars.array().log();
  ub_hpars.array() = ub_hpars.array().log();

  VectorXd hpars_z_guess = 0.5 * (lb_hpars + ub_hpars);

  // création des observations.
  VectorXd theta_true(dim_theta); // valeur de theta utilisée pour générer les observations
  theta_true << 0.55, 0.75, 0.3;
  double noise = 5e-2;
  int ndata = 20;
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
  DoE doe_init(lb_t, ub_t, 100, 10); // doe halton de 100 points
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
  MainDensity.SetKernel(Kernel_Z_SE);
  MainDensity.SetHparsBounds(lb_hpars, ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
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
  for(int i=0;i<dim_theta;i++){
      COV_init(i, i) = pow(0.05, 2);
  }
  cout << "COV_init : " << endl
       << COV_init << endl;

  MatrixXd Bounds_hpars_HGPs(2, 5); // 5 hpars HGPs. std, noise, 3 longueurs de corrélation. à changer en fct de theta.
  Bounds_hpars_HGPs(0, 0) = 0.5;    // std
  Bounds_hpars_HGPs(1, 0) = 10;
  Bounds_hpars_HGPs(0, 1) = 1e-8; // lcor 1
  Bounds_hpars_HGPs(1, 1) = 10;
  Bounds_hpars_HGPs(0, 2) = 1e-8; // stdnoise
  Bounds_hpars_HGPs(1, 2) = 1e-5;
  Bounds_hpars_HGPs(0, 3) = 1e-8; // lcor 2
  Bounds_hpars_HGPs(1, 3) = 10;
  Bounds_hpars_HGPs(0, 4) = 1e-8; // lcor 2
  Bounds_hpars_HGPs(1, 4) = 10;
  VectorXd Hpars_guess_HGPs = 0.5 * (Bounds_hpars_HGPs.row(0) + Bounds_hpars_HGPs.row(1)).transpose();

  // paramètres MCMC
  int nombre_steps_mcmc = 1e5;
  int nombre_samples_collected = 500; // 1 sample tous les 200.
  int nautocor = 500;

  double time_opti_fmp = 1e-2;
  double time_opti_hgps = 5;

  // Calibration phase. cool beans
  /*récupération des échantllons post FMP précis et calcul des scores. Nécessaire pour évaluer l'erreur avec la fonction bofplusscores.*/
  vector<VectorXd> thetas_post;
  vector<VectorXd> hpars_post;
  vector<double> scores_post;
  {
    string fname_post = "results/fmp/samp.gnu";
    thetas_post = ReadVector(fname_post); // on prend l'ensemble des points du sample.
    DensityOpt Dopt(MainDensity);
    for (int i = 0; i < thetas_post.size(); i++)
    {
      VectorXd hpars = Dopt.HparsOpt(thetas_post[i], hpars_z_guess, time_opti_fmp);
      hpars_post.push_back(hpars);
      scores_post.push_back(Dopt.loglikelihood_theta(thetas_post[i], hpars));
    }
  }
  
  /*Faire une slice en theta, calculer les vrais hpars optimaux, et le score. On comparera ensuite les valeurs obtenues pour chaque hGP.*/
  vector<VectorXd> thetas_slice;
  vector<VectorXd> hpars_slice;
  vector<double> scores_slice;
  {
    int nslice=100;
    for(int i=0;i<nslice;i++){
      VectorXd theta=theta_true;
      theta(0)=(1.0*i)/nslice;
      thetas_slice.push_back(theta);
    }
    DensityOpt Dopt(MainDensity);
    for (int i = 0; i < thetas_slice.size(); i++)
    {
      VectorXd hpars = Dopt.HparsOpt(thetas_slice[i], hpars_z_guess, time_opti_fmp);
      hpars_slice.push_back(hpars);
      scores_slice.push_back(Dopt.loglikelihood_theta(thetas_slice[i], hpars));
    }
    string fname="results/slice.gnu";
    WriteVectors(thetas_slice,hpars_slice,scores_slice,fname);
  }
  

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
      VectorXd hpar1 = Dopt.HparsOpt(thetas1[i], hpars_z_guess, time_opti_fmp);
      hpars1.push_back(hpar1);
    }
    // afficher les points de construction
    string fconst = "results/" + foldname + "/constr.gnu";
    WriteVectors(thetas1, hpars1, fconst);
    // construction des surrogates des hpars optimaux.
    Dopt.BuildHGPs(thetas1, hpars1, Kernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, Hpars_guess_HGPs, time_opti_hgps);

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
    auto get_hpars_and_var = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v = Dopt.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };

    auto get_score = [&Dopt](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dopt.loglikelihood_theta(theta, hpars);
    };

    auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas1, &Dopt, &hpars1, &get_score, &scores_post](ofstream &ofile)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd score = evaluate_surrogate_bofplusscores(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
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
    //prédictions sur les slices.
    vector<VectorXd> hpars_slice_pred;
    vector<double> scores_slice_pred;
    for(int i=0;i<thetas_slice.size();i++){
      VectorXd hp=get_hpars_and_var(thetas_slice[i]).first;
      double score=get_score(thetas_slice[i],hp);
      hpars_slice_pred.push_back(hp);
      scores_slice_pred.push_back(score);
    }
    string fslice="results/"+foldname+"/slice.gnu";
    WriteVectors(thetas_slice,hpars_slice_pred,scores_slice_pred,fslice);
  }

  /*Algorithme de sélection automatique de points*/
  {
    generator.seed(0);
    string foldname = "is10";
    // paramètres de l'algorithme
    int npts_init = 10;
    int npts_per_iter = 10;
    int nsteps_mcmc = 1e5;
    int nsamples_mcmc = 200;
    // Construction d'un DoE initial. Le même que pour QMC.
    DoE doe_init(lb_t, ub_t, npts_init, 1);
    DensityOpt Dopt(MainDensity);
    vector<VectorXd> thetas_training;
    vector<VectorXd> hpars_training;
    auto tinit = doe_init.GetGrid();
    for (const auto theta : doe_init.GetGrid())
    {
      VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, time_opti_fmp);
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
    auto add_points = [&Dopt, npts_per_iter, nsteps_mcmc, nsamples_mcmc, &X_init_mcmc, &COV_init, &generator, hpars_z_guess, &thetas_training, &hpars_training, time_opti_fmp]()
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

      vector<VectorXd> selected_thetas(npts_per_iter);
      vector<double> weights(candidate_thetas.size());
      // tirage sans remise pondéré par les poids.
      auto hpars_opt_hgps = Dopt.GetHparsHGPs();
      // muter cout le temps de construction des hgps
      cout.setstate(ios_base::failbit);
      auto begin = chrono::steady_clock::now();
      for (int i = 0; i < npts_per_iter; i++)
      {
        // calcul des  poids, sauf ceux qui sont déjà à 0 (déjà tiré)
        for (int j = 0; j < weights.size(); j++)
        {
          if ((weights[j] != 0.0) || (i == 0))
          {
            weights[j] = Dopt.EstimatePredError(candidate_thetas[j]);
          }
        }
        std::discrete_distribution<int> distribution(weights.begin(), weights.end());
        int drawn = distribution(generator);
        weights[drawn] = 0;
        // réalisation de l'optimisation au points tiré
        selected_thetas[i] = candidate_thetas[drawn];
        VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, time_opti_fmp);
        thetas_training.push_back(selected_thetas[i]);
        hpars_training.push_back(hpars);
        // on inclut ce point dans les hGPs, mais on ne change pas les hyperparamètres des hGPs. Les routines d'optimisation sont utilisées mais juste pour que ça soit pratique.
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
        Dopt.SetHGPs(hpars_opt_hgps);
      }
      // on remet cout.
      cout.clear();
      auto end = chrono::steady_clock::now();
      cout << "time for resampling step : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds." << endl;
    };

    auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd score = evaluate_surrogate_bofplusscores(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
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
    while (thetas_training.size() < 100)
    {
      // construction hGPs
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
      Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, time_opti_hgps);
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
    //calibration
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
        //prédictions sur les slices.
    vector<VectorXd> hpars_slice_pred;
    vector<double> scores_slice_pred;
    for(int i=0;i<thetas_slice.size();i++){
      VectorXd hp=get_hpars_and_var(thetas_slice[i]).first;
      double score=get_score(thetas_slice[i],hp);
      hpars_slice_pred.push_back(hp);
      scores_slice_pred.push_back(score);
    }
    string fslice="results/"+foldname+"/slice.gnu";
    WriteVectors(thetas_slice,hpars_slice_pred,scores_slice_pred,fslice);
  }



exit(0);
    /*Loop de l'algorithme de sélection automatique de points, sans resampling step. On draw le nouveau set directement de la MCMC.*/
  {
    int nrepet = 10;
    for (int k = 0; k < nrepet; k++)
    {
      cout << "starting dro s" << k << endl;
      string foldname = "dros"+to_string(k);
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
        VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, time_opti_fmp);
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
      auto add_points = [&Dopt, nsteps_mcmc, nsamples_mcmc, &X_init_mcmc, &COV_init, &generator, hpars_z_guess, &thetas_training, &hpars_training, time_opti_fmp]()
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
          VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, time_opti_fmp);
          thetas_training.push_back(selected_thetas[i]);
          hpars_training.push_back(hpars);
        }
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
        Dopt.SetHGPs(hpars_opt_hgps);
      };

      auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
      {
        //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
        VectorXd score = evaluate_surrogate_bofplusscores(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
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
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
        Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, time_opti_hgps);
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
  
  exit(0);
  
  /*Loop pour faire plusieurs QMC.*/
    /*Loop de l'algorithme de sélection automatique de points*/
  {
    int nrepet = 10;
    for (int k = 0; k < nrepet; k++)
    {
      cout << "computing QMC number " << k << endl;
      string foldname = "qmcs"+to_string(k);
      // paramètres de l'algorithme
      int nptsmax = 100;
      int npts_init = 10;
      int npts_per_iter = 10;
      // Construction d'un DoE initial. Le même que pour QMC.
      int indstart=1+500*k;
      DoE doe_qmc(lb_t, ub_t, nptsmax, indstart); // la totalité des points
      DensityOpt Dopt(MainDensity);
      vector<VectorXd> thetas_training; //contient les points courant
      vector<VectorXd> hpars_training;      
      vector<VectorXd> thetas_qmc; //contient tous les points
      vector<VectorXd> hpars_qmc;
      auto tinit = doe_qmc.GetGrid();
      for (const auto theta : doe_qmc.GetGrid())
      {
        VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, time_opti_fmp);
        thetas_qmc.push_back(theta);
        hpars_qmc.push_back(hpars);
      }
      for(int i=0;i<npts_init;i++){
        thetas_training.push_back(thetas_qmc[i]);
        hpars_training.push_back(hpars_qmc[i]);
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
      auto add_points = [&thetas_training, &hpars_training,npts_per_iter,&Dopt,&thetas_qmc,&hpars_qmc]()
      {
        //rajouter npts_per_iter points du theta_qmc
        auto hpars_opt_hgps = Dopt.GetHparsHGPs();
        int size=thetas_training.size();
        for (int i = size; i < size+npts_per_iter; i++)
        {
          if(i>= thetas_qmc.size()){break;}
          thetas_training.push_back(thetas_qmc[i]);
          hpars_training.push_back(hpars_qmc[i]);
        }
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
        Dopt.SetHGPs(hpars_opt_hgps);
      };

      auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
      {
        //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
        VectorXd score = evaluate_surrogate_bofplusscores(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
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
      while (thetas_training.size() < 100)
      {
        // construction hGPs
        cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
        Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, time_opti_hgps);
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

  /*Loop de l'algorithme de sélection automatique de points*/
  {
    int nrepet = 10;
    for (int k = 0; k < nrepet; k++)
    {
      string foldname = "is10s"+to_string(k);
      // paramètres de l'algorithme
      int npts_init = 10;
      int npts_per_iter = 10;
      int nsteps_mcmc = 1e5;
      int nsamples_mcmc = 200;
      // Construction d'un DoE initial. Le même que pour QMC.
      DoE doe_init(lb_t, ub_t, npts_init, 1);
      DensityOpt Dopt(MainDensity);
      vector<VectorXd> thetas_training;
      vector<VectorXd> hpars_training;
      auto tinit = doe_init.GetGrid();
      for (const auto theta : doe_init.GetGrid())
      {
        VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, time_opti_fmp);
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
      auto add_points = [&Dopt, npts_per_iter, nsteps_mcmc, nsamples_mcmc, &X_init_mcmc, &COV_init, &generator, hpars_z_guess, &thetas_training, &hpars_training, time_opti_fmp]()
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

        vector<VectorXd> selected_thetas(npts_per_iter);
        vector<double> weights(candidate_thetas.size());
        // tirage sans remise pondéré par les poids.
        auto hpars_opt_hgps = Dopt.GetHparsHGPs();
        // muter cout le temps de construction des hgps
        cout.setstate(ios_base::failbit);
        auto begin = chrono::steady_clock::now();
        for (int i = 0; i < npts_per_iter; i++)
        {
          // calcul des  poids, sauf ceux qui sont déjà à 0 (déjà tiré)
          for (int j = 0; j < weights.size(); j++)
          {
            if ((weights[j] != 0.0) || (i == 0))
            {
              weights[j] = Dopt.EstimatePredError(candidate_thetas[j]);
            }
          }
          std::discrete_distribution<int> distribution(weights.begin(), weights.end());
          int drawn = distribution(generator);
          weights[drawn] = 0;
          // réalisation de l'optimisation au points tiré
          selected_thetas[i] = candidate_thetas[drawn];
          VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, time_opti_fmp);
          thetas_training.push_back(selected_thetas[i]);
          hpars_training.push_back(hpars);
          // on inclut ce point dans les hGPs, mais on ne change pas les hyperparamètres des hGPs. Les routines d'optimisation sont utilisées mais juste pour que ça soit pratique.
          Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
          Dopt.SetHGPs(hpars_opt_hgps);
        }
        // on remet cout.
        cout.clear();
        auto end = chrono::steady_clock::now();
        cout << "time for resampling step : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds." << endl;
      };

      auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
      {
        //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
        VectorXd score = evaluate_surrogate_bofplusscores(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
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
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
        Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, time_opti_hgps);
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



exit(0);
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
      VectorXd hpar1 = Dopt.HparsOpt(thetas1[i], hpars_z_guess, time_opti_fmp);
      hpars1.push_back(hpar1);
    }
    // afficher les points de construction
    string fconst = "results/" + foldname + "/constr.gnu";
    WriteVectors(thetas1, hpars1, fconst);
    // construction des surrogates des hpars optimaux.
    Dopt.BuildHGPs(thetas1, hpars1, Kernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, Hpars_guess_HGPs, time_opti_hgps);

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
    auto get_hpars_and_var = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v = Dopt.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };

    auto get_score = [&Dopt](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dopt.loglikelihood_theta(theta, hpars);
    };

    auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas1, &Dopt, &hpars1, &get_score, &scores_post](ofstream &ofile)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd score = evaluate_surrogate_bofplusscores(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
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

  /*QMC de 50 points*/
  {
    string foldname = "qmc50";
    DensityOpt Dopt(MainDensity);
    int ngrid = 50;
    DoE doe1(lb_t, ub_t, ngrid, 1);
    auto thetas1 = doe1.GetGrid();
    vector<VectorXd> hpars1;
    for (int i = 0; i < ngrid; i++)
    {
      VectorXd hpar1 = Dopt.HparsOpt(thetas1[i], hpars_z_guess, time_opti_fmp);
      hpars1.push_back(hpar1);
    }
    // afficher les points de construction
    string fconst = "results/" + foldname + "/constr.gnu";
    WriteVectors(thetas1, hpars1, fconst);
    // construction des surrogates des hpars optimaux.
    Dopt.BuildHGPs(thetas1, hpars1, Kernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, Hpars_guess_HGPs, time_opti_hgps);

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
    auto get_hpars_and_var = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v = Dopt.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };

    auto get_score = [&Dopt](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dopt.loglikelihood_theta(theta, hpars);
    };

    auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas1, &Dopt, &hpars1, &get_score, &scores_post](ofstream &ofile)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd score = evaluate_surrogate_bofplusscores(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
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



  /*Algorithme de sélection automatique de points*/
  {
    generator.seed(9999);
    string foldname = "is10s2";
    // paramètres de l'algorithme
    int npts_init = 10;
    int npts_per_iter = 10;
    int nsteps_mcmc = 1e5;
    int nsamples_mcmc = 200;
    // Construction d'un DoE initial. Le même que pour QMC.
    DoE doe_init(lb_t, ub_t, npts_init, 1);
    DensityOpt Dopt(MainDensity);
    vector<VectorXd> thetas_training;
    vector<VectorXd> hpars_training;
    auto tinit = doe_init.GetGrid();
    for (const auto theta : doe_init.GetGrid())
    {
      VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, time_opti_fmp);
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
    auto add_points = [&Dopt, npts_per_iter, nsteps_mcmc, nsamples_mcmc, &X_init_mcmc, &COV_init, &generator, hpars_z_guess, &thetas_training, &hpars_training, time_opti_fmp]()
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

      vector<VectorXd> selected_thetas(npts_per_iter);
      vector<double> weights(candidate_thetas.size());
      // tirage sans remise pondéré par les poids.
      auto hpars_opt_hgps = Dopt.GetHparsHGPs();
      // muter cout le temps de construction des hgps
      cout.setstate(ios_base::failbit);
      auto begin = chrono::steady_clock::now();
      for (int i = 0; i < npts_per_iter; i++)
      {
        // calcul des  poids, sauf ceux qui sont déjà à 0 (déjà tiré)
        for (int j = 0; j < weights.size(); j++)
        {
          if ((weights[j] != 0.0) || (i == 0))
          {
            weights[j] = Dopt.EstimatePredError(candidate_thetas[j]);
          }
        }
        std::discrete_distribution<int> distribution(weights.begin(), weights.end());
        int drawn = distribution(generator);
        weights[drawn] = 0;
        // réalisation de l'optimisation au points tiré
        selected_thetas[i] = candidate_thetas[drawn];
        VectorXd hpars = Dopt.HparsOpt(selected_thetas[i], hpars_z_guess, time_opti_fmp);
        thetas_training.push_back(selected_thetas[i]);
        hpars_training.push_back(hpars);
        // on inclut ce point dans les hGPs, mais on ne change pas les hyperparamètres des hGPs. Les routines d'optimisation sont utilisées mais juste pour que ça soit pratique.
        Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
        Dopt.SetHGPs(hpars_opt_hgps);
      }
      // on remet cout.
      cout.clear();
      auto end = chrono::steady_clock::now();
      cout << "time for resampling step : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds." << endl;
    };

    auto write_performance_hgps = [&thetas_post, &hpars_post, &get_hpars_and_var, &thetas_training, &Dopt, &hpars_training, &get_score, &scores_post](ofstream &ofile)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd score = evaluate_surrogate_bofplusscores(thetas_post, hpars_post, scores_post, get_hpars_and_var, get_score);
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
      Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32);
      Dopt.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps, time_opti_hgps);
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

  exit(0);

  /*expensive FMP calibration*/
  {
    DensityOpt Dopt(MainDensity);
    auto in_bounds = [&Dopt](VectorXd const &X)
    {
      return Dopt.in_bounds_pars(X);
    };
    auto get_hpars_opti = [Dopt, &hpars_z_guess, &time_opti_fmp](VectorXd const &X)
    {
      vector<VectorXd> p(1);
      p[0] = Dopt.HparsOpt(X, hpars_z_guess, time_opti_fmp);
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

  exit(0);

  /*FMP calibration avec le test set*/
  {
    DensityOpt Dopt(MainDensity);

    // construction des surrogates des hpars optimaux.
    Dopt.BuildHGPs(thetas_test, hpars_test, Kernel_GP_Matern32);
    Dopt.OptimizeHGPs(Bounds_hpars_HGPs, Hpars_guess_HGPs, 1);

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
    cout << "Beginning FMP calibration with QMC surrogate..." << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples[i] = get_hpars_opti(samples[i])[0];
    }
    // diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/fmpqmc/autocor.gnu");

    // write samples
    string fnamesamp = "results/fmpqmc/samp.gnu";
    string fnameallsamp = "results/fmpqmc/allsteps.gnu";
    WriteVectors(samples, hparsofsamples, fnamesamp);
    WriteVector(visited_steps, fnameallsamp);

    // predictions
    Dopt.SetNewSamples(samples);
    Dopt.SetNewHparsOfSamples(hparsofsamples);
    string fnamepred = "results/fmpqmc/preds.gnu";
    string fnamesF = "results/fmpqmc/sampsF.gnu";
    string fnamesZ = "results/fmpqmc/sampsZ.gnu";
    Dopt.WritePredictions(XPREDS, fnamepred);
    Dopt.WriteSamplesFandZ(XPREDS, fnamesF, fnamesZ);

    // test. Juste afficher la variance de prédiction avec les quantités pour voir si ça marche bien.
    auto get_hpars_and_var = [Dopt, &hpars_z_guess](VectorXd const &X)
    {
      VectorXd h = Dopt.EvaluateHparOpt(X);
      VectorXd v = Dopt.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };

    auto v = evaluate_surrogate_bof(thetas_test, hpars_test, get_hpars_and_var);

    cout << "final score best hgps : " << v.transpose() << endl;

    // je calcule toutes les prédictions du GP sur les points de test.
    vector<VectorXd> preds_test;
    vector<VectorXd> stds_test;
    for (int i = 0; i < thetas_test.size(); i++)
    {
      auto p = get_hpars_and_var(thetas_test[i]);
      preds_test.push_back(p.first);
      VectorXd v = p.second.array().sqrt();
      stds_test.push_back(v);
    }
    string fnname = "results/hgps/best.gnu";
    WriteVectors(thetas_test, preds_test, stds_test, fnname);
  }

  exit(0);
}
