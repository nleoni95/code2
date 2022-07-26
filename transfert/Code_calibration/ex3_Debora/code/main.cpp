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

typedef map<int, VectorXd> map_doe;             // key : int, value : vectorXd
typedef map<int, vector<VectorXd>> map_results; // key : int, value : vecteur de VectorXd
typedef map<string, VectorXd> map_exp;          // contient les valeurs expérimentales

double const flux_nominal = 128790;

/* Functions for reading the data files*/
int line_count(string const &filename)
{
  // renvoie le nombre de lignes dans un fichier
  ifstream ifile(filename);
  int nlines = 0;
  if (ifile)
  {
    ifile.unsetf(ios_base::skipws);
    nlines = count(istream_iterator<char>(ifile), istream_iterator<char>(), '\n');
  }
  return nlines;
}

map_doe read_doe(string const &filename)
{
  // lecture du DoE à filename et écriture dans la map
  map_doe m;
  ifstream ifile(filename);
  if (ifile)
  {
    string line;
    while (getline(ifile, line))
    {
      if (line[0] == '#')
      {
        continue;
      }
      if (line.empty())
      {
        continue;
      }
      // décomposition de la line en des mots
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>()); // on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      // traitement des mots. Le premier est le num du cas, les 3-4-5-6-7 sont les paramètres.
      VectorXd param(5);
      for (int i = 2; i < 7; i++)
      {
        param(i - 2) = stod(words[i]);
      }
      param(0) /= flux_nominal;
      int key = stoi(words[0]);
      m.insert(make_pair(key, param));
    }
  }
  else
  {
    cerr << "DoE file doesn't exist" << endl;
  }
  cout << " Size of initial DoE: " << m.size() << " points." << endl;

  return m;
}

vector<VectorXd> read_singleresult(string const &filename)
{
  vector<VectorXd> v(5);
  int nlines = line_count(filename) - 1; // ignorer la 1ère ligne
  int current_line = 0;
  VectorXd X(40);
  VectorXd Alpha(40);
  VectorXd D(40);
  VectorXd V1(40);
  VectorXd V2(40);
  ifstream ifile(filename);
  if (ifile)
  {
    string line;
    while (getline(ifile, line))
    {
      if (line[0] == 'X')
      {
        continue;
      }
      // décomposition de la line en mots
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>()); // on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      // traitement des mots.
      X(current_line) = stod(words[0]);
      Alpha(current_line) = stod(words[1]);
      D(current_line) = stod(words[2]);
      V1(current_line) = stod(words[3]);
      V2(current_line) = stod(words[4]);
      current_line++;
    }
  }
  else
  {
    vector<VectorXd> vempty;
    return vempty;
  }
  v[0] = X;
  v[1] = Alpha;
  v[2] = D;
  v[3] = V1;
  v[4] = V2;
  return v;
}

map_results read_results_qmc(string const &filename)
{
  map_results m;
  for (int i = 1; i < 2041; i++)
  {
    string fullname = "../data/qmc/" + to_string(i) + "/" + filename;
    vector<VectorXd> v = read_singleresult(fullname);
    if (!v.empty())
    {
      m.insert(make_pair(i, v));
    }
  }
  cout << m.size() << " simulations read." << endl;
  return m;
}

map_results read_results_lhs(string const &filename)
{
  map_results m;
  for (int i = 1; i < 2041; i++)
  {
    string fullname = "../data/lhs/" + to_string(i) + "/" + filename;
    vector<VectorXd> v = read_singleresult(fullname);
    if (!v.empty())
    {
      m.insert(make_pair(i, v));
    }
  }
  cout << m.size() << " simulations read." << endl;
  return m;
}

map_exp read_exp_data(string const &filename)
{
  map_exp m;
  int nlines = line_count(filename) - 1;
  int current_line = 0;
  VectorXd X(49);
  VectorXd alpha(49);
  VectorXd D(49);
  VectorXd V(49);
  ifstream ifile(filename);
  if (ifile)
  {
    string line;
    while (getline(ifile, line))
    {
      if (line[0] == 'p')
      {
        continue;
      }
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>()); // on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      // traitement des mots.
      X(current_line) = stod(words[0]);
      alpha(current_line) = stod(words[1]);
      D(current_line) = stod(words[2]);
      V(current_line) = stod(words[3]);
      current_line++;
    }
  }
  //écriture des données dans la map
  m.insert(make_pair("X", X));
  m.insert(make_pair("Alpha", alpha));
  m.insert(make_pair("D", D));
  m.insert(make_pair("V", V));
  return m;
}

/*Functions for calibration*/

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // Noyau pour l'erreur de modèle.
  // hpars: sigma edm, sigma obs, lcor.
  double d1 = abs(x(0) - y(0)) / exp(hpar(2));
  double cor = pow(exp(hpar(0)), 2);
  cor *= (1 + sqrt(5) * d1 + 5 * pow(d1, 2) / 3) * exp(-sqrt(5) * d1); // x1
  return cor;
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double d1 = abs(x(0) - y(0)) / hpar(1);
  double d2 = abs(x(1) - y(1)) / hpar(3);
  double d3 = abs(x(2) - y(2)) / hpar(4);
  double d4 = abs(x(3) - y(3)) / hpar(5);
  double d5 = abs(x(4) - y(4)) / hpar(6);
  double cor = -d1 - d2 - d3 - d4 - d5;
  cor = exp(cor) * (1 + d1) * (1 + d2) * (1 + d3) * (1 + d4) * (1 + d5);
  return pow(hpar(0), 2) * cor;
}

double DKernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar, int p)
{
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
  else if (p <= 6)
  { // dérivée par rapport à une longueur de corrélation
    double cor = pow(hpar(0), 2);
    VectorXd d(5);
    d(0) = abs(x(0) - y(0)) / hpar(1);
    for (int i = 1; i < 5; i++)
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
    for (int i = 3; i < 7; i++)
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

double logprior_hpars(VectorXd const &hpars)
{
  return 0;
}

double logprior_pars(VectorXd const &pars)
{
  return 0;
}

VectorXd RtoGP(const VectorXd &X)
{
  // Linear transformation to real parameter space to GP space in [0,1]^5.
  if (X(0) < 0.9 || X(0) > 1.1)
  {
    cerr << "erreur de dimension rtogp 0" << endl;
  }
  if (X.size() != 5)
  {
    cerr << "erreur de dimension rtogp" << endl;
  }
  VectorXd Xgp(5);
  Xgp(0) = (X(0) - 0.9) / 0.2;
  list<int> l = {1, 2, 3, 4};
  for (int i : l)
  {
    if (X(i) > 2 || X(i) < 0.5)
    {
      cerr << "erreur de dimension rtogp" << i << " : " << X(i) << endl;
    }
    Xgp(i) = (X(i) - 0.5) / 1.5;
  }
  return Xgp;
}

VectorXd GPtoR(const VectorXd &Xgp)
{
  // reverse transformation
  if (Xgp(0) < 0 || Xgp(0) > 1)
  {
    cerr << "erreur de dimension gptor" << endl;
  }
  if (Xgp.size() != 5)
  {
    cerr << "erreur de dimension gptor" << endl;
  }
  VectorXd X(5);
  X(0) = 0.9 + 0.2 * Xgp(0);
  list<int> l2 = {1, 2, 3, 4};
  for (int i : l2)
  {
    if (X(i) > 1 || X(i) < 0)
    {
      cerr << "erreur de dimension gptor X(i)" << X(i) << endl;
    }
    X(i) = 0.5 + 1.5 * Xgp(i);
  }
  return X;
}

vector<vector<DATA>> PerformPCA(map_doe const &m, map_results const &mr, int qte, MatrixXd &VP, MatrixXd &Acoefs, VectorXd &featureMeans, int nmodes)
{
  int ncalcs = mr.size();
  int nrayons = mr.at(1)[0].size();
  MatrixXd U(nrayons, ncalcs);
  MatrixXd P(5, ncalcs);
  for (int i = 0; i < ncalcs; i++)
  {
    auto it = next(mr.cbegin(), i);
    U.col(i) = (*it).second[qte];
    P.col(i) = RtoGP(m.at((*it).first));
  }
  featureMeans = U.rowwise().mean();
  U = U.colwise() - featureMeans;
  MatrixXd Covmatrix = U * U.transpose();
  Covmatrix /= (ncalcs);
  SelfAdjointEigenSolver<MatrixXd> eig(Covmatrix);
  VectorXd lambdas = eig.eigenvalues();
  MatrixXd vecpropres = eig.eigenvectors();
  MatrixXd VPs = vecpropres.rightCols(nmodes);
  VectorXd lambdas_red = lambdas.bottomRows(nmodes);
  lambdas_red.reverseInPlace();
  VP = VPs.rowwise().reverse();
  cout << "Sélection de " << nmodes << " modes." << endl;
  cout << "VP principales : " << lambdas_red.transpose() << endl;
  cout << "Quantité d'énergie conservée : " << 100 * lambdas_red.array().sum() / lambdas.array().sum() << " %" << endl;
  MatrixXd A = VP.transpose() * U;
  VectorXd Ascale = lambdas_red.array().sqrt();
  Acoefs = Ascale.asDiagonal();
  MatrixXd normedA = Acoefs.inverse() * A;
  vector<vector<DATA>> vd(nmodes);
  for (int j = 0; j < nmodes; j++)
  {
    vector<DATA> v(ncalcs);
    for (int i = 0; i < ncalcs; i++)
    {
      DATA dat;
      dat.SetX(P.col(i));
      dat.SetValue(normedA(j, i));
      v[i] = dat;
    }
    vd[j] = v;
  }
  return vd;
}

VectorXd EvaluateMeanGPPCA(vector<GP> const &vgp, VectorXd const &Target, MatrixXd const &VP, MatrixXd const &Acoefs, VectorXd const &featureMeans)
{
  int nmodes = Acoefs.cols();
  int nrayons = VP.rows();
  VectorXd meansgps(vgp.size());
  VectorXd varsgps(vgp.size());
  for (int i = 0; i < vgp.size(); i++)
  {
    meansgps(i) = vgp[i].EvalMean(Target);
  }
  return featureMeans + VP * Acoefs * meansgps;
}

VectorXd eval_erreur_validation(MatrixXd const &M_truth, MatrixXd const &M_projected, MatrixXd const &M_predicted)
{
  //évaluation des erreurs de validation et répartition de l'erreur.
  int ncalcs = M_truth.cols();
  auto ps = [](MatrixXd const &A, MatrixXd const &B) -> double
  {
    return (A.transpose() * B).trace();
  };
  double disttruth_proj = sqrt(ps(M_truth - M_projected, M_truth - M_projected));
  double distproj_GP = sqrt(ps(M_predicted - M_projected, M_predicted - M_projected));
  double disttotale = sqrt(ps(M_predicted - M_truth, M_predicted - M_truth));
  double prop_err_projection = pow(disttruth_proj, 2) / pow(disttotale, 2);
  double prop_err_GP = pow(distproj_GP, 2) / pow(disttotale, 2);
  cout << "répartition de l'erreur entre projection et GP : " << prop_err_projection << ", " << prop_err_GP << endl;
  double pct_moyen_erreur = 100 * disttotale / sqrt(ps(M_truth, M_truth));
  cout << "pct moyen erreur L2 : " << pct_moyen_erreur << endl;
  VectorXd err(3);
  err << prop_err_projection, prop_err_GP, pct_moyen_erreur;
  return err;
}

VectorXd compute_erreurs_validation(int qte, map_doe const &m_lhs, map_results const &mr_lhs, vector<GP> const &vgp, MatrixXd const &VP, MatrixXd const &Acoefs, VectorXd const &featureMeans)
{
  int ncalcs = mr_lhs.size();
  int nmodes = Acoefs.cols();
  int nrayons = VP.rows();
  MatrixXd M_truth(nrayons, ncalcs);
  MatrixXd P_truth(5, ncalcs);
  for (int i = 0; i < ncalcs; i++)
  {
    auto it = next(mr_lhs.cbegin(), i);
    M_truth.col(i) = (*it).second[qte];
    P_truth.col(i) = RtoGP(m_lhs.at((*it).first));
  }
  MatrixXd M_projected(nrayons, ncalcs);
  MatrixXd M_truth_centered = M_truth.colwise() - featureMeans;
  MatrixXd M_truth_multiplied = VP * VP.transpose() * (M_truth_centered);
  M_projected = (M_truth_multiplied).colwise() + featureMeans;
  MatrixXd M_predicted(nrayons, ncalcs);
  for (int i = 0; i < ncalcs; i++)
  {
    VectorXd ParamEval = P_truth.col(i);
    M_predicted.col(i) = EvaluateMeanGPPCA(vgp, ParamEval, VP, Acoefs, featureMeans);
  }
  auto afficher_erreurs = [M_truth, M_projected, M_predicted, ncalcs](int nstart, int nend)
  {
    MatrixXd M_truth_2 = M_truth.block(nstart, 0, nend - nstart + 1, ncalcs);
    MatrixXd M_projected_2 = M_projected.block(nstart, 0, nend - nstart + 1, ncalcs);
    MatrixXd M_predicted_2 = M_predicted.block(nstart, 0, nend - nstart + 1, ncalcs);
    VectorXd v = eval_erreur_validation(M_truth_2, M_projected_2, M_predicted_2);
    return v;
  };
  cout << "sur tout le domaine : " << endl;
  return afficher_erreurs(0, nrayons - 1);
}

VectorXd interpolate(VectorXd const &Yorig, VectorXd const &Xorig, VectorXd const &Xnew)
{
  // interpolate the results of numerical simulations over the experimental locations
  if (Yorig.size() != Xorig.size())
  {
    cerr << "erreur d'interpolation : taille différente." << Yorig.size() << " " << Xorig.size() << endl;
  }
  VectorXd Ynew(Xnew.size());
  for (int i = 0; i < Xnew.size(); i++)
  {
    double ynext = 0;
    double yprev = 0;
    double xnext = 0;
    double xprev = 0;
    if (Xnew(i) < Xorig(0))
    {
      ynext = Yorig(0);
      xnext = Xorig(0);
      xprev = 2 * Xnew(i) - Xorig(0);
      double slope = (Yorig(1) - Yorig(0)) / (Xorig(1) - Xorig(0));
      yprev = ynext - slope * (xnext - xprev);
    }
    else if (Xnew(i) > Xorig(Xorig.size() - 1))
    {
      yprev = Yorig(Xorig.size() - 1);
      xprev = Xorig(Xorig.size() - 1);
      xnext = 2 * Xnew(i) - xprev;
      double slope = (Yorig(Xorig.size() - 1) - Yorig(Xorig.size() - 2)) / (Xorig(Xorig.size() - 1) - Xorig(Xorig.size() - 2));
      ynext = yprev - slope * (xprev - xnext);
    }
    else
    {
      int indice = 0;
      while (Xnew(i) > Xorig(indice))
      {
        indice++;
      }
      ynext = Yorig(indice);
      xnext = Xorig(indice);
      yprev = Yorig(indice - 1);
      xprev = Xorig(indice - 1);
    }
    double m = (ynext - yprev) / (xnext - xprev);
    double b = ynext - m * xnext;
    Ynew(i) = m * Xnew(i) + b;
  }
  return Ynew;
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

void Nodups(std::vector<VectorXd> &v)
{
  // remove duplicate elements in a vector
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
  VectorXd MAP(5);
  MAP << 0.2, 0.53, 0.27, 0.9, 0.47;
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

int main(int argc, char **argv)
{

  std::default_random_engine generator;
  generator.seed(16031995);
  map_doe m = read_doe("design_qmc_full.dat");
  map_doe m_lhs = read_doe("design_lhs_full.dat");
  map_results mr = read_results_qmc("clean_profile.dat");
  map_results mr_lhs = read_results_lhs("clean_profile.dat");
  map_exp me = read_exp_data("../data/exp/clean_exp.dat");

  /* Beginning of surrogate construction for Neptune_CFD
  The corresponding GPs are already optimized, with hyperparameters stored in results/hpars_gp_deb_alpha.gnu and results/hpars_gp_deb_diam.gnu.
  The code reads from these files.*/

  double ftol_rel_gp_neptune = 1e-4;
  /*surrogate for the void fraction*/
  int nmodes_gp_alpha = 4;
  vector<GP> vgp_a;
  vector<VectorXd> hpars_gp_deb_alpha;
  MatrixXd VP_a;
  MatrixXd Acoefs_a;
  VectorXd featureMeans_a;
  {
    int ncalcs = mr.size();
    int nrayons = mr.at(1)[0].size();
    MatrixXd VP(nrayons, nmodes_gp_alpha);
    MatrixXd Acoefs(nmodes_gp_alpha, nmodes_gp_alpha);
    VectorXd featureMeans(nrayons);
    vector<vector<DATA>> full_data = PerformPCA(m, mr, 1, VP, Acoefs, featureMeans, nmodes_gp_alpha);
    int nhpars_gp = 7;
    MatrixXd Bounds_hpars_gp(2, nhpars_gp);
    Bounds_hpars_gp(0, 0) = 1E-3;
    Bounds_hpars_gp(1, 0) = 1e2;
    Bounds_hpars_gp(0, 2) = 1E-3;
    Bounds_hpars_gp(1, 2) = 2E-3;
    list<int> l = {1, 3, 4, 5, 6};
    for (int i : l)
    {
      Bounds_hpars_gp(0, i) = 1E-2;
      Bounds_hpars_gp(1, i) = 2;
    }
    VectorXd hpars_gp_guess(7);
    for (int i = 0; i < nhpars_gp; i++)
    {
      hpars_gp_guess(i) = 0.5 * (Bounds_hpars_gp(1, i) + Bounds_hpars_gp(0, i));
    }
    hpars_gp_deb_alpha = ReadVector("hpars_gp_deb_alpha.gnu");
    for (int i = 0; i < nmodes_gp_alpha; i++)
    {
      GP gp(Kernel_GP_Matern32);
      gp.SetDKernel(DKernel_GP_Matern32);
      gp.SetData(full_data[i]);
      gp.SetGP(hpars_gp_deb_alpha[i]);
      vgp_a.push_back(gp);
    }
    VP_a = VP;
    Acoefs_a = Acoefs;
    featureMeans_a = featureMeans;
  }

  /*surrogate for the bubble diameter*/
  int nmodes_gp_diam = 6;
  vector<GP> vgp_d;
  vector<VectorXd> hpars_gp_deb_diam;
  MatrixXd VP_d;
  MatrixXd Acoefs_d;
  VectorXd featureMeans_d;
  {
    int ncalcs = mr.size();
    int nrayons = mr.at(1)[0].size();
    MatrixXd VP(nrayons, nmodes_gp_diam);
    MatrixXd Acoefs(nmodes_gp_diam, nmodes_gp_diam);
    VectorXd featureMeans(nrayons);
    vector<vector<DATA>> full_data = PerformPCA(m, mr, 2, VP, Acoefs, featureMeans, nmodes_gp_diam);
    int nhpars_gp = 7;
    MatrixXd Bounds_hpars_gp(2, nhpars_gp);
    Bounds_hpars_gp(0, 0) = 1E-3;
    Bounds_hpars_gp(1, 0) = 1e2;
    Bounds_hpars_gp(0, 2) = 1E-3;
    Bounds_hpars_gp(1, 2) = 2E-3;
    list<int> l = {1, 3, 4, 5, 6};
    for (int i : l)
    {
      Bounds_hpars_gp(0, i) = 1E-2;
      Bounds_hpars_gp(1, i) = 2;
    }
    VectorXd hpars_gp_guess(7);
    for (int i = 0; i < nhpars_gp; i++)
    {
      hpars_gp_guess(i) = 0.5 * (Bounds_hpars_gp(1, i) + Bounds_hpars_gp(0, i));
    }
    hpars_gp_deb_diam = ReadVector("hpars_gp_deb_diam.gnu");
    for (int i = 0; i < nmodes_gp_diam; i++)
    {
      GP gp(Kernel_GP_Matern32);
      gp.SetDKernel(DKernel_GP_Matern32);
      gp.SetData(full_data[i]);
      gp.SetGP(hpars_gp_deb_diam[i]);
      vgp_d.push_back(gp);
    }
    VP_d = VP;
    Acoefs_d = Acoefs;
    featureMeans_d = featureMeans;
  }

  /*End of surrogate construction*/
  /*Interpolate CFD data to experimental measurements*/
  VectorXd Xexpe = me["X"];
  VectorXd Yexpe_diam = me["D"];
  VectorXd Yexpe_alpha = me["Alpha"];
  VectorXd Xgrid_num = mr[1][0];
  VectorXd Yexpe_interpol_diam = interpolate(Yexpe_diam, Xexpe, Xgrid_num);
  VectorXd Yexpe_interpol_alpha = interpolate(Yexpe_alpha, Xexpe, Xgrid_num);
  int ndata = Xgrid_num.size();
  vector<VectorXd> Xgrid_exp(ndata);
  for (int i = 0; i < ndata; i++)
  {
    VectorXd xobs(1);
    xobs << Xgrid_num(i);
    Xgrid_exp[i] = xobs;
  }
  string obsfile_alpha = "results/obsalpha.gnu";
  string obsfile_diam = "results/obsdiam.gnu";
  WriteObs(Xgrid_exp, Yexpe_interpol_alpha, obsfile_alpha);
  WriteObs(Xgrid_exp, Yexpe_interpol_diam, obsfile_diam);

  int dim_theta = 5;
  int dim_hpars = 3;

  /* Bounds for parameters and hyperparameters */
  VectorXd lb_t(dim_theta);
  VectorXd ub_t(dim_theta);
  for (int i = 0; i < dim_theta; i++)
  {
    lb_t(i) = 0;
    ub_t(i) = 1;
  }
  VectorXd lb_hpars_alpha(dim_hpars);
  VectorXd ub_hpars_alpha(dim_hpars);
  lb_hpars_alpha << 1e-4, 1e-4, 1e-4;
  ub_hpars_alpha << 1, 1e-1, 1e-1;
  VectorXd lb_hpars_diam(dim_hpars);
  VectorXd ub_hpars_diam(dim_hpars);
  lb_hpars_diam << 2e-5, 1e-8, 5e-4;
  ub_hpars_diam << 5e-3, 3e-5, 1e-1;
  lb_hpars_alpha.array() = lb_hpars_alpha.array().log();
  ub_hpars_alpha.array() = ub_hpars_alpha.array().log();
  lb_hpars_diam.array() = lb_hpars_diam.array().log();
  ub_hpars_diam.array() = ub_hpars_diam.array().log();
  VectorXd hpars_z_guess_alpha = 0.5 * (lb_hpars_alpha + ub_hpars_alpha);
  VectorXd hpars_z_guess_diam = 0.5 * (lb_hpars_diam + ub_hpars_diam);

  /*Bounds for hGP hyperparameters*/
  MatrixXd Bounds_hpars_HGPs(2, 7); // 7 hpars HGPs. std, noise, 3 longueurs de corrélation.
  Bounds_hpars_HGPs(0, 0) = 0.1;
  Bounds_hpars_HGPs(1, 0) = 10;
  for (int i = 1; i < 7; i++)
  {
    Bounds_hpars_HGPs(0, i) = 1e-1;
    Bounds_hpars_HGPs(1, i) = 10;
  }
  Bounds_hpars_HGPs(0, 2) = 1e-5; // stdnoise
  Bounds_hpars_HGPs(1, 2) = 1e-1;
  VectorXd Hpars_guess_HGPs = 0.5 * (Bounds_hpars_HGPs.row(0) + Bounds_hpars_HGPs.row(1)).transpose();
  vector<VectorXd> v_hpars_guess_hgps;
  for (int i = 0; i < lb_hpars_alpha.size() + lb_hpars_diam.size(); i++)
  {
    v_hpars_guess_hgps.push_back(Hpars_guess_HGPs);
  }
  /*Define prior mean, and model F for both Density objects.*/
  auto lambda_priormean_alpha = [](vector<VectorXd> const &X, VectorXd const &hpars)
  {
    VectorXd b = VectorXd::Zero(X.size());
    return b;
  };
  auto lambda_priormean_diam = [](vector<VectorXd> const &X, VectorXd const &hpars)
  {
    VectorXd b = VectorXd::Zero(X.size());
    return b;
  };

  auto lambda_model_diam = [&vgp_d, &VP_d, &featureMeans_d, &Acoefs_d](vector<VectorXd> const &X, VectorXd const &theta) -> VectorXd
  {
    return EvaluateMeanGPPCA(vgp_d, theta, VP_d, Acoefs_d, featureMeans_d);
  };

  auto lambda_model_alpha = [&vgp_a, &VP_a, &featureMeans_a, &Acoefs_a](vector<VectorXd> const &X, VectorXd const &theta) -> VectorXd
  {
    return EvaluateMeanGPPCA(vgp_a, theta, VP_a, Acoefs_a, featureMeans_a);
  };

  /* MCMC parameters */
  MatrixXd COV_init = pow(0.1, 2) * MatrixXd::Identity(5, 5);
  VectorXd Xinit(5);
  Xinit << 0.5, 0.5, 0.5, 0.5, 0.5;

  int nombre_steps_mcmc = 1e4;
  int nombre_samples_collected = 500;
  int nautocor = 2000;
  int npts_init = 50;
  DoE doe_init(lb_t, ub_t, npts_init, 1);

  Density MainDensity_alpha(doe_init);
  MainDensity_alpha.SetLogPriorPars(logprior_pars);
  MainDensity_alpha.SetLogPriorHpars(logprior_hpars);
  MainDensity_alpha.SetZKernel(Kernel_Z_Matern52);
  MainDensity_alpha.SetFModel(lambda_model_alpha);
  MainDensity_alpha.SetZPriorMean(lambda_priormean_alpha);
  MainDensity_alpha.SetHparsBounds(lb_hpars_alpha, ub_hpars_alpha);
  MainDensity_alpha.SetObservations(Xgrid_exp, Yexpe_interpol_alpha);
  MainDensity_alpha.SetLearnedOutputerr(1);

  Density MainDensity_diam(doe_init);
  MainDensity_diam.SetLogPriorPars(logprior_pars);
  MainDensity_diam.SetLogPriorHpars(logprior_hpars);
  MainDensity_diam.SetZKernel(Kernel_Z_Matern52);
  MainDensity_diam.SetFModel(lambda_model_diam);
  MainDensity_diam.SetZPriorMean(lambda_priormean_diam);
  MainDensity_diam.SetHparsBounds(lb_hpars_diam, ub_hpars_diam);
  MainDensity_diam.SetObservations(Xgrid_exp, Yexpe_interpol_diam);
  MainDensity_diam.SetLearnedOutputerr(1);

  double ftol_rel_fmp = 1e-12;
  double ftol_rel_hgps = 1e-5;
  double ftol_calc_map = 1e-4;

  /* Beginning of calibration */

  /* full FMP calibration */
  {
    // paramètres de l'algorithme
    string foldname = "fmp";
    cout << "début double calibration fmp avec nsteps =" << nombre_steps_mcmc << endl;
    DensityOpt Dopt_alpha(MainDensity_alpha);
    DensityOpt Dopt_diam(MainDensity_diam);
    VectorXd times = VectorXd::Zero(1);

    // début de l'algorithme
    ofstream timefile("results/" + foldname + "/time.gnu");
    timefile << "#Columns : 1. Coût total (ms)" << endl;

    // calibration finale
    auto get_hpars_opti = [&Dopt_alpha, &Dopt_diam, &hpars_z_guess_alpha, &hpars_z_guess_diam, &ftol_rel_fmp](VectorXd const &X)
    {
      vector<VectorXd> p(2);
      p[0] = Dopt_alpha.HparsOpt(X, hpars_z_guess_alpha, ftol_rel_fmp);
      p[1] = Dopt_diam.HparsOpt(X, hpars_z_guess_diam, ftol_rel_fmp);
      return p;
    };
    auto compute_score_opti = [&Dopt_alpha, &Dopt_diam](vector<VectorXd> p, VectorXd const &X)
    {
      double ll1 = Dopt_alpha.loglikelihood_theta(X, p[0]);
      double ll2 = Dopt_diam.loglikelihood_theta(X, p[1]);
      return ll1 + ll2 + logprior_pars(X);
    };
    auto in_bounds = [&Dopt_alpha](VectorXd const &X)
    {
      return Dopt_alpha.in_bounds_pars(X);
    };

    auto begin = chrono::steady_clock::now();
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, Xinit, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsofsamples_alpha(nombre_samples_collected);
    vector<VectorXd> hparsofsamples_diam(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsofsamples_alpha[i] = get_hpars_opti(samples[i])[0];
      hparsofsamples_diam[i] = get_hpars_opti(samples[i])[1];
    }
    auto end = chrono::steady_clock::now();
    times(0) = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
    timefile << times(0) << endl;
    ofstream fsamp("results/" + foldname + "/samples.gnu");
    WriteVector(samples, fsamp);
    double corlength = Selfcor_diagnosis(visited_steps, nautocor, 1, "results/" + foldname + "/autocor.gnu");

    //écriture MAP, moyenne et var calculées sur toute la chaîne.
    auto score = [&get_hpars_opti, &compute_score_opti](VectorXd const &X)
    {
      return compute_score_opti(get_hpars_opti(X), X);
    };
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
    string fnameMAP = "results/" + foldname + "/map.gnu";
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
    /* compute predictions of the model at MAP */
    VectorXd fpredalpha = lambda_model_alpha(Xgrid_exp, map);
    VectorXd fpreddiam = lambda_model_diam(Xgrid_exp, map);
    string fnamepred = "results/" + foldname + "/predmap.gnu";
    ofstream opred(fnamepred);
    for (int i = 0; i < Xgrid_exp.size(); i++)
    {
      opred << Xgrid_exp[i](0) << " " << fpredalpha(i) << " " << fpreddiam(i) << endl;
    }
  }

  // HP-AS calibration
  {
    string foldname = "hpas";
    // paramètres de l'algorithme
    int ninit = 50;
    int ndraws = 100;                       // nombre de répétitions pour estimer la variance lors du calcul des poids.
    int nmax = 250;                         // nombre de points total dans le surrogate
    int npts_per_iter = 100;                // nombre de pts rajoutés à chaque itération
    int nsamples_mcmc = 20 * npts_per_iter; // samples MCMC à chaque itération
    int nsteps_mcmc = 100 * nsamples_mcmc;  // longueur MCMC à chaque itération

    cout << "début calibration HP-AS" << endl;
    auto begin = chrono::steady_clock::now();
    DensityOpt Dopt_alpha(MainDensity_alpha);
    DensityOpt Dopt_diam(MainDensity_diam);
    VectorXd times = VectorXd::Zero(5);
    vector<VectorXd> thetas_training;
    vector<VectorXd> hpars_training_alpha;
    vector<VectorXd> hpars_training_diam;

    DoE doe_lhs_initial(lb_t, ub_t, ninit, generator);
    auto t1 = chrono::steady_clock::now();
    for (const auto theta : doe_lhs_initial.GetGrid())
    {
      VectorXd hpars_alpha = Dopt_alpha.HparsOpt(theta, hpars_z_guess_alpha, ftol_rel_fmp);
      VectorXd hpars_diam = Dopt_diam.HparsOpt(theta, hpars_z_guess_diam, ftol_rel_fmp);
      thetas_training.push_back(theta);
      hpars_training_alpha.push_back(hpars_alpha);
      hpars_training_diam.push_back(hpars_diam);
    }
    auto t2 = chrono::steady_clock::now();
    times(2) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count(); // mesure du temps pour les optimisations initiales.

    // lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
    auto add_points = [&Dopt_alpha, &Dopt_diam, nsteps_mcmc, &Xinit, &COV_init, &generator, &hpars_z_guess_alpha, &hpars_z_guess_diam, &thetas_training, &hpars_training_alpha, &hpars_training_diam, ftol_rel_fmp, ndraws, nsamples_mcmc, npts_per_iter](string foldname, VectorXd &times)
    {
      auto get_hpars_opti = [&Dopt_alpha, &Dopt_diam](VectorXd const &X)
      {
        vector<VectorXd> p(2);
        p[0] = Dopt_alpha.EvaluateHparOpt(X);
        p[1] = Dopt_diam.EvaluateHparOpt(X);
        return p;
      };
      auto compute_score_opti = [&Dopt_alpha, &Dopt_diam](vector<VectorXd> p, VectorXd const &X)
      {
        // version sans prior des paramètres
        double ll1 = Dopt_alpha.loglikelihood_theta(X, p[0]);
        double ll2 = Dopt_diam.loglikelihood_theta(X, p[1]);
        return ll1 + ll2 + logprior_pars(X);
      };
      auto in_bounds = [&Dopt_alpha](VectorXd const &X)
      {
        return Dopt_alpha.in_bounds_pars(X);
      };
      auto t1 = chrono::steady_clock::now();
      vector<VectorXd> allsteps = Run_MCMC(nsteps_mcmc, Xinit, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
      vector<VectorXd> candidate_thetas;
      for (int i = 0; i < nsamples_mcmc; i++)
      {
        candidate_thetas.push_back(allsteps[i * (allsteps.size() / nsamples_mcmc)]);
      }
      Nodups(candidate_thetas);
      auto t2 = chrono::steady_clock::now();
      times(0) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
      Selfcor_diagnosis(allsteps, 5000, 1, "results/" + foldname + "/autocor.gnu");
      vector<VectorXd> selected_thetas(npts_per_iter);
      vector<double> weights(candidate_thetas.size());
      // weighted resampling
      auto begin = chrono::steady_clock::now();
      vector<vector<VectorXd>> draws_alpha = Dopt_alpha.SampleHparsOpt(candidate_thetas, ndraws, generator);
      vector<vector<VectorXd>> draws_diam = Dopt_diam.SampleHparsOpt(candidate_thetas, ndraws, generator);
      for (int j = 0; j < weights.size(); j++)
      {
        vector<double> scores;
        for (int k = 0; k < ndraws; k++)
        {
          double ll1 = Dopt_alpha.loglikelihood_theta(candidate_thetas[j], draws_alpha[j][k]);
          double ll2 = Dopt_diam.loglikelihood_theta(candidate_thetas[j], draws_diam[j][k]);
          scores.push_back(ll1 + ll2);
        }
        double mean = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
        double sq_sum = std::inner_product(scores.begin(), scores.end(), scores.begin(), 0.0);
        double var = sq_sum / scores.size() - mean * mean;
        weights[j] = var;
      }
      auto t3 = chrono::steady_clock::now();
      times(1) = chrono::duration_cast<chrono::milliseconds>(t3 - begin).count();
      for (int i = 0; i < npts_per_iter; i++)
      {
        std::discrete_distribution<int> distribution(weights.begin(), weights.end());
        int drawn = distribution(generator);
        weights[drawn] = 0;
        selected_thetas[i] = candidate_thetas[drawn];
        VectorXd hpars_alpha = Dopt_alpha.HparsOpt(selected_thetas[i], hpars_z_guess_alpha, ftol_rel_fmp);
        VectorXd hpars_diam = Dopt_diam.HparsOpt(selected_thetas[i], hpars_z_guess_diam, ftol_rel_fmp);
        thetas_training.push_back(selected_thetas[i]);
        hpars_training_alpha.push_back(hpars_alpha);
        hpars_training_diam.push_back(hpars_diam);
      }
      cout.clear();
      auto end = chrono::steady_clock::now();
      times(2) = chrono::duration_cast<chrono::milliseconds>(end - t3).count();
      cout << "time for resampling step : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds." << endl;
    };

    // début de l'algorithme
    ofstream timefile("results/" + foldname + "/time.gnu");
    timefile << "#Columns : 1. npts 2. MCMC, 3. Resampling, 4. Optimisation des nouveaux points, 5. Mise à jour des GPs, 5. Coût total (1+2+3+4)" << endl;
    vector<VectorXd> hpars_opt_hgps_alpha;
    vector<VectorXd> hpars_opt_hgps_diam;

    // contruction initiale des hGPs
    {
      // build GPs
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      auto t1 = chrono::steady_clock::now();
      Dopt_alpha.BuildHGPs(thetas_training, hpars_training_alpha, Kernel_GP_Matern32, DKernel_GP_Matern32);
      Dopt_diam.BuildHGPs(thetas_training, hpars_training_diam, Kernel_GP_Matern32, DKernel_GP_Matern32);
      Dopt_alpha.OptimizeHGPs(Bounds_hpars_HGPs, v_hpars_guess_hgps, ftol_rel_hgps);
      Dopt_diam.OptimizeHGPs(Bounds_hpars_HGPs, v_hpars_guess_hgps, ftol_rel_hgps);
      hpars_opt_hgps_alpha = Dopt_alpha.GetHparsHGPs();
      hpars_opt_hgps_diam = Dopt_diam.GetHparsHGPs();
      auto t2 = chrono::steady_clock::now();
      times(3) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
      times(4) += times(0) + times(1) + times(2) + times(3);
      timefile << thetas_training.size() << " ";
      for (int o = 0; o < times.size(); o++)
      {
        timefile << times(o) << " ";
      }
      timefile << endl;
    }
    // construction des hGPs
    while (thetas_training.size() < nmax)
    {
      // ajout de nouveaux training points
      add_points(foldname, times);
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      auto t1 = chrono::steady_clock::now();
      Dopt_alpha.BuildHGPs(thetas_training, hpars_training_alpha, Kernel_GP_Matern32, DKernel_GP_Matern32);
      Dopt_diam.BuildHGPs(thetas_training, hpars_training_diam, Kernel_GP_Matern32, DKernel_GP_Matern32);
      Dopt_alpha.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps_alpha, ftol_rel_hgps);
      Dopt_diam.OptimizeHGPs(Bounds_hpars_HGPs, hpars_opt_hgps_diam, ftol_rel_hgps);
      hpars_opt_hgps_alpha = Dopt_alpha.GetHparsHGPs();
      hpars_opt_hgps_diam = Dopt_diam.GetHparsHGPs();
      auto t2 = chrono::steady_clock::now();
      times(3) = chrono::duration_cast<chrono::milliseconds>(t2 - t1).count();
      times(4) += times(0) + times(1) + times(2) + times(3);
      timefile << thetas_training.size() << " ";
      for (int o = 0; o < times.size(); o++)
      {
        timefile << times(o) << " ";
      }
      timefile << endl;
    }

    // calibration
    auto get_hpars_opti = [&Dopt_alpha, &Dopt_diam](VectorXd const &X)
    {
      vector<VectorXd> p(2);
      p[0] = Dopt_alpha.EvaluateHparOpt(X);
      p[1] = Dopt_diam.EvaluateHparOpt(X);
      return p;
    };
    auto compute_score_opti = [&Dopt_alpha, &Dopt_diam](vector<VectorXd> p, VectorXd const &X)
    {
      double ll1 = Dopt_alpha.loglikelihood_theta(X, p[0]);
      double ll2 = Dopt_diam.loglikelihood_theta(X, p[1]);
      return ll1 + ll2 + logprior_pars(X);
    };
    auto in_bounds = [&Dopt_alpha](VectorXd const &X)
    {
      return Dopt_alpha.in_bounds_pars(X);
    };
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, Xinit, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
    }
    ofstream fsamp("results/" + foldname + "/samples.gnu");
    WriteVector(samples, fsamp);
    // calcul MAP, moyenne et var sur la chaîne.
    auto score = [&get_hpars_opti, &compute_score_opti](VectorXd const &X)
    {
      return compute_score_opti(get_hpars_opti(X), X);
    };

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
    string fnameMAP = "results/" + foldname + "/map.gnu";
    ofstream omap(fnameMAP);
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
    /* compute predictions of the model at MAP */
    VectorXd fpredalpha = lambda_model_alpha(Xgrid_exp, map);
    VectorXd fpreddiam = lambda_model_diam(Xgrid_exp, map);
    string fnamepred = "results/" + foldname + "/predmap.gnu";
    ofstream opred(fnamepred);
    for (int i = 0; i < Xgrid_exp.size(); i++)
    {
      opred << Xgrid_exp[i](0) << " " << fpredalpha(i) << " " << fpreddiam(i) << endl;
    }
  }
}