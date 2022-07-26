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

int neval = 1;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0, 1);
std::normal_distribution<double> distN(0, 1);
vector<DATA> data;
vector<VectorXd> Grid;

int dim_x = 1;
int dim_theta = 3;

// tableau regroupant tous les inputs des cas expérimentaux 3,4,5,6,8, etc.
// dans l'ordre : p,v,DTsub,diam,Tsat,rhof,muf,rhog,cpf,kf,hfg,sigma,fric.
//                0,1,  2,   3,    4,   5,  6,   7,  8,  9, 10,  11,   12
map<int, vector<double>> map_expe = {
    {3, {2.000000e+00, 3.660000e+00, 2.780000e+01, 1.956000e-02, 1.202115e+02, 9.637275e+02, 3.059678e-04, 1.129006e+00, 4.207407e+03, 6.743324e-01, 2.201557e+06, 5.492552e-02, 1.527583e-02}},
    {4, {2.000000e+00, 3.000000e-01, 2.780000e+01, 1.956000e-02, 1.202115e+02, 9.637275e+02, 3.059678e-04, 1.129006e+00, 4.207407e+03, 6.743324e-01, 2.201557e+06, 5.492552e-02, 2.639084e-02}},
    {5, {2.000000e+00, 1.220000e+00, 2.780000e+01, 1.956000e-02, 1.202115e+02, 9.637275e+02, 3.059678e-04, 1.129006e+00, 4.207407e+03, 6.743324e-01, 2.201557e+06, 5.492552e-02, 1.910965e-02}},
    {6, {4.000000e+00, 1.220000e+00, 5.550000e+01, 1.956000e-02, 1.436125e+02, 9.667158e+02, 3.214146e-04, 2.162668e+00, 4.202403e+03, 6.721438e-01, 2.133333e+06, 5.009629e-02, 1.930011e-02}},
    {8, {4.000000e+00, 1.220000e+00, 8.330000e+01, 1.956000e-02, 1.436125e+02, 9.831798e+02, 4.642762e-04, 2.162668e+00, 4.182234e+03, 6.512592e-01, 2.133333e+06, 5.009629e-02, 2.084200e-02}},
    {14, {2.000000e+00, 1.220000e+00, 2.780000e+01, 1.956000e-02, 1.202115e+02, 9.637275e+02, 3.059678e-04, 1.129006e+00, 4.207407e+03, 6.743324e-01, 2.201557e+06, 5.492552e-02, 1.910965e-02}},
    {15, {4.000000e+00, 3.000000e-01, 5.550000e+01, 1.956000e-02, 1.436125e+02, 9.667158e+02, 3.214146e-04, 2.162668e+00, 4.202403e+03, 6.721438e-01, 2.133333e+06, 5.009629e-02, 2.669458e-02}},
    {16, {4.000000e+00, 3.000000e-01, 2.780000e+01, 1.956000e-02, 1.436125e+02, 9.465546e+02, 2.411237e-04, 2.162668e+00, 4.238815e+03, 6.829080e-01, 2.133333e+06, 5.009629e-02, 2.500814e-02}},
    {18, {4.000000e+00, 1.220000e+00, 1.110000e+01, 1.956000e-02, 1.436125e+02, 9.327430e+02, 2.085730e-04, 2.162668e+00, 4.269601e+03, 6.849851e-01, 2.133333e+06, 5.009629e-02, 1.774809e-02}},
    {20, {6.000000e+00, 3.000000e-01, 2.780000e+01, 1.956000e-02, 1.588324e+02, 9.341209e+02, 2.111695e-04, 3.168816e+00, 4.266018e+03, 6.850530e-01, 2.085638e+06, 4.684359e-02, 2.430147e-02}},
    {21, {6.000000e+00, 1.220000e+00, 2.780000e+01, 1.956000e-02, 1.588324e+02, 9.341209e+02, 2.111695e-04, 3.168816e+00, 4.266018e+03, 6.850530e-01, 2.085638e+06, 4.684359e-02, 1.778827e-02}},
    {22, {4.000000e+00, 3.350000e+00, 5.550000e+01, 1.067000e-02, 1.436125e+02, 9.667158e+02, 3.214146e-04, 2.162668e+00, 4.202403e+03, 6.721438e-01, 2.133333e+06, 5.009629e-02, 1.772091e-02}},
    {23, {4.000000e+00, 1.220000e+00, 5.550000e+01, 1.067000e-02, 1.436125e+02, 9.667158e+02, 3.214146e-04, 2.162668e+00, 4.202403e+03, 6.721438e-01, 2.133333e+06, 5.009629e-02, 2.207772e-02}}};

double ravik_model(double DTsup, VectorXd const &params, int case_nr)
{
  // modèle MITB
  // params de dimension 3 : en 0 l'angle de contact, en 1 et 2 les 2 coefs de la corrélation du diamètre.
  vector<double> conds = map_expe.at(case_nr);
  double Tsat = conds[4];
  double DTsub = conds[2];
  double Twall = Tsat + DTsup;
  double Tbulk = Tsat - DTsub;
  double angle = params(0) * M_PI / 180;
  double vel = conds[1];
  double p = conds[0];
  double Dh = conds[3];
  double rhof = conds[5];
  double muf = conds[6];
  double rhog = conds[7];
  double cpf = conds[8];
  double kf = conds[9];
  double hfg = conds[10];
  double sigma = conds[11];
  double Re = rhof * vel * Dh / muf;
  double Pr = muf * cpf / kf;
  double Jasub = rhof * cpf * DTsub / (rhog * hfg);
  double Jasup = rhof * cpf * DTsup / (rhog * hfg);
  double etaf = kf / (rhof * cpf);
  double fric = conds[12];
  double NuD = ((fric / 8) * (Re - 1000) * Pr) / (1 + 12.7 * sqrt(fric / 8) * (pow(Pr, 2. / 3.) - 1));
  double hfc = NuD * kf / Dh;
  double Dd = params(1) * pow(((rhof - rhog) / rhog), 0.27) * pow(Jasup, params(2)) * pow(1 + Jasub, -0.3) * pow(vel, -0.26);
  double twait = 6.1e-3 * pow(Jasub, 0.6317) / DTsup;
  double chi = max(0., 0.05 * DTsub / DTsup);
  double c1 = 1.243 / sqrt(Pr);
  double c2 = 1.954 * chi;
  double c3 = -1 * min(abs(c2), 0.5 * c1);
  double K = (c1 + c3) * Jasup * sqrt(etaf);
  double tgrowth = pow(0.25 * Dd / K, 2);
  double freq = 1. / (twait + tgrowth);
  double N0 = freq * tgrowth * 3.1415 * pow(0.5 * Dd, 2);
  // ishiihibiki
  double pp = p * 1e5;
  double Tg = Tsat + DTsup + 273.15;
  double TTsat = Tsat + 273.15;
  double rhoplus = log10((rhof - rhog) / rhog);
  double frhoplus = -0.01064 + 0.48246 * rhoplus - 0.22712 * pow(rhoplus, 2) + 0.05468 * pow(rhoplus, 3);
  double Rc = (2 * sigma * (1 + (rhog / rhof))) / (pp * (exp(hfg * (Tg - TTsat) / (462 * Tg * TTsat)) - 1));
  double Npp = (4.72E5) * (1 - exp(-(pow(angle, 2)) / (8 * (pow(0.722, 2))))) * (exp(frhoplus * (2.5E-6) / Rc) - 1);
  double Nppb;
  if (N0 * Npp < exp(-1))
  {
    Nppb = Npp;
  }
  else if (N0 * Npp < exp(1))
  {
    Nppb = (0.2689 * N0 * Npp + 0.2690) / N0;
  }
  else
  {
    Nppb = (log(N0 * Npp) - log(log(N0 * Npp))) / N0;
  }
  double Ca = (muf * K) / (sigma * sqrt(tgrowth));
  double Ca0 = 2.16 * 1E-4 * (pow(DTsup, 1.216));
  double rappD = max(0.1237 * pow(Ca, -0.373) * sin(angle), 1.);

  double Dinception = rappD * Dd;
  double Dml = Dinception / 2.;
  double deltaml = 4E-6 * sqrt(Ca / Ca0);

  double phiml = rhof * hfg * freq * Nppb * (deltaml * (pow(Dml, 2)) * (3.1415 / 12.) * (2 - (pow(rappD, 2) + rappD)));
  double phiinception = 1.33 * 3.1415 * pow(Dinception / 2., 3) * rhog * hfg * freq * Nppb;
  double phie = phiml + phiinception;
  double Dlo = 1.2 * Dd;
  double Asl = (Dlo + Dd) / (2 * sqrt(Nppb));
  double tstar = (pow(kf, 2)) / ((pow(hfc, 2)) * 3.1415 * etaf);
  tstar = min(tstar, twait);
  double Ssl = min(1., Asl * Nppb * tstar * freq);
  double phisc = 2 * hfc * Ssl * (DTsup + DTsub);
  double phifc = (1 - Ssl) * hfc * (DTsup + DTsub);

  return phisc + phifc + phie;
}

double optfuncfitpol(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* function to be optimized for least squares fit*/
  auto p{static_cast<pair<VectorXd *, VectorXd *> *>(data)};
  VectorXd xobs = *(p->first);
  VectorXd y = *(p->second);
  VectorXd pred(xobs.size());
  for (int i = 0; i < xobs.size(); i++)
  {
    if (xobs(i) < x[1])
    {
      pred(i) = x[0] * xobs(i);
    }
    else
    {
      pred(i) = x[0] * xobs(i) + pow(xobs(i) - x[1], x[2]);
    }
  }
  VectorXd v = y - pred;
  double sqres = v.array().square().sum();
  return -1 * sqres;
};

VectorXd Compute_derivatives_y(vector<VectorXd> const &obs_locs, VectorXd const &Yobs, double ftol_rel)
{
  // Compute the derivatives of the true process at the observation points obs_locs.
  // Using a polynomial fit p(x) = a * x + (x - b)^c
  VectorXd xguess(3);
  xguess(0) = 10, xguess(1) = 5, xguess(2) = 7;
  VectorXd lb(3);
  lb << 0, 5, 0;
  VectorXd ub(3);
  ub << 100000, 15, 10;
  VectorXd obslocs(obs_locs.size());
  for (int i = 0; i < obslocs.size(); i++)
  {
    obslocs(i) = obs_locs[i](0);
  }
  auto p = make_pair(&obslocs, &Yobs);
  optroutine(optfuncfitpol, &p, xguess, lb, ub, ftol_rel);
  cout << "best fit polynomial : " << xguess.transpose() << endl;
  // calcul des dérivées
  auto fprime = [xguess](double x)
  {
    if (x < xguess(1))
    {
      return xguess(0);
    }
    else
    {
      return xguess(0) + xguess(2) * pow(x - xguess(1), xguess(2) - 1);
    }
  };
  auto f = [xguess](double x)
  {
    if (x < xguess(1))
    {
      return xguess(0) * x;
    }
    else
    {
      return xguess(0) * x + pow(x - xguess(1), xguess(2));
    }
  };
  VectorXd derivobs(obs_locs.size());
  for (int i = 0; i < obs_locs.size(); i++)
  {
    derivobs(i) = fprime(obs_locs[i](0));
  }
  return derivobs;
}

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // Kernel Matern 5/2
  double d = abs(x(0) - y(0));
  double l = exp(hpar(1));
  return pow(exp(hpar(0)), 2) * (1 + (d / l) + (1. / 3) * pow(d / l, 2)) * exp(-d / l); // 5/2
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  // noyau pour le hGP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3
  double d1 = abs(x(0) - y(0)) / hpar(1);
  double d2 = abs(x(1) - y(1)) / hpar(3);
  double d3 = abs(x(2) - y(2)) / hpar(4);
  double cor = -d1 - d2 - d3;
  cor = exp(cor) * (1 + d1) * (1 + d2) * (1 + d3);
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
  else if (p <= 4)
  { // dérivée par rapport à une longueur de corrélation
    double cor = pow(hpar(0), 2);
    VectorXd d(3);
    d(0) = abs(x(0) - y(0)) / hpar(1);
    for (int i = 1; i < 3; i++)
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
    for (int i = 3; i < 5; i++)
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
    cerr << "erreur calcul dérivée. indice demandé : " << p << ". Maximum : 4" << endl;
    exit(0);
  }
}

VectorXd RtoGP(VectorXd const &X)
{
  // transforme theta dans l'espace réel en [0,1]. transformation affine.
  int dim = X.size();
  VectorXd lb_t(dim);
  lb_t(0) = 20;
  lb_t(1) = 0.5 * 18.9E-6;
  lb_t(2) = 0.5 * 0.75; // 0.2 avant
  VectorXd ub_t(dim);
  ub_t(0) = 60;
  ub_t(1) = 1.5 * 18.9E-6;
  ub_t(2) = 1.5 * 0.75;
  VectorXd Xgp(dim);
  for (int i = 0; i < dim; i++)
  {
    Xgp(i) = (X(i) - lb_t(i)) / (ub_t(i) - lb_t(i));
  }
  return Xgp;
}

VectorXd GPtoR(VectorXd const &X)
{
  // transforme theta dans l'espace gp vers l'espace réel. transformation affine.
  int dim = X.size();
  VectorXd lb_t(dim);
  lb_t(0) = 20;
  lb_t(1) = 0.5 * 18.9E-6;
  lb_t(2) = 0.5 * 0.75; // 0.2 avant
  VectorXd ub_t(dim);
  ub_t(0) = 60;
  ub_t(1) = 1.5 * 18.9E-6;
  ub_t(2) = 1.5 * 0.75;
  VectorXd Xr(dim);
  for (int i = 0; i < dim; i++)
  {
    Xr(i) = lb_t(i) + (ub_t(i) - lb_t(i)) * X(i);
  }
  return Xr;
}

double logprior_hpars(VectorXd const &hpars)
{
  return 0;
}

double lognorm(double x, double mean, double std)
{
  return -0.5 * pow((x - mean) / std, 2);
}

double logprior_pars(VectorXd const &pars)
{
  // prior gaussien sur les paramètres.
  double d = 0;
  for (int i = 0; i < 3; i++)
  {
    d += lognorm(pars(i), 0.5, 0.3);
  }
  return d;
}

pair<vector<VectorXd>, VectorXd> read_csv(const string &filename)
{
  // lit les data d'un csv.
  ifstream ifile(filename);
  string line;
  vector<string> res;
  getline(ifile, line);
  while (getline(ifile, line))
  {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ','))
    {
      res.push_back(cell);
    }
  }
  vector<VectorXd> Xlocations;
  VectorXd Yobs(res.size() / 2);
  for (int i = 0; i < res.size() / 2; i++)
  {
    DATA dat;
    VectorXd x(1);
    x << stod(res[2 * i]);
    Xlocations.push_back(x);
    Yobs(i) = stod(res[2 * i + 1]);
  }
  cout << "filename : " << filename << endl;
  cout << "obs loaded : " << Xlocations.size() << endl;
  return make_pair(Xlocations, Yobs);
}

void WriteObs(string filename, vector<DATA> &data)
{
  ofstream ofile(filename);
  for (auto const &d : data)
  {
    ofile << d.GetX()(0) << " " << d.Value() << endl;
  }
  ofile.close();
}

double optfuncKOH_pooled(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* fonction à optimiser pour trouver les hpars koh.*/
  auto ptp = (tuple<const vector<MatrixXd> *, vector<Density> *> *)data; // cast
  auto tp = *ptp;
  const vector<MatrixXd> *Residus_v = get<0>(tp);
  vector<Density> *d = get<1>(tp);
  // transformer x en vector de vectorXd pour avoir tous les hpars
  vector<VectorXd> hpars_v;
  int c = 0;
  for (int i = 0; i < d->size(); i++)
  {
    VectorXd h(d->at(i).GetBoundsHpars().first.size());
    for (int j = 0; j < h.size(); j++)
    {
      h(j) = x[c];
      c++;
    }
    hpars_v.push_back(h);
  }
  vector<LDLT<MatrixXd>> ldlt_v;
  for (int i = 0; i < d->size(); i++)
  {
    auto xlocs = *(d->at(i).GetXlocations());
    MatrixXd M = d->at(i).Gamma(xlocs, hpars_v[i]);
    if (d->at(i).GetPresenceInputerr() > 0)
    {
      M += pow(d->at(i).GetInputerr(hpars_v[i]), 2) * d->at(i).GetDerMatrix();
    }
    LDLT<MatrixXd> L(M);
    ldlt_v.push_back(L);
  }

  vector<double> prob(Residus_v->at(0).cols());
  for (int i = 0; i < prob.size(); i++)
  {
    double g = 0;
    for (int j = 0; j < d->size(); j++)
    {
      VectorXd priormean = d->at(j).EvaluatePMean(*(d->at(j).GetXlocations()), hpars_v[j]);
      double ll = d->at(j).loglikelihood_fast(Residus_v->at(j).col(i) - priormean, ldlt_v[j]);
      g += ll;
    }
    prob[i] = g;
  }
  double logvstyp = -200;
  for (int i = 0; i < prob.size(); i++)
  {
    double l = prob[i];
    VectorXd theta = d->at(0).GetGrid()->at(i);
    double logprior = d->at(0).EvaluateLogPPars(theta);
    double f = exp(l + logprior - logvstyp);
    if (isinf(f))
    {
      cerr << "erreur myoptfunc_koh : increase logvstyp. Try the value: " << l + logprior << endl;
    }
    prob[i] = f;
  }

  // calcul de l'intégrale. suppose un grid régulier. modifier si on veut un prior pour les paramètres.
  // multiplication des priors pour chaques hyperparamètres !
  double res = accumulate(prob.begin(), prob.end(), 0.0);
  res /= prob.size();
  return res;
};

vector<VectorXd> HparsKOH_pooled(vector<Density> &vDens, VectorXd const &hpars_guess, double ftol_rel)
{
  // optimisation KOH poolée entre toutes les densités. Il faut faire quoi alors ? On est obligé de calculer toutes les valeurs y-ftheta avant de passer à l'optfct. Sinon ça prend trop de temps.
  auto begin = chrono::steady_clock::now();
  int dim = vDens.size();
  int dim_hpars = hpars_guess.size();
  vector<MatrixXd> residus_v(dim);
  for (int i = 0; i < dim; i++)
  {
    VectorXd yobs = *(vDens[i].GetObs());
    vector<VectorXd> xlocations = *(vDens[i].GetXlocations());
    MatrixXd Residustheta(yobs.size(), vDens[i].GetGrid()->size());
    for (int j = 0; j < Residustheta.cols(); j++)
    {
      VectorXd theta = vDens[i].GetGrid()->at(j);
      Residustheta.col(j) = yobs - vDens[i].EvaluateModel(xlocations, theta);
    }
    residus_v[i] = Residustheta;
  }
  auto tp = make_tuple(&residus_v, &vDens);
  // création des bornes des hpars et du guess. tout est fait en vector pour pouvoir gérer les tailles sans s'embêter.
  VectorXd guess(hpars_guess.size() * vDens.size());
  VectorXd lb_hpars(hpars_guess.size() * vDens.size());
  VectorXd ub_hpars(hpars_guess.size() * vDens.size());
  for (int i = 0; i < vDens.size(); i++)
  {
    auto p = vDens[i].GetBoundsHpars();
    for (int j = 0; j < hpars_guess.size(); j++)
    {
      lb_hpars(hpars_guess.size() * i + j) = p.first(j);
      ub_hpars(hpars_guess.size() * i + j) = p.second(j);
      guess(hpars_guess.size() * i + j) = hpars_guess(j);
    }
  }
  double critmax = optroutine(optfuncKOH_pooled, &tp, guess, lb_hpars, ub_hpars, ftol_rel);
  cout << "fin de l'opt koh. critère: " << critmax << endl;
  vector<VectorXd> ret;
  int c = 0;
  for (int i = 0; i < dim; i++)
  {
    VectorXd v(dim_hpars);
    for (int j = 0; j < dim_hpars; j++)
    {
      v(j) = guess[c];
      c++;
    }
    ret.push_back(v);
  }
  auto end = chrono::steady_clock::now();
  cout << "time : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s." << endl;
  return ret;
}


const double Big = -1.e16;

int main(int argc, char **argv)
{
  default_random_engine generator(1);

  /*Paramètres de la simulation*/
  int nombre_steps_mcmc = 1e5;
  int nombre_samples_collected = 1000; 
  int nautocor = 2000;

  double ftol_rel_hgps = 1e-4; // 10 secondes par optimisation opti
  double ftol_rel_fmp = 1e-4;
  double ftol_rel_koh = 1e-4;
  double ftol_rel_derivatives = 1e-4;
  double inputerr = 0.37; // correspond à 0.1 Kelvin d'incertitude en input.

  /*Bounds for parameters and hyperparameters*/
  VectorXd lb_hpars(2);
  lb_hpars(0) = 1e4;
  lb_hpars(1) = 1;
  VectorXd ub_hpars(2);
  ub_hpars(0) = 4e6;
  ub_hpars(1) = 25;
  VectorXd hpars_z_guess = 0.5 * (lb_hpars + ub_hpars);
  lb_hpars.array() = lb_hpars.array().log();
  ub_hpars.array() = ub_hpars.array().log();
  hpars_z_guess.array() = hpars_z_guess.array().log();

  VectorXd lb_t(3);
  lb_t(0) = 0;
  lb_t(1) = 0.5 * 18.9E-6;
  lb_t(2) = 0.5 * 0.75;
  VectorXd ub_t(3);
  ub_t(0) = 90;
  ub_t(1) = 1.5 * 18.9E-6;
  ub_t(2) = 1.5 * 0.75;
  lb_t = RtoGP(lb_t); // transfer bounds from physical space to GP space
  ub_t = RtoGP(ub_t);

  /*Bounds for hGPs*/
  MatrixXd Bounds_hpars_gp(2, 5);
  Bounds_hpars_gp(0, 0) = 1E-4;
  Bounds_hpars_gp(1, 0) = 1e4; // variance
  Bounds_hpars_gp(0, 2) = 1E-4;
  Bounds_hpars_gp(1, 2) = 1e4; // sigma obs
  list<int> l = {1, 3, 4};
  for (int i : l)
  {
    Bounds_hpars_gp(0, i) = 1E-2;
    Bounds_hpars_gp(1, i) = 5; // lcors.
  }
  VectorXd hpars_gp_guess(5);
  for (int i = 0; i < 5; i++)
  {
    hpars_gp_guess(i) = 0.5 * (Bounds_hpars_gp(1, i) + Bounds_hpars_gp(0, i));
  }
  hpars_gp_guess(0) = 1;    // var edm
  hpars_gp_guess(2) = 1e-3; // var obs
  vector<VectorXd> v_hpars_gp_guess;
  for (int i = 0; i < hpars_z_guess.size(); i++)
  {
    v_hpars_gp_guess.push_back(hpars_gp_guess);
  }
  // vector<int> cases={3,4,5,6,8,14,15,16,18,20,21,22,23}; // calibration with all experiments
  // vector<int> cases={3,4,5,6,15,16,18,20,21}; // calibration with all good experiments
  vector<int> cases = {6}; // calibration with experiment nr 6

  int vsize = cases.size();
  // Points de prédiction, et aussi de construction de surrogate du modèle.
  int samp_size = 60;
  vector<VectorXd> X_predictions(samp_size);
  for (int i = 0; i < samp_size; i++)
  {
    VectorXd x(1);
    x << 0.01 + 35 * double(i) / double(samp_size);
    X_predictions[i] = x;
  }

  auto lambda_priormean = [](vector<VectorXd> const &X, VectorXd const &hpars)
  {
    VectorXd b = VectorXd::Zero(X.size());
    return b;
  };

  DoE doe_init(lb_t, ub_t, 300, 1);
  vector<Density> vDens;
  for (int i = 0; i < cases.size(); i++)
  {
    auto lambda_Fmodel = [&cases, i](vector<VectorXd> const &X, VectorXd const &theta) mutable
    {
      VectorXd pred(X.size());
      for (int j = 0; j < X.size(); j++)
      {
        pred(j) = ravik_model(X[j](0), GPtoR(theta), cases[i]);
      }
      return pred;
    };
    string foldname = "../data/";
    auto obs = read_csv(foldname + "Kennel" + to_string(cases[i]) + ".csv");
    ofstream fobs("results/obs" + to_string(cases[i]) + ".gnu");
    WriteVectors(obs.first, obs.second, fobs);
    fobs.close();
    cout << "Building density for case " << cases[i] << endl;
    Density MainDensity(doe_init);
    MainDensity.SetFModel(lambda_Fmodel);
    MainDensity.SetZKernel(Kernel_Z_Matern52);
    MainDensity.SetHparsBounds(lb_hpars, ub_hpars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetZPriorMean(lambda_priormean);
    MainDensity.SetObservations(obs.first, obs.second);
    MainDensity.SetFixedOutputerr(-3);                                                                                // low value for logarithm of output error, for stability
    MainDensity.SetFixedInputerr(log(inputerr), Compute_derivatives_y(obs.first, obs.second, ftol_rel_derivatives)); 
    vDens.push_back(MainDensity);
  }
  VectorXd X_init_mcmc = 0.5 * VectorXd::Ones(dim_theta);
  MatrixXd COV_init = MatrixXd::Identity(dim_theta, dim_theta);
  COV_init(0, 0) = pow(0.04, 2);
  COV_init(1, 1) = pow(0.17, 2);
  COV_init(2, 2) = pow(0.07, 2);
  /*Beginning of calibration*/
  /* Construction of hGPs on a QMC DoE*/

  /* KOH calibration */
  {
    auto hparskoh_pooled = HparsKOH_pooled(vDens, hpars_z_guess, ftol_rel_koh);
    cout << "hparskoh: " << endl;
    for (int i = 0; i < cases.size(); i++)
    {
      cout << "case " << cases[i] << " : " << hparskoh_pooled[i].transpose() << endl;
    }
    auto in_bounds = [&vDens](VectorXd const &X)
    {
      return vDens[0].in_bounds_pars(X);
    };
    auto get_hpars_kohs = [&hparskoh_pooled](VectorXd const &X)
    {
      return hparskoh_pooled;
    };

    auto compute_score_kohs = [&vDens, &vsize](vector<VectorXd> p, VectorXd const &X)
    {
      double res = 0;
      for (int i = 0; i < vsize; i++)
      {
        double d = vDens[i].loglikelihood_theta(X, p[i]);
        res += d;
      }
      res += vDens[0].EvaluateLogPPars(X);
      return res;
    };
    auto allsteps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_kohs, get_hpars_kohs, in_bounds, generator);
    Selfcor_diagnosis(allsteps, 5000, 1, "results/koh/autocor.gnu");
    vector<VectorXd> theta_sample;
    for (int i = 0; i < nombre_samples_collected; i++)
    {
      theta_sample.push_back(allsteps[i * (allsteps.size() / nombre_samples_collected)]);
    }
    for (int i = 0; i < vsize; i++)
    {
      vDens[i].SetNewSamples(theta_sample);
      vector<VectorXd> vhpars;
      for (int j = 0; j < theta_sample.size(); j++)
      {
        vhpars.push_back(hparskoh_pooled[i]);
      }
      vDens[i].SetNewHparsOfSamples(vhpars);
    }
    string fsamp = "results/koh/samples.gnu";
    ofstream ofile(fsamp);
    ofile << "# samples obtained from cases ";
    for (int i = 0; i < cases.size(); i++)
    {
      ofile << cases[i] << ", ";
    }
    ofile << endl;
    WriteVector(theta_sample, ofile);
    ofile.close();
    for (int i = 0; i < vsize; i++)
    {
      string fnamepred = "results/koh/preds" + to_string(cases[i]) + ".gnu";
      vDens[i].WritePredictions(X_predictions, fnamepred);
    }
  }

  /* HP-QMC calibration */
  {
    cout << "begin HP-QMC calibration" << endl;
    vector<DensityOpt> vDopt;
    for (int i = 0; i < vsize; i++)
    {
      DensityOpt Dopt(vDens[i]);
      // utilisation du doe_init pour construire les hGPs (il est distribué selon un QMC)
      vector<VectorXd> thetas_training = doe_init.GetGrid();
      vector<VectorXd> hpars_training;
      for (const auto theta : thetas_training)
      {
        VectorXd hpars = Dopt.HparsOpt(theta, hpars_z_guess, ftol_rel_fmp);
        hpars_training.push_back(hpars);
      }
      Dopt.BuildHGPs(thetas_training, hpars_training, Kernel_GP_Matern32, DKernel_GP_Matern32);
      Dopt.OptimizeHGPs(Bounds_hpars_gp, v_hpars_gp_guess, ftol_rel_hgps);
      vDopt.push_back(Dopt);
    }

    auto in_bounds = [&vDens](VectorXd const &X)
    {
      return vDens[0].in_bounds_pars(X);
    };
    auto get_hpars_opti = [&vDopt, &vsize, &lb_hpars, &ub_hpars](VectorXd const &X)
    {
      vector<VectorXd> p(vsize);
      for (int i = 0; i < vsize; i++)
      {
        p[i] = vDopt[i].EvaluateHparOpt(X);
      }
      return p;
    };
    auto compute_score_opti = [&vDopt, &vsize](vector<VectorXd> p, VectorXd const &X)
    {
      double res = 0;
      for (int i = 0; i < vsize; i++)
      {
        double d = vDopt[i].loglikelihood_theta(X, p[i]);
        res += d;
      }
      res += vDopt[0].EvaluateLogPPars(X);
      return res;
    };

    auto allsteps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    Selfcor_diagnosis(allsteps, 5000, 1, "results/hpqmc/autocor.gnu");
    vector<VectorXd> theta_sample;
    for (int i = 0; i < nombre_samples_collected; i++)
    {
      theta_sample.push_back(allsteps[i * (allsteps.size() / nombre_samples_collected)]);
    }
    // Set des samples dans chaque densité
    for (int i = 0; i < vsize; i++)
    {
      vDopt[i].SetNewSamples(theta_sample);
      vector<VectorXd> vhpars;
      for (int j = 0; j < theta_sample.size(); j++)
      {
        vhpars.push_back(vDopt[i].HparsOpt(theta_sample[j], hpars_z_guess, ftol_rel_fmp));
      }
      vDopt[i].SetNewHparsOfSamples(vhpars);
    }
    string fsamp = "results/hpqmc/samples.gnu";
    ofstream ofile(fsamp);
    WriteVector(theta_sample, ofile);
    ofile.close();
    for (int i = 0; i < vsize; i++)
    {
      string fnamepred = "results/hpqmc/preds" + to_string(cases[i]) + ".gnu";
      vDopt[i].WritePredictions(X_predictions, fnamepred);
    }
  }

  exit(0);
}
