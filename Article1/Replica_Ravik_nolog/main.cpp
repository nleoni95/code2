// Dans ce fichier : on met en oeuvre différents algorithmes d'optimisation pour KOH. Voir lequel est le meilleur.
// On met en place une quadrature pour évaluer de manière précise l'intégrale KOH.
// On regarde maintenant la sensibilité aux observations.
// On essaye avec un hpar supplémentaire : moyenne de model bias constante
// Problème avec au moins trois hyperparamètres, et la totalité des expériences. On va faire baver KOH.
// on refait RAVIK en propre mais sans log sur le kernel: l'algo d'optimisation n'aime pas.

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
//pour stocker les valeurs calculées d'hyperparamètres optimaux. On a un vectorxd qui est le theta, et une map (int vectorxd) qui permet de retrouver l'hyperparamètre optimal (vectorxd) du cas i (int)

int neval = 1;
//std::default_random_engine generator;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0, 1);
std::normal_distribution<double> distN(0, 1);
vector<DATA> data;
vector<VectorXd> Grid;

int dim_x = 1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
int dim_theta = 3;

// importation des données Python:
//dans l'ordre : p,v,DTsub,diam,Tsat,rhof,muf,rhog,cpf,kf,hfg,sigma,fric.
//               0,1,  2,   3,    4,   5,  6,   7,  8,  9, 10,  11,   12
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

// MODELE RAVIK.

double ravik_model_physique(double DTsup, VectorXd const &params, int case_nr)
{
  //implémentation du MIT Boilng model.
  //params contient en 0 : l'angle de contact, en 1 et 2 les 2 coefs de la corrélation du diamètre.
  //ne pas oublier de convertir params dans le domaine correct.
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
  //ishiihibiki
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

double gaussprob(double x, double mu, double sigma)
{
  //renvoie la probabilité gaussienne
  return 1. / (sqrt(2 * 3.14 * pow(sigma, 2))) * exp(-0.5 * pow((x - mu) / sigma, 2));
}

double Kernel_Z_Matern52_noexp(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //squared exponential
  double s = hpar(0);
  double l = hpar(1);
  double d = abs(x(0) - y(0));
  return pow(s, 2) * (1 + (d / l) + (1. / 3) * pow(d / l, 2)) * exp(-d / l); //5/2
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
  cor *= (1 + abs(x(0) - y(0)) / hpar(1)) * exp(-abs(x(0) - y(0)) / hpar(1)); //par1
  cor *= (1 + abs(x(1) - y(1)) / hpar(3)) * exp(-abs(x(1) - y(1)) / hpar(3)); //par2
  cor *= (1 + abs(x(2) - y(2)) / hpar(4)) * exp(-abs(x(2) - y(2)) / hpar(4)); //par3
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

VectorXd RtoGP(VectorXd const &X)
{
  //transforme theta dans l'espace réel en [0,1]. transformation affine.
  int dim = X.size();
  //bornes de l'intervalle ici. il faut que la moyenne somme à 2*valeur nominale pour que 0.5 soit la moyenne.
  VectorXd lb_t(dim);
  lb_t(0) = 20;
  lb_t(1) = 0.5 * 18.9E-6;
  lb_t(2) = 0.5 * 0.75; //0.2 avant
  VectorXd ub_t(dim);
  ub_t(0) = 60;
  ub_t(1) = 1.5 * 18.9E-6;
  ub_t(2) = 1.5 * 0.75;
  for (int i = 0; i < dim; i++)
  {
    //if(X(i)<lb_t(i) || X(i)>ub_t(i)){cerr << "erreur de dimension rtogp " << i <<endl;}
  }
  VectorXd Xgp(dim);
  for (int i = 0; i < dim; i++)
  {
    Xgp(i) = (X(i) - lb_t(i)) / (ub_t(i) - lb_t(i));
  }
  return Xgp;
}

VectorXd GPtoR(VectorXd const &X)
{
  //transforme theta dans l'espace gp vers l'espace réel. transformation affine.
  //bornes de l'intervalle ici.
  int dim = X.size();
  //bornes de l'intervalle ici. il faut que la moyenne somme à 2*valeur nominale pour que 0.5 soit la moyenne.
  VectorXd lb_t(dim);
  lb_t(0) = 20;
  lb_t(1) = 0.5 * 18.9E-6;
  lb_t(2) = 0.5 * 0.75; //0.2 avant
  VectorXd ub_t(dim);
  ub_t(0) = 60;
  ub_t(1) = 1.5 * 18.9E-6;
  ub_t(2) = 1.5 * 0.75;
  for (int i = 0; i < dim; i++)
  {
    //if(X(i)<0 || X(i)>1){cerr << "erreur de dimension gptor " << i <<endl;}
  }
  VectorXd Xr(dim);
  for (int i = 0; i < dim; i++)
  {
    Xr(i) = lb_t(i) + (ub_t(i) - lb_t(i)) * X(i);
  }
  return Xr;
}
//loi inverse gamma pour l
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

double logprior_hpars(VectorXd const &hpars)
{
  return 0;
}

double logprior_hpars_invgamma(VectorXd const &hpars)
{
  double alpha = 50;
  double beta = 500;
  return loginvgammaPdf(hpars(1), alpha, beta);
}

double lognorm(double x, double mean, double std)
{
  return -0.5 * pow((x - mean) / std, 2);
}

double logprior_pars(VectorXd const &pars)
{
  //prior gaussien sur les paramètres. ils seront dans l'espace (0,1.)
  double d = 0;
  for (int i = 0; i < 3; i++)
  {
    d += lognorm(pars(i), 0.5, 0.3);
  }
  return d;
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

void PrintVector(vector<VectorXd> &X, VectorXd &values, const char *file_name)
{
  FILE *out = fopen(file_name, "w");
  for (int i = 0; i < X.size(); i++)
  {
    fprintf(out, "%e %e\n", X[i](0), values(i));
  }
  fclose(out);
}

std::vector<std::string> getNextLineAndSplitIntoTokens(std::istream &str)
{
  std::vector<std::string> result;
  std::string line;
  std::getline(str, line);

  std::stringstream lineStream(line);
  std::string cell;

  while (std::getline(lineStream, cell, ','))
  {
    result.push_back(cell);
  }
  return result;
}

vector<DATA> read_csv(const string &filename)
{
  //lit les data d'un csv.
  ifstream ifile(filename);
  string line;
  vector<string> res;
  getline(ifile, line); //skipper première ligne.
  while (getline(ifile, line))
  {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ','))
    {
      res.push_back(cell);
    }
  }
  vector<DATA> t;
  for (int i = 0; i < res.size() / 2; i++)
  {
    DATA dat;
    VectorXd x(1);
    x << stod(res[2 * i]);
    dat.SetX(x);
    dat.SetValue(stod(res[2 * i + 1]));
    t.push_back(dat);
  }
  cout << "filename : " << filename << endl;
  cout << "obs loaded : " << t.size() << endl;
  return t;
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

void writeVector(string const &filename, vector<VectorXd> const &v)
{
  ofstream ofile(filename);
  int size = v[0].size();
  for (int i = 0; i < v.size(); i++)
  {
    for (int j = 0; j < size; j++)
    {
      ofile << v[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

vector<VectorXd> readVector(string const &filename)
{
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
      VectorXd hpars(words.size());
      for (int i = 0; i < hpars.size(); i++)
      {
        hpars(i) = stod(words[i]);
      }
      v.push_back(hpars);
    }
  }
  return v;
}

void writeVectors(string const &filename, vector<VectorXd> &v1, vector<VectorXd> &v2)
{
  if (!v1.size() == v2.size())
  {
    cerr << "erreur de dimension dans writeVectors." << v1.size() << " " << v2.size() << endl;
  }
  ofstream ofile(filename);
  int size1 = v1[0].size();
  int size2 = v2[0].size();
  for (int i = 0; i < v1.size(); i++)
  {
    for (int j = 0; j < size1; j++)
    {
      ofile << v1[i](j) << " ";
    }
    for (int j = 0; j < size2; j++)
    {
      ofile << v2[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

void compute_Rubin(vector<vector<VectorXd>> &v)
{
  auto begin = chrono::steady_clock::now();
  //calculer le critère de Rubin pour la variance inter chaînes.
  //centrer les échantillons
  int m = v.size();       //nombre de chaînes
  int n = v[0].size();    //taille des chaînes
  int d = v[0][0].size(); //dimension des chaînes
  vector<VectorXd> means;
  for (int i = 0; i < m; i++)
  {
    VectorXd mean = VectorXd::Zero(d);
    for (int j = 0; j < n; j++)
    {
      mean += v[i][j];
    }
    mean /= n;
    means.push_back(mean);
  }
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      v[i][j] -= means[i];
    }
  }
  VectorXd totalmean = VectorXd::Zero(d);
  for (int i = 0; i < m; i++)
  {
    totalmean += means[i];
  }
  totalmean /= m;
  VectorXd B = VectorXd::Zero(d); //between-sequence variance
  for (int i = 0; i < m; i++)
  {
    B.array() += (means[i] - totalmean).array().square();
  }
  B *= (1.0 * n) / (m - 1.0);
  VectorXd W = VectorXd::Zero(d);
  for (int i = 0; i < m; i++)
  {
    VectorXd sjs = VectorXd::Zero(d);
    for (int j = 0; j < n; j++)
    {
      sjs.array() += (v[i][j] - means[i]).array().square();
    }
    sjs /= n - 1.0;
    W += sjs;
  }
  W /= 1.0 * m;
  cout << "B : " << B.transpose() << endl;
  cout << "W : " << W.transpose() << endl;
  VectorXd var = W * (n - 1.0) / (n * 1.0) + B * (1.0 / n);
  VectorXd R = (var.array() / W.array()).sqrt();
  cout << "R : " << R.transpose() << endl;

  auto end = chrono::steady_clock::now();
  cout << "Rubin criterion over.  "
       << " time : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s " << endl;
}

vector<vector<VectorXd>> revert_vector(vector<vector<VectorXd>> const &v)
{
  //inverse les deux vecteurs.
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

int optroutine(nlopt::vfunc optfunc, void *data_ptr, vector<double> &x, vector<double> const &lb_hpars, vector<double> const &ub_hpars, double max_time)
{
  //routine d'optimisation sans gradient

  //paramètres d'optimisation
  double ftol_large = 1e-5;
  double xtol_large = 1e-3;
  double ftol_fin = 1e-15;
  double xtol_fin = 1e-7;
  // 1 optimiseur local et un global.
  nlopt::opt local_opt(nlopt::LD_LBFGS, x.size());
  local_opt.set_max_objective(optfunc, data_ptr);
  local_opt.set_ftol_rel(ftol_large);
  local_opt.set_xtol_rel(xtol_large);
  local_opt.set_lower_bounds(lb_hpars);
  local_opt.set_upper_bounds(ub_hpars);

  nlopt::opt opt(nlopt::GD_MLSL_LDS, x.size());
  opt.set_max_objective(optfunc, data_ptr);
  opt.set_lower_bounds(lb_hpars);
  opt.set_upper_bounds(ub_hpars);
  //pas de contrainte de temps.
  opt.set_maxtime(max_time); //20 secondes au max.
  opt.set_local_optimizer(local_opt);
  double msup;                     /* the maximum objective value, upon return */
  int fin = opt.optimize(x, msup); //messages d'arrêt
  if (!fin == 3)
  {
    cout << "opti hpars message d'erreur : " << fin << endl;
  }
  //on relance une opti locale à partir du max. trouvé.
  local_opt.set_ftol_rel(ftol_fin);
  local_opt.set_xtol_rel(xtol_fin);
  fin = local_opt.optimize(x, msup); //messages d'arrêt
  if (!fin == 3)
  {
    cout << "opti hpars message d'erreur : " << fin << endl;
  }
  return fin;
}

vector<VectorXd> HparsKOH_separate(vector<Density> &vDens, VectorXd const &hpars_guess, double max_time)
{
  //optimisation KOH pour chaque densité séparément. Attention, le max_time est divisé entre chaque optimisation.
  //un seul hpars_guess. on pourra changer ça si besoin.
  auto begin = chrono::steady_clock::now();
  int dim = vDens.size();
  double indiv_time = max_time / dim;
  vector<VectorXd> hpars_koh;
  for (int i = 0; i < dim; i++)
  {
    VectorXd h = vDens[i].HparsKOH(hpars_guess, indiv_time);
    hpars_koh.push_back(h);
  }
  return hpars_koh;
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

void drawPrior(default_random_engine &generator, string filename)
{
  //tire des échantillons de la prior, MVN. 50000 points comme les autres. les affiche dans un fichier.
  int npts = 50000;
  VectorXd mean = 0.5 * VectorXd::Ones(3);
  MatrixXd cov = pow(0.3, 2) * MatrixXd::Identity(3, 3);
  MatrixXd sqrtCOV = cov.llt().matrixL();
  vector<VectorXd> v(npts);
  for (int i = 0; i < npts; i++)
  {
    VectorXd N(3);
    for (int j = 0; j < 3; j++)
    {
      N(j) = distN(generator);
    }
    VectorXd x = mean + sqrtCOV * N;
    v[i] = x;
  }
  writeVector(filename, v);
}

double optfuncfitpol(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* moindres carrés pour une fct de la forme spéciale.*/
  auto p = (Density *)data; //recast
  VectorXd y = p->GetYobs();
  const vector<VectorXd> xobsvec = p->GetXlocs();
  VectorXd xobs(xobsvec.size());
  for (int i = 0; i < xobs.size(); i++)
  {
    xobs(i) = xobsvec[i](0);
  }
  // calcul des prédictions de la fonction
  VectorXd pred(xobs.size());
  for (int i = 0; i < xobs.size(); i++)
  {
    if (xobs(i) < x[1])
    {
      pred(i) = x[0] * xobs(i) + x[3];
    }
    else
    {
      pred(i) = x[0] * xobs(i) + pow(xobs(i) - x[1], x[2]) + x[3];
    }
  }
  VectorXd v = y - pred;
  double sqres = v.array().square().sum();
  return -1 * sqres;
};

pair<VectorXd, VectorXd> Compute_derivatives_f(Density *p, std::vector<Eigen::VectorXd> const &obs_locs, std::vector<Eigen::VectorXd> const &preds_locs, double max_time)
{
  //calcul des dérivées aux points d'observation et aux points de prédictions. Pour cela : on fait des différences finies aux points d'obs. Si plus complexe : on devra fitter un polynôme. Mais même pas besoin je pense. ah bah si pour des points quelconques...
  //on va fitter 1 fois 1 modèle bon pour estimer. Et après ça roule. en least squares bien sûr.
  VectorXd xguess(4);
  xguess(0) = 10, xguess(1) = 5, xguess(2) = 7, xguess(3) = 0;
  VectorXd lb(4);
  lb << 0, 5, 0, 0;
  VectorXd ub(4);
  ub << 100000, 15, 10, 3e5;
  optroutine(optfuncfitpol, p, xguess, lb, ub, max_time);
  cout << "best fit polynomial : " << xguess.transpose() << endl;
  //calcul des dérivées maintenant.
  auto fprime = [&xguess](double x)
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
  auto f = [&xguess](double x)
  {
    if (x < xguess(1))
    {
      return xguess(0) * x + xguess(3);
    }
    else
    {
      return xguess(0) * x + pow(x - xguess(1), xguess(2)) + xguess(3);
    }
  };

  VectorXd derobs(obs_locs.size());
  for (int i = 0; i < obs_locs.size(); i++)
  {
    derobs(i) = fprime(obs_locs[i](0));
  }
  VectorXd derp(preds_locs.size());
  for (int i = 0; i < preds_locs.size(); i++)
  {
    derp(i) = fprime(preds_locs[i](0));
  }
  cout << "derivatives at obs pts : " << derobs.transpose() << endl;
  cout << "derivatives at preds pts : " << derp.transpose() << endl;
  return make_pair(derobs, derp);
}

double optfuncKOH_pooled(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* fonction à optimiser pour trouver les hpars koh.*/
  auto ptp = (tuple<const vector<MatrixXd> *, vector<Density> *> *)data; //cast
  auto tp = *ptp;
  const vector<MatrixXd> *Residus_v = get<0>(tp);
  vector<Density> *d = get<1>(tp);
  //transformer x en vector de vectorXd pour avoir tous les hpars
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
  //il faut que toutes les densités aient le même grid en theta.
  vector<LDLT<MatrixXd>> ldlt_v;
  //vecteur de tous les LDLTS.
  for (int i = 0; i < d->size(); i++)
  {
    const vector<VectorXd> *xconv = d->at(i).GetXlocations();
    MatrixXd M = d->at(i).Gamma(*xconv, hpars_v[i]) + pow(d->at(i).GetOutputerr(hpars_v[i]), 2) * MatrixXd::Identity(xconv->size(), xconv->size());
    //présence d'input error donc on rajoute ici
    if (d->at(i).GetPresenceInputerr())
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
      double ll = d->at(j).loglikelihood_fast(Residus_v->at(j).col(i), ldlt_v[j]);
      g += ll;
    }
    prob[i] = g;
  }
  double logvstyp = -200;
  //normalisation et passage à l'exponentielle.
  //le max doit être le même sur toutes les itérations... d'où la nécessité de le définir à l'extérieur !!!
  for (int i = 0; i < prob.size(); i++)
  {
    //passage à l'exponentielle.
    //on suppose que le logprior des paramètres est le même pour tous, et correspond à celui de la première densité.
    double l = prob[i];
    VectorXd theta = d->at(0).GetGrid()->at(i);
    double logprior = d->at(0).EvaluateLogPPars(theta);
    double f = exp(l + logprior - logvstyp);
    if (isinf(f))
    {
      cerr << "erreur myoptfunc_koh : infini. Valeur de la fonction : " << l + logprior << endl;
    }
    prob[i] = f;
  }

  //calcul de l'intégrale. suppose un grid régulier. modifier si on veut un prior pour les paramètres.
  //multiplication des priors pour chaques hyperparamètres !
  double res = accumulate(prob.begin(), prob.end(), 0.0);
  res /= prob.size();
  return res;
};

vector<VectorXd> HparsKOH_pooled(vector<Density> &vDens, VectorXd const &hpars_guess, double max_time)
{
  //optimisation KOH poolée entre toutes les densités.  On est obligé de calculer toutes les valeurs y-ftheta avant de passer à l'optfct. Sinon ça prend trop de temps.
  auto begin = chrono::steady_clock::now();
  int dim = vDens.size();
  int dim_hpars = hpars_guess.size();
  vector<MatrixXd> residus_v(dim);
  for (int i = 0; i < dim; i++)
  {
    VectorXd expvalues = vDens[i].GetYobs();
    const vector<VectorXd> xobsvec = vDens[i].GetXlocs();
    MatrixXd Residustheta(expvalues.size(), vDens[i].GetGrid()->size());
    for (int j = 0; j < Residustheta.cols(); j++)
    {
      VectorXd theta = vDens[i].GetGrid()->at(j);
      Residustheta.col(j) = expvalues - vDens[i].EvaluateModel(xobsvec, theta);
    }
    residus_v[i] = Residustheta;
  }
  auto tp = make_tuple(&residus_v, &vDens);
  //création des bornes des hpars et du guess. tout est fait en vector pour pouvoir gérer les tailles sans s'embêter.
  vector<double> lb_hpars, ub_hpars, guess;
  for (int i = 0; i < dim; i++)
  {
    auto p = vDens[i].GetBoundsHpars();
    for (int j = 0; j < hpars_guess.size(); j++)
    {
      lb_hpars.push_back(p.first(j));
      ub_hpars.push_back(p.second(j));
      guess.push_back(hpars_guess(j));
    }
  }

  int fin = optroutine(optfuncKOH_pooled, &tp, guess, lb_hpars, ub_hpars, max_time);
  vector<double> grad; //vide
  cout << "fin de l'opt koh pooled. crit max :" << optfuncKOH_pooled(guess, grad, &tp) << endl;
  //il faut repasser guess en vector<vectorXd>.
  vector<VectorXd> ret; //=hpars_guess;
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

void Nodups(std::vector<VectorXd> &v)
{
  auto end = v.end();
  for (auto it = v.begin(); it != end; ++it)
  {
    end = std::remove(it + 1, end, *it);
  }

  v.erase(end, v.end());
}

const double Big = -1.e16;

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    cout << "Usage :\n";
    cout << "\t Number of obs, seed_obs\n";
    exit(0);
  }


  VectorXd Xmin = 0.5 * VectorXd::Ones(3);
  Xmin << 0, 0.5 * 18.9E-6, 0.5 * 0.75;
  VectorXd Xmax = 0.5 * VectorXd::Ones(3);
  Xmax << 90, 2 * 18.9E-6, 2 * 0.75;
  cout << RtoGP(Xmin).transpose() << endl;
  cout << RtoGP(Xmax).transpose() << endl;

  int nd = atoi(argv[1]);
  uint32_t seed_obs = atoi(argv[2]); //
  //int cas=6;
  default_random_engine generator(123456);

  /*Paramètres de la simulation*/
  //pour la MCMC
  int nombre_steps_mcmc = 5e5;
  int nombre_samples_collected = 1000; //1 sample tous les 500.
  int nautocor = 3000;

  int time_opt_opti = 10;      // 10 secondes par optimisation opti
  int time_opt_koh_loc = 600;  // 10 minutes par optimisation KOH locale
  int time_opt_koh_glo = 7200; // 2h pour optimisation KOH globale
  double inputerr = 0.37;      //sigma=0.1K d'incertitude en input.
  // Bornesup des paramètres et hyperparamètres
  // lower bound : theta vaut zero, chacun des paramètres vaut la moitié de son coefficient.
  //upper bound : theta vaut 90, chacun des paramètres vaut 2* sa valeur
  VectorXd lb_t_true(dim_theta);
  lb_t_true << 0, 0.2 * 18.9E-6, 0.2 * 0.75;
  VectorXd ub_t_true(dim_theta);
  ub_t_true << 90, 2 * 18.9E-6, 2 * 0.75;

  VectorXd lb_t = RtoGP(lb_t_true);
  VectorXd ub_t = RtoGP(ub_t_true);

  cout << "lb_t : " << lb_t.transpose() << endl;
  cout << "ub_t : " << ub_t.transpose() << endl;

  VectorXd lb_hpars(2);
  lb_hpars(0) = 1e4;
  lb_hpars(1) = 1; //l et sigma.
  VectorXd ub_hpars(2);
  ub_hpars(0) = 4e6;
  ub_hpars(1) = 25;

  VectorXd hpars_z_guess = 0.5 * (lb_hpars + ub_hpars);

  //bornes pour hGPs
  MatrixXd Bounds_hpars_gp(2, 5);
  Bounds_hpars_gp(0, 0) = 1E-4;
  Bounds_hpars_gp(1, 0) = 1e4; //variance
  Bounds_hpars_gp(0, 2) = 1E-4;
  Bounds_hpars_gp(1, 2) = 1e4; //sigma obs
  list<int> l = {1, 3, 4};
  for (int i : l)
  {
    Bounds_hpars_gp(0, i) = 1E-2;
    Bounds_hpars_gp(1, i) = 5; //lcors.
  }
  VectorXd hpars_gp_guess(5);
  for (int i = 0; i < 5; i++)
  {
    hpars_gp_guess(i) = 0.5 * (Bounds_hpars_gp(1, i) + Bounds_hpars_gp(0, i));
  }
  hpars_gp_guess(0) = 1;    //var edm
  hpars_gp_guess(2) = 1e-3; //var obs

  //vector<int> cases={3,4,5,6,8,14,15,16,18,20,21,22,23}; // total
  vector<int> cases = {3, 4, 5, 6, 15, 16,18, 20, 21}; // good cases
                                                       //vector<int> cases={3,4,6,16}; // total
                                                       //vector<int> cases={6};
  //vector<int> cases = {3,6};
  int vsize = cases.size();
  //Points de prédiction, et aussi de construction de surrogate du modèle.
  int samp_size = 60; //80 avant
  vector<VectorXd> X_predictions(samp_size);
  VectorXd XPREDS(samp_size);
  for (int i = 0; i < samp_size; i++)
  {
    VectorXd x(1);
    x << 0.01 + 35 * double(i) / double(samp_size);
    X_predictions[i] = x;
    XPREDS(i) = x(0);
  }

  auto lambda_priormean = [](vector<VectorXd> const &X, VectorXd const &hpars)
  {
    VectorXd b = VectorXd::Zero(X.size());
    return b;
  };

  //construction du DoE initial en LHS (generator) ou Grid Sampling(sans generator)
  DoE doe_init(lb_t, ub_t, 400, 10); //doe halton. 300 points en dimension 3 (3 paramètres incertains).
  doe_init.WriteGrid("results/save/grid.gnu");

  int predsize = 60;
  vector<VectorXd> prediction_points;
  for (int i = 0; i < predsize; i++)
  {
    VectorXd x(1);
    x << 0.01 + 35 * double(i) / double(predsize);
    prediction_points.push_back(x);
  }

  vector<Density> vDens;
  for (int i = 0; i < cases.size(); i++)
  {
    auto lambda_model = [&cases, i](vector<VectorXd> const &X, VectorXd const &theta) mutable
    {
      //renvoie toutes les prédictions du modèle aux points donnés par X.
      VectorXd pred(X.size());
      for (int j = 0; j < X.size(); j++)
      {
        pred(j) = ravik_model_physique(X[j](0), GPtoR(theta), cases[i]);
      }
      return pred;
    };

    string intro = "/home/catB/nl255551/Documents/Code/Ravik/courbes_ravik/";
    auto data = read_csv(intro + "Kennel" + to_string(cases[i]) + ".csv");
    VectorXd Yobs(data.size());
    vector<VectorXd> location_points;
    for (int j = 0; j < data.size(); j++)
    {
      Yobs(j) = data[j].Value();
      location_points.push_back(data[j].GetX());
    }

    //test: affichage des observations et des prédictions nominales du modèle.
    VectorXd Xinit = 0.5 * VectorXd::Ones(3);

    string filename_fit = "results/fit" + to_string(cases[i]) + ".gnu";
    string fder = "results/der.gnu";
    Density MainDensity(doe_init);
    MainDensity.SetModel(lambda_model);
    MainDensity.SetKernel(Kernel_Z_Matern52_noexp);
    //MainDensity.SetKernelDerivatives(D1Kernel_Z_Matern52,D2Kernel_Z_Matern52,D1Kernel_Z_Matern52);
    MainDensity.SetHparsBounds(lb_hpars, ub_hpars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetPriorMean(lambda_priormean);
    MainDensity.SetObservations(location_points, Yobs);
    MainDensity.SetOutputerr(false, 10, 1); //outputerr à 10 pour être consistant.
    //MainDensity.Compute_derivatives_f(location_points,location_points,10,fder);
    auto derivatives = Compute_derivatives_f(&MainDensity, location_points, prediction_points, 10);
    //calcul des dérivées de f. retourner sous la forme vector<VectorXd>.
    MainDensity.SetInputerr(false, inputerr, 0, derivatives.first, derivatives.second); //fixé.
    vDens.push_back(MainDensity);
  }
  //test: calcul des hpars optimaux pour le vecteur 0,0,0. Il faut que ça donne le même que dans le cas simple.
  // on a fait le test: 10-2 en time opt est suffisant.

  VectorXd X_init_mcmc = 0.5 * VectorXd::Ones(dim_theta);
  MatrixXd COV_init = MatrixXd::Identity(dim_theta, dim_theta);
  COV_init(0, 0) = pow(0.2, 2);
  COV_init(1, 1) = pow(0.2, 2);
  COV_init(2, 2) = pow(0.2, 2);
  cout << "COV_init : " << endl
       << COV_init << endl;

  //début des calibrations. cool beans.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


  //Phase FMP

  vector<DensityOpt> vDopt;
  vector<VectorXd> thetas_train;
  vector<vector<VectorXd>> hpars_train;

    //étape 1 fmp: construction de DoE QMC de 100 points. ne doti pas être run en même temps que étape 1bis.
  {
    DoE doe_coarse(lb_t, ub_t, 100, 10); //doe halton. 300 points en dimension 3 (3 paramètres incertains).
    thetas_train = doe_coarse.GetGrid();
    if (!vDopt.size() == 0)
    {
      cerr << "erreur !!!!!!!! vDopt déjà construit" << endl;
      exit(0);
    }
    for (int i = 0; i < vsize; i++)
    {
      DensityOpt Dopt(vDens[i]);
      Dopt.SetDoE(doe_coarse);
      string fname = "results/hparsopt/init/";
      vector<VectorXd> hopt_train;
      for (int j = 0; j < thetas_train.size(); j++)
      {
        VectorXd hopt = Dopt.HparsOpt(thetas_train[j], hpars_z_guess, 0.01);
        hopt_train.push_back(hopt);
      }
      hpars_train.push_back(hopt_train);
      Dopt.BuildHGPs(thetas_train, hopt_train, Kernel_GP_Matern32);
      Dopt.OptimizeHGPs(Bounds_hpars_gp, hpars_gp_guess, 1);
      WriteVectors(thetas_train, hopt_train, fname + "hopt" + to_string(cases[i]) + ".gnu");
      vDopt.push_back(Dopt);
    }
  }

  //étape 2 fmp: iterative sampling.
  { //paramètres de l'algorithme iterative sampling.
    int npts_init = 100;
    int npts_per_iter = 25;
    int nsteps_mcmc = 1e5;
    int nsamples_mcmc = 5000;
    int niter = 4;
    double time_opti_fine = 1e-2; //avec gradients.
    double time_opti_hgps = 1;

    auto get_hpars = [&vDopt](VectorXd const &theta)
    {
      vector<VectorXd> h;
      for (int i = 0; i < vDopt.size(); i++)
      {
        VectorXd ho = vDopt[i].EvaluateHparOpt(theta);
        h.push_back(ho);
      }
      return h;
    };

    auto compute_score = [&vDopt](vector<VectorXd> const &h, VectorXd const &theta)
    {
      double res = 0;
      for (int i = 0; i < vDopt.size(); i++)
      {
        res += vDopt[i].loglikelihood_theta(theta, h[i]);
      }
      res += logprior_pars(theta);
      return res;
    };

    auto in_bounds = [&vDopt](VectorXd const &X)
    {
      return vDopt[0].in_bounds_pars(X);
    };

    auto compute_weight = [&vDopt](VectorXd const &theta)
    {
      //compute le poids de resampling.
      double weight = 0;
      for (int i = 0; i < vDopt.size(); i++)
      {
        weight += vDopt[i].EstimatePredError(theta);
      }
      return weight;
    };

    auto resample = [&compute_weight, &generator](int npts, vector<VectorXd> const &candidate_set)
    {
      //première chose à faire: enlever les duplicates du candidate_set.
      vector<VectorXd> nodups = candidate_set;

      Nodups(nodups);
      vector<double> weights(nodups.size());
      for (int i = 0; i < weights.size(); i++)
      {
        weights[i] = compute_weight(nodups[i]);
      }
      //tirage sans remise pondéré par les poids.
      vector<VectorXd> selected_set;
      for (int i = 0; i < npts; i++)
      {
        std::discrete_distribution<int> distribution(weights.begin(), weights.end());
        int drawn = distribution(generator);
        weights[drawn] = 0;
        selected_set.push_back(nodups[drawn]);
      }
      return selected_set;
    };

    //algorithme principal.

    auto begin = chrono::steady_clock::now();
    for (int i = 0; i < niter; i++)
    {
      //run MCMC
      vector<VectorXd> allsteps = Run_MCMC(nsteps_mcmc, X_init_mcmc, COV_init, compute_score, get_hpars, in_bounds, generator);
      vector<VectorXd> candidate_set(nsamples_mcmc);
      for (int j = 0; j < candidate_set.size(); j++)
      {
        candidate_set[j] = allsteps[j * (allsteps.size() / candidate_set.size())];
      }
      //resample from the candidate set
      vector<VectorXd> selected_set = resample(npts_per_iter, candidate_set);
      //compute optimal hpars
      for (const auto theta : selected_set)
      {
        for (int j = 0; j < vDopt.size(); j++)
        {
          VectorXd hopt = vDopt[j].HparsOpt(theta, hpars_z_guess, time_opti_fine);
          hpars_train[j].push_back(hopt);
        }
        thetas_train.push_back(theta);
      }
      //write training points.
      for (int j = 0; j < cases.size(); j++)
      {
        string sp = "results/hparsopt/" + to_string(thetas_train.size()) + "/hopt" + to_string(cases[j]) + ".gnu";
        WriteVectors(thetas_train, hpars_train[j], sp);
      }
      //update hGPs. 10s par hgp.
      for (int j = 0; j < cases.size(); j++)
      {
        vDopt[j].BuildHGPs(thetas_train, hpars_train[j], Kernel_GP_Matern32);
        vDopt[j].OptimizeHGPs(Bounds_hpars_gp, hpars_gp_guess, time_opti_hgps);
      }
    }

    auto end = chrono::steady_clock::now();
    cout << "temps pour algo : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s" << endl;
  }

  // étape 3fmp: MCMC fmp sur l'ensemble des densités.
  //on en profite pour construire la matrice de covariance, qui sera utilisée comme pt de départ pour Bayes.
  {
    //paramètres de la MCMC FMP.
    int nsteps_mcmc = 5e5;
    int nsamples_mcmc = 3000;

    auto get_hpars = [&vDopt](VectorXd const &theta)
    {
      vector<VectorXd> h;
      for (int i = 0; i < vDopt.size(); i++)
      {
        VectorXd ho = vDopt[i].EvaluateHparOpt(theta);
        h.push_back(ho);
      }
      return h;
    };

    auto compute_score = [&vDopt](vector<VectorXd> const &h, VectorXd const &theta)
    {
      double res = 0;
      for (int i = 0; i < vDopt.size(); i++)
      {
        res += vDopt[i].loglikelihood_theta(theta, h[i]);
      }
      res += logprior_pars(theta);
      return res;
    };

    auto in_bounds = [&vDopt](VectorXd const &X)
    {
      return vDopt[0].in_bounds_pars(X);
    };
    //run mcmc et extraire échantillons.
    vector<VectorXd> allsteps = Run_MCMC(nsteps_mcmc, X_init_mcmc, COV_init, compute_score, get_hpars, in_bounds, generator);
    vector<VectorXd> selected_samples(nsamples_mcmc);
    for (int j = 0; j < selected_samples.size(); j++)
    {
      selected_samples[j] = allsteps[j * (allsteps.size() / selected_samples.size())];
    }
    //diagnostic
    Selfcor_diagnosis(allsteps, nautocor, 1, "results/fmp/autocor.gnu");
    //je veux construire un vecteur avec tous les samples et l'écrire. ça sera + facile pour la covariance aussi. les hpars sont passés en exponentielle.
    vector<VectorXd> selected_samples_withhpars;
    vector<vector<VectorXd>> selected_hpars_vec(vsize);
    for (const auto &theta : selected_samples)
    {
      vector<VectorXd> h = get_hpars(theta);
      VectorXd maxsample(3 + 2 * h.size());
      maxsample.head(3) = theta;
      for (int i = 0; i < h.size(); i++)
      {
        maxsample(3 + 2 * i) = exp(h[i](0));
        maxsample(3 + 2 * i + 1) = exp(h[i](1));
      }
      selected_samples_withhpars.push_back(maxsample);
      for (int i = 0; i < vsize; i++)
      {
        selected_hpars_vec[i].push_back(h[i]);
      }
    }
    for (int i = 0; i < vDens.size(); i++)
    {
      vDopt[i].SetNewSamples(selected_samples);
      vDopt[i].SetNewHparsOfSamples(selected_hpars_vec[i]);
      string fn = "results/fmp/preds" + to_string(cases[i]) + ".gnu";
      string f1 = "results/fmp/sampsf" + to_string(cases[i]) + ".gnu";
      string f2 = "results/fmp/sampsz" + to_string(cases[i]) + ".gnu";
      vDopt[i].WritePredictions(prediction_points, fn);
      vDopt[i].WriteSamplesFandZ(prediction_points, f1, f2);
    }
    writeVector("results/fmp/sample.gnu", selected_samples_withhpars);

    //construction cov_init_from_fmp
    MatrixXd COV_init_from_FMP = MatrixXd::Zero(selected_samples_withhpars[0].size(), selected_samples_withhpars[0].size());
    VectorXd X_init_from_FMP = VectorXd::Zero(selected_samples_withhpars[0].size());
    for (int j = 0; j < selected_samples_withhpars.size(); j++)
    {
      X_init_from_FMP += selected_samples_withhpars[j];
    }
    X_init_from_FMP /= selected_samples_withhpars.size();
    for (int j = 0; j < selected_samples_withhpars.size(); j++)
    {
      VectorXd x = selected_samples_withhpars[j] - X_init_from_FMP;
      COV_init_from_FMP += x * x.transpose();
    }
    cout << "COV_init_from_FMP : " << endl;
    cout << COV_init_from_FMP << endl;
    cout << "X_init_from_FMP : " << endl;
    cout << X_init_from_FMP.transpose() << endl;
    vector<VectorXd> xsave;
    vector<VectorXd> Msave;
    xsave.push_back(X_init_from_FMP);
    for (int i = 0; i < COV_init_from_FMP.cols(); i++)
    {
      Msave.push_back(COV_init_from_FMP.col(i));
    }
    string f1 = "results/Xinit.gnu";
    string f2 = "results/Minit.gnu";
    writeVector(f1, xsave);
    writeVector(f2, Msave);
  }

exit(0);

  // calibration BAYES_backupsingle
  {
    VectorXd lb_hpars_noexp(2);
    VectorXd ub_hpars_noexp(2);
    lb_hpars_noexp = lb_hpars.array().exp();
    ub_hpars_noexp = ub_hpars.array().exp();
    //construction de Densities avec les hpars pas dans l'espace log.
    vector<Density> vDens_fb;
    for (int i = 0; i < vDens.size(); i++)
    {
      Density D(vDens[i]);
      D.SetKernel(Kernel_Z_Matern52_noexp);
      D.SetHparsBounds(lb_hpars_noexp, ub_hpars_noexp);
      vDens_fb.push_back(D);
    }
    //paramètres mcmc bayes
    int nsteps_mcmc = 5e5;
    int nsamples_mcmc = 3000;
    //lecture cov_init et xinit.
    string f1 = "results/Xinit.gnu";
    string f2 = "results/Minit.gnu";
    vector<VectorXd> xsave = readVector(f1);
    vector<VectorXd> Msave = readVector(f2);
    VectorXd X_fromFMP = xsave[0];
    MatrixXd COV_fromFMP(Msave.size(), Msave.size());
    for (int i = 0; i < COV_fromFMP.cols(); i++)
    {
      COV_fromFMP.col(i) = Msave[i];
    }

    //tirages points de départ.
    int nstarting = 10;
    vector<VectorXd> starting_points;
    starting_points.push_back(xsave[0]);
    for (int i = 0; i < nstarting - 1; i++)
    {
      VectorXd S(xsave[0].size());
      for (int j = 0; j < 3; j++)
      {
        S(j) = lb_t(j) + (ub_t(j) - lb_t(j)) * distU(generator);
      }
      for (int j = 3; j < S.size(); j++)
      {
        int u;
        if (j % 2)
        { //j impair donc edm
          u = 0;
        }
        else
        {
          u = 1;
        }
        S(j) = lb_hpars_noexp(u) + (ub_hpars_noexp(u) - lb_hpars_noexp(u)) * distU(generator);
      }
      starting_points.push_back(S);
    }

    auto in_bounds_fb = [&vDens_fb, &vsize, &lb_hpars_noexp, &ub_hpars_noexp](VectorXd const &X)
    {
      VectorXd theta = X.head(3);
      vector<VectorXd> hp;
      for (int i = 0; i < vsize; i++)
      {
        VectorXd h(lb_hpars_noexp.size());
        for (int j = 0; j < h.size(); j++)
        {
          h(j) = X(3 + j + i * h.size());
        }
        hp.push_back(h);
      }
      bool in = true;
      for (int i = 0; i < vsize; i++)
      {
        for (int j = 0; j < lb_hpars_noexp.size(); j++)
        {
          if (hp[i](j) < lb_hpars_noexp(j) || hp[i](j) > ub_hpars_noexp(j))
          {
            in = false;
          }
        }
      }
      if (!vDens_fb[0].in_bounds_pars(X))
      {
        return false;
      }
      return in;
    };

    auto get_hpars_fb = [](VectorXd const &X)
    {
      //renvoyer un vecteur sans intérêt.
      vector<VectorXd> v(1);
      return v;
    };
    auto compute_score_fb = [&vsize, &lb_hpars, &vDens_fb](vector<VectorXd> p, VectorXd const &X)
    {
      //il faut décomposer X en thetas/ hpars. 3 est la dimension des paramètres
      //attention : suppose logprior_hpars nul.
      VectorXd theta = X.head(3);
      vector<VectorXd> hp;
      for (int i = 0; i < vsize; i++)
      {
        VectorXd h(lb_hpars.size());
        for (int j = 0; j < h.size(); j++)
        {
          h(j) = X(3 + j + i * h.size());
        }
        hp.push_back(h);
      }
      double res = 0;
      for (int i = 0; i < vsize; i++)
      {
        double d = vDens_fb[i].loglikelihood_theta(theta, hp[i]);
        res += d;
      }
      res += logprior_pars(theta);
      return res;
    };
    cout << "run mcmc from starting point XinitMCMC. COV init divisée par 25." << endl;
    COV_fromFMP /= 25;
    vector<VectorXd> allsteps = Run_MCMC_hundred(200 * nsteps_mcmc, starting_points[1], COV_fromFMP, compute_score_fb, get_hpars_fb, in_bounds_fb, generator);

    vector<VectorXd> selected_samples(nsamples_mcmc);
    for (int j = 0; j < selected_samples.size(); j++)
    {
      selected_samples[j] = allsteps[j * (allsteps.size() / selected_samples.size())];
    }
    //je veux construire un vecteur avec tous les samples et l'écrire. ça sera + facile pour la covariance aussi. les hpars sont passés en exponentielle.
    writeVector("results/bayes/sample.gnu", selected_samples);
    //diagnostic
    Selfcor_diagnosis(allsteps, nautocor, 0.5, "results/bayes/autocor.gnu");
    cout << "autocor on the sel sample." << endl;
    Selfcor_diagnosis(selected_samples, nautocor, 1, "results/bayes/autocorss.gnu");
    //il faut reconstruire un vecteur d'hpars...
    vector<VectorXd> selected_thetas;
    vector<vector<VectorXd>> selected_hpars_vec(vsize);
    //initialisation du vecteur d'hpars.
    for (int k = 0; k < selected_samples.size(); k++)
    {
      VectorXd t = selected_samples[k].head(3);
      selected_thetas.push_back(t);
      for (int i = 0; i < vsize; i++)
      {
        VectorXd h(lb_hpars.size());
        for (int j = 0; j < h.size(); j++)
        {
          h(j) = selected_samples[k](3 + j + i * h.size());
        }
        selected_hpars_vec[i].push_back(h);
      }
    }

    //on tente les prédictions à la fin.
    for (int i = 0; i < vDens_fb.size(); i++)
    {
      vDens_fb[i].SetNewSamples(selected_thetas);
      vDens_fb[i].SetNewHparsOfSamples(selected_hpars_vec[i]);
      string fn = "results/bayes/preds" + to_string(cases[i]) + ".gnu";
      string f1 = "results/bayes/sampsf" + to_string(cases[i]) + ".gnu";
      string f2 = "results/bayes/sampsz" + to_string(cases[i]) + ".gnu";
      vDens_fb[i].WritePredictions(prediction_points, fn);
      vDens_fb[i].WriteSamplesFandZ(prediction_points, f1, f2);
    }
  }

  exit(0);

  //calibration KOH pooled.
  //MCMC KOH pooled.
  {
    vector<VectorXd> hparskoh_pooled = HparsKOH_pooled(vDens, hpars_z_guess, 300); //300 secondes semblent suffisantes aux tests.
    int nsteps_mcmc = 5e5;
    int nsamples_mcmc = 3000;

    auto get_hpars = [&hparskoh_pooled](VectorXd const &theta)
    {
      return hparskoh_pooled;
    };

    auto compute_score = [&vDens](vector<VectorXd> const &h, VectorXd const &theta)
    {
      double res = 0;
      for (int i = 0; i < vDens.size(); i++)
      {
        res += vDens[i].loglikelihood_theta(theta, h[i]);
      }
      res += logprior_pars(theta);
      return res;
    };

    auto in_bounds = [&vDens](VectorXd const &X)
    {
      return vDens[0].in_bounds_pars(X);
    };
    //run mcmc et extraire échantillons.
    vector<VectorXd> allsteps = Run_MCMC(nsteps_mcmc, X_init_mcmc, COV_init, compute_score, get_hpars, in_bounds, generator);
    vector<VectorXd> selected_samples(nsamples_mcmc);
    for (int j = 0; j < selected_samples.size(); j++)
    {
      selected_samples[j] = allsteps[j * (allsteps.size() / selected_samples.size())];
    }
    //diagnostic
    Selfcor_diagnosis(allsteps, nautocor, 1, "results/koh/autocor.gnu");
    //je veux construire un vecteur avec tous les samples et l'écrire. ça sera + facile pour la covariance aussi. les hpars sont passés en exponentielle.
    vector<VectorXd> selected_samples_withhpars;
    vector<vector<VectorXd>> selected_hpars_vec(vsize);
    for (const auto &theta : selected_samples)
    {
      vector<VectorXd> h = get_hpars(theta);
      VectorXd maxsample(3 + 2 * h.size());
      maxsample.head(3) = theta;
      for (int i = 0; i < h.size(); i++)
      {
        maxsample(3 + 2 * i) = exp(h[i](0));
        maxsample(3 + 2 * i + 1) = exp(h[i](1));
      }
      selected_samples_withhpars.push_back(maxsample);
      for (int i = 0; i < vsize; i++)
      {
        selected_hpars_vec[i].push_back(h[i]);
      }
    }
    for (int i = 0; i < vDens.size(); i++)
    {
      vDens[i].SetNewSamples(selected_samples);
      vDens[i].SetNewHparsOfSamples(selected_hpars_vec[i]);
      string fn = "results/koh/preds" + to_string(cases[i]) + ".gnu";
      string f1 = "results/koh/sampsf" + to_string(cases[i]) + ".gnu";
      string f2 = "results/koh/sampsz" + to_string(cases[i]) + ".gnu";
      vDens[i].WritePredictions(prediction_points, fn);
      vDens[i].WriteSamplesFandZ(prediction_points, f1, f2);
    }
    writeVector("results/koh/sample.gnu", selected_samples_withhpars); //OK.
  }






  exit(0);

    //étape 1 bis fmp: construction des vDopt, thetas,_train, hpars_train à partir de fichiers. ne doit pa être run en même temps que étape 1.
  {
    if (!vDopt.size() == 0)
    {
      cerr << "erreur !!!!!!!! vDopt déjà construit" << endl;
      exit(0);
    }
    string foldname = "results/hparsopt/200/";
    //build thetas_train
    auto temp = ReadVector(foldname + "hopt3.gnu");
    for (const auto v : temp)
    {
      VectorXd t = v.head(3);
      thetas_train.push_back(t);
    }
    cout << "thetas_train loaded. size : " << thetas_train.size() << endl;
    for (int i = 0; i < vsize; i++)
    {
      DensityOpt Dopt(vDens[i]);
      temp = ReadVector(foldname + "hopt" + to_string(cases[i]) + ".gnu");
      vector<VectorXd> hopt_train;
      for (const auto v : temp)
      {
        VectorXd h = v.tail(lb_hpars.size());
        hopt_train.push_back(h);
      }
      Dopt.BuildHGPs(thetas_train, hopt_train, Kernel_GP_Matern32);
      Dopt.OptimizeHGPs(Bounds_hpars_gp, hpars_gp_guess, 1);
      vDopt.push_back(Dopt);
      hpars_train.push_back(hopt_train);
      if (!hopt_train.size() == thetas_train.size())
      {
        cerr << "erreur taille" << endl;
        exit(0);
      }
    }
  }



  exit(0);
}
