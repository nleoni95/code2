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

typedef map<int, VectorXd> map_doe;             //key : int, value : vectorXd
typedef map<int, vector<VectorXd>> map_results; //key : int, value : vecteur de VectorXd
typedef map<string, VectorXd> map_exp;          //contient les valeurs expérimentales

double const flux_nominal = 128790;

std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0, 1);
std::normal_distribution<double> distN(0, 1);
vector<DATA> data;
vector<VectorXd> Grid;

double computer_model(const double &x, const VectorXd &t)
{
  //n'oublions pas que t sont dans [0,1]. Il faut les remettre au goût du jour.
  return x * sin(2 * t(0) * x) + (x + 0.15) * (1 - t(1));
}

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //squared exponential
  double d = abs(x(0) - y(0));
  return pow(hpar(0), 2) * exp(-0.5 * pow(d / hpar(1), 2));
}

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //squared exponential
  double d = abs(x(0) - y(0));
  return pow(hpar(0), 2) * (1 + (d / hpar(2)) + (1. / 3) * pow(d / hpar(2), 2)) * exp(-d / hpar(2)); //5/2
}

double gradKernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar, int index)
{
  //dérivées du kernel matern 52.
  double d = abs(x(0) - y(0));
  if (index == 0)
  {
    //sedm
    return 2 * hpar(0) * (1 + (d / hpar(2)) + (1. / 3) * pow(d / hpar(2), 2)) * exp(-d / hpar(2));
  }
  if (index == 1)
  {
    //sobs
    if (x(0) == y(0))
    {
      return 2 * hpar(1);
    }
    else
    {
      return 0;
    }
  }
  if (index == 2)
  {
    //lcor
    double X = d / hpar(2);
    return pow(hpar(0), 2) * exp(-X) * pow(X, 2) * (d + hpar(2)) / (3 * pow(hpar(2), 2));
  }
  cout << "error grad " << endl;
  return 0;
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double d1 = abs(x(0) - y(0)) / hpar(1);
  double d2 = abs(x(1) - y(1)) / hpar(3);
  double d3 = abs(x(2) - y(2)) / hpar(4);
  double d4 = abs(x(3) - y(3)) / hpar(5);
  double d5 = abs(x(4) - y(4)) / hpar(6);
  double cor = -d1 - d2 - d3 - d4 - d5;
  cor = exp(cor) * (1 + d1) * (1 + d2) * (1 + d3) * (1 + d4) * (1 + d5);
  return pow(hpar(0), 2) * cor;
}

double Kernel_GP_SQexp(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double d1 = pow((x(0) - y(0)) / hpar(1), 2);
  double d2 = pow((x(1) - y(1)) / hpar(3), 2);
  double d3 = pow((x(2) - y(2)) / hpar(4), 2);
  double d4 = pow((x(3) - y(3)) / hpar(5), 2);
  double d5 = pow((x(4) - y(4)) / hpar(6), 2);
  double cor = -d1 - d2 - d3 - d4 - d5;
  cor = exp(cor);
  return pow(hpar(0), 2) * cor;
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

double gradlogprior_hpars(VectorXd const &hpars, int i)
{
  return 0;
}

double logprior_pars(VectorXd const &pars)
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

VectorXd RtoGP(const VectorXd &X)
{
  //passage de l'espace réel à l'espace GP dans [0,1]. transfo linéaire
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
  //passage de l'espace GP [0,1] à l'espace réel. transfo linéaire
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
/* DEBORA-related functions*/

int line_count(string const &filename)
{
  //renvoie le nombre de lignes dans un fichier
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
  //lecture du DoE à filename et écriture dans la map
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
      //décomposition de la line en des mots
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>()); //on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      //traitement des mots. Le premier est le num du cas, les 3-4-5-6-7 sont les paramètres.
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
  //lit les résultats dans un fichier donné et les rend sous forme vector de VectorXd.
  //renvoie un vecteur vide si le fichier n'existe pas.
  //architecture des fichiers lus : 1ère ligne à ignorer.
  //colonnes : X, alpha, Dbul.
  vector<VectorXd> v(5);
  int nlines = line_count(filename) - 1; //ignorer la 1ère ligne
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
      //décomposition de la line en mots
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>()); //on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      //traitement des mots.
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

map_results read_results_qmc(string const &foldname)
{
  //lecture de tous les résultats de calcul et écriture dans une map_results.
  //l'argument sera par exemple "clean_profile.dat"
  map_results m;
  for (int i = 1; i < 2041; i++)
  {
    string fullname = foldname + "/" + to_string(i) + "/clean_profile.dat";
    vector<VectorXd> v = read_singleresult(fullname);
    if (!v.empty())
    {
      m.insert(make_pair(i, v));
    }
  }
  cout << m.size() << " simulations read." << endl;
  return m;
}

map_results read_results_lhs(string const &foldname)
{
  //lecture de tous les résultats de calcul et écriture dans une map_results.
  //l'argument sera par exemple "clean_profile.dat"
  map_results m;
  for (int i = 1; i < 2041; i++)
  {
    string fullname = foldname + "/" + to_string(i) + "/clean_profile.dat";
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
  //lecture du fichier de données expérimentales
  map_exp m;
  int nlines = line_count(filename) - 1; //on retire la première ligne
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
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>()); //on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      //traitement des mots.
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

MatrixXd EvaluateMeanGPPCA(vector<GP> const &vgp, VectorXd const &Target, MatrixXd const &VP, MatrixXd const &Acoefs, VectorXd const &featureMeans)
{
  //renvoie moyenne et variance prédites par un vecteur vgp, dans le cadre PCA. Au paramètre Target, en coordonnées GP. Les points d'évaluation sont les mêmes que ceux utilisés pour la construction de la base.
  //prédiction des coeffcients moyens et des variances moyennes
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

VectorXd interpolate(VectorXd const &Yorig, VectorXd const &Xorig, VectorXd const &Xnew)
{
  //interpolation des données Yorig, définies sur Xorig, sur le nouveau grid Xnew.
  //Les grids sont supposés ordonnés.
  if (Yorig.size() != Xorig.size())
  {
    cerr << "erreur d'interpolation : taille différente." << Yorig.size() << " " << Xorig.size() << endl;
  }
  VectorXd Ynew(Xnew.size());
  for (int i = 0; i < Xnew.size(); i++)
  {
    //check si on est au-delà des bornes de Xnew
    double ynext = 0; //coordonnées dans l'espace d'origine
    double yprev = 0;
    double xnext = 0;
    double xprev = 0;
    if (Xnew(i) < Xorig(0))
    {
      //on créé une valeur deux fois plus loin à partir de la pente estimée
      ynext = Yorig(0);
      xnext = Xorig(0);
      xprev = 2 * Xnew(i) - Xorig(0);
      double slope = (Yorig(1) - Yorig(0)) / (Xorig(1) - Xorig(0));
      yprev = ynext - slope * (xnext - xprev);
    }
    else if (Xnew(i) > Xorig(Xorig.size() - 1))
    {
      //pareil, on créée une valeur deux fois plus loin.
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
      //indice devient l'indice du immédiatement supérieur.
      ynext = Yorig(indice);
      xnext = Xorig(indice);
      yprev = Yorig(indice - 1);
      xprev = Xorig(indice - 1);
    }
    //interpolation linéaire
    double m = (ynext - yprev) / (xnext - xprev);
    double b = ynext - m * xnext;
    Ynew(i) = m * Xnew(i) + b;
  }
  return Ynew;
}

vector<vector<DATA>> PerformPCA(map_doe const &m, map_results const &mr, int qte, MatrixXd &VP, MatrixXd &Acoefs, VectorXd &featureMeans, int nmodes)
{
  //réalise la PCA de la quantité qte. 1=tdv, 2=diametre. VP = vecteurs propres réduits, Acoefs = normalisation des coefficients appris. nmodes = nombre de modes choisis.
  //construction de la matrice des données
  int ncalcs = mr.size();           //nombre de calculs réussis
  int nrayons = mr.at(1)[0].size(); //nombre de points de mesure en rayon.
  MatrixXd U(nrayons, ncalcs);
  MatrixXd P(5, ncalcs); //contient les paramètres des DoE, les colonnes correspondents aux colonnes de U.
  for (int i = 0; i < ncalcs; i++)
  {
    auto it = next(mr.cbegin(), i);
    U.col(i) = (*it).second[qte];        //1 car on regarde le taux de vide.
    P.col(i) = RtoGP(m.at((*it).first)); //on store les valeurs des paramètres correspondant aux calculs, dans les coordonnées GP.
  }
  //on retranche à chaque colonne la moyenne des colonnes https://stackoverflow.com/questions/33531505/principal-component-analysis-with-eigen-library
  featureMeans = U.rowwise().mean(); //vecteur colonne de taille nrayons
  U = U.colwise() - featureMeans;
  MatrixXd Covmatrix = U * U.transpose(); //taille nrayons,nrayons
  Covmatrix /= (ncalcs);
  //décomp. valeurs propres et vp
  SelfAdjointEigenSolver<MatrixXd> eig(Covmatrix);
  //valeurs propres
  VectorXd lambdas = eig.eigenvalues();     //nrayons
  MatrixXd vecpropres = eig.eigenvectors(); //(nrayons,nrayons)
  //cout << "lambdas : " << lambdas.transpose() << endl;
  //cout << "ev : " << vecpropres << endl;
  //vérification : vecpropres.transpose()*vecpropres vaut l'identité.

  //sélection de nsel modes
  MatrixXd VPs = vecpropres.rightCols(nmodes);       //(nrayons,nmodes)
  VectorXd lambdas_red = lambdas.bottomRows(nmodes); //nmodes
  //on reverse les vecteurs propres et valeurs propres pour que les principaux se trouvent à la position 0.
  lambdas_red.reverseInPlace();
  VP = VPs.rowwise().reverse();
  cout << "Sélection de " << nmodes << " modes." << endl;
  cout << "VP principales : " << lambdas_red.transpose() << endl;
  cout << "Quantité d'énergie conservée : " << 100 * lambdas_red.array().sum() / lambdas.array().sum() << " %" << endl;
  //vérification qu'on a bien choisi des vecteurs propres : on a bien vecred.transpose()*vecred=Id
  //calcul de la matrice des coefficients à apprendre
  MatrixXd A = VP.transpose() * U; //(nmodes,ncalcs)
  //les lignes de A sont déjà du même ordre de grandeur.
  //remarque : les lignes de A somment à 0..
  VectorXd Ascale = lambdas_red.array().sqrt();
  Acoefs = Ascale.asDiagonal(); //matrice diagonale avec les ordres de grandeur de A.
  MatrixXd normedA = Acoefs.inverse() * A;
  //on exporte le tout sous forme de vecteur<DATA>
  vector<vector<DATA>> vd(nmodes);
  for (int j = 0; j < nmodes; j++)
  {
    vector<DATA> v(ncalcs);
    for (int i = 0; i < ncalcs; i++)
    {
      DATA dat;
      dat.SetX(P.col(i));
      dat.SetValue(normedA(j, i)); //P déjà en coordonnées gp.
      v[i] = dat;
    }
    vd[j] = v;
  }
  return vd;
}

void eval_erreur_validation(MatrixXd const &M_truth, MatrixXd const &M_projected, MatrixXd const &M_predicted)
{
  //évaluation des erreurs de validation et répartition de l'erreur.
  int ncalcs = M_truth.cols();
  //définition du produit scalaire
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
}

void compute_erreurs_validation(int qte, map_doe const &m_lhs, map_results const &mr_lhs, vector<GP> const &vgp, MatrixXd const &VP, MatrixXd const &Acoefs, VectorXd const &featureMeans)
{
  //on récupère les erreurs sur le dataset de validation.
  //étape 1 : mettre les données de validation dans une matrice
  int ncalcs = mr_lhs.size();
  int nmodes = Acoefs.cols();
  int nrayons = VP.rows();
  MatrixXd M_truth(nrayons, ncalcs); //valeurs calculées sur le dataset
  MatrixXd P_truth(5, ncalcs);       //valeurs des 5 paramètres sur le dataset (coordonnées GP)
  for (int i = 0; i < ncalcs; i++)
  {
    auto it = next(mr_lhs.cbegin(), i);
    M_truth.col(i) = (*it).second[qte];            //1 car on regarde le taux de vide.
    P_truth.col(i) = RtoGP(m_lhs.at((*it).first)); //on store les valeurs des paramètres correspondant aux calculs, dans les coordonnées GP.
  }
  //projeter le dataset sur la base VP.
  MatrixXd M_projected(nrayons, ncalcs);
  MatrixXd M_truth_centered = M_truth.colwise() - featureMeans;
  MatrixXd M_truth_multiplied = VP * VP.transpose() * (M_truth_centered);
  M_projected = (M_truth_multiplied).colwise() + featureMeans; //M_proj=featureMeans+VPtVP(M_truth-featureMeans)
  //calcul des prédictions moyennes GP
  MatrixXd M_predicted(nrayons, ncalcs);
  for (int i = 0; i < ncalcs; i++)
  {
    VectorXd ParamEval = P_truth.col(i);                                              //paramètres du calcul i (coords GP)
    M_predicted.col(i) = EvaluateMeanGPPCA(vgp, ParamEval, VP, Acoefs, featureMeans); //on prend seulement les prédictions moyennes.
  }
  //calcul des erreurs. Faisons sur tout le domaine.
  auto afficher_erreurs = [M_truth, M_projected, M_predicted, ncalcs](int nstart, int nend) -> void
  {
    MatrixXd M_truth_2 = M_truth.block(nstart, 0, nend - nstart + 1, ncalcs);
    MatrixXd M_projected_2 = M_projected.block(nstart, 0, nend - nstart + 1, ncalcs);
    MatrixXd M_predicted_2 = M_predicted.block(nstart, 0, nend - nstart + 1, ncalcs);
    eval_erreur_validation(M_truth_2, M_projected_2, M_predicted_2);
  };
  cout << "sur tout le domaine : " << endl;
  afficher_erreurs(0, nrayons - 1);
  cout << "à la paroi : " << endl;
  afficher_erreurs(26, 39);
  cout << "au milieu du canal : " << endl;
  afficher_erreurs(10, 25);
  cout << "au coeur du canal : " << endl;
  afficher_erreurs(0, 9);
}

/*Concerne l'étude de l'algo adaptatif*/

VectorXd evaluate_surrogate(vector<VectorXd> const &thetas_ref, vector<VectorXd> const &hpars_ref, function<pair<VectorXd, VectorXd>(VectorXd const &)> const &get_hpars_and_var)
{
  //évaluation de la qualité d'un surrogate par la fonction get_hpars qu'il propose. On pourra en profiter pour vérifier que pour la densité fmp e référence ça fait bien zéro.
  //prenons l'erreur moyenne L2 relative ? Non. On va faire la moyenne + variance. averaged absolute individual standardized error. Et la moyenne a posteriori de cette quantité.
  //calcul de l'erreur dans chaque dimension.
  if (!thetas_ref.size() == hpars_ref.size())
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

VectorXd evaluate_surrogate_log(vector<VectorXd> const &thetas_ref, vector<VectorXd> const &hpars_ref, function<pair<VectorXd, VectorXd>(VectorXd const &)> const &get_hpars_and_var)
{
  //le hpars (0) est en échelle log.
  //évaluation de la qualité d'un surrogate par la fonction get_hpars qu'il propose. On pourra en profiter pour vérifier que pour la densité fmp e référence ça fait bien zéro.
  //prenons l'erreur moyenne L2 relative ? Non. On va faire la moyenne + variance. averaged absolute individual standardized error. Et la moyenne a posteriori de cette quantité.
  //calcul de l'erreur dans chaque dimension.
  if (!thetas_ref.size() == hpars_ref.size())
  {
    cerr << "erreur : différentes tailles evaluate_surrogate !" << endl;
    exit(0);
  }
  auto hparsrefmod = hpars_ref;
  for (int i = 0; i < hparsrefmod.size(); i++)
  {
    hparsrefmod[i](0) = log(hparsrefmod[i](0));
  }
  VectorXd err = VectorXd::Zero(hparsrefmod[0].size());
  for (int i = 0; i < thetas_ref.size(); i++)
  {
    auto p = get_hpars_and_var(thetas_ref[i]);
    VectorXd std = p.second.array().sqrt();
    VectorXd v = (hparsrefmod[i] - p.first).cwiseQuotient(std).array().abs();
    err += v;
  }
  return err / thetas_ref.size();
}

VectorXd evaluate_surrogate_bof(vector<VectorXd> const &thetas_ref, vector<VectorXd> const &hpars_ref, function<pair<VectorXd, VectorXd>(VectorXd const &)> const &get_hpars_and_var)
{
  //évaluation de la qualité d'un surrogate par la fonction get_hpars qu'il propose. On pourra en profiter pour vérifier que pour la densité fmp e référence ça fait bien zéro.
  //On va juste faire l'erreur L2 relative. relou mais avec la variance OLM ça marche pas trop bien.
  if (!thetas_ref.size() == hpars_ref.size())
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
  return err / thetas_ref.size();
}

VectorXd evaluate_surrogate_bofplusscores(vector<VectorXd> const &thetas_ref, vector<VectorXd> const &hpars_ref, vector<double> const &scores_ref, function<pair<VectorXd, VectorXd>(VectorXd const &)> const &get_meanpred, function<double(VectorXd const &, VectorXd const &)> const &get_score)
{
  //On va juste faire l'erreur L2 relative. relou mais avec la variance OLM ça marche pas trop bien.
  //dernière composante : l'erreur L2 sur la fct likelihood elle-même. Plus interprétable.
  if (!thetas_ref.size() == hpars_ref.size())
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
int main(int argc, char **argv)
{
  default_random_engine generator(123456);

  map_doe m = read_doe("../../DEBORA/data/qmc/design_qmc_full.dat");     //read the QMC points
  map_doe m_lhs = read_doe("../../DEBORA/data/lhs/design_lhs_full.dat"); //read the LHS points
  map_results mr = read_results_qmc("../../DEBORA/data/qmc");            //read the QMC results
  map_results mr_lhs = read_results_lhs("../../DEBORA/data/lhs");        //read the LHS results
  map_exp me = read_exp_data("../../DEBORA/data/clean_exp.dat");         //read the observations (X, alpha, D)

  /*Construction du surrogate DEBORA*/
  /*Surrogate pour le taux de vide*/
  vector<GP> vgp_a(3);
  MatrixXd VP_a;
  MatrixXd Acoefs_a;
  VectorXd featureMeans_a;
  {
    //initialisation des variables
    int nmodes = 3;
    int ncalcs = mr.size();           //nombre de calculs réussis
    int nrayons = mr.at(1)[0].size(); //nombre de points de mesure en rayon.
    MatrixXd VP(nrayons, nmodes);
    MatrixXd Acoefs(nmodes, nmodes);
    VectorXd featureMeans(nrayons);
    vector<vector<DATA>> full_data = PerformPCA(m, mr, 1, VP, Acoefs, featureMeans, nmodes); //PCA du taux de vide

    //construction des gps
    vector<GP> vgp(nmodes);

    int nhpars_gp = 7;
    MatrixXd Bounds_hpars_gp(2, nhpars_gp);
    Bounds_hpars_gp(0, 0) = 1E-3;
    Bounds_hpars_gp(1, 0) = 1e2; //variance
    Bounds_hpars_gp(0, 2) = 1E-3;
    Bounds_hpars_gp(1, 2) = 2E-3; //sigma obs
    list<int> l = {1, 3, 4, 5, 6};
    for (int i : l)
    {
      Bounds_hpars_gp(0, i) = 1E-2;
      Bounds_hpars_gp(1, i) = 2; //lcors.
    }
    VectorXd hpars_gp_guess(7);
    for (int i = 0; i < nhpars_gp; i++)
    {
      hpars_gp_guess(i) = 0.5 * (Bounds_hpars_gp(1, i) + Bounds_hpars_gp(0, i));
    }

    VectorXd hpars_gp0(7);
    hpars_gp0 << 0.77, 1.06, 2e-3, 1.84, 1.46, 0.15, 0.58; //matern 3 modes

    GP gp0(Kernel_GP_Matern32);
    gp0.SetData(full_data[0]);
    gp0.SetGP(hpars_gp0);

    VectorXd hpars_gp1(7);
    hpars_gp1 << 1.22, 0.54, 2e-3, 0.93, 0.79, 0.18, 1.02; //matern 3 modes

    GP gp1(Kernel_GP_Matern32);
    gp1.SetData(full_data[1]);
    gp1.SetGP(hpars_gp1);

    VectorXd hpars_gp2(7);
    hpars_gp2 << 0.93, 0.39, 2e-3, 0.67, 0.55, 0.16, 0.83; //matern 3 modes

    GP gp2(Kernel_GP_Matern32);
    gp2.SetData(full_data[2]);
    gp2.SetGP(hpars_gp2);

    vgp[0] = gp0;
    vgp[1] = gp1;
    vgp[2] = gp2;

    compute_erreurs_validation(1, m_lhs, mr_lhs, vgp, VP, Acoefs, featureMeans);

    //On recopie tout dans des variables extérieures
    vgp_a = vgp;
    VP_a = VP;
    Acoefs_a = Acoefs;
    featureMeans_a = featureMeans;
  }

  /*Surrogate pour le diamètre de bulle*/
  vector<GP> vgp_d(5);
  MatrixXd VP_d;
  MatrixXd Acoefs_d;
  VectorXd featureMeans_d;
  {
    //initialisation des variables
    int nmodes = 5;
    int ncalcs = mr.size();           //nombre de calculs réussis
    int nrayons = mr.at(1)[0].size(); //nombre de points de mesure en rayon.
    MatrixXd VP(nrayons, nmodes);
    MatrixXd Acoefs(nmodes, nmodes);
    VectorXd featureMeans(nrayons);
    vector<vector<DATA>> full_data = PerformPCA(m, mr, 2, VP, Acoefs, featureMeans, nmodes); //PCA du diamètre

    //construction des gps
    vector<GP> vgp(nmodes);

    int nhpars_gp = 7;
    MatrixXd Bounds_hpars_gp(2, nhpars_gp);
    Bounds_hpars_gp(0, 0) = 1E-3;
    Bounds_hpars_gp(1, 0) = 1e2; //variance
    Bounds_hpars_gp(0, 2) = 1E-3;
    Bounds_hpars_gp(1, 2) = 2E-3; //sigma obs
    list<int> l = {1, 3, 4, 5, 6};
    for (int i : l)
    {
      Bounds_hpars_gp(0, i) = 1E-2;
      Bounds_hpars_gp(1, i) = 2; //lcors.
    }
    VectorXd hpars_gp_guess(7);
    for (int i = 0; i < nhpars_gp; i++)
    {
      hpars_gp_guess(i) = 0.5 * (Bounds_hpars_gp(1, i) + Bounds_hpars_gp(0, i));
    }

    VectorXd hpars_gp0(7);
    hpars_gp0 << 1.18, 1.54, 2e-3, 0.91, 1.13, 0.3, 0.97; //matern 3 modes

    GP gp0(Kernel_GP_Matern32);
    gp0.SetData(full_data[0]);
    gp0.SetGP(hpars_gp0);

    VectorXd hpars_gp1(7);
    hpars_gp1 << 1.8, 0.73, 2e-3, 0.71, 0.827, 0.26, 1.14; //matern 3 modes

    GP gp1(Kernel_GP_Matern32);
    gp1.SetData(full_data[1]);
    gp1.SetGP(hpars_gp1);

    VectorXd hpars_gp2(7);
    hpars_gp2 << 0.97, 0.26, 2e-3, 0.29, 0.37, 0.21, 0.96; //matern 3 modes

    GP gp2(Kernel_GP_Matern32);
    gp2.SetData(full_data[2]);
    gp2.SetGP(hpars_gp2);

    VectorXd hpars_gp3(7);
    hpars_gp3 << 1.08, 0.25, 2e-3, 0.34, 0.37, 0.23, 0.98;

    GP gp3(Kernel_GP_Matern32);
    gp3.SetData(full_data[3]);
    gp3.SetGP(hpars_gp3);

    VectorXd hpars_gp4(7);
    hpars_gp4 << 0.98, 0.20, 2e-3, 0.38, 0.38, 0.17, 0.69;

    GP gp4(Kernel_GP_Matern32);
    gp4.SetData(full_data[4]);
    gp4.SetGP(hpars_gp4);

    VectorXd hpars_gp5(7);
    hpars_gp5 << 0.89, 0.79, 1e-3, 0.26, 1.2, 0.69, 1.66;

    vgp[0] = gp0;
    vgp[1] = gp1;
    vgp[2] = gp2;
    vgp[3] = gp3;
    vgp[4] = gp4;

    compute_erreurs_validation(2, m_lhs, mr_lhs, vgp, VP, Acoefs, featureMeans);
    //On recopie tout dans des variables extérieures
    vgp_d = vgp;
    VP_d = VP;
    Acoefs_d = Acoefs;
    featureMeans_d = featureMeans;
  }

  /*Récupération des observations et interpolation sur le maillage*/
  VectorXd Yexp_diam = interpolate(me["D"], me["X"], mr[1][0]);
  VectorXd Yexp_alpha = interpolate(me["Alpha"], me["X"], mr[1][0]);
  vector<VectorXd> location_points;
  for (int i = 0; i < mr[1][0].size(); i++)
  {
    VectorXd x(1);
    x << mr[1][0](i);
    location_points.push_back(x);
  }

  string exp1 = "results/obsalpha.gnu";
  string exp2 = "results/obsdiam.gnu";
  WriteObs(location_points, Yexp_alpha, exp1);
  WriteObs(location_points, Yexp_diam, exp2);

  /*Construction des fonctions d'appel*/
  //Attention ! Le surrogate n'est construit que sur 40 points radiaux. On ne peut donc pas l'interroger sur d'autres points. Les points de prédiction sont donc obligatoirement égaux aux points d'observation, et la structure de la lambdafunction pour le modèle est simplifiée (ne dépend pas du point X)
  auto lambda_model_diam = [&vgp_d, &VP_d, &featureMeans_d, &Acoefs_d](vector<VectorXd> const &X, VectorXd const &theta) -> VectorXd
  {
    return EvaluateMeanGPPCA(vgp_d, theta, VP_d, Acoefs_d, featureMeans_d);
  };

  auto lambda_model_alpha = [&vgp_a, &VP_a, &featureMeans_a, &Acoefs_a](vector<VectorXd> const &X, VectorXd const &theta) -> VectorXd
  {
    return EvaluateMeanGPPCA(vgp_a, theta, VP_a, Acoefs_a, featureMeans_a);
  };

  auto lambda_priormean_alpha = [&location_points](vector<VectorXd> const &X, VectorXd const &hpars)
  {
    return VectorXd::Zero(location_points.size());
  };

  auto lambda_priormean_diam = [&location_points](vector<VectorXd> const &X, VectorXd const &hpars)
  {
    return VectorXd::Zero(location_points.size());
  };

  /*Paramètres de simulation*/
  //pour la MCMC
  int nombre_steps_mcmc = 1e5;
  int nombre_samples_collected = 1000; //1 sample tous les 100.
  int nautocor = 2000;

  int time_opt_opti = 10;      // 10 secondes par optimisation opti
  int time_opt_koh_loc = 600;  // 10 minutes par optimisation KOH locale
  int time_opt_koh_glo = 7200; // 2h pour optimisation KOH globale

  // Bornes sup des paramètres et hyperparamètres
  int dim_theta = 5;
  int dim_hpars_alpha = 3;
  int dim_hpars_diam = 3;

  VectorXd lb_t = VectorXd::Zero(dim_theta);
  VectorXd ub_t = VectorXd::Ones(dim_theta);
  /* before modif
  VectorXd lb_hpars_alpha(dim_hpars_alpha);
  VectorXd ub_hpars_alpha(dim_hpars_alpha);
  lb_hpars_alpha << 1e-4, 1e-4, 1e-4;
  ub_hpars_alpha << 1, 1e-1, 1e-1;
  VectorXd lb_hpars_diam(dim_hpars_diam);
  VectorXd ub_hpars_diam(dim_hpars_diam);
  lb_hpars_diam << 2e-5, 1e-8, 5e-4;
  ub_hpars_diam << 5e-3, 3e-5, 1e-1;
  */
  VectorXd lb_hpars_alpha(dim_hpars_alpha);
  VectorXd ub_hpars_alpha(dim_hpars_alpha);
  lb_hpars_alpha << 1e-2, 1e-3, 1e-3;
  ub_hpars_alpha << 1, 1e-1, 1e-1;
  VectorXd lb_hpars_diam(dim_hpars_diam);
  VectorXd ub_hpars_diam(dim_hpars_diam);
  lb_hpars_diam << 1e-4, 1e-7, 1e-3;
  ub_hpars_diam << 1e-2, 1e-4, 1e-1;

  VectorXd hpars_z_guess_alpha = 0.5 * (lb_hpars_alpha + ub_hpars_alpha);
  VectorXd hpars_z_guess_diam = 0.5 * (lb_hpars_diam + ub_hpars_diam);

  //bounds for HGPs hyperparameters
  int nhpars_gp = 7;
  MatrixXd Bounds_hpars_hgps(2, 7);
  Bounds_hpars_hgps(0, 0) = 1e-1;
  Bounds_hpars_hgps(1, 0) = 3; //variance
  Bounds_hpars_hgps(0, 2) = 1E-10;
  Bounds_hpars_hgps(1, 2) = 1e-9; //exp noise
  list<int> l = {1, 3, 4, 5, 6};
  for (int i : l)
  {
    Bounds_hpars_hgps(0, i) = 1E-5;
    Bounds_hpars_hgps(1, i) = 8; //cor lengths
  }
  VectorXd hpars_hgp_guess = 0.5 * (Bounds_hpars_hgps.row(0) + Bounds_hpars_hgps.row(1)).transpose();

  //building initial DoE
  DoE doe_init(lb_t, ub_t, 100, 10); //doe halton de 100 points
  doe_init.WriteGrid("results/grid.gnu");

  Density Dalpha(doe_init);
  Dalpha.SetModel(lambda_model_alpha);
  Dalpha.SetKernel(Kernel_Z_Matern52);
  Dalpha.SetHparsBounds(lb_hpars_alpha, ub_hpars_alpha);
  Dalpha.SetLogPriorHpars(logprior_hpars);
  Dalpha.SetLogPriorPars(logprior_pars);
  Dalpha.SetPriorMean(lambda_priormean_alpha);
  Dalpha.SetObservations(location_points, Yexp_alpha);
  Dalpha.SetOutputerr(true, 1e-4, 1);

  Density Ddiam(doe_init);
  Ddiam.SetModel(lambda_model_diam);
  Ddiam.SetKernel(Kernel_Z_Matern52);
  Ddiam.SetHparsBounds(lb_hpars_diam, ub_hpars_diam);
  Ddiam.SetLogPriorHpars(logprior_hpars);
  Ddiam.SetLogPriorPars(logprior_pars);
  Ddiam.SetPriorMean(lambda_priormean_diam);
  Ddiam.SetObservations(location_points, Yexp_diam);
  Ddiam.SetOutputerr(true, 1e-4, 1);

  VectorXd X_init_mcmc = 0.5 * VectorXd::Ones(dim_theta);
  MatrixXd COV_init = pow(0.05, 2) * MatrixXd::Identity(5, 5);
  cout << "COV_init : " << endl
       << COV_init << endl;

  ///TEST PHASE OPTI
  //on sait que hparsopt grad est bon pour 1e-3 secondes d'optimisation. Comparons, sur 500 points, 1e-3, 1e-2 et 1e-1.

  /* test sur les optimisations
  {
    auto thetas=doe_init.GetGrid();
    cout << "comparaison avec nthetas = " << thetas.size() << endl;
    DensityOpt Doalpha(Dalpha);
    DensityOpt Dodiam(Ddiam);
    Doalpha.SetKernelGrads(gradKernel_Z_Matern52);
    Dodiam.SetKernelGrads(gradKernel_Z_Matern52);
    Doalpha.SetLogpriorGrads(gradlogprior_hpars);
    Dodiam.SetLogpriorGrads(gradlogprior_hpars);

    auto eval = [&Doalpha, &Dodiam, &hpars_z_guess_alpha, &hpars_z_guess_diam](VectorXd const &X, double time)
    {
      //cout << "time : " << time << endl;
      VectorXd hparsopt_alpha = Doalpha.HparsOpt(X, hpars_z_guess_alpha, time);
      VectorXd hparsopt_diam = Dodiam.HparsOpt(X, hpars_z_guess_diam, time);
      auto p = make_pair(hparsopt_alpha, hparsopt_diam);
      //cout << "hpars alpha : " << p.first.transpose() << endl;
      //cout << "hpars diam : " << p.second.transpose() << endl;
      double ll1 = Doalpha.loglikelihood_theta(X, p.first);
      double ll2 = Dodiam.loglikelihood_theta(X, p.second);
      double lp = logprior_pars(X);
      return ll1 + ll2 + lp;
      //cout << "score : " << ll1 + ll2 + lp << endl
    };
    auto eval_grad = [&Doalpha, &Dodiam, &hpars_z_guess_alpha, &hpars_z_guess_diam](VectorXd const &X, double time)
    {
      //cout << "(grad) time : " << time << endl;
      VectorXd hparsopt_alpha = Doalpha.HparsOpt_withgrad(X, hpars_z_guess_alpha, time);
      VectorXd hparsopt_diam = Dodiam.HparsOpt_withgrad(X, hpars_z_guess_diam, time);
      auto p = make_pair(hparsopt_alpha, hparsopt_diam);
      //cout << "hpars alpha : " << p.first.transpose() << endl;
      //cout << "hpars diam : " << p.second.transpose() << endl;
      double ll1 = Doalpha.loglikelihood_theta(X, p.first);
      double ll2 = Dodiam.loglikelihood_theta(X, p.second);
      double lp = logprior_pars(X);
      return ll1 + ll2 + lp;
      //cout << "score : " << ll1 + ll2 + lp << endl
    };

    vector<double> v = {1e-3, 1e-2, 1e-1};
    for (auto time : v)
    {
      cout << "time : " << time << endl;
      double score=0;
      for(auto t:thetas)
      {
        score+=eval(t,time);
      }
      cout << "score nograd : " << score << endl;
      score=0;
      for(auto t:thetas)
      {
        score+=eval_grad(t,time);
      }
      cout << "score grad : " << score << endl << endl;
    }
    exit(0);
  }

*/

  /*Algorithme de sélection automatique de points*/
  {
    //paramètres de l'algorithme
    int npts_init = 50;
    int npts_per_iter = 50;
    int nsteps_mcmc = 1e5;
    int nsamples_mcmc = 1000;
    int niter_max = 20;
    double time_opti_fine = 1e-2; //avec gradients.
    double time_opti_hgps = 10;
    //Construction d'un DoE initial et calcul des hpars optimaux dessus.
    DoE doe_init(lb_t, ub_t, npts_init, 10);
    DensityOpt Doalpha(Dalpha);
    DensityOpt Dodiam(Ddiam);
    Doalpha.SetKernelGrads(gradKernel_Z_Matern52);
    Dodiam.SetKernelGrads(gradKernel_Z_Matern52);
    Doalpha.SetLogpriorGrads(gradlogprior_hpars);
    Dodiam.SetLogpriorGrads(gradlogprior_hpars);
    vector<VectorXd> thetas_training;
    vector<VectorXd> halpha_training;
    vector<VectorXd> hdiam_training;
    auto tinit = doe_init.GetGrid();
    for (const auto theta : doe_init.GetGrid())
    {
      VectorXd halpha = Doalpha.HparsOpt_withgrad(theta, hpars_z_guess_alpha, time_opti_fine);
      VectorXd hdiam = Dodiam.HparsOpt_withgrad(theta, hpars_z_guess_diam, time_opti_fine);
      thetas_training.push_back(theta);
      halpha_training.push_back(halpha);
      hdiam_training.push_back(hdiam);
    }

    //read the true FMP sample
    vector<VectorXd> thetas_true;
    vector<VectorXd> halpha_true;
    vector<VectorXd> hdiam_true;
    string gname = "results/fmp/samp.gnu";
    vector<VectorXd> vals = ReadVector(gname);
    for (auto const &v : vals)
    {
      auto theta = v.head(5);
      auto hdiam = v.tail(3);
      VectorXd halpha(3);
      halpha << v(5), v(6), v(7);
      thetas_true.push_back(theta);
      hdiam_true.push_back(hdiam);
      halpha_true.push_back(halpha);
    }
    //compute scores for the true FMP sample
    vector<double> scores_true_alpha;
    vector<double> scores_true_diam;
    for (int i = 0; i < thetas_true.size(); i++)
    {
      scores_true_alpha.push_back(Doalpha.loglikelihood_theta(thetas_true[i], halpha_true[i]) + logprior_hpars(halpha_true[i]));
      scores_true_diam.push_back(Dodiam.loglikelihood_theta(thetas_true[i], hdiam_true[i]) + logprior_hpars(hdiam_true[i]));
    }

    //lambda fcts pour l'évaluation de la qualité des surrogates.
    auto get_score_alpha = [&Doalpha](VectorXd const &theta, VectorXd const &hpars)
    {
      return Doalpha.loglikelihood_theta(theta, hpars) + logprior_hpars(hpars);
    };
    auto get_score_diam = [&Dodiam](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dodiam.loglikelihood_theta(theta, hpars) + logprior_hpars(hpars);
    };

    auto get_hpars_and_var_alpha = [&Doalpha](VectorXd const &X)
    {
      VectorXd h = Doalpha.EvaluateHparOpt(X);
      VectorXd v = Doalpha.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };
    auto get_hpars_and_var_diam = [&Dodiam](VectorXd const &X)
    {
      VectorXd h = Dodiam.EvaluateHparOpt(X);
      VectorXd v = Dodiam.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };

    //lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
    auto add_points = [&Doalpha, &Dodiam, npts_per_iter, nsteps_mcmc, nsamples_mcmc, &X_init_mcmc, &COV_init, &generator, hpars_z_guess_diam, hpars_z_guess_alpha, &thetas_training, &halpha_training, &hdiam_training, time_opti_fine]()
    {
      auto get_hpars_opti = [&Doalpha, &Dodiam, &hpars_z_guess_alpha, &hpars_z_guess_diam](VectorXd const &X)
      {
        vector<VectorXd> p(2);
        p[0] = Doalpha.EvaluateHparOpt(X);
        p[1] = Dodiam.EvaluateHparOpt(X);
        return p;
      };
      auto compute_score_opti = [&Doalpha, &Dodiam](vector<VectorXd> h, VectorXd const &X)
      {
        double ll1 = Doalpha.loglikelihood_theta(X, h[0]);
        double ll2 = Dodiam.loglikelihood_theta(X, h[1]);
        double lp = logprior_pars(X);
        return ll1 + ll2 + lp;
      };
      auto in_bounds = [&Doalpha](VectorXd const &X)
      {
        return Doalpha.in_bounds_pars(X);
      };

      vector<VectorXd> allsteps = Run_MCMC(nsteps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
      vector<VectorXd> candidate_thetas(nsamples_mcmc);
      for (int i = 0; i < candidate_thetas.size(); i++)
      {
        candidate_thetas[i] = allsteps[i * (allsteps.size() / candidate_thetas.size())];
      }
      vector<VectorXd> selected_thetas(npts_per_iter);
      vector<double> weights(nsamples_mcmc);
      for (int i = 0; i < weights.size(); i++)
      {
        weights[i] = Doalpha.EstimatePredError(candidate_thetas[i]) + Dodiam.EstimatePredError(candidate_thetas[i]);
      }
      //tirage sans remise pondéré par les poids.
      for (int i = 0; i < npts_per_iter; i++)
      {
        std::discrete_distribution<int> distribution(weights.begin(), weights.end());
        int drawn = distribution(generator);
        weights[drawn] = 0;
        selected_thetas[i] = candidate_thetas[drawn];
      }
      //on réalise les optimisations aux points sélectionnés. et on les rajoute au total de points.
      for (const auto theta : selected_thetas)
      {
        VectorXd halpha = Doalpha.HparsOpt_withgrad(theta, hpars_z_guess_alpha, time_opti_fine);
        VectorXd hdiam = Dodiam.HparsOpt_withgrad(theta, hpars_z_guess_diam, time_opti_fine);
        thetas_training.push_back(theta);
        halpha_training.push_back(halpha);
        hdiam_training.push_back(hdiam);
      }
    };

    auto write_performance_hgps = [&thetas_true, &halpha_true, &hdiam_true, &get_hpars_and_var_alpha, &get_hpars_and_var_diam, &thetas_training, &Doalpha, &Dodiam, &halpha_training, &hdiam_training, &get_score_alpha, &get_score_diam, &scores_true_alpha, &scores_true_diam](ofstream &ofile_alpha, ofstream &ofile_diam)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd scorealpha = evaluate_surrogate_bofplusscores(thetas_true, halpha_true, scores_true_alpha, get_hpars_and_var_alpha, get_score_alpha);
      VectorXd scorediam = evaluate_surrogate_bofplusscores(thetas_true, hdiam_true, scores_true_diam, get_hpars_and_var_diam, get_score_diam);
      ofile_alpha << thetas_training.size() << " ";
      ofile_diam << thetas_training.size() << " ";
      for (int i = 0; i < scorealpha.size(); i++)
      {
        ofile_alpha << scorealpha(i) << " ";
      }
      ofile_alpha << endl;
      for (int i = 0; i < scorediam.size(); i++)
      {
        ofile_diam << scorediam(i) << " ";
      }
      ofile_diam << endl;
    };

    //début de l'algorithme
    ofstream ofstream_alpha("results/score_alpha_is.gnu");
    ofstream ofstream_diam("results/score_diam_is.gnu");
    vector<VectorXd> hpars_opt_hgps_alpha;
    vector<VectorXd> hpars_opt_hgps_diam;
    for (int i = 0; i < hpars_z_guess_alpha.size(); i++)
    {
      hpars_opt_hgps_alpha.push_back(hpars_hgp_guess);
    }
    for (int i = 0; i < hpars_z_guess_diam.size(); i++)
    {
      hpars_opt_hgps_diam.push_back(hpars_hgp_guess);
    }
    while (thetas_training.size() <= 2000)
    {
      //construction hGPs
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      Doalpha.BuildHGPs(thetas_training, halpha_training, Kernel_GP_Matern32);
      Dodiam.BuildHGPs(thetas_training, hdiam_training, Kernel_GP_Matern32);
      Doalpha.OptimizeHGPs(Bounds_hpars_hgps, hpars_opt_hgps_alpha, time_opti_hgps);
      Dodiam.OptimizeHGPs(Bounds_hpars_hgps, hpars_opt_hgps_diam, time_opti_hgps);
      hpars_opt_hgps_alpha = Doalpha.GetHparsHGPs();
      hpars_opt_hgps_diam = Dodiam.GetHparsHGPs();
      cout << "hpars de hgps : " << endl;
      for (int i = 0; i < hpars_opt_hgps_alpha.size(); i++)
      {
        cout << hpars_opt_hgps_alpha[i].transpose() << endl;
        cout << hpars_opt_hgps_diam[i].transpose() << endl;
      }

      //évaluation de leur score
      write_performance_hgps(ofstream_alpha, ofstream_diam);

      //ajout de nouveaux points
      add_points();

      //remove les duplicates de thetas. ne marche pas en cas de triplés. En cas de triplés : un autre point sera supprimé sans le mértier. pas trop grave.
      vector<VectorXd> dup = thetas_training;
      vector<int> dups;
      for (int i = 0; i < thetas_training.size(); i++)
      {
        for (int j = i + 1; j < thetas_training.size(); j++)
        {
          if (thetas_training[i] == thetas_training[j])
          {
            dups.push_back(j);
          }
        }
      }
      sort(dups.begin(), dups.end());
      for (int i = dups.size() - 1; i >= 0; i--)
      {
        cout << "duplicate found !" << endl;
        cout << thetas_training[dups[i]].transpose() << endl;
        thetas_training.erase(thetas_training.begin() + dups[i]);
        halpha_training.erase(halpha_training.begin() + dups[i]);
        hdiam_training.erase(hdiam_training.begin() + dups[i]);
      }

      //écriture des points d'apprentissage dans un fichier
      string o1("results/trainingalpha.gnu");
      string o2("results/trainingdiam.gnu");
      WriteVectors(thetas_training, halpha_training, o1);
      WriteVectors(thetas_training, hdiam_training, o2);
    }
    ofstream_alpha.close();
    ofstream_diam.close();
  }

  exit(0);

  //Calibration phase

  /*Test : je rajoute progressivment des points d'un DOE lhs.*/
  {
    double time_opti_hgps = 10;
    double time_opti_fine = 1e-2; //avec gradients.
    int npts_per_iter = 50;
    //Construction d'un DoE initial et calcul des hpars optimaux dessus.
    DoE doe_qmc(lb_t, ub_t, 2000, 10);
    vector<VectorXd> thetas_qmc = doe_qmc.GetGrid();
    DensityOpt Doalpha(Dalpha);
    DensityOpt Dodiam(Ddiam);
    Doalpha.SetKernelGrads(gradKernel_Z_Matern52);
    Dodiam.SetKernelGrads(gradKernel_Z_Matern52);
    Doalpha.SetLogpriorGrads(gradlogprior_hpars);
    Dodiam.SetLogpriorGrads(gradlogprior_hpars);
    vector<VectorXd> thetas_training;
    vector<VectorXd> halpha_training;
    vector<VectorXd> hdiam_training;
    //read the true FMP sample
    vector<VectorXd> thetas_true;
    vector<VectorXd> halpha_true;
    vector<VectorXd> hdiam_true;
    string gname = "results/fmp/samp.gnu";
    vector<VectorXd> vals = ReadVector(gname);
    for (auto const &v : vals)
    {
      auto theta = v.head(5);
      auto hdiam = v.tail(3);
      VectorXd halpha(3);
      halpha << v(5), v(6), v(7);
      thetas_true.push_back(theta);
      hdiam_true.push_back(hdiam);
      halpha_true.push_back(halpha);
    }
    //compute scores for the true FMP sample
    vector<double> scores_true_alpha;
    vector<double> scores_true_diam;
    for (int i = 0; i < thetas_true.size(); i++)
    {
      scores_true_alpha.push_back(Doalpha.loglikelihood_theta(thetas_true[i], halpha_true[i]) + logprior_hpars(halpha_true[i]));
      scores_true_diam.push_back(Dodiam.loglikelihood_theta(thetas_true[i], hdiam_true[i]) + logprior_hpars(hdiam_true[i]));
    }

    //lambda fcts pour l'évaluation de la qualité des surrogates.
    auto get_score_alpha = [&Doalpha](VectorXd const &theta, VectorXd const &hpars)
    {
      return Doalpha.loglikelihood_theta(theta, hpars) + logprior_hpars(hpars);
    };
    auto get_score_diam = [&Dodiam](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dodiam.loglikelihood_theta(theta, hpars) + logprior_hpars(hpars);
    };
    auto get_hpars_and_var_alpha = [&Doalpha](VectorXd const &X)
    {
      VectorXd h = Doalpha.EvaluateHparOpt(X);
      VectorXd v = Doalpha.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };
    auto get_hpars_and_var_diam = [&Dodiam](VectorXd const &X)
    {
      VectorXd h = Dodiam.EvaluateHparOpt(X);
      VectorXd v = Dodiam.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };

    //lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
    auto add_points = [&Doalpha, &Dodiam, npts_per_iter, &thetas_qmc, &thetas_training, &halpha_training, &hdiam_training, time_opti_fine, hpars_z_guess_alpha, hpars_z_guess_diam]()
    {
      //juste rajouter npts_per_iter points depuis le thetas_true.
      int npts = thetas_training.size();
      for (int i = 0; i < npts_per_iter; i++)
      {
        if (npts + i >= thetas_qmc.size())
        {
          break;
        }
        VectorXd theta = thetas_qmc[npts + i];
        thetas_training.push_back(theta);
        VectorXd ha = Doalpha.HparsOpt_withgrad(theta, hpars_z_guess_alpha, time_opti_fine);
        VectorXd hd = Dodiam.HparsOpt_withgrad(theta, hpars_z_guess_diam, time_opti_fine);
        halpha_training.push_back(ha);
        hdiam_training.push_back(hd);
      }
    };

    auto write_performance_hgps = [&thetas_true, &halpha_true, &hdiam_true, &get_hpars_and_var_alpha, &get_hpars_and_var_diam, &thetas_training, &Doalpha, &Dodiam, &halpha_training, &hdiam_training, &get_score_alpha, &get_score_diam, &scores_true_alpha, &scores_true_diam](ofstream &ofile_alpha, ofstream &ofile_diam)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd scorealpha = evaluate_surrogate_bofplusscores(thetas_true, halpha_true, scores_true_alpha, get_hpars_and_var_alpha, get_score_alpha);
      VectorXd scorediam = evaluate_surrogate_bofplusscores(thetas_true, hdiam_true, scores_true_diam, get_hpars_and_var_diam, get_score_diam);
      ofile_alpha << thetas_training.size() << " ";
      ofile_diam << thetas_training.size() << " ";
      for (int i = 0; i < scorealpha.size(); i++)
      {
        ofile_alpha << scorealpha(i) << " ";
      }
      ofile_alpha << endl;
      for (int i = 0; i < scorediam.size(); i++)
      {
        ofile_diam << scorediam(i) << " ";
      }
      ofile_diam << endl;
    };

    //début de l'algorithme
    ofstream ofstream_alpha("results/score_alpha_qmc.gnu");
    ofstream ofstream_diam("results/score_diam_qmc.gnu");
    vector<VectorXd> hpars_opt_hgps_alpha;
    vector<VectorXd> hpars_opt_hgps_diam;
    for (int i = 0; i < hpars_z_guess_alpha.size(); i++)
    {
      hpars_opt_hgps_alpha.push_back(hpars_hgp_guess);
    }
    for (int i = 0; i < hpars_z_guess_diam.size(); i++)
    {
      hpars_opt_hgps_diam.push_back(hpars_hgp_guess);
    }
    for (int i = 0; i < 40; i++)
    {
      if (thetas_training.size() == thetas_qmc.size())
      {
        break;
      }
      //ajout de nouveaux points
      add_points();

      vector<VectorXd> dup = thetas_training;
      vector<int> dups;
      for (int i = 0; i < thetas_training.size(); i++)
      {
        for (int j = i + 1; j < thetas_training.size(); j++)
        {
          if (thetas_training[i] == thetas_training[j])
          {
            dups.push_back(j);
          }
        }
      }
      sort(dups.begin(), dups.end());
      for (int i = dups.size() - 1; i >= 0; i--)
      {
        cout << "duplicate found !" << endl;
        thetas_training.erase(thetas_training.begin() + dups[i]);
        halpha_training.erase(halpha_training.begin() + dups[i]);
        hdiam_training.erase(hdiam_training.begin() + dups[i]);
      }

      //construction hGPs
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      Doalpha.BuildHGPs(thetas_training, halpha_training, Kernel_GP_Matern32);
      Dodiam.BuildHGPs(thetas_training, hdiam_training, Kernel_GP_Matern32);
      Doalpha.OptimizeHGPs(Bounds_hpars_hgps, hpars_opt_hgps_alpha, time_opti_hgps);
      Dodiam.OptimizeHGPs(Bounds_hpars_hgps, hpars_opt_hgps_diam, time_opti_hgps);
      hpars_opt_hgps_alpha = Doalpha.GetHparsHGPs();
      hpars_opt_hgps_diam = Dodiam.GetHparsHGPs();
      cout << "hpars de hgps : " << endl;
      for (int i = 0; i < hpars_opt_hgps_alpha.size(); i++)
      {
        cout << hpars_opt_hgps_alpha[i].transpose() << endl;
        cout << hpars_opt_hgps_diam[i].transpose() << endl;
      }

      //évaluation de leur score
      write_performance_hgps(ofstream_alpha, ofstream_diam);

      //écriture des points d'apprentissage dans un fichier
      string o1("results/trainingalpha.gnu");
      string o2("results/trainingdiam.gnu");
      WriteVectors(thetas_training, halpha_training, o1);
      WriteVectors(thetas_training, hdiam_training, o2);
    }
  }
  exit(0);

  /*Test : je rajoute progressivment des points du true sample dans le training set.*/
  {
    double time_opti_hgps = 2;
    int npts_per_iter = 50;
    //Construction d'un DoE initial et calcul des hpars optimaux dessus.
    DensityOpt Doalpha(Dalpha);
    DensityOpt Dodiam(Ddiam);
    Doalpha.SetKernelGrads(gradKernel_Z_Matern52);
    Dodiam.SetKernelGrads(gradKernel_Z_Matern52);
    Doalpha.SetLogpriorGrads(gradlogprior_hpars);
    Dodiam.SetLogpriorGrads(gradlogprior_hpars);
    vector<VectorXd> thetas_training;
    vector<VectorXd> halpha_training;
    vector<VectorXd> hdiam_training;
    //read the true FMP sample
    vector<VectorXd> thetas_true;
    vector<VectorXd> halpha_true;
    vector<VectorXd> hdiam_true;
    string gname = "results/fmp/samp.gnu";
    vector<VectorXd> vals = ReadVector(gname);
    for (auto const &v : vals)
    {
      auto theta = v.head(5);
      auto hdiam = v.tail(3);
      VectorXd halpha(3);
      halpha << v(5), v(6), v(7);
      thetas_true.push_back(theta);
      hdiam_true.push_back(hdiam);
      halpha_true.push_back(halpha);
    }
    //compute scores for the true FMP sample
    vector<double> scores_true_alpha;
    vector<double> scores_true_diam;
    for (int i = 0; i < thetas_true.size(); i++)
    {
      scores_true_alpha.push_back(Doalpha.loglikelihood_theta(thetas_true[i], halpha_true[i]) + logprior_hpars(halpha_true[i]));
      scores_true_diam.push_back(Dodiam.loglikelihood_theta(thetas_true[i], hdiam_true[i]) + logprior_hpars(hdiam_true[i]));
    }

    //lambda fcts pour l'évaluation de la qualité des surrogates.
    auto get_score_alpha = [&Doalpha](VectorXd const &theta, VectorXd const &hpars)
    {
      return Doalpha.loglikelihood_theta(theta, hpars) + logprior_hpars(hpars);
    };
    auto get_score_diam = [&Dodiam](VectorXd const &theta, VectorXd const &hpars)
    {
      return Dodiam.loglikelihood_theta(theta, hpars) + logprior_hpars(hpars);
    };
    auto get_hpars_and_var_alpha = [&Doalpha](VectorXd const &X)
    {
      VectorXd h = Doalpha.EvaluateHparOpt(X);
      VectorXd v = Doalpha.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };
    auto get_hpars_and_var_diam = [&Dodiam](VectorXd const &X)
    {
      VectorXd h = Dodiam.EvaluateHparOpt(X);
      VectorXd v = Dodiam.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };

    //lambda fonction pour rajouter un certain nombre de points et calculer les hpars optimaux.
    auto add_points = [&Doalpha, &Dodiam, npts_per_iter, &thetas_true, &halpha_true, &hdiam_true, &thetas_training, &halpha_training, &hdiam_training]()
    {
      //juste rajouter npts_per_iter points depuis le thetas_true.
      int npts = thetas_training.size();
      for (int i = 0; i < npts_per_iter; i++)
      {
        if (npts + i >= thetas_true.size())
        {
          break;
        }
        thetas_training.push_back(thetas_true[npts + i]);
        halpha_training.push_back(halpha_true[npts + i]);
        hdiam_training.push_back(hdiam_true[npts + i]);
      }
    };

    auto write_performance_hgps = [&thetas_true, &halpha_true, &hdiam_true, &get_hpars_and_var_alpha, &get_hpars_and_var_diam, &thetas_training, &Doalpha, &Dodiam, &halpha_training, &hdiam_training, &get_score_alpha, &get_score_diam, &scores_true_alpha, &scores_true_diam](ofstream &ofile_alpha, ofstream &ofile_diam)
    {
      //évaluer les scores des surrogates et mettre leur performance dans un fichier, ainsi que le nb de pts de construction.
      VectorXd scorealpha = evaluate_surrogate_bofplusscores(thetas_true, halpha_true, scores_true_alpha, get_hpars_and_var_alpha, get_score_alpha);
      VectorXd scorediam = evaluate_surrogate_bofplusscores(thetas_true, hdiam_true, scores_true_diam, get_hpars_and_var_diam, get_score_diam);
      ofile_alpha << thetas_training.size() << " ";
      ofile_diam << thetas_training.size() << " ";
      for (int i = 0; i < scorealpha.size(); i++)
      {
        ofile_alpha << scorealpha(i) << " ";
      }
      ofile_alpha << endl;
      for (int i = 0; i < scorediam.size(); i++)
      {
        ofile_diam << scorediam(i) << " ";
      }
      ofile_diam << endl;
    };

    //début de l'algorithme
    ofstream ofstream_alpha("results/bestscore_alpha.gnu");
    ofstream ofstream_diam("results/bestscore_diam.gnu");
    vector<VectorXd> hpars_opt_hgps_alpha;
    vector<VectorXd> hpars_opt_hgps_diam;
    for (int i = 0; i < hpars_z_guess_alpha.size(); i++)
    {
      hpars_opt_hgps_alpha.push_back(hpars_hgp_guess);
    }
    for (int i = 0; i < hpars_z_guess_diam.size(); i++)
    {
      hpars_opt_hgps_diam.push_back(hpars_hgp_guess);
    }
    for (int i = 0; i < 20; i++)
    {

      if (thetas_training.size() == thetas_true.size())
      {
        break;
      }
      //ajout de nouveaux points
      add_points();

      vector<VectorXd> dup = thetas_training;
      vector<int> dups;
      for (int i = 0; i < thetas_training.size(); i++)
      {
        for (int j = i + 1; j < thetas_training.size(); j++)
        {
          if (thetas_training[i] == thetas_training[j])
          {
            dups.push_back(j);
          }
        }
      }
      sort(dups.begin(), dups.end());
      for (int i = dups.size() - 1; i >= 0; i--)
      {
        cout << "duplicate found !" << endl;
        thetas_training.erase(thetas_training.begin() + dups[i]);
        halpha_training.erase(halpha_training.begin() + dups[i]);
        hdiam_training.erase(hdiam_training.begin() + dups[i]);
      }

      //construction hGPs
      cout << "construction hGPs avec " << thetas_training.size() << " points..." << endl;
      Doalpha.BuildHGPs(thetas_training, halpha_training, Kernel_GP_Matern32);
      Dodiam.BuildHGPs(thetas_training, hdiam_training, Kernel_GP_Matern32);
      Doalpha.OptimizeHGPs(Bounds_hpars_hgps, hpars_opt_hgps_alpha, time_opti_hgps);
      Dodiam.OptimizeHGPs(Bounds_hpars_hgps, hpars_opt_hgps_diam, time_opti_hgps);
      hpars_opt_hgps_alpha = Doalpha.GetHparsHGPs();
      hpars_opt_hgps_diam = Dodiam.GetHparsHGPs();
      cout << "hpars de hgps : " << endl;
      for (int i = 0; i < hpars_opt_hgps_alpha.size(); i++)
      {
        cout << hpars_opt_hgps_alpha[i].transpose() << endl;
        cout << hpars_opt_hgps_diam[i].transpose() << endl;
      }

      //évaluation de leur score
      write_performance_hgps(ofstream_alpha, ofstream_diam);

      //écriture des points d'apprentissage dans un fichier
      string o1("results/trainingalpha.gnu");
      string o2("results/trainingdiam.gnu");
      WriteVectors(thetas_training, halpha_training, o1);
      WriteVectors(thetas_training, hdiam_training, o2);
    }
  }
  exit(0);

  /*Build surrogates on hGPs, and test their precision. IN LOG SCALE*/
  {
    //read the true FMP sample
    vector<VectorXd> thetas_true;
    vector<VectorXd> halpha_true;
    vector<VectorXd> hdiam_true;
    string gname = "results/fmp/samp.gnu";
    vector<VectorXd> vals = ReadVector(gname);
    for (auto const &v : vals)
    {
      thetas_true.push_back(v.head(5));
      hdiam_true.push_back(v.tail(3));
      VectorXd x(3);
      x << v(5), v(6), v(7);
      halpha_true.push_back(x);
    }

    //build a prior reference sample.
    int size_surro = 500;
    DoE doe_sur(lb_t, ub_t, size_surro, 10); //doe halton de 100 points
    DoE doe_val(lb_t, ub_t, 500, 98968);     //doe halton de 100 points
    DensityOpt Doalpha(Dalpha);
    DensityOpt Dodiam(Ddiam);
    Doalpha.SetKernelGrads(gradKernel_Z_Matern52);
    Dodiam.SetKernelGrads(gradKernel_Z_Matern52);
    Doalpha.SetLogpriorGrads(gradlogprior_hpars);
    Dodiam.SetLogpriorGrads(gradlogprior_hpars);

    auto get_true_hpars_opti = [&Doalpha, &Dodiam, &hpars_z_guess_alpha, &hpars_z_guess_diam](VectorXd const &X)
    {
      vector<VectorXd> p(2);
      p[0] = Doalpha.HparsOpt_withgrad(X, hpars_z_guess_alpha, 1e-3);
      p[1] = Dodiam.HparsOpt_withgrad(X, hpars_z_guess_diam, 1e-3);
      return p;
    };

    //calcule les hpars optimaux sur doe_sur, puis construit les hGPs pour aller avec.
    vector<VectorXd> thetas_sur = doe_sur.GetGrid();
    vector<VectorXd> thetas_val = doe_val.GetGrid();
    vector<VectorXd> hpars_sur_alpha;
    vector<VectorXd> hpars_sur_diam;
    vector<VectorXd> hpars_val_alpha;
    vector<VectorXd> hpars_val_diam;
    for (VectorXd t : thetas_sur)
    {
      auto p = get_true_hpars_opti(t);
      hpars_sur_alpha.push_back(p[0]);
      hpars_sur_diam.push_back(p[1]);
    }
    for (VectorXd t : thetas_val)
    {
      auto p = get_true_hpars_opti(t);
      hpars_val_alpha.push_back(p[0]);
      hpars_val_diam.push_back(p[1]);
    }
    Doalpha.BuildHGPs(thetas_sur, hpars_sur_alpha, Kernel_GP_Matern32);
    Dodiam.BuildHGPs(thetas_sur, hpars_sur_diam, Kernel_GP_Matern32);

    Doalpha.OptimizeHGPs(Bounds_hpars_hgps, hpars_hgp_guess, 20);
    Dodiam.OptimizeHGPs(Bounds_hpars_hgps, hpars_hgp_guess, 20);

    auto get_hpars_and_var_alpha = [&Doalpha](VectorXd const &X)
    {
      VectorXd h = Doalpha.EvaluateHparOpt(X);
      VectorXd v = Doalpha.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };
    auto get_hpars_and_var_diam = [&Dodiam](VectorXd const &X)
    {
      VectorXd h = Dodiam.EvaluateHparOpt(X);
      VectorXd v = Dodiam.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };
    cout << "erreur moyenne a priori pour surrogate de taille " << size_surro << endl;
    auto test_alpha = evaluate_surrogate(thetas_val, hpars_val_alpha, get_hpars_and_var_alpha);
    auto test_diam = evaluate_surrogate(thetas_val, hpars_val_diam, get_hpars_and_var_diam);
    cout << test_alpha.transpose() << endl;
    cout << test_diam.transpose() << endl;

    cout << "erreur moyenne a posteriori pour surrogate de taille " << size_surro << endl;
    test_alpha = evaluate_surrogate(thetas_true, halpha_true, get_hpars_and_var_alpha);
    test_diam = evaluate_surrogate(thetas_true, hdiam_true, get_hpars_and_var_diam);
    cout << test_alpha.transpose() << endl;
    cout << test_diam.transpose() << endl;

    //évaluons seulement le surrogate alpha. on affiche les valeurs normales, sa prédiction, et sa variance. OK échelle log.
    string fname = "results/perf.gnu";
    ofstream ofile(fname);
    for (int i = 0; i < thetas_val.size(); i++)
    {
      auto p = get_hpars_and_var_alpha(thetas_val[i]);
      ofile << hpars_val_alpha[i].transpose() << endl;
      ofile << p.first.transpose() << endl;
      VectorXd v = p.second.array().sqrt();
      ofile << v.transpose() << endl
            << endl;
    }
  }

  exit(0);

  /*Build surrogates on hGPs, and test their precision.*/
  {
    //read the true FMP sample
    vector<VectorXd> thetas_true;
    vector<VectorXd> halpha_true;
    vector<VectorXd> hdiam_true;
    string gname = "results/fmp/samp.gnu";
    vector<VectorXd> vals = ReadVector(gname);
    for (auto const &v : vals)
    {
      thetas_true.push_back(v.head(5));
      hdiam_true.push_back(v.tail(3));
      VectorXd x(3);
      x << v(5), v(6), v(7);
      halpha_true.push_back(x);
    }

    //build a prior reference sample.
    int size_surro = 100;
    DoE doe_sur(lb_t, ub_t, size_surro, 10); //doe halton de 100 points
    DoE doe_val(lb_t, ub_t, 500, 98968);     //doe halton de 100 points
    DensityOpt Doalpha(Dalpha);
    DensityOpt Dodiam(Ddiam);
    Doalpha.SetKernelGrads(gradKernel_Z_Matern52);
    Dodiam.SetKernelGrads(gradKernel_Z_Matern52);
    Doalpha.SetLogpriorGrads(gradlogprior_hpars);
    Dodiam.SetLogpriorGrads(gradlogprior_hpars);

    auto get_true_hpars_opti = [&Doalpha, &Dodiam, &hpars_z_guess_alpha, &hpars_z_guess_diam](VectorXd const &X)
    {
      vector<VectorXd> p(2);
      p[0] = Doalpha.HparsOpt_withgrad(X, hpars_z_guess_alpha, 1e-3);
      p[1] = Dodiam.HparsOpt_withgrad(X, hpars_z_guess_diam, 1e-3);
      return p;
    };

    //calcule les hpars optimaux sur doe_sur, puis construit les hGPs pour aller avec.
    vector<VectorXd> thetas_sur = doe_sur.GetGrid();
    vector<VectorXd> thetas_val = doe_val.GetGrid();
    vector<VectorXd> hpars_sur_alpha;
    vector<VectorXd> hpars_sur_diam;
    vector<VectorXd> hpars_val_alpha;
    vector<VectorXd> hpars_val_diam;
    for (VectorXd t : thetas_sur)
    {
      auto p = get_true_hpars_opti(t);
      hpars_sur_alpha.push_back(p[0]);
      hpars_sur_diam.push_back(p[1]);
    }
    for (VectorXd t : thetas_val)
    {
      auto p = get_true_hpars_opti(t);
      hpars_val_alpha.push_back(p[0]);
      hpars_val_diam.push_back(p[1]);
    }
    Doalpha.BuildHGPs(thetas_sur, hpars_sur_alpha, Kernel_GP_Matern32);
    Dodiam.BuildHGPs(thetas_sur, hpars_sur_diam, Kernel_GP_Matern32);
    Doalpha.OptimizeHGPs(Bounds_hpars_hgps, hpars_hgp_guess, 10);
    Dodiam.OptimizeHGPs(Bounds_hpars_hgps, hpars_hgp_guess, 10);

    auto get_hpars_and_var_alpha = [&Doalpha](VectorXd const &X)
    {
      VectorXd h = Doalpha.EvaluateHparOpt(X);
      VectorXd v = Doalpha.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };
    auto get_hpars_and_var_diam = [&Dodiam](VectorXd const &X)
    {
      VectorXd h = Dodiam.EvaluateHparOpt(X);
      VectorXd v = Dodiam.EvaluateVarHparOpt(X);
      return make_pair(h, v);
    };
    cout << "erreur moyenne a priori pour surrogate de taille " << size_surro << endl;
    auto test_alpha = evaluate_surrogate(thetas_val, hpars_val_alpha, get_hpars_and_var_alpha);
    auto test_diam = evaluate_surrogate(thetas_val, hpars_val_diam, get_hpars_and_var_diam);
    cout << test_alpha.transpose() << endl;
    cout << test_diam.transpose() << endl;

    cout << "erreur moyenne a posteriori pour surrogate de taille " << size_surro << endl;
    test_alpha = evaluate_surrogate(thetas_true, halpha_true, get_hpars_and_var_alpha);
    test_diam = evaluate_surrogate(thetas_true, hdiam_true, get_hpars_and_var_diam);
    cout << test_alpha.transpose() << endl;
    cout << test_diam.transpose() << endl;
  }

  exit(0);

  /*expensive FMP calibration*/
  {
    DensityOpt Doalpha(Dalpha);
    DensityOpt Dodiam(Ddiam);
    Doalpha.SetKernelGrads(gradKernel_Z_Matern52);
    Dodiam.SetKernelGrads(gradKernel_Z_Matern52);
    Doalpha.SetLogpriorGrads(gradlogprior_hpars);
    Dodiam.SetLogpriorGrads(gradlogprior_hpars);
    auto in_bounds = [&Doalpha](VectorXd const &X)
    {
      return Doalpha.in_bounds_pars(X);
    };
    auto get_hpars_opti = [&Doalpha, &Dodiam, &hpars_z_guess_alpha, &hpars_z_guess_diam](VectorXd const &X)
    {
      vector<VectorXd> p(2);
      p[0] = Doalpha.HparsOpt_withgrad(X, hpars_z_guess_alpha, 1e-3);
      p[1] = Dodiam.HparsOpt_withgrad(X, hpars_z_guess_diam, 1e-3);
      return p;
    };
    auto compute_score_opti = [&Doalpha, &Dodiam](vector<VectorXd> const &p, VectorXd const &X)
    {
      double d = Doalpha.loglikelihood_theta(X, p[0]);
      double d2 = Dodiam.loglikelihood_theta(X, p[1]);
      return d + d2 + Doalpha.EvaluateLogPPars(X);
    };
    cout << "Beginning FMP calibration..." << endl;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, X_init_mcmc, COV_init, compute_score_opti, get_hpars_opti, in_bounds, generator);
    vector<VectorXd> samples(nombre_samples_collected);
    vector<VectorXd> hparsalphaofsamples(nombre_samples_collected);
    vector<VectorXd> hparsdiamofsamples(nombre_samples_collected);
    for (int i = 0; i < samples.size(); i++)
    {
      samples[i] = visited_steps[i * (visited_steps.size() / samples.size())];
      hparsalphaofsamples[i] = get_hpars_opti(samples[i])[0];
      hparsdiamofsamples[i] = get_hpars_opti(samples[i])[1];
    }
    //diagnostic
    Selfcor_diagnosis(visited_steps, nautocor, 1, "results/fmp/autocor.gnu");

    //write samples
    string fnamesamp = "results/fmp/samp.gnu";
    string fnameallsamp = "results/fmp/allsteps.gnu";
    WriteVectors(samples, hparsalphaofsamples, hparsdiamofsamples, fnamesamp);
    WriteVector(visited_steps, fnameallsamp);

    //predictions
    Doalpha.SetNewSamples(samples);
    Doalpha.SetNewHparsOfSamples(hparsalphaofsamples);
    Dodiam.SetNewSamples(samples);
    Dodiam.SetNewHparsOfSamples(hparsdiamofsamples);
    string fnamepred = "results/fmp/predsalpha.gnu";
    string fnamesF = "results/fmp/sampsFalpha.gnu";
    string fnamesZ = "results/fmp/sampsZalpha.gnu";
    string fnamepred2 = "results/fmp/predsdiam.gnu";
    string fnamesF2 = "results/fmp/sampsFdiam.gnu";
    string fnamesZ2 = "results/fmp/sampsZdiam.gnu";
    Doalpha.WritePredictions(location_points, fnamepred); //on doit utiliser les points d'observation pour prédire.
    Doalpha.WriteSamplesFandZ(location_points, fnamesF, fnamesZ);
    Dodiam.WritePredictions(location_points, fnamepred2); //on doit utiliser les points d'observation pour prédire.
    Dodiam.WriteSamplesFandZ(location_points, fnamesF2, fnamesZ2);
  }

  exit(0);
}
