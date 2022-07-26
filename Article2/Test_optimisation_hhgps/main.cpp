// Test pour tester les algorithmes d'optimisation pour les hhGPs.
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

double myoptfunc_gp_nograd(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* This is the function you optimize for defining the GP. Also gives gradient. */
    auto ptp=(pair<GP*,int*>*) data;
  GP *proc=ptp->first; //Pointer to the GP
  int *count=ptp->second;
  *count+=1;
  Eigen::VectorXd p(x.size()); //Parameters to be optimized
  for (int i = 0; i < (int)x.size(); i++)
    p(i) = x[i];                 //Setting the proposed value of the parameters
  double value = -1*proc->SetGP(p); //Evaluate the function. Attention ! moi je maximise.
  return value;
};

double myoptfunc_gp_grad(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* This is the function you optimize for defining the GP. Also gives gradient. */
  auto ptp=(pair<GP*,int*>*) data;
  GP *proc=ptp->first; //Pointer to the GP
  int *count=ptp->second;
  *count+=1;  
  Eigen::VectorXd p(x.size()); //Parameters to be optimized
  for (int i = 0; i < (int)x.size(); i++)
    p(i) = x[i];                 //Setting the proposed value of the parameters
  double value = -1*proc->SetGP(p); //Evaluate the function. Attention ! moi je maximise.
  Eigen::VectorXd Grad= -1*proc->DerivLL();
  for(int i=0;i<Grad.size();i++){
    grad[i]=Grad(i);
  }
  return value;
};

void optroutine_grad(void *gp_ptr, nlopt::algorithm alg, string alg_string, double xtol_rel, double ftol_rel, double max_time, VectorXd Xinit, VectorXd const &lb_h, VectorXd const &ub_h, ofstream &ofile){
  //routine d'optimisation, avec écriture de tous les résultats dans un fichier.
  vector<double> x = VXDtoV(Xinit);
  vector<double> lb = VXDtoV(lb_h);
  vector<double> ub = VXDtoV(ub_h);
  nlopt::opt local_opt(alg,x.size());
  int count=0;
  auto p=make_pair(gp_ptr,&count);
  local_opt.set_max_objective(myoptfunc_gp_grad, &p);
  local_opt.set_ftol_rel(ftol_rel);
  local_opt.set_xtol_rel(xtol_rel);
  local_opt.set_lower_bounds(lb);
  local_opt.set_upper_bounds(ub);
  local_opt.set_maxtime(max_time);
  double msup;           /* the maximum objective value, upon return */
  //on relance une opti locale à partir du max. trouvé.
  auto begin = chrono::steady_clock::now();
  int message_arret = local_opt.optimize(x, msup); //messages d'arrêt : ftol = 3, xtol =4, time=6.
  auto end = chrono::steady_clock::now();
  double real_time = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
  VectorXd Xmax=VtoVXD(x);
  //calcul du gradient au meilleur point.
  GP *proc = (GP *) gp_ptr;
  double msup2=-1*proc->SetGP(Xmax);
  VectorXd grad=-1*proc->DerivLL();
  //écriture dans le fichier
  ofile << "Calcul avec algorithme : " << alg_string << ". xtol = " << xtol_rel << ", ftol = " << ftol_rel << ", max_time = " << max_time << endl;
  ofile << "Point de départ : " << endl << Xinit.transpose() << endl;
  ofile << "Argument maximum trouvé : " << endl << Xmax.transpose() << endl;
  ofile << "Critère max : " << endl << msup << " " << msup2 << endl;
  ofile << "Gradient au max : " << endl << grad.transpose() << endl;
  ofile << "Message d'arrêt : " << endl << message_arret << endl;
  ofile << "Temps réel d'optimisation : " << endl << real_time << endl;
  ofile << "Nombre d'évaluations de fonction : " << endl << count << endl;
}


void optroutine_nograd(void *gp_ptr, nlopt::algorithm alg, string alg_string, double xtol_rel, double ftol_rel, double max_time, VectorXd Xinit, VectorXd const &lb_h, VectorXd const &ub_h, ofstream &ofile){
  //routine d'optimisation, avec écriture de tous les résultats dans un fichier.
  vector<double> x = VXDtoV(Xinit);
  vector<double> lb = VXDtoV(lb_h);
  vector<double> ub = VXDtoV(ub_h);
  nlopt::opt local_opt(alg,x.size());
    int count=0;
  auto p=make_pair(gp_ptr,&count);
  local_opt.set_max_objective(myoptfunc_gp_nograd, &p);
  local_opt.set_ftol_rel(ftol_rel);
  local_opt.set_xtol_rel(xtol_rel);
  local_opt.set_lower_bounds(lb);
  local_opt.set_upper_bounds(ub);
  local_opt.set_maxtime(max_time);
  double msup;           /* the maximum objective value, upon return */
  //on relance une opti locale à partir du max. trouvé.
  auto begin = chrono::steady_clock::now();
  int message_arret = local_opt.optimize(x, msup); //messages d'arrêt : ftol = 3, xtol =4, time=6.
  auto end = chrono::steady_clock::now();
  double real_time = chrono::duration_cast<chrono::milliseconds>(end - begin).count();
  VectorXd Xmax=VtoVXD(x);
  GP *proc = (GP *) gp_ptr;
  double msup2=-1*proc->SetGP(Xmax);
  //écriture dans le fichier
  ofile << "Calcul avec algorithme : " << alg_string << ". xtol = " << xtol_rel << ", ftol = " << ftol_rel << ", max_time = " << max_time << endl;
  ofile << "Point de départ : " << endl << Xinit.transpose() << endl;
  ofile << "Argument maximum trouvé : " << endl << Xmax.transpose() << endl;
  ofile << "Critère max : " << endl << msup << " " << msup2 << endl;
  ofile << "Message d'arrêt : " << endl << message_arret << endl;
  ofile << "Temps réel d'optimisation : " << endl << real_time << endl;
  ofile << "Nombre d'évaluations de fonction : " << endl << count << endl;
}


int main(int argc, char **argv)
{



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
  default_random_engine generator(123456);
  VectorXd theta_true(dim_theta); // valeur de theta utilisée pour générer les observations
  theta_true << 0.55, 0.75, 0.3, 0.4, 0.6, 0.9;
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

  vector<VectorXd> best_hpars_hgps; // on y met les meilleurs hpars, trouvés à partir d'un calcul fin.
  {
    VectorXd h1(8), h2(8), h3(8), h4(8), h5(8), h6(8), h7(8);
    h1 << 1.53, 0.25, 5e-2, 0.64, 0.25, 0.22, 0.33, 0.30;
    h2 << 1.34, 0.21, 1e-2, 0.30, 0.19, 0.18, 0.29, 0.20;
    h3 << 1.79, 0.17, 1.3e-2, 0.31, 0.28, 0.23, 0.43, 0.39;
    h4 << 1.87, 0.27, 3e-2, 0.81, 0.33, 0.23, 0.49, 0.48;
    h5 << 1.53, 0.21, 2.4e-3, 0.30, 0.19, 0.17, 0.38, 0.27;
    h6 << 1.96, 0.23, 1e-2, 0.47, 0.29, 0.21, 0.35, 0.28;
    h7 << 1.78, 0.23, 1e-2, 0.57, 0.25, 0.21, 0.35, 0.25;
    best_hpars_hgps.push_back(h1);
    best_hpars_hgps.push_back(h2);
    best_hpars_hgps.push_back(h3);
    best_hpars_hgps.push_back(h4);
    best_hpars_hgps.push_back(h5);
    best_hpars_hgps.push_back(h6);
    best_hpars_hgps.push_back(h7);
  }


  // Calibration phase. cool beans
  /*récupération des échantllons post FMP précis et calcul des scores. Nécessaire pour évaluer l'erreur avec la fonction bofplusscores.*/


  double time_opti_fmp=0.05;

  /*récupération de 500 pts thetapost*/
  vector<VectorXd> thetas_post;
  vector<VectorXd> hpars_post;
  vector<double> scores_post;
  {
    string fname_post = "results/fmp/samp.gnu";
    thetas_post = ReadVector(fname_post); // on prend l'ensemble des points du sample.
    int size=thetas_post.size()/2;

    DensityOpt Dopt(MainDensity);
    for (int i = 0; i < thetas_post.size(); i++)
    {
      VectorXd hpars = Dopt.HparsOpt(thetas_post[i], hpars_z_guess, time_opti_fmp);
      hpars_post.push_back(hpars);
      scores_post.push_back(Dopt.loglikelihood_theta(thetas_post[i], hpars));
    }
  }

  //création d'un hGP sur un hpar (déjà en log scale).
  int which_hpar=0;
  GP gp(Kernel_GP_Matern32);
  vector<DATA> data_gp;
  for(int i=0;i<thetas_post.size();i++){
    DATA d; d.SetX(thetas_post[i]); d.SetValue(hpars_post[i](which_hpar));
    data_gp.push_back(d);
  }
  gp.SetData(data_gp);
  gp.SetDKernel(DKernel_GP_Matern32);


  //vérification de gradient SetGP avec comparaison différences finies.
  /*
  {
    VectorXd x=0.5*VectorXd::Ones(6);
    VectorXd y=0.6*VectorXd::Ones(6);
    VectorXd hp_ref=Hpars_guess_HGPs;
    cout << "hpars ref : " << hp_ref << endl;
    VectorXd dx=hp_ref/100;
    for(int i=0;i<8;i++){
      VectorXd hpp=hp_ref;
      VectorXd hpm=hp_ref;
      hpp(i)+=dx(i);
      hpm(i)-=dx(i);
      double dm=gp.SetGP(hpm);
      double dp=gp.SetGP(hpp);
      double df=(dp-dm)/(2*dx(i));
      gp.SetGP(hp_ref);
      VectorXd gradf=gp.DerivLL();
      cout << "hpar : " << i << ", dx : " << dx(i) << endl;
      cout << " df = " << df << ", gradf = " << gradf(i) << endl; 
    }
  }
  */


  //test de la routine. sans gradient déjà.
  string fname="results/res.gnu";
  double ftol=1e-4;
  ofstream ofile(fname);
  optroutine_grad(&gp,nlopt::LD_LBFGS,"LD_LBFGS",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;  
  optroutine_grad(&gp,nlopt::LD_TNEWTON_PRECOND_RESTART,"LD_TNEWTON_PRECOND_RESTART",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;  
  optroutine_grad(&gp,nlopt::LD_TNEWTON_PRECOND,"LD_TNEWTON_PRECOND",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl; 
  optroutine_grad(&gp,nlopt::LD_TNEWTON_RESTART,"LD_TNEWTON_RESTART",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl; 
  optroutine_grad(&gp,nlopt::LD_TNEWTON,"LD_TNEWTON",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl; 
  optroutine_grad(&gp,nlopt::LD_MMA,"LD_MMA",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;
  //optroutine_grad(&gp,nlopt::LD_AUGLAG,"LD_AUGLAG",1e-3,1e-5,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;
  optroutine_nograd(&gp,nlopt::LN_SBPLX,"LN_SBPLX",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;
  optroutine_nograd(&gp,nlopt::LN_NEWUOA,"LN_NEWUOA",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;
  optroutine_nograd(&gp,nlopt::LN_BOBYQA,"LN_BOBYQA",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;
  ofile << "global algorithms : " << endl;
  optroutine_nograd(&gp,nlopt::GN_DIRECT_NOSCAL,"GN_DIRECT_NOSCAL",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;
  //optroutine_grad(&gp,nlopt::GD_STOGO,"GD_STOGO",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;  
  optroutine_nograd(&gp,nlopt::GN_CRS2_LM,"GN_CRS2_LM",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;
  optroutine_nograd(&gp,nlopt::GN_ISRES,"GN_ISRES",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;
  optroutine_nograd(&gp,nlopt::GN_AGS,"GN_AGS",1e-10,ftol,60,Hpars_guess_HGPs,Bounds_hpars_HGPs.row(0),Bounds_hpars_HGPs.row(1),ofile);
  ofile << endl << endl;
  exit(0);
}
