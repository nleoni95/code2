//exemple très simple où l'on cherche sur des problèmes simples à calculer la postérieure p(theta,psi) pour voir si elle est gaussienne.
//exemple de Tuo2015

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

double gaussprob(double x, double mu, double sigma)
{
  //renvoie la probabilité gaussienne
  return 1. / (sqrt(2 * 3.14 * pow(sigma, 2))) * exp(-0.5 * pow((x - mu) / sigma, 2));
}


double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //squared exponential
  double d = abs(x(0) - y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0), 2) * (1 + (d / hpar(2))) * exp(-d / hpar(2)); //3/2
}

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar)
{
  //kernel squared exponential. hpars : sigma2/l, sigma.
  double s=pow(10,hpar(0));
  double l=hpar(1);
  double d = abs(x(0) - y(0));
  return pow(s, 2) * exp(-0.5 * pow(d /l, 2)); //3/2
}

tuple<VectorXd, MatrixXd> GaussFit(vector<VectorXd> const &samples)
{
  //fit d'une gaussienne multivariée sur un sample.
  int d = samples[0].size();
  VectorXd mean = VectorXd::Zero(d);
  MatrixXd SecondMoment = MatrixXd::Zero(d, d);
  for_each(samples.begin(), samples.end(), [&SecondMoment, &mean](VectorXd const &x) mutable
           {
             mean += x;
             SecondMoment += x * x.transpose();
           });
  mean /= samples.size();
  MatrixXd Var = SecondMoment / samples.size() - mean * mean.transpose();
  auto tp = make_tuple(mean, Var);
  return tp;
}

tuple<double, double> GaussFit(vector<double> const &samples)
{
  //fit d'une gaussienne multivariée sur un sample.
  double mean = 0;
  double SecondMoment = 0;
  for (int i = 0; i < samples.size(); i++)
  {
    double x = samples[i];
    mean += x;
    SecondMoment += pow(x, 2);
  }
  mean /= samples.size();
  double Var = SecondMoment / samples.size() - pow(mean, 2);
  auto tp = make_tuple(mean, Var);
  return tp;
}

MatrixXd QQplot(vector<VectorXd> const &samples, default_random_engine &generator)
{
  //calcul d'un QQ plot. On rend une matrice avec samples(0).size() colonnes et autant de lignes que du nombre de quantiles choisi.
  //si j'ai bien compris, la première colonne est les quantiles de la loi normale. chaque colonne ensuite correspond aux quantiles du premier paramètre, du premier hpar, etc.
  //on met le tout dans un vector car je ne sais faire les QQplot qu'une dimension à la fois.
  int nquantiles = 50; //on choisit de calculer 50 quantiles
  normal_distribution<double> distN(0, 1);
  int ndim = samples[0].size();
  MatrixXd res(nquantiles, ndim + 1);
  //tirage d'un échantillon de loi normale 1D
  vector<double> sample_normal(samples.size());
  transform(sample_normal.begin(), sample_normal.end(), sample_normal.begin(), [&generator, &distN](double d)
            { return distN(generator); });
  sort(sample_normal.begin(), sample_normal.end());
  VectorXd quant_normal(nquantiles);
  for (int i = 0; i < nquantiles; i++)
  {
    double q = (i + 0.5) / (1.0 * nquantiles); // on ne prend ni le quantile 0 ni le quantile 100
    int n = q * sample_normal.size();
    quant_normal(i) = sample_normal[n];
  }
  res.col(0) = quant_normal;
  for (int j = 0; j < ndim; j++)
  {
    //création du sample réduit
    vector<double> sample_1D(samples.size());
    for (int i = 0; i < samples.size(); i++)
    {
      sample_1D[i] = samples[i](j);
    }
    //on centre, on réduit et on trie
    auto tpg = GaussFit(sample_1D);
    double m = get<0>(tpg);
    double s = sqrt(get<1>(tpg));
    transform(sample_1D.begin(), sample_1D.end(), sample_1D.begin(), [m, s](double x)
              {
                double r = (x - m) / s;
                return r;
              });
    sort(sample_1D.begin(), sample_1D.end());
    VectorXd quant_1D(nquantiles);
    for (int i = 0; i < nquantiles; i++)
    {
      double q = (i + 0.5) / (1.0 * nquantiles); // on ne prend ni le quantile 0 ni le quantile 100
      int n = q * sample_1D.size();
      quant_1D(i) = sample_1D[n];
    }
    //on met les deux vecteurs de quantiles dans une même matrice. quantiles théoriques d'abord.
    MatrixXd M(nquantiles, 2);
    M.col(0) = quant_normal;
    M.col(1) = quant_1D;
    res.col(j + 1) = quant_1D;
  }
  return res;
}

void WriteObs(vector<VectorXd> const &X, VectorXd const &obs, string filename)
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
const double Big = -1.e16;

int main(int argc, char **argv)
{

  double noise = 0.1; //sigma noise
  int nombre_steps_mcmc = 3e4;
  int nombre_samples_collected = 200; //on garde la moitié des samples de la mcmc
  int dim_theta = 1;
  int dim_hpars = 2;

  // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(dim_theta);
  lb_t(0) = -2; //0.2 avant
  VectorXd ub_t(dim_theta);
  ub_t(0) = 2;

  VectorXd lb_hpars(dim_hpars);
  lb_hpars(0) = -20;
  lb_hpars(1) = 0;
  VectorXd ub_hpars(dim_hpars);
  ub_hpars(0) = 20;
  ub_hpars(1) = 20;

  VectorXd hpars_z_guess=0.5*(lb_hpars+ub_hpars);

  VectorXd X_init(dim_theta + dim_hpars);
  X_init(0) = -1;
  X_init(1)=-1;
  X_init(2)=0;
  X_init.tail(dim_hpars) = 0.5 * (lb_hpars + ub_hpars);
  MatrixXd COV_init = MatrixXd::Identity(3,3);
  COV_init(0, 0) = pow(0.2, 2); //pour KOH separate : 1e-2 partout fonctionne bien.
  COV_init(1, 1) = pow(0.4, 2);
  COV_init(2, 2) = pow(0.4, 2);
  //fonctions pour obtenir des observations.

  auto true_fct_scalar = [](double x)
  {
    double x0 = x * 2 * M_PI;
    //true underlying process
    return sin(x0) * exp(x0 / 10);
  };

  auto lambda_model = [](vector<VectorXd> const &Xprofile, VectorXd const &theta)
  {
    //le vecteur Xprofile contient tous les x scalaires. Il faut renvoyer une prédiction de même taille que Xprofile.
    VectorXd res(Xprofile.size());
    for (int i = 0; i < res.size(); i++)
    {
      double x = Xprofile[i](0) * 2 * M_PI;
      res(i) = sin(x) * exp(x / 10) - abs(theta(0) + 1) * (sin(theta(0) * x) + cos(theta(0) * x));
    }
    return res;
  };

  //grid initial dans [0,1]
  int nobs = 20;
  VectorXd initgrid(nobs);
  for (int i = 0; i < nobs; i++)
  {
    double x = i * 1.0 / nobs;
    initgrid(i) = x;
  }
  //on tire les observations sur le grid increasing max, déjà.
  int seed = 42; //seed pour les observations
  default_random_engine generator(seed);

  auto run_analysis = [noise, &generator, nombre_steps_mcmc, nombre_samples_collected, lb_t, ub_t, lb_hpars, ub_hpars, dim_theta, dim_hpars, lambda_model, X_init, COV_init,hpars_z_guess](string foldname, VectorXd const &Xobs, VectorXd const &Yobs)
  {
    //fonction qui trouve un échantillon de la postérieure à partir du grid, des observations, et écrit tout dans un dossier.
    //compute moyenne theta, std, moyenne sigma, std, et renvoie tout ça.
    int nobs = Xobs.size();
    vector<VectorXd> Xobsvec; // Xobs sous une forme différente.
    for (int i = 0; i<nobs;i++){
      VectorXd X(1); X << Xobs(i); Xobsvec.push_back(X);
    }
    string endname = ".gnu";
    int dim_x = 1;      // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
    int samp_size = 80; //80 avant
    vector<VectorXd> X_predictions(samp_size);
    for (int i = 0; i < samp_size; i++)
    {
      VectorXd x(1); x << Xobs(nobs - 1) * double(i) / double(samp_size);
      X_predictions[i]=x;
    }

    //pour la MCMC

    int nautocor = 500;
    int nsel = nombre_samples_collected; //c'est la même variable avec 2 noms

    auto logprior_pars = [](VectorXd const &p)
    {
      return 0;
    };

    auto logprior_hpars = [](VectorXd const &h)
    {
      //pour avoir uniform prior en l.
      return h(0);
    };
    auto lambda_priormean = [](vector<VectorXd> const &X, VectorXd const &h)
    {
      return VectorXd::Zero(X.size());
    };

    //calcul des samples bayes et de l'approximation gaussienne/
    DoE doe_init(lb_t, ub_t, 1, 10); // DoE Halton
    string fname_data = foldname + "obs" + endname;
    WriteObs(Xobsvec,Yobs,fname_data);

    Density Dens(doe_init);
    Dens.SetModel(lambda_model);
    Dens.SetKernel(Kernel_Z_SE); //n'utilisons pas les dérivées pour ce cas.
    Dens.SetHparsBounds(lb_hpars, ub_hpars);
    Dens.SetLogPriorHpars(logprior_hpars);
    Dens.SetLogPriorPars(logprior_pars);
    Dens.SetPriorMean(lambda_priormean);
    Dens.SetObservations(Xobsvec,Yobs);
    Dens.SetOutputerr(false,noise,0);
    auto get_hpars=[](VectorXd const &theta){
      //non utilisé car calibration bayes.
      vector<VectorXd> p(1);
      return p;
    };

    auto compute_score = [&Dens, dim_theta, dim_hpars,&logprior_hpars](vector<VectorXd> const &p, VectorXd const &X)
    {
      VectorXd theta(dim_theta);
      theta = X.head(dim_theta);
      VectorXd hpars(dim_hpars);
      hpars = X.tail(dim_hpars);
      return Dens.loglikelihood_theta(theta, hpars)+logprior_hpars(hpars); 
    };

    auto in_bounds = [lb_hpars, ub_hpars, lb_t, ub_t, dim_theta, dim_hpars](VectorXd const &X)
    {
      VectorXd theta = X.head(dim_theta);
      VectorXd hpars = X.tail(dim_hpars);
      for (int i = 0; i < dim_theta; i++)
      {
        if (theta(i) > ub_t(i) || theta(i) < lb_t(i))
        {
          return false;
        }
      }
      for (int i = 0; i < dim_hpars; i++)
      {
        if (hpars(i) > ub_hpars(i) || hpars(i) < lb_hpars(i))
        {
          return false;
        }
      }
      return true;
    };
    VectorXd Xinit=X_init;
    MatrixXd Covinit=COV_init;
    vector<VectorXd> visited_steps = Run_MCMC(nombre_steps_mcmc, Xinit, Covinit, compute_score,get_hpars, in_bounds, generator);
    vector<VectorXd> selected_thetas;
    vector<VectorXd> selected_hpars;
    //on collecte nsel samples.
    for (int i = 0; i < nsel; i++)
    {
      VectorXd s = visited_steps[i * (visited_steps.size() / nsel)];
      selected_thetas.push_back(s.head(dim_theta));
      selected_hpars.push_back(s.tail(dim_hpars));
    }
    //Dens.SetNewSamples(selected_thetas);
    //Dens.SetNewHparsOfSamples(selected_hpars);
    //vérification du mélange de la MCMC

    //Dens.WritePredictions(X_predictions,foldname+ "preds"+endname);

    string fname = foldname + "autocor" + endname;
    Selfcor_diagnosis(visited_steps,nautocor,1, fname);
    WriteVectors(selected_thetas,selected_hpars,foldname+"samples"+endname);

    //calcul moyenne theta, std, moyenne sigma, std.
    //attention au scale de s. sert à rien de faire cv l à priori puisque il n'a pas d'influence.

    int chainsize=visited_steps.size();
    double tmean=0;
    double tstd=0;
    double smean=0;
    double sstd=0;
    double smedian=0;
    vector<double> vec_s;

    for(int i=0;i<chainsize;i++){
      tmean+=visited_steps[i](0);
      smean+=pow(10,visited_steps[i](1));
      vec_s.push_back(pow(10,visited_steps[i](1)));
    }

    tmean/=chainsize; smean/=chainsize;
    for(int i=0;i<chainsize;i++){
      tstd+=pow(visited_steps[i](0)-tmean,2);
      sstd+=pow(pow(10,visited_steps[i](1))-smean,2);
    }
    tstd=sqrt(tstd/chainsize);
    sstd=sqrt(sstd/chainsize);
    //trouver la médiane
    size_t n=vec_s.size()/2;
    nth_element(vec_s.begin(),vec_s.begin()+n,vec_s.end());
    smedian=vec_s[n];
    //on met tout ds un vectorXd
    VectorXd res(5);
    res << tmean, tstd,smean,sstd,smedian;
    return res; //renvoie les params du fit gaussien sur les samples.
  };

  auto draw_pts = [&generator](vector<int> pts_depart, int taille_finale)
  {
    //pour retirer un certain nombre de points d'une liste d'indices.
    uniform_real_distribution<double> distU(0, 1);
    vector<int> pts_arrivee = pts_depart;
    while (pts_arrivee.size() > taille_finale)
    {
      int ind_remove = pts_arrivee.size() * distU(generator);
      pts_arrivee.erase(pts_arrivee.begin() + ind_remove);
    };
    return pts_arrivee;
  };

  auto create_obsFD_total = [true_fct_scalar, noise, &generator, draw_pts](int nobs)
  {
    //créer les grid FD 4, 5, 6, max. nobs=20 donc max : 900 observations.
    VectorXd gridmaxFD(nobs * 45);
    VectorXd yobsmaxFD(nobs * 45);
    for (int i = 0; i < nobs * 45; i++)
    {
      double x = i * 1.0 / (nobs * 45);
      gridmaxFD(i) = x;
      yobsmaxFD(i) = true_fct_scalar(gridmaxFD(i)) + noise * distN(generator);
    }
    vector<int> ind45(nobs * 45);
    for (int i = 0; i < ind45.size(); i++)
    {
      ind45[i] = i;
    }
    vector<int> ind30 = draw_pts(ind45, nobs * 30);
    VectorXd grid6FD(nobs * 30);
    VectorXd yobs6FD(nobs * 30);
    //mettre les nouveaux pts
    for (int i = 0; i < nobs * 30; i++)
    {
      grid6FD(i) = gridmaxFD(ind30[i]);
      yobs6FD(i) = yobsmaxFD(ind30[i]);
    }
    //tirer les points à enlever
    for (int i = 0; i < nobs * 30; i++)
    {
      ind30[i] = i;
    }
    vector<int> ind15 = draw_pts(ind30, nobs * 15);
    VectorXd grid5FD(nobs * 15);
    VectorXd yobs5FD(nobs * 15);
    //mettre les nouveaux pts
    for (int i = 0; i < nobs * 15; i++)
    {
      grid5FD(i) = grid6FD(ind15[i]);
      yobs5FD(i) = yobs6FD(ind15[i]);
    }
    for (int i = 0; i < nobs * 15; i++)
    {
      ind15[i] = i;
    }
    vector<int> ind = draw_pts(ind15, nobs);
    VectorXd grid4FD(nobs * 5);
    VectorXd yobs4FD(nobs * 5);
    for (int i = 0; i < nobs * 5; i++)
    {
      grid4FD(i) = grid5FD(ind[i]);
      yobs4FD(i) = yobs5FD(ind[i]);
    }
    return make_tuple(grid4FD, yobs4FD, grid5FD, yobs5FD, grid6FD, yobs6FD, gridmaxFD, yobsmaxFD);
  };


  auto create_big_FD_grid = [true_fct_scalar, noise, &generator](int nobs){
    //créé le grid FD de taille maximale.
    VectorXd X(nobs);
    VectorXd Y(nobs);
    for (int i = 0; i < nobs; i++)
    {
      double x = i * 1.0 / (nobs);
      X(i) = x;
      Y(i) = true_fct_scalar(X(i)) + noise * distN(generator);
    }
    return make_pair(X,Y);
  };

  auto extract_FD_grid = [&generator,draw_pts](VectorXd const & Xprev, VectorXd const & Yprev,int nobs){
    //créé un nouveau grid FD de taille nobs.
    if(nobs > Xprev.size()){cerr << "erreur ! taille grid trop grande" << endl;}
    if(nobs == Xprev.size()){cerr << "extracting same grid size ?" << endl;}
    VectorXd X(nobs);
    VectorXd Y(nobs);
    vector<int> indprev;
    for(int i=0;i<Xprev.size();i++){
      indprev.push_back(i);
    }
    vector<int> indnew=draw_pts(indprev,nobs);
    for(int i=0;i<nobs;i++){
      X(i)=Xprev(indnew[i]);
      Y(i)=Yprev(indnew[i]);
    }
    return make_pair(X,Y);
  };

  //je veux deux fonctions. L'une créé une masse d'observations, genre 200. Et la seconde tire aléatoirement un certain nb de points depuis un grid précédent.
  generator.seed(56974364);


  int nobsmax=1000;
  auto gridmax=create_big_FD_grid(nobsmax);
  auto gridcurrent=gridmax;
  string fnamed = "results/cvgence.gnu";
  ofstream ofile(fnamed);
  string fnamef = "results/4FD/";
  vector<int> indexes; 
/*
  indexes.push_back(1000);
  indexes.push_back(900);
  indexes.push_back(700);
  indexes.push_back(600);
  
  indexes.push_back(500);
  indexes.push_back(400);
  indexes.push_back(300);
  
  indexes.push_back(200);
  indexes.push_back(190);
  indexes.push_back(180);
  indexes.push_back(170);
  indexes.push_back(160);
  indexes.push_back(150);
  indexes.push_back(140);
  indexes.push_back(130);
  indexes.push_back(120);
  indexes.push_back(110);
  indexes.push_back(100);
  indexes.push_back(90);
  indexes.push_back(80);
  indexes.push_back(70);
  indexes.push_back(60);
  indexes.push_back(50);
  indexes.push_back(40);
  indexes.push_back(30);
  indexes.push_back(20);
  indexes.push_back(10);
  */
  for(int i=100;i>4;i-=1){
    cout << "nobs : " << i << endl;
    gridcurrent=extract_FD_grid(gridcurrent.first,gridcurrent.second,i);
    VectorXd v=run_analysis(fnamef,gridcurrent.first, gridcurrent.second);
    ofile << i << " " << v(0) << " " << v(1) << " " << v(2) << " " << v(3) <<" " << v(4) << endl;
  }
  exit(0);




  auto tup = create_obsFD_total(nobs);
  cout << endl << "FD4" << endl;
  string fname = "results/4FD/";
  run_analysis(fname,get<0>(tup), get<1>(tup));

  exit(0);

  cout << endl << "FD5" << endl;
  fname = "results/5FD/";
  run_analysis(fname,get<2>(tup), get<3>(tup));

  cout << endl <<"FD6" << endl;
  fname = "results/6FD/";
  run_analysis(fname,get<4>(tup), get<5>(tup));

  cout << endl <<"FD7" << endl;
  fname = "results/7FD/";
  run_analysis(fname,get<6>(tup), get<7>(tup));

  exit(0);


  /*

  auto tupRD=create_obsRD_total(nobs);
  cout << "RD4" << endl;
  fname="results/4RD/";
  runanal_RD(get<0>(tupRD),get<1>(tupRD),fname);

  cout << "RD5" << endl;
  fname="results/5RD/";
  runanal_RD(get<0>(tupRD),get<2>(tupRD),fname);

  cout << "RD6" << endl;
  fname="results/6RD/";
  runanal_RD(get<0>(tupRD),get<3>(tupRD),fname);

  cout << "RD7" << endl;
  fname="results/7RD/";
  runanal_RD(get<0>(tupRD),get<4>(tupRD),fname);
  */

  exit(0);

  /*
  foldname="results/3ID/";
  tp=run_analysis_MCMCadapt(foldname,grid3ID,yobs_3ID);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;


  foldname="results/4ID/";
  tp=run_analysis_MCMCadapt(foldname,grid4ID,yobs_4ID);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;
  
*/

  /*

  foldname="results/3FD/";
  tp=run_analysis_MCMCadapt(foldname,grid3FD,yobs_3FD);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;


  foldname="results/4FD/";
  tp=run_analysis_MCMCadapt(foldname,grid4FD,yobs_4FD);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;
  

  vector<VectorXd> yobs_RD2(2);
  vector<VectorXd> yobs_RD4(4);
  for(int i=0;i<2;i++){
    yobs_RD2[i]=yobs_RD8[i];
    yobs_RD4[i]=yobs_RD8[i];
    yobs_RD4[i+2]=yobs_RD8[i+2];
  }

  foldname="results/2RD/";
  tp=run_analysis_RD(foldname,initgrid,yobs_RD2);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;


  foldname="results/3RD/";
  tp=run_analysis_RD(foldname,initgrid,yobs_RD4);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;


  foldname="results/4RD/";
  tp=run_analysis_RD(foldname,initgrid,yobs_RD8);
  t=get<0>(tp)(0);
  s=get<0>(tp)(1);
  tstd=get<1>(tp)(0,0);
  sstd=get<1>(tp)(1,1);
  cout << " t : " << t << " " << sqrt(tstd) << endl;
  cout << " s : " << s << " " << sqrt(sstd) << endl;
  
*/
  exit(0);
}
