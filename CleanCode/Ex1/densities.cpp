#include "densities.h"
#include <ctime>
#include <list>
#include <random>
#include <chrono>

using namespace std;
using namespace Eigen;

//utility functions
Eigen::VectorXd VtoVXD(std::vector<double> const &v)
{
  //turn vector<double> into VectorXd
  Eigen::VectorXd X(v.size());
  for (int i = 0; i < v.size(); i++)
  {
    X(i) = v[i];
  }
  return X;
}

std::vector<double> VXDtoV(Eigen::VectorXd const &X)
{
  //turn VectorXd into vector<double>
  std::vector<double> v(X.size());
  for (int i = 0; i < v.size(); i++)
  {
    v[i] = X(i);
  }
  return v;
}

double FindQuantile(double pct, VectorXd const &X)
{
  //returns the n-th element in a VectorXd
  int n = pct * X.size();
  vector<double> x = VXDtoV(X);
  nth_element(x.begin(), x.begin() + n, x.end());
  return x[n];
}

void WriteVector(vector<VectorXd> const &v, string const &filename)
{
  //write a vector to a file
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

double optroutine(nlopt::vfunc optfunc, void *data_ptr, VectorXd &X, VectorXd const &lb_hpars, VectorXd const &ub_hpars, double max_time)
{
  //routine d'optimisation sans gradient
  vector<double> x = VXDtoV(X);
  vector<double> lb_hpars_opt = VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt = VXDtoV(ub_hpars);
  //paramètres d'optimisation
  double ftol_large = 1e-5;
  double xtol_large = 1e-3;
  double ftol_fin = 1e-15;
  double xtol_fin = 1e-7;
  // 1 optimiseur local et un global.
  nlopt::opt local_opt(nlopt::LN_SBPLX, x.size());

  local_opt.set_max_objective(optfunc, data_ptr);
  local_opt.set_ftol_rel(ftol_large);
  local_opt.set_xtol_rel(xtol_large);
  local_opt.set_lower_bounds(lb_hpars_opt);
  local_opt.set_upper_bounds(ub_hpars_opt);
  local_opt.set_maxtime(max_time);
  nlopt::opt opt(nlopt::GD_MLSL_LDS, x.size());
  opt.set_max_objective(optfunc, data_ptr);
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxtime(max_time); //limite de temps : 60 sec.
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
  X = VtoVXD(x);
  return msup;
}

double optroutine_withgrad(nlopt::vfunc optfunc, void *data_ptr, VectorXd &X, VectorXd const &lb_hpars, VectorXd const &ub_hpars, double max_time)
{
  //routine d'optimisation avec gradient
  vector<double> x = VXDtoV(X);
  vector<double> lb_hpars_opt = VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt = VXDtoV(ub_hpars);
  //paramètres d'optimisation
  double ftol_large = 1e-10;
  double xtol_large = 1e-5;
  double ftol_fin = 1e-20;
  double xtol_fin = 1e-15;
  // 1 optimiseur local et un global.
  nlopt::opt local_opt(nlopt::LD_MMA, x.size()); //MMA en temps normal.

  local_opt.set_max_objective(optfunc, data_ptr);
  local_opt.set_ftol_rel(ftol_fin);
  //local_opt.set_xtol_rel(xtol_fin);
  local_opt.set_lower_bounds(lb_hpars_opt);
  local_opt.set_upper_bounds(ub_hpars_opt);

  nlopt::opt opt(nlopt::GD_MLSL_LDS, x.size());
  opt.set_max_objective(optfunc, data_ptr);
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxtime(max_time); //2 secondes au max.
  opt.set_local_optimizer(local_opt);
  double msup;                     /* the maximum objective value, upon return */
  int fin = opt.optimize(x, msup); //messages d'arrêt
  if (!fin == 3)
  {
    cout << "opti hpars message de fin : " << fin << endl;
  }
  //on relance une opti locale à partir du max. trouvé.
  local_opt.set_ftol_rel(ftol_fin);
  //local_opt.set_xtol_rel(xtol_fin);
  fin = local_opt.optimize(x, msup); //messages d'arrêt
  if (!fin == 3)
  {
    cout << "opti hpars message de fin : " << fin << endl;
  }
  X = VtoVXD(x);
  return msup;
}

double optfuncKOH(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* fonction à optimiser pour trouver les hpars koh.*/
  auto ptp = (tuple<const MatrixXd *, Density *> *)data; //cast
  auto tp = *ptp;
  const MatrixXd *Residus = get<0>(tp);
  const Density *d = get<1>(tp);
  const vector<VectorXd> *xconv = d->GetXlocations();
  VectorXd hpars = VtoVXD(x);
  double logvstyp = 30; //typical value of the log-likelihood function. Necessary when taking the exponential to avoid out of bounds numbers. Error messages appear when this value is too low.
  vector<double> prob(Residus->cols());
  MatrixXd G = d->Gamma(*xconv, hpars);
  LDLT<MatrixXd> ldlt(G);
  VectorXd pmean = d->EvaluatePMean(*xconv, hpars); 
  for (int i = 0; i < Residus->cols(); i++)
  {
    double g = d->loglikelihood_fast(Residus->col(i) - pmean, ldlt);
    prob[i] = g;
  }
  for (int i = 0; i < prob.size(); i++)
  {
    double p = prob[i];
    VectorXd theta = d->GetGrid()->at(i);
    double logprior = d->EvaluateLogPPars(theta);
    double f = exp(p + logprior - logvstyp);
    if (isinf(f))
    {
      cerr << "error in optfuncKOH. Try increasing the value of logvstyp. Try the value : " << p + logprior << endl;
      exit(0);
    }
    prob[i] = f;
  }
  double res = accumulate(prob.begin(), prob.end(), 0.0);
  res *= exp(d->EvaluateLogPHpars(hpars));
  return log(res);
};

double optfuncOpt_nograd(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  //on inclut l'uncertainty X ici.
  /* fonction à optimiser pour trouver les hpars optimaux. Normelemnt l'optimisation est seulement en dimension 2 car l'input uncertainty ne joue pas.*/
  //cast du null pointer
  pair<const VectorXd *, const DensityOpt *> *p = (pair<const VectorXd *, const DensityOpt *> *)data;
  const VectorXd *obs = p->first;
  const DensityOpt *d = p->second;
  VectorXd hpars = VtoVXD(x);
  const vector<VectorXd> *xconv = d->GetXlocations();
  LDLT<MatrixXd> ldlt(d->Gamma(*xconv, hpars));
  double ll = d->loglikelihood_fast(*obs, ldlt);
  double lp = d->EvaluateLogPHpars(hpars);
  return ll + lp;
};

double optfuncOpt_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* fonction à optimiser pour trouver les hpars optimaux.*/
  //écrite pour 2 hyperparamètres.
  //cast du null pointer
  pair<const VectorXd *, const DensityOpt *> *p = (pair<const VectorXd *, const DensityOpt *> *)data;
  const VectorXd *obs = p->first; //contient déjà yobs-ftheta.
  const DensityOpt *d = p->second;
  VectorXd hpars = VtoVXD(x);
  VectorXd obsmodif = *obs;
  const vector<VectorXd> *xconv = d->GetXlocations();
  LDLT<MatrixXd> ldlt(d->Gamma(*xconv, hpars)); //incX ici
  double ll = d->loglikelihood_fast(obsmodif, ldlt);
  double lp = d->EvaluateLogPHpars(hpars);
  //calcul des matrices des gradients
  if (!grad.size() == 0)
  {
    int nd = xconv->size();
    MatrixXd DG1 = MatrixXd::Zero(nd, nd);
    MatrixXd DG2 = MatrixXd::Zero(nd, nd);
    for (int i = 0; i < nd; i++)
    {
      for (int j = i; j < nd; j++)
      {
        //DG1(i,j)=d->m_DKernel1((*xconv)[i],(*xconv)[j],hpars);
        //DG2(i,j)=d->m_DKernel2((*xconv)[i],(*xconv)[j],hpars);
        if (i != j)
        {
          DG1(j, i) = DG1(i, j);
          DG2(j, i) = DG2(i, j);
        }
      }
    }
    MatrixXd Kinv = ldlt.solve(MatrixXd::Identity(nd, nd));
    VectorXd alpha = Kinv * obsmodif;
    MatrixXd aat = alpha * alpha.transpose();
    grad[0] = 0.5 * ((aat - Kinv) * DG1).trace(); // pas de prior
    grad[1] = 0.5 * ((aat - Kinv) * DG2).trace(); // pas de prior non plus
  }
  return ll + lp;
};

double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* This is the function you optimize for defining the GP */
  GP *proc = (GP *)data;       //Pointer to the GP
  Eigen::VectorXd p(x.size()); //Parameters to be optimized
  for (int i = 0; i < (int)x.size(); i++)
    p(i) = x[i];                 //Setting the proposed value of the parameters
  double value = proc->SetGP(p); //Evaluate the function
  if (!grad.empty())
  { //Cannot compute gradient : stop!
    std::cout << "Asking for gradient, I stop !" << std::endl;
    exit(1);
  }
  return value;
};

void Run_Burn_Phase_MCMC(int nburn, MatrixXd &COV_init, VectorXd &Xcurrento, function<double(vector<VectorXd>, VectorXd const &)> const &compute_score, function<vector<VectorXd>(VectorXd const &)> const &get_hpars, function<bool(VectorXd)> const &in_bounds, default_random_engine &generator)
{
  //phase de burn.
  int dim_mcmc = COV_init.cols();
  uniform_real_distribution<double> distU(0, 1);
  normal_distribution<double> distN(0, 1);
  VectorXd Xinit = Xcurrento;
  MatrixXd COV = COV_init;
  MatrixXd sqrtCOV = COV.llt().matrixL();
  vector<VectorXd> hparscurrent = get_hpars(Xinit);
  double finit = compute_score(hparscurrent, Xinit);
  VectorXd Xcurrent = Xinit;
  double fcurrent = finit;
  int naccept = 0;
  VectorXd acc_means = VectorXd::Zero(dim_mcmc);
  MatrixXd acc_var = MatrixXd::Zero(dim_mcmc, dim_mcmc);
  auto begin = chrono::steady_clock::now();
  cout << "fcurrent : " << fcurrent << endl;
  for (int i = 0; i < nburn; i++)
  {
    VectorXd Step(dim_mcmc);
    for (int j = 0; j < dim_mcmc; j++)
    {
      Step(j) = distN(generator);
    }
    VectorXd Xcandidate = Xcurrent + sqrtCOV * Step;
    if (in_bounds(Xcandidate))
    {
      vector<VectorXd> hparscandidate = get_hpars(Xcandidate);
      double fcandidate = compute_score(hparscandidate, Xcandidate);
      //cout << fcandidate;
      if (fcandidate > fcurrent || fcandidate - fcurrent > log(distU(generator)))
      {
        naccept++;
        Xcurrent = Xcandidate;
        fcurrent = fcandidate;
        hparscurrent = hparscandidate;
        //cout << " +";
      }
      //cout << endl;
    }
    acc_means += Xcurrent;
    acc_var += Xcurrent * Xcurrent.transpose();
  }
  double acc_rate = (double)(naccept) / (double)(nburn);
  MatrixXd CovProp = (pow(2.38, 2) / (double)(dim_mcmc)) * (acc_var / (nburn - 1) - acc_means * acc_means.transpose() / pow(1.0 * nburn, 2));
  auto end = chrono::steady_clock::now();
  cout << "burn phase over. "
       << " time : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s, accept rate : " << 100 * acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl
       << CovProp << endl;
  Xcurrento = Xcurrent;
  COV_init = CovProp;
}

vector<VectorXd> Run_MCMC(int nsteps, VectorXd const &Xinit, MatrixXd const &COV_init, function<double(vector<VectorXd>, VectorXd const &)> const &compute_score, function<vector<VectorXd>(VectorXd const &)> const &get_hpars, function<bool(VectorXd)> const &in_bounds, default_random_engine &generator)
{
  //Metropolis-Hastings algorithm with burn phase. Returns all visited steps (not including the burn phase).
  cout << "starting MCMC with " << nsteps << " samples." << endl;
  int dim_mcmc = Xinit.size();
  vector<VectorXd> allsamples;
  uniform_real_distribution<double> distU(0, 1);
  normal_distribution<double> distN(0, 1);
  MatrixXd COV = COV_init;
  VectorXd Xinit0 = Xinit;
  Run_Burn_Phase_MCMC(0.1 * nsteps, COV, Xinit0, compute_score, get_hpars, in_bounds, generator);
  MatrixXd sqrtCOV = COV.llt().matrixL();
  vector<VectorXd> hparsstart = get_hpars(Xinit0);
  double finit = compute_score(hparsstart, Xinit0);
  VectorXd Xcurrent = Xinit0;
  double fcurrent = finit;
  vector<VectorXd> hparscurrent = hparsstart;
  int naccept = 0;
  auto begin = chrono::steady_clock::now();
  for (int i = 0; i < nsteps; i++)
  {
    VectorXd Step(dim_mcmc);
    for (int j = 0; j < dim_mcmc; j++)
    {
      Step(j) = distN(generator);
    }
    VectorXd Xcandidate = Xcurrent + sqrtCOV * Step;
    if (in_bounds(Xcandidate))
    {
      vector<VectorXd> hparscandidate = get_hpars(Xcandidate);
      double fcandidate = compute_score(hparscandidate, Xcandidate);
      if (fcandidate > fcurrent || fcandidate - fcurrent > log(distU(generator)))
      {
        naccept++;
        Xcurrent = Xcandidate;
        fcurrent = fcandidate;
        hparscurrent = hparscandidate;
      }
    }
    allsamples.push_back(Xcurrent);
  }
  auto end = chrono::steady_clock::now();
  double acc_rate = (double)(naccept) / (double)(nsteps);
  cout << "MCMC phase over. time : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s, accept rate : " << 100 * acc_rate << " pct. " << endl;
  return allsamples;
}

VectorXd Lagged_mean(vector<VectorXd> const &v, int n)
{
  int dim_mcmc = v[0].size();
  VectorXd ans = VectorXd::Zero(dim_mcmc);
  for (int i = 0; i < v.size() - n; i++)
  {
    ans += v[i].cwiseProduct(v[i + n]);
  }
  return ans / (double(v.size() - n));
}

void Selfcor_diagnosis(vector<VectorXd> const &samples, int nstepsmax, double proportion, string const &filename)
{
  if (proportion > 1)
  {
    cerr << "error : cannot select that many samples" << endl;
    return;
  }
  int dim_mcmc = samples[0].size();
  int nselected = proportion * samples.size();
  cout << "self-correlation computed on " << nselected << " samples." << endl;
  vector<VectorXd> allmcmcsamples_selected(nselected);
  VectorXd mean = VectorXd::Zero(dim_mcmc);
  VectorXd var = VectorXd::Zero(dim_mcmc);
  for (int i = 0; i < nselected; i++)
  {
    allmcmcsamples_selected[i] = samples[i + samples.size() - nselected];
    mean += allmcmcsamples_selected[i];
  }
  mean /= nselected;
  for (int i = 0; i < nselected; i++)
  {
    allmcmcsamples_selected[i] -= mean;
    var += allmcmcsamples_selected[i].cwiseProduct(allmcmcsamples_selected[i]);
  }
  var /= nselected;
  ofstream ofile(filename);
  VectorXd integ = VectorXd::Zero(dim_mcmc);
  for (int i = 0; i < nstepsmax; i++)
  {
    VectorXd cor = Lagged_mean(allmcmcsamples_selected, i).cwiseQuotient(var);
    for (int j = 0; j < cor.size(); j++)
    {
      ofile << cor(j) << " ";
    }
    integ += cor;
    ofile << endl;
  }
  ofile.close();
  cout << "1D correlation lengths :" << endl;
  cout << integ.transpose() << endl;
  cout << "maximum : " << integ.maxCoeff() << endl;
}

DoE::DoE(){

};

DoE::DoE(VectorXd const &lb, VectorXd const &ub, int n) : m_lb_pars(lb), m_ub_pars(ub)
{
  //Build a DoE with a regular grid
  int dim = lb.size();
  int npts = pow(n, dim);
  VectorXd theta_courant(dim);
  VectorXd ind_courant(dim);
  for (int i = 0; i < npts; i++)
  {
    ind_courant = indices(i, n, dim);
    for (int j = 0; j < dim; j++)
    {
      theta_courant(j) = m_lb_pars(j) + (ind_courant(j) + 0.5) * (m_ub_pars(j) - m_lb_pars(j)) / double(n);
    }
    m_grid.push_back(theta_courant);
  }
};

DoE::DoE(VectorXd const &lb, VectorXd const &ub, int ntotal, default_random_engine &generator) : m_lb_pars(lb), m_ub_pars(ub)
{
  //Build a DoE with an uniform LHS
  uniform_real_distribution<double> distU(0, 1);
  int dim = lb.size();
  std::vector<VectorXd> perm(dim);
  for (int i = 0; i < dim; i++)
  {
    perm[i] = Randpert(ntotal, generator);
  }
  VectorXd theta_courant(dim);
  for (int i = 0; i < ntotal; i++)
  {
    for (int j = 0; j < dim; j++)
    {
      theta_courant(j) = lb(j) + (ub(j) - lb(j)) * (perm[j](i) + distU(generator)) / double(ntotal);
    }
    m_grid.push_back(theta_courant);
  }
};

DoE::DoE(VectorXd const &lb, VectorXd const &ub, int npoints, int first_element) : m_lb_pars(lb), m_ub_pars(ub)
{
  //Build a DoE with a QMC Halton sequence
  int dim = lb.size();
  MatrixXd H = halton_sequence(first_element, first_element + npoints - 1, dim);
  for (int i = 0; i < npoints; i++)
  {
    VectorXd theta = lb + (ub - lb).cwiseProduct(H.col(i));
    m_grid.push_back(theta);
  }
};

VectorXd DoE::Randpert(int n, default_random_engine &generator) const
{
  VectorXd result(n);
  std::uniform_real_distribution<double> distU(0, 1);
  for (int i = 0; i < n; i++)
  {
    result(i) = i;
  }
  for (int i = n - 1; i > 0; i--)
  {
    int j = int(floor(distU(generator) * (i + 1)));
    double a = result(i);
    result(i) = result(j);
    result(j) = a;
  }
  return result;
};

VectorXd DoE::indices(int const s, int const n, int const d)
{
  VectorXd multiindice(d);
  int indloc;
  int remainder = s;
  for (int pp = d - 1; pp > -1; pp--)
  {
    indloc = (int)remainder % n; //On commence par le coefficient le plus à droite.
    multiindice(pp) = indloc;
    remainder = (remainder - indloc) / n;
  }
  return multiindice;
};

void DoE::WriteGrid(string const &filename) const
{
  //écriture des thetas juste pour être sûr.
  ofstream ofile(filename);
  for (VectorXd const &v : m_grid)
  {
    for (int i = 0; i < v.size(); i++)
    {
      ofile << v(i) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

Density::Density(){

};

Density::Density(Density const &d)
{
  m_model = d.m_model;
  m_priormean = d.m_priormean;
  m_logpriorhpars = d.m_logpriorhpars;
  m_logpriorpars = d.m_logpriorpars;
  m_Kernel = d.m_Kernel;
  m_Kernel_derivatives = d.m_Kernel_derivatives;
  m_dim_hpars = d.m_dim_hpars;
  m_dim_pars = d.m_dim_pars;
  m_lb_hpars = d.m_lb_hpars;
  m_ub_hpars = d.m_ub_hpars;
  m_lb_pars = d.m_lb_pars;
  m_ub_pars = d.m_ub_pars;
  m_Grid = d.m_Grid;
  m_Xlocations = d.m_Xlocations;
  m_observations = d.m_observations;
  m_samples = d.m_samples;
  m_hparsofsamples = d.m_hparsofsamples;
  m_allmcmcsamples = d.m_allmcmcsamples;
  m_inputerr = d.m_inputerr;
  m_derivatives_obs = d.m_derivatives_obs;
  m_derivatives_preds = d.m_derivatives_preds;
};

Density::Density(DoE const &g)
{
  //construction à partir d'un grid (nécessaire).
  m_Grid = g;
  m_lb_pars = m_Grid.m_lb_pars;
  m_ub_pars = m_Grid.m_ub_pars;
  m_dim_pars = m_lb_pars.size();
}

void Density::SetDoE(DoE const &g)
{
  //set new grid
  m_Grid = g;
  m_lb_pars = m_Grid.m_lb_pars;
  m_ub_pars = m_Grid.m_ub_pars;
  m_dim_pars = m_lb_pars.size();
}

void Density::SetObservations(vector<VectorXd> const &Xlocations, VectorXd const &observations)
{
  if (!Xlocations.size() == observations.size())
  {
    cerr << "erreur : mauvais nombre d'observations" << endl;
  }
  else
  {
    m_Xlocations = Xlocations;
    m_observations = observations;
  }
}

Eigen::MatrixXd Density::Gamma(vector<VectorXd> const &locs, Eigen::VectorXd const &hpar) const
{
  // Renvoie la matrice de corrélation avec  bruit
  //bruit en nugget juste pour être sûr de l'inversibilité.
  int nd = locs.size();
  double noisey = hpar(0); //bruit
  Eigen::MatrixXd A(nd, nd);
  for (int i = 0; i < nd; i++)
  {
    for (int j = i; j < nd; j++)
    {
      A(i, j) = m_Kernel(locs[i], locs[j], hpar);
      if (i != j)
      {
        A(j, i) = A(i, j);
      }
      else
      {
        A(i, j) += pow(noisey, 2); //kernel.
      }
    }
  }
  return A;
}

double Density::loglikelihood_fast(VectorXd const &obs, LDLT<MatrixXd> const &ldlt) const
{
  //calcul de la LL à la dernière étape.
  VectorXd Alpha = ldlt.solve(obs);
  int nd = obs.size();
  //return -0.5*obs.dot(Alpha)-0.5*(ldlt.vectorD().array().log()).sum()-0.5*nd*log(2*3.1415);
  return -0.5 * obs.dot(Alpha) - 0.5 * (ldlt.vectorD().array().log()).sum();
}

double Density::loglikelihood_theta_fast(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars, Eigen::LDLT<Eigen::MatrixXd> const &ldlt) const
{
  //écrite pour un AUGDATA de taille 1.
  VectorXd obs = m_observations - m_model(m_Xlocations, theta); //pas de priormean pour aller plus vite.
  return loglikelihood_fast(obs, ldlt);
}

double Density::loglikelihood_theta(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar) const
{
  //évaluation de la LDLT de Gamma. a priori écrite pour un AUGDATA de taille 1 également. Si la taille change, on pourrait faire plusieurs gamma (plusieurs hpars.)
  MatrixXd G = Gamma(m_Xlocations, hpar);
  LDLT<MatrixXd> ldlt(G);
  return loglikelihood_theta_fast(theta, hpar, ldlt);
}

LDLT<MatrixXd> Density::GetLDLT(VectorXd const &hpars)
{
  MatrixXd G = Gamma(m_Xlocations, hpars);
  LDLT<MatrixXd> ldlt(G);
  return ldlt;
}

VectorXd Density::HparsKOH(VectorXd const &hpars_guess, double max_time) const
{
  auto begin = chrono::steady_clock::now();
  VectorXd guess = hpars_guess;
  MatrixXd Residustheta(m_observations.size(), m_Grid.m_grid.size());
  for (int i = 0; i < m_Grid.m_grid.size(); i++)
  {
    VectorXd theta = m_Grid.m_grid[i];
    Residustheta.col(i) = -m_model(m_Xlocations, theta) + m_observations; //priormean will be included in optfuncKOH.
  }
  auto tp = make_tuple(&Residustheta, this);
  double msup = optroutine(optfuncKOH, &tp, guess, m_lb_hpars, m_ub_hpars, max_time);
  auto end = chrono::steady_clock::now();
  cout << guess.transpose() << endl;
  cout << "KOH optimisation over. Time : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s." << endl;
  return guess;
}

bool Density::in_bounds_pars(VectorXd const &pars) const
{
  for (int i = 0; i < m_dim_pars; i++)
  {
    if (pars(i) < m_lb_pars(i) || pars(i) > m_ub_pars(i))
    {
      return false;
    }
  }
  return true;
}

bool Density::in_bounds_hpars(VectorXd const &hpars) const
{
  for (int i = 0; i < m_dim_hpars; i++)
  {
    if (hpars(i) < m_lb_hpars(i) || hpars(i) > m_ub_hpars(i))
    {
      return false;
    }
  }
  return true;
}

void Density::WriteSamples(string const &filename) const
{
  ofstream ofile(filename);
  for (int i = 0; i < m_samples.size(); i++)
  {
    for (int j = 0; j < m_samples[0].size(); j++)
    {
      ofile << m_samples[i](j) << " ";
    }
    for (int j = 0; j < m_hparsofsamples[0].size(); j++)
    {
      ofile << m_hparsofsamples[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

VectorXd Density::meanF(vector<VectorXd> const &X) const
{
  //prédiction moyenne du modèle
  VectorXd mpf = VectorXd::Zero(X.size());
  for (VectorXd const &s : m_samples)
  {
    mpf += m_model(X, s);
  }
  return mpf / m_samples.size();
}

MatrixXd Density::varF(vector<VectorXd> const &X) const
{
  //variance de f seule.
  VectorXd mpf = meanF(X);
  MatrixXd SecondMoment = MatrixXd::Zero(mpf.size(), mpf.size());
  for_each(m_samples.begin(), m_samples.end(), [&SecondMoment, &X, this](VectorXd const &theta) mutable
           {
             VectorXd Pred = this->m_model(X, theta);
             SecondMoment += Pred * Pred.transpose();
           });
  MatrixXd var = SecondMoment / m_samples.size() - mpf * mpf.transpose();
  return var;
}

VectorXd Density::meanZCondTheta(vector<VectorXd> const &X, VectorXd const &theta, VectorXd const &hpars) const
{
  //prédiction moyenne de Z sur Xprofile, pour le i-ème theta de m_samples.
  VectorXd y(X.size());
  y = m_observations - m_model(m_Xlocations, theta) - m_priormean(m_Xlocations, hpars);
  MatrixXd G = Gamma(m_Xlocations, hpars);
  MatrixXd K = MatrixXd::Zero(X.size(), m_Xlocations.size());
  for (int i = 0; i < K.rows(); i++)
  {
    for (int j = 0; j < K.cols(); j++)
    {
      K(i, j) = m_Kernel(X[i], m_Xlocations[j], hpars);
    }
  }
  LDLT<MatrixXd> ldlt(G);
  VectorXd predmean = m_priormean(X, hpars) + K * ldlt.solve(y);
  return predmean;
}

MatrixXd Density::varZCondTheta(vector<VectorXd> const &X, VectorXd const &theta, VectorXd const &hpars) const
{
  //variance prédictive de z sur Xprofile, pour le i-ème theta de m_samples.
  MatrixXd G = Gamma(m_Xlocations, hpars);
  MatrixXd K = MatrixXd::Zero(X.size(), m_Xlocations.size());
  for (int i = 0; i < K.rows(); i++)
  {
    for (int j = 0; j < K.cols(); j++)
    {
      K(i, j) = m_Kernel(X[i], m_Xlocations[j], hpars);
    }
  }
  MatrixXd Kprior = MatrixXd::Zero(X.size(), X.size());
  for (int i = 0; i < Kprior.rows(); i++)
  {
    for (int j = 0; j < Kprior.cols(); j++)
    {
      Kprior(i, j) = m_Kernel(X[i], X[j], hpars);
    }
  }
  LDLT<MatrixXd> ldlt(G);
  MatrixXd VarPred = Kprior - K * ldlt.solve(K.transpose());
  return VarPred;
}

MatrixXd Density::PredFZ(vector<VectorXd> const &X) const
{
  //predictions avec f+z. Première colonne : moyenne, Deuxième colonne : variance de E[f+z]. Troisième colonne : espérance de var[z].
  //récupération des valeurs de E[f+z|theta]
  //Calcul de la composante E[Var z] également.
  VectorXd mean = VectorXd::Zero(X.size());
  MatrixXd SecondMoment = MatrixXd::Zero(X.size(), X.size());
  MatrixXd Evarz = MatrixXd::Zero(X.size(), X.size());
  for (int i = 0; i < m_samples.size(); i++)
  {
    VectorXd theta = m_samples[i];
    VectorXd hpars = m_hparsofsamples[i];
    VectorXd fpred = m_model(X, theta);
    VectorXd zpred = meanZCondTheta(X, theta, hpars);
    mean += fpred + zpred;
    SecondMoment += (fpred + zpred) * (fpred + zpred).transpose();
    Evarz += varZCondTheta(X, theta, hpars);
  }
  mean /= m_samples.size();
  SecondMoment /= m_samples.size();
  Evarz /= m_samples.size();
  MatrixXd VarEfz = SecondMoment - mean * mean.transpose();
  MatrixXd res(X.size(), 3);
  res.col(0) = mean;
  res.col(1) = VarEfz.diagonal();
  res.col(2) = Evarz.diagonal();
  return res;
}

VectorXd Density::DrawZCondTheta(vector<VectorXd> const &X, VectorXd const &theta, VectorXd const &hpars_z, default_random_engine &generator) const
{
  //tirage d'une réalisation de z pour un theta et des hpars donnés.
  normal_distribution<double> distN(0, 1);
  VectorXd mean = meanZCondTheta(X, theta, hpars_z);
  MatrixXd Cov = varZCondTheta(X, theta, hpars_z);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> DEC(Cov);
  Eigen::VectorXd D = DEC.eigenvalues();
  for (unsigned i = 0; i < D.rows(); i++)
    D(i) = sqrt(fabs(D(i)));
  VectorXd sample(mean.size());
  for (int i = 0; i < sample.size(); i++)
  {
    sample(i) = distN(generator) * D(i);
  }
  return mean + DEC.eigenvectors() * sample;
}

void Density::WritePredictions(vector<VectorXd> const &X, string const &filename) const
{
  // write predictions of f and f+z, with uncertainty. f+z is normally distributed so its variance is explicit. For uncertainty about f, we use all posterior samples to compute 95% confidence intervals. 
  default_random_engine generator{static_cast<long unsigned int>(time(0))};
  uniform_int_distribution<int> U(0, m_samples.size() - 1);

  //quantities around f + z
  MatrixXd Predictions = PredFZ(X); //col0 : moyenne fz, col1: varefz, col2: evarz

  //quantities around f
  VectorXd meanf = meanF(X);
  MatrixXd fvalues(meanf.rows(), m_samples.size());
  for (int i = 0; i < m_samples.size(); i++)
  {
    fvalues.col(i) = m_model(X, m_samples[i]);
  }
  for (int i = 0; i < fvalues.rows(); i++)
  {
    VectorXd V = fvalues.row(i).transpose();
    vector<double> v = VXDtoV(V);
    std::sort(v.begin(), v.end());
    fvalues.row(i) = VtoVXD(v).transpose();
  }
  VectorXd quant2p5(meanf.rows()); //these quantiles are the limits of the 95% confidence interval.
  VectorXd quant97p5(meanf.rows());
  for (int i = 0; i < meanf.rows(); i++)
  {
    VectorXd R = fvalues.row(i);
    quant2p5(i) = R(int(0.025 * R.size()));
    quant97p5(i) = R(int(0.975 * R.size()));
  }

  ofstream ofile(filename);
  //columns printed in the file : X(maybe multiple columns), E_theta[f+z], sqrt(Var[f+z]), sqrt(E_theta[Var[z]]), E_theta[f], t_2.5[f], t_97.5[f].
  for (int i = 0; i < Predictions.rows(); i++)
  {
    for (int j=0; j < X[i].size();j++){
      ofile << X[i](j) << " ";
    }
    ofile << Predictions(i,0) << " " << sqrt(Predictions(i,1)+Predictions(i,2)) << " " << sqrt(Predictions(i,2)) << " " << meanf(i) << " " << quant2p5(i) << " " << quant97p5(i) << endl;
  }
  ofile.close();
}

void Density::WriteSamplesFandZ(vector<VectorXd> const &X, string const &filenameF, string const &filenameZ) const
{
  //draw samples of f and corresponding samples of z.
  int ndraws = 10;
  cout << "drawing " << ndraws <<" samples of f and z" << endl;
  default_random_engine generator{static_cast<long unsigned int>(time(0))};
  uniform_int_distribution<int> U(0, m_samples.size() - 1);
  vector<VectorXd> fs(ndraws);
  vector<VectorXd> zs(ndraws);
  for (int i = 0; i < ndraws; i++)
  {
    int r = U(generator);
    fs[i] = m_model(X, m_samples[r]);
    zs[i] = DrawZCondTheta(X, m_samples[r], m_hparsofsamples[r], generator);
  }
  ofstream ofile(filenameF);
  for (int i = 0; i < ndraws; i++)
  {
    for (int j = 0; j < X.size(); j++)
    {
      ofile << fs[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
  ofile.open(filenameZ);
  for (int i = 0; i < ndraws; i++)
  {
    for (int j = 0; j < X.size(); j++)
    {
      ofile << zs[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

DensityOpt::DensityOpt(Density const &d) : Density(d)
{
  m_samples.clear();
  m_hparsofsamples.clear();
  m_allmcmcsamples.clear();
}

VectorXd DensityOpt::HparsOpt(VectorXd const &theta, VectorXd hpars_guess, double max_time) const
{
  VectorXd obsmtheta = m_observations - m_model(m_Xlocations, theta);
  auto p = make_pair(&obsmtheta, this);
  //int fin=optroutine_withgrad(optfuncOpt_withgrad,&p,guess,m_lb_hpars,m_ub_hpars,max_time);
  //version sans gradient quand la prior mean est compliquée...
  //on optimise les paramètres du noyau
  int fin = optroutine(optfuncOpt_nograd, &p, hpars_guess, m_lb_hpars, m_ub_hpars, max_time);
  return hpars_guess;
}

VectorXd DensityOpt::EvaluateHparOpt(VectorXd const &theta) const
{
  //évaluation de l'hpar optimal par les GPs. On utilise uniquement la prédiction moyenne.
  int nmodes = m_vgp_hpars_opti.size();
  VectorXd meansgps(nmodes);
  for (int i = 0; i < nmodes; i++)
  {
    meansgps(i) = m_vgp_hpars_opti[i].EvalMean(theta);
  }
  VectorXd Ymean = m_featureMeans + m_VP * m_Acoefs * meansgps;
  return Ymean;
}

std::vector<Eigen::VectorXd> DensityOpt::Compute_optimal_hpars(double max_time, string filename)
{
  //calcul de tous les hpars optimaux sur m_grid, et rangement dans m_hpars_opti.
  //on les écrit ensuite dans le fichier correspondant.
  m_hpars_opti.clear();
  VectorXd hpars_guess = 0.5 * (m_lb_hpars + m_ub_hpars);
  const vector<VectorXd> *grid = GetGrid();
  auto begin = chrono::steady_clock::now();
  transform(grid->begin(), grid->end(), back_inserter(m_hpars_opti), [&hpars_guess, this, max_time](VectorXd const &theta) mutable
            {
              VectorXd hpars_opt = HparsOpt(theta, hpars_guess, max_time);
              hpars_guess = hpars_opt; //warm restart
              AUGDATA dat;
              dat.SetX(theta);
              dat.SetValue(hpars_opt);
              return dat;
            });
  auto end = chrono::steady_clock::now();
  //affichage dans un fichier
  ofstream ofile(filename);
  for (AUGDATA const &d : m_hpars_opti)
  {
    VectorXd X = d.GetX();
    VectorXd hpars = d.Value();
    //cout << "theta : " << X.transpose() << endl;
    //cout << "hparsopt : " << hpars.transpose() << endl;
    for (int i = 0; i < X.size(); i++)
    {
      ofile << X(i) << " ";
    }
    for (int i = 0; i < hpars.size(); i++)
    {
      ofile << hpars(i) << " ";
    }
    ofile << endl;
  }
  ofile.close();
  //calcul de quelques statistiques sur ces hpars optimaux obtenus. moyenne, et matrice de covariance des données.
  VectorXd mean = VectorXd::Zero(m_dim_hpars);
  MatrixXd SecondMoment = MatrixXd::Zero(m_dim_hpars, m_dim_hpars);
  double mean_lp = 0;
  for (AUGDATA const &a : m_hpars_opti)
  {
    mean_lp += loglikelihood_theta(a.GetX(), a.Value());
    mean += a.Value();
    SecondMoment += a.Value() * a.Value().transpose();
  }
  mean_lp /= m_hpars_opti.size();
  mean /= m_hpars_opti.size();
  SecondMoment /= m_hpars_opti.size();
  MatrixXd Var = SecondMoment - mean * mean.transpose();
  cout << " fin de calcul des hpars opti sur le grid. Moyenne : " << mean.transpose() << endl;
  cout << "moyenne des logposts:" << mean_lp << endl;
  cout << "temps de calcul : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s" << endl;
  //on resize les hpars dans un vector<VectorXd> pour les renvoyer
  vector<VectorXd> hparsfound;
  for (AUGDATA const &d : m_hpars_opti)
  {
    hparsfound.push_back(d.Value());
  }
  return hparsfound;
}

vector<VectorXd> DensityOpt::Return_optimal_hpars(double max_time) const
{
  //calcul de tous les hpars optimaux sur m_grid.
  vector<VectorXd> res;
  VectorXd hpars_guess = 0.5 * (m_lb_hpars + m_ub_hpars);
  const vector<VectorXd> *grid = GetGrid();
  auto begin = chrono::steady_clock::now();
  transform(grid->begin(), grid->end(), back_inserter(res), [&hpars_guess, this, max_time](VectorXd const &theta) mutable
            {
              VectorXd hpars_opt = HparsOpt(theta, hpars_guess, max_time);
              hpars_guess = hpars_opt; //warm restart
              return hpars_opt;
            });
  auto end = chrono::steady_clock::now();
  cout << "fin des optimisations. temps : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s." << endl;
  return res;
}

void DensityOpt::Test_hGPs(int npoints, double max_time)
{
  //calcul d'un nouveau grid de thetas, optimisation sur ces points, et évaluation de la qualité de prédiction des GPs sur ces points.
  default_random_engine generator;
  generator.seed(16);
  int nthetas = npoints;
  auto ps = [](MatrixXd const &A, MatrixXd const &B) -> double
  {
    return (A.transpose() * B).trace();
  };
  VectorXd hpars_guess = 0.5 * (m_lb_hpars + m_ub_hpars);
  uniform_real_distribution<double> distU(0, 1);
  vector<VectorXd> newgrid; //taille nthetas
  MatrixXd Hopt(m_dim_hpars, nthetas);
  if (m_newgrid.size() == 0)
  {
    //construction des valeurs
    for (int i = 0; i < nthetas; i++)
    {
      VectorXd t(3);
      t << distU(generator), distU(generator), distU(generator);
      newgrid.push_back(t);
    }
    for (int i = 0; i < nthetas; i++)
    {
      Hopt.col(i) = HparsOpt(newgrid[i], hpars_guess, max_time);
    }
    m_newgrid = newgrid;
    m_Hopt_newgrid = Hopt;
  }
  else
  {
    //valeurs précédemment construites
    newgrid = m_newgrid;
    Hopt = m_Hopt_newgrid;
  }
  MatrixXd Hoptpred(m_dim_hpars, nthetas);
  for (int i = 0; i < nthetas; i++)
  {
    Hoptpred.col(i) = EvaluateHparOpt(newgrid[i]);
  }
  MatrixXd Hoptv = Hopt.row(0);
  MatrixXd Hoptpredv = Hoptpred.row(0);
  double erredm = sqrt(ps(Hoptv - Hoptpredv, Hoptv - Hoptpredv) / ps(Hoptv, Hoptv));
  Hoptv = Hopt.row(1);
  Hoptpredv = Hoptpred.row(1);
  double errexp = sqrt(ps(Hoptv - Hoptpredv, Hoptv - Hoptpredv) / ps(Hoptv, Hoptv));
  cout << "erreur relative des hGPs sur le grid de validation : edm : " << erredm * 100 << " pct,lcor : " << errexp * 100 << endl;
}

Eigen::VectorXd DensityOpt::Test_hGPs_on_sample(std::vector<Eigen::VectorXd> const &theta_ref, std::vector<Eigen::VectorXd> const &hpars_ref) const
{
  //test de la performance des hgps sur un sample donné.
  if (theta_ref.size() != hpars_ref.size())
  {
    cerr << "erreur de taille" << endl;
  }
  cout << "test hgps on a sample size " << theta_ref.size() << endl;
  vector<VectorXd> approx_hpars(theta_ref.size());
  transform(theta_ref.begin(), theta_ref.end(), approx_hpars.begin(), [this](VectorXd const &theta)
            { return EvaluateHparOpt(theta); });
  vector<double> true_logliks(theta_ref.size());
  vector<double> approx_logliks(theta_ref.size());
  for (int i = 0; i < theta_ref.size(); i++)
  {
    true_logliks[i] = loglikelihood_theta(theta_ref[i], hpars_ref[i]) + m_logpriorhpars(hpars_ref[i]);
    approx_logliks[i] = loglikelihood_theta(theta_ref[i], approx_hpars[i]) + m_logpriorhpars(approx_hpars[i]);
  }
  //calcul de l'erreur moyenne en log-vraisemblance
  double errmoy = 0;
  for (int i = 0; i < true_logliks.size(); i++)
  {
    errmoy += pow(true_logliks[i] - approx_logliks[i], 2);

    if (true_logliks[i] - approx_logliks[i] > 1)
    {
      cerr << "erreur ll true sous-estimation. " << endl;
      cerr << "hpars true : " << hpars_ref[i].transpose() << endl;
      cerr << "hpars approx : " << approx_hpars[i].transpose() << endl;
      cerr << "ll true : " << true_logliks[i] << ", ll approx : " << approx_logliks[i] << endl;
      cerr << "in bounds hpars approx ? " << in_bounds_hpars(approx_hpars[i]) << endl;
    }
    /*
  if(approx_logliks[i]>0.1+true_logliks[i]){cerr << "erreur ll true surestimation. " <<endl;
  cerr<< "hpars true : " << hpars_ref[i].transpose()<< endl;
  cerr<< "hpars approx : " <<approx_hpars[i].transpose()<< endl;
  cerr<< "ll true : " << true_logliks[i]<< ", ll approx : "<< approx_logliks[i] <<endl;
  cerr <<"in bounds hpars approx ? "<<in_bounds_hpars(approx_hpars[i])<< endl;}*/
  }
  //valeur rms
  errmoy = sqrt(errmoy / true_logliks.size()); //erreur moyenne en log-probabilité.
  //calcul de l'erreur moyenne sur chaque hpar
  VectorXd errmoy_hpars = VectorXd::Zero(m_dim_hpars);
  VectorXd cumsum_hpars = VectorXd::Zero(m_dim_hpars);
  for (int i = 0; i < true_logliks.size(); i++)
  {
    errmoy_hpars.array() += ((hpars_ref[i] - approx_hpars[i]).array().square());
    cumsum_hpars.array() += ((hpars_ref[i]).array().square());
  }
  errmoy_hpars.array() = 100 * (errmoy_hpars.cwiseQuotient(cumsum_hpars)).array().sqrt(); //assignation à array ou au vectorxd directement ?
  VectorXd res(m_dim_hpars + 1);
  for (int i = 0; i < m_dim_hpars; i++)
  {
    res(i) = errmoy_hpars(i);
  }
  res(m_dim_hpars) = errmoy;
  cout << "testou : " << endl;
  cout << "hpars true : " << hpars_ref[50].transpose() << endl;
  cout << "hpars estimated : " << approx_hpars[50].transpose() << endl;
  cout << "ll true : " << true_logliks[50] << endl;
  cout << "ll estimated : " << approx_logliks[50] << endl;
  return res;
}

void DensityOpt::BuildHGPs_noPCA(double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), MatrixXd const &Bounds_hpars_GPs, VectorXd const &Hpars_guess_GPs)
{
  //initialisation des HGPs sans faire de PCA (un hGP par hyperparamètre directement)
  //on normalise quand même la valeur des GPs pour qu'ils soient sur le même range/
  //récupération des data pour les GPs.
  if (m_hpars_opti.size() == 0)
  {
    cerr << "erreur : hpars optis non calculés !" << endl;
  }
  cout << "construction des hGPs individuels sans PCA : " << endl;
  int nmodes = m_dim_hpars;
  MatrixXd U(m_dim_hpars, m_hpars_opti.size()); //matrice des données
  MatrixXd P(m_dim_pars, m_hpars_opti.size());  //matrice des thetas
  for (int i = 0; i < m_hpars_opti.size(); i++)
  {
    U.col(i) = m_hpars_opti[i].Value();
    P.col(i) = m_hpars_opti[i].GetX();
  }
  m_featureMeans = U.rowwise().mean();
  U = U.colwise() - m_featureMeans;
  //calcul des STDs de chaque feature et normalisation.
  VectorXd stds = VectorXd::Zero(m_dim_hpars);
  for (int i = 0; i < m_dim_hpars; i++)
  {
    stds(i) = sqrt(U.row(i).array().square().sum() / m_hpars_opti.size());
    if (stds(i) == 0.)
    {
      stds(i) += 0.01 * m_featureMeans(i);
      cout << "correction d'hpars constants." << endl;
    }
  }
  m_VP = MatrixXd::Identity(m_dim_hpars, m_dim_hpars);
  m_Acoefs = stds.asDiagonal();
  MatrixXd normedA = m_Acoefs.inverse() * U;
  //on met sous forme vector<vector> DATA pour la passer aux GPs
  vector<vector<DATA>> vd(nmodes);
  for (int j = 0; j < nmodes; j++)
  {
    vector<DATA> v(m_hpars_opti.size());
    for (int i = 0; i < m_hpars_opti.size(); i++)
    {
      DATA dat;
      dat.SetX(P.col(i));
      dat.SetValue(normedA(j, i));
      v[i] = dat;
    }
    vd[j] = v;
  }
  vector<GP> vgp(nmodes);
  for (int i = 0; i < nmodes; i++)
  {
    GP gp(Kernel_GP);
    gp.SetData(vd[i]);
    gp.SetGP(Hpars_guess_GPs);
    vgp[i] = gp;
  }
  m_vgp_hpars_opti = vgp;
  m_Bounds_hpars_GPs = Bounds_hpars_GPs;
}

VectorXd DensityOpt::opti_1gp(int i, VectorXd &hpars_guess)
{
  cout << "optimisation du gp pour hpars numero " << i << endl;
  auto begin = chrono::steady_clock::now();
  m_vgp_hpars_opti[i].OptimizeGP(myoptfunc_gp, &m_Bounds_hpars_GPs, &hpars_guess, hpars_guess.size());
  auto end = chrono::steady_clock::now();
  hpars_guess = m_vgp_hpars_opti[i].GetPar();
  cout << "par after opt : " << hpars_guess.transpose() << endl;
  cout << "temps pour optimisation : " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " s" << endl;
  return hpars_guess;
}

void DensityOpt::opti_allgps(VectorXd const &hpars_guess)
{
  if (m_vhpars_pour_gp.empty())
  {
    for (int i = 0; i < m_vgp_hpars_opti.size(); i++)
    {
      m_vhpars_pour_gp.push_back(hpars_guess);
    }
  }
  if (!m_vhpars_pour_gp.size() == m_vgp_hpars_opti.size())
  {
    cerr << "problem in vhpars size" << endl;
  }
  for (int i = 0; i < m_vgp_hpars_opti.size(); i++)
  {
    VectorXd h = opti_1gp(i, m_vhpars_pour_gp[i]);
    m_vhpars_pour_gp[i] = h;
  }
}

void DensityOpt::update_hGPs_noPCA(vector<VectorXd> const &new_thetas, vector<VectorXd> const &new_hpars, double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), MatrixXd const &Bounds_hpars_GPs, VectorXd const &Hpars_guess_GPs)
{
  //on fait une optimisation fine sur les new thetas, puis on les rajoute aux hGPs. On optimise les hyperparamètres avec le tout.
  cout << "updating hGPs with " << new_thetas.size() << "new points..." << endl;
  for (int i = 0; i < new_thetas.size(); i++)
  {
    AUGDATA dat;
    dat.SetX(new_thetas[i]), dat.SetValue(new_hpars[i]);
    m_hpars_opti.push_back(dat);
  }
  cout << "new number of points for hGPs : " << m_hpars_opti.size() << endl;
  BuildHGPs_noPCA(Kernel_GP, Bounds_hpars_GPs, Hpars_guess_GPs);
}

std::vector<Eigen::VectorXd> DensityOpt::update_hGPs_noPCA(vector<VectorXd> const &new_thetas, double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), MatrixXd const &Bounds_hpars_GPs, VectorXd const &Hpars_guess_GPs, double max_time)
{
  //on fait une optimisation fine sur les new thetas, puis on les rajoute aux hGPs. On optimise les hyperparamètres avec le tout.
  cout << "updating hGPs with " << new_thetas.size() << "new points..." << endl;
  vector<VectorXd> new_hpars(new_thetas.size());
  VectorXd hpars_guess = 0.5 * (m_lb_hpars + m_ub_hpars);
  for (int i = 0; i < new_thetas.size(); i++)
  {
    new_hpars[i] = HparsOpt(new_thetas[i], hpars_guess, max_time);
  }
  for (int i = 0; i < new_thetas.size(); i++)
  {
    AUGDATA dat;
    dat.SetX(new_thetas[i]), dat.SetValue(new_hpars[i]);
    m_hpars_opti.push_back(dat);
  }
  cout << "new number of points for hGPs : " << m_hpars_opti.size() << endl;
  BuildHGPs_noPCA(Kernel_GP, Bounds_hpars_GPs, Hpars_guess_GPs);
  return new_hpars;
}

double DensityOpt::myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  /* This is the function you optimize for defining the GP */
  GP *proc = (GP *)data;       //Pointer to the GP
  Eigen::VectorXd p(x.size()); //Parameters to be optimized
  for (int i = 0; i < (int)x.size(); i++)
    p(i) = x[i];                 //Setting the proposed value of the parameters
  double value = proc->SetGP(p); //Evaluate the function
  if (!grad.empty())
  { //Cannot compute gradient : stop!
    std::cout << "Asking for gradient, I stop !" << std::endl;
    exit(1);
  }
  return value;
};

void DensityOpt::WritehGPs(string const &filename) const
{
  //écriture des hyperparamètres optimaux des GPs
  //écrit pour 3 GPs avec 7 hpars chacun.
  ofstream ofile(filename);
  for (int i = 0; i < m_vhpars_pour_gp.size(); i++)
  {
    for (int j = 0; j < m_vhpars_pour_gp[0].size(); j++)
    {
      ofile << m_vhpars_pour_gp[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

void DensityOpt::ReadhGPs(string const &filename)
{
  //ne marche pas avec priormean
  ifstream ifile(filename);
  if (ifile)
  {
    string line;
    while (getline(ifile, line))
    {
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)), istream_iterator<string>());
      VectorXd hpars(7);
      for (int i = 0; i < 7; i++)
      {
        hpars(i) = stod(words[i]);
      }
      m_vhpars_pour_gp.push_back(hpars);
    }
    cout << "number of GPs loaded : " << m_vhpars_pour_gp.size() << endl;
  }
  else
  {
    cerr << "empty file : " << filename << endl;
  }
  if (m_vgp_hpars_opti.empty())
  {
    cerr << "erreur : hGPs opti non initialisés" << endl;
  }
  //on applique les hyperparamètres aux hGPs.
  for (int i = 0; i < m_vgp_hpars_opti.size(); i++)
  {
    m_vgp_hpars_opti[i].SetGP(m_vhpars_pour_gp[i]);
  }
  ifile.close();
}

double DensityOpt::EstimatePredError(VectorXd const &theta) const
{
  //estimateur de l'erreur de prédiction des hGPs en un point theta. Fait à partir de la variance de prédiction des hGPs.
  VectorXd var = EvaluateVarHparOpt(theta);
  VectorXd stds_scale = m_Acoefs.diagonal().array().square();
  return var.cwiseQuotient(stds_scale).array().sum();
}

VectorXd DensityOpt::EvaluateVarHparOpt(VectorXd const &theta) const
{
  //estimateur de l'erreur de prédiction des hGPs en un point theta. Fait à partir de la variance de prédiction des hGPs.
  int nmodes = m_vgp_hpars_opti.size();
  VectorXd vargps(nmodes); //on stocke les variances de prédiction de chaque GP.
  for (int i = 0; i < nmodes; i++)
  {
    vargps(i) = m_vgp_hpars_opti[i].Eval(theta)(1);
  }
  //variances de prédiction pour chaque hyperparamètre
  MatrixXd scaledVP = (m_Acoefs * m_VP).array().square();
  VectorXd Variances = scaledVP * vargps;
  //on renvoie la somme des variances, sans scaling pour le moment car je ne sais pas comment faire.

  return Variances;
}