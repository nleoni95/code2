#include "densities.h"
#include <ctime>
#include <list>
#include <random>
#include <chrono>

using namespace std;
using namespace Eigen;

Eigen::VectorXd VtoVXD(std::vector<double> const &v)
{
  Eigen::VectorXd X(v.size());
  for (int i = 0; i < v.size(); i++)
  {
    X(i) = v[i];
  }
  return X;
}

std::vector<double> VXDtoV(Eigen::VectorXd const &X)
{
  std::vector<double> v(X.size());
  for (int i = 0; i < v.size(); i++)
  {
    v[i] = X(i);
  }
  return v;
}

vector<VectorXd> VdtoVVXd(vector<double> &v)
{
  vector<VectorXd> V(v.size());
  for (int i = 0; i < v.size(); i++)
  {
    VectorXd VXd(1);
    VXd << v[i];
    V[i] = VXd;
  }
  return V;
}

void WriteVector(vector<VectorXd> const &v, ofstream &ofile)
{
  int size = v[0].size();
  for (int i = 0; i < v.size(); i++)
  {
    for (int j = 0; j < size; j++)
    {
      ofile << v[i](j) << " ";
    }
    ofile << endl;
  }
}

void WriteVectors(vector<VectorXd> const &v1, vector<double> const &v2, ofstream &ofile)
{
  if (!(v1.size() == v2.size()))
  {
    cerr << "warning : vecteurs de taille différente" << endl;
  }
  int size = min(v1.size(), v2.size());
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < v1[i].size(); j++)
    {
      ofile << v1[i](j) << " ";
    }
    ofile << v2[i] << endl;
  }
}

void WriteVectors(vector<VectorXd> const &v1, VectorXd const &v2, ofstream &ofile)
{
  if (!(v1.size() == v2.size()))
  {
    cerr << "warning : vecteurs de taille différente" << endl;
  }
  for (int i = 0; i < v1.size(); i++)
  {
    for (int j = 0; j < v1[i].size(); j++)
    {
      ofile << v1[i](j) << " ";
    }
    ofile << v2(i) << endl;
  }
}

void WriteVectors(vector<VectorXd> const &v1, vector<VectorXd> const &v2, ofstream &ofile)
{
  if (!(v1.size() == v2.size()))
  {
    cerr << "warning : vecteurs de taille différente" << endl;
  }
  int size = min(v1.size(), v2.size());
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < v1[i].size(); j++)
    {
      ofile << v1[i](j) << " ";
    }
    for (int j = 0; j < v2[i].size(); j++)
    {
      ofile << v2[i](j) << " ";
    }
    ofile << endl;
  }
}

void WriteVectors(vector<VectorXd> const &v1, vector<VectorXd> const &v2, vector<double> const &v3, ofstream &ofile)
{
  if (!(v1.size() == v2.size()) || !(v1.size() == v3.size()))
  {
    cerr << "warning : vecteurs de taille différente" << endl;
  }
  int size = min(min(v1.size(), v2.size()), v3.size());
  for (int i = 0; i < size; i++)
  {
    for (int j = 0; j < v1[i].size(); j++)
    {
      ofile << v1[i](j) << " ";
    }
    for (int j = 0; j < v2[i].size(); j++)
    {
      ofile << v2[i](j) << " ";
    }

    ofile << v3[i] << " ";

    ofile << endl;
  }
}

vector<VectorXd> ReadVector(string const &filename)
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
      vector<double> values;
      for (int i = 0; i < words.size(); i++)
      {
        values.push_back(stod(words[i]));
      }
      v.push_back(VtoVXD(values));
    }
    cout << "number of lines in the file : " << v.size() << endl;
  }
  else
  {
    cerr << "warning : file " << filename << "not found." << endl;
  }
  return v;
}

double optroutine(nlopt::vfunc optfunc, void *data_ptr, VectorXd &X, VectorXd const &lb_hpars, VectorXd const &ub_hpars, double ftol_rel)
{
  vector<double> x = VXDtoV(X);
  vector<double> lb_hpars_opt = VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt = VXDtoV(ub_hpars);
  nlopt::opt local_opt(nlopt::LN_SBPLX, x.size());
  local_opt.set_max_objective(optfunc, data_ptr);
  local_opt.set_ftol_rel(ftol_rel);
  local_opt.set_lower_bounds(lb_hpars_opt);
  local_opt.set_upper_bounds(ub_hpars_opt);
  double msup;
  local_opt.optimize(x, msup);
  X = VtoVXD(x);
  return msup;
}

double optroutine_withgrad(nlopt::vfunc optfunc, void *data_ptr, VectorXd &X, VectorXd const &lb_hpars, VectorXd const &ub_hpars, double ftol_rel)
{
  vector<double> x = VXDtoV(X);
  vector<double> lb_hpars_opt = VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt = VXDtoV(ub_hpars);
  nlopt::opt local_opt(nlopt::LD_TNEWTON_PRECOND_RESTART, x.size());
  local_opt.set_max_objective(optfunc, data_ptr);
  local_opt.set_ftol_rel(ftol_rel);
  local_opt.set_lower_bounds(lb_hpars_opt);
  local_opt.set_upper_bounds(ub_hpars_opt);
  double msup;
  try
  {
    local_opt.optimize(x, msup); 
  }
  catch (const runtime_error &error)
  {
    cout << " optimization failed. Keeping initial value : " << X.transpose() << endl;
    msup = 0;
  }
  X = VtoVXD(x);
  return msup;
}

double optfuncKOH(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  auto ptp = (tuple<const MatrixXd *, double *,Density *> *)data; 
  auto tp = *ptp;
  const MatrixXd *Residus = get<0>(tp);
  double * logvstyp= get<1>(tp);
  const Density *d = get<2>(tp);
  const vector<VectorXd> *xconv = d->GetXlocations();
  VectorXd hpars = VtoVXD(x);
  vector<double> prob(Residus->cols());
  MatrixXd G = d->Gamma(*xconv, hpars) + pow(d->GetOutputerr(hpars), 2) * MatrixXd::Identity(xconv->size(), xconv->size());
  if (d->GetPresenceInputerr())
  {
    G += pow(d->GetInputerr(hpars), 2) * d->GetDerMatrix();
  }
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
    double f = exp(p + logprior - *logvstyp);
    if (isinf(f))
    {
      cerr << "Error in optfuncKOH. Try increasing the value of logvstyp. Try the value : " << p + logprior << " or higher" <<endl;
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
  pair<const VectorXd *, const DensityOpt *> *p = (pair<const VectorXd *, const DensityOpt *> *)data;
  const VectorXd *obs = p->first;
  const DensityOpt *d = p->second;
  VectorXd hpars = VtoVXD(x);
  const vector<VectorXd> *xconv = d->GetXlocations();
  MatrixXd G = d->Gamma(*xconv, hpars) + pow(d->GetOutputerr(hpars), 2) * MatrixXd::Identity(xconv->size(), xconv->size());
  if (d->GetPresenceInputerr())
  {
    G += pow(d->GetInputerr(hpars), 2) * d->GetDerMatrix();
  }
  LDLT<MatrixXd> ldlt(G);
  double ll = d->loglikelihood_fast(*obs, ldlt);
  double lp = d->EvaluateLogPHpars(hpars);
  return ll + lp;
};

double optfuncOpt_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data)
{

  pair<const VectorXd *, const DensityOpt *> *p = (pair<const VectorXd *, const DensityOpt *> *)data;
  const VectorXd *obs = p->first; 
  const DensityOpt *d = p->second;
  VectorXd hpars = VtoVXD(x);
  VectorXd obsmodif = *obs;
  const vector<VectorXd> *xconv = d->GetXlocations();
  MatrixXd G = d->Gamma(*xconv, hpars) + pow(d->GetOutputerr(hpars), 2) * MatrixXd::Identity(xconv->size(), xconv->size());
  if (d->GetPresenceInputerr())
  {
    G += pow(d->GetInputerr(hpars), 2) * d->GetDerMatrix();
  }
  LDLT<MatrixXd> ldlt(G);
  double ll = d->loglikelihood_fast(obsmodif, ldlt);
  double lp = d->EvaluateLogPHpars(hpars);
  auto kernel_gradients = d->GetZKernelGrads();
  auto logprior_gradients = d->GetLogpriorHparsGrads();
  if (!(grad.size() == 0))
  {
    int nd = xconv->size();
    MatrixXd Kinv = ldlt.solve(MatrixXd::Identity(nd, nd));
    VectorXd alpha = Kinv * obsmodif;
    MatrixXd aat = alpha * alpha.transpose();
    for (int n = 0; n < hpars.size(); n++)
    {
      MatrixXd DG = MatrixXd::Zero(nd, nd);
      for (int i = 0; i < nd; i++)
      {
        for (int j = i; j < nd; j++)
        {
          DG(i, j) = kernel_gradients((*xconv)[i], (*xconv)[j], hpars, n);
          if (i != j)
          {
            DG(j, i) = DG(i, j);
          }
        }
      }
      grad[n] = 0.5 * ((aat - Kinv) * DG).trace() + logprior_gradients(hpars, n);
    }
  }
  return ll + lp;
};

double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  GP *proc = (GP *)data;
  Eigen::VectorXd p(x.size());
  for (int i = 0; i < x.size(); i++)
  {
    p(i) = x[i];
  }
  return -1 * proc->SetGP(p);
};

double myoptfunc_gp_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
  GP *gp = (GP *) data;
  Eigen::VectorXd p(x.size());
  for (int i = 0; i < x.size(); i++)
  {
    p(i) = x[i];
  }
  double value = -1 * gp->SetGP(p); //nécessaire d'appeler SetGP avant DerivLL
  Eigen::VectorXd Grad = -1 * gp->DerivLL();
  for (int i = 0; i < Grad.size(); i++)
  {
    grad[i] = Grad(i);
  }
  return value;
};

void OptimizeGPBis(GP &gp, VectorXd &guess, VectorXd const &lb, VectorXd const &ub, double ftol)
{
  auto p = &gp;
  optroutine_withgrad(myoptfunc_gp_withgrad, p, guess, lb, ub, ftol);
  cout << "val critère : " << -1 * gp.SetGP(guess) << endl;
  gp.SetGP(guess);
};

void Run_Burn_Phase_MCMC(int nburn, MatrixXd &COV_init, VectorXd &Xcurrento, function<double(vector<VectorXd>, VectorXd const &)> const &compute_score, function<vector<VectorXd>(VectorXd const &)> const &get_hpars, function<bool(VectorXd)> const &in_bounds, default_random_engine &generator)
{
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
  cout << "theta : " << Xcurrent.transpose() << endl;
  cout << "hpars : " << hparscurrent[0].transpose() << endl;
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
      if (fcandidate > fcurrent || fcandidate - fcurrent > log(distU(generator)))
      {
        naccept++;
        Xcurrent = Xcandidate;
        fcurrent = fcandidate;
        hparscurrent = hparscandidate;
      }
    }
    acc_means += Xcurrent;
    acc_var += Xcurrent * Xcurrent.transpose();
  }
  double acc_rate = (double)(naccept) / (double)(nburn);
  MatrixXd CovProp = (pow(2.38, 2) / (double)(dim_mcmc)) * (acc_var / (nburn - 1) - acc_means * acc_means.transpose() / pow(1.0 * nburn, 2));
  auto end = chrono::steady_clock::now();
  cout << "burn phase over. "
       << " time : " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms, accept rate : " << 100 * acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl
       << CovProp << endl;
  Xcurrento = Xcurrent;
  COV_init = CovProp;
}

vector<VectorXd> Run_MCMC(int nsteps, VectorXd &Xinit, MatrixXd COV_init, function<double(vector<VectorXd>, VectorXd const &)> const &compute_score, function<vector<VectorXd>(VectorXd const &)> const &get_hpars, function<bool(VectorXd)> const &in_bounds, default_random_engine &generator)
{
  /* Metropolis-Hastings algorithm with burn phase. Returns all visited steps (not including the burn phase). */
  cout << "starting MCMC with " << nsteps << " steps." << endl;
  int dim_mcmc = Xinit.size();
  vector<VectorXd> allsamples;
  uniform_real_distribution<double> distU(0, 1);
  normal_distribution<double> distN(0, 1);
  Run_Burn_Phase_MCMC(0.1 * nsteps, COV_init, Xinit, compute_score, get_hpars, in_bounds, generator);
  MatrixXd sqrtCOV = COV_init.llt().matrixL();
  vector<VectorXd> hparsstart = get_hpars(Xinit);
  double finit = compute_score(hparsstart, Xinit);
  VectorXd Xcurrent = Xinit;
  double fcurrent = finit;
  vector<VectorXd> hparscurrent = hparsstart;
  int naccept = 0;
  int noob = 0;
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
    else
    {
      noob++;
    }
    allsamples.push_back(Xcurrent);
  }
  auto end = chrono::steady_clock::now();
  double acc_rate = (double)(naccept) / (double)(nsteps);
  double oob_rate = (double)(noob) / (double)(nsteps);
  cout << "MCMC phase over. time : " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms, accept rate : " << 100 * acc_rate << " pct, oob rate :" << 100 * oob_rate << endl;
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

double Selfcor_diagnosis(vector<VectorXd> const &samples, int nstepsmax, double proportion, string const &filename)
{
  if ((proportion > 1) || (proportion < 0))
  {
    cerr << "error : pick a proportion of samples between 0 and 1" << endl;
    return 0;
  }
  int dim_mcmc = samples[0].size();
  int nselected = proportion * samples.size();
  cout << "computing self-correlation on " << nselected << " samples." << endl;
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
  cout << "maximum of all lengths: " << integ.maxCoeff() << endl;
  return integ.maxCoeff();
}

DoE::DoE(){

};

DoE::DoE(VectorXd const &lb, VectorXd const &ub, int n) : m_lb_pars(lb), m_ub_pars(ub)
{
  /*Construction d'un DoE en Grid Sampling (points réguliers) */
  int dim = lb.size();
  int npts = pow(n, dim);
  VectorXd theta_courant(dim);
  VectorXd ind_courant(dim);
  for (int i = 0; i < npts; i++)
  {
    ind_courant = Multiindex(i, n, dim);
    for (int j = 0; j < dim; j++)
    {
      theta_courant(j) = m_lb_pars(j) + (ind_courant(j) + 0.5) * (m_ub_pars(j) - m_lb_pars(j)) / double(n);
    }
    m_grid.push_back(theta_courant);
  }
};

DoE::DoE(VectorXd const &lb, VectorXd const &ub, int ntotal, default_random_engine &generator) : m_lb_pars(lb), m_ub_pars(ub)
{
  /*Construction d'un DoE en LHS */
  uniform_real_distribution<double> distU(0, 1);
  int dim = lb.size();
  std::vector<VectorXd> perm(dim);
  for (int i = 0; i < dim; i++)
  {
    perm[i] = Randperm(ntotal, generator);
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
  /*Construction d'un DoE en QMC avec suite de Halton */
  int dim = lb.size();
  MatrixXd H = halton_sequence(first_element, first_element + npoints - 1, dim);
  for (int i = 0; i < npoints; i++)
  {
    VectorXd theta = lb + (ub - lb).cwiseProduct(H.col(i));
    m_grid.push_back(theta);
  }
};

VectorXd DoE::Randperm(int n, default_random_engine &generator)
{
  /* Permutation aléatoire de (1,n) */
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

VectorXd DoE::Multiindex(int const s, int const n, int const d)
{
  VectorXd multiindice(d);
  int indloc;
  int remainder = s;
  for (int pp = d - 1; pp > -1; pp--)
  {
    indloc = (int)remainder % n;
    multiindice(pp) = indloc;
    remainder = (remainder - indloc) / n;
  }
  return multiindice;
};

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
  m_outputerr = d.m_outputerr;
  m_indexinputerr = d.m_indexinputerr;
  m_indexoutputerr = d.m_indexoutputerr;
  m_derivatives_obs = d.m_derivatives_obs;
  m_presence_inputerr = d.m_presence_inputerr;
  m_presence_outputerr = d.m_presence_outputerr;
};

Density::Density(DoE const &g)
{
  m_Grid = g;
  m_lb_pars = m_Grid.m_lb_pars;
  m_ub_pars = m_Grid.m_ub_pars;
  m_dim_pars = m_lb_pars.size();
}

void Density::SetNewDoE(DoE const &g)
{
  m_Grid = g;
  m_lb_pars = m_Grid.m_lb_pars;
  m_ub_pars = m_Grid.m_ub_pars;
  m_dim_pars = m_lb_pars.size();
}

void Density::SetObservations(vector<VectorXd> const &Xlocations, VectorXd const &observations)
{
  if (!(Xlocations.size() == observations.size()))
  {
    cerr << "erreur : mauvais nombre d'observations" << endl;
  }
  else
  {
    m_Xlocations = Xlocations;
    m_observations = observations;
  }
}

void Density::SetFixedInputerr(double logstdinputerr, VectorXd model_derivatives_at_obs)
{
  m_presence_inputerr = 1;
  m_inputerr = logstdinputerr;
  VectorXd derivativessquared = model_derivatives_at_obs.array().square();
  m_derivatives_obs = derivativessquared.asDiagonal();
}
void Density::SetLearnedInputerr(int index, VectorXd model_derivatives_at_obs)
{
  m_presence_inputerr = 2;
  m_indexinputerr = index;
  VectorXd derivativessquared = model_derivatives_at_obs.array().square();
  m_derivatives_obs = derivativessquared.asDiagonal();
}

void Density::SetFixedOutputerr(double logstdoutputerr)
{
  m_presence_outputerr = 1;
  m_outputerr = logstdoutputerr;
}

void Density::SetLearnedOutputerr(int index)
{
  m_presence_outputerr = 2;
  m_indexoutputerr = index;
}

double Density::GetOutputerr(VectorXd const &hpars) const
{
  if (m_presence_outputerr == 0)
  {return 0;
  }
  else if (m_presence_outputerr == 1)
    {
    return exp(m_outputerr);
  }
  else if (m_presence_outputerr == 2)
  {
    return exp(hpars(m_indexoutputerr));
  }
  else{
    cerr << "erreur : m_presence_outputerr = " << m_presence_outputerr << endl;
  }
  return 0;
}

double Density::GetInputerr(VectorXd const &hpars) const
{
  if (m_presence_inputerr == 0)
  {return 0;
  }
  else if (m_presence_inputerr == 1)
    {
    return exp(m_inputerr);
  }
  else if (m_presence_inputerr == 2)
  {
    return exp(hpars(m_indexinputerr));
  }
  else{
    cerr << "erreur : m_presence_inputerr = " << m_presence_inputerr << endl;
  }
  return 0;
}

Eigen::MatrixXd Density::Gamma(vector<VectorXd> const &locs, Eigen::VectorXd const &hpar) const
{
  int nd = locs.size();
  Eigen::MatrixXd A = MatrixXd::Zero(nd, nd);
  for (int i = 0; i < nd; i++)
  {
    for (int j = i; j < nd; j++)
    {
      A(i, j) = m_Kernel(locs[i], locs[j], hpar);
      if (i != j)
      {
        A(j, i) = A(i, j);
      }
    }
  }
  return A;
}

double Density::loglikelihood_fast(VectorXd const &obs, LDLT<MatrixXd> const &ldlt) const
{
  VectorXd Alpha = ldlt.solve(obs);
  return -0.5 * obs.dot(Alpha) - 0.5 * (ldlt.vectorD().array().log()).sum() - 0.5 * obs.size() * log(2 * 3.1415);
}

double Density::loglikelihood_theta_fast(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars, Eigen::LDLT<Eigen::MatrixXd> const &ldlt) const
{
  VectorXd obs = m_observations - m_model(m_Xlocations, theta) - m_priormean(m_Xlocations, hpars);
  return loglikelihood_fast(obs, ldlt);
}

double Density::loglikelihood_theta(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar) const
{
  MatrixXd G = Gamma(m_Xlocations, hpar) + pow(GetOutputerr(hpar), 2) * MatrixXd::Identity(m_Xlocations.size(), m_Xlocations.size());
  if (m_presence_inputerr>0)
  {
    G += pow(GetInputerr(hpar), 2) * m_derivatives_obs;
  }
  LDLT<MatrixXd> ldlt(G);
  return loglikelihood_theta_fast(theta, hpar, ldlt);
}

VectorXd Density::HparsKOH(VectorXd const &hpars_guess, double logvs_typ ,double ftol_rel) const
{
  auto begin = chrono::steady_clock::now();
  VectorXd guess = hpars_guess;
  MatrixXd Residustheta(m_observations.size(), m_Grid.m_grid.size());
  for (int i = 0; i < m_Grid.m_grid.size(); i++)
  {
    VectorXd theta = m_Grid.m_grid[i];
    Residustheta.col(i) = -m_model(m_Xlocations, theta) + m_observations;
  }
  auto tp = make_tuple(&Residustheta,&logvs_typ, this);
  optroutine(optfuncKOH, &tp, guess, m_lb_hpars, m_ub_hpars, ftol_rel);
  auto end = chrono::steady_clock::now();
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

VectorXd Density::meanF(vector<VectorXd> const &X) const
{
  VectorXd mpf = VectorXd::Zero(X.size());
  for (VectorXd const &s : m_samples)
  {
    mpf += m_model(X, s);
  }
  return mpf / m_samples.size();
}

MatrixXd Density::varF(vector<VectorXd> const &X) const
{
  VectorXd mpf = meanF(X);
  MatrixXd SecondMoment = MatrixXd::Zero(mpf.size(), mpf.size());
  for_each(m_samples.begin(), m_samples.end(), [&SecondMoment, &X, this](VectorXd const &theta) mutable
           {
             VectorXd Pred = this->m_model(X, theta);
             SecondMoment += Pred * Pred.transpose(); });
  MatrixXd var = SecondMoment / m_samples.size() - mpf * mpf.transpose();
  return var;
}

VectorXd Density::meanZCondTheta(vector<VectorXd> const &X, VectorXd const &theta, VectorXd const &hpars) const
{
  VectorXd y(X.size());
  y = m_observations - m_model(m_Xlocations, theta) - m_priormean(m_Xlocations, hpars);
  MatrixXd G = Gamma(m_Xlocations, hpars) + pow(GetOutputerr(hpars), 2) * MatrixXd::Identity(m_Xlocations.size(), m_Xlocations.size());
  if (m_presence_inputerr>0)
  {
    G += pow(GetInputerr(hpars), 2) * m_derivatives_obs;
  }
  LDLT<MatrixXd> ldlt(G);
  MatrixXd K = MatrixXd::Zero(X.size(), m_Xlocations.size());
  for (int i = 0; i < K.rows(); i++)
  {
    for (int j = 0; j < K.cols(); j++)
    {
      K(i, j) = m_Kernel(X[i], m_Xlocations[j], hpars);
    }
  }
  VectorXd predmean = m_priormean(X, hpars) + K * ldlt.solve(y);
  return predmean;
}

MatrixXd Density::varZCondTheta(vector<VectorXd> const &X, VectorXd const &theta, VectorXd const &hpars) const
{
  MatrixXd G = Gamma(m_Xlocations, hpars) + pow(GetOutputerr(hpars), 2) * MatrixXd::Identity(m_Xlocations.size(), m_Xlocations.size());
  if (m_presence_inputerr>0)
  {
    G += pow(GetInputerr(hpars), 2) * m_derivatives_obs;
  }
  LDLT<MatrixXd> ldlt(G);
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
  MatrixXd VarPred = Kprior - K * ldlt.solve(K.transpose());
  return VarPred;
}

MatrixXd Density::PredFZ(vector<VectorXd> const &X) const
{
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
  default_random_engine generator;
  int seed = time(0);
  generator.seed(seed);
  uniform_int_distribution<int> U(0, m_samples.size() - 1);
  MatrixXd Predictions = PredFZ(X); 
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
  VectorXd quant2p5(meanf.rows());
  VectorXd quant97p5(meanf.rows());
  for (int i = 0; i < meanf.rows(); i++)
  {
    VectorXd R = fvalues.row(i);
    quant2p5(i) = R(int(0.025 * R.size()));
    quant97p5(i) = R(int(0.975 * R.size()));
  }

  ofstream ofile(filename);
   for (int i = 0; i < Predictions.rows(); i++)
  {
    for (int j = 0; j < X[i].size(); j++)
    {
      ofile << X[i](j) << " ";
    }
    ofile << Predictions(i, 0) << " " << sqrt(Predictions(i, 1) + Predictions(i, 2)) << " " << sqrt(Predictions(i, 2)) << " " << meanf(i) << " " << quant2p5(i) << " " << quant97p5(i) << endl;
  }
  ofile.close();
}

void Density::WriteSamplesFandZ(vector<VectorXd> const &X, string const &filenameF, string const &filenameZ,int nsamples) const
{
  cout << "drawing " << nsamples << " samples of f and z" << endl;
  default_random_engine generator;
  int seed = time(0);
  generator.seed(seed);
  uniform_int_distribution<int> U(0, m_samples.size() - 1);
  vector<VectorXd> fs(nsamples);
  vector<VectorXd> zs(nsamples);
  for (int i = 0; i < nsamples; i++)
  {
    int r = U(generator);
    fs[i] = m_model(X, m_samples[r]);
    zs[i] = DrawZCondTheta(X, m_samples[r], m_hparsofsamples[r], generator);
  }
  ofstream ofile(filenameF);
  for (int i = 0; i < nsamples; i++)
  {
    for (int j = 0; j < X.size(); j++)
    {
      ofile << fs[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
  ofile.open(filenameZ);
  for (int i = 0; i < nsamples; i++)
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

VectorXd DensityOpt::HparsOpt(VectorXd const &theta, VectorXd hpars_guess, double ftol_rel) const
{
  VectorXd obsmtheta = m_observations - m_model(m_Xlocations, theta);
  auto p = make_pair(&obsmtheta, this);
  int fin = optroutine(optfuncOpt_nograd, &p, hpars_guess, m_lb_hpars, m_ub_hpars, ftol_rel);
  return hpars_guess;
}

VectorXd DensityOpt::HparsOpt_withgrad(VectorXd const &theta, VectorXd hpars_guess, double ftol_rel) const
{
  VectorXd obsmtheta = m_observations - m_model(m_Xlocations, theta);
  auto p = make_pair(&obsmtheta, this);
  int fin = optroutine_withgrad(optfuncOpt_withgrad, &p, hpars_guess, m_lb_hpars, m_ub_hpars, ftol_rel);
  return hpars_guess;
}

VectorXd DensityOpt::EvaluateHparOpt(VectorXd const &theta) const
{
  VectorXd meansgps(m_hGPs.size());
  for (int i = 0; i < m_hGPs.size(); i++)
  {
    meansgps(i) = m_hGPs[i].EvalMean(theta);
  }
  VectorXd scaledmeans = meansgps;
  return scaledmeans;
}

vector<vector<VectorXd>> DensityOpt::SampleHparsOpt(vector<VectorXd> const &thetas, int nsamples, default_random_engine &generator) const
{
  int nthetas = thetas.size();
  vector<vector<VectorXd>> res;
  vector<MatrixXd> res_malformatte; 
  for (int i = 0; i < m_hGPs.size(); i++)
  {
    MatrixXd M = m_hGPs[i].SampleGPDirect(thetas, nsamples, generator); 
    res_malformatte.push_back(M);
  }
  for (int i = 0; i < nthetas; i++)
  {
    vector<VectorXd> v;
    for (int j = 0; j < nsamples; j++)
    {
      VectorXd V(m_hGPs.size());
      for (int k = 0; k < m_hGPs.size(); k++)
      {
        V(k) = res_malformatte[k](i, j);
      }
      v.push_back(V);
    }
    res.push_back(v);
  }
  return res;
}

void DensityOpt::BuildHGPs(std::vector<Eigen::VectorXd> const &thetas, std::vector<Eigen::VectorXd> const &hpars_optimaux, double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), double (*DKernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int))
{
  m_hGPs.clear();
  vector<vector<DATA>> vdata;
  for (int i = 0; i < m_dim_hpars; i++)
  {
    vector<DATA> data_hgp;
    for (int j = 0; j < thetas.size(); j++)
    {
      DATA d;
      d.SetValue(hpars_optimaux[j](i));
      d.SetX(thetas[j]);
      data_hgp.push_back(d);
    }
    vdata.push_back(data_hgp);
  }
  cout << "construction of " << m_dim_hpars << " hGPs with " << thetas.size() << " points." << endl;
  vector<GP> vgp;
  for (int i = 0; i < m_dim_hpars; i++)
  {
    GP gp(Kernel_GP);
    gp.SetData(vdata[i]);
    gp.SetDKernel(DKernel_GP);
    vgp.push_back(gp);
  }
  m_hGPs = vgp;
}

void DensityOpt::OptimizeHGPs(Eigen::MatrixXd Bounds_hpars_GPs, vector<VectorXd> Hpars_guess_GPs, double ftol)
{
  for (int i = 0; i < m_hGPs.size(); i++)
  {
    m_hGPs[i].SetGP(Hpars_guess_GPs[i]);
    cout << "optimization of HGP number " << i << endl;
    auto begin = chrono::steady_clock::now();
    OptimizeGPBis(m_hGPs[i], Hpars_guess_GPs[i], Bounds_hpars_GPs.row(0), Bounds_hpars_GPs.row(1), ftol);
    auto end = chrono::steady_clock::now();
    cout << "optimisation over in " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds." << endl;
    cout << "result : " << m_hGPs[i].GetPar().transpose() << endl;
  }
}

void DensityOpt::SetHGPs(vector<VectorXd> Hpars_guess_GPs)
{
  if (Hpars_guess_GPs.size() != m_hGPs.size())
  {
    cerr << "erreur size sethgps : " << Hpars_guess_GPs.size() << ", " << m_hGPs.size() << endl;
  }
  for (int i = 0; i < m_hGPs.size(); i++)
  {
    m_hGPs[i].SetGP(Hpars_guess_GPs[i]);
  }
}
