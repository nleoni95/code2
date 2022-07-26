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

void WriteVectors(vector<VectorXd> const &v1, vector<VectorXd> const &v2, string const &filename)
{
  //write two vectors of same size to a file
  if (!(v1.size() == v2.size()))
  {
    cerr << "warning : vecteurs de taille différente" << endl;
  }
  int size = min(v1.size(), v2.size());
  ofstream ofile(filename);
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
  ofile.close();
}

void WriteVectors(vector<VectorXd> const &v1, vector<VectorXd> const &v2, vector<VectorXd> const &v3, string const &filename)
{
  //write two vectors of same size to a file
  if (!(v1.size() == v2.size()) || !(v1.size() == v3.size()))
  {
    cerr << "warning : vecteurs de taille différente" << endl;
  }
  int size = min(min(v1.size(), v2.size()), v3.size());
  ofstream ofile(filename);
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
    for (int j = 0; j < v3[i].size(); j++)
    {
      ofile << v3[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

vector<VectorXd> ReadVector(string const &filename)
{
  //lecture d'un fichier de donénes espacées d'une espace. renvoie un vector de taille le nombre de lignes du fichier, chaque VectorXd contient toutes les entrées d'une ligne.
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
      for(int i=0;i<words.size();i++){
        values.push_back(stod(words[i]));
      }
      v.push_back(VtoVXD(values));
    }
    cout << "number of lines in the file : " << v.size() << endl;
  }
  else{
    cerr << "warning : file " << filename << "not found." << endl;
  }
  return v;
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
  double msup;           /* the maximum objective value, upon return */
  opt.optimize(x, msup); //messages d'arrêt
  //on relance une opti locale à partir du max. trouvé.
  local_opt.set_ftol_rel(ftol_fin);
  local_opt.set_xtol_rel(xtol_fin);
  local_opt.optimize(x, msup); //messages d'arrêt
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
  double ftol_large = 1e-5;
  double xtol_large = 1e-3;
  double ftol_fin = 1e-15;
  double xtol_fin = 1e-7;
  // 1 optimiseur local et un global.
  nlopt::opt local_opt(nlopt::LD_MMA, x.size());
  local_opt.set_max_objective(optfunc, data_ptr);
  local_opt.set_ftol_rel(ftol_large);
  local_opt.set_xtol_rel(xtol_large);
  local_opt.set_lower_bounds(lb_hpars_opt);
  local_opt.set_upper_bounds(ub_hpars_opt);
  local_opt.set_maxtime(max_time);
  nlopt::opt opt(nlopt::G_MLSL_LDS, x.size());
  opt.set_max_objective(optfunc, data_ptr);
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxtime(max_time); //limite de temps : 60 sec.
  opt.set_local_optimizer(local_opt);
  double msup;                     /* the maximum objective value, upon return */
  int fin = opt.optimize(x, msup); //messages d'arrêt
  //cout << "max. après MLSL : " << VtoVXD(x).transpose() << endl;
  //cout << "arrêt : " << fin << endl;
  //on relance une opti locale à partir du max. trouvé.
  local_opt.set_ftol_rel(ftol_fin);
  local_opt.set_xtol_rel(xtol_fin);
  fin = local_opt.optimize(x, msup); //messages d'arrêt
  //cout << "max. après opt locale : " << VtoVXD(x).transpose() << endl;
  //cout << "arrêt : " << fin << endl;
  //vector<double> grad=x;
  //optfunc(x,grad,data_ptr);
  //cout << "gradient en ce point : " << VtoVXD(grad).transpose() << endl << endl;
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
  /* fonction à optimiser pour trouver les hpars optimaux.*/
  //écrite pour 2 hyperparamètres.
  //cast du null pointer
  pair<const VectorXd *, const DensityOpt *> *p = (pair<const VectorXd *, const DensityOpt *> *)data;
  const VectorXd *obs = p->first; //contient déjà yobs-ftheta.
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
  //compute gradient matrices
  auto kernel_gradients = d->GetKernelGrads();
  auto logprior_gradients = d->GetLogpriorGrads();
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
  /* This is the function you optimize for defining the GP */
  GP *proc = (GP *) data;       //Pointer to the GP
  Eigen::VectorXd p(x.size()); //Parameters to be optimized
  for (int i = 0; i < (int)x.size(); i++)
    p(i) = x[i];                 //Setting the proposed value of the parameters
  double value = -1*proc->SetGP(p); //Evaluate the function. Attention ! moi je maximise.
  if (!grad.empty())
  { //Cannot compute gradient : stop!
    std::cout << "Asking for gradient, I stop !" << std::endl;
    exit(1);
  }
  return value;
};

void OptimizeGPBis(GP & gp, VectorXd &guess, VectorXd const & lb,VectorXd const & ub, double time){
	//version réécrite Nicolas
	// std::cout << "Optimize Gaussian process for " << np << " hyperparameters\n";
  optroutine(myoptfunc_gp,&gp,guess,lb,ub,time);
  cout << "val critère : " << -1*gp.SetGP(guess) << endl;
	gp.SetGP(guess);
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

vector<VectorXd> Run_MCMC(int nsteps, VectorXd &Xinit, MatrixXd COV_init, function<double(vector<VectorXd>, VectorXd const &)> const &compute_score, function<vector<VectorXd>(VectorXd const &)> const &get_hpars, function<bool(VectorXd)> const &in_bounds, default_random_engine &generator)
{
  //Metropolis-Hastings algorithm with burn phase. Returns all visited steps (not including the burn phase).
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
  int noob=0;
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
        else{
      noob++;
    }
    allsamples.push_back(Xcurrent);
  }
  auto end = chrono::steady_clock::now();
  double acc_rate = (double)(naccept) / (double)(nsteps);
  double oob_rate = (double)(noob) / (double)(nsteps);
  cout << "MCMC phase over. time : " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " ms, accept rate : " << 100 * acc_rate << " pct, oob rate :" << 100*oob_rate << endl;
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
  m_outputerr = d.m_outputerr;
  m_indexinputerr = d.m_indexinputerr;
  m_indexoutputerr = d.m_indexoutputerr;
  m_derivatives_obs = d.m_derivatives_obs;
  m_presence_inputerr = d.m_presence_inputerr;
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

void Density::SetInputerr(bool b, double value, int index, VectorXd derivatives_at_obs)
{
  // call if the input error is present, and choose if it is learned or fixed.
  // if b is true, it is learned (then value is useless)
  // if b is false, it is fixed (then index is useless)
  // value : standard deviation of the error if it is fixed.
  // index : position of the inputerr in the vector of hyperparameters if it is learned.
  // if value is 0 : then the two vectorXd arguments are useless.
  // if value is nonnegative : the two vectorXd must contain the derivatives of the true process at the observation points and the predicted points, in the same order as them. In particular, check that they have the same size.
  if (b)
  {
    //inputerr is learned
    m_presence_inputerr = true;
    m_inputerr = -1;
    m_indexinputerr = index;
    VectorXd derivativessquared = derivatives_at_obs.array().square();
    m_derivatives_obs = derivativessquared.asDiagonal();
  }
  else
  {
    //input error is fixed in the problem
    m_presence_inputerr = true;
    m_inputerr = value;
    VectorXd derivativessquared = derivatives_at_obs.array().square();
    m_derivatives_obs = derivativessquared.asDiagonal();
  }
}

void Density::SetOutputerr(bool b, double value, int index)
{
  //choose if the output error is present, and if it is learned or fixed.
  // if b is true, it is learned (then value is useless)
  // if b is false, it is fixed (then index is useless)
  // value : standard deviation of the error if it is fixed.
  // index : position of the inputerr in the vector of hyperparameters if it is learned.
  if (b)
  {
    m_outputerr = -1;
    m_indexoutputerr = index;
  }
  else
  {
    if (value < 0)
    {
      cerr << "error : negative output error" << endl;
      exit(0);
    }
    m_outputerr = value;
  }
}

double Density::GetOutputerr(VectorXd const &hpars) const
{
  if (m_outputerr == -1)
  {
    //outputerr is learned
    return hpars(m_indexoutputerr);
  }
  else
  {
    return m_outputerr;
  }
}

double Density::GetInputerr(VectorXd const &hpars) const
{
  if (m_inputerr == -1)
  {
    //inputerr is learned
    return hpars(m_indexinputerr);
  }
  else
  {
    return m_inputerr;
  }
}

Eigen::MatrixXd Density::Gamma(vector<VectorXd> const &locs, Eigen::VectorXd const &hpar) const
{
  // Non-noisy correlation matrix
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
  MatrixXd G = Gamma(m_Xlocations, hpar) + pow(GetOutputerr(hpar), 2) * MatrixXd::Identity(m_Xlocations.size(), m_Xlocations.size());
  if (m_presence_inputerr)
  {
    G += pow(GetInputerr(hpar), 2) * m_derivatives_obs;
  }
  LDLT<MatrixXd> ldlt(G);
  return loglikelihood_theta_fast(theta, hpar, ldlt);
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
  MatrixXd G = Gamma(m_Xlocations, hpars) + pow(GetOutputerr(hpars), 2) * MatrixXd::Identity(m_Xlocations.size(), m_Xlocations.size());
  if (m_presence_inputerr)
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
  //variance prédictive de z sur Xprofile, pour le i-ème theta de m_samples.
  MatrixXd G = Gamma(m_Xlocations, hpars) + pow(GetOutputerr(hpars), 2) * MatrixXd::Identity(m_Xlocations.size(), m_Xlocations.size());
  if (m_presence_inputerr)
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
  default_random_engine generator;
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
    for (int j = 0; j < X[i].size(); j++)
    {
      ofile << X[i](j) << " ";
    }
    ofile << Predictions(i, 0) << " " << sqrt(Predictions(i, 1) + Predictions(i, 2)) << " " << sqrt(Predictions(i, 2)) << " " << meanf(i) << " " << quant2p5(i) << " " << quant97p5(i) << endl;
  }
  ofile.close();
}

void Density::WriteSamplesFandZ(vector<VectorXd> const &X, string const &filenameF, string const &filenameZ) const
{
  //draw samples of f and corresponding samples of z.
  int ndraws = 10;
  cout << "drawing " << ndraws << " samples of f and z" << endl;
  default_random_engine generator;
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
  int fin = optroutine(optfuncOpt_nograd, &p, hpars_guess, m_lb_hpars, m_ub_hpars, max_time);
  return hpars_guess;
}

VectorXd DensityOpt::HparsOpt_withgrad(VectorXd const &theta, VectorXd hpars_guess, double max_time) const
{
  VectorXd obsmtheta = m_observations - m_model(m_Xlocations, theta);
  auto p = make_pair(&obsmtheta, this);
  int fin = optroutine_withgrad(optfuncOpt_withgrad, &p, hpars_guess, m_lb_hpars, m_ub_hpars, max_time);
  return hpars_guess;
}

VectorXd DensityOpt::EvaluateHparOpt(VectorXd const &theta) const
{
  //évaluation de l'hpar optimal par les GPs. On utilise uniquement la prédiction moyenne.
  VectorXd meansgps(m_hGPs.size());
  for (int i = 0; i < m_hGPs.size(); i++)
  {
    meansgps(i) = m_hGPs[i].EvalMean(theta);
  }
  VectorXd scaledmeans = m_scales_hGPs.cwiseProduct(meansgps) + m_means_hGPs;
  return scaledmeans;
}

void DensityOpt::BuildHGPs(std::vector<Eigen::VectorXd> const &thetas, std::vector<Eigen::VectorXd> const &hpars_optimaux, double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &))
{
  m_hGPs.clear();
  if (!(thetas.size() == hpars_optimaux.size()))
  {
    cerr << "error : thetas and optimal hpars with different sizes !" << endl;
    exit(0);
  }
  //compute means and stds to scale the data.
  VectorXd means = VectorXd::Zero(hpars_optimaux[0].size());
  VectorXd stds = VectorXd::Zero(hpars_optimaux[0].size());
  for (int i = 0; i < thetas.size(); i++)
  {
    means += hpars_optimaux[i];
  }
  means /= thetas.size();
  for (int i = 0; i < thetas.size(); i++)
  {
    VectorXd v = (hpars_optimaux[i] - means).array().square();
    stds += v;
  }
  stds = stds.array().sqrt();
  stds /= sqrt(thetas.size());
  //split the data sets into vectors of DATA for GPs.
  //test
  //means=VectorXd::Ones(3);
  //stds=VectorXd::Ones(3);
  vector<vector<DATA>> vdata;
  for (int i = 0; i < m_dim_hpars; i++)
  {
    vector<DATA> data_hgp;
    for (int j = 0; j < thetas.size(); j++)
    {
      DATA d;
      d.SetValue(((hpars_optimaux[j](i) - means(i)) / stds(i)));
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
    vgp.push_back(gp);
  }
  m_hGPs = vgp;
  m_means_hGPs = means;
  m_scales_hGPs = stds;
}

void DensityOpt::OptimizeHGPs(Eigen::MatrixXd Bounds_hpars_GPs, Eigen::VectorXd Hpars_guess_GPs,double time)
{
  for (int i = 0; i < m_hGPs.size(); i++)
  {
    m_hGPs[i].SetGP(Hpars_guess_GPs);
    cout << "optimization of HGP number " << i << endl;
    //cout << "guess : " << Hpars_guess_GPs.transpose() << endl;
    auto begin = chrono::steady_clock::now();
    OptimizeGPBis(m_hGPs[i],Hpars_guess_GPs, Bounds_hpars_GPs.row(0),Bounds_hpars_GPs.row(1),time);
    auto end = chrono::steady_clock::now();
    cout << "optimisation over in " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds." << endl;
    cout << "result : " << m_hGPs[i].GetPar().transpose() << endl;
  }
}

void DensityOpt::OptimizeHGPs(Eigen::MatrixXd Bounds_hpars_GPs, vector<VectorXd> Hpars_guess_GPs,double time)
{
  for (int i = 0; i < m_hGPs.size(); i++)
  {
    m_hGPs[i].SetGP(Hpars_guess_GPs[i]);
    cout << "optimization of HGP number " << i << endl;
    //cout << "guess : " << Hpars_guess_GPs.transpose() << endl;
    auto begin = chrono::steady_clock::now();
    OptimizeGPBis(m_hGPs[i],Hpars_guess_GPs[i], Bounds_hpars_GPs.row(0),Bounds_hpars_GPs.row(1),time);
    auto end = chrono::steady_clock::now();
    cout << "optimisation over in " << chrono::duration_cast<chrono::seconds>(end - begin).count() << " seconds." << endl;
    cout << "result : " << m_hGPs[i].GetPar().transpose() << endl;
  }
}

void DensityOpt::SetHGPs(vector<VectorXd> Hpars_guess_GPs)
{
  for (int i = 0; i < m_hGPs.size(); i++)
  {
    m_hGPs[i].SetGP(Hpars_guess_GPs[i]);
  }
}



double DensityOpt::EstimatePredError(VectorXd const &theta) const
{
  //scaled prediction variances for fair comparison
  auto vhgps=GetHparsHGPs();
  VectorXd vargps(m_hGPs.size());
  for (int i = 0; i < m_hGPs.size(); i++)
  {
    vargps(i) = m_hGPs[i].Eval(theta)(1)+pow(vhgps[i](2),2);
  }
  return vargps.array().sum();
}

VectorXd DensityOpt::EvaluateVarHparOpt(VectorXd const &theta) const
{
  //on suppose que la variance est située à la position 2 du vecteur hyperparamètres.
  auto vhgps=GetHparsHGPs();
  int nmodes = m_hGPs.size();
  VectorXd vargps(nmodes);
  for (int i = 0; i < nmodes; i++)
  {
    vargps(i) = m_hGPs[i].Eval(theta)(1)+pow(vhgps[i](2),2);
  }
  VectorXd stds2 = m_scales_hGPs.array().square();
  return stds2.cwiseProduct(vargps);
}