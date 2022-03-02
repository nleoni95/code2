#include "densities.h"
#include <ctime>
#include <list>
#include <random>
#include <chrono>

using namespace std;
using namespace Eigen;

//utilitaire
Eigen::VectorXd VtoVXD(std::vector<double> const &v){
    //conversion vector double - vectorxd
    Eigen::VectorXd X(v.size()); for(int i=0;i<v.size();i++) {X(i)=v[i];} return X;
    }

std::vector<double> VXDtoV(Eigen::VectorXd const &X){
    //conversion vectorxd - vector double
    std::vector<double> v(X.size()); for(int i=0;i<v.size();i++) {v[i]=X(i);} return v;
    }
std::vector<Eigen::VectorXd> Get_data_locations(std::vector<AUGDATA> const & data){
    //récupération des locations des data
    std::vector<Eigen::VectorXd> v;
    std::transform(data.begin(),data.end(),std::back_inserter(v),[](AUGDATA const &d){return d.GetX();});
    return v;
}

/* Fonctions de la classe DoE*/
DoE::DoE()
{

};

DoE::DoE(VectorXd const &lb, VectorXd const &ub, int n): m_lb_pars(lb),m_ub_pars(ub)
{
  //n correspond au nombre de points par dimension
  //initialisation en grid régulier ayant le même nombre de points par dimension.
  int dim=lb.size();
  int npts=pow(n,dim);
  VectorXd theta_courant(dim);
  VectorXd ind_courant(dim);
  for(int i=0;i<npts;i++){
    ind_courant=indices(i,n,dim);
    for (int j=0;j<dim;j++){
      theta_courant(j)=m_lb_pars(j)+(ind_courant(j)+0.5)*(m_ub_pars(j)-m_lb_pars(j))/double(n);
    }
    m_grid.push_back(theta_courant);
  }
};

DoE::DoE(VectorXd const &lb, VectorXd const &ub, int ntotal, default_random_engine &generator): m_lb_pars(lb),m_ub_pars(ub)
{
  uniform_real_distribution<double> distU(0,1);
  int dim=lb.size();
  //Construction en LHS uniforme.
  // n correspond au nombre de points dans le grid.
  // division de chaque dimension en npoints : on génère m_dimension permutations de {0,npoints-1}.
  std::vector<VectorXd> perm(dim);
  for (int i=0;i<dim;i++){
    perm[i]=Randpert(ntotal,generator);
  }
  // calcul des coordonnées de chaque point par un LHS.
  VectorXd theta_courant(dim);
  for(int i=0;i<ntotal;i++){
    for (int j=0;j<dim;j++){
      theta_courant(j)=lb(j)+(ub(j)-lb(j))*(perm[j](i)+distU(generator))/double(ntotal);
    }
    m_grid.push_back(theta_courant);
  }
};

DoE::DoE(VectorXd const &lb, VectorXd const &ub, int npoints, int first_element): m_lb_pars(lb),m_ub_pars(ub)
{
  int dim=lb.size();
  MatrixXd H=halton_sequence(first_element,first_element+npoints-1,dim);
  for(int i=0;i<npoints;i++){
    VectorXd theta=lb+(ub-lb).cwiseProduct(H.col(i));
    m_grid.push_back(theta);
  }
};

VectorXd DoE::Randpert(int n,default_random_engine & generator)const{
  VectorXd result(n);
  std::uniform_real_distribution<double> distU(0,1);
  for (int i=0;i<n;i++){
    result(i)=i;
  }
  for (int i=n-1;i>0;i--){
    int j=int(floor(distU(generator)*(i+1)));
    double a=result(i);
    result(i)=result(j);
    result(j)=a;
  }
  return result;
};

VectorXd DoE::indices(int const s, int const n, int const d)
{
  //renvoie le multi-indice correspondant à l'indice courant s dans un tableau de dimension d et de taille n dans chaque direction.
  VectorXd multiindice(d);
  int indloc;
  int remainder=s;
  for(int pp=d-1;pp>-1;pp--){
    indloc=(int) remainder % n; //On commence par le coefficient le plus à droite.
    multiindice(pp)=indloc;
    remainder=(remainder-indloc)/n;
  }
  return multiindice;
};

void DoE::WriteGrid(string const & filename) const{
  //écriture des thetas juste pour être sûr.
  ofstream ofile(filename);
  for (VectorXd const & v:m_grid){
    for(int i=0;i<v.size();i++){
      ofile << v(i) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}


Density::Density()
{

};

Density::Density(Density const &d)
{
    //constructeur par copie
    m_model=d.m_model;
    m_priormean=d.m_priormean;
    m_logpriorhpars=d.m_logpriorhpars;
    m_logpriorpars=d.m_logpriorpars;
    m_Kernel=d.m_Kernel;
    m_DKernel1=d.m_DKernel1;
    m_DKernel2=d.m_DKernel2;
    m_DKernel3=d.m_DKernel3;
    m_dim_hpars=d.m_dim_hpars;
    m_dim_pars=d.m_dim_pars;
    m_lb_hpars=d.m_lb_hpars;
    m_ub_hpars=d.m_ub_hpars;
    m_lb_pars=d.m_lb_pars;
    m_ub_pars=d.m_ub_pars;
    m_Grid=d.m_Grid;
    m_data_exp=d.m_data_exp;
    m_Xprofile=d.m_Xprofile;
    m_Xprofile_converted=d.m_Xprofile_converted;
    m_samples=d.m_samples;
    m_hparsofsamples=d.m_hparsofsamples;
    m_allmcmcsamples=d.m_allmcmcsamples;
    m_inputerr=d.m_inputerr;
    m_derivatives_obs=d.m_derivatives_obs;
    m_derivatives_preds=d.m_derivatives_preds;
    m_incx_obs=d.m_incx_obs;
};

Density::Density(DoE const & g)
{
    //construction à partir d'un grid (nécessaire).
    m_Grid=g;
    m_lb_pars=m_Grid.m_lb_pars;
    m_ub_pars=m_Grid.m_ub_pars;
    m_dim_pars=m_lb_pars.size();
}

void Density::SetNewDoE(DoE const & g){
  //set new grid
  m_Grid=g;
  m_lb_pars=m_Grid.m_lb_pars;
  m_ub_pars=m_Grid.m_ub_pars;
  m_dim_pars=m_lb_pars.size();
}

Eigen::MatrixXd Density::IncX(vector<VectorXd> const & locs) const {
  //Calcul de la matrice supplémentaire liée à l'incertitude sur les inputs. On rend une matrice diagonale dont les coefficients sont les carrés des dérivées du true process, pris aux points d'observations. Eh mais on pourrait simplement calculer les dérivées du modèle aux points d'observations puisqu'il est rapide à évaluer ! supposons qu'il est dans le vecteur m_modelderivatives.
  //première étape: calcul des dérivées. Ah mais ça dépend de theta à chaque fois... donc obligés de recalculer.
  //
  int nd=locs.size();
  Eigen::MatrixXd A=MatrixXd::Zero(nd,nd);
  for(int i=0; i<nd; i++){
    if(nd==m_derivatives_preds.size()){
      A(i,i)=pow(m_inputerr,2)*pow(m_derivatives_preds(i),2);
    }
    else if(nd==m_derivatives_obs.size()){
      A(i,i)=pow(m_inputerr,2)*pow(m_derivatives_obs(i),2);
    }
    else{
      cerr << "oups." << endl;
    }
  }
  return A;
}

void Density::Compute_derivatives_f(std::vector<Eigen::VectorXd> const & obs_locs,std::vector<Eigen::VectorXd> const & preds_locs,double max_time,string const & filename){
  //calcul des dérivées aux points d'observation et aux points de prédictions. Pour cela : on fait des différences finies aux points d'obs. Si plus complexe : on devra fitter un polynôme. Mais même pas besoin je pense. ah bah si pour des points quelconques...
  //on va fitter 1 fois 1 modèle bon pour estimer. Et après ça roule. en least squares bien sûr.
  VectorXd xguess(4); xguess(0)=10,xguess(1)=5,xguess(2)=7,xguess(3)=0;
  VectorXd lb(4); lb << 0,5,0,0;
  VectorXd ub(4); ub << 100000,15,10,3e5;
  auto p=this;
  optroutine(optfuncfitpol,p,xguess,lb,ub,max_time);
  cout << "best fit polynomial : " << xguess.transpose() << endl;
  //calcul des dérivées maintenant.
  auto fprime=[&xguess](double x){
    if(x<xguess(1)){
      return xguess(0);
    }
    else{
      return xguess(0)+xguess(2)*pow(x-xguess(1),xguess(2)-1);
    }
  };
  auto f=[&xguess](double x){
    if(x<xguess(1)){
      return xguess(0)*x+xguess(3);
    }
    else{
      return xguess(0)*x+pow(x-xguess(1),xguess(2))+xguess(3);
    }
  };
  
  VectorXd derobs(obs_locs.size());
  for(int i=0;i<obs_locs.size();i++){
    derobs(i)=fprime(obs_locs[i](0));
  }
    VectorXd derp(preds_locs.size());
  for(int i=0;i<preds_locs.size();i++){
    derp(i)=fprime(preds_locs[i](0));
  }
  m_derivatives_obs=derobs;
  m_derivatives_preds=derp;
  cout << "derivatives at obs pts : " << derobs.transpose() << endl;
  cout << "derivatives at pred pts : " << derp.transpose() << endl;
  //juste plot du fit poly pour voir.
  ofstream ofile(filename);
  for(int i=0;i<preds_locs.size();i++){
    ofile << preds_locs[i](0) << " " <<f(preds_locs[i](0)) << endl;
  }

}

double Density::optfuncfitpol(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* moindres carrés pour une fct de la forme spéciale.*/
  auto p=(Density*) data; //recast
  VectorXd y=p->m_data_exp[0].Value();
  VectorXd xobs=p->m_data_exp[0].GetX();
  // calcul des prédictions de la fonction
  VectorXd pred(xobs.size());
  for(int i=0;i<xobs.size();i++){
    if(xobs(i)<x[1]){
      pred(i)=x[0]*xobs(i)+x[3];
    }
    else{
      pred(i)=x[0]*xobs(i)+pow(xobs(i)-x[1],x[2])+x[3];
    }
  }
  VectorXd v=y-pred;
  double sqres=v.array().square().sum();
  return -1*sqres;
};



Eigen::VectorXd Density::modderiv(vector<VectorXd> const & locs, Eigen::VectorXd const &theta,Eigen::VectorXd const &hpar) const {
  //Calcul des dérivées du modèle en chaque point. Le modèle prend un vecteur en entrée donc il suffit de l'évaluer 1 fois supplémentaire.
  //attention on calcule bien les dérivées de f+z car sinon le modèle seul représente mal le true process...
  double dx=0.05; //distance pour évaluer les dérivées.
  VectorXd X(locs.size()); for(int i=0;i<X.size();i++){X(i)=locs[i](0);}
  VectorXd Xd(2*X.size());
  for(int i=0;i<X.size();i++){
    Xd(2*i)=X(i)-dx;
    Xd(2*i+1)=X(i)+dx;
  }
  VectorXd mod=m_model(Xd,theta)+meanZCondTheta_nox(Xd,theta,hpar);
  VectorXd modderiv(X.size());
  for(int i=0;i<X.size();i++){
    modderiv(i)=0.5*(mod(2*i+1)-mod(2*i))/dx;
  }
  return modderiv;
}

Eigen::MatrixXd Density::Gamma(vector<VectorXd> const & locs, Eigen::VectorXd const &hpar) const {
  // Renvoie la matrice de corrélation avec  bruit
  //bruit en nugget juste pour être sûr de l'inversibilité.
  int nd=locs.size();
  double noisey=10; //bruit super élevé pour avoir le bon ordre de grandeur.
  Eigen::MatrixXd A(nd,nd);
  for(int i=0; i<nd; i++){
    for(int j=i; j<nd; j++){
      A(i,j) = m_Kernel(locs[i],locs[j], hpar);
      if(i!=j){
	A(j,i) = A(i,j);
      }else{
	A(i,j) += pow(noisey,2);		//kernel.
      }
    }
  }
  return A;
}

double Density::loglikelihood_fast(VectorXd const &obs,LDLT<MatrixXd> const &ldlt) const{
    //calcul de la LL à la dernière étape.
    VectorXd Alpha=ldlt.solve(obs);
    int nd=obs.size();
    //return -0.5*obs.dot(Alpha)-0.5*(ldlt.vectorD().array().log()).sum()-0.5*nd*log(2*3.1415);
    return -0.5*obs.dot(Alpha)-0.5*(ldlt.vectorD().array().log()).sum();
}

double Density::loglikelihood_theta_fast(Eigen::VectorXd const &theta,Eigen::VectorXd const &hpars, Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const{
    //écrite pour un AUGDATA de taille 1.
    VectorXd obs=m_data_exp[0].Value()-m_model(m_data_exp[0].GetX(),theta); //pas de priormean pour aller plus vite.
    return loglikelihood_fast(obs,ldlt);
}

double Density::loglikelihood_theta(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar)const{
    //évaluation de la LDLT de Gamma. a priori écrite pour un AUGDATA de taille 1 également. Si la taille change, on pourrait faire plusieurs gamma (plusieurs hpars.)
    MatrixXd G=Gamma(m_Xprofile_converted,hpar);
    LDLT<MatrixXd> ldlt(G);
    return loglikelihood_theta_fast(theta,hpar,ldlt);
}

double Density::loglikelihood_theta_incx(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar)const{
    //évaluation de la LDLT de Gamma. a priori écrite pour un AUGDATA de taille 1 également. Si la taille change, on pourrait faire plusieurs gamma (plusieurs hpars.)
    MatrixXd G=Gamma(m_Xprofile_converted,hpar);
    LDLT<MatrixXd> ldlt(G+m_incx_obs);
    return loglikelihood_theta_fast(theta,hpar,ldlt);
}



double Density::loglikelihood_obs_i(Eigen::VectorXd const &obsv, int i,Eigen::VectorXd const &hpars)const{
  //calcule la vraisemblance de l'observation i. gaussienne.
  //on suppose que le vecteur yobs-f_theta-priormean a été calculé en amont.
  double obs=obsv(i);
  double var=m_Kernel(m_Xprofile_converted[i],m_Xprofile_converted[i],hpars)+pow(hpars(1),2);
  return -0.5*log(2*M_PI)-0.5*log(var)-0.5*pow(obs,2)/var;
}

LDLT<MatrixXd> Density::GetLDLT(VectorXd const & hpars){
    MatrixXd G=Gamma(m_Xprofile_converted,hpars);
    LDLT<MatrixXd> ldlt(G);
    return ldlt;
}

int Density::optroutine_light(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars)const{
  //routine d'optimisation unique, pour clarifier le code.
  vector<double> x=VXDtoV(X);
  vector<double> lb_hpars_opt=VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt=VXDtoV(ub_hpars);
  //paramètres d'optimisation
  int maxeval=2000;
  int popsize=80;
  double ftol=1e-4;
  // 1 opti globale et 1 opti locale.
  //glo
  nlopt::opt opt(nlopt::LN_SBPLX, x.size());
  opt.set_max_objective(optfunc, data_ptr); 
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxeval(maxeval);
  opt.set_ftol_rel(ftol);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  X=VtoVXD(x);
  return fin;
}

int Density::optroutine_heavy(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars)const{
  default_random_engine generator;
  uniform_real_distribution<double> distU(0,1);
  //routine d'optimisation unique, pour clarifier le code.
  vector<double> x=VXDtoV(X);
  vector<double> lb_hpars_opt=VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt=VXDtoV(ub_hpars);
  //paramètres d'optimisation
  int maxeval=2000;
  int popsize=80;
  double ftol=1e-6;
  nlopt::opt opt(nlopt::LN_AUGLAG_EQ, x.size());
  opt.set_max_objective(optfunc, data_ptr); 
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxeval(maxeval);
  opt.set_ftol_rel(ftol);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  //on recommence tant qu'on n'a pas stagné 50 fois.
  double msup_max=msup;
  double msup_prev=msup;
  int n_stagne=50;
  int n_current=0;
  int n_recommence=0;
  while(n_current<n_stagne){
    n_current++;
    vector<double> x_start(x.size());
    for(int i=0;i<x.size();i++){
      x_start[i]=lb_hpars_opt[i]+(ub_hpars_opt[i]-lb_hpars_opt[i])*distU(generator);
    }
    opt.optimize(x_start,msup);
    if(msup>msup_max){
      x=x_start;
      msup_max=msup;
      n_current=0;
      n_recommence++;
    }
  }
  X=VtoVXD(x);
  return fin;
}

int Density::optroutine(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars, double max_time)const{
    //routine d'optimisation sans gradient
  vector<double> x=VXDtoV(X);
  vector<double> lb_hpars_opt=VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt=VXDtoV(ub_hpars);
  //paramètres d'optimisation
  double ftol_large=1e-5;
  double xtol_large=1e-3;
  double ftol_fin=1e-15;
  double xtol_fin=1e-7;
  // 1 optimiseur local et un global.
  nlopt::opt local_opt(nlopt::LN_SBPLX,x.size());

  local_opt.set_max_objective(optfunc, data_ptr); 
  local_opt.set_ftol_rel(ftol_large);
  local_opt.set_xtol_rel(xtol_large);
  local_opt.set_lower_bounds(lb_hpars_opt);
  local_opt.set_upper_bounds(ub_hpars_opt);

  nlopt::opt opt(nlopt::GD_MLSL_LDS, x.size());
  opt.set_max_objective(optfunc, data_ptr);
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxtime(max_time); //limite de temps : 60 sec.
  opt.set_local_optimizer(local_opt);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  //on relance une opti locale à partir du max. trouvé.
  local_opt.set_ftol_rel(ftol_fin);
  local_opt.set_xtol_rel(xtol_fin);
  fin=local_opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  X=VtoVXD(x);
  return fin;
}

int Density::optroutine_withgrad(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars, double max_time)const{
  //routine d'optimisation avec gradient
  vector<double> x=VXDtoV(X);
  vector<double> lb_hpars_opt=VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt=VXDtoV(ub_hpars);
  //paramètres d'optimisation
  double ftol_large=1e-10;
  double xtol_large=1e-5;
  double ftol_fin=1e-20;
  double xtol_fin=1e-15;
  // 1 optimiseur local et un global.
  nlopt::opt local_opt(nlopt::LD_MMA,x.size());//MMA en temps normal.

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
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  //on relance une opti locale à partir du max. trouvé.
  local_opt.set_ftol_rel(ftol_fin);
  //local_opt.set_xtol_rel(xtol_fin);
  fin=local_opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  //calcul du point final et affichage du gradient.
  /*
  {
    vector<double> grad(x.size());
    double score_final=optfunc(x,grad,data_ptr);
    cout << "x : ";
    for(int i=0;i<x.size();i++){
      cout << x[i] << " ";
    }
    cout <<endl << "grad : " ;
    for(int i=0;i<grad.size();i++){
      cout << grad[i] << " ";
    }
    cout << endl << "score : " << score_final << endl;
  }
  */
  X=VtoVXD(x);
  return fin;
}

int Density::optroutine_lightwithgrad(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars, double max_time) const{
  //routine d'optimisation avec gradient. une simple optimisation locale.
  vector<double> x=VXDtoV(X);
  vector<double> lb_hpars_opt=VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt=VXDtoV(ub_hpars);
  //paramètres d'optimisation
  double ftol_fin=1e-20;
  double xtol_fin=1e-15;
  // seulement 1 opti locale
  //glo
  nlopt::opt opt(nlopt::LD_MMA, x.size());
  opt.set_max_objective(optfunc, data_ptr); 
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxtime(max_time);
  //opt.set_population(popsize);
  opt.set_ftol_rel(ftol_fin);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  X=VtoVXD(x);
  return fin;
}

int Density::optroutine_lightnograd(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars, double max_time) const{
  //routine d'optimisation sans gradient. une simple optimisation locale.
  vector<double> x=VXDtoV(X);
  vector<double> lb_hpars_opt=VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt=VXDtoV(ub_hpars);
  //paramètres d'optimisation
  double ftol_fin=1e-20;
  double xtol_fin=1e-15;
  // seulement 1 opti locale
  //glo
  nlopt::opt opt(nlopt::LN_SBPLX, x.size());
  opt.set_max_objective(optfunc, data_ptr); 
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxtime(max_time);
  //opt.set_population(popsize);
  opt.set_ftol_rel(ftol_fin);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  X=VtoVXD(x);
  return fin;
}


VectorXd Density::HparsKOH(VectorXd const & hpars_guess, double max_time) const {
  //calcule les hyperparamètres optimaux selon KOH.
  auto begin=chrono::steady_clock::now();
  VectorXd guess=hpars_guess;
  //On prépare les évaluations de surrogate pour la routine opt, pour accélérer la fonction.
  MatrixXd Residustheta(m_data_exp[0].Value().size(),m_Grid.m_grid.size());
  for(int i=0;i<m_Grid.m_grid.size();i++){
    VectorXd theta=m_Grid.m_grid[i];
    Residustheta.col(i)=-m_model(m_data_exp[0].GetX(),theta)+m_data_exp[0].Value(); //tout y-ftheta stocké ici. Le calcul de la priormean se fait dans optfuncKOH.
  }
  auto tp=make_tuple(&Residustheta,this);
  int fin=optroutine(optfuncKOH,&tp,guess,m_lb_hpars,m_ub_hpars,max_time);
  auto end=chrono::steady_clock::now();
  cout << "fin de l'opt koh : message " << fin << endl;
  cout << guess.transpose() << endl;
  cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s." << endl;
  return guess;
}

VectorXd Density::HparsLOOCV(VectorXd const & hpars_guess, double max_time) const{
  //calcule les hyperparamètres optimaux selon KOH.
  VectorXd guess=hpars_guess;
  //On prépare les évaluations de surrogate pour la routine opt, pour accélérer la fonction.
  MatrixXd Residustheta(m_data_exp[0].Value().size(),m_Grid.m_grid.size());
  for(int i=0;i<m_Grid.m_grid.size();i++){
    VectorXd theta=m_Grid.m_grid[i];
    Residustheta.col(i)=-m_model(m_data_exp[0].GetX(),theta)+m_data_exp[0].Value(); //y-ftheta
  }
  auto tp=make_tuple(&Residustheta,this);
  int fin=optroutine(optfuncLOOCV,&tp,guess,m_lb_hpars,m_ub_hpars,max_time);
  cout << "fin de l'opt loocv : message " << fin << endl;
  return guess;
}

double Density::optfuncLOOCV(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* fonction à optimiser pour trouver les hpars loocv. On le fait selon la philosophie KOH, c'est-à-dire on intègre la densité loocv contre la prior de theta.*/
  auto ptp=(tuple<const MatrixXd *,Density*>*) data; //cast
  auto tp=*ptp;
  const MatrixXd *Residus=get<0>(tp);
  const Density *d=get<1>(tp);
  const vector<VectorXd> *xconv=d->GetXconverted();
  VectorXd hpars=VtoVXD(x);
  double logvstyp=30; //à choisir.
  //calcul de la logvs sur le grid
  vector<double> prob(Residus->cols());
  MatrixXd G=d->Gamma(*xconv,hpars);
  LDLT<MatrixXd> ldlt(G);
  MatrixXd Ginv=ldlt.solve(MatrixXd::Identity(G.cols(),G.cols()));
  //au lieu du transform, j'essaye une boucle simple.
  for(int i=0;i<Residus->cols();i++){
    //boucle sur les theta
    //on rentre la valeur de la fonction loocv pour une valeur de theta (donc c'est une somme)
    //on choisit y.
    VectorXd alpha=ldlt.solve(Residus->col(i));
    double g=0;
    for(int j=0;j<alpha.size();j++){
      //boucle sur les observations
      g+=log(Ginv(j,j))-pow(alpha(j),2)/Ginv(j,j);
    }
    prob[i]=g;
  }
  //pas de passage à l'exponentielle. on va juste faire la moyenne.
  //calcul de l'intégrale. suppose un grid régulier. modifier si on veut un prior pour les paramètres.
  double res=accumulate(prob.begin(),prob.end(),0.0); res/=prob.size();
  //avec prior sur les hpars.
  res+=d->m_logpriorhpars(hpars);
  return res;
};


double Density::optfuncKOH(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* fonction à optimiser pour trouver les hpars koh.*/
  auto ptp=(tuple<const MatrixXd *,Density*>*) data; //cast
  auto tp=*ptp;
  const MatrixXd *Residus=get<0>(tp);
  const Density *d=get<1>(tp);
  const vector<VectorXd> *xconv=d->GetXconverted();
  VectorXd hpars=VtoVXD(x);
  double logvstyp=30; //à choisir.
  //calcul de la logvs sur le grid
  vector<double> prob(Residus->cols());
  MatrixXd G=d->Gamma(*xconv,hpars)+d->IncX(*xconv);
  LDLT<MatrixXd> ldlt(G);
  VectorXd pmean=d->EvaluatePMean(d->GetXprofile(),hpars); //on utilise bien la prior mean.
  //au lieu du transform, j'essaye une boucle simple.
  for(int i=0;i<Residus->cols();i++){
    double g=d->loglikelihood_fast(Residus->col(i)-pmean,ldlt);
    prob[i]=g;
  }
  //normalisation et passage à l'exponentielle.
  //le max doit être le même sur toutes les itérations... d'où la nécessité de le définir à l'extérieur !!!
  for(int i=0;i<prob.size();i++){
    double p=prob[i];
    VectorXd theta=d->GetGrid()->at(i);
    double logprior=d->EvaluateLogPPars(theta);
    double f=exp(p+logprior-logvstyp);
    if(isinf(f)){cerr << "erreur myoptfunc_koh : infini. Valeur de la fonction : " << p+logprior << endl;}
    prob[i]=f;
  }

  //calcul de l'intégrale. suppose un grid régulier. modifier si on veut un prior pour les paramètres.

  double res=accumulate(prob.begin(),prob.end(),0.0);
  res*=exp(d->m_logpriorhpars(hpars));
  return res;
};

VectorXd Density::HparsKOHFromData(VectorXd const & hpars_guess,vector<VectorXd> const & thetas, vector<VectorXd> const & values) const{
  //calcule les hyperparamètres optimaux selon KOH, à partir des données en entrée.
  VectorXd guess=hpars_guess;
  auto tp=make_tuple(&thetas,&values,this);
  int fin=optroutine_light(optfuncKOHFromData,&tp,guess,m_lb_hpars,m_ub_hpars);
  cout << "fin de l'opt koh from data : message " << fin << endl;
  return guess;
}



double Density::optfuncKOHFromData(const std::vector<double> &x, std::vector<double> &grad, void *data){
    /* fonction à optimiser pour trouver les hpars koh.*/
  VectorXd hpars=VtoVXD(x);
  auto ptp=(tuple<const vector<VectorXd>*,const vector<VectorXd>*, Density*> *) data; //cast
  auto tp=*ptp;

  const vector<VectorXd>* thetas=get<0>(tp);
  const vector<VectorXd>* values=get<1>(tp);
  Density *d=get<2>(tp);
  const vector<VectorXd> *xconv=d->GetXconverted();
  const vector<AUGDATA> *expdata=d->GetExpData();
  VectorXd obs=((*expdata)[0]).Value();
  VectorXd X=((*expdata)[0]).GetX();

  double logvstyp=30; //à choisir.
  //calcul de la logvs sur le grid
  vector<double> prob(thetas->size());
  MatrixXd G=d->Gamma(*xconv,hpars);
  LDLT<MatrixXd> ldlt(G);
  for(int i=0;i<thetas->size();i++){
    VectorXd theta=(*thetas)[i];
    //construction des data
    VectorXd obstheta=obs-d->m_model(X,theta);
    double g=d->loglikelihood_fast(obstheta,ldlt);
    prob[i]=g;
  }
  transform(prob.begin(),prob.end(),prob.begin(),[logvstyp](const double &d)->double {
    double f=exp(d-logvstyp);
    if(isinf(f)){cerr << "erreur myoptfunc_koh fromdata : infini" << endl;}
    return f;
  });
  //calcul de l'intégrale. suppose un grid régulier. modifier si on veut un prior pour les paramètres.
  double res=accumulate(prob.begin(),prob.end(),0.0); res/=prob.size();
  res*=exp(d->m_logpriorhpars(hpars));
  return res;

};

/*
VectorXd Density::HparsNOEDM(VectorXd const & hpars_guess) {
    //calcule juste l'erreur expérimentale optimale sans erreur de modèle.
    //calcul d'une valeur typique
    const double logvstyp=30;

    //paramètres de l'optimisation
    int maxeval=5000;
    vector<double> lb_hpars(1);
    lb_hpars[0]=m_lb_hpars[1]; //récupération des bornes pour sobs
    vector<double> ub_hpars(1);
    ub_hpars[0]=m_ub_hpars[1];
    vector<double> x(1);
    x[0]=hpars_guess(1); //guess.
    nlopt::opt opt(nlopt::GN_ISRES, 1);
    opt.set_max_objective(optfuncNOEDM, this); 
    opt.set_lower_bounds(lb_hpars);
    opt.set_upper_bounds(ub_hpars);
    opt.set_maxeval(maxeval);
    opt.set_population(2000);
    opt.set_ftol_rel(1e-4);
    double msup; 
    int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 0: erreur
    VectorXd res(3); //on renvoie quand même un vectorXd de taille 3 pour l'utiliser avec kernel.
    res(0)=0; res(1)=x[0]; res(2)=1;
    return res;
}*/

double Density::optfuncNOEDM(const std::vector<double> &x, std::vector<double> &grad, void *data){
  //ne pas utiliser avec une priormean.
    /* fonction à optimiser pour trouver les hpars koh.*/
  VectorXd hpars(3);
  hpars << 0,x[0],1;
  Density *d = (Density *) data; //cast
  const vector<VectorXd> * grid=d->GetGrid(); //const ptr vers le grid.
  const vector<VectorXd> *xconv=d->GetXconverted();
  double logvstyp=30; //à choisir.
  //calcul de la logvs sur le grid
  vector<double> prob(grid->size());
  MatrixXd G=d->Gamma(*xconv,hpars);
  LDLT<MatrixXd> ldlt(G);
  transform(grid->cbegin(),grid->cend(),prob.begin(),[d,&ldlt,hpars](VectorXd const & theta)->double{
      double g=d->loglikelihood_theta_fast(theta,hpars,ldlt);
      //cout << "g :" << g << endl;
      //cout << "theta " << theta.transpose() << endl;
      return g;
  });
  //normalisation et passage à l'exponentielle.
  //le max doit être le même sur toutes les itérations... d'où la nécessité de le définir à l'extérieur !!!
  transform(prob.begin(),prob.end(),prob.begin(),[logvstyp](const double &d)->double {
    double f=exp(d-logvstyp);
    if(isinf(f)){cerr << "erreur myoptfunc_noedm : infini" << endl;}
    return f;
  });
  //calcul de l'intégrale. suppose un grid régulier.
  double res=accumulate(prob.begin(),prob.end(),0.0); res/=prob.size();
  //on ne rajoute pas de prior sur les hpars si ça ne concerne pas sexp.
  //res*=exp(d->m_logpriorhpars(hpars));
  //cout << "hpars testés : " << x[0] << " int : " << res << endl;
  return res;
};




bool Density::in_bounds_pars(VectorXd const & pars) const{
  for(int i=0;i<m_dim_pars;i++){
    if(pars(i)<m_lb_pars(i) || pars(i)>m_ub_pars(i)){
      //if(pars(i)<m_lb_pars(i)){cout << "oob- : " << i <<endl;}
      //else{cout << "oob+ : " << i <<endl;}
      return false;
    }
  }
  return true;
}

bool Density::in_bounds_hpars(VectorXd const & hpars) const{
  for(int i=0;i<m_dim_hpars;i++){
    if(hpars(i)<m_lb_hpars(i) || hpars(i)>m_ub_hpars(i)){
      //if(hpars(i)<m_lb_hpars(i)){cout << "oob- : " << i+m_dim_pars<<endl;}
      //else{cout << "oob+ : " << i+m_dim_pars<<endl;}
      return false;
    }
  }
  return true;
}

double Density::Run_Burn_phase_MCMC(int nburn, MatrixXd & COV_init,VectorXd const & hpars, VectorXd & Xcurrento, default_random_engine & generator) {
  //phase de burn et calcul de la nouvelle matrice de covariance.
  //on renvoie la covariance scalée et l'état actuel.
  //on peut calculer une seule fois la matrice Gamma puisque les hpars restent identiques.
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  LDLT<MatrixXd> ldlt(Gamma(m_Xprofile_converted,hpars));

  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  VectorXd Xinit=Xcurrento;
  double finit=loglikelihood_theta_fast(Xinit,hpars,ldlt)+m_logpriorpars(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(m_dim_pars);
  MatrixXd acc_var=MatrixXd::Zero(m_dim_pars,m_dim_pars);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      double fcandidate=loglikelihood_theta_fast(Xcandidate,hpars,ldlt)+m_logpriorpars(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
    m_allmcmcsamples.push_back(Xcurrent);
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(m_dim_pars))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2)+1e-10*MatrixXd::Identity(m_dim_pars,m_dim_pars));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
  return acc_rate;
}

void Density::Run_MCMC_fixed_hpars(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,VectorXd const & hpars,default_random_engine & generator){
  //MCMC à hpars fixés.

  cout << "running mcmc fixedhpars with " << nsteps << " steps." <<endl;
  //vidons les samples juste au cas où.
  m_samples.clear(); m_hparsofsamples.clear(); m_allmcmcsamples.clear();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  LDLT<MatrixXd> ldlt(Gamma(m_Xprofile_converted,hpars));
  auto compute_score=[this,hpars,ldlt](VectorXd const & X)-> double {
    return this->loglikelihood_theta_fast(X,hpars,ldlt)+this->m_logpriorpars(X);
  };
  MatrixXd COV=COV_init;
  MatrixXd COVPred=COV;
  Run_Burn_phase_MCMC(nsteps*0.1,COV,hpars,Xinit,generator);
  //COV=scale_covmatrix(COV,Xinit,compute_score,0,generator,"results/diag/scalekoh.gnu");
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      m_samples.push_back(Xcurrent);
      m_hparsofsamples.push_back(hpars);
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}

double Density::Run_Burn_phase_FullMCMC(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento, default_random_engine & generator) {
//phase de burn et calcul de la nouvelle matrice de covariance.
  //on renvoie la covariance scalée et l'état actuel.
  //on peut calculer une seule fois la matrice Gamma puisque les hpars restent identiques.
  int dim_mcmc=m_dim_pars+m_dim_hpars;
  auto compute_score=[this](VectorXd const & X)-> double {
    VectorXd hpars=X.tail(m_dim_hpars);
    VectorXd theta=X.head(m_dim_pars);
    return this->loglikelihood_theta(theta,hpars)+this->m_logpriorhpars(hpars)+this->m_logpriorpars(theta);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  VectorXd Xinit=Xcurrento;
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(dim_mcmc);
  MatrixXd acc_var=MatrixXd::Zero(dim_mcmc,dim_mcmc);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate.head(m_dim_pars)) & in_bounds_hpars(Xcandidate.tail(m_dim_hpars))){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
    m_allmcmcsamples.push_back(Xcurrent);
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(dim_mcmc))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
  return acc_rate;
}

void Density::Run_FullMCMC(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,default_random_engine & generator){
//MCMC à hpars variables
  cout << "running mcmc bayes with " << nsteps << " steps, with burn." <<endl;
  m_samples.clear(); m_hparsofsamples.clear(); m_allmcmcsamples.clear();
  int dim_mcmc=m_dim_pars+m_dim_hpars;
  auto compute_score=[this](VectorXd const & X)-> double {
    VectorXd hpars=X.tail(m_dim_hpars);
    VectorXd theta=X.head(m_dim_pars);
    return this->loglikelihood_theta(theta,hpars)+this->m_logpriorhpars(hpars)+this->m_logpriorpars(theta);
  };
  m_samples.clear(); m_hparsofsamples.clear();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  Run_Burn_phase_FullMCMC(nsteps*0.1,COV,Xinit,generator);
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate.head(m_dim_pars)) && in_bounds_hpars(Xcandidate.tail(m_dim_hpars))){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>1000){
      m_samples.push_back(Xcurrent.head(m_dim_pars));
      m_hparsofsamples.push_back(Xcurrent.tail(m_dim_hpars));
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}

void Density::Run_FullMCMC_noburn(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,default_random_engine & generator){
//MCMC à hpars variables
  cout << "running mcmc bayes with " << nsteps << " steps, no burn." <<endl;
  m_samples.clear(); m_hparsofsamples.clear(); m_allmcmcsamples.clear();
  int dim_mcmc=m_dim_pars+m_dim_hpars;
  auto compute_score=[this](VectorXd const & X)-> double {
    VectorXd hpars=X.tail(m_dim_hpars);
    VectorXd theta=X.head(m_dim_pars);
    return this->loglikelihood_theta(theta,hpars)+this->m_logpriorhpars(hpars)+this->m_logpriorpars(theta);
  };
  m_samples.clear(); m_hparsofsamples.clear();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate.head(m_dim_pars)) && in_bounds_hpars(Xcandidate.tail(m_dim_hpars))){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>1000){
      m_samples.push_back(Xcurrent.head(m_dim_pars));
      m_hparsofsamples.push_back(Xcurrent.tail(m_dim_hpars));
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}

void Density::Autocor_diagnosis(int nstepsmax, string const & filename) const
{
  //diagnostic d'autocorrélation, jusqu'à une distance de nstepsmax pas dans la mcmc.
  //on oublie les 10% premières steps dans le diagnostic, pour la phase de burn.
  //on centre les données
  int dim_mcmc=m_allmcmcsamples[0].size();
  int nselected=0.2*m_allmcmcsamples.size(); //on estime sur les derniers 10%
  cout << "autocorrélation calculée sur les 20 derniers pourcents des samples !" << endl;
  vector<VectorXd> allmcmcsamples_selected(nselected);
  VectorXd mean=VectorXd::Zero(dim_mcmc);
  VectorXd var=VectorXd::Zero(dim_mcmc); //variance composante par composante
  for (int i=0;i<nselected;i++){
    allmcmcsamples_selected[i]=m_allmcmcsamples[i+m_allmcmcsamples.size()-nselected];
    mean+=allmcmcsamples_selected[i];
  }
  mean/=nselected;
  for (int i=0;i<nselected;i++){
    allmcmcsamples_selected[i]-=mean;
    var+=allmcmcsamples_selected[i].cwiseProduct(allmcmcsamples_selected[i]);
  }
  var/=nselected; //calcul de la variance composante par composante.
  ofstream ofile(filename);
  VectorXd integ=VectorXd::Zero(dim_mcmc);
  FILE* out= fopen("results/autocor.gnu","w");
  for (int i=0;i<nstepsmax;i++){
    VectorXd cor=Lagged_mean(allmcmcsamples_selected,i).cwiseQuotient(var);
    for(int j=0;j<cor.size();j++){
      ofile << cor(j) << " ";
    }
    integ+=cor;
    ofile << endl;
  }
  ofile.close();
  cout << "intégrales avec nautocor = " << nstepsmax << endl;
  cout << integ.transpose() << endl;
  cout << "maximum : " << integ.maxCoeff() << endl;
}

int Density::max_ll()const{
  //renvoie l'adresse du MAP parmi les samples contenus dans m_sample.
  //la fonction de référence est loglikelihood seule.
  cout << "searching for MAP among " << m_samples.size() << " samples.." << endl;
  vector<double> score(m_samples.size());
  for(int i=0;i<m_samples.size();i++){
    score[i]=loglikelihood_theta(m_samples[i],m_hparsofsamples[i]);
  }
  double max=score[0];
  int imax=0;
  for(int i=0;i<score.size();i++){
    if(score[i]>max){
      max=score[i];
      imax=i;
    }
  }
  cout << "indice MAP : " << imax << endl;
  return imax;
}


void Density::WriteObsWithUncertainty(std::string const filename) const{
  //écriture des observations dans un fichier, mais avec leur incertitude en x et leur incertitude en y. 1 seul écart-type.
  VectorXd yobs=m_data_exp[0].Value();
  VectorXd xobs=m_data_exp[0].GetX();
  VectorXd incx=m_inputerr*VectorXd::Ones(xobs.size());
  MatrixXd M=IncX(m_Xprofile_converted);
  VectorXd incy(xobs.size());
  for(int i=0;i<incy.size();i++){
    incy(i)=sqrt(M(i,i));
  }
  ofstream ofile(filename);
  for(int i=0;i<xobs.size();i++){
    ofile << xobs(i) << " " << yobs(i) << " " << incx(i) << " " << incy(i) << endl;
  }
}

VectorXd Density::Lagged_mean(vector<VectorXd> const & v,int n) const{
  int dim_mcmc=v[0].size();
  VectorXd ans=VectorXd::Zero(dim_mcmc);
  for (int i=0;i<v.size()-n;i++){
    ans+=v[i].cwiseProduct(v[i+n]);
  }
  return ans/(double (v.size()-n));
}

void Density::WriteMCMCSamples(string const & filename)const{
  ofstream ofile(filename);
  for(int i=0;i<m_allmcmcsamples.size();i++){
    for(int j=0;j<m_allmcmcsamples[0].size();j++){
      ofile << m_allmcmcsamples[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

void Density::FindVPs(MatrixXd const &M) const{
  //décomposition en valeurs propres de la matrice M.
  SelfAdjointEigenSolver<MatrixXd> eig(M);
  VectorXd lambdas=eig.eigenvalues();
  MatrixXd vecpropres=eig.eigenvectors(); //vecteurs propres stockés dans les colonnes. Le mode principal est la dernière colonne.
  lambdas.reverseInPlace();
  MatrixXd vecpropresr=vecpropres.rowwise().reverse();
  cout << "valeurs propres : " << lambdas.transpose() << endl;
  cout << "modes associés : " << endl << vecpropresr << endl;
}

void Density::WriteSamples(string const & filename)const{
  ofstream ofile(filename);
  for(int i=0;i<m_samples.size();i++){
    for(int j=0;j<m_samples[0].size();j++){
      ofile << m_samples[i](j) << " ";
    }
    for(int j=0;j<m_hparsofsamples[0].size();j++){
      ofile << m_hparsofsamples[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

void Density::ReadSamples(string const & filename){
  //écrit pour la dimension 5 et 3 hpars pour kernel_z. attention à ce que m_dim_pars et m_dim_hpars soient bien sélectionnées.
  m_samples.clear();
  m_hparsofsamples.clear();
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      
      VectorXd theta(m_dim_pars);
      VectorXd hpars(m_dim_hpars);
      for(int i=0;i<m_dim_pars;i++){
        theta(i)=stod(words[i]);
      }
      for(int i=0;i<m_dim_hpars;i++){
        hpars(i)=stod(words[i+m_dim_pars]); //c'est bien pars ici.
      }
      m_samples.push_back(theta);
      m_hparsofsamples.push_back(hpars);
    }
    cout << "number of samples loaded : " << m_samples.size() <<". dims : " << m_dim_pars << " and " << m_dim_hpars <<"."<<endl;
  }
  else{
    cerr << "empty file" << endl;
  }

  ifile.close();
}

VectorXd Density::meanF(VectorXd const & X) const {
  //prédiction moyenne du modèle
  VectorXd mpf=VectorXd::Zero(X.size());
  for(VectorXd const &s:m_samples){
    mpf+=m_model(X,s);
  }
  return mpf/m_samples.size();
}

MatrixXd Density::varF(VectorXd const & X) const {
  //variance de f seule.
  VectorXd mpf=meanF(X);
  MatrixXd SecondMoment=MatrixXd::Zero(mpf.size(),mpf.size());
  for_each(m_samples.begin(),m_samples.end(),[&SecondMoment,&X,this](VectorXd const &theta)mutable{
    VectorXd Pred=this->m_model(X,theta);
    SecondMoment+=Pred*Pred.transpose();
  });
  MatrixXd var=SecondMoment/m_samples.size()-mpf*mpf.transpose();
  //on renvoie toute la matrice et non la diagonale.
  return var;
}
VectorXd Density::meanZCondTheta_nox(VectorXd const & X,VectorXd const & theta, VectorXd const & hpars) const{
  //prédiction moyenne de Z sur Xprofile, pour le i-ème theta de m_samples.
  VectorXd y(X.size());
  y=m_data_exp[0].Value()-m_model(m_Xprofile,theta)-m_priormean(m_Xprofile,hpars);
  //puisque la prédiction du modèle et les observations sont aux mêmes points, ça simplifie grandement la tâche.
  MatrixXd G=Gamma(m_Xprofile_converted,hpars);
  //corrélation entre les obs et les points demandés.
  MatrixXd K=MatrixXd::Zero(X.size(),m_Xprofile_converted.size());
  for(int i=0;i<K.rows();i++){
    VectorXd x(1);x(0)=X(i);
    for(int j=0;j<K.cols();j++){
      K(i,j)=m_Kernel(x,m_Xprofile_converted[j],hpars);
    }
  }
  LDLT<MatrixXd> ldlt(G);
  //construction de la matrice de covariance, entre les mêmes points, mais sans le bruit expérimental ni l'incertitude d'input.
  VectorXd predmean=m_priormean(X,hpars)+K*ldlt.solve(y);
  return predmean;
}

VectorXd Density::meanZCondTheta(VectorXd const & X,VectorXd const & theta, VectorXd const & hpars) const{
  //prédiction moyenne de Z sur Xprofile, pour le i-ème theta de m_samples.
  VectorXd y(X.size());
  y=m_data_exp[0].Value()-m_model(m_Xprofile,theta)-m_priormean(m_Xprofile,hpars);
  //puisque la prédiction du modèle et les observations sont aux mêmes points, ça simplifie grandement la tâche.
  MatrixXd G=Gamma(m_Xprofile_converted,hpars)+IncX(m_Xprofile_converted);
  //corrélation entre les obs et les points demandés.
  MatrixXd K=MatrixXd::Zero(X.size(),m_Xprofile_converted.size());
  for(int i=0;i<K.rows();i++){
    VectorXd x(1);x(0)=X(i);
    for(int j=0;j<K.cols();j++){
      K(i,j)=m_Kernel(x,m_Xprofile_converted[j],hpars);
    }
  }
  LDLT<MatrixXd> ldlt(G);
  //construction de la matrice de covariance, entre les mêmes points, mais sans le bruit expérimental ni l'incertitude d'input.
  VectorXd predmean=m_priormean(X,hpars)+K*ldlt.solve(y);
  return predmean;
}

MatrixXd Density::varZCondTheta(VectorXd const & X,VectorXd const & theta, VectorXd const & hpars) const{
  //variance prédictive de z sur Xprofile, pour le i-ème theta de m_samples.

  MatrixXd G=Gamma(m_Xprofile_converted,hpars)+IncX(m_Xprofile_converted);
    //corrélation entre les obs et les points demandés.
  MatrixXd K=MatrixXd::Zero(X.size(),m_Xprofile_converted.size());
  for(int i=0;i<K.rows();i++){
    VectorXd x(1);x(0)=X(i);
    for(int j=0;j<K.cols();j++){
      K(i,j)=m_Kernel(x,m_Xprofile_converted[j],hpars);
    }
  }
  MatrixXd Kprior=MatrixXd::Zero(X.size(),X.size());
  for(int i=0;i<Kprior.rows();i++){
    VectorXd x(1);x(0)=X(i);
    for(int j=0;j<Kprior.cols();j++){
      VectorXd y(1);y(0)=X(j);
      Kprior(i,j)=m_Kernel(x,y,hpars);
    }
  }
  //puisque la prédiction du modèle et les observations sont aux mêmes points, ça simplifie grandement la tâche.
  LDLT<MatrixXd> ldlt(G);
  //construction de la matrice de covariance, entre les mêmes points, mais sans le bruit expérimental.
  MatrixXd VarPred=Kprior-K*ldlt.solve(K.transpose());
  //on renvoie seulement la diagonale (prédiction moyenne)
  return VarPred;
}

MatrixXd Density::PredFZ(VectorXd const & X) const{
  //predictions avec f+z. Première colonne : moyenne, Deuxième colonne : variance de E[f+z]. Troisième colonne : espérance de var[z].
  //récupération des valeurs de E[f+z|theta]
  //Calcul de la composante E[Var z] également.
  VectorXd mean=VectorXd::Zero(X.size());
  MatrixXd SecondMoment=MatrixXd::Zero(X.size(),X.size());
  MatrixXd Evarz=MatrixXd::Zero(X.size(),X.size());
  for(int i=0;i<m_samples.size();i++){
    VectorXd theta=m_samples[i];
    VectorXd hpars=m_hparsofsamples[i];
    VectorXd fpred=m_model(X,theta);
    VectorXd zpred=meanZCondTheta(X,theta,hpars);
    mean+=fpred+zpred;
    SecondMoment+=(fpred+zpred)*(fpred+zpred).transpose();
    Evarz+=varZCondTheta(X,theta,hpars);
  }
  mean/=m_samples.size();
  SecondMoment/=m_samples.size();
  Evarz/=m_samples.size();
  MatrixXd VarEfz=SecondMoment-mean*mean.transpose();
  MatrixXd res(X.size(),3);
  res.col(0)=mean; res.col(1)=VarEfz.diagonal(); res.col(2)=Evarz.diagonal();
  return res;
}

VectorXd Density::DrawZCondTheta(VectorXd const & X, VectorXd const & theta, VectorXd const &hpars_z, default_random_engine & generator) const{
  //tirage d'une réalisation de z pour un theta et des hpars donnés.
  normal_distribution<double> distN(0,1);
  VectorXd mean=meanZCondTheta(X,theta,hpars_z);
  MatrixXd Cov=varZCondTheta(X,theta,hpars_z);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> DEC(Cov);
  Eigen::VectorXd D = DEC.eigenvalues();
	for(unsigned i=0; i<D.rows(); i++) D(i) = sqrt(fabs(D(i)));
	std::cout << "Dmax : " << D.maxCoeff() << " Dmin " << D.minCoeff() << std::endl;
  VectorXd sample(mean.size());
  for(int i=0;i<sample.size();i++){sample(i)=distN(generator)*D(i);}
  return mean+DEC.eigenvectors()*sample;
}

/*
VectorXd Density::DrawZCondTheta(VectorXd const & X, VectorXd const & theta, VectorXd const &hpars_z, default_random_engine & generator) const{
  //tirage d'une réalisation de z pour un theta et des hpars donnés.
  normal_distribution<double> distN(0,1);
  VectorXd mean=meanZCondTheta(X,theta,hpars_z);
  MatrixXd Cov=varZCondTheta(X,theta,hpars_z);
  Eigen::LDLT<Eigen::MatrixXd> DEC(Cov);
	Eigen::VectorXd D = DEC.vectorD();
  for(int i=0; i<D.rows(); i++) D(i) = sqrt(fabs(D(i)));
  std::cout << "Dmax : " << D.maxCoeff() << " Dmin " << D.minCoeff() << std::endl;

  VectorXd sample(mean.size());
  for(int i=0;i<sample.size();i++){sample(i)=distN(generator)*D(i);}
  return mean+DEC.matrixL()*sample;
}*/

/*
VectorXd Density::DrawZCondTheta(VectorXd const & X, VectorXd const & theta, VectorXd const &hpars_z, default_random_engine & generator) const{
  //tirage d'une réalisation de z pour un theta et des hpars donnés. vieille version avec LLT
  normal_distribution<double> distN(0,1);
  VectorXd mean=meanZCondTheta(X,theta,hpars_z);
  MatrixXd Cov=varZCondTheta(X,theta,hpars_z);
  VectorXd N(mean.size());
  for(int i=0;i<N.size();i++){N(i)=distN(generator);}
  MatrixXd sqrtCOV=Cov.llt().matrixL();
  return mean+sqrtCOV*N;
}
*/


void Density::WriteOneCalcul(VectorXd const & X, VectorXd const & theta, VectorXd const & hpars_z, string const & filename) const{
  //ne marche pas si X différent de m_Xprofile
  VectorXd meanf=m_model(X,theta);
  VectorXd meanz=meanZCondTheta(X,theta,hpars_z);
  ofstream ofile(filename);
  for(int i=0;i<meanf.rows();i++){
    ofile << m_Xprofile(i) << " " << m_data_exp[0].Value()(i) <<" "<< meanf(i) << " " << meanz(i) << endl;
  }
  ofile.close();
}


void Density::WritePredictions(VectorXd const & X, string const & filename) const{
  //de plus, tirage de multiples échantillons de z.
  default_random_engine generator{static_cast<long unsigned int>(time(0))};
  uniform_int_distribution<int> U(0,m_samples.size()-1);
  vector<VectorXd> tirages(10);
  for(int i=0;i<tirages.size();i++){
    int r=U(generator);
    VectorXd samp=m_model(X,m_samples[r])+DrawZCondTheta(X,m_samples[r],m_hparsofsamples[r],generator); //drawzcondtheta contient la prior mean
    tirages[i]=samp;
  }
  MatrixXd Predictions=PredFZ(X); //col0 : moyenne fz, col1: varefz, col2: evarz
  ofstream ofile(filename);
  for(int i=0;i<Predictions.rows();i++){
    ofile << X(i) << " " << Predictions(i,0) << " " << sqrt(Predictions(i,1)+Predictions(i,2)) << " " << sqrt(Predictions(i,2)) << " ";
    for(int j=0;j<tirages.size();j++){
      ofile << tirages[j](i) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

double Density::FindQuantile(double pct, VectorXd const &X) const{
  //retourne le n-ième élément du vectorXd
  int n=pct*X.size();
  vector<double> x=VXDtoV(X);
  nth_element(x.begin(),x.begin()+n,x.end());
  return x[n];
}

void Density::WritePredictionsF(VectorXd const & X, string const & filename) const{
  //écriture des prédictions de F, avec theta selon la postérieure. Les intervalles de confiance sont affichés d'après la vraie distribution des valeurs de f (et non en calculant la variance)
  //modifié, pas de predfmap.
  cout << "computing predictions with " << m_samples.size() << " samples..." << endl;
  VectorXd meanf=meanF(X); //prédiction moyenne
  MatrixXd fvalues(meanf.rows(),m_samples.size());
  for(int i=0;i<m_samples.size();i++){
    fvalues.col(i)=m_model(X,m_samples[i]);
  }
  //on trie chaque ligne de fvalues, une seule fois.
  for(int i=0;i<fvalues.rows();i++){
    VectorXd V=fvalues.row(i).transpose();
    vector<double> v=VXDtoV(V);
    std::sort(v.begin(),v.end());
    fvalues.row(i)=VtoVXD(v).transpose();
  }

  //vecteurs des quantiles
  //int ind_max=max_ll();
  //VectorXd predf_map=m_model(X,m_samples[ind_max]);
  VectorXd quant2p5(meanf.rows()); //95pct de la masse
  VectorXd quant97p5(meanf.rows());
  VectorXd quant25(meanf.rows()); //50 pct de la masse
  VectorXd quant75(meanf.rows());
  for(int i=0;i<meanf.rows();i++){
    VectorXd R=fvalues.row(i);
    quant2p5(i)=R(int(0.025*R.size()));
    quant97p5(i)=R(int(0.975*R.size()));
    quant25(i)=R(int(0.25*R.size()));
    quant75(i)=R(int(0.75*R.size()));
  }
  ofstream ofile(filename);
  for(int i=0;i<meanf.rows();i++){
    ofile << X(i) <<" "<< meanf(i) << " " << quant25(i) << " " << quant75(i) << " " << quant2p5(i) << " " << quant97p5(i) << endl;
  }
  ofile.close();
}

void Density::WritePriorPredictions(VectorXd const & X, string const & filename, default_random_engine & generator) {
  //prédictions à priori, avec un prior uniforme sur les paramètres. Je ne sais pas encore traiter z, donc je laisse comme ça.
  uniform_real_distribution<double> distU(0,1); 
  vector<VectorXd> samples=m_samples;
  for (int i=0;i<5000;i++){
    VectorXd X(m_samples[0].size());
    for(int j=0;j<X.size();j++){
      X(j)=distU(generator);
    }
    m_samples[i]=X;    
  }
  WritePredictionsF(X,filename);
  m_samples=samples;
}

void Density::WritePriorPredictionsF(VectorXd const & X, string const & filename, vector<VectorXd> const & prior_sample) {
  //prédictions à priori, avec un prior uniforme sur les paramètres. Je ne sais pas encore traiter z, donc je laisse comme ça.
  //avec prior normal !!!!
  vector<VectorXd> samples=m_samples;
  m_samples=prior_sample;
  WritePredictionsF(X,filename);
  m_samples=samples;
}

void Density::WriteFinePredictions(VectorXd const & X, string const & filename) const{
  //écriture des prédictions de F, avec theta selon la postérieure. Les intervalles de confiance sont affichés d'après la vraie distribution des valeurs de f (et non en calculant la variance).
  //une colonne sur 2 correspond à F+Z avec le même exercice (prédiction moyenne du GP).
  //la première ligne du fichier correspond à la vérité expérimentale.
  cout << "computing fine predictions with " << m_samples.size() << " samples..." << endl;
  list<int> l={3,20,35};
  cout << "chosen points : ";
  for (int i:l){cout << i << " ";}
  cout << endl;
  int explength=m_data_exp[0].Value().size(); //40 points je crois.
  VectorXd expvalues=m_data_exp[0].Value();
  MatrixXd fvalues(explength,m_samples.size());
  for(int i=0;i<m_samples.size();i++){
    fvalues.col(i)=m_model(X,m_samples[i]);
  }
  MatrixXd zvalues(explength,m_samples.size());
  for(int i=0;i<m_samples.size();i++){
    zvalues.col(i)=meanZCondTheta(X,m_samples[i],m_hparsofsamples[i]);
  }
  MatrixXd selected_values(2*l.size(),m_samples.size());
  auto it=l.begin();
  for(int i=0;i<l.size();i++){
    selected_values.row(2*i)=fvalues.row(*it);
    selected_values.row(2*i+1)=fvalues.row(*it)+zvalues.row(*it);
    it++;
  }
  ofstream ofile(filename);
  //on écrit la vérité
  it=l.begin();
  for(int i=0;i<l.size();i++){
      ofile << expvalues(*it) << " " << expvalues(*it) << " ";
      it++;
  }
  ofile << endl;
  //on écrit les prédictions.
  for(int i=0;i<m_samples.size();i++){
    for(int j=0;j<2*l.size();j++){
      ofile << selected_values(j,i) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

void Density::WriteFinePriorPredictions(VectorXd const & X, string const & filename, default_random_engine & generator) {
  //prédictions à priori, avec un prior uniforme sur les paramètres. Je ne sais pas encore traiter z, donc je laisse comme ça.
  uniform_real_distribution<double> distU(0,1); 
  vector<VectorXd> samples=m_samples;
  for (int i=0;i<m_samples.size();i++){
    VectorXd X(m_samples[0].size());
    for(int j=0;j<X.size();j++){
      X(j)=distU(generator);
    }
    m_samples[i]=X;    
  }
  WriteFinePredictions(X,filename);
  m_samples=samples;
}

double Density::AIC()const{
  //on prend le couple theta/hpars au max. de likelihood.
  int i=max_ll();
  return -2*(loglikelihood_theta(m_samples[i],m_hparsofsamples[i])-m_samples[i].size());
}

double Density::DIC()const{
  //calcul de la moyenne à posteriori.
  VectorXd theta_post=VectorXd::Zero(m_samples[0].size());
  VectorXd hpars_post=VectorXd::Zero(m_hparsofsamples[0].size());
  double logs=0;
  for(int i=0;i<m_samples.size();i++){
    theta_post+=m_samples[i];
    hpars_post+=m_hparsofsamples[i];
    logs+=loglikelihood_theta(m_samples[i],m_hparsofsamples[i]);
  }
  theta_post/=m_samples.size();
  hpars_post/=m_samples.size();
  logs/=m_samples.size();
  double dic =2*loglikelihood_theta(theta_post,hpars_post)-4*logs;
  return dic;
}

double DensityOpt::DIC()const{
  //calcul de la moyenne à posteriori.
  //le hpars post est l'optimal (et non la moyenne des samples)
  VectorXd theta_post=VectorXd::Zero(m_samples[0].size());
  double logs=0;
  for(int i=0;i<m_samples.size();i++){
    theta_post+=m_samples[i];
    logs+=loglikelihood_theta(m_samples[i],m_hparsofsamples[i]);
  }
  theta_post/=m_samples.size();
  VectorXd hpars_post=HparsOpt(theta_post,0.5*(m_lb_hpars+m_ub_hpars),10);
  logs/=m_samples.size();
  double dic =2*loglikelihood_theta(theta_post,hpars_post)-4*logs;
  return dic;
}

double Density::WAIC2()const{
  //on calcule les vraisemblances pour toutes les observations individuelles, pour tous les samples.
  MatrixXd M(m_Xprofile.size(),m_samples.size());
  for(int i=0;i<m_samples.size();i++){
    VectorXd theta=m_samples[i];
    VectorXd hpars=m_hparsofsamples[i];
    VectorXd obs=m_data_exp[0].Value()-m_model(m_Xprofile,theta)-m_priormean(m_Xprofile,hpars);
    for(int j=0;j<m_Xprofile.size();j++){
      M(j,i)=loglikelihood_obs_i(obs,j,hpars);
    }
  }
  //calcul de la variance par colonnes.
  auto calc_var=[](VectorXd const &v){
    double mean=v.mean();
    double var=0;
    for(int i=0;i<v.size();i++){var+=pow(v(i)-mean,2);}
    return var/v.size();
  };
  double pwaic2=0;
  for(int i=0;i<m_Xprofile.size();i++){
    pwaic2+=calc_var(M.row(i).transpose());
  }
  int i=max_ll();
  return -2*(loglikelihood_theta(m_samples[i],m_hparsofsamples[i])-pwaic2);
}



DensityOpt::DensityOpt(Density const & d) : Density(d){
  m_samples.clear();
  m_hparsofsamples.clear();
  m_allmcmcsamples.clear();
}

VectorXd DensityOpt::HparsOpt(VectorXd const & theta, VectorXd const & hpars_guess, double max_time) const{
  VectorXd guess=hpars_guess;
  //guess(2)=1e-3; //modif nico
  //calcul des données f-theta
  VectorXd obsmtheta(m_Xprofile.size());
  VectorXd Yexp=m_data_exp[0].Value();
  VectorXd Fpred=m_model(m_data_exp[0].GetX(),theta);
  obsmtheta=Yexp-Fpred; //la priormean sera mise dans optfuncopt
  auto p=make_pair(&obsmtheta,this);
  //int fin=optroutine_withgrad(optfuncOpt_withgrad,&p,guess,m_lb_hpars,m_ub_hpars,max_time);
  //version sans gradient quand la prior mean est compliquée...
  //on optimise les paramètres du noyau
  int fin=optroutine(optfuncOpt_nograd,&p,guess,m_lb_hpars,m_ub_hpars,max_time);
  return guess;
}
VectorXd DensityOpt::HparsOpt_quicknograd(VectorXd const & theta, VectorXd const & hpars_guess, double max_time) const{
  VectorXd guess=hpars_guess;
  //on fait une simple optimisation locale, et avec le gradient.
  //guess(2)=1e-3; //modif nico
  //calcul des données f-theta
  VectorXd obsmtheta(m_Xprofile.size());
  VectorXd Yexp=m_data_exp[0].Value();
  VectorXd Fpred=m_model(m_data_exp[0].GetX(),theta);
  obsmtheta=Yexp-Fpred; //la priormean sera mise dans optfuncopt
  auto p=make_pair(&obsmtheta,this);
  //int fin=optroutine_withgrad(optfuncOpt_withgrad,&p,guess,m_lb_hpars,m_ub_hpars,max_time);
  //version sans gradient quand la prior mean est compliquée...
  //on optimise les paramètres du noyau
  int fin=optroutine_lightnograd(optfuncOpt_nograd,&p,guess,m_lb_hpars,m_ub_hpars,max_time);
  return guess;
}



VectorXd DensityOpt::HparsOpt_quickwithgrad(VectorXd const & theta, VectorXd const & hpars_guess, double max_time) const{
  VectorXd guess=hpars_guess;
  //on fait une simple optimisation locale, et avec le gradient.
  //guess(2)=1e-3; //modif nico
  //calcul des données f-theta
  VectorXd obsmtheta(m_Xprofile.size());
  VectorXd Yexp=m_data_exp[0].Value();
  VectorXd Fpred=m_model(m_data_exp[0].GetX(),theta);
  obsmtheta=Yexp-Fpred; //la priormean sera mise dans optfuncopt
  auto p=make_pair(&obsmtheta,this);
  //int fin=optroutine_withgrad(optfuncOpt_withgrad,&p,guess,m_lb_hpars,m_ub_hpars,max_time);
  //version sans gradient quand la prior mean est compliquée...
  //on optimise les paramètres du noyau
  int fin=optroutine_lightwithgrad(optfuncOpt_withgrad,&p,guess,m_lb_hpars,m_ub_hpars,max_time);
  return guess;
}


double DensityOpt::optfuncOpt_nograd(const std::vector<double> &x, std::vector<double> &grad, void *data){
  //on inclut l'uncertainty X ici.
  /* fonction à optimiser pour trouver les hpars optimaux. Normelemnt l'optimisation est seulement en dimension 2 car l'input uncertainty ne joue pas.*/
  //cast du null pointer
  pair<const VectorXd*,const DensityOpt*> *p=(pair<const VectorXd*,const DensityOpt*> *) data;
  const VectorXd *obs=p->first;
  const DensityOpt *d=p->second;
  VectorXd hpars=VtoVXD(x);
  VectorXd obsmodif=*obs;
  const vector<VectorXd> *xconv=d->GetXconverted();
  LDLT<MatrixXd> ldlt(d->Gamma(*xconv,hpars)+d->IncX(*xconv));
  double ll=d->loglikelihood_fast(obsmodif,ldlt);
  double lp=d->m_logpriorhpars(hpars);
  //cout << "opt1. hpars testés : " << hpars.transpose() << endl;
  //cout << "ll :" << ll << endl;
  return ll+lp;
};

double DensityOpt::optfuncOpt_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* fonction à optimiser pour trouver les hpars optimaux.*/
  //écrite pour 2 hyperparamètres.
  //cast du null pointer
  pair<const VectorXd*,const DensityOpt*> *p=(pair<const VectorXd*,const DensityOpt*> *) data;
  const VectorXd *obs=p->first; //contient déjà yobs-ftheta.
  const DensityOpt *d=p->second;
  VectorXd hpars=VtoVXD(x);
  VectorXd obsmodif=*obs;
  const vector<VectorXd> *xconv=d->GetXconverted();
  LDLT<MatrixXd> ldlt(d->Gamma(*xconv,hpars)+ d->Get_IncX()); //incX ici
  double ll=d->loglikelihood_fast(obsmodif,ldlt);
  double lp=d->m_logpriorhpars(hpars);
  //calcul des matrices des gradients
  if(!grad.size()==0){
    int nd=xconv->size();
    MatrixXd DG1=MatrixXd::Zero(nd,nd);
    MatrixXd DG2=MatrixXd::Zero(nd,nd);
    for(int i=0; i<nd; i++){
      for(int j=i; j<nd; j++){
        DG1(i,j)=d->m_DKernel1((*xconv)[i],(*xconv)[j],hpars);
        DG2(i,j)=d->m_DKernel2((*xconv)[i],(*xconv)[j],hpars);
        if(i!=j){
          DG1(j,i) = DG1(i,j);
          DG2(j,i) = DG2(i,j);
        }
      }
    }
    MatrixXd Kinv=ldlt.solve(MatrixXd::Identity(nd,nd));
    VectorXd alpha=Kinv*obsmodif;
    MatrixXd aat=alpha*alpha.transpose();
    grad[0]=0.5*((aat-Kinv)*DG1).trace(); // pas de prior
    grad[1]=0.5*((aat-Kinv)*DG2).trace(); // pas de prior non plus
  }
  return ll+lp;
};

VectorXd DensityOpt::EvaluateHparOpt(VectorXd const & theta) const {
  //évaluation de l'hpar optimal par les GPs. On utilise uniquement la prédiction moyenne.
  int nmodes=m_vgp_hpars_opti.size();
  VectorXd meansgps(nmodes);
  for(int i=0;i<nmodes;i++){
    meansgps(i)=m_vgp_hpars_opti[i].EvalMean(theta);
  }
  VectorXd Ymean=m_featureMeans+m_VP*m_Acoefs*meansgps;
  return Ymean;
}

std::vector<Eigen::VectorXd> DensityOpt::Compute_optimal_hpars(double max_time,string filename){
  //calcul de tous les hpars optimaux sur m_grid, et rangement dans m_hpars_opti.
  //on les écrit ensuite dans le fichier correspondant.
  m_hpars_opti.clear();
  VectorXd hpars_guess=0.5*(m_lb_hpars+m_ub_hpars);
  const vector<VectorXd> *grid=GetGrid();
  auto begin=chrono::steady_clock::now();
  transform(grid->begin(),grid->end(),back_inserter(m_hpars_opti),[&hpars_guess,this,max_time](VectorXd const & theta) mutable {
    VectorXd hpars_opt=HparsOpt(theta, hpars_guess,max_time);
    hpars_guess=hpars_opt; //warm restart
    AUGDATA dat; dat.SetX(theta); dat.SetValue(hpars_opt);
    return dat;
  });
  auto end=chrono::steady_clock::now();
  //affichage dans un fichier
  ofstream ofile(filename);
  for(AUGDATA const &d:m_hpars_opti){
    VectorXd X=d.GetX();
    VectorXd hpars=d.Value();
    //cout << "theta : " << X.transpose() << endl;
    //cout << "hparsopt : " << hpars.transpose() << endl;
    for(int i=0;i<X.size();i++){
      ofile << X(i) << " ";
    }
    for(int i=0;i<hpars.size();i++){
      ofile << hpars(i) << " ";
    }
    ofile << endl;
    }
  ofile.close();
  //calcul de quelques statistiques sur ces hpars optimaux obtenus. moyenne, et matrice de covariance des données.
  VectorXd mean=VectorXd::Zero(m_dim_hpars);
  MatrixXd SecondMoment=MatrixXd::Zero(m_dim_hpars,m_dim_hpars);
  double mean_lp=0;
  for(AUGDATA const & a:m_hpars_opti){
    mean_lp+=loglikelihood_theta(a.GetX(),a.Value());
    mean+=a.Value();
    SecondMoment+=a.Value()*a.Value().transpose();
  }
  mean_lp/=m_hpars_opti.size();
  mean/=m_hpars_opti.size();
  SecondMoment/=m_hpars_opti.size();
  MatrixXd Var=SecondMoment-mean*mean.transpose();
  cout << " fin de calcul des hpars opti sur le grid. Moyenne : " << mean.transpose() << endl;
  cout << "moyenne des logposts:" << mean_lp << endl;
  cout << "temps de calcul : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;
  //on resize les hpars dans un vector<VectorXd> pour les renvoyer
  vector<VectorXd> hparsfound;
  for(AUGDATA const &d:m_hpars_opti){
    hparsfound.push_back(d.Value());
  }
  return hparsfound;

}

vector<VectorXd> DensityOpt::Return_optimal_hpars(double max_time) const{
  //calcul de tous les hpars optimaux sur m_grid. 
  vector<VectorXd> res;
  VectorXd hpars_guess=0.5*(m_lb_hpars+m_ub_hpars);
  const vector<VectorXd> *grid=GetGrid();
  auto begin=chrono::steady_clock::now();
  transform(grid->begin(),grid->end(),back_inserter(res),[&hpars_guess,this,max_time](VectorXd const & theta) mutable {
    VectorXd hpars_opt=HparsOpt(theta, hpars_guess,max_time);
    hpars_guess=hpars_opt; //warm restart
    return hpars_opt;
  });
  auto end=chrono::steady_clock::now();
  cout << "fin des optimisations. temps : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s." << endl;
  return res;
}

void DensityOpt::Test_hGPs(int npoints, double max_time) {
  //calcul d'un nouveau grid de thetas, optimisation sur ces points, et évaluation de la qualité de prédiction des GPs sur ces points.
  default_random_engine generator; generator.seed(16);
  int nthetas=npoints;
  auto ps=[](MatrixXd const &A, MatrixXd const &B)->double{
    return (A.transpose()*B).trace();
  };
  VectorXd hpars_guess=0.5*(m_lb_hpars+m_ub_hpars);
  uniform_real_distribution<double> distU(0,1); 
  vector<VectorXd> newgrid; //taille nthetas
  MatrixXd Hopt(m_dim_hpars,nthetas);
  if(m_newgrid.size()==0){
    //construction des valeurs
    for(int i=0;i<nthetas;i++){
      VectorXd t(3);
      t << distU(generator),distU(generator),distU(generator);
      newgrid.push_back(t);
    }
    for(int i=0;i<nthetas;i++){
      Hopt.col(i)=HparsOpt(newgrid[i],hpars_guess,max_time);
    }
    m_newgrid=newgrid;
    m_Hopt_newgrid=Hopt;
  }
  else{
    //valeurs précédemment construites
    newgrid=m_newgrid;
    Hopt=m_Hopt_newgrid;
  }
  MatrixXd Hoptpred(m_dim_hpars,nthetas);
  for(int i=0;i<nthetas;i++){
    Hoptpred.col(i)=EvaluateHparOpt(newgrid[i]);
  }
  MatrixXd Hoptv=Hopt.row(0);
  MatrixXd Hoptpredv=Hoptpred.row(0);
  double erredm=sqrt(ps(Hoptv-Hoptpredv,Hoptv-Hoptpredv)/ps(Hoptv,Hoptv));
  Hoptv=Hopt.row(1);
  Hoptpredv=Hoptpred.row(1);
  double errexp=sqrt(ps(Hoptv-Hoptpredv,Hoptv-Hoptpredv)/ps(Hoptv,Hoptv)); 
  cout << "erreur relative des hGPs sur le grid de validation : edm : " << erredm*100 << " pct,lcor : " << errexp*100 <<  endl;
}

Eigen::VectorXd DensityOpt::Test_hGPs_on_sample(std::vector<Eigen::VectorXd> const & theta_ref,std::vector<Eigen::VectorXd> const & hpars_ref) const{
  //test de la performance des hgps sur un sample donné.
  if(theta_ref.size()!=hpars_ref.size()){cerr << "erreur de taille" << endl;}
  cout << "test hgps on a sample size " << theta_ref.size() << endl;
  vector<VectorXd> approx_hpars(theta_ref.size());
  transform(theta_ref.begin(),theta_ref.end(),approx_hpars.begin(),[this](VectorXd const &theta){
    return EvaluateHparOpt(theta);
  });
  vector<double> true_logliks(theta_ref.size());
  vector<double> approx_logliks(theta_ref.size());
  for(int i=0;i<theta_ref.size();i++){
    true_logliks[i]=loglikelihood_theta(theta_ref[i],hpars_ref[i])+m_logpriorhpars(hpars_ref[i]);
    approx_logliks[i]=loglikelihood_theta(theta_ref[i],approx_hpars[i])+m_logpriorhpars(approx_hpars[i]);
  }
    //calcul de l'erreur moyenne en log-vraisemblance
  double errmoy=0;
  for(int i=0;i<true_logliks.size();i++){errmoy+=pow(true_logliks[i]-approx_logliks[i],2);
  
  if(true_logliks[i]-approx_logliks[i]>1){cerr << "erreur ll true sous-estimation. " <<endl;
  cerr<< "hpars true : " << hpars_ref[i].transpose()<< endl;
  cerr<< "hpars approx : " <<approx_hpars[i].transpose()<< endl;
  cerr<< "ll true : " << true_logliks[i]<< ", ll approx : "<< approx_logliks[i] <<endl;
  cerr <<"in bounds hpars approx ? "<<in_bounds_hpars(approx_hpars[i])<< endl;}
  /*
  if(approx_logliks[i]>0.1+true_logliks[i]){cerr << "erreur ll true surestimation. " <<endl;
  cerr<< "hpars true : " << hpars_ref[i].transpose()<< endl;
  cerr<< "hpars approx : " <<approx_hpars[i].transpose()<< endl;
  cerr<< "ll true : " << true_logliks[i]<< ", ll approx : "<< approx_logliks[i] <<endl;
  cerr <<"in bounds hpars approx ? "<<in_bounds_hpars(approx_hpars[i])<< endl;}*/
  }
  //valeur rms
  errmoy=sqrt(errmoy/true_logliks.size()); //erreur moyenne en log-probabilité.
  //calcul de l'erreur moyenne sur chaque hpar
  VectorXd errmoy_hpars=VectorXd::Zero(m_dim_hpars);
  VectorXd cumsum_hpars=VectorXd::Zero(m_dim_hpars);
  for(int i=0;i<true_logliks.size();i++){
    errmoy_hpars.array()+=((hpars_ref[i]-approx_hpars[i]).array().square());
    cumsum_hpars.array()+=((hpars_ref[i]).array().square());
  }
  errmoy_hpars.array()=100*(errmoy_hpars.cwiseQuotient(cumsum_hpars)).array().sqrt(); //assignation à array ou au vectorxd directement ?
  VectorXd res(m_dim_hpars+1);
  for(int i=0;i<m_dim_hpars;i++){
    res(i)=errmoy_hpars(i);
  }
  res(m_dim_hpars)=errmoy;
  cout << "testou : " << endl;
  cout << "hpars true : " << hpars_ref[50].transpose() << endl;
  cout << "hpars estimated : " << approx_hpars[50].transpose() << endl;
  cout << "ll true : " << true_logliks[50] << endl;
  cout << "ll estimated : " << approx_logliks[50] << endl;
  return res;
}

void DensityOpt::BuildHGPs_noPCA(double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), MatrixXd const & Bounds_hpars_GPs,VectorXd const & Hpars_guess_GPs){
  //initialisation des HGPs sans faire de PCA (un hGP par hyperparamètre directement)
  //on normalise quand même la valeur des GPs pour qu'ils soient sur le même range/
  //récupération des data pour les GPs.
  if(m_hpars_opti.size()==0){cerr << "erreur : hpars optis non calculés !" << endl;}
  cout << "construction des hGPs individuels sans PCA : " << endl;
  int nmodes=m_dim_hpars;
  MatrixXd U(m_dim_hpars,m_hpars_opti.size()); //matrice des données
  MatrixXd P(m_dim_pars,m_hpars_opti.size()); //matrice des thetas
  for(int i=0;i<m_hpars_opti.size();i++){
    U.col(i)=m_hpars_opti[i].Value();
    P.col(i)=m_hpars_opti[i].GetX();
  }
  m_featureMeans=U.rowwise().mean();
  U=U.colwise()-m_featureMeans;
  //calcul des STDs de chaque feature et normalisation.
  VectorXd stds=VectorXd::Zero(m_dim_hpars);
  for(int i=0;i<m_dim_hpars;i++){
    stds(i)=sqrt(U.row(i).array().square().sum()/m_hpars_opti.size());
    if(stds(i)==0.){stds(i)+=0.01*m_featureMeans(i); cout << "correction d'hpars constants." << endl;}
  }
  m_VP=MatrixXd::Identity(m_dim_hpars,m_dim_hpars);
  m_Acoefs=stds.asDiagonal();
  MatrixXd normedA=m_Acoefs.inverse()*U;
  //on met sous forme vector<vector> DATA pour la passer aux GPs
  vector<vector<DATA>> vd(nmodes);
  for(int j=0;j<nmodes;j++){
    vector<DATA> v(m_hpars_opti.size());
    for(int i=0;i<m_hpars_opti.size();i++){
      DATA dat; dat.SetX(P.col(i)); dat.SetValue(normedA(j,i));
      v[i]=dat;
    }
    vd[j]=v;
  }
  vector<GP> vgp(nmodes);
  for(int i=0;i<nmodes;i++){
    GP gp(Kernel_GP);
    gp.SetData(vd[i]);
    gp.SetGP(Hpars_guess_GPs);
    vgp[i]=gp;
  }
  m_vgp_hpars_opti=vgp;
  m_Bounds_hpars_GPs=Bounds_hpars_GPs;
}

VectorXd DensityOpt::opti_1gp(int i, VectorXd & hpars_guess){
  cout << "optimisation du gp pour hpars numero " << i <<endl;
  auto begin=chrono::steady_clock::now();
  m_vgp_hpars_opti[i].OptimizeGP(myoptfunc_gp,&m_Bounds_hpars_GPs,&hpars_guess,hpars_guess.size());
  auto end=chrono::steady_clock::now();
  hpars_guess=m_vgp_hpars_opti[i].GetPar();
  cout  << "par after opt : " << hpars_guess.transpose() << endl;
  cout << "temps pour optimisation : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;
  return hpars_guess;
}

void DensityOpt::opti_allgps(VectorXd const &hpars_guess){
  if(m_vhpars_pour_gp.empty()){
    for(int i=0;i<m_vgp_hpars_opti.size();i++){
      m_vhpars_pour_gp.push_back(hpars_guess);
    }
  }
  if(!m_vhpars_pour_gp.size()==m_vgp_hpars_opti.size()){cerr << "problem in vhpars size" << endl;}
  for(int i=0;i<m_vgp_hpars_opti.size();i++){
    VectorXd h=opti_1gp(i,m_vhpars_pour_gp[i]);
    m_vhpars_pour_gp[i]=h;
  }
}

void DensityOpt::update_hGPs_noPCA(vector<VectorXd> const &new_thetas,vector<VectorXd> const &new_hpars,double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), MatrixXd const & Bounds_hpars_GPs,VectorXd const & Hpars_guess_GPs){
  //on fait une optimisation fine sur les new thetas, puis on les rajoute aux hGPs. On optimise les hyperparamètres avec le tout.
  cout << "updating hGPs with " << new_thetas.size() << "new points..." << endl;
  for(int i=0;i<new_thetas.size();i++){
    AUGDATA dat; dat.SetX(new_thetas[i]),dat.SetValue(new_hpars[i]);
    m_hpars_opti.push_back(dat);
  }
  cout << "new number of points for hGPs : " << m_hpars_opti.size() << endl;
  BuildHGPs_noPCA(Kernel_GP,Bounds_hpars_GPs,Hpars_guess_GPs);
}

std::vector<Eigen::VectorXd> DensityOpt::update_hGPs_noPCA(vector<VectorXd> const &new_thetas,double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), MatrixXd const & Bounds_hpars_GPs,VectorXd const & Hpars_guess_GPs, double max_time){
  //on fait une optimisation fine sur les new thetas, puis on les rajoute aux hGPs. On optimise les hyperparamètres avec le tout.
  cout << "updating hGPs with " << new_thetas.size() << "new points..." << endl;
  vector<VectorXd> new_hpars(new_thetas.size());
  VectorXd hpars_guess=0.5*(m_lb_hpars+m_ub_hpars);
  for(int i=0;i<new_thetas.size();i++){
    new_hpars[i]=HparsOpt(new_thetas[i],hpars_guess,max_time);
  }
  for(int i=0;i<new_thetas.size();i++){
    AUGDATA dat; dat.SetX(new_thetas[i]),dat.SetValue(new_hpars[i]);
    m_hpars_opti.push_back(dat);
  }
  cout << "new number of points for hGPs : " << m_hpars_opti.size() << endl;
  BuildHGPs_noPCA(Kernel_GP,Bounds_hpars_GPs,Hpars_guess_GPs);
  return new_hpars;
}

double DensityOpt::myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
	/* This is the function you optimize for defining the GP */
	GP* proc = (GP*) data;											//Pointer to the GP
	Eigen::VectorXd p(x.size());									//Parameters to be optimized
	for(int i=0; i<(int) x.size(); i++) p(i) = x[i];				//Setting the proposed value of the parameters
	double value = proc->SetGP(p);									//Evaluate the function
	if (!grad.empty()) {											//Cannot compute gradient : stop!
		std::cout << "Asking for gradient, I stop !" << std::endl; exit(1);
	}
	return value;
};

double DensityOpt::Run_Burn_phase_MCMC_opti_expensive(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento, default_random_engine & generator, double max_time) {
  //phase de burn et calcul de la nouvelle matrice de covariance.
  //on renvoie la covariance scalée et l'état actuel.
  //on peut calculer une seule fois la matrice Gamma puisque les hpars restent identiques.
  auto compute_score=[this](VectorXd const & hpars,VectorXd const & Xtest)-> double {
    return this->loglikelihood_theta(Xtest,hpars)+this->m_logpriorpars(Xtest);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  VectorXd Xinit=Xcurrento;
  VectorXd hpars_guess=0.5*(m_lb_hpars+m_ub_hpars);
  VectorXd hpars_init=HparsOpt(Xinit,hpars_guess,max_time);
  double finit=compute_score(hpars_init,Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  VectorXd hpars_current=hpars_init;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(m_dim_pars);
  MatrixXd acc_var=MatrixXd::Zero(m_dim_pars,m_dim_pars);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      VectorXd hparscandidate=HparsOpt(Xcandidate,hpars_current,max_time);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hpars_current=hparscandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
    m_allmcmcsamples.push_back(Xcurrent);
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(m_dim_pars))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2)+1e-10*MatrixXd::Identity(m_dim_pars,m_dim_pars));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
  return acc_rate;
}

void DensityOpt::Run_MCMC_opti_expensive(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,default_random_engine & generator, double max_time){
  //MCMC à hpars variables. pas de scaling.
  cout << "running mcmc opti expensive with " << nsteps << " steps." <<endl;
  m_samples.clear(); m_hparsofsamples.clear(); m_allmcmcsamples.clear();
  auto compute_score=[this](VectorXd const & hpars,VectorXd const & Xtest)-> double {
    return this->loglikelihood_theta(Xtest,hpars)+this->m_logpriorpars(Xtest);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  Run_Burn_phase_MCMC_opti_expensive(nsteps*0.1,COV,Xinit,generator,max_time);
  VectorXd hpars_guess =(m_lb_hpars+m_ub_hpars)*0.5;
  VectorXd hpars_init=HparsOpt(Xinit,hpars_guess,max_time);
  //COV=scale_covmatrix(COV,Xinit,compute_score,0,generator,"results/diag/scaleopt.gnu");
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(hpars_init,Xinit);
  VectorXd hparscurrent=hpars_init;
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      VectorXd hparscandidate=HparsOpt(Xcandidate,hparscurrent,max_time);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      m_samples.push_back(Xcurrent);
      m_hparsofsamples.push_back(hparscurrent);
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}

double DensityOpt::Run_Burn_phase_MCMC_opti_hGPs(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento, default_random_engine & generator) {
  //phase de burn et calcul de la nouvelle matrice de covariance.
  //on renvoie la covariance scalée et l'état actuel.
  //on peut calculer une seule fois la matrice Gamma puisque les hpars restent identiques.
  auto compute_score=[this](VectorXd const & hpars,VectorXd const & Xtest)-> double {
    return this->loglikelihood_theta(Xtest,hpars)+this->m_logpriorpars(Xtest);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  VectorXd Xinit=Xcurrento;
  VectorXd hpars_init=EvaluateHparOpt(Xinit);
  double finit=compute_score(hpars_init,Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  VectorXd hpars_current=hpars_init;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(m_dim_pars);
  MatrixXd acc_var=MatrixXd::Zero(m_dim_pars,m_dim_pars);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      VectorXd hparscandidate=EvaluateHparOpt(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hpars_current=hparscandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
    m_allmcmcsamples.push_back(Xcurrent);
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(m_dim_pars))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2)+1e-10*MatrixXd::Identity(m_dim_pars,m_dim_pars));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
  return acc_rate;
} 

void DensityOpt::Run_MCMC_opti_hGPs(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,default_random_engine & generator){

  cout << "running mcmc opti_hgps with " << nsteps << " steps." <<endl;
  //MCMC à hpars variables. pas de scaling.
  m_samples.clear(); m_hparsofsamples.clear(); m_allmcmcsamples.clear();
  auto compute_score=[this](VectorXd const & hpars,VectorXd const & Xtest)-> double {
    return this->loglikelihood_theta(Xtest,hpars)+this->m_logpriorpars(Xtest);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  Run_Burn_phase_MCMC_opti_hGPs(nsteps*0.1,COV,Xinit,generator);
  VectorXd hpars_init=EvaluateHparOpt(Xinit);
  //COV=scale_covmatrix(COV,Xinit,compute_score,0,generator,"results/diag/scaleopt.gnu");
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(hpars_init,Xinit);
  VectorXd hparscurrent=hpars_init;
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      VectorXd hparscandidate=EvaluateHparOpt(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      m_samples.push_back(Xcurrent);
      m_hparsofsamples.push_back(hparscurrent);
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}


void DensityOpt::WritehGPs(string const & filename)const{
  //écriture des hyperparamètres optimaux des GPs
  //écrit pour 3 GPs avec 7 hpars chacun.
  ofstream ofile(filename);
  for(int i=0;i<m_vhpars_pour_gp.size();i++){
    for(int j=0;j<m_vhpars_pour_gp[0].size();j++){
      ofile << m_vhpars_pour_gp[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

void DensityOpt::ReadhGPs(string const & filename){
  //ne marche pas avec priormean
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      VectorXd hpars(7);
      for(int i=0;i<7;i++){
        hpars(i)=stod(words[i]);
      }
      m_vhpars_pour_gp.push_back(hpars);
    }
    cout << "number of GPs loaded : " << m_vhpars_pour_gp.size() << endl;
  }
  else{
    cerr << "empty file : " << filename <<endl;
  }
  if(m_vgp_hpars_opti.empty()){cerr << "erreur : hGPs opti non initialisés" << endl;}
  //on applique les hyperparamètres aux hGPs.
  for(int i=0;i<m_vgp_hpars_opti.size();i++){
    m_vgp_hpars_opti[i].SetGP(m_vhpars_pour_gp[i]);
  }
  ifile.close();
}

double DensityOpt::EstimatePredError(VectorXd const & theta) const {
  //estimateur de l'erreur de prédiction des hGPs en un point theta. Fait à partir de la variance de prédiction des hGPs.
  VectorXd var=EvaluateVarHparOpt(theta);
  VectorXd stds_scale=m_Acoefs.diagonal().array().square();
  return var.cwiseQuotient(stds_scale).array().sum();
}

VectorXd DensityOpt::EvaluateVarHparOpt(VectorXd const & theta) const {
  //estimateur de l'erreur de prédiction des hGPs en un point theta. Fait à partir de la variance de prédiction des hGPs.
  int nmodes=m_vgp_hpars_opti.size();
  VectorXd vargps(nmodes); //on stocke les variances de prédiction de chaque GP.
  for(int i=0;i<nmodes;i++){
    vargps(i)=m_vgp_hpars_opti[i].Eval(theta)(1);
  }
  //variances de prédiction pour chaque hyperparamètre
  MatrixXd scaledVP=(m_Acoefs*m_VP).array().square();
  VectorXd Variances=scaledVP*vargps;
  //on renvoie la somme des variances, sans scaling pour le moment car je ne sais pas comment faire.
  
  return Variances;
}

Densities::Densities(){}

Densities::Densities(Densities const & d){
  m_Densities_vec=d.m_Densities_vec;
  m_dim=d.m_dim;
  m_dim_pars=d.m_dim_pars;
  m_samples=d.m_samples;
  m_hparsofsamples_v=d.m_hparsofsamples_v;
  m_allmcmcsamples=d.m_allmcmcsamples;
  m_logpriorpars=d.m_logpriorpars;
}

Densities::Densities(vector<Density> const &v){
  //seule manière d'initialiser.
  m_Densities_vec=v;
  m_dim=v.size();
}
std::vector<Eigen::LDLT<Eigen::MatrixXd>> Densities::compute_ldlts(std::vector<Eigen::VectorXd> const & hpars_v){
  //calcule les ldlts pour chacune des densités aux hpars hpars.
  if(!hpars_v.size()==m_dim){cerr << "erreur de dimension !" << endl;}
  vector<LDLT<MatrixXd>> v(m_dim);
  for(int i=0;i<m_dim;i++){
    v[i]=m_Densities_vec[i].GetLDLT(hpars_v[i]);
  }
  return v;
}

vector<VectorXd> Densities::HparsKOH_separate(vector<VectorXd> const & hpars_guess_vec, double max_time) {
  auto begin=chrono::steady_clock::now();
  if(!hpars_guess_vec.size()==m_dim){cerr << "erreur de dimension !" << endl;}
  vector<VectorXd> hpars(m_dim); 
  for(int i=0;i<m_dim;i++){
    VectorXd hpars_guess=hpars_guess_vec[i];
    hpars[i]=m_Densities_vec[i].HparsKOH(hpars_guess,max_time);
  }
  auto end=chrono::steady_clock::now();
  cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s." << endl;
  return hpars;
}

vector<VectorXd> Densities::HparsKOH_pooled(vector<VectorXd> const & hpars_guess_vec, double max_time) {
  auto begin=chrono::steady_clock::now();
  //optimisation KOH poolée entre toutes les densités.
  if(!hpars_guess_vec.size()==m_dim){cerr << "erreur de dimension !" << endl;}
  vector<MatrixXd> residus_v(m_dim);
  for(int i=0;i<m_dim;i++){
    VectorXd expvalues=m_Densities_vec[i].GetExpData()->at(0).Value();
    VectorXd xvalues=m_Densities_vec[i].GetXprofile();
    MatrixXd Residustheta(expvalues.size(),m_Densities_vec[i].GetGrid()->size());
    for(int j=0;j<Residustheta.cols();j++){
      VectorXd theta=m_Densities_vec[i].GetGrid()->at(j);
      Residustheta.col(j)=expvalues-m_Densities_vec[i].EvaluateModel(xvalues,theta);
    }
    residus_v[i]=Residustheta;
  }
  auto tp=make_tuple(&residus_v,this);
  //création des bornes des hpars et du guess. tout est fait en vector pour pouvoir gérer les tailles sans s'embêter.
  vector<double> lb_hpars,ub_hpars,guess;
  for(int i=0;i<m_dim;i++){
    auto p=m_Densities_vec[i].GetBoundsHpars();
    for(int j=0;j<hpars_guess_vec[i].size();j++){
      lb_hpars.push_back(p.first(j));
      ub_hpars.push_back(p.second(j));
      guess.push_back(hpars_guess_vec[i](j));
    }
  }

  int fin=optroutine(optfuncKOH_pooled,&tp,guess,lb_hpars,ub_hpars,max_time);
  cout << "fin de l'opt koh pooled : message " << fin << endl;
  //il faut repasser guess en vector<vectorXd>.
  vector<VectorXd> ret=hpars_guess_vec;
  int c=0;
  for(int i=0;i<m_dim;i++){
    for(int j=0;j<ret[i].size();j++){
      ret[i](j)=guess[c];
      c++;
    }
  }
  auto end=chrono::steady_clock::now();
  cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s." << endl;
  return ret;
}

int Densities::optroutine(nlopt::vfunc optfunc,void *data_ptr, vector<double> &x, vector<double> const & lb_hpars, vector<double> const & ub_hpars, double max_time){
    //routine d'optimisation sans gradient

  //paramètres d'optimisation
  double ftol_large=1e-5;
  double xtol_large=1e-3;
  double ftol_fin=1e-15;
  double xtol_fin=1e-7;
  // 1 optimiseur local et un global.
  nlopt::opt local_opt(nlopt::LD_LBFGS,x.size());
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
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  //on relance une opti locale à partir du max. trouvé.
  local_opt.set_ftol_rel(ftol_fin);
  local_opt.set_xtol_rel(xtol_fin);
  fin=local_opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  return fin;
}

double Densities::optfuncKOH_pooled(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* fonction à optimiser pour trouver les hpars koh.*/
  auto ptp=(tuple<const vector<MatrixXd> *,Densities*>*) data; //cast
  auto tp=*ptp;
  const vector<MatrixXd> *Residus_v=get<0>(tp);
  Densities *d=get<1>(tp);
  const vector<Density> * D_v=d->GetDensities_v();
  //transformer x en vector de vectorXd pour avoir tous les hpars
  vector<VectorXd> hpars_v;
  int c=0;
  for(int i=0;i<d->GetDim();i++){
    VectorXd h(D_v->at(i).GetBoundsHpars().first.size());
    for(int j=0;j<h.size();j++){
      h(j)=x[c];
      c++;
    }
    hpars_v.push_back(h);
  }
  //il faut que toutes les densités aient le même grid en theta.
  vector<LDLT<MatrixXd>> ldlt_v=d->compute_ldlts(hpars_v);
  vector<double> prob(Residus_v->at(0).cols());
  for(int i=0;i<prob.size();i++){
    double g=0;
    for(int j=0;j<d->GetDim();j++){
      VectorXd priormean=D_v->at(j).EvaluatePMean(D_v->at(j).GetXprofile(),hpars_v[j]);
      double ll=D_v->at(j).loglikelihood_fast(Residus_v->at(j).col(i)-priormean,ldlt_v[j]);
      g+=ll;
    }
    prob[i]=g;
  }
  double logvstyp=-200;
  //normalisation et passage à l'exponentielle.
  //le max doit être le même sur toutes les itérations... d'où la nécessité de le définir à l'extérieur !!!
  for(int i=0;i<prob.size();i++){
    //passage à l'exponentielle.
    //on suppose que le logprior des paramètres est le même pour tous, et correspond à celui de la première densité.
    double d=prob[i];
    VectorXd theta=D_v->at(0).GetGrid()->at(i);
    double logprior=D_v->at(0).EvaluateLogPPars(theta);
    double f=exp(d+logprior-logvstyp);
    if(isinf(f)){cerr << "erreur myoptfunc_koh : infini. Valeur de la fonction : " << d+logprior << endl;}
    prob[i]=f;
  }

  //calcul de l'intégrale. suppose un grid régulier. modifier si on veut un prior pour les paramètres.
  //multiplication des priors pour chaques hyperparamètres !
  double res=accumulate(prob.begin(),prob.end(),0.0); res/=prob.size();
  for(int i=0;i<d->GetDim();i++){
  res*=exp(D_v->at(i).EvaluateLogPHpars(hpars_v[i]));}
  return res;
};


double Densities::loglikelihood_theta(VectorXd const & theta, vector<VectorXd> const & hpars_v)const{
  double res=0;
  for(int i=0;i<m_dim;i++){
    res+=m_Densities_vec[i].loglikelihood_theta(theta,hpars_v[i]);
  }
  return res;
}

double Densities::loglikelihood_theta_fast(VectorXd const & theta, vector<VectorXd> const & hpars_v,vector<LDLT<MatrixXd>> const & ldlt_v) const {
  double res=0;
  for(int i=0;i<m_dim;i++){
    double ll=m_Densities_vec[i].loglikelihood_theta_fast(theta,hpars_v[i],ldlt_v[i]);
    res+=ll;
  }
  return res;
}

bool Densities::in_bounds_pars(Eigen::VectorXd const & pars) const{
  //on suppose que toutes les densités on les mêmes bornes de paramètres.
  return m_Densities_vec[0].in_bounds_pars(pars);
}

double Densities::Run_Burn_phase_MCMC_fixed_hpars(int nburn, MatrixXd & COV_init,vector<VectorXd> const & hpars_v, std::vector<Eigen::LDLT<Eigen::MatrixXd>> const & ldlt_v, VectorXd & Xcurrento, default_random_engine & generator) {
  //phase de burn et calcul de la nouvelle matrice de covariance.
  //on renvoie la covariance scalée et l'état actuel.
  //on peut calculer une seule fois la matrice Gamma puisque les hpars restent identiques.
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  auto compute_score=[this,hpars_v,ldlt_v](VectorXd const & theta)-> double {
    double s=loglikelihood_theta_fast(theta,hpars_v,ldlt_v)+m_logpriorpars(theta);
    return s;
  };

  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  VectorXd Xinit=Xcurrento;
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(m_dim_pars);
  MatrixXd acc_var=MatrixXd::Zero(m_dim_pars,m_dim_pars);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
    m_allmcmcsamples.push_back(Xcurrent);
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(m_dim_pars))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2)+1e-10*MatrixXd::Identity(m_dim_pars,m_dim_pars));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
  return acc_rate;
}


void Densities::Run_MCMC_fixed_hpars(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,vector<VectorXd> const & hpars_v,default_random_engine & generator){
  //MCMC à hpars fixés.

  cout << "running mcmc fixedhpars with " << nsteps << " steps." <<endl;
  //vidons les samples juste au cas où.
  m_samples.clear(); m_hparsofsamples_v.clear(); 
  vector<vector<VectorXd>> hsubs(m_dim);
  m_hparsofsamples_v=hsubs;
  m_allmcmcsamples.clear();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  auto ldlt_v=compute_ldlts(hpars_v);
  auto compute_score=[this,hpars_v,ldlt_v](VectorXd const & theta)-> double {
    double s=loglikelihood_theta_fast(theta,hpars_v,ldlt_v)+m_logpriorpars(theta);
    return s;
  };
  MatrixXd COV=COV_init;
  MatrixXd COVPred=COV;
  Run_Burn_phase_MCMC_fixed_hpars(nsteps*0.1,COV,hpars_v,ldlt_v,Xinit,generator);
  //COV=scale_covmatrix(COV,Xinit,compute_score,0,generator,"results/diag/scalekoh.gnu");
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      m_samples.push_back(Xcurrent);
      for(int j=0;j<m_dim;j++){
        m_hparsofsamples_v[j].push_back(hpars_v[j]);
      }
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}


void Densities::Run_MCMC_adapt(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,std::vector<Eigen::VectorXd> const & hpars_v,double lambda, double gamma, default_random_engine & generator){
  //algorithme andrieu global AM with global adaptive scaling (algorithme 4)


  cout << "running mcmc koh with " << nsteps << " steps, adaptative algorithm, gamma = "<< gamma << endl;
  //MCMC à hpars variables. pas de scaling.
  m_samples.clear(); m_hparsofsamples_v.clear(); 
  vector<vector<VectorXd>> hsubs(m_dim);
  m_hparsofsamples_v=hsubs;
  m_allmcmcsamples.clear();
  double alphastar=0.234; //valeur conseillée dans l'article. taux d'acceptation optimal.

  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);

  auto ldlt_v=compute_ldlts(hpars_v);
  auto compute_score=[this,hpars_v,ldlt_v](VectorXd const & theta)-> double {
    double s=loglikelihood_theta_fast(theta,hpars_v,ldlt_v)+m_logpriorpars(theta);
    return s;
  };

   auto draw_prop=[this](double lambda,MatrixXd COV,default_random_engine & generator, normal_distribution<double> & distN ){
    //tire une proposal de matrice de cov sqrt(lambda)*COV*sqrt(lambda)
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    MatrixXd sqrtCOV=COV.llt().matrixL();
    VectorXd s=sqrt(lambda)*sqrtCOV*Step;
    return s;
  };

  auto update_params=[this,gamma,alphastar](VectorXd & mu, MatrixXd & COV,double & lambda,double alpha,VectorXd Xcurrent){
    //update les paramètres de l'algo MCMC.
    lambda*=exp(gamma*(alpha-alphastar));
    COV=COV+gamma*((Xcurrent-mu)*(Xcurrent-mu).transpose()-COV);
    COV+=1e-10*MatrixXd::Identity(m_dim_pars,m_dim_pars);
    mu=mu+gamma*(Xcurrent-mu);
  };

  MatrixXd COV=COV_init;
  cout << "cov : " <<COV << endl;
  VectorXd mu=Xinit;
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  double alpha=0;
  for(int i=0;i<nsteps;i++){
    VectorXd Xcandidate=Xcurrent+draw_prop(lambda,COV,generator,distN);
    if(in_bounds_pars(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      alpha=min(exp(fcandidate-fcurrent),1.);
      double c=distU(generator);
      if(alpha>=c){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(isnan(alpha)){alpha=1e-20;};
    update_params(mu,COV,lambda,alpha,Xcurrent);
    if(i%(nsteps/nsamples)==0 && i>0){
      m_samples.push_back(Xcurrent);
      for(int j=0;j<m_dim;j++){
        m_hparsofsamples_v[j].push_back(hpars_v[j]);
      }
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}

void Densities::WriteSamples(string const & filename)const{
  ofstream ofile(filename);
  for(int i=0;i<m_samples.size();i++){
    for(int j=0;j<m_samples[0].size();j++){
      ofile << m_samples[i](j) << " ";
    }
    for(int j=0;j<m_dim;j++){
      VectorXd h=m_hparsofsamples_v[j][i];
      for(int k=0;k<h.size();k++){
        ofile << h(k) << " ";
      }
    }
    ofile << endl;
  }
  ofile.close();
}

void Densities::WriteAllSamples(string const & filename)const{
  ofstream ofile(filename);
  for(int i=0;i<m_allmcmcsamples.size();i++){
    for(int j=0;j<m_allmcmcsamples[0].size();j++){
      ofile << m_allmcmcsamples[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
}

VectorXd Densities::Lagged_mean(vector<VectorXd> const & v,int n) const{
  int dim_mcmc=v[0].size();
  VectorXd ans=VectorXd::Zero(dim_mcmc);
  for (int i=0;i<v.size()-n;i++){
    ans+=v[i].cwiseProduct(v[i+n]);
  }
  return ans/(double (v.size()-n));
}

void Densities::Autocor_diagnosis(int nstepsmax, string const & filename) const
{
  //diagnostic d'autocorrélation, jusqu'à une distance de nstepsmax pas dans la mcmc.
  //on oublie les 10% premières steps dans le diagnostic, pour la phase de burn.
  //on centre les données
  int dim_mcmc=m_allmcmcsamples[0].size();
  int nselected=0.9*m_allmcmcsamples.size();
  vector<VectorXd> allmcmcsamples_selected(nselected);
  VectorXd mean=VectorXd::Zero(dim_mcmc);
  VectorXd var=VectorXd::Zero(dim_mcmc); //variance composante par composante
  for (int i=0;i<nselected;i++){
    allmcmcsamples_selected[i]=m_allmcmcsamples[i+m_allmcmcsamples.size()-nselected];
    mean+=allmcmcsamples_selected[i];
  }
  mean/=nselected;
  for (int i=0;i<nselected;i++){
    allmcmcsamples_selected[i]-=mean;
    var+=allmcmcsamples_selected[i].cwiseProduct(allmcmcsamples_selected[i]);
  }
  var/=nselected; //calcul de la variance composante par composante.
  ofstream ofile(filename);
  VectorXd integ=VectorXd::Zero(dim_mcmc);
  FILE* out= fopen("results/autocor.gnu","w");
  for (int i=0;i<nstepsmax;i++){
    VectorXd cor=Lagged_mean(allmcmcsamples_selected,i).cwiseQuotient(var);
    for(int j=0;j<cor.size();j++){
      ofile << cor(j) << " ";
    }
    integ+=cor;
    ofile << endl;
  }
  ofile.close();
  cout << "intégrales avec nautocor = " << nstepsmax << endl;
  cout << integ.transpose() << endl;
  cout << "maximum : " << integ.maxCoeff() << endl;
}

DensitiesOpt::DensitiesOpt(Densities const & ds):Densities(ds){
  m_DensityOpt_vec.clear();
  auto vec=ds.GetDensities_v();
  for(int i=0;i<vec->size();i++){
    DensityOpt d(vec->at(i));
    m_DensityOpt_vec.push_back(d);
  }
  m_grid=*m_DensityOpt_vec[0].GetGrid();
  //affichage temporaire pour s'assurer que le grid est bien recopié
  cout << m_grid[0].transpose() << endl;
  //cout << m_grid[50].transpose() << endl;
}



vector<vector<VectorXd>> DensitiesOpt::compute_optimal_hpars(double max_time) const {
  //renvoie les hyperparamètres optimaux calculés sur le grid.
  //calcul de ces hyperparamètres optimaux.
  //on considère que ce sont tous les mêmes noyaux.
  //on considère que chaque densité correspond bien à la liste cases.
  vector<vector<VectorXd>> res;
  for(int i=0;i<m_DensityOpt_vec.size();i++){
    vector<VectorXd> hopt=m_DensityOpt_vec[i].Return_optimal_hpars(max_time);
    res.push_back(hopt);
  }
  return res;
}

void DensitiesOpt::write_optimal_hpars(vector<vector<VectorXd>> const & hpars, vector<VectorXd> const & grid_theta,string filename) const {
  //écriture des hpars optimaux dans un fichier. on ne garde pas en mémoire à quels cas ça correspond.
  ofstream ofile(filename);
  int ncases=hpars.size();
  int ngrid=hpars[0].size(); //=grid_theta.size()
  for(int i=0;i<ngrid;i++){
    //écriture du theta
    for(int j=0;j<grid_theta[i].size();j++){
      ofile << grid_theta[i](j) << " ";
    }
    //boucle sur les cas
    for(int j=0;j<ncases;j++){
      //boucle sur la dimension des hpars
      for(int k=0;k<hpars[j][i].size();k++){
        ofile << hpars[j][i](k) << " ";
      }
    }
    ofile << endl;
  }
  ofile.close();
}

void DensitiesOpt::read_optimal_hpars(vector<vector<VectorXd>> & hparsv, vector<VectorXd> & grid_theta,string filename,int ncases, int dimhp) const {
  //lecture des hpars optimaux dans un fichier. on n'a pas l'info à quels cas ça correspond.
  //les paramètres hpars et grid_theta doivent être vides pour le bon fonctionnement.
  //pour hparsv : on veut que ça soit un vecteur de taille ncases, dont chaque élément est de taille grid_theta.size().
  grid_theta.clear();
  vector<vector<VectorXd>> hp(ncases);
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      int number_cases=ncases;
      VectorXd theta(3);
      for(int i=0;i<theta.size();i++){
        theta(i)=stod(words[i]);
      }
      grid_theta.push_back(theta);
      //lecture des hpars
      vector<VectorXd> v;
      for(int i=0;i<number_cases;i++){
        VectorXd hpars(dimhp); //a modifier...
        for(int j=0;j<dimhp;j++){
          hpars(j)=stod(words[3+j+i*dimhp]);
        }
        hp[i].push_back(hpars);
      }
    }
  }
  hparsv=hp;
  cout << "number of thetas in the file : " << grid_theta.size() << endl;
  cout << "number of cases loaded : " << hparsv.size() << endl;
}

void DensitiesOpt::update_hGPs_with_hpars(std::vector<Eigen::VectorXd> const & grid_theta,std::vector<std::vector<Eigen::VectorXd>>  const & hparsv,double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &),Eigen::MatrixXd const & Bounds_hpars_GPs,Eigen::VectorXd const & Hpars_guess_GPs){
  //construction des hGPs à partir des training points donnés.
  for(int i=0;i<hparsv.size();i++){
    m_DensityOpt_vec[i].update_hGPs_noPCA(grid_theta,hparsv[i],Kernel_GP,Bounds_hpars_GPs,Hpars_guess_GPs);
  }
}

void DensitiesOpt::opti_allgps(VectorXd const & hpars_guess){
  //optimisation de tous les hGPs de toutes les densityopt.
  for(auto &d:m_DensityOpt_vec){
    d.opti_allgps(hpars_guess);
  }
}

std::vector<Eigen::VectorXd> DensitiesOpt::HparsOpt(Eigen::VectorXd const & theta,std::vector<Eigen::VectorXd> const & hpars_guess_vec, double max_time) const{
  std::vector<Eigen::VectorXd> h=hpars_guess_vec;
  for(int i=0;i<m_dim;i++){
    h[i]=m_DensityOpt_vec[i].HparsOpt(theta,hpars_guess_vec[i],max_time);
  }
  return h;
}

void DensitiesOpt::opti_allgps(std::vector<Eigen::VectorXd> const & hpars_guess_v){
  for(int i=0;i<m_dim;i++){
    m_DensityOpt_vec[i].opti_allgps(hpars_guess_v[i]);
  }
}

std::vector<Eigen::VectorXd> DensitiesOpt::EvaluateHparsOpt(Eigen::VectorXd const & theta) const{
  vector<VectorXd> h;
  for(int i=0;i<m_dim;i++){
    h.push_back(m_DensityOpt_vec[i].EvaluateHparOpt(theta));
  }
  return h;
}

void DensitiesOpt::Test_hGPs(int npoints, double max_time){
  for(int i=0;i<m_dim;i++){
    m_DensityOpt_vec[i].Test_hGPs(npoints, max_time);
  }
}
MatrixXd DensitiesOpt::Burn_phase_test(int nburn, MatrixXd const & COV_init, VectorXd & Xcurrento, default_random_engine & generator,string const & filename) {
  //phase de burn à afficher, qui a un rôle de débug. On affiche tous les samples visités dans le fichier filename.
  vector<VectorXd> visited_samples;
  vector<double> lls;
  auto compute_score=[this](vector<VectorXd> const & hpars,VectorXd const & Xtest)-> double {
    return loglikelihood_theta(Xtest,hpars)+m_logpriorpars(Xtest);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  VectorXd Xinit=Xcurrento;
  vector<VectorXd> hpars_init=EvaluateHparsOpt(Xinit);
  double finit=compute_score(hpars_init,Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  vector<VectorXd> hpars_current=hpars_init;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(m_dim_pars);
  MatrixXd acc_var=MatrixXd::Zero(m_dim_pars,m_dim_pars);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      vector<VectorXd> hparscandidate=EvaluateHparsOpt(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hpars_current=hparscandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
    visited_samples.push_back(Xcurrent);
    lls.push_back(fcurrent);
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(m_dim_pars))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2)+1e-10*MatrixXd::Identity(m_dim_pars,m_dim_pars));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  cout << "écriture des samples dans le fichier " << filename << endl;
  ofstream ofile(filename);
  for(int i=0;i<visited_samples.size();i++){
    for(int j=0;j<visited_samples[i].size();j++){
      ofile << visited_samples[i](j) << " ";
    }
    ofile << lls[i] << endl;
  }
  ofile.close();
  return CovProp;
}


double DensitiesOpt::Run_Burn_phase_MCMC_opti_hGPs(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento, default_random_engine & generator) {
  //phase de burn et calcul de la nouvelle matrice de covariance.
  //on renvoie la covariance scalée et l'état actuel.
  //on peut calculer une seule fois la matrice Gamma puisque les hpars restent identiques.
  auto compute_score=[this](vector<VectorXd> const & hpars,VectorXd const & Xtest)-> double {
    return loglikelihood_theta(Xtest,hpars)+m_logpriorpars(Xtest);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  VectorXd Xinit=Xcurrento;
  vector<VectorXd> hpars_init=EvaluateHparsOpt(Xinit);
  double finit=compute_score(hpars_init,Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  vector<VectorXd> hpars_current=hpars_init;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(m_dim_pars);
  MatrixXd acc_var=MatrixXd::Zero(m_dim_pars,m_dim_pars);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      vector<VectorXd> hparscandidate=EvaluateHparsOpt(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hpars_current=hparscandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
    m_allmcmcsamples.push_back(Xcurrent);
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(m_dim_pars))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2)+1e-10*MatrixXd::Identity(m_dim_pars,m_dim_pars));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
  return acc_rate;
}

void DensitiesOpt::Run_MCMC_opti_hGPs(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,default_random_engine & generator){

  cout << "running mcmc opti_hgps with " << nsteps << " steps." <<endl;
  //MCMC à hpars variables. pas de scaling.
  m_samples.clear(); m_hparsofsamples_v.clear(); 
  vector<vector<VectorXd>> hsubs(m_dim);
  m_hparsofsamples_v=hsubs;
  m_allmcmcsamples.clear();
  auto compute_score=[this](vector<VectorXd> const & hpars,VectorXd const & Xtest)-> double {
    return loglikelihood_theta(Xtest,hpars)+m_logpriorpars(Xtest);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  Run_Burn_phase_MCMC_opti_hGPs(nsteps*0.1,COV,Xinit,generator);
  vector<VectorXd> hpars_init=EvaluateHparsOpt(Xinit);

  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(hpars_init,Xinit);
  vector<VectorXd> hparscurrent=hpars_init;
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      vector<VectorXd> hparscandidate=EvaluateHparsOpt(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      m_samples.push_back(Xcurrent);
      for(int j=0;j<m_dim;j++){
        m_hparsofsamples_v[j].push_back(hparscurrent[j]);
      }
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}

void DensitiesOpt::Run_MCMC_opti_noburn(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,default_random_engine & generator){

  cout << "running mcmc opti_hgps with " << nsteps << " steps, no burn phase." <<endl;
  //MCMC à hpars variables. pas de scaling.
  m_samples.clear(); m_hparsofsamples_v.clear(); 
  vector<vector<VectorXd>> hsubs(m_dim);
  m_hparsofsamples_v=hsubs;
  m_allmcmcsamples.clear();
  auto compute_score=[this](vector<VectorXd> const & hpars,VectorXd const & Xtest)-> double {
    return loglikelihood_theta(Xtest,hpars)+m_logpriorpars(Xtest);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  vector<VectorXd> hpars_init=EvaluateHparsOpt(Xinit);
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(hpars_init,Xinit);
  vector<VectorXd> hparscurrent=hpars_init;
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds_pars(Xcandidate)){
      vector<VectorXd> hparscandidate=EvaluateHparsOpt(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      m_samples.push_back(Xcurrent);
      for(int j=0;j<m_dim;j++){
        m_hparsofsamples_v[j].push_back(hparscurrent[j]);
      }
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}

void DensitiesOpt::Run_MCMC_opti_adapt(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,double lambda, double gamma, default_random_engine & generator){
  //algorithme andrieu global AM with global adaptive scaling (algorithme 4)


  cout << "running mcmc opti_hgps with " << nsteps << " steps, adaptative algorithm, gamma = "<< gamma << endl;
  //MCMC à hpars variables. pas de scaling.
  m_samples.clear(); m_hparsofsamples_v.clear(); 
  vector<vector<VectorXd>> hsubs(m_dim);
  m_hparsofsamples_v=hsubs;
  m_allmcmcsamples.clear();
  double alphastar=0.234; //valeur conseillée dans l'article. taux d'acceptation optimal.

  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);

  auto compute_score=[this](vector<VectorXd> const & hpars,VectorXd const & Xtest)-> double {
    //calcule le log-score d'un point Xtest
    return loglikelihood_theta(Xtest,hpars)+m_logpriorpars(Xtest);
  };

   auto draw_prop=[this](double lambda,MatrixXd COV,default_random_engine & generator, normal_distribution<double> & distN ){
    //tire une proposal de matrice de cov sqrt(lambda)*COV*sqrt(lambda)
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    MatrixXd sqrtCOV=COV.llt().matrixL();
    VectorXd s=sqrt(lambda)*sqrtCOV*Step;
    return s;
  };

  auto update_params=[this,gamma,alphastar](VectorXd & mu, MatrixXd & COV,double & lambda,double alpha,VectorXd Xcurrent){
    //update les paramètres de l'algo MCMC.
    lambda*=exp(gamma*(alpha-alphastar));
    COV=COV+gamma*((Xcurrent-mu)*(Xcurrent-mu).transpose()-COV);
    COV+=1e-10*MatrixXd::Identity(m_dim_pars,m_dim_pars);
    mu=mu+gamma*(Xcurrent-mu);
  };

  MatrixXd COV=COV_init;
  cout << "cov : " <<COV << endl;
  VectorXd mu=Xinit;
  vector<VectorXd> hpars_init=EvaluateHparsOpt(Xinit);
  double finit=compute_score(hpars_init,Xinit);
  vector<VectorXd> hparscurrent=hpars_init;
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  double lambdan=lambda;
  auto begin=chrono::steady_clock::now();
  double alpha=0;
  for(int i=0;i<nsteps;i++){
    VectorXd Xcandidate=Xcurrent+draw_prop(lambdan,COV,generator,distN);
    //cout << "candidate : " << Xcandidate.transpose() << endl;
    if(in_bounds_pars(Xcandidate)){
      vector<VectorXd> hparscandidate=EvaluateHparsOpt(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      alpha=min(exp(fcandidate-fcurrent),1.);
      double c=distU(generator);
      if(alpha>=c){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
        //cout <<"accepted : "<< alpha << " " << lambdan << " " <<mu.transpose() << endl;
      }
      else{//cerr << "not accepted : " << fcurrent << " " << fcandidate << " " <<alpha << " "<< c <<endl;
      }
    }
    else {
      //cerr << "not in bounds " << lambdan << endl;
    }
    if(isnan(alpha)){alpha=1e-20;};
    update_params(mu,COV,lambdan,alpha,Xcurrent);
    if(i%(nsteps/nsamples)==0 && i>0){
      m_samples.push_back(Xcurrent);
      for(int j=0;j<m_dim;j++){
        m_hparsofsamples_v[j].push_back(hparscurrent[j]);
      }
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}

void DensitiesOpt::Run_MCMC_opti_adapt2(int nsteps,int nsamples, VectorXd & Xinit, MatrixXd const & COV_init,VectorXd lambda, double gamma, default_random_engine & generator){
  //algorithme andrieu global AM with componentwise adaptive scaling (algorithme 6)
  //si p est la dimension de la chaîne, chaque étape requiert p+1 évaluations de la vraisemblance. 


  cout << "running mcmc opti_hgps with " << nsteps << " steps, adaptative algorithm with cpwise scaling, gamma = "<< gamma << endl;
  //MCMC à hpars variables. pas de scaling.
  m_samples.clear(); m_hparsofsamples_v.clear(); 
  vector<vector<VectorXd>> hsubs(m_dim);
  m_hparsofsamples_v=hsubs;
  m_allmcmcsamples.clear();
  double alphastarstar=0.44; //valeur conseillée dans l'article. taux d'acceptation optimal par dimension.

  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);

  auto compute_score=[this](vector<VectorXd> const & hpars,VectorXd const & Xtest)-> double {
    //calcule le log-score d'un point Xtest
    return loglikelihood_theta(Xtest,hpars)+m_logpriorpars(Xtest);
  };

  auto eval_alphas=[this,compute_score](vector<VectorXd> & optimal_hpars,VectorXd Xcurrent,VectorXd Step, double fcurrent,double & fcandidate){
    //renvoie tous les alphas résultants des m_dim_pars évaluations de la log_vraisemblance.
    VectorXd alphas(m_dim_pars+1);
    optimal_hpars=EvaluateHparsOpt(Xcurrent+Step);
    fcandidate=compute_score(optimal_hpars,Xcurrent+Step);
    alphas(m_dim_pars)=min(1.,exp(fcandidate-fcurrent));
    //toutes les évaluations supplémentaires de la log-vraisemblance
    for(int i=0;i<m_dim_pars;i++){
      VectorXd X=Xcurrent; X(i)+=Step(i);
      vector<VectorXd> hpars=EvaluateHparsOpt(X);
      double alpha=min(1.,exp(compute_score(hpars,X)-fcurrent));
      if(isnan(alpha)) {alpha=1e-20;} //cas où alpha est trop faible
      alphas(i)=alpha;
    }
    return alphas;
  };

    auto draw_prop=[this](VectorXd lambda,MatrixXd COV,default_random_engine & generator, normal_distribution<double> & distN ){
    //tire une proposal de matrice de cov sqrt(lambda)*COV*sqrt(lambda)
    VectorXd Step(m_dim_pars); for(int j=0;j<m_dim_pars;j++){Step(j)=distN(generator);}
    MatrixXd sqrtCOV=COV.llt().matrixL();
    VectorXd sqrtlambda(lambda.size());
    sqrtlambda.array()=lambda.array().sqrt();
    VectorXd s=sqrtlambda.asDiagonal()*sqrtCOV*Step;
    return s;
  };

  auto update_params=[this,gamma,alphastarstar](VectorXd & mu, MatrixXd & COV,VectorXd & lambda,VectorXd alpha,VectorXd Xcurrent){
    //update les paramètres de l'algo MCMC.
    for(int i=0;i<m_dim_pars;i++){
      lambda(i)*=exp(gamma*(alpha(i)-alphastarstar));
    }
    COV=COV+gamma*((Xcurrent-mu)*(Xcurrent-mu).transpose()-COV);
    COV+=1e-10*MatrixXd::Identity(m_dim_pars,m_dim_pars);
    mu=mu+gamma*(Xcurrent-mu);
  };

  MatrixXd COV=COV_init;
  cout << "cov : " <<COV << endl;
  VectorXd mu=Xinit;
  vector<VectorXd> hpars_init=EvaluateHparsOpt(Xinit);
  double finit=compute_score(hpars_init,Xinit);
  vector<VectorXd> hparscurrent=hpars_init;
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  VectorXd alphas(m_dim_pars+1);
  double fcandidate;
  for(int i=0;i<nsteps;i++){
    VectorXd Step=draw_prop(lambda,COV,generator,distN);
    if(in_bounds_pars(Xcurrent+Step)){
      vector<VectorXd> hparscandidate;
      double fcandidate;
      alphas=eval_alphas(hparscandidate,Xcurrent,Step,fcurrent,fcandidate);
      double c=distU(generator);
      if(alphas(m_dim_pars)>=c){
        naccept++;
        Xcurrent+=Step;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    update_params(mu,COV,lambda,alphas,Xcurrent);
    if(i%(nsteps/nsamples)==0 && i>0){
      m_samples.push_back(Xcurrent);
      for(int j=0;j<m_dim;j++){
        m_hparsofsamples_v[j].push_back(hparscurrent[j]);
      }
    }
    m_allmcmcsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << m_samples.size() << endl;
  cout << m_samples[0].transpose() << endl;
  cout << m_samples[1].transpose() << endl;
  cout << m_samples[2].transpose() << endl;
  cout << m_samples[3].transpose() << endl;
  cout << m_samples[4].transpose() << endl;
}


void Densities::WritePredictionsF(VectorXd const & X, string const & filename) {
  //on assigne les nouveaux samples/hparsofsamples. Ensuite, on utilise les fonctions 1D avec des noms de fichiers bien choisis.
  for(int i=0;i<m_dim;i++){
    m_Densities_vec[i].SetNewSamples(m_samples);    m_Densities_vec[i].SetNewHparsOfSamples(m_hparsofsamples_v[i]);
    string filename2=filename+to_string(i)+".gnu";
    m_Densities_vec[i].WritePredictionsF(X,filename2);
  }
}

void Densities::WritePredictions(VectorXd const & X, string const & filename) {
  //on assigne les nouveaux samples/hparsofsamples. Ensuite, on utilise les fonctions 1D avec des noms de fichiers bien choisis.
  for(int i=0;i<m_dim;i++){
    m_Densities_vec[i].SetNewSamples(m_samples);
    m_Densities_vec[i].SetNewHparsOfSamples(m_hparsofsamples_v[i]);
    string filename2=filename+to_string(i)+".gnu";
    m_Densities_vec[i].WritePredictions(X,filename2);
  }
}



















