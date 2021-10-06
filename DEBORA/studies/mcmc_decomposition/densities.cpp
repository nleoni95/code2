#include "densities.h"
#include <ctime>
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
    ofile << v(0) << " " <<  v(1) << " " << v(2) << " " << v(3) << " " << v(4) << endl;
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
    m_data_exp=d.m_data_exp;
    m_Grid=d.m_Grid;
    m_data_exp=d.m_data_exp;
    m_Xprofile=d.m_Xprofile;
    m_Xprofile_converted=d.m_Xprofile_converted;
    m_samples=d.m_samples;
    m_hparsofsamples=d.m_hparsofsamples;
    m_allmcmcsamples=d.m_allmcmcsamples;
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

Eigen::MatrixXd Density::Gamma(vector<VectorXd> const & locs, Eigen::VectorXd const &hpar) const {
  // Renvoie la matrice de corrélation avec  bruit
  int nd=locs.size();
  Eigen::MatrixXd A(nd,nd);
  for(int i=0; i<nd; i++){
    for(int j=i; j<nd; j++){
      A(i,j) = m_Kernel(locs[i],locs[j], hpar);
      if(i!=j){
	A(j,i) = A(i,j);
      }else{
	A(i,j) += hpar(1)*hpar(1);		//bruit exp toujours à la position 1
      }
    }
  }
  return A;
}

double Density::loglikelihood_fast(VectorXd const &obs,LDLT<MatrixXd> const &ldlt) const{
    //calcul de la LL à la dernière étape.
    VectorXd Alpha=ldlt.solve(obs);
    int nd=obs.size();
    return -0.5*obs.dot(Alpha)-0.5*(ldlt.vectorD().array().log()).sum()-0.5*nd*log(2*3.1415);
}

double Density::loglikelihood_theta_fast(std::vector<AUGDATA> const &exp_data, Eigen::VectorXd const &theta, Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const{
    //écrite pour un AUGDATA de taille 1.
    VectorXd X=exp_data[0].GetX();
    VectorXd Vals=exp_data[0].Value();
    VectorXd obs(Vals.size());
    obs=Vals-m_model(X,theta); //problème ici. On voudrait mettre la priormean, sauf que ça dépend d'hpars qui ne sont pas connus dans cette fonction. On pourrait le corriger en créant les data dans le fonction d'avant, mais cela ne permettrait pas le découplage pour optfunc_opt. On reste sans priormean.
    return loglikelihood_fast(obs,ldlt);
}

double Density::loglikelihood_theta(std::vector<AUGDATA> const &exp_data, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar)const{
    //évaluation de la LDLT de Gamma. a priori écrite pour un AUGDATA de taille 1 également. Si la taille change, on pourrait faire plusieurs gamma (plusieurs hpars.)
    MatrixXd G=Gamma(m_Xprofile_converted,hpar);
    LDLT<MatrixXd> ldlt(G);
    return loglikelihood_theta_fast(exp_data,theta,ldlt);
}

int Density::optroutine_light(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars){
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

int Density::optroutine_heavy(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars){
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
  //pour stabilité
  x[1]+=1e-7*distU(generator);
  X=VtoVXD(x);
  return fin;
}

int Density::optroutine(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars){
  //routine d'optimisation unique, pour clarifier le code.
  vector<double> x=VXDtoV(X);
  vector<double> lb_hpars_opt=VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt=VXDtoV(ub_hpars);
  //paramètres d'optimisation
  int maxeval=10000;
  int popsize=80;
  double ftol=1e-6;
  // 1 opti globale et 1 opti locale.
  //glo
  nlopt::opt opt(nlopt::GN_ISRES, x.size());
  opt.set_max_objective(optfunc, data_ptr); 
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxeval(maxeval);
  //opt.set_population(popsize);
  opt.set_ftol_rel(ftol);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  //loc
  nlopt::opt opt2(nlopt::LN_SBPLX, x.size());
  opt2.set_max_objective(optfunc, data_ptr); 
  opt2.set_lower_bounds(lb_hpars_opt);
  opt2.set_upper_bounds(ub_hpars_opt);
  opt2.set_maxeval(maxeval);
  opt2.set_ftol_rel(ftol);
  fin=opt2.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  X=VtoVXD(x);
  return fin;
}

int Density::optroutine_withgrad(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars,nlopt::algorithm alg){
  //routine d'optimisation avec gradient
  vector<double> x=VXDtoV(X);
  vector<double> lb_hpars_opt=VXDtoV(lb_hpars);
  vector<double> ub_hpars_opt=VXDtoV(ub_hpars);
  //paramètres d'optimisation
  int maxeval=10000;
  int popsize=80;
  double ftol=1e-6;
  // 1 opti globale et 1 opti locale.
  //glo
  nlopt::opt opt(nlopt::GD_MLSL, x.size());
  opt.set_max_objective(optfunc, data_ptr); 
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  opt.set_maxeval(maxeval);
  //opt.set_population(popsize);
  opt.set_ftol_rel(ftol);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  nlopt::opt opt2(nlopt::LD_AUGLAG, x.size());
  opt2.set_max_objective(optfunc, data_ptr); 
  opt2.set_lower_bounds(lb_hpars_opt);
  opt2.set_upper_bounds(ub_hpars_opt);
  opt2.set_maxeval(maxeval);
  opt2.set_ftol_rel(ftol);
  fin=opt2.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  X=VtoVXD(x);
  return fin;
}

VectorXd Density::HparsKOH(VectorXd const & hpars_guess) {
  //calcule les hyperparamètres optimaux selon KOH.
  VectorXd guess=hpars_guess;
  //On prépare les évaluations de surrogate pour la routine opt, pour accélérer la fonction.
  MatrixXd Residustheta(m_data_exp[0].Value().size(),m_Grid.m_grid.size());
  for(int i=0;i<m_Grid.m_grid.size();i++){
    VectorXd theta=m_Grid.m_grid[i];
    Residustheta.col(i)=-m_model(m_data_exp[0].GetX(),theta)+m_data_exp[0].Value(); //y-ftheta
  }
  auto tp=make_tuple(&Residustheta,this);
  int fin=optroutine(optfuncKOH,&tp,guess,m_lb_hpars,m_ub_hpars);
  cout << "fin de l'opt koh : message " << fin << endl;
  return guess;
}

/*
VectorXd Density::HparsKOH(VectorXd const & hpars_guess) {
  //calcule les hyperparamètres optimaux selon KOH.

    int maxeval=5000;
    vector<double> lb_hpars=VXDtoV(m_lb_hpars); //on fait l'opti sur les hpars et pas sur le log.
    vector<double> ub_hpars=VXDtoV(m_ub_hpars);
    vector<double> x=VXDtoV(hpars_guess); //guess.
    nlopt::opt opt(nlopt::GN_ISRES, x.size());   
    opt.set_max_objective(optfuncKOH, this); 
    opt.set_lower_bounds(lb_hpars);
    opt.set_upper_bounds(ub_hpars);
    opt.set_maxeval(maxeval);
    opt.set_population(2000);
    opt.set_ftol_rel(1e-4);
    double msup;
    int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 0: erreur
    return VtoVXD(x);
}
*/

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
  MatrixXd G=d->Gamma(*xconv,hpars);
  LDLT<MatrixXd> ldlt(G);
  //au lieu du transform, j'essaye une boucle simple.
  for(int i=0;i<Residus->cols();i++){
    double g=d->loglikelihood_fast(Residus->col(i),ldlt);
    prob[i]=g;
  }
  //normalisation et passage à l'exponentielle.
  //le max doit être le même sur toutes les itérations... d'où la nécessité de le définir à l'extérieur !!!
  transform(prob.begin(),prob.end(),prob.begin(),[logvstyp](const double &d)->double {
    double f=exp(d-logvstyp);
    if(isinf(f)){cerr << "erreur myoptfunc_koh : infini" << endl;}
    return f;
  });
  //calcul de l'intégrale. suppose un grid régulier. modifier si on veut un prior pour les paramètres.
  double res=accumulate(prob.begin(),prob.end(),0.0); res/=prob.size();
  res*=exp(d->m_logpriorhpars(hpars));
  return res;
};

VectorXd Density::HparsKOHFromData(VectorXd const & hpars_guess,vector<VectorXd> const & thetas, vector<VectorXd> const & values) {
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


VectorXd Density::HparsNOEDM(VectorXd const & hpars_guess) {
  VectorXd guess(1);
  guess << hpars_guess(1);
  VectorXd lb_hpars(1);
  VectorXd ub_hpars(1);
  lb_hpars << m_lb_hpars(1);
  ub_hpars << m_ub_hpars(1);
  optroutine_light(optfuncNOEDM,this,guess,lb_hpars,ub_hpars);
  VectorXd ret(3);
  ret << 0,guess(0),1;
  return ret;
}

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
    /* fonction à optimiser pour trouver les hpars koh.*/
  VectorXd hpars(3);
  hpars << 0,x[0],1;
  Density *d = (Density *) data; //cast
  const vector<VectorXd> * grid=d->GetGrid(); //const ptr vers le grid.
  const vector<AUGDATA> * expdata=d->GetExpData();
  const vector<VectorXd> *xconv=d->GetXconverted();
  double logvstyp=30; //à choisir.
  //calcul de la logvs sur le grid
  vector<double> prob(grid->size());
  MatrixXd G=d->Gamma(*xconv,hpars);
  LDLT<MatrixXd> ldlt(G);
  transform(grid->cbegin(),grid->cend(),prob.begin(),[d,&ldlt,expdata](VectorXd const & theta)->double{
      double g=d->loglikelihood_theta_fast(*expdata,theta,ldlt);
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
      return false;
    }
  }
  return true;
}

bool Density::in_bounds_hpars(VectorXd const & hpars) const{
  for(int i=0;i<m_dim_hpars;i++){
    if(hpars(i)<m_lb_hpars(i) || hpars(i)>m_ub_hpars(i)){
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
  VectorXd Xinit(m_dim_pars); for(int i=0;i<m_dim_pars;i++){Xinit(i)=m_lb_pars(i)+distU(generator)*(m_ub_pars(i)-m_lb_pars(i));}
  double finit=loglikelihood_theta_fast(m_data_exp,Xinit,ldlt)+m_logpriorpars(Xinit);
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
      double fcandidate=loglikelihood_theta_fast(m_data_exp,Xcandidate,ldlt)+m_logpriorpars(Xcandidate);
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

void Density::Run_MCMC_fixed_hpars(int nsteps,int nsamples, MatrixXd const & COV_init,VectorXd const & hpars,default_random_engine & generator){
  //MCMC à hpars fixés.
  //vidons les samples juste au cas où.
  m_samples.clear(); m_hparsofsamples.clear(); m_allmcmcsamples.clear();

  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  LDLT<MatrixXd> ldlt(Gamma(m_Xprofile_converted,hpars));
  auto compute_score=[this,hpars,ldlt](VectorXd const & X)-> double {
    return this->loglikelihood_theta_fast(*(this->GetExpData()),X,ldlt)+this->m_logpriorpars(X);
  };
  VectorXd Xinit(m_dim_pars);
  MatrixXd COV=COV_init;
  MatrixXd COVPred=COV;
  Run_Burn_phase_MCMC(nsteps*0.1,COV,hpars,Xinit,generator);
  COV=scale_covmatrix(COV,Xinit,compute_score,0,generator,"results/diag/scalekoh.gnu");
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
    VectorXd hpars=X.tail(m_dim_hpars); //jsp si on peut y accéder comme ça
    VectorXd theta=X.head(m_dim_pars);
    return this->loglikelihood_theta(*(this->GetExpData()),theta,hpars)+this->m_logpriorhpars(hpars)+this->m_logpriorpars(theta);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  VectorXd Xinit(dim_mcmc); 
  for(int i=0;i<m_dim_pars;i++){Xinit(i)=m_lb_pars(i)+distU(generator)*(m_ub_pars(i)-m_lb_pars(i));}
  for(int i=0;i<m_dim_hpars;i++){Xinit(i+m_dim_pars)=m_lb_hpars(i)+distU(generator)*(m_ub_hpars(i)-m_lb_hpars(i));}
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
    if(in_bounds_pars(Xcandidate.head(m_dim_pars)) && in_bounds_hpars(Xcandidate.tail(m_dim_hpars))){
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
  MatrixXd CovProp=(pow(2.38,2)/(double)(dim_mcmc))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2)+1e-10*MatrixXd::Identity(dim_mcmc,dim_mcmc));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
  return acc_rate;
}

double Density::Run_Burn_phase_FullMCMC_Controlled(int nburn, MatrixXd & COV_return, VectorXd & Xcurrento, default_random_engine & generator) {
//phase de burn et calcul de la nouvelle matrice de covariance.
  //on renvoie la covariance scalée et l'état actuel.
  //on peut calculer une seule fois la matrice Gamma puisque les hpars restent identiques.
  int dim_mcmc=m_dim_pars+m_dim_hpars;
  auto compute_score=[this](VectorXd const & X)-> double {
    VectorXd hpars=X.tail(m_dim_hpars); //jsp si on peut y accéder comme ça
    VectorXd theta=X.head(m_dim_pars);
    return this->loglikelihood_theta(*(this->GetExpData()),theta,hpars)+this->m_logpriorhpars(hpars)+this->m_logpriorpars(theta);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd sqrtCOV=COV_return.llt().matrixL();
  VectorXd Xinit(dim_mcmc); 
  for(int i=0;i<m_dim_pars;i++){Xinit(i)=m_lb_pars(i)+distU(generator)*(m_ub_pars(i)-m_lb_pars(i));}
  for(int i=0;i<m_dim_hpars;i++){Xinit(i+m_dim_pars)=m_lb_hpars(i)+distU(generator)*(m_ub_hpars(i)-m_lb_hpars(i));}
  cout << "Xinit : " << Xinit.transpose() << endl;
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  VectorXd running_mean=VectorXd::Zero(dim_mcmc);
  MatrixXd running_cov=MatrixXd::Zero(dim_mcmc,dim_mcmc);
  MatrixXd scaled_cov(dim_mcmc,dim_mcmc);
  auto begin=chrono::steady_clock::now();
  for(int i=1;i<=nburn;i++){
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
    VectorXd v=Xcurrent-running_mean;
    running_mean+=(1/(double (i)))*v;
    running_cov+=(1/(double (i)))*(v*v.transpose()-running_cov);
    scaled_cov=(pow(2.38,2)/(double(dim_mcmc)))*running_cov;
    sqrtCOV=scaled_cov.llt().matrixL();
    m_allmcmcsamples.push_back(Xcurrent);
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << scaled_cov << endl;
  Xcurrento=Xcurrent;
  COV_return=scaled_cov;
  return acc_rate;
}


void Density::Run_FullMCMC(int nsteps,int nsamples, MatrixXd const & COV_init,default_random_engine & generator){
//MCMC à hpars variables
  m_samples.clear(); m_hparsofsamples.clear(); m_allmcmcsamples.clear();
  int dim_mcmc=m_dim_pars+m_dim_hpars;
  auto compute_score=[this](VectorXd const & X)-> double {
    VectorXd hpars=X.tail(m_dim_hpars);
    VectorXd theta=X.head(m_dim_pars);
    return this->loglikelihood_theta(*(this->GetExpData()),theta,hpars)+this->m_logpriorhpars(hpars)+this->m_logpriorpars(theta);
  };
  m_samples.clear(); m_hparsofsamples.clear();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  VectorXd Xinit(dim_mcmc);
  for(int i=0;i<m_dim_pars;i++){Xinit(i)=m_lb_pars(i)+distU(generator)*(m_ub_pars(i)-m_lb_pars(i));}
  for(int i=0;i<m_dim_hpars;i++){Xinit(i+m_dim_pars)=m_lb_hpars(i)+distU(generator)*(m_ub_hpars(i)-m_lb_hpars(i));}
  MatrixXd COV=COV_init;
  Run_Burn_phase_FullMCMC(nsteps*0.1,COV,Xinit,generator);
  COV=scale_covmatrix(COV,Xinit,compute_score,0,generator,"results/diag/scalefullbayes.gnu");
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
    if(i%(nsteps/nsamples)==0 && i>0){
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

double Density::measure_acc_rate_1D(int nsteps, VectorXd const & direction, double step_size, VectorXd const & Xorigin, function<double(VectorXd)> const & compute_score, default_random_engine & generator)const{
  //mesure de l'accept_rate dans la direction direction, à partir du point Xorigin.
  normal_distribution<double> distN(0,1);
  uniform_real_distribution<double> distU(0,1);
  VectorXd Xinit=Xorigin;
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  for(int i=0;i<nsteps;i++){
    VectorXd Xcandidate=Xcurrent+distN(generator)*step_size*direction;
    if(in_bounds_pars(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
  }
  double acc_rate=(double)(naccept)/(double)(nsteps);
  return acc_rate;
}

void Density::estimate_acc_rate(int nsteps, MatrixXd const & modes, VectorXd const & lambdas, VectorXd const & Xorigin, function<double(VectorXd)> const & compute_score, default_random_engine & generator)const{
  //estimation de l'accept_rate global à partir des mesures 1D.
  int dim_mcmc=Xorigin.size();
  vector<double> acc_rates(dim_mcmc);
  double tot_acc_rate=1;
  cout << "acc rates estimés : ";
  for(int i=0;i<dim_mcmc;i++){
    double acc_rate=measure_acc_rate_1D(nsteps,modes.col(i),sqrt(lambdas(i)),Xorigin,compute_score,generator);
    cout << acc_rate << " " ;
    acc_rates[i]=acc_rate;
    tot_acc_rate*=acc_rate;
  }
  cout << endl << " total acc_rate estimé : " << tot_acc_rate << endl;
}

void Density::get_acc_rate(MatrixXd const & COV, VectorXd const & Xcurrent, function<double(VectorXd)> const & compute_score, default_random_engine & generator) const{
  SelfAdjointEigenSolver<MatrixXd> eig(COV);
  VectorXd lambdas=eig.eigenvalues();
  lambdas.reverseInPlace();
  MatrixXd modes=eig.eigenvectors().rowwise().reverse();
  cout << "valeurs propres : " << lambdas.transpose() << endl;
  cout << "modes associés : " << endl << modes << endl;
  estimate_acc_rate(1e4,modes,lambdas,Xcurrent,compute_score,generator);
}

MatrixXd Density::scale_covmatrix(MatrixXd const & COV, VectorXd const & Xcurrent, function<double(VectorXd)> const & compute_score, double true_accrate, default_random_engine & generator,string const filename) const{
  //on suppose qu'à la fin de la phase de burn, les directions principales ont été corectement estimées. On rescale la matrice de covariance dans ces directions.
  //décomp de la cov. initiale.
  ofstream ofile(filename);
  ofile << endl << "covmatrix avant scaling" << endl;
  ofile << "covmatrix : " << endl << COV <<endl;
  int dim_mcmc=Xcurrent.size();
  double atarget=0.3; //on vise un alpha global de 30 pct.
  double atarget1D=pow(atarget,1/(1.0*dim_mcmc));
  SelfAdjointEigenSolver<MatrixXd> eig(COV);
  VectorXd lambdas=eig.eigenvalues();
  lambdas.reverseInPlace();
  MatrixXd modes=eig.eigenvectors().rowwise().reverse();
  ofile << "valeurs propres : " << lambdas.transpose() << endl;
  ofile << "modes associés : " << endl << modes << endl;
  //récupération des acc rates estimés en 1D.
  vector<double> acc_rates_1D(dim_mcmc);
  double estimated_global_ac=1;
  ofile << "acc rates 1D estimés : " << endl;
  for(int i=0;i<dim_mcmc;i++){
    double a=measure_acc_rate_1D(1e4,modes.col(i),sqrt(lambdas(i)),Xcurrent,compute_score,generator);
    estimated_global_ac*=a;
    acc_rates_1D[i]=a;
    ofile << a << " ";
  }
  ofile << endl << "estimated global acc rate before correction : " << estimated_global_ac << endl;
  ofile << "target acc rate in 1D : " << atarget1D << endl;
  VectorXd new_lambdas(dim_mcmc);
  double scale=pow(true_accrate/estimated_global_ac,1/(1.0*dim_mcmc));
  if(scale==0){scale=1;}//pour utiliser la fonction sans connaître l'acc_rate.
  for(int i=0;i<dim_mcmc;i++){
    acc_rates_1D[i]*=scale;
  }
  for(int i=0;i<dim_mcmc;i++){
    double ratio=pow(acc_rates_1D[i]*(1-atarget1D)/(atarget1D*(1-acc_rates_1D[i])),2/1.23);
    new_lambdas(i)=lambdas(i)*ratio;
  }
  ofile << "nouvelles valeurs propres : " << new_lambdas.transpose() << endl;
  //constitution de la nouvelle matrice de covariance
  MatrixXd L=new_lambdas.asDiagonal();
  MatrixXd NewCOV=modes*L*modes.inverse();
  ofile << "nouvelle Covmatrix : " << endl << NewCOV<< endl;
  ofile.close();
  return NewCOV;
}




void Density::Autocor_diagnosis(int nstepsmax, string const & filename)
{
  //diagnostic d'autocorrélation, jusqu'à une distance de nstepsmax pas dans la mcmc.
  //on centre les données
  int dim_mcmc=m_allmcmcsamples[0].size();
  VectorXd mean=VectorXd::Zero(dim_mcmc);
  VectorXd var=VectorXd::Zero(dim_mcmc); //variance composante par composante
  for (int i=0;i<m_allmcmcsamples.size();i++){
    mean+=m_allmcmcsamples[i];
  }
  mean/=m_allmcmcsamples.size();
  for (int i=0;i<m_allmcmcsamples.size();i++){
    m_allmcmcsamples[i]-=mean;
    var+=m_allmcmcsamples[i].cwiseProduct(m_allmcmcsamples[i]);
  }
  var/=m_allmcmcsamples.size(); //calcul de la variance composante par composante.
  ofstream ofile(filename);
  FILE* out= fopen("results/autocor.gnu","w");
  for (int i=0;i<nstepsmax;i++){
    VectorXd cor=Lagged_mean(i).cwiseQuotient(var);
    for(int j=0;j<cor.size();j++){
      ofile << cor(j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
  for (int i=0;i<m_allmcmcsamples.size();i++){
    m_allmcmcsamples[i]+=mean;
  }
}

VectorXd Density::Lagged_mean(int n) const{
  int dim_mcmc=m_allmcmcsamples[0].size();
  VectorXd ans=VectorXd::Zero(dim_mcmc);
  for (int i=0;i<m_allmcmcsamples.size()-n;i++){
    ans+=m_allmcmcsamples[i].cwiseProduct(m_allmcmcsamples[i+n]);
  }
  return ans/(double (m_allmcmcsamples.size()-n));
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
  //écrit pour la dimension 5 et 3 hpars pour kernel_z.
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      VectorXd theta(5);
      VectorXd hpars(3);
      for(int i=0;i<5;i++){
        theta(i)=stod(words[i]);
      }
      for(int i=0;i<3;i++){
        hpars(i)=stod(words[i+5]);
      }
      m_samples.push_back(theta);
      m_hparsofsamples.push_back(hpars);
    }
    cout << "number of samples loaded : " << m_samples.size() << endl;
  }
  else{
    cerr << "empty file" << endl;
  }

  ifile.close();
}

VectorXd Density::MAP_given_hpars(VectorXd const & hpars) const{
  //il faut calculer les densités à chaque point.
  vector<double> lp(m_samples.size());
  transform(m_samples.begin(),m_samples.end(),lp.begin(),[&hpars, this](VectorXd const & theta)-> double {
    return this->loglikelihood_theta(*(this->GetExpData()),theta,hpars)+this->m_logpriorhpars(hpars);
  });
  auto it= std::max_element(lp.begin(),lp.end()); // ? jsp pourquoi ça marche.
  int index=distance(lp.begin(),it);
  return m_samples[index];
}

VectorXd Density::Mean() const{
  VectorXd acc=VectorXd::Zero(m_dim_pars);
  for_each(m_samples.begin(),m_samples.end(),[&acc](const VectorXd &X)mutable{
    acc+=X;
  });
  return acc/m_samples.size();
}

MatrixXd Density::Variance() const{
  VectorXd mean=Mean();
  MatrixXd SecondMoment=MatrixXd::Zero(m_dim_pars,m_dim_pars);
  for_each(m_samples.begin(),m_samples.end(),[&SecondMoment](VectorXd const &X)mutable{
    SecondMoment+=X*X.transpose();
  });
  MatrixXd var=SecondMoment/m_samples.size()-mean*mean.transpose();
  return var;
}

double Density::Evidence(double normconst) const{
  cout << "calculating evidence on " << m_samples.size() << "samples..." << endl;
  double evidence=0;
  for(int i=0;i<m_samples.size();i++){
    double l=loglikelihood_theta(m_data_exp,m_samples[i],m_hparsofsamples[i]);
    double e=exp(l-normconst);
    if(isinf(e)){cerr << "erreur calcul evidence : infini " << l << endl;}
    evidence+=e;
  }
  return evidence/m_samples.size();
}

VectorXd Density::meanF(VectorXd const & X) const {
  //prédiction moyenne du modèle
  VectorXd mpf=VectorXd::Zero(m_Xprofile.size());
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

VectorXd Density::meanZCondTheta(VectorXd const & X,VectorXd const & theta, VectorXd const & hpars) const{
  //prédiction moyenne de Z sur Xprofile, pour le i-ème theta de m_samples.
  VectorXd y(m_Xprofile.size());
  y=m_data_exp[0].Value()-m_model(X,theta);
  //puisque la prédiction du modèle et les observations sont aux mêmes points, ça simplifie grandement la tâche.
  MatrixXd G=Gamma(m_Xprofile_converted,hpars);
  LDLT<MatrixXd> ldlt(G);
  //construction de la matrice de covariance, entre les mêmes points, mais sans le bruit expérimental.
  MatrixXd Kstar=G-pow(hpars(1),2)*MatrixXd::Identity(m_Xprofile.size(),m_Xprofile.size());
  VectorXd predmean=Kstar*ldlt.solve(y);
  return predmean;
}

MatrixXd Density::varZCondTheta(VectorXd const & X,VectorXd const & theta, VectorXd const & hpars) const{
  //variance prédictive de z sur Xprofile, pour le i-ème theta de m_samples.

  MatrixXd G=Gamma(m_Xprofile_converted,hpars);
  MatrixXd Kstar=G-pow(hpars(1),2)*MatrixXd::Identity(m_Xprofile.size(),m_Xprofile.size());
  //puisque la prédiction du modèle et les observations sont aux mêmes points, ça simplifie grandement la tâche.
  LDLT<MatrixXd> ldlt(G);
  //construction de la matrice de covariance, entre les mêmes points, mais sans le bruit expérimental.
  MatrixXd VarPred=Kstar-Kstar*ldlt.solve(Kstar);
  //on renvoie seulement la diagonale (prédiction moyenne)
  return VarPred;
}

MatrixXd Density::PredFZ(VectorXd const & X) const{
  //predictions avec f+z. Première colonne : moyenne, Deuxième colonne : variance de E[f+z]. Troisième colonne : espérance de var[z].
  //récupération des valeurs de E[f+z|theta]
  //Calcul de la composante E[Var z] également.
  VectorXd mean=VectorXd::Zero(m_Xprofile.size());
  MatrixXd SecondMoment=MatrixXd::Zero(m_Xprofile.size(),m_Xprofile.size());
  MatrixXd Evarz=MatrixXd::Zero(m_Xprofile.size(),m_Xprofile.size());
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
  MatrixXd res(m_Xprofile.size(),3);
  res.col(0)=mean; res.col(1)=VarEfz.diagonal(); res.col(2)=Evarz.diagonal();
  return res;
}

VectorXd Density::DrawZCondTheta(VectorXd const & X, VectorXd const & theta, VectorXd const &hpars_z, default_random_engine & generator) const{
  //tirage d'une réalisation de z pour un theta et des hpars donnés.
  normal_distribution<double> distN(0,1);
  VectorXd mean=meanZCondTheta(X,theta,hpars_z);
  MatrixXd Cov=varZCondTheta(X,theta,hpars_z);
  VectorXd N(mean.size());
  for(int i=0;i<N.size();i++){N(i)=distN(generator);}
  MatrixXd sqrtCOV=Cov.llt().matrixL();
  return mean+sqrtCOV*N;
}


void Density::WriteOneCalcul(VectorXd const & X, VectorXd const & theta, VectorXd const & hpars_z, string const & filename) const{
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
  VectorXd samp1=DrawZCondTheta(X,m_samples[5],m_hparsofsamples[5],generator);
  VectorXd samp2=DrawZCondTheta(X,m_samples[5],m_hparsofsamples[5],generator);
  VectorXd samp3=DrawZCondTheta(X,m_samples[5],m_hparsofsamples[5],generator);
  VectorXd samp4=DrawZCondTheta(X,m_samples[5],m_hparsofsamples[5],generator);
  VectorXd meanf=meanF(X);
  VectorXd varf=varF(X).diagonal();
  MatrixXd Predictions=PredFZ(X); //col0 : moyenne fz, col1: varefz, col2: evarz
  ofstream ofile(filename);
  for(int i=0;i<Predictions.rows();i++){
    ofile << m_Xprofile(i) << " " << m_data_exp[0].Value()(i) <<" "<< meanf(i) << " " << sqrt(varf(i)) << " " << Predictions(i,0) << " " << sqrt(Predictions(i,1)+Predictions(i,2)) << " " << sqrt(Predictions(i,2)) << " " << samp1(i)<< " " << samp2(i)<< " " << samp3(i)<< " " << samp4(i) << endl;
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
  cout << "computing predictions with " << m_samples.size() << " samples..." << endl;
  VectorXd meanf=meanF(X); //prédiction moyenne
  MatrixXd fvalues(meanf.rows(),m_samples.size());
  for(int i=0;i<m_samples.size();i++){
    fvalues.col(i)=m_model(X,m_samples[i]);
  }
  //vecteurs des quantiles
  VectorXd quant2p5(meanf.rows()); //95pct de la masse
  VectorXd quant97p5(meanf.rows());
  VectorXd quant25(meanf.rows()); //50 pct de la masse
  VectorXd quant75(meanf.rows());
  for(int i=0;i<meanf.rows();i++){
    quant2p5(i)=FindQuantile(0.025,fvalues.row(i));
    quant97p5(i)=FindQuantile(0.975,fvalues.row(i));
    quant25(i)=FindQuantile(0.25,fvalues.row(i));
    quant75(i)=FindQuantile(0.75,fvalues.row(i));
  }
  ofstream ofile(filename);
  for(int i=0;i<meanf.rows();i++){
    ofile << m_Xprofile(i) << " " << m_data_exp[0].Value()(i) <<" "<< meanf(i) << " " << quant25(i) << " " << quant75(i) << " " << quant2p5(i) << " " << quant97p5(i) << endl;
  }
  ofile.close();
}

void Density::WritePriorPredictions(VectorXd const & X, string const & filename, default_random_engine & generator) {
  //prédictions à priori, avec un prior uniforme sur les paramètres. Je ne sais pas encore traiter z, donc je laisse comme ça.
  uniform_real_distribution<double> distU(0,1); 
  vector<VectorXd> samples=m_samples;
  for (int i=0;i<m_samples.size();i++){
    VectorXd X(5);
    X << distU(generator),distU(generator),distU(generator),distU(generator),distU(generator);
    m_samples[i]=X;    
  }
  WritePredictions(X,filename);
  m_samples=samples;
}

void Density::WritePriorPredictionsF(VectorXd const & X, string const & filename, default_random_engine & generator) {
  //prédictions à priori, avec un prior uniforme sur les paramètres. Je ne sais pas encore traiter z, donc je laisse comme ça.
  uniform_real_distribution<double> distU(0,1); 
  vector<VectorXd> samples=m_samples;
  for (int i=0;i<m_samples.size();i++){
    VectorXd X(5);
    X << distU(generator),distU(generator),distU(generator),distU(generator),distU(generator);
    m_samples[i]=X;    
  }
  WritePredictionsF(X,filename);
  m_samples=samples;
}

DensityOpt::DensityOpt(Density const & d) : Density(d){
  m_samples.clear();
  m_hparsofsamples.clear();
  m_allmcmcsamples.clear();
}

VectorXd DensityOpt::HparsOpt(VectorXd const & theta, VectorXd const & hpars_guess) {
  VectorXd guess=hpars_guess;
  //guess(2)=1e-3; //modif nico
  //calcul des données f-theta
  VectorXd obsmtheta(m_Xprofile.size());
  VectorXd Yexp=m_data_exp[0].Value();
  VectorXd Fpred=m_model(m_data_exp[0].GetX(),theta);
  obsmtheta=Yexp-Fpred; //encore une fois, on devrait mettre la priormean.
  auto p=make_pair(&obsmtheta,this);
  int fin=optroutine_heavy(optfuncOpt,&p,guess,m_lb_hpars,m_ub_hpars);
  return guess;
}

VectorXd DensityOpt::HparsOpt(VectorXd const & theta, VectorXd const & hpars_guess,nlopt::algorithm alg) {
  VectorXd guess=hpars_guess;
  //guess(2)=1e-3; //modif nico
  //calcul des données f-theta
  VectorXd obsmtheta(m_Xprofile.size());
  VectorXd Yexp=m_data_exp[0].Value();
  VectorXd Fpred=m_model(m_data_exp[0].GetX(),theta);
  obsmtheta=Yexp-Fpred; //encore une fois, on devrait mettre la priormean.
  auto p=make_pair(&obsmtheta,this);
  int fin=optroutine_withgrad(optfuncOpt_withgrad,&p,guess,m_lb_hpars,m_ub_hpars,alg);
  return guess;
}

double DensityOpt::optfuncOpt(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* fonction à optimiser pour trouver les hpars optimaux.*/
  //cast du null pointer
  pair<const VectorXd*,const DensityOpt*> *p=(pair<const VectorXd*,const DensityOpt*> *) data;
  const VectorXd *obs=p->first;
  const DensityOpt *d=p->second;
  VectorXd hpars=VtoVXD(x);
  const vector<VectorXd> *xconv=d->GetXconverted();
  LDLT<MatrixXd> ldlt(d->Gamma(*xconv,hpars));
  double ll=d->loglikelihood_fast(*obs,ldlt);
  double lp=d->m_logpriorhpars(hpars);
  //cout << "hpars testés : " << hpars.transpose() << endl;
  //cout << "ll :" << ll << ", lp : " << lp << endl;
  return ll+lp;
};

double DensityOpt::optfuncOpt_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* fonction à optimiser pour trouver les hpars optimaux.*/
  //cast du null pointer
  pair<const VectorXd*,const DensityOpt*> *p=(pair<const VectorXd*,const DensityOpt*> *) data;
  const VectorXd *obs=p->first;
  const DensityOpt *d=p->second;
  VectorXd hpars=VtoVXD(x);
  const vector<VectorXd> *xconv=d->GetXconverted();
  LDLT<MatrixXd> ldlt(d->Gamma(*xconv,hpars));
  double ll=d->loglikelihood_fast(*obs,ldlt);
  double lp=d->m_logpriorhpars(hpars);
  //calcul des matrices des gradients
  int nd=xconv->size();
  MatrixXd DG1=MatrixXd::Zero(nd,nd);
  MatrixXd DG2=MatrixXd::Zero(nd,nd);
  MatrixXd DG3=MatrixXd::Zero(nd,nd);

  for(int i=0; i<nd; i++){
    for(int j=i; j<nd; j++){
      DG1(i,j)=d->m_DKernel1((*xconv)[i],(*xconv)[j],hpars);
      DG3(i,j)=d->m_DKernel3((*xconv)[i],(*xconv)[j],hpars);
      if(i!=j){
	      DG1(j,i) = DG1(i,j);
        DG3(j,i) = DG3(i,j);
      }else{
        DG2(i,j)+=2*hpars(1);
      }
    }
  }
  VectorXd alpha=ldlt.solve(*obs);
  MatrixXd aat=alpha*alpha.transpose();
  MatrixXd Kinv=ldlt.solve(MatrixXd::Identity(nd,nd));
  if(!grad.size()==0){
    grad[0]=0.5*((aat-Kinv)*DG1).trace()-2/hpars(0); //le prior
    grad[1]=0.5*((aat-Kinv)*DG2).trace();
    grad[2]=0.5*((aat-Kinv)*DG3).trace();
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

void DensityOpt::Compute_optimal_hpars(){
  //calcul de tous les hpars optimaux sur m_grid, et rangement dans m_hpars_opti.
  m_hpars_opti.clear();
  VectorXd hpars_guess=0.5*(m_lb_hpars+m_ub_hpars);
  const vector<VectorXd> *grid=GetGrid();
  transform(grid->begin(),grid->end(),back_inserter(m_hpars_opti),[&hpars_guess,this](VectorXd const & theta) mutable {
    VectorXd hpars_opt=HparsOpt(theta, hpars_guess);
    hpars_guess=hpars_opt; //warm restart
    AUGDATA dat; dat.SetX(theta); dat.SetValue(hpars_opt);
    return dat;
  });
  //affichage dans un fichier
  ofstream ofile("results/hparsopt.gnu");
  for(AUGDATA const &d:m_hpars_opti){
    VectorXd X=d.GetX();
    VectorXd hpars=d.Value();
    ofile << X(0) << " " << X(1) << " "<< X(2) << " "<< X(3) << " "<< X(4) << " " << hpars(0) << " " << hpars(1) << " " << hpars(2) << endl;
  }
  ofile.close();
  //calcul de quelques statistiques sur ces hpars optimaux obtenus. moyenne, et matrice de covariance des données.
  VectorXd mean=VectorXd::Zero(m_dim_hpars);
  MatrixXd SecondMoment=MatrixXd::Zero(m_dim_hpars,m_dim_hpars);
  double mean_lp=0;
  for(AUGDATA const & a:m_hpars_opti){
    mean_lp+=loglikelihood_theta(m_data_exp,a.GetX(),a.Value());
    mean+=a.Value();
    SecondMoment+=a.Value()*a.Value().transpose();
  }
  mean_lp/=m_hpars_opti.size();
  mean/=m_hpars_opti.size();
  SecondMoment/=m_hpars_opti.size();
  MatrixXd Var=SecondMoment-mean*mean.transpose();
  cout << " fin de calcul des hpars opti sur le grid. Moyenne : " << mean.transpose() << endl;
  cout << "stds  : " << sqrt(Var(0,0)) << " " << sqrt(Var(1,1)) << " " <<sqrt(Var(2,2)) << " " << endl;
  cout << "moyenne des logposts:" << mean_lp << endl;
}

void DensityOpt::Compute_optimal_hpars(nlopt::algorithm alg){
  //calcul de tous les hpars optimaux sur m_grid, et rangement dans m_hpars_opti.
  m_hpars_opti.clear();
  VectorXd hpars_guess=0.5*(m_lb_hpars+m_ub_hpars);
  const vector<VectorXd> *grid=GetGrid();
  transform(grid->begin(),grid->end(),back_inserter(m_hpars_opti),[&hpars_guess,this,alg](VectorXd const & theta) mutable {
    VectorXd hpars_opt=HparsOpt(theta, hpars_guess,alg);
    hpars_guess=hpars_opt; //warm restart
    AUGDATA dat; dat.SetX(theta); dat.SetValue(hpars_opt);
    return dat;
  });
  //affichage dans un fichier
  ofstream ofile("results/hparsopt.gnu");
  for(AUGDATA const &d:m_hpars_opti){
    VectorXd X=d.GetX();
    VectorXd hpars=d.Value();
    ofile << X(0) << " " << X(1) << " "<< X(2) << " "<< X(3) << " "<< X(4) << " " << hpars(0) << " " << hpars(1) << " " << hpars(2) << endl;
  }
  ofile.close();
  //calcul de quelques statistiques sur ces hpars optimaux obtenus. moyenne, et matrice de covariance des données.
  VectorXd mean=VectorXd::Zero(m_dim_hpars);
  MatrixXd SecondMoment=MatrixXd::Zero(m_dim_hpars,m_dim_hpars);
  double mean_lp=0;
  for(AUGDATA const & a:m_hpars_opti){
    mean_lp+=loglikelihood_theta(m_data_exp,a.GetX(),a.Value());
    mean+=a.Value();
    SecondMoment+=a.Value()*a.Value().transpose();
  }
  mean_lp/=m_hpars_opti.size();
  mean/=m_hpars_opti.size();
  SecondMoment/=m_hpars_opti.size();
  MatrixXd Var=SecondMoment-mean*mean.transpose();
  cout << " fin de calcul des hpars opti sur le grid. Moyenne : " << mean.transpose() << endl;
  cout << "stds  : " << sqrt(Var(0,0)) << " " << sqrt(Var(1,1)) << " " <<sqrt(Var(2,2)) << " " << endl;
  cout << "moyenne des logposts:" << mean_lp << endl;
}


void DensityOpt::Test_hGPs() {
  //calcul d'un nouveau grid de thetas, optimisation sur ces points, et évaluation de la qualité de prédiction des GPs sur ces points.
  default_random_engine generator; generator.seed(16);
  int nthetas=2000;
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
      VectorXd t(5);
      t << distU(generator),distU(generator),distU(generator),distU(generator),distU(generator);
      newgrid.push_back(t);
    }
    for(int i=0;i<nthetas;i++){
      Hopt.col(i)=HparsOpt(newgrid[i],hpars_guess,nlopt::LN_AUGLAG_EQ);
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
  Hoptv=Hopt.row(2);
  Hoptpredv=Hoptpred.row(2);
  double errlcor=sqrt(ps(Hoptv-Hoptpredv,Hoptv-Hoptpredv)/ps(Hoptv,Hoptv)); 
  cout << "erreur relative des hGPs sur le grid de validation : edm : " << erredm*100 << " pct, exp : " << errexp*100 << ", lcor : " << errlcor*100 <<  endl;
}

void DensityOpt::BuildHGPs(double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), MatrixXd const & Bounds_hpars_GPs,VectorXd const & Hpars_guess_GPs,int nmodes){
  //PCA des hpars opti, et initialisation des HGPs.
  //récupération des data pour les GPs.
  if(m_hpars_opti.size()==0){cerr << "erreur : hpars optis non calculés !" << endl;}
  if(nmodes>m_dim_hpars){cerr << "erreur : impossible de sélectionner autant de modes !" << endl;}
  MatrixXd U(m_dim_hpars,m_hpars_opti.size()); //matrice des données
  MatrixXd P(m_dim_pars,m_hpars_opti.size()); //matrice des thetas
  for(int i=0;i<m_hpars_opti.size();i++){
    U.col(i)=m_hpars_opti[i].Value();
    P.col(i)=m_hpars_opti[i].GetX();
  }
  m_featureMeans=U.rowwise().mean();
  U=U.colwise()-m_featureMeans;
  MatrixXd Covmatrix=U*U.transpose();
  Covmatrix/=((double) m_hpars_opti.size());
  SelfAdjointEigenSolver<MatrixXd> eig(Covmatrix);
  VectorXd lambdas_red=eig.eigenvalues().bottomRows(nmodes); 
  MatrixXd vecpropres=eig.eigenvectors().rightCols(nmodes);
  m_VP=vecpropres.rowwise().reverse();
  cout << "Sélection de " << nmodes << " modes pour les hpars opt" << endl;
  cout << "modes : " << endl << m_VP << endl;
  cout << "Quantité d'énergie conservée : " << 100*lambdas_red.array().sum()/eig.eigenvalues().array().sum() << " %" << endl;
  //matrice de coefficients à apprendre
  MatrixXd A=m_VP.transpose()*U;
  VectorXd Ascale=lambdas_red.array().sqrt();
  m_Acoefs=Ascale.asDiagonal();
  MatrixXd normedA=m_Acoefs.inverse()*A;
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

void DensityOpt::opti_1gp(int i, VectorXd & hpars_guess){
  cout << "optimisation du gp pour hpars numero " << i <<endl;
  auto begin=chrono::steady_clock::now();
  m_vgp_hpars_opti[i].OptimizeGP(myoptfunc_gp,&m_Bounds_hpars_GPs,&hpars_guess,hpars_guess.size());
  auto end=chrono::steady_clock::now();
  hpars_guess=m_vgp_hpars_opti[i].GetPar();
  cout  << "par after opt : " << hpars_guess.transpose() << endl;
  cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;
  m_vhpars_pour_gp.push_back(hpars_guess);
}

void DensityOpt::opti_allgps(VectorXd const &hpars_guess){
  VectorXd h=hpars_guess;
  for(int i=0;i<m_vgp_hpars_opti.size();i++){
    opti_1gp(i,h);
  }
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


double DensityOpt::Run_Burn_phase_MCMC_opti(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento, default_random_engine & generator) {
  //phase de burn et calcul de la nouvelle matrice de covariance.
  //on renvoie la covariance scalée et l'état actuel.
  //on peut calculer une seule fois la matrice Gamma puisque les hpars restent identiques.
  auto compute_score=[this](VectorXd const & X)-> double {
    VectorXd hpars=this->EvaluateHparOpt(X);
    return this->loglikelihood_theta(*(this->GetExpData()),X,hpars)+this->m_logpriorpars(X);
  };
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  VectorXd Xinit(m_dim_pars); for(int i=0;i<m_dim_pars;i++){Xinit(i)=m_lb_pars(i)+distU(generator)*(m_ub_pars(i)-m_lb_pars(i));}
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
// ATTENTION C'EST MODIFIE
void DensityOpt::Run_MCMC_opti(int nsteps,int nsamples, MatrixXd const & COV_init,default_random_engine & generator){
  //MCMC à hpars variables
  m_samples.clear(); m_hparsofsamples.clear(); m_allmcmcsamples.clear();
  auto compute_score=[this](VectorXd const & X)-> double {
    VectorXd hpars=this->EvaluateHparOpt(X);
    return this->loglikelihood_theta(*(this->GetExpData()),X,hpars)+this->m_logpriorpars(X);
  };
  m_samples.clear(); m_hparsofsamples.clear();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  VectorXd Xinit(m_dim_pars);
  MatrixXd COV=COV_init;
  Run_Burn_phase_MCMC_opti(nsteps*0.1,COV,Xinit,generator);
  COV=scale_covmatrix(COV,Xinit,compute_score,0,generator,"results/diag/scaleopt.gnu");
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
      m_hparsofsamples.push_back(EvaluateHparOpt(Xcurrent));
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












