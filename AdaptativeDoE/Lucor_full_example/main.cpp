// Tests pour l'adaptative DoE. Dans ce fichier, on utilise tout le framework réalité/modèle.
// On fait de l'adaptative DoE en utilisant le framework OLM/Lucor. On va faire une mcmc avec le surrogate de l'étape k pour obtenir des échantillons pour l'étape suivante.
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <functional>
#include <unordered_set>
#include "densities.h"

using namespace std;
using namespace Eigen;
int neval=0;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);
const int dimx=1;
const int dimtheta=1;
const int dimgp=dimx+dimtheta;
VectorXd hpars_noedm(3);

typedef pair<VectorXd,VectorXd> Vec_aug; //DATA augmentée, comprenant les hyperparamètres optimaux à l'observation DATA.
typedef tuple<const vector<DATA>*, const GP*,const MCMC_opti*,const vector<VectorXd>*,const double*> tuple_KOH; //à passer dans les arguments de myoptfunc_koh.


VectorXd randpert(int const n){
  //renvoie une permutation aléatoire de {0,1,...,n-1}.. Mélange de Fisher-Yates.
  VectorXd result(n);
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
}

vector<VectorXd> InitGridRect(VectorXd const &lb_t,VectorXd const &ub_t, int const n){
  // Construction du grid initial de thetas. Il faut construire au préalable les vecteurs lb_t et ub_t car on ne peut pas les déclarer de manière globale.
  // n correspond au nombre de points total.
  int dim_theta=1;
  int npoints=n*dim_theta;
  vector<VectorXd> grid;
  VectorXd theta_courant(1);
  VectorXd ind_courant(1);
  for(int i=0;i<npoints;i++){
    theta_courant(0)=lb_t(0)+(i+0.5)*(ub_t(0)-lb_t(0))/double(n);
    grid.push_back(theta_courant);
  }
  return grid;  
}

vector<VectorXd> InitGridUnif(VectorXd const &lb_t,VectorXd const &ub_t, int const n){
  // Construction du grid initial de thetas. On le fait une bonne fois pour toutes. Il faut construire au préalable les vecteurs lb_t et ub_t car on ne peut pas les déclarer de manière globale.
  // n correspond au nombre de points par dimension.
  int dim_theta=lb_t.size();
  int npoints=n*dim_theta;
  vector<VectorXd> grid;
  VectorXd theta_courant(dim_theta);
  VectorXd ind_courant(dim_theta);
  for(int i=0;i<npoints;i++){
    for (int j=0;j<dim_theta;j++){
      theta_courant(j)=lb_t(j)+distU(generator)*(ub_t(j)-lb_t(j));
    }
    grid.push_back(theta_courant);
  }
  return grid;  
}

vector<VectorXd> InitGridLHS(VectorXd const &lb_t,VectorXd const &ub_t, int const npoints){
  // Construction du grid initial de thetas. Il faut construire au préalable les vecteurs lb_t et ub_t car on ne peut pas les déclarer de manière globale.
  //   // n correspond au nombre de points TOTAL
  int dim_theta=lb_t.size();
  // division de chaque dimension en npoints : on génère dim_theta permutations de {0,npoints-1}.
  std::vector<VectorXd> perm(dim_theta);
  for (int i=0;i<dim_theta;i++){
    perm[i]=randpert(npoints);
  }
  // calcul des coordonnées de chaque point par un LHS.
  vector<VectorXd> grid;
  VectorXd theta_courant(dim_theta);
  for(int i=0;i<npoints;i++){
    for (int j=0;j<dim_theta;j++){
      theta_courant(j)=lb_t(j)+(ub_t(j)-lb_t(j))*(perm[j](i)+distU(generator))/double(npoints);
    }
    grid.push_back(theta_courant);    
  }
  return grid;
}

double truth_function(VectorXd const &x){
  //fonction vérité
  return x(0);
};

double my_model(VectorXd const &x, VectorXd const &theta){
  //modèle
  return x(0)*sin(2.0*theta(0)*x(0))+(x(0)+0.15)*(1-theta(0));
};

/* Evaluate Kernel of the Stochastic process for the two points x and y, given the parameters in par:
	- par(0) is the variance,
	- par(1) is the correlation length
*/
double Kernel_GP(Eigen::VectorXd const &x, Eigen::VectorXd const &y, const Eigen::VectorXd &par){
  // std /lcorx / sobs / lcortheta
  double sqdistx=(x-y).head(dimx).squaredNorm();
  double sqdistt=(x-y).tail(dimtheta).squaredNorm();
  double kx=exp(-0.5*sqdistx/pow(par(1),2));
  double kt=exp(-0.5*sqdistt/pow(par(3),2));
	return pow(par(0),2)*kx*kt; /* matern 1/2 */ //on multiplie les deux kernels
}

double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &xp, Eigen::VectorXd const &hpar){
  //Kernel de l'erreur de modèle. hpar(0) = sig_edm, hpar(1) = sig_exp, hpar(2) = lcor
  return hpar(0)*hpar(0)*exp(-pow((x(0)-xp(0))/hpar(2),2)*.5); /* squared exponential kernel */
  double d=abs(x(0)-xp(0));
  return pow(hpar(0),2)*(1+((2.24*d)/hpar(2))+1.66*pow(d/hpar(2),2))*exp(-(2.24*d)/hpar(2)); /*Matern 5/2*/
  return pow(hpar(0),2)*exp(-(d)/hpar(2)); /*Matern 1/2*/
  return pow(hpar(0),2)*(1+(1.732*d)/hpar(2))*exp(-(1.732*d)/hpar(2)); /*Matern 3/2*/ // marche bien
  return hpar(0)*hpar(0)*exp(-pow((x(0)-xp(0))/hpar(2),2)*.5); /* squared exponential kernel */
};

double PriorMean(VectorXd const &x, VectorXd const &hpars){
  return 0;
}

double logprior_hpars(VectorXd const &hpars){
  return 0;//-2*(log(hpars(1)))-2*(log(hpars(0)));
}

double logprior_pars(VectorXd const &pars){
 return 0;
}

VectorXd RtoGP(const VectorXd &X, const VectorXd &theta){
  //concatène les deux vecteurs pour un format adapté au GP OLM
  if(X.size()!=dimx || theta.size()!=dimtheta){
    cerr << "erreur rtogp. dimx : " << X.size() << ", dimt : " << theta.size() << endl;
  }
  VectorXd XGP(dimgp);
  XGP.head(dimx)=X;
  XGP.tail(dimtheta)=theta;
  return XGP;
}

double myoptfunc_koh(const std::vector<double> &x, std::vector<double> &grad, void *data){
	/* fonction à optimiser pour trouver les hpars koh.*/
  VectorXd hpars(x.size());
  hpars(0)=exp(x[0]);
  hpars(1)=exp(x[1]);
  hpars(2)=x[2];
  //j'ai besoin de la mcmc(calcul ll), gp (calcul f), obs (calcul data).
  tuple_KOH* tp=(tuple_KOH*) data;
  //je vais essayer de faire en sorte de ne pas recopier les objets, surtout le GP.
  const vector<DATA>* yobs= get<0>(*tp); 
  const GP* gp=get<1>(*tp);
  const MCMC_opti* mcmc=get<2>(*tp);
  const vector<VectorXd>* grid=get<3>(*tp);
  const double *logvstyp=get<4>(*tp);
  //calcul de la logvs sur le grid.
  vector<double> prob(grid->size());
  transform(grid->begin(),grid->end(),prob.begin(),[yobs,gp,mcmc,hpars](VectorXd const & theta)->double{
    //calcul des data
    vector<DATA> residuals(yobs->size());
    transform(yobs->begin(),yobs->end(),residuals.begin(),[gp,theta](DATA const &d)->DATA{
      VectorXd X=d.GetX();
      DATA dat; dat.SetX(X); 
      dat.SetValue(d.Value()-gp->EvalMean(RtoGP(X,theta)));
      return dat;
    });
    return mcmc->loglikelihood(&residuals,hpars);
  });
  //normalisation et passage à l'exponentielle.
  //le max doit être le même sur toutes les itérations... d'où la nécessité de le définir à l'extérieur !!!
  transform(prob.begin(),prob.end(),prob.begin(),[logvstyp](const double &d)->double {
    double f=exp(d-*logvstyp);
    if(isinf(f)){cerr << "erreur myoptfunc_koh : infini" << endl;}
    return f;
  });
  //calcul de l'intégrale. suppose un grid régulier.
  double res=accumulate(prob.begin(),prob.end(),0.0); res/=prob.size();
  //cout << "hpars testés : " << hpars.transpose() << " int : " << res << endl;
  return res;
};

double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
	/* This is the function you optimize for defining the GP */
	GP* proc = (GP*) data;											//Pointer to the GP
	Eigen::VectorXd p(x.size());									//Parameters to be optimized
	for(int i=0; i<(int) x.size(); i++) p(i) = x[i];				//Setting the proposed value of the parameters
	double value = proc->SetGP(p);									//Evaluate the function
	if (!grad.empty()) {											//Cannot compute gradient : stop!
		std::cout << "Asking for gradient, I stop !" << std::endl; exit(1);
	}
	neval++;														//increment the number of evaluation count
	return value;
};

VectorXd Hpars_koh(const GP &gp, const MCMC_opti & mcmc, const vector<DATA> & yobs, vector<double> const & lb_hpars,vector<double> const & ub_hpars, const VectorXd &hpars_guess, const VectorXd & lb_t, const VectorXd & ub_t){
  //construction d'un grid pour les thetas
  vector<VectorXd> grid=InitGridRect(lb_t,ub_t,300); //100 pts dans chaque dimension.
  //calcul d'une valeur typique pour la logvskoh
  const double logvstyp=30;
  //construction du tuple_koh
  tuple_KOH tp = make_tuple(&yobs,&gp,&mcmc,&grid,&logvstyp);
  //paramètres de l'optimisation
  int maxeval=5000;
  vector<double> lb_hpars_koh=lb_hpars; lb_hpars_koh[0]=log(lb_hpars_koh[0]); lb_hpars_koh[1]=log(lb_hpars_koh[1]);
  vector<double> ub_hpars_koh=ub_hpars; ub_hpars_koh[0]=log(ub_hpars_koh[0]); ub_hpars_koh[1]=log(ub_hpars_koh[1]);
  vector<double> x(hpars_guess.size());
  x[0]=log(hpars_guess(0));
  x[1]=log(hpars_guess(1));
  x[2]=hpars_guess(2);
  nlopt::opt opt(nlopt::LN_SBPLX, x.size());    /* algorithm and dimensionality */
  opt.set_max_objective(myoptfunc_koh, &tp); /*l'opti se fait sur le log des std.*/
  opt.set_lower_bounds(lb_hpars_koh);
  opt.set_upper_bounds(ub_hpars_koh);
  opt.set_maxeval(maxeval);
  opt.set_ftol_rel(1e-4);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 0: erreur
  VectorXd res(x.size());
  res(0)=exp(x[0]);
  res(1)=exp(x[1]);
  res(2)=x[2];
  return res;
}

double test_gp(GP &gp,VectorXd lb_t, VectorXd ub_t, VectorXd lb_x, VectorXd ub_x){
  //calcul de la norme L2 du GP par rapport au modèle, pour vérifier sa validité.
  int dx=100;
  int dt=100;
  vector<VectorXd> gridX=InitGridRect(lb_x,ub_x,dx);
  vector<VectorXd> gridT=InitGridRect(lb_t,ub_t,dt);
  double L2=0;
  for(int i=0;i<dx;i++){
    for(int j=0;j<dt;j++){
      VectorXd X=gridX[i];
      VectorXd T=gridT[j];
      VectorXd I(dimgp);
      I.head(dimx)=X;
      I.tail(dimtheta)=T;
      L2+=pow(gp.EvalMean(I)-my_model(X,T),2);
    }
  }
  L2/=(dx*dt);
  return L2;
}

double ll_approx_edm(GP &gp, VectorXd thetaprime, VectorXd hpars_typical, MCMC_opti &mcmc, vector<DATA> &yobs,default_random_engine &generator){
  //renvoie la vraisemblance d'une valeur de theta, à ses hyperparamètres optimisés, en utilisant un GP pour f.
  //transformation nécessaire pour utiliser le GP
  vector<VectorXd> vI(yobs.size());
  for (int i=0;i<yobs.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=yobs[i].GetX();
    I.tail(dimtheta)=thetaprime;
    vI[i]=I;
  }
  //calcul des données obs-ftheta et optimisation des hyperparamètres.
  vector<DATA> datamoy;
  for (int i=0;i<yobs.size();i++){
    VectorXd I(dimgp);
    I=vI[i];
    double Pred=gp.EvalMean(I);
    DATA dat; dat.SetX(vI[i].head(dimx)); dat.SetValue(yobs[i].Value()-Pred);
    datamoy.push_back(dat);  
  }
  VectorXd hparsopti=mcmc.Opti_hpars(datamoy,hpars_typical);
  return mcmc.loglikelihood(&datamoy,hparsopti);
}

double ll_approx_noedm(GP &gp, VectorXd thetaprime, VectorXd hpars_typical, MCMC_opti &mcmc, vector<DATA> &yobs,default_random_engine &generator){
  //renvoie la vraisemblance d'une valeur de theta, sans erreur de modèle, en utilisant un GP pour f.
  //transformation nécessaire pour utiliser le GP
  vector<VectorXd> vI(yobs.size());
  for (int i=0;i<yobs.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=yobs[i].GetX();
    I.tail(dimtheta)=thetaprime;
    vI[i]=I;
  }
  //calcul des données obs-ftheta et optimisation des hyperparamètres.
  vector<DATA> datamoy;
  for (int i=0;i<yobs.size();i++){
    VectorXd I(dimgp);
    I=vI[i];
    double Pred=gp.EvalMean(I);
    DATA dat; dat.SetX(vI[i].head(dimx)); dat.SetValue(yobs[i].Value()-Pred);
    datamoy.push_back(dat);  
  }
  VectorXd hparsnoedm(3); //hpars de l'edm
  hparsnoedm << 0,0.01,1; //sedm = 0 et sobs = la vraie valeur. On n'en a pas dans ce problème mais on met 0.01.
  return mcmc.loglikelihood(&datamoy,hparsnoedm);
}


void printGP(GP &gp, const char* file_name,VectorXd lb_x, VectorXd ub_x){
  //affiche le GP dans un fichier
  int ndim=1;
  FILE* out = fopen(file_name,"w");
  fprintf(out,"#Observation du pg. ndim premieres colonnes : coordonnes, avant-derniere colonne : moyenne pg calibre, derniere colonne : 1 sd du pg calibre \n");
  for (unsigned is = 0; is < 500; is++)
    {
      VectorXd x(ndim);
      for (unsigned id = 0; id < ndim; id++){
	x(id) =lb_x(0)+(ub_x(0)-lb_x(0))*double(is)/500;
	fprintf(out,"%e ",x(id));
      }
      VectorXd eval = gp.Eval(x);
      fprintf(out,"%e %e %e\n",eval(0),eval(0)+sqrt(eval(1)),eval(0)-sqrt(eval(1)));
    }
  fclose(out);
}

//faire une MCMC "classique", en utilisant un surrogate pour f. On renvoie un ensemble de DATA que sont les échantillons obtenus.
vector<VectorXd> DoMcmcOpti(GP const &gp, vector<DATA> const &yobs, vector<Vec_aug> const & theta_aug, VectorXd Xinit, MatrixXd COVinit, MCMC_opti & mcmc, int nsamples, default_random_engine & generator){
  int nsteps=100000; //1e6 steps
  int nburn=1000; //nombre d'étapes de burn. On ne choisit pas de samples là-dedans.
  int dimmcmc=Xinit.size();
  vector<VectorXd> v;
  //lambda functions ^^
  //choix d'hpars
  auto opthpars=[theta_aug,& generator]()-> VectorXd{
    int indice=distU(generator)*theta_aug.size();
    return theta_aug[indice].second;
  };
  //calcul de loglikelihood
  auto ll=[&gp, &yobs, &mcmc](VectorXd &theta, VectorXd &hpars)-> double {
    //création des data
    vector<DATA> data(yobs.size());
    transform(yobs.begin(),yobs.end(),data.begin(),[&gp,&theta](DATA const &d)->DATA{
      VectorXd X=d.GetX();
      DATA dat; dat.SetX(X); dat.SetValue(d.Value()-gp.EvalMean(RtoGP(X,theta)));
      return dat;
    });
    return mcmc.loglikelihood(&data,hpars);
  };
  MatrixXd sqrtCOV=COVinit.llt().matrixL();
  VectorXd Xcurrent=Xinit;
  VectorXd hparscurrent=opthpars();
  int naccept=0;
  double fcurrent=ll(Xcurrent,hparscurrent);
  auto begin=chrono::steady_clock::now();
  cout << "Running MCMC with " << nsteps << " steps..." << endl;
  cout << "opthpars : " << hparscurrent.transpose() << endl;
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dimmcmc);
    for (int j=0;j<dimmcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    double fcandidate=ll(Xcandidate,hparscurrent); //pas de priors pour le moment
    if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
      if(Xcandidate(0)>-0.5 && Xcandidate(0)<1.5){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        //hparscurrent=opthpars();
      }
    }
    if(i%(nsteps/nsamples)==0 && i>nburn){v.push_back(Xcurrent);}
  }
  
  double ns_d=nsteps*1.0;
  double acc_rate= naccept /ns_d;
  auto end=chrono::steady_clock::now();
  cout << "chaîne terminée. " << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct" << endl;
  cout << "nombre d'échantillons retenus : " << v.size() << endl;
  return v;
}

vector<VectorXd> DoMcmcHparsFixed(GP const &gp, vector<DATA> const &yobs, VectorXd const &hpars_fixed, VectorXd Xinit, MatrixXd COVinit, MCMC_opti & mcmc, int nsamples, default_random_engine & generator){
  int nsteps=100000; //1e6 steps
  int nburn=1000; //nombre d'étapes de burn. On ne choisit pas de samples là-dedans.
  int dimmcmc=Xinit.size();
  vector<VectorXd> v;
  //lambda functions ^^
  //calcul de loglikelihood
  auto ll=[&gp, &yobs, &mcmc](VectorXd &theta, VectorXd &hpars)-> double {
    //création des data
    vector<DATA> data(yobs.size());
    transform(yobs.begin(),yobs.end(),data.begin(),[&gp,&theta](DATA const &d)->DATA{
      VectorXd X=d.GetX();
      DATA dat; dat.SetX(X); dat.SetValue(d.Value()-gp.EvalMean(RtoGP(X,theta)));
      return dat;
    });
    return mcmc.loglikelihood(&data,hpars);
  };
  MatrixXd sqrtCOV=COVinit.llt().matrixL();
  VectorXd Xcurrent=Xinit;
  VectorXd hparscurrent=hpars_fixed;
  int naccept=0;
  double fcurrent=ll(Xcurrent,hparscurrent);
  auto begin=chrono::steady_clock::now();
  cout << "Running MCMC with " << nsteps << " steps..." << endl;
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dimmcmc);
    for (int j=0;j<dimmcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    double fcandidate=ll(Xcandidate,hparscurrent); //pas de priors pour le moment
    if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
      if(Xcandidate(0)>-0.5 && Xcandidate(0)<1.5){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        //hparscurrent=opthpars();
      }
    }
    if(i%(nsteps/nsamples)==0 && i>nburn){v.push_back(Xcurrent);}
  }
  
  double ns_d=nsteps*1.0;
  double acc_rate= naccept /ns_d;
  auto end=chrono::steady_clock::now();
  cout << "chaîne terminée. " << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct" << endl;
  cout << "nombre d'échantillons retenus : " << v.size() << endl;
  return v;
}

VectorXd DrawRandOptHpars(GP const &gp, vector<DATA> const &yobs, MCMC_opti const & mcmc, vector<Vec_aug> const & Augdata, default_random_engine &generator){
  //évalue la vraisemblance de chaque vecteur augmenté (theta+ hpars opt de theta). Choisir un hpars aléatoirement parmi cette liste, en fonction de la vraisemblance du point.
  vector<double> probs;
  for(Vec_aug const &adata:Augdata){
    //construction des data associées à theta
    VectorXd theta=adata.first;
    VectorXd hpars_opt_theta=adata.second;
    vector<DATA> residuals(yobs.size());
    transform(yobs.begin(),yobs.end(),residuals.begin(),[gp,theta](DATA const &d)->DATA{
      VectorXd X=d.GetX();
      DATA dat; dat.SetX(X); 
      dat.SetValue(d.Value()-gp.EvalMean(RtoGP(X,theta)));
      return dat;
    });
    double loglik=mcmc.loglikelihood(&residuals,hpars_opt_theta);
    probs.push_back(loglik);
  }
  //passage à l'exponentielle
  transform(probs.begin(),probs.end(),probs.begin(),[](double d)-> double{
    double f=exp(d);
    if (isinf(f)){cerr << "erreur : infini" << endl;}
    return f;
  });
  //random choix d'un hpars parmi la liste
  double cumsum=accumulate(probs.begin(),probs.end(),0.0);
  double threshold=cumsum*distU(generator);
  double cur_sum=0;
  int index=0;
  for(int i=0;i<probs.size();i++){
    cur_sum+=probs[i];
    if(cur_sum>=threshold){
      index=i;
      break;
    }
  }
  return Augdata[index].second;
}

const double Big = -1.e16;

int main( int argc, char **argv){
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distU(0,1);
  std::normal_distribution<double> distN(0,1);
  generator.seed(16);

  // Inputs:
  int ntheta_init = 6; //taille du DoE initial
  const int nobs= 10; //nombre d'observations de la réalité
  double sig_noise=1e-2; //bruit d'observation


  //Bornes de x
  VectorXd lb_x(dimx); lb_x(0)=0;
  VectorXd ub_x(dimx); ub_x(0)=1;
  //bornes de theta
  VectorXd lb_t(dimtheta); lb_t(0)=-0.5;
  VectorXd ub_t(dimtheta); ub_t(0)=1.5;

  //bornes des hpars DE Z (erreur de modèle)
  const int dim_hparsz=3;
  std::vector<double> lb_hpars(dim_hparsz); lb_hpars[0]=0.005;lb_hpars[1]=0.005;lb_hpars[2]=0.1; //-5 avant
  std::vector<double> ub_hpars(dim_hparsz); ub_hpars[0]=2;ub_hpars[1]=2;ub_hpars[2]=2; //bornes sur l'optimisation des hpars. edm exp lcor
  VectorXd hpars_typical_z(dim_hparsz);
  for(int i=0;i<dim_hparsz;i++){hpars_typical_z(i)=lb_hpars[i];}
  //hpars représentatifs de no edm
  hpars_noedm(0)=0; hpars_noedm(1)=sig_noise;hpars_noedm(2)=666;

  //bornes des hpars DU SURROGATE GP(f)
  const int nhpars_gp = 4; //nombre de hpars pour le GPOLM. 
	MatrixXd Bounds_hpars_gp(2,nhpars_gp);
	Bounds_hpars_gp(0,0) = 0.1; Bounds_hpars_gp(1,0) = 1; //stdedm
	Bounds_hpars_gp(0,1) = 0.05; Bounds_hpars_gp(1,1) = 5; //lcorx
	Bounds_hpars_gp(0,2) = 1e-8; Bounds_hpars_gp(1,2) = 1e-7; //sigma exp (très faible)
  Bounds_hpars_gp(0,3) = 0.05; Bounds_hpars_gp(1,3) = 5; //lcortheta
  VectorXd hpars_typical_gp(nhpars_gp); for (int i=0;i<nhpars_gp;i++){hpars_typical_gp(i)=Bounds_hpars_gp(0,i);}

  //génération des observations. yobs contient (X, obs à X)
  vector<DATA> yobs;
  VectorXd yobsXd(nobs); //on stocke aussi la valeur des observations dans un vectorXd, pratique.
  for (int i=0;i<nobs;i++){
    DATA dat;
    VectorXd x(1);
    x=lb_x+(ub_x-lb_x)*distU(generator);
    double f=truth_function(x)+sig_noise*distN(generator);
    dat.SetX(x); dat.SetValue(f);
    yobs.push_back(dat);
    yobsXd(i)=f;
  }

  //création d'une instance de densité car c'est plus pratique
  DoE doe_null(lb_t,ub_t,2);// gridsize peu importe
  doe_null.Fill(my_model,&yobs);
  Density MainDensity(doe_null);
  MainDensity.SetModel(my_model);
  MainDensity.SetKernel(Kernel);
  MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(PriorMean);
  MCMC test(MainDensity,1);
  MCMC_opti mcmc(test,1);

  //Construction du DoE initial, un LHS.
  vector<VectorXd> gridLHSINIT=InitGridLHS(lb_t,ub_t,ntheta_init);

  //On évalue le modèle sur le DoE initial.
  //data_GP contient ((X,theta),f(X,theta)).

  FILE* out = fopen("results/observations.gnu","w");
  fprintf(out,"#Fichier du LHS initial. ndim premieres colonnes : coordonnes, derniere colonne : réalisation du modèle\n");
  vector<DATA> data_GP;
  vector<Vec_aug> Augdata;
  for(const VectorXd & theta:gridLHSINIT){
    vector<DATA> residuals(nobs);
    for(int j=0;j<nobs;j++){
      VectorXd X=yobs[j].GetX();
      VectorXd I(dimgp);
      I.head(dimx)=X;
      I.tail(dimtheta)=theta;
      for(unsigned id=0; id<dimgp; id++){fprintf(out,"%e ",I(id));}
      double model_eval=my_model(I.head(dimx),I.tail(dimtheta));
      DATA dat; dat.SetX(I); dat.SetValue(model_eval);
      fprintf(out,"%e \n",dat.Value());
      data_GP.push_back(dat);
      DATA dat2; dat2.SetX(X); dat2.SetValue(yobs[j].Value()-model_eval);
      residuals[j]=dat2;
    }
    //optimisation des hyperparamètres de z pour cette valeur de theta.
    VectorXd hparsopt=mcmc.Opti_hpars(residuals,hpars_typical_z);
    //stockage dans le vecteur Augdata.
    Augdata.push_back(make_pair(theta,hparsopt));
  }
  fclose(out);


  //utilities
  //vecteurs où on récup les normes l2 successives
  vector<double> L2opt;
  vector<double> L2noedm;
  vector<double> L2koh;
  //on récup les métamodèles successifs
  vector<GP> GPsopt;
  vector<GP> GPsnoedm;
  vector<GP> GPskoh;
  //on récup les points visités
  vector<VectorXd> total_visited_points_opt;
  vector<VectorXd> total_visited_points_noedm;
  vector<VectorXd> total_visited_points_koh;
  for(const VectorXd & theta:gridLHSINIT){
    total_visited_points_opt.push_back(theta);
    total_visited_points_noedm.push_back(theta);
    total_visited_points_koh.push_back(theta);
  }


  //Augdata n'est utilisé que par la méthode opti. Par contre data_GP est commun aux autres, donc on le copie.
  vector<DATA> data_GP_opti;
  copy(data_GP.begin(),data_GP.end(),back_inserter(data_GP_opti));

  int nrepet=10; //on répète 5 fois
  int nsamples=3; // on récupère 1 point à chaque fois.

  /*Boucle principale du DoE opti*/
  {
    for(int irepet=0;irepet<nrepet;irepet++){
      //data_GP pour les données du GP, Augdata pour les hpars optimaux.
      //construction du processus gaussien de f à partir du DoE existant.
      GP gp(Kernel_GP);
      gp.SetData(data_GP_opti);
      gp.SetGP(hpars_typical_gp);
      
      cout << "optimisation du GP :" << endl;
      auto begin=chrono::steady_clock::now();
      gp.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_typical_gp,nhpars_gp);
      cout  << "par (Guess) : " << hpars_typical_gp.transpose() << endl;
	    hpars_typical_gp = gp.GetPar();
	    cout  << "par (Optim) (sigmaedm,lcorphi,sigmaobs,lcor2,lcor3,lcor4,lcor5) : " << hpars_typical_gp.transpose() << endl;
      auto end=chrono::steady_clock::now();
      cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;


      //évaluation du GP en norme L2
      double error=test_gp(gp,lb_t,ub_t,lb_x,ub_x);
      L2opt.push_back(error);
      cout << "at step " << irepet << ", surrogate error : " << error << endl;

      //évaluation, à l'aide du gp actuel, des vraisemblances de chaque point, prises avec leurs hpars optimaux. Cela constitue un score qui permet de choisir les hpars pour la MCMC ensuite.
      VectorXd hpars_opti=DrawRandOptHpars(gp,yobs,mcmc,Augdata,generator);
      cout << "hpars opti :" << hpars_opti.transpose() << endl;

      //MCMC opti pour récupérer les nouveaux points d'évaluation :
      VectorXd Tinit(dimtheta);
      Tinit(0)=0.5;
      MatrixXd COVinit=MatrixXd::Zero(dimtheta,dimtheta);
      COVinit(0,0)=pow(0.1,2);
      vector<VectorXd> newpoints=DoMcmcHparsFixed(gp,yobs,hpars_opti,Tinit,COVinit,mcmc,nsamples,generator);

      //évaluations de modèle et optimisation des hyperparamètres
      cout << "nouveaux points : ";
      for(const VectorXd & theta:newpoints){
        total_visited_points_opt.push_back(theta);
        cout << theta.transpose() << " ";
        vector<DATA> residuals(nobs);
        for(int j=0;j<nobs;j++){
          VectorXd X=yobs[j].GetX();
          VectorXd I=RtoGP(X,theta);
          double model_eval=my_model(I.head(dimx),I.tail(dimtheta));
          DATA dat; dat.SetX(I); dat.SetValue(model_eval);
          data_GP_opti.push_back(dat);
          DATA dat2; dat2.SetX(X); dat2.SetValue(yobs[j].Value()-model_eval);
          residuals[j]=dat2;
        }
        //optimisation des hyperparamètres de z pour cette valeur de theta.
        VectorXd hparsopt=mcmc.Opti_hpars(residuals,hpars_typical_z);
        //stockage dans le vecteur Augdata.
        Augdata.push_back(make_pair(theta,hparsopt));
      }
      cout << endl;
      GPsopt.push_back(gp);
    }
  }

  vector<DATA> data_GP_noedm;
  copy(data_GP.begin(),data_GP.end(),back_inserter(data_GP_noedm));

  /*Boucle principale du DoE sans EDM*/
  {
    for(int irepet=0;irepet<nrepet;irepet++){
      //data_GP pour les données du GP, Augdata pour les hpars optimaux.
      //construction du processus gaussien de f à partir du DoE existant.
      GP gp(Kernel_GP);
      gp.SetData(data_GP_noedm);
      gp.SetGP(hpars_typical_gp);
      
      cout << "optimisation du GP :" << endl;
      auto begin=chrono::steady_clock::now();
      gp.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_typical_gp,nhpars_gp);
      cout  << "par (Guess) : " << hpars_typical_gp.transpose() << endl;
	    hpars_typical_gp = gp.GetPar();
	    cout  << "par (Optim) (sigmaedm,lcorphi,sigmaobs,lcor2,lcor3,lcor4,lcor5) : " << hpars_typical_gp.transpose() << endl;
      auto end=chrono::steady_clock::now();
      cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;

      //évaluation du GP en norme L2
      double error=test_gp(gp,lb_t,ub_t,lb_x,ub_x);
      L2noedm.push_back(error);
      cout << "at step " << irepet << ", surrogate error : " << error << endl;

      //MCMC sans EDM pour récupérer les nouveaux points d'évaluation :
      VectorXd Tinit(dimtheta);
      Tinit(0)=0.5;
      MatrixXd COVinit=MatrixXd::Zero(dimtheta,dimtheta);
      COVinit(0,0)=pow(0.1,2);
      vector<VectorXd> newpoints=DoMcmcHparsFixed(gp,yobs,hpars_noedm,Tinit,COVinit,mcmc,nsamples,generator);

      //évaluations de modèle
      cout << "nouveaux points : ";
      for(const VectorXd & theta:newpoints){
        total_visited_points_noedm.push_back(theta);
        cout << theta.transpose() << " ";
        for(int j=0;j<nobs;j++){
          VectorXd X=yobs[j].GetX();
          VectorXd I=RtoGP(X,theta);
          double model_eval=my_model(I.head(dimx),I.tail(dimtheta));
          DATA dat; dat.SetX(I); dat.SetValue(model_eval);
          data_GP_noedm.push_back(dat);
        }
      }
      cout << endl;
      GPsnoedm.push_back(gp);
    }
  }

  vector<DATA> data_GP_koh;
  copy(data_GP.begin(),data_GP.end(),back_inserter(data_GP_koh));

  /*Boucle principale du DoE KOH*/
  vector<VectorXd> successive_hpars_koh;
  {
    for(int irepet=0;irepet<nrepet;irepet++){
      //data_GP pour les données du GP, Augdata pour les hpars optimaux.
      //construction du processus gaussien de f à partir du DoE existant.
      GP gp(Kernel_GP);
      gp.SetData(data_GP_koh);
      gp.SetGP(hpars_typical_gp);
      
      cout << "optimisation du GP :" << endl;
      auto begin=chrono::steady_clock::now();
      gp.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_typical_gp,nhpars_gp);
      cout  << "par (Guess) : " << hpars_typical_gp.transpose() << endl;
	    hpars_typical_gp = gp.GetPar();
	    cout  << "par (Optim) (sigmaedm,lcorphi,sigmaobs,lcor2,lcor3,lcor4,lcor5) : " << hpars_typical_gp.transpose() << endl;
      auto end=chrono::steady_clock::now();
      cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;

      //évaluation du GP en norme L2
      double error=test_gp(gp,lb_t,ub_t,lb_x,ub_x);
      L2koh.push_back(error);
      cout << "at step " << irepet << ", surrogate error : " << error << endl;

      //calcul hpars KOH
      VectorXd kohhpars=Hpars_koh(gp,mcmc,yobs,lb_hpars,ub_hpars,hpars_typical_z,lb_t,ub_t);
      cout << "hpars koh :" << kohhpars.transpose() << endl;
      successive_hpars_koh.push_back(kohhpars);

      //MCMC KOH pour récupérer les nouveaux points d'évaluation :
      VectorXd Tinit(dimtheta);
      Tinit(0)=0.5;
      MatrixXd COVinit=MatrixXd::Zero(dimtheta,dimtheta);
      COVinit(0,0)=pow(0.1,2);
      vector<VectorXd> newpoints=DoMcmcHparsFixed(gp,yobs,kohhpars,Tinit,COVinit,mcmc,nsamples,generator);
      //évaluations de modèle
      cout << "nouveaux points : ";
      for(const VectorXd & theta:newpoints){
        total_visited_points_koh.push_back(theta);
        cout << theta.transpose() << " ";
        for(int j=0;j<nobs;j++){
          VectorXd X=yobs[j].GetX();
          VectorXd I=RtoGP(X,theta);
          double model_eval=my_model(I.head(dimx),I.tail(dimtheta));
          DATA dat; dat.SetX(I); dat.SetValue(model_eval);
          data_GP_koh.push_back(dat);
        }
      }
      cout << endl;
      GPskoh.push_back(gp);
    }
  }



  /*print des normes L2*/
  ofstream ofile("results/L2err.gnu");
  for(int i=0;i<nrepet;i++){
    ofile << L2opt[i] << " " << L2koh[i] << " " << L2noedm[i] << endl;
  }
  ofile.close();

  /*print des GPs*/
  //grid en x
  VectorXd tc(1); tc(0)=0.5; //theta_chosen
  VectorXd x_min(1); x_min(0)=0;
  VectorXd x_max(1); x_max(0)=1;
  vector<VectorXd> grid_x=InitGridRect(x_min,x_max,50);
  {
     ofile.open("results/gpopt.gnu");
     for_each(grid_x.begin(),grid_x.end(),[&ofile,GPsopt,tc](VectorXd const &X)-> void{
       ofile << X(0) << " ";
       for_each(GPsopt.begin(),GPsopt.end(),[&ofile,X,tc](const GP & gp)->void{
         VectorXd I=RtoGP(X,tc);
         VectorXd Pred=gp.Eval(I);
         ofile << Pred(0) << " " <<sqrt(Pred(1)) << " ";
       });
       ofile << endl;
     });
     ofile.close();
  }
  {
     ofile.open("results/gpkoh.gnu");
     for_each(grid_x.begin(),grid_x.end(),[&ofile,GPskoh,tc](VectorXd const &X)-> void{
       ofile << X(0) << " ";
       for_each(GPskoh.begin(),GPskoh.end(),[&ofile,X,tc](const GP & gp)->void{
         VectorXd I=RtoGP(X,tc);
         VectorXd Pred=gp.Eval(I);
         ofile << Pred(0) << " " <<sqrt(Pred(1)) << " ";
       });
       ofile << endl;
     });
     ofile.close();
  }
  {
     ofile.open("results/gpnoedm.gnu");
     for_each(grid_x.begin(),grid_x.end(),[&ofile,GPsnoedm,tc](VectorXd const &X)-> void{
       ofile << X(0) << " ";
       for_each(GPsnoedm.begin(),GPsnoedm.end(),[&ofile,X,tc](const GP & gp)->void{
         VectorXd I=RtoGP(X,tc);
         VectorXd Pred=gp.Eval(I);
         ofile << Pred(0) << " " <<sqrt(Pred(1)) << " ";
       });
       ofile << endl;
     });
     ofile.close();
  }

  {
     ofile.open("results/refsol.gnu");
     for_each(grid_x.begin(),grid_x.end(),[&ofile,tc](VectorXd const &X)-> void{
       ofile << X(0) << " " << my_model(X,tc) << endl;
     });
     ofile.close();
  }

  /*print des points visités successifs. attention c'est écrit pour la 1D.*/
  {
    ofile.open("results/visited_thetas.gnu");
    for(int i=0;i<total_visited_points_koh.size();i++){
      ofile << total_visited_points_opt[i] << " " << total_visited_points_koh[i] << " " << total_visited_points_noedm[i] << endl;
    }
    ofile.close();
  }

  /*print des hyperparamètres KOH ?*/
  {
    ofile.open("results/hparskoh.gnu");
    for(const VectorXd &h:successive_hpars_koh){
      ofile << h.transpose() << endl;
    }
    ofile.close();
  }

  exit(0);
 

  /*affichage des GPs*/
  /*
  VectorXd theta_aff(1);
  theta_aff(0)=0.9;
  vector<VectorXd> lhs_x=InitGridLHS(lb_x,ub_x,200);
  out=fopen("results/gp.gnu","w");
  for (int i=0;i<lhs_x.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=lhs_x[i];
    I.tail(dimtheta)=theta_aff;
    VectorXd Pred=gp1.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    VectorXd Pred2=gp2.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    fprintf(out,"%e %e %e\n",I(0),Pred(0),Pred2(0));
  }
  fclose(out);
  //affichage en -0.1
  theta_aff(0)=-0.1;
  out=fopen("results/gp2.gnu","w");
  for (int i=0;i<lhs_x.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=lhs_x[i];
    I.tail(dimtheta)=theta_aff;
    VectorXd Pred=gp1.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    VectorXd Pred2=gp2.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    fprintf(out,"%e %e %e\n",I(0),Pred(0),Pred2(0));
  }
  fclose(out);
  //affichage en 0.5
  theta_aff(0)=0.5;
  out=fopen("results/gp3.gnu","w");
  for (int i=0;i<lhs_x.size();i++){
    VectorXd I(dimgp);
    I.head(dimx)=lhs_x[i];
    I.tail(dimtheta)=theta_aff;
    VectorXd Pred=gp1.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    VectorXd Pred2=gp2.Eval(I); //prédiction du gp. case 1 : moyenne, case 2 : variance.
    fprintf(out,"%e %e %e\n",I(0),Pred(0),Pred2(0));
  }
  fclose(out);
  */




  exit(0);
}

