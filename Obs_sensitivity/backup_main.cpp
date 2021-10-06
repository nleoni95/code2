// Dans ce fichier : on met en oeuvre différents algorithmes d'optimisation pour KOH. Voir lequel est le meilleur.
// On met en place une quadrature pour évaluer de manière précise l'intégrale KOH.
// On regarde maintenant la sensibilité aux observations.
data.

#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <random>
#include <functional>
#include "pcb++.h"
#include "cub++.h"
#include "vstoch++.h"
#include "mstoch++.h"
#include "sampler.h"
#include <ctime>
#include "halton.cpp"
#include "densities.h"


using namespace std;
using namespace Eigen;

int neval=1;
//std::default_random_engine generator;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);
vector<DATA> data;
vector<VectorXd> Grid;


int gridstep=140; //Finesse du grid en theta
int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
int dim_theta=1;
int alpha_ig=5; // paramètres du prior sur lcor
double beta_ig=0.4;


VectorXd indices(int const s, int const n, int const p){
  //renvoie le multi-indice correspondant à l'indice courant s dans un tableau de dimension p et de taille n dans chaque direction.
  VectorXd multiindice(p);
  int indloc;
  int remainder=s;
  for(int pp=p-1;pp>-1;pp--){
    indloc=(int) remainder % n; //On commence par le coefficient le plus à droite.
    multiindice(pp)=indloc;
    remainder=(remainder-indloc)/n;
  }
  return multiindice;
}

int grosindice(VectorXd const &v, int const n){
  // n : taile du tableau dans une dimension.
  //renvoie le gros indice qui correspond au multiindice v.
  //on compte bien de la façon suivante : 000 001 010 011 etc.
  int p=v.size(); //dimension du tableau
  int ans=0;
  for (int i=0;i<p;i++){
    ans+=v(p-1-i)*pow(n,i);
  }
  return ans;
}

vector<VectorXd> InitGrid(VectorXd const &lb_t,VectorXd const &ub_t, int const n){
  // Construction du grid initial de thetas. On le fait une bonne fois pour toutes. Il faut construire au préalable les vecteurs lb_t et ub_t car on ne peut pas les déclarer de manière globale.
  // n correspond au nombre de points par dimension.
  int dim_theta=lb_t.size();
  int npoints=n*dim_theta;
  vector<VectorXd> grid;
  VectorXd theta_courant(dim_theta);
  VectorXd ind_courant(dim_theta);
  for(int i=0;i<npoints;i++){
    ind_courant=indices(i,n,dim_theta);
    for (int j=0;j<dim_theta;j++){
      theta_courant(j)=lb_t(j)+(ind_courant(j)+0.5)*(ub_t(j)-lb_t(j))/double(n);
    }
    grid.push_back(theta_courant);
  }
  return grid;  
}

VectorXd draw_hpars(vector<double> const &lb, vector<double> const &ub){
  //Tirage de hpars selon la prior. edm / exp / lcor. Les deux premiers uniformes tronqués, la dernière inv gamma.
  VectorXd hpars(3);
  hpars(0)=lb[0]+(ub[0]-lb[0])*distU(generator);
  hpars(1)=lb[1]+(ub[1]-lb[1])*distU(generator);
  gamma_distribution<double> dist(alpha_ig,1./beta_ig);
  hpars(2)=1./dist(generator);
  return hpars;
}

double my_function(VectorXd const &x){
  return x(0);
};

double my_model(VectorXd const &x,VectorXd const &theta){
  return x(0)*sin(2.0*theta(0)*x(0))+(x(0)+0.15)*(1-theta(0));
}

double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &xp, Eigen::VectorXd const &hpar){
  //Fonction Kernel sans bruit/
  // hpar(0) = sig_edm
  // hpar(1) = sig_exp
  // hpar(2) = lcor
   return hpar(0)*hpar(0)*exp(-pow((x(0)-xp(0))/hpar(2),2)*.5 ); /* squared exponential kernel */
};

Eigen::MatrixXd Gamma(void *data, Eigen::VectorXd const &hpar){
  // Renvoie la matrice de corrélation avec  bruit
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  Eigen::MatrixXd A(nd,nd);
  for(int i=0; i<nd; i++){
    for(int j=i; j<nd; j++){
      A(i,j) = Kernel((*data2)[i].GetX(),(*data2)[j].GetX(), hpar);
      if(i!=j){
	A(j,i) = A(i,j);
      }else{
	A(i,j) += hpar(1)*hpar(1);					//Noise correlation
      }
    }
  }
  return A;
}

double loglikelihood_fast(VectorXd const &obs, VectorXd const &Alpha, Eigen::VectorXd const &hpar, LDLT<MatrixXd> const &ldlt){
  //renvoie la log-vraisemblance des hyperparamètres hpar, étant donné les données data.
  int nd=obs.size();
  return -0.5*obs.dot(Alpha)-0.5*(ldlt.vectorD().array().log()).sum() -0.5*nd*log(2*3.1415);
}

double loglikelihood(void *data, Eigen::VectorXd const &hpar){
  //renvoie la log-vraisemblance des hyperparamètres hpar, étant donné les données data.
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  MatrixXd G=Gamma(data2,hpar);
  VectorXd obs(nd);
  for(unsigned i=0; i<nd; i++) obs(i) = (*data2)[i].Value(); // copie des valeurs observées dans un VectorXd
  LDLT<MatrixXd> ldlt(G);
  VectorXd Alpha=ldlt.solve(obs);
  return loglikelihood_fast(obs,Alpha,hpar,ldlt);
}

double logpost(void *data, Eigen::VectorXd const &hpar){
  //renvoie la log-postérieure (à une constante additive près) des hyperparamètres hpar, étant donné les données data.
  double logprior_sigexp=0;//-log(hpar(1)); //-log(hpar(1)); // prior 1/sigma_exp.
  double logprior_sigedm=0; //-log(hpar(0)); // prior 1/sigma_edm.
  double logprior_l=log(pow(beta_ig,alpha_ig)*pow(hpar(2),-alpha_ig-1)*exp(-beta_ig/hpar(2))/tgamma(alpha_ig)); // prior inverse-gamma sur lcor. Moyenne 1/2, Mode 1/4. Toute petite queue vers 0 et queue assez large vers inf;
  return loglikelihood(data,hpar)+logprior_l;// + logprior_sigexp + logprior_sigedm + logprior_l;
}

double loglikelihood_theta_fast(void *data, Eigen::VectorXd const &hpar, LDLT<MatrixXd> const &ldlt, VectorXd const &theta){
  //Besoin de calculer l'inversion matricielle. 
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  VectorXd obs_theta(nd);
  for (int i=0;i<nd;i++){obs_theta(i)=(*data2)[i].Value()-my_model((*data2)[i].GetX(),theta);}
  VectorXd Alpha=ldlt.solve(obs_theta);
  return loglikelihood_fast(obs_theta,Alpha,hpar,ldlt);
}

double loglikelihood_theta(void *data, Eigen::VectorXd const &hpar, VectorXd const &theta){
  //renvoie la log-vraisemblance des hyperparamètres hpar, étant donné les données data et les paramètres du modèle theta.
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  std::vector<DATA> data3;
  for(unsigned ie=0; ie<nd; ie++){
    DATA dat; dat.SetX((*data2)[ie].GetX()); dat.SetValue((*data2)[ie].Value()-my_model((*data2)[ie].GetX(),theta));
    data3.push_back(dat);
  }
  return loglikelihood(&data3,hpar);
}

double logpost_theta(void *data, Eigen::VectorXd const &hpar,VectorXd const &theta){
  //renvoie la log-postérieure des hyperparamètres hpar, étant donné les données data et les paramètres du modèle theta.
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  std::vector<DATA> data3;
  for(unsigned ie=0; ie<nd; ie++){
    DATA dat; dat.SetX((*data2)[ie].GetX()); dat.SetValue((*data2)[ie].Value()-my_model((*data2)[ie].GetX(),theta));
    data3.push_back(dat);
  }
  return logpost(&data3,hpar);
}


double optfunc__nograd(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /*Fonction à optimiser, revoie la postérieure sans calcul du gradient. On ne peut pas la merger avec logpost_theta car elle a besoin de theta.*/
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  Eigen::VectorXd hpar(3);
  for(unsigned p=0; p<3; p++) {hpar(p) =x[p];} // copie des hyperparamètres dans un VectorXd
  //cout << "hpars opti tested : " << hpar.transpose() << endl;
  return logpost(data2,hpar);
}

double optfunc__koh(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /*Fonction à optimiser, Kennedy O'Hagan. On cherche à estimer l'intégrale moyennée sur un grid uniforme (avec priors uniformes) */
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  Eigen::VectorXd hpar(3);
  for(int p=0; p<3; p++) {hpar(p) =x[p];} // copie des hyperparamètres dans un VectorXd
  MatrixXd G=Gamma(data2,hpar); //les valeurs de data2 ne sont pas correctes car non retranchées de ftheta. Cependant on n'utilise que les X pour calculer G.
  LDLT<MatrixXd> ldlt(G);
  double avg=0;
  int ngrid=Grid.size();
  for (int i=0;i<ngrid;i++){
    avg+=exp(loglikelihood_theta_fast(data2,hpar,ldlt,Grid[i]));
  }
  double logprior_l=log(pow(beta_ig,alpha_ig)*pow(hpar(2),-alpha_ig-1)*exp(-beta_ig/hpar(2))/tgamma(alpha_ig));
  avg=avg*exp(logprior_l);
  avg=avg/(double) ngrid;

  // cout << "hpars koh tested : " << hpar.transpose() <<", value :" << avg << endl;

  return avg;
}

double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* This is the function you optimize for defining the GP. */
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

std::vector<DATA> vectornormalize(const vector<DATA> &v){
  //normalise un vecteur : le maximum est mis à 1
  std::vector<DATA> vp;
  int ns=v.size();
  double max_local=-1e35;
  for (int i=0;i<ns;i++){
    if(v[i].Value()>max_local){max_local=v[i].Value();}
  }
  for (int i=0;i<ns;i++){
    DATA dat; dat.SetX(v[i].GetX()); dat.SetValue(v[i].Value()-max_local);
    vp.push_back(dat);
  }
  return vp;
}

std::vector<DATA> vectorweight1(const vector<DATA> &v){
  //normalise une densité. L'intégrale est mise à 1. Marche parce qu'on est dans un intervalle [0,1].
  std::vector<DATA> vp;
  int ns=v.size();
  double weight=0;
  for (int i=0;i<ns;i++){
    weight+=v[i].Value();
  }
  weight=weight/(double) ns;
  for (int i=0;i<ns;i++){
    DATA dat; dat.SetX(v[i].GetX()); dat.SetValue(v[i].Value()/weight);
    vp.push_back(dat);
  }
  return vp;
}

std::vector<DATA> vectorexp(const vector<DATA> &v){
  //convertit un vecteur de logp en vecteur de probabilités
  std::vector<DATA> vp;
  int ns=v.size();
  for (int i=0;i<ns;i++){
    DATA dat; dat.SetX(v[i].GetX()); dat.SetValue(exp(v[i].Value()));
    vp.push_back(dat);
  }
  return vp;
}

std::vector<DATA> vectorlog(const vector<DATA> &v){
  //convertit un vecteur de p en vecteur de logprobabilités
  std::vector<DATA> vp;
  int ns=v.size();
  for (int i=0;i<ns;i++){
    DATA dat; dat.SetX(v[i].GetX()); dat.SetValue(log(v[i].Value()));
    vp.push_back(dat);
  }
  return vp;
}

int indmax(const vector<DATA> &v){
  // retrouve l'indice maximum du vecteur v.
  int ns=v.size();
  int imax=0;
  double maxloc=v[0].Value();
  for (int i=0;i<ns;i++){
    if(v[i].Value()>maxloc){maxloc=v[i].Value(); imax=i;}
  }
  return imax;
}

vector<DATA> BuildPost(vector<DATA> data, vector<VectorXd> const grid, Eigen::VectorXd const &hpar){
  // Construit une logpostérieure sur le grid, avec une seule valeur d'hyperparamètres. Non normalisée.
  vector<DATA> lpost;
  int ngrid=grid.size();
  for (int i;i<ngrid;i++){
    DATA dat; dat.SetX(grid[i]); dat.SetValue(logpost_theta(&data,hpar,grid[i]));
    lpost.push_back(dat);
  }
  return lpost;
}

void WritePost(const std::vector<DATA> &v, const char* file_name){
  //Ecrit un vector dans un fichier. Pour les samples de la mcmc.
  //Marche avec les VectorXd de n'importe quelle taile.
  FILE* out=fopen(file_name,"w");
  int ndim=v[0].GetX().size(); //nombre de dimensions de VectorXd (2 en général)
  fprintf(out,"#Evaluation fine de la logpost. Obs : Seed : Edm : bruitexp : GridTheta : Tempscalcul: col1:t1 col2:t2 col3:lp \n");
  for (int i=0;i<v.size();i++){
    for (int j=0;j<ndim;j++){
      fprintf(out,"%e ",v[i].GetX()(j));
    }
    fprintf(out,"%e\n",v[i].Value());
  }
  fclose(out);
}

vector<DATA> ReadPost(const char* file_name){
  // Lire un vecteur de DATA dans un fichier. En général c'est plutôt un vecteur de probabilités, mais le format DATA se prête bien à faire ça.
  // La fonction est écrite pour une probabilité en dimension 1.
  double f;
  VectorXd X(1);
  std::vector<DATA> v;
  string line;
  ifstream file (file_name);
  if (file.is_open()){
    getline(file,line); //skip la première ligne sur laquelle on a mis des informations
    while (getline(file,line)){
      stringstream(line) >> X(0) >> f;
      DATA dat; dat.SetX(X); dat.SetValue(f);
      v.push_back(dat);
    }
  }
  file.close();
  return v;
}

double KLDiv(const vector<DATA> &p, const vector<DATA> &q){
  // Calcule la kldiv de p par rapport à q. d'après wiki, p représente les données, observations, ou distrib de proba calculée avec précision. q représente son approximation.
  double kl=0;
  for (int i=0;i<p.size();i++){
    kl+=p[i].Value()*log(p[i].Value()/q[i].Value());
  }
  return kl/(double) p.size();
}

double Evidence(const std::vector<DATA> &lp){
  // Calcule l'evidence d'un modèle ayant généré la logpost lp.
  int ns=lp.size();
  double evi=0;
  for (int i;i<ns;i++){
    evi+=exp(lp[i].Value());
  }
  return evi/(double) ns; // car maintenant on intègre à 1.
}

double Entropy(const vector<DATA> &p){
  int ns=p.size();
  double ent=0;
  for (int i=0;i<ns;i++){
    ent-=log(p[i].Value())*p[i].Value();
  }
  return ent/(double) ns;
}

VectorXd MAP(const vector<DATA> &p){
  return p[indmax(p)].GetX();
}

VectorXd Mean(const vector<DATA> &p){
  int dim=p[0].GetX().size();
  VectorXd AVG=VectorXd::Zero(dim);
  for (int i=0;i<p.size();i++){
    AVG+=p[i].GetX()*p[i].Value();
  }
  return AVG/(double) p.size();
}

double esplog(const vector<DATA> &pbayes, const vector<DATA> &lp){
  //calcule le terme en esplog. On calcule sur la vraisemblance (lp ne doit pas être normalisée)
  if(!pbayes.size()==lp.size()){cout << "erreur de dimension !" << endl;}
  double esplog=0;
  for (int i=0;i<pbayes.size();i++){
    esplog+=pbayes[i].Value()*lp[i].Value();
  }
  esplog=esplog/pbayes.size();
  return esplog;
}

double logint(const vector<DATA> &lp){
  //calcule le terme en logint. On calcule sur la vraisemblance (lp ne doit pas être normalisée)
  double logint=0;
  vector<DATA> p=vectorexp(lp);
  for (int i=0;i<p.size();i++){
    logint+=p[i].Value();
  }
  logint=log(logint)-log(p.size());
  return logint;
}

//fonctions pour MCMC

double f_mcmc(void *data, const VectorXd &t_mcmc){
  VectorXd theta(1);
  VectorXd hpars(3);
  theta(0)=t_mcmc(0);
  hpars(0)=t_mcmc(1);
  hpars(1)=t_mcmc(2);
  hpars(2)=t_mcmc(3);
  return logpost_theta(data,hpars,theta);
}

bool in_bounds(const VectorXd &t_mcmc, const VectorXd &lb_mcmc, const VectorXd &ub_mcmc){
  // Vérifie si le vecteur t_mcmc est dans les clous.
  for (int i=0;i<4;i++){
    if (t_mcmc(i)<lb_mcmc(i) || t_mcmc(i)>ub_mcmc(i)){
      return false;
    }
  }
  return true;
}

VectorXd moy_mcmc(const vector<VectorXd> &thetas, const vector<double> &lp){
  //Calcul d'une moyenne. Attention, formalisme très tuned à notre code : lp contient des logprobabilités non normalisées, thetas est de dimension 4.
  //On va tout d'abord normaliser ces probas.
  int ns=lp.size();
  vector<double> p(ns);
  double weight=0;
  for (int i=0;i<ns;i++){
    p[i]=exp(lp[i]);
    weight+=p[i];
  }
  for (int i=0;i<ns;i++){
    p[i]=p[i]/weight;
  }
  VectorXd Mean=VectorXd::Zero(4);
  for (int i=0;i<ns;i++){
    Mean+=thetas[i]*p[i];
  }
  return Mean;
}

MatrixXd cov_mcmc(const vector<VectorXd> &thetas, const vector<double> &lp){
  //Calcul d'une variance. Attention formalisme très tuné à notre code (voir fonction précédente)
  int ns=lp.size();
  vector<double> p(ns);
  double weight=0;
  for (int i=0;i<ns;i++){
    p[i]=exp(lp[i]);
    weight+=p[i];
  }
  for (int i=0;i<ns;i++){
    p[i]=p[i]/weight;
  }
  VectorXd Mean=moy_mcmc(thetas,lp);
  MatrixXd COV=MatrixXd::Zero(4,4);
  for (int i=0;i<ns;i++){
    COV+=(thetas[i]-Mean)*(thetas[i]-Mean).transpose()*p[i];
  }
  return COV;
}

double ellipse_dist(const VectorXd X, const VectorXd O, const MatrixXd COV){
  // vérifie si le point X est dans l'intervalle de confiance de la distribution gaussienne de maximum O et de matrice de covariance COV. On l'utilise en 4 dimensions. Donc le seuil à 95% est 9.488, à 90% c'est 7.779.
  LDLT<MatrixXd> ldlt(COV);
  VectorXd Alpha=ldlt.solve(X-O);
  return (X-O).dot(Alpha);
}

VectorXd sample_gauss(const VectorXd O, const MatrixXd COV){
  //tire un échantillon d'une gaussienne multidimensionnelle
  int nd=O.size();
  VectorXd N(nd);
  MatrixXd SqrtCOV=COV.llt().matrixL();
  for(int i=0;i<nd;i++){
    N(i)=distN(generator);
  }
  return O + SqrtCOV*N; 
};


const double Big = -1.e16;


int main(int argc, char **argv){
  

  if(argc != 3){
    cout << "Usage :\n";
    cout << "\t Number of obs, seed_obs\n";
    exit(0);
  }
  int nd  = atoi(argv[1]);
  uint32_t seed_obs=atoi(argv[2]);//12 marche bien
  // Construction du grid initial
  VectorXd lb_t(dim_theta);
  VectorXd ub_t(dim_theta);
  lb_t(0)=-0.5;
  ub_t(0)=1.5;

  VectorXd lb(2);
  VectorXd ub(2);
  lb(0)=-0.5;
  ub(0)=1.5;
  lb(1)=-0.5;
  ub(1)=1.5;
  DoE test(lb,ub,20);
   for (int i=0;i<400;i++){
   cout << test.GetGrid()[i].transpose() << " " << test.GetWeights()[i] << endl;
  }
  exit(0);



  

  std::vector<double> lb_hpars(3); lb_hpars[0]=-5;lb_hpars[1]=-6;lb_hpars[2]=-3; //-5 avant
  std::vector<double> ub_hpars(3); ub_hpars[0]=0;ub_hpars[1]=-1;ub_hpars[2]=log(2); //bornes sur l'optimisation des hpars. edm exp lcor.
  for (int i=0;i<3;i++){lb_hpars[i]=exp(lb_hpars[i]);ub_hpars[i]=exp(ub_hpars[i]);}// on fait l'opti sur les paramètres directement.
  Grid=InitGrid(lb_t,ub_t,6);
  int ngrid=Grid.size();
  int ndim = 1;
  double xp_noise=0.01;
  int maxeval=200000;
  const char * c_obs=to_string(nd).c_str();
  const char * c_seedobs=to_string(seed_obs).c_str();
  string shortprefixe=string("results/")+c_obs+string("_")+c_seedobs;
  cout << shortprefixe << endl;
  FILE* out = fopen((string(shortprefixe)+string("_observations.gnu")).c_str(),"w");
  /*Generate the observations */
  generator.seed(seed_obs);
  fprintf(out,"#Fichier des observations. ndim premieres colonnes : coordonnes, derniere colonne : observation (bruitee)\n");
  for(unsigned d=0; d<nd; d++){
    VectorXd x(1);
    x(0) = (double) d/(double)nd;
    fprintf(out,"%e ",x(0));
    double f = my_function(x) + distN(generator)*xp_noise;
    DATA dat; dat.SetX(x); dat.SetValue(f);
    fprintf(out,"%e \n",dat.Value());
    data.push_back(dat);
  }
  fclose(out);

  double ftol=1e-4;
  cout << "Calcul hpars KOH" << endl;

  // Calcul des hyperparamètres KOH.

  VectorXd hpars_max_koh(3);
  double time_exec_koh;
  {
    clock_t c_start = std::clock();
    //1ère étape : optimisation globale par algo génétique
    {
      int pop=2000; // population size
      cout << "Début de l'optimisation globale... Pop size : " << pop << endl;
      /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
      std::vector<double> x(3);
      for (int j=0;j<3;j++){x[j]=exp(log(lb_hpars[j])+(log(ub_hpars[j])-log(lb_hpars[j]))*distU(generator));} // initial guess
      nlopt::opt opt(nlopt::GN_ISRES, 3);    /* algorithm and dimensionality */
      opt.set_max_objective(optfunc__koh, &data);
      opt.set_lower_bounds(lb_hpars);
      opt.set_upper_bounds(ub_hpars);
      opt.set_maxeval(maxeval);
      opt.set_population(pop);
      opt.set_ftol_rel(ftol);
      double msup; /* the maximum objective value, upon return */
      int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 
      for(int i=0;i<3;i++){hpars_max_koh(i)=x[i];}
      cout << "Message d'arrêt : " << fin <<". max trouvé : " << hpars_max_koh.transpose() << ", valeur du critère : " << msup <<endl;
    }
    //2ème étape : optimisation locale par sbplx en partant du point trouvé précédemment.
    {
      cout << "Début de l'optimisation locale..." << endl;
      double msuploc;
      /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
      std::vector<double> x(3);
      for (int j=0;j<3;j++){x[j]=hpars_max_koh(j);}
      nlopt::opt opt(nlopt::LN_SBPLX, 3);    /* algorithm and dimensionality */
      opt.set_max_objective(optfunc__koh, &data);
      opt.set_lower_bounds(lb_hpars);
      opt.set_upper_bounds(ub_hpars);
      opt.set_maxeval(maxeval);
      opt.set_ftol_rel(ftol);
      double msup; /* the maximum objective value, upon return */
      int fin=opt.optimize(x, msup);
      for(int i=0;i<3;i++){hpars_max_koh(i)=x[i];}
      msuploc=msup;
      cout << "Message d'arrêt : " << fin  << ". Max : " << hpars_max_koh.transpose() << ", valeur du critère : " << msuploc << endl;
      int niter=100; // On recommence l'opti tant qu'on n'a pas stagné 100 fois.
      int nopt=0;
      while (nopt<niter){
	nopt++;
	for (int j=0;j<3;j++){x[j]=exp(log(lb_hpars[j])+(log(ub_hpars[j])-log(lb_hpars[j]))*distU(generator));}
	int fin= opt.optimize(x, msup);
	if(msup>msuploc){
	  nopt=0;
	  msuploc=msup;
	  for(int k=0;k<3;k++){
	    hpars_max_koh(k)=x[k];
	  }
	  cout << "Message d'arrêt : " << fin  << ". Nouveau max : " << hpars_max_koh.transpose() << ", valeur du critère : " << msuploc << endl;
	}
      }
      double max_vrais_koh=msuploc;
      cout << "hyperparametres KOH : (edm, exp, lcor) : " << hpars_max_koh.transpose() << " a la vraisemblance : " << max_vrais_koh << endl << endl;
    }
    clock_t c_end = std::clock();
    double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    time_exec_koh=time_elapsed_ms / 1000.0;
    std::cout << "Temps pour opti KOH : " << time_exec_koh << " s\n";
  }

  //Comparaison des méthodes en construisant la post sur un grid (assez fin) de thetas. Ou alors KOH direct ? Non. je commence par tout faire puis une fois que tout marche on fera des cas individuels.
  vector<VectorXd> FGrid=InitGrid(lb_t,ub_t,gridstep);
  int fsize=FGrid.size();
  //Calcul des hpars opti sur le finegrid
  cout << "Début optimisation Opti..." << endl;
  std::vector<VectorXd> opti_hpars; // les hyperparamètres optimaux sont là-dedans
  std::vector<DATA> lp_opti; // la valeur de la logpost estimée est là
  double time_exec_opti=0;
  bool opti_exists;
  {
    std::ifstream fs((string(shortprefixe)+string("_lp_opti.gnu")).c_str());
    opti_exists=fs.is_open();
  }
  cout << "opti exists : " << opti_exists << endl;

  if(opti_exists){
    lp_opti=vectorlog(ReadPost((string(shortprefixe)+string("_lp_opti.gnu")).c_str()));
  }
  else
    {
      clock_t c_start = std::clock();
      for (int z=0;z<fsize;z++){
	//Creation des data
	std::vector<DATA> data2;
	for(int ie=0; ie<nd; ie++){
	  DATA dat; dat.SetX(data[ie].GetX()); dat.SetValue(data[ie].Value()-my_model(data[ie].GetX(),FGrid[z]));
	  data2.push_back(dat); // on construit les données y-f_t
	}

	VectorXd hpars_max_opti(3);
	//Etape 1 : opti globale
	{
	  int pop=2000; // population size
	  //	cout << "Début de l'optimisation globale... Pop size : " << pop << endl;
	  std::vector<double> x(3);
	  for (int j=0;j<3;j++){x[j]=exp(log(lb_hpars[j])+(log(ub_hpars[j])-log(lb_hpars[j]))*distU(generator));} // initial guess
	  nlopt::opt opt(nlopt::GN_ISRES, 3);    /* algorithm and dimensionality */
	  opt.set_max_objective(optfunc__nograd, &data2);
	  opt.set_lower_bounds(lb_hpars);
	  opt.set_upper_bounds(ub_hpars);
	  opt.set_maxeval(maxeval);
	  opt.set_population(pop);
	  opt.set_ftol_rel(ftol);		
	  double msup; /* the maximum objective value, upon return */
	  int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 
	  for(int i=0;i<3;i++){hpars_max_opti(i)=x[i];}
	  //	cout << "Message d'arrêt : " << fin <<". max trouvé : " << hpars_max_opti.transpose() << ", valeur du critère : " << msup <<endl;
	}
	//Etape 2 : opti locale
	{
	  //	cout << "Début de l'optimisation locale..." << endl;
	  double msuploc;
	  /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
	  std::vector<double> x(3);
	  for (int j=0;j<3;j++){x[j]=hpars_max_opti(j);}
	  nlopt::opt opt(nlopt::LN_SBPLX, 3);    /* algorithm and dimensionality */
	  opt.set_max_objective(optfunc__nograd, &data2);
	  opt.set_lower_bounds(lb_hpars);
	  opt.set_upper_bounds(ub_hpars);
	  opt.set_maxeval(maxeval);
	  opt.set_ftol_rel(ftol);
	  double msup; /* the maximum objective value, upon return */
	  int fin=opt.optimize(x, msup);
	  for(int i=0;i<3;i++){hpars_max_opti(i)=x[i];}
	  msuploc=msup;
	  //	cout << "Message d'arrêt : " << fin  << ". Max : " << hpars_max_opti.transpose() << ", valeur du critère : " << msuploc << endl;
	  int niter=100; // On recommence l'opti tant qu'on n'a pas stagné 100 fois.
	  int nopt=0;
	  while (nopt<niter){
	    nopt++;
	    for (int j=0;j<3;j++){x[j]=exp(log(lb_hpars[j])+(log(ub_hpars[j])-log(lb_hpars[j]))*distU(generator));}
	    int fin= opt.optimize(x, msup);
	    if(msup>msuploc){
	      nopt=0;
	      msuploc=msup;
	      for(int k=0;k<3;k++){
		hpars_max_opti(k)=x[k];
	      }
	      //  cout << "Message d'arrêt : " << fin  << ". Nouveau max : " << hpars_max_opti.transpose() << ", valeur du critère : " << msuploc << endl;
	    }
	  }
	  double max_vrais_opti=msuploc;
	  //	cout << "hyperparametres KOH : (edm, exp, lcor) : " << hpars_max_koh.transpose() << " a la vraisemblance : " << max_vrais_koh << endl << endl;

	  opti_hpars.push_back(hpars_max_opti);
	  DATA dat; dat.SetX(FGrid[z]); dat.SetValue(msuploc);
	  lp_opti.push_back(dat);
	}
	data2.clear();
      }
      clock_t c_end = std::clock();
      time_exec_opti = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
      std::cout << "Temps pour opti Opti : " << time_exec_opti / 1000.0 << " s\n";
    }
  //Calcul de la postérieure bayésienne complète sur le FGrid.
  
  cout << "Calcul post. bayésienne..." << endl;
  vector<DATA> p_bayes;
  double time_exec_bayes=0;

  bool bayes_exists;
  {
    std::ifstream fs((string(shortprefixe)+string("_p_bayes.gnu")).c_str());
    bayes_exists=fs.is_open();
  }
  cout << "bayes exxists : " << bayes_exists << endl;
  if(bayes_exists){
    p_bayes=ReadPost((string(shortprefixe)+string("_p_bayes.gnu")).c_str());
  }
  else
    {
      int nsim_bayes=2000000; // nombre de tirages par valeur de paramètres
      int seed_bayes=66;
      generator.seed(seed_bayes);
      VectorXd probs=VectorXd::Zero(fsize);
      clock_t c_start = std::clock();
      //Tirage des hyperparamètres et évaluation des décompositions de Cholesky.
      generator.seed(seed_bayes);
      for (int i=0;i<nsim_bayes;i++){
	VectorXd hpars=draw_hpars(lb_hpars,ub_hpars);
	MatrixXd G=Gamma(&data,hpars);
	LDLT<MatrixXd> ldlt(G);
	for (int j=0;j<fsize;j++){
	  probs(j)+=exp(loglikelihood_theta_fast(&data,hpars,ldlt,FGrid[j]));
	}
      }
      for (int j=0;j<fsize;j++){
	probs(j)=probs(j)/(double) nsim_bayes;
	DATA dat; dat.SetX(FGrid[j]); dat.SetValue(probs(j));
	p_bayes.push_back(dat);      
      }
      clock_t c_end = std::clock();
      double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
      std::cout << "Temps pour calcul Bayes : " << time_elapsed_ms / 1000.0 << " s\n";
      time_exec_bayes=time_elapsed_ms / 1000.0;
    }

  // Ecriture des hpars dans un fichier
  if(!opti_exists)
    {
      out=fopen((string(shortprefixe)+string("_hpars_opti.gnu")).c_str(),"w");
      fprintf(out,"#Hpars obtenus par optimisation. col1:t1 col2:t2 col3:edm, col4:exp, col5:lcor\n");
      for (int i=0;i<fsize;i++){
	fprintf(out,"%e %e %e %e\n",FGrid[i](0),opti_hpars[i](0),opti_hpars[i](1),opti_hpars[i](2));
      }
      fclose(out);
    }
  p_bayes=vectorweight1(p_bayes);
  WritePost(p_bayes,(string(shortprefixe)+string("_p_bayes.gnu")).c_str());
  
  //Normalisation et écriture
  vector<DATA> lp_koh=BuildPost(data,FGrid,hpars_max_koh);
  double evkoh=Evidence(lp_koh);
  double esplogkoh=esplog(p_bayes,lp_koh);
  double esplogopti=esplog(p_bayes,lp_opti);
  double logintkoh=logint(lp_koh);
  double logintopti=logint(lp_opti);

  lp_opti=vectorweight1(vectorexp(lp_opti));
  WritePost(lp_opti,(string(shortprefixe)+string("_lp_opti.gnu")).c_str());

  lp_koh=vectorweight1(vectorexp(lp_koh));
  WritePost(lp_koh,(string(shortprefixe)+string("_lp_koh.gnu")).c_str());

  double klbo=KLDiv(p_bayes,lp_opti);
  double klbk=KLDiv(p_bayes,lp_koh);
  cout << "KLDiv de bayes à opti : " << klbo << endl;
  cout << "KLDiv de bayes à koh : " << klbk << endl << endl;
  cout << "esplogkoh : " << esplogkoh << endl;
  cout << "esplogopti : " << esplogopti << endl;
  cout << "logintkoh : " << logintkoh << endl;
  cout << "logintopti : " << logintopti << endl;
  cout << "checksum : " << klbk-klbo << " = " << esplogopti-esplogkoh+logintkoh-logintopti << endl;
  cout << "Bayes factor : " << exp(logintopti-logintkoh) << endl;
  
  double entbayes=Entropy(p_bayes);
  double entkoh=Entropy(lp_koh);
  double entopti=Entropy(lp_opti);
  
  VectorXd mapopti=MAP(lp_opti);
  VectorXd mapkoh=MAP(lp_koh);
  VectorXd mapbayes=MAP(p_bayes);
  
  VectorXd meanopti=Mean(lp_opti);
  VectorXd meankoh=Mean(lp_koh);
  VectorXd meanbayes=Mean(p_bayes);
  //Ecriture d'un fichier summary. On y met : caractéristiques du calcul, temps d'exécution, hpars koh, kldivs, entropies.
  
  {
    FILE* out=fopen("results/summary.gnu","a");
    fprintf(out,"%d %d %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e %e\n",nd,seed_obs,time_exec_koh,time_exec_opti,time_exec_bayes,hpars_max_koh(0),hpars_max_koh(1),hpars_max_koh(2),klbo,klbk,entopti,entkoh,entbayes,evkoh,mapopti(0),mapkoh(0),mapbayes(0),meanopti(0),meankoh(0),meanbayes(0));
    fclose(out);
  }
  //Calcul de la KLDiv entre prior et posterior, puis l'inverse.
  vector<DATA> unif;
  for (int i=0;i<FGrid.size();i++){
    DATA dat; dat.SetX(FGrid[i]); dat.SetValue(1);
    unif.push_back(dat);
  }
  unif=vectorweight1(unif);
  cout << "KL de prior à post" << KLDiv(unif,p_bayes) << endl;
  cout << "KL de post à prior" << KLDiv(p_bayes,unif) << endl;

  // Début d'une MCMC en 4 dimensions.
  // avec un VecteurXd dans l'ordre suivant : theta/edm/exp/lcor
  int nchain=2000000;
  VectorXd t_init(4);
  t_init(0)=0;
  t_init(1)=hpars_max_koh(0);
  t_init(2)=hpars_max_koh(1);
  t_init(3)=hpars_max_koh(2);
  
  MatrixXd COV=MatrixXd::Zero(4,4);
  COV(0,0)=pow(0.2,2);
  COV(1,1)=pow(0.05,2);
  COV(2,2)=pow(0.001,2);
  COV(3,3)=pow(0.1,2);
  MatrixXd sqrtCOV=COV.llt().matrixL();

  VectorXd lb_mcmc(4);
  VectorXd ub_mcmc(4);
  lb_mcmc(0)=lb_t(0); ub_mcmc(0)=ub_t(0);
  lb_mcmc(1)=lb_hpars[0]; ub_mcmc[1]=ub_hpars[0];
  lb_mcmc(2)=lb_hpars[1]; ub_mcmc[2]=ub_hpars[1];
  lb_mcmc(3)=lb_hpars[2]; ub_mcmc[3]=ub_hpars[2];
  
  vector<VectorXd> theta_samples;
  vector<double> f_samples;
  
  double f_init=f_mcmc(&data,t_init);
  VectorXd t_current=t_init;
  double f_current=f_init;
  VectorXd t_candidate(4);
  double f_candidate;
  int accept_count=0;

  for (int i=0;i<nchain;i++){
    VectorXd Step(4);
    for (int j=0;j<4;j++){Step[j]=distN(generator);}
    t_candidate=t_current+sqrtCOV*Step;
    f_candidate=f_mcmc(&data,t_candidate);
    if(f_candidate>f_current && in_bounds(t_candidate,lb_mcmc,ub_mcmc)){
      accept_count+=1;
      t_current=t_candidate;
      f_current=f_candidate;
    }
    else if(f_candidate-f_current>log(distU(generator))  && in_bounds(t_candidate,lb_mcmc,ub_mcmc)) {
      accept_count+=1;
      t_current=t_candidate;
      f_current=f_candidate;
    } 
    theta_samples.push_back(t_current);
    f_samples.push_back(f_current);
  }
  cout << "acc rate : " << 100*double(accept_count)/double(nchain)<< endl;
  //Ecriture des échantillons obtenus
  {
    FILE* out=fopen((string(shortprefixe)+string("_samples_mcmc.gnu")).c_str(),"w");
    for (int i=0;i<theta_samples.size();i++){
      fprintf(out,"%e %e %e %e %e\n",theta_samples[i](0),theta_samples[i](1),theta_samples[i](2),theta_samples[i](3),f_samples[i]);
    }
    fclose(out);
  }
  
  //Sélection de quelques échantillons indépendants.
  int nindepsamp=3000;
  vector<VectorXd> thetaselect;
  vector<double> fselect;

  for (int i=0;i<theta_samples.size();i++){
    if (i>nindepsamp && i%(theta_samples.size()/nindepsamp)==0){
      thetaselect.push_back(theta_samples[i]);
      fselect.push_back(f_samples[i]);
    }
  }

  cout << "nombre d'échantillons indep : " << thetaselect.size() << endl;

  {
    FILE* out=fopen((string(shortprefixe)+string("_samples_mcmc_select.gnu")).c_str(),"w");
    for (int i=0;i<thetaselect.size();i++){
      fprintf(out,"%e %e %e %e %e\n",thetaselect[i](0),thetaselect[i](1),thetaselect[i](2),thetaselect[i](3),fselect[i]);
    }
    fclose(out);
  }

  //Tri des échantillons en deux catégories sur theta avec séparation sur 0.5
  vector<VectorXd> thetaselectleft;
  vector<double> fselectleft;
  vector<VectorXd> thetaselectright;
  vector<double> fselectright;

  for (int i=0;i<thetaselect.size();i++){
    if (thetaselect[i](0)<0.5){
      thetaselectleft.push_back(thetaselect[i]);
      fselectleft.push_back(fselect[i]);
    }
    else{
      thetaselectright.push_back(thetaselect[i]);
      fselectright.push_back(fselect[i]);
    }
  }
  cout << "nombre d'échantillons à gauche : " << thetaselectleft.size() << " et à droite : " << thetaselectright.size() << endl;
  

  VectorXd t_moyleft=moy_mcmc(thetaselectleft,fselectleft);
  VectorXd t_moyright=moy_mcmc(thetaselectright,fselectright);
  cout << " moyenne à gauche : " << t_moyleft.transpose() << endl;
  cout << " moyenne à droite : " << t_moyright.transpose() << endl;


  //Find le max à gauche et le max à droite en utilisant tous les échantillons parcourus.
  VectorXd t_maxleft(4);
  VectorXd t_maxright(4);
  double f_maxleft=0;
  double f_maxright=0;
  for (int i=0;i<theta_samples.size();i++){
    if(f_samples[i]>f_maxleft && theta_samples[i](0)<0.5){
      f_maxleft=f_samples[i];
      t_maxleft=theta_samples[i];
    }
    else if (f_samples[i]>f_maxright && theta_samples[i](0)>0.5){
      f_maxright=f_samples[i];
      t_maxright=theta_samples[i];
    }
  }

  cout << " max à gauche : " << t_maxleft.transpose() << "p : " << f_maxleft << endl;
  cout << " max à droite : " << t_maxright.transpose() << "p : " << f_maxright << endl;
  MatrixXd COVLEFT=cov_mcmc(thetaselectleft,fselectleft);
  MatrixXd COVRIGHT=cov_mcmc(thetaselectright,fselectright);
  
  cout << "matrice cov à gauche : \n" << COVLEFT << endl;
  cout << "matrice cov à droite : \n" << COVRIGHT << endl;

  //estimation de pi1 et pi2. On va utiliser les max pour fitter nos gaussiennes.
  double detleft=pow(COVLEFT.llt().matrixL().determinant(),2);
  double detright=pow(COVRIGHT.llt().matrixL().determinant(),2);
  double ratio=sqrt(detleft/detright);

  double fmaxleft=exp(f_mcmc(&data,t_maxleft));
  double fmaxright=exp(f_mcmc(&data,t_maxright));
  double pi1=(ratio*fmaxleft/fmaxright)/(1+ratio*fmaxleft/fmaxright);
  cout << "pi1 : " << pi1 << endl;

  //estimation de la densité bayes pour une valeur de theta (en VectorXd de dimension 1)
  vector<DATA> pbayes_estimate;
  {
    for (int i=0;i<fsize;i++){
      VectorXd thcourant=FGrid[i];
      DATA dat; dat.SetX(thcourant); dat.SetValue((1./sqrt(2*3.1415))*(pi1/sqrt(COVLEFT(0,0))*exp(-0.5*pow(thcourant(0)-t_maxleft(0),2)/COVLEFT(0,0))+(1-pi1)/sqrt(COVRIGHT(0,0))*exp(-0.5*pow(thcourant(0)-t_maxright(0),2)/COVRIGHT(0,0))));
      pbayes_estimate.push_back(dat);      
    }
  }
  //calcul de la covariance réduite de KOH , en supposant qu'il a choisi le maximum left :
  MatrixXd COVLEFT_hpars=COVLEFT.block(1,1,3,3);
  VectorXd COVLEFT_cross=COVLEFT.col(0).tail(3); //les 3 derniers éléments de la première colonne
  MatrixXd COVRIGHT_hpars=COVRIGHT.block(1,1,3,3);
  VectorXd COVRIGHT_cross=COVRIGHT.col(0).tail(3); //les 3 derniers éléments de la première colonne
  double VarKoh=COVLEFT(0,0)-COVLEFT_cross.transpose()*COVLEFT_hpars.llt().solve(MatrixXd::Identity(3,3))*COVLEFT_cross;
  cout << "var koh : " << VarKoh << endl;

  //calcul du critère de séparation de la marginale des hyperparamètres :
  

  //estimation de la densité KOH :
  vector<DATA> pkoh_estimate;
  {
    for (int i=0;i<fsize;i++){
      VectorXd thcourant=FGrid[i];
      DATA dat; dat.SetX(thcourant); dat.SetValue((1./sqrt(2*3.1415))*(pi1/sqrt(VarKoh)*exp(-0.5*pow(thcourant(0)-t_maxleft(0),2)/VarKoh)));
      pkoh_estimate.push_back(dat);      
    }
  }
  //estimation de la densité Opti :
  vector<DATA> popti_estimate;
  {
    for (int i=0;i<fsize;i++){
      VectorXd thcourant=FGrid[i];
      DATA dat; dat.SetX(thcourant); dat.SetValue((1./sqrt(2*3.1415))*(pi1/sqrt(detleft)*exp(-0.5*pow(thcourant(0)-t_maxleft(0),2)/COVLEFT(0,0))+(1-pi1)/sqrt(detright)*exp(-0.5*pow(thcourant(0)-t_maxright(0),2)/COVRIGHT(0,0))));
      popti_estimate.push_back(dat);      
    }
  }


  popti_estimate=vectorweight1(popti_estimate);
  WritePost(popti_estimate,(string(shortprefixe)+string("_p_opti_estimate.gnu")).c_str());
  pkoh_estimate=vectorweight1(pkoh_estimate);
  WritePost(pkoh_estimate,(string(shortprefixe)+string("_p_koh_estimate.gnu")).c_str());
  pbayes_estimate=vectorweight1(pbayes_estimate);
  WritePost(pbayes_estimate,(string(shortprefixe)+string("_p_bayes_estimate.gnu")).c_str());

  //Calcul du critère Bayes pour choisir entre les maxima :
  double critbayesleft=pi1/sqrt(COVLEFT(0,0));
  double critbayesright=(1-pi1)/sqrt(COVRIGHT(0,0));
  
  //Calcul du critère KOH :
  double critkohleft=pi1/(COVLEFT_hpars-COVLEFT_cross*COVLEFT_cross.transpose()/COVLEFT(0,0)).llt().matrixL().determinant();//c'est bien la racine du déterminant.
  double critkohright=(1-pi1)/(COVRIGHT_hpars-COVRIGHT_cross*COVRIGHT_cross.transpose()/COVRIGHT(0,0)).llt().matrixL().determinant();//c'est bien la racine du déterminant.

  cout << "crit bayes left : " << critbayesleft << ", right : " << critbayesright << endl;
  cout << "crit koh left : " << critkohleft << ", right : " << critkohright << endl;

  //test pour voir si les ellipses sont confondues.
  {
    cout << "recherche de points en commun..." << endl;
    int ntest=10000; //on teste 10000 points
    int in95=0;
    int in90=0;
    int distmax=1000;
    VectorXd Xmax=VectorXd::Zero(4);
    for (int i=0;i<ntest;i++){
      VectorXd Xtest=sample_gauss(t_maxleft,COVLEFT);
      int distleft=ellipse_dist(Xtest,t_maxleft,COVLEFT);
      int distright=ellipse_dist(Xtest,t_maxright,COVRIGHT);
      if(std::max(distleft,distright)<distmax){
	distmax=std::max(distleft,distright);
	Xmax=Xtest;
      }
      if(distright<7.779 && distleft<7.779){
	cout << "point en commun à 90% : " << Xtest.transpose();
      }
      if(distright<9.488 && distleft<9.488){
	cout << "point en commun à 95% : " << Xtest.transpose();
      }
    }
    cout << "point le plus proche : " << Xmax.transpose() << " à la distance " << distmax << endl;
  }

  //critère de séparation de la marginale des hyperparametres. On prend le maximum de gauche comme référence (car il est plus grand)
  {
    MatrixXd diag=COVLEFT.ldlt().vectorD();
    MatrixXd O=COVLEFT.ldlt().matrixL().transpose();
    cout << "O : " << O << endl;
    MatrixXd Betainv=MatrixXd::Zero(4,4);
    Betainv(0,0)=sqrt(diag(0));
    Betainv(1,1)=sqrt(diag(1));
    Betainv(2,2)=sqrt(diag(2));
    Betainv(3,3)=sqrt(diag(3));
    MatrixXd Beta=MatrixXd::Zero(4,4);
    Beta(0,0)=1./sqrt(diag(0));
    Beta(1,1)=1./sqrt(diag(1));
    Beta(2,2)=1./sqrt(diag(2));
    Beta(3,3)=1./sqrt(diag(3));
    MatrixXd newCOV=Beta*O*COVRIGHT*O.transpose()*Beta;
    VectorXd newcenter=Beta*O*(t_maxright-t_maxleft);
    cout << "newCOV : " << newCOV << endl;
    cout << "newcenter : " << newcenter << endl;

    //test de recherche de points.
    int ntest=10000; //on teste 10000 points
    double dmax=1000000;
    VectorXd Xmax(4);
    for (int i=0;i<ntest;i++){
      VectorXd Xtest=sample_gauss(newcenter,newCOV);
      double dist=Xtest.squaredNorm();
      if(dist<dmax && ellipse_dist(Xtest,newcenter,newCOV)<9.488){ //ellipse à 95%
	cout << ". " << endl;
	dmax=dist;
	Xmax=Xtest;
      }
    }
  
    cout << "Point le plus proche : " << Xmax.transpose() << " à la distance " << dmax-9.488 << endl;
    cout << "Point le plus proche : " << (t_maxleft+O.transpose()*Betainv*Xmax).transpose() << endl;
  }

  {//Le plus simple est plutôt de mesurer la distance en theta. on va dire à 95 pct (2 sigma) :
    cout << "distance en theta : " << (t_maxright(0)-2*sqrt(COVRIGHT(0,0)))-(t_maxleft(0)+2*sqrt(COVLEFT(0,0))) << endl;
  }

  //On optimise avec BayesOpt pour trouver le maximum (pour le moment)

  
  


  
  exit(0);
};
