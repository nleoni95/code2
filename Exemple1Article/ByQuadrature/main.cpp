// Dans ce fichier : on met en oeuvre différents algorithmes d'optimisation pour KOH. Voir lequel est le meilleur.
// On reproduit l'exemple 1 de l'article avec notre nouveau fichier pour être solide.


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

int neval=1;
//std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);
vector<DATA> data;
vector<VectorXd> Grid;


int gridstep=140; //Finesse du grid en theta
int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
int dim_theta=1;




double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=x(0)-y(0);
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(2),2)); //squaredexp
}

double PriorMean(VectorXd const &x, VectorXd const &hpars){
  return 0;
}

double logprior_hpars(VectorXd const &hpars){
  if(hpars(2)<0){return -999;}
  double alpha_ig=5.5;
  double beta_ig=0.3;
  return log(pow(beta_ig,alpha_ig)*pow(hpars(2),-alpha_ig-1)*exp(-beta_ig/hpars(2))/tgamma(alpha_ig));//-log(hpars(0));
}



double logprior_pars(VectorXd const &pars){
  //prior uniforme sur les pramètres
  return 0;
}

void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
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
  //je sais que le calcul n'est pas bon car on pondère par les probabilités au lieu de moyenner. C'est 
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

double reconstructed_prior(double x, double moy, double var){
  //évaluation d'une gaussienne de moyenne moy et de variance var
  return (1./(sqrt(2*M_PI*var))*exp(-0.5*pow(x-moy,2)/var));
}

double post_bayes_rec(double x, double p1, double moyleft, double varleft, double moyright, double varright){
  //évaluation de la post bayes
  double left=p1*reconstructed_prior(x,moyleft,varleft);
  double right=(1-p1)*reconstructed_prior(x,moyright,varright);
  return left+right;
}

double post_fmp_rec(double x, double p1, double moyleft, double varleft, double moyright, double varright){
    //évaluation de la post fmp
  double left=p1*reconstructed_prior(x,moyleft,varleft);
  double right=(1-p1)*reconstructed_prior(x,moyright,varright);
  return left+right;
}

double post_koh_rec(double x, double moy, double var){
    //évaluation de la post fmp
  return reconstructed_prior(x,moy,var);
}

const double Big = -1.e16;

class Comparator {
  //nécessaire pour créer une map avec des clefs VectorXi.
public:
  bool operator()(const VectorXi &v1, const VectorXi & v2)const{
    return v1.array().sum()<v2.array().sum();
  }
};

auto comp=[](const VectorXi &v1, const VectorXi & v2){
  //opérateur de comparaison vector Xi. On compare les premiers coefs, puis les deuxièmes, etc jusqu'au dernier.
  if(!v1.size()==v2.size()){cerr << "erreur comparaison vectorXi!!" <<endl;}
  for(int i=0;i<v1.size();i++){
    if(v1(i)<v2(i)){
      return true;
    }
    if(v1(i)>v2(i)){
      return false;
    }
  }
  return false;
  };

typedef map<VectorXi,VectorXd,decltype(comp)> map_VXd; // pour faire un grid.
typedef map<VectorXi,double,decltype(comp)> map_double; // pour stocker une densité.


void fill_grid(map_VXd & map,VectorXi const & refinement, VectorXd const & lb, VectorXd const & ub){
//fill a map with bounds and refinement in each direction
  int d=refinement.size();
  VectorXi ind=VectorXi::Zero(d);
  //parcourir tous les vecteurs d'entiers.
  //mettons déjà le point plein de zéros
  VectorXd pointzero(d); for(int i=0;i<d;i++){pointzero(i)=lb(i);}
  map.insert(make_pair(ind,pointzero));
  while(ind!=refinement){
    //changement de l'indice
    for(int i=1;i<=d;i++){
      if(ind(d-i)==refinement(d-i)){
        ind(d-i)=0; 
      }
      else{
        ind(d-i)++;
        break;
      }
    }
    //construction du point du grid
    VectorXd point(d); for(int i=0;i<d;i++){point(i)=lb(i)+(ub(i)-lb(i))*ind(i)/refinement(i);}
    map.insert(make_pair(ind,point));
  }
}

void WriteObs(string filename,vector<AUGDATA> &augdata){
  ofstream ofile(filename);
  AUGDATA a=augdata[0];
  VectorXd x=a.GetX();
  VectorXd y=a.Value();
  for(int i=0;i<x.size();i++){
    ofile << x(i) << " " << y(i) << endl;
  }
  ofile.close();
}

void WriteDensity(map_VXd const & map_grid, map_double const & map_density,string filename){
  //écriture d'une map dans un fichier. normalement les ensemble de keys de chacune des maps doivent être identiques.
  ofstream ofile(filename);
  for_each(map_grid.begin(),map_grid.end(),[&ofile,&map_density](pair<VectorXi,VectorXd> const &p){
    for(int i=0;i<p.second.size();i++){
      ofile << p.second(i) << " " ;
    }
    VectorXi Xi=p.first;
    ofile << map_density.at(Xi) << endl;
  });
}

void Normalisation_map_double(map_double &map, double interval_length){
  //on met l'intégrale d'une map 1D à 1, en précisant la longueur de l'intervalle.
  //calcul de l'intégrale et normalisation
  double integral=0;
  for_each(map.begin(),map.end(),[&integral](pair<VectorXi,double> p){
    integral+=p.second;
  });
  integral*=interval_length/map.size();
  cout << "integral: "<< integral << endl;
  for_each(map.begin(),map.end(),[integral,&map](pair<VectorXi,double> p){
    double l=p.second/integral;
    map[p.first]=l;
  });
}

void Take_exp_map_double(map_double &map){
  //on prend l'exponentielle des éléments de la map.
  //calcul de l'intégrale et normalisation
  double max_logkoh=max_element(map.begin(),map.end(),[](const pair<VectorXi,double>& p1,const pair<VectorXi,double>& p2){
    return p1.second<p2.second;
  })->second;
  for_each(map.begin(),map.end(),[max_logkoh,&map](pair<VectorXi,double> p){
    double l=p.second;
    l=exp(l-max_logkoh);
    map[p.first]=l;
  });
}




int main(int argc, char **argv){
  


  int nobs  = 15;
  uint32_t seed_obs=69;//lol
  auto begin=chrono::steady_clock::now();
  auto end=chrono::steady_clock::now();
  // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(1); lb_t(0)=-0.5;
  VectorXd ub_t(1); ub_t(0)=1.5; 

  VectorXd lb_hpars(3); lb_hpars(0)=0;lb_hpars(1)=4e-3;lb_hpars(2)=1e-2; //-5 avant
  VectorXd ub_hpars(3); ub_hpars(0)=0.7;ub_hpars(1)=3e-2;ub_hpars(2)=1.5; //bornes sur l'optimisation des hpars. edm exp lcor

  VectorXd lb_hpars_noedm(3); lb_hpars_noedm(0)=0;lb_hpars_noedm(1)=1e-3;lb_hpars_noedm(2)=0.1; //-5 avant
  VectorXd ub_hpars_noedm(3); ub_hpars_noedm(0)=1e-8;ub_hpars_noedm(1)=1;ub_hpars_noedm(2)=0.2; //bornes sur l'optimisation des hpars. edm exp lcor
  
  VectorXd lb_total(4),ub_total(4);
  lb_total << lb_t(0),lb_hpars(0),lb_hpars(1),lb_hpars(2);
  ub_total << ub_t(0),ub_hpars(0),ub_hpars(1),ub_hpars(2);

    //création du grid pour les theta, et pour theta+hpars
  map_VXd grid_theta(comp);
  map_VXd grid_total(comp);
  VectorXi refinement_total(4); refinement_total << 100,70,40,70;
  VectorXi refinement_theta(1); refinement_theta << refinement_total(0);
  fill_grid(grid_total,refinement_total,lb_total,ub_total);
  fill_grid(grid_theta,refinement_theta,lb_t,ub_t);
  cout << grid_total.size() << endl;
  
  //maintenant il faut des bonnes fonctions pour évaluer la vraisemblance et les priors. c'est vraiment tout ce dont on a besoin. je crois que je vais les faire individuellement...
  //KOH nécessite une quadrature. Mais pas Opt.

  std::default_random_engine generator(seed_obs);
  auto lambda_truefct=[&generator](VectorXd const & Xprofile){
      std::normal_distribution<double> distN(0,1);
      double noise=1e-2;
      VectorXd y(Xprofile.size());
      for(int i=0;i<y.size();i++){
        y(i)=Xprofile(i)+noise*distN(generator);
      }
      return y;
    };

    auto lambda_model=[](VectorXd const & Xprofile, VectorXd const & theta){
      //le vecteur Xprofile contient tous les x scalaires. Il faut renvoyer une prédiction de même taille que Xprofile.
      VectorXd m(Xprofile.size());
      for(int i=0;i<m.size();i++){
        double x=Xprofile(i);
        m(i)=x*sin(2*x*theta(0))+(x+0.15)*(1-theta(0));
      }
      return m;
    };

    auto logprior_pars=[](VectorXd const &p){
      return 0;
    };

    auto logprior_hpars=[](VectorXd const &h){
      //logprior NUL
      double alpha_ig=5.5;
      double beta_ig=0.3;
      double v=log(pow(beta_ig,alpha_ig)*pow(h(2),-alpha_ig-1)*exp(-beta_ig/h(2))/tgamma(alpha_ig));
      return 0; 
      //return v;
    };
    auto lambda_priormean=[](VectorXd const &X,VectorXd const &h){
      return VectorXd::Zero(X.size());
    };

    VectorXd x(nobs);
    for(int i=0;i<nobs;i++){
      x(i)=(i+1)/(nobs+1.0);
    }
    VectorXd y=lambda_truefct(x);
    AUGDATA a; a.SetX(x); a.SetValue(y);
    vector<AUGDATA> augdata; //vecteur de taille 1
    augdata.push_back(a);

  
    DoE doe_init(lb_t,ub_t,300,10); // DoE Halton. ça détemrine la précision de l'intégrale pour le calcul des hyperparamètres KOH.
    string foldname="results/";
    WriteObs(foldname+"obs.gnu",augdata);

    Density Dens(doe_init);
    Dens.SetModel(lambda_model);
    Dens.SetKernel(Kernel_Z_SE); //n'utilisons pas les dérivées pour ce cas.
    Dens.SetHparsBounds(lb_hpars,ub_hpars);
    Dens.SetLogPriorHpars(logprior_hpars);
    Dens.SetLogPriorPars(logprior_pars);
    Dens.SetPriorMean(lambda_priormean);
    Dens.SetDataExp(augdata);
    Dens.SetXprofile(augdata[0].GetX()); 
    DensityOpt DensOpt(Dens);
  

  //ok on a tout ce qui est nécessaire pour calculer la vraisemblance et les hpars optimaux je crois. mettons qd mm les data dans un format augdata ? 
  //KOH
  begin=chrono::steady_clock::now();
  VectorXd hpars_koh=0.5*(lb_hpars+ub_hpars);
  hpars_koh=Dens.HparsKOH(hpars_koh,20);
  cout << "hpars koh : " << hpars_koh.transpose() << endl;
  //remplissage de la densité KOH.
  map_double kohlogpost(comp);
  for_each(grid_theta.begin(),grid_theta.end(),[hpars_koh,Dens,&kohlogpost](pair<VectorXi,VectorXd> const &p){
    VectorXi Xi=p.first;
    double logp=Dens.loglikelihood_theta(p.second,hpars_koh); //pas de prior sur theta
    kohlogpost.insert(make_pair(
      Xi,logp
    ));
  });
  //on prend l'exponentielle et on normalise chaque densité.
  //déjà : trouver le maximum de chaque densité pour la normalisation.
  Take_exp_map_double(kohlogpost);
  //calcul des intégrales et normalisation...
  Normalisation_map_double(kohlogpost,ub_t(0)-lb_t(0));
  WriteDensity(grid_theta,kohlogpost,"results/pkoh.gnu");
  end=chrono::steady_clock::now();
  cout << "time KOH : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;

  //NOEDM
  begin=chrono::steady_clock::now();
  Dens.SetHparsBounds(lb_hpars_noedm,ub_hpars_noedm);
  VectorXd hpars_noedm=0.5*(lb_hpars_noedm+ub_hpars_noedm);
  hpars_noedm=Dens.HparsKOH(hpars_noedm,10);
  cout << "hpars noedm : " << hpars_noedm.transpose() << endl;


  //remplissage de la densité noedm.
  map_double noedmlogpost(comp);
  for_each(grid_theta.begin(),grid_theta.end(),[hpars_noedm,Dens,&noedmlogpost](pair<VectorXi,VectorXd> const &p){
    VectorXi Xi=p.first;
    double logp=Dens.loglikelihood_theta(p.second,hpars_noedm); //pas de prior sur theta
    noedmlogpost.insert(make_pair(
      Xi,logp
    ));
  });
  //on prend l'exponentielle et on normalise chaque densité.
  //déjà : trouver le maximum de chaque densité pour la normalisation.
  Take_exp_map_double(noedmlogpost);
  Normalisation_map_double(noedmlogpost,ub_t(0)-lb_t(0));
  WriteDensity(grid_theta,noedmlogpost,"results/pnoedm.gnu");
  end=chrono::steady_clock::now();
  cout << "time NOEDM : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;


  //OPT  
  //calcul des hpars optimaux 

  begin=chrono::steady_clock::now();
  VectorXd hpars_test=0.5*(lb_hpars+ub_hpars);
  map_VXd hparsopt(comp);
  for_each(grid_theta.begin(),grid_theta.end(),[hpars_test,DensOpt,&hparsopt](pair<VectorXi,VectorXd> const &p){
    VectorXi Xi=p.first;
    VectorXd hopt=DensOpt.HparsOpt(p.second,hpars_test,0.5);
    hparsopt.insert(make_pair(Xi,hopt));
  });
  //remplissage de la densité.
  map_double optlogpost(comp);
  for_each(grid_theta.begin(),grid_theta.end(),[hparsopt,Dens,&optlogpost](pair<VectorXi,VectorXd> const &p){
    VectorXi Xi=p.first;
    double logp=Dens.loglikelihood_theta(p.second,hparsopt.at(p.first)); //pas de prior sur theta
    optlogpost.insert(make_pair(
      Xi,logp
    ));
  });
  //calcul max, exponentielle et normalisation
  Take_exp_map_double(optlogpost);
  Normalisation_map_double(optlogpost,ub_t(0)-lb_t(0));
  WriteDensity(grid_theta,optlogpost,"results/popt.gnu");
  end=chrono::steady_clock::now();
  cout << "time OPT : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;

  //affichage des hpars optimaux

ofstream ofile("results/hparsopt.gnu");
for_each(grid_theta.begin(),grid_theta.end(),[&ofile,hparsopt](pair<VectorXi,VectorXd> const &p){
  VectorXi indice=p.first;
  VectorXd h=hparsopt.at(indice);
  VectorXd t=p.second;
  ofile <<t(0)<< " " << h(0) << " " << h(1) << " " << h(2) << endl;
});
ofile.close();




  //calcul Bayes. 
  //remplissage de la densité Bayes
  begin=chrono::steady_clock::now();
  
  map_double bayeslogpost(comp);
  for_each(grid_total.begin(),grid_total.end(),[Dens,&bayeslogpost,logprior_hpars](pair<VectorXi,VectorXd> const &p){
    VectorXd theta=p.second.head(1);
    VectorXd hpars=p.second.tail(3);
    double logp=Dens.loglikelihood_theta(theta,hpars)+logprior_hpars(hpars); //pas de prior sur theta
    VectorXi Xi=p.first;
    bayeslogpost.insert(make_pair(
      Xi,logp
    ));
  });
  
  Take_exp_map_double(bayeslogpost);
  //calcul de la marginale bayes en theta.
  map_double bayesposttheta(comp);
  //tous les éléments sont mis à zero
  for_each(grid_theta.begin(),grid_theta.end(),[&bayesposttheta](pair<VectorXi,VectorXd> const &p){
    VectorXi Xi=p.first;
    double a=0;
    bayesposttheta.insert(make_pair(Xi,a));
  });

  //on somme progressivement
  for_each(bayeslogpost.begin(),bayeslogpost.end(),[&bayesposttheta](pair<VectorXi,double> p){
    VectorXi theta_indice=p.first.head(1);
    bayesposttheta[theta_indice]+=p.second;
  });
  
  VectorXi ttt(1); ttt(0)=5; //pour test
  cout << bayesposttheta[ttt] << endl;

  //calcul de l'intégrale et normalisation

Normalisation_map_double(bayesposttheta,ub_t(0)-lb_t(0));
WriteDensity(grid_theta,bayesposttheta,"results/pbayes.gnu");
end=chrono::steady_clock::now();
cout << "time BAYES : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;


//figures post-traitement

//affichage des postérieures des hyperparamètres. pratique à afficher sur les figures.
//faisons des objets 1D. plus simple.
VectorXi refinement_edm(1); refinement_edm << refinement_total(1); //attention quand la dim hpars change.
VectorXi refinement_exp(1); refinement_exp << refinement_total(2);
VectorXi refinement_lcor(1); refinement_lcor << refinement_total(3);
VectorXd lb_edm(1),lb_exp(1),lb_lcor(1); lb_edm << lb_hpars(0); lb_exp << lb_hpars(1); lb_lcor << lb_hpars(2); 
VectorXd ub_edm(1),ub_exp(1),ub_lcor(1); ub_edm << ub_hpars(0); ub_exp << ub_hpars(1); ub_lcor << ub_hpars(2); 
map_VXd grid_edm(comp);
map_VXd grid_exp(comp);
map_VXd grid_lcor(comp);
fill_grid(grid_edm,refinement_edm,lb_edm,ub_edm);
fill_grid(grid_exp,refinement_exp,lb_exp,ub_exp);
fill_grid(grid_lcor,refinement_lcor,lb_lcor,ub_lcor);
map_double margbayes_edm(comp);
map_double margbayes_exp(comp);
map_double margbayes_lcor(comp);
//initialisation des densités à zero
//tous les éléments sont mis à zero
  for_each(grid_edm.begin(),grid_edm.end(),[&margbayes_edm](pair<VectorXi,VectorXd> const &p){
    VectorXi Xi=p.first;
    double a=0;
    margbayes_edm.insert(make_pair(Xi,a));
  });
    for_each(grid_exp.begin(),grid_exp.end(),[&margbayes_exp](pair<VectorXi,VectorXd> const &p){
    VectorXi Xi=p.first;
    double a=0;
    margbayes_exp.insert(make_pair(Xi,a));
  });
    for_each(grid_lcor.begin(),grid_lcor.end(),[&margbayes_lcor](pair<VectorXi,VectorXd> const &p){
    VectorXi Xi=p.first;
    double a=0;
    margbayes_lcor.insert(make_pair(Xi,a));
  });
  //je vais faire un compteur pour m'assurer que tous les élements sont incrémentés autant.
  //on remplit tout à partir de la densité bayes complète.
  for_each(bayeslogpost.begin(),bayeslogpost.end(),[&margbayes_edm,&margbayes_exp,&margbayes_lcor](pair<VectorXi,double> p){
    VectorXi ind_edm(1); ind_edm << p.first(1);
    VectorXi ind_exp(1); ind_exp << p.first(2);
    VectorXi ind_lcor(1); ind_lcor << p.first(3);
    margbayes_edm[ind_edm]+=p.second;
    margbayes_exp[ind_exp]+=p.second;
    margbayes_lcor[ind_lcor]+=p.second;
  });
  
  Normalisation_map_double(margbayes_edm,ub_edm(0)-lb_edm(0));
  Normalisation_map_double(margbayes_exp,ub_exp(0)-lb_exp(0));
  Normalisation_map_double(margbayes_lcor,ub_lcor(0)-lb_lcor(0));

  WriteDensity(grid_edm,margbayes_edm,"results/margbayesedm.gnu");
  WriteDensity(grid_exp,margbayes_exp,"results/margbayesexp.gnu");
  WriteDensity(grid_lcor,margbayes_lcor,"results/margbayeslcor.gnu");

  //affichage du logprior de l dans un fichier.

  map_double prior_lcor(comp);
    for_each(grid_lcor.begin(),grid_lcor.end(),[&prior_lcor,logprior_hpars](pair<VectorXi,VectorXd> const &p){
    VectorXd hpars=VectorXd::Zero(3);
    hpars(2)=p.second(0);
    double logp=logprior_hpars(hpars); //pas de prior sur theta
    VectorXi Xi=p.first;
    prior_lcor.insert(make_pair(
      Xi,logp
    ));
  });
  Take_exp_map_double(prior_lcor);
  Normalisation_map_double(prior_lcor,ub_lcor(0)-lb_lcor(0));
  WriteDensity(grid_lcor,prior_lcor,"results/priorlcor.gnu");


  exit(0);

//deuxième approche. On trouve les deux modes de la densité par optimisation, puis on fait l'approximation de Laplace en ces modes, puis on recalcule les densités de cette manière, comme validation de nos formules.
//trouver les modes de la densité en 4D. On suppose qu'il y en a deux.





  /*




  int samp_size=100;
  

  
  Dopt.Build();
  cout << "MAP : " << Dopt.MAP().transpose() << endl;
  cout << "Mean : " << Dopt.Mean().transpose() << endl;
  cout << Dopt.Cov() << endl;
  
  Dopt.WritePost("results/popt.gnu");
  Dopt.WriteHpars("results/hopt.gnu");
  VectorXd temp=Dopt.DrawSample(generator);
  PrintVector(X,temp,"results/predopt1.gnu");
  temp=Dopt.DrawSample(generator);
  PrintVector(X,temp,"results/predopt2.gnu");
  temp=Dopt.DrawSample(generator);
  PrintVector(X,temp,"results/predopt3.gnu");
  cout << "Pred Opti over." << endl;
  
 
  {
    //tri des samples en deux bins. critère : 0.5
    vector<VectorXd> thetaselect=mcmc.GetSelectedSamples();
    vector<double> fselect=mcmc.GetSelectedValues();
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
    //Find le max à gauche et le max à droite en utilisant les selected samples.
    VectorXd t_maxleft(4);
    VectorXd t_maxright(4);
    double f_maxleft=0;
    double f_maxright=0;
    for (int i=0;i<thetaselect.size();i++){
      if(fselect[i]>f_maxleft && thetaselect[i](0)<0.5){
        f_maxleft=fselect[i];
        t_maxleft=thetaselect[i];
      }
      else if (fselect[i]>f_maxright && thetaselect[i](0)>0.5){
        f_maxright=fselect[i];
        t_maxright=thetaselect[i];
      }
    }

    cout << " max à gauche : " << t_maxleft.transpose() << "logp : " << f_maxleft << endl; //
    cout << " max à droite : " << t_maxright.transpose() << "logp : " << f_maxright << endl;
    MatrixXd COVLEFT=cov_mcmc(thetaselectleft,fselectleft);
    MatrixXd COVRIGHT=cov_mcmc(thetaselectright,fselectright);
  
    //cout << "matrice cov à gauche : \n" << COVLEFT << endl;
    //cout << "matrice cov à droite : \n" << COVRIGHT << endl;

    double detleft=pow(COVLEFT.llt().matrixL().determinant(),2);
    double detright=pow(COVRIGHT.llt().matrixL().determinant(),2);
    double ratio=sqrt(detleft/detright);

    f_maxleft=exp(f_maxleft);
    f_maxright=exp(f_maxright);
    double pi1=(double) thetaselectleft.size()/((double) thetaselectleft.size()+thetaselectright.size());
    cout << "pi1 : " << pi1 << endl;
    //Calcul du critère KOH :
    MatrixXd COVLEFT_hpars=COVLEFT.block(1,1,3,3);
    VectorXd COVLEFT_cross=COVLEFT.col(0).tail(3); //les 3 derniers éléments de la première colonne
    MatrixXd COVRIGHT_hpars=COVRIGHT.block(1,1,3,3);
    VectorXd COVRIGHT_cross=COVRIGHT.col(0).tail(3); //les 3 derniers éléments de la première colonne
    double critkohleft=pi1/(COVLEFT_hpars-COVLEFT_cross*COVLEFT_cross.transpose()/COVLEFT(0,0)).llt().matrixL().determinant();//c'est bien la racine du déterminant.
    double critkohright=(1-pi1)/(COVRIGHT_hpars-COVRIGHT_cross*COVRIGHT_cross.transpose()/COVRIGHT(0,0)).llt().matrixL().determinant();//c'est bien la racine du déterminant.
    cout << "crit koh left : " << critkohleft << ", right : " << critkohright << endl;
    bool choice_koh_left=(critkohleft>critkohright); //trouver si KOH choisit left ou right
    VectorXd t_maxkoh(4);
    double var_maxkoh(0);
    if(choice_koh_left){t_maxkoh=t_maxleft; var_maxkoh=COVLEFT(0,0)-COVLEFT_cross.transpose()*COVLEFT_hpars.llt().solve(MatrixXd::Identity(3,3))*COVLEFT_cross;}

    else{t_maxkoh=t_maxright; var_maxkoh=COVRIGHT(0,0)-COVRIGHT_cross.transpose()*COVRIGHT_hpars.llt().solve(MatrixXd::Identity(3,3))*COVRIGHT_cross;}

    double varbayesleft=COVLEFT(0,0);
    double varbayesright=COVRIGHT(0,0);
    double varfmpleft=varbayesleft;
    double varfmpright=varbayesright;
    double pi1fmp=pi1*(sqrt(COVLEFT(0,0)/detleft))/(pi1*(sqrt(COVLEFT(0,0)/detleft))+(1-pi1)*(sqrt(COVRIGHT(0,0)/detright)));
    cout << "pi1fmp : " << pi1fmp << endl;
    cout << "varright : " << varbayesright << ", varkohright : " << var_maxkoh << endl;
    cout << "COVLEFT : " << COVLEFT << endl;
    cout << "COVRIGHT : " << COVRIGHT << endl;

    FILE* out =fopen("results/recons.gnu","w");

    for (int i=0;i<140;i++){
      double theta=-0.5+2*((double) i)/140.;
      double p_bayes=post_bayes_rec(theta,pi1,t_maxleft(0),varbayesleft,t_maxright(0),varbayesright);
      double p_fmp=post_fmp_rec(theta,pi1fmp,t_maxleft(0),varfmpleft,t_maxright(0),varfmpright);
      double p_koh=post_koh_rec(theta,t_maxkoh(0),var_maxkoh);
      fprintf(out, "%e %e %e %e\n",theta,p_fmp,p_koh,p_bayes);    
    }
    fclose(out);

  }



  exit(0);
  DensityBayes DBayes(MainDensity);
  DBayes.SetSampleHpars(sample_hpars);
  DBayes.Build();
  cout << DBayes.MAP().transpose() << endl;
  cout << DBayes.Mean().transpose() << endl;
  cout << DBayes.Cov() << endl;
  cout << DBayes.Entropy() << endl;
  DBayes.WritePost("resultsdensity/pbayes.gnu");
  exit(0);
  }
  */
}
