// Tests pour l'adaptative DoE. Dans ce fichier, on approche juste une fonction avec un PG. On n'a pas tout le framework réalité/modèle.
//Code utilisé pour approcher une fonction (my_function) par un processsus gaussien dans le framework OLM. On utilise l'algorithme Bayesian Optimisation pour trouver les points d'acquisition. On teste différentes fonctions d'acquisition : Expected Improvement, et mon critère.


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
#include "gp++.h"

using namespace std;
using namespace Eigen;
int neval=0;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);
VectorXd lb_t(1);
VectorXd ub_t(1);



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
  // Construction du grid initial de thetas. On le fait une bonne fois pour toutes. Il faut construire au préalable les vecteurs lb_t et ub_t car on ne peut pas les déclarer de manière globale.
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
  // Construction du grid initial de thetas. On le fait une bonne fois pour toutes. Il faut construire au préalable les vecteurs lb_t et ub_t car on ne peut pas les déclarer de manière globale.
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

double my_function(VectorXd const &x){
  return x(0)*sin(x(0));
};

/* Evaluate Kernel of the Stochastic process for the two points x and y, given the parameters in par:
	- par(0) is the variance,
	- par(1) is the correlation length
*/
double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &y, const Eigen::VectorXd &par){
	return par(1)*exp( -((x-y)/par(0)).squaredNorm()*.5 ); /* squared exponential kernel */
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


double MaxValGP(GP &gp){
  //renvoie la valeur maximale d'un GP observée jusqu'à présent (utile pour l'expected improvement)
  double max_courant=gp.FDat(0);
  for (int i=1;i<gp.NDat();i++){
    if(gp.FDat(i)>max_courant){
      max_courant=gp.FDat(i);
    }
  }
  return max_courant;
}
double EI(GP &gp, VectorXd &xprime){
  //calcule le critère d'expected improvement au point x.
  double fmax(MaxValGP(gp));
  VectorXd Pred=gp.Eval(xprime); //prédiction du gp. case 1 : moyenne, case 2 : variance.
  double Z=(Pred(0)-fmax)/sqrt(Pred(1));
  return (Pred(0)-fmax)*0.5*(1+erf(Z/sqrt(2)))+sqrt(Pred(1))*exp(-0.5*pow(Z,2));
}

double mycriteria(GP &gp, VectorXd &xprime){
  //calcule mon critère.
  VectorXd Pred=gp.Eval(xprime); //prédiction du gp. case 1 : moyenne, case 2 : variance.
  double m(1./Pred(1));
  Eigen::LDLT<Eigen::MatrixXd> ldlt=gp.GetLDLT();
  //calcule le vecteur k à xprime
  VectorXd kp(gp.NDat());
  for (int i=0;i<kp.size();i++){kp(i)=gp.Kij(gp.XDat(i),xprime);}
  VectorXd alpha=ldlt.solve(kp); //=K^-1*kp
  //calcul de l'intégrale. On créé un grid de thetastar.
  VectorXd lb_t(1);
  VectorXd ub_t(1);
  lb_t(0)=0;
  ub_t(0)=10;
  const int gridsize=40; //20 points pour calculer l'intégrale
  vector<VectorXd> Grid=InitGridRect(lb_t,ub_t,gridsize);
  double critere(0);
  double weight(0);
  for (const auto& xstar :Grid){
    VectorXd ks(gp.NDat());
    for (int i=0;i<ks.size();i++){ks(i)=gp.Kij(gp.XDat(i),xstar);}
    double g(gp.EvalMean(xstar));
    //double g(my_function(xstar)); si on veut la réponse
    critere+=pow(gp.Kij(xprime,xstar)-ks.transpose()*alpha,2)*g;
    weight+=g;
  }
  critere*=m/weight;
  return critere; 
}

double optfunc__ei(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /*Renvoie l'EI*/
  GP* gp2 = static_cast<GP*>(data); // cast du null pointer en type désiré
  VectorXd X(1);
  X(0)=x[0];
  return EI(*gp2,X);
}

double optfunc__crit(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /*Renvoie l'EI*/
  GP* gp2 = static_cast<GP*>(data); // cast du null pointer en type désiré
  VectorXd X(1);
  X(0)=x[0];
  return mycriteria(*gp2,X);
}

void printGP(GP &gp, const char* file_name){
  //affiche le GP dans un fichier
  int ndim=1;
  FILE* out = fopen(file_name,"w");
  fprintf(out,"#Observation du pg. ndim premieres colonnes : coordonnes, avant-derniere colonne : moyenne pg calibre, derniere colonne : 1 sd du pg calibre \n");
  for (unsigned is = 0; is < 500; is++)
    {
      VectorXd x(ndim);
      for (unsigned id = 0; id < ndim; id++){
	x(id) =lb_t(0)+(ub_t(0)-lb_t(0))*double(is)/500;
	fprintf(out,"%e ",x(id));
      }
      VectorXd eval = gp.Eval(x);
      fprintf(out,"%e %e %e %e %e\n",eval(0),eval(0)+sqrt(eval(1)),eval(0)-sqrt(eval(1)),EI(gp,x),mycriteria(gp,x));
    }
  fclose(out);
}

const double Big = -1.e16;


int main( int argc, char **argv){
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distU(0,1);
  std::normal_distribution<double> distN(0,1);

  // Inputs:
  if(argc != 2){
    cout << "Usage :\n";
    cout << "\t Number of data \n";
    exit(0);
  }
  int npts_init   = atoi(argv[1]);
  const int ndim = 1;

  //Bornes de x

  lb_t(0)=0;
  ub_t(0)=10;
  //Construction du LHS initial
  vector<VectorXd> gridLHS=InitGridLHS(lb_t,ub_t,npts_init);

  VectorXd par(3);
  par(0) = 0.3; par(1) = 9.; //cor length et variance a priori.
  par(2)=1e-2;

  //Réalisation d'observations sur le LHS initial

  FILE* out = fopen("results/observations.gnu","w");
  fprintf(out,"#Fichier des observations. ndim premieres colonnes : coordonnes, derniere colonne : observation (bruitee)\n");
  vector<DATA> data;
  for(const auto& x:gridLHS){
    for(unsigned id=0; id<ndim; id++){
      fprintf(out,"%e ",x(id));
    }
    double f = my_function(x);
    DATA dat; dat.SetX(x); dat.SetValue(f);
    fprintf(out,"%e \n",dat.Value());
    data.push_back(dat);
  }
  fclose(out);
  
  GP gp(Kernel);
  gp.SetData(data);
  gp.SetGP(par);

  int maxeval(2000);
  double ftol(1e-4);
  //boucle d'optimisation
  int nobs_max=50;
  clock_t c_start=std::clock();
  for (int nobs=npts_init;nobs<=nobs_max;nobs++){
    
    string shortprefixe=string("results/ei_")+std::to_string(nobs);
    printGP(gp,(string(shortprefixe)+string("_pg_out.gnu")).c_str());
    VectorXd xNext(1);

    {
      std::vector<double> lb(1),ub(1);
      lb[0]=lb_t(0);
      ub[0]=ub_t(0);
      int pop=2000; // population size
      /*Optimisation de l'EI.*/
      std::vector<double> x(1);
      x[0]=lb[0]+(ub[0]-lb[0])*distU(generator);
      nlopt::opt opt(nlopt::GN_ISRES, 1);    /* algorithm and dimensionality */
      opt.set_max_objective(optfunc__ei, &gp);
      opt.set_lower_bounds(lb);
      opt.set_upper_bounds(ub);
      opt.set_maxeval(maxeval);
      opt.set_population(pop);
      opt.set_ftol_rel(ftol);
      double msup; /* the maximum objective value, upon return */
      int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 
      xNext(0)=x[0];
      cout << "Message d'arrêt : " << fin <<". max trouvé au point : " << xNext << endl;
    }


    //Réaliser l'observation
    DATA dat; dat.SetX(xNext); dat.SetValue(my_function(xNext));
    data.push_back(dat);
    gp.SetData(data);
    gp.SetGP(par);
    VectorXd eval = gp.Eval(xNext);
    FILE* out = fopen("results/observations.gnu","a");
    fprintf(out,"%e %e\n",dat.GetX()(0),dat.Value());
    fclose(out);
  }
  clock_t c_end = std::clock();
  double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  double time_opti=time_elapsed_ms / 1000.0;
  std::cout << "Temps pour opti : " << time_opti << " s\n";
  exit(0);

  /*Sample the GP*/
  int ns = 6;
  vector<VectorXd> Target;
  int itmx = 50;
  for(unsigned it=0; it<itmx; it++){
    VectorXd xi = VectorXd(ndim);
    for (unsigned id = 0; id < ndim; id++){
      xi(id)=10*double(it)/itmx;
    }
    Target.push_back(xi);
  }
  MatrixXd Sample = gp.SampleGPDirect(Target, ns, generator);
  out = fopen("results/gp_samples.gnu","w");
  fprintf(out,"#samples du pg. ndim premieres colonnes : coordonnes, chaque colonne ensuite est une real du pg a posteriori\n");
  for(unsigned it=0; it<itmx; it++){
    for (unsigned id = 0; id < ndim; id++){
      fprintf(out,"%e ",Target[it](id));
    }
    for(unsigned is=0; is<ns; is++){
      fprintf(out,"%e ",Sample(it,is));
    }
    fprintf(out,"\n");
  }
  fclose(out);

  exit(0);
};
