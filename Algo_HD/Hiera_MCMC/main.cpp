// Dans ce fichier : on met en oeuvre différents algorithmes d'optimisation pour KOH. Voir lequel est le meilleur.
// On reproduit l'exemple 1 de l'article avec notre nouveau fichier pour être solide.


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
#include "halton.hpp"
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


double my_function(VectorXd const &x){
  return x(0);
};

double my_model(VectorXd const &x,VectorXd const &theta){
  return x(0)*sin(2.0*theta(0)*x(0))+(x(0)+0.15)*(1-theta(0));
}

double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &xp, Eigen::VectorXd const &hpar){
  //Fonction Kernel sans bruit. hpar(0) = sig_edm, hpar(1) = sig_exp, hpar(2) = lcor
   return hpar(0)*hpar(0)*exp(-pow((x(0)-xp(0))/hpar(2),2)*.5 ); /* squared exponential kernel */
};

double PriorMean(VectorXd const &x, VectorXd const &hpars){
  return 0;
}

double logprior_hpars(VectorXd const &hpars){
  if(hpars(2)<=0){return -999;}
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

void PrintVector(vector<VectorXd> &X, const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e\n",X[i](0));
  }
  fclose(out);
}

MatrixXd Gamma(void const *data, Eigen::VectorXd const &hpar){
  // Renvoie la matrice de corrélation avec  bruit
  vector<DATA>* data2 = (vector<DATA>*) data; // cast du null pointer en type désiré
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

double loglikelihood_theta(void *data, Eigen::VectorXd const &hpar, VectorXd const &theta){
  //besoin de réécrire cette fonction pour être plus générale que celle de Density. On veut pouvoir évaluer le modèle à un theta et X quelconque.
  //renvoie la log-vraisemblance des hyperparamètres hpar, étant donné les données data et les paramètres du modèle theta.
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  std::vector<DATA> data3;
  for(unsigned ie=0; ie<nd; ie++){
    DATA dat; dat.SetX((*data2)[ie].GetX()); dat.SetValue((*data2)[ie].Value()-(my_model((*data2)[ie].GetX(),theta)+PriorMean((*data2)[ie].GetX(),hpar)));
    data3.push_back(dat);
  }
  return loglikelihood(&data3,hpar);
}


VectorXd random_elem(vector<VectorXd> const &v){
  int randomindex=rand()%v.size();
  return v[randomindex];
}
vector<double> RandHpars(default_random_engine &generator, vector<double> &lb, vector<double> &ub){
  vector<double> x(3);
  x[0]=lb[0]+(ub[0]-lb[0])*distU(generator);
  x[1]=lb[1]+(ub[1]-lb[1])*distU(generator);
  x[2]=lb[2]+(ub[2]-lb[2])*distU(generator);
  return x;
}

double optfunc(const std::vector<double> &x, std::vector<double> &grad, void *data){
  Eigen::VectorXd hpars(3);
  for(int p=0; p<3; p++) {hpars(p) =x[p];} // copie des hyperparamètres dans un VectorXd
  //maintenant qu'on connaît les hyperparamètres, il faut soustraire la moyenne a priori aux données. /!
  double d=loglikelihood(data,hpars)+logprior_hpars(hpars);
  return d;
}

VectorXd hpars_opti(VectorXd &theta,VectorXd &hpars_precedent) {
  //optimisation des hpars
   std::default_random_engine generator;
  std::uniform_real_distribution<double> distU(0,1);
  int m_dim_hpars=3;
  std::vector<double> lb_hpars(3); lb_hpars[0]=exp(-5);lb_hpars[1]=exp(-6);lb_hpars[2]=exp(-3); //-5 avant
  std::vector<double> ub_hpars(3); ub_hpars[0]=exp(0);ub_hpars[1]=exp(-1);ub_hpars[2]=2; //bornes sur l'optimisation des hpars. edm exp lcor
  int maxeval=5000;
  double ftol=1e-4;
  int nd=data.size(); //nombre d'observations
  //construction de la densité opt. On fait une boucle sur les theta.
       	//Creation des data
	  std::vector<DATA> data2;
	  for(int ie=0; ie<nd; ie++){
	    DATA dat; 
      dat.SetX(data[ie].GetX());
      dat.SetValue((data[ie].Value())-my_model(data[ie].GetX(),theta));//observations moins évaluations du modèle. on ne met pas prior mean car c'est fait dans optfunc.
      data2.push_back(dat); // on construit les données y-f_t      
    }    
    VectorXd hpars_max_opti(3);
    {
	  int pop=2000; // population size
	  //	cout << "Début de l'optimisation globale... Pop size : " << pop << endl;
	  std::vector<double> x(m_dim_hpars);
    
    for (int j=0;j<m_dim_hpars;j++){
        x[j]=hpars_precedent(j); //warm_restart
    } 
    // initial guess}
    nlopt::opt opt(nlopt::GN_ISRES, m_dim_hpars);    /* algorithm and dimensionality */
	  opt.set_max_objective(optfunc, &data2);
	  opt.set_lower_bounds(lb_hpars);
	  opt.set_upper_bounds(ub_hpars);
	  opt.set_maxeval(maxeval);
	  opt.set_population(pop);
	  opt.set_ftol_rel(ftol);		
	  double msup; /* the maximum objective value, upon return */
	  int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 
	  for(int i=0;i<m_dim_hpars;i++){hpars_max_opti(i)=x[i];}
	  //cout << "Message d'arrêt : " << fin <<". max trouvé : " << hpars_max_opti.transpose() << ", valeur du critère : " << msup <<endl;
  	}
	//Etape 2 : opti locale
	  {
  	  //	cout << "Début de l'optimisation locale..." << endl;
	    double msuploc;
	    /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
	    std::vector<double> x(m_dim_hpars);
	    for (int j=0;j<m_dim_hpars;j++){x[j]=hpars_max_opti(j);}
	    nlopt::opt opt(nlopt::LN_SBPLX, m_dim_hpars);    /* algorithm and dimensionality */
	    opt.set_max_objective(optfunc, &data2);
	    opt.set_lower_bounds(lb_hpars);
	    opt.set_upper_bounds(ub_hpars);
	    opt.set_maxeval(maxeval);
	    opt.set_ftol_rel(ftol);
	    double msup; /* the maximum objective value, upon return */
	    int fin=opt.optimize(x, msup);
	    for(int i=0;i<m_dim_hpars;i++){hpars_max_opti(i)=x[i];}
	    msuploc=msup;
	    //cout << "Message d'arrêt : " << fin  << ". Max : " << hpars_max_opti.transpose() << ", valeur du critère : " << msuploc << endl;
	    int niter_optimisations=100; // On recommence l'opti tant qu'on n'a pas stagné 100 fois.
	    int nopt=0;
	    while (nopt<niter_optimisations){
	      nopt++;
	      x=RandHpars(generator,lb_hpars,ub_hpars);
	      int fin= opt.optimize(x, msup);
	      if(msup>msuploc){
	        nopt=0;
	        msuploc=msup;
	        for(int k=0;k<m_dim_hpars;k++){
		        hpars_max_opti(k)=x[k];
	        }
	       //cout << "Message d'arrêt : " << fin  << ". Nouveau max : " << hpars_max_opti.transpose() << ", valeur du critère : " << msuploc << endl;
	      }
  	  }
	  //	cout << "hyperparametres KOH : (edm, exp, lcor) : " << hpars_max_koh.transpose() << " a la vraisemblance : " << max_vrais_koh << endl << endl;    }
	  data2.clear();
    return hpars_max_opti;
    }

}





const double Big = -1.e16;


int main(int argc, char **argv){
  

  if(argc != 3){
    cout << "Usage :\n";
    cout << "\t Number of obs, seed_obs\n";
    exit(0);
  }
  int nd  = atoi(argv[1]);
  uint32_t seed_obs=atoi(argv[2]);//

  // Bornes des paramètres et hyperparamètres
  VectorXd lb_t(1); lb_t(0)=-0.5;
  VectorXd ub_t(1); ub_t(0)=1.5; 

  std::vector<double> lb_hpars(3); lb_hpars[0]=exp(-5);lb_hpars[1]=exp(-6);lb_hpars[2]=exp(-3); //-5 avant
  std::vector<double> ub_hpars(3); ub_hpars[0]=exp(0);ub_hpars[1]=exp(-1);ub_hpars[2]=2; //bornes sur l'optimisation des hpars. edm exp lcor
  
  generator.seed(seed_obs);
  //construction des observations et écriture dans un fichier
  FILE* out = fopen("results/observations.gnu","w");
  double xp_noise=0.01;
  
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
  
  vector<VectorXd> X(150); for (int i=0;i<X.size();i++){VectorXd x(1); x(0)=double(i)/double(X.size()); X[i]=x;}
  //construction du DoE initial en grid
  DoE doe_init(lb_t,ub_t,140);//,generator);
  doe_init.Fill(my_model,&data);
  doe_init.Fill_Predictions(X);

  //configuration de l'instance de base de densité
  Density MainDensity(doe_init);
  MainDensity.SetModel(my_model);
  MainDensity.SetKernel(Kernel);
  MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetPriorMean(PriorMean);

  int samp_size=100;
  

  /* Partie MCMC */
  {
    //MCMC full
  int nchain=2000000;
  VectorXd t_init(4);
  t_init(0)=1;
  t_init(1)=0.07;
  t_init(2)=0.01;
  t_init(3)=0.20;
  
  MatrixXd COV=MatrixXd::Zero(4,4);
  COV(0,0)=pow(0.13,2);
  COV(1,1)=pow(0.03,2);
  COV(2,2)=pow(0.005,2);
  COV(3,3)=pow(0.03,2);
  MCMC mcmc(MainDensity,nchain);
  mcmc.Run(t_init,COV,generator);
  mcmc.SelectSamples(200);
  cout << "map :" << mcmc.MAP().transpose() << endl;
  cout << "mean :" << mcmc.Mean().transpose() << endl;
  cout << "cov : " << mcmc.Cov() << endl;
  mcmc.WriteSelectedSamples("results/mcmcselectedsamples.gnu");
  mcmc.WriteAllSamples("results/mcmcallsamples.gnu");
  mcmc.WritePredictions("results/predmcmc.gnu");
  mcmc.WritePredictionsFZ("results/predmcmcFZ.gnu");
  
  }
  vector<VectorXd> selected_samples;

  {
    int nchain=2000000;
    int nselection=2000;
    int naccept(0);
    //MCMC sur la totale (fait à la main)
    VectorXd t_init(4);
    t_init(0)=1;
    t_init(1)=0.07;
    t_init(2)=0.01;
    t_init(3)=0.20;
    MatrixXd COV=MatrixXd::Zero(4,4);
    COV(0,0)=pow(0.13,2);
    COV(1,1)=pow(0.03,2);
    COV(2,2)=pow(0.005,2);
    COV(3,3)=pow(0.03,2);
    MatrixXd sqrtCOV=COV.llt().matrixL();
    VectorXd Xcurrent=t_init;
    VectorXd Xcandidate=Xcurrent;
    double fcurrent=loglikelihood_theta(&data,Xcurrent.tail(3),Xcurrent.head(1))+logprior_hpars(Xcurrent.tail(3))+logprior_pars(Xcurrent.head(1));
    double fcandidate(0);
    clock_t c_start = std::clock();
    for (int i=0;i<nchain;i++){
    VectorXd Step(4);
    for (int j=0;j<Step.size();j++){Step[j]=distN(generator);}
    Xcandidate=Xcurrent+sqrtCOV*Step;
    fcandidate=loglikelihood_theta(&data,Xcandidate.tail(3),Xcandidate.head(1))+logprior_hpars(Xcandidate.tail(3))+logprior_pars(Xcandidate.head(1));
    if(Xcandidate(0)>-0.5 && Xcandidate(0)<1.5){
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(i>5000 && i%(nselection)==0){
      selected_samples.push_back(Xcurrent);
    }
    //ici ajouter les valeurs de la chaîne si l'on souhaite les conserver
  }
  clock_t c_end = std::clock();
  double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  double time_mcmc=time_elapsed_ms / 1000.0;
  std::cout << "Temps pour MCMC full: " << time_mcmc << " s\n";
  cout << "accept rate : " << 100*double(naccept)/double(nchain) << endl;
  PrintVector(selected_samples,"results/ssfull.gnu");

  }
  {
    //MCMC sur les conditionnelles (en 1 dimension)
  double theta_seuil=0.5;
  VectorXd hpars_left(3);
  hpars_left << 9.45E-2,8E-3,0.35;
  VectorXd hpars_right(3);
  hpars_right << 5.73E-2,8E-3,0.20;


  vector<VectorXd> selected_samples_left;
  vector<VectorXd> selected_samples_right;
  vector<VectorXd> selected_samples_joint;
  //MCMC à gauche
  int nchain=2000000;
  int nselection=2000;
  int naccept(0);
  VectorXd t_init(1);
  t_init(0)=0;
  double cov_propleft=pow(0.49,2);
  VectorXd Xcurrent=t_init;
  VectorXd Xcandidate(1);
  double fcurrent=loglikelihood_theta(&data,hpars_left,Xcurrent)+logprior_hpars(hpars_left)+logprior_pars(Xcurrent);
  double fcandidate(0);
  double step(0);
  clock_t c_start = std::clock();
  for (int i=0;i<nchain;i++){
    Xcandidate(0)=Xcurrent(0)+distN(generator)*sqrt(cov_propleft);
    fcandidate=loglikelihood_theta(&data,hpars_left,Xcandidate)+logprior_hpars(hpars_left)+logprior_pars(Xcandidate);
    if(Xcandidate(0)>-0.5 && Xcandidate(0)<1.5){
      if(fcandidate>fcurrent | fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(i>5000 && i%(nselection)==0){
      selected_samples_left.push_back(Xcurrent);
    }
  }
  cout << "accept rate mcmc left : " << 100*double(naccept)/double(nchain) << endl;
  clock_t c_end = std::clock();
  std::cout << "Temps pour MCMC left : " << (c_end-c_start) / CLOCKS_PER_SEC << " s\n";
  PrintVector(selected_samples_left,"results/ssleft.gnu");

  //MCMC right
  double cov_propright=pow(0.22,2);
  naccept=0;
  Xcurrent=t_init;
  fcurrent=loglikelihood_theta(&data,hpars_right,Xcurrent)+logprior_hpars(hpars_right)+logprior_pars(Xcurrent);
  for (int i=0;i<nchain;i++){
    Xcandidate(0)=Xcurrent(0)+distN(generator)*sqrt(cov_propright);
    fcandidate=loglikelihood_theta(&data,hpars_right,Xcandidate)+logprior_hpars(hpars_right)+logprior_pars(Xcandidate);
    if(Xcandidate(0)>-0.5 && Xcandidate(0)<1.5){
      if(fcandidate>fcurrent | fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(i>5000 && i%(nselection)==0){
      selected_samples_right.push_back(Xcurrent);
    }
  }
  cout << "accept rate mcmc right : " << 100*double(naccept)/double(nchain) << endl;
  c_start = std::clock();
  std::cout << "Temps pour MCMC left : " << -(c_end-c_start) / CLOCKS_PER_SEC << " s\n";
  PrintVector(selected_samples_right,"results/ssright.gnu");


  //Gibbs sampling joint
  Xcurrent=random_elem(selected_samples_left);
  for (int i=0;i<nchain;i++){
    if(Xcurrent(0)<theta_seuil){
      Xcurrent=random_elem(selected_samples_left);
    }
    if(Xcurrent(0)>=theta_seuil){
      Xcurrent=random_elem(selected_samples_right);
    }
    if(i>5000 && i%(nselection)==0){
      selected_samples_joint.push_back(Xcurrent);
    }
  }
  PrintVector(selected_samples_joint,"results/ssjoint.gnu");

  }
  // MCMC avec optimisation
    {
    int nchain=2000000;
    int nselection=2000; //on sélectionne tous les 2000 steps donc 1000 pts
    int nopti=20000; //on optimise tous les 20000 steps donc 100 optis.
    int naccept(0);
    //MCMC sur la totale (fait à la main)
    VectorXd t_init(4);
    t_init(0)=1;
    t_init(1)=0.07;
    t_init(2)=0.01;
    t_init(3)=0.20;
    MatrixXd COV=MatrixXd::Zero(4,4);
    COV(0,0)=pow(0.30,2);
    COV(1,1)=pow(0.03,2);
    COV(2,2)=pow(0.005,2);
    COV(3,3)=pow(0.03,2);
    MatrixXd sqrtCOV=COV.llt().matrixL();
    VectorXd Xcurrent=t_init;
    VectorXd Xcandidate=Xcurrent;
    double fcurrent=loglikelihood_theta(&data,Xcurrent.tail(3),Xcurrent.head(1))+logprior_hpars(Xcurrent.tail(3))+logprior_pars(Xcurrent.head(1));
    double fcandidate(0);
    clock_t c_start = std::clock();
    for (int i=0;i<nchain;i++){
      if(i%(nopti)==0){
        VectorXd hpars_prec=Xcurrent.tail(3);
        Xcurrent.tail(3)=hpars_opti(Xcurrent,hpars_prec);
        //cout << "nouveaux hpars opti : " << Xcurrent.tail(3).transpose() << "a theta : "<< Xcurrent.head(1) << endl;
      }     
    VectorXd Step(4);
    for (int j=0;j<Step.size();j++){Step[j]=distN(generator);}
    Xcandidate=Xcurrent+sqrtCOV*Step;
    Xcandidate.tail(3)=Xcurrent.tail(3);
    fcandidate=loglikelihood_theta(&data,Xcandidate.tail(3),Xcandidate.head(1))+logprior_hpars(Xcandidate.tail(3))+logprior_pars(Xcandidate.head(1));
    if(Xcandidate(0)>-0.5 && Xcandidate(0)<1.5){
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    if(i>5000 && i%(nselection)==0){
      selected_samples.push_back(Xcurrent);
    }
    //ici ajouter les valeurs de la chaîne si l'on souhaite les conserver
  }
  clock_t c_end = std::clock();
  double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  double time_mcmc=time_elapsed_ms / 1000.0;
  std::cout << "Temps pour MCMC opti1: " << time_mcmc << " s\n";
  cout << "accept rate : " << 100*double(naccept)/double(nchain) << endl;
  PrintVector(selected_samples,"results/ssopti1.gnu");

  }

  exit(0);
}
