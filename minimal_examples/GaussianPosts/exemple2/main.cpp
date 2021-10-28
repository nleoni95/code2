//exemple très simple où l'on cherche sur des problèmes simples à calculer la postérieure p(theta,psi) pour voir si elle est gaussienne.
//estimation de la longueur de corrélation cette fois-ci.

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
//pour stocker les valeurs calculées d'hyperparamètres optimaux. On a un vectorxd qui est le theta, et une map (int vectorxd) qui permet de retrouver l'hyperparamètre optimal (vectorxd) du cas i (int)

int neval=1;
//std::default_random_engine generator;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);
vector<DATA> data;
vector<VectorXd> Grid;




double gaussprob(double x,double mu, double sigma){
  //renvoie la probabilité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}

double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(2),2)); //3/2
}

double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2)))*exp(-d/hpar(2)); //3/2
}


double kernel(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double s=0.05;
  double d=abs(x(0)-y(0));
  return pow(s,2)*exp(-0.5*pow(d/hpar(0),2)); //3/2
}

double Kernel_Z_Quad(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau linéaire
  double d=(x(0)-hpar(3))*(y(0)-hpar(3));
  return pow(hpar(0),2)+pow(hpar(2),2)*d+pow(hpar(4),2)*pow(d,2);
  //hpar 0 : ordonnée en x=0. Equivalent à cst prior mean.
  // hpars 2 : pente de la fonction
  // hpars 3 : point en x où l'incertitude sera nulle.
  // hpars 4 : coeff du second degré.
}

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2))+(1./3)*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor par1, 3:lcor par2, 4:lcor par3.
  double cor=pow(hpar(0),2);
  cor*=(1+abs(x(0)-y(0))/hpar(1))*exp(-abs(x(0)-y(0))/hpar(1)); //phi
  cor*=(1+abs(x(1)-y(1))/hpar(3))*exp(-abs(x(1)-y(1))/hpar(3)); //BK
  cor*=(1+abs(x(2)-y(2))/hpar(4))*exp(-abs(x(2)-y(2))/hpar(4)); //COAL
  return cor;
}

void Run_Burn_Phase_MCMC(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento,function<double(VectorXd const &)> const & compute_score,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //phase de burn.
  int dim_mcmc=COV_init.cols();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  VectorXd Xinit=Xcurrento;
  MatrixXd COV=COV_init;
  MatrixXd sqrtCOV=COV.llt().matrixL();
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
    if(in_bounds(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(dim_mcmc))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
}

tuple<vector<VectorXd>,vector<double>> Run_MCMC(int nsteps,VectorXd const & Xinit,MatrixXd const & COV_init,function<double(VectorXd const &)> const & compute_score,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //MCMC générique. renvoie un tuple <samples,hparsofsamples,allsamples>
  int dim_mcmc=Xinit.size();
  vector<VectorXd> allsamples;
  vector<double> scores_of_samples;
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  VectorXd Xinit0=Xinit;
  Run_Burn_Phase_MCMC(nsteps*0.1,COV,Xinit0,compute_score,in_bounds,generator);
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(Xinit0);
  VectorXd Xcurrent=Xinit0;
  double fcurrent=finit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds(Xcandidate)){
      double fcandidate=compute_score(Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    allsamples.push_back(Xcurrent);
    scores_of_samples.push_back(fcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;

  cout << allsamples[0].transpose() << endl;
  cout << allsamples[nsteps/2].transpose() << endl;
  auto tp=make_tuple(allsamples,scores_of_samples);
  return tp;
}

tuple<vector<VectorXd>,vector<double>> Run_MCMC_adapt(int nsteps, VectorXd & Xinit, MatrixXd const & COV_init,function<double(VectorXd const &)> const & compute_score,function<bool(VectorXd)> const & in_bounds,double lambda, double gamma, default_random_engine & generator){
  //algorithme andrieu global AM with global adaptive scaling (algorithme 4)
  //paramètres initiaux conseillés :     double lambda=pow(2.38,2)/dim_mcmc; double gamma=0.01;
  cout << "running mcmc adapt with " << nsteps << " steps, adaptative algorithm, gamma = "<< gamma << endl;
  double alphastar=0.234; //valeur conseillée dans l'article. taux d'acceptation optimal.
  vector<VectorXd> allsamples;
  vector<double> scores_of_samples;
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  int dim_mcmc=Xinit.size();
   auto draw_prop=[dim_mcmc](double lambda,MatrixXd COV,default_random_engine & generator, normal_distribution<double> & distN ){
    //tire une proposal de matrice de cov sqrt(lambda)*COV*sqrt(lambda)
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    MatrixXd sqrtCOV=COV.llt().matrixL();
    VectorXd s=sqrt(lambda)*sqrtCOV*Step;
    return s;
  };

  auto update_params=[dim_mcmc,gamma,alphastar](VectorXd & mu, MatrixXd & COV,double & lambda,double alpha,VectorXd Xcurrent){
    //update les paramètres de l'algo MCMC.
    lambda*=exp(gamma*(alpha-alphastar));
    COV=COV+gamma*((Xcurrent-mu)*(Xcurrent-mu).transpose()-COV);
    COV+=1e-10*MatrixXd::Identity(dim_mcmc,dim_mcmc);
    mu=mu+gamma*(Xcurrent-mu);
  };

  MatrixXd COV=COV_init;
  cout << "cov : " <<COV << endl;
  VectorXd mu=Xinit;
  double finit=compute_score(Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  int naccept=0;
  double lambdan=lambda;
  auto begin=chrono::steady_clock::now();
  double alpha=0;
  for(int i=0;i<nsteps;i++){
    VectorXd Xcandidate=Xcurrent+draw_prop(lambdan,COV,generator,distN);
    //cout << "candidate : " << Xcandidate.transpose() << endl;
    if(in_bounds(Xcandidate)){
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
    update_params(mu,COV,lambdan,alpha,Xcurrent);
    allsamples.push_back(Xcurrent);
    scores_of_samples.push_back(fcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << allsamples[0].transpose() << endl;
  cout << allsamples[nsteps/2].transpose() << endl;
  auto tp=make_tuple(allsamples,scores_of_samples);
  return tp;
}


void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
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

tuple<VectorXd,MatrixXd> GaussFit(vector<VectorXd> const & samples){
  //fit d'une gaussienne multivariée sur un sample.
  int d=samples[0].size();
  VectorXd mean=VectorXd::Zero(d);
  MatrixXd SecondMoment=MatrixXd::Zero(d,d);
  for_each(samples.begin(),samples.end(),[&SecondMoment,&mean](VectorXd const &x)mutable{
    mean+=x;
    SecondMoment+=x*x.transpose();
  });
  mean/=samples.size();
  MatrixXd Var=SecondMoment/samples.size()-mean*mean.transpose();
  auto tp=make_tuple(mean,Var);
  return tp;
}

tuple<double,double> GaussFit(vector<double> const & samples){
  //fit d'une gaussienne multivariée sur un sample.
  double mean=0;
  double SecondMoment=0;
  for(int i=0;i<samples.size();i++){
    double x=samples[i];
    mean+=x;
    SecondMoment+=pow(x,2);
  }
  mean/=samples.size();
  double Var=SecondMoment/samples.size()-pow(mean,2);
  auto tp=make_tuple(mean,Var);
  return tp;
}

MatrixXd QQplot(vector<VectorXd> const & samples, default_random_engine & generator){
  //calcul d'un QQ plot. On rend une matrice avec samples.size() colonnes et autant de lignes que du nombre de quantiles choisi.
  //on met le tout dans un vector car je ne sais faire les QQplot qu'une dimension à la fois.
  int nquantiles=50; //on choisit de calculer 20 quantiles
  normal_distribution<double> distN(0,1);
  int ndim=samples[0].size();
  MatrixXd res(nquantiles,ndim+1);
  //tirage d'un échantillon de loi normale 1D
  vector<double> sample_normal(samples.size());
  transform(sample_normal.begin(),sample_normal.end(),sample_normal.begin(),[&generator,&distN](double d){
    return distN(generator);
  });
  sort(sample_normal.begin(),sample_normal.end());
  VectorXd quant_normal(nquantiles);
  for(int i=0;i<nquantiles;i++){
    double q=(i+0.5)/(1.0*nquantiles); // on ne prend ni le quantile 0 ni le quantile 100
    int n=q*sample_normal.size();
    quant_normal(i)=sample_normal[n];
  }
  res.col(0)=quant_normal;
  for(int j=0;j<ndim;j++){
    //création du sample réduit
    vector<double> sample_1D(samples.size()); for(int i=0;i<samples.size();i++){sample_1D[i]=samples[i](j);}
    //on centre, on réduit et on trie
    auto tpg=GaussFit(sample_1D);
    double m=get<0>(tpg);
    double s=sqrt(get<1>(tpg));
    transform(sample_1D.begin(),sample_1D.end(),sample_1D.begin(),[m,s](double x){
      double r=(x-m)/s;
      return r;
    });
    sort(sample_1D.begin(),sample_1D.end());
    VectorXd quant_1D(nquantiles);
    for(int i=0;i<nquantiles;i++){
      double q=(i+0.5)/(1.0*nquantiles); // on ne prend ni le quantile 0 ni le quantile 100
      int n=q*sample_1D.size();
      quant_1D(i)=sample_1D[n];
    }
    //on met les deux vecteurs de quantiles dans une même matrice. quantiles théoriques d'abord.
    MatrixXd M(nquantiles,2); M.col(0)=quant_normal; M.col(1)=quant_1D;
    res.col(j+1)=quant_1D;
  }
  return res;
}


const double Big = -1.e16;


int main(int argc, char **argv){
  if(argc != 3){
    cout << "Usage :\n";
    cout << "\t Number of obs, seed_obs\n";
    exit(0);
  }
  auto run_analysis=[](string foldname,int nobs,int seed){
    string endname=to_string(seed)+".gnu";
    int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
    int dim_theta=1;
    int dim_hpars=1;
    

    // Bornes des paramètres et hyperparamètres
    VectorXd lb_t(dim_theta); lb_t(0)=0; //0.2 avant
    VectorXd ub_t(dim_theta); ub_t(0)=1;

    //hpars. hpars(1) est forcément la sigma exp.
    VectorXd lb_hpars(dim_hpars); lb_hpars(0)=1e-4;
    VectorXd ub_hpars(dim_hpars); ub_hpars(0)=5;

    int samp_size=80; //80 avant
    VectorXd X_predictions(samp_size); for (int i=0;i<samp_size;i++){X_predictions(i)=double(i)/double(samp_size);}  
    int time_opt_opti=10; // temps pour optimisations opti
    int time_opt_koh_loc=20; // temps pour optimisation KOH
    
    default_random_engine generator(seed);
    //true function
    auto true_fct=[&generator](VectorXd const & xobs){
      vector<VectorXd> xobs_v(xobs.size());
      for(int i=0;i<xobs_v.size();i++){
        VectorXd X(1); X(0)=xobs(i) ; xobs_v[i]=X;
      }
      double theta=0.5;
      VectorXd hpars(2);
      hpars << 0.1,1e-4;
      GP gp(kernel);
      gp.SetGP(hpars);
      MatrixXd m=gp.SampleGP(xobs_v,1,generator); //j'espère 1 colonne
      //je vais faire mon sample moi-même ?
      VectorXd v=theta*xobs+m.col(0);
      return v;
    };

    //pour le nom de cas

    //pour la MCMC
    int nombre_steps_mcmc=1e5;
    int nombre_samples_collected=5e4;
    int nautocor=500;
    int nsel=nombre_samples_collected;

    VectorXd X_init(dim_theta+dim_hpars);
    X_init.head(dim_theta)=0.5*(lb_t+ub_t);
    X_init.tail(dim_hpars)=0.5*(lb_hpars+ub_hpars);
    MatrixXd COV_init=MatrixXd::Identity(dim_theta+dim_hpars,dim_theta+dim_hpars);
    COV_init(0,0)=pow(5e-2,2); //pour KOH separate : 1e-2 partout fonctionne bien.
    COV_init(1,1)=pow(5e-1,2);


    

    auto lambda_model=[](VectorXd const & Xprofile, VectorXd const & theta){
      //le vecteur Xprofile contient tous les x scalaires. Il faut renvoyer une prédiction de même taille que Xprofile.
      VectorXd v=theta(0)*Xprofile;
      return v;
    };

    auto logprior_pars=[](VectorXd const &p){
      return 0;
    };

    auto logprior_hpars=[](VectorXd const &h){
      return 0;
    };
    auto lambda_priormean=[](VectorXd const &X,VectorXd const &h){
      return VectorXd::Zero(X.size());
    };

  
    vector<AUGDATA> augdata; //remplir les données ici.

    VectorXd x(nobs);
    //construction du vecteur x des observations
    for(int i=0;i<nobs;i++){
      x(i)=i/(nobs*1.0);
    }
    VectorXd y=true_fct(x);
    AUGDATA a; a.SetX(x); a.SetValue(y);
    augdata.push_back(a);
    //calcul des samples bayes et de l'approximation gaussienne/
    DoE doe_init(lb_t,ub_t,200,10); // DoE Halton
    string fname_data=foldname+"obs"+endname;
    WriteObs(fname_data,augdata);

    Density Dens(doe_init);
    Dens.SetModel(lambda_model);
    Dens.SetKernel(kernel); //n'utilisons pas les dérivées pour ce cas.
    Dens.SetHparsBounds(lb_hpars,ub_hpars);
    Dens.SetLogPriorHpars(logprior_hpars);
    Dens.SetLogPriorPars(logprior_pars);
    Dens.SetPriorMean(lambda_priormean);



    Dens.SetDataExp(augdata);
    Dens.SetXprofile(augdata[0].GetX()); 


    //phase Full Bayes
    
      //on a écrit une fonction de MCMC. Il faut juste lui donner la fonction de scoring.
      auto scoring_function=[Dens,dim_theta,dim_hpars](VectorXd const & X){
        VectorXd theta(dim_theta); theta=X.head(dim_theta);
        VectorXd hpars(dim_hpars); hpars=X.tail(dim_hpars);
        return Dens.loglikelihood_theta(theta,hpars); //priors uniformes
      };

      auto in_bounds=[lb_hpars,ub_hpars,lb_t,ub_t,dim_theta,dim_hpars](VectorXd const &X){
        VectorXd theta=X.head(dim_theta);
        VectorXd hpars=X.tail(dim_hpars);
        for(int i=0;i<dim_theta;i++){
          if(theta(i)>ub_t(i) || theta(i)<lb_t(i)){
            return false;
          }
        }
        for(int i=0;i<dim_hpars;i++){
          if(hpars(i)>ub_hpars(i) || hpars(i)<lb_hpars(i)){
            return false;
          }
        }
        return true;
      };
      double lambda=pow(2.38,2)/2;
      double gamma=0.01;
      auto res=Run_MCMC_adapt(nombre_steps_mcmc,X_init,COV_init,scoring_function,in_bounds,lambda,gamma,generator); //tuple qui contient tous les samples et toutes les données.
      //sélection des samples 
      tuple<vector<VectorXd>,vector<double>> selected_res;
      
      vector<VectorXd> selected_samples;
      vector<double> selected_ll;
      //sélection uniforme
      /*
      for(int i=0;i<nsel;i++){
        int id=i*get<0>(res).size()/nsel;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      */
      //sélection des derniers samples
      for(int i=0;i<nsel;i++){
        int id=get<0>(res).size()-1-i;
        selected_samples.push_back(get<0>(res)[id]);
        selected_ll.push_back(get<1>(res)[id]);
      }
      
      vector<VectorXd> sel_thetas(nsel);
      vector<VectorXd> sel_hpars(nsel);  
      for(int i=0;i<nsel;i++){
        VectorXd X=selected_samples[i];
        sel_thetas[i]=X.head(dim_theta);
        sel_hpars[i]=X.tail(dim_hpars);
      }
      //on met dans la densité
      Dens.SetNewAllSamples(get<0>(res));
      Dens.SetNewSamples(sel_thetas);
      Dens.SetNewHparsOfSamples(sel_hpars);
      //vérification du mélange de la MCMC

      string fname=foldname+"autocor"+endname;
      Dens.Autocor_diagnosis(nautocor,fname);

      // écriture des samples et aussi quelques prédictions pour test visuel
      Dens.WriteSamples(foldname+"samples"+endname);
      Dens.WritePredictions(X_predictions,foldname+"preds"+endname);

      //fit d'une gaussienne
      auto tp_gauss=GaussFit(selected_samples);
      cout << "gaussfit : " << get<0>(tp_gauss).transpose() << endl;
      cout << get<1>(tp_gauss) << endl;

      //écriture dans un fichier
      fname=foldname+"gaussfit"+endname;
      ofstream ofile(fname);
      ofile << " moyenne gaussienne : " << endl << get<0>(tp_gauss).transpose() << endl;
      ofile << "matrice de variance : " << endl << get<1>(tp_gauss) << endl;

      ofile.close();

      //calcul des quantiles normalisés et affichage dans un fichier
      MatrixXd Quant=QQplot(selected_samples,generator);
      fname=foldname+"qqplot"+endname;
      ofile.open(fname);
      for(int i=0;i<Quant.rows();i++){
        for(int j=0;j<Quant.cols();j++){
          ofile << Quant(i,j) << " ";
        }
        ofile << endl;
      }
      ofile.close();
    
    return make_tuple(get<0>(tp_gauss),get<1>(tp_gauss)); //renvoie les params du fit gaussien sur les samples.
  };
  string foldname="results/obs80/";
  list<int> lseeds={42,654,332,465,221};
  double tmean=0;
  double tstd=0;
  double smean=0;
  double sstd=0;

  for(int i:lseeds){
    auto tp=run_analysis(foldname,80,i);
    tmean+=get<0>(tp)(0);
    smean+=get<0>(tp)(1);
    tstd+=get<1>(tp)(0,0);
    sstd+=get<1>(tp)(1,1);
  }
  tmean/=lseeds.size();
  smean/=lseeds.size();
  tstd/=lseeds.size();
  sstd/=lseeds.size();
  cout << " t : " << tmean << " " << sqrt(tstd) << endl;
  cout << " s : " << smean << " " << sqrt(sstd) << endl;
  //run_analysis doit rendre la moyenne de theta et sa variance. On affiche ensuite ça en fct des observations.

  exit(0);
/*
  //phase KOH
  {
    VectorXd hparskoh=Dens.HparsKOH(0.5*(lb_hpars+ub_hpars),time_opt_koh_loc);
    cout << "hpars koh : " << hparskoh.transpose() << endl;
    VectorXd X_init_theta=X_init.head(dim_theta);
    MatrixXd COV_init_theta=COV_init.topLeftCorner(dim_theta,dim_theta);
    Dens.Run_MCMC_fixed_hpars(nombre_steps_mcmc,1,X_init_theta,COV_init_theta,hparskoh,generator);
    vector<VectorXd> allsamples_koh=Dens.GetAllSamples();
    int s=allsamples_koh.size();
    vector<VectorXd> selected_samples;
    //sélection des derniers samples
    for(int i=0;i<nsel;i++){
      int id=allsamples_koh.size()-1-i;
      selected_samples.push_back(allsamples_koh[id]);
    }
    
    vector<VectorXd> sel_thetas(nsel);
    vector<VectorXd> sel_hpars(nsel);  
    for(int i=0;i<nsel;i++){
      VectorXd X=selected_samples[i];
      sel_thetas[i]=X.head(dim_theta);
      sel_hpars[i]=hparskoh;
    }
    //on met dans la densité
    Dens.SetNewSamples(sel_thetas);
    Dens.SetNewHparsOfSamples(sel_hpars);
    //vérification du mélange de la MCMC

    string fname=foldname+"autocorkoh.gnu";
    Dens.Autocor_diagnosis(nautocor,fname);

    // écriture des samples et aussi quelques prédictions pour test visuel
    Dens.WriteSamples(foldname+"sampleskoh.gnu");
    Dens.WritePredictions(X_predictions,foldname+"predskoh.gnu");

    //fit d'une gaussienne
    auto tp_gauss=GaussFit(selected_samples);
    cout << "gaussfit : " << get<0>(tp_gauss).transpose() << endl;
    cout << get<1>(tp_gauss) << endl;

    //écriture dans un fichier
    fname=foldname+"gaussfitkoh.gnu";
    ofstream ofile(fname);
    ofile << " moyenne gaussienne : " << endl << get<0>(tp_gauss).transpose() << endl;
    ofile << "matrice de variance : " << endl << get<1>(tp_gauss) << endl;

    ofile.close();
  }
  //phase OPTI
  {
    cout << "début phase opti" << endl;
    DensityOpt DensOpt(Dens);
    DensOpt.BuildHGPs_noPCA();
    cout << "hpars koh : " << hparskoh.transpose() << endl;
    VectorXd X_init_theta=X_init.head(dim_theta);
    MatrixXd COV_init_theta=COV_init.topLeftCorner(dim_theta,dim_theta);
    Dens.Run_MCMC_fixed_hpars(nombre_steps_mcmc,1,X_init_theta,COV_init_theta,hparskoh,generator);
    vector<VectorXd> allsamples_koh=Dens.GetAllSamples();
    int s=allsamples_koh.size();
    vector<VectorXd> selected_samples;
    //sélection des derniers samples
    for(int i=0;i<nsel;i++){
      int id=allsamples_koh.size()-1-i;
      selected_samples.push_back(allsamples_koh[id]);
    }
    
    vector<VectorXd> sel_thetas(nsel);
    vector<VectorXd> sel_hpars(nsel);  
    for(int i=0;i<nsel;i++){
      VectorXd X=selected_samples[i];
      sel_thetas[i]=X.head(dim_theta);
      sel_hpars[i]=hparskoh;
    }
    //on met dans la densité
    Dens.SetNewSamples(sel_thetas);
    Dens.SetNewHparsOfSamples(sel_hpars);
    //vérification du mélange de la MCMC

    string fname=foldname+"autocorkoh.gnu";
    Dens.Autocor_diagnosis(nautocor,fname);

    // écriture des samples et aussi quelques prédictions pour test visuel
    Dens.WriteSamples(foldname+"sampleskoh.gnu");
    Dens.WritePredictions(X_predictions,foldname+"predskoh.gnu");

    //fit d'une gaussienne
    auto tp_gauss=GaussFit(selected_samples);
    cout << "gaussfit : " << get<0>(tp_gauss).transpose() << endl;
    cout << get<1>(tp_gauss) << endl;

    //écriture dans un fichier
    fname=foldname+"gaussfitkoh.gnu";
    ofstream ofile(fname);
    ofile << " moyenne gaussienne : " << endl << get<0>(tp_gauss).transpose() << endl;
    ofile << "matrice de variance : " << endl << get<1>(tp_gauss) << endl;

    ofile.close();

  }
*/
 
}

