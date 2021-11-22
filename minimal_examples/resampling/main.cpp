//exemple pour resampling (chapitre 4/5, exemple 1). On reprend l'exemple du premier article et on fait les 3 méthodes KOH Opti Bayes. Puis on resample les échantillons Opti pour voir si la postérieure obtenue est meilleure.


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
  double d=x(0)-y(0);
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(2),2)); //3/2
}

double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential

  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2)))*exp(-d/hpar(2)); //3/2
}


double kernel(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double l=0.1;
  double d=abs(x(0)-y(0));
  return pow(hpar(0),2)*exp(-0.5*pow(d/l,2)); //3/2
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
  cor*=(1+abs(x(0)-y(0))/hpar(2))*exp(-abs(x(0)-y(0))/hpar(2)); //lcor en theta
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


void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
}

void PrintVector(vector<VectorXd> &X, vector<double> &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values[i]);
  }
  fclose(out);
}

void PrintVector(string filename,vector<VectorXd> &X){
  ofstream ofile(filename);
  for(int i=0;i<X.size();i++){
    for(int j=0;j<X[i].size();j++){
      ofile << X[i](j) << " ";
    }
    ofile << endl;
  }
  ofile.close();
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

tuple<vector<VectorXd>,vector<VectorXd>> ReadSamples(string const & filename){
  //écrit pour la dimension 5 et 3 hpars pour kernel_z. attention à ce que m_dim_pars et m_dim_hpars soient bien sélectionnées.
  vector<VectorXd> samples;
  vector<VectorXd> hparsv;
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      
      VectorXd theta(1);
      VectorXd hpars(3);
      for(int i=0;i<1;i++){
        theta(i)=stod(words[i]);
      }
      for(int i=0;i<3;i++){
        hpars(i)=stod(words[i+1]); //c'est bien pars ici.
      }
      samples.push_back(theta);
      hparsv.push_back(hpars);
    }
    cout << "number of samples loaded : " << samples.size() <<"."<<endl;
  }
  else{
    cerr << "empty file" << endl;
  }
  ifile.close();
  return make_tuple(samples,hparsv);
}


tuple<vector<VectorXd>,vector<double>> ReadWeights(string const & filename){
  //écrit pour la dimension 5 et 3 hpars pour kernel_z. attention à ce que m_dim_pars et m_dim_hpars soient bien sélectionnées.
  vector<VectorXd> samples;
  vector<double> hparsv;
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>());
      
      VectorXd theta(1);
      double hpars;
      for(int i=0;i<1;i++){
        theta(i)=stod(words[i]);
      }
      for(int i=0;i<1;i++){
        hpars=stod(words[i+1]); //c'est bien pars ici.
      }
      samples.push_back(theta);
      hparsv.push_back(hpars);
    }
    cout << "number of samples loaded : " << samples.size() <<"."<<endl;
  }
  else{
    cerr << "empty file" << endl;
  }
  ifile.close();
  return make_tuple(samples,hparsv);
}



const double Big = -1.e16;


int main(int argc, char **argv){

  int seed_obs=111111;//5546243; 111111 marche de ouf.
  int seed_mcmc= 1111111;  // results1 6842244;
  default_random_engine generator(seed_obs);

  //paramètres MCMC
    int dim_x=1; // On considère que x vit seulement dans [0,1]. A normaliser si jamais on va en plus grande dimension.
    int dim_theta=1;
    int dim_hpars=3;
    

    // Bornes des paramètres et hyperparamètres
    VectorXd lb_t(dim_theta); lb_t(0)=-0.5; //0.2 avant
    VectorXd ub_t(dim_theta); ub_t(0)=1.5;
    //hpars. hpars(1) est forcément la sigma exp.
    VectorXd lb_hpars(dim_hpars); lb_hpars(0)=1e-3;lb_hpars(1)=1e-3;lb_hpars(2)=0.1; // edm, exp, lcor
    VectorXd ub_hpars(dim_hpars); ub_hpars(0)=1;ub_hpars(1)=1;ub_hpars(2)=5;


    int nombre_steps_mcmc=3e6;
    int nombre_samples_collected=1e4;
    int nautocor=2000;
    int nsel=nombre_samples_collected;




    VectorXd X_init_theta(dim_theta);
    X_init_theta(0)=0;
    MatrixXd COV_init_theta=MatrixXd::Identity(1,1);
    COV_init_theta(0,0)=pow(3e-1,2);

    VectorXd X_init_bayes(dim_theta+dim_hpars);
    X_init_bayes(0)=0.5;
    X_init_bayes(1)=0.3;
    X_init_bayes(2)=1e-2;
    X_init_bayes(3)=0.3;
    MatrixXd COV_init_bayes=MatrixXd::Identity(dim_theta+dim_hpars,dim_theta+dim_hpars);
    COV_init_bayes(0,0)=pow(1e-1,2);
    COV_init_bayes(1,1)=pow(5e-2,2);
    COV_init_bayes(2,2)=pow(5e-3,2);
    COV_init_bayes(3,3)=pow(5e-2,2);    



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
      return v;
    };
    auto lambda_priormean=[](VectorXd const &X,VectorXd const &h){
      return VectorXd::Zero(X.size());
    };

    auto in_bounds_theta=[lb_hpars,ub_hpars,lb_t,ub_t,dim_theta,dim_hpars](VectorXd const &X){
        VectorXd theta=X.head(dim_theta);    
        for(int i=0;i<dim_theta;i++){
          if(theta(i)>ub_t(i) || theta(i)<lb_t(i)){
            return false;
          }
        }
        return true;
      };
auto in_bounds_bayes=[lb_hpars,ub_hpars,lb_t,ub_t,dim_theta,dim_hpars](VectorXd const &X){
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

  
    vector<AUGDATA> augdata; //remplir les données ici.

    
    //construction des observations
    int nobs=8;
    VectorXd x(nobs);
    for(int i=0;i<nobs;i++){
      x(i)=(i+1)/(nobs+1.0);
    }
    VectorXd y=lambda_truefct(x);
    AUGDATA a; a.SetX(x); a.SetValue(y);
    augdata.push_back(a);
    //calcul des samples/
    DoE doe_init(lb_t,ub_t,300,10); // DoE Halton
    string foldname="results/";
    WriteObs(foldname+"obs.gnu",augdata);

    Density Dens(doe_init);
    Dens.SetModel(lambda_model);
    Dens.SetKernel(Kernel_Z_SE); //n'utilisons pas les dérivées pour ce cas. si, on va mettre les dérivées.
    Dens.SetHparsBounds(lb_hpars,ub_hpars);
    Dens.SetLogPriorHpars(logprior_hpars);
    Dens.SetLogPriorPars(logprior_pars);
    Dens.SetPriorMean(lambda_priormean);
    Dens.SetDataExp(augdata);
    Dens.SetXprofile(augdata[0].GetX()); 


    //KOH
    /*
    VectorXd hpars_koh=0.5*(lb_hpars+ub_hpars);
    hpars_koh=Dens.HparsKOH(hpars_koh,120);
    cout << "hpars koh : " << hpars_koh.transpose() << endl;
    auto scoring_function_koh=[Dens,hpars_koh](VectorXd const & theta){
        return Dens.loglikelihood_theta(theta,hpars_koh); //priors uniformes sur theta
    };
    generator.seed(seed_mcmc);
    auto res_koh=Run_MCMC(nombre_steps_mcmc,X_init_theta,COV_init_theta,scoring_function_koh,in_bounds_theta,generator);
    vector<VectorXd> allsamples_koh=get<0>(res_koh);
    vector<VectorXd> selectedsamples_koh(nombre_samples_collected);
    vector<VectorXd> selectedhpars_koh(nombre_samples_collected);
    for(int i=0;i<nombre_samples_collected;i++){
      int bigind=i*(allsamples_koh.size()/selectedsamples_koh.size());
      selectedsamples_koh[i]=allsamples_koh[bigind];
      selectedhpars_koh[i]=hpars_koh;
    }

    Dens.SetNewAllSamples(allsamples_koh);
    Dens.SetNewSamples(selectedsamples_koh);
    Dens.SetNewHparsOfSamples(selectedhpars_koh);
    Dens.Autocor_diagnosis(nautocor,foldname+"autocorkoh.gnu");
    Dens.WriteSamples(foldname+"sampkoh.gnu");
    

    //Bayes
    auto scoring_function_bayes=[Dens,dim_theta,dim_hpars,logprior_hpars](VectorXd const & X){
        VectorXd theta(dim_theta); theta=X.head(dim_theta);
        VectorXd hpars(dim_hpars); hpars=X.tail(dim_hpars);
        return Dens.loglikelihood_theta(theta,hpars)+logprior_hpars(hpars); //priors uniformes
    };

    generator.seed(seed_mcmc);
    auto res_bayes=Run_MCMC(3*nombre_steps_mcmc,X_init_bayes,COV_init_bayes,scoring_function_bayes,in_bounds_bayes,generator);
    vector<VectorXd> allsamples_bayes(3*nombre_steps_mcmc);
    vector<VectorXd> selectedsamples_bayes(nombre_samples_collected);
    vector<VectorXd> selectedhpars_bayes(nombre_samples_collected);
    for(int i=0;i<3*nombre_steps_mcmc;i++){
      allsamples_bayes[i]=get<0>(res_bayes)[i];
    }
    for(int i=0;i<nombre_samples_collected;i++){
      int bigind=i*(allsamples_bayes.size()/selectedsamples_bayes.size());
      selectedsamples_bayes[i]=allsamples_bayes[bigind];
      selectedhpars_bayes[i]=get<0>(res_bayes)[bigind].tail(dim_hpars);
    }
    Dens.SetNewAllSamples(allsamples_bayes);
    Dens.SetNewSamples(selectedsamples_bayes);
    Dens.SetNewHparsOfSamples(selectedhpars_bayes);
    Dens.Autocor_diagnosis(nautocor,foldname+"autocorbayes.gnu");
    Dens.WriteSamples(foldname+"sampbayes.gnu");

    */

    //Opt
    MatrixXd Bounds_hpars_gp(2,3); //taille (2;nhpars_gp)
    Bounds_hpars_gp(0,0)=1e-3; Bounds_hpars_gp(1,0)=1e2; //variance
    Bounds_hpars_gp(0,1)=1e-3; Bounds_hpars_gp(1,1)=2e-3; //nugget
    Bounds_hpars_gp(0,2)=1e-3; Bounds_hpars_gp(1,2)=5; //lcor en theta
    VectorXd hpars_gp_guess(3);
    hpars_gp_guess=0.5*(Bounds_hpars_gp.row(0)+Bounds_hpars_gp.row(1)).transpose(); 


    //les surrogates hpars sont construits sur le doe_init.
    DensityOpt DensOpt(Dens);
    VectorXd hpars_guess(3);
    hpars_guess << 0.1,4e-3,0.25;
  
   // DensOpt.Compute_optimal_hpars(1); //2 secondes par optimisation
   // DensOpt.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess); //hpars des surrogates pour hpars !
   // DensOpt.opti_allgps(hpars_gp_guess);
   // DensOpt.WritehGPs(foldname+"hGPs.gnu");
    //DensOpt.Test_hGPs(100,2);
    auto scoring_function_opt=[DensOpt](VectorXd const & theta){
        VectorXd hpars=DensOpt.EvaluateHparOpt(theta);
        return DensOpt.loglikelihood_theta(theta,hpars); //priors uniformes
    };

    auto scoring_function_opt_expensive=[DensOpt,hpars_guess](VectorXd const & theta){
        VectorXd hpars=DensOpt.HparsOpt(theta,hpars_guess,1e-4);
        return DensOpt.loglikelihood_theta(theta,hpars); //priors uniformes
    };
    generator.seed(seed_mcmc);
    /*
    auto res_opt=Run_MCMC(nombre_steps_mcmc,X_init_theta,COV_init_theta,scoring_function_opt_expensive,in_bounds_theta,generator);
    vector<VectorXd> allsamples_opt=get<0>(res_opt);
    vector<VectorXd> selectedsamples_opt(nombre_samples_collected);
    vector<VectorXd> selectedhpars_opt(nombre_samples_collected);
    
    for(int i=0;i<nombre_samples_collected;i++){
      int bigind=i*(allsamples_opt.size()/selectedsamples_opt.size());
      selectedsamples_opt[i]=allsamples_opt[bigind];
      selectedhpars_opt[i]=DensOpt.HparsOpt(selectedsamples_opt[i],hpars_guess,1e-2);
    }
    DensOpt.SetNewAllSamples(allsamples_opt);
    DensOpt.SetNewSamples(selectedsamples_opt);
    DensOpt.SetNewHparsOfSamples(selectedhpars_opt);
    DensOpt.Autocor_diagnosis(nautocor,foldname+"autocoropt.gnu");
    DensOpt.WriteSamples(foldname+"sampopt.gnu");
    */

   //lecture
   auto tp=ReadSamples(foldname+"sampopt.gnu");
   auto selectedsamples_opt=get<0>(tp);
   auto selectedhpars_opt=get<1>(tp);


    cout << "calcul des poids..." << endl;
      int number_new_samples=250;
      /*
      vector<double> weights(selectedsamples_opt.size());
      for(int i=0;i<weights.size();i++){
        double w=0;
        for(int j=0;j<selectedsamples_opt.size();j++){
          w+=exp(DensOpt.loglikelihood_theta(selectedsamples_opt[i],selectedhpars_opt[j]));
        }
        w/=exp(DensOpt.loglikelihood_theta(selectedsamples_opt[i],selectedhpars_opt[i]));
        weights[i]=w;
      }
      //afficher les weights dans un fichier
    PrintVector(selectedsamples_opt,weights,"results/weights.gnu");
    */
    vector<double> weights=get<1>(ReadWeights(foldname+"weights.gnu"));

    auto resampling=[selectedsamples_opt,DensOpt,selectedhpars_opt,&generator,number_new_samples,weights](string filename){
      //étape de resampling
      auto weights2=weights;
      vector<VectorXd> resampled(number_new_samples);
      for(int i=0; i<number_new_samples; ++i){
          std::discrete_distribution<int> distribution(weights2.begin(), weights2.end()); 
          int number = distribution(generator);
          weights2[number] = 0; // the weight associates to the sampled value is set to 0
          resampled[i]=selectedsamples_opt[number];
      }
      //afficher les nouveaux samples maintenant
      PrintVector(filename,resampled);
  };
  /*
  //on fait plusieurs tirages de poids pour montrer la variabilité selon l'aléatoire.
  //faisons-en 5.
  for(int i=0;i<5;i++){
    string fname=foldname+"resamples"+to_string(i)+".gnu";
    resampling(fname);
  }
  */

  //écriture d'hyperparamètres optimaux pour jolies figures.

  vector<VectorXd> opthpars(200);
  for(int i=0;i<200;i++){
    VectorXd t(1);
    t(0)=-0.5+2*(i*1.0)/200;
    VectorXd h=DensOpt.HparsOpt(t,hpars_guess,3);
    VectorXd res(4);
    res << t(0),h(0),h(1),h(2);
    opthpars[i]=res;
  }
  PrintVector(foldname+"hparsoptfine.gnu",opthpars);

  exit(0);
  //on calcule un resample
  {
    auto weights2=weights;
      vector<VectorXd> resampled(number_new_samples);
      for(int i=0; i<number_new_samples; ++i){
          std::discrete_distribution<int> distribution(weights2.begin(), weights2.end()); 
          int number = distribution(generator);
          weights2[number] = 0; // the weight associates to the sampled value is set to 0
          resampled[i]=selectedsamples_opt[number];
    }
    //on calcule les hpars optimaux sur ce resample
    vector<VectorXd> resampled_hpars(number_new_samples);
    for(int i=0;i<number_new_samples;i++){
      resampled_hpars[i]=DensOpt.HparsOpt(resampled[i],hpars_guess,1);
    }
    //on calcule les poids sur ce resample. Attention. Le dénominateur ne peut plus être loglikelihood theta car ce n'est plus la densité dont est originaire le point. Il faut reconstruire un estimateur de cette densité à partir de l'échantillon resampled.
    //KDE de la nouvelle densité
    double width=0.1;
    auto kde=[width,resampled](VectorXd const & theta){
      auto k=[](double x){
        return (1/sqrt(2*M_PI))*exp(-0.5*pow(x,2));
      };
      double res=0;
      for(int i=0;i<resampled.size();i++){
        res+=k((resampled[i](0)-theta(0))/width);
      }
      res/=width*resampled.size();
      return res;
    };
    


    vector<double> weightsnew(resampled.size());
      for(int i=0;i<weightsnew.size();i++){
        double w=0;
        for(int j=0;j<resampled.size();j++){
          w+=exp(DensOpt.loglikelihood_theta(resampled[i],resampled_hpars[j]));
        }
        w/=kde(resampled[i]);
        weightsnew[i]=w;
      }
      //afficher les weights dans un fichier
PrintVector(resampled,weightsnew,"results/weightsnew.gnu");

//afficher que le KDE est bon
vector<double> kde_values(resampled.size());
for(int i=0;i<kde_values.size();i++){
  kde_values[i]=kde(resampled[i]);
}
PrintVector(resampled,kde_values,"results/kde.gnu");
  }
  exit(0); 
}

