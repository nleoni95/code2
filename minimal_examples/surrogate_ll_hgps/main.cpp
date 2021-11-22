//exemple minimal pour coder le refinement adaptatif de surrogate, avec intelligent sampling (MCMCs, sélection de points,...)
//donc le densities.ccp et le densities.h sont améliorés.

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

void PrintVector(string filename,vector<VectorXd> &X,vector<VectorXd> &Y){
  ofstream ofile(filename);
  for(int i=0;i<X.size();i++){
    for(int j=0;j<X[i].size();j++){
      ofile << X[i](j) << " ";
    }
    for(int j=0;j<Y[i].size();j++){
      ofile << Y[i](j) << " ";
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

double myoptfct_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
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
    VectorXd lb_hpars(dim_hpars); lb_hpars(0)=1e-3;lb_hpars(1)=1e-7;lb_hpars(2)=0.1; // edm, exp, lcor
    VectorXd ub_hpars(dim_hpars); ub_hpars(0)=1;ub_hpars(1)=1;ub_hpars(2)=5;


    int nombre_steps_mcmc=1e5;
    int nombre_samples_collected=200;
    int nautocor=2000;
    int nsel=nombre_samples_collected;




    VectorXd X_init_theta(dim_theta);
    X_init_theta(0)=0;
    MatrixXd COV_init_theta=MatrixXd::Identity(1,1);
    COV_init_theta(0,0)=pow(3e-1,2);

   



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
return 0;
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


  
    vector<AUGDATA> augdata; //remplir les données ici.

    
    //construction des observations
    int nobs=8;
    VectorXd x(nobs);
    for(int i=0;i<nobs;i++){
      x(i)=(i+1)/(nobs+1.0);
    }
    VectorXd Xpred(50);
    for(int i=0;i<Xpred.size();i++){
      Xpred(i)=(i+1)/(Xpred.size()+1.0);
    }
    VectorXd y=lambda_truefct(x);
    AUGDATA a; a.SetX(x); a.SetValue(y);
    augdata.push_back(a);
    //calcul des samples. DOE initial de 8 points/
    DoE doe_init(lb_t,ub_t,10,666); // DoE Halton. indice de départ = 10 dans les version précédentes.
    
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

    //Opt
    MatrixXd Bounds_hpars_gp(2,3); //taille (2;nhpars_gp)
    Bounds_hpars_gp(0,0)=1e-3; Bounds_hpars_gp(1,0)=1e1; //variance
    Bounds_hpars_gp(0,1)=1e-3; Bounds_hpars_gp(1,1)=1e1; //nugget
    Bounds_hpars_gp(0,2)=1e-3; Bounds_hpars_gp(1,2)=5; //lcor en theta
    VectorXd hpars_gp_guess(3);
    hpars_gp_guess << 1e-3,1e-3,0.25;

    DensityOpt DensOpt(Dens);
    VectorXd hpars_guess_z(3);
    hpars_guess_z << 0.1,1e-3,0.25;

    //phase construction du surrogate hGPs
    //time
    auto t1=chrono::steady_clock::now();
    //évaluation des hpars
  
    DensOpt.Compute_optimal_hpars(1); //1 seconde par optimisation
    DensOpt.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess); //hpars des surrogates pour hpars !
    DensOpt.opti_allgps(hpars_gp_guess);
    //time
    auto t2=chrono::steady_clock::now();
    //MCMC

    auto tp=DensOpt.intel_Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,X_init_theta,COV_init_theta,generator);
    DensOpt.intel_autocor_diagnosis(get<0>(tp),2000,foldname+"autocor.gnu");
    auto sample=get<1>(tp);
    auto v_hgps=DensOpt.GethGPs();
    for(int i=0;i<sample.size();i++){
      auto vvar=DensOpt.intel_updated_var(v_hgps,sample,sample[i]);
      //cout << vvar[0] << " " <<vvar[1] << " " <<vvar[2] << endl;
      double score=std::accumulate(vvar.begin(),vvar.end(),0.0)/vvar.size();
      cout << sample[i](0) << ", " << score << endl;
    }

    auto t3=chrono::steady_clock::now();

    cout << "temps total " <<  chrono::duration_cast<chrono::seconds>(t3-t2).count() << " s" << endl;

    exit(0);
}

