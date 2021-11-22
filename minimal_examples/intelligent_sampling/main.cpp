//exemple pour surrogate ll ou hgps. dans l'exemple, on fait tourner l'exemple 1 avec Opti full (très peu de temps pour chaque optimisation), surrogate hGPs et surrogate ll. On compare les temps de calcul et les samples finaux. Il faut également rovuer un moyen de comparer la précision des surrogates sur la loglik.
//but de cet exemple : montrer qu'on gagne du temps avec les surrogates, sans perdre en précision. Sur la comparaison ll/hgps, on veut montrer qu'on a une précision à peu près pareil.


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
    VectorXd lb_hpars(dim_hpars); lb_hpars(0)=1e-3;lb_hpars(1)=1e-3;lb_hpars(2)=0.1; // edm, exp, lcor
    VectorXd ub_hpars(dim_hpars); ub_hpars(0)=1;ub_hpars(1)=1;ub_hpars(2)=5;


    int nombre_steps_mcmc=3e6;
    int nombre_samples_collected=5e3;
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
    //calcul des samples/
    DoE doe_init(lb_t,ub_t,50,666); // DoE Halton. indice de départ = 10 dans les version précédentes.
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
    Bounds_hpars_gp(0,0)=1e-3; Bounds_hpars_gp(1,0)=1e2; //variance
    Bounds_hpars_gp(0,1)=1e-3; Bounds_hpars_gp(1,1)=2e-3; //nugget
    Bounds_hpars_gp(0,2)=1e-3; Bounds_hpars_gp(1,2)=5; //lcor en theta
    VectorXd hpars_gp_guess(3);
    hpars_gp_guess=0.5*(Bounds_hpars_gp.row(0)+Bounds_hpars_gp.row(1)).transpose(); 

    DensityOpt DensOpt(Dens);
    VectorXd hpars_guess_z(3);
    hpars_guess_z << 0.1,4e-3,0.25;

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

   //DensOpt.WritehGPs(foldname+"hGPs.gnu");
    //DensOpt.Test_hGPs(100,2);
    auto scoring_function_opt=[DensOpt](VectorXd const & theta){
        VectorXd hpars=DensOpt.EvaluateHparOpt(theta);
        return DensOpt.loglikelihood_theta(theta,hpars); //priors uniformes
    };
    generator.seed(seed_mcmc);
    auto res_hgps=Run_MCMC(nombre_steps_mcmc,X_init_theta,COV_init_theta,scoring_function_opt,in_bounds_theta,generator);
    //time
    auto t3=chrono::steady_clock::now();

    //stockage des samples (timé en tant que phase de prédictions)
    vector<VectorXd> allsamples_hgps=get<0>(res_hgps);
    vector<VectorXd> selectedsamples_hgps(nombre_samples_collected);
    vector<VectorXd> selectedhpars_hgps(nombre_samples_collected);
    
    for(int i=0;i<nombre_samples_collected;i++){
      int bigind=i*(allsamples_hgps.size()/selectedsamples_hgps.size());
      selectedsamples_hgps[i]=allsamples_hgps[bigind];
      selectedhpars_hgps[i]=DensOpt.HparsOpt(selectedsamples_hgps[i],hpars_guess_z,1e-4);
    }

    DensOpt.SetNewAllSamples(allsamples_hgps);
    DensOpt.SetNewSamples(selectedsamples_hgps);
    DensOpt.SetNewHparsOfSamples(selectedhpars_hgps);
    DensOpt.Autocor_diagnosis(nautocor,foldname+"autocorhgps.gnu");
    DensOpt.WriteSamples(foldname+"samphgps.gnu");
    DensOpt.WritePredictions(Xpred,foldname+"predshgps.gnu");

    auto t3bis=chrono::steady_clock::now();

    double temps1=chrono::duration_cast<chrono::seconds>(t2-t1).count();
    double temps2=chrono::duration_cast<chrono::seconds>(t3-t2).count();
    double temps3=chrono::duration_cast<chrono::seconds>(t3bis-t3).count();
    cout << "temps hgps modeleval+buildgp " <<  chrono::duration_cast<chrono::seconds>(t2-t1).count() << " s" << endl;
    cout << "temps hgps mcmc " <<  chrono::duration_cast<chrono::seconds>(t3-t2).count() << " s" << endl;
    cout << "temps preds hgps " <<  chrono::duration_cast<chrono::seconds>(t3bis-t3).count() << " s" << endl;




    //phase construction d'un surrogate pour la log-likelihood.

    //on prend les mêmes points que le surrogate hGPs
    //time
    auto t4=chrono::steady_clock::now();
    //évaluations de la loglikelihood
    vector<VectorXd> thetas_doe=doe_init.GetGrid();
    vector<double> ll_points(thetas_doe.size());
    for(int i=0;i<ll_points.size();i++){
      VectorXd h=DensOpt.HparsOpt(thetas_doe[i],hpars_guess_z,1);
      ll_points[i]=DensOpt.loglikelihood_theta(thetas_doe[i],h);
    }
    //construction du GP
    vector<DATA> data_ll(thetas_doe.size());
    for(int i=0;i<data_ll.size();i++){
      DATA dat; dat.SetX(thetas_doe[i]); dat.SetValue(ll_points[i]);
      data_ll[i]=dat;
    }
    GP gp_ll(Kernel_GP_Matern32);
    gp_ll.SetData(data_ll);
    gp_ll.SetGP(hpars_gp_guess);
    gp_ll.OptimizeGP(myoptfct_gp,&Bounds_hpars_gp,&hpars_gp_guess,hpars_gp_guess.size());
    hpars_gp_guess=gp_ll.GetPar();
    cout  << "par after opt : " << hpars_gp_guess.transpose() << endl;
    //time
    auto t5=chrono::steady_clock::now();
    //MCMC
    auto scoring_function_hll=[DensOpt,gp_ll](VectorXd const & theta){
      return gp_ll.EvalMean(theta);
    };
    generator.seed(seed_mcmc);
    auto res_hll=Run_MCMC(nombre_steps_mcmc,X_init_theta,COV_init_theta,scoring_function_hll,in_bounds_theta,generator);
    //time
    auto t6=chrono::steady_clock::now();

    //traitement des samples

    vector<VectorXd> allsamples_hll=get<0>(res_hll);
    vector<VectorXd> selectedsamples_hll(nombre_samples_collected);
    vector<VectorXd> selectedhpars_hll(nombre_samples_collected);
    
    for(int i=0;i<nombre_samples_collected;i++){
      int bigind=i*(allsamples_hll.size()/selectedsamples_hll.size());
      selectedsamples_hll[i]=allsamples_hll[bigind];
      selectedhpars_hll[i]=DensOpt.HparsOpt(selectedsamples_hll[i],hpars_guess_z,1e-4);
    }

    DensOpt.SetNewAllSamples(allsamples_hll);
    DensOpt.SetNewSamples(selectedsamples_hll);
    DensOpt.SetNewHparsOfSamples(selectedhpars_hll);
    DensOpt.Autocor_diagnosis(nautocor,foldname+"autocorhll.gnu");
    DensOpt.WriteSamples(foldname+"samphll.gnu");
    DensOpt.WritePredictions(Xpred,foldname+"predshll.gnu");


    auto t6bis=chrono::steady_clock::now();

    double temps4=chrono::duration_cast<chrono::seconds>(t5-t4).count();
    double temps5=chrono::duration_cast<chrono::seconds>(t6-t5).count();
    double temps6=chrono::duration_cast<chrono::seconds>(t6bis-t6).count();

    cout << "temps gpll modeleval+buildgp " <<  temps4 << " s" << endl;
    cout << "temps gpll mcmc " <<  temps5 << " s" << endl;
    cout << "temps gpll preds " <<  temps6 << " s" << endl;



    //phase de la méthode Opt complète
    //time

    auto t7=chrono::steady_clock::now();
    //MCMC
    auto scoring_function_optfull=[DensOpt,hpars_guess_z](VectorXd const & theta){
        VectorXd hpars=DensOpt.HparsOpt(theta,hpars_guess_z,1e-4);
        return DensOpt.loglikelihood_theta(theta,hpars); //priors uniformes
    };
    //auto res_optfull=Run_MCMC(nombre_steps_mcmc,X_init_theta,COV_init_theta,scoring_function_optfull,in_bounds_theta,generator);
    //time

    auto t8=chrono::steady_clock::now();
    //phase non mesurée du traitement des samples
    
    auto allsamples_optfull=ReadSamples(foldname+"sampoptfullref.gnu");
    vector<VectorXd> selectedsamples_optfull=get<0>(allsamples_optfull);
    vector<VectorXd> selectedhpars_optfull=get<1>(allsamples_optfull);
    /*
    vector<VectorXd> allsamples_optfull=get<0>(res_optfull);
    vector<VectorXd> selectedsamples_optfull(nombre_samples_collected);
    vector<VectorXd> selectedhpars_optfull(nombre_samples_collected);
    
    for(int i=0;i<nombre_samples_collected;i++){
      int bigind=i*(allsamples_optfull.size()/selectedsamples_optfull.size());
      selectedsamples_optfull[i]=allsamples_optfull[bigind];
      selectedhpars_optfull[i]=DensOpt.HparsOpt(selectedsamples_optfull[i],hpars_guess_z,1e-4);
    }
    */

    //DensOpt.SetNewAllSamples(allsamples_optfull);
    DensOpt.SetNewSamples(selectedsamples_optfull);
    DensOpt.SetNewHparsOfSamples(selectedhpars_optfull);
    //DensOpt.Autocor_diagnosis(nautocor,foldname+"autocoroptfull.gnu");
    DensOpt.WriteSamples(foldname+"sampoptfull.gnu");
    DensOpt.WritePredictions(Xpred,foldname+"predsoptfull.gnu");


    auto t8bis=chrono::steady_clock::now();

    double tempsA=chrono::duration_cast<chrono::seconds>(t8-t7).count();
    double tempsAA=chrono::duration_cast<chrono::seconds>(t8bis-t8).count();


    cout << "temps mcmc full" <<  tempsA << " s" << endl;
    cout << "temps preds full" <<  tempsAA << " s" << endl;


    //écriture des temps de calcul
    ofstream outfile(foldname+"tempscalcul.gnu");
    outfile << temps1 << endl << temps2 << endl << temps3 << endl << temps4 << endl << temps5 << endl << temps6 << endl << tempsA << endl << tempsAA << endl;
    outfile.close(); 


    

    //maintenant : calcul de la précision de la log-vraisemblance.
    //on peut faire erreur a priori et erreur à posteriori.

    //erreur a priori: calcul sur les thetas du doe_init.

    auto compute_error_ll=[DensOpt,gp_ll,hpars_guess_z](vector<VectorXd> const & thetas, ofstream &ofile){
      //calcul de l'erreur moyenne entre la vraie loglik et les deux surrogates (surrogate direct et surrogate par hgps.)
      auto surro_ll_hgps=[DensOpt] (VectorXd const & theta)  {
        VectorXd h=DensOpt.EvaluateHparOpt(theta);
        return DensOpt.loglikelihood_theta(theta,h);
      }; 
      double error_hgps=0;
      double error_hll=0;
      double denom=0;
      for(int i=0;i<thetas.size();i++){
        VectorXd true_hpars=DensOpt.HparsOpt(thetas[i],hpars_guess_z,1e-2);
        double valref=DensOpt.loglikelihood_theta(thetas[i],true_hpars);
        double valhgps=surro_ll_hgps(thetas[i]);
        double valgpll=gp_ll.EvalMean(thetas[i]);
        denom+=pow(valref,2);
        error_hgps+=pow(valref-valhgps,2);
        error_hll+=pow(valref-valgpll,2);
      }
      error_hgps/=denom;
      error_hll/=denom;
      ofile << "error hgps sur la ll : " << error_hgps << ", error hll : " << error_hll << endl;
      return 0;
    };

    auto compute_error_hGPs=[DensOpt,hpars_guess_z](vector<VectorXd> const & thetas, ofstream &ofile){
      //calcul de l'erreur moyenne des hGPs sur le grid des theta.
      vector<double> err_hgps(3);
      vector<double> denom(3);
      for(int i=0;i<3;i++){
        err_hgps[i]=0; denom[i]=0;
      }
      for(int i=0;i<thetas.size();i++){
        VectorXd htrue=DensOpt.HparsOpt(thetas[i],hpars_guess_z,1e-2);
        VectorXd happrox=DensOpt.EvaluateHparOpt(thetas[i]);
        for (int j=0;j<3;j++){
          err_hgps[j]+=pow(htrue(j)-happrox(j),2);
          denom[j]+=pow(htrue(j),2);
        }
      }
      for(int i=0;i<3;i++){
        err_hgps[i]/=denom[i];
      }
      ofile << "erreur hGPs : ";
      for(int i=0;i<3;i++){
        ofile << err_hgps[i] << " ";
      }
      ofile << endl;
      return 0;
    };



    //écrire les fonctions likelihood et hpars sur un grid de theta pour pouvoir les comparer visuellement.
    // les thetas sont bien sûr thetas_doe.
    //en fait, prenons plus fin que thetas_doe sinon la ll est moche. Prenons 300 de manière générale.


    DoE doe_comparison(lb_t,ub_t,300,10); // DoE Halton
    vector<VectorXd> thetas_comp=doe_comparison.GetGrid();
    vector<VectorXd> likelihoods(thetas_comp.size());
    vector<VectorXd> hpars(thetas_comp.size());
    cout << "calcul des approximations..." << endl;
    for (int i=0;i<thetas_comp.size();i++){
      //calcul des likelihoods.
      VectorXd true_hpars=DensOpt.HparsOpt(thetas_comp[i],hpars_guess_z,1);
      VectorXd hpars_approx=DensOpt.EvaluateHparOpt(thetas_comp[i]);
      VectorXd hhp(6); hhp.head(3)=true_hpars; hhp.tail(3)=hpars_approx;
      double valref=DensOpt.loglikelihood_theta(thetas_comp[i],true_hpars);
      double valhgps=DensOpt.loglikelihood_theta(thetas_comp[i],hpars_approx);
      double valgpll=gp_ll.EvalMean(thetas_comp[i]);
      VectorXd ll(3); ll<< valref,valhgps,valgpll;
      hpars[i]=hhp;
      likelihoods[i]=ll;
    }
  PrintVector(foldname+"graphs_ll.gnu",thetas_comp,likelihoods);
  PrintVector(foldname+"graphs_hpars.gnu",thetas_comp,hpars);

  //calcul erreurs. C'est mieux que tout soit fait sur le même sample des thetas.
  vector<VectorXd> samples_opt_ref=get<0>(ReadSamples(foldname+"sampoptfullref.gnu"));

    ofstream oufile(foldname+"errors.gnu");
    cout << "erreurs a priori " << endl;
    compute_error_ll(thetas_comp,oufile);
    compute_error_hGPs(thetas_comp,oufile);
    cout << "erreurs a posteriori " << endl;
    compute_error_ll(selectedsamples_optfull,oufile);
    compute_error_hGPs(selectedsamples_optfull,oufile);
    oufile.close();

  exit(0); 
}

