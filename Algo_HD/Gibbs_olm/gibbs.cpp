#include "gibbs.h"
#include "densities.h"

using namespace std;
using namespace Eigen;

KDE::KDE(std::vector<Eigen::VectorXd> const &v)
{
    m_data=v;
    //on centre les données.
    VectorXd sum_of_elements=VectorXd::Zero(m_data[0].size());
    for (auto& x : m_data){
        sum_of_elements += x;
    }
    m_means=sum_of_elements/m_data.size();
    VectorXd var_estimate=VectorXd::Zero(m_data[0].size());
    for (auto& x : m_data){
        x-=m_means;
        var_estimate+=x.cwiseProduct(x);
    }
    m_std=(var_estimate/m_data.size()).cwiseSqrt();
    for (auto& x : m_data){
        x=x.cwiseQuotient(m_std);
    }
    m_bandwidth=VectorXd::Zero(m_data[0].size());
    for (int i=0;i<m_bandwidth.size();i++){
        m_bandwidth(i)=pow(m_data.size(),-1./(m_data[0].size()+4.));//scott's rule, uniform bandwidth
    }
    cout << "data std : " << m_std(0) << endl;
};

double KDE::Evaluate(VectorXd const &X) const
{
    double result(0);
    int ndim=m_data[0].size();
    VectorXd norm_X=(X-m_means).cwiseQuotient(m_std);
    for (int i=0;i<m_data.size();i++){
        result+=(1./(pow(6.28,ndim/2.)*pow(m_bandwidth(0),ndim)))*exp(-0.5*((m_data[i]-norm_X).cwiseQuotient(m_bandwidth)).squaredNorm());
    }
    return result/m_data.size();
}

VectorXd KDE::Sample(std::default_random_engine &generator) const{
    //tirage d'un point du KDE
    std::normal_distribution<double> distN(0,1);
    std::uniform_int_distribution<int> distI(0,m_data.size()-1); //génération d'un entier aléatoire
    int n_selected=distI(generator);
    VectorXd normal(m_data[0].size());
    for (int i=0;i<m_data[0].size();i++){
        normal(i)=distN(generator); //vecteur d'échantillons de lois normales
    }
    return m_means+m_std.cwiseProduct(m_data[n_selected])+m_std.cwiseProduct(m_bandwidth.cwiseProduct(normal));
}

void KDE::FidTest(default_random_engine &generator) const{
  vector<VectorXd> sim_sample(m_data.size());
  VectorXd samp_mean=VectorXd::Zero(m_data[0].size());
  VectorXd samp_var=VectorXd::Zero(m_data[0].size());
  for (int i=0;i<m_data.size();i++){
    sim_sample[i]=Sample(generator);
    samp_mean+=sim_sample[i];
  }
  samp_mean/=m_data.size();
  for (int i=0;i<m_data.size();i++){
    sim_sample[i]-=samp_mean;
    samp_var+=sim_sample[i].cwiseProduct(sim_sample[i]);
  }
  samp_var/=m_data.size();
  //calcul des erreurs relatives. On va rester en dimension 1 pour le moment.
  cout << "mrelerror=" << 100*(samp_mean(0)-m_means(0))/m_means(0) << "%" << endl;
  cout << "stdrelerror=" << 100*(sqrt(samp_var(0))-m_std(0))/m_std(0) << "%" << endl;
}

MCMC_par::MCMC_par(Density const &d, VectorXd const &hpars,int nchain) : Density(d)
{
    m_nchain=nchain;
    m_dim_mcmc=m_dim_pars;
    m_hpars=hpars;
     //inversion de la matrice des observations
    MatrixXd G=Gamma(&m_obs,m_hpars);
    LDLT<MatrixXd> ldlt(G);
    m_ldlt=ldlt;
}

double MCMC_par::loglikelihood_theta_fast(VectorXd const &theta) const
{
  int nd=m_obs.size();
  VectorXd obs_theta(nd);
  for (int i=0;i<nd;i++){
      obs_theta(i)=m_obs[i].Value()-(m_my_model(m_obs[i].GetX(),theta)-m_priormean(m_obs[i].GetX(),m_hpars));
    }
  VectorXd Alpha=m_ldlt.solve(obs_theta);
  return loglikelihood_fast(obs_theta,Alpha,m_hpars,m_ldlt);
}

void MCMC_par::SelectSamples(int nsamples)
{
  //renvoie nsamples pris uniformément de la MCMC.
    for (int i=0;i<m_all_samples.size();i++){
        if (i>(m_all_samples.size()/ nsamples) && i%(m_all_samples.size()/ nsamples)==0){
          m_selected_samples.push_back(m_all_samples[i]);
          m_selected_values.push_back(m_all_values[i]);
        }
    }
}
void MCMC_par::Autocorrelation_diagnosis(int n)
{
  //on centre les données
  VectorXd mean=VectorXd::Zero(m_dim_mcmc);
  VectorXd var=VectorXd::Zero(m_dim_mcmc); //variance composante par composante
  for (int i=0;i<m_all_samples.size();i++){
    mean+=m_all_samples[i];
  }
  mean/=m_all_samples.size();
  for (int i=0;i<m_all_samples.size();i++){
    m_all_samples[i]-=mean;
    var+=m_all_samples[i].cwiseProduct(m_all_samples[i]);
  }
  var/=m_all_samples.size(); //calcul de la variance composante par composante.
  //affichage du diagnostic dans autocor.gnu
  FILE* out= fopen("results/autocor.gnu","w");
  for (int i=0;i<n;i++){
    VectorXd cor=Lagged_mean(i).cwiseQuotient(var);
    for (int j=0;j<m_dim_mcmc;j++){
        fprintf(out, "%e ",cor(j));
    }
    fprintf(out,"\n");
  }
  fclose(out);
}

Eigen::VectorXd MCMC_par::Lagged_mean(int n) const{
  VectorXd ans=VectorXd::Zero(m_dim_mcmc);
  for (int i=0;i<m_all_samples.size()-n;i++){
    ans+=m_all_samples[i].cwiseProduct(m_all_samples[i+n]);
  }
  return ans/(double (m_all_samples.size()-n));
}

void MCMC_par::Run(Eigen::VectorXd const &Xinit, Eigen::MatrixXd const &COV_init,default_random_engine &generator)
{
  int naccept(0);
  std::normal_distribution<double> distN(0,1);
  std::uniform_real_distribution<double> distU(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
 

  cout << "Running MCMC with " << m_nchain << " steps at hpars " << m_hpars.transpose() << endl;
  if(!COV_init.cols()==m_dim_mcmc){cout <<"erreur de dimension MCMC" << endl;}
  VectorXd Xcurrent=Xinit;
  VectorXd Xcandidate(m_dim_mcmc);
  double fcurrent=loglikelihood_theta_fast(Xcurrent)+m_logpriorhpars(m_hpars)+m_logpriorpars(Xcurrent);
  double fcandidate(0);
  clock_t c_start = std::clock();
  for (int i=0;i<m_nchain;i++){
    VectorXd Step(m_dim_mcmc);
    for (int j=0;j<Step.size();j++){Step[j]=distN(generator);}
    Xcandidate=Xcurrent+sqrtCOV*Step;
    fcandidate=loglikelihood_theta_fast(Xcandidate)+m_logpriorhpars(m_hpars)+m_logpriorpars(Xcandidate);
    if (in_bounds(Xcandidate)){
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    m_all_samples.push_back(Xcurrent);
    m_all_values.push_back(fcurrent);
    //ici ajouter les valeurs de la chaîne si l'on souhaite les conserver
  }
  clock_t c_end = std::clock();
  double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  double time_mcmc=time_elapsed_ms / 1000.0;
  std::cout << "Temps pour MCMC : " << time_mcmc << " s\n";
  cout << "accept rate : " << 100*double(naccept)/double(m_nchain) << endl;
};

bool MCMC_par::in_bounds(Eigen::VectorXd &X)
{
  for (int i=0;i<m_Grid.GetDimension();i++){
    if (X(i)<m_Grid.GetParsLb()(i) || X(i)>m_Grid.GetParsUb()(i)){return false;}
  }
  return true;
}

void MCMC_par::WriteAllSamples(const char* file_name) const
{
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<m_all_samples.size();i++){
    for (int j=0;j<m_dim_mcmc;j++){
      fprintf(out,"%e ",m_all_samples[i](j));
    }
    fprintf(out,"%e\n",m_all_values[i]);
  }
  fclose(out);
};

void MCMC_par::WriteSelectedSamples(const char* file_name) const
{
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<m_selected_samples.size();i++){
    for (int j=0;j<m_dim_mcmc;j++){
      fprintf(out,"%e ",m_selected_samples[i](j));
    }
    fprintf(out,"%e\n",m_selected_values[i]);
  }
  fclose(out);
};



