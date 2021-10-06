#ifndef DENSITIES_H
#define DENSITIES_H


#include <vector>
#include <random>
#include <Eigen/Dense>
#include <iostream>
#include "gp++.h"

class DoE
{
 public:
  DoE();
  DoE(Eigen::VectorXd lb, Eigen::VectorXd ub,int n); //constructeur en grid sampling
  DoE(Eigen::VectorXd lb, Eigen::VectorXd ub,int ntotal,std::default_random_engine &generator); //constructeur en LHS
  
  static Eigen::VectorXd indices(int const s, int const n, int const d);
  static int grosindice(Eigen::VectorXd const &v, int const n);
  static Eigen::VectorXd Randpert(int const n);
  std::vector<Eigen::VectorXd> GetGrid() const {return m_grid;};
  std::vector<double> GetWeights() const {return m_weights;};
  int GetDimension() const{return m_dimension;};
  Eigen::VectorXd GetParsLb() const {return m_lb;};
  Eigen::VectorXd GetParsUb() const {return m_ub;};

 private:
  Eigen::VectorXd m_lb; //bornes inf des pars.
  Eigen::VectorXd m_ub; //bornes sup des pars.
  int m_dimension; //dimension des pars
  std::vector<Eigen::VectorXd> m_grid; //valeurs des pars
  std::vector<double> m_weights; //poids de la quadrature
};

class Density
{
  public:
    Density();
    //constructeur avec grid/ Initialise m_npts également.
    Density(DoE g);
    //constructeur par copie
    Density(Density &d);
    void WritePost(const char* file_name);
    double Entropy() const;
    Eigen::VectorXd MAP() const;
    Eigen::VectorXd Mean() const;
    Eigen::MatrixXd Cov() const;
    double KLDiv(Density &d) const;

    //fonctions pour calculer la ddp
    Eigen::MatrixXd Gamma(void const *data, Eigen::VectorXd const &hpar) const;
    double loglikelihood_fast(Eigen::VectorXd const &obs, Eigen::VectorXd const &Alpha, Eigen::VectorXd const &hpar, Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const;
    double loglikelihood(void *data, Eigen::VectorXd const &hpar) const;
    double loglikelihood_theta_fast(void *data, Eigen::VectorXd const &hpar, Eigen::LDLT<Eigen::MatrixXd> const &ldlt, Eigen::VectorXd const &theta)const;
    double loglikelihood_theta(void *data, Eigen::VectorXd const &hpar, Eigen::VectorXd const &theta)const;
  

    //fonctions de set
    void SetLogPriorPars(double (*logprior)(Eigen::VectorXd const &)){m_logpriorpars=logprior;};
    void SetLogPriorHpars(double (*logprior)(Eigen::VectorXd const &)){m_logpriorhpars=logprior;};
    void SetKernel(double (*Kernel)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &)){m_Kernel=Kernel;};
    void SetModel(double (*my_model)(Eigen::VectorXd const &,Eigen::VectorXd const &)){m_my_model=my_model;}
    void SetPriorMean(double (*PriorMean)(Eigen::VectorXd const &,Eigen::VectorXd const &)){m_priormean=PriorMean;}
  
    void SetHparsBounds(std::vector<double> &lb_hpars,std::vector<double> &ub_hpars){m_dim_hpars=lb_hpars.size(); m_lb_hpars=lb_hpars;m_ub_hpars=ub_hpars;};
    void SetObs(std::vector<DATA> &obs){m_obs=obs;};

    //tire une réalisation d'une MVN de moyenne Mean et de covariance COV.
    static Eigen::VectorXd DrawMVN(Eigen::VectorXd &Mean, Eigen::MatrixXd &COV, std::default_random_engine &generator); 
    //build la densité à partir d'une valeur fixe d'hpars
    void Build(Eigen::VectorXd const &hpars);


    //accesseurs
    DoE GetDoE(){return m_Grid;};
    std::vector<double> GetValues(){return m_values;};
    std::vector<DATA> GetObs(){return m_obs;};
    int GetNpts(){return m_npts;};

    //évaluations
    double LogPriorHpars(Eigen::VectorXd &hpars) const {return m_logpriorhpars(hpars);};
    double MyModel(Eigen::VectorXd const & x,Eigen::VectorXd const & pars)const{return m_my_model(x,pars);};
    
  protected:
    //met l'intégrale à 1 
    void vectorweight1();
    //transforme les valeurs par un log
    void vectorlog(); 
    //transforme les valeurs par un exp
    void vectorexp();
    // retourne l'indice de la valeur maximale
    int indmax() const;

    int m_dim_hpars; //dimension des hpars
    int m_dim_pars; //dimension des pars
    int m_npts; //nombre total de points dans le grid
    std::vector<double> m_values; //valeurs de la logprob
    DoE m_Grid; //valeurs du grid
    std::vector<DATA> m_obs; //observations ici
    std::vector<double> m_lb_hpars; //borne inf des hpars
    std::vector<double> m_ub_hpars;
    double (*m_Kernel)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &);
    double (*m_logpriorhpars)(Eigen::VectorXd const &);
    double (*m_logpriorpars)(Eigen::VectorXd const &);
    double (*m_my_model)(Eigen::VectorXd const &,Eigen::VectorXd const &);
    double (*m_priormean)(Eigen::VectorXd const &,Eigen::VectorXd const &);
};

class DensityKOH : public Density
{
  public:
  DensityKOH(DoE g);
  DensityKOH(Density &d);
  //calcule les hpars optimaux, remplit la densité, passe à l'exp et normalise.
  void Build();
  //calcule les hpars optimaux
  Eigen::VectorXd HparsKOH();

  //prédiction.
  double MeanPred(Eigen::VectorXd &x) const;
  Eigen::VectorXd MeanPred(std::vector<Eigen::VectorXd> &vectorx) const;
  double VarPred(Eigen::VectorXd &x) const;
  Eigen::MatrixXd VarPred(std::vector<Eigen::VectorXd> &vectorx) const;
  Eigen::VectorXd DrawSample(std::vector<Eigen::VectorXd> &vectorx,std::default_random_engine &generator) const;
  //accesseurs
  Eigen::VectorXd GetHpars() const {return m_hpars;};
  protected:
  //la fonction d'optimisation
  static double optfunc(std::vector<double> const &x, std::vector<double> &grad, void *data);
  Eigen::VectorXd m_hpars;
};

class DensityOpt : public Density
{
  public:
  DensityOpt(DoE g);
  DensityOpt(Density &d);
  void Build();
  void WriteHpars(const char* file_name) const;
  double MeanPred(Eigen::VectorXd &x) const;
  Eigen::VectorXd MeanPred(std::vector<Eigen::VectorXd> &vectorx) const;
  double VarPred(Eigen::VectorXd &x) const;
  Eigen::MatrixXd VarPred(std::vector<Eigen::VectorXd> &x) const;
  Eigen::VectorXd DrawSample(std::vector<Eigen::VectorXd> &vectorx,std::default_random_engine &generator) const;
  Eigen::VectorXd MeanPredCondTheta(std::vector<Eigen::VectorXd> &vectorx,int i) const;
  Eigen::MatrixXd VarPredCondTheta(std::vector<Eigen::VectorXd> &vectorx,int i) const;

  std::vector<Eigen::VectorXd> GetHpars() const {return m_hpars_opti;};
  protected:
  //fonction d'optimisation
  static double optfunc(std::vector<double> const &x, std::vector<double> &grad, void *data);
  //méthode d'optimisation. Remplit la densité.
  //contient les hpars optimaux (remplis par la méthode Build)
  std::vector<Eigen::VectorXd> m_hpars_opti;
};

struct AugmentedDensityOpt
{
  AugmentedDensityOpt(DensityOpt *d, std::vector<DATA> *obs){D=d; newobs=obs;};
  DensityOpt *D;
  std::vector<DATA> *newobs;
};

class DensityBayes : public Density
{
  public:
  DensityBayes(Density &d);
  void SetSampleHpars(Eigen::VectorXd (*sample_hpars)(std::default_random_engine &)){m_sample_hpars=sample_hpars;};
  void Build();
  protected:
  Eigen::VectorXd (*m_sample_hpars)(std::default_random_engine &generator);
};

class DensityCV : public DensityKOH
{
  public:
  DensityCV(Density &d);
  static double optfunc(std::vector<double> const &x, std::vector<double> &grad, void *data);
  //calcule les hpars optimaux
  Eigen::VectorXd HparsCV();
  void Build();
};   

class MCMC : public Density{
  //on construit une classe un peu spéciale qui permettra de faire une MCMC globale.
  //l'héritage nous permet de récupérer toutes les fonctions Kernel, logvs, etc.
  //le paramètre généralisé X est dans l'ordre (pars,hpars).
  public:
  MCMC(Density &d,int nchain);
  void Run(Eigen::VectorXd &Xinit, Eigen::MatrixXd &COV,std::default_random_engine &generator);
  void SelectSamples(int nsamples);
  std::vector<Eigen::VectorXd> GetSelectedSamples() const {return m_selected_samples;}
  std::vector<double> GetSelectedValues() const {return m_selected_values;}

  //fonctions de calcul sur les échantillons sélectionnés.
  Eigen::VectorXd Mean() const;
  Eigen::MatrixXd Cov() const;
  Eigen::VectorXd MAP() const;

  //fonctions de prédiction.
  Eigen::VectorXd MeanPredCondX(std::vector<Eigen::VectorXd> &vectorx,Eigen::VectorXd const &X) const;
  Eigen::MatrixXd VarPredCondX(std::vector<Eigen::VectorXd> &vectorx,Eigen::VectorXd const &X) const;
  
  Eigen::VectorXd MeanPred(std::vector<Eigen::VectorXd> &vectorx) const;
  Eigen::MatrixXd VarPred(std::vector<Eigen::VectorXd> &vectorx) const;
  Eigen::VectorXd DrawSample(std::vector<Eigen::VectorXd> &vectorx,std::default_random_engine &generator) const;


  protected:
  bool in_bounds(Eigen::VectorXd &X) const;

  int m_nchain; //nombre d'itérations de la MCMC
  int m_dim_mcmc; //dimension theta+hpars
  int m_naccept; //nombre d'étapes acceptées
  std::vector<Eigen::VectorXd> m_all_samples; //tous les états de la MCMC.
  std::vector<Eigen::VectorXd> m_selected_samples; //échantillon récupéré de la chaîne
  std::vector<double> m_all_values; //toutes les valeurs de la MCMC
  std::vector<double> m_selected_values; // valeurs récupérées de la chaîne.
};


#endif /*DENSITIES_H*/
