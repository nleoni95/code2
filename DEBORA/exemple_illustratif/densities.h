/*Nouvelle version de densities. Dans celle-ci, on considère une prédiction vectorielle du modèle. On va chercher à obtenir des échantillons des densités par des MCMC, au lieu de les évaluer par quadrature. */
#ifndef DENSITIES_H
#define DENSITIES_H

#include <vector>
#include <map>
#include <random>
#include <string>
#include <iostream>
#include <fstream>
#include <functional>
#include <algorithm>
#include <utility>
#include <iterator>
#include <Eigen/Dense>
#include "gp++.h"
#include "my_samplers.h"



class AUGDATA {
public:
	AUGDATA(){};
	AUGDATA(Eigen::VectorXd const &x, Eigen::VectorXd const &f){ X=x; F=f;};
	AUGDATA(AUGDATA const &d){ X = d.X; F= d.F;};
	void operator = (const AUGDATA d){ X = d.X; F= d.F;};
	Eigen::VectorXd GetX() const { return X; };
	Eigen::VectorXd Value() const { return F; };
    void SetX(Eigen::VectorXd x) { X=x;};
    void SetValue(Eigen::VectorXd f) { F=f;};
    std::vector<DATA> split()const {std::vector<DATA> v(F.size());
        DATA dat; dat.SetX(X);
        for(int i=0;i<F.size();i++){dat.SetValue(F(i)); v[i]=dat;}
        return v;
    };
private:
	Eigen::VectorXd X;
	Eigen::VectorXd F;
};


class DoE
{
    friend class Density;
    public:
    DoE();
    DoE(Eigen::VectorXd const &lb, Eigen::VectorXd const &ub,int n); //constructeur en grid sampling
    DoE(Eigen::VectorXd const &lb, Eigen::VectorXd const &ub,int ntotal,std::default_random_engine &generator); //constructeur en LHS
    DoE(Eigen::VectorXd const &lb, Eigen::VectorXd const &ub,int npoints,int first_element); //constructeur en QMC

    Eigen::VectorXd Randpert(int n,std::default_random_engine & generator) const;
    Eigen::VectorXd indices(int const s, int const n, int const d);
    void WriteGrid(std::string const & filename) const;
    std::vector<Eigen::VectorXd> GetGrid() const {return m_grid;}

    protected:
    Eigen::VectorXd m_lb_pars; //bornes inf des paramètres
    Eigen::VectorXd m_ub_pars; //bornes sup des paramètres
    std::vector<Eigen::VectorXd> m_grid; //valeurs des paramètres
};

class Density
{
    public:
    //construction
    Density();
    Density(DoE const &g);
    Density(Density const &d);

    //fonctions de set à appeler dans le main.
    void SetLogPriorPars(std::function<double(Eigen::VectorXd const &)> logpriorpars){m_logpriorpars=logpriorpars;};
    void SetLogPriorHpars(std::function<double(Eigen::VectorXd const &)> logpriorhpars){m_logpriorhpars=logpriorhpars;};
    void SetKernel(std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &,Eigen::VectorXd const &)> Kernel){m_Kernel=Kernel;};
    void SetKernelDerivatives(std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &,Eigen::VectorXd const &)> DKernel1,std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &,Eigen::VectorXd const &)> DKernel2,std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &,Eigen::VectorXd const &)> DKernel3){m_DKernel1=DKernel1,m_DKernel2=DKernel2,m_DKernel3=DKernel3;};
    void SetModel(std::function<Eigen::VectorXd(Eigen::VectorXd const &,Eigen::VectorXd const &)> my_model){m_model=my_model;}
    void SetPriorMean(std::function<Eigen::VectorXd(Eigen::VectorXd const &,Eigen::VectorXd const &)> priormean){m_priormean=priormean;}
    void SetHparsBounds(Eigen::VectorXd const &lb_hpars,Eigen::VectorXd const &ub_hpars){m_dim_hpars=lb_hpars.size(); m_lb_hpars=lb_hpars;m_ub_hpars=ub_hpars;};
    void SetDataExp(std::vector<AUGDATA> &obs){m_data_exp=obs;};
    void SetXprofile(Eigen::VectorXd const &X){m_Xprofile=X;
        std::vector<Eigen::VectorXd> Xc; for(int i=0;i<X.size();i++){
            Eigen::VectorXd x(1);x(0)=X(i);Xc.push_back(x);
        } m_Xprofile_converted=Xc;
    };
    void SetNoise(double stdnoise){m_noise=stdnoise;}
    void SetNewDoE(DoE const & g);
    void SetNewSamples(std::vector<Eigen::VectorXd> const & s){m_samples=s;}
    void SetNewHparsOfSamples(std::vector<Eigen::VectorXd> const & s){m_hparsofsamples=s;}
    void SetNewAllSamples(std::vector<Eigen::VectorXd> const & s){m_allmcmcsamples=s;}
    //fonctions pour évaluation de la vraisemblance.
    Eigen::MatrixXd Gamma(std::vector<Eigen::VectorXd> const & locs, Eigen::VectorXd const &hpar) const;
    double loglikelihood_fast(Eigen::VectorXd const &obs,Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const;
    double loglikelihood_theta_fast(Eigen::VectorXd const &theta,Eigen::VectorXd const &hpars, Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const;
    double loglikelihood_theta(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar)const;
    double loglikelihood_obs_i(Eigen::VectorXd const & obsv,int i, Eigen::VectorXd const & hpars) const;

    //fonctions de MCMC
    double Run_Burn_phase_MCMC(int nburn, Eigen::MatrixXd & COV_init,Eigen::VectorXd const & hpars,Eigen::VectorXd &Xcurrent,std::default_random_engine & generator);
    void Run_MCMC_fixed_hpars(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,Eigen::VectorXd const & hpars,std::default_random_engine & generator);
    double Run_Burn_phase_FullMCMC(int nburn, Eigen::MatrixXd & COV_init,Eigen::VectorXd &Xcurrent,std::default_random_engine & generator);
    void Run_FullMCMC(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,std::default_random_engine & generator);
    void Run_FullMCMC_noburn(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,std::default_random_engine & generator);

    int max_ll() const;
    Eigen::VectorXd map()const;
    double Var() const;





    //diagnostic MCMC
    Eigen::VectorXd Lagged_mean(std::vector<Eigen::VectorXd> const &v, int n) const;
    void Autocor_diagnosis(int nstepsmax, std::string const & filename) const;
    void WriteMCMCSamples(std::string const & filename) const;

    void FindVPs(Eigen::MatrixXd const &M) const;
    
    bool in_bounds_pars(Eigen::VectorXd const & pars) const;
    bool in_bounds_hpars(Eigen::VectorXd const & hpars) const;

    //routine d'optimisation
    int optroutine(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars, double max_time) const;
    int optroutine_withgrad(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars, double max_time)const;
    int optroutine_lightwithgrad(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars)const;
    int optroutine_light(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars)const;
    int optroutine_heavy(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars)const;

    //calcul des hpars KOH
    Eigen::VectorXd HparsKOH(Eigen::VectorXd const & hpars_guess, double max_time) const;
    static double optfuncKOH(const std::vector<double> &x, std::vector<double> &grad, void *data);

    Eigen::VectorXd HparsKOHFromData(Eigen::VectorXd const & hpars_guess,std::vector<Eigen::VectorXd> const & thetas, std::vector<Eigen::VectorXd> const & values) const;
    static double optfuncKOHFromData(const std::vector<double> &x, std::vector<double> &grad, void *data);

    //Eigen::VectorXd HparsNOEDM(Eigen::VectorXd const & hpars_guess)const;
    static double optfuncNOEDM(const std::vector<double> &x, std::vector<double> &grad, void *data);

    Eigen::VectorXd HparsLOOCV(Eigen::VectorXd const & hpars_guess, double max_time) const;
    static double optfuncLOOCV(const std::vector<double> &x, std::vector<double> &grad, void *data);

    //fonctions pour calcul de hessienne

    //static Eigen::MatrixXd ComputeHessian(Eigen::VectorXd const & ref_point,Eigen::VectorXd const & step_size, function<Eigen::VectorXd(Eigen::VectorXd)>const & fprime);
    
    //sauvegarde et lecture
    void WriteSamples(std::string const & filename)const;
    void ReadSamples(std::string const & filename);

    //fonctions de prédiction. écrites pour un seul point expérimental X.
    //et on va utiliser Xprofile..
    Eigen::VectorXd meanF(Eigen::VectorXd const & X) const;
    Eigen::VectorXd meanZCondTheta(Eigen::VectorXd const & X,Eigen::VectorXd const & theta, Eigen::VectorXd const &hpars_z) const;
    Eigen::MatrixXd varZCondTheta(Eigen::VectorXd const & X,Eigen::VectorXd const & theta, Eigen::VectorXd const &hpars_z) const;
    Eigen::MatrixXd varF(Eigen::VectorXd const & X) const;
    Eigen::MatrixXd PredFZ(Eigen::VectorXd const &X) const;
    Eigen::VectorXd DrawZCondTheta(Eigen::VectorXd const & X,Eigen::VectorXd const & theta, Eigen::VectorXd const &hpars_z, std::default_random_engine & generator) const;

    double FindQuantile(double pct, Eigen::VectorXd const &X) const;

    void WriteOneCalcul(Eigen::VectorXd const &X, Eigen::VectorXd const & theta, Eigen::VectorXd const & hpars_z, std::string const & filename)const ;
    int WritePredictions(Eigen::VectorXd const &X,std::string const & filename) const;
    int WritePredictionsF(Eigen::VectorXd const &X,std::string const & filename) const;
    void WritePriorPredictions(Eigen::VectorXd const &X,std::string const & filename,std::default_random_engine & generator);
    void WritePriorPredictionsF(Eigen::VectorXd const &X,std::string const & filename,std::default_random_engine & generator);
    void WriteFinePredictions(Eigen::VectorXd const &X,std::string const & filename) const;
    void WriteFinePriorPredictions(Eigen::VectorXd const &X,std::string const & filename,std::default_random_engine & generator);


    //comparaison de modèles
    double AIC()const;
    double DIC()const;
    double WAIC2()const;

    //accès extérieur
    const std::vector<Eigen::VectorXd> *GetGrid() const {return &m_Grid.m_grid;}
    const std::vector<AUGDATA> *GetExpData() const {return &m_data_exp;}
    const std::vector<Eigen::VectorXd> *GetXconverted() const{return &m_Xprofile_converted;}
    const Eigen::VectorXd GetXprofile() const{return m_Xprofile;}
    Eigen::LDLT<Eigen::MatrixXd> GetLDLT(Eigen::VectorXd const & hpars);
    std::pair<Eigen::VectorXd,Eigen::VectorXd> GetBoundsHpars()const{return std::make_pair(m_lb_hpars,m_ub_hpars);}
    Eigen::VectorXd EvaluateModel(Eigen::VectorXd const &X, Eigen::VectorXd const & theta) const {return m_model(X,theta);}
    Eigen::VectorXd EvaluatePMean(Eigen::VectorXd const &X, Eigen::VectorXd const & hpars) const {return m_priormean(X,hpars);}
    double EvaluateLogPHpars(Eigen::VectorXd const & hpars) const {return m_logpriorhpars(hpars);}
    double EvaluateLogPPars(Eigen::VectorXd const & pars) const {return m_logpriorpars(pars);}

    protected:
    double m_noise;
    std::function<Eigen::VectorXd(Eigen::VectorXd const &,Eigen::VectorXd const &)> m_model;
    std::function<Eigen::VectorXd(Eigen::VectorXd const &,Eigen::VectorXd const &)> m_priormean;
    std::function<double(Eigen::VectorXd const &)> m_logpriorpars;
    std::function<double(Eigen::VectorXd const &)> m_logpriorhpars;
    std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &,Eigen::VectorXd const &)> m_Kernel;
    std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &,Eigen::VectorXd const &)> m_DKernel1;
    std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &,Eigen::VectorXd const &)> m_DKernel2;
    std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &,Eigen::VectorXd const &)> m_DKernel3;

    int m_dim_hpars;
    int m_dim_pars;
    //bornes des hpars
    Eigen::VectorXd m_lb_hpars;
    Eigen::VectorXd m_ub_hpars; 
    Eigen::VectorXd m_lb_pars;
    Eigen::VectorXd m_ub_pars;

    DoE m_Grid;

    std::vector<AUGDATA> m_data_exp; //dans mon cas : il est de taille 1.
    Eigen::VectorXd m_Xprofile; //les X sur lesquels est défini le premier profil. A faire évoluer éventuellement en un vecteur de VectorXd.
    std::vector<Eigen::VectorXd> m_Xprofile_converted; //conversion du Xprofile en vecteur de vectorsXd de dimension 1.

    std::vector<Eigen::VectorXd> m_samples;
    std::vector<Eigen::VectorXd> m_hparsofsamples;
    std::vector<Eigen::VectorXd> m_allmcmcsamples;
};

class DensityOpt : public Density{
    public:
    DensityOpt(Density const & d);
    //calcul des hpars optimaux
    //Eigen::VectorXd HparsOpt(Eigen::VectorXd const & theta, Eigen::VectorXd const & hpars_guess);
    Eigen::VectorXd HparsOpt(Eigen::VectorXd const & theta, Eigen::VectorXd const & hpars_guess,double max_time)const;
    Eigen::VectorXd HparsOpt_quick(Eigen::VectorXd const & theta, Eigen::VectorXd const & hpars_guess,double max_time)const;
    static double optfuncOpt(const std::vector<double> &x, std::vector<double> &grad, void *data);
    static double optfuncOpt_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data);
    
    Eigen::VectorXd EvaluateHparOpt(Eigen::VectorXd const & theta) const;

    //version densityopt

    double DIC()const;

    //Calcul des hpars optimaux sur le grid.
    void Compute_optimal_hpars(double max_time);

    //Initialisation des GPs. 
    void BuildHGPs(double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &),Eigen::MatrixXd const & Bounds_hpars_GPs,Eigen::VectorXd const & Hpars_guess_GPs,int nmodes);
    void Test_hGPs(int npoints, double max_time);
    Eigen::VectorXd Test_hGPs_on_sample(std::vector<Eigen::VectorXd> const & theta_ref,std::vector<Eigen::VectorXd> const & hpars_ref) const;
    //Optimisation du GP pour les hpars. Il faut autant de GPs que d'hpars de z.
    static double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data);
    Eigen::VectorXd opti_1gp(int i,Eigen::VectorXd & hpars_guess);
    void opti_allgps(Eigen::VectorXd const & hpars_guess);
    void update_hGPs(std::vector<Eigen::VectorXd> const &new_thetas,double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &),Eigen::MatrixXd const & Bounds_hpars_GPs,Eigen::VectorXd const & Hpars_guess_GPs,int nmodes, double max_time);

    //Fonctions de MCMC

    double Run_Burn_phase_MCMC_opti_expensive(int nburn, Eigen::MatrixXd & COV_init,Eigen::VectorXd &Xcurrent,std::default_random_engine & generator, double max_time);
    void Run_MCMC_opti_expensive(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,std::default_random_engine & generator, double max_time);

    double Run_Burn_phase_MCMC_opti_hGPs(int nburn, Eigen::MatrixXd & COV_init,Eigen::VectorXd &Xcurrent,std::default_random_engine & generator);
    void Run_MCMC_opti_hGPs(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,std::default_random_engine & generator);


    //pour mon opti modif
    //double get_post_hpar(Eigen::VectorXd const & theta_prop, Eigen::VectorXd const & hpars_current, Eigen::VectorXd const & step);

    //sauvegarde des hpars des gps.
    void WritehGPs(std::string const & filename)const;
    void ReadhGPs(std::string const & filename);

    protected:
    std::vector<AUGDATA> m_hpars_opti;
    std::vector<GP> m_vgp_hpars_opti;
    std::vector<Eigen::VectorXd> m_vhpars_pour_gp;

    Eigen::MatrixXd m_Bounds_hpars_GPs; //des mêmes bornes d'optimisation pour tous les GPs.

    //quantités pour HGPs
    Eigen::MatrixXd m_VP;
    Eigen::MatrixXd m_Acoefs;
    Eigen::VectorXd m_featureMeans;

    //quantités pour le test_HGPs.
    std::vector<Eigen::VectorXd> m_newgrid;
    Eigen::MatrixXd m_Hopt_newgrid;

};

class Densities{
    //liste de densités.
    public:
    Densities();
    Densities(Densities const &d);
    Densities(std::vector<Density> const &v);
    void SetLogPriorPars(std::function<double(Eigen::VectorXd const &)> logpriorpars){m_logpriorpars=logpriorpars;};
    void SetDimPars(int i){m_dim_pars=i;}
    bool in_bounds_pars(Eigen::VectorXd const & pars) const;

    int optroutine(nlopt::vfunc optfunc,void *data_ptr, std::vector<double> &x, std::vector<double> const & lb_hpars, std::vector<double> const & ub_hpars,double max_time);

    static double optfuncKOH_pooled(const std::vector<double> &x, std::vector<double> &grad, void *data);


    //on cherche les hyperparamètres optimaux pour chacune des density
    std::vector<Eigen::VectorXd> HparsKOH_separate(std::vector<Eigen::VectorXd> const & hpars_guess_vec, double max_time);
    std::vector<Eigen::VectorXd> HparsKOH_pooled(std::vector<Eigen::VectorXd> const & hpars_guess_vec, double max_time);

    //run une MCMC avec l'ensemble des densités
    double loglikelihood_theta_fast(Eigen::VectorXd const & theta, std::vector<Eigen::VectorXd> const & hpars_v, std::vector<Eigen::LDLT<Eigen::MatrixXd>> const & ldlt_v) const;

    double loglikelihood_theta(Eigen::VectorXd const & theta, std::vector<Eigen::VectorXd> const & hpars_v) const;

    std::vector<Eigen::LDLT<Eigen::MatrixXd>> compute_ldlts(std::vector<Eigen::VectorXd> const & hpars_v);

    double Run_Burn_phase_MCMC_fixed_hpars(int nburn, Eigen::MatrixXd & COV_init,std::vector<Eigen::VectorXd> const & hpars_v, std::vector<Eigen::LDLT<Eigen::MatrixXd>> const & ldlt_v,Eigen::VectorXd &Xcurrent,std::default_random_engine & generator);
    void Run_MCMC_fixed_hpars(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,std::vector<Eigen::VectorXd> const & hpars_v,std::default_random_engine & generator);

    Eigen::VectorXd Lagged_mean(std::vector<Eigen::VectorXd> const &v, int n) const;
    void Autocor_diagnosis(int nstepsmax, std::string const & filename) const;
    void WriteSamples(std::string const & filename)const;
    void WriteAllSamples(std::string const & filename)const;

    void WritePredictions(Eigen::VectorXd const &X,std::string const & filename);
    void WritePredictionsF(Eigen::VectorXd const &X,std::string const & filename);

    int GetDim()const{return m_dim;}
    const std::vector<Density> *GetDensities_v() const {return &m_Densities_vec;}


    protected:
    std::vector<Density> m_Densities_vec;
    int m_dim; //taille de la liste de densities.
    int m_dim_pars;

    std::vector<Eigen::VectorXd> m_samples;
    std::vector<std::vector<Eigen::VectorXd>> m_hparsofsamples_v;
    std::vector<Eigen::VectorXd> m_allmcmcsamples;
    std::function<double(Eigen::VectorXd const &)> m_logpriorpars;
};

class DensitiesOpt : public Densities{
    public:
    DensitiesOpt(Densities const &ds);

    void compute_optimal_hpars(double max_time);
    std::vector<Eigen::VectorXd> HparsOpt(Eigen::VectorXd const & theta,std::vector<Eigen::VectorXd> const & hpars_guess_vec, double max_time) const;

    std::vector<Eigen::VectorXd> EvaluateHparsOpt(Eigen::VectorXd const & theta) const;

    //Pour les hGPs. 

    void opti_allgps(std::vector<Eigen::VectorXd> const & hpars_guess_v);
    void BuildHGPs(double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &),Eigen::MatrixXd const & Bounds_hpars_GPs,Eigen::VectorXd const & Hpars_guess_GPs,int nmodes);
    void Test_hGPs(int npoints,double max_time);

    //mcmc
    double Run_Burn_phase_MCMC_opti_hGPs(int nburn, Eigen::MatrixXd & COV_init,Eigen::VectorXd &Xcurrent,std::default_random_engine & generator);
    void Run_MCMC_opti_hGPs(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,std::default_random_engine & generator);



    protected:
    std::vector<DensityOpt> m_DensityOpt_vec;
};

#endif /*DENSITIES_H*/