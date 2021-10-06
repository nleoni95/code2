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
    void SetPriorMean(std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &)> priormean){m_priormean=priormean;}
    void SetHparsBounds(Eigen::VectorXd const &lb_hpars,Eigen::VectorXd const &ub_hpars){m_dim_hpars=lb_hpars.size(); m_lb_hpars=lb_hpars;m_ub_hpars=ub_hpars;};
    void SetDataExp(std::vector<AUGDATA> &obs){m_data_exp=obs;};
    void SetXprofile(Eigen::VectorXd const &X){m_Xprofile=X;
        std::vector<Eigen::VectorXd> Xc; for(int i=0;i<X.size();i++){
            Eigen::VectorXd x(1);x(0)=X(i);Xc.push_back(x);
        } m_Xprofile_converted=Xc;
    };
    void SetNewDoE(DoE const & g);
    void SetNewSamples(std::vector<Eigen::VectorXd> const & s){m_samples=s;}
    void SetNewHparsOfSamples(std::vector<Eigen::VectorXd> const & s){m_hparsofsamples=s;}
    void SetNewAllSamples(std::vector<Eigen::VectorXd> const & s){m_allmcmcsamples=s;}
    //fonctions pour évaluation de la vraisemblance.
    Eigen::MatrixXd Gamma(std::vector<Eigen::VectorXd> const & locs, Eigen::VectorXd const &hpar) const;
    double loglikelihood_fast(Eigen::VectorXd const &obs,Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const;
    double loglikelihood_theta_fast(std::vector<AUGDATA> const &exp_data, Eigen::VectorXd const &theta, Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const;
    double loglikelihood_theta_fast_decomposition(std::vector<AUGDATA> const &exp_data, Eigen::VectorXd const &theta, Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const;
    double loglikelihood_fast_terme_complexity(Eigen::VectorXd const &obs,Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const;
    double loglikelihood_fast_terme_fit(Eigen::VectorXd const &obs,Eigen::LDLT<Eigen::MatrixXd> const &ldlt)const;
    double loglikelihood_theta_decomposition(std::vector<AUGDATA> const &exp_data, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar)const;
    double loglikelihood_theta(std::vector<AUGDATA> const &exp_data, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar)const;

    //fonctions de MCMC
    double Run_Burn_phase_MCMC(int nburn, Eigen::MatrixXd & COV_init,Eigen::VectorXd const & hpars,Eigen::VectorXd &Xcurrent,std::default_random_engine & generator);
    void Run_MCMC_fixed_hpars(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,Eigen::VectorXd const & hpars,std::default_random_engine & generator);
    double Run_Burn_phase_FullMCMC(int nburn, Eigen::MatrixXd & COV_init,Eigen::VectorXd &Xcurrent,std::default_random_engine & generator);
    void Run_FullMCMC(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,std::default_random_engine & generator);
    void Run_FullMCMC_noburn(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,std::default_random_engine & generator);
    
    double measure_acc_rate_1D(int nsteps, Eigen::VectorXd const & direction, double step_size, Eigen::VectorXd const & Xorigin, std::function<double(Eigen::VectorXd)> const & compute_score, std::default_random_engine & generator) const;

    void estimate_acc_rate(int nsteps, Eigen::MatrixXd const & modes, Eigen::VectorXd const & lambdas, Eigen::VectorXd const & Xorigin, std::function<double(Eigen::VectorXd)> const & compute_score, std::default_random_engine & generator) const;

    void get_acc_rate(Eigen::MatrixXd const & COV, Eigen::VectorXd const & Xcurrent, std::function<double(Eigen::VectorXd)> const & compute_score, std::default_random_engine & generator) const;

    Eigen::MatrixXd scale_covmatrix(Eigen::MatrixXd const & COV, Eigen::VectorXd const & Xcurrent, std::function<double(Eigen::VectorXd)> const & compute_score, double true_accrate, std::default_random_engine & generator,std::string const filename) const;
 
    //loglikelihood_study
    void loglikelihood_study(Eigen::VectorXd const & theta, Eigen::VectorXd const & hpars) const;
    void loglikelihood_grid(Eigen::VectorXd const & theta, Eigen::VectorXd const & lb_hparsgrid, Eigen::VectorXd const & ub_hparsgrid, double sobs, int ngrid, std::string filename) const;



    //diagnostic MCMC
    Eigen::VectorXd Lagged_mean(std::vector<Eigen::VectorXd> const &v, int n) const;
    void Autocor_diagnosis(int nstepsmax, std::string const & filename) const;
    void WriteMCMCSamples(std::string const & filename) const;

    void FindVPs(Eigen::MatrixXd const &M) const;
    
    bool in_bounds_pars(Eigen::VectorXd const & pars) const;
    bool in_bounds_hpars(Eigen::VectorXd const & hpars) const;

    //routine d'optimisation
    int optroutine(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars);
    int optroutine_withgrad(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars);
    int optroutine_lightwithgrad(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars);
    int optroutine_light(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars);
    int optroutine_heavy(nlopt::vfunc optfunc,void *data_ptr, Eigen::VectorXd &x, Eigen::VectorXd const & lb_hpars, Eigen::VectorXd const & ub_hpars);

    //calcul des hpars KOH
    Eigen::VectorXd HparsKOH(Eigen::VectorXd const & hpars_guess);
    static double optfuncKOH(const std::vector<double> &x, std::vector<double> &grad, void *data);

    Eigen::VectorXd HparsKOHFromData(Eigen::VectorXd const & hpars_guess,std::vector<Eigen::VectorXd> const & thetas, std::vector<Eigen::VectorXd> const & values);
    static double optfuncKOHFromData(const std::vector<double> &x, std::vector<double> &grad, void *data);

    Eigen::VectorXd HparsNOEDM(Eigen::VectorXd const & hpars_guess);
    static double optfuncNOEDM(const std::vector<double> &x, std::vector<double> &grad, void *data);

    Eigen::VectorXd HparsLOOCV(Eigen::VectorXd const & hpars_guess);
    static double optfuncLOOCV(const std::vector<double> &x, std::vector<double> &grad, void *data);

    //fonctions pour calcul de hessienne

    //static Eigen::MatrixXd ComputeHessian(Eigen::VectorXd const & ref_point,Eigen::VectorXd const & step_size, function<Eigen::VectorXd(Eigen::VectorXd)>const & fprime);
    


    //fonctions d'étude de la postérieure
    Eigen::VectorXd MAP_given_hpars(Eigen::VectorXd const & hpars) const;
    Eigen::VectorXd Mean() const;
    Eigen::MatrixXd Variance() const;
    double Evidence(double normconst) const;

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
    void WritePredictions(Eigen::VectorXd const &X,std::string const & filename) const;
    void WritePredictionsF(Eigen::VectorXd const &X,std::string const & filename) const;
    void WritePriorPredictions(Eigen::VectorXd const &X,std::string const & filename,std::default_random_engine & generator);
    void WritePriorPredictionsF(Eigen::VectorXd const &X,std::string const & filename,std::default_random_engine & generator);
    void WriteFinePredictions(Eigen::VectorXd const &X,std::string const & filename) const;
    void WriteFinePriorPredictions(Eigen::VectorXd const &X,std::string const & filename,std::default_random_engine & generator);


    //accès extérieur
    const std::vector<Eigen::VectorXd> *GetGrid() const {return &m_Grid.m_grid;}
    const std::vector<AUGDATA> *GetExpData() const {return &m_data_exp;}
    const std::vector<Eigen::VectorXd> *GetXconverted() const{return &m_Xprofile_converted;}
    Eigen::VectorXd EvaluateModel(Eigen::VectorXd const &X, Eigen::VectorXd const & theta) const {return m_model(X,theta);}
    double EvaluateLogPHpars(Eigen::VectorXd const & hpars) const {return m_logpriorhpars(hpars);}

    protected:
    std::function<Eigen::VectorXd(Eigen::VectorXd const &,Eigen::VectorXd const &)> m_model;
    std::function<double(Eigen::VectorXd const &,Eigen::VectorXd const &)> m_priormean;
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
    Eigen::VectorXd HparsOpt(Eigen::VectorXd const & theta, Eigen::VectorXd const & hpars_guess);
    Eigen::VectorXd HparsOpt_quick(Eigen::VectorXd const & theta, Eigen::VectorXd const & hpars_guess);
    static double optfuncOpt(const std::vector<double> &x, std::vector<double> &grad, void *data);
    static double optfuncOpt_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data);
    Eigen::VectorXd EvaluateHparOpt(Eigen::VectorXd const & theta) const;

    //Calcul des hpars optimaux sur le grid.
    void Compute_optimal_hpars();
    //avec dérivée
    void Compute_optimal_hpars(nlopt::algorithm alg);

    //Initialisation des GPs. 
    void BuildHGPs(double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &),Eigen::MatrixXd const & Bounds_hpars_GPs,Eigen::VectorXd const & Hpars_guess_GPs,int nmodes);
    void Test_hGPs() ;
    //Optimisation du GP pour les hpars. Il faut autant de GPs que d'hpars de z.
    static double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data);
    void opti_1gp(int i,Eigen::VectorXd & hpars_guess);
    void opti_allgps(Eigen::VectorXd const & hpars_guess);

    //Fonctions de MCMC

    double Run_Burn_phase_MCMC_opti_expensive(int nburn, Eigen::MatrixXd & COV_init,Eigen::VectorXd &Xcurrent,std::default_random_engine & generator);
    void Run_MCMC_opti_expensive(int nsteps,int nsamples,Eigen::VectorXd & Xinit, Eigen::MatrixXd const & COV_init,std::default_random_engine & generator);

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

#endif /*DENSITIES_H*/