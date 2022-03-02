/*Nouvelle version de densities. Dans celle-ci, on considère une prédiction vectorielle du modèle. On va chercher à obtenir des échantillons des densités par des MCMC, au lieu de les évaluer par quadrature. */
#ifndef DENSITIES_H
#define DENSITIES_H

#include <vector>
#include <map>
#include <list>
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

/*utility functions*/
Eigen::VectorXd VtoVXD(std::vector<double> const &v);
std::vector<double> VXDtoV(Eigen::VectorXd const &X);
double FindQuantile(double pct, Eigen::VectorXd const &X);
void WriteVector(std::vector<Eigen::VectorXd> const &v, std::string const &filename);
double optroutine(nlopt::vfunc optfunc, void *data_ptr, Eigen::VectorXd &X, Eigen::VectorXd const &lb_hpars, Eigen::VectorXd const &ub_hpars, double max_time);
double optroutine_withgrad(nlopt::vfunc optfunc, void *data_ptr, Eigen::VectorXd &X, Eigen::VectorXd const &lb_hpars, Eigen::VectorXd const &ub_hpars, double max_time);

/*functions related to optimization*/
double optfuncKOH(const std::vector<double> &x, std::vector<double> &grad, void *data);
double optfuncOpt_nograd(const std::vector<double> &x, std::vector<double> &grad, void *data);
double optfuncOpt_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data);
double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data);

/*MCMC-related functions*/
void Run_Burn_Phase_MCMC(int nburn, Eigen::MatrixXd &COV_init, Eigen::VectorXd &Xcurrento, std::function<double(std::vector<Eigen::VectorXd>, Eigen::VectorXd const &)> const &compute_score, std::function<std::vector<Eigen::VectorXd>(Eigen::VectorXd const &)> const &get_hpars, std::function<bool(Eigen::VectorXd)> const &in_bounds, std::default_random_engine &generator);

std::vector<Eigen::VectorXd> Run_MCMC(int nsteps, Eigen::VectorXd const &Xinit, Eigen::MatrixXd const &COV_init, std::function<double(std::vector<Eigen::VectorXd>, Eigen::VectorXd const &)> const &compute_score, std::function<std::vector<Eigen::VectorXd>(Eigen::VectorXd const &)> const &get_hpars, std::function<bool(Eigen::VectorXd)> const &in_bounds, std::default_random_engine &generator);

//diagnosis functions
Eigen::VectorXd Lagged_mean(std::vector<Eigen::VectorXd> const &v, int n);
void Selfcor_diagnosis(std::vector<Eigen::VectorXd> const &samples, int nstepsmax, double proportion, std::string const &filename);

class AUGDATA
{
public:
    AUGDATA(){};
    AUGDATA(Eigen::VectorXd const &x, Eigen::VectorXd const &f)
    {
        X = x;
        F = f;
    };
    AUGDATA(AUGDATA const &d)
    {
        X = d.X;
        F = d.F;
    };
    void operator=(const AUGDATA d)
    {
        X = d.X;
        F = d.F;
    };
    Eigen::VectorXd GetX() const { return X; };
    Eigen::VectorXd Value() const { return F; };
    void SetX(Eigen::VectorXd x) { X = x; };
    void SetValue(Eigen::VectorXd f) { F = f; };
    std::vector<DATA> split() const
    {
        std::vector<DATA> v(F.size());
        DATA dat;
        dat.SetX(X);
        for (int i = 0; i < F.size(); i++)
        {
            dat.SetValue(F(i));
            v[i] = dat;
        }
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
    DoE(Eigen::VectorXd const &lb, Eigen::VectorXd const &ub, int n);                                             //constructeur en grid sampling
    DoE(Eigen::VectorXd const &lb, Eigen::VectorXd const &ub, int ntotal, std::default_random_engine &generator); //constructeur en LHS
    DoE(Eigen::VectorXd const &lb, Eigen::VectorXd const &ub, int npoints, int first_element);                    //constructeur en QMC

    void WriteGrid(std::string const &filename) const;
    std::vector<Eigen::VectorXd> GetGrid() const { return m_grid; }

    //utility
    Eigen::VectorXd Randpert(int n, std::default_random_engine &generator) const;
    Eigen::VectorXd indices(int const s, int const n, int const d);
    
protected:
    Eigen::VectorXd m_lb_pars;           //bornes inf des paramètres
    Eigen::VectorXd m_ub_pars;           //bornes sup des paramètres
    std::vector<Eigen::VectorXd> m_grid; //valeurs des paramètres
};

class Density
{
public:
    //construction
    Density();
    Density(DoE const &g);
    Density(Density const &d);

    //necessary calls to initialize the object
    void SetLogPriorPars(std::function<double(Eigen::VectorXd const &)> logpriorpars) { m_logpriorpars = logpriorpars; };
    void SetLogPriorHpars(std::function<double(Eigen::VectorXd const &)> logpriorhpars) { m_logpriorhpars = logpriorhpars; };
    void SetKernel(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &)> Kernel) { m_Kernel = Kernel; };
    void SetModel(std::function<Eigen::VectorXd(std::vector<Eigen::VectorXd> const &, Eigen::VectorXd const &)> my_model) { m_model = my_model; }
    void SetPriorMean(std::function<Eigen::VectorXd(std::vector<Eigen::VectorXd> const &, Eigen::VectorXd const &)> priormean) { m_priormean = priormean; }
    void SetHparsBounds(Eigen::VectorXd const &lb_hpars, Eigen::VectorXd const &ub_hpars)
    {
        m_dim_hpars = lb_hpars.size();
        m_lb_hpars = lb_hpars;
        m_ub_hpars = ub_hpars;
    };
    void SetObservations(std::vector<Eigen::VectorXd> const &Xlocations, Eigen::VectorXd const &observations);

    //non-necessary calls, for convenience.
    void SetDoE(DoE const &g);
    void SetNewSamples(std::vector<Eigen::VectorXd> const &s) { m_samples = s; }
    void SetNewHparsOfSamples(std::vector<Eigen::VectorXd> const &s) { m_hparsofsamples = s; }
    void SetNewAllSamples(std::vector<Eigen::VectorXd> const &s) { m_allmcmcsamples = s; }

    //likelihood-related functions
    Eigen::MatrixXd Gamma(std::vector<Eigen::VectorXd> const &locs, Eigen::VectorXd const &hpar) const;
    double loglikelihood_fast(Eigen::VectorXd const &obs, Eigen::LDLT<Eigen::MatrixXd> const &ldlt) const;
    double loglikelihood_theta_fast(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars, Eigen::LDLT<Eigen::MatrixXd> const &ldlt) const;
    double loglikelihood_theta(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar) const;

    bool in_bounds_pars(Eigen::VectorXd const &pars) const;
    bool in_bounds_hpars(Eigen::VectorXd const &hpars) const;

    //calcul des hpars KOH
    Eigen::VectorXd HparsKOH(Eigen::VectorXd const &hpars_guess, double max_time) const;

    //write the samples
    void WriteSamples(std::string const &filename) const;

    //predictions
    void WritePredictions(std::vector<Eigen::VectorXd> const &Xpredictions, std::string const &filename) const;
    void WritePredictionsF(std::vector<Eigen::VectorXd> const &Xpredictions, std::string const &filename) const;
    void WriteSamplesFandZ(std::vector<Eigen::VectorXd> const &Xpredictions, std::string const &filenameF, std::string const &filenameZ) const;

    //external access
    const std::vector<Eigen::VectorXd> *GetGrid() const { return &m_Grid.m_grid; }
    const std::vector<Eigen::VectorXd> *GetXlocations() const { return &m_Xlocations; }
    Eigen::LDLT<Eigen::MatrixXd> GetLDLT(Eigen::VectorXd const &hpars);
    std::pair<Eigen::VectorXd, Eigen::VectorXd> GetBoundsHpars() const { return std::make_pair(m_lb_hpars, m_ub_hpars); }
    Eigen::VectorXd EvaluateModel(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta) const { return m_model(X, theta); }
    Eigen::VectorXd EvaluatePMean(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &hpars) const { return m_priormean(X, hpars); }
    double EvaluateLogPHpars(Eigen::VectorXd const &hpars) const { return m_logpriorhpars(hpars); }
    double EvaluateLogPPars(Eigen::VectorXd const &pars) const { return m_logpriorpars(pars); }

    //internal methods
    Eigen::VectorXd meanF(std::vector<Eigen::VectorXd> const &X) const;
    Eigen::VectorXd meanZCondTheta(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars_z) const;
    Eigen::MatrixXd varZCondTheta(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars_z) const;
    Eigen::MatrixXd varF(std::vector<Eigen::VectorXd> const &X) const;
    Eigen::MatrixXd PredFZ(std::vector<Eigen::VectorXd> const &X) const;
    Eigen::VectorXd DrawZCondTheta(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars_z, std::default_random_engine &generator) const;

protected:
    DoE m_Grid;
    double m_inputerr;
    Eigen::VectorXd m_derivatives_obs;
    Eigen::VectorXd m_derivatives_preds;

    //model, priors, kernel and its derivatives
    std::function<Eigen::VectorXd(std::vector<Eigen::VectorXd> const &, Eigen::VectorXd const &)> m_model;
    std::function<Eigen::VectorXd(std::vector<Eigen::VectorXd> const &, Eigen::VectorXd const &)> m_priormean;
    std::function<double(Eigen::VectorXd const &)> m_logpriorpars;
    std::function<double(Eigen::VectorXd const &)> m_logpriorhpars;
    std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &)> m_Kernel;
    std::vector<std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &)>> m_Kernel_derivatives;

    //bounds of parameters & hyperparameters
    Eigen::VectorXd m_lb_hpars;
    Eigen::VectorXd m_ub_hpars;
    Eigen::VectorXd m_lb_pars;
    Eigen::VectorXd m_ub_pars;
    int m_dim_hpars;
    int m_dim_pars;

    //observations and their locations
    std::vector<Eigen::VectorXd> m_Xlocations;
    Eigen::VectorXd m_observations;

    //samples from the MCMC
    std::vector<Eigen::VectorXd> m_samples;
    std::vector<Eigen::VectorXd> m_hparsofsamples;
    std::vector<Eigen::VectorXd> m_allmcmcsamples;
};

class DensityOpt : public Density
{
public:
    DensityOpt(Density const &d);

    //necessary call to use derivatives to compute optimal hyperparameters
    void SetKernelDerivatives(std::vector<std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &)>> vector_derivatives) {}

    //calcul des hpars optimaux
    Eigen::VectorXd HparsOpt(Eigen::VectorXd const &theta, Eigen::VectorXd hpars_guess, double max_time) const;
    Eigen::VectorXd HparsOpt_withgrad(Eigen::VectorXd const &theta, Eigen::VectorXd hpars_guess, double max_time) const;

    Eigen::VectorXd EvaluateHparOpt(Eigen::VectorXd const &theta) const;

    //Calcul des hpars optimaux sur le grid.
    std::vector<Eigen::VectorXd> Compute_optimal_hpars(double max_time, std::string filename);
    std::vector<Eigen::VectorXd> Return_optimal_hpars(double max_time) const;

    //Initialisation des GPs.
    void BuildHGPs_noPCA(double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), Eigen::MatrixXd const &Bounds_hpars_GPs, Eigen::VectorXd const &Hpars_guess_GPs);
    void Test_hGPs(int npoints, double max_time);
    Eigen::VectorXd Test_hGPs_on_sample(std::vector<Eigen::VectorXd> const &theta_ref, std::vector<Eigen::VectorXd> const &hpars_ref) const;
    //Optimisation du GP pour les hpars. Il faut autant de GPs que d'hpars de z.
    static double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data);
    Eigen::VectorXd opti_1gp(int i, Eigen::VectorXd &hpars_guess);
    void opti_allgps(Eigen::VectorXd const &hpars_guess);
    //version où on donne les hpars
    void update_hGPs_noPCA(std::vector<Eigen::VectorXd> const &new_thetas, std::vector<Eigen::VectorXd> const &new_hpars, double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), Eigen::MatrixXd const &Bounds_hpars_GPs, Eigen::VectorXd const &Hpars_guess_GPs);
    //version où on calcule les hpars. et on les rend aussi.
    std::vector<Eigen::VectorXd> update_hGPs_noPCA(std::vector<Eigen::VectorXd> const &new_thetas, double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), Eigen::MatrixXd const &Bounds_hpars_GPs, Eigen::VectorXd const &Hpars_guess_GPs, double max_time);

    Eigen::VectorXd EvaluateVarHparOpt(Eigen::VectorXd const &theta) const;
    double EstimatePredError(Eigen::VectorXd const &theta) const;

    //sauvegarde des hpars des gps.
    void WritehGPs(std::string const &filename) const;
    void ReadhGPs(std::string const &filename);

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
