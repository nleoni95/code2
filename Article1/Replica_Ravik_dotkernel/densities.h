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

typedef std::pair<Eigen::VectorXd, Eigen::VectorXd> AUGDATA;

/*utility functions*/
Eigen::VectorXd VtoVXD(std::vector<double> const &v);
std::vector<double> VXDtoV(Eigen::VectorXd const &X);
double FindQuantile(double pct, Eigen::VectorXd const &X);
void WriteVector(std::vector<Eigen::VectorXd> const &v, std::string const &filename);
void WriteVectors(std::vector<Eigen::VectorXd> const &v1, std::vector<Eigen::VectorXd> const &v2, std::string const &filename);
void WriteVectors(std::vector<Eigen::VectorXd> const &v1, std::vector<Eigen::VectorXd> const &v2,std::vector<Eigen::VectorXd> const &v3, std::string const &filename);
std::vector<Eigen::VectorXd> ReadVector(std::string const &filename);
double optroutine(nlopt::vfunc optfunc, void *data_ptr, Eigen::VectorXd &X, Eigen::VectorXd const &lb_hpars, Eigen::VectorXd const &ub_hpars, double max_time);
double optroutine_withgrad(nlopt::vfunc optfunc, void *data_ptr, Eigen::VectorXd &X, Eigen::VectorXd const &lb_hpars, Eigen::VectorXd const &ub_hpars, double max_time);

/*functions related to optimization*/
double optfuncKOH(const std::vector<double> &x, std::vector<double> &grad, void *data);
double optfuncOpt_nograd(const std::vector<double> &x, std::vector<double> &grad, void *data);
double optfuncOpt_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data);
double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data);

/*MCMC-related functions*/
void Run_Burn_Phase_MCMC(int nburn, Eigen::MatrixXd &COV_init, Eigen::VectorXd &Xcurrento, std::function<double(std::vector<Eigen::VectorXd>, Eigen::VectorXd const &)> const &compute_score, std::function<std::vector<Eigen::VectorXd>(Eigen::VectorXd const &)> const &get_hpars, std::function<bool(Eigen::VectorXd)> const &in_bounds, std::default_random_engine &generator);

std::vector<Eigen::VectorXd> Run_MCMC(int nsteps, Eigen::VectorXd  &Xinit, Eigen::MatrixXd &COV_init, std::function<double(std::vector<Eigen::VectorXd>, Eigen::VectorXd const &)> const &compute_score, std::function<std::vector<Eigen::VectorXd>(Eigen::VectorXd const &)> const &get_hpars, std::function<bool(Eigen::VectorXd)> const &in_bounds, std::default_random_engine &generator);

std::vector<Eigen::VectorXd> Run_MCMC_hundred(int nsteps, Eigen::VectorXd  &Xinit, Eigen::MatrixXd &COV_init, std::function<double(std::vector<Eigen::VectorXd>, Eigen::VectorXd const &)> const &compute_score, std::function<std::vector<Eigen::VectorXd>(Eigen::VectorXd const &)> const &get_hpars, std::function<bool(Eigen::VectorXd)> const &in_bounds, std::default_random_engine &generator);

//diagnosis functions
Eigen::VectorXd Lagged_mean(std::vector<Eigen::VectorXd> const &v, int n);
void Selfcor_diagnosis(std::vector<Eigen::VectorXd> const &samples, int nstepsmax, double proportion, std::string const &filename);

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

    //necessary calls to include input and output Gaussian error, with fixed value or learned.
    void SetOutputerr(bool learned, double value, int index);
    void SetInputerr(bool learned, double value, int index, Eigen::VectorXd derivatives_at_obs,Eigen::VectorXd derivatives_at_preds);

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
    void WriteSamplesFandZ(std::vector<Eigen::VectorXd> const &Xpredictions, std::string const &filenameF, std::string const &filenameZ) const;

    //derivatives

    //external access
    const std::vector<Eigen::VectorXd> *GetGrid() const { return &m_Grid.m_grid; }
    const std::vector<Eigen::VectorXd> *GetXlocations() const { return &m_Xlocations; }
    const std::vector<Eigen::VectorXd> GetXlocs() const { return m_Xlocations; }
    std::pair<Eigen::VectorXd, Eigen::VectorXd> GetBoundsHpars() const { return std::make_pair(m_lb_hpars, m_ub_hpars); }
    Eigen::VectorXd EvaluateModel(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta) const { return m_model(X, theta); }
    Eigen::VectorXd EvaluatePMean(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &hpars) const { return m_priormean(X, hpars); }
    double EvaluateLogPHpars(Eigen::VectorXd const &hpars) const { return m_logpriorhpars(hpars); }
    double EvaluateLogPPars(Eigen::VectorXd const &pars) const { return m_logpriorpars(pars); }
    double GetInputerr(Eigen::VectorXd const &hpars) const;
    double GetOutputerr(Eigen::VectorXd const &hpars) const;
    bool GetPresenceInputerr() const {return m_presence_inputerr;}
    Eigen::MatrixXd GetDerMatrix() const {return m_derivatives_obs;}
    Eigen::VectorXd GetYobs() const {return m_observations;}

protected:
    //internal methods
    Eigen::VectorXd meanF(std::vector<Eigen::VectorXd> const &X) const;
    Eigen::VectorXd meanZCondTheta(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars_z) const;
    Eigen::MatrixXd varZCondTheta(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars_z) const;
    Eigen::MatrixXd varF(std::vector<Eigen::VectorXd> const &X) const;
    Eigen::MatrixXd PredFZ(std::vector<Eigen::VectorXd> const &X) const;
    Eigen::VectorXd DrawZCondTheta(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars_z, std::default_random_engine &generator) const;

    //input error and output error
    bool m_presence_inputerr = false;
    double m_inputerr = 0;
    double m_outputerr = 0;
    int m_indexinputerr;
    int m_indexoutputerr;
    //derivatives of the true process at the observation points
    Eigen::MatrixXd m_derivatives_obs;
    Eigen::MatrixXd m_derivatives_preds;

    DoE m_Grid;

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

    //two calls that are necessary to use the gradient-based algorithm (HparsOpt_withgrad)
    void SetKernelGrads(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> kernel_gradients){m_kernel_gradients=kernel_gradients;}
    void SetLogpriorGrads(std::function<double(Eigen::VectorXd const &, int i)> logprior_gradients){m_logprior_gradients=logprior_gradients;}

    //compute optimal hyperparameters
    Eigen::VectorXd HparsOpt(Eigen::VectorXd const &theta, Eigen::VectorXd hpars_guess, double max_time) const;          //without using gradients
    Eigen::VectorXd HparsOpt_withgrad(Eigen::VectorXd const &theta, Eigen::VectorXd hpars_guess, double max_time) const; //using gradients
    Eigen::VectorXd EvaluateHparOpt(Eigen::VectorXd const &theta) const;                                                 //by interrogating the GPs

    //Construction des hGPs. faisons sans normalisation pour le moment.
    void BuildHGPs(std::vector<Eigen::VectorXd> const & thetas,std::vector<Eigen::VectorXd> const & hpars_optimaux,
    double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &));
    void OptimizeHGPs(Eigen::MatrixXd Bounds_hpars_GPs, Eigen::VectorXd Hpars_guess_GPs,double time);
    void OptimizeHGPs(Eigen::MatrixXd Bounds_hpars_GPs, std::vector<Eigen::VectorXd> Hpars_guess_GPs,double time);

    Eigen::VectorXd EvaluateVarHparOpt(Eigen::VectorXd const &theta) const;
    double EstimatePredError(Eigen::VectorXd const &theta) const;

    std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> GetKernelGrads()const{return m_kernel_gradients;}
    std::function<double(Eigen::VectorXd const &, int i)> GetLogpriorGrads()const{return m_logprior_gradients;}
    std::vector<Eigen::VectorXd> GetHparsHGPs() const {std::vector<Eigen::VectorXd> v; for(int i=0;i<m_hGPs.size();i++){v.push_back(m_hGPs[i].GetPar());} return v;}

protected:
    std::vector<GP> m_hGPs;
    Eigen::VectorXd m_means_hGPs;
    Eigen::VectorXd m_scales_hGPs;
    std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> m_kernel_gradients;
    std::function<double(Eigen::VectorXd const &, int i)> m_logprior_gradients;
};

#endif /*DENSITIES_H*/
