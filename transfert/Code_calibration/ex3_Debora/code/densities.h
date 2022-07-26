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


Eigen::VectorXd VtoVXD(std::vector<double> const &v);
std::vector<double> VXDtoV(Eigen::VectorXd const &X);
std::vector<Eigen::VectorXd> VdtoVVXd(std::vector<double> &v);

void WriteVector(std::vector<Eigen::VectorXd> const &v, std::ofstream &ofile);
void WriteVectors(std::vector<Eigen::VectorXd> const &v1, std::vector<Eigen::VectorXd> const &v2, std::ofstream &ofile);
void WriteVectors(std::vector<Eigen::VectorXd> const &v1, std::vector<double> const &v2, std::ofstream &ofile);
void WriteVectors(std::vector<Eigen::VectorXd> const &v1, Eigen::VectorXd const &v2, std::ofstream &ofile);
void WriteVectors(std::vector<Eigen::VectorXd> const &v1, std::vector<Eigen::VectorXd> const &v2, std::vector<double> const &v3, std::ofstream &ofile);
std::vector<Eigen::VectorXd> ReadVector(std::string const &filename);

/*Fonctions d'optimisation*/

double optroutine(nlopt::vfunc optfunc, void *data_ptr, Eigen::VectorXd &X, Eigen::VectorXd const &lb_hpars, Eigen::VectorXd const &ub_hpars, double ftol_rel);
double optroutine_withgrad(nlopt::vfunc optfunc, void *data_ptr, Eigen::VectorXd &X, Eigen::VectorXd const &lb_hpars, Eigen::VectorXd const &ub_hpars, double ftol_rel);

double optfuncKOH(const std::vector<double> &x, std::vector<double> &grad, void *data);
double optfuncOpt_nograd(const std::vector<double> &x, std::vector<double> &grad, void *data);
double optfuncOpt_withgrad(const std::vector<double> &x, std::vector<double> &grad, void *data);
double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data);

void OptimizeGPBis(GP &gp, Eigen::VectorXd &guess, Eigen::VectorXd const &lb, Eigen::VectorXd const &ub, double ftol);

/*Fonctions MCMC*/
void Run_Burn_Phase_MCMC(int nburn, Eigen::MatrixXd &COV_init, Eigen::VectorXd &Xcurrento, std::function<double(std::vector<Eigen::VectorXd>, Eigen::VectorXd const &)> const &compute_score, std::function<std::vector<Eigen::VectorXd>(Eigen::VectorXd const &)> const &get_hpars, std::function<bool(Eigen::VectorXd)> const &in_bounds, std::default_random_engine &generator);
std::vector<Eigen::VectorXd> Run_MCMC(int nsteps, Eigen::VectorXd &Xinit, Eigen::MatrixXd COV_init, std::function<double(std::vector<Eigen::VectorXd>, Eigen::VectorXd const &)> const &compute_score, std::function<std::vector<Eigen::VectorXd>(Eigen::VectorXd const &)> const &get_hpars, std::function<bool(Eigen::VectorXd)> const &in_bounds, std::default_random_engine &generator);
Eigen::VectorXd Lagged_mean(std::vector<Eigen::VectorXd> const &v, int n);
double Selfcor_diagnosis(std::vector<Eigen::VectorXd> const &samples, int nstepsmax, double proportion, std::string const &filename);

class DoE
{
    friend class Density;

public:
    DoE();
    DoE(Eigen::VectorXd const &lb, Eigen::VectorXd const &ub, int n);                                             // construire un DoE uniforme
    DoE(Eigen::VectorXd const &lb, Eigen::VectorXd const &ub, int npts, std::default_random_engine &generator); // construire un DoE LHS
    DoE(Eigen::VectorXd const &lb, Eigen::VectorXd const &ub, int npts, int first_element);                    // construire un DoE QMC

    std::vector<Eigen::VectorXd> GetGrid() const { return m_grid; }

    Eigen::VectorXd Randperm(int n, std::default_random_engine &generator);
    Eigen::VectorXd Multiindex(int const s, int const n, int const d);


protected:
    Eigen::VectorXd m_lb_pars;           // bornes inf des paramètres
    Eigen::VectorXd m_ub_pars;           // bornes sup des paramètres
    std::vector<Eigen::VectorXd> m_grid; // valeurs du DoE
};

class Density
{
public:
    Density();
    Density(DoE const &g);
    Density(Density const &d);

    /*Appels obligatoires */
    void SetObservations(std::vector<Eigen::VectorXd> const &Xlocations, Eigen::VectorXd const &observations);
    void SetFModel(std::function<Eigen::VectorXd(std::vector<Eigen::VectorXd> const &, Eigen::VectorXd const &)> my_model) { m_model = my_model; }
    void SetZKernel(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &)> Kernel) { m_Kernel = Kernel; };
    void SetZPriorMean(std::function<Eigen::VectorXd(std::vector<Eigen::VectorXd> const &, Eigen::VectorXd const &)> priormean) { m_priormean = priormean; }
    void SetLogPriorPars(std::function<double(Eigen::VectorXd const &)> logpriorpars) { m_logpriorpars = logpriorpars; };
    void SetLogPriorHpars(std::function<double(Eigen::VectorXd const &)> logpriorhpars) { m_logpriorhpars = logpriorhpars; };
    void SetHparsBounds(Eigen::VectorXd const &lb_hpars, Eigen::VectorXd const &ub_hpars)
    {
        m_dim_hpars = lb_hpars.size();
        m_lb_hpars = lb_hpars;
        m_ub_hpars = ub_hpars;
    };

    /*Appels pour fixer ou apprendre l'erreur sur l'input ou l'output des observations.
    Si pas d'appel à ces méthodes : les erreurs sont à 0. */
    void SetFixedOutputerr(double logstdoutputerr);
    void SetLearnedOutputerr(int index);
    void SetFixedInputerr(double logstdinputerr, Eigen::VectorXd model_derivatives_at_obs);
    void SetLearnedInputerr(int index, Eigen::VectorXd mode_derivatives_at_obs);

    /* Appels non nécessaires (pour commodité) */
    void SetNewDoE(DoE const &g);
    void SetNewSamples(std::vector<Eigen::VectorXd> const &s) { m_samples = s; }
    void SetNewHparsOfSamples(std::vector<Eigen::VectorXd> const &s) { m_hparsofsamples = s; }
    void SetNewAllSamples(std::vector<Eigen::VectorXd> const &s) { m_allmcmcsamples = s; }

    /* Calcul de la vraisemblance */
    Eigen::MatrixXd Gamma(std::vector<Eigen::VectorXd> const &locs, Eigen::VectorXd const &hpar) const;
    double loglikelihood_fast(Eigen::VectorXd const &obs, Eigen::LDLT<Eigen::MatrixXd> const &ldlt) const;
    double loglikelihood_theta_fast(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars, Eigen::LDLT<Eigen::MatrixXd> const &ldlt) const;
    double loglikelihood_theta(Eigen::VectorXd const &theta, Eigen::VectorXd const &hpar) const;

    /* Tests si paramètres et hyperparamètres sont dans les bornes */
    bool in_bounds_pars(Eigen::VectorXd const &pars) const;
    bool in_bounds_hpars(Eigen::VectorXd const &hpars) const;

    /* Calcul des hyperparamètres KOH */
    Eigen::VectorXd HparsKOH(Eigen::VectorXd const &hpars_guess, double logvs_typ ,double ftol_rel) const;

    /* Calcul et écriture de predictions */
    void WritePredictions(std::vector<Eigen::VectorXd> const &Xpredictions, std::string const &filename) const;
    void WriteSamplesFandZ(std::vector<Eigen::VectorXd> const &Xpredictions, std::string const &filenameF, std::string const &filenameZ,int nsamples) const;

    /* Accès externe */
    const std::vector<Eigen::VectorXd> *GetGrid() const { return &m_Grid.m_grid;}
    const std::vector<Eigen::VectorXd> *GetXlocations() const { return &m_Xlocations;}
    const Eigen::VectorXd *GetObs() const { return &m_observations;}
    std::pair<Eigen::VectorXd, Eigen::VectorXd> GetBoundsHpars() const { return std::make_pair(m_lb_hpars, m_ub_hpars); }
    Eigen::VectorXd EvaluateModel(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta) const { return m_model(X, theta); }
    Eigen::VectorXd EvaluatePMean(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &hpars) const { return m_priormean(X, hpars); }
    double EvaluateLogPHpars(Eigen::VectorXd const &hpars) const { return m_logpriorhpars(hpars); }
    double EvaluateLogPPars(Eigen::VectorXd const &pars) const { return m_logpriorpars(pars); }
    double GetInputerr(Eigen::VectorXd const &hpars) const;
    double GetOutputerr(Eigen::VectorXd const &hpars) const;
    int GetPresenceInputerr() const { return m_presence_inputerr; }
    Eigen::MatrixXd GetDerMatrix() const { return m_derivatives_obs; }

protected:
    /* Prédictions */
    Eigen::VectorXd meanF(std::vector<Eigen::VectorXd> const &X) const;
    Eigen::VectorXd meanZCondTheta(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars_z) const;
    Eigen::MatrixXd varZCondTheta(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars_z) const;
    Eigen::MatrixXd varF(std::vector<Eigen::VectorXd> const &X) const;
    Eigen::MatrixXd PredFZ(std::vector<Eigen::VectorXd> const &X) const;
    Eigen::VectorXd DrawZCondTheta(std::vector<Eigen::VectorXd> const &X, Eigen::VectorXd const &theta, Eigen::VectorXd const &hpars_z, std::default_random_engine &generator) const;

    int m_presence_inputerr = 0;  // présence d'input error. 0 : nulle, 1 : à valeur fixe, 2 : apprise.
    int m_presence_outputerr = 0;  // présence d'output error. 0 : nulle, 1 : à valeur fixe, 2 : apprise.
    double m_inputerr;         // logarithme de la std. d'input error
    double m_outputerr;        // logarithme de la std de l'output error
    int m_indexinputerr;               // indice dans hpars de l'input error
    int m_indexoutputerr;              // indice dans hpars de l'output error
    Eigen::MatrixXd m_derivatives_obs; // dérivées de y aux points d'observation
    DoE m_Grid;                        // Ensemble des paramètres servant à calculer HparsKOH.

    // Computer model, priors, dérivées
    std::function<Eigen::VectorXd(std::vector<Eigen::VectorXd> const &, Eigen::VectorXd const &)> m_model;     // modèle F
    std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &)> m_Kernel; // noyau de Z
    std::function<Eigen::VectorXd(std::vector<Eigen::VectorXd> const &, Eigen::VectorXd const &)> m_priormean; // prior mean de Z
    std::function<double(Eigen::VectorXd const &)> m_logpriorpars;                                             // log de la densité a priori des paramètres
    std::function<double(Eigen::VectorXd const &)> m_logpriorhpars;                                            // log de la densité a priori des hyperparamètres

    std::vector<std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &)>> m_Kernel_derivatives; // dérivée du noyau de Z par rapport aux hyperparamètres

    Eigen::VectorXd m_lb_hpars; // borne inf hyperparamètres
    Eigen::VectorXd m_ub_hpars; // borne sup hyperparamètres
    Eigen::VectorXd m_lb_pars;  // borne inf paramètres
    Eigen::VectorXd m_ub_pars;  // borne sup paramètres
    int m_dim_hpars;            // dimension hyperparamètres
    int m_dim_pars;             // dimension paramètres

    std::vector<Eigen::VectorXd> m_Xlocations; // points d'observations
    Eigen::VectorXd m_observations;            // valeurs des observations

    std::vector<Eigen::VectorXd> m_allmcmcsamples; // échantillon complet de la MCMC
    std::vector<Eigen::VectorXd> m_samples;        //échantillon de la MCMC après thinning
    std::vector<Eigen::VectorXd> m_hparsofsamples; // hpars correspondant à l'échantillon après thinning
};

class DensityOpt : public Density
{
public:
    /* Obligatoire : Construction par copie d'un objet Density */
    DensityOpt(Density const &d);

    /* Appels nécessaires pour utiliser la méthode de recherche d'hyperparamètres optimaux avec gradients (HparsOpt_withgrad) */
    void SetZKernelGrads(std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> Zkernel_gradients) { m_Zkernel_gradients = Zkernel_gradients; }
    void SetLogpriorHparsGrads(std::function<double(Eigen::VectorXd const &, int i)> logpriorhpars_gradients) { m_logpriorhpars_gradients = logpriorhpars_gradients; }

    /* Construction des hGPs */
    void BuildHGPs(std::vector<Eigen::VectorXd> const &thetas, std::vector<Eigen::VectorXd> const &hpars_optimaux, double (*Kernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &), double (*DKernel_GP)(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int));
    void OptimizeHGPs(Eigen::MatrixXd Bounds_hpars_GPs, std::vector<Eigen::VectorXd> Hpars_guess_GPs, double ftol);
    void SetHGPs(std::vector<Eigen::VectorXd> Hpars_guess_GPs);

    /* Calcul d'hyperparamètres optimaux */
    Eigen::VectorXd HparsOpt(Eigen::VectorXd const &theta, Eigen::VectorXd hpars_guess, double ftol_rel) const;                                                      // sans utiliser le gradient
    Eigen::VectorXd HparsOpt_withgrad(Eigen::VectorXd const &theta, Eigen::VectorXd hpars_guess, double ftol_rel) const;                                             // avec utilisation du gradient
    Eigen::VectorXd EvaluateHparOpt(Eigen::VectorXd const &theta) const;                                                                                             // prédiction avec hGPs (prédiction moyenne)
    std::vector<std::vector<Eigen::VectorXd>> SampleHparsOpt(std::vector<Eigen::VectorXd> const &thetas, int nsamples, std::default_random_engine &generator) const; // tirage des hGPs

    /* Accès extérieur */
    std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> GetZKernelGrads() const { return m_Zkernel_gradients; }
    std::function<double(Eigen::VectorXd const &, int i)> GetLogpriorHparsGrads() const { return m_logpriorhpars_gradients; }
    std::vector<Eigen::VectorXd> GetHparsHGPs() const
    {
        std::vector<Eigen::VectorXd> v;
        for (int i = 0; i < m_hGPs.size(); i++)
        {
            v.push_back(m_hGPs[i].GetPar());
        }
        return v;
    }

protected:
    std::vector<GP> m_hGPs;                                                                                                      // hGPs
    std::function<double(Eigen::VectorXd const &, Eigen::VectorXd const &, Eigen::VectorXd const &, int i)> m_Zkernel_gradients; // gradients du noyau de Z par rapport aux hpars
    std::function<double(Eigen::VectorXd const &, int i)> m_logpriorhpars_gradients;                                             // gradients du logprior des hpars de Z par rapport aux hpars
};

#endif /*DENSITIES_H*/
