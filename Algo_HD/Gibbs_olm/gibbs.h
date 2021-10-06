#ifndef GIBBS_H
#define GIBBS_H

#include "densities.h"
#include <ctime>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <iostream>
#include <string>

class KDE
{
    public:
    KDE(std::vector<Eigen::VectorXd> const &v);
    Eigen::VectorXd Sample(std::default_random_engine &generator) const; //tire un point du KDE
    double Evaluate(Eigen::VectorXd const &X) const; //évalue la densité du KDE.
    void FidTest(std::default_random_engine &generator) const; //teste si le KDE est fidèle à ses échantillons.


    protected:
    std::vector<Eigen::VectorXd> m_data; //données centrées et normalisées
    Eigen::VectorXd m_means;
    Eigen::VectorXd m_std;
    Eigen::VectorXd m_bandwidth;
};

class MCMC_par : public Density
{
    //MCMC spécifiquement sur les paramètres, à valeurs d'hpars fixées.
    public:
    MCMC_par(Density const &d, Eigen::VectorXd const &hpars, int nchain);
    void Run(Eigen::VectorXd const &Xinit, Eigen::MatrixXd const &COV,std::default_random_engine &generator);
    void SelectSamples(int nsamples);

    //affichage 
    void WriteAllSamples(const char* file_name) const;
    void WriteSelectedSamples(const char* file_name) const;

    //accesseurs
    std::vector<Eigen::VectorXd> GetSelectedSamples() const {return m_selected_samples;}
  
    //tests d'autocorrélation
    void Autocorrelation_diagnosis(int n); // /!\ modifie les m_all_samples pour centrer les données !
    Eigen::VectorXd Lagged_mean(int n) const; //renvoie la moyenne courante avec un lag de n.

    double loglikelihood_theta_fast(Eigen::VectorXd const &theta) const;
    Eigen::VectorXd GetHpars() const {return m_hpars;};

    protected:
    bool in_bounds(Eigen::VectorXd &X);
    
    Eigen::LDLT<Eigen::MatrixXd> m_ldlt;
    Eigen::VectorXd m_hpars; //hyperparamètres fixés pour la MCMC
    int m_nchain; //nombre d'itérations de la MCMC
    int m_dim_mcmc; //dimension theta
    std::vector<Eigen::VectorXd> m_all_samples; //tous les états de la MCMC.
    std::vector<Eigen::VectorXd> m_selected_samples; //échantillon récupéré de la chaîne
    std::vector<double> m_all_values; //toutes les valeurs de la MCMC
    std::vector<double> m_selected_values; // valeurs récupérées de la chaîne.
};



#endif /*GIBBS_H*/