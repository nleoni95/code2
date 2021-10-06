#ifndef GIBBS_H
#define GIBBS_H

#include <vector>
#include <random>
#include <Eigen/Dense>

class KDE
{
    public:
    KDE(std::vector<Eigen::VectorXd> const &v);
    Eigen::VectorXd Sample(std::default_random_engine &generator) const; //tire un point du KDE
    double Evaluate(Eigen::VectorXd const &X) const; //évalue la densité du KDE.


    protected:
    std::vector<Eigen::VectorXd> m_data; //données centrées et normalisées
    Eigen::VectorXd m_means;
    Eigen::VectorXd m_std;
    Eigen::VectorXd m_bandwidth;
};



#endif /*GIBBS_H*/