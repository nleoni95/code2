#include "gibbs.h"

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
        m_bandwidth[i]=pow(m_data.size(),-1./(m_data[0].size()+4));//scott's rule, uniform bandwidth
    }
};

double KDE::Evaluate(VectorXd const &X) const
{
    double result(0);
    VectorXd norm_X=(X-m_means).cwiseQuotient(m_std);
    for (int i=0;i<m_data.size();i++){
        result+=exp(-0.5*((m_data[i]-norm_X).cwiseQuotient(m_bandwidth)).squaredNorm());
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
        normal[i]=distN(generator); //vecteur d'échantillons de lois normales
    }
    return m_means+m_std.cwiseProduct(m_data[n_selected])+m_bandwidth.cwiseProduct(normal);
}