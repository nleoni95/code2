#ifndef DATA_H_
#define DATA_H_
#include "data++.h"
#endif


class TPS {
public:
	TPS(){};

	void SetData(const Eigen::VectorXd &X, double F){
			Xpt.push_back(X);
			value.push_back(F);
			nd = Xpt.size();
			std::cout << "The TPS will use " << nd << " support points\n";
	};

	void SetData(const std::vector<DATA> &data){
		Xpt.clear();
        value.clear();
		for(int i=0; i<data.size(); i++){
			Xpt.push_back(data[i].X);
			value.push_back(data[i].F);
		}
		nd = Xpt.size();
		std::cout << "The TPS will use " << nd << " observations\n";
	};

	/*Set the TPS system */
	void Set(){	
		nd = Xpt.size();						//Number of support points
		dim= Xpt[0].size();
		ntot = nd+dim+1;
		Eigen::MatrixXd ThinPb = Eigen::MatrixXd::Zero(ntot,ntot);
		for(unsigned u=0; u<nd; u++){
			for(unsigned v=0; v<nd; v++){
				double r = (Xpt[u]-Xpt[v]).norm();
				ThinPb(u,v) = r*r*log(r);
				if(u==v) ThinPb(u,v) = 0;
			}
			ThinPb(u,nd) 	= 1.;
			for(unsigned id=0; id<dim; id++) ThinPb(u,nd+1+id) 	= Xpt[u](id);
			ThinPb(nd,u) 	= 1.;
			for(unsigned id=0; id<dim; id++) ThinPb(nd+1+id,u) 	= Xpt[u](id);
		}
		LU.compute(ThinPb);
		Eigen::VectorXd Y = Eigen::VectorXd::Zero(ntot);
		for(unsigned it=0; it<nd; it++) Y(it) = value[it]; 
		Alpha = LU.solve(Y);				/* Solve for the GP coordinates*/
		return;
	};

	void SetForNewValues(Eigen::VectorXd const &F){
		Eigen::VectorXd Y = Eigen::VectorXd::Zero(ntot);
		for(unsigned it=0; it<nd; it++) Y(it) = F(it); 
		Alpha = LU.solve(Y);
	}


	double Eval(Eigen::VectorXd const &x){
		Eigen::VectorXd Px(ntot);
		for(unsigned i=0; i<Xpt.size(); i++){
			Px(i) = Phi_TPS(x,Xpt[i]);
		}
		Px(Xpt.size()) = 1;
		for(unsigned i=0; i<x.rows(); i++) Px(Xpt.size()+1+i) = x(i);
		return Px.dot(Alpha);
	};

	double KernelTPS(Eigen::VectorXd const &x, Eigen::VectorXd const &y){
		double l=.5;
		return exp( -((x-y)/l).squaredNorm()*.5 );
	};

	double Phi_TPS(Eigen::VectorXd const &x, Eigen::VectorXd const &y){
		double r = (x-y).norm();
		if(r>0) r = r*r*log(r);
		return r; 
	}

    /* Retrieve dimension of the TPS */
    unsigned Dim() const { return dim; };

    /* Retrieve number of observations */
    unsigned Nobs() const { return Xpt.size(); };

    /* Retrieve location of i-th observation */
    Eigen::VectorXd Xobs(unsigned const i) const { return Xpt[i]; };

    /* Retrieve value of i-th observation */
    double Fobs(unsigned const i) const { return value[i]; };
private:
	int nd;
	unsigned dim;
	unsigned ntot;
	Eigen::FullPivLU<Eigen::MatrixXd> LU;
	Eigen::VectorXd Alpha;
	std::vector<Eigen::VectorXd> Xpt;
	std::vector<double> value;
};
