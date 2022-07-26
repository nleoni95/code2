#ifndef GLH_H_
#include "gauss.h"
#endif

#ifndef GP_H_
#include "gp++.h"
#endif

#ifndef EGO_H_
#define EGO_H_
#endif

/* EGP class */

double AEI_opt(const std::vector<double> &x, std::vector<double> &grad, void *data);


class EGO : public GP {
public:
	EGO(){ 
		KERNEL = NULL;
		n_free = true;
	};

	EGO(double (*K)(Eigen::VectorXd const & , Eigen::VectorXd const & ,  const Eigen::VectorXd &)) : GP(K){ 
	};

	void SetBest(){
		best = value[0];
		for(unsigned i=1; i<value.size(); i++) best = fmin(value[i],best);
	};

	double Best() const {return best;};

	Eigen::VectorXd FindMinAEI(Eigen::MatrixXd Bounds){
		dim = Bounds.rows();
	   	std::vector<double> lb(dim); 
		std::vector<double> ub(dim);
    	for(unsigned id=0; id<dim; id++){
        	lb[id] = Bounds(0,id);
        	ub[id] = Bounds(1,id);        
    	}
		std::vector<double> x(dim); 
   		nlopt::opt opt(nlopt::LN_SBPLX, dim);    /* algorithm and dimensionality */
		opt.set_lower_bounds(lb);
		opt.set_upper_bounds(ub);
		opt.set_max_objective(AEI_opt, this);
		opt.set_xtol_rel(1e-6);
    	opt.set_maxeval(1000);
		std::cout << "Starting optimization\n";
    	Eigen::VectorXd X(x.size());									//Parameters to be optimized
	  	std::cout << "Start optimization (10 random starting points)\n";
		dim = x.size();
		double bestmax = -1;
		VectorXd Xbest = VectorXd::Zero(dim);
    	for(unsigned is=0; is<30; is++){
        	Eigen::VectorXd Xi = (Eigen::VectorXd::Ones(dim) + Eigen::VectorXd::Random(dim))*.5;
			for(unsigned id=0; id<dim; id++) x[id] = Bounds(0,id)*Xi(id)+Bounds(1,id)*(1.-Xi(id));
    		double minf; /* the minimum objective value, upon return */
        	if (opt.optimize(x, minf) < 0) printf("nlopt failed!\n");
        	std::cout << "Optimization is successful ";
	    	for(unsigned i=0; i<x.size(); i++){
				X(i) = x[i];				//Setting the proposed value of the
				printf("%e ",X(i));
			}
			printf(" minf : %e \n",minf);
			if(minf>bestmax){
				Xbest = X; bestmax = minf;
			}
    	}
    	return Xbest;
	};
	double AEI(Eigen::VectorXd &X) const {
		Eigen::VectorXd Vpred = Eval(X);	
		double s = sqrt(Vpred(1));
		double diff = (Best() - Vpred(0))/s;
		double EI = (diff*cdf_normal(diff) + pdfg(diff))*s;  
		EI *=( 1.-sqrt( Sig2()/ (Vpred(1)+Sig2()) ));	
		return EI;
	};

private:
	unsigned dim;
	unsigned nd;
	double best;
};


double AEI_opt(const std::vector<double> &x, std::vector<double> &grad, void *data){
//    std::cout << " In EGO AEI Extremum function \n";
    EGO* ego = (EGO*) data;			//Pointer to the GP
	Eigen::VectorXd X(x.size());	
	for(unsigned i=0; i<x.size(); i++) X(i) = x[i];
	Eigen::VectorXd Vpred = ego->Eval(X);
	double s = sqrt(Vpred(1));
	double diff = (ego->Best() - Vpred(0))/s;
	double EI = (diff*cdf_normal(diff) + pdfg(diff))*s;  
	EI *=( 1.-sqrt( ego->Sig2()/ (Vpred(1)+ego->Sig2()) ));
	if (!grad.empty()) {										//Cannot compute gradient : stop!
		std::cout << "Asking for gradient, I stop !" << std::endl; exit(1);
	}
	return EI;
};