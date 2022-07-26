#include <nlopt.hpp>

int neval=0;

/* The class for the observations */
class DATA {
	friend class GP;
	friend class GPNF;
	friend class TPS;

public:
	DATA(){};
	DATA(Eigen::VectorXd const &x, double const &f){ X=x; F=f;};
	DATA(DATA const &d){ X = d.X; F= d.F;};
	void operator = (const DATA d){ X = d.X; F= d.F;};
	Eigen::VectorXd GetX() const { return X; };
	double Value() const { return F; };
    void SetX(Eigen::VectorXd x) { X=x;};
    void SetValue(double f) { F=f;};
private:
	Eigen::VectorXd X;
	double F;
};


/* Purpose : select the reduced set of columns of A minimizing the Frobenius error*/
std::vector<unsigned> ColSelect(const Eigen::MatrixXd &A, double frac){
	std::default_random_engine Rng;
	std::uniform_real_distribution<double> Unif(0,1);
	int ns = A.cols();
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::VectorXd s = svd.singularValues();
	double st = s.sum();
	double s0 = 0;
	int nr = 0;
	while(s0<st*sqrt(frac)){
		s0 += s(nr); nr++;
		if(nr>=A.cols()) break; 
	}
	nr = fmin(nr*6,A.cols());
	std::cout << s.transpose() << std::endl;
	std::cout<< "Need to make selection of " << nr << " columns over " << ns << "\n";

	Eigen::VectorXd Pr(ns);
	Eigen::VectorXd Prc = Eigen::VectorXd::Zero(ns);
	for(int is=0; is<ns; is++){
		Pr(is) = V.row(is).squaredNorm() / (double)(nr);
		Prc(is) += Pr(is);
		if(is<ns-1) Prc(is+1) = Prc(is);
	}
	Eigen::VectorXi Drawn = Eigen::VectorXi::Zero(ns);
	std::vector<unsigned> draw;
	while(draw.size()<nr){
		double xp = Prc(ns-1)*Unif(Rng);
		int is = 0;
		while(Prc(is)<xp) is++;
		if(Drawn(is)==0){
			draw.push_back(is);
			Drawn(is) = 1;
		}
	}
	Eigen::MatrixXd C(A.rows(),nr);
	for(int ik=0; ik<nr; ik++) C.col(ik) = A.col(draw[ik]);
	Eigen::JacobiSVD<Eigen::MatrixXd> SVD(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
	std::cout << SVD.singularValues().transpose() << std::endl;
	return draw;
};


/* GP model class */

class GP {
public:
	GP(){ KERNEL = NULL;};
	GP(double (*K)(Eigen::VectorXd const & , Eigen::VectorXd const & ,  const Eigen::VectorXd &)){ KERNEL = K;};

	void SetKernel(double (*K)(Eigen::VectorXd const & , Eigen::VectorXd const & , const Eigen::VectorXd &)){
			KERNEL = K;
	};

	void SetData(const Eigen::VectorXd &X, double F){
			Xpt.push_back(X);
			value.push_back(F);
			nd = Xpt.size();
			std::cout << "The GP will use " << nd << " observations\n";
	};

	void SetData(const std::vector<DATA> &data){
		Xpt.clear();
        value.clear();

		for(int i=0; i<data.size(); i++){
			Xpt.push_back(data[i].X);
			value.push_back(data[i].F);
		}
		nd = Xpt.size();
		std::cout << "The GP will use " << nd << " observations\n";
	};

	/*Set the Gaussian process for the parameters in par */
	double Set(const Eigen::VectorXd &par){	
		PAR = par;
		nd = Xpt.size();						//Number of data points
		unsigned np = par.rows();
		Eigen::MatrixXd A(nd,nd);				//Correlation perator
		Eigen::VectorXd Y(nd);					//Observations
		sigsn = pow(par(np-1),2);		//Noise variance
		for(int i=0; i<nd; i++){
			for(int j=i; j<nd; j++){
				A(i,j) = KERNEL(Xpt[i], Xpt[j], par);	//Two points correlation
				if(i!=j){
					A(j,i) = A(i,j);
				}else{
					A(i,j) += sigsn;					//Noise correlation
				}
			}
			Y(i) = value[i];							//Noisy observation
		}
		ldlt.compute(A);  					/* Decompose Correlation */
		Alpha = ldlt.solve(Y);				/* Solve for the GP coordinates*/
		/* Compute log of SLE optimization */
		logp = -Y.dot(Alpha)*.5 - (ldlt.vectorD().array().log()).sum() 
		- (double)(nd)*log(M_PI*2)*.5;	
		return -logp;
	};

	/* Evaluate GP at point x, mean prediction only */
	double EvalFast(Eigen::VectorXd const &x) const {
		double val = 0;
		for(int i=0; i<nd; i++){
			val += KERNEL(x,Xpt[i],PAR)*Alpha(i);
		}
		return val;
	};

	/* Evaluate GP at point x, mean and variance of prediction*/
	Eigen::VectorXd Eval(Eigen::VectorXd const &x) const {
		Eigen::VectorXd kstar(nd);
		for(int i=0; i<nd; i++) kstar(i) = KERNEL(x,Xpt[i],PAR);
		Eigen::VectorXd Out(2);
		Out(0) = kstar.dot(Alpha);
		Eigen::VectorXd v = ldlt.solve(kstar);
		Out(1) = KERNEL(x,x,PAR) - kstar.dot(v);
		return Out;
	};

    /* Retrieve dimension of the GP */
    unsigned Dim() const { return Xpt[0].rows(); };

    /* Retrieve number of observations */
    unsigned Nobs() const { return Xpt.size(); };

    /* Retrieve location of i-th observation */
    Eigen::VectorXd Xobs(unsigned const i) const { return Xpt[i]; };

    /* Retrieve value of i-th observation */
    double Fobs(unsigned const i) const { return value[i]; };

	double LOGP() const { return logp;};

	void OptimizeSLE(nlopt::vfunc myoptfunc_gp, unsigned const np = 4){
		nlopt::opt opt(nlopt::LN_SBPLX, np); /* algorithm and dimensionality */
		std::vector<double> lb(np); 
		std::vector<double> ub(np);
		std::vector<double> x(np); 
		std::cout << "Optimize Gaussian process for " << np << " hyperparameters\n";
		if(np==3){
			lb[0] = 1.e-4; 	lb[1] = 1.e-5; lb[2] = 1.e-5;/* lower bounds */
 			ub[0] = 1.e6; 	ub[1] = 1.e0; 	ub[2] = 1.e-4; /* upper bounds */
			x[0] = 1; x[1] = .1;  x[2] = lb[2];
		}else if(np==4){
			lb[0] = 1.e-4; lb[1] = 1.e-3; lb[2] = 1.1; lb[3] = 1.e-5;/* lower bounds */
 			ub[0] = 1.e6; ub[1] = 1.e1; ub[2] = 2.; ub[3] = 5.; /* upper bounds */
			x[0] = 1; x[1] = 1; x[2] = 1.5; x[3] = lb[2];			
		} else {
			std::cout<< "Invalid number of parameters \n";
		}
		opt.set_lower_bounds(lb);
		opt.set_upper_bounds(ub);
		opt.set_min_objective(myoptfunc_gp, this);
		opt.set_xtol_rel(1e-4);
		neval = 0;
		std::cout << "Starting optimization\n";
		double minf; /* the minimum objective value, upon return */
		if (opt.optimize(x, minf) < 0) {
			printf("nlopt failed!\n");
		}
		else {
			if(np==3){
				printf("found minimum at L = %8.6g\n \t\t Std = %8.7g \n \t\t Noise = %8.6g\n" , x[0], x[1], x[2]);
			}else{
				printf("found minimum at L = %8.6g\n \t\t Std = %8.7g \n \t\t Gamma= %6.5g \n\t\t Noise = %8.6g\n" , x[0], x[1], x[2], x[3]);				
			}
		}
		std::cout<<"Number of function evaluations " << neval << std::endl;
		PAR = Eigen::VectorXd(np);
		for(unsigned p=0; p<np; p++) PAR(p) =x[p];
		Set(PAR);
	};

	double EvalSelFast(Eigen::VectorXd const &x) const {
		double val = 0;
		for(int i=0; i<Sel.size(); i++){
			val += KERNEL(x,Xpt[Sel[i]],PAR)*BS(i);
		}
		return val;
	};

	Eigen::VectorXd EvalSel(Eigen::VectorXd const &x) const {
		Eigen::VectorXd kstar(Sel.size());
		for(int i=0; i<Sel.size(); i++){
			kstar(i) = KERNEL(x,Xpt[Sel[i]],PAR);
		}
		Eigen::VectorXd Out(2);
		Out(0) = kstar.dot(BS);
		Out(1) = KERNEL(x,x,PAR) - kstar.dot(Rinv*kstar);
		return Out;
	};

	void Select(double prec=.999){
        if(prec>1) prec = .9999;
        if(prec<0) prec = .1;
		Eigen::MatrixXd K(nd,nd);
		Eigen::VectorXd Y(nd);
		for(int i=0; i<nd; i++){
			for(int j=i; j<nd; j++){
				K(i,j) = KERNEL(Xpt[i],Xpt[j],PAR);
				K(j,i) = K(i,j);
			}
			Y(i) = value[i];
		}
		Sel.clear();
		Sel = ColSelect(K, prec); //This is the reduced set of features.
		Eigen::MatrixXd Z(nd,Sel.size());
		for(int i=0; i<Sel.size(); i++) Z.col(i) = K.col(Sel[i]);
		Eigen::MatrixXd R = Z.transpose()*Z/(sigsn) + (Eigen::MatrixXd) Eigen::VectorXd::Ones(Z.cols()).asDiagonal();
		Rinv = R.inverse();
		BS = Rinv*Z.transpose()*Y/sigsn;
		std::cout<< BS.transpose() << std::endl;
	};

    double Kvalue (Eigen::VectorXd const x, Eigen::VectorXd const y) const {
        return KERNEL(x,y,PAR);
    };
    
    std::vector<unsigned> Selected() const {
        return Sel;
    }


	Eigen::MatrixXd CompCov(std::vector<Eigen::VectorXd> const &Target, Eigen::VectorXd &Mean){
		/* Compute the Covariance at the Target points */
		Eigen::MatrixXd CoV(Target.size(),Target.size());
		Mean = Eigen::VectorXd(Target.size());
		std::vector<Eigen::VectorXd> ks;
		for(unsigned it=0; it<Target.size(); it++){
			Eigen::VectorXd kstar(nd);
			Eigen::VectorXd Xi = Target[it];
			for(int i=0; i<nd; i++) kstar(i) = KERNEL(Xi,Xpt[i],PAR);
			Mean(it) = kstar.dot(Alpha);
			ks.push_back(kstar);
			for(unsigned jt=it; jt<Target.size(); jt++){
				Eigen::VectorXd Xj = Target[jt];
				CoV(it,jt) = KERNEL(Xi,Xj,PAR);
			}
		}
		for(unsigned it=0; it<Target.size(); it++){
			Eigen::VectorXd Vi = ldlt.solve(ks[it]);	
			for(unsigned jt=it; jt<Target.size(); jt++){
				CoV(it,jt) -= Vi.dot(ks[jt]);
				if(it!=jt) CoV(jt,it) = CoV(it,jt);
			}
		}
		return CoV;
	};

	Eigen::MatrixXd SampleTarget(std::vector<Eigen::VectorXd> const &Target, int ns, std::default_random_engine &gen){
		Eigen::VectorXd Mean;
		Eigen::MatrixXd CoV = CompCov(Target,Mean);
		Eigen::LDLT<Eigen::MatrixXd> DEC(CoV);
		Eigen::VectorXd D = DEC.vectorD();
		for(unsigned i=0; i<D.rows(); i++) D(i) = sqrt(fabs(D(i)));
		std::cout << "Dmax : " << D.maxCoeff() << " Dmin " << D.minCoeff() << std::endl;
		std::normal_distribution<double> distN(0,1);
		Eigen::MatrixXd Samples(Target.size(),ns);
		for(unsigned s=0; s<ns; s++){
			Eigen::VectorXd Eta(CoV.cols());
			for(unsigned i=0; i<CoV.cols(); i++) Eta(i) = distN(gen)*D(i);
			Samples.col(s) = DEC.matrixL()*Eta + Mean;
		}
		return Samples;
	};

	Eigen::MatrixXd SampleTarget(std::vector<Eigen::VectorXd> const &Target, Eigen::MatrixXd const &Rea, std::default_random_engine &gen){
		Eigen::VectorXd Mean(Target.size());
		// Eigen::MatrixXd CoV = CompCov(Target,Mean);
		// Eigen::LDLT<Eigen::MatrixXd> DEC(CoV);
		// Eigen::VectorXd D = DEC.vectorD();
		// for(unsigned i=0; i<D.rows(); i++) D(i) = sqrt(fabs(D(i)));
		// std::cout << "Dmax : " << D.maxCoeff() << " Dmin " << D.minCoeff() << std::endl;
		// std::normal_distribution<double> distN(0,1);
		std::vector<Eigen::VectorXd> kst;
		for(unsigned it=0; it<Target.size(); it++){
			Eigen::VectorXd kstar(nd);
			Eigen::VectorXd Xi = Target[it];
			for(int i=0; i<nd; i++) kstar(i) = KERNEL(Xi,Xpt[i],PAR);
			kst.push_back(kstar);
		}
		/* Engaging the actual sampling */
		std::cout << "Here we go \n";
		Eigen::MatrixXd Samples(Target.size(),Rea.cols());
		for(unsigned s=0; s<Rea.cols(); s++){
			Eigen::VectorXd Alp = ldlt.solve(Rea.col(s));
			for(unsigned it=0; it<Target.size(); it++) Mean(it) = kst[it].dot(Alp);
			// Eigen::VectorXd Eta(CoV.cols());
			// for(unsigned i=0; i<CoV.cols(); i++) Eta(i) = distN(gen)*D(i);
			Samples.col(s) = Mean;	// + DEC.matrixL()*Eta;
		}
		return Samples;
	};

	Eigen::MatrixXd EvalNew(std::vector<Eigen::VectorXd> const &Target, Eigen::MatrixXd const &Rea){
		Eigen::VectorXd Mean(Target.size());
		std::vector<Eigen::VectorXd> kst;
		for(unsigned it=0; it<Target.size(); it++){
			Eigen::VectorXd kstar(nd);
			Eigen::VectorXd Xi = Target[it];
			for(int i=0; i<nd; i++) kstar(i) = KERNEL(Xi,Xpt[i],PAR);
			kst.push_back(kstar);
		}
		/* Engaging the actual sampling */
		Eigen::MatrixXd Samples(Target.size(),Rea.cols());
		for(unsigned s=0; s<Rea.cols(); s++){
			Eigen::VectorXd Alp = ldlt.solve(Rea.col(s));
			for(unsigned it=0; it<Target.size(); it++) Mean(it) = kst[it].dot(Alp);
			Samples.col(s) = Mean;
		}
		return Samples;
	};

	Eigen::VectorXd GetPar() const {return PAR;};

private:
	int nd;
	double sigsn;
	double logp;
	Eigen::LDLT<Eigen::MatrixXd> ldlt;
	Eigen::VectorXd Alpha;
	std::vector<Eigen::VectorXd> Xpt;
	std::vector<double> value;
	double (*KERNEL)(const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &);
	Eigen::VectorXd PAR;
	std::vector<unsigned> Sel;
	Eigen::VectorXd BS;
	Eigen::MatrixXd Rinv;
};



/* Noise Free GP model class */

class GPNF {
public:
	GPNF(){ KERNEL = NULL;};
	GPNF(double (*K)(Eigen::VectorXd const & , Eigen::VectorXd const & ,  const Eigen::VectorXd &)){ KERNEL = K;};

	void SetKernel(double (*K)(Eigen::VectorXd const & , Eigen::VectorXd const & , const Eigen::VectorXd &)){
			KERNEL = K;
	};

	void SetData(const Eigen::VectorXd &X, double F){
			Xpt.push_back(X);
			value.push_back(F);
			nd = Xpt.size();
			// std::cout << "The GP will use " << nd << " observations\n";
	};

	void SetData(const std::vector<DATA> &data){
		Xpt.clear();
        value.clear();
		for(int i=0; i<data.size(); i++){
			Xpt.push_back(data[i].X);
			value.push_back(data[i].F);
		}
		nd = Xpt.size();
		std::cout << "The GP will use " << nd << " observations\n";
	};

	void SetValuesAtData(const Eigen::VectorXd &F){
		if(F.size()!=Xpt.size()){
			std::cout << "can only set as many data as data points !\n";
			return;
		}
		value.clear();
		for(unsigned i=0; i<Xpt.size(); i++) value.push_back(F(i));
	};
	/*Set the Gaussian process for the parameters in par */
	double Set(const Eigen::VectorXd &par){	
		PAR = par;
		nd = Xpt.size();						//Number of data points
		unsigned np = par.rows();
		Eigen::MatrixXd A(nd,nd);				//Correlation perator
		Eigen::VectorXd Y(nd);					//Observations
		for(int i=0; i<nd; i++){
			for(int j=i; j<nd; j++){
				A(i,j) = KERNEL(Xpt[i], Xpt[j], par);	//Two points correlation
				if(i!=j){
					A(j,i) = A(i,j);
				}else{
					A(j,i) += 1.e-10; 
				}
			}
			Y(i) = value[i];							//Noisy observation
		}
		ldlt.compute(A);  					/* Decompose Correlation */ 
		Alpha = ldlt.solve(Y);				/* Solve for the GP coordinates*/
		/* Compute log of SLE optimization */
		logp = -Y.dot(Alpha)*.5 - (ldlt.vectorD().array().log()).sum() 
		- (double)(nd)*log(M_PI*2)*.5;	
		return -logp;
	};

	/* Evaluate GP at point x, mean prediction only */
	double EvalFast(Eigen::VectorXd const &x) const {
		double val = 0;
		for(int i=0; i<nd; i++){
			val += KERNEL(x,Xpt[i],PAR)*Alpha(i);
		}
		return val;
	};
	/* Evaluate GP at point x, mean and variance of prediction*/
	Eigen::VectorXd Eval(Eigen::VectorXd const &x) const {
		Eigen::VectorXd kstar(nd);
		for(int i=0; i<nd; i++) kstar(i) = KERNEL(x,Xpt[i],PAR);
		Eigen::VectorXd Out(2);
		Out(0) = kstar.dot(Alpha);
		Eigen::VectorXd v = ldlt.solve(kstar);
		Out(1) = KERNEL(x,x,PAR) - kstar.dot(v);
		return Out;
	};

    /* Retrieve dimension of the GP */
    unsigned Dim() const { return Xpt[0].rows(); };

    /* Retrieve number of observations */
    unsigned Nobs() const { return Xpt.size(); };

    /* Retrieve location of i-th observation */
    Eigen::VectorXd Xobs(unsigned const i) const { return Xpt[i]; };

    /* Retrieve value of i-th observation */
    double Fobs(unsigned const i) const { return value[i]; };

	double LOGP() const { return logp;};

	void OptimizeSLE(nlopt::vfunc myoptfunc_gp){
		unsigned const np = 2;
		nlopt::opt opt(nlopt::LN_SBPLX, np); /* algorithm and dimensionality */
		std::vector<double> lb(np); 
		std::vector<double> ub(np);
		std::vector<double> x(np); 
		// std::cout << "Optimize Gaussian process for " << np << " hyperparameters\n";
		lb[0] = 1.e-4; 	lb[1] = 1.e-5; /* lower bounds */
		ub[0] = 1.e2; 	ub[1] = 6.e0;  /* upper bounds */
		x[0] = 1.; x[1] = .1;
		opt.set_lower_bounds(lb);
		opt.set_upper_bounds(ub);
		opt.set_min_objective(myoptfunc_gp, this);
		opt.set_xtol_rel(1e-4);
		neval = 0;
		std::cout << "Starting optimization of Noise-Free process\n";
		double minf; /* the minimum objective value, upon return */
		if (opt.optimize(x, minf) < 0) {
			printf("nlopt failed!\n");
		}else {
			printf("found minimum at L = %8.6g \t Std = %8.7g in %4d evaluations\n ",x[0],x[1],neval);
		}
		PAR = Eigen::VectorXd(np);
		for(unsigned p=0; p<np; p++) PAR(p) =x[p];
		Set(PAR);
	};


    double Kvalue (Eigen::VectorXd const x, Eigen::VectorXd const y) const {
        return KERNEL(x,y,PAR);
    };
	
	Eigen::MatrixXd CompCov(std::vector<Eigen::VectorXd> const &Target, Eigen::VectorXd &Mean){
		/* Compute the Covariance at the Target points */
		Eigen::MatrixXd CoV(Target.size(),Target.size());
		Mean = Eigen::VectorXd(Target.size());
		std::vector<Eigen::VectorXd> ks;
		for(unsigned it=0; it<Target.size(); it++){
			Eigen::VectorXd kstar(nd);
			Eigen::VectorXd Xi = Target[it];
			for(int i=0; i<nd; i++) kstar(i) = KERNEL(Xi,Xpt[i],PAR);
			Mean(it) = kstar.dot(Alpha);
			ks.push_back(kstar);
			for(unsigned jt=it; jt<Target.size(); jt++){
				Eigen::VectorXd Xj = Target[jt];
				CoV(it,jt) = KERNEL(Xi,Xj,PAR);
			}
		}
		for(unsigned it=0; it<Target.size(); it++){
			Eigen::VectorXd Vi = ldlt.solve(ks[it]);	
			for(unsigned jt=it; jt<Target.size(); jt++){
				CoV(it,jt) -= Vi.dot(ks[jt]);
				if(it!=jt) CoV(jt,it) = CoV(it,jt);
			}
		}
		return CoV;
	};

	Eigen::MatrixXd SampleTarget(std::vector<Eigen::VectorXd> const &Target, int ns, std::default_random_engine &gen){
		Eigen::VectorXd Mean;
		Eigen::MatrixXd CoV = CompCov(Target,Mean);
		Eigen::LDLT<Eigen::MatrixXd> DEC(CoV);
		Eigen::VectorXd D = DEC.vectorD();
		for(unsigned i=0; i<D.rows(); i++) D(i) = sqrt(fabs(D(i)));
		std::cout << "Dmax : " << D.maxCoeff() << " Dmin " << D.minCoeff() << std::endl;
		std::normal_distribution<double> distN(0,1);
		Eigen::MatrixXd Samples(Target.size(),ns);
		for(unsigned s=0; s<ns; s++){
			Eigen::VectorXd Eta(CoV.cols());
			for(unsigned i=0; i<CoV.cols(); i++) Eta(i) = distN(gen)*D(i);
			Samples.col(s) = DEC.matrixL()*Eta + Mean;
		}
		return Samples;
	};

	Eigen::MatrixXd SampleTarget(std::vector<Eigen::VectorXd> const &Target, Eigen::MatrixXd const &Rea){
		Eigen::VectorXd Mean(Target.size());
		std::vector<Eigen::VectorXd> kst;
		for(unsigned it=0; it<Target.size(); it++){
			Eigen::VectorXd kstar(nd);
			Eigen::VectorXd Xi = Target[it];
			for(int i=0; i<nd; i++) kstar(i) = KERNEL(Xi,Xpt[i],PAR);
			kst.push_back(kstar);
		}
		/* Engaging the actual sampling */
		Eigen::MatrixXd Samples(Target.size(),Rea.cols());
		for(unsigned s=0; s<Rea.cols(); s++){
			Eigen::VectorXd Alp = ldlt.solve(Rea.col(s));
			for(unsigned it=0; it<Target.size(); it++) Mean(it) = kst[it].dot(Alp);
			Samples.col(s) = Mean;
		}
		return Samples;
	};

	Eigen::MatrixXd EvalNew(std::vector<Eigen::VectorXd> const &Target, Eigen::MatrixXd const &Rea){
		Eigen::VectorXd Mean(Target.size());
		std::vector<Eigen::VectorXd> kst;
		for(unsigned it=0; it<Target.size(); it++){
			Eigen::VectorXd kstar(nd);
			Eigen::VectorXd Xi = Target[it];
			for(int i=0; i<nd; i++) kstar(i) = KERNEL(Xi,Xpt[i],PAR);
			kst.push_back(kstar);
		}
		/* Engaging the actual sampling */
		Eigen::MatrixXd Samples(Target.size(),Rea.cols());
		for(unsigned s=0; s<Rea.cols(); s++){
			Eigen::VectorXd Alp = ldlt.solve(Rea.col(s));
			for(unsigned it=0; it<Target.size(); it++) Mean(it) = kst[it].dot(Alp);
			Samples.col(s) = Mean;
		}
		return Samples;
	};
	Eigen::VectorXd GetPar() const {return PAR;};
	Eigen::LDLT<Eigen::MatrixXd> GetLDLT() const { return ldlt;};
private:
	int nd;
	double logp;
	Eigen::LDLT<Eigen::MatrixXd> ldlt;
	Eigen::VectorXd Alpha;
	std::vector<Eigen::VectorXd> Xpt;
	std::vector<double> value;
	double (*KERNEL)(const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &);
	Eigen::VectorXd PAR;
};


//double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data);
// 	/* This is the function you optimize for defining the GP */	
// double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
// 	GP* proc = (GP*) data;											//Pointer to the GP
// 	Eigen::VectorXd p(x.size());									//Parameters to be optimized
// 	for(int i=0; i<(int) x.size(); i++) p(i) = x[i];				//Setting the proposed value of the parameters
// 	double value = proc->Set(p);									//Evaluate the function
// 	if (!grad.empty()) {											//Cannot compute gradient : stop!
// 		std::cout << "Asking for gradient, I stop !" << std::endl; exit(1);
// 	}
// 	neval++;														//increment the number of evaluation count
// 	return value;
// };
//
// double Kernel(Eigen::VectorXd const &x, Eigen::VectorXd const &y, const Eigen::VectorXd &par){
// 	return par(0)*exp( -((x-y)/par(1)).squaredNorm()*.5 ); /* squared exponential kernel */
// };
//
//	Typical procedure:
//		vector<Data> data; 
//	You set your data (couples of points coordinate and function value)
//		GP proc(Kernel);
//		proc.SetData(data);
//		proc.OptimizeSLE();
//		proc.Select();
//
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

