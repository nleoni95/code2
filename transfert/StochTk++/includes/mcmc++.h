typedef double (*pfun_t)(VectorXd const &, void *);
double Tiny = -1.e-16;


class MCMC{

public:
    MCMC(MatrixXd &CovInit, pfun_t pin, void *din, unsigned na = 5000){
		nadapt 		= na;
    	plog 		= pin;
    	d 			= CovInit.cols();
    	CovProp		= CovInit;
    	nstandard 	= std::normal_distribution<double>(0,1);
    	ustandard 	= std::uniform_real_distribution<double>(0,1);
    	Tac 		= VectorXi::Zero(nadapt);
    	this->DecCov();
		  data = din;
    };

    void Seed(VectorXd const &Xini){ 
    	xCur 	= Xini; 
		xBest 	= xCur;
    	pCur 	= plog(xCur,data);
		pBest   = pCur;
    	mea  	= xCur;
    	acc 	= xCur*xCur.transpose();
    	count 	= 1;
    };

	VectorXd DoStep(unsigned const nstep){
		for(unsigned i=0; i<nstep; i++) doOneStep();
		return xCur;
	};

    void doOneStep(){
    	VectorXd xTemp 	= xCur + PropStep();
    	double pTemp 	= plog(xTemp,data);
		if(pTemp>pBest){
			pBest = pTemp;
			xBest = xTemp;
		}
		if(pTemp==Tiny){
	  		Tac(count%nadapt) = 0;			
		}else{
	    	double r0 		= ustandard(generator);
	    	if( pTemp-pCur>log(r0)){
    			xCur = xTemp;
    			pCur = pTemp;
	    		Tac(count%nadapt) = 1;
    		}else{
    			Tac(count%nadapt) = 0;
    		}
		}
    };

    void Burn(unsigned const nburn){
		for(unsigned i=0; i<nburn; i++){
			BurnOneStep();
			if((i>300) &&(i%nadapt==0)) AdaptCov();
		}
		printf("Accept rate %6.4f \n",(double)Tac.sum()/(double)(nadapt));
    };
	VectorXd GetBest() const { return xBest;}; 
    VectorXd GetCur() const { return xCur;};							//Get the current state of the chain
	void SetCovProp(MatrixXd const &Cov_in){ CovProp = Cov_in;};//Enforce covariance of the proposal
	MatrixXd GetCovProp() const { return CovProp; };			//Retrieve the covariance of the proposal

private:
	unsigned nadapt;
	unsigned d;																//Dimension of the chain
	pfun_t plog;															//Log-likelihood function
	VectorXd xCur;															//Current state of the chain
	VectorXd xBest;
	double pCur;															//Log-likelihood of the current state
	double pBest;
	unsigned count;															//Current number of steps 
	MatrixXd acc;															//Accumulated sum over the steps 2nd moments
	VectorXd mea;															//Accumulated steps mean
	VectorXi Tac;															//Counter for acceptance rate
	MatrixXd CovProp;														//Covariance of the proposal step
  	Eigen::LLT<MatrixXd> CovDec;											//Decomposition of the proposal covariance
 	std::default_random_engine generator;									//Random number generator
  	std::normal_distribution<double> nstandard;								//and distributions
  	std::uniform_real_distribution<double> ustandard;
	void *data;

/*	Utilities */
  	void DecCov(){ CovDec.compute(CovProp*(2.38*2.38)/(double)(d)); };		//Decompose the (scaled) covariance proposal

  	VectorXd PropStep(){
  		VectorXd dX = VectorXd(d);
  		for(unsigned i=0; i<d; i++) dX(i) = nstandard(generator);
  		dX = CovDec.matrixL()*dX;
		for(unsigned i=0; i<d; i++) dX(i) += nstandard(generator)*1.e-6;
  		return dX;
  	};

    void BurnOneStep(){
    	VectorXd xTemp = xCur + PropStep();
    	double pTemp = plog(xTemp,data);

		if( pTemp == Tiny){
			Tac(count%nadapt) = 0;
		}else{
	    	double r0 = ustandard(generator);
    		if( pTemp-pCur>log(r0)){
    			xCur = xTemp;
    			pCur = pTemp;
	    		Tac(count%nadapt) = 1;
    		}else{
    			Tac(count%nadapt) = 0;
    		}
		}
    	mea += xCur;
    	acc += xCur*xCur.transpose();
    	count++;
    };

	void AdaptCov(){////////////////// what's for?
		double c1 	= (double) (count);
		double c2 	= c1*c1;
    	CovProp 	= acc/(double)(count-1);
		CovProp    -= (mea*mea.transpose())/c2;
		CovProp    += 1.e-10 * Eigen::MatrixXd::Identity(d,d);
		VectorXd Mn = mea / c1;
		printf("Mean : %12.6e ",Mn(0));
		for(unsigned im=1; im<Mn.rows(); im++) printf("%12.6e ",Mn(im));
		printf(" Rate : %12.6e \n", (double)Tac.sum() / (double)nadapt);
		DecCov();
  };
};
