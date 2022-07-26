#ifndef KDE_H_
#define KDE_H_
#endif

using namespace std;
using namespace Eigen;


class KDE {
	
public:
	//! \brief default constructor
	KDE(){};
	//! \brief constructor from a basis and set of points
	KDE(MatrixXd const &Xi, char Typ = 'G'){
		nd = Xi.rows();
		Kc = pow(2.*M_PI, -.5*nd );
		ns = Xi.cols();
		Samp = Xi;
		t = Typ;
		cout << "Instantiate KDE in " << nd << " dimensions " << " with " << ns << " Samples\n";
		if(t == 'G'){				//Use isotropic kernel with Optimal Silverman's BW
			MatrixXd MS = MeanSigma();
			double cof = 1./(double)(nd+4);
			double c1 = pow( 4./(double)(nd+2), cof) * pow( (double) ns, -cof);
			H = MatrixXd::Zero(nd,nd); 
			Kd = 1.;
			for(int id=0; id<nd; id++){
				H(id,id) = pow(c1 * MS(id,1),-2);
				Kd *= H(id,id);
			}
			Kd = sqrt(Kd)/(double) ns;
		}else{
			cout << "Unknown KDE method \n"; 
			exit(1);
		}
	};

	double dP_KDE(VectorXd const &x) const {
		double p = 0;
		VectorXd ps(ns);
#pragma omp parallel for schedule (dynamic)		
		for(int is=0; is<ns; is++){
			VectorXd Dx = x - Samp.col(is);
			ps(is) = exp( - Dx.dot(H*Dx)*.5 );
		}
		return ps.sum()*(Kc*Kd);
	};
	
	//! \brief returns number of samples	
	int Ns() const{ return ns;};
	
	//! \brief returns number of regressors
	int Dim() const{ return nd;};
	
	
private:	
	MatrixXd Samp;
	MatrixXd H;
	char t;
	int nd;
	int ns;
	double Kc, Kd;
	MatrixXd MeanSigma() const {
		 MatrixXd MS(nd,2);
		 double cofm = 1. / (double)(ns  );		 
		 double cofs = 1. / (double)(ns-1);
		for(int id=0; id<nd; id++){
			MS(id,0) = Samp.row(id).sum()*cofm;
			MS(id,1) = sqrt( fabs(Samp.row(id).squaredNorm()*cofs - MS(id,0)*MS(id,0)) );
		}
		return MS;
	};

};
