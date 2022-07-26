#ifndef PCB_H_
#include "pcb++.h"
#endif

#ifndef Regress_H_
#define Regress_H_
#endif

using namespace std;
using namespace Eigen;


template <class B>
class Regress {
	
	template<class BV> friend class LARS;

public:
	//! \brief default constructor
	Regress(){};

	//! \brief constructor from a basis and set of points
	Regress(B *bin, MatrixXd const &Xi){
		base = bin;
		int nd   = base->Ndim();
		np = base->Npol();
		nr = Xi.cols();
		Z = MatrixXd(nr,np);
		for(int ir=0; ir<nr; ir++){
			VectorXd xi = Xi.col(ir);
			Z.row(ir) = base->Comp_Psi(xi);
		}
		Fish = (Z.transpose()*Z).inverse();
	};

	//! \brief constructor from a set of regressor values
	Regress(MatrixXd const &Psi){
		nr = Psi.rows();
		np = Psi.cols();
		Z = Psi;
		Fish = (Z.transpose()*Z).inverse();
	};

	//! \brief returns basis pointer
	B* Base() const{ return base;};
	
	//! \brief returns number of regression points	
	int Dim() const{ return nr;};
	
	//! \brief returns number of regressors
	int Npol() const{ return np;};
	
	//! \brief solve the regression problem
	VectorXd Solve(VectorXd const &R) const {
		if(R.rows() != nr){
			cout << "Non Matching number of data for regression \n" << endl;
			exit(1);
		}
		return Fish*(Z.transpose()*R);
	};
	
	
private:	
	MatrixXd Z, Fish;
	B* base;
	int nr;
	int np;
};


template <class B>
class LARS {
	
public:
	//! \brief default constructor
	LARS(){};

	//! \brief constructor from a basis and set of points
	LARS(B *bin, MatrixXd const &Xi,  MatrixXd const &Xc){
		base = bin;
		int nd = base->Ndim();
		np = base->Npol();
		nr = Xi.cols();
		nc = Xc.cols();
		Z  = MatrixXd(nr,np);
		Zc = MatrixXd(nc,np);
		
		for(int ir=0; ir<nr; ir++){
			VectorXd xi = Xi.col(ir);
			Z.row(ir) = base->Comp_Psi(xi);
		}
		for(int ir=0; ir<nc; ir++){
			VectorXd xi = Xc.col(ir);
			Zc.row(ir) = base->Comp_Psi(xi);
		}
		N = VectorXd(np); 
		for(int ip=0; ip<np; ip++) N(ip) = Z.col(ip).norm();
		
	};


	//! \brief constructor from a sets of polynomials
	LARS(MatrixXd const &Psir,  MatrixXd const &Psic){
		base = NULL;
		np = Psir.cols();
		nr = Psir.rows();
		nc = Psic.rows();
		Z  = Psir;
		Zc = Psic;
		if(np != Psic.cols()){
			cout << "LARS error : Data and CV set have non matching dimensions\n";
			exit(0);
		}
		N = VectorXd(np); 
		for(int ip=0; ip<np; ip++) N(ip) = Z.col(ip).norm();
	};


	//! \brief returns basis pointer
	B* Base() const{ return base;};
	
	//! \brief returns number of regression points	
	int Dim() const{ return nr;};
	int Dimc() const{ return nc;};
	int Dimt() const{ return nr+nc;};
	
	//! \brief returns number of regressors
	int Npol() const{ return np;};
	
	//! \brief solve the regression problem
	VectorXd Solve(VectorXd const &R, VectorXd const &C, int &k) const {
		if(R.rows() != nr || C.rows() != nc){
			cout << "Non Matching number of data for LARS \n" << endl;
			exit(1);
		}
		
		double resk = C.norm();
		VectorXi I = VectorXi::Zero(np);
		VectorXd c = VectorXd::Zero(np);
		k=0;

		while(k<np){
			//Compute current residual
			VectorXd Res = R;
			for(int ip=0; ip<k; ip++) Res -= Z.col(I(ip))*c(I(ip));
			VectorXd Rcv = C;
			for(int ip=0; ip<k; ip++) Rcv -= Zc.col(I(ip))*c(I(ip));
			if(Rcv.norm()>resk){
				c(I(k-1)) = 0;
				break;
			}
			resk = Rcv.norm();
			//Project the residual:
			VectorXd PR = Z.transpose()*Res;
			VectorXd PC(np);
			for(int ip=0; ip<np; ip++) PC(ip) = fabs(PR(ip)/N(ip));
			for(int ip=0; ip<k; ip++) PC(I(ip)) = 0;
			int mx; PC.maxCoeff(&mx);					//Find element the most aligned with the residual
			I(k) = mx;
			MatrixXd s(k+1,k+1); VectorXd r(k+1);
			for(int i=0; i<k+1; i++){
				for(int j=0; j<k+1; j++){
					s(i,j) = Z.col(I(i)).dot(Z.col(I(j)));
				}
				r(i) = Z.col(I(i)).dot(R);
			}
			VectorXd x = s.inverse()*r;
			for(int i=0; i<k+1; i++) c(I(i)) = x(i);
			k++;
		};
		
		/* Finalize by solving over the full data set */
		MatrixXd s(k,k); VectorXd r(k);
		for(int i=0; i<k; i++){
			for(int j=0; j<k; j++){
				s(i,j) = Z.col(I(i)).dot(Z.col(I(j))) + Zc.col(I(i)).dot(Zc.col(I(j)));
			}
			r(i) = Z.col(I(i)).dot(R) + Zc.col(I(i)).dot(C);
		}
		VectorXd x = s.inverse()*r;
		for(int i=0; i<k; i++) c(I(i)) = x(i);
		return c;		
	};

	//! \brief solve the regression problem
	VectorXd Solve(VectorXd const &R, int &k, int kmax=160) const {
		if(R.rows() != nr ){
			cout << "Non Matching number of data for LARS \n" << endl;
			exit(1);
		}
		
		double resk = R.norm();
		VectorXi I = VectorXi::Zero(np);
		VectorXd c = VectorXd::Zero(np);
		k=0;
		if(kmax >np) kmax = np;
		VectorXd ResLoo = VectorXd::Ones(kmax)*1.e6;
		while(k<kmax){
			//Compute current residual
			VectorXd Res = R;
			for(int ip=0; ip<k; ip++) Res -= Z.col(I(ip))*c(I(ip));
			//Project the residual:
			VectorXd PR = Z.transpose()*Res;
			VectorXd PC(np);
			for(int ip=0; ip<np; ip++) PC(ip) = fabs(PR(ip)/N(ip));
			for(int ip=0; ip<k; ip++) PC(I(ip)) = 0;
			int mx; PC.maxCoeff(&mx);					//Find element the most aligned with the residual
			I(k) = mx;
			
			{
				VectorXd r(k+1);
				MatrixXd Zk(nr,k+1);
				for(int i=0; i<k+1; i++){
					Zk.col(i) = Z.col(I(i));
					r(i)      = Z.col(I(i)).dot(R);
				}
				MatrixXd F = (Zk.transpose()*Zk).inverse();
				VectorXd x = F*r;
				VectorXd br = R - Zk*x;
				F = Zk * F * Zk.transpose();
				double rloo1 = 0;
				for(int o=0; o<nr; o++) rloo1 += pow(br(o)/(1.-F(o,o)),2);
				ResLoo(k) = rloo1;				
				for(int i=0; i<k+1; i++) c(I(i)) = x(i);
			}
			k++;
		};		
		ResLoo.minCoeff(&k);
		{
			k ++;
			VectorXd r(k);
			MatrixXd Zk(nr,k);
			for(int i=0; i<k; i++){
				Zk.col(i) = Z.col(I(i));
				r(i) = Z.col(I(i)).dot(R);
			}
			MatrixXd F = (Zk.transpose()*Zk).inverse();
			VectorXd x = F*r;
			c *=0;
			for(int i=0; i<k; i++) c(I(i)) = x(i);
		}	
		return c;		
	};


	//! \brief solve the regression problem
	VectorXd SolveKF(VectorXd const &R, int &k, int kfold=2) const {
		if(R.rows() != nr ){
			cout << "Non Matching number of data for LARS \n" << endl;
			exit(1);
		}
		vector<int> ok[kfold];
		RandomLib::Random Rng; Rng.Reseed();
		for(int o=0; o<nr; o++){
			ok[Rng.Integer(kfold)].push_back(o);
		}
		for(int kf=0; kf<kfold; kf++) ok[kf].push_back(nr);

//		for(int kf=0; kf<kfold; kf++) printf("Sample size for %2d fold is %5d \n",kf,ok[kf].size()-1);
//		cout << "Total number of samples : " << nr << endl;
		MatrixXd Zf[kfold], Zc[kfold];
		VectorXd Rf[kfold], Rc[kfold];
		VectorXd Nr[kfold];
		VectorXi I[kfold];
		VectorXd c[kfold];
		
		for(int kf=0; kf<kfold; kf++){
			int sizef = ok[kf].size()-1;
			Zf[kf] = MatrixXd(nr-sizef,np);
			Rf[kf] = VectorXd(nr-sizef);
			Zc[kf] = MatrixXd(sizef,np);
			Rc[kf] = VectorXd(sizef);
			int io=0, ic=0;
			for(int i=0; i<nr; i++){
				if(i!=ok[kf][io]){
					Zf[kf].row(ic) = Z.row(i); 
					Rf[kf](ic) = R(i);
					ic++;
				}else{
					Zc[kf].row(io) = Z.row(i);
					Rc[kf](io) = R(i);
					io++;
				}
			}
			cout << " io - ic " << io << " " << ic << " size CV " << ok[kf].size() << endl;
			Nr[kf] = VectorXd(np);
			for(int ip=0; ip<np;ip++) Nr[kf](ip) = Zf[kf].col(ip).norm();
			I[kf] = VectorXi::Zero(np);
			c[kf] = VectorXd::Zero(np);
		}	
//		cout << "K-fold problem is ready \n";
		int km=0;
		int kopt;
		VectorXd Error = VectorXd::Ones(np)*1.e10;
		while(km<np){
			VectorXd Ekf(kfold);
			for(int kf=0; kf<kfold; kf++){
				VectorXd Res = Rf[kf];
				for(int ip=0; ip<km; ip++) Res -= Zf[kf].col(I[kf](ip))*c[kf](I[kf](ip));
				VectorXd PR = Zf[kf].transpose()*Res;
				VectorXd PC(np);
				for(int ip=0; ip<np; ip++) PC(ip) = fabs(PR(ip)/Nr[kf](ip));
				for(int ip=0; ip<km; ip++) PC(I[kf](ip)) = 0;
				int mx; PC.maxCoeff(&mx);					//Find element the most aligned with the residual
				I[kf](km) = mx;	
				VectorXd r(km+1);
				MatrixXd Zk(Zf[kf].rows(),km+1);
				for(int i=0; i<km+1; i++){
					Zk.col(i) = Zf[kf].col(I[kf](i));
					r(i)      = Zf[kf].col(I[kf](i)).dot(Rf[kf]);
				}
				MatrixXd F = (Zk.transpose()*Zk).inverse();
				VectorXd x = F*r;
				VectorXd rc = Rc[kf];
				for(int i=0; i<km+1; i++) rc -= Zc[kf].col(I[kf](i))*x(i);
				Ekf(kf) = rc.squaredNorm() / Rc[kf].rows();
			}
			cout << Ekf.transpose() << endl;
			Error(km) = Ekf.sum()/(double)(kfold);
			Error.minCoeff(&kopt);
			if(km-kopt>5) break; 
			km++;	
		}
		kopt++;
		cout << "Optimal rank is " << kopt << endl;
		VectorXd Copt = VectorXd::Zero(np);
		VectorXi Iopt = VectorXi::Zero(kopt);
		k=0;
		while(k<kopt){
			//Compute current residual
			VectorXd Res = R;
			for(int ip=0; ip<k; ip++) Res -= Z.col(Iopt(ip))*Copt(Iopt(ip));
			//Project the residual:
			VectorXd PR = Z.transpose()*Res;
			VectorXd PC(np);
			for(int ip=0; ip<np; ip++) PC(ip) = fabs(PR(ip)/N(ip));
			for(int ip=0; ip<k; ip++) PC(Iopt(ip)) = 0;
			int mx; PC.maxCoeff(&mx);					//Find element the most aligned with the residual
			Iopt(k) = mx;
			{
				VectorXd r(k+1);
				MatrixXd Zk(nr,k+1);
				for(int i=0; i<k+1; i++){
					Zk.col(i) = Z.col(Iopt(i));
					r(i)      = Z.col(Iopt(i)).dot(R);
				}
				MatrixXd F = (Zk.transpose()*Zk).inverse();
				VectorXd x = F*r;
				for(int i=0; i<k+1; i++) Copt(Iopt(i)) = x(i);
			}
			k++;
		};		
		return Copt;
	};

	double LOO_Error(MatrixXd const &z, VectorXd const &Ob) const {
		MatrixXd F = (z.transpose()*z).inverse();
		VectorXd b = z.transpose()*Ob;
		VectorXd xt = F*b;
		VectorXd br = Ob - z*xt;
		MatrixXd F1 = z*F*z.transpose();
		double rloo1 = 0;
		for(int o=0; o<z.rows(); o++) rloo1 += pow(br(o)/(1-F1(o,o)),2);
		/*
		double rloo2 = 0;
		for(int o=0; o<z.rows(); o++){
			VectorXd u = z.row(o); 
			VectorXd p = b - u*Ob(o);
			VectorXd y = F*p;
			VectorXd q = F*u;
			VectorXd x = y + q *( u.dot(y) )/(1-u.dot(q)) ;
			rloo2 += pow((z*x-Ob)(o),2);
		}
		cout << "LOO error estimation : " << rloo1 << " and " << rloo2 << endl;
//		rloo /= (double)(z.rows());
		*/
		return (rloo1);
	};
	
private:	
	MatrixXd Z, Zc;
	VectorXd N;
	B* base;
	int nr, nc, np;
};
