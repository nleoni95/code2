#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <Eigen/Dense>
#include "pcb++.h"
#include "mra++.h"
using namespace std;
using namespace Eigen;

void MRA::DoDetails(){
	cout << "*** Do the MRA operators:\n";
	//Define quadrature and integration points
	nq2s = Nq1d()*2;			//Quadrature size
	x2s = VectorXd::Zero(nq2s);
	w2s = VectorXd::Zero(nq2s);
	gauss_legendre(Nq1d(), x2s, w2s);
	x2s *= 0.5; w2s *= 0.5;
	for(int iq=0; iq<Nq1d(); iq++){						//make it a two-sided quadature
		x2s(iq+Nq1d()) = x2s(iq) + .5;
		w2s(iq+Nq1d()) = w2s(iq);
	}
	//Make the 1D detail function
	llleg = MatrixXd::Zero(nord+1,nord+1);
	rlleg = MatrixXd::Zero(nord+1,nord+1);
	for(int iq=0; iq<Nq1d(); iq++){  //just half integration !!!!!
		VectorXd PLG  = pleg_at_x(x2s(iq)*2.);  //Legendre polynomial
		VectorXd PLTL = pleg_at_x(x2s(iq));
		VectorXd PLTR = pleg_at_x(x2s(iq+Nq1d()));
		for(int io=0; io<=nord; io++){	//for each of the function
			llleg.row(io) += PLG*(PLTL(io)*w2s(iq));
			rlleg.row(io) += PLG*(PLTR(io)*w2s(iq));
		}
	}
	llleg *= 2;
	rlleg *= 2;
	//Construct the multi-index set of the Detail basis in each direction:
	Prd = new list<op_mra> [Ndim()];
	Prp = new list<op_mra> [Ndim()];
	Red = new list<op_mra> [Ndim()];
	for(int dir=0; dir<Ndim(); dir++){
		//First we define the projection on restriction operators
		MatrixXd PL = MatrixXd::Zero(npol,npol);
		MatrixXd PR = MatrixXd::Zero(npol,npol);
		for(int ip=0; ip<npol; ip++){
			VectorXi a = Alp(ip);
			for(int jp=0; jp<npol; jp++){
				VectorXi b = Alp(jp);
				VectorXi dif = a-b; dif(dir) = 0; dif = dif.array().abs();
				if(dif.maxCoeff() == 0 ){
					PL(jp,ip) += llleg(a(dir),b(dir));
					PR(jp,ip) += rlleg(a(dir),b(dir));
				}
			}
		}
		op_mra proj;
		for(int ip=0; ip<npol; ip++){
			proj.ip = ip;
			for(int jp=0; jp<npol; jp++){
				proj.jp = jp;
				if(fabs(PL(ip,jp)) > 1.e-10){
					proj.cl = PL(ip,jp);
					proj.cr = PR(ip,jp);
					Prp[dir].push_back(proj);
				} else if(fabs(PR(ip,jp)) > 1.e-10 ) { cout << "Clearly wrong !!!! " << endl; exit(1);}
			}
		}
		//The projection operator for continuous functions is set.
		cout << "\tDirection " << dir << "\t Size of Projection op " << Prp[dir].size() << endl;

		MatrixXd Dl = MatrixXd::Zero(npol,npol);
		MatrixXd Dr = MatrixXd::Zero(npol,npol);
//Construct a set of vectors spanning the detail space by applying Pred o Rest operator
		for(int ip=0; ip<npol; ip++){
			VectorXd Xl = VectorXd::Zero(npol); Xl(ip) = 1;
			VectorXd Xr = VectorXd::Zero(npol); Xr(ip) =-1.;
//Remove component // to the Legendre Basis
			for(int jp=0; jp<npol; jp++){
				double cof = (Xl.dot(PL.col(jp)) + Xr.dot(PR.col(jp)))*.5;
				Xl -= PL.col(jp)*cof; Xr -= PR.col(jp)*cof;
			}
			double norm = (Xl.squaredNorm() + Xr.squaredNorm())*.5;
			norm = 1/sqrt(norm);
			Xl *= norm; Xr *= norm;
			Dl.col(ip) = Xl; Dr.col(ip) = Xr;
		}
//Orthonormalize the set of vectors to make it an orthonormal basis
		for(int ip=0; ip<npol; ip++){
			double norm = 1./sqrt((Dl.col(ip).squaredNorm() + Dr.col(ip).squaredNorm())*.5);
			Dl.col(ip) *=norm; Dr.col(ip) *=norm;
			for(int jp=ip+1; jp<npol; jp++){
				double cof = (Dl.col(ip).dot(Dl.col(jp)) + Dr.col(ip).dot(Dr.col(jp)))*.5;
				Dl.col(jp) -= Dl.col(ip)*cof;
				Dr.col(jp) -= Dr.col(ip)*cof;
			}
			VectorXd Xl = Dl.col(ip); VectorXd Xr = Dr.col(ip);
		}
//Define the multidimensional detail functions as multidimensional Legendre expansions on left and right part
		for(int ip=0; ip<npol; ip++){
			proj.ip = ip;
			for(int jp=0; jp<npol; jp++){
				proj.jp = jp;
				if(fabs(Dl(ip,jp)) >1.e-10 || fabs(Dr(ip,jp) > 1.e-10)){
					proj.cl = Dl(ip,jp);
					proj.cr = Dr(ip,jp);
					Prd[dir].push_back(proj);
				}
			}
		}
//Do Restriction of details:
		for(int ip=0; ip<npol; ip++){
			VectorXd C = VectorXd::Zero(npol); C(ip) = -1; VectorXd R = VectorXd::Zero(npol);
			VectorXd P = Restrict(C,R,dir); Predict(P,C,R,dir); C(ip) +=1;
			for(int jp=0; jp<npol; jp++) PL(jp,ip) = (C.dot(Dl.col(jp)) + R.dot(Dr.col(jp)))*.5;
			C *=0; R *=0; R(ip) = -1; P = Restrict(C,R,dir); Predict(P,C,R,dir); R(ip) +=1;
			for(int jp=0; jp<npol; jp++) PR(jp,ip) = (C.dot(Dl.col(jp)) + R.dot(Dr.col(jp)))*.5;
		}
		for(int ip=0; ip<npol; ip++){
			proj.ip = ip;
			for(int jp=0; jp<npol; jp++){
				proj.jp = jp;
				if(fabs(PL(ip,jp))>1.e-10){
					proj.cl = PL(ip,jp);
					proj.cr = PR(ip,jp);
					Red[dir].push_back(proj);
				}
			}
		}
	}
	cout <<"\tDetails and operators are all set!\n";
};
