#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <Eigen/Dense>
#include <math.h>
#include "pcb++.h"
using namespace std;
using namespace Eigen;

VectorXd PCB::Derive_PCE(VectorXd const &pc, int const dir){
	/// Input: PC coefficients and direction of derivation \n
	/// Return: PC coefficients of the derivative
	VectorXd dp = VectorXd::Zero(npol);
	list<op_pc>::iterator it;
	for(it=Deriv[dir].begin(); it!=Deriv[dir].end(); it++) dp((*it).i) += pc((*it).j)*((*it).c);
	return dp;
};

MatrixXd PCB::Derive_PCE(MatrixXd const &pc, int const dir){
	/// Input: matrix (npol,.) of PC coefficients and direction of derivation \n
	/// Return: matrix (npol,.) of PC coefficients of the derivative
	MatrixXd dp = MatrixXd::Zero(npol,pc.cols());
	list<op_pc>::iterator it;
	for(it=Deriv[dir].begin(); it!=Deriv[dir].end(); it++) dp.row((*it).i) += pc.row((*it).j)*((*it).c);
	return dp;
};

void PCB::BasisDer(){
/// Construct the spectral derivative tensor of the basis.
	cout << "\tConstruct Derivative tensors:" << endl;
	MatrixXd dleg, dher;
	VectorXd px, dpx;
	double eps = 1.e-8;
	if(Ctype =='U' || Ctype == 'M'){
		dleg = MatrixXd::Zero(nord+1,nord+1);
		for(int iq=0; iq<nq1d; iq++){
			px  = pleg_at_x(xleg[iq]); dpx = dpleg_at_x(xleg[iq]);
			for(int i1=0; i1<=nord; i1++)
				for(int i2=0; i2<=nord; i2++)dleg(i1,i2) += px[i1]*dpx[i2]*wleg[iq];
		}
		for(int i1=0; i1<=nord; i1++)
			for(int i2=0; i2<=nord; i2++) if(fabs(dleg(i1,i2))<=eps) dleg(i1,i2)=0.;
//		cout << "Derivative operator for Hermite polynomials " << endl << dleg << endl;	
	}
	if (Ctype =='N' || Ctype == 'M'){
		dher = MatrixXd::Zero(nord+1,nord+1);
		for(int iq=0; iq<nq1d; iq++){
			px  = pher_at_x(xher(iq)); dpx = dpher_at_x(xher(iq));
			for(int i1=0; i1<=nord; i1++)
				for(int i2=0; i2<=nord; i2++) dher(i1,i2) += px[i1]*dpx[i2]*wher[iq];
		}
		for(int i1=0; i1<=nord; i1++)
			for(int i2=0; i2<=nord; i2++) if(fabs(dher(i1,i2))<=eps) dher(i1,i2)=0.;
//		cout << "Derivative operator for Hermite polynomials " << endl << dher << endl;
	}
	Deriv = new list<op_pc> [ndim];
	for(int id=0; id<ndim; id++){
		MatrixXd Der = MatrixXd::Zero(npol,npol);
		for(int ip=0; ip<npol; ip++){
			if(alp(ip,id)>0){
				for(int jp=0; jp<npol; jp++){
					int dist= (alp.row(ip).array()-alp.row(jp).array()).abs().sum();
					if(dist==fabs(alp(ip,id)-alp(jp,id))){
					  if(type[id] == 'U'){
							Der(jp,ip) += dleg(alp(jp,id),alp(ip,id));
						}else{
							Der(jp,ip) += dher(alp(jp,id),alp(ip,id));
						}
					}
				}
			}
		}
		op_pc elem; elem.l=0;
		for(int ip=0; ip<npol; ip++){
			elem.i = ip;
			for(int jp=0; jp<npol; jp++) if(Der(ip,jp) != 0){ elem.j = jp; elem.c = Der(ip,jp); Deriv[id].push_back(elem);}
		}
	}
};
