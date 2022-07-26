#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <Eigen/Dense>
#include <math.h>
#include "pcb++.h"
using namespace std;
using namespace Eigen;

VectorXd PCB::Comp_Psi(VectorXd const &xi){
	/// Input : xi, ndim-coordinate
	/// Returns : npol-vector of polynomials values
	VectorXd p1d, psi = VectorXd::Ones(npol);
  	for(int id=0; id<ndim; id++){
		if(type[id] == 'U'){
    			p1d = pleg_at_x(xi(id));
		}else{
			p1d = pher_at_x(xi(id));
		}
    	for(int k=0; k<npol; k++) psi(k) *= p1d(alp(k,id));
  	}
	return psi;
};

MatrixXd PCB::Comp_Psi(MatrixXd const &xi){
	/// Input : matrix (ndim,.) whose columns are ndim-coordinates
	/// Returns : matrix (.,npol) whore rows contain the polynmials values
	MatrixXd psi = MatrixXd::Ones(xi.cols(),npol);
	MatrixXd p1d;
  	for(int id=0; id<ndim; id++){
		if(type[id] == 'U'){
    			p1d = pleg_at_x(xi.row(id));
		}else{
			p1d = pher_at_x(xi.row(id));
		}
    		for(int k=0; k<npol; k++) psi.col(k).array() *= p1d.col(alp(k,id)).array();
  	}
	return psi;
};

VectorXd PCB::pleg_at_x(double const x){
	/// Input : location
	/// Returns : vector of polynomial values
	VectorXd p = VectorXd::Zero(nord+1);
	double xx = x*2 - 1.;
	p(0) = 1.;
	if(nord==0) return p;
	p(1) = xx;
	for(int i=2; i<=nord; i++) p(i)=((2*i-1)*p(i-1)*xx-(i-1)*p(i-2))/i;
	for(int i=1;i<=nord;i++) p(i) *= sqrt(2*i+1);
	return p;
};
 
MatrixXd PCB::pleg_at_x(const VectorXd &x){
	/// Input : Vector of locations
	/// Returns : matrix of polynomial values
	int n=x.rows();
	VectorXd xx = x*2; xx -= VectorXd::Ones(n);
	MatrixXd p(n,nord+1);
	p.col(0) = VectorXd::Ones(n);
	if(nord==0) return p;
	p.col(1) = xx;
	for(int i=2; i<=nord; i++) p.col(i).array()= (p.col(i-1).array()*xx.array()*(2*i-1) -p.col(i-2).array()*(i-1))/i;
	for(int i=1; i<=nord; i++) p.col(i) *= sqrt(2*i+1);
	return p;
};

VectorXd PCB::dpleg_at_x(double const x){
	/// Input : location
	/// Returns : vector of polynomial derivative values
	double xx = x*2 - 1.;
	VectorXd p = VectorXd(nord+1);
	VectorXd d = VectorXd(nord+1);
	p(0)  = 1.; d(0) = 0.;
	if(nord==0) return d;
	p(1)  = xx; d(1) = 1.;
	for(int i=2; i<=nord; i++){
		d(i) = ( (2*i-1)*(d(i-1)*xx+p(i-1)) - (i-1)*d(i-2) )/i;
		p(i) = ( (2*i-1)*p(i-1)*xx          - (i-1)*p(i-2) )/i;
	}
	for(int i=1;i<=nord;i++) d(i) *= (sqrt(8*i+4));
	return d;
};

MatrixXd PCB::dpleg_at_x(const VectorXd &x){
	/// Input : Vector of locations
	/// Returns : matrix of polynomial derivatives values
	int n=x.rows();
	VectorXd xx = x*2; xx -= VectorXd::Ones(n);
	MatrixXd p(n,nord+1);
	MatrixXd d(n,nord+1);
	p.col(0) = VectorXd::Ones(n);
	d.col(0) = VectorXd::Zero(n);
	if(nord==0) return p;
	p.col(1) = xx;
	d.col(1) = p.col(0);
	for(int i=2; i<=nord; i++){
		p.col(i).array()= (p.col(i-1).array()*xx.array()*(2*i-1) - p.col(i-2).array()*(i-1))/i;
		d.col(i).array()= ((d.col(i-1).array()*xx.array()+p.col(i-1).array())*(2*i-1) - d.col(i-2).array()*(i-1))/i;
	}
	for(int i=1; i<=nord; i++) d.col(i) *= sqrt(8*i+4);
	return d;
};

VectorXd PCB::pher_at_x(double const  x){
	/// Input : location
	/// Return : vector of polynomial values
	VectorXd p(nord+1);
	p(0) = 1.;
	if(nord==0) return p;
	p(1) = x;
	for(int i=2; i<=nord; i++) p(i) = p(i-1)*x-(i-1)*p(i-2);
	double c =1.;
	for(int i=2;i<=nord;i++){
		c *= (double)(i);
		p(i) *= (1./sqrt(c));
	}
	return p;
};

MatrixXd PCB::pher_at_x(VectorXd const  &x){
	/// Input : Vector of locations
	/// Returns : matrix of polynomial values
	MatrixXd p(x.rows(),nord+1);
	p.col(0) = VectorXd::Ones(x.rows());
	if(nord==0) return p;
	p.col(1) = x;
	for(int i=2; i<=nord; i++) p.col(i) = p.col(i-1).array()*x.array()-(i-1)*p.col(i-2).array();
	double c =1.;
	for(int i=2;i<=nord;i++){
		c *= (double)(i);
		p.col(i) *= (1./sqrt(c));
	}
	return p;
};

VectorXd PCB::dpher_at_x(double const  x){
	/// Input : location
	/// Returns : vector of polynomial derivative values
	VectorXd p(nord+1); VectorXd d(nord+1);
	p(0) = 1.; d(0) = 0.;
	if(nord==0) return d;
	p(1) = x; d(1) = 1.;
	if(nord==1) return (d*2);
	for(int i=2; i<=nord; i++){
		p(i) = p(i-1)*x-(i-1)*p(i-2);
		d(i) = p(i-1) + d(i-1)*x -(i-1)*d(i-2);
	}
	double c =1.;
	for(int i=2;i<=nord;i++){
		c *= (double)(i);
		d(i) *= (1./sqrt(c));
	}
	return d;
};

MatrixXd PCB::dpher_at_x(VectorXd const &x){
	/// Input : Vector of locations
	/// Returns : matrix of polynomial derivative values
	MatrixXd p(x.rows(),nord+1); VectorXd d(x.rows(),nord+1);
	p.col(0) = VectorXd::Ones(x.rows()); d.col(0) = VectorXd::Zero(x.rows());
	if(nord==0) return d;
	p.col(1) = x; d.col(1) = p.col(0);
	if(nord==1) return (d*2);
	for(int i=2; i<=nord; i++){
		p.col(i) = p.col(i-1).array()*x.array()-(i-1)*p.col(i-2).array();
		d.col(i) = p.col(i-1).array() + d.col(i-1).array()*x.array() -(i-1)*d.col(i-2).array();
	}
	double c =1.;
	for(int i=2;i<=nord;i++){
		c *= (double)(i);
		d.col(i) *= (1./sqrt(c));
	}
	return d;
};
