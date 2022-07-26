#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <Eigen/Dense>
#include <math.h>
#include "pcb++.h"
using namespace std;
using namespace Eigen;

bool Comp_VectorXi(VectorXi U, VectorXi V){
	//! Brief Comparison of multi-indexes for sorting the polynomials
	if(U.sum() < V.sum()) return true;
	if(U.sum() > V.sum()) return false;
	for(int id=0; id<U.rows(); id++){
		if(U(id) > V(id)) return true;
		if(U(id) < V(id)) return false;
	}
	return true;
};

void PCB::TotalDegree(){
	//! Tensorization with respect to total degree <= nord
	list<VectorXi>::iterator p;
	list<VectorXi> Pol;
	VectorXi pol = VectorXi::Zero(ndim);
	while(pol(0)<=nord){Pol.push_back(pol); pol(0)++;}
	for(int id=1; id<ndim; id++){
		int npol = Pol.size(); p = Pol.begin();
		for(int el=0; el<npol; el++){
			pol = (*p); pol(id) ++;
			while(pol.sum()<=nord){ Pol.push_back(pol); pol(id)++;}
			p++;
		}
	}
	Pol.sort(Comp_VectorXi);
	npol = Pol.size(); alp = MatrixXi(npol,ndim);
	int ip=0; for(p=Pol.begin(); p!=Pol.end(); p++){ alp.row(ip) = (*p); ip++;}
};

void PCB::TotalDegree(VectorXd const &W){
	//! Tensorization with respect to total degree <= nord and directional weights W
	if(W.minCoeff()<=0){
		cout << "Weights for weighted L-1 tensorization must be positive \n";
		exit(1);
	}
	VectorXd Weff = VectorXd::Constant(ndim,1/W.minCoeff()).array()*W.array();
	list<VectorXi>::iterator p;
	list<VectorXi> Pol;
	VectorXi pol = VectorXi::Zero(ndim);
	while(Weff(0)*pol(0)<=nord){ Pol.push_back(pol); pol(0)++;}
	for(int id=1; id<ndim; id++){
		int npol = Pol.size();
		p = Pol.begin();
		for(int el=0; el<npol; el++){
			pol = (*p); pol(id) ++;
			double l1 =0; for(int d=0; d<ndim; d++) l1+=Weff(d)*pol(d);
			while( l1<=nord){
				Pol.push_back(pol); pol(id)++;
				l1 =0; for(int d=0; d<ndim; d++) l1+=Weff(d)*pol(d);
			}
			p++;
		}
	}
	Pol.sort(Comp_VectorXi);
	npol = Pol.size(); alp = MatrixXi(npol,ndim);
	int ip=0; for(p=Pol.begin(); p!=Pol.end(); p++){ alp.row(ip) = (*p); ip++;}
};

void PCB::HyperCross(){
	list<VectorXi>::iterator p;
	list<VectorXi> Pol;
	VectorXi pol = VectorXi::Zero(ndim);
	while(pol(0)<=nord){ Pol.push_back(pol); pol(0)++;}
	for(int id=1; id<ndim; id++){
		int npol = Pol.size();
		p = Pol.begin();
		for(int el=0; el<npol; el++){
			pol = (*p); pol(id) ++;
			int l1 =1; for(int d=0; d<ndim; d++) l1*=(pol(d)+1);
			while( l1<=nord+1){
				Pol.push_back(pol); pol(id)++;
				l1 =1; for(int d=0; d<ndim; d++) l1*=(pol(d)+1);
			}
			p++;
		}
	}
	Pol.sort(Comp_VectorXi);
	npol = Pol.size(); alp = MatrixXi(npol,ndim);
	int ip=0; for(p=Pol.begin(); p!=Pol.end(); p++){ alp.row(ip) = (*p); ip++;}
};

void PCB::FullTens(){
	list<VectorXi>::iterator p;
	list<VectorXi> Pol;
	VectorXi pol = VectorXi::Zero(ndim);
	while(pol(0)<=nord){ Pol.push_back(pol); pol(0)++;}
	for(int id=1; id<ndim; id++){
		int npol = Pol.size();
		p = Pol.begin();
		for(int el=0; el<npol; el++){
			pol = (*p); pol(id) ++;
			while( pol(id)<=nord){ Pol.push_back(pol); pol(id)++;}
			p++;
		}
	}
	Pol.sort(Comp_VectorXi);
	npol = Pol.size(); alp = MatrixXi(npol,ndim);
	int ip=0; for(p=Pol.begin(); p!=Pol.end(); p++){ alp.row(ip) = (*p); ip++;}
};

void PCB::FullTens(VectorXd const &W){
	if(W.minCoeff()<=0){
		cout << "Weights for full tensorization must be positive \n"; exit(1);
	}
	VectorXd Weff = VectorXd::Constant(ndim,1/W.minCoeff()).array()*W.array();
	list<VectorXi>::iterator p;
	list<VectorXi> Pol;
	VectorXi pol = VectorXi::Zero(ndim);
	while(Weff(0)*pol(0)<=nord){ Pol.push_back(pol); pol(0)++;}
	for(int id=1; id<ndim; id++){
		int npol = Pol.size();
		p = Pol.begin();
		for(int el=0; el<npol; el++){
			pol = (*p); pol(id) ++;
			while( Weff(id)*pol(id)<=nord){ Pol.push_back(pol); pol(id)++;}
			p++;
		}
	}
	Pol.sort(Comp_VectorXi);
	npol = Pol.size(); alp = MatrixXi(npol,ndim);
	int ip=0; for(p=Pol.begin(); p!=Pol.end(); p++){ alp.row(ip) = (*p); ip++;}
};

PCB::PCB(){
	/// Default constructor: \n
	ndim =-1;
	sprod=1;
	striple=1;
	sderive=1;
};

PCB::PCB(int ndi, int nor, char tin[], int const t_tens, bool d, bool p, bool t){
	/// Input parameters are: \n
	/// dimension of the stochastic space\n
	///	maximal polynomial order\n
	///	type of measure in each dimension\n
	/// type of tensorization (0: total degree, 1: full tensorization, 2: hyperbolic-cross)\n
	/// boolean for derivative operators
	/// boolean for product operator
	/// boolean for triple product operator
//	cout << "*** Create a new basis :\n";
	ndim = ndi; nord = nor;
	sderive = d; sprod= p; striple= t;
	type_tens = t_tens;
	for(int id=0; id<ndim; id++) type.push_back(tin[id]);
	Ctype = type[0]; for(int id=1; id<ndim; id++) if(type[id]!=Ctype) Ctype='M';
	Basis1D();
	BasisND();
	if(sprod   == 0) BasisProd();
	if(striple == 0) BasisTProd();
	if(sderive == 0) BasisDer();
};

PCB::PCB(int ndi, int nor, char tin[], int const t_tens, VectorXd const &We, bool d, bool p, bool t){
	/// Input parameters are: \n
	/// dimension of the stochastic space\n
	///	maximal polynomial order\n
	///	type of measure in each dimension\n
	/// type of tensorization (0: total degree, 1: full tensorization)\n
	/// weights for tensorization
	/// boolean for derivative operators
	/// boolean for product operator
	/// boolean for triple product operator
//	cout << "*** Create a new basis :\n";
	ndim = ndi; nord = nor;
	sderive = d; sprod= p; striple= t;
	type_tens = t_tens;
	for(int id=0; id<ndim; id++) type.push_back(tin[id]);
	Ctype = type[0]; for(int id=1; id<ndim; id++) if(type[id]!=Ctype) Ctype='M';
	Basis1D();
	BasisND(We);
	if(sprod   == 0) BasisProd();
	if(striple == 0) BasisTProd();
	if(sderive == 0) BasisDer();
};

PCB::PCB(int ndi, int nor, char tin, int const t_tens, bool d, bool p, bool t){
	/// Input parameters are: \n
	/// dimension of the stochastic space\n
	///	maximal polynomial order\n
	///	type of measure for all dimensions\n
	/// type of tensorization (0: total degree, 1: full tensorization)\n
	/// boolean for derivative operators
	/// boolean for product operator
	/// boolean for triple product operator
	if(tin!='U' && tin!='N'){ cout << "For this constructor use type 'U' or 'N' for the basis" << endl; exit(1);}
	sderive = d; sprod= p; striple= t;
	ndim = ndi; nord = nor; type_tens = t_tens;
	for(int id=0; id<ndim; id++) type.push_back(tin);
	Ctype = type[0]; for(int id=1; id<ndim; id++) if(type[id]!=Ctype) Ctype='M';
	Basis1D();
	BasisND();
	if(sprod   == 0) BasisProd();
	if(striple == 0) BasisTProd();
	if(sderive == 0) BasisDer();
};

PCB::PCB(int ndi, int nor, char tin, int const t_tens, VectorXd const &We, bool d, bool p, bool t){
	/// Input parameters are: \n
	/// dimension of the stochastic space\n
	///	maximal polynomial order\n
	///	type of measure for all dimensions\n
	/// type of tensorization (0: total degree, 1: full tensorization)\n
	/// weights for tensorization
	/// boolean for derivative operators
	/// boolean for product operator
	/// boolean for triple product operator
	cout << "*** Create a new basis :\n";
	if(tin!='U' && tin!='N'){ cout << "For this constructor use type 'U' or 'N' for the basis" << endl; exit(1);}
	sderive = d; sprod= p; striple= t;
	ndim = ndi; nord = nor; type_tens = t_tens;
	for(int id=0; id<ndim; id++) type.push_back(tin);
	Ctype = type[0]; for(int id=1; id<ndim; id++) if(type[id]!=Ctype) Ctype='M';
	for(int id=0; id<ndim; id++) printf("\tDirection %2d : Type %c\n",id+1,type[id]);
	cout <<"\tType of basis : " << Ctype << endl;
	Basis1D();
	BasisND(We);
	cout << "\tNord " << nord << " Ndim " << ndim << " Npol " << npol << endl;
	if(sprod   == 0) BasisProd();
	if(striple == 0) BasisTProd();
	if(sderive == 0) BasisDer();
};

PCB::PCB(list<VectorXi> &tens, char tin, bool d, bool p, bool t){
	/// Input parameters are: \n
	/// list of tensorization (multi-indexes)\n
	///	type of measure for all dimensions\n
	/// boolean for derivative operators
	/// boolean for product operator
	/// boolean for triple product operator
	if(tin!='U' && tin!='N'){ cout << "For this constructor use type 'U' or 'N' for the basis" << endl; exit(1);}
	sderive = d; sprod= p; striple= t;
	list<VectorXi>::iterator T; T = tens.begin(); ndim = (*T).rows();

	for(int id=0; id<ndim; id++) type.push_back(tin);
	Ctype = type[0]; for(int id=1; id<ndim; id++) if(type[id]!=Ctype) Ctype='M';	
	npol = tens.size(); alp = MatrixXi(npol,ndim); int ip=0;
	for(T=tens.begin(); T!=tens.end(); T++){ alp.row(ip) = (*T); ip++;}
	nord = alp.maxCoeff();
	Basis1D();
	if(sprod   == 0) BasisProd();
	if(striple == 0) BasisTProd();
	if(sderive == 0) BasisDer();
};

PCB::PCB(list<VectorXi> &tens, char tin[], bool d, bool p, bool t){
	/// Input parameters are: \n
	/// list of tensorization (multi-indexes)\n
	/// type of measure in each dimension\n
	/// boolean for derivative operators
	/// boolean for product operator
	/// boolean for triple product operator
	sderive = d; sprod= p; striple= t;
	list<VectorXi>::iterator T; T = tens.begin(); ndim = (*T).rows();
	for(int id=0; id<ndim; id++) type.push_back(tin[id]);
	Ctype = type[0]; for(int id=1; id<ndim; id++) if(type[id]!=Ctype) Ctype='M';
	npol = tens.size(); alp = MatrixXi(npol,ndim); int ip=0;
	for(T=tens.begin(); T!=tens.end(); T++){ alp.row(ip) = (*T); ip++;}
	nord = alp.maxCoeff();
	Basis1D();
	if(sprod   == 0) BasisProd();
	if(striple == 0) BasisTProd();
	if(sderive == 0) BasisDer();
};


void PCB::Basis1D(){
	nq1d = (nord+1)*2;
	xleg = VectorXd(nq1d); wleg = VectorXd(nq1d);
	gauss_legendre(nq1d, xleg, wleg);
	xher = VectorXd(nq1d); wher = VectorXd(nq1d);
	gauss_hermite(nq1d, xher, wher);
	if(Ctype!='U' && Ctype!='N' && Ctype!='M'){ cout << "basis is not of type U, N or M --> Stop "<<endl; exit(1);}
};


void PCB::BasisND(){
	if(type_tens==0){
//		cout << "\tUse total degree tensorization of 1D polynomials\n"; 
		TotalDegree();
	}else if(type_tens==1){
//		cout << "\tUse full tensorization of 1D polynomials\n"; 
		FullTens();
	}else if(type_tens==2){
//		cout << "\tUse hyperbolic cross tensorization of 1D polynomials\n";
 		HyperCross();
	}else{
		cout << "Don't know this type of tensorization\n" << endl; exit(1);
	}
};

void PCB::BasisND(VectorXd const &We){
	if(type_tens==0){
		cout << "\tUse weighted total degree tensorization of 1D polynomials\n"; 
		TotalDegree(We);
	}else if(type_tens==1){
		cout << "\tUse weighted full tensorization of 1D polynomials\n"; 
		FullTens(We);
	}else{
		cout << "Don't know this type of Weighted tensorization\n" << endl; exit(1);
	}
};

void PCB::clear(){
	if(sprod   == 0) delete [] Prod;
	if(striple == 0) delete [] Triple;
	if(sderive == 0) delete [] Deriv;
	npol=0;
	nord=0;
	ndim=0;
	Ctype = 'U';
};

PCB::~PCB(){
//  		cout << "*** PCB Destructor\n";
  	clear();
};

