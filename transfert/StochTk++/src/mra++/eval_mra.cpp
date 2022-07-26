#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <Eigen/Dense>
#include "pcb++.h"
#include "mra++.h"
using namespace std;
using namespace Eigen;

VectorXd MRA::Comp_Det(VectorXd const &xi,int const dir){
	//! Input: vector of coordinates xi
	//! Returns: Vector of detail functions values
	VectorXd xl=xi;
	VectorXd W = VectorXd::Zero(npol);
	if(xl(dir)<.5){
		xl(dir) *= 2;
		VectorXd Pl = Comp_Psi(xl);
		for(list<op_mra>::iterator it=Prd[dir].begin(); it!=Prd[dir].end(); it++) W((*it).jp) += Pl((*it).ip)*(*it).cl;
	}else{
		xl(dir) = xl(dir)*2-1;
		VectorXd Pl = Comp_Psi(xl);
		for(list<op_mra>::iterator it=Prd[dir].begin(); it!=Prd[dir].end(); it++) W((*it).jp) += Pl((*it).ip)*(*it).cr;
	}
  return W;
};


