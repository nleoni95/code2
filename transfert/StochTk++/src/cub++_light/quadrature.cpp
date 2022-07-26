#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <list>
#include <set>
#include <Eigen/Dense>
#include "cub++_light.h"
using namespace Eigen;
using namespace std;

bool Comp_set(VectorXi U, VectorXi V){
  if(U.sum() < V.sum()) return true;
  if(U.sum() > V.sum()) return false;
  for(int id=0; id<U.rows(); id++){
	if(U(id) < V(id) ) return true;
	if(U(id) > V(id) ) return false;
  }
  return false;
};

void RULEQUAD::prep_fejer(){
	Npt(0) = 1;
	Npn(0) = 1;
	for(int il=1; il<nl; il++){
		Npt(il) = Npt(il-1)*2+1;
		Npn(il) = Npt(il) - Npt(il-1);
	}
	int Nq = Npt(nl-1);
//Prepare points:
	Xp = VectorXd(Nq);
	for(int i=0; i<Nq; i++) Xp(i) = (1.-cos( M_PI*(i+1)/(double)(Nq+1)))*.5;
//Prepare weights:
	Wght = MatrixXd::Zero(Nq,nl);
	for(int il=0; il<nl; il++){
		int nq = Npt(il);
		VectorXd tet(nq);
		for(int i=0; i<nq; i++) tet(i) = M_PI*(i+1)/(double)(nq+1);
		double sw = 0.;
		for(int i=0; i<nq; i++){
			double s=0.;
			for(int j=1; j<= (nq+1)/2; j++) s += sin((2*j-1)*tet(i))/ (double)(2*j-1);
			Wght(i,il) = 2.*sin(tet(i))/(double)(Npt(il)+1)*s;
			sw += Wght(i,il);
		}
	}
	Wghto = Wght;
	for(int il=nl-1; il>0; il--)
	{
		for(int i=0; i<Npt(il-1); i++) Wght(2*i+1,il) -= Wght(i,il-1);
	}
	Xind = MatrixXi::Zero(Nq,nl);
	for(int i=0; i<Npt(nl-1); i++) Xind(i,nl-1) = i;
	for(int il=nl-2; il>=0; il--)
		for(int i=0; i<Npt(il); i++) Xind(i,il) = Xind(2*i+1,il+1);
	Appl = MatrixXi::Ones(Nq,nl)*(-1);
	int inc = Npt(nl-1)/Npt(0)/2+1;
	for(int il=0; il<nl; il++){
		int ic=0;
		for(int i=inc-1; i<Npt(nl-1); i+=inc){
			Appl(i,il) = ic;
			ic++;
		}
		inc *=.5;
	}
	cout << "\t\t Fejer's quadrature nodes & weights are ready\n";
	cout << "\t\t # of points per level  : " << Npt.transpose() << endl;
};
