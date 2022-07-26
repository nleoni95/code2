#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <list>
#include <set>
#include <Eigen/Dense>
#include "cub++.h"
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
	int Nq = Npt(nl-1); // total number of points of the highest quadrature level (nl-1)
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
	// std::cout << "Here are the FJ weights Wghto:\n" << Wghto << std::endl;
	
	for(int il=nl-1; il>0; il--)
	{
		for(int i=0; i<Npt(il-1); i++) Wght(2*i+1,il) -= Wght(i,il-1);
	}
	// std::cout << "Here are the FJ weight differences Wght:\n" << Wght << std::endl;
	
	// Global and local indexing
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
	
	cout << "\t\t 1D Fejer's quadrature nodes & weights are ready\n";
	cout << "\t\t # of points per level  : " << Npt.transpose() << endl;
};


void RULEQUAD::prep_clenshaw(){
	Npt(0) = 1;	Npn(0) = 1;
	if(nl>1){
	Npt(1) = 3; Npn(1) = 2;
	}
	if(nl>2){	
		for(int il=2; il<nl; il++){
			Npt(il) = Npt(il-1)*2-1; // total number of points
			Npn(il) = Npt(il) - Npt(il-1); // incremental number of points
		}
	}
	int Nq = Npt(nl-1);
	// std::cout << "Total number of points Nq:\n" << Nq << std::endl;
	
//Prepare points:
	Xp = VectorXd(Nq);
	if (Nq>1) for(int i=0; i<Nq; i++) Xp(i) = (1.-cos( M_PI*(i)/(double)(Nq-1)))*.5;
	else Xp(0) = 1.;
	// std::cout << "Here are the CC points:\n" << Xp << std::endl;
	
//Prepare weights:
	Wght = MatrixXd::Zero(Nq,nl);
	for(int il=0; il<nl; il++){
		int nq = Npt(il); 
		double d1 = 1./(nq-1.); double d2 = 1./nq/(nq-2.);
		// double sw = 0.; // uncomment to check the weight sum
		for(int i=1; i<nq-1; i++){ // excludes the boundary points
			double s=0.;
			for(int j=1; j<= (nq-3)/2; j++){
				s += cos(2.*M_PI*j*i*d1)/ (double)(4*j*j-1);
			}
			Wght(i,il) = 1.-cos(M_PI*i)*d2-2.*s; Wght(i,il) = Wght(i,il)*d1;
			// sw += Wght(i,il); // uncomment to check the weight sum
		}
		Wght(0,il) = d2*0.5; Wght(nq-1,il) = Wght(0,il) ; // boundary points
		// sw += 2.*Wght(0,il); std::cout << "Sum check:\n" << sw << std::endl; // uncomment to check the weight sum
	}
	Wght(0,0) = 1.0;
	
	Wghto = Wght;
	
	// std::cout << "Here are the CC weights Wghto:\n" << Wghto << std::endl;
	
	for(int il=nl-1; il>1; il--)
		for(int i=0; i<Npt(il-1); i++) Wght(2*i,il) -= Wght(i,il-1);
	Wght(1,1) -= Wght(0,0); // takes care of the first 2 levels
	
	// std::cout << "Here are the CC weight differences Wght:\n" << Wght << std::endl;
	
	// Global and local indexing	
	Xind = MatrixXi::Zero(Nq,nl);
	for(int i=0; i<Npt(nl-1); i++) Xind(i,nl-1) = i;
	
	for(int il=nl-2; il>0; il--)
		for(int i=0; i<Npt(il); i++) Xind(i,il) = Xind(2*i,il+1);
	Xind(0,0) = Xind(1,1);
			
	Appl = MatrixXi::Ones(Nq,nl)*(-1);

	int inc = Npt(nl-1)/Npt(0)/2;	
	Appl(inc,0) = 0;
	
	for(int il=1; il<nl; il++){
		int ic=0;
		for(int i=0; i<Npt(nl-1); i+=inc){
			Appl(i,il) = ic; ic++;
		}
		inc = floor(inc*.5);		
	}
		
	cout << "\t\t 1D Clenshaw-Curtis quadrature nodes & weights are ready\n";
	cout << "\t\t # of points per level  : " << Npt.transpose() << endl;
};