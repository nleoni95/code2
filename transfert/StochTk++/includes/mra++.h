#ifndef PCB_H_
#include "pcb++.h"
#endif

#ifndef MRA_H_
#define MRA_H_
#endif
#ifndef PCB_H_
#include "pcb++.h"
#endif

using namespace std;
using namespace Eigen;

struct op_mra {
	int ip, jp;
	double cl, cr;
};

//! \brief The MRA class builds on the PCB class to define multi-resolution analysis tools,
//!	in particular the projection and restriction operators in each direction. \n
//! It  also provide utilities to apply this operators at an elemental level
class MRA : public PCB {
	template <class B, class T> friend class Stoch;
	template <class B, class T> friend class VStoch;
	template <class T> friend class TREE;
private:
	int ndim;					/*!< dimension of the stochatsic space */
	int npol;					/*!< dimension of the (Legendre) polynomial basis */
	int nord;					/*!< maximal order of the polynomial basis */
	int nq2s;					/*!< dimension of the 2-side quadrature */
	VectorXd x2s; 				/*!< Vector of quadrature points */
	VectorXd w2s;				/*!< Vector of quadrature weights */
	MatrixXd llleg, rlleg;
	list<op_mra> *Prd;			/*!< Prediction operators */
	list<op_mra> *Red;			/*!< Reduction operators */
	list<op_mra> *Prp;			/*!< Details projection operators */
public:
	//! \brief Constructor
	MRA(int nd, int no, int ttens, bool td, bool tp) : PCB(nd, no,'U', ttens, td, tp){
		//! Input: number of dimensions, maximal order, type of tensorization, booleans for derivative and product operator
		npol = Npol();
		ndim = Ndim();
		nord = Nord();
		DoDetails();
	};
	//! \brief Destructor
	~MRA(){
		cout << "*** MRA destructor\n";
		delete [] Prd;
		delete [] Prp;
		delete [] Red;
	};

	//! \brief Computes the detail functions at point x in [0,1]x....x[0,1]
	VectorXd Comp_Det(VectorXd const &xi,int const dir);
	//! \brief Prediction of children from parent and detail (synthesis)
	template <class T> void Predict(T const &P, T const &D, T &L, T &R, int const dir);
	//! \brief Prediction of children from parent only	(prolongation)
	template <class T> void Predict(T const &P, T &L, T &R, int const dir);

	//! \brief Prediction of left child from parent only (prolongation)
	template <class T> T PredictLeft(T const &P, int const dir);

	//! \brief Prediction of right child from parent only (prolongation)
	template <class T> T PredictRight(T const &P, int const dir);

	//! \brief Restrict children into parent and detail	 (injection)
	template <class T> void Restrict(T const &L, T const &R, T &P, T &D, int const dir);
	//! \brief Restrict children to parent (projection)
	template <class T> T Restrict(T const &L, T const &R, int const dir);
	//! \brief Extract detail from pair of children
	template <class T> T GetDet(T const &L, T const &R, int const dir);

private:
	//! \brief Create the MRA operators
	void DoDetails();
};

template <class T>
void MRA::Predict(T const &P, T const &D, T &L, T &R, int const dir){
	//! Input: Parent, Detail\n
	//! Ouput: Left and Right children
	L=P; L*=0; R=P; R*=0;
	list<op_mra>::iterator it;
	for(it=Prp[dir].begin(); it!=Prp[dir].end(); it++){
		L((*it).ip) += P((*it).jp)*(*it).cl;
		R((*it).ip) += P((*it).jp)*(*it).cr;
	}
	for(it=Prd[dir].begin(); it!=Prd[dir].end(); it++){
		L((*it).ip) += D((*it).jp)*(*it).cl;
		R((*it).ip) += D((*it).jp)*(*it).cr;
	}
};

template <class T>
void MRA::Predict(T const &P, T &L, T &R, int const dir){
	//! Input: Parent\n
	//! Ouput: Left and Right children
	L=P; L*=0; R=P;  R*=0;
	list<op_mra>::iterator it;
	for(it=Prp[dir].begin(); it!=Prp[dir].end(); it++){
		L((*it).ip) += P((*it).jp)*(*it).cl;
		R((*it).ip) += P((*it).jp)*(*it).cr;
	}
};

template <class T>
T MRA::PredictRight(T const &P, int const dir){
	//! Input: Parent\n
	//! Ouput: Right child
	T R = P;  R*=0;
	list<op_mra>::iterator it;
	for(it=Prp[dir].begin(); it!=Prp[dir].end(); it++){
		R((*it).ip) += P((*it).jp)*(*it).cr;
	}
	return R;
};

template <class T>
T MRA::PredictLeft(T const &P, int const dir){
	//! Input: Parent\n
	//! Ouput: Left child
	T L=P; L*=0; 
	list<op_mra>::iterator it;
	for(it=Prp[dir].begin(); it!=Prp[dir].end(); it++){
		L((*it).ip) += P((*it).jp)*(*it).cl;
	}
	return L;
};



template <class T>
void MRA::Restrict(T const &L, T const &R, T &P, T &D, int const dir){
	//! Input: Left and Right children\n
	//! Ouput: Parent and Detail
	P = L; P*=0; D = R; D*=0;
	list<op_mra>::iterator it;
	for(it=Prp[dir].begin(); it!=Prp[dir].end(); it++){
		P((*it).jp) += L((*it).ip)*(*it).cl + R((*it).ip)*(*it).cr;
	}
	for(it=Red[dir].begin(); it!=Red[dir].end(); it++){
		D((*it).ip) += L((*it).jp)*(*it).cl + R((*it).jp)*(*it).cr;
	}
	P *= 0.5;
};



template <class T>
	//! Input: Left and Right children\n
	//! returns: Detail
T MRA::GetDet(T const &L, T const &R, int const dir){
	T D = L; D*=0;
	list<op_mra>::iterator it;
	for(it=Red[dir].begin(); it!=Red[dir].end(); it++){
		D((*it).ip) += L((*it).jp)*(*it).cl + R((*it).jp)*(*it).cr;
	}
	return D;
};

template <class T>
T MRA::Restrict(T const &L, T const &R, int const dir){
	//! Input: Left and Right children\n
	//! returns: Parent
	T P = L; P*=0;
	list<op_mra>::iterator it;
	for(it=Prp[dir].begin(); it!=Prp[dir].end(); it++){
		P((*it).jp) += L((*it).ip)*(*it).cl + R((*it).ip)*(*it).cr;
	}
	return P *= 0.5;
};
