#ifndef Stoch_H_
#include "stoch++.h"
#endif

#ifndef Vstoch_H_
#include "vstoch++.h"
#endif

#ifndef Mstoch_H_
#define Mstoch_H_
#endif


template <class B, class T> class MStoch;


template <class B, class T> class MStoch {

private:
	B *base;
	T *Mmod;
	
public:
	//! \brief basic constructor
	MStoch(){ base = NULL;};
	//! \brief constructor form a basis
	MStoch(B *b){ base = b; Mmod = new T[base->Npol()];};
	//! \brief copy constructor
	MStoch(const MStoch<B,T> &V){ base = V.base; Mmod = new T[base->Npol()]; for(int ip=0; ip<base->Npol(); ip++) Mmod[ip] = V[ip];};
	//! \brief destructor
	~MStoch(){delete[] Mmod;};
	
	MStoch(B *b, const T &M){
		base = b; Mmod = new T[base->Npol()];
		for(int ip=0; ip<base->Npol(); ip++) Mmod[ip] = M;
	};
	
	//! \brief overloaded = operator
	void operator=(const MStoch<B,T> &V){  
		if(Mmod==NULL) delete Mmod;
		base = V.base;
		Mmod = new T[base->Npol()];
		for(int ip=0; ip<base->Npol(); ip++) Mmod[ip] = V.Mmod[ip]; 
	};
	
	void operator*=(double const v) {
		for(int i=0; i<base->Npol(); i++){
			Mmod[i] *= v;
		}
	};
	
	
	//! \brief retrieve stochastoc matrix's column
	VStoch<B,double> col(const int ic) const {
		VStoch<B,double> temp(base,Mmod[0].rows());
		for(int ip=0; ip<base->Npol(); ip++) temp.set_mode(ip,Mmod[ip].col(ic));
		return temp;
	};
	
	//! \brief set stochastic matrix's mode	
	void set_mode(int i, const T &V){ Mmod[i] = V;};
	//! \brief add to stochastic matrix's mode	
	void add_mode(int i, const T &V){ Mmod[i]+= V;};
	//! \brief substract to stochastic matrix's mode	
	void sub_mode(int i, const T &V){ Mmod[i]-= V;};
	//! \brief set stochastic matrix's component	
	void set_entry(const int i, const int j, Stoch<B,double> &V){ for(int ip=0; ip<base->Npol(); ip++) Mmod[ip](i,j) = V.u(ip);};
	//! \brief add to stochastic vector's component
	void add_entry(int i, int j, Stoch<B,double> &V){ for(int ip=0; ip<base->Npol(); ip++) Mmod[ip](i,j) += V.u(ip);};
	//! \brief substract to stochastic vector's component		
	void sub_entry(int i, int j, Stoch<B,double> &V){ for(int ip=0; ip<base->Npol(); ip++) Mmod[ip](i,j) -= V.u(ip);};
	//! \brief retrieve stochastic vector's mode		
	T operator[](const int i) const { return Mmod[i];};
	
	T Mean() const { return Mmod[0];};
	T Std() const { 
		T ret = Mmod[0]*0;
		for(int ip=1; ip<base->Npol(); ip++){
			ret.array() += Mmod[ip].array().square(); 
		}
		ret.array() = ret.array().sqrt();
		return ret;
	};


	
	Stoch<B,double> operator ()(const int i, const int j) const {
		Stoch<B,double> temp(base);
		for(int ip=0; ip<base->Npol(); ip++) temp.u(ip) = Mmod[ip](i,j); 
		return temp;
	};
	
	int Rows() const {return Mmod[0].rows();};
	int Cols() const {return Mmod[0].cols();};
	int Npol() const {return base->Npol();};
	int Ndim() const {return base->Ndim();};
	B* Base() const {return base;};
	
	T Value_at_xi(VectorXd const xi) const { 
		VectorXd Psi = base->Comp_Psi(xi);
		T temp = Mmod[0];
		for(int ip=1; ip<base->Npol(); ip++) temp += Mmod[ip]*Psi(ip);
		return temp;
	};
	T operator()(VectorXd const &xi) const {
		VectorXd Psi = base->Comp_Psi(xi);
		T temp = Mmod[0];
		for(int ip=1; ip<base->Npol(); ip++) temp += Mmod[ip]*Psi(ip);
		return temp;
	};
	
	VStoch<B,double> operator*(const VStoch<B, double> &V) const{
		VStoch<B,double> temp = V;
		for(int i=0; i<base->Npol(); i++){ temp.u.col(i) *= 0;
			for(list<op_pc>::iterator l=base->Prod[i].begin(); l!=base->Prod[i].end();l++)
				temp.u.col(i) += Mmod[(*l).i]*(V.u.col((*l).j)*(*l).c);
			}
		return temp;
	};

	VStoch<B,cmplx> operator*(const VStoch<B, cmplx> &V) const{
		VStoch<B,cmplx> temp = V;
		for(int i=0; i<base->Npol(); i++){ temp.u.col(i) *= 0;
			for(list<op_pc>::iterator l=base->Prod[i].begin(); l!=base->Prod[i].end();l++)
				temp.u.col(i) += Mmod[(*l).i]*(V.u.col((*l).j)*(*l).c);
			}
		return temp;
	};
	
	
};

