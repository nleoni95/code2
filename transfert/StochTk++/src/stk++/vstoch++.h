#ifndef Stoch_H_
#include "stoch++.h"
#endif

#ifndef Vstoch_H_
#define Vstoch_H_
#endif

template <class B, class T> class VStoch;


//! \brief Recast type stochastic double vector to stochastic complex vector !!
template <class B>
VStoch<B,cmplx> _cmplx(const VStoch<B,double> &DS){
	VStoch<B,cmplx> CS(DS.Base(),DS.Size());
	MatrixXd u = DS.u;
	CS.u = u.cast<cmplx>();
	return CS;
};
//! \brief Recast type stochastic complex vector to itself !!
template <class B>
VStoch<B,cmplx> _cmplx(const VStoch<B,cmplx> &DS){
	VStoch<B,cmplx> CS(DS);
	return CS;
};
//! \brief Extract real part of a stochastic complex vector
template <class B>	VStoch<B,double> _real(const VStoch<B,cmplx> &V){
	VStoch<B,double> temp(V.Base(),V.Size()); temp.u = V.u.real(); return temp;
};
//! \brief Extract imaginary part of a stochastic complex vector
template <class B>	VStoch<B,double> _imag(const VStoch<B,cmplx> &V){
	VStoch<B,double> temp(V.Base(),V.Size()); temp.u = V.u.imag(); return temp;
};
//! \brief Extract real part of a stochastic real --> is itself
template <class B>	VStoch<B,double> _real(const VStoch<B,double> &V){
	VStoch<B,double> temp(V); return temp;
};
//! \brief Extract imaginary part of a stochastic real --> is zero
template <class B>	VStoch<B,double> _imag(const VStoch<B,double> &V){
	VStoch<B,double> temp(V); temp.u *= 0; return temp;
};
//! \brief Set conjugate of a stochastic complex vector
template <class B>	VStoch<B,cmplx> _conj(const VStoch<B,cmplx> &V){
	VStoch<B,cmplx> temp(V.Base(),V.Size()); temp.u = V.u.conjugate(); return temp;
};
//! \brief Set conjugate of a stochastic double vector
template <class B>	VStoch<B,cmplx> _conj(const VStoch<B,double> &V){
	VStoch<B,cmplx> temp(V.Base(),V.Size()); 
	temp.u = MatrixXc::Zero(V.u.rows(),V.u.cols());
	temp.u.real() = V.u; 
	return temp;
};

template <class B, class T> class VStoch {

private:
	B *base;
	int nc;

public:
	Matrix<T,Dynamic,Dynamic> u;
	//! \brief basic constructor
	VStoch(){ base = NULL; nc=0;};
	//! \brief constructor form a basis ans size
	VStoch(B *b, int const n){ base = b; nc= n; u = Matrix<T,Dynamic,Dynamic>::Zero(nc,base->npol);};
	//! \brief copy constructor
	VStoch(const VStoch<B,T> &V){ base = V.base; nc = V.nc; u = V.u;};
	//! \brief destructor
	~VStoch(){};

	//! \brief overloaded = operator
	void operator=(const VStoch<B,T> &V){ base = V.base; nc = V.nc; u = V.u;};
	//! \brief overloaded operator + for 2 stochastic vectors (with same type)
	VStoch<B,T> operator+(const VStoch<B,T> &V) const { VStoch<B,T> temp(*this); temp.u+= V.u; return temp;};
	//! \brief overloaded operator - for 2 stochastic vectors (with same type)
	VStoch<B,T> operator-(const VStoch<B,T> &V) const { VStoch<B,T> temp(*this); temp.u-= V.u; return temp;};
	//! \brief overloaded operator += for 2 stochastic vectors (with same type)
	void operator+=(const VStoch<B,T> &V) { u += V.u;};
	//! \brief overloaded operator -= for 2 stochastic vectors (with same type)
	void operator-=(const VStoch<B,T> &V) { u -= V.u;};
	//! \brief overloaded operator + for addition of a deterministic vector (of same type)
	VStoch<B,T> operator+(const Matrix<T,Dynamic,Dynamic> &V) const {
		VStoch<B,T> temp(*this); temp.u.col(0) += V; return temp;
	};
	//! \brief overloaded operator - for substraction of a deterministic vector (of same type)
	VStoch<B,T> operator-(const Matrix<T,Dynamic,Dynamic> &V) const {
			VStoch<B,T> temp(*this); temp.u.col(0) -= V; return temp;
	};
	//! \brief overloaded operator += for addition of a deterministic vector (of same type)
	void operator+=(const Matrix<T,Dynamic,Dynamic> &V) { u.col(0) += V;};
	//! \brief overloaded operator -= for sybstrction of a deterministic vector (of same type)
	void operator-=(const Matrix<T,Dynamic,Dynamic> &V) { u.col(0) -= V;};

	//! \brief overloaded * with a stochastic scalar (of same type)
	VStoch<B,T> operator*(const Stoch<B,T> &V) const{
		VStoch<B,T> temp(*this);
		for(int i=0; i<base->Npol(); i++){ temp.u.col(i) *= 0;
			for(list<op_pc>::iterator l=base->Prod[i].begin(); l!=base->Prod[i].end();l++)
				temp.u.col(i) += u.col((*l).i)*(V.u[(*l).j]*(*l).c);
			}
		return temp;
	};
	//! \brief overloaded / by a stochastic scalar (of same type)
	VStoch<B,T> operator/(const Stoch<B,T> &V) const{
		VStoch<B,T> temp(*this); Stoch<B,T> VI = V.Inv();
		for(int i=0; i<base->Npol(); i++){ temp.u.col(i) *= 0;
			for(list<op_pc>::iterator l=base->Prod[i].begin(); l!=base->Prod[i].end();l++)
				temp.u.col(i) += u.col((*l).i)*(VI.u[(*l).j]*(*l).c);
		}
		return temp;
	};
	//! \brief overloaded *= with a stochastic scalar (of same type)
	void operator*=(const Stoch<B,T> &V) { u = ((*this)*V).u;};
	//! \brief overloaded /= with a stochastic scalar (of same type)
	void operator/=(const Stoch<B,T> &V) { u = ((*this)/V).u;};

	//! \brief overloaded multiplication with a determinstic scalar (of same type)
	VStoch<B,T> operator*(const T V) const { VStoch<B,T> temp(*this); temp.u *= V; return temp;};
	//! \brief overloaded division with a determinstic scalar (of same type)
	VStoch<B,T> operator/(const T V) const { VStoch<B,T> temp(*this); temp.u /= V; return temp;};
	//! \brief overloaded *= with a deterministic scalar (of same type)	
	void operator*=(const T &V) { u *= V;};
	//! \brief overloaded /= with a deterministic scalar (of same type)	
	void operator/=(const T &V) { u /= V;};

	//! \brief set stochastic vector's mode	
	void set_mode(int i, const Matrix<T,Dynamic,1> &V){ u.col(i) = V;};
	//! \brief add to stochastic vector's mode	
	void add_mode(int i, const Matrix<T,Dynamic,1> &V){ u.col(i)+= V;};
	//! \brief substract to stochastic vector's mode	
	void sub_mode(int i, const Matrix<T,Dynamic,1> &V){ u.col(i)-= V;};
	//! \brief set stochastic vector's component	
	void set_entry(int i, Stoch<B,T> &V){ u.row(i) = V.u;};
	//! \brief add to stochastic vector's component	
	void add_entry(int i, Stoch<B,T> &V){ u.row(i)+= V.u;};
	//! \brief substract to stochastic vector's component		
	void sub_entry(int i, Stoch<B,T> &V){ u.row(i)-= V.u;};

	//! \brief set stochastic vector's component mode	
	void set_entry_mode(int i, int m, T V){ u(i,m) = V;};
	//! \brief add to stochastic vector's component	mode
	void add_entry_mode(int i, int m, T V){ u(i,m) += V;};
	//! \brief substract to stochastic vector's component mode		
	void sub_entry_mode(int i, int m, T V){ u(i,m)-= V;};

	//! \brief retrieve stochastic vector's mode		
	Matrix<T,Dynamic,Dynamic> operator[](const int i) const { return u.col(i);};
	//! \brief retrieve stochastic vector's component		
	Stoch<B,T> operator()(int const i) const { 
		Stoch<B,T> temp(base);
		for(int ip=0; ip<base->npol; ip++) temp.set_mode(ip,u(i,ip)); 
		return temp;
	};
	//! \brief retrieve stochastic vector's basis		
	B*  Base() const { return base;};
	//! \brief retrieve stochastic vector's number of components			
	int Size() const { return nc;};
	//! \brief retrieve stochastic vector's number of components			
	int Rows() const { return nc;};
	//! \brief retrieve stochastic vector's number of modes			
	int Npol() const {return base->Npol();};
	//! \brief retrieve stochastic vector's number of dimensions				
	int Ndim() const {return base->Ndim();};
	//! \brief retrieve stochastic vector's number of dimensions				
	int Nord() const {return base->Nord();};
	
	//! \brief retrieve mean of stochsastic vector
	Matrix<T,Dynamic,1> Mean() const { return u.col(0);};
	//! \brief retrieve standard deviation of stochsastic vector
	Matrix<T,Dynamic,1> Std() const { Matrix<T,Dynamic,1> temp = Matrix<T,Dynamic,1>::Zero(nc);
		for(int i=0; i<nc; i++) temp(i) = (*this)(i).Std();
		return temp;
	};
	//! \brief retrieve variance of stochsastic vector
	Matrix<T,Dynamic,1> Var() const { Matrix<T,Dynamic,1> temp = Matrix<T,Dynamic,1>::Zero(nc);
		for(int i=0; i<nc; i++) temp(i) = (*this)(i).Var();
		return temp;
	};
	//! \brief retrieve component norm of stochsastic vector
	Matrix<T,Dynamic,1> CNorm() const { Matrix<T,Dynamic,1> temp = Matrix<T,Dynamic,1>::Zero(nc);
		for(int i=0; i<nc; i++) temp(i) = (*this)(i).Norm();
		return temp;
	};
	//! \brief retrieve squared component norm of stochsastic vector
	Matrix<T,Dynamic,1> CNorm2() const { Matrix<T,Dynamic,1> temp = Matrix<T,Dynamic,1>::Zero(nc);
		for(int i=0; i<nc; i++) temp(i) = (*this)(i).Norm2();
		return temp;
	};
	
	//! \brief retrieve a 1st order sensitivity indice of stochsastic vector
	Matrix<T,Dynamic,1> S1(const int j) const { 
		Matrix<T,Dynamic,1> temp = Matrix<T,Dynamic,1>::Zero(nc);
		for(int i=0; i<nc; i++) temp(i) = (*this)(i).S1(j);
		return temp;
	};
	//! \brief retrieve a total sensitivity indice of stochsastic vector
	Matrix<T,Dynamic,1> ST(const int j) const { 
		Matrix<T,Dynamic,1> temp = Matrix<T,Dynamic,1>::Zero(nc);
		for(int i=0; i<nc; i++) temp(i) = (*this)(i).ST(j);
		return temp;
	};
	//! \brief retrieve 1st order sensitivity indices of stochsastic vector
	Matrix<T,Dynamic,Dynamic> S1() const { 
		Matrix<T,Dynamic,Dynamic> temp = Matrix<T,Dynamic,Dynamic>::Zero(nc,base->Ndim());
		for(int i=0; i<nc; i++) temp.row(i) = (*this)(i).S1();
		return temp;
	};
	//! \brief retrieve total sensitivity indices of stochsastic vector
	Matrix<T,Dynamic,Dynamic> ST() const { 
		Matrix<T,Dynamic,Dynamic> temp = Matrix<T,Dynamic,Dynamic>::Zero(nc,base->Ndim());
		for(int i=0; i<nc; i++) temp.row(i) = (*this)(i).ST();
		return temp;
	};
	
	//! \brief Average the Stoch in all directions but the id-th
	VStoch<B,T> Marginalize_others(int const id) const {
		VStoch<B,T> temp(base,nc);
		for(int ip=0; ip<base->Npol(); ip++) if(base->alp.row(ip).sum()- base->alp(ip,id) ==0) temp.u.col(ip) = u.col(ip);
		return temp;
	};
	//! \brief Average the Stoch in the id-th diretion
	VStoch<B,T> Marginalize(int const id) const {
		VStoch<B,T> temp(base,nc);
		for(int ip=0; ip<base->Npol(); ip++) if( base->alp(ip,id) ==0 ) temp.u.col(ip) = u.col(ip);
		return temp;
	};
	
	
	//Evaluation
	//! \brief Value of Stochactic vector at point xi
	Matrix<T,Dynamic,1> Value_at_xi(VectorXd const xi) const { return u*base->Comp_Psi(xi);};
	//! \brief Value of Stoch from values of the polynomials
	Matrix<T,Dynamic,1> Value_at_psi(VectorXd const psi) const {return u*psi;};
	Matrix<T,Dynamic,1> operator()(VectorXd const &xi) const { return u*base->Comp_Psi(xi);};
	
	VStoch<B,T> Derive(int const id) const {
		VStoch<B,T> temp(base,nc);
		for(list<op_pc>::iterator l=base->Deriv[id].begin(); l!=base->Deriv[id].end();l++)
			temp.add_mode((*l).i,u.col((*l).j)*(*l).c);
		return temp;
	};
	
	void Grad(Stoch<B,T> const &U){
		base = U.base; nc = base->ndim; u = Matrix<T,Dynamic,Dynamic>::Zero(nc,base->npol);
		VectorXd du(base->npol);
		for(int id=0; id<base->ndim; id++){
			for(list<op_pc>::iterator l=base->Deriv[id].begin(); l!=base->Deriv[id].end();l++) u(id,(*l).i) += U.u[(*l).j]*(*l).c;
		}
	};
	
	
	//! \brief retrieve real part of stochastic Vector
	VStoch<B,double> real() const { return _real(*this);};
	//! \brief retrieve imaginary part of stochastic Vector
	VStoch<B,double> imag() const { return _imag(*this);};
	//! \brief retrieve imaginary part of stochastic Vector
	VStoch<B,cmplx> conjug() const { return _conjug(*this);};
	//! \brief Recast to stochastic complex vector
	VStoch<B,cmplx> complex() const { return _cmplx(*this);};	
	
	double Norm() const { double s=0; for(int ip=0; ip<base->npol; ip++) s += u.col(ip).squaredNorm(); return sqrt(s);};
	double Norm2()const { double s=0; for(int ip=0; ip<base->npol; ip++) s += u.col(ip).squaredNorm(); return s;};
	
	//! \brief overloaded dot product 
	Stoch<B,double> dot(const VStoch<B,double> &V) const{ 
		Stoch<B,double> DOT(V.Base()); 
		for(int i=0; i<nc; i++) DOT += (*this)(i)*V(i);
		return DOT;
	};
	//! \brief overloaded dot product 
	Stoch<B,cmplx> dot(const VStoch<B,cmplx> &V) const{ 
		Stoch<B,cmplx> DOT(V.Base()); 
		for(int i=0; i<nc; i++) DOT += (*this)(i).conjug()*V(i);
		return DOT;
	};
	//! \brief overloaded dot product 
	Stoch<B,cmplx> dot(const VectorXc &V) const{ 
		Stoch<B,cmplx> DOT(base); 
		for(int ip=0; ip<base->Npol(); ip++){
			VectorXc Xc = (*this)[ip];
			cmplx dip = Xc.adjoint()*V;
			DOT.set_mode(ip,dip);
		}
		return DOT;
	};

	//! \brief overloaded dot product 
	Stoch<B,double> dotreal(const VectorXd &V) const{ 
		Stoch<B,double> DOT(base); 
		for(int ip=0; ip<base->Npol(); ip++){
			VectorXd Xd = (*this)[ip];
			double dip = Xd.dot(V);
			DOT.set_mode(ip,dip);
		}
		return DOT;
	};
	
};

