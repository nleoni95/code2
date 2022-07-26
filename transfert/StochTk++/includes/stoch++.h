#ifndef Stoch_H_
#define Stoch_H_
#endif

using namespace std;
using namespace Eigen;

typedef complex<double> cmplx;
typedef Matrix<cmplx,Dynamic,1> VectorXc;
typedef Matrix<cmplx,Dynamic,Dynamic> MatrixXc;

template <class B, class T> class Stoch;
template <class B, class T> class VStoch;

//! \brief Recast type stochastic <B,double> to stochastic complex<double>
template <class B>
Stoch<B,cmplx> _cmplx(const Stoch<B,double> &DS){
	VectorXd d = DS.getPC();
	VectorXc c = d.cast<cmplx>();
	Stoch<B,cmplx> CS(DS.Base());
	for(int ip=0; ip<CS.Npol(); ip++) CS.set_mode(ip,c(ip));
	return CS;
};
//! \brief Recast type stochastic complex<B,double> to itself !!
template <class B>
Stoch<B,cmplx> _cmplx(const Stoch<B,cmplx> &DS){
	Stoch<B,cmplx> CS(DS);
	return CS;
};
//! \brief Extract real part of a stochastic complex
template <class B>	Stoch<B,double> _real(const Stoch<B,cmplx> &V){
	Stoch<B,double> temp(V.Base()); temp.u = V.u.real(); return temp;
};
//! \brief Extract imaginary part of a stochastic complex
template <class B>	Stoch<B,double> _imag(const Stoch<B,cmplx> &V){
	Stoch<B,double> temp(V.Base()); temp.u = V.u.imag(); return temp;
};
//! \brief Extract real part of a stochastic real --> is itself
template <class B>	Stoch<B,double> _real(const Stoch<B,double> &V){
	Stoch<B,double> temp(V); return temp;
};
//! \brief Extract imaginary part of a stochastic real --> is zero
template <class B>	Stoch<B,double> _imag(const Stoch<B,double> &V){
	Stoch<B,double> temp(V); temp.u *= 0; return temp;
};
//! \brief Set conjugate of a stochastic complex
template <class B>	Stoch<B,cmplx> _conj(const Stoch<B,cmplx> &V){
	Stoch<B,cmplx> temp(V.Base()); temp.u = V.u.conjugate(); return temp;
};
//! \brief Set conjugate of a stochastic double
template <class B>	Stoch<B,cmplx> _conj(const Stoch<B,double> &V){
	Stoch<B,cmplx> temp(V.Base()); 
	temp.u = VectorXc::Zero(V.u.rows());
	temp.u.real() = V.u; 
	return temp;
};
//! \brief Compute variance of a stochastic double
template <class B>
double _var(const Stoch<B,double> &V){
	double v = V.u.squaredNorm() - V.u(0)*V.u(0);
	return v;
};
//! \brief Compute variances of real and imaginary part of a stochastic complex
template <class B>
cmplx _var(const Stoch<B,cmplx> &V){
	cmplx v (V.u.real().squaredNorm() - V.u.real()(0)*V.u.real()(0),V.u.imag().squaredNorm() - V.u.imag()(0)*V.u.imag()(0));
	return v;
};
//! \brief Compute standard deviation of a stochastic double
template <class B>
double _std(const Stoch<B,double> &V){
	double v = fabs(V.u.squaredNorm() - V.u(0)*V.u(0));
	return sqrt(v);
};
//! \brief Compute standard deviations of real and imaginary part of a stochastic complex
template <class B>
cmplx _std(const Stoch<B,cmplx> &V){
	double vr = sqrt(fabs(V.u.real().squaredNorm() - V.u.real()(0)*V.u.real()(0)));
	double vi = sqrt(fabs(V.u.imag().squaredNorm() - V.u.imag()(0)*V.u.imag()(0)));
	return cmplx(vr,vi);
};
//! \brief Compute 1-st order sensitivity index in direction i, of a stochastic double
template <class B> double  _s1(const int i, const Stoch<B,double> &V){
	if(i<0 || i>=V.Base()->Ndim()) return 0; 
	double var = V.Var();
	if(var==0) return 0;
	double s=0;
	for(int ip=1; ip<V.Base()->Npol(); ip++)
		if(V.Base()->Alp(ip).sum() == V.Base()->Alp(ip,i)) s+=(V.u(ip)*V.u(ip));
	return s / var;
};

//! \brief Compute 1-st order sensitivity indices in direction i, of the real and imaginary parts
//! \brief of a stochastic complex
template <class B> cmplx  _s1(const int i, const Stoch<B,cmplx> &V){
	if(i<0 || i>=V.Base()->Ndim()) return cmplx(0,0); cmplx s(0,0);
	for(int ip=1; ip<V.Base()->Npol(); ip++)
		if(V.Base()->Alp(ip).sum() == V.Base()->Alp(ip,i))
			s+=cmplx(V.u.real()(ip)*V.u.real()(ip), V.u.imag()(ip)*V.u.imag()(ip));
		cmplx var = V.Var();
		double sr = real(s)/real(var); double	si = imag(s)/imag(var);
		if(real(var)==0) sr=0; 
		if(imag(var)==0) si=0;
	return cmplx(sr,si);
};
//! \brief Compute all 1-st order sensitivity indices of a stochastic double
template <class B> VectorXd  _s1(const Stoch<B,double> &V){
	VectorXd s1 = VectorXd::Zero(V.Base()->Ndim());
	double var = V.Var();
	if(var==0) return s1;
	for(int ip=1; ip<V.Base()->Npol(); ip++){
		int it = V.Base()->Alp(ip).sum();
		for(int i=0; i<V.Base()->Ndim(); i++)
			if( it == V.Base()->Alp(ip,i)) s1(i)+=(V.u(ip)*V.u(ip));
	}
	return (s1 / var);
};
//! \brief Compute all 1-st order sensitivity indices of the real and imaginary parts
//! \brief of a stochastic complex
template <class B> VectorXc  _s1(const Stoch<B,cmplx> &V){
	VectorXc s1 = VectorXc::Zero(V.Base()->Ndim());
	cmplx var = V.Var();
	if(abs(var)==0) return s1;
	for(int ip=1; ip<V.Base()->Npol(); ip++){
		int it = V.Base()->Alp(ip).sum();
		for(int i=0; i<V.Base()->Ndim(); i++) if( it == V.Base()->Alp(ip,i))
			s1(i)+= cmplx(V.u.real()(ip)*V.u.real()(ip) , V.u.imag()(ip)*V.u.imag()(ip));
	}
	if(real(var) != 0 ) s1.real()/=real(var);
	if(imag(var) != 0 ) s1.imag()/=imag(var);
	return s1;
};

//! \brief Compute the total sensitivity index in direction i of a stochastic double
template <class B> double  _st(const int i, const Stoch<B,double> &V){
	if(i<0 || i>=V.Base()->Ndim()) return 0; double s=0;
	double var = V.Var();
	if(var==0) return 0;
	for(int ip=1; ip<V.Base()->Npol(); ip++)
		if(V.Base()->Alp(ip,i)!=0) s+=(V.u(ip)*V.u(ip));
	return s / V.Var();
};

//! \brief Compute the total sensitivity indices in direction i of
//! \brief real and imaginary parts of a stochastic complex
template <class B> cmplx  _st(const int i, const Stoch<B,cmplx> &V){
	if(i<0 || i>=V.Base()->Ndim()) return cmplx(0,0); cmplx s(0,0);
	cmplx var = V.Var();
	if(abs(var)==0) return s;
	for(int ip=1; ip<V.Base()->Npol(); ip++)
		if(V.Base()->Alp(ip,i) != 0) s+=cmplx(V.u.real()(ip)*V.u.real()(ip), V.u.imag()(ip)*V.u.imag()(ip));
	double sr, si;
	if(real(var) != 0 ) sr = real(s)/real(var);
	if(imag(var) != 0 ) si = imag(s)/imag(var);
	return cmplx(sr,si);
};

//! \brief Compute all the total sensitivity indices of a stochastic double
template <class B> VectorXd _st(const Stoch<B,double> &V){
	VectorXd s = VectorXd::Zero(V.Base()->Ndim());
	double var = V.Var();
	if(var==0) return s;
	for(int ip=1; ip<V.Base()->Npol(); ip++)
		for(int i=0; i<V.Base()->Ndim(); i++) if(V.Base()->Alp(ip,i)!=0) s(i)+=(V.u(ip)*V.u(ip));
	return s / V.Var();
};

//! \brief Compute all the total sensitivity indices of the
//! \brief real and imaginary parts of a stochastic complex
template <class B> VectorXc _st(const Stoch<B,cmplx> &V){
	VectorXc s = VectorXc::Zero(V.Base()->Ndim());
	cmplx var = V.Var();
	if(abs(var)==0) return s;
	for(int ip=1; ip<V.Base()->Npol(); ip++)
		for(int i=0; i<V.Base()->Ndim(); i++) if(V.Base()->Alp(ip,i)!=0)
			s(i)+=cmplx(V.u(ip).real()*V.u(ip).real(), V.u(ip).imag()*V.u(ip).imag());
	if(real(var)!=0) s.real()/=real(var); 
	if(imag(var)!=0) s.imag()/=imag(var);
	return s;
};


//! \brief Class for stochastic scalars (double and complex<double>)
template <class B, class T>
class Stoch {

	template<class BV, class TV> friend class VStoch;

private:
	//!	\brief Pointer to the basis of the stochastic object
	B *base;

public:
	//! \brief expansion coefficients
	Matrix<T,Dynamic,1> u;
	//! \brief type of basis
	typedef B Btype;
	//! \brief numeric type of stochastic
	typedef T Ttype;
	//! \brief basic constructor
	Stoch(){ base = NULL;}
	//! \brief constructor from basis
	Stoch(B *bas){ base = bas; u = Matrix<T,Dynamic,1>::Zero(base->Npol());};
	//! \brief copy constructor
	Stoch(const Stoch &A){ base = A.base; u = A.u;};
	//! \brief destructor
	~Stoch(){};
	//Overloaded basic operators
	//! \brief overloaded = operator for two stochastic objects
	void operator = (const Stoch V){ base = V.base; u= V.u;};
	//! \brief overloaded + operator for two stochastic objects
	Stoch operator+(const Stoch V) const { Stoch<B,T> temp(*this); temp.u+= V.u; return temp;};
	//! \brief overloaded - operator for two stochastic objects
	Stoch operator-(const Stoch V) const { Stoch<B,T> temp(*this); temp.u-= V.u; return temp;};
	//! \brief overloaded + operator for a stochastic object and deterministic one
	Stoch operator+(const T V) const { Stoch<B,T> temp(*this); temp.u(0) += V;   return temp;};
	//! \brief overloaded - operator for a stochastic object and deterministic one
	Stoch operator-(const T V) const { Stoch<B,T> temp(*this); temp.u(0) -= V;   return temp;};
	//! \brief overloaded += operator for two stochastic objects
	void operator+=(const Stoch V){ u += V.u;};
	//! \brief overloaded -= operator for two stochastic objects
	void operator-=(const Stoch V){ u-= V.u;};
	//! \brief overloaded += operator for a stochastic object and deterministic one
	void operator+=(const T V){ u(0) += V;};
	//! \brief overloaded -= operator for a stochastic object and deterministic one
	void operator-=(const T V){ u(0) -= V;  };
	//! \brief overloaded * operator for two stochastic objects
	Stoch operator*(const Stoch V) const {
		Stoch<B,T> temp(base);
		for(int i=0; i<base->Npol(); i++){
			for(list<op_pc>::iterator l=base->Prod[i].begin(); l!=base->Prod[i].end();l++)
				temp.u(i) += u[(*l).i]*V.u[(*l).j]*(*l).c;
		}
		return temp;
	};
	//! \brief overloaded * operator for  a stochastic object and deterministic one
	Stoch operator*(const T V) const { Stoch<B,T> temp(base); temp.u = u*V; return temp;};
	//! \brief overloaded *= operator for two stochastic objects
	void operator*=(const Stoch V){ Stoch<B,T> temp = (*this)*V; u=temp.u;};
	//! \brief overloaded *= operator for  a stochastic object and deterministic one
	void operator*=(const T V){ u *= V;};
	//! \brief overloaded / operator for two stochastic objects
	Stoch operator/(const Stoch V) const { Stoch<B,T> temp = (*this)*V.Inv(); return temp;};
	//! \brief overloaded / operator for  a stochastic object and deterministic one
	Stoch operator/(const T V) const { Stoch temp(*this); temp.u /= V; return temp;};
	//! \brief overloaded /= operator for two stochastic objects
	void operator/=(const Stoch V){ Stoch<B,T> temp = (*this)*V.Inv(); u=temp.u;};
	//! \brief overloaded /= operator for  a stochastic object and deterministic one
	void operator/=(const T V){u/=V;};

	//Analyze:
	//! \brief Returns the mean of the stochastic object
	T Mean() const {return u(0);};
	//! \brief Returns the standard deviation of the Stoch
	T Std() const {return _std(*this);};
	//! \brief Returns the variance of the Stoch
	T Var() const {return _var(*this);};
	//! \brief Returns the L2-norm of the Stoch
	double Norm() const { return u.norm();};
	//! \brief Returns the L2-norm squared of the Stoch
	double Norm2() const { return u.squaredNorm();};
	//! \brief Returns the 1st order sensitivity index associated to direction i
	T S1(const int i) const{return _s1(i,*this);};
	//! \brief Returns all the 1st order sensitivity indices
	Matrix<T,Dynamic,1> S1() const { return _s1(*this);};
	//! \brief Returns the total order sensitivity index associated to direction i
	T ST(const int i) const { return _st(i, *this);};
	//! \brief Returns all the total order sensitivity indices
	Matrix<T,Dynamic,1> ST() const { return _st(*this);};
	//! \brief Average the Stoch in all directions but the id-th
	Stoch<B,T> Marginalize_others(int const id) const {
		Stoch<B,T> temp = (*this);
		for(int ip=0; ip<base->Npol(); ip++) if(base->alp.row(ip).sum()- base->alp(ip,id) !=0) temp.u(ip) =0;
		return temp;
	};
	//! \brief Average the Stoch in the id-th diretion
	Stoch<B,T> Marginalize(int const id) const {
		Stoch<B,T> temp = (*this);
		for(int ip=0; ip<base->Npol(); ip++) if( base->alp(ip,id) !=0 ) temp.u(ip) = 0.;
		return temp;
	};
	//! \brief Screen output of the stochastic modes
	void view_modes(string c) const { cout << c << u.transpose() << endl;};
	//! \brief Summary of Stoch: mean and standard deviation
	void view_abs(string c) const { cout << c << " : Mean : " << Mean() << " Std : " << Std() << endl;};

	//Accessors:
	//! \brief Returns the real part
	Stoch<B,double> real() const { return _real(*this);};
	//! \brief Returns the imaginary part
	Stoch<B,double> imag() const { return _imag(*this);};
	//! \brief Returns the conjugate
	Stoch<B,cmplx> conjug() const { return _conj(*this);};
	//! \brief Recast to complex
	Stoch<B,cmplx> complex() const { return _cmplx(*this);};	
	
	//! \brief Returns the i-th stochastic coefficient
	T operator[](int const i) const { return u(i);};
	//! \brief Returns the i-th stochastic coefficient
	T operator()(VectorXd const &xi) const { return base->Comp_Psi(xi).dot(u);;};


	//! \brief returns the full set of expansion coefficients
	Matrix<T,Dynamic,1> getPC() const {return u;};
	//! \brief Returns a pointer to the basis
	B* Base() const {return base;};
	//! \brief Returns the PC basis dimension
	int Npol() const {return base->Npol();};
	//! \brief Returns the dimension of the stochastic space
	int Ndim() const {return base->Ndim();};
	//! \brief Returns the polynomial order of the stochastic space
	int Nord() const {return base->Nord();};

	//Modifyers:
	//! \brief Set value of stochastic mode i
	void set_mode(int i, T c){ u(i) = c;};
	//! \brief Add value to a stochastic mode i
	void add_mode(int i, T c){ u(i) += c;};
	//! \brief Substract value to a stochastic mode
	void sub_mode(int i, T c){ u(i) -= c;};
	//! \brief Initialize a linear Stoch with prescribed mean and standard deviation
	//! carried in specific direction
	void set_MeanStd(const T mean, const T std, int const j){
		if(j<=0 || j>base->Ndim()){ cout << "Invalid Initialization by set_MeanStd" << endl;
		cout << "Dimension of the stochastic space " << base->Ndim() << " called dimension " << j<< endl;exit(1);}
		u = Matrix<T,Dynamic,1>::Zero(base->Npol()); u(0) = mean; u(j) = std;
	};

	void set_URange(const T low, const T high, int const j){
		if(j<=0 || j>base->Ndim()){ cout << "Invalid Initialization by Range" << endl;
		cout << "Dimension of the stochastic space " << base->Ndim() << " called dimension " << j<< endl;exit(1);}
		if(base->Type(j-1)!='U'){ cout << "Invalid Initialization given a range" << endl;
		cout << "while not uniformly distributed in direction " << j << " instead " << base->Type(j-1)<< endl;exit(1);}		
		u = Matrix<T,Dynamic,1>::Zero(base->Npol()); 
		u(0) = (high+low)*.5; u(j) = (high-low)/(2*base->pleg_at_x(1.)(1));
	};

	//Evaluation
	//! \brief Value of Stoch at point xi
	T Value_at_xi(VectorXd const xi) const { return base->Comp_Psi(xi).dot(u);};
	//! \brief Value of Stoch from values of the polynomials
	T Value_at_psi(VectorXd const psi) const {return psi.dot(u);};

	//Operations:
	//! \brief Inverse
	Stoch<B,T> Inv() const;
	//! \brief In place inversion
	void Inv_inplace(){ Stoch<B,T> Temp = (*this); (*this) = Temp.Inv();};
	//! \brief Exponential
	Stoch<B,T> Exp() const;
	//! \brief In place exponentiation
	void Exp_inplace(){ Stoch<B,T> Temp = (*this); (*this) = Temp.Exp();};
	//! \brief Square root (with positive mean)
	Stoch<B,T> Sqroot() const;
	//! \brief In place square root extraction.
	void Sqroot_inplace(){ Stoch<B,T> Temp = (*this); (*this) = Temp.Sqroot();};
	//! \brief Derivative with respect to given direction
	Stoch<B,T> Derive(int const id) const {
		Stoch<B,T> temp(base);
		for(list<op_pc>::iterator l=base->Deriv[id].begin(); l!=base->Deriv[id].end();l++)
			temp.add_mode((*l).i,u[(*l).j]*(*l).c);
		return temp;
	};
	
};

template <class B, class T>
Stoch<B,T> Stoch<B,T>::Inv() const {
	int npol = base->Npol();
	Matrix<T,Dynamic,Dynamic> A = Matrix<T,Dynamic,Dynamic>::Zero(npol,npol);
	Matrix<T,Dynamic,1> R = Matrix<T,Dynamic,1>::Zero(npol); 
	Stoch<B,T> E(base);
	R(0)=1;
	for(int i=0; i< npol; i++){
		for(list<op_pc>::iterator l=base->OProd(i).begin(); l!=base->OProd(i).end(); l++){
			A(i,(*l).i) += u((*l).j)*(*l).c;
		}
	}
	E.u = A.lu().solve(R);
	//	FullPivLU<Matrix<T,Dynamic,Dynamic> > LUA(A);   /* Full LU for debugging */
	//	if(LUA.isInvertible()==0){ cout <<"ILL PROBLEM FOR INVERSION\n"; exit(1);}
	//	E.u = LUA.solve(R);
	return E;
};

template <class B, class T>
Stoch<B,T> Stoch<B,T>::Sqroot() const {
	Stoch<B,T> Y(base);
	Stoch<B,T> temp = (*this);
	Y.set_mode(0,sqrt(  Mean() ) );
	Stoch<B,T> R = temp - Y*Y;
	double R0 = temp.Norm(); if(R0 == 0) R0 = 1;
	int it =0;
	while(R.Norm()/R0 > 1.e-10 && R.Norm()>1.e-10 && it < 500 ){
		Y += ((R/Y)*.25); R = temp - Y*Y; it++;
	}
	if(it == 200){ cout << "Non Converged STOCHASTIC SQROOT (Stoch)\n"; 
	cout << "Iteration " << it << endl; Y.view_abs("Current iterate :");
	R.view_abs("Error"); exit(0);
	}
	return Y;
};

template <class B, class T>
Stoch<B,T> Stoch<B,T>::Exp() const {
	int npol = base->Npol();
	Matrix<T,Dynamic,Dynamic> A = Matrix<T,Dynamic,Dynamic>::Zero(npol,npol);
	for(int i=0; i<npol; i++)
		for(list<op_pc>::iterator l=base->OProd(i).begin(); l!=base->OProd(i).end(); l++) A(i,(*l).i) += u((*l).j)*(*l).c;
		Eigen::SelfAdjointEigenSolver<Matrix<T,Dynamic,Dynamic> > Eig(A);
	Matrix<T,Dynamic,Dynamic> BP = Eig.eigenvectors();
	VectorXd Ev = Eig.eigenvalues();
	Matrix<T,Dynamic,1> Yt = BP.row(0);
	for(int ip=0; ip<npol; ip++) Yt(ip) *= exp(Ev(ip));
	Stoch<B,T> E(base);
	E.u = BP*Yt;
	return E;
};

