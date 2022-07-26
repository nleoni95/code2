#ifndef PCB_H_
#include "pcb++.h"
#endif

#ifndef CUB_H_
#define CUB_H_
#endif

#include <list>
#include <set>
using namespace Eigen;
using namespace std;

//! \brief Compare two polynomial multi-indices
bool Comp_set(VectorXi U, VectorXi V);

//! \brief Templated cubature points class

//! The PTC class defines nodes of the cubarure rules. A node has
//! a coordinate X(), a weight W(). It carries two values. One is Value(),
//! with template type, which is the quantity we eventually want to project,
//! and fval(), of type double, is the quantity used to drive the adaptation
//! process (i.e. excesses are computed based on fval, so the cubature rule is
//! adapted to the integration of fval).
//! \note the class Type must have a basic constructor.
template <class Type> class PTC {
	template <class T> friend class CUB;
private:
	VectorXd x;							//!< coordinate of the point
	VectorXi t;							//!< tensorization owning the point
	VectorXi ip;						//!< multi-index of the point
	double w;							//!< weight of the point
	double fval;						//!< point's value to be used for adaptation
	Type value;							//!< function value at this point
	int set;							//!< flag for function set
public:
	//! \brief basic constructor for a point in ndim space
	PTC(int const ndim){
		x = VectorXd::Zero(ndim);
		t = VectorXi::Zero(ndim);
		ip= VectorXi::Zero(ndim);
		w = 1.;
		fval = 0;
		set = 0;
	};
	//! \brief destructor
	~PTC(){};
	PTC &operator=(const PTC &rhs){
		this->x = rhs.x;
		this->w = rhs.w;
		this->t = rhs.t;
		this->ip= rhs.ip;
		this->fval = rhs.fval;
		this->value = rhs.value;
		this->set = rhs.set;
		return *this;
	};
	int operator==(const PTC &rhs) const{
		for(int id=0; id<this->x.rows(); id++)
			if( this->x(id) != rhs.x(id)) return 0;
		return 1;
	};
	int operator<(const PTC &rhs) const{
		for(int id=0; id<=0; id++) if(this->x(id) > rhs.x(id)) return 1;
		return 0;
	};
	//! \brief Returns points coordinates
	VectorXd X(){return x;};
	//! \brief Returns points weight
	double W(){return w;};
	//! \brief Return point value
	Type Value(){return value;};
	//! \brief Return point fval used for adaptation
	Type Fval(){return fval;};
	//! \brief Set point value
	void SetValue(Type V){value = V;};
	//! \brief Set point fval
	void SetFVal(double f){fval = f;};
	void SetX(VectorXd X){ x = X;};
	void SetW(double W){ w = W;};
};

//! \brief Cubature tensorization class
class TENS {
	template <class T> friend class CUB;
  	friend ostream &operator<<(ostream & o, const TENS &a){
	  	o << "(" << a.lmd.sum() << ") : " << a.lmd.transpose() <<  endl; return o;};
	friend void Act_to_Old();
	friend void PlotTENS(list<TENS> &old, list<TENS> &act, list<TENS> &tst, FILE *out);
	void PlotTENS(list<TENS> &old, FILE *out);
private:
	VectorXi lmd;						//!< multi-index of tensorization
	double excess;						//!< excess of tensorization
	int smolyak;						//!< Smolyak coefficient of the tensorization
public:
	//! \brief default constructor
	TENS(){};
	//! \brief basic constructor for a ndim tensorization
	TENS(int const ndim){ lmd  = VectorXi::Ones(ndim); smolyak = 0; excess=0;};
	//! \brief constructor from a multi-index
	TENS(const VectorXi LMD){  lmd  = LMD; smolyak = 0; excess=0;};
	//! \brief destructor
	~TENS(){};
//Operator overloading
  	TENS &operator=(const TENS &rhs){
		this->lmd = rhs.lmd;
		this->excess= rhs.excess;
		this->smolyak = rhs.smolyak;
		return *this;
	};
  	int operator==(const TENS &T2) const{ if(Dist(T2) != 0) return 0; return 1;};
  	int operator<(const TENS &rhs) const {
		for(int id=0; id<this->lmd.rows(); id++){
	    	if(this->lmd(id) >= rhs.lmd(id)) return 0;
		}
		return 1;
	};
	//! \brief Distance with another tensorization
	int Dist(const TENS &rhs) const{ return (this->lmd - rhs.lmd).array().abs().sum();};
	//! \brief Display tensorizations
	void ListTENS(list<TENS> &liste) const {
	  list<TENS>::iterator it; for(it=liste.begin(); it != liste.end(); it++) cout << *it;

	};
};

class RULEQUAD {
	template <class T> friend class CUB;
private:
	int nl;						//!< total number of level
	VectorXi Npt;				//!< number of points per levels
	VectorXi Npn;				//!< number of new points at each level
	VectorXd Xp;				//!< full set of quadrature points
	MatrixXi Xind;				//!< indexes of points at each level
	MatrixXi Appl;				//!< local index of points at each level
	MatrixXd Wght;				//!< points'weight at each level
	MatrixXd Wghto;				//!< weight's differences at each level
public:
	//! \brief constructors for definition of 1d quadratures
	//! \note Only Féjer and Clenshaw-Curtis so far... the selection switch is "manual" for now
	RULEQUAD(){ nl = 9; Npt = VectorXi(nl); Npn = VectorXi(nl); prep_clenshaw();};
	// RULEQUAD(){ nl = 9; Npt = VectorXi(nl); Npn = VectorXi(nl); prep_fejer();}; // nl is the total number of levels; the highest level is nl-1
	
	~RULEQUAD(){};
private :
	void prep_fejer();
	void prep_clenshaw();
};

//! \brief Templated cubature class
template <typename Type> class CUB {
private:
	int ndim;						//!< dimension of the integration domain
	bool finalize;					//!< boolean for finalized cubature
	RULEQUAD rule;					//!< 1-D quadrature rules
public:
	list<TENS> old;  				//!< list of old tensorizations
	list<TENS> act;  				//!< list of active tensorizations
	list<TENS> tst;  				//!< list of test tensorizations
	list<PTC<Type> > p_act;   		//!< list of active points
	list<PTC<Type> > p_tst;   		//!< list of test points
	list<VectorXi> poly;			//!< list of multi-indices of exactely integrated polynomials
	//! \brief basic constructor for a n-dimensional cubature
	CUB(int const n){ ndim = n; finalize =0;};
	//! \brief Initialize the cubature level through Smolyak scheme, up to level lev
	void Initialize(int lev);
	//! \brief Enrich the cubature rule inserting the new tensorization with largest excess
	double Adapt();
	//! \brief Compute the function value at the unset cubature points
	int Set_Fval(double (*ftocall)(VectorXd const &Xin, Type &sol, void *), void *);
	//! \brief Reset the function value at the cubature points
	int Set_Fval_New(double (*ftocall)(VectorXd const &Xin, Type &sol, void *), void *);
	//! \brief Finalize the cubature rule
	void Finalize();
	//! \brief Check weights consistency
	double Check_Weights();
	//! \brief Determine the polynomial set achieved by the cubature
	void Set_Poly();
	//! \brief Perform the NISP projection for the provided basis
	void Do_NISP(PCB &base, Type []);
	//! \brief Perform the pseudo-spectral NISP projection
	void Do_Pseudo_NISP(PCB &base, Type []);
	//! \brief Retrieve the pseudo-spectral NISP projection operator
	MatrixXd Get_Pseudo_NISP();


	//! \brief Output  to file the cubature points
	void PlotPtc(FILE *out);
	//! \brief Output  to file the cubature tensorizations
	void PlotTens(FILE *out);
	//!brief steal the cubature points
	MatrixXd GetCubPts();
	//!brief steal the cubature weights
	VectorXd GetCubWgts();
	
private:
	//! \brief Add to the list the new points of the tensorization
	void SetPoints(TENS *t);
	//! \brief Make active the tensorization
	void Add_To_Active(TENS &tens);
	//! \brief Returns integral of fval;
	void Set_Old();
	double Compute_Integral();
	//! \brief Update the weights for the tensorization
	void Update_Act_Weights(TENS *t);
	//! \brief check consistency of the weights
	double Check_Act_Weights();
	//! \brief Compute the unset excesses
	void Set_Excesses();
	//! \brief Construct test tensorizations from provided tensorization
	void Do_Test(TENS &tens);
	//! \brief Construct test tensorizations from provided tensorization
	void Do_Test_Max(TENS &tens, int const lev_mx);
};

template <typename Type>
void CUB<Type>::Do_Test(TENS &tens){
	int ndim = tens.lmd.rows();
	for(int id=0; id<ndim; id++){
		TENS test = tens; test.lmd(id) ++;
		if(test.lmd(id) <= 9){
			int flag=1;
			for(list<TENS>::iterator it = tst.begin(); it!= tst.end(); it++) if( (*it == test) ) {flag = 0; break;};
			if(flag == 1){ 					//Is new, check if is admissible (has predecessors)
				for(int jd=0; jd<ndim; jd++){
					int flag2 = 1;
					if(test.lmd(jd) !=1){
						if( (jd!=id) ){
							flag2=0;
							for(list<TENS>::iterator it = act.begin(); it!= act.end(); it++)
								if( ( test.Dist(*it) == 1) && (*it).lmd(jd) == test.lmd(jd)-1 ) {flag2 = 1; break;};
						}
					}
					if(flag2==0){flag=0; break;}
				}
				if(flag==1) tst.push_back(test);						//Yes it is
			}
		}
	}
};

template <typename Type>
void CUB<Type>::Do_Test_Max(TENS &tens, int const lev_mx){
//Attempt to create successors to this tensorization, maintaining admissibility, while not exceeding a maximum level
	int ndim = tens.lmd.rows();
	if(tens.lmd.sum() >= lev_mx) return;
	for(int id=0; id<ndim; id++){
		TENS test = tens;
		test.lmd(id) ++;
		int flag=1; //check if the tensorization isn't already in the nex list
		for(list<TENS>:: iterator it = tst.begin(); it!= tst.end(); it++) if( (*it == test) ) {flag = 0; break;};
		if(flag == 1){ //check that this tensorization is admissible (has predecessors in the active set)
			for(int jd=0; jd<ndim; jd++){
				int flag2 = 1;
				if(test.lmd(jd) !=0){
					if( (jd!=id) ){
						flag2=0;
						for(list<TENS>::iterator it = act.begin(); it!= act.end(); it++)
							if( ( test.Dist(*it) == 1) && (*it).lmd(jd) == test.lmd(jd)-1 ) {flag2 = 1; break;};
					}
				}
				if(flag2==0){flag=0; break;}
			}
			if(flag==1) tst.push_back(test);		// Tensorization is admissible, add to the nex list
		}
	}
};

template <typename Type>
void CUB<Type>::Do_Pseudo_NISP(PCB &base, Type Coeff[]){ //Do Pseudo spectral NISP projection
	if(finalize==0) Finalize();
	int npol = base.npol;
	if(int(poly.size())!=npol){
		cout << "WARNING: the basis may not be adapted to the cubature rule.\n";
	}
	typename list<PTC<Type> >::iterator p;
	p = p_act.begin(); for(int ip=0; ip<npol; ip++) Coeff[ip].set_to_zero();		//Initialize to zero the PC coefficients
	VectorXd psi;
	for(p=p_act.begin(); p!=p_act.end(); p++){ 									// For each active point
		psi = base.Comp_Psi((*p).x);											//		compute polynomials
		for(list<TENS>::iterator it=act.begin(); it!=act.end(); it++){  		//		For each cubature tensorization
			VectorXi Dist = (*it).lmd - (*p).t;
			if(Dist.minCoeff() >= 0){ 											// 			point is owned by the tensorization
				VectorXi Odr = VectorXi(ndim);
				for(int id=0; id<ndim; id++) Odr(id) =	(rule.Npt((*it).lmd(id)-1)-1)/2;// 	set the max polynomial order
				double w = 1;													//			compute the weight for this tensorization
				for(int id=0; id<ndim; id++){
					int ip = (*p).ip(id); 										//			Index of the points
					int il = rule.Appl(ip,(*it).lmd(id)-1); 					//			local point index at this level
					w *= rule.Wghto(il,(*it).lmd(id)-1);
				}
				w*=(*it).smolyak;												//          account for Smolyak coefficient
				for(int ip=0; ip<base.npol; ip++)								// 			add contribution
					if( (Odr-base.Alp(ip)).minCoeff() >=0  ) Coeff[ip] += (*p).value*(w*psi[ip]);
			}
		}
	}
	return;
};


template <typename Type>
MatrixXd CUB<Type>::Get_Pseudo_NISP(){ //Retrieve the Pseudo spectral NISP projection operator
	if(finalize==0) Finalize();
	PCB base = PCB(poly,'U');
	int npol = base.Npol();
	MatrixXd Proj = MatrixXd::Zero(poly.size(),p_act.size());
	VectorXd psi;
	typename list<PTC<Type> >::iterator p;
	p = p_act.begin();															//Initialize to zero the PC coefficients
	int iq=0;
	for(p=p_act.begin(); p!=p_act.end(); p++){ 									// For each active point
		psi = base.Comp_Psi((*p).x);											//		compute polynomials
		for(list<TENS>::iterator it=act.begin(); it!=act.end(); it++){  		//		For each cubature tensorization
			VectorXi Dist = (*it).lmd - (*p).t;
			if(Dist.minCoeff() >= 0){ 											// 			point is owned by the tensorization
				VectorXi Odr = VectorXi(ndim);
				for(int id=0; id<ndim; id++) Odr(id) =	(rule.Npt((*it).lmd(id)-1)-1)/2; // 	set the max polynomial order
				double w = 1;													//			compute the weight for this tensorization
				for(int id=0; id<ndim; id++){
					int ip = (*p).ip(id); 										//			Index of the points
					int il = rule.Appl(ip,(*it).lmd(id)-1); 					//			local point index at this level
					w *= rule.Wghto(il,(*it).lmd(id)-1);
				}
				w*=(*it).smolyak;												//          account for Smolyak coefficient
				for(int ip=0; ip<base.npol; ip++)								// 			add contribution
					if( (Odr-base.Alp(ip)).minCoeff() >=0  ) Proj(ip,iq) += w*psi[ip]; //Coeff[ip] += (*p).value*(w*psi[ip]);
			}
		}
		iq++;
	}
	return Proj;
};

template <typename Type>
void CUB<Type>::Do_NISP(PCB &base, Type Coeff[]){  //Perform the NISP Projection
	if(finalize==0) Finalize();
	int npol = base.npol;
	if(int(poly.size())!=npol){
		cout << "WARNING: the basis may not be adapted to the cubature rule.\n";
	}
	typename list<PTC<Type> >::iterator p;
	p = p_act.begin(); for(int ip=0; ip<npol; ip++) Coeff[ip].set_to_zero();	//Initialize to zero the PC coefficients
	VectorXd psi;
	for(p=p_act.begin(); p!=p_act.end(); p++){										//For each active points
		psi = base.Comp_Psi((*p).x);											//	evaluate the polynomials
		Type contr = (*p).value*(*p).w;											//	point value time weight
		for(int ip=0; ip<npol; ip++) Coeff[ip] += contr*psi(ip);				//	add point contribution to PC coefficient
	}
	return;
};

template <typename Type>
MatrixXd CUB<Type>::GetCubPts(){
	if(finalize==0) cout << "Warning the cubature rule has not been finalized\n";
	MatrixXd Pts(ndim,p_act.size()+p_tst.size());
	typename list<PTC<Type> >::iterator p;
	int c=0;
	for(p=p_act.begin(); p!=p_act.end(); p++){										//For each active points
		Pts.col(c) = (*p).x;
		c++;
	}
	for(p=p_tst.begin(); p!=p_tst.end(); p++){										//For each test points
		Pts.col(c) = (*p).x;
		c++;
	}
	return Pts;
};


template <typename Type>
VectorXd CUB<Type>::GetCubWgts(){
	if(finalize==0) cout << "Warning the cubature rule has not been finalized\n";
	VectorXd Wgt(p_act.size()+p_tst.size());
	typename list<PTC<Type> >::iterator p;
	int c=0;
	for(p=p_act.begin(); p!=p_act.end(); p++){										//For each active points
		Wgt(c) = (*p).w;
		c++;
	}
	for(p=p_tst.begin(); p!=p_tst.end(); p++){										//For each test points
		Wgt(c) = (*p).w;
		c++;
	}
	return Wgt;
};


template <typename Type>
void CUB<Type>::Set_Poly(){
	if(finalize==0) Finalize();
	bool(*fn_pt)(VectorXi,VectorXi) = Comp_set;
	set<VectorXi,bool(*)(VectorXi,VectorXi)> Pol(fn_pt);
	set<VectorXi>::iterator tpol;
	list<TENS>::iterator it;
	list<VectorXi>::iterator p; 	

	for(it=act.begin(); it!=act.end(); it++){ 					//	For each tensorization of the cubature rule
		VectorXi ord = (*it).lmd;
		for(int id=0; id<ndim; id++)
			ord(id) = (rule.Npt(ord(id)-1))/2;					// 		max polynomial order for this tensorization
		list<VectorXi> Plo;										// 		list of polynomials whose squares are exactely integrated
		VectorXi pol = VectorXi::Zero(ndim);					//		proceed by tensorization in N-dimension
		for(int io=0; io<=ord(0); io++){ pol(0) = io; Plo.push_back(pol);}
		for(int id=1; id<ndim; id++){
			int npol = Plo.size();
			for(int io=1; io<=ord(id); io++){
				p = Plo.begin();
				for(int el=0; el<npol; el++){
					VectorXi tt = (*p); tt(id) = io; Plo.push_back(tt); p++;
				}
			}
		}
		for(p=Plo.begin(); p!=Plo.end(); p++) Pol.insert((*p)); //		add these polynomials to the current SET Pol
	}
	poly.clear();
	for(tpol=Pol.begin(); tpol!=Pol.end(); tpol++) poly.push_back(*tpol);
	poly.sort(Comp_VectorXi);		//Sort
};

template <typename Type>
void CUB<Type>::Set_Excesses(){									// Compute the excesses for adaptivity
	list<TENS>::iterator it; typename list<PTC<Type> >::iterator p;
	for(it=tst.begin(); it!=tst.end(); it++){					// For all test tensorizations
		double excess=0;
		for(p=p_tst.begin(); p!= p_tst.end(); p++) 					//	contribution of test points owned by the tensorization
			if( (*p).t == (*it).lmd ) excess += (*p).fval*(*p).w;
		for(p=p_act.begin(); p!=p_act.end(); p++){					//	contribution of active points owned by dominated tensorizations
			VectorXi mi = (*p).t - (*it).lmd;
			if( mi.maxCoeff() <= 0){							//		this point is owned by a dominated tensorization
				double w=1.;
				for(int id=0; id<mi.rows(); id++){
					int ind = rule.Appl((*p).ip(id),(*it).lmd(id)-1);
					w *= rule.Wght(ind,(*it).lmd(id)-1);
				}
				excess += ((*p).fval*w);
			}
		}
		(*it).excess = excess;									//	affect tensorization excess
	}															// Next tensorization
};

template <typename Type>
double CUB<Type>::Compute_Integral(){
	typename list<PTC<Type> >::iterator pts;
	double f_int = 0.;
	for(pts=p_act.begin(); pts!=p_act.end(); pts++) f_int += ((*pts).fval*(*pts).w);
	for(list<TENS>::iterator it=tst.begin(); it!=tst.end(); it++) f_int += (*it).excess;
	return f_int;
};

template <typename Type>
int CUB<Type>::Set_Fval(double (*ftocall)(VectorXd const &Xin, Type &sol, void *), void *user){
//Set the function value at the cubature nodes, only if not already computed
	typename list<PTC<Type> >::iterator p;
	int neval = 0;
	for(p=p_act.begin(); p!=p_act.end(); p++){
		if((*p).set == 0){
		  Type sol; (*p).fval = ftocall( (*p).x, sol, user );
		  (*p).set = 1; (*p).value = sol; neval++;
		}
	}
	for(p=p_tst.begin(); p!=p_tst.end(); p++){
		if((*p).set == 0){
		  Type sol; (*p).fval = ftocall( (*p).x, sol, user );
		  (*p).set = 1; (*p).value = sol; neval++;
		}
	}
	Set_Excesses();
	return neval;
};

template <typename Type>
int CUB<Type>::Set_Fval_New(double (*ftocall)(VectorXd const &Xin, Type &sol, void *), void *user){
//Set the function value at the cubature nodes: recompute value in any case.
	typename list<PTC<Type> >::iterator p;
	int neval = 0;
	for(p=p_act.begin(); p!=p_act.end(); p++){
	  Type sol;
	  (*p).fval = ftocall( (*p).x, sol, user );
	  (*p).set = 1;
	  (*p).value = sol;
	  neval++;
	}
	for(p=p_tst.begin(); p!=p_tst.end(); p++){
	  Type sol;
	  (*p).fval = ftocall( (*p).x, sol, user );
	  (*p).set = 1;
	  (*p).value = sol;
	  neval++;
	}
	return neval;
};

template <typename Type>
void CUB<Type>::SetPoints(TENS *t){	//Create points owned by tensorization t to the list of points ptc.
	typename list<PTC<Type> >::iterator p;
	VectorXi lmd = (*t).lmd; 								//copy the tensorization
	int nstrt = p_tst.size(); 								//initial number of points in the list
	PTC<Type> pn = PTC<Type>(ndim);							//create a generic cubature point
	pn.t = lmd;												//affect the new tensorization for the created points
 	for(int i=0; i< rule.Npn( lmd(0)-1); i++){				//construct the New points of the tensorization
		pn.x(0)  = rule.Xp( rule.Xind(2*i,lmd(0)-1) );
		pn.ip(0) = rule.Xind(2*i,lmd(0)-1);
		pn.w     = rule.Wght(    2*i,lmd(0)-1);
		p_tst.push_back(pn);
	}
	for(int id=1; id<ndim; id++){ 							// by tensorizations along the directions
		p=p_tst.begin(); advance(p,nstrt);
		while( p!= p_tst.end()){
			(*p).x(id)  = rule.Xp( rule.Xind(0,lmd(id) -1 ) );
			(*p).ip(id) = rule.Xind(0,lmd(id) -1 );
			(*p).w     *= rule.Wght(    0,lmd(id) -1);
			p++;
		}
		int n = p_tst.size(); p = p_tst.begin(); advance(p,nstrt); int c=0;
		while( c < n-nstrt){
			for(int i=1; i< rule.Npn( lmd(id) -1 ); i++){
				PTC<Type> pn = *p;
				pn.x(id)  =  rule.Xp( rule.Xind(2*i,lmd(id) -1) );
				pn.ip(id) =  rule.Xind(2*i,lmd(id) -1);
				pn.w     *=  (rule.Wght(     2*i,lmd(id) -1)/rule.Wght(    0,lmd(id) -1));
				p_tst.push_back(pn);
			}
			p++; c++;
		}
	}
};

template <typename Type>
void CUB<Type>::Update_Act_Weights(TENS *t){ //Update the weights of the points after the tensorization in argument becomes active
	typename list<PTC<Type> >::iterator p;
	VectorXi lmd = (*t).lmd;
	for(p=p_act.begin(); p!=p_act.end(); p++){   					// For all the active points
		VectorXi mi = (*p).t - (*t).lmd;						// 		Distance of the tensorization owning the point with t
		if( mi.maxCoeff() <= 0){ 								//		If point's tensorization is dominated
			double w=1.;										//			compute the weight due to tensorization t
			for(int id=0; id<ndim; id++){						//				(... too lazy to comment; basically,
				int ind = rule.Appl((*p).ip(id),(*t).lmd(id)-1);//						tensorization of 1-d weights)
				w *= rule.Wght(ind,(*t).lmd(id)-1);
			}
			(*p).w += w;   //update the weight.					//			update the point's weight
		}
	}															// Next point
};

template <typename Type>
double CUB<Type>::Check_Act_Weights(){	//check if the sum of weights for the points in active set is indeed =1
	double sum =0.;
	typename list<PTC<Type> >::iterator p;
	for(p=p_act.begin(); p!=p_act.end(); p++) sum += (*p).w;
	return sum;
};

template <typename Type>
double CUB<Type>::Check_Weights(){	//check if the sum of weights for the points in active set is indeed =1
	double sum =0.; typename list<PTC<Type> >::iterator p;
	for(p=p_act.begin(); p!=p_act.end(); p++) sum += (*p).w; return sum;
};

template <typename Type>
double CUB<Type>::Adapt(){
	int ntest = tst.size();
	if(ntest==0) return 0.;
	VectorXd Ex(ntest);
	list<TENS>::iterator it;
	int i=0;
	for(it=tst.begin(); it!=tst.end(); it++){
		Ex(i) = fabs((*it).excess);
		i++;
	}
	Ex.maxCoeff(&i);
	if(Ex(i) < 1.e-14){
		cout << "No other tensorization found \n\n";
		return Ex(i);
	}
	it = tst.begin(); advance(it,i);
	TENS selct = (*it);
	cout << "Adapt tensorization : " << selct.lmd.transpose() << " having an excess : " << Ex(i) << endl;
	Add_To_Active(selct);
	return Ex(i);
};

template <typename Type>
void CUB<Type>::Initialize(int nlev){
	cout << "\t*** Initialize a cubature rule up to level " << nlev << endl;
	list<TENS>::iterator it;
	list<TENS>::iterator jt;
	typename list<PTC<Type> >::iterator pts;
	TENS t = TENS(ndim);
	t.lmd = VectorXi::Ones(ndim);				// start with tensorization (1...1)
	tst.push_back(t);
	SetPoints(&t);
	Add_To_Active(t);

	for(int lev=0; lev <nlev-1; lev++){			// For each subsequent levels
		int nit=tst.size();
		for(int j=0; j<nit;j++){ 				//	For all the current test tensorizations
			it = tst.begin();
			TENS selct = (*it);					// 		save this tensorization for future use
			Add_To_Active(selct);
		}										//	Next test tensorization
	}											// Next level
	Set_Old();							// Do the set of old tensorizations
};

template <typename Type>
void CUB<Type>::Add_To_Active(TENS &tens){					// Push the tensorization in argument from the test set.
	list<TENS>::iterator it, jt;							// to the active set
	typename list<PTC<Type> >::iterator pts;
	for(it=tst.begin(); it!=tst.end(); it++){				// For all element of the set of test tensorization
		if(tens==(*it)){									//	If it is the tensorization to be pushed
			act.push_back(*it);								// 		push it to active set
			TENS selct = (*it);								//
			Update_Act_Weights(&selct);    			    	// 		update the weights of the active set of nodes
			if ( fabs(Check_Act_Weights() )-1. > 1.e-10){   		// sanity check !
				printf("Sum of weight %12.6e deviates from 1 ---> STOP \n",Check_Act_Weights());exit(0);}
			for(pts=p_tst.begin(); pts!=p_tst.end(); pts++){	// 		For each points in the set of test points
				if( (*pts).t == selct.lmd ){				//			If it is owned by the test tensorization
					p_act.push_back((*pts)); p_tst.erase(pts);	//				push it to the set of active points
					pts--;
				}
			}
			tst.erase(it);									//		remove it from the set of test tensorizations
			int nt_o = tst.size();							//		record updated size of set of test tensorization
			Do_Test(selct); //selct.Do_Test(act,tst); 						//		update the set of test tensorizations, preserving adminissibility
			for(int i=nt_o; i<int(tst.size()); i++){		//		For all new test tensorizations
				jt = tst.begin(); advance(jt,i);
				selct = (*jt); SetPoints(&selct);		//			create the new points owned by this new test tensorization
			}
			break;											// 		Job done !
		}
	}														// Next test tensorization
	Set_Old();												// Update set of old tensorizations
};

template <typename Type>
void CUB<Type>::Set_Old(){
//! Push strictly dominated tensorizations from active to old set.
	list<TENS>::iterator tac, tes;
	if(int(act.size()) == 0) return;					// None to check
	tac = act.begin(); int ndim = (*tac).lmd.rows();
	for(tac=act.begin(); tac != act.end(); tac++){ 		//For each tensorizations, if it is dominated, then we make it "old"
		int flag = 1; 									//	assumed it is strictly dominated
		for(int id=0; id<ndim; id++){					//	seek for successors in all directions
			int flag2 = 0;
			for(tes= act.begin(); tes != act.end(); tes++){
				int d = (*tes).Dist(*tac);
				if( d==1 && (*tes).lmd(id) == (*tac).lmd(id)+1 ){flag2 = 1; break;} //found a successor in that direction
			}
			if(flag2 == 0){ flag = 0; break;} 			//found no successor, thus isn't strictly dominated: can't be old.
		}
		if( flag==1 ){									// ok, make it old then.
			old.push_back(*tac); act.erase(tac); tac --;
		}
	}
};

template <typename Type>
void CUB<Type>::Finalize(){
	cout << "\t*** Finalize the cubature rule\n";
	list<TENS>::iterator it, jt;
	typename list<PTC<Type> >::iterator pts;
	cout << "Start :\n";
	/*
	for(int i=0; i<ii; i++){							//For all the test tensorizations
		it = tst.begin(); TENS selct = (*it);			//	save this tensorization for later use
		act.push_back(*it);								// 	push this tensorization to active set
//		cout << "Do weights:\n";
		Update_Act_Weights(&selct);    			    	//	update the weights of active set of nodes
//		cout << "Do points:\n";
		for(pts=p_tst.begin(); pts!=p_tst.end(); pts++){//	For all points in the set of test points
			if( (*pts).t == selct.lmd ){				//		if owned by this tensorization
//				cout << "Push \n";
				p_act.push_back((*pts));				//			push it to the set of active points
//				cout << "Erase\n";
				p_tst.erase(pts); 
//				cout << "Decrease\n";
				if(pts!=p_tst.begin()) pts--;				//			remove from the set of test points
			}
//			cout <<"Next\n";
			if(p_tst.size()==0) break;
		}
//		cout << "erase\n";
		tst.erase(it);									//	remove this tensorization from the set of test ones
	}
	*/
	for(it=tst.begin(); it!=tst.end(); it++){
		TENS selct = (*it);								//	save this tensorization for later use
		act.push_back(*it);								// 	push this tensorization to active set
		Update_Act_Weights(&selct);    			    	//	update the weights of active set of nodes
		for(pts=p_tst.begin(); pts!=p_tst.end(); pts++){//	For all points in the set of test points
			if( (*pts).t == selct.lmd ){				//		if owned by this tensorization
				p_act.push_back((*pts));				//			push it to the set of active points
			}
		}		
	}													// Next tensorization
	tst.clear();
	p_tst.clear();
	if ( fabs(Check_Act_Weights() )-1. > 1.e-10){   	// sanity check on weights
		printf("Sum of weight %12.6e deviates from 1 ---> STOP \n",Check_Act_Weights()); exit(1);
	};
	/*
	while(old.size() !=0){								// Move all tensorizations in the old set to the active set
		it = old.begin(); act.push_back(*it); old.remove(*it);
	}
	*/
	for(it=old.begin(); it!=old.end(); it++){
		act.push_back(*it);
	}
	old.clear();
	cout << "Set Smolyak coefficients\n";
//	Now the final step: compute the Smolyak's coefficients of the tensorizations (they are all in the active set)
	for(it=act.begin(); it!=act.end(); it++){			// For all tensorizations
		for(jt=act.begin(); jt!=act.end(); jt++){		//		For all others
			VectorXi Dist = (*it).lmd - (*jt).lmd;		//			Distance between the two tensorizations
			if(Dist.minCoeff()>=-1 && Dist.maxCoeff()<=0){//		If jt dominates it but by no more than 1 in every direction
				if(Dist.sum()%2 ==0){ 					//				increment the Smolyak coefficients accordingly
					(*it).smolyak ++;					//						(I am too lazy to detail!!!)
				}else{
					(*it).smolyak --;
				}
			}
		}												//		Next other
	}													// Next tensorization
	finalize = 1;
	cout << "Rule is finalized\n";
};

template <typename Type>
void CUB<Type>::PlotPtc(FILE *out){
	typename list<PTC<Type> >::iterator it;
	for(it=p_act.begin(); it != p_act.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%e ",(*it).x(id)); fprintf(out,"\n");
	}
	// fprintf(out,"\n\n");
	fprintf(out,"%%%%%%\n"); // changed for Matlab format compatibility
	for(it=p_tst.begin(); it != p_tst.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%e ",(*it).x(id)); fprintf(out,"\n");
	}
};

template <typename Type>
void CUB<Type>::PlotTens(FILE *out){
	list<TENS>::iterator it;
	for(it=old.begin(); it != old.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%d ",(*it).lmd(id)); fprintf(out,"\n");
	}
	// fprintf(out,"\n\n");
	fprintf(out,"%%%%%%\n"); // changed for Matlab format compatibility
	for(it=act.begin(); it != act.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%d ",(*it).lmd(id)); fprintf(out,"\n");
	}
	// fprintf(out,"\n\n");
	fprintf(out,"%%%%%%\n"); // changed for Matlab format compatibility
	for(it=tst.begin(); it != tst.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%d ",(*it).lmd(id)); fprintf(out,"\n");
	}
};

