#ifndef PCB_H_
#include "pcb++.h"
#endif

#ifndef COL_H_
#define COL_H_
#endif

using namespace Eigen;
using namespace std;

//! \brief Compare two polynomial multi-indices
bool Comp_set(VectorXi U, VectorXi V);

//! \brief Templated interpolation points class

//! The PTC class defines nodes of the cubarure rules. A node has
//! a coordinate X(), a weight W(). It carries two values. One is Value(),
//! with template type, which is the quantity we eventually want to project,
//! and fval(), of type double, is the quantity used to drive the adaptation
//! process (i.e. excesses are computed based on fval, so the interpolation rule is
//! adapted to the integration of fval).
//! \note the class Type must have a basic constructor.
template <class Type> class PTC {
	template <class T> friend class COL;
private:
	VectorXd x;							//!< coordinate of the point
	VectorXi t;							//!< tensorization owning the point
	VectorXi ip;						//!< multi-index of the point
	double fval;						//!< point's value to be used for adaptation
	Type value;							//!< function value at this point
	int set;							//!< flag for function set
public:
	//! \brief basic constructor for a point in ndim space
	PTC(int const ndim){
		x = VectorXd::Zero(ndim);
		t = VectorXi::Zero(ndim);
		ip= VectorXi::Zero(ndim);
		fval = 0;
		set = 0;
	};
	//! \brief destructor
	~PTC(){};
	PTC &operator=(const PTC &rhs){
		this->x = rhs.x;
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
	//! \brief Return point value
	Type Value(){return value;};
	//! \brief Return point fval used for adaptation
	Type Fval(){return fval;};
	//! \brief Set point value
	void SetValue(Type V){value = V;};
	//! \brief Set point fval
	void SetFVal(double f){fval = f;};
};

//! \brief interpolation tensorization class
class TENS {
	template <class T> friend class COL;
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

class INTERPOL {
	template <class T> friend class COL;
private:
	int nl;						//!< maximal number of level
	VectorXi Npt;				//!< number of points per levels
	VectorXi Npn;				//!< number of new points at each level
	VectorXd Xp;				//!< full set of quadrature points
	MatrixXi Xind;				//!< indexes of points at each level
	MatrixXi Appl;				//!< local index of points at each level
	MatrixXd Wght;				//!< points'weight at each level
	MatrixXd Wghto;				//!< weight's differences at each level
public:
	//! \brief constructors for definition of 1d quadratures
	//! \note Only Féjer so far...
	INTERPOL(){ nl = 9; Npt = VectorXi(nl); Npn = VectorXi(nl); prep_fejer();};
	~INTERPOL(){};
	double fint(double x, int i, int l);
	int NPlev(int l){ return Npt(l);};
	double XPlev(int i, int l){ return Xp(Xind(i,l));};
	double WPlev(int i, int l){ return Wght(Xind(i,l));};
private :
	void prep_fejer();
};

//! \brief Templated interpolation class
template <typename Type> class COL : INTERPOL {
private:
	int ndim;						//!< dimension of the integration domain
	bool finalize;					//!< boolean for finalized interpolation
public:
	int NPlev(int l){ return Npt(l);};
	double XPlev(int i, int l){ return Xp(Xind(i,l));};
	double WPlev(int i, int l){ return Wght(i,l);};
	VectorXd XPlev(int l){ 
		VectorXd XP(Npt(l)); 
		for(int i=0; i<Npt(l); i++) XP(i) = Xp(Xind(i,l));
		return XP;
	};
	VectorXd WPlev(int l){ 
		VectorXd WP(Npt(l)); 
		for(int i=0; i<Npt(l); i++) WP(i) = Wght(i,l);
		return WP;
	};
	list<TENS> old;  				//!< list of old tensorizations
	list<TENS> act;  				//!< list of active tensorizations
	list<TENS> tst;  				//!< list of test tensorizations
	list<PTC<Type> > p_act;   		//!< list of active points
	list<PTC<Type> > p_tst;   		//!< list of test points
	list<VectorXi> poly;			//!< list of multi-indices of exactely integrated polynomials
	//! \brief basic constructor for a n-dimensional interpolation
	COL(int const n){ ndim = n; finalize =0;};
	//! \brief Initialize the interpolation level through Smolyak scheme, up to level lev
	void Initialize(int lev);
	//! \brief Enrich the interpolation rule inserting the new tensorization with largest excess
	double Adapt();
	//! \brief Compute the function value at the unset interpolation points
	int Set_Fval(double (*ftocall)(VectorXd const &Xin, Type &sol, void *), void *);
	//! \brief Reset the function value at the interpolation points
	int Set_Fval_New(double (*ftocall)(VectorXd const &Xin, Type &sol, void *), void *);
	//! \brief Finalize the interpolation rule
	void Finalize();
	void PlotPtc(FILE *out);
	//! \brief Output  to file the interpolation tensorizations
	void PlotTens(FILE *out);
private:
	//! \brief Add to the list the new points of the tensorization
	void SetPoints(TENS *t);
	//! \brief Make active the tensorization
	void Add_To_Active(TENS &tens);
	//! \brief Returns integral of fval;
	void Set_Old();
	double Compute_Integral();
	//! \brief Compute the unset excesses
	void Set_Excesses();
	//! \brief Construct test tensorizations from provided tensorization
	void Do_Test(TENS &tens);
	//! \brief Construct test tensorizations from provided tensorization
	void Do_Test_Max(TENS &tens, int const lev_mx);
};

template <typename Type>
void COL<Type>::Do_Test(TENS &tens){
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
void COL<Type>::Do_Test_Max(TENS &tens, int const lev_mx){
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
void COL<Type>::Set_Excesses(){									// Compute the excesses for adaptivity
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
					int ind = Appl((*p).ip(id),(*it).lmd(id)-1);
//					w *= Wght(ind,(*it).lmd(id)-1);
				}
//				excess += ((*p).fval*w);
			}
		}
		(*it).excess = excess;									//	affect tensorization excess
	}															// Next tensorization
};


template <typename Type>
int COL<Type>::Set_Fval(double (*ftocall)(VectorXd const &Xin, Type &sol, void *), void *user){
//Set the function value at the interpolation nodes, only if not already computed
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
//	Set_Excesses();
	return neval;
};

template <typename Type>
int COL<Type>::Set_Fval_New(double (*ftocall)(VectorXd const &Xin, Type &sol, void *), void *user){
//Set the function value at the interpolation nodes: recompute value in any case.
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
void COL<Type>::SetPoints(TENS *t){	//Create points owned by tensorization t to the list of points ptc.
	typename list<PTC<Type> >::iterator p;
	VectorXi lmd = (*t).lmd; 								//copy the tensorization
	int nstrt = p_tst.size(); 								//initial number of points in the list
	PTC<Type> pn = PTC<Type>(ndim);							//create a generic interpolation point
	pn.t = lmd;												//affect the new tensorization for the created points
 	for(int i=0; i< Npn( lmd(0)-1); i++){				//construct the New points of the tensorization
		pn.x(0)  = Xp( Xind(2*i,lmd(0)-1) );
		pn.ip(0) = Xind(2*i,lmd(0)-1);
		p_tst.push_back(pn);
	}
	for(int id=1; id<ndim; id++){ 							// by tensorizations along the directions
		p=p_tst.begin(); advance(p,nstrt);
		while( p!= p_tst.end()){
			(*p).x(id)  = Xp( Xind(0,lmd(id) -1 ) );
			(*p).ip(id) = Xind(0,lmd(id) -1 );
			p++;
		}
		int n = p_tst.size(); p = p_tst.begin(); advance(p,nstrt); int c=0;
		while( c < n-nstrt){
			for(int i=1; i< Npn( lmd(id) -1 ); i++){
				PTC<Type> pn = *p;
				pn.x(id)  =  Xp( Xind(2*i,lmd(id) -1) );
				pn.ip(id) =  Xind(2*i,lmd(id) -1);
				p_tst.push_back(pn);
			}
			p++; c++;
		}
	}
};

template <typename Type>
double COL<Type>::Adapt(){
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
void COL<Type>::Initialize(int nlev){
	cout << "\t*** Initialize a interpolation rule up to level " << nlev << endl;
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
void COL<Type>::Add_To_Active(TENS &tens){					// Push the tensorization in argument from the test set.
	list<TENS>::iterator it, jt;							// to the active set
	typename list<PTC<Type> >::iterator pts;
	for(it=tst.begin(); it!=tst.end(); it++){				// For all element of the set of test tensorization
		if(tens==(*it)){									//	If it is the tensorization to be pushed
			act.push_back(*it);								// 		push it to active set
			TENS selct = (*it);								//
//			Update_Act_Weights(&selct);    			    	// 		update the weights of the active set of nodes
//			if ( fabs(Check_Act_Weights() )-1. > 1.e-10){   		// sanity check !
//				printf("Sum of weight %12.6e deviates from 1 ---> STOP \n",Check_Act_Weights());exit(0);}
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
void COL<Type>::Set_Old(){
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
void COL<Type>::Finalize(){
	cout << "\t*** Finalize the interpolation rule\n";
	list<TENS>::iterator it, jt;
	typename list<PTC<Type> >::iterator pts;
	int ii = tst.size();
	for(int i=0; i<ii; i++){							//For all the test tensorizations
		it = tst.begin(); TENS selct = (*it);			//	save this tensorization for later use
		act.push_back(*it);								// 	push this tensorization to active set
//		Update_Act_Weights(&selct);    			    	//	update the weights of active set of nodes
		for(pts=p_tst.begin(); pts!=p_tst.end(); pts++){//	For all points in the set of test points
			if( (*pts).t == selct.lmd ){				//		if owned by this tensorization
				p_act.push_back((*pts));				//			push it to the set of active points
				p_tst.erase(pts); pts--;				//			remove from the set of test points
			}
		}
		tst.erase(it);									//	remove this tensorization from the set of test ones
	}													// Next tensorization
	while(old.size() !=0){								// Move all tensorizations in the old set to the active set
		it = old.begin(); act.push_back(*it); old.remove(*it);
	}
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
};

template <typename Type>
void COL<Type>::PlotPtc(FILE *out){
	typename list<PTC<Type> >::iterator it;
	for(it=p_act.begin(); it != p_act.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%e ",(*it).x(id)); fprintf(out,"\n");
	}
	fprintf(out,"\n\n");
	for(it=p_tst.begin(); it != p_tst.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%e ",(*it).x(id)); fprintf(out,"\n");
	}
};

template <typename Type>
void COL<Type>::PlotTens(FILE *out){
	list<TENS>::iterator it;
	for(it=old.begin(); it != old.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%d ",(*it).lmd(id)); fprintf(out,"\n");
	}
	fprintf(out,"\n\n");
	for(it=act.begin(); it != act.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%d ",(*it).lmd(id)); fprintf(out,"\n");
	}
	fprintf(out,"\n\n");
	for(it=tst.begin(); it != tst.end(); it++){
		for(int id=0; id<ndim; id++) fprintf(out,"%d ",(*it).lmd(id)); fprintf(out,"\n");
	}
};

