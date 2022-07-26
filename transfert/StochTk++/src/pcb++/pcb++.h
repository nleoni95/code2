#ifndef PCB_H_
#define PCB_H_
#endif

#ifndef GLH_H_
#include "gauss.h"
#endif

#include <list>
#include <vector>


#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

//! \brief Comparaison operator between polynomial tensorizations
bool Comp_VectorXi(VectorXi U, VectorXi V);

//! \struct op_pc
//! \brief Define a generic structure with integer indices (up to 3) and a double, to be used
//! to store element of tensors, up to third order ones.
struct op_pc{
	unsigned i;		/*!< first index */
	unsigned j;		/*!< second index */
	unsigned l;		/*!< third index */
	double c;		/*!< entry */
};

//! \class PCB
//! \brief Define a Generic Polynomial Chaos basis and set up operators (product, triple product and derivation) and polynomial evaluation utilities.
class PCB{
	friend class MRA;
	template <class T> friend class CUB;
	template <class B, class T> friend class Stoch;
	template <class B, class T> friend class VStoch;
	template <class B, class T> friend class MStoch;

private:
	int npol;	/*!< Dimension of the PC basis */
	int ndim;   /*!< Number of stochastic dimensions */
	int nord;	/*!< Max partial degree of the basis */
	int nq1d;	/*!< Dimension of 1D quadrature rule */
	char Ctype;	/*!< Type of basis (based on "U"niform, "N"ormal, "M"ixed measures) */
//	char *type;  /*!< Type along each stochastic dimension */
	vector<char> type;
	int type_tens; /*!< Type of tensorization */
	MatrixXi alp;  /*!< Multi-index of polynomials */
//! \brief Gauss-Legendre integration points and weights.
//! \note Legendre basis is defined for the interval [0,1].
	VectorXd xleg;	/*!< Gauss-Legendre points */
	VectorXd wleg;	/*!< Gauss-Legendre weights */
	VectorXd xher;	/*!< Gauss-Hermite points */
	VectorXd wher;	/*!< Gauss-Hermite weights */
	bool sprod;		/*!< Boolean for product definition */
	list<op_pc> *Prod;	/*!< Product opertor definition */
	bool striple; 		/*!< Boolean for triple product definition */
	list<op_pc> *Triple;/*!< Triple roduct opertor definition */
	bool sderive; 		/*!< Boolean for derivation definition */
	list<op_pc> *Deriv;/*!< Derivation opertor definition */

public:
//! \brief Returns the dimension of the basis
	int Npol() const {return npol;};
//! \brief Returns the number of stochastic dimensions
	int Ndim() const {return ndim;};
//! \brief Returns the maximal partial order
	int Nord() const {return nord;};
//! \brief Returns the number of quadrature points in 1D
	int Nq1d() const {return nq1d;};
//! \brief Returns the basis type
//! \note Can be "U"niform, "N"ormal or "M"ixed.
	char Type() const {return Ctype;};
//! \brief Returns the type of measure (U or N) in direction i.
	char Type(const int i) const {return type[i];};
//! \brief Returns the whole set of multi-indices.

//! \brief Returns the type of tensorization
	int Tens() const {return type_tens; };
	MatrixXi Alp() const {return alp;};
//! \brief Returns the multi-index of the i-th polynomial.
	VectorXi Alp(const int i) const {return alp.row(i);};
//! \brief Returns the polynomial degree in the d-th dimension of the i-th polynomial.
	int Alp(const int i, const int d) const {return alp(i,d);};
//! \brief Returns the set of 1-D Gauss points for uniform measure.
	VectorXd Xleg() const {return xleg;};
//! \brief Returns the set of 1-D Gauss weights for uniform measure
	VectorXd Wleg() const {return wleg;};
//! \brief Returns the set of 1-D Gauss points for normal measure
	VectorXd Xher() const {return xher;};
//! \brief Returns the set of 1-D Gauss weights for normal measure
	VectorXd Wher() const {return wher;}
//! \brief Returns the i-th Gauss point for uniform measure
	double Xleg(const int i) const {return xleg(i);};
//! \brief Returns the i-th Gauss weight for uniform measure
	double Wleg(const int i) const {return wleg(i);};
//! \brief Returns the i-th Gauss point for normal measure
	double Xher(const int i) const {return xher(i);};
//! \brief Returns the i-th Gauss weight for Normal measure
	double Wher(const int i) const {return wher(i);};
//! \brief list the product operator for the i-th mode
	list<op_pc> LProd(const int i) const { 
		list<op_pc> cop;
		op_pc ter;
		for(list<op_pc>::iterator l=Prod[i].begin(); l!=Prod[i].end();l++){
			ter.i = (*l).i; ter.j = (*l).j; ter.c = (*l).c;
			cop.push_back(ter);
		}
		return cop;
	};
//! \brief return the product operator for the i-th mode
	list<op_pc>& OProd(const int i) const { return Prod[i];};
//! \brief Returns a reference to the derivation operator along the d-th dimension
	list<op_pc>& ODeriv(const int d) const {return Deriv[d];};

//! \brief Default constructor
	PCB();
//! \brief Constructor for isotropic - different measures basis
	PCB(int ndim, int nord, char t[]	, int const t_tens, bool deriv=1, bool sprod=1, bool tprod=1);
//! \brief Constructor for isotropic / unique measure basis
	PCB(int ndim, int nord, char typ   	, int const t_tens, bool deriv=1, bool sprod=1, bool tprod=1);
//! \brief Constructor for anisotropic - different measures basis
	PCB(int ndim, int nord, char t[]	, int const t_tens, VectorXd const &W, bool deriv=1, bool sprod=1, bool tprod=1);
//! \brief Constructor for anisotropic - unique measure basis
	PCB(int ndim, int nord, char typ   	, int const t_tens, VectorXd const &W, bool deriv=1, bool sprod=1, bool tprod=1);
//! \brief Constructor from multi-index list - different measures basis
	PCB(list<VectorXi> &tens, char t[]	, bool deriv=1, bool sprod=1, bool tprod=1);
//! \brief Constructor from multi-index list - unique measure basis
	PCB(list<VectorXi> &tens, char t	, bool deriv=1, bool sprod=1, bool tprod=1);

//! \brief Destructor.
	~PCB();
//! \brief Free all memory and reset the basis.
	void clear();

//! \brief Normalized Legendre polyomials at x (up to order nord)
	VectorXd pleg_at_x(double const x);
//! \brief Normalized Legendre polyomials at set of x (up to order nord)
	MatrixXd pleg_at_x(const VectorXd &x);
//! \brief Derivatives of normalized Legendre polyomials at x  (up to order nord)
	VectorXd dpleg_at_x(double const x);
//! \brief Derivatives of normalized Legendre polyomials at set of x (up to order nord)
	MatrixXd dpleg_at_x(const VectorXd &x);

//! \brief Normalized Hermite polyomials at x (up to order nord)
	VectorXd pher_at_x(double const x);
//! \brief Normalized Hermite polyomials at set of x (up to order nord)
	MatrixXd pher_at_x(VectorXd const  &x);
//! \brief Derivatives of normalized Hermite polyomials at x  (up to order nord)
	VectorXd dpher_at_x(double const x);
//! \brief Derivatives of normalized Hermite polyomials at set of x (up to order nord)
	MatrixXd dpher_at_x(VectorXd const  &x);

//! \brief PCs' values at an (ndim) point
	VectorXd Comp_Psi(VectorXd const &xi);
//! \brief PCs' values at a set of (ndim) points
	MatrixXd Comp_Psi(MatrixXd const &xi);

//! \brief Set PC expansion of derivative in direction d
	VectorXd Derive_PCE(VectorXd const &pc, int const d);
//! \brief Set PC expansion of derivative in direction d
	MatrixXd Derive_PCE(MatrixXd const &pc, int const d);

//! \brief Set derivation operators
	void SetDerive(){ if(sderive==1){ BasisDer(); sderive = 0;}};
//! \brief Set product operator
	void SetProd(){ if(sprod==1){ BasisProd(); sprod = 0;}};
//! \brief Set triple product operator
	void SetTProd(){ if(striple==1){ BasisTProd(); striple = 0;}};

private:
//! \brief Set data for 1D polynomials and integration rules.
	void Basis1D();
//! \brief Construct the N-dimensional multi-indexes for isotropic basis
	void BasisND();
//! \brief Construct the N-dimensional multi-indexes for anisotropic basis
	void BasisND(VectorXd const &We);

//! \brief Multi-indices for total degree tensozorization, isotropic
	void TotalDegree();
//! \brief Multi-indices for total degree tensozorization, anisotropic
	void TotalDegree(VectorXd const &W);
//! \brief Multi-indices for hyperbolic cross tensozorization, isotropic
	void HyperCross();
//! \brief Multi-indices for full tensozorization, isotropic
	void FullTens();
//! \brief Multi-indices for full tensozorization, anisotropic
	void FullTens(VectorXd const &W);

//! \brief Construct the product operator
	void BasisProd();
//! \brief Construct the triple product operator
	void BasisTProd();
//! \brief Construct the derivation operators
	void BasisDer();
};

