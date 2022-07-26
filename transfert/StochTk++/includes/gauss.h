#ifndef GLH_H_
#define GLH_H_
#endif

using namespace Eigen;

//! \brief Define Gauss-Legendre quadrature
void gauss_legendre(int npts, VectorXd &, VectorXd &);
//! \brief Define Gauss-Hermite quadrature
void gauss_hermite( int npts, VectorXd &, VectorXd &);
double pdfg( double x);
double cdf_normal(double x);
double inv_cdf_normal(double x);
void u_hermite_recur ( double *p2, double *dp2, double *p1, double x, int order );
void u_hermite_root ( double *x, int order, double *dp2, double *p1 );
double r8_epsilon(void );
double r8_gamma( double x );
double r8_huge( void );
