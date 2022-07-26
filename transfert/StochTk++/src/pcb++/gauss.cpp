#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <Eigen/Dense>
#include "gauss.h"

using namespace std;
using namespace Eigen;

double pdfg(double x){
/// PDF of standard Normal distribution
	double  pi = 3.1415926535897932385E0;
	return exp(-x*x/2 )/sqrt(2*pi);
};

double cdf_normal(double x){
/// CDF of standard Normal distribution
	x = x / sqrt(2);
	return (erf(x)+1)*.5;
};

double inv_cdf_normal(double x){
	double y=0;
	double dist;
	dist = cdf_normal(y) -x;
	while(fabs(dist)>=1.e-10){
		y = y - dist / pdfg(y);
		dist = cdf_normal(y)-x;
	}
	return y;
};

void gauss_legendre(int npts, VectorXd &xleg, VectorXd &wleg){
/// \note for integration over [0,1] with weight 1\n
/// Input : number of points \n
///	Output : vector of points and weights.
	int     m, i, j;
	double  t, t1, pp, p1, p2, p3;
	double pi = 3.1415926535897932384626434;
	double  eps = 3.e-14;			/* limit for accuracy */
	m = (npts+1)/2;
	pp = 0.;
	xleg = VectorXd(npts);
	wleg = VectorXd(npts);
	for(i=1; i<=m; i++){
		t  = cos(pi*(i-0.25)/(npts+0.5));
		t1 = 1;
		while((fabs(t-t1))>=eps){
			p1 = 1.0;
			p2 = 0.0;
			for(j=1; j<=npts; j++){
				p3 = p2;
				p2 = p1;
				p1 = ((2*j-1)*t*p2-(j-1)*p3)/j;
			}
			pp = npts*(t*p1-p2)/(t*t-1);
			t1 = t;
			t  = t1 - p1/pp;
		}
		xleg[i-1]    = -t;
		xleg[npts-i] = t;
		wleg[i-1]    = 2.0/((1-t*t)*pp*pp);
		wleg[npts-i] = wleg[i-1];
	}
	for(i=0; i<npts; i++){
		xleg[i] = (xleg[i]+1)/2;
		wleg[i] = wleg[i]/2;
	}
};

void gauss_hermite( int npts, VectorXd &xher, VectorXd &wher){
/// \note Integration with weight exp( -x*x/2 ) / sqrt(2pi)\n
/// Input : number of points \n
///	Output : vector of points and weights.
	double cc, dp2, p1, s, temp, x=0;
	double pi = 3.1415926535897932384626434;
	cc = 1.7724538509 * r8_gamma ( (double) (npts) ) / pow (2.0,npts-1);
	s = pow ( 2.0 * (double)(npts)+1.0,1.0/6.0);
	xher = VectorXd(npts);
	wher = VectorXd(npts);

	for (int i = 0; i<(npts+1)/2; i++ ){
		if (i == 0){
			x = s*s*s - 1.85575/s;
		}else if(i==1){
			x = x - 1.14 * pow ((double)(npts), 0.426) / x;
		}else if (i==2){
			x = 1.86*x - 0.86 * xher[0];
		}else if (i==3){
			x = 1.91*x - 0.91 * xher[1];
		}else{
			x = 2.0*x - xher[i-2];
		}
		u_hermite_root ( &x, npts, &dp2, &p1 );
		xher[i] = x;
		wher[i] = ( cc / dp2 ) / p1;
		xher[npts-i-1] = -x;
		wher[npts-i-1] = wher[i];
	}
	for (int i = 1; i <= npts/2; i++ ){
		temp          = xher[i-1];
		xher[i-1]     = xher[npts-i];
		xher[npts-i] = temp;
	}
	for (int i = 0; i < npts; i++ ){
		xher[i] = xher[i]*sqrt(2.);
		wher[i] = wher[i]/sqrt(pi);
	}
	return;
};

void u_hermite_recur ( double *p2, double *dp2, double *p1, double x, int order ){
	double dq0, dq1, dq2, q0, q1, q2;
	q1 = 1.0;
	dq1 = 0.0;
	q2 = x;
	dq2 = 1.0;
	for (int i = 2; i<=order; i++ ){
		q0  = q1;
		dq0 = dq1;
		q1  = q2;
		dq1 = dq2;
		q2  = x*q1       - 0.5*( ( double )( i ) - 1.0 )*q0;
		dq2 = x*dq1 + q1 - 0.5*( ( double )( i ) - 1.0 )*dq0;
	}
	*p2 = q2;
	*dp2 = dq2;
	*p1 = q1;
	return;
};

void u_hermite_root ( double *x, int order, double *dp2, double *p1 ){
	double d, eps,p2;
	int step_max = 10;
	eps = r8_epsilon ( );
	for (int step = 1; step <= step_max; step++ ){
		u_hermite_recur ( &p2, dp2, p1, *x, order );
		d = p2 / ( *dp2 );
		*x -= d;
		if (fabs(d)<=eps*(fabs( *x )+ 1.0) )return;
	}
	return;
};

double r8_epsilon ( void ){
// R8_EPSILON returns the R8 roundoff unit.
	double value=1.;
	while ( 1.0 < (1.0+value)) value = value / 2.0;
	value = 2.0 * value;
	return value;
}

double r8_gamma( double x ){
//    R8_GAMMA evaluates Gamma(X) for a real argument.
//    This routine calculates the gamma function for a real argument X.
//    Computation is based on an algorithm outlined in reference 1.
//    The program uses rational functions that approximate the gamma
//    function to at least 20 significant decimal digits.  Coefficients
//    for the approximation over the interval (1,2) are unpublished.
//    Those for the approximation for 12 <= X are from reference 2.
//  Parameters:
//    Input, double X, the argument of the function.
//    Output, double R8_GAMMA, the value of the function.

//  Coefficients for minimax approximation over (12, INF).
	double c[7] = {-1.910444077728E-03, 8.4171387781295E-04, -5.952379913043012E-04, 7.93650793500350248E-04, -2.777777777777681622553E-03, 8.333333333333333331554247E-02, 5.7083835261E-03};
	double eps = 2.22E-16;
	double fact;
	double half = 0.5;
	int i;
	int n;
	double one = 1.0;
	double p[8] = {-1.71618513886549492533811E+00, 2.47656508055759199108314E+01, -3.79804256470945635097577E+02, 6.29331155312818442661052E+02, 8.66966202790413211295064E+02, -3.14512729688483675254357E+04, -3.61444134186911729807069E+04, 6.64561438202405440627855E+04 };
	bool parity;
	double pi = 3.1415926535897932384626434;
	double q[8] = {-3.08402300119738975254353E+01, 3.15350626979604161529144E+02, -1.01515636749021914166146E+03,
		-3.10777167157231109440444E+03, 2.25381184209801510330112E+04, 4.75584627752788110767815E+03, -1.34659959864969306392456E+05, -1.15132259675553483497211E+05 };
	double res;
	double sqrtpi = 0.9189385332046727417803297;
	double sum;
	double twelve = 12.0;
	double two = 2.0;
	double value;
	double xbig = 171.624;
	double xden;
	double xinf = 1.79E+308;
	double xminin = 2.23E-308;
	double xnum;
	double y;
	double y1;
	double ysq;
	double z;
	double zero = 0.0;;
	parity = false;
	fact = one;
	n = 0;
	y = x;
//  Argument is negative.
	if ( y <= zero ){
		y = - x;
		y1 = ( double ) ( int ) ( y );
		res = y - y1;
		if ( res != zero ){
			if ( y1 != ( double ) ( int ) ( y1 * half ) * two ){
				parity = true;
			}
			fact = - pi / sin ( pi * res );
			y = y + one;
		}else{
			res = xinf;
			value = res;
			return value;
		}
	}
//  Argument is positive.
	if ( y < eps ){
		if ( xminin <= y ){
			res = one / y;
		}else{
			res = xinf;
			value = res;
			return value;
		}
	}
	else if ( y < twelve ){
		y1 = y;
		if ( y < one ){
			z = y;
			y = y + one;
		}else{
			n = (int)(y) - 1;
			y = y - (double)(n);
			z = y - one;
		}
		xnum = zero;
		xden = one;
		for ( i = 0; i < 8; i++ ){
			xnum = ( xnum + p[i] ) * z;
			xden = xden * z + q[i];
		}
		res = xnum / xden + one;
		if ( y1 < y ){
			res = res / y1;
		} else if( y<y1 ){
			for ( i = 1; i <= n; i++ ){
				res = res * y;
				y = y + one;
			}
		}
	}else{
		if ( y <= xbig ){
			ysq = y * y;
			sum = c[6];
			for ( i = 0; i < 6; i++ )
				sum = sum / ysq + c[i];
			sum = sum / y - y + sqrtpi;
			sum = sum + ( y - half ) * log ( y );
			res = exp ( sum );
		}else{
			res = xinf;
			value = res;
			return value;
		}
	}
	if ( parity )res = - res;
	if ( fact != one )res = fact / res;
	value = res;
	return value;
};

double r8_huge (){//    R8_HUGE returns a "huge" R8.
	double value;
	value = 1.0E+30;
	return value;
};

