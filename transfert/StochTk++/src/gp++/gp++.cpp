#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <math.h>
#include <random>
#include <time.h>
#include <functional>
#include <Eigen/Dense>
#include "gp++.h"

double GP_Extremum(const std::vector<double> &x, std::vector<double> &grad, void *data)
{
	// std::cout << " In GP Extremum function \n";
	GP *proc = (GP *)data;		 // Pointer to the GP
	Eigen::VectorXd X(x.size()); // Parameters to be optimized
	for (unsigned i = 0; i < x.size(); i++)
		X(i) = x[i];				  // Setting the proposed value of the parameters
	double value = proc->EvalMean(X); // Evaluate the function
	if (!grad.empty())
	{ // Cannot compute gradient : stop!
		std::cout << "Asking for gradient, I stop !" << std::endl;
		exit(1);
	}
	return value;
};

Eigen::VectorXd GP::FindExtremum(Eigen::MatrixXd Bounds)
{
	std::vector<double> lb(dim);
	std::vector<double> ub(dim);
	for (unsigned id = 0; id < dim; id++)
	{
		lb[id] = Bounds(0, id);
		ub[id] = Bounds(1, id);
	}
	std::vector<double> x(dim);
	nlopt::opt opt(nlopt::LN_SBPLX, dim); /* algorithm and dimensionality */
	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(ub);
	opt.set_min_objective(GP_Extremum, this);
	opt.set_xtol_rel(1e-4);
	opt.set_maxeval(1000);
	std::cout << "Starting optimization\n";
	Eigen::VectorXd X(x.size()); // Parameters to be optimized
	std::cout << "Start optimization\n";
	for (unsigned is = 0; is < 10; is++)
	{
		Eigen::VectorXd Xi = (Eigen::VectorXd::Ones(dim) + Eigen::VectorXd::Random(dim)) * .5;
		for (unsigned i = 0; i < x.size(); i++)
			x[i] = Xi(i);
		std::cout << Xi.transpose() << std::endl;
		double minf; /* the minimum objective value, upon return */
		if (opt.optimize(x, minf) < 0)
			printf("nlopt failed!\n");
		std::cout << "Optimization is successful\n";
		for (unsigned i = 0; i < x.size(); i++)
			X(i) = x[i]; // Setting the proposed value of the
		std::cout << X.transpose() << std::endl
				  << std::endl;
	}
	return X;
};

/*Set the Gaussian process using the parameters in par */
double GP::SetGP(const Eigen::VectorXd &par)
{
	PAR = par;
	nd = Xpt.size(); // Number of data points
	unsigned np = par.rows();
	if (np == 2)
	{
		n_free = true;
	}
	else
	{
		n_free = false;
	}
	Eigen::MatrixXd A(nd, nd); // Correlation operator
	Eigen::VectorXd Y(nd);	   // Observations
	if (n_free == false)
	{
		sigsn = pow(par(2), 2); // Noise variance
	}
	else
	{
		sigsn = 1.e-8;
	}
	for (int i = 0; i < nd; i++)
	{
		for (int j = i; j < nd; j++)
		{
			A(i, j) = KERNEL(Xpt[i], Xpt[j], par); // Two points correlation
			if (i != j)
			{
				A(j, i) = A(i, j);
			}
			else
			{
				A(i, j) += sigsn; // Noise correlation
			}
		}
		Y(i) = value[i]; // Noisy observation
	}
	ldlt.compute(A);	   /* Decompose Correlation */
	Alpha = ldlt.solve(Y); /* Solve for the GP coordinates*/
	/* Compute log of SLE optimization */
	logp = -Y.dot(Alpha) - (ldlt.vectorD().array().log()).sum() - (double)(nd)*log(M_PI * 2);
	return -logp / 2.;
};

/*Compute derivative of the SetGP result (derivative of the log-likelihood). Must be called after SetGP because it does not recompute the ldlt of the covariance matrix. Requires also a prior call to SetDKernel.*/
Eigen::VectorXd GP::DerivLL() const
{
	// construction of derivative covariance matrices
	std::vector<Eigen::MatrixXd> DM_vec;
	for (int p = 0; p < PAR.size(); p++)
	{
		Eigen::MatrixXd M(Xpt.size(), Xpt.size());
		for (int i = 0; i < Xpt.size(); i++)
		{
			for (int j = i; j < Xpt.size(); j++)
			{
				M(i, j) = DKERNEL(Xpt[i], Xpt[j], PAR, p); // d??riv??e par rapport au param??tre num??ro p
				if (i != j)
				{
					M(j, i) = M(i, j);
				}
			}
		}
		DM_vec.push_back(M);
	}
	Eigen::MatrixXd Kinv=ldlt.solve(Eigen::MatrixXd::Identity(Xpt.size(),Xpt.size()));
	Eigen::VectorXd grad(PAR.size());
	for(int p=0;p<PAR.size();p++){
		grad(p)=Alpha.transpose().dot(DM_vec[p]*Alpha);
		double traceterm=0;
		for(int i=0;i<Xpt.size();i++){
			for(int j=0;j<Xpt.size();j++){
				traceterm+=Kinv(i,j)*DM_vec[p](j,i);
			}
		}
		grad(p)-=traceterm;
	}
	return -grad/2;
}

Eigen::MatrixXd GP::GetKobs(Eigen::VectorXd &Mean) const
{
	Eigen::MatrixXd K(nd, nd); // Correlation Operator
	for (int i = 0; i < nd; i++)
	{
		for (int j = i; j < nd; j++)
		{
			K(i, j) = KERNEL(Xpt[i], Xpt[j], PAR); // Two points correlation
			if (i != j)
				K(j, i) = K(i, j);
		}
	}
	Mean = K * Alpha;
	return K;
};

Eigen::MatrixXd GP::GetKobs() const
{
	Eigen::MatrixXd K(nd, nd); // Correlation Operator
	for (int i = 0; i < nd; i++)
	{
		for (int j = i; j < nd; j++)
		{
			K(i, j) = KERNEL(Xpt[i], Xpt[j], PAR); // Two points correlation
			if (i != j)
				K(j, i) = K(i, j);
		}
	}
	return K;
};

void GP::OptimizeGP(nlopt::vfunc myoptfunc_gp, Eigen::MatrixXd const *Bounds, Eigen::VectorXd const *Guess, unsigned np)
{
	if (Bounds != NULL)
		np = Bounds->cols();
	if (Guess != NULL)
		np = Guess->rows();
	// std::cout << "Optimize Gaussian process for " << np << " hyperparameters\n";
	std::vector<double> lb(np);
	std::vector<double> ub(np);
	std::vector<double> x(np);

	if (Bounds == NULL)
	{
		if (np == 2)
		{
			lb[0] = 1.e-4;
			ub[0] = 1.e2; // length
			lb[1] = 1.e-5;
			ub[1] = 1.e1; // variance
			n_free = true;
		}
		else if (np == 3)
		{
			lb[0] = 1.e-4;
			ub[0] = 1.e2; // length
			lb[1] = 1.e-5;
			ub[1] = 1.e1; // variance
			lb[2] = 1.e-5;
			ub[2] = 1.e0; // noise intensity
			n_free = false;
		}
		else if (np == 4)
		{
			lb[0] = 1.e-4;
			ub[0] = 1.e2; // length
			lb[1] = 1.e-5;
			ub[1] = 1.e1; // variance
			lb[2] = 1.e-5;
			ub[2] = 1.e0; // noise intensity
			lb[3] = 1.1;
			ub[3] = 2; // Roughness parameter
			n_free = false;
		}
	}
	else
	{
		n_free = false;
		if (np == 2)
			n_free = true;
		for (unsigned ip = 0; ip < np; ip++)
		{
			lb[ip] = (*Bounds)(0, ip);
			ub[ip] = (*Bounds)(1, ip);
		}
	}
	// for(unsigned ip=0; ip<np; ip++) std::cout<< lb[ip] << "  " << ub[ip] << std::endl;
	if (Guess == NULL)
	{
		x[0] = lb[0];
		x[1] = lb[1];
		if (np > 2)
		{
			x[2] = lb[2];
		}
		if (np > 3)
		{
			x[3] = ub[3];
		}
	}
	else
	{
		for (unsigned ip = 0; ip < np; ip++)
		{
			x[ip] = (*Guess)(ip);
		}
	}
	nlopt::opt opt(nlopt::LN_SBPLX, np); /* algorithm and dimensionality */
	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(ub);
	opt.set_min_objective(myoptfunc_gp, this);
	opt.set_xtol_rel(1e-3);
	opt.set_maxeval(1000);
	// std::cout << "Starting optimization\n";
	double minf; /* the minimum objective value, upon return */
	if (opt.optimize(x, minf) < 0)
	{
		printf("nlopt failed!\n");
	}
	else
	{
		// printf("Successful optimization !\n");
	}
	PAR = Eigen::VectorXd(np);
	for (unsigned p = 0; p < np; p++)
		PAR(p) = x[p];
	SetGP(PAR);
};

double GP::EvalMean(Eigen::VectorXd const &x) const
{
	double val = 0;
	for (int i = 0; i < nd; i++)
		val += KERNEL(x, Xpt[i], PAR) * Alpha(i);
	return val;
};

Eigen::VectorXd GP::Eval(Eigen::VectorXd const &x) const
{
	Eigen::VectorXd kstar(nd);
	for (int i = 0; i < nd; i++)
		kstar(i) = KERNEL(x, Xpt[i], PAR);
	Eigen::VectorXd Out(2);
	Out(0) = kstar.dot(Alpha);
	Eigen::VectorXd v = ldlt.solve(kstar);
	Out(1) = fabs(KERNEL(x, x, PAR) - kstar.dot(v));
	return Out;
};

// void GP::Select(double prec)
// {
// 	if (prec > 1)
// 		prec = .9999;
// 	if (prec < 0)
// 		prec = .1;
// 	Eigen::MatrixXd K(nd, nd);
// 	Eigen::VectorXd Y(nd);
// 	for (int i = 0; i < nd; i++)
// 	{
// 		for (int j = i; j < nd; j++)
// 		{
// 			K(i, j) = KERNEL(Xpt[i], Xpt[j], PAR);
// 			K(j, i) = K(i, j);
// 		}
// 		Y(i) = value[i];
// 	}
// 	Sel.clear();
// 	Sel = ColSelect(K, prec); //This is the reduced set of features.
// 	std::cout << "After Selection the GP will be constructed using " << Sel.size() << " data\n";
// 	Eigen::MatrixXd Z(nd, Sel.size());
// 	for (int i = 0; i < Sel.size(); i++)
// 		Z.col(i) = K.col(Sel[i]);
// 	Eigen::MatrixXd R = Z.transpose() * Z / (sigsn) + (Eigen::MatrixXd)Eigen::VectorXd::Ones(Z.cols()).asDiagonal();
// 	Rinv = R.inverse();
// 	BS = Rinv * Z.transpose() * Y / sigsn;
// };

void GP::Select(double prec)
{
	if (prec > 1)
		prec = .9999;
	if (prec < 0)
		prec = .1;
	Eigen::MatrixXd K(nd, nd);
	Eigen::VectorXd Y(nd);
	for (int i = 0; i < nd; i++)
	{
		for (int j = i; j < nd; j++)
		{
			K(i, j) = KERNEL(Xpt[i], Xpt[j], PAR);
			if (i != j)
			{
				K(j, i) = K(i, j);
			}
		}
		Y(i) = value[i];
	}
	Sel.clear();
	Sel = ColSelect(K, prec); // This is the reduced set of features.
	int nsel = Sel.size();
	std::cout << "After Selection the GP will be constructed using " << nsel << " data\n";
	Eigen::MatrixXd R(nsel, nsel);
	Eigen::VectorXd Yr(nsel);
	for (int i = 0; i < nsel; i++)
	{
		int isel = Sel[i];
		for (int j = i; j < nsel; j++)
		{
			int jsel = Sel[j];
			R(i, j) = KERNEL(Xpt[isel], Xpt[jsel], PAR);
			if (i != j)
			{
				R(j, i) = R(i, j);
			}
			else
			{
				R(i, i) += sigsn;
			}
		}
		Yr(i) = value[isel];
	}
	Rinv = R.inverse();
	BS = Rinv * Yr;
};

/* Generate (correlated) samples of the GP at the target points */
Eigen::MatrixXd GP::SampleGP(std::vector<Eigen::VectorXd> const &Target, int ns, std::default_random_engine &gen) const
{
	Eigen::VectorXd Mean;
	Eigen::MatrixXd CoV = CompCov(Target, Mean);
	Eigen::LDLT<Eigen::MatrixXd> DEC(CoV);
	Eigen::VectorXd D = DEC.vectorD();
	for (unsigned i = 0; i < D.rows(); i++)
		D(i) = sqrt(fabs(D(i)));
	// std::cout << "Dmax : " << D.maxCoeff() << " Dmin " << D.minCoeff() << std::endl;
	std::normal_distribution<double> distN(0, 1);
	Eigen::MatrixXd Samples(Target.size(), ns);
	for (unsigned s = 0; s < ns; s++)
	{
		Eigen::VectorXd Eta(CoV.cols());
		for (unsigned i = 0; i < CoV.cols(); i++)
			Eta(i) = distN(gen) * D(i);
		Samples.col(s) = DEC.matrixL() * Eta + Mean;
	}
	return Samples;
};

/* Make correlated sample over a large number of target (sample exactely over the firt nsup and then interpolate) */
Eigen::MatrixXd GP::SampleGPLarge(std::vector<Eigen::VectorXd> const &Target, unsigned nsup, unsigned ns, std::default_random_engine &gen) const
{
	Eigen::MatrixXd Sini;
	{ /*Generate the sample values at the first nsup targets */
		std::vector<Eigen::VectorXd> T0(nsup);
		for (unsigned it = 0; it < nsup; it++)
			T0[it] = Target[it];
		Sini = SampleGP(T0, ns, gen);
		Eigen::MatrixXd A(nsup, nsup); // Correlation operator
		for (unsigned i = 0; i < nsup; i++)
		{
			for (unsigned j = i; j < nsup; j++)
			{
				A(i, j) = KERNEL(T0[i], T0[j], PAR); // Two points correlation
				if (i != j)
				{
					A(j, i) = A(i, j);
				}
				else
				{
					A(i, j) += sigsn;
				}
			}
		}
		Eigen::LDLT<Eigen::MatrixXd> Dec(A);
		Sini = Dec.solve(Sini);
	}
	Eigen::MatrixXd Sample = Eigen::MatrixXd::Zero(Target.size(), ns);
	for (unsigned it = 0; it < nsup; it++)
	{
		for (unsigned jt = 0; jt < Target.size(); jt++)
		{
			Sample.row(jt) += Sini.row(it) * KERNEL(Target[it], Target[jt], PAR);
		}
	}
	return Sample;
};

/* Make direct correlated samples */
Eigen::MatrixXd GP::SampleGPDirect(std::vector<Eigen::VectorXd> const &Target, unsigned ns, std::default_random_engine &gen) const
{
	Eigen::VectorXd Mean;
	Eigen::MatrixXd CoV = CompCov(Target, Mean);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> DEC(CoV);
	Eigen::VectorXd D = DEC.eigenvalues();
	for (unsigned i = 0; i < D.rows(); i++)
		D(i) = sqrt(fabs(D(i)));
	std::cout << "Dmax : " << D.maxCoeff() << " Dmin " << D.minCoeff() << std::endl;
	std::normal_distribution<double> distN(0, 1);
	Eigen::MatrixXd Samples(Target.size(), ns);
	for (unsigned s = 0; s < ns; s++)
	{
		Eigen::VectorXd Eta(CoV.cols());
		for (unsigned i = 0; i < CoV.cols(); i++)
			Eta(i) = distN(gen) * D(i);
		Samples.col(s) = DEC.eigenvectors() * Eta + Mean;
	}
	return Samples;
};

/* compute the mean and covariance matrix of the GP at the target points */
Eigen::MatrixXd GP::CompCov(std::vector<Eigen::VectorXd> const &Target, Eigen::VectorXd &Mean) const
{
	Eigen::MatrixXd CoV(Target.size(), Target.size());
	Mean = Eigen::VectorXd(Target.size());
	std::vector<Eigen::VectorXd> ks;
	for (unsigned it = 0; it < Target.size(); it++)
	{
		Eigen::VectorXd kstar(nd);
		Eigen::VectorXd Xi = Target[it];
		for (int i = 0; i < nd; i++)
			kstar(i) = KERNEL(Xi, Xpt[i], PAR);
		Mean(it) = kstar.dot(Alpha);
		ks.push_back(kstar);
		for (unsigned jt = it; jt < Target.size(); jt++)
		{
			Eigen::VectorXd Xj = Target[jt];
			CoV(it, jt) = KERNEL(Xi, Xj, PAR);
		}
	}
	for (unsigned it = 0; it < Target.size(); it++)
	{
		Eigen::VectorXd Vi = ldlt.solve(ks[it]);
		for (unsigned jt = it; jt < Target.size(); jt++)
		{
			CoV(it, jt) -= Vi.dot(ks[jt]);
			if (it != jt)
				CoV(jt, it) = CoV(it, jt);
		}
	}
	return CoV;
};

double GP::EvalSelMean(Eigen::VectorXd const &x) const
{
	double val = 0;
	for (int i = 0; i < Sel.size(); i++)
		val += KERNEL(x, Xpt[Sel[i]], PAR) * BS(i);
	return val;
};

Eigen::VectorXd GP::EvalSel(Eigen::VectorXd const &x) const
{
	Eigen::VectorXd kstar(Sel.size());
	for (unsigned i = 0; i < Sel.size(); i++)
		kstar(i) = KERNEL(x, Xpt[Sel[i]], PAR);
	Eigen::VectorXd Out(2);
	Out(0) = kstar.dot(BS);
	Out(1) = fabs(KERNEL(x, x, PAR) - kstar.dot(Rinv * kstar));
	return Out;
};

Eigen::VectorXd GP::GetKvec(Eigen::VectorXd const &x) const
{
	Eigen::VectorXd kv(nd);
	for (unsigned d = 0; d < nd; d++)
		kv(d) = KERNEL(x, Xpt[d], PAR);
	return kv;
};

/* Purpose : select the reduced set of columns of A minimizing the Frobenius error*/
std::vector<unsigned> ColSelect(const Eigen::MatrixXd &A, double frac)
{
	std::default_random_engine Rng;
	std::uniform_real_distribution<double> Unif(0, 1);
	int ns = A.cols();
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::VectorXd s = svd.singularValues();
	double st = s.sum();
	double s0 = 0;
	int nr = 0;
	while (s0 < st * sqrt(frac))
	{
		s0 += s(nr);
		nr++;
		if (nr >= A.cols())
			break;
	}
	nr = fmin(nr * 6, A.cols());
	std::cout << s.transpose() << std::endl;
	std::cout << "Need to make selection of " << nr << " columns over " << ns << "\n";

	Eigen::VectorXd Pr(ns);
	Eigen::VectorXd Prc = Eigen::VectorXd::Zero(ns);
	for (int is = 0; is < ns; is++)
	{
		Pr(is) = V.row(is).squaredNorm() / (double)(nr);
		Prc(is) += Pr(is);
		if (is < ns - 1)
			Prc(is + 1) = Prc(is);
	}
	Eigen::VectorXi Drawn = Eigen::VectorXi::Zero(ns);
	std::vector<unsigned> draw;
	while (draw.size() < nr)
	{
		double xp = Prc(ns - 1) * Unif(Rng);
		int is = 0;
		while (Prc(is) < xp)
			is++;
		if (Drawn(is) == 0)
		{
			draw.push_back(is);
			Drawn(is) = 1;
		}
	}
	Eigen::MatrixXd C(A.rows(), nr);
	for (int ik = 0; ik < nr; ik++)
		C.col(ik) = A.col(draw[ik]);
	Eigen::JacobiSVD<Eigen::MatrixXd> SVD(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
	std::cout << SVD.singularValues().transpose() << std::endl;
	return draw;
};
