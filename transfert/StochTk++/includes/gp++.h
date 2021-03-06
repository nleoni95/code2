#ifndef GP_H_
#define GP_H_
#endif

#ifndef DATA_H_
#define DATA_H_
#include "data++.h"
#endif

#include <nlopt.hpp>

std::vector<unsigned> ColSelect(const Eigen::MatrixXd &A, double frac);

/* GP model class */

class GP
{
	friend class EGO;

public:
	GP()
	{
		KERNEL = NULL;
		n_free = true;
	};

	GP(double (*K)(Eigen::VectorXd const &, Eigen::VectorXd const &, const Eigen::VectorXd &))
	{
		KERNEL = K;
		n_free = true;
	};

	void SetKernel(double (*K)(Eigen::VectorXd const &, Eigen::VectorXd const &, const Eigen::VectorXd &))
	{
		KERNEL = K;
	};

	void SetData(const Eigen::VectorXd &X, double F)
	{
		Xpt.clear();
		value.clear();
		Xpt.push_back(X);
		value.push_back(F);
		nd = Xpt.size();
		dim = Xpt[0].size();
		// std::cout << "The GP will use " << nd << " observations\n";
	};

	void AddData(const Eigen::VectorXd &X, double F)
	{
		Xpt.push_back(X);
		value.push_back(F);
		nd = Xpt.size();
		dim = Xpt[0].size();
		// std::cout << "The GP will use " << nd << " observations\n";
	};

	void SetData(const std::vector<DATA> &data)
	{
		Xpt.clear();
		value.clear();
		for (int i = 0; i < data.size(); i++)
		{
			Xpt.push_back(data[i].X);
			value.push_back(data[i].F);
		}
		nd = Xpt.size();
		dim = Xpt[0].size();
		std::cout << "The GP will use " << nd << " observations\n";
	};

	void AddData(const std::vector<DATA> &data)
	{
		for (int i = 0; i < data.size(); i++)
		{
			Xpt.push_back(data[i].X);
			value.push_back(data[i].F);
		}
		nd = Xpt.size();
		dim = Xpt[0].size();
		// std::cout << "The GP will use " << nd << " observations\n";
	};

	void ChangeDataValues(const Eigen::VectorXd &Fnew)
	{
		if (Fnew.size() != Xpt.size())
		{
			std::cout << "Cannot change the values when sizes are different\n";
			return;
		}
		value.clear();
		for (unsigned i = 0; i < Xpt.size(); i++)
			value.push_back(Fnew(i));
	};

	/*Set the Gaussian process using the parameters in par */
	double SetGP(const Eigen::VectorXd &par);
	/* Optimize the GP parameters */
	void OptimizeGP(nlopt::vfunc myoptfunc_gp, Eigen::MatrixXd const *Bounds, Eigen::VectorXd const *Guess, unsigned np = 2);
	/* Select a subset of data */
	void Select(double prec = .999);

	Eigen::VectorXd FindExtremum(Eigen::MatrixXd Bounds);

	/* Evaluation of the GP mean */
	double EvalMean(Eigen::VectorXd const &x) const;
	/* Evaluate GP at point x, mean and variance of prediction*/
	Eigen::VectorXd Eval(Eigen::VectorXd const &x) const;
	/* Evaluation of the GP mean */
	double EvalSelMean(Eigen::VectorXd const &x) const;
	Eigen::VectorXd PredictObservations() const;
	/* Evaluation of the GP mean and variance */
	Eigen::VectorXd EvalSel(Eigen::VectorXd const &x) const;
	Eigen::VectorXd GetAlpha() const { return Alpha; };
	/* Retrieve dimension of the GP */
	unsigned Dim() const { return dim; };
	/* Retrieve number of observations */
	unsigned NDat() const { return Xpt.size(); };
	/* Retrieve location of i-th observation */
	Eigen::VectorXd XDat(unsigned const i) const { return Xpt[i]; };
	/* Retrieve value of i-th observation */
	double FDat(unsigned const i) const { return value[i]; };
	/* Retrieve the log MSE */
	double GetLogP() const { return logp; };
	/* Retrieve parameters of the GP */
	Eigen::VectorXd GetPar() const { return PAR; };
	/* Retrieve noise variance */
	double Sig2() const { return sigsn; };
	/* A priori correlation between two points */
	double Kij(Eigen::VectorXd const x, Eigen::VectorXd const y) const { return KERNEL(x, y, PAR); };
	/* Retrieve the LDLT decomposition of the GP Model */
	Eigen::LDLT<Eigen::MatrixXd> GetLDLT() const { return ldlt; };
	/* Retrieve the selected nodes */
	std::vector<unsigned> Selected() const { return Sel; };

	/* Compute Covariance matrix and mean values of the GP at the target points */
	Eigen::MatrixXd CompCov(std::vector<Eigen::VectorXd> const &Target, Eigen::VectorXd &Mean) const;
	/* Sample the GP at the target points */
	Eigen::MatrixXd SampleGP(std::vector<Eigen::VectorXd> const &Target, int ns, std::default_random_engine &gen) const;
	/* Sample the GP at a large set of target points */
	Eigen::MatrixXd SampleGPLarge(std::vector<Eigen::VectorXd> const &Target, unsigned nsup, unsigned ns, std::default_random_engine &gen) const;

	/* Sample the GP directly */
	Eigen::MatrixXd SampleGPDirect(std::vector<Eigen::VectorXd> const &Target, unsigned ns, std::default_random_engine &gen) const;
	Eigen::VectorXd GetKvec(Eigen::VectorXd const &x) const;
	Eigen::MatrixXd GetKobs() const;
	Eigen::MatrixXd GetKobs(Eigen::VectorXd &Mean) const;
	unsigned RedDim() { return Sel.size(); /*number of reduced data*/ };
	std::vector<unsigned> GetSel() { return Sel; };

	/* Function for derivatives*/
	void SetDKernel(double (*DK)(Eigen::VectorXd const &, Eigen::VectorXd const &, const Eigen::VectorXd &, int))
	{
		DKERNEL = DK;
	};
	Eigen::VectorXd DerivLL() const;

private:
	unsigned dim;
	unsigned nd;
	bool n_free;
	double sigsn;
	double logp;
	Eigen::LDLT<Eigen::MatrixXd> ldlt;
	Eigen::VectorXd Alpha;
	std::vector<Eigen::VectorXd> Xpt;
	std::vector<double> value;
	double (*KERNEL)(const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &);
	double (*DKERNEL)(const Eigen::VectorXd &, const Eigen::VectorXd &, const Eigen::VectorXd &, int i);
	Eigen::VectorXd PAR;
	std::vector<unsigned> Sel;
	Eigen::VectorXd BS;
	Eigen::MatrixXd Rinv;
};
