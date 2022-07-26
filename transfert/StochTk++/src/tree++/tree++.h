#ifndef MRA_H_
#include "mra++.h"
#endif

#ifndef Vstoch_H_
#include "vstoch++.h"
#endif

#include <vector>

#ifndef TREE_H_
#define TREE_H_
#endif

const int cut_min = 3;
const int cut_init = 0;
const int cut_max = 40;
const int cut_max_d = 6;

using namespace Eigen;
using namespace std;

template <class TL> class TREE;	//forward declaration

/*
	TEMPLATE CLASS NODE
*/



template <class TN> class NODE {
	
	template <class T> friend class TREE;

public:
	int dir;
	int index;
	int lindex;
	VectorXd binf;
	VectorXd bsup;
	VectorXi ncut;
	NODE<TN> *parent;
	NODE<TN> *left;
	NODE<TN> *right;
	TN sol;
	TN det;

public:
//Default constructor
	NODE(){ dir=0; index=0; parent =NULL; left=NULL; right=NULL; lindex=-1;};
//Copy constructor
	NODE(const NODE<TN> &U){
		dir    = U.dir;
		index  = U.index;
		lindex = U.lindex;
		binf   = U.binf;
		bsup   = U.bsup;
		ncut    = U.ncut;
		parent  = U.parent;
		left    = U.left;
		right   = U.right;
		sol     = U.sol;
		det     = U.det;
	};
	void operator=(const NODE<TN> &U){
		dir    = U.dir;
		index  = U.index;
		lindex = U.lindex;
		binf   = U.binf;
		bsup   = U.bsup;
		ncut    = U.ncut;
		parent  = U.parent;
		left    = U.left;
		right   = U.right;
		sol     = U.sol;
		det     = U.det;
	};
	template <class T> friend ostream& operator<<(ostream & out, const NODE<T> & B);
	int Dir() const {return dir;};
	int Index() const {return index;};
	int LIndex() const {return lindex;};
	VectorXd Binf() const {return binf;};
	VectorXd Bsup() const {return bsup;};
	double Binf(int const i) const {return binf(i);};
	double Bsup(int const i) const {return bsup(i);};
	VectorXi Ncut() const {return ncut;};
	int Ncut(const int i) const {return ncut(i);};
	int Level() const {return ncut.sum();};
	double Vol() const {double vol=1; for(int id=0; id<binf.rows(); id++) vol*= (bsup(id)-binf(id)); return vol;};
	NODE<TN> *Par() const {return parent;};
	NODE<TN> *Left() const {return left;};
	NODE<TN> *Right() const {return right;};
	TN Sol() const {return sol;};
	TN Det() const {return det;};
	void SetSol(TN const &s) { sol = s;};
	void SetDet(TN const &d) { det = d;};
};



template <class TN> class PairLeaf {
public:
	NODE<TN> *left;
	NODE<TN> *right;
};


/*
	TEMPLATE CLASS TREE
*/

template <class Tr> class TREE {
private:
	NODE<Tr>* root;
	int nnode;
	int nleaf;
	int ndim;
	int nord;
	int npol;
	MRA* mra;

public:
	int Nodes() const {return nnode;};
	int Leafs() const {return nleaf;};
	int INodes() const {return nnode-nleaf;};
	int Ndim() const {return ndim;};
	int Npol() const {return npol;};
	int Nord() const {return nord;};
	MRA* Mra() const {return mra;};
	NODE<Tr>* Root() const {return root;};

	TREE();
	TREE(TREE<Tr> &S); //copy constructor
	~TREE();
	void clear();
	void IniTree(MRA* basis);
	void IniTree(MRA* basis, int mdir);
	void IniTree(MRA* basis, int mdir[]);
	void IniTree(MRA *basis, MatrixXd const &Xobs, int n_per_dom);
	void IniTree(MRA *basis, MatrixXd const &Xobs, const VectorXd &Dat, int n_per_dom);
	void Part_Node_Centroid(NODE<Tr> *node, int nc, int lis[], MatrixXd const &Xc);

	void CopyTree(TREE<Tr> &S);
	void RandomTree(int acc);
	void NewBasis(MRA *bnew){
		if(bnew->Ndim() != mra->Ndim()){ cout << "WARNING : can't change basis with different dimensions\n"; return;}
		mra = bnew;
	};
	void PruneTree(double eps);
	void SetSupports();
	void ListLeafs(NODE<Tr> *pleaf[]);
	void SetLeafs();
	void SetLeafIndex();
	int FindLeafIndex(VectorXd const &xp);

	void ListNodes(NODE<Tr> *pnode[]);
	void ListINodes(NODE<Tr> *pnode[]);
	void SetNodeIndex();
	NODE<Tr>* FindNode(VectorXd const &xp);

	VectorXd Comp_Psi(NODE<Tr> *node, VectorXd const & Xi){
		//Scale coordinates:
		VectorXd Xl = Xi - node->Binf();
		Xl = Xl.array()/(node->Bsup()-node->Binf()).array();
		if( Xl.minCoeff()<0 || Xl.maxCoeff() > 1) return VectorXd::Zero(npol);
		return mra->Comp_Psi(Xl);
	};

	VectorXd Comp_Det(NODE<Tr> *node, VectorXd const & Xi){
		//Scale coordinates:
		if(node->dir==0) return VectorXd::Zero(npol);
		VectorXd Xl = Xi - node->Binf();
		Xl = Xl.array()/(node->Bsup()-node->Binf()).array();
		if( Xl.minCoeff()<0 || Xl.maxCoeff() > 1) return VectorXd::Zero(npol);
		return mra->Comp_Det(Xl,node->dir-1);
	};
	
	double SValue_at_xi(VectorXd const & Xi){
		NODE<Tr>* node = FindNode(Xi);
		VectorXd Xl = Xi - node->Binf();
		Xl = Xl.array()/(node->Bsup()-node->Binf()).array();
		return node->sol.Value_at_xi(Xl);
	};
	
	SOLMod Value_at_xi(VectorXd const & Xi){
		NODE<Tr>* node = FindNode(Xi);
		VectorXd Xl = Xi - node->Binf();
		Xl = Xl.array()/(node->Bsup()-node->Binf()).array();
		return node->sol.Value_at_xi(Xl);
	}

	VectorXd VValue_at_xi(VectorXd const & Xi){
		NODE<Tr>* node = FindNode(Xi);
		VectorXd Xl = Xi - node->Binf();
		Xl = Xl.array()/(node->Bsup()-node->Binf()).array();
		return node->sol.Value_at_xi(Xl);
	};

	void PredictTree();
	void PredictTreeNoDetail();
	void RestrictTree();
	void SynthTree();
	void RegressTree(MatrixXd const &Xi, VectorXd const &Ob);
	void RegressTreeAdapt(MatrixXd const &Xi, VectorXd const &Ob);
	void OptimTree(TREE<Tr> &Orig);
	void UnionTrees(TREE<Tr> &S, TREE<Tr> &T);
	void ShakeTree();
	void ShakeTree2();
	double SumDet(NODE<Tr> *leaf);
	void SCoarsenTree(double const eta);
	void VCoarsenTree(double const eta);
	void EnrichFull();
	void Enrich(VectorXi const &Dir);
	void PlotLeafs(string name);
	void PlotTree(char *name);
	void ListTree();

	std::vector<int> Intersect(VectorXd const &Binf, VectorXd const &Bsup, NODE<Tr> *pl[]);

	double SMean() {
		double mean;
		NODE<Tr>* leafs[Leafs()];
		ListLeafs(leafs);
		mean = leafs[0]->sol.Mean()*leafs[0]->Vol();
		for(int il=1; il<Leafs(); il++) mean += leafs[il]->sol.Mean()*leafs[il]->Vol(); 
		return mean;
	};
	
	VectorXd VMean() {
		VectorXd mean;
		NODE<Tr>* leafs[Leafs()];
		ListLeafs(leafs);
		mean = leafs[0]->sol.Mean()*leafs[0]->Vol();
		for(int il=1; il<Leafs(); il++) mean += leafs[il]->sol.Mean()*leafs[il]->Vol(); 
		return mean;
	};

	double SStd() {
		double std, mean;
		NODE<Tr>* leafs[Leafs()];
		ListLeafs(leafs);
		mean = SMean();
		std = leafs[0]->sol.Norm2()*leafs[0]->Vol();
		for(int il=1; il<Leafs(); il++) std += leafs[il]->sol.Norm2()*leafs[il]->Vol(); 
		std = sqrt(std-mean*mean);
		return std;
	};

	double SVar() {
		double var, mean;
		NODE<Tr>* leafs[Leafs()];
		ListLeafs(leafs);
		mean = SMean();
		var = leafs[0]->sol.Norm2()*leafs[0]->Vol();
		for(int il=1; il<Leafs(); il++) var += leafs[il]->sol.Norm2()*leafs[il]->Vol(); 
		var = (var-mean*mean);
		return var;
	};
	
	VectorXd VStd() {
		VectorXd std, mean;
		NODE<Tr>* leafs[Leafs()];
		ListLeafs(leafs);
		mean = VMean();
		std = leafs[0]->sol.CNorm2()*leafs[0]->Vol();
		for(int il=1; il<Leafs(); il++) std += leafs[il]->sol.CNorm2()*leafs[il]->Vol(); 
		std = (std.array()-mean.array().square()).sqrt();
		return std;
	};

	double SNorm2() {
		double snorm;
		NODE<Tr>* leafs[Leafs()];
		ListLeafs(leafs);
		snorm = leafs[0]->sol.Norm2()*leafs[0]->Vol();
		for(int il=1; il<Leafs(); il++) snorm += leafs[il]->sol.Norm2()*leafs[il]->Vol(); 
		return snorm;
	};

private:
	void IniRoot();
	void IniTree(NODE<Tr> *node);
	void IniTree(NODE<Tr> *node, int mdir);
	void IniTree(NODE<Tr> *node, int mdir[]);
	void RandomTree(NODE<Tr> *node, int acc);
	void IniTreePart(NODE<Tr> *node, MatrixXd &Obs, int n_per_dom);
	void SplitPart(NODE<Tr> *node, const MatrixXd &Obs, int n_per_dom);
	void SplitPart(NODE<Tr> *node, list<int> liste, const MatrixXd &Obs, int n_per_dom);
	void SplitPart(NODE<Tr> *node, list<int> liste, const MatrixXd &Obs, const VectorXd &Dat, int n_per_dom);
	void SetSupports(NODE<Tr> *node);
	void SetLeafs(NODE<Tr> *node);
	void ListLeafs(NODE<Tr> *node, int &nl, NODE<Tr> *pleaf[]);
	void SetLeafIndex(NODE<Tr> *node,int *index);
	void SetPairLeaf(NODE<Tr> *node, list<PairLeaf<Tr> > &pairs);
	void RemoveNodes(NODE<Tr> *node);
	void RemoveChildren(NODE<Tr> *node);	
	void AddChildren(NODE<Tr> *node, int idir);
	void ListNodes(NODE<Tr> *node, int &nn, NODE<Tr> *pnode[]);
	void ListINodes(NODE<Tr> *node, int &nn, NODE<Tr> *pnode[]);
	void CopyNodes(NODE<Tr> *t, NODE<Tr> *s);
	void SetNodeIndex(NODE<Tr> *node, int *index);
	NODE<Tr>* FindNode(NODE<Tr> *node, VectorXd const &xp);

	void PredNode(NODE<Tr> *t);
	void PredNodeNoDetail(NODE<Tr> *t);
	void PredNodeEnrich(NODE<Tr> *t);	
	void RestNode(NODE<Tr> *t);
	void Restrict(NODE<Tr> *n);	
	void SynthNode(NODE<Tr> *t);
	void RegressNode(NODE<Tr> *n, list<int> &l, MatrixXd const &Xi, VectorXd &Ob);
	VectorXd Regress(NODE<Tr> *n, list<int> &l, MatrixXd const &Xi, VectorXd const &Ob);
	void RegressNodeA(NODE<Tr> *n, list<int> &l, MatrixXd const &Xi, VectorXd &Ob);
	
	int Intersect(NODE<Tr> *n1, NODE<Tr>* n2);
	void SetLocSol(NODE<Tr> *n1, VectorXd &binf, VectorXd &bsup, VectorXd &sol);
	void SetGloSol(NODE<Tr> *n1, VectorXd &binf, VectorXd &bsup, VectorXd &sol);
	int OptimLR(const int id, NODE<Tr>* node, TREE<Tr> &Orig, NODE<Tr>* leafs[], VectorXd &soll, VectorXd &solr);
	void OptimNode(NODE<Tr> *node, TREE<Tr> &Orig, NODE<Tr>* leafs[]);
	void PruneNodes(NODE<Tr>* node,double eps);
	void UnionTrees(NODE<Tr> *node);
	void ExtUnion(NODE<Tr> *tu, NODE<Tr> *t2, VectorXd const &xm, int idir);
	void ExtUnion2(NODE<Tr> *tu, VectorXd const &binf, VectorXd const &bsup);
	int is_intersecting(NODE<Tr> *node, VectorXd const &Xinf, VectorXd const &Xsup);
	void ShakeTree(NODE<Tr> *node);

	void PlotLeaf3D(NODE<Tr> *node, FILE *output);
	void PlotLeaf2D(NODE<Tr> *node, FILE *output);
	void PlotLeaf(NODE<Tr> *node, FILE *output);
	void PlotNode(NODE<Tr> *node, FILE *output);
	void ListTree(NODE<Tr> *node);
};
