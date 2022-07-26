#ifndef ATREE_H_
#include "adapt.h"
#endif

#ifndef RTREE_H_
#define RTREE_H_
#endif

/*
SECTION RELATED TO REGRESSION OVER TREES
*/

template <class T>
void TREE<T>::PruneTree(double eps){
	cout << "Clean the tree by coarsening from bottom\n";
	PruneNodes(root,eps);
	SetLeafs();
	SetNodeIndex();
};

template <class T>
void TREE<T>::PruneNodes(NODE<T>* node, double eps){
	if(node->left != NULL){
		PruneNodes(node->left,eps);
		PruneNodes(node->right,eps);
	}
	if(node->left!=NULL){
		if(node->left->dir==0 && node->right->dir==0){
			if(node->det.norm()<=eps){
				RemoveNodes(node->left);
				RemoveNodes(node->right);
				node->dir = 0;
				node->det*= 0;
			}
		}
	}
	return;
};

template<class T>
void TREE<T>::RegressTreeAdapt(MatrixXd const &Xi, VectorXd const &Ob){
	IniRoot();
	root->sol = VectorXd::Zero(mra->Npol());
	root->det = VectorXd::Zero(mra->Npol());
	list<int> lp;
	VectorXd O = Ob;
	for(int is=0; is<Xi.cols(); is++) lp.push_back(is);
	RegressNodeA(root, lp, Xi, O);
	SetNodeIndex();
	SetLeafs();
	PredictTree();
};


template <class T>
void TREE<T>::RegressNodeA(NODE<T> *node, list<int> &lp, MatrixXd const &Xi, VectorXd &Ob){
	node->det = VectorXd::Zero(mra->Npol());
	cout << "\n \t *** Adaptive Regression over a node with " << lp.size() << " points\n";
	if(int(lp.size())<mra->Npol()*1.5+1){
		node->det = VectorXd::Zero(mra->Npol());
		cout << "Exhausted observations : --> Predict." << endl; PredNode(node);
		return;
	}
	VectorXd Rp = Regress(node, lp, Xi, Ob);	//regress the node
	VectorXd Rl[ndim];
	VectorXd Rr[ndim];
	VectorXd RP[ndim];
	VectorXd Ind = VectorXd::Zero(ndim);
	for(int d=0; d<ndim; d++){
		list<int> left, right;
		list<int>::iterator it;
		double xc = (node->Binf(d)+node->Bsup(d))*.5;
		for(it=lp.begin(); it!=lp.end(); it++){
			if(Xi(d,(*it)) <= xc){
				left.push_back((*it));
			}else{
				right.push_back((*it));
			}
		}
		cout << "Direction " << d << " size of children problems " << left.size() << " & " << right.size()<<endl;
		if(int(left.size())<mra->Npol()*1.5+1 || int(right.size())<mra->Npol()*1.5+1 ){
			Ind(d) = 0;
		}else{
			AddChildren(node,d+1);
			Rl[d] = Regress(node->left , left , Xi, Ob);
			Rr[d] = Regress(node->right, right, Xi, Ob);
			RP[d] = mra->Restrict(Rl[d],Rr[d],d);
			Ind(d) = (Rp-RP[d]).norm();
			RemoveNodes(node->left);
			RemoveNodes(node->right);
			node->dir = 0;
		}

	}
	cout << "Indicator : " << Ind.transpose() << endl;
	int d;
	double diff = Ind.maxCoeff(&d);
//	cout << "diff and direction " << diff << " " << d << endl;
	if(Ind(d)==0){
		node->sol += Rp;
		return;
	}
	list<int> left, right;
	list<int>::iterator it;
	double xc = (node->Binf(d)+node->Bsup(d))*.5;
	for(it=lp.begin(); it!=lp.end(); it++){
		if(Xi(d,(*it)) <= xc){
			left.push_back((*it));
		}else{
			right.push_back((*it));
		}
	}

	if(Ind(d)/(Rp.norm()+1.e-12) < 1.e-3){
		cout << "Normalized diff is small " << diff << " \n";
		node->sol += Rp;
		double rm = 0.;
		for(it=lp.begin(); it!=lp.end(); it++){
			Ob((*it)) -= Rp.dot(Comp_Psi(node,Xi.col((*it)))); rm += pow(Ob((*it)),2);
		}
		if(rm > 1.e-10){
			cout << "but significant residual\n";
			AddChildren(node,d+1);
			node->left->sol = VectorXd::Zero(mra->Npol());
			node->left->det = VectorXd::Zero(mra->Npol());
			node->right->sol = VectorXd::Zero(mra->Npol());
			node->right->det = VectorXd::Zero(mra->Npol());
			cout << "call for the children\n";
			RegressNodeA(node->left ,left ,Xi, Ob);
			RegressNodeA(node->right,right,Xi, Ob);
			mra->Restrict(node->left->sol,node->right->sol,Rl[d],Rr[d],d);
			node->sol += Rl[d];
			node->det  = Rr[d];
		}else{
			cout << "Insignificant residual\n";
			return;
		}
	}else{
		cout << "Normalized diff is large " << diff << "\n";
		AddChildren(node,d+1);
		node->left->sol = VectorXd::Zero(mra->Npol());
		node->left->det = VectorXd::Zero(mra->Npol());
		node->right->sol = VectorXd::Zero(mra->Npol());
		node->right->det = VectorXd::Zero(mra->Npol());
		mra->Predict(node->sol,node->left->sol,node->right->sol,d);
		RegressNodeA(node->left ,left ,Xi, Ob);
		RegressNodeA(node->right,right,Xi, Ob);
		mra->Restrict(node->left->sol,node->right->sol,node->sol,node->det,d);
		cout << "Going up, norm is " << node->sol.norm() << " and " << node->det.norm()<<endl;
	}
};



template <class T>
VectorXd TREE<T>::Regress(NODE<T> *node, list<int> &lp, MatrixXd const &Xi, VectorXd const &Ob){
	int npol = mra->Npol();
	VectorXd Rhs = VectorXd::Zero(npol);
	if(lp.size()==0) return Rhs;
	MatrixXd Reg = MatrixXd::Zero(npol,npol);
	list<int>::iterator it,jt;
	cout << "Number of regression points : " << lp.size() << endl;
	for(it=lp.begin(); it !=lp.end(); it++){
		VectorXd Psi= Comp_Psi(node,Xi.col((*it)));
		Reg += Psi*Psi.transpose();
		Rhs += Psi*Ob((*it));
	}
//	VectorXd Sol = Reg.jacobiSvd(ComputeThinU | ComputeThinV).solve(Rhs);
	FullPivLU<MatrixXd> LUR(Reg);
	if(LUR.isInvertible()==0){ cout <<" NON INVERTIBLE \n"; return Rhs*0;}
	VectorXd Sol = LUR.solve(Rhs);
//	cout << "\n Residual of Regression problem is : " << (Reg*Sol - Rhs).norm() << " norm sol " << Sol.norm();
//	cout << " Rhs " << Rhs.norm() << endl;
//	cout << Reg << endl;
	return Sol;
};

template <class T>
void TREE<T>::RegressTree(MatrixXd const &Xi, VectorXd const &Ob){
	list<int> lp;
	VectorXd O = Ob;
	for(int is=0; is<Xi.cols(); is++) lp.push_back(is);
	RegressNode(root, lp, Xi, O);
};

template <class T>
void TREE<T>::RegressNode(NODE<T> *node, list<int> &lp, MatrixXd const &Xi, VectorXd &Ob){
	//1) solve the regression problems over the node.
	if(int(lp.size())<mra->Npol()*1.5+1){
		node->det = VectorXd::Zero(mra->Npol());
		cout << "Exhausted observations : --> Predict." << endl; PredNode(node);
		return;
	}
	if(node->dir==0){
		node->sol += Regress(node, lp, Xi, Ob);
		node->det = node->sol*0;
	}else{
		VectorXd Rp = Regress(node, lp, Xi, Ob);
		int d = node->dir-1;
		list<int> left, right;
		list<int>::iterator it;
		double xc = (node->Binf(d)+node->Bsup(d))*.5;
		for(it=lp.begin(); it!=lp.end(); it++) if(Xi(d,(*it)) <= xc){ left.push_back((*it)); }else{ right.push_back((*it));}
		if(int(left.size())<mra->Npol()*1.5+1 || int(right.size())<mra->Npol()*1.5+1 ){
			PredNode(node);
			node->det = VectorXd::Zero(mra->Npol());
			node->sol += Rp;
			PredNode(node);
			return;
		}
		VectorXd Rl = Regress(node->left , left , Xi, Ob);
		VectorXd Rr = Regress(node->right, right, Xi, Ob);
		VectorXd RP(mra->Npol()), RD(mra->Npol());
		mra->Restrict(Rl,Rr,RP,RD,d);
		if((Rp-RP).norm()/(Rp.norm()+1.e-12) < 1.e-3){ //Global essentially captures the remainer
			node->sol += Rp; double rm = 0.;
			for(it=lp.begin(); it!=lp.end(); it++){
				Ob((*it)) -= Rp.dot(Comp_Psi(node,Xi.col((*it))));
				rm += pow(Ob((*it)),2);
			}
			if(rm > 1.e-10){
				RegressNode(node->left ,left ,Xi, Ob);
				RegressNode(node->right,right,Xi, Ob);
				mra->Restrict(node->left->sol,node->right->sol,Rl,Rr,d);
				node->sol += Rl;
				node->det = Rr;
			}else{
				node->det*=0;
				PredNode(node);
			}
		}else{
			RegressNode(node->left ,left ,Xi, Ob);
			RegressNode(node->right,right,Xi, Ob);
			mra->Restrict(node->left->sol,node->right->sol,node->sol,node->det,d);
		}
	}
};
