#ifndef BTREE_H_
#include "basic.h"
#endif

#ifndef ATREE_H_
#define ATREE_H_
#endif

#include <vector>

using namespace std;

/*
	SECTION RELATED TO SOLUTION REPRESENTATION OVER THE TREE
*/


template <class T>
void TREE<T>::PredictTree(){
	PredNode(root);
};

template <class T>
void TREE<T>::PredictTreeNoDetail(){
	PredNodeNoDetail(root);
};


template <class T>
void TREE<T>::PredNode(NODE<T> *node){
	if(node->left!=NULL){	//Node has children
		int dir = node->dir-1;
		T Cl = node->sol*0;
		T Cr = node->sol*0;
		list<op_mra>::iterator it;
		for(it=mra->Prp[dir].begin(); it!=mra->Prp[dir].end(); it++){
			Cl.add_mode((*it).ip,node->sol[(*it).jp]*(*it).cl);
			Cr.add_mode((*it).ip,node->sol[(*it).jp]*(*it).cr);
		}
		for(it=mra->Prd[dir].begin(); it!=mra->Prd[dir].end(); it++){
			Cl.add_mode((*it).ip,node->det[(*it).jp]*(*it).cl);
			Cr.add_mode((*it).ip,node->det[(*it).jp]*(*it).cr);
		}
		node->left->sol  = Cl;
		node->right->sol = Cr;
		PredNode(node->left);
		PredNode(node->right);
	}else{
		node->det = node->sol*0;
	}
};

template <class T>
void TREE<T>::PredNodeNoDetail(NODE<T> *node){
	if(node->left!=NULL){	//Node has children
		int dir = node->dir-1;
		T Cl = node->sol*0;
		T Cr = node->sol*0;
		list<op_mra>::iterator it;
		for(it=mra->Prp[dir].begin(); it!=mra->Prp[dir].end(); it++){
			Cl.add_mode((*it).ip,node->sol[(*it).jp]*(*it).cl);
			Cr.add_mode((*it).ip,node->sol[(*it).jp]*(*it).cr);
		}
		node->left->sol  = Cl;
		node->right->sol = Cr;
		PredNodeNoDetail(node->left);
		PredNodeNoDetail(node->right);
	}else{
		node->det = node->sol*0;
	}
};

template <class T>
void TREE<T>::PredNodeEnrich(NODE<T> *node){
	if(node->left!=NULL){	//Node has children
		int dir = node->dir-1;
		T Cl = node->sol*0;
		T Cr = node->sol*0;
		list<op_mra>::iterator it;
		for(it=mra->Prp[dir].begin(); it!=mra->Prp[dir].end(); it++){
			Cl.add_mode((*it).ip,node->sol[(*it).jp]*(*it).cl);
			Cr.add_mode((*it).ip,node->sol[(*it).jp]*(*it).cr);
		}
		node->left->sol  = Cl;
		node->right->sol = Cr;
		PredNodeNoDetail(node->left);
		PredNodeNoDetail(node->right);
	}
};

template <class T>
void TREE<T>::RestrictTree(){
	RestNode(root);
};

template <class T>
void TREE<T>::RestNode(NODE<T> *node){
	if(node->left==NULL){
		node->det = node->sol*0;
		return;
	}
	RestNode(node->left);
	RestNode(node->right);
	Restrict(node);
};

template <class T> 
void TREE<T>::Restrict(NODE<T>* node){
	node->sol = node->left->sol*0;
	node->det = node->left->sol*0;
	list<op_mra>::iterator it;
	int dir = node->dir-1;
	for(it=mra->Prp[dir].begin(); it!=mra->Prp[dir].end(); it++){
		node->sol.add_mode((*it).jp, node->left->sol[(*it).ip]*(*it).cl + node->right->sol[(*it).ip]*(*it).cr);
	}
	node->sol *= 0.5;
	node->det = node->sol*0;
	for(it=mra->Red[dir].begin(); it!=mra->Red[dir].end(); it++){
		node->det.add_mode((*it).ip, node->left->sol[(*it).jp]*(*it).cl + node->right->sol[(*it).jp]*(*it).cr);
	}
};


template <class T> double TREE<T>::SumDet(NODE<T> *leaf){
	NODE<T> nod = leaf;
	double sd = nod->det.squaredNorm()/nod->Vol()/nod->Vol();
	while(nod->parent != NULL){
		nod = nod->parent;
		sd += nod->det.squaredNorm()/nod->Vol()/nod->Vol();
	}
	return sd;
};

template <class T> void TREE<T>::VCoarsenTree(double const eta){
	cout << "\t Coarsening:\n";
	list<PairLeaf<T> > pairs;
	NODE<T> *node;
	typename list<PairLeaf<T>, std::allocator<PairLeaf<T> > >::iterator l;
	int nbleaf = Leafs();
	do{
		nbleaf = Leafs();
		if(nbleaf==1) break;		
		node = root;
		pairs.clear();
		SetPairLeaf(root,pairs);
		if(pairs.size()==0) break;
		for(l=pairs.begin(); l!=pairs.end(); l++){
			node = (*l).left->parent;
			Restrict(node);
			double Var = node->det.Std().sum();//*sqrt(node->Vol());
			if(Var<eta) RemoveChildren(node);
		}
		cout << "# of leafs : " << Leafs() << endl;
	}while(nbleaf != Leafs());
	SetLeafs();
	cout << "done with coarsening\n";
};

template <class T> void TREE<T>::SCoarsenTree(double const eta){
	cout << "\t Coarsening:\n";
	list<PairLeaf<T> > pairs;
	NODE<T> *node;
	typename list<PairLeaf<T>, std::allocator<PairLeaf<T> > >::iterator l;
	int nbleaf = Leafs();
	do{
		nbleaf = Leafs();
		if(nbleaf==1) break;		
		node = root;
		pairs.clear();
		SetPairLeaf(root,pairs);
		if(pairs.size()==0) break;
		for(l=pairs.begin(); l!=pairs.end(); l++){
			node = (*l).left->parent;
			Restrict(node);
			double Var = node->det.Std();//*sqrt(node->Vol());
			if(Var<eta) RemoveChildren(node);
		}
		cout << "# of leafs : " << Leafs() << endl;
	}while(nbleaf != Leafs());
	SetLeafs();
	cout << "done with coarsening\n";
};

template <class T> void TREE<T>::EnrichFull(){
	cout <<"\t Enrichment\n";
	NODE<T> *node;
	for(int id=0; id<ndim; id++){   		//Force enrichment once along each dimension
		SetLeafs();
		NODE<T>* leafs[nleaf];
		ListLeafs(leafs);
		for(int l=0; l<nleaf; l++){
			node = leafs[l];
			if(node->ncut(id)<cut_max_d){
				AddChildren(node,id+1);
				PredNodeEnrich(node);
			}
		}	
	}
	SetLeafs();
	cout <<"Number of leafs after enrichment: " << nleaf << endl;
};

template <class T> 
vector<int> TREE<T>::Intersect(VectorXd const &Binf, VectorXd const &Bsup, NODE<T> *pl[]){
	vector<int> inters;
	for(int l=0; l<nleaf; l++){
		VectorXd Dsup = Bsup - pl[l]->Binf();
		if(Dsup.minCoeff()>0){
			Dsup = pl[l]->Bsup() - Binf;
			if(Dsup.minCoeff()>0) inters.push_back(l);
		}
	}
	return inters;
};

template <class T> void TREE<T>::Enrich(VectorXi const &Dir){
	NODE<T> *node;
	SetLeafs();
	NODE<T>* leafs[nleaf]; ListLeafs(leafs);
	for(int l=0; l<nleaf; l++){
		if(Dir(l)>0){
			int id = Dir(l)-1;
			node = leafs[l];
			if(node->ncut(id)<cut_max_d){
				AddChildren(node,id+1);
				PredNodeEnrich(node);
			}
		}	
	}
	SetLeafs();
//	cout <<"Number of leafs after enrichment: " << nleaf << endl;
};


template <class T> void TREE<T>::RemoveChildren(NODE<T> *node){
	RemoveNodes(node->left);	
	RemoveNodes(node->right);
	node->dir =0;
	nleaf--;
};

template <class T> void TREE<T>::SetPairLeaf(NODE<T> *node, list<PairLeaf<T> > &pairs){
	if(node->left != NULL){
		SetPairLeaf(node->left, pairs);
		SetPairLeaf(node->right, pairs);
	}else{
		if(node->parent->left==node){ //this is a left pair
			if(node->parent->right->left == NULL){ //sister is also a leaf
				PairLeaf<T> a_pair; a_pair.left = node; a_pair.right = node->parent->right->left;
				pairs.push_back(a_pair);
			}
		}
	}
};


//CONSTRUCT A TREE GIVEN A LIST OF CENTROIDS:

template <class T> void TREE<T>::Part_Node_Centroid(NODE<T> *node, int nc, int lis[], MatrixXd const &Xc){
	if(nc == 1){  						/* the node is actually a leaf as it contains a single centroid */
		return;
	}
	/* Otherwise : Decide of a valid direction to split the node: */
	int flag=0, mm=0, mp=0;
	int ml[nc], pl[nc];
	for(int id=0; id<ndim; id++){
		double xx = (node->binf(id)+node->bsup(id))*.5;
		flag = 0; mm=0; mp=0;
		for(int i=0; i<nc; i++){
			if(Xc(id,lis[i]) == xx){
				flag = -1;							//This is not a valid direction to break
				break;
			}else if(Xc(id,lis[i]) < xx ){
				ml[mm] = lis[i];
				mm++;
			}else{
				pl[mp] = lis[i];
				mp++;
			}
		}
		if(flag==0){ //id is a valid direction
			flag = id;
			break;
		}
	}
	
	if(flag==-1){
		cout << "Found no possible splitting directions in Part_Node_Centroid !" << endl;
		exit(1);
	} 
	AddChildren(node, flag+1); /* split the node in the flag-th direction */
	Part_Node_Centroid(node->left, mm, ml, Xc);
	Part_Node_Centroid(node->right, mp, pl, Xc);
};

