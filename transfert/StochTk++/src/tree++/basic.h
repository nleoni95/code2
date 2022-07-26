#ifndef TREE_H_
#include "tree++.h"
#endif

#ifndef BTREE_H_
#define BTREE_H_
#endif

using namespace std;
/*
	SECTION RELATED TO TREE INITIALIZATION
*/


template <class T> TREE<T>::TREE(){
	root   	= NULL;
	ndim    = 0;
	nord 	= 0;
	npol  	= 0;
	nnode  	= 0;
	nleaf  	= 0;
	mra 	= NULL;
};

template <class T> TREE<T>::~TREE(){
	clear();
	if(root!=NULL){ delete root; root = NULL;}
	ndim   	= 0;
	nord 	= 0;
	npol	= 0;
	mra = NULL;
};

template <class T> void TREE<T>::clear(){
	if(root==NULL){ return;}
	RemoveNodes(root);
	nnode 	= 0;
	nleaf 	= 0;
};

template <class T> TREE<T>::TREE(TREE<T> &S){
	root   	= NULL;
	mra 	= S.Mra();
	ndim    = mra->Ndim();
	nord    = mra->Nord();
	npol    = mra->Npol();
	nnode  = 0;
	nleaf  = 0;
	IniRoot();
	CopyNodes(S.root,root);
	SetLeafs(root);
};

template <class T> void TREE<T>::IniTree(MRA *basis){
	mra = basis;
	ndim = mra->Ndim();
	nord = mra->Nord();
	npol = mra->Npol();
	IniRoot();
	IniTree(root);
	SetLeafs();
};

template <class T> void TREE<T>::IniTree(NODE<T> *node){
	if(node==NULL) return;
	for(int id=0; id<ndim; id++){
		if(node->ncut[id]<cut_init){ AddChildren(node,id+1); break;}
	}
	IniTree(node->left);
	IniTree(node->right);
};

template<class T> void TREE<T>::IniTree(MRA *basis, int mdir[]){
	mra = basis;
	ndim = mra->Ndim();
	nord = mra->Nord();
	npol = mra->Npol();
	IniRoot();
	IniTree(root,mdir);
	SetLeafs();
};

template<class T> void TREE<T>::IniTree(NODE<T> *node, int mdir[]){
	if(node==NULL) return;
	for(int id=0; id<ndim; id++){
		if(node->ncut[id]<mdir[id]){ AddChildren(node,id+1); break;}
	}
	IniTree(node->left,mdir);
	IniTree(node->right,mdir);
};

template <class T> void TREE<T>::IniTree(MRA *basis, int mdir){
	mra = basis;
	ndim = mra->Ndim();
	nord = mra->Nord();
	npol = mra->Npol();
	cout << "Do Root" << endl;
	IniRoot();
	cout << "Do Tree" << endl;
	IniTree(root,mdir);
	SetLeafs();
};

template <class T> void TREE<T>::IniTree(NODE<T> *node, int mdir){
	if(node==NULL) return;
	for(int id=0; id<ndim; id++){
		if(node->ncut[id]<mdir){ AddChildren(node,id+1); break;}
	}
	IniTree(node->left,mdir);
	IniTree(node->right,mdir);
};

template <class T> void TREE<T>::IniRoot(){
	if(root!=NULL){clear();}
	nnode =1;
	nleaf =0;
	NODE<T> *t = new NODE<T>;
	t->parent= NULL;
	t->left  = NULL;
	t->right = NULL;
	t->index = nnode;
	t->dir = 0;
	t->binf = VectorXd::Zero(ndim);
	t->bsup = VectorXd::Zero(ndim);
	t->ncut = VectorXi::Zero(ndim);
	for(int i=0; i<ndim; i++){
		t->binf[i]=0.;
		t->bsup[i]=1.;
		t->ncut[i]=0;
	}
	root = t;
	return;
};

template <class T> void TREE<T>::RandomTree(int acc){
	NODE<T> *node = root;
	int cas;
	if(node->left!=NULL) return;
	cas = rand()%(100);
	if(nnode<=10) cas=0;
	if(cas<=acc){
		int dir = rand()%ndim +1;
		if(node->ncut[dir-1]<cut_max){
			AddChildren(node,dir);
			RandomTree(node->left, acc);
			RandomTree(node->right, acc);
		}
	}
	SetLeafs();
};

template <class T> void TREE<T>::RandomTree(NODE<T> *node, int acc){
	int cas;
	if(node->left!=NULL){
		return;
	}else{
		cas = rand()%(100);
		if(nnode<=10) cas=0;
		if(cas<=acc){
			int dir = rand()%ndim +1;
			if(node->ncut[dir-1]<cut_max){AddChildren(node,dir); RandomTree(node->left, acc); RandomTree(node->right, acc);}
		}
	}
};

template <class T>
void TREE<T>::IniTree(MRA *basis, const MatrixXd &Xobs, int n_per_dom){
	mra = basis;
	ndim = mra->Ndim();
	nord = mra->Nord();
	npol = mra->Npol();
	IniRoot();
	list<int> liste;
	for(int i=0; i<Xobs.cols(); i++) liste.push_back(i);
	SplitPart(root, liste, Xobs, n_per_dom);
		/*
		MatrixXd Obs = Xobs;
		SplitPart(root,Obs, n_per_dom);
		*/
	SetLeafs();
};

template <class T>
void TREE<T>::IniTree(MRA *basis, const MatrixXd &Xobs, const VectorXd &Dat, int n_per_dom){
	mra = basis;
	ndim = mra->Ndim();
	nord = mra->Nord();
	npol = mra->Npol();
	IniRoot();
	list<int> liste;
	for(int i=0; i<Xobs.cols(); i++) liste.push_back(i);
	SplitPart(root, liste, Xobs, Dat, n_per_dom);
	SetLeafs();
};
template <class T> void TREE<T>::SplitPart(NODE<T> *node, const MatrixXd &Obs, int n_per_dom){
// Analyse possible splitting directions:
	if (Obs.cols() <= 1 ) return;
	VectorXd indicator = VectorXd::Zero(ndim);
	VectorXd xm = (node->binf+node->bsup)*.5;
	for(int i=0; i<Obs.cols(); i++) indicator.array() += (Obs.col(i).array()-xm.array()).square();
	indicator = indicator.array().abs();
	double maxd = indicator.maxCoeff();
	while( maxd > 0 ){
		int id = 0;
		while( indicator(id) < maxd) {id++;}
		double xpart = xm(id);
		int ninf = 0;
		for(int i=0; i<Obs.cols(); i++) if(Obs(id,i) < xpart){ ninf++;}

		if(ninf >= n_per_dom && ninf <= Obs.cols()-n_per_dom ){
			MatrixXd Oinf = MatrixXd(ndim,ninf);
			MatrixXd Osup = MatrixXd(ndim,Obs.cols()-ninf);
			ninf = 0;
			int nsup = 0;
			for(int i=0; i<Obs.cols(); i++){
				if(Obs(id,i) < xpart)
					{ Oinf.col(ninf) = Obs.col(i); ninf++;}
				else
					{ Osup.col(nsup) = Obs.col(i); nsup++;}
			}
	// Create the two children:
			id++;
			AddChildren(node,id);
			SplitPart(node->left, Oinf, n_per_dom);
			SplitPart(node->right,Osup, n_per_dom);
			return;
		}else{
			indicator(id) = 0;
			maxd = indicator.maxCoeff();
		}
	}
};

template <class T> void TREE<T>::SplitPart(NODE<T> *node, list<int> liste, const MatrixXd &Obs, int n_per_dom){
// Analyse possible splitting directions:

	if(int(liste.size())<=n_per_dom*2) return;
	list<int>::iterator it;
	VectorXd Xm = (node->Binf()+node->Bsup())*.5;
	VectorXi bal= VectorXi::Zero(ndim);
	for(it=liste.begin(); it!=liste.end(); it++)
		for(int id=0; id<ndim; id++) if(Obs(id,(*it)) <= Xm(id)) bal(id)++;
	int lm = liste.size()/2;
	bal -= VectorXi::Constant(ndim,lm);
	bal.array() = bal.array().abs();
	bal += VectorXi::Ones(ndim);
	for(int id=0; id<ndim; id++) bal(id)+= (node->Level()+1-node->Ncut(id));
	int id;
	double bmx = bal.maxCoeff();
	for(int id=0; id<ndim; id++) if(bal(id)<bmx) bal(id)=0;
	int nb = rand()%bal.sum()+1;
	id=0; int ct=bal(id);
//	cout << "Ncut " << node->Ncut().transpose() << " bal " << bal.transpose() << endl;
	while(ct<nb){id++; ct+=bal(id);};
//	cout << "nbd at splitpart "<< bal.transpose()<< " Direction d="<< id << " nb " << nb << endl;
	list<int> left;
	list<int> right;
	for(it=liste.begin(); it!=liste.end(); it++)
		if(Obs(id,(*it)) <= Xm(id)){
			left.push_back((*it));
		}else{
			right.push_back((*it));
		}
	id++;
	AddChildren(node,id);
	SplitPart(node->left, left,  Obs, n_per_dom);
	SplitPart(node->right,right, Obs, n_per_dom);
};

template <class T> void TREE<T>::SplitPart(NODE<T> *node, list<int> liste, const MatrixXd &Obs, const VectorXd &Dat, int n_per_dom){
// Analyse possible splitting directions:

	if(int(liste.size())<n_per_dom*2) return;	//splitting would necessarily result in a node with to few observations
	list<int>::iterator it;
	VectorXd Xm = (node->Binf()+node->Bsup())*.5;
	VectorXd Xl = VectorXd::Zero(ndim);
	VectorXd Xr = VectorXd::Zero(ndim);
	VectorXi Nl = VectorXi::Zero(ndim);
	for(it=liste.begin(); it!=liste.end(); it++)
		for(int d=0; d<ndim; d++) if(Obs(d,(*it)) <= Xm(d)){ Xl(d) += Dat((*it)); Nl(d)++;}else{ Xr(d) += Dat((*it));}
	for(int d=0; d<ndim; d++){
		if(Nl(d)<n_per_dom || liste.size()-Nl(d)<n_per_dom){
			Xl(d) =0;
		}else{
			Xl(d) = fabs(Xl(d)/Nl(d) - Xr(d)/(liste.size()-Nl(d)));
		}
	}
	double bmx = Xl.maxCoeff();
	VectorXi Bal = VectorXi::Zero(ndim);
	for(int id=0; id<ndim; id++) if(Xl(id)=bmx) Bal(id)++;
	int id=0; int nb = rand()%Bal.sum()+1;
	int ct=Bal(id);
	while(ct<nb){id++; ct+=Bal(id);};
	list<int> left;
	list<int> right;
	for(it=liste.begin(); it!=liste.end(); it++)
		if(Obs(id,(*it)) <= Xm(id)){
			left.push_back((*it));
		}else{
			right.push_back((*it));
		}
	id++;
	AddChildren(node,id);
	SplitPart(node->left, left,  Obs, Dat, n_per_dom);
	SplitPart(node->right,right, Obs, Dat, n_per_dom);
};

template <class T> void TREE<T>::CopyTree(TREE<T> &S){
	mra = S.MRA();
	ndim = mra->Ndim();
	nord = mra->Nord();
	npol = mra->Npol();
	IniRoot();
	CopyNodes(S.root,root);
	SetLeafs(root);
};

template <class T> void TREE<T>::SetSupports(){
	NODE<T> *node = root;
	if(node==NULL) return;
	if(node->parent!=NULL){
		node->binf = node->parent->binf;
		node->bsup = node->parent->bsup;
		int id = node->parent->dir - 1;
		if(node->parent->left==node){
			node->bsup[id] = (node->binf[id]+node->bsup[id])*.5;
		}else{
			node->binf[id] = (node->binf[id]+node->bsup[id])*.5;
		}
	}else{
		node->binf = VectorXd::Zero(ndim);
		node->bsup = VectorXd::Ones(ndim);
	}
	SetSupports(node->left);
	SetSupports(node->right);
};

template <class T> void TREE<T>::SetSupports(NODE<T> *node)
{
	if(node==NULL) return;
	if(node->parent!=NULL){
		node->binf = node->parent->binf;
		node->bsup = node->parent->bsup;
		int id = node->parent->dir - 1;
		if(node->parent->left==node){
			node->bsup[id] = (node->binf[id]+node->bsup[id])*.5;
		}else{
			node->binf[id] = (node->binf[id]+node->bsup[id])*.5;
		}
	}else{
		node->binf = VectorXd::Zero(ndim);
		node->bsup = VectorXd::Ones(ndim);
	}
	SetSupports(node->left);
	SetSupports(node->right);
};

/*
	SECTION RELATED TO LEAFS /
*/

template <class T> void TREE<T>::SetLeafs(){
	//Construct the set of leafs of a Tree
	nleaf = 0;
	SetLeafs(root); int index =0;
	SetLeafIndex(root,&index);
};

template <class T> void TREE<T>::SetLeafs(NODE<T> *node){
	if(node->left==NULL){ 					//node is actually a leaf:
		node->lindex = nleaf; nleaf ++;		//increment the number of leafs
	}else{									//the node has descendent:
		SetLeafs(node->left);
		SetLeafs(node->right);
	}
};

template <class T> void TREE<T>::ListLeafs(NODE<T> *pleaf[]){
	int nl=0;
	ListLeafs(root, nl, pleaf);
};

template <class T> void TREE<T>::ListLeafs(NODE<T> *node, int &nl, NODE<T> *pleaf[]){
	if(node->dir==0){					//node is a leaf:
		pleaf[nl] = node; node->lindex = nl ++;
	}else{								//node has descendent:
		ListLeafs(node->left, nl, pleaf);
		ListLeafs(node->right, nl, pleaf);
	}
};

template <class T> void TREE<T>::SetLeafIndex(){
	int index =0; SetLeafIndex(root,&index);
};

template <class T> void TREE<T>::SetLeafIndex(NODE<T> *node,int *index){
  if(node->dir==0){
    node->lindex = *index; (*index) ++; return;
  }
  SetLeafIndex(node->left,index);
  SetLeafIndex(node->right,index);
};

template <class T> int TREE<T>::FindLeafIndex(VectorXd const &xp){
	NODE<T> *node = root;
	if(node->dir==0) return node->lindex;
	int id = node->dir-1;
	if(xp[id]<(node->binf(id)+node->bsup(id))*.5){
		return FindNode(node->left,xp)->lindex;
	}else{
		return FindNode(node->right,xp)->lindex;
	}
};

/*
	SECTION RELATED TO NODES
*/
template <class T> void TREE<T>::AddChildren(NODE<T> *node, int idir)
{
	if(node->left==NULL){
	//create the left child
		node->dir = idir;
		NODE<T> *tl = new NODE<T>;
		nnode++;
		tl->dir = 0;
		tl->parent = node;
		tl->left  = NULL;
		tl->right = NULL;
		tl->index = nnode;
		tl->binf= node->binf;
		tl->bsup= node->bsup;
		tl->ncut= node->ncut;
		tl->ncut[idir-1] ++;
		tl->bsup[idir-1] = (tl->binf[idir-1]+tl->bsup[idir-1])*.5;
		node->left = tl;
	}
	if(node->right==NULL){
		NODE<T> *tr = new NODE<T>;
		nnode++;
		tr->dir=0;
		tr->parent = node;
		tr->left  = NULL;
		tr->right = NULL;
		tr->index = nnode;
		tr->binf= node->binf;
		tr->bsup= node->bsup;
		tr->ncut = node->ncut;
		tr->ncut[idir-1] ++;
		tr->binf[idir-1] = (tr->binf[idir-1]+tr->bsup[idir-1])*.5;
		node->right = tr;
	}
};

template <class T> void TREE<T>::RemoveNodes(NODE<T> *node){
	if(node == NULL) return;
	RemoveNodes(node->left);
	RemoveNodes(node->right);
	if(node->parent != NULL){
		if(node->parent->left==node){
			node->parent->left = NULL;
		}else{
			node->parent->right = NULL;
		}
		nnode = nnode-1;
		delete node;
		node = NULL;
	}else{ //This is the root node:
		nnode = nnode-1;
		delete node;
		node = NULL;
		root = NULL;
	}
};

template <class T> void TREE<T>::CopyNodes(NODE<T> *t, NODE<T> *s)
{
	s->dir    = t->dir;
	s->index  = t->index;
	s->lindex = t->lindex;
	s->ncut   = t->ncut;
	s->binf   = t->binf;
	s->bsup   = t->bsup;
	if(t->left!=NULL){
		nnode ++;
		NODE<T> *nl = new NODE<T>;
		nl->parent = s;
		nl->left   = NULL;
		nl->right  = NULL;
		s->left    = nl;
		CopyNodes(t->left,s->left);
		nnode ++;
		NODE<T> *nr = new NODE<T>;
		nr->parent = s;
		nr->left   = NULL;
		nr->right  = NULL;
		s->right   = nr;
		CopyNodes(t->right,s->right);
	}else{
		s->left  = NULL;
		s->right = NULL;
	}
};

template <class T> void TREE<T>::ListNodes(NODE<T> *pnode[]){
	int nn=0;
	ListNodes(root, nn, pnode);
};

template <class T> void TREE<T>::ListINodes(NODE<T> *pnode[]){
	int nn=0;
	ListINodes(root, nn, pnode);
};

template <class T> void TREE<T>::ListNodes(NODE<T> *node, int &nn, NODE<T> *pnode[]){
	pnode[nn] = node; nn++;
	if(node->left!=NULL) ListNodes(node->left, nn, pnode);
	if(node->right!=NULL) ListNodes(node->right, nn, pnode);
};

template <class T> void TREE<T>::ListINodes(NODE<T> *node, int &nn, NODE<T> *pnode[]){
	if(node->left!=NULL){
		pnode[nn] = node; nn++;
		ListINodes(node->left, nn, pnode);
		ListINodes(node->right, nn, pnode);
	}
};


template <class T> void TREE<T>::SetNodeIndex(){
	int index = 0;
	SetNodeIndex(root,&index);
	return;
};

template <class T> void TREE<T>::SetNodeIndex(NODE<T> *node,int *index){
	if(node==NULL) return;
	node->index = *index;
	(*index)++;
	SetNodeIndex(node->left,index);
	SetNodeIndex(node->right,index);
	return;
};

template <class T> NODE<T>* TREE<T>::FindNode(VectorXd const &xp){
	return FindNode(root,xp);
};

template <class T> NODE<T>* TREE<T>::FindNode(NODE<T> *node, VectorXd const &xp){
	if(node->dir==0) return node;
	int id = node->dir-1;
	if(xp(id)<(node->binf(id)+node->bsup(id))*.5) return FindNode(node->left,xp);
	return FindNode(node->right,xp);
};

template <class T> void TREE<T>::PlotLeafs(string name){
	char *fileName = (char*)name.c_str();
	FILE *output =fopen(fileName, "w");
	PlotLeaf(root,output);
	fclose(output);
	return;
};

template <class T> void TREE<T>::PlotLeaf(NODE<T> *node, FILE *output){
	if(node->dir!=0){
		PlotLeaf(node->left, output);
		PlotLeaf(node->right, output);
	}else{
		if(ndim==2){
			PlotLeaf2D(node, output);
		}else if(ndim==3){
			PlotLeaf3D(node, output);
		}else{
			for(int id = 0 ; id < ndim; id++){
				fprintf(output," %e ",node->binf[id]*0.5+node->bsup[id]*.5);
				fprintf(output,"\n");
			}
		}
	}
	return;
};

template <class T> void TREE<T>::PlotLeaf2D(NODE<T> *node, FILE *output)
{
	fprintf(output,"%e %e 0\n",node->binf[0],node->binf[1]);
	fprintf(output,"%e %e 0\n",node->bsup[0],node->binf[1]);
	fprintf(output,"\n\n");
	fprintf(output,"%e %e 0\n",node->bsup[0],node->binf[1]);
	fprintf(output,"%e %e 0\n",node->bsup[0],node->bsup[1]);
	fprintf(output,"\n\n");
	fprintf(output,"%e %e 0\n",node->bsup[0],node->bsup[1]);
	fprintf(output,"%e %e 0\n",node->binf[0],node->bsup[1]);
	fprintf(output,"\n\n");
	fprintf(output,"%e %e 0\n",node->binf[0],node->bsup[1]);
	fprintf(output,"%e %e 0\n",node->binf[0],node->binf[1]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	return;
};

template <class T>	void TREE<T>::PlotLeaf3D(NODE<T> *node, FILE *output)
{
	fprintf(output,"%e %e %e \n",node->binf[0],node->binf[1],node->binf[2]);
	fprintf(output,"%e %e %e \n",node->bsup[0],node->binf[1],node->binf[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n",node->binf[0],node->binf[1],node->binf[2]);
	fprintf(output,"%e %e %e \n",node->binf[0],node->bsup[1],node->binf[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n",node->bsup[0],node->binf[1],node->binf[2]);
	fprintf(output,"%e %e %e \n",node->bsup[0],node->bsup[1],node->binf[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n",node->bsup[0],node->bsup[1],node->binf[2]);
	fprintf(output,"%e %e %e \n",node->binf[0],node->bsup[1],node->binf[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n", node->binf[0],node->binf[1],node->bsup[2]);
	fprintf(output,"%e %e %e \n",node->bsup[0],node->binf[1],node->bsup[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n",node->binf[0],node->binf[1],node->bsup[2]);
	fprintf(output,"%e %e %e \n",node->binf[0],node->bsup[1],node->bsup[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n",node->bsup[0],node->binf[1],node->bsup[2]);
	fprintf(output,"%e %e %e \n",node->bsup[0],node->bsup[1],node->bsup[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n",node->bsup[0],node->bsup[1],node->bsup[2]);
	fprintf(output,"%e %e %e \n",node->binf[0],node->bsup[1],node->bsup[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n", node->binf[0],node->binf[1],node->binf[2]);
	fprintf(output,"%e %e %e \n",node->binf[0],node->binf[1],node->bsup[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n",node->binf[0],node->bsup[1],node->binf[2]);
	fprintf(output,"%e %e %e \n",node->binf[0],node->bsup[1],node->bsup[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n",node->bsup[0],node->bsup[1],node->binf[2]);
	fprintf(output,"%e %e %e \n",node->bsup[0],node->bsup[1],node->bsup[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	fprintf(output,"%e %e %e \n",node->bsup[0],node->binf[1],node->binf[2]);
	fprintf(output,"%e %e %e \n",node->bsup[0],node->binf[1],node->bsup[2]);
	fprintf(output,"\n");
	fprintf(output,"\n");
	return;
};

template <class T> void TREE<T>::PlotTree(char *name)
{
	FILE *output;
	SetNodeIndex();
	if((output=fopen(name,"w"))!=NULL){
		fprintf(output,"digraph G {\n");
		PlotNode(root,output);
		fprintf(output,"}\n");
		if(fclose(output)==EOF){
			cout << "File was not closed " << endl;
		}
	}else{
		cout << "file " << name << " couldn't be opened " << endl;
	}
};

template <class T> void TREE<T>::PlotNode(NODE<T> *node, FILE *output){
	if(node->det.Norm() >=1.e-3){
		fprintf(output,"%6d [shape=box,style=filled,fillcolor=red];\n",node->index);
	}else if (node->det.Norm() >=1.e-6){
		fprintf(output,"%6d [shape=box,style=filled,fillcolor=gold];\n",node->index);
	}else if(node->det.Norm() >0){
		fprintf(output,"%6d [shape=ellipse,style=filled,fillcolor=chartreuse];\n",node->index);
	}else{
		fprintf(output,"%6d [shape=invtriangle,style=filled,fillcolor=chartreuse];\n",node->index);
	}
	if(node->left!=NULL){
		fprintf(output," %6d -> %6d [label=\"%2d\"];\n",node->index, node->left->index, node->dir);
		PlotNode(node->left, output);
	}
	if(node->right!=NULL){
		fprintf(output," %6d -> %6d [label=\"%2d\"];\n",node->index, node->right->index, node->dir);
		PlotNode(node->right, output);
	}
	return;
};

template <class T> void TREE<T>::ListTree(){ ListTree(root); };

template <class T> void TREE<T>::ListTree(NODE<T> *node){
  if(node==NULL) return;
  cout << "Domain :"; for(int id=0; id< ndim; id++) printf("[%6.4f,%6.4f]",node->binf[id],node->bsup[id]);
  cout << "\n\tChildren in direction " << node->dir << endl;
  ListTree(node->left);
  ListTree(node->right);
};



