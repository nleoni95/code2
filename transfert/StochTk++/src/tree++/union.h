template <class T> 
void TREE<T>::UnionTrees(TREE<T> &S1, TREE<T> &S2){
	IniRoot(S1.Mra());
	mra = S1.Mra();
	CopyNodes(S1.root,root);
	NODE<T> *Tleafs[S2.nleaf];
	S2.ListLeafs(Tleafs);
	for(int il=0; il<S2.nleaf; il++) UnionTrees(Tleafs[il]);
	SetLeafs(root);
};

template <class T> void TREE<T>::UnionTrees(NODE<T> *node){
	ExtUnion2(root,node->binf, node->bsup);
};

template <class T> void TREE<T>::ExtUnion2(NODE<T> *node_u, VectorXd const &binf, VectorXd const &bsup){
	if( is_intersecting(node_u, binf, bsup) == 0) return;
	if(node_u->dir==0){  //the node is intersecting the leaf and has no child:
		VectorXd ninf(ndim);
		VectorXd nsup(ndim);
		for(int id=0; id<ndim; id++){
			ninf(id) = max(node_u->binf(id),binf(id));
			nsup(id) = min(node_u->bsup(id),bsup(id));
		}
		VectorXd nc = (node_u->binf+node_u->bsup)*.5;
		VectorXd ic = (ninf+nsup)*.5;
		ic -= nc;
		if( ic.norm() == 0 ) return; //node_u is actually contained in node.
		int dir = 0;
		for(int id=1; id<ndim; id++) if( fabs(ic(id)) > fabs(ic(dir)) ) dir = id;
		//splitting node_u in direction dir:
		node_u->dir = dir+1;
		NODE<T> *nl = new NODE<T>;
		nl->dir = 0;
		nl->parent = node_u;
		nnode++;
		nl->index = nnode;
		nl->left = NULL;
		nl->right = NULL;
		nl->ncut = node_u->ncut;
		nl->binf = node_u->binf;
		nl->bsup = node_u->bsup;
		nl->ncut[dir]++;
		nl->bsup[dir] = (nl->bsup[dir]+nl->binf[dir])*.5;
		node_u->left = nl;
		NODE<T> *nr = new NODE<T>;
		nr->dir = 0;
		nr->parent = node_u;
		nnode++;
		nr->index = nnode;
		nr->left = NULL;
		nr->right= NULL;
		nr->ncut = node_u->ncut;
		nr->binf = node_u->binf;
		nr->bsup = node_u->bsup;
		nr->ncut[dir]++;
		nr->binf[dir] = (nr->bsup[dir]+nr->binf[dir])*.5;
		node_u->right = nr;

		ExtUnion2(node_u->left, binf, bsup);
		ExtUnion2(node_u->right,binf, bsup);
	}else{
		ExtUnion2(node_u->left, binf, bsup);
		ExtUnion2(node_u->right,binf, bsup);
	}
};

template <class T> void TREE<T>::ExtUnion(NODE<T> *tu, NODE<T> *t2, VectorXd const &xm, int idir){
	if(tu->binf[idir-1]>=xm(idir-1)) return;
	if(tu->bsup[idir-1]<=xm(idir-1)) return;
	for(int id=0; id<ndim; id++){
		if(id!=idir-1){
			if(tu->binf[id]>=t2->bsup[id]) return;
			if(tu->bsup[id]<=t2->binf[id]) return;
		}
	}

	if(tu->dir==0){
	//create two children
		tu->dir = idir;
		NODE<T> *nl = new NODE<T>;
		nl->dir = 0;
		nl->parent = tu;
		nnode++;
		nl->index = nnode;
		nl->left = NULL;
		nl->right = NULL;
		nl->ncut = tu->ncut;
		nl->binf = tu->binf;
		nl->bsup = tu->bsup;
		nl->ncut[idir-1]++;
		nl->bsup[idir-1] = (nl->bsup[idir-1]+nl->binf[idir-1])*.5;
		tu->left = nl;
		NODE<T> *nr = new NODE<T>;
		nr->dir = 0;
		nr->parent = tu;
		nnode++;
		nr->index = nnode;
		nr->left = NULL;
		nr->right= NULL;
		nr->ncut = tu->ncut;
		nr->binf = tu->binf;
		nr->bsup = tu->bsup;
		nr->ncut[idir-1]++;
		nr->binf[idir-1] = (nr->bsup[idir-1]+nr->binf[idir-1])*.5;
		tu->right = nr;
		return;
	}else{
	//seek for children
		ExtUnion(tu->left,t2,xm,idir);
		ExtUnion(tu->right,t2,xm,idir);
	}
};

template <class T> int TREE<T>::is_intersecting(NODE<T> *node, VectorXd const &Xinf, VectorXd const &Xsup){
	for(int jd=0; jd<ndim; jd++)
		if(node->binf(jd) >= Xsup(jd) || node->bsup(jd) <= Xinf(jd) ) return 0;
	return 1;
};

template <class T> void TREE<T>::ShakeTree(){
//	Set_Sol_Nodes();
	ShakeTree(root);
	return;
};

template <class T> void TREE<T>::ShakeTree(NODE<T> *node){
	if(node==NULL) return;
	if(node->dir==0) return;
	if(node->left->dir!=node->right->dir){
		ShakeTree(node->left);
		ShakeTree(node->right);
	}else{
		if(node->left->dir==0) return;
		int id1 = node->dir;
		int id2 = node->left->dir;
		if(id1!=id2 && node->bsup(id1-1)-node->binf(id1-1)<=.5){
			int choix = rand()%2;
			if (choix==0){
				node->dir        =id2;
				node->left->dir  =id1;
				node->right->dir =id1;
				node->left ->ncut[id1-1]--;
				node->left ->ncut[id2-1]++;
				node->right->ncut[id1-1]--;
				node->right->ncut[id2-1]++;
				node->left->binf = node->binf;
				node->left->bsup = node->bsup;
				node->right->binf = node->binf;
				node->right->bsup = node->bsup;
				node->left->bsup(id2-1)  = (node->binf(id2-1)+node->bsup(id2-1))*.5;
				node->right->binf(id2-1) = (node->binf(id2-1)+node->bsup(id2-1))*.5;
				node->left->right->parent = node->right;
				node->right->left->parent = node->left;
				NODE<T> *p = node->left->right;
				node->left->right = node->right->left;
				node->right->left = p;
			}
		}
		ShakeTree(node->left);
		ShakeTree(node->right);
	}
	return;
};

template <class T> void TREE<T>::ShakeTree2(){
	NODE<T> *pl[nleaf]; ListLeafs(pl);
	for(int il=0; il<nleaf; il++){
		int d_p = pl[il]->parent->dir;
		if(pl[il]->parent->parent != NULL){
			int d_pp = pl[il]->parent->parent->dir;
			if(d_p != d_pp){
				if(pl[il]->parent->parent->left->dir==pl[il]->parent->parent->right->dir){
					if(rand()%2 ==0){
						int id2 = d_p;
						int id1 = d_pp;
						NODE<T> *node = pl[il]->parent->parent;
						node->dir        =id2;
						node->left->dir  =id1;
						node->right->dir =id1;
						node->left ->ncut[id1-1]--;
						node->left ->ncut[id2-1]++;
						node->right->ncut[id1-1]--;
						node->right->ncut[id2-1]++;
						node->left->binf = node->binf;
						node->left->bsup = node->bsup;
						node->right->binf = node->binf;
						node->right->bsup = node->bsup;
						node->left->bsup(id2-1)  = (node->binf(id2-1)+node->bsup(id2-1))*.5;
						node->right->binf(id2-1) = (node->binf(id2-1)+node->bsup(id2-1))*.5;
						node->left->right->parent = node->right;
						node->right->left->parent = node->left;
						NODE<T> *p = node->left->right;
						node->left->right = node->right->left;
						node->right->left = p;
					}
				}
			}
		}
	}
	return;
};

