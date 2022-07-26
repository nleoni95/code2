template <class T>
void TREE<T>::OptimTree(TREE<T> &Orig){
	cout << "Optimize the tree partition\n";
	NODE<T>* leafs[Orig.Leafs()];
	Orig.ListLeafs(leafs);
	if(root->left!=NULL){
		RemoveNodes(root->left);
		RemoveNodes(root->right);
	}
	root->sol = Orig.root->sol;
	OptimNode(root,Orig,leafs);
	RestrictTree();
	SetNodeIndex();
//	SetLeafs();
};

template <class T>
int TREE<T>::Intersect(NODE<T> *n1, NODE<T>* n2){
	for(int d=0; d<ndim; d++){
		if(n1->Binf(d) >= n2->Bsup(d) || n1->Bsup(d) <= n2->Binf(d)) return 0;
	}
	return 1;
};

template <class T>
void TREE<T>::SetLocSol(NODE<T> *n1, VectorXd &inf, VectorXd &sup, VectorXd &sol){
//	cout << "set lo\n";
	double eps = pow(2.,-15);
	for(int d=0; d<ndim; d++){
//		printf("Traget in [%6.5f - %6.5f]\n",n1->Binf(d),n1->Bsup(d));
//		printf("Start at [%6.5f - %6.5f]\n",inf(d),sup(d));
		VectorXd sl,sr;
		while( sup(d) > n1->Bsup(d) || inf(d) < n1->Binf(d) ){
			if( (inf(d)+sup(d))*.5 - eps < n1->Binf(d)){
//			cout <<"Make a right\n";
				mra->Predict(sol,sl,sr,d); sol = sr; inf(d) = (inf(d)+sup(d))*.5;
			}else{
//			cout <<"Make a left\n";
				mra->Predict(sol,sl,sr,d); sol = sl; sup(d) = (inf(d)+sup(d))*.5;
			}
//			printf("Reset to [%6.5f - %6.5f]\n",inf(d),sup(d));
			if(sup(d)-inf(d)<1.e-6) exit(1);
		}
	}			//So now,Si is the local expansion over the intersection support.
//	cout << "Done\n";
};

template <class T>
void TREE<T>::SetGloSol(NODE<T> *n1, VectorXd &binf, VectorXd &bsup, VectorXd &sol){
//	cout << "set glo\n";
	for(int d=0; d<ndim; d++){
		double ds = bsup(d) - binf(d);
		VectorXd sl, sr;
		while( ds < (n1->Bsup(d)-n1->Binf(d)) ){
			int ic = binf(d)/ds;
			if(ic%2==0){ //this is a left child
				sl = sol; sr = sl*0;
				sol = mra->Restrict(sl,sr,d);
				bsup(d) += ds; ds *=2;
			}else{
				sr = sol; sl = sr*0;
				sol = mra->Restrict(sl,sr,d);
				binf(d) -= ds; ds *=2;
			}
		}
	}			//So now,Si is the local expansion over the intersection support.
};

template <class T>
int TREE<T>::OptimLR(const int id, NODE<T>* node, TREE<T> &Orig, NODE<T>* leafs[], VectorXd &soll, VectorXd &solr){
//	cout << "\tDetails of a child\n";
	AddChildren(node,id+1);
	node->left->sol  = VectorXd::Zero(mra->Npol());
	node->right->sol = VectorXd::Zero(mra->Npol());
	soll = VectorXd(mra->Npol());
	solr = VectorXd(mra->Npol());
	int nl=0, nr=0;
	for(int il=0; il<Orig.Leafs(); il++){
		int flag=Intersect(leafs[il],node->left);
		if(flag==1){								//leaf is intersecting the child
			VectorXd binf = leafs[il]->Binf();
			VectorXd bsup = leafs[il]->Bsup();
//			for(int d=0; d<ndim; d++) printf("[%4.3f %4.3f] [%4.3f %4.3f]\n",node->left->Binf(d),node->left->Bsup(d),binf(d),bsup(d)); cout <<"\n";
			SetLocSol(node->left, binf, bsup, soll);
			SetGloSol(node->left, binf, bsup, soll);
			node->left->sol += soll;
			nl++;
		}
		flag=Intersect(leafs[il],node->right);
		if(flag==1){								//leaf is intersecting the child
			VectorXd binf = leafs[il]->Binf();
			VectorXd bsup = leafs[il]->Bsup();
			SetLocSol(node->right, binf, bsup, solr);
			SetGloSol(node->right, binf, bsup, solr);
			node->right->sol+= solr;
			nr++;
		}
	}
	soll = node->left->sol;
	solr = node->right->sol;
	RemoveNodes(node->left);
	RemoveNodes(node->right);
//	cout <<"\t# of left and right leafs :" << nl << " - " << nr << endl;
	return max(nl,nr);
};

template <class T>
void TREE<T>::OptimNode(NODE<T> *node, TREE<T> &Orig, NODE<T>* leafs[]){
	cout << "Optimizing a node :\n" << (node->Level()) << "\n";
	VectorXd nsub = VectorXd::Zero(ndim);
	VectorXd soll, solr, sp, det;
	for(int id=0; id<ndim; id++){
		OptimLR(id, node, Orig, leafs, soll, solr);
		mra->Restrict(soll,solr,sp,det,id);
		nsub(id) = det.norm();
	}
	int d; nsub.maxCoeff(&d);
	node->sol = sp;
	if(nsub(d)<1.e-12){
		node->dir=0; node->det = node->sol*0;
		cout << "No details " <<endl;  return;
		RemoveNodes(node->left); RemoveNodes(node->right);
	}

	int nl = OptimLR(d, node, Orig, leafs, soll, solr);
	node->sol = mra->Restrict(soll,solr,d);
	cout <<"Node solution is "<< node->sol.norm() << endl;
	if(nl==1){
		node->dir=0; node->det = node->sol*0;
		cout << "No CHILDREN " <<endl;  return;
		RemoveNodes(node->left); RemoveNodes(node->right);
		return;
	}
	AddChildren(node,d+1);
	node->left->sol = soll;
	node->right->sol = solr;
	OptimNode(node->left, Orig, leafs);
	OptimNode(node->right, Orig, leafs);
	mra->Restrict(node->left->sol,node->right->sol,node->sol,node->det,d);
	cout << "Norm of details : " << node->det.norm() << endl;
};

template<class T> void TREE<T>::SynthTree()
{
	SynthNode(root);
};
template<class T> void TREE<T>::SynthNode(NODE<T>* node)
{
	if(node->dir!=0){
		VectorXd Xl,Xr; mra->Predict(node->sol,Xl,Xr,node->dir-1);
		node->left->sol  += Xl;
		node->right->sol += Xr;
		SynthNode(node->left);
		SynthNode(node->right);
		mra->Restrict(node->left->sol,node->right->sol,node->sol,node->det,node->dir-1);
	}
};
