//création d'un DoE par QMC, suite de Halton

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <math.h>
#include <random>
#include <time.h>
#include <functional>
#include <Eigen/Dense>
#include "halton.cpp"
#include <typeinfo>

using namespace Eigen;

//script que j'utilise pour tirer des éléments d'une suite de Halton.

const int npts=2040; // nombre de points tirés
const int nburn=1; // indice du point de départ
const int ndim=5; //dimension du plan d'expériences
const int nend=nburn+npts-1; // indice du dernier point

int main (){
	VectorXd lowlims(ndim); //limites basses des paramètres
	VectorXd highlims(ndim); //limites hautes des paramètres
	double FluxNominal=128790.0;
	lowlims << 0.90*FluxNominal,0.5,0.5,0.5,0.5;
	highlims << 1.1*FluxNominal,2,2,2,2;


double *h=halton_sequence(nburn+1,nburn+npts,ndim); //tableau de taille ndim*(npts) ;ou a3*(a2-a1+1)



MatrixXd M(npts,ndim); //stockage des élements de la suite
for (int i=0;i<npts;i++){
	for(int j=0;j<ndim;j++){
		M(i,j)=lowlims(j)+(highlims(j)-lowlims(j))*h[ndim*i+j];
	}
}
cout << "tirage de " << npts << " points de dimension " << ndim << " selon une suite de Halton" << endl;
string name=to_string(nburn)+"-"+to_string(nend);
string filename="designs/design_qmc"+name+".dat";

FILE* out = fopen(filename.c_str(),"w");
fprintf(out, "#NAME: DoE %s\n",name.c_str());
fprintf(out,"#TITLE: DoE\n");
fprintf(out,"#COLUMN_NAMES: Numero | NomEssai | FluxExp | BK | COAL | NUCL | MT\n");
fprintf(out,"\n");
for (int i=0;i<npts;i++){
	fprintf(out,"%i Cas%i ",i+nburn,i+nburn);
	for (int j=0;j<ndim;j++){
		fprintf(out, "%e ",M(i,j));
	}
	fprintf(out,"\n");
	}
fclose(out);
}
