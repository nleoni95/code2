//création d'un DoE par LHS

#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <math.h>
#include <random>
#include <time.h>
#include <functional>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

//script que j'utilise pour tirer des éléments d'une suite de Halton.

const int npts=2040; // nombre de points tirés
const int ndim=5; //dimension du plan d'expériences

VectorXd Randpert(int n,default_random_engine &generator){
	//renvoie une permutation aléatoire de (1:n). algo sympa
	VectorXd result(n);
	std::uniform_real_distribution<double> distU(0,1);
    for (int i=0;i<n;i++){
      result(i)=i;
    }
    for (int i=n-1;i>0;i--){
      int j=int(floor(distU(generator)*(i+1)));
      double a=result(i);
      result(i)=result(j);
      result(j)=a;
    }
    return result;
};

int main (){
	VectorXd lowlims(ndim); //limites basses des paramètres
	VectorXd highlims(ndim); //limites hautes des paramètres
	double FluxNominal=128790.0;
	lowlims << 0.9*FluxNominal,0.5,0.5,0.5,0.5;
	highlims << 1.1*FluxNominal,2,2,2,2;

	uniform_real_distribution<double> distU(0,1);
	default_random_engine(generator);

	generator.seed(666);
	vector<VectorXd> permutations(ndim); //on stocke les permutations
	for(int i=0;i<ndim;i++){
		permutations[i]=Randpert(npts,generator);
	}
	MatrixXd M(npts,ndim); //stockage des élements de la suite
	for (int i=0;i<npts;i++){
		for(int j=0;j<ndim;j++){
			M(i,j)=lowlims(j)+(highlims(j)-lowlims(j))*(permutations[j](i)+distU(generator))/double(npts);
		}
	}

cout << "tirage de " << npts << " points de dimension " << ndim << " selon un LHS" << endl;

for (int ifile=0;ifile<17;ifile++){
	int istart=1+120*ifile;
	int iend=120+120*ifile;
	string name=to_string(istart)+"-"+to_string(iend);
	string filename="designs/design_lhs"+name+".dat";
	FILE* out = fopen(filename.c_str(),"w");
	fprintf(out, "#NAME: DoE%s\n",name.c_str());
	fprintf(out,"#TITLE: DoELHS\n");
	fprintf(out,"#COLUMN_NAMES: Numero | NomEssai | FluxExp | BK | COAL | NUCL | MT\n");
	fprintf(out,"\n");
	for (int i=istart;i<=iend;i++){
		fprintf(out,"%i Cas%i ",i,i);
		for (int j=0;j<ndim;j++){
			fprintf(out, "%e ",M(i-1,j));
		}
		fprintf(out,"\n");
		}
	fclose(out);
}
//aussi création du fichier global.
{
int istart=1;
int iend=2040;
string name=to_string(istart)+"-"+to_string(iend);
string filename="designs/design_lhs"+name+".dat";
	FILE* out = fopen(filename.c_str(),"w");
	fprintf(out, "#NAME: DoE%s\n",name.c_str());
	fprintf(out,"#TITLE: DoELHS\n");
	fprintf(out,"#COLUMN_NAMES: Numero | NomEssai | FluxExp | BK | COAL | NUCL | MT\n");
	fprintf(out,"\n");
	for (int i=istart;i<=iend;i++){
		fprintf(out,"%i Cas%i ",i,i);
		for (int j=0;j<ndim;j++){
			fprintf(out, "%e ",M(i-1,j));
		}
		fprintf(out,"\n");
		}
	fclose(out);
}
}
