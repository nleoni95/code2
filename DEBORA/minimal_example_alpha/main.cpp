// Exemple minimal de calibration pour le taux de vide seul.


#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>
#include <nlopt.hpp>
#include <Eigen/Dense>
#include <random>
#include <functional>
#include <iterator>
#include <chrono>
#include "densities.h"
#include "pcb++.h"
#include "cub++.h"
#include "vstoch++.h"
#include "mstoch++.h"


using namespace std;
using namespace Eigen;

typedef map<int,VectorXd> map_doe; //key : int, value : vectorXd
typedef map<int,vector<VectorXd>> map_results; //key : int, value : vecteur de VectorXd
typedef map<string,VectorXd> map_exp; //contient les valeurs expérimentales

int neval=1;
std::default_random_engine generator;
std::uniform_real_distribution<double> distU(0,1);
std::normal_distribution<double> distN(0,1);

double const flux_nominal=128790;

int dim_theta=5;

int line_count(string const &filename){
  //renvoie le nombre de lignes dans un fichier
  ifstream ifile(filename);
  int nlines=0;
  if(ifile){
    ifile.unsetf(ios_base::skipws);
    nlines=count(istream_iterator<char>(ifile),istream_iterator<char>(),'\n');
  }
  return nlines;
}

map_doe read_doe(string const &filename){
  //lecture du DoE à filename et écriture dans la map
  map_doe m;
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      if (line[0]=='#'){continue;}
      if (line.empty()){continue;}
      //décomposition de la line en des mots
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>()); //on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      //traitement des mots. Le premier est le num du cas, les 3-4-5-6-7 sont les paramètres.
      VectorXd param(5);
      for (int i=2;i<7;i++){
        param(i-2)=stod(words[i]);
      }
      param(0)/=flux_nominal;
      int key=stoi(words[0]);
      m.insert(make_pair(key,param));
    }
  }
  else{ cerr << "DoE file doesn't exist" << endl;}
  cout << " Size of initial DoE: " << m.size() << " points."<< endl;

  return m;
}

vector<VectorXd> read_singleresult(string const &filename){
  //lit les résultats dans un fichier donné et les rend sous forme vector de VectorXd.
  //renvoie un vecteur vide si le fichier n'existe pas.
  //architecture des fichiers lus : 1ère ligne à ignorer.
  //colonnes : X, alpha, Dbul.
  vector<VectorXd> v(5);
  int nlines=line_count(filename)-1; //ignorer la 1ère ligne
  int current_line=0;
  VectorXd X(40);
  VectorXd Alpha(40);
  VectorXd D(40);
  VectorXd V1(40);
  VectorXd V2(40);
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      if (line[0]=='X'){continue;}
      //décomposition de la line en mots
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>()); //on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      //traitement des mots.
      X(current_line)=stod(words[0]);
      Alpha(current_line)=stod(words[1]);
      D(current_line)=stod(words[2]);
      V1(current_line)=stod(words[3]);
      V2(current_line)=stod(words[4]);
      current_line++;
    }
  }
  else {
    vector<VectorXd> vempty; return vempty;}
  v[0]=X;
  v[1]=Alpha;
  v[2]=D;
  v[3]=V1;
  v[4]=V2;
  return v;
}

map_results read_results_qmc(string const &filename){
  //lecture de tous les résultats de calcul et écriture dans une map_results.
  //l'argument sera par exemple "clean_profile.dat"
  map_results m;
  for (int i=1;i<2041;i++){
    string fullname="data/qmc/"+to_string(i)+"/"+filename;
    vector<VectorXd> v=read_singleresult(fullname);
    if(!v.empty()){
      m.insert(make_pair(i,v));
    }
  }
  cout << m.size() << " simulations read." << endl;
  return m;
}

map_exp read_exp_data(string const &filename){
  //lecture du fichier de données expérimentales
  map_exp m;
  int nlines=line_count(filename)-1; //on retire la première ligne
  int current_line=0;
  VectorXd X(49);
  VectorXd alpha(49);
  VectorXd D(49);
  VectorXd V(49);
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      if (line[0]=='p'){continue;}
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>()); //on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      //traitement des mots.
      X(current_line)=stod(words[0]);
      alpha(current_line)=stod(words[1]);
      D(current_line)=stod(words[2]);
      V(current_line)=stod(words[3]);
      current_line++;
    }
  }
  //écriture des données dans la map
  m.insert(make_pair("X",X));
  m.insert(make_pair("Alpha",alpha));
  m.insert(make_pair("D",D));
  m.insert(make_pair("V",V));
  return m;
}


double Kernel_Z_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le terme correctif Z
  //squared exponential
  double d=abs(x(0)-y(0));
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(2),2));
}

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2))+0.33*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D1Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à sigma_md
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return 2*hpar(0)*(1+(d/hpar(2))+0.33*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D2Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à sigma_obs
  return 0;
}

double D3Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à lcor
  double d=abs(x(0)-y(0));
  double X=d/hpar(2);
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*exp(-X)*0.33*(X+pow(X,2))*X/hpar(2);
}

double Kernel_GP_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour les GPs du surrogate du code de calcul et les GPs pour les hyperparamètres optimaux (aussi appelés hGPs).
  // 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double cor=pow(hpar(0),2);
  cor*=exp(-0.5*pow((x(0)-y(0))/hpar(1),2)); //phi
  cor*=exp(-0.5*pow((x(1)-y(1))/hpar(3),2)); //BK
  cor*=exp(-0.5*pow((x(2)-y(2))/hpar(4),2)); //COAL
  cor*=exp(-0.5*pow((x(3)-y(3))/hpar(5),2)); //NUCL
  cor*=exp(-0.5*pow((x(4)-y(4))/hpar(6),2)); //MT
  return cor;
}

double Kernel_GP_Matern(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double cor=pow(hpar(0),2);
  cor*=exp(-0.5*abs(x(0)-y(0))/hpar(1)); //phi
  cor*=exp(-0.5*abs(x(1)-y(1))/hpar(3)); //BK
  cor*=exp(-0.5*abs(x(2)-y(2))/hpar(4)); //COAL
  cor*=exp(-0.5*abs(x(3)-y(3))/hpar(5)); //NUCL
  cor*=exp(-0.5*abs(x(4)-y(4))/hpar(6)); //MT
  return cor;
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  double cor=pow(hpar(0),2);
  cor*=(1+abs(x(0)-y(0))/hpar(1))*exp(-abs(x(0)-y(0))/hpar(1)); //phi
  cor*=(1+abs(x(1)-y(1))/hpar(3))*exp(-abs(x(1)-y(1))/hpar(3)); //BK
  cor*=(1+abs(x(2)-y(2))/hpar(4))*exp(-abs(x(2)-y(2))/hpar(4)); //COAL
  cor*=(1+abs(x(3)-y(3))/hpar(5))*exp(-abs(x(3)-y(3))/hpar(5)); //NUCL
  cor*=(1+abs(x(4)-y(4))/hpar(6))*exp(-abs(x(4)-y(4))/hpar(6)); //MT
  return cor;
}

double logprior_hpars(VectorXd const &hpars){
  //prior en 1/sigma_edm^2
  return -2*log(hpars(0));
}

double logprior_pars(VectorXd const &pars){
  //prior uniforme sur les pramètres
  return 0;
}

double PriorMean(VectorXd const & X, VectorXd const &hpars){
  return 0;
}

void PrintVector(VectorXd &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X(i),values(i));
  }
  fclose(out);
}


double tauxparoi(vector<VectorXd> const &v){
  //renvoie le taux à la paroi d'un calcul v (si clean_profile est modifié, à modifier en conséquence).
  return v[1](39);
}
double tauxcoeur(vector<VectorXd> const &v){
  //renvoie le taux au coeur d'un calcul v (si clean_profile est modifié, à modifier en conséquence).
  return v[1](0);
}
double diamparoi(vector<VectorXd> const &v){
  //renvoie le diam à la paroi d'un calcul v (si clean_profile est modifié, à modifier en conséquence).
  return v[2](39);
}
double diamparoi(VectorXd const &diam, VectorXd const &x){
  vector<VectorXd> v(2);
  v[1]=diam; v[0]=x;
  return diamparoi(v);
}
double diamcoeur(vector<VectorXd> const &v){
  //renvoie le diam au coeur d'un calcul v (si clean_profile est modifié, à modifier en conséquence).
  return v[2](0);
}
double diamcoeur(VectorXd const &diam, VectorXd const &x){
  vector<VectorXd> v(2);
  v[1]=diam; v[0]=x;
  return diamcoeur(v);
}
double tauxmoyen(vector<VectorXd> const &v){
  //renvoie le taux au coeur d'un calcul v (si clean_profile est modifié, à modifier en conséquence).
  //j'applique l'équation de Julien sans réfléchir.
  int s=v[1].size();
  double alphamoy=0;
  for (int i=0;i<s-1;i++){
    alphamoy+=0.5*(v[1](i)+v[1](i+1))*(pow(v[0](i+1),2)-pow(v[0](i),2));
  }
  alphamoy/=pow(v[0](s-1),2);
  return alphamoy;
}
double tauxmoyen(VectorXd const &alpha, VectorXd const &x){
  vector<VectorXd> v(2);
  v[1]=alpha ; v[0]=x;
  return tauxmoyen(v);
}


VectorXd RtoGP(const VectorXd & X){
  //passage de l'espace réel à l'espace GP dans [0,1]. transformation linéaire.
  //"espace réel" : 0.9<X(0)<1.1 , et  0.5<X(i)<1.5 pour i de 1 à 4. 
  if(X(0)<0.9 || X(0)>1.1){cerr << "erreur de dimension rtogp 0" << endl;}
  if(X.size()!=5){cerr << "erreur de dimension rtogp" << endl;}
  VectorXd Xgp(5);
  Xgp(0)=(X(0)-0.9)/0.2;
  list<int> l={1,2,3,4};
  for (int i:l){
    if(X(i)>2 || X(i)<0.5){cerr << "erreur de dimension rtogp" << i <<" : " << X(i) << endl;}
    Xgp(i)=(X(i)-0.5)/1.5;
  }
  return Xgp;
}

VectorXd GPtoR(const VectorXd & Xgp){
  //passage de l'espace GP [0,1] à l'espace réel. transformation linéaire
  if(Xgp(0)<0 || Xgp(0)>1){cerr << "erreur de dimension gptor" << endl;}
  if(Xgp.size()!=5){cerr << "erreur de dimension gptor" << endl;}
  VectorXd X(5);
  X(0)=0.9+0.2*Xgp(0);
  list<int> l2={1,2,3,4};
  for (int i:l2){
    if(X(i)>1 || X(i)<0){cerr << "erreur de dimension gptor X(i)" << X(i) <<  endl;}
    X(i)=0.5+1.5*Xgp(i);
  }
  return X;
}


//tirage d'échantillons de vecteur de gp.
vector<vector<DATA>> PerformPCA(map_doe const &m, map_results const &mr, int qte, MatrixXd & VP, MatrixXd &Acoefs, VectorXd &featureMeans, int nmodes){
  //réalise la PCA de la quantité qte. 1=tdv, 2=diametre. VP = vecteurs propres réduits, Acoefs = normalisation des coefficients appris. nmodes = nombre de modes choisis.
  //construction de la matrice des données
  int ncalcs=mr.size(); //nombre de calculs réussis
  int nrayons=mr.at(1)[0].size(); //nombre de points de mesure en rayon.
  MatrixXd U(nrayons,ncalcs);
  MatrixXd P(5,ncalcs); //contient les paramètres des DoE, les colonnes correspondents aux colonnes de U.
  for(int i=0;i<ncalcs;i++){
    auto it = next(mr.cbegin(),i);
    U.col(i)=(*it).second[qte]; 
    P.col(i)=RtoGP(m.at((*it).first)); //on store les valeurs des paramètres correspondant aux calculs, dans les coordonnées GP.
  }
  //on centre les données https://stackoverflow.com/questions/33531505/principal-component-analysis-with-eigen-library
  featureMeans=U.rowwise().mean(); //vecteur colonne de taille nrayons
  U=U.colwise()-featureMeans;
  MatrixXd Covmatrix=U*U.transpose(); //taille nrayons,nrayons
  //décomp. valeurs propres et vp
  SelfAdjointEigenSolver<MatrixXd> eig(Covmatrix);
  VectorXd lambdas=eig.eigenvalues(); //nrayons
  MatrixXd vecpropres=eig.eigenvectors(); //(nrayons,nrayons)
  //sélection de nsel modes
  VP=vecpropres.rightCols(nmodes); //(nrayons,nmodes)
  VectorXd lambdas_red=lambdas.bottomRows(nmodes); //nmodes
  //on reverse les vecteurs propres et valeurs propres pour que les principaux se trouvent à la position 0.
  lambdas_red.reverseInPlace();
  MatrixXd VPr=VP.rowwise().reverse();
  VP=VPr;
  cout << "Sélection de " << nmodes << " modes." << endl;
  cout << "VP principales : " << lambdas_red.transpose()<< endl;
  cout << "Quantité d'énergie conservée : " << 100*lambdas_red.array().sum()/lambdas.array().sum() << " %" << endl;
  //matrice des coefficients à apprendre et scaling par les sqrt(lambda)
  MatrixXd A=VP.transpose()*U; //(nmodes,ncalcs)
  VectorXd Ascale=lambdas_red.array().sqrt();
  Acoefs=Ascale.asDiagonal(); 
  MatrixXd normedA=Acoefs.inverse()*A;
  //return de vector<DATA> pour aller dans le framework GP.
  vector<vector<DATA>> vd(nmodes);
  for(int j=0;j<nmodes;j++){
    vector<DATA> v(ncalcs);
    for(int i=0;i<ncalcs;i++){
      DATA dat; dat.SetX(P.col(i)); dat.SetValue(normedA(j,i)); 
      v[i]=dat;
    }
    vd[j]=v;
  }
  return vd;
}

MatrixXd EvaluateMeanVarGPPCA(vector<GP> const &vgp, VectorXd const & Target, MatrixXd const &VP, MatrixXd const & Acoefs, VectorXd const & featureMeans){
  //renvoie moyenne et variance prédites par un vecteur vgp, dans le cadre PCA. Au paramètre Target, en coordonnées GP. 
  //prédiction des coeffcients moyens et des variances moyennes
  int nmodes=Acoefs.cols();
  int nrayons=VP.rows();
  VectorXd meansgps(vgp.size());
  MatrixXd varsgps=MatrixXd::Zero(vgp.size(),vgp.size());
  for (int i=0;i<vgp.size();i++){
    VectorXd MeanVar=vgp[i].Eval(Target);
    meansgps(i)=MeanVar(0);
    varsgps(i,i)=MeanVar(1);
  }
  VectorXd Ymean=featureMeans+VP*Acoefs*meansgps;
  MatrixXd Yvar=VP*Acoefs*varsgps*Acoefs.transpose()*VP.transpose(); 
  VectorXd YVAR=Yvar.diagonal();
  //première colonne : moyennes, deuxième colonne : variances.
  MatrixXd result(nrayons,2); result.col(0)=Ymean, result.col(1)=YVAR;
  return result;
}

MatrixXd DrawSamplesGPPCA(int ns, vector<GP> const &vgp, VectorXd const & Target, MatrixXd const &VP, MatrixXd const & Acoefs, VectorXd const & featureMeans, default_random_engine & generator){
  //tirages de samples du surrogate du code de calcul.
  //ns : nombre de samples
  int nmodes=Acoefs.cols();
  int nrayons=VP.rows();
  vector<VectorXd> Target2(1); //passage en vector pour le framework du GP
  Target2[0]=Target;
  MatrixXd SamplesGP(nmodes,ns);
  for(int i=0;i<nmodes;i++){
    SamplesGP.row(i)=vgp[i].SampleGPDirect(Target2,ns,generator);
  }
  MatrixXd results(nrayons,ns);
  results=VP*Acoefs*SamplesGP;
  results=results.colwise()+featureMeans;
  return results;
}


VectorXd interpolate(VectorXd const & Yorig,VectorXd const & Xorig,VectorXd const & Xnew){
  //interpolation des données Yorig, définies sur Xorig, sur le nouveau grid Xnew.
  //fonction utile pour passer de l'espace "mesures expérimentales" (49 points de mesure dans un rayon)  à l'espace "prédictions du code de calcul" (40 points environ ?)
  //Les grids sont supposés ordonnés.
  if(Yorig.size()!=Xorig.size()){cerr <<"erreur d'interpolation : taille différente." << Yorig.size() << " "<<Xorig.size() << endl;}
  VectorXd Ynew(Xnew.size());
  for(int i=0;i<Xnew.size();i++){
    //check si on est au-delà des bornes de Xnew
    double ynext=0; //coordonnées dans l'espace d'origine
    double yprev=0;
    double xnext=0;
    double xprev=0;
    if(Xnew(i)<Xorig(0)){
      //on créé une valeur deux fois plus loin à partir de la pente estimée
      ynext=Yorig(0);
      xnext=Xorig(0);
      xprev=2*Xnew(i)-Xorig(0);
      double slope=(Yorig(1)-Yorig(0))/(Xorig(1)-Xorig(0));
      yprev=ynext-slope*(xnext-xprev);
    }
    else if(Xnew(i)>Xorig(Xorig.size()-1)){
      //pareil, on créée une valeur deux fois plus loin.
      yprev=Yorig(Xorig.size()-1);
      xprev=Xorig(Xorig.size()-1);
      xnext=2*Xnew(i)-xprev;
      double slope=(Yorig(Xorig.size()-1)-Yorig(Xorig.size()-2))/(Xorig(Xorig.size()-1)-Xorig(Xorig.size()-2));
      ynext=yprev-slope*(xprev-xnext);
    }
    else{
      int indice=0;
      while(Xnew(i)>Xorig(indice)){
        indice++;
      }
      //indice devient l'indice du immédiatement supérieur.
      ynext=Yorig(indice);
      xnext=Xorig(indice);
      yprev=Yorig(indice-1);
      xprev=Xorig(indice-1);
    }
    //interpolation linéaire
    double m=(ynext-yprev)/(xnext-xprev);
    double b=ynext-m*xnext;
    Ynew(i)=m*Xnew(i)+b;
  }
  return Ynew;
}

void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
  }
  fclose(out);
}

const double Big = -1.e16;


int main(int argc, char **argv){
  generator.seed(12345);

  //lecture du design QMC full.
  map_doe m=read_doe("design_qmc_full.dat");
  //lecture des résultats de calcul.
  map_results mr=read_results_qmc("clean_profile.dat");
  //lecture des données expérimentales. clés : X, Alpha, D.
  map_exp me=read_exp_data("clean_exp.dat");

  double diam_xmin=3E-3;
  double diam_xmax=me["X"](48);


  /*POD pour le taux de vide alpha*/
  vector<GP> vgp_a(3);
  MatrixXd VP_a;
  MatrixXd Acoefs_a;
  VectorXd featureMeans_a;
  
  {
    //initialisation des variables
    int nmodes=3;
    int ncalcs=mr.size(); //nombre de calculs réussis
    int nrayons=mr.at(1)[0].size(); //nombre de points de mesure en rayon.
    MatrixXd VP(nrayons,nmodes);
    MatrixXd Acoefs(nmodes,nmodes);
    VectorXd featureMeans(nrayons);
    vector<vector<DATA>> full_data=PerformPCA(m,mr,1,VP,Acoefs,featureMeans,nmodes); //PCA du taux de vide

    //construction des gps
    vector<GP> vgp(nmodes);

    int nhpars_gp=7;
    MatrixXd Bounds_hpars_gp(2,nhpars_gp);
    Bounds_hpars_gp(0,0)=1E-3; Bounds_hpars_gp(1,0)=1e2; //variance
    Bounds_hpars_gp(0,2)=1E-3; Bounds_hpars_gp(1,2)=2E-3; //sigma obs
    list<int> l={1,3,4,5,6};
    for (int i:l){
      Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=2; //lcors.
    }
    VectorXd hpars_gp_guess(7);
    for (int i=0;i<nhpars_gp;i++){
      hpars_gp_guess(i)=0.5*(Bounds_hpars_gp(1,i)+Bounds_hpars_gp(0,i));
    }

    VectorXd hpars_gp0(7);
    hpars_gp0 << 0.204,1.17,1e-3,2,2,0.83,1.44; //matern 3 modes

    GP gp0(Kernel_GP_Matern);
    gp0.SetData(full_data[0]);
    gp0.SetGP(hpars_gp0);
   
    VectorXd hpars_gp1(7);
    hpars_gp1 << 0.406,0.53,1e-3,1.72,1.32,0.84,1.99; //matern 3 modes

    GP gp1(Kernel_GP_Matern);
    gp1.SetData(full_data[1]);
    gp1.SetGP(hpars_gp1);

    VectorXd hpars_gp2(7);
    hpars_gp2 << 0.47,0.41,1e-3,1.11,0.87,0.69,1.94; //matern 3 modes

    GP gp2(Kernel_GP_Matern);
    gp2.SetData(full_data[2]);
    gp2.SetGP(hpars_gp2);

    vgp[0]=gp0;
    vgp[1]=gp1;
    vgp[2]=gp2;   

    //On recopie tout dans des variables extérieures
    vgp_a=vgp;
    VP_a=VP;
    Acoefs_a=Acoefs;
    featureMeans_a=featureMeans;
  }
  

  //Calibration alpha

  {
    //On a défini AUGDATA, qui est un DATA avec une Value VectorXd plutôt que double.
    //La fonction lambda_model renvoie directement les prédictions du surrogate du code de calcul aux points de mesure expérimentaux. On utilise donc une fonction lambda_model qui retourne une prédiction vectorielle, et pour laquelle le paramètre X n'est pas utilisé.

    cout << endl << "Début partie calibration..." << endl;
    //interpolation des données expérimentales sur le maillage du code de calcul.
    VectorXd Xexpe=me["X"];
    VectorXd Yexpe=me["Alpha"];
    VectorXd Xgrid_num=mr[1][0];
    VectorXd Yexpe_interpol=interpolate(Yexpe,Xexpe,Xgrid_num);

    //on les range dans un vecteur<AUGDATA> de dimension 1.
    int ndata=Xgrid_num.size(); //nrayons!
    vector<AUGDATA> data_exp(1);
    {
      VectorXd useless=VectorXd::Random(1);
      AUGDATA dat; dat.SetX(useless); dat.SetValue(Yexpe_interpol);
      data_exp[0]=dat;
    }

  int dim_theta=5;
  int dim_hpars=3;

  //bornes des paramètres de f et des hpars de z.
  
  VectorXd lb_t(dim_theta);
  VectorXd ub_t(dim_theta);
  for(int i=0;i<dim_theta;i++){
    lb_t(i)=0;
    ub_t(i)=1;
  }

  //hpars z : sedm, sobs, lcor.
  VectorXd lb_hpars(dim_hpars);
  VectorXd ub_hpars(dim_hpars);
  lb_hpars << 1e-4,1e-3,1e-4;
  ub_hpars << 1,1e-1,2e-2;

  //définition du modèle ! //coefs dans [0,1]
  auto lambda_model=[&vgp_a,&VP_a,&featureMeans_a,&Acoefs_a](VectorXd const & X,VectorXd const & theta)->VectorXd{
    return EvaluateMeanVarGPPCA(vgp_a,theta,VP_a,Acoefs_a,featureMeans_a).col(0);
  };

  //définition des bornes pour les hGPs.
  int nhpars_gp=7;
  MatrixXd Bounds_hpars_gp(2,nhpars_gp);
  Bounds_hpars_gp(0,0)=1E-7; Bounds_hpars_gp(1,0)=1e2; //sigma_md
  Bounds_hpars_gp(0,2)=1E-7; Bounds_hpars_gp(1,2)=1e-4; //sigma obs
  list<int> l={1,3,4,5,6};
  for (int i:l){
    Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=2; //lcors.
  }
  VectorXd hpars_gp_guess=0.5*(Bounds_hpars_gp.row(0)+Bounds_hpars_gp.row(1)).transpose();

  // paramètres pour la MCMC

  int nombre_steps_mcmc=1e6; //nombre total de steps
  int nombre_samples_collected=1500; //nombre de samples récupérés
  int nautocor=4000; //distance en steps sur laquelle on évalue l'autocorrélation

  VectorXd Xinit_mcmc(5);
  Xinit_mcmc << 0.5,0.5,0.5,0.5,0.5;
  MatrixXd COV_init=pow(0.1,2)*MatrixXd::Identity(5,5);
  MatrixXd COV_init_bayes=MatrixXd::Identity(8,8);
  COV_init_bayes.topLeftCorner(5,5)=COV_init;
  COV_init_bayes(5,5)*=pow(5e-2,2); //sedm
  COV_init_bayes(6,6)*=pow(3e-4,2); //sobs
  COV_init_bayes(7,7)*=pow(8e-4,2); //lcor
  cout << "COV_init_bayes : " << endl << COV_init_bayes << endl;
  
  //construction du grid
  DoE doe_init(lb_t,ub_t,100,1); //grid QMC Halton avec 1500 points.
  doe_init.WriteGrid("results/save/grid.gnu");

  //configuration de l'instance de base de densité
  Density MainDensity(doe_init);
  MainDensity.SetLogPriorPars(logprior_pars);
  MainDensity.SetLogPriorHpars(logprior_hpars);
  MainDensity.SetKernel(Kernel_Z_Matern52);
  MainDensity.SetKernelDerivatives(D1Kernel_Z_Matern52,D2Kernel_Z_Matern52,D3Kernel_Z_Matern52);
  MainDensity.SetModel(lambda_model);
  MainDensity.SetPriorMean(PriorMean);
  MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
  MainDensity.SetDataExp(data_exp);
  MainDensity.SetXprofile(Xgrid_num); //Xgrid_num contient les valeurs des rayons où il y a eu mesure.

  
  /*Calibration KOH*/
  
  
  cout << "début calibration KOH" << endl;
  VectorXd hpars_koh(dim_hpars);
  hpars_koh=0.5*(lb_hpars+ub_hpars);
  hpars_koh=MainDensity.HparsKOH(hpars_koh); //optimisation
  cout << "hpars koh :" << hpars_koh.transpose() << endl;

  MainDensity.Run_MCMC_fixed_hpars(nombre_steps_mcmc,nombre_samples_collected,Xinit_mcmc,COV_init,hpars_koh,generator);
  MainDensity.Autocor_diagnosis(nautocor,"results/diag/autocorkoh.gnu");
  //écriture de tous les steps MCMC
  MainDensity.WriteMCMCSamples("results/diag/allsampkoh.gnu");
  //écriture des samples choisis
  MainDensity.WriteSamples("results/save/sampkohalpha.gnu");
  //écriture des prédictions de F seule
  MainDensity.WritePredictionsF(VectorXd::Random(1),"results/preds/alphakohF.gnu");
  //écriture des prédictions de F+Z
  MainDensity.WritePredictions(VectorXd::Random(1),"results/preds/alphakoh.gnu");
  //écriture des densités prédictives de F en 3 points : coeur, milieu, paroi.
  MainDensity.WriteFinePredictions(VectorXd::Random(1),"results/preds/alphakohfine.gnu");
  //écriture des prédictions de F a priori.
  MainDensity.WritePriorPredictionsF(VectorXd::Random(1),"results/preds/alphaprior.gnu",generator);
  


   
  /*Calibration Opti*/
  
  cout << "début Opt :" << endl;
  auto begin_opt=chrono::steady_clock::now();
  DensityOpt DensOpt(MainDensity);
  //calcul des hyperparamètres optimaux sur le grid doe_init (1500 points QMC actuellement)
  DensOpt.Compute_optimal_hpars();
  //construction des hGPs
  DensOpt.Build_hGPs(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess,3);
  //optimisation des hyperparamètres des hGPs
  DensOpt.opti_allgps(hpars_gp_guess);
  //la fonction Test_hGPs permet de générer un grid de 2000 points d'hyperparamètres optimaux pour tester les performances. Prend environ 20 minutes.
  //DensOpt.Test_hGPs();
  //écriture des hyperparamètres optimaux
  DensOpt.WritehGPs("results/save/hGPsalpha.gnu");
  //run de la chaîne opti.
  DensOpt.Run_MCMC_opti_hGPs(nombre_steps_mcmc,nombre_samples_collected,Xinit_mcmc,COV_init,generator);
  DensOpt.Autocor_diagnosis(nautocor,"results/diag/autocoropt.gnu");
  DensOpt.WriteSamples("results/save/sampoptalpha.gnu");
  DensOpt.WriteMCMCSamples("results/diag/allsampopt.gnu");

  DensOpt.WritePredictionsF(VectorXd::Random(1),"results/preds/alphaoptF.gnu");
  DensOpt.WritePredictions(VectorXd::Random(1),"results/preds/alphaopt.gnu");
  DensOpt.WriteFinePredictions(VectorXd::Random(1),"results/preds/alphaoptfine.gnu");




  /*Calibration Bayes*/
  cout << endl << "Début full bayes : " << endl;
  VectorXd Xinit_bayes(8);
  Xinit_bayes << 0.5,0.7,0.2,0.5,0.5,1e-1,2e-3,3e-3;
  MainDensity.Run_FullMCMC(nombre_steps_mcmc,nombre_samples_collected,Xinit_bayes,COV_init_bayes,generator);
  MainDensity.WritePredictionsF(VectorXd::Random(1),"results/preds/alphafbF.gnu");
  MainDensity.WritePredictions(VectorXd::Random(1),"results/preds/alphafb.gnu");
  MainDensity.WriteFinePredictions(VectorXd::Random(1),"results/preds/alphafbfine.gnu");
  MainDensity.WriteSamples("results/save/sampfbalpha.gnu");
  MainDensity.Autocor_diagnosis(nautocor,"results/diag/autocorfb.gnu");
  MainDensity.WriteMCMCSamples("results/diag/allsampfb.gnu");

  
  }
exit(0);

  
};