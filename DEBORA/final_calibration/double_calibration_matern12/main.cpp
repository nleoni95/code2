// Dans ce fichier : on lit les données du calcul et on les post-traite à notre souhait.
// Calibration avec le taux de vide et le diamètre en même temps.
// Master file pour la calibration double.


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
    string fullname="../../data/qmc/"+to_string(i)+"/"+filename;
    vector<VectorXd> v=read_singleresult(fullname);
    if(!v.empty()){
      m.insert(make_pair(i,v));
    }
  }
  cout << m.size() << " simulations read." << endl;
  return m;
}

map_results read_results_lhs(string const &filename){
  //lecture de tous les résultats de calcul et écriture dans une map_results.
  //l'argument sera par exemple "clean_profile.dat"
  map_results m;
  for (int i=1;i<2041;i++){
    string fullname="../../data/lhs/"+to_string(i)+"/"+filename;
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

double gaussprob(double x,double mu, double sigma){
  //renvoie la probabilité gaussienne
  return 1./(sqrt(2*3.14*pow(sigma,2)))*exp(-0.5*pow((x-mu)/sigma,2));
}


double Kernel_Z(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  return pow(hpar(0),2)*exp(-0.5*pow(d/hpar(2),2));
}

double Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //squared exponential
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2))+(1./3)*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D1Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à sigma
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return 2*hpar(0)*(1+(d/hpar(2))+(1./3)*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double D2Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  return 0;
}

double D3Kernel_Z_Matern52(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à lcor
  double d=abs(x(0)-y(0));
  double X=d/hpar(2);
  return pow(hpar(0),2)*exp(-X)*pow(X,2)*(d+hpar(2))/(3*pow(hpar(2),2));
}

double Kernel_Z_Matern12(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //matern 5/2.
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*exp(-d/hpar(2)); //5/2
}

double D1Kernel_Z_Matern12(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à sigma
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return 2*hpar(0)*exp(-d/hpar(2)); //5/2
}

double D2Kernel_Z_Matern12(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  return 0;
}

double D3Kernel_Z_Matern12(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à lcor
  double d=abs(x(0)-y(0));
  double X=d/hpar(2);
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*exp(-X)*X/hpar(2);
}

double Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //matern 5/2.
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*(1+(d/hpar(2)))*exp(-d/hpar(2)); //5/2
}

double D1Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à sigma
  double d=abs(x(0)-y(0));
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return 2*hpar(0)*(1+(d/hpar(2)))*exp(-d/hpar(2)); //5/2
}

double D2Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  return 0;
}

double D3Kernel_Z_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //dérivée par rapport à lcor
  double d=abs(x(0)-y(0));
  double X=d/hpar(2);
  //return pow(hpar(0),2)*exp(-d/hpar(2));
  return pow(hpar(0),2)*exp(-X)*(X)*X/hpar(2);
}


double Kernel_GP_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double cor=0;
  cor+=pow((x(0)-y(0))/hpar(1),2); //phi
  cor+=pow((x(1)-y(1))/hpar(3),2); //BK
  cor+=pow((x(2)-y(2))/hpar(4),2); //COAL
  cor+=pow((x(3)-y(3))/hpar(5),2); //NUCL
  cor+=pow((x(4)-y(4))/hpar(6),2); //MT
  return pow(hpar(0),2)*exp(-0.5*cor);
}

double Kernel_GP_Matern12(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double cor=0;
  cor+=0.5*abs(x(0)-y(0))/hpar(1); //phi
  cor+=0.5*abs(x(1)-y(1))/hpar(3); //BK
  cor+=0.5*abs(x(2)-y(2))/hpar(4); //COAL
  cor+=0.5*abs(x(3)-y(3))/hpar(5); //NUCL
  cor+=0.5*abs(x(4)-y(4))/hpar(6); //MT
  return pow(hpar(0),2)*exp(-cor);
}

double Kernel_GP_Matern32(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double d1=abs(x(0)-y(0))/hpar(1);
  double d2=abs(x(1)-y(1))/hpar(3);
  double d3=abs(x(2)-y(2))/hpar(4);
  double d4=abs(x(3)-y(3))/hpar(5);
  double d5=abs(x(4)-y(4))/hpar(6);
  double cor=-d1-d2-d3-d4-d5;
  cor=exp(cor)*(1+d1)*(1+d2)*(1+d3)*(1+d4)*(1+d5);
  return pow(hpar(0),2)*cor;
}

double logprior_hpars(VectorXd const &hpars){
  //edm, exp, lcor
  return -2*log(hpars(0));
}

double logprior_pars(VectorXd const &pars){
  //prior uniforme sur les pramètres
  return 0;
}

//myoptfunc_gp est définie dans densities.cpp


void PrintVector(VectorXd &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X(i),values(i));
  }
  fclose(out);
}


VectorXd RtoGP(const VectorXd & X){
  //passage de l'espace réel à l'espace GP dans [0,1]. transfo linéaire
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
  //passage de l'espace GP [0,1] à l'espace réel. transfo linéaire
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
    U.col(i)=(*it).second[qte]; //1 car on regarde le taux de vide.
    P.col(i)=RtoGP(m.at((*it).first)); //on store les valeurs des paramètres correspondant aux calculs, dans les coordonnées GP.
  }
  //on retranche à chaque colonne la moyenne des colonnes https://stackoverflow.com/questions/33531505/principal-component-analysis-with-eigen-library
  featureMeans=U.rowwise().mean(); //vecteur colonne de taille nrayons
  U=U.colwise()-featureMeans;
  MatrixXd Covmatrix=U*U.transpose(); //taille nrayons,nrayons
  Covmatrix/=(ncalcs);
  //décomp. valeurs propres et vp
  SelfAdjointEigenSolver<MatrixXd> eig(Covmatrix);
  //valeurs propres
  VectorXd lambdas=eig.eigenvalues(); //nrayons
  MatrixXd vecpropres=eig.eigenvectors(); //(nrayons,nrayons)
  //cout << "lambdas : " << lambdas.transpose() << endl;
  //cout << "ev : " << vecpropres << endl;
  //vérification : vecpropres.transpose()*vecpropres vaut l'identité.

  //sélection de nsel modes
  MatrixXd VPs=vecpropres.rightCols(nmodes); //(nrayons,nmodes)
  VectorXd lambdas_red=lambdas.bottomRows(nmodes); //nmodes
  //on reverse les vecteurs propres et valeurs propres pour que les principaux se trouvent à la position 0.
  lambdas_red.reverseInPlace();
  VP=VPs.rowwise().reverse();
  cout << "Sélection de " << nmodes << " modes." << endl;
  cout << "VP principales : " << lambdas_red.transpose()<< endl;
  cout << "Quantité d'énergie conservée : " << 100*lambdas_red.array().sum()/lambdas.array().sum() << " %" << endl;
  //vérification qu'on a bien choisi des vecteurs propres : on a bien vecred.transpose()*vecred=Id
  //calcul de la matrice des coefficients à apprendre
  MatrixXd A=VP.transpose()*U; //(nmodes,ncalcs)
  //les lignes de A sont déjà du même ordre de grandeur.
  //remarque : les lignes de A somment à 0..
  VectorXd Ascale=lambdas_red.array().sqrt();
  Acoefs=Ascale.asDiagonal(); //matrice diagonale avec les ordres de grandeur de A.
  MatrixXd normedA=Acoefs.inverse()*A;
  //on exporte le tout sous forme de vecteur<DATA>
  vector<vector<DATA>> vd(nmodes);
  for(int j=0;j<nmodes;j++){
    vector<DATA> v(ncalcs);
    for(int i=0;i<ncalcs;i++){
      DATA dat; dat.SetX(P.col(i)); dat.SetValue(normedA(j,i)); //P déjà en coordonnées gp.
      v[i]=dat;
    }
    vd[j]=v;
  }
  return vd;
}

MatrixXd EvaluateMeanVarGPPCA(vector<GP> const &vgp, VectorXd const & Target, MatrixXd const &VP, MatrixXd const & Acoefs, VectorXd const & featureMeans){
  //renvoie moyenne et variance prédites par un vecteur vgp, dans le cadre PCA. Au paramètre Target, en coordonnées GP. Les points d'évaluation sont les mêmes que ceux utilisés pour la construction de la base.
  //prédiction des coeffcients moyens et des variances moyennes
  int nmodes=Acoefs.cols();
  int nrayons=VP.rows();
  VectorXd meansgps(vgp.size());
  VectorXd varsgps(vgp.size());
  for (int i=0;i<vgp.size();i++){
    VectorXd MeanVar=vgp[i].Eval(Target);
    meansgps(i)=MeanVar(0);
    varsgps(i)=MeanVar(1);
  }
  MatrixXd VP2=square(VP.array()); //coefficient-wise square
  VectorXd Ymean=featureMeans+VP*Acoefs*meansgps;
  VectorXd Yvar=VP2*Acoefs*Acoefs*varsgps; 
  MatrixXd result(nrayons,2); result.col(0)=Ymean, result.col(1)=Yvar;
  return result;
}

MatrixXd EvaluateMeanGPPCA(vector<GP> const &vgp, VectorXd const & Target, MatrixXd const &VP, MatrixXd const & Acoefs, VectorXd const & featureMeans){
  //renvoie moyenne et variance prédites par un vecteur vgp, dans le cadre PCA. Au paramètre Target, en coordonnées GP. Les points d'évaluation sont les mêmes que ceux utilisés pour la construction de la base.
  //prédiction des coeffcients moyens et des variances moyennes
  int nmodes=Acoefs.cols();
  int nrayons=VP.rows();
  VectorXd meansgps(vgp.size());
  VectorXd varsgps(vgp.size());
  for (int i=0;i<vgp.size();i++){
    meansgps(i)=vgp[i].EvalMean(Target);
  }
  return featureMeans+VP*Acoefs*meansgps;
}


MatrixXd DrawSamplesGPPCA(int ns, vector<GP> const &vgp, VectorXd const & Target, MatrixXd const &VP, MatrixXd const & Acoefs, VectorXd const & featureMeans, default_random_engine & generator){
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

VectorXd EvaluateVarGPPCAbySampling(vector<GP> const &vgp, VectorXd const & Target, MatrixXd const &VP, MatrixXd const & Acoefs, VectorXd const & featureMeans,default_random_engine & generator){
  //renvoie variance prédites par un vecteur vgp, dans le cadre PCA. Au paramètre Target, en coordonnées GP. Les points d'évaluation sont les mêmes que ceux utilisés pour la construction de la base.
  //évaluation de la variance par tirage d'échantillons pour voir si correspondance avec l'autre méthode.
  int nrayons=VP.rows();
  int nsamples=1e6;
  MatrixXd Samples=DrawSamplesGPPCA(nsamples,vgp,Target,VP,Acoefs,featureMeans,generator);
  VectorXd VAR(nrayons);
  for (int i=0;i<nrayons;i++){
    //on recopie l'ensemble des samples dans un vector
    vector<double> v(nsamples); for(int j=0;j<nsamples;j++){v[j]=Samples(i,j);}
    //on évalue la variance de la série
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double var = (sq_sum / v.size() - mean * mean);
    VAR(i)=var;
  }
  return VAR;
}

void eval_erreur_validation(MatrixXd const & M_truth, MatrixXd const & M_projected, MatrixXd const & M_predicted){
  //évaluation des erreurs de validation et répartition de l'erreur.
  int ncalcs=M_truth.cols();
  //définition du produit scalaire
  auto ps=[](MatrixXd const &A, MatrixXd const &B)->double{
    return (A.transpose()*B).trace();
  };
  double disttruth_proj=sqrt(ps(M_truth-M_projected,M_truth-M_projected));
  double distproj_GP=sqrt(ps(M_predicted-M_projected,M_predicted-M_projected));
  double disttotale=sqrt(ps(M_predicted-M_truth,M_predicted-M_truth));
  double prop_err_projection=pow(disttruth_proj,2)/pow(disttotale,2);
  double prop_err_GP=pow(distproj_GP,2)/pow(disttotale,2);
  cout << "répartition de l'erreur entre projection et GP : " << prop_err_projection << ", " << prop_err_GP << endl;
  double pct_moyen_erreur=100*disttotale/sqrt(ps(M_truth,M_truth));
  cout << "pct moyen erreur L2 : " << pct_moyen_erreur << endl;
}

void compute_erreurs_validation(int qte, map_doe const & m_lhs, map_results const & mr_lhs ,vector<GP> const &vgp, MatrixXd const &VP, MatrixXd const & Acoefs, VectorXd const & featureMeans){
  //on récupère les erreurs sur le dataset de validation.
  //étape 1 : mettre les données de validation dans une matrice
  int ncalcs=mr_lhs.size();
  int nmodes=Acoefs.cols();
  int nrayons=VP.rows();
  MatrixXd M_truth(nrayons,ncalcs); //valeurs calculées sur le dataset
  MatrixXd P_truth(5,ncalcs); //valeurs des 5 paramètres sur le dataset (coordonnées GP)
  for(int i=0;i<ncalcs;i++){
    auto it = next(mr_lhs.cbegin(),i);
    M_truth.col(i)=(*it).second[qte]; //1 car on regarde le taux de vide.
    P_truth.col(i)=RtoGP(m_lhs.at((*it).first)); //on store les valeurs des paramètres correspondant aux calculs, dans les coordonnées GP.
  }
  //projeter le dataset sur la base VP.
  MatrixXd M_projected(nrayons,ncalcs);
  MatrixXd M_truth_centered=M_truth.colwise()-featureMeans;
  MatrixXd M_truth_multiplied=VP*VP.transpose()*(M_truth_centered);
  M_projected=(M_truth_multiplied).colwise()+featureMeans; //M_proj=featureMeans+VPtVP(M_truth-featureMeans)
  //calcul des prédictions moyennes GP
  MatrixXd M_predicted(nrayons, ncalcs);
  for(int i=0;i<ncalcs;i++){
    VectorXd ParamEval=P_truth.col(i); //paramètres du calcul i (coords GP)
    M_predicted.col(i)=EvaluateMeanGPPCA(vgp,ParamEval,VP,Acoefs,featureMeans); //on prend seulement les prédictions moyennes.
  }
  //calcul des erreurs. Faisons sur tout le domaine.
  auto afficher_erreurs=[M_truth,M_projected,M_predicted,ncalcs](int nstart, int nend)-> void{
    MatrixXd M_truth_2=M_truth.block(nstart,0,nend-nstart+1,ncalcs);
    MatrixXd M_projected_2=M_projected.block(nstart,0,nend-nstart+1,ncalcs);
    MatrixXd M_predicted_2=M_predicted.block(nstart,0,nend-nstart+1,ncalcs); 
    eval_erreur_validation(M_truth_2,M_projected_2,M_predicted_2);
  };
  cout << "sur tout le domaine : " << endl;
  afficher_erreurs(0,nrayons-1);
  cout << "à la paroi : " << endl;
  afficher_erreurs(26,39);
  cout << "au milieu du canal : " << endl;
  afficher_erreurs(10,25);
  cout << "au coeur du canal : " << endl;
  afficher_erreurs(0,9);

}

VectorXd interpolate(VectorXd const & Yorig,VectorXd const & Xorig,VectorXd const & Xnew){
  //interpolation des données Yorig, définies sur Xorig, sur le nouveau grid Xnew.
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

//fonctions de MCMC. Autant en faire une unique, et on spécifie seulement la fonction de vraisemblance. Dans ce cas la fonction de vraisemblance doit prendre en argument l'état actuel, pour pouvoir faire opti modif.

void Run_Burn_Phase_MCMC(int nburn, MatrixXd & COV_init, VectorXd & Xcurrento,function<double(pair<VectorXd,VectorXd>, VectorXd const &)> const & compute_score, function<pair<VectorXd,VectorXd>(VectorXd)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //phase de burn.
  int dim_mcmc=COV_init.cols();
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  VectorXd Xinit=Xcurrento;
  pair<VectorXd,VectorXd> hparsinit=get_hpars(Xinit);
  MatrixXd COV=COV_init;
  MatrixXd sqrtCOV=COV.llt().matrixL();
  double finit=compute_score(hparsinit,Xinit);
  VectorXd Xcurrent=Xinit;
  double fcurrent=finit;
  pair<VectorXd,VectorXd> hparscurrent=hparsinit;
  int naccept=0;
  VectorXd acc_means=VectorXd::Zero(dim_mcmc);
  MatrixXd acc_var=MatrixXd::Zero(dim_mcmc,dim_mcmc);
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nburn;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds(Xcandidate)){
      pair<VectorXd,VectorXd> hparscandidate=get_hpars(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    acc_means+=Xcurrent;
    acc_var+=Xcurrent*Xcurrent.transpose();
  }
  double acc_rate=(double)(naccept)/(double)(nburn);
  MatrixXd CovProp=(pow(2.38,2)/(double)(dim_mcmc))*(acc_var/(nburn-1)-acc_means*acc_means.transpose()/pow(1.0*nburn,2));
  auto end=chrono::steady_clock::now();
  cout << "burn phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "new cov matrix : " << endl << CovProp << endl;
  Xcurrento=Xcurrent;
  COV_init=CovProp;
}

tuple<vector<VectorXd>,vector<pair<VectorXd,VectorXd>>,vector<VectorXd>,vector<double>> Run_MCMC(int nsteps,int nsamples, VectorXd const & Xinit,MatrixXd const & COV_init,function<double(pair<VectorXd,VectorXd>, VectorXd const &)> const & compute_score, function<pair<VectorXd,VectorXd>(VectorXd)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //MCMC générique. renvoie un tuple <samples,hparsofsamples,allsamples>
  int dim_mcmc=Xinit.size();
  vector<VectorXd> samples;
  vector<pair<VectorXd,VectorXd>> hparsofsamples;
  vector<VectorXd> allsamples;
  vector<double> scores_of_samples;
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  VectorXd Xinit0=Xinit;
  Run_Burn_Phase_MCMC(nsteps*0.1,COV,Xinit0,compute_score,get_hpars,in_bounds,generator);
  //scaling
  cout << "no scaling" << endl;
  //COV=scale_covmatrix(COV,Xinit0,compute_score,get_hpars,in_bounds,0,generator,"results/scaling.gnu");
  MatrixXd sqrtCOV=COV.llt().matrixL();
  pair<VectorXd,VectorXd> hparsinit=get_hpars(Xinit0);
  double finit=compute_score(hparsinit,Xinit0);
  VectorXd Xcurrent=Xinit0;
  double fcurrent=finit;
  pair<VectorXd,VectorXd> hparscurrent=hparsinit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds(Xcandidate)){
      pair<VectorXd,VectorXd> hparscandidate=get_hpars(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      samples.push_back(Xcurrent);
      hparsofsamples.push_back(hparscurrent);
      scores_of_samples.push_back(fcurrent);
    }
    allsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << samples.size() << endl;
  cout << samples[0].transpose() << endl;
  cout << samples[1].transpose() << endl;
  cout << samples[2].transpose() << endl;
  cout << samples[3].transpose() << endl;
  cout << samples[4].transpose() << endl;
  auto tp=make_tuple(samples,hparsofsamples,allsamples,scores_of_samples);
  return tp;
}


tuple<vector<VectorXd>,vector<pair<VectorXd,VectorXd>>,vector<VectorXd>> Run_MCMC_noburn(int nsteps,int nsamples, VectorXd const & Xinit,MatrixXd const & COV_init,function<double(pair<VectorXd,VectorXd>, VectorXd const &)> const & compute_score, function<pair<VectorXd,VectorXd>(VectorXd)> const & get_hpars,function<bool(VectorXd)> const & in_bounds,default_random_engine & generator){
  //MCMC générique. renvoie un tuple <samples,hparsofsamples,allsamples>
  int dim_mcmc=Xinit.size();
  vector<VectorXd> samples;
  vector<pair<VectorXd,VectorXd>> hparsofsamples;
  vector<VectorXd> allsamples;
  uniform_real_distribution<double> distU(0,1);
  normal_distribution<double> distN(0,1);
  MatrixXd COV=COV_init;
  VectorXd Xinit0=Xinit;
  //Run_Burn_Phase_MCMC(nsteps*0.1,COV,Xinit0,compute_score,get_hpars,in_bounds,generator);
  //scaling
  cout << "no scaling, no burn." << endl;
  //COV=scale_covmatrix(COV,Xinit0,compute_score,get_hpars,in_bounds,0,generator,"results/scaling.gnu");
  MatrixXd sqrtCOV=COV.llt().matrixL();
  pair<VectorXd,VectorXd> hparsinit=get_hpars(Xinit0);
  double finit=compute_score(hparsinit,Xinit0);
  VectorXd Xcurrent=Xinit0;
  double fcurrent=finit;
  pair<VectorXd,VectorXd> hparscurrent=hparsinit;
  int naccept=0;
  auto begin=chrono::steady_clock::now();
  for(int i=0;i<nsteps;i++){
    VectorXd Step(dim_mcmc); for(int j=0;j<dim_mcmc;j++){Step(j)=distN(generator);}
    VectorXd Xcandidate=Xcurrent+sqrtCOV*Step;
    if(in_bounds(Xcandidate)){
      pair<VectorXd,VectorXd> hparscandidate=get_hpars(Xcandidate);
      double fcandidate=compute_score(hparscandidate,Xcandidate);
      if(fcandidate>fcurrent || fcandidate-fcurrent>log(distU(generator))){
        naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
        hparscurrent=hparscandidate;
      }
    }
    if(i%(nsteps/nsamples)==0 && i>0){
      samples.push_back(Xcurrent);
      hparsofsamples.push_back(hparscurrent);
    }
    allsamples.push_back(Xcurrent);
  }
  auto end=chrono::steady_clock::now();
  double acc_rate=(double)(naccept)/(double)(nsteps);
  cout << "MCMC phase over. " << " time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s, accept rate : " << 100*acc_rate << " pct. " << endl;
  cout << "number of samples : " << samples.size() << endl;
  cout << samples[0].transpose() << endl;
  cout << samples[1].transpose() << endl;
  cout << samples[2].transpose() << endl;
  cout << samples[3].transpose() << endl;
  cout << samples[4].transpose() << endl;
  auto tp=make_tuple(samples,hparsofsamples,allsamples);
  return tp;
}


Eigen::VectorXd VtoVXD2(std::vector<double> const &v){
    //conversion vector double - vectorxd
    Eigen::VectorXd X(v.size()); for(int i=0;i<v.size();i++) {X(i)=v[i];} return X;
    }

std::vector<double> VXDtoV2(Eigen::VectorXd const &X){
    //conversion vectorxd - vector double
    std::vector<double> v(X.size()); for(int i=0;i<v.size();i++) {v[i]=X(i);} return v;
    }

VectorXd optroutine(nlopt::vfunc optfunc,void *data_ptr, VectorXd &X, VectorXd const & lb_hpars, VectorXd const & ub_hpars,double max_time){
  //routine d'optimisation pour KOH double.
  vector<double> x=VXDtoV2(X);
  vector<double> lb_hpars_opt=VXDtoV2(lb_hpars);
  vector<double> ub_hpars_opt=VXDtoV2(ub_hpars);
  //paramètres d'optimisation
  int maxeval=10000;
  int popsize=80;
  double ftol_large=1e-7;
  double xtol_large=1e-4;
  double ftol_fin=1e-15;
  double xtol_fin=1e-7;
  // 1 optimiseur local et un global.
  nlopt::opt local_opt(nlopt::LN_SBPLX,x.size());
  local_opt.set_max_objective(optfunc, data_ptr); 
  local_opt.set_ftol_rel(ftol_fin);
  local_opt.set_xtol_rel(xtol_fin);
  local_opt.set_lower_bounds(lb_hpars_opt);
  local_opt.set_upper_bounds(ub_hpars_opt);

  nlopt::opt opt(nlopt::GD_MLSL_LDS, x.size());
  opt.set_max_objective(optfunc, data_ptr);
  opt.set_lower_bounds(lb_hpars_opt);
  opt.set_upper_bounds(ub_hpars_opt);
  //pas de contrainte de temps.
  opt.set_maxtime(max_time); //20 secondes au max.
  opt.set_local_optimizer(local_opt);
  double msup; /* the maximum objective value, upon return */
  int fin=opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  //on relance une opti locale à partir du max. trouvé.
  local_opt.set_ftol_rel(ftol_fin);
  local_opt.set_xtol_rel(xtol_fin);
  fin=local_opt.optimize(x, msup); //messages d'arrêt 
  if (!fin==3){cout << "opti hpars message d'erreur : " << fin << endl;}
  return VtoVXD2(x);
}
//on peut réutiliser les fonctions de diagnostic. Il suffira de faire une fonction set_all_mcmc_samples. et aussi set_samples et set_hparsofsamples pour faire la prédiction directement.

double optfuncKOH_Double(const std::vector<double> &x, std::vector<double> &grad, void *data){
  /* fonction à optimiser pour trouver les hpars koh avec deux fonctions.*/
  //alpha=1, diam=2.
  auto ptp=(tuple<const MatrixXd *,const MatrixXd *,Density*,Density*>*) data; //cast
  auto tp=*ptp;
  const MatrixXd *Residus1=get<0>(tp);
  const MatrixXd *Residus2=get<1>(tp);
  const Density *d1=get<2>(tp);
  const Density *d2=get<3>(tp);
  const vector<VectorXd> *xconv1=d1->GetXconverted();
  const vector<VectorXd> *xconv2=d2->GetXconverted();
  VectorXd h=VtoVXD2(x);
  VectorXd hpars_alpha=h.head(3);
  VectorXd hpars_diam=h.tail(3);
  double logvstyp=30; //à choisir.
  //calcul de la logvs sur le grid
  vector<double> prob1(Residus1->cols());
  vector<double> prob2(Residus2->cols());
  MatrixXd G1=d1->Gamma(*xconv1,hpars_alpha);
  MatrixXd G2=d2->Gamma(*xconv2,hpars_diam);
  LDLT<MatrixXd> ldlt1(G1);
  LDLT<MatrixXd> ldlt2(G2);
  //VectorXd pmean=VectorXd::Zero(Residus1->rows());
  //au lieu du transform, j'essaye une boucle simple.
  for(int i=0;i<Residus1->cols();i++){
    double g1=d1->loglikelihood_fast(Residus1->col(i),ldlt1);
    double g2=d2->loglikelihood_fast(Residus2->col(i),ldlt2);
    prob1[i]=g1;
    prob2[i]=g2;
  }
  //normalisation et passage à l'exponentielle.
  //le max doit être le même sur toutes les itérations... d'où la nécessité de le définir à l'extérieur !!!
  transform(prob1.begin(),prob1.end(),prob1.begin(),[logvstyp](const double &d)->double {
    double f=exp(d-logvstyp);
    if(isinf(f)){cerr << "erreur myoptfunc_koh : infini prob1" << endl;}
    return f;
  });
  transform(prob2.begin(),prob2.end(),prob2.begin(),[logvstyp](const double &d)->double {
    double f=exp(d-logvstyp);
    if(isinf(f)){cerr << "erreur myoptfunc_koh : infini prob2" << endl;}
    return f;
  });
  //calcul de l'intégrale. suppose un grid régulier. modifier si on veut un prior pour les paramètres.
  double res(0);
  for(int i=0;i<prob1.size();i++){
    res+=prob1[i]*prob2[i];
  }
  res*=exp(d1->EvaluateLogPHpars(hpars_alpha))*exp(d2->EvaluateLogPHpars(hpars_diam));
  return res;
};

VectorXd HparsKOH_double(VectorXd const & hpars_guess,VectorXd const & lb_hpars,VectorXd const & ub_hpars,Density const & D1, Density const & D2,vector<AUGDATA> const & data_exp_1,vector<AUGDATA> const & data_exp_2){
  //calcul de hpars koh avec les deux fonctions de vraisemblance.
  //les deux densités sont construites sur le même grid.
  VectorXd guess=hpars_guess;
  vector<VectorXd> Grid=*(D1.GetGrid());
  MatrixXd Residustheta1(data_exp_1[0].Value().size(),Grid.size());
  MatrixXd Residustheta2(data_exp_2[0].Value().size(),Grid.size());
  for(int i=0;i<Grid.size();i++){
    VectorXd theta=Grid[i];
    Residustheta1.col(i)=data_exp_1[0].Value()-D1.EvaluateModel(data_exp_1[0].GetX(),theta);
    Residustheta2.col(i)=data_exp_2[0].Value()-D2.EvaluateModel(data_exp_2[0].GetX(),theta);
  }
  auto tp=make_tuple(&Residustheta1,&Residustheta2,&D1,&D2);
  guess=optroutine(optfuncKOH_Double,&tp,guess,lb_hpars,ub_hpars,500); //max_time en secondes
  return guess;
}


tuple<vector<VectorXd>,vector<VectorXd>,vector<VectorXd>> read_sample_reference(string &filename){
  //lit un vecteur de thetas, hpars alpha, hpars diam.
  vector<VectorXd> grid_verif;
  vector<VectorXd> halpha_verif;
  vector<VectorXd> hdiam_verif;
  ifstream ifile(filename);
  if(ifile){
    string line;
    while(getline(ifile,line)){
      if (line[0]=='#'){continue;}
      if (line.empty()){continue;}
      //décomposition de la line en des mots
      istringstream iss(line);
      vector<string> words((istream_iterator<string>(iss)),istream_iterator<string>()); //on récupère les mots. Ne fonctionne qu'avec des espaces séparateurs.
      VectorXd theta(5);
      for (int i=0;i<5;i++){
        theta(i)=stod(words[i]);
      }
      VectorXd hpars_alpha(3);
      for(int i=0;i<3;i++){
        hpars_alpha(i)=stod(words[i+5]);
      }
      
      VectorXd hpars_diam(3);
      for(int i=0;i<3;i++){
        hpars_diam(i)=stod(words[i+8]);
      }
      grid_verif.push_back(theta);
      halpha_verif.push_back(hpars_alpha);
      hdiam_verif.push_back(hpars_diam);
    }
  }
  cout << "number of samples loaded in the verif grid : " << grid_verif.size() << endl;
  auto tp=make_tuple(grid_verif,halpha_verif,hdiam_verif);
  return tp;
}

double myoptfunc_gp(const std::vector<double> &x, std::vector<double> &grad, void *data){
	/* This is the function you optimize for defining the GP */
	GP* proc = (GP*) data;											//Pointer to the GP
	Eigen::VectorXd p(x.size());									//Parameters to be optimized
	for(int i=0; i<(int) x.size(); i++) p(i) = x[i];				//Setting the proposed value of the parameters
	double value = proc->SetGP(p);									//Evaluate the function
	if (!grad.empty()) {											//Cannot compute gradient : stop!
		std::cout << "Asking for gradient, I stop !" << std::endl; exit(1);
	}
	return value;
};

int find_max_ll_double_densities(function<double(pair<VectorXd,VectorXd>, VectorXd const &)> const & compute_score, vector<VectorXd> &vthetas,vector<pair<VectorXd,VectorXd>> &vhpars){
  //renvoie l'indice du maximum de vraisemblance de la densité jointe 
  vector<double> vlik(vthetas.size());
  for(int i=0;i<vthetas.size();i++){
    vlik[i]=compute_score(vhpars[i],vthetas[i]);
  }
  auto itmax=max_element(vlik.begin(),vlik.end());
  int index=distance(vlik.begin(),itmax);
  cout << "findmaxll : indice " << index << ", valeur au max : " << vlik[index] << "." << endl;
  cout << "valeurs annexes : " << vlik[index-1] << ", " << vlik[index+1] << ", " << vlik[0] << endl;
  return index;
}

int find_max_ll_double_densities(function<double(pair<VectorXd,VectorXd>, VectorXd const &)> const & compute_score, vector<VectorXd> &samplesfb){
  //renvoie l'indice du maximum de vraisemblance de la densité jointe 
  //p ne sert à rien pour fb : tout est dasn els samples
  auto p=make_pair(VectorXd::Ones(1),VectorXd::Ones(1));
  vector<double> vlik(samplesfb.size());
  for(int i=0;i<samplesfb.size();i++){
    vlik[i]=compute_score(p,samplesfb[i]);
  }
  auto itmax=max_element(vlik.begin(),vlik.end());
  return distance(vlik.begin(),itmax);
}

double estimate_evidence(function<double(pair<VectorXd,VectorXd>, VectorXd const &)> const & compute_score, function<pair<VectorXd,VectorXd>(VectorXd)> const & get_hpars,int npoints, default_random_engine & generator){
  uniform_real_distribution<double> distU(0,1);
  vector<VectorXd> thetas(npoints);
  for(int i=0;i<npoints;i++){
    VectorXd theta(5);
    for(int j=0;j<theta.size();j++){
      theta(j)=distU(generator);
    }
    thetas[i]=theta;
  }
  double integral=0;
  for(int i=0;i<npoints;i++){
    auto hpars=get_hpars(thetas[i]);
    integral+=exp(compute_score(hpars,thetas[i]));
  }
  return integral/npoints;
}

const double Big = -1.e16;


int main(int argc, char **argv){
  generator.seed(5564984);

  //lecture du design QMC full. toutes les clés existent
  map_doe m=read_doe("design_qmc_full.dat");
  //contient le design LHS
  map_doe m_lhs=read_doe("design_lhs_full.dat");
  //lecture des calculs. seules les clés correspondant à des calculs convergés existent.
  map_results mr=read_results_qmc("clean_profile.dat");
  //contiendra les résultats des calculs LHS
  map_results mr_lhs=read_results_lhs("clean_profile.dat");
  //lecture des données expérimentales. clés : X, Alpha, D.
  map_exp me=read_exp_data("clean_exp.dat");


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
    hpars_gp0 << 0.77,1.06,2e-3,1.84,1.46,0.15,0.58; //matern 3 modes

    GP gp0(Kernel_GP_Matern32);
    gp0.SetData(full_data[0]);
    gp0.SetGP(hpars_gp0);

   
    VectorXd hpars_gp1(7);
    hpars_gp1 << 1.22,0.54,2e-3,0.93,0.79,0.18,1.02; //matern 3 modes

    GP gp1(Kernel_GP_Matern32);
    gp1.SetData(full_data[1]);
    gp1.SetGP(hpars_gp1);


    VectorXd hpars_gp2(7);
    hpars_gp2 << 0.93,0.39,2e-3,0.67,0.55,0.16,0.83; //matern 3 modes

    GP gp2(Kernel_GP_Matern32);
    gp2.SetData(full_data[2]);
    gp2.SetGP(hpars_gp2);


    vgp[0]=gp0;
    vgp[1]=gp1;
    vgp[2]=gp2;   


    compute_erreurs_validation(1,m_lhs,mr_lhs,vgp,VP,Acoefs,featureMeans);

    //On recopie tout dans des variables extérieures
    vgp_a=vgp;
    VP_a=VP;
    Acoefs_a=Acoefs;
    featureMeans_a=featureMeans;
  }
  

  /*POD pour le diamètre de bulle*/
  vector<GP> vgp_d(5);
  MatrixXd VP_d;
  MatrixXd Acoefs_d;
  VectorXd featureMeans_d;
  {
    //initialisation des variables
    int nmodes=5;
    int ncalcs=mr.size(); //nombre de calculs réussis
    int nrayons=mr.at(1)[0].size(); //nombre de points de mesure en rayon.
    MatrixXd VP(nrayons,nmodes);
    MatrixXd Acoefs(nmodes,nmodes);
    VectorXd featureMeans(nrayons);
    vector<vector<DATA>> full_data=PerformPCA(m,mr,2,VP,Acoefs,featureMeans,nmodes); //PCA du diamètre

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
    hpars_gp0 << 1.18,1.54,2e-3,0.91,1.13,0.3,0.97; //matern 3 modes

    GP gp0(Kernel_GP_Matern32);
    gp0.SetData(full_data[0]);
    gp0.SetGP(hpars_gp0);


    VectorXd hpars_gp1(7);
    hpars_gp1 << 1.8,0.73,2e-3,0.71,0.827,0.26,1.14; //matern 3 modes

    GP gp1(Kernel_GP_Matern32);
    gp1.SetData(full_data[1]);
    gp1.SetGP(hpars_gp1);


    VectorXd hpars_gp2(7);
    hpars_gp2 << 0.97,0.26,2e-3,0.29,0.37,0.21,0.96; //matern 3 modes

    GP gp2(Kernel_GP_Matern32);
    gp2.SetData(full_data[2]);
    gp2.SetGP(hpars_gp2);


    VectorXd hpars_gp3(7);
    hpars_gp3 << 1.08,0.25,2e-3,0.34,0.37,0.23,0.98;

    GP gp3(Kernel_GP_Matern32);
    gp3.SetData(full_data[3]);
    gp3.SetGP(hpars_gp3);



    VectorXd hpars_gp4(7);
    hpars_gp4 << 0.98,0.20,2e-3,0.38,0.38,0.17,0.69;

    GP gp4(Kernel_GP_Matern32);
    gp4.SetData(full_data[4]);
    gp4.SetGP(hpars_gp4);

    VectorXd hpars_gp5(7);
    hpars_gp5 << 0.89,0.79,1e-3,0.26,1.2,0.69,1.66;

    vgp[0]=gp0;
    vgp[1]=gp1;
    vgp[2]=gp2;
    vgp[3]=gp3;
    vgp[4]=gp4;

    

    compute_erreurs_validation(2,m_lhs,mr_lhs,vgp,VP,Acoefs,featureMeans);
    //On recopie tout dans des variables extérieures
    vgp_d=vgp;
    VP_d=VP;
    Acoefs_d=Acoefs;
    featureMeans_d=featureMeans;
  }
 

  /*Application croisée*/

  {
    //Récupération des données expérimentales sous la forme d'un vector<DATA>. J'ai dû modifier un peu le densities.cpp pour marcher. Nouveautés : 
    // le m_vectorx doit coïncider avec les X du profil. C'est un vecteur<double> maintenant.
    // les données expérimentales doivent toujours être sous la forme d'un vecteur<DATA>. Les X sont de taille 1 et correspondent au vectorx.
    // Kernel est une fonction de deux double pour ne pas se tromper.
    // de même priormean est une fonction d'un double (x) et d'un vectorxd (hpars)
    // en fait il n'y a pas de x expérimental dans l'histoire...


    //interpolation des données expérimentales sur le maillage.
    VectorXd Xexpe=me["X"];
    VectorXd Yexpe_diam=me["D"];
    VectorXd Yexpe_alpha=me["Alpha"];
    VectorXd Xgrid_num=mr[1][0];
    VectorXd Yexpe_interpol_diam=interpolate(Yexpe_diam,Xexpe,Xgrid_num);
    VectorXd Yexpe_interpol_alpha=interpolate(Yexpe_alpha,Xexpe,Xgrid_num);

    //on les range dans un vecteur<AUGDATA>.
    int ndata=Xgrid_num.size(); //nrayons!
    vector<AUGDATA> data_exp_diam(1);
    vector<AUGDATA> data_exp_alpha(1);
    {
      AUGDATA dat; dat.SetX(Xgrid_num); dat.SetValue(Yexpe_interpol_diam);
      data_exp_diam[0]=dat;
      dat.SetValue(Yexpe_interpol_alpha);
      data_exp_alpha[0]=dat;
    }

    //affichage des observations dans un fichier
    ofstream outobs("results/obs.gnu");
    for(int i=0;i<data_exp_alpha[0].GetX().size();i++){
      outobs << data_exp_alpha[0].GetX()(i) << " " << data_exp_alpha[0].Value()(i) << " " << data_exp_diam[0].Value()(i) << endl;
    }

    outobs.close();


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
    VectorXd lb_hpars_alpha(dim_hpars);
    VectorXd ub_hpars_alpha(dim_hpars);
    lb_hpars_alpha << 1e-4,1e-4,1e-4;
    ub_hpars_alpha << 1,1e-1,1e-1;

    VectorXd hpars_z_guess_alpha(dim_hpars);
    hpars_z_guess_alpha << 4.7e-2,2.1e-3,1.4e-3;

    //diam
    VectorXd lb_hpars_diam(dim_hpars);
    VectorXd ub_hpars_diam(dim_hpars);
    lb_hpars_diam << 2e-5,1e-8,5e-4; 
    ub_hpars_diam << 5e-3,3e-5,1e-1;

    VectorXd hpars_z_guess_diam(dim_hpars);
    hpars_z_guess_diam << 3e-4,6.7e-7,6e-3;

    int size_xexpe=data_exp_alpha[0].Value().size();
    //lambda priormean
    auto lambda_priormean_alpha=[size_xexpe](VectorXd const & X, VectorXd const & hpars){
      return VectorXd::Zero(size_xexpe);
    };
    auto lambda_priormean_diam=[size_xexpe](VectorXd const & X, VectorXd const & hpars){
      return VectorXd::Zero(size_xexpe);
    };

    //définition du modèle ! //coefs dans 0-1
    auto lambda_model_diam=[&vgp_d,&VP_d,&featureMeans_d,&Acoefs_d](VectorXd const & X,VectorXd const & theta)->VectorXd{
      return EvaluateMeanGPPCA(vgp_d,theta,VP_d,Acoefs_d,featureMeans_d);
    };

    auto lambda_model_alpha=[&vgp_a,&VP_a,&featureMeans_a,&Acoefs_a](VectorXd const & X,VectorXd const & theta)->VectorXd{
      return EvaluateMeanGPPCA(vgp_a,theta,VP_a,Acoefs_a,featureMeans_a);
    };

    //définition des bornes pour les hGPs.
    int nhpars_gp=7;
    MatrixXd Bounds_hpars_gp(2,nhpars_gp);
    Bounds_hpars_gp(0,0)=1E-4; Bounds_hpars_gp(1,0)=1e4; //variance
    Bounds_hpars_gp(0,2)=1E-4; Bounds_hpars_gp(1,2)=1e4; //sigma obs
    list<int> l={1,3,4,5,6};
    for (int i:l){
      Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=5; //lcors.
    }
    VectorXd hpars_gp_guess(7);
    for (int i=0;i<nhpars_gp;i++){
      hpars_gp_guess(i)=0.5*(Bounds_hpars_gp(1,i)+Bounds_hpars_gp(0,i));
    }
    hpars_gp_guess(0)=1; //var edm
    hpars_gp_guess(2)=1e-3; //var obs


    // pour la MCMC
    MatrixXd COV_init=pow(0.1,2)*MatrixXd::Identity(5,5);
    cout << "COV_init : " << endl << COV_init << endl;
    VectorXd Xinit(5);
    Xinit << 0.5,0.5,0.5,0.5,0.5;

    int nombre_steps_mcmc=1e6;
    int nombre_samples_collected=3000;
    int nautocor=5000;

    //construction du grid
    int npts_init=2000;
    DoE doe_init(lb_t,ub_t,npts_init,1);
    //afichage du grid qmc
    doe_init.WriteGrid("results/save/grid.gnu");

    //instance de base de densité pour alpha
    Density MainDensity_alpha(doe_init);
    MainDensity_alpha.SetLogPriorPars(logprior_pars);
    MainDensity_alpha.SetLogPriorHpars(logprior_hpars);
    MainDensity_alpha.SetKernel(Kernel_Z_Matern12);
    MainDensity_alpha.SetModel(lambda_model_alpha);
    MainDensity_alpha.SetPriorMean(lambda_priormean_alpha);
    MainDensity_alpha.SetHparsBounds(lb_hpars_alpha,ub_hpars_alpha);
    MainDensity_alpha.SetDataExp(data_exp_alpha);
    MainDensity_alpha.SetXprofile(Xgrid_num);

    //instance de base de densité pour diam
    Density MainDensity_diam(doe_init);
    MainDensity_diam.SetLogPriorPars(logprior_pars);
    MainDensity_diam.SetLogPriorHpars(logprior_hpars);
    MainDensity_diam.SetKernel(Kernel_Z_Matern12);
    MainDensity_diam.SetModel(lambda_model_diam);
    MainDensity_diam.SetPriorMean(lambda_priormean_diam);
    MainDensity_diam.SetHparsBounds(lb_hpars_diam,ub_hpars_diam);
    MainDensity_diam.SetDataExp(data_exp_diam);
    MainDensity_diam.SetXprofile(Xgrid_num);


    auto in_bounds=[&MainDensity_alpha,&MainDensity_diam](VectorXd const & X){
      if(X.size()==5){
        return MainDensity_alpha.in_bounds_pars(X);
      }
      if(X.size()==11){
        VectorXd Xmid(3);
        Xmid << X(5),X(6),X(7);
        return MainDensity_alpha.in_bounds_pars(X.head(5)) && MainDensity_diam.in_bounds_hpars(X.tail(3)) && MainDensity_alpha.in_bounds_hpars(Xmid);
      }
      cerr << "nonconform X size" << endl;
      return false;
    };

    // Calcul des hpars KOH avec les deux fonctions de vraisemblance.
    VectorXd hpars_double_guess(6);
    hpars_double_guess << hpars_z_guess_alpha(0),hpars_z_guess_alpha(1),hpars_z_guess_alpha(2),hpars_z_guess_diam(0),hpars_z_guess_diam(1),hpars_z_guess_diam(2);
    VectorXd lb_hpars_double(6);
    lb_hpars_double << lb_hpars_alpha(0),lb_hpars_alpha(1),lb_hpars_alpha(2),lb_hpars_diam(0),lb_hpars_diam(1),lb_hpars_diam(2);
    VectorXd ub_hpars_double(6);
    ub_hpars_double << ub_hpars_alpha(0),ub_hpars_alpha(1),ub_hpars_alpha(2),ub_hpars_diam(0),ub_hpars_diam(1),ub_hpars_diam(2);

    //phase KOH
  /*
    {
    cout << "hpars koh double (guess): " << hpars_double_guess.transpose() << endl;
    auto begin=chrono::steady_clock::now();
    

    VectorXd hparskohdouble(6);
    //hparskohdouble << 0.098,0.0020,0.0018,0.000364,7.4e-7,0.0045;
    hparskohdouble=HparsKOH_double(hpars_double_guess,lb_hpars_double,ub_hpars_double,MainDensity_alpha,MainDensity_diam,data_exp_alpha,data_exp_diam);
    cout << "hpars koh double after optimisation : " << hparskohdouble.transpose() << endl;
      cout << endl << "Début KOH" << endl;
      VectorXd hparskoh_alpha=hparskohdouble.head(3);
      VectorXd hparskoh_diam=hparskohdouble.tail(3);
      cout << "hpars koh alpha : " << hparskoh_alpha.transpose() << endl;
      cout << "hpars koh diam : " << hparskoh_diam.transpose() << endl;

      auto get_hpars_koh=[&hparskoh_alpha,&hparskoh_diam](VectorXd const & X){
        auto p=make_pair(hparskoh_alpha,hparskoh_diam);
        return p;
      };

      //calcul des LDLT pour KOH.
      //à mon avis ces 2 vecteurs sont les mêmes, mais bon
      const vector<VectorXd> *xconv_alpha=MainDensity_alpha.GetXconverted();
      const vector<VectorXd> *xconv_diam=MainDensity_diam.GetXconverted();
      MatrixXd G1=MainDensity_alpha.Gamma(*xconv_alpha,hparskoh_alpha);
      MatrixXd G2=MainDensity_diam.Gamma(*xconv_diam,hparskoh_diam);
      LDLT<MatrixXd> LDLT1(G1);
      LDLT<MatrixXd> LDLT2(G2);

      auto compute_score_koh=[&MainDensity_diam,&MainDensity_alpha,&data_exp_alpha,&data_exp_diam,&LDLT1,&LDLT2,&hparskoh_alpha,&hparskoh_diam](pair<VectorXd,VectorXd> p, VectorXd const &X){
        //on ne se sert pas de p car on connaît déjà les hpars.
        double ll1=MainDensity_alpha.loglikelihood_theta_fast(X,hparskoh_alpha,LDLT1);
        double ll2=MainDensity_diam.loglikelihood_theta_fast(X,hparskoh_diam,LDLT2);
        double lp=logprior_pars(X);
        return ll1+ll2+lp;
      };


      //run de la MCMC

      auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,Xinit,COV_init,compute_score_koh,get_hpars_koh,in_bounds,generator);

      auto end=chrono::steady_clock::now();
      cout << "Temps pour KOH : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << endl;

      vector<VectorXd> samples_koh=get<0>(res);
      vector<VectorXd> allsamples_koh=get<2>(res);
      vector<pair<VectorXd,VectorXd>> vhpars_koh=get<1>(res);
      vector<VectorXd> vhpars_koh_alpha(vhpars_koh.size());
      vector<VectorXd> vhpars_koh_diam(vhpars_koh.size());
      for(int i=0;i<vhpars_koh.size();i++){
        auto p=vhpars_koh[i];
        vhpars_koh_alpha[i]=p.first;
        vhpars_koh_diam[i]=p.second;
      }

      //Set des samples
      MainDensity_alpha.SetNewSamples(samples_koh);
      MainDensity_diam.SetNewSamples(samples_koh);    
      MainDensity_alpha.SetNewAllSamples(allsamples_koh);
      MainDensity_diam.SetNewAllSamples(allsamples_koh);
      MainDensity_alpha.SetNewHparsOfSamples(vhpars_koh_alpha);
      MainDensity_diam.SetNewHparsOfSamples(vhpars_koh_diam);

      //diagnostic
      MainDensity_alpha.Autocor_diagnosis(nautocor,"results/diag/autocorkoh.gnu");

      //écriture des samples
      MainDensity_alpha.WriteSamples("results/save/sampkohalpha.gnu");
      MainDensity_diam.WriteSamples("results/save/sampkohdiam.gnu"); 

      //prédictions
      MainDensity_alpha.WritePredictionsF(Xgrid_num,"results/preds/predskohalphaF.gnu");
      MainDensity_diam.WritePredictionsF(Xgrid_num,"results/preds/predskohdiamF.gnu");
      MainDensity_alpha.WritePredictions(Xgrid_num,"results/preds/predskohalpha.gnu");
      MainDensity_diam.WritePredictions(Xgrid_num,"results/preds/predskohdiam.gnu");
      auto end2=chrono::steady_clock::now();
      cout << "Temps pour prédictions KOH : " << chrono::duration_cast<chrono::seconds>(end2-end).count() << endl;
    }


    exit(0);
*/

    ///phase opti avec choix de points de training
      
    {
      cout << "début double calibration opti avec nsteps =" << nombre_steps_mcmc <<endl;
      auto begin=chrono::steady_clock::now();
      DensityOpt DensOpt_alpha(MainDensity_alpha);     
      DensityOpt DensOpt_diam(MainDensity_diam);

      DoE doe_light(lb_t,ub_t,50,1); // DoE light de 50 points.

      DensOpt_alpha.SetNewDoE(doe_light);
      DensOpt_diam.SetNewDoE(doe_light);

      DensOpt_alpha.Compute_optimal_hpars(0.5);
      DensOpt_diam.Compute_optimal_hpars(0.5);

      DensOpt_alpha.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
      DensOpt_alpha.opti_allgps(hpars_gp_guess);
      //DensOpt_alpha.Test_hGPs();

      DensOpt_diam.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
      DensOpt_diam.opti_allgps(hpars_gp_guess);
      //DensOpt_diam.Test_hGPs();

   
      

      auto get_hpars_opti=[&DensOpt_alpha,&DensOpt_diam](VectorXd const & X){
        VectorXd hparsopt_alpha=DensOpt_alpha.EvaluateHparOpt(X);
        VectorXd hparsopt_diam=DensOpt_diam.EvaluateHparOpt(X);
        auto p=make_pair(hparsopt_alpha,hparsopt_diam);
        return p;
      };
        
            
      auto compute_score_opti=[&data_exp_alpha,&data_exp_diam,&DensOpt_alpha,&DensOpt_diam](pair<VectorXd,VectorXd> p, VectorXd const &X){
        double ll1=DensOpt_alpha.loglikelihood_theta(X,p.first);
        double ll2=DensOpt_diam.loglikelihood_theta(X,p.second);
        double lp=logprior_pars(X);
        return ll1+ll2+lp;
      };

      //phase de rajout d'hyperparamètres par MCMC, avec sélection par la variance de prédiction.
      
      //test de la performance du grid initial.

      auto add_npoints=[&DensOpt_diam,&DensOpt_alpha,&nombre_steps_mcmc,&Xinit,&COV_init,&compute_score_opti,&get_hpars_opti,&in_bounds,&Bounds_hpars_gp,&hpars_gp_guess](int npoints,default_random_engine &generator, int & npts_total, int nsamples_mcmc){
          //ajout de n points obtenus à partir d'une MCMC.
          npts_total+=npoints;
          auto begin=chrono::steady_clock::now();
          //récupération de nsamples_mcmc points de la MCMC.
          auto res=Run_MCMC(nombre_steps_mcmc,nsamples_mcmc,Xinit,COV_init,compute_score_opti,get_hpars_opti,in_bounds,generator);
          auto begin_hgps=chrono::steady_clock::now();
          vector<VectorXd> samples_opti=get<0>(res);
          vector<double> scores_of_samples=get<3>(res);
          //calcul du critère en chaque point.
          vector<VectorXd> selected_thetas;
          vector<double> weights(samples_opti.size());
          for(int j=0;j<weights.size();j++){
            double a=DensOpt_alpha.EstimatePredError(samples_opti[j]);
            double b=DensOpt_diam.EstimatePredError(samples_opti[j]);
            weights[j]=a+b;
          }
          for(int i=0;i<npoints;i++){
            std::discrete_distribution<int> distribution(weights.begin(), weights.end());
            int drawn = distribution(generator);
            weights[drawn]=0;
            selected_thetas.push_back(samples_opti[drawn]);
          }
          //simple check
          cout << "rajout de " << selected_thetas.size() << " points :" << endl;
          //rajout des points 
          DensOpt_alpha.update_hGPs_noPCA(selected_thetas,Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess,0.5);
          DensOpt_diam.update_hGPs_noPCA(selected_thetas,Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess,0.5);
          DensOpt_alpha.opti_allgps(hpars_gp_guess);
          DensOpt_diam.opti_allgps(hpars_gp_guess);
          auto end_hgps=chrono::steady_clock::now();
        };

      int npts_total=npts_init;
      //rajout de 200 points
      add_npoints(200,generator,npts_total,4000);
      //rerajout de 200 points
      add_npoints(200,generator,npts_total,4000);
      
        

      //calibration
      
      auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,Xinit,COV_init,compute_score_opti,get_hpars_opti,in_bounds,generator);
      
      auto end=chrono::steady_clock::now();
      cout << "Temps pour OPT : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << endl;
      
      
      
      auto samples_opti=get<0>(res);
      vector<VectorXd> allsamples_opti=get<2>(res);
      vector<pair<VectorXd,VectorXd>> vhpars_opti=get<1>(res);
      vector<VectorXd> vhpars_opti_alpha(vhpars_opti.size());
      vector<VectorXd> vhpars_opti_diam(vhpars_opti.size());
      for(int i=0;i<vhpars_opti.size();i++){
        auto p=vhpars_opti[i];
        vhpars_opti_alpha[i]=p.first;
        vhpars_opti_diam[i]=p.second;
      }
        //Set des samples
        DensOpt_alpha.SetNewSamples(samples_opti);
        DensOpt_diam.SetNewSamples(samples_opti);
        DensOpt_alpha.SetNewAllSamples(allsamples_opti);
        DensOpt_diam.SetNewAllSamples(allsamples_opti);
        DensOpt_alpha.SetNewHparsOfSamples(vhpars_opti_alpha);
        DensOpt_diam.SetNewHparsOfSamples(vhpars_opti_diam);

        //diagnostic
        DensOpt_alpha.Autocor_diagnosis(nautocor,"results/diag/autocoropt+.gnu");

        //écriture des samples
        DensOpt_alpha.WriteSamples("results/save/sampopt+alpha.gnu");
        DensOpt_diam.WriteSamples("results/save/sampopt+diam.gnu");

        DensOpt_alpha.WriteMCMCSamples("results/save/allsampopt+alpha.gnu");
        DensOpt_diam.WriteMCMCSamples("results/save/allsampopt+diam.gnu");  

        //prédictions
        DensOpt_alpha.WritePredictionsF(Xgrid_num,"results/preds/predsopt+alphaF.gnu");
        DensOpt_diam.WritePredictionsF(Xgrid_num,"results/preds/predsopt+diamF.gnu");

        DensOpt_alpha.WritePredictions(Xgrid_num,"results/preds/predsopt+alpha.gnu");
        DensOpt_diam.WritePredictions(Xgrid_num,"results/preds/predsopt+diam.gnu");


      //le calcul de l'évidence doit se faire quand les hpars sont les plus précis possibles.
     
    
      //test des 200 points sur l'échantillon obtenu.
      
        auto thetas_ref=DensOpt_alpha.GetSample();
        cout << thetas_ref.size() << endl;
        thetas_ref.erase(thetas_ref.begin(),thetas_ref.begin()+2800);
        vector<VectorXd> halpha_ref(thetas_ref.size());
        vector<VectorXd> hdiam_ref(thetas_ref.size());
        for(int i=0;i<thetas_ref.size();i++){
          halpha_ref[i]=DensOpt_alpha.HparsOpt(thetas_ref[i],hpars_z_guess_alpha,1);
          hdiam_ref[i]=DensOpt_diam.HparsOpt(thetas_ref[i],hpars_z_guess_diam,1);
        }
        //cout << DensOpt_alpha.Test_hGPs_on_sample(thetas_ref,halpha_ref).transpose() << endl;
        //cout << DensOpt_diam.Test_hGPs_on_sample(thetas_ref,hdiam_ref).transpose() << endl;
      
    }

    exit(0);
  


     /// phase full bayes
    
    {
        cout << "début double calibration full bayes" << endl;

      auto begin=chrono::steady_clock::now();

        auto get_hpars_fb=[](VectorXd const & X){
          //non utilisée
          auto p=make_pair(VectorXd::Random(3),VectorXd::Random(3));
          return p;
        };

        auto compute_score_fb=[&MainDensity_diam,&MainDensity_alpha](pair<VectorXd,VectorXd> p, VectorXd const &X){
          //on ne se sert pas de p ici
          VectorXd theta=X.head(5);
          VectorXd hpars_diam=X.tail(3);
          VectorXd hpars_alpha(3);
          hpars_alpha << X(5),X(6),X(7);
          double ll1=MainDensity_alpha.loglikelihood_theta(theta,hpars_alpha);
          double ll2=MainDensity_diam.loglikelihood_theta(theta,hpars_diam);
          double lp1=logprior_hpars(hpars_alpha);
          double lp2=logprior_hpars(hpars_diam);
          return ll1+ll2+lp1+lp2;
        };

        //paramètres initiaux de la MCMC
        VectorXd Xinit_fb(11);
        Xinit_fb << Xinit(0),Xinit(1),Xinit(2),Xinit(3),Xinit(4),hpars_z_guess_alpha(0),hpars_z_guess_alpha(1),hpars_z_guess_alpha(2),hpars_z_guess_diam(0),hpars_z_guess_diam(1),hpars_z_guess_diam(2);

        MatrixXd COV_init_fb=MatrixXd::Identity(11,11);
        COV_init_fb.topLeftCorner(5,5)=0.5*COV_init;
        COV_init_fb(5,5)*=pow(8e-2,2); //sedm bougé
        COV_init_fb(6,6)*=pow(1e-4,2); //sobs
        COV_init_fb(7,7)*=pow(5e-4,2); //lcor 
        COV_init_fb(8,8)*=pow(4e-4,2); //sedm avant 1e-7
        COV_init_fb(9,9)*=pow(4e-8,2); //sobs avant 1e-8
        COV_init_fb(10,10)*=pow(2e-3,2); //lcor avant 1e-3

        COV_init_fb*=0.5;

        cout << "COV init fb : " << endl << COV_init_fb << endl;

        //run de la MCMC
        cout << "nombre steps mcmc full bayes :" << 8*nombre_steps_mcmc << endl;

      auto res=Run_MCMC(8*nombre_steps_mcmc,nombre_samples_collected,Xinit_fb,COV_init_fb,compute_score_fb,get_hpars_fb,in_bounds,generator);
      auto end=chrono::steady_clock::now();
      cout << "Temps pour BAYES : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << endl;
        //diviser les samples en theta et hpars
        vector<VectorXd> rough_samples_fb=get<0>(res);
        int nsamples_fb=rough_samples_fb.size();
        vector<VectorXd> allsamples_fb=get<2>(res);
        vector<VectorXd> clean_samples_fb(nsamples_fb);
        vector<VectorXd> vhpars_fb_alpha(nsamples_fb);
        vector<VectorXd> vhpars_fb_diam(nsamples_fb);

        for(int i=0;i<nsamples_fb;i++){
          VectorXd X=rough_samples_fb[i];
          VectorXd theta=X.head(5);
          VectorXd hpars_diam=X.tail(3);
          VectorXd hpars_alpha(3);
          hpars_alpha << X(5),X(6),X(7);
          clean_samples_fb[i]=theta;
          vhpars_fb_alpha[i]=hpars_alpha;
          vhpars_fb_diam[i]=hpars_diam;
        }

        //Set des samples
        MainDensity_alpha.SetNewSamples(clean_samples_fb);
        MainDensity_diam.SetNewSamples(clean_samples_fb);
        MainDensity_alpha.SetNewAllSamples(allsamples_fb);
        MainDensity_diam.SetNewAllSamples(allsamples_fb);
        MainDensity_alpha.SetNewHparsOfSamples(vhpars_fb_alpha);
        MainDensity_diam.SetNewHparsOfSamples(vhpars_fb_diam);

        //diagnostic
        MainDensity_alpha.Autocor_diagnosis(nautocor,"results/diag/autocorfb.gnu");

        //écriture des samples
        MainDensity_alpha.WriteSamples("results/save/sampfbalpha.gnu");
        MainDensity_diam.WriteSamples("results/save/sampfbdiam.gnu"); 
        MainDensity_alpha.WriteMCMCSamples("results/save/allsampfbalpha.gnu");
        MainDensity_diam.WriteMCMCSamples("results/save/allsampfbdiam.gnu");



        //prédictions
        MainDensity_alpha.WritePredictionsF(Xgrid_num,"results/preds/predsfbalphaF.gnu");
        MainDensity_diam.WritePredictionsF(Xgrid_num,"results/preds/predsfbdiamF.gnu");
        MainDensity_alpha.WritePredictions(Xgrid_num,"results/preds/predsfbalpha.gnu");
        MainDensity_diam.WritePredictions(Xgrid_num,"results/preds/predsfbdiam.gnu");

    }

  exit(0);


    
   



  ///phase opti avec gros grid QMC de 1500 points.
    {
      cout << "début double calibration opti avec gros QMC nsteps =" << nombre_steps_mcmc <<endl;
      DensityOpt DensOpt_alpha(MainDensity_alpha);     
      DensityOpt DensOpt_diam(MainDensity_diam);

      //construction d'un grid de vérification pour les hGPs, en alpha et diam.
      DoE doe_verif_hpars(lb_t,ub_t,500,1);
      vector<VectorXd> thetas_verif=doe_verif_hpars.GetGrid();
      vector<VectorXd> halpha_verif(thetas_verif.size());
      vector<VectorXd> hdiam_verif(thetas_verif.size());
      for(int i=0;i<thetas_verif.size();i++){
        halpha_verif[i]=DensOpt_alpha.HparsOpt(thetas_verif[i],hpars_z_guess_alpha,0.5);
        hdiam_verif[i]=DensOpt_diam.HparsOpt(thetas_verif[i],hpars_z_guess_diam,0.5);
      }


      DoE doe_new_opti(lb_t,ub_t,1500,1000);
      DensOpt_alpha.SetNewDoE(doe_new_opti);
      DensOpt_diam.SetNewDoE(doe_new_opti);
      
      DensOpt_alpha.Compute_optimal_hpars(1);
      DensOpt_diam.Compute_optimal_hpars(1);

      DensOpt_alpha.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
      DensOpt_alpha.opti_allgps(hpars_gp_guess);
      cout << DensOpt_alpha.Test_hGPs_on_sample(thetas_verif,halpha_verif).transpose() << endl;

      DensOpt_diam.BuildHGPs_noPCA(Kernel_GP_Matern32,Bounds_hpars_gp,hpars_gp_guess);
      DensOpt_diam.opti_allgps(hpars_gp_guess);
      cout << DensOpt_diam.Test_hGPs_on_sample(thetas_verif,halpha_verif).transpose() << endl;
      

      auto get_hpars_opti=[&DensOpt_alpha,&DensOpt_diam](VectorXd const & X){
        VectorXd hparsopt_alpha=DensOpt_alpha.EvaluateHparOpt(X);
        VectorXd hparsopt_diam=DensOpt_diam.EvaluateHparOpt(X);
        auto p=make_pair(hparsopt_alpha,hparsopt_diam);
        return p;
      };
        
            
      auto compute_score_opti=[&data_exp_alpha,&data_exp_diam,&DensOpt_alpha,&DensOpt_diam](pair<VectorXd,VectorXd> p, VectorXd const &X){
        double ll1=DensOpt_alpha.loglikelihood_theta(X,p.first);
        double ll2=DensOpt_diam.loglikelihood_theta(X,p.second);
        double lp=logprior_pars(X);
        return ll1+ll2+lp;
      };
     
      //calibration
      {
        auto res=Run_MCMC(nombre_steps_mcmc,nombre_samples_collected,Xinit,COV_init,compute_score_opti,get_hpars_opti,in_bounds,generator);
        auto samples_opti=get<0>(res);
        vector<VectorXd> allsamples_opti=get<2>(res);
        vector<pair<VectorXd,VectorXd>> vhpars_opti=get<1>(res);
        vector<VectorXd> vhpars_opti_alpha(vhpars_opti.size());
        vector<VectorXd> vhpars_opti_diam(vhpars_opti.size());
        for(int i=0;i<vhpars_opti.size();i++){
          auto p=vhpars_opti[i];
          vhpars_opti_alpha[i]=p.first;
          vhpars_opti_diam[i]=p.second;
        }
        //Set des samples
        MainDensity_alpha.SetNewSamples(samples_opti);
        MainDensity_diam.SetNewSamples(samples_opti);
        MainDensity_alpha.SetNewAllSamples(allsamples_opti);
        MainDensity_diam.SetNewAllSamples(allsamples_opti);
        MainDensity_alpha.SetNewHparsOfSamples(vhpars_opti_alpha);
        MainDensity_diam.SetNewHparsOfSamples(vhpars_opti_diam);

        //diagnostic
        MainDensity_alpha.Autocor_diagnosis(nautocor,"results/diag/autocoropt.gnu");

        //écriture des samples
        MainDensity_alpha.WriteSamples("results/save/sampoptalpha.gnu");
        MainDensity_diam.WriteSamples("results/save/sampoptdiam.gnu"); 

        //prédictions
        MainDensity_alpha.WritePredictionsF(Xgrid_num,"results/preds/predsoptalphaF.gnu");
        MainDensity_diam.WritePredictionsF(Xgrid_num,"results/preds/predsoptdiamF.gnu");

        MainDensity_alpha.WritePredictions(Xgrid_num,"results/preds/predsoptalpha.gnu");
        MainDensity_diam.WritePredictions(Xgrid_num,"results/preds/predsoptdiam.gnu");



      }         
    }
  exit(0);
  }
}