// Dans ce fichier : on lit les données du calcul et on les post-traite à notre souhait.
//Application croisée : les samples alpha pour une prédiction sur le diamètre, et les samples diam pour une prédiction sur alpha.


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

vector<int> KMEANS(vector<DATA> const &data, int const nk){
  /* Cluster the vector of data into nk sets */
  int ndat = data.size();
  int ndim = data[0].GetX().rows();
  std::vector<int> Memb(ndat); /* Memberships */
  if(nk>=data.size()){
    std::cout << "You ask for " << nk << " clusters for " << data.size() << " data \n";
    return Memb;
  }

  Eigen::MatrixXd Loc(ndim, ndat); /* Data Points */
  for (int id = 0; id < ndat; id++)
    Loc.col(id) = data[id].GetX();
  for (int id = 0; id < ndim; id++)
  {
    double min_d = Loc.row(id).minCoeff();
    double max_d = Loc.row(id).maxCoeff();
    for (int d = 0; d < ndat; d++)
      Loc(id, d) = (Loc(id, d) - min_d) / (max_d - min_d);
  }

  Eigen::MatrixXd Cent(ndim, nk); /* Initialize Centroids  */
  for (int ic = 0; ic < nk; ic++)
    Cent.col(ic) = Loc.col(ic);


  Eigen::VectorXd dist(nk);
  for (int id = 0; id < ndat; id++)
  {
    for (int ic = 0; ic < nk; ic++)
    {
      dist(ic) = (Loc.col(id) - Cent.col(ic)).squaredNorm();
    }
    dist.minCoeff(&Memb[id]);
  }

  int Chgt = 1;
  while (Chgt > 0)
  {
    /* Set centroids */
    for (int ic = 0; ic < nk; ic++)
    {
      int nc = 0;
      Eigen::VectorXd Xc = Eigen::VectorXd::Zero(ndim);
      for (int id = 0; id < ndat; id++)
      {
        if (Memb[id] == ic)
        {
          Xc += Loc.col(id);
          nc++;
        }
      }
      Cent.col(ic) = Xc / (double)(nc);
    }
    /* Set Memberships */
    Chgt = 0;
    for (int id = 0; id < ndat; id++)
    {
      for (int ic = 0; ic < nk; ic++)
      {
        dist(ic) = (Loc.col(id) - Cent.col(ic)).squaredNorm();
      }
      int memb;
      dist.minCoeff(&memb);
      if (memb != Memb[id])
      {
        Memb[id] = memb;
        Chgt++;
      }
    }
    //std::cout << "Changes in memberships " << Chgt << std::endl;
  }
  return Memb;
};

VectorXd linfit(vector<DATA> const &data){
  //renvoie les coefficients de la régression linéaire du vecteur data. le vecteur de données doit être centré sinon la régression linéaire sera mauvaise.
  int dimx=data[0].GetX().size();
  //extraction des data
  VectorXd Y(data.size());
  for (int i=0;i<data.size();i++){
    Y(i)=data[i].Value();
  }
  //construction de la matrice de régression
  MatrixXd X(data.size(),dimx);
  for (int i=0;i<data.size();i++){
    VectorXd Pos=data[i].GetX();
    for(int j=0;j<dimx;j++){
      X(i,j)=Pos(j);
    }
  }
  //résolution par SVD
  VectorXd coefs=X.bdcSvd(ComputeThinU | ComputeThinV).solve(Y);
  return coefs;
}

vector<VectorXd> get_gradients(vector<DATA> const &data, int nk){
  //divise le dataset en nk clusters, et récupère le vecteur gradient pour chacun des clusters.
  vector<int> appartenances=KMEANS(data,nk);
  vector<VectorXd> gradients(nk);
  for(int c=0;c<nk;c++){
    vector<DATA> data_in_clusters;
    //remplissage du data_in_clusters
    for(int i=0;i<data.size();i++){
      if(appartenances[i]==c){ data_in_clusters.push_back(data[i]);}
    }
    //cout << "cluster " << c << " contient " << data_in_clusters.size() << " data." << endl;
    VectorXd grad=linfit(data_in_clusters);
    //cout << "son gradient : " << grad.transpose() << endl;
    gradients[c]=grad;
  }
  return gradients;
}

MatrixXd CovMatGradients(vector<VectorXd> const &v){
  //calcule la matrice de covariance de l'ensemble des vecteurs
  int dimx=v[0].size();
  //cout << "dimx :" << dimx << endl;
  VectorXd Mean=VectorXd::Zero(dimx);
  for(VectorXd const &x:v){
   // cout << "yo" << x.transpose() <<endl;
    Mean+=x;
  }
  Mean/=v.size();
  MatrixXd Cov=MatrixXd::Zero(dimx,dimx);
  for(VectorXd const &x:v){
    VectorXd xc=x-Mean;
    Cov+=xc*xc.transpose();
  }
  Cov/=v.size();
  return Cov;
}

void principal_directions(vector<DATA> const &data, int const nk){
  //nk : nombre de clusters
  //on centre les données dans un autre vector<DATA>.
  vector<DATA> centered_data(data.size());
  double mean=0;
  for(DATA const &d:data){
    mean+=d.Value();
  }
  mean/=data.size();
  for(int i=0;i<data.size();i++){
    DATA dat; dat.SetX(data[i].GetX()); dat.SetValue(data[i].Value()-mean);
    centered_data[i]=dat;
  }
  //affiche les directions principales du dataset
  vector<VectorXd> gradients=get_gradients(centered_data,nk);
  MatrixXd M=CovMatGradients(gradients);
  SelfAdjointEigenSolver<MatrixXd> eig(M);
  //sorted eigenvalues : 
  //cout << "sorted ev :" << eig.eigenvalues().transpose().rightCols(2) << endl;
  //sorted eigenvectors : 
  //cout << "sorted vec : " <<  eig.eigenvectors() << endl;
  //best Ndim basis : 
  MatrixXd EV=eig.eigenvalues().transpose().rightCols(2); //(1 ligne, 2 colonnes)
  cout << "principal direction : " << eig.eigenvectors().rightCols(1).transpose() << endl;
  cout << "first 2 ev : " << EV << ", rapport : " << EV(0,1)/EV(0,0) << endl;
}

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
  return pow(hpar(0),2)*(1+(d/hpar(2))+0.33*pow(d/hpar(2),2))*exp(-d/hpar(2)); //5/2
}

double Kernel_GP_SE(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double cor=pow(hpar(0),2);
  cor*=exp(-0.5*pow((x(0)-y(0))/hpar(1),2)); //phi
  cor*=exp(-0.5*pow((x(1)-y(1))/hpar(3),2)); //BK
  cor*=exp(-0.5*pow((x(2)-y(2))/hpar(4),2)); //COAL
  cor*=exp(-0.5*pow((x(3)-y(3))/hpar(5),2)); //NUCL
  cor*=exp(-0.5*pow((x(4)-y(4))/hpar(6),2)); //MT
  return cor;
}

double Kernel_GP_Matern(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  double cor=pow(hpar(0),2);
  cor*=exp(-0.5*abs(x(0)-y(0))/hpar(1)); //phi
  cor*=exp(-0.5*abs(x(1)-y(1))/hpar(3)); //BK
  cor*=exp(-0.5*abs(x(2)-y(2))/hpar(4)); //COAL
  cor*=exp(-0.5*abs(x(3)-y(3))/hpar(5)); //NUCL
  cor*=exp(-0.5*abs(x(4)-y(4))/hpar(6)); //MT
  return cor;
}



double logprior_hpars(VectorXd const &hpars){
  //edm, exp, lcor
  return -2*log(hpars(0));
}

double logprior_pars(VectorXd const &pars){
  //prior uniforme sur les pramètres
  return 0;
}

double PriorMean(VectorXd const & X, VectorXd const &hpars){
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

double ecart_minmax(VectorXd const &D1, VectorXd const &X1, VectorXd const &D2, VectorXd const &X2, double Xmin, double Xmax){
  //renvoie la norme L2 entre les fonctions D1 et D2, définies sur les maillages X1 et X2. Maillage de référence : X1. On se restreint à l'intervalle Xmin - Xmax.
  //simple test de taille
  if(!D1.size()==X1.size() || !D2.size()==X2.size()){cerr << "différence de taille entre vecteurs" << endl;}
  if(Xmin >= Xmax){cerr << "min et max inversés" << endl;}
  int size1=X1.size();
  int size2=X2.size();
  int istart=0;
  int iend=size1-1;
  //détermination des indices istart et iend
  for (int i=0;i<size1;i++){
    if(X1(i)<Xmin){istart=i;}
    else{
      break;}
  }

  for (int i=size1-1;i>=0;i--){
    if(X1(i)>Xmax){iend=i;}
    else {break;}
  } 
  //cout << "istart et iend" << istart << " " << iend << endl;

  //calcul de l'erreur l2
  double l2=0;
  for(int i=istart;i<=iend;i++){
    //interpolation de D2 sur le grid X1.
    double xvalue=X1(i);
    double d2value;
    if(X2(0)>xvalue){d2value=D2(0);}
    else if(X2(size2-1)<xvalue){d2value=D2(size2-1);}
    else{
      //la valeur se trouve entre deux X2.
      int j=0;
      while(X2(j)<xvalue){
        j++;
      }
      j--;
      //interpolation linéaire
      double m=(D2(j+1)-D2(j))/(X2(j+1)-X2(j));
      double b=D2(j)-m*X2(j);
      d2value=m*xvalue+b;
    }
    l2+=pow(D1(i)-d2value,2);
  }
  l2/=(iend-istart+1);
  return l2;

}

double ecart(VectorXd const &D1, VectorXd const &X1, VectorXd const &D2, VectorXd const &X2){
  //renvoie la norme L2 entre les fonctions D1 et D2, définies sur les maillages X1 et X2. Maillage de référence : X1. On suppose un ordonnement des X.
  double Xmin=X1(0);
  double Xmax=X1(X1.size()-1);
  return ecart_minmax(D1,X1,D2,X2,Xmin,Xmax);
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

double l2_diam(map_results const & mr, map_exp const & me, int i){
  //renvoie la norme l2 de l'écart de diamètre du calcul i.
  return ecart(me.at("D"),me.at("X"),mr.at(i)[2],mr.at(i)[0]);
}

double l2_diam_minmax(map_results const & mr, map_exp const & me, int i,double xmin, double xmax){
  return ecart_minmax(me.at("D"),me.at("X"),mr.at(i)[2],mr.at(i)[0],xmin,xmax);
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

double tdv_moyen_given_vari(GP & gp, int indice, double var, default_random_engine & generator){
  //calcul du taux de vide moyen pour une autre variable donnée.
  //selon l'indice : 0= phi, 1= BK, 2=COAL, 3=NUCL, 4=MT.
  //attention var doit être en coord gp (de 0 à 1)
  double tdv=0;
  VectorXd X(5);
  X(indice)=var;
  set<int,greater<int>> s={0,1,2,3,4};
  s.erase(indice);
  int nrepet=1000;
  for (int i=0;i<nrepet;i++){
    for (int j:s){
      X(j)=distU(generator);
    }
    tdv+=gp.EvalMean(X);
  }
  tdv/=nrepet;
  return tdv;
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
    MatrixXd MeanVar=EvaluateMeanVarGPPCA(vgp,ParamEval,VP,Acoefs,featureMeans);
    M_predicted.col(i)=MeanVar.col(0); //on prend seulement les prédictions moyennes.
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

  double diam_xmin=3E-3;
  double diam_xmax=me["X"](48);
  cout << l2_diam(mr,me,1) << " " << l2_diam(mr,me,2) <<endl;
  cout << l2_diam_minmax(mr,me,1,diam_xmin,diam_xmax) << " " << l2_diam_minmax(mr,me,2,diam_xmin,diam_xmax) << endl;

  ofstream file("results/stats.gnu");
  file << "# tp, tc, tm, dp, dc\n";
  for (auto &v:mr){
    double tp=tauxparoi(v.second);
    double tc=tauxcoeur(v.second);
    double tm=tauxmoyen(v.second);
    double dp=diamparoi(v.second);
    double dc=diamcoeur(v.second);
    double l2diamfull=ecart(me.at("D"),me.at("X"),v.second[2],v.second[0]); //erreur L2 sur les diamètres
    double l2diampartial=ecart_minmax(me.at("D"),me.at("X"),v.second[2],v.second[0],diam_xmin,diam_xmax);
    double l2V1=ecart(me.at("V"),me.at("X"),v.second[3],v.second[0]); //erreur L2 sur la vitesse V1
    double l2V2=ecart(me.at("V"),me.at("X"),v.second[4],v.second[0]); //erreur L2 sur la vitesse V2
    int calcnr=v.first;
    file << tp << " " << tc << " " << tm << " "<< dp << " "<< dc << " "
     << calcnr << " " << l2diamfull << " " << l2diampartial << " " 
     << l2V1 << " " << l2V2 << " "
     <<m[calcnr].transpose() <<  endl;
  }
  file.close();





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

    //tentative de prédiction. Calculons le taux de vide prédit sur un calcul de validation
    VectorXd Xeval=mr_lhs[365][0]; //me["X"];//;
    VectorXd Yeval=mr_lhs[666][1];//me["Alpha"];//mr[365][1];
    VectorXd ParamEval(5);
    ParamEval << 0.5,0.5,0.5,0.5,0.5;
    ParamEval=RtoGP(m_lhs[666]); //paramètres du calcul numéro 365

    MatrixXd Pred=EvaluateMeanVarGPPCA(vgp,ParamEval,VP,Acoefs,featureMeans);
    MatrixXd Samples=DrawSamplesGPPCA(3,vgp,ParamEval,VP,Acoefs,featureMeans,generator);
    ofstream ofile("results/alphaPCA.gnu");
    for(int i=0;i<Xeval.size();i++){
      ofile << Xeval(i) <<" " << Yeval(i) << " " << Pred(i,0) << " " << sqrt(Pred(i,1)) <<" "<< Samples(i,0)<<" " << Samples(i,1)<<" " << Samples(i,2) << endl;
    }
    ofile.close();
    compute_erreurs_validation(1,m_lhs,mr_lhs,vgp,VP,Acoefs,featureMeans);

    //On recopie tout dans des variables extérieures
    vgp_a=vgp;
    VP_a=VP;
    Acoefs_a=Acoefs;
    featureMeans_a=featureMeans;
  }
  

  /*POD pour le diamètre de bulle*/
  vector<GP> vgp_d(3);
  MatrixXd VP_d;
  MatrixXd Acoefs_d;
  VectorXd featureMeans_d;
  {
    //initialisation des variables
    int nmodes=3;
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
    hpars_gp0 << 0.278,1.93,1e-3,1.363,1.38,2,2; //matern 3 modes

    GP gp0(Kernel_GP_Matern);
    gp0.SetData(full_data[0]);
    gp0.SetGP(hpars_gp0);

    VectorXd hpars_gp1(7);
    hpars_gp1 << 0.42,0.52,1e-3,0.69,0.76,1.25,2; //matern 3 modes

    GP gp1(Kernel_GP_Matern);
    gp1.SetData(full_data[1]);
    gp1.SetGP(hpars_gp1);

    VectorXd hpars_gp2(7);
    hpars_gp2 << 0.51,0.25,1e-3,0.29,0.39,1.24,2; //matern 3 modes

    GP gp2(Kernel_GP_Matern);
    gp2.SetData(full_data[2]);
    gp2.SetGP(hpars_gp2);

    vgp[0]=gp0;
    vgp[1]=gp1;
    vgp[2]=gp2;   

    //tentative de prédiction. Calculons le taux de vide prédit sur un calcul de validation
    VectorXd Xeval=mr_lhs[365][0]; //me["X"];//;
    VectorXd Yeval=mr_lhs[365][2];//me["Alpha"];//mr[365][1];
    VectorXd ParamEval(5);
    ParamEval << 0.5,0.5,0.5,0.5,0.5;
    ParamEval=RtoGP(m_lhs[365]); //paramètres du calcul numéro 365

    MatrixXd Pred=EvaluateMeanVarGPPCA(vgp,ParamEval,VP,Acoefs,featureMeans);
    MatrixXd Samples=DrawSamplesGPPCA(3,vgp,ParamEval,VP,Acoefs,featureMeans,generator);
    ofstream ofile("results/diamPCA.gnu");
    for(int i=0;i<Xeval.size();i++){
      ofile << Xeval(i) <<" " << Yeval(i) << " " << Pred(i,0) << " " << sqrt(Pred(i,1)) << " " << Samples(i,0)<<" " << Samples(i,1)<<" " << Samples(i,2) << endl;
    }
    ofile.close();

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
      VectorXd useless=VectorXd::Random(1);
      AUGDATA dat; dat.SetX(useless); dat.SetValue(Yexpe_interpol_diam);
      data_exp_diam[0]=dat;
      dat.SetValue(Yexpe_interpol_alpha);
      data_exp_alpha[0]=dat;
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
  //lb_hpars << 1e-4,1e-4,1e-4; tdv
  //ub_hpars << 1e-1,1e-1,2e-2; tdv

      //hpars z : sedm, sobs, lcor.
    VectorXd lb_hpars_alpha(dim_hpars);
    VectorXd ub_hpars_alpha(dim_hpars);
    lb_hpars_alpha << 1e-4,1e-4,1e-4;
    ub_hpars_alpha << 1,1e-1,1e-1;

    VectorXd lb_hpars_diam(dim_hpars+1);
    VectorXd ub_hpars_diam(dim_hpars+1);
    lb_hpars_diam << 2e-5,1e-8,5e-4,-5e-4; 
    ub_hpars_diam << 5e-3,3e-5,1e-1,5e-4;

  //définition du modèle ! //coefs dans 0-1
  auto lambda_model_diam=[&vgp_d,&VP_d,&featureMeans_d,&Acoefs_d](VectorXd const & X,VectorXd const & theta)->VectorXd{
    return EvaluateMeanVarGPPCA(vgp_d,theta,VP_d,Acoefs_d,featureMeans_d).col(0);
  };

  auto lambda_model_alpha=[&vgp_a,&VP_a,&featureMeans_a,&Acoefs_a](VectorXd const & X,VectorXd const & theta)->VectorXd{
    return EvaluateMeanVarGPPCA(vgp_a,theta,VP_a,Acoefs_a,featureMeans_a).col(0);
  };

  //définition des bornes pour les hGPs.
  int nhpars_gp=7;
  MatrixXd Bounds_hpars_gp(2,nhpars_gp);
  Bounds_hpars_gp(0,0)=1E-7; Bounds_hpars_gp(1,0)=1e2; //variance
  Bounds_hpars_gp(0,2)=1E-7; Bounds_hpars_gp(1,2)=1e-4; //sigma obs
  list<int> l={1,3,4,5,6};
  for (int i:l){
    Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=2; //lcors.
  }
  VectorXd hpars_gp_guess(7);
  for (int i=0;i<nhpars_gp;i++){
    hpars_gp_guess(i)=0.5*(Bounds_hpars_gp(1,i)+Bounds_hpars_gp(0,i));
  }
  hpars_gp_guess(0)=1; //var edm
  hpars_gp_guess(2)=1e-5; //var obs

  VectorXd useless=VectorXd::Random(1);

  //construction du grid
  DoE doe_init(lb_t,ub_t,1500,1);
  //afichage du grid lhs
  doe_init.WriteGrid("results/save/grid.gnu");
  int normconst_evidence=30;


  VectorXd X_predictions=Xgrid_num; //pas le choix par rapport à la construction de la fct modèle.

  int size_xexpe=data_exp_alpha[0].Value().size();
  auto lambda_priormean_alpha=[size_xexpe](VectorXd const & X, VectorXd const & hpars){
      return VectorXd::Zero(size_xexpe);
    };
  auto lambda_priormean_diam=[size_xexpe](VectorXd const & X, VectorXd const & hpars){
      return hpars(3)*VectorXd::Ones(size_xexpe);
    };
  
  /*
  {
    //construction de densité sur diamètre pour prédictions sur alpha
    //instance de base de densité pour alpha
    Density MainDensity(doe_init);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetKernel(Kernel_Z_Matern52);
    MainDensity.SetModel(lambda_model_alpha);
    MainDensity.SetPriorMean(lambda_priormean_alpha);
    MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
    MainDensity.SetDataExp(data_exp_alpha);
    MainDensity.SetXprofile(Xgrid_num);
    cout << endl << "Début appli croisée..." << endl;
    
    //load les samples de diam
    MainDensity.ReadSamples("samples_single/sampkohalpha.gnu");
    MainDensity.WritePredictions(X_predictions,"results/preds_single/alphakoh.gnu");
    MainDensity.WritePredictionsF(X_predictions,"results/preds_single/alphakohF.gnu");


    MainDensity.ReadSamples("samples_single/sampoptalpha.gnu");
    MainDensity.WritePredictions(X_predictions,"results/preds_single/alphaopt.gnu");
    MainDensity.WritePredictionsF(X_predictions,"results/preds_single/alphaoptF.gnu");


    MainDensity.ReadSamples("samples_single/sampfbalpha.gnu");
    MainDensity.WritePredictions(X_predictions,"results/preds_single/alphafb.gnu");
    MainDensity.WritePredictionsF(X_predictions,"results/preds_single/alphafbF.gnu");

  }
  */

  

/////////////////
/*
  {
    //prédictions de diamètre à partir de samples alpha.
    
    //instance de base de densité pour diam
    Density MainDensity(doe_init);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetKernel(Kernel_Z_Matern52);
    MainDensity.SetModel(lambda_model_diam);
    MainDensity.SetPriorMean(lambda_priormean_diam);
    MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
    MainDensity.SetDataExp(data_exp_diam);
    MainDensity.SetXprofile(Xgrid_num);

    
    //load les samples de alpha
    MainDensity.ReadSamples("samples/sampkohalpha.gnu");
    MainDensity.WritePredictionsF(X_predictions,"results/crosspreds/diamkohF.gnu");
 
    MainDensity.ReadSamples("samples/sampoptalpha.gnu");
    MainDensity.WritePredictionsF(X_predictions,"results/crosspreds/diamoptF.gnu");


    MainDensity.ReadSamples("samples/sampfbalpha.gnu");
    MainDensity.WritePredictionsF(X_predictions,"results/crosspreds/diamfbF.gnu");

  }
  */
  
  
  //prédictions de alpha avec un sample de la densité double.
  
  
  
  {
    //instance de base de densité pour alpha
    Density MainDensity(doe_init);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetKernel(Kernel_Z_Matern52);
    MainDensity.SetModel(lambda_model_alpha);
    MainDensity.SetPriorMean(lambda_priormean_alpha);
    MainDensity.SetHparsBounds(lb_hpars_alpha,ub_hpars_alpha);
    MainDensity.SetDataExp(data_exp_alpha);
    MainDensity.SetXprofile(Xgrid_num);

    //load les samples de alpha
    //le DIC Opt est calculé de la même manière que les autres, bon...
    MainDensity.ReadSamples("samples_double/sampkohalpha.gnu");
    MainDensity.WritePredictionsF(X_predictions,"results/preds_double/alphakohF.gnu");
    cout << "méthode KOH : " << endl;
    cout << "AIC : " << MainDensity.AIC() << ", DIC : " << MainDensity.DIC() << ", " << "WAIC2 : " << MainDensity.WAIC2() << endl;
    
    MainDensity.ReadSamples("samples_double/sampoptalpha.gnu");
    MainDensity.WritePredictionsF(X_predictions,"results/preds_double/alphaoptF.gnu");
    cout << "méthode OPT : " << endl;
    cout << "AIC : " << MainDensity.AIC() << ", DIC : " << MainDensity.DIC() << ", " << "WAIC2 : " << MainDensity.WAIC2() << endl;

    MainDensity.ReadSamples("samples_double/sampfbalpha.gnu");
    MainDensity.WritePredictionsF(X_predictions,"results/preds_double/alphafbF.gnu");
    cout << "méthode Bayes : " << endl;
    cout << "AIC : " << MainDensity.AIC() << ", DIC : " << MainDensity.DIC() << ", " << "WAIC2 : " << MainDensity.WAIC2() << endl;
  }
  

 exit(0);


  
  //prédictions du diamètre avec un sample de la densité double.
  
  
  {
    //instance de base de densité pour diam
    Density MainDensity(doe_init);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetKernel(Kernel_Z_Matern52);
    MainDensity.SetModel(lambda_model_diam);
    MainDensity.SetPriorMean(lambda_priormean_diam);
    MainDensity.SetHparsBounds(lb_hpars_diam,ub_hpars_diam);
    MainDensity.SetDataExp(data_exp_diam);
    MainDensity.SetXprofile(Xgrid_num);
    

    //load les samples de diam
    MainDensity.ReadSamples("samples/sampkohdiam.gnu");

    MainDensity.WritePredictionsF(Xgrid_num,"results/preds/diamkohF.gnu");

    
    
    MainDensity.ReadSamples("samples/sampoptdiam.gnu");

    MainDensity.WritePredictionsF(Xgrid_num,"results/preds/diamoptF.gnu");


    MainDensity.ReadSamples("samples/sampfbdiam.gnu");

    MainDensity.WritePredictionsF(Xgrid_num,"results/preds/diamfbF.gnu");


  }
  
  
  exit(0);
  /*
  /// calcul evidence alpha
  {
     //instance de base de densité pour alpha
    Density MainDensity(doe_init);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetKernel(Kernel_Z_Matern);
    MainDensity.SetModel(lambda_model_alpha);
    MainDensity.SetPriorMean(PriorMean);
    MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
    MainDensity.SetDataExp(data_exp_alpha);
    MainDensity.SetXprofile(Xgrid_num);
    cout << endl << "Début calcul evidence alpha..." << endl;

        //load les samples de alpha
    MainDensity.ReadSamples("samples/sampkoha.gnu");
    double e=MainDensity.Evidence(normconst_evidence);
    cout << "Evidence KOH sur alpha : " << e << endl;

    MainDensity.ReadSamples("samples/sampopta.gnu");
    e=MainDensity.Evidence(normconst_evidence);
    cout << "Evidence Opt sur alpha : " << e << endl;

    MainDensity.ReadSamples("samples/sampfba.gnu");
    e=MainDensity.Evidence(normconst_evidence);
    cout << "Evidence FV sur alpha : " << e << endl;

    MainDensity.ReadSamples("samples/sampnoedma.gnu");
    e=MainDensity.Evidence(normconst_evidence);
    cout << "Evidence NoEdm sur alpha :" << e << endl;


  }

  /// calcul evidence diam
  {
     //instance de base de densité pour diam
    Density MainDensity(doe_init);
    MainDensity.SetLogPriorPars(logprior_pars);
    MainDensity.SetLogPriorHpars(logprior_hpars);
    MainDensity.SetKernel(Kernel_Z_Matern);
    MainDensity.SetModel(lambda_model_diam);
    MainDensity.SetPriorMean(PriorMean);
    MainDensity.SetHparsBounds(lb_hpars,ub_hpars);
    MainDensity.SetDataExp(data_exp_diam);
    MainDensity.SetXprofile(Xgrid_num);
    cout << endl << "Début calcul evidence diam..." << endl;

        //load les samples de diam
    MainDensity.ReadSamples("samples/sampkohdiam.gnu");
    double e=MainDensity.Evidence(normconst_evidence);
    cout << "Evidence KOH sur diam : " << e << endl;

    MainDensity.ReadSamples("samples/sampoptdiam.gnu");
    e=MainDensity.Evidence(normconst_evidence);
    cout << "Evidence Opt sur diam : " << e << endl;

    MainDensity.ReadSamples("samples/sampfbdiam.gnu");
    e=MainDensity.Evidence(normconst_evidence);
    cout << "Evidence FV sur diam : " << e << endl;

    MainDensity.ReadSamples("samples/sampnoedmdiam.gnu");
    e=MainDensity.Evidence(normconst_evidence);
    cout << "Evidence NoEdm sur diam :" << e << endl;


  }
*/

}
exit(0);

  
};