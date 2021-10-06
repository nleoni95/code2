// Dans ce fichier : on lit les données du calcul et on les post-traite à notre souhait.
// On lit également les données expérimentales.
// On peut faire quelques plots des données de calcul
// On créée ensuite un GP qu'on sauvegarde dans un fichier tiers qui sera lu plus tard.
// Il faut également récupérer le point du DoE correspondant.
// backup du main comprenant polfit, régression polynomiale, régression logistique.

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
#include "pcb++.h"
#include "cub++.h"
#include "vstoch++.h"
#include "mstoch++.h"
#include "gp++.h"


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

std::vector<int> KMEANS(vector<DATA> const &data, int const nk){
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
    string fullname="../data/qmc/"+to_string(i)+"/"+filename;
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
    string fullname="../data/lhs/"+to_string(i)+"/"+filename;
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

double my_model(VectorXd const &x, VectorXd const &theta){

}
double Kernel_GP(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  //attention les lcor ne sont pas au carré. Je corrigerai plus tard...
  list<int> l={1,3,4,5,6};
  double cor=pow(hpar(0),2);
  cor*=exp(-0.5*pow(x(0)-y(0),2)/hpar(1)); //phi
  cor*=exp(-0.5*pow(x(1)-y(1),2)/hpar(3)); //BK
  cor*=exp(-0.5*pow(x(2)-y(2),2)/hpar(4)); //COAL
  cor*=exp(-0.5*pow(x(3)-y(3),2)/hpar(5)); //NUCL
  cor*=exp(-0.5*pow(x(4)-y(4),2)/hpar(6)); //MT
  return cor;
}

double Kernel_GP_Matern(VectorXd const &x, VectorXd const &y, VectorXd const &hpar){
  //noyau pour le GP. 0:intensity, 2:noise, 1:lcor phi, 3:lcor BK, 4:lcor COAL, 5:lcor NUCL, 6: lcor MT
  list<int> l={1,3,4,5,6};
  double cor=pow(hpar(0),2);
  cor*=exp(-0.5*abs(x(0)-y(0))/hpar(1)); //phi
  cor*=exp(-0.5*abs(x(1)-y(1))/hpar(3)); //BK
  cor*=exp(-0.5*abs(x(2)-y(2))/hpar(4)); //COAL
  cor*=exp(-0.5*abs(x(3)-y(3))/hpar(5)); //NUCL
  cor*=exp(-0.5*abs(x(4)-y(4))/hpar(6)); //MT
  return cor;
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
	neval++;														//increment the number of evaluation count
	return value;
};

double logprior_hpars(VectorXd const &hpars){
  //edm, exp, lcor
  if(hpars(2)<=0){return -999;}
  double alpha_ig=10;
  double beta_ig=140;
  return log(pow(beta_ig,alpha_ig)*pow(hpars(2),-alpha_ig-1)*exp(-beta_ig/hpars(2))/tgamma(alpha_ig));
  return 0;//-2*(log(hpars(1)))-2*(log(hpars(0)));
}

double logprior_pars(VectorXd const &pars){
  //prior uniforme sur les pramètres
  double moy0=40;
  double moy1=18.9e-6;
  double moy2=0.6;
  double sig0=2*5;
  double sig1=18.9e-6/10.; //double sig1=2*1e-6;
  double sig2=0.6/10.;
  return log(gaussprob(pars(0),moy0,sig0)*gaussprob(pars(1),moy1,sig1)*gaussprob(pars(2),moy2,sig2));
}

void PrintVector(vector<VectorXd> &X, VectorXd &values,const char* file_name){
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<X.size();i++){
    fprintf(out,"%e %e\n",X[i](0),values(i));
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


//fonctions pour faire un fit logistique
VectorXd EvaluateLogfit(VectorXd const &x, VectorXd const & coefs){
  //évaluation de la fonction logfit sur un ensemble de points x.
  // fonction : y=alpha*(1+exp(-beta -gama*x))
  if(!coefs.size()==3){cerr << "erreur de dimension Evaluate Logfit. Dimension : " << coefs.size();}
  VectorXd y(x.size());
  for (int i=0;i<y.size();i++){
    y(i)=coefs(0)/(1+coefs(1)*exp(-(coefs(2)*x(i))));
  }
  return y;
}

double myoptfunc_logfit(const std::vector<double> &x, std::vector<double> &grad, void *data){
	/* Fonction à optimiser, logfit par moindres carrés. */
  pair<VectorXd,VectorXd>* p=(pair<VectorXd,VectorXd>*) data; //recast du pointeur
  VectorXd xv(x.size());  for(int i=0;i<x.size();i++){xv(i)=x[i];} //transformation du vecteur en vectorxd
  VectorXd Yfit=EvaluateLogfit(p->second,xv);
  return ((p->first)-Yfit).squaredNorm();
};

VectorXd logisticfit(VectorXd const &y, VectorXd const &x){
  //calcule le logistic fit pour une courbe donnée. Ne faisons qu'une opti locale.
  //création du pointeur pour passer les données
  std::pair<VectorXd,VectorXd> p=make_pair(y,x);
  int dim=3; //3 coefficients
  std::vector<double> lb(dim); 
	std::vector<double> ub(dim);
  lb[0]=0.1;ub[0]=10; //alpha
  lb[1]=3;ub[1]=15; //beta
  lb[2]=1;ub[2]=1E3; //gamma
	std::vector<double> par(dim);
  for(int i=0;i<dim;i++){par[i]=0.5*(lb[i]+ub[i]);}
  nlopt::opt opt(nlopt::LN_SBPLX, dim);    /* algorithm and dimensionality */
	opt.set_lower_bounds(lb);
	opt.set_upper_bounds(ub);
  opt.set_min_objective(myoptfunc_logfit, &p);
	opt.set_xtol_rel(1e-4);
  opt.set_maxeval(1000);
  double minf; /* the minimum objective value, upon return */
  if (opt.optimize(par, minf) < 0) printf("nlopt failed!\n");
  VectorXd parv(dim);
  for(int i=0;i<dim;i++){parv(i)=par[i];}
  return parv;
}
//fonctions pour faire un fit polynomial

VectorXd EvaluateMeanGPLogFit(VectorXd const &xevals_pol, vector<GP> const &vgp, VectorXd const & Target){
  //évaluation de la prédiction moyenne en un point, sur un vecteur, à partir d'un vecteur de GPs.
  VectorXd coefsmean(3);
  for(int i=0;i<3;i++){coefsmean(i)=vgp[i].EvalMean(Target);}
  VectorXd Yfit=EvaluateLogfit(xevals_pol,coefsmean);
  return Yfit;
}

MatrixXd DrawSamplesGPLogFit(VectorXd const &xevals_pol, vector<GP> const &vgp, VectorXd const & Target, int ns,default_random_engine &generator){
  //tirage de samples à partir d'un vecteur de GPs.
  vector<VectorXd> Target2(1); //passage en vector pour la fonction olm
  Target2[0]=Target;
  MatrixXd coefs(3,ns);
  for (int i=0;i<vgp.size();i++){
    coefs.row(i)=vgp[i].SampleGPDirect(Target2,ns,generator); //on peut stocker dans une colonne car Target2 est de taille 1
  }
  MatrixXd Yfit(xevals_pol.size(),ns);
  for (int i=0;i<ns;i++){
    Yfit.col(i)=EvaluateLogfit(xevals_pol,coefs.col(i));
  }
  return Yfit;
}

VectorXd EvaluateVarGPLogFit(VectorXd const &xevals_pol, vector<GP> const &vgp, VectorXd const & Target, int ns,default_random_engine &generator){
  //évaluation de la variance sur un vecteur de points. Puisqu'il n'y a pas d'expression explicite, à faire par tirage de samples (nombre : ns)
  //tirage des samples
  MatrixXd Samples=DrawSamplesGPLogFit(xevals_pol,vgp,Target,ns,generator); //de taille (xevals_pol.size(),ns)

  VectorXd VAR(xevals_pol.size());
  for (int i=0;i<xevals_pol.size();i++){
    //on recopie l'ensemble des samples dans un vector
    vector<double> v(ns); for(int j=0;j<ns;j++){v[j]=Samples(i,j);}
    //on évalue la variance de la série
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double var = (sq_sum / v.size() - mean * mean);
    VAR(i)=var;
  }
  return VAR;
}

//on donne un vecteur d, un vecteur x, un degré poly, et on fait le fit.
VectorXd polfit(VectorXd const &y, VectorXd const &x, int order){
  if(!y.size()==x.size() || order <0){cerr << "erreur polfit" << endl;}
  //construction de la matrice de régression
  MatrixXd X(x.size(),order+1);
  for(int i=0;i<x.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(x(i),j);
    }
  }
  //solution par SVD
  VectorXd coefs=X.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
  return coefs;
}

VectorXd polfitD(VectorXd const &y, VectorXd const &x, int order){
  if(!y.size()==x.size() || order <0){cerr << "erreur polfit" << endl;}
  //pour diamètre
  double diamref=0.006;
  //construction de la matrice de régression
  MatrixXd X(x.size(),order+1);
  for(int i=0;i<x.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(x(i)-diamref,j);
    }
  }
  //solution par SVD
  VectorXd coefs=X.bdcSvd(ComputeThinU | ComputeThinV).solve(y);
  return coefs;
}

//on donne un vecteur x, un vecteur de coefficients, et on renvoie une évaluation du polfit sur ce vecteur.
VectorXd EvaluatePolfit(VectorXd const & x, VectorXd const & coefs){
  //construction de la matrice de régression
  //pour diamètre
  double diamref=0.006;
  int order=coefs.size()-1;
  MatrixXd X(x.size(),order+1);
  for(int i=0;i<x.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(x(i)-diamref,j);
    }
  }
  return X*coefs;
}

VectorXd EvaluatePolfitD(VectorXd const & x, VectorXd const & coefs){
  //construction de la matrice de régression
  int order=coefs.size()-1;
  MatrixXd X(x.size(),order+1);
  for(int i=0;i<x.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(x(i),j);
    }
  }
  return X*coefs;
}

//prédiction moyenne en utilisant un vecteur de GPs sur un fit polynomial.
VectorXd EvaluateMeanGP(VectorXd const &xevals_pol, vector<GP> const &vgp, VectorXd const & Target){
  // xevals_pol : les points d'évaluation du polynôme, Target : la valeur du paramètre pour le gp.
  int order=vgp.size()-1;
  //construction de la matrice de régression
  MatrixXd X(xevals_pol.size(),order+1);
  for(int i=0;i<xevals_pol.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(xevals_pol(i),j);
    }
  }
  //calcul de la moyenne de chaque gp
  VectorXd coefs(order+1);
  for (int i=0;i<vgp.size();i++){
    coefs(i)=vgp[i].EvalMean(Target);
  }
  return X*coefs;
}

//prédiction moyenne et la variance en utilisant un vecteur de GPs sur un fit polynomial.
MatrixXd 
EvaluateMeanVarGP(VectorXd const &xevals_pol, vector<GP> const &vgp, VectorXd const & Target){
  // xevals_pol : les points d'évaluation du polynôme, Target : la valeur du paramètre pour le gp.
  // on renvoie une matrice de xeval_pol.size() lignes et 2 colonnes. la première colonne est la moyenne du gp, la seconde est la variance en ce point.
  int order=vgp.size()-1;
  //construction de la matrice de régression pour la moyenne
  MatrixXd X(xevals_pol.size(),order+1);
  for(int i=0;i<xevals_pol.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(xevals_pol(i),j);
    }
  }
  //construction de la matrice de régression pour la variance
  MatrixXd X2=square(X.array()); //coefficient-wise square
  //check rapide si j'ai pas fait de conneries
  for(int i=0;i<xevals_pol.size();i++){
    for(int j=0;j<order+1;j++){
      if(!X2(i,j)==pow(X(i,j),2)){cerr << "erreur : " << pow(X(i,j),2) << " , " << X2(i,j) << endl; }
      X(i,j)=pow(xevals_pol(i),j);
    }
  }
  //calcul de la moyenne de chaque gp
  VectorXd meansgps(order+1);
  VectorXd varsgps(order+1);
  for (int i=0;i<vgp.size();i++){
    VectorXd MeanVar=vgp[i].Eval(Target);
    meansgps(i)=MeanVar(0);
    varsgps(i)=MeanVar(1);
  }
  VectorXd moy=X*meansgps;
  VectorXd var=X2*varsgps;
  MatrixXd res(xevals_pol.size(),2); res.col(0)=moy; res.col(1)=var;
  return res;
}

VectorXd EvaluateMeanGPD(VectorXd const &xevals_pol, vector<GP> const &vgp, VectorXd const & Target){
  // xevals_pol : les points d'évaluation du polynôme, Target : la valeur du paramètre pour le gp.
  int order=vgp.size()-1;
  double diamref=0.006;
  //construction de la matrice de régression
  MatrixXd X(xevals_pol.size(),order+1);
  for(int i=0;i<xevals_pol.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(xevals_pol(i)-diamref,j);
    }
  }
  //calcul de la moyenne de chaque gp
  VectorXd coefs(order+1);
  for (int i=0;i<vgp.size();i++){
    coefs(i)=vgp[i].EvalMean(Target);
  }
  return X*coefs;
}

//prédiction moyenne et la variance en utilisant un vecteur de GPs sur un fit polynomial.
MatrixXd 
EvaluateMeanVarGPD(VectorXd const &xevals_pol, vector<GP> const &vgp, VectorXd const & Target){
  // xevals_pol : les points d'évaluation du polynôme, Target : la valeur du paramètre pour le gp.
  // on renvoie une matrice de xeval_pol.size() lignes et 2 colonnes. la première colonne est la moyenne du gp, la seconde est la variance en ce point.
  int order=vgp.size()-1;
  double diamref=0.006;
  //construction de la matrice de régression pour la moyenne
  MatrixXd X(xevals_pol.size(),order+1);
  for(int i=0;i<xevals_pol.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(xevals_pol(i)-diamref,j);
    }
  }
  //construction de la matrice de régression pour la variance
  MatrixXd X2=square(X.array()); //coefficient-wise square
  //check rapide si j'ai pas fait de conneries
  for(int i=0;i<xevals_pol.size();i++){
    for(int j=0;j<order+1;j++){
      if(!X2(i,j)==pow(X(i,j),2)){cerr << "erreur : " << pow(X(i,j),2) << " , " << X2(i,j) << endl; }
      X(i,j)=pow(xevals_pol(i)-diamref,j);
    }
  }
  //calcul de la moyenne de chaque gp
  VectorXd meansgps(order+1);
  VectorXd varsgps(order+1);
  for (int i=0;i<vgp.size();i++){
    VectorXd MeanVar=vgp[i].Eval(Target);
    meansgps(i)=MeanVar(0);
    varsgps(i)=MeanVar(1);
  }
  VectorXd moy=X*meansgps;
  VectorXd var=X2*varsgps;
  MatrixXd res(xevals_pol.size(),2); res.col(0)=moy; res.col(1)=var;
  return res;
}

//tirage d'échantillons de vecteur de gp.
MatrixXd DrawSamplesGP(VectorXd const &xevals_pol, vector<GP> const &vgp, VectorXd const & Target, int ns, default_random_engine &generator){
  // xevals_pol : les points d'évaluation du polynôme, xeval_par : la valeur du paramètre pour le gp.
  // faisons pour un seul point dans l'espace des paramètres gp, sinon il faut rajouter une dimension au problème.
  int order=vgp.size()-1;
  //construction de la matrice de régression
  MatrixXd X(xevals_pol.size(),order+1);
  for(int i=0;i<xevals_pol.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(xevals_pol(i),j);
    }
  }
  //calcul de la moyenne de chaque gp
  MatrixXd coefs(order+1,ns); //matrice regroupant les échantillons
  vector<VectorXd> Target2(1); //passage en vector pour la fonction olm
  Target2[0]=Target;
  for (int i=0;i<vgp.size();i++){
    coefs.row(i)=vgp[i].SampleGPDirect(Target2,ns,generator);
  }

  return X*coefs;
}

MatrixXd DrawSamplesGPD(VectorXd const &xevals_pol, vector<GP> const &vgp, VectorXd const & Target, int ns, default_random_engine &generator){
  // xevals_pol : les points d'évaluation du polynôme, xeval_par : la valeur du paramètre pour le gp.
  // faisons pour un seul point dans l'espace des paramètres gp, sinon il faut rajouter une dimension au problème.
  double diamref=0.006;
  int order=vgp.size()-1;
  //construction de la matrice de régression
  MatrixXd X(xevals_pol.size(),order+1);
  for(int i=0;i<xevals_pol.size();i++){
    for(int j=0;j<order+1;j++){
      X(i,j)=pow(xevals_pol(i)-diamref,j);
    }
  }
  //calcul de la moyenne de chaque gp
  MatrixXd coefs(order+1,ns); //matrice regroupant les échantillons
  vector<VectorXd> Target2(1); //passage en vector pour la fonction olm
  Target2[0]=Target;
  for (int i=0;i<vgp.size();i++){
    coefs.row(i)=vgp[i].SampleGPDirect(Target2,ns,generator);
  }
  return X*coefs;
}

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
  VP=vecpropres.rightCols(nmodes); //(nrayons,nmodes)
  VectorXd lambdas_red=lambdas.bottomRows(nmodes); //nmodes
  //on reverse les vecteurs propres et valeurs propres pour que les principaux se trouvent à la position 0.
  lambdas_red.reverseInPlace();
  VectorXd main_mode=VP.col(0); //obligés de faire ainsi car sinon la clonne 0 est mauvaise... bug de eigen ?
  VP=VP.rowwise().reverse();
  VP.col(nmodes-1)=main_mode;
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
  int nsamples=1e5;
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

double erreurf_olm(MatrixXd const & A, MatrixXd const &B){
  //calcul de la distance entre les deux prédictions d'après la définition d'erreur OLM.
  int ncalcs=A.cols();
  auto scalprod=[](MatrixXd const &A, MatrixXd const &B)->double{
    return (A.transpose()*B).trace();
  };
  return scalprod(A-B,A-B)/(ncalcs*scalprod(A,A));
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

const double Big = -1.e16;


int main(int argc, char **argv){
  generator.seed(17);

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

  /*POD pour le taux de vide*/
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
  }

  /*POD pour le diamètre de bulle*/
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
  }

  

  exit(0);
  /*Construction du p.g. avec 1 objectif scalaire : taux de vide moyen*/
  /*
  {
    double true_tv=0.208;
    vector<DATA> data_gp;
    for (auto &v:mr){
      VectorXd X=m[v.first];
      double tvmoyen=tauxmoyen(v.second);
      DATA dat; dat.SetX(RtoGP(X)); dat.SetValue(tvmoyen);
      data_gp.push_back(dat);
    }
    //hpars GP. Puisqu'on a plus de 2 hpars, le numéro 3 est forcément le sigma noise.
    int nhpars_gp=7;
    MatrixXd Bounds_hpars_gp(2,nhpars_gp);
    Bounds_hpars_gp(0,0)=1E-3; Bounds_hpars_gp(1,0)=1; //variance
    Bounds_hpars_gp(0,2)=1E-3; Bounds_hpars_gp(1,2)=2E-3; //sigma obs
    list<int> l={1,3,4,5,6};
    for (int i:l){
      Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=0.5; //lcors.
    }
    VectorXd hpars_gp_guess(7);
    for (int i=0;i<nhpars_gp;i++){
      hpars_gp_guess(i)=0.5*(Bounds_hpars_gp(1,i)+Bounds_hpars_gp(0,i));
    }
    auto begin=chrono::steady_clock::now();
    GP gp(Kernel_GP);
    gp.SetData(data_gp);
    gp.SetGP(hpars_gp_guess);
    cout << "optimisation..." << endl;

    gp.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_gp_guess,nhpars_gp);
    cout  << "par (Guess) : " << hpars_gp_guess.transpose() << endl;
	  hpars_gp_guess = gp.GetPar();
	  cout  << "par (Optim) (sigmaedm,lcorphi,sigmaobs,lcor2,lcor3,lcor4,lcor5) : " << hpars_gp_guess.transpose() << endl;
    auto end=chrono::steady_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;
  }
  */
  //construction des DATA
  //construction du processus gaussien


  /* PCA des données*/
  //tout recopier dans un vecteur<DATA> d'abord

  
  /*
  {
    vector<DATA> simus;
    //faisons sur le taux de vide moyen. je devrais trouver une grosse dépendance en phi.
    for(const auto &v:mr){
      DATA dat;
      VectorXd X=m[v.first];
      double value=tauxmoyen(v.second);
      dat.SetValue(value); dat.SetX(X);
      simus.push_back(dat);
    }
    cout << "5 clusters : " << endl;
    auto begin=chrono::steady_clock::now();
    principal_directions(simus,5);
    auto end=chrono::steady_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;
    
    cout << "10 clusters : " << endl;
    begin=chrono::steady_clock::now();
    principal_directions(simus,10);
    end=chrono::steady_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;

    cout << "20 clusters : " << endl;
    begin=chrono::steady_clock::now();
    principal_directions(simus,20);
    end=chrono::steady_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl; 
  }
  exit(0);
  */

  /* Construction de PGs pour le taux de vide, par fit logistique*/
  /*
  {
    vector<GP> vgp(3); //poly d'ordre 3, donc 4 gps.
    //construction des data.
    vector<DATA> data_gp0;
    vector<DATA> data_gp1;
    vector<DATA> data_gp2;
    for (auto &v:mr){
      VectorXd X=m[v.first];
      VectorXd coefs=logisticfit(v.second[1],v.second[0]);
      DATA dat; dat.SetX(RtoGP(X)); dat.SetValue(coefs(0));
      data_gp0.push_back(dat);
      dat.SetValue(coefs(1));
      data_gp1.push_back(dat);
      dat.SetValue(coefs(2));
      data_gp2.push_back(dat);
    }   

    //hpars GP. on essaye avec les mêmes bornes d'opti pour le moment.
    int nhpars_gp=7;
    MatrixXd Bounds_hpars_gp(2,nhpars_gp);
    Bounds_hpars_gp(0,0)=1E-3; Bounds_hpars_gp(1,0)=1E5; //variance
    Bounds_hpars_gp(0,2)=1E-3; Bounds_hpars_gp(1,2)=2E-3; //sigma obs
    list<int> l={1,3,4,5,6};
    for (int i:l){
      Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=3; //lcors.
    }
    VectorXd hpars_gp_guess(7);
    //valeurs guessed avec les optimisations précédentes.
    hpars_gp_guess(0)=1e7;
    hpars_gp_guess(1)=0.32;
    hpars_gp_guess(2)=0.002;
    hpars_gp_guess(3)=1.95;
    hpars_gp_guess(4)=1.93;
    hpars_gp_guess(5)=0.01;
    hpars_gp_guess(6)=1.53;

    VectorXd hpars_gp0(7);
    hpars_gp0 << 0.258,0.01,0.002,0.11,0.127,0.01,0.51;

    GP gp0(Kernel_GP);
    gp0.SetData(data_gp0);
    gp0.SetGP(hpars_gp0);
    
    VectorXd hpars_gp1(7);
    hpars_gp1 << 5.11,0.01,0.002,0.44,0.183,0.01,0.27;


    GP gp1(Kernel_GP);
    gp1.SetData(data_gp1);
    gp1.SetGP(hpars_gp1);
    
    VectorXd hpars_gp2(7);
    hpars_gp2 << 214.5,0.01,0.0019,0.06,0.14,0.01,0.46;
    GP gp2(Kernel_GP);
    gp2.SetData(data_gp2);
    gp2.SetGP(hpars_gp2);

    vgp[0]=gp0;
    vgp[1]=gp1;
    vgp[2]=gp2;

    //tentative de prédiction. Calculons le taux de vide prédit au point nominal, ainsi que quelques samples.
    VectorXd Xeval=mr[365][0]; //me["X"];//;
    VectorXd Yeval=mr[365][1];//me["Alpha"];//mr[365][1];
    VectorXd ParamEval(5);
    ParamEval << 0.5,0.5,0.5,0.5,0.5;
    ParamEval=RtoGP(m[365]); //paramètres du calcul numéro 365

    VectorXd Pred=EvaluateMeanGPLogFit(Xeval,vgp,ParamEval);
    MatrixXd Samples=DrawSamplesGPLogFit(Xeval,vgp,ParamEval,3,generator);
    auto begin=chrono::steady_clock::now();
    VectorXd Var=EvaluateVarGPLogFit(Xeval,vgp,ParamEval,1000,generator);
    auto end=chrono::steady_clock::now();
    cout << "time for 1000 samples : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;

    ofstream ofile("results/predGP.gnu");
    for(int i=0;i<Xeval.size();i++){
      ofile << Xeval(i) << " " << Pred(i) <<" " << sqrt(Var(i)) << " " << Samples.col(0)(i) << " " << Samples.col(1)(i) << " " << Samples.col(2)(i) << endl;
    }
    ofile.close();

    //Evaluation du vgp taux de vide sur le DoE LHS
    
    {
      VectorXd Xpred=mr_lhs[1][0]; //les x de prédiction du dts.
      double err_tv=0;
      double err_tp=0;
      for(const auto &v:mr_lhs){
        //taux de vide moyen sur le dataset validation
        double tv=tauxmoyen(v.second);
        double tp=tauxparoi(v.second);
        //paramètre correspondant
        VectorXd Theta=m_lhs[v.first];
        //prédiction moyenne du vgp.
        VectorXd Pred =EvaluateMeanGPLogFit(Xeval,vgp,RtoGP(Theta));
        //calcul du taux de vide moyen
        double tv2=tauxmoyen(Pred,Xpred);
        double tp2=Pred(Pred.size()-1);
        //calcul de l'erreur relative : 
        double err_rel_tv=abs(tv-tv2)/tv;
        double err_rel_tp=abs(tp-tp2)/tp;
        //ajout : 
        err_tv+=err_rel_tv;
        err_tp+=err_rel_tp;
      }
      err_tv/=mr_lhs.size();
      err_tp/=mr_lhs.size();
      cout << "erreur relative logfit tv : " << err_tv*100 << " pct" << endl;
      cout << "erreur relative logfit tp : " << err_tp*100 << " pct" << endl;
    }    
  }
  */
  
  /*Construction de pgs pour reproduire le profil de taux de vide par POLFIT*/
  {
    vector<GP> vgp(4); //poly d'ordre 3, donc 4 gps.
    //construction des data.
    vector<DATA> data_gp0;
    vector<DATA> data_gp1;
    vector<DATA> data_gp2;
    vector<DATA> data_gp3;
    for (auto &v:mr){
      VectorXd X=m[v.first];
      VectorXd coefs=polfit(v.second[1],v.second[0],3); //fit du profil de taux de vide. l'entier est l'ordre du polynome ! pas la dimension de coefs.
      DATA dat; dat.SetX(RtoGP(X)); dat.SetValue(coefs(0));
      data_gp0.push_back(dat);
      dat.SetValue(coefs(1));
      data_gp1.push_back(dat);
      dat.SetValue(coefs(2));
      data_gp2.push_back(dat);
      dat.SetValue(coefs(3));
      data_gp3.push_back(dat);
    }    
    //hpars GP. on essaye avec les mêmes bornes d'opti pour le moment.
    int nhpars_gp=7;
    MatrixXd Bounds_hpars_gp(2,nhpars_gp);
    Bounds_hpars_gp(0,0)=1E-3; Bounds_hpars_gp(1,0)=5E6; //variance
    Bounds_hpars_gp(0,2)=1E-3; Bounds_hpars_gp(1,2)=2E-3; //sigma obs
    list<int> l={1,3,4,5,6};
    for (int i:l){
      Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=3; //lcors.
    }
    VectorXd hpars_gp_guess(7);
    //valeurs guessed avec les optimisations précédentes.
    hpars_gp_guess(0)=0.1;
    hpars_gp_guess(1)=0.32;
    hpars_gp_guess(2)=0.002;
    hpars_gp_guess(3)=1.95;
    hpars_gp_guess(4)=1.93;
    hpars_gp_guess(5)=0.01;
    hpars_gp_guess(6)=1.53;
    VectorXd hpars_gp0(7);
    hpars_gp0 << 0.12,0.36,0.002,2.28,2.15,0.01,1.73;

    GP gp0(Kernel_GP);
    gp0.SetData(data_gp0);
    gp0.SetGP(hpars_gp0);
    /*
    cout << "optimisation du GP 0" << endl;
    gp0.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_gp_guess,nhpars_gp);
    cout  << "par (Guess) : " << hpars_gp_guess.transpose() << endl;
	  hpars_gp_guess = gp0.GetPar();
	  cout  << "par (Optim) (sigmaedm,lcorphi,sigmaobs,lcor2,lcor3,lcor4,lcor5) : " << hpars_gp_guess.transpose() << endl;
    */

    VectorXd hpars_gp1(7);
    hpars_gp1 << 10.15,0.023,0.002,0.3,0.1,0.01,0.57;


    GP gp1(Kernel_GP);
    gp1.SetData(data_gp1);
    gp1.SetGP(hpars_gp1);
    /*
    cout << "optimisation du GP 1" << endl;

    gp1.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_gp_guess,nhpars_gp);
    cout  << "par (Guess) : " << hpars_gp_guess.transpose() << endl;
	  hpars_gp_guess = gp1.GetPar();
	  cout  << "par (Optim) (sigmaedm,lcorphi,sigmaobs,lcor2,lcor3,lcor4,lcor5) : " << hpars_gp_guess.transpose() << endl;
    end=chrono::steady_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;
    */

    VectorXd hpars_gp2(7);
    hpars_gp2 << 3946.5,0.05,0.002,0.42,0.11,0.01,0.43;
    GP gp2(Kernel_GP);
    gp2.SetData(data_gp2);
    gp2.SetGP(hpars_gp2);
    /*
    cout << "optimisation du GP 2" << endl;

    gp2.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_gp_guess,nhpars_gp);
    cout  << "par (Guess) : " << hpars_gp_guess.transpose() << endl;
	  hpars_gp_guess = gp2.GetPar();
	  cout  << "par (Optim) (sigmaedm,lcorphi,sigmaobs,lcor2,lcor3,lcor4,lcor5) : " << hpars_gp_guess.transpose() << endl;
    end=chrono::steady_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;
    */
    VectorXd hpars_gp3(7);
    hpars_gp3 << 1.13E6,0.044,0.0013,0.17,0.124,0.01,0.38;
    GP gp3(Kernel_GP);
    gp3.SetData(data_gp3);
    gp3.SetGP(hpars_gp3);
    /*
    cout << "optimisation du GP 3" << endl;

    gp3.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_gp3,nhpars_gp);
    cout  << "par (Guess) : " << hpars_gp3.transpose() << endl;
	  hpars_gp3 = gp3.GetPar();
	  cout  << "par (Optim) (sigmaedm,lcorphi,sigmaobs,lcor2,lcor3,lcor4,lcor5) : " << hpars_gp3.transpose() << endl;
    end=chrono::steady_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;
    */

    vgp[0]=gp0;
    vgp[1]=gp1;
    vgp[2]=gp2;
    vgp[3]=gp3;

    //tentative de prédiction. Calculons le taux de vide prédit au point nominal, ainsi que quelques samples.
    VectorXd Xeval=mr[365][0]; //me["X"];//;
    VectorXd Yeval=mr[365][1];//me["Alpha"];//mr[365][1];
    VectorXd ParamEval(5);
    ParamEval << 0.5,0.5,0.5,0.5,0.5;
    ParamEval=RtoGP(m[365]); //paramètres du calcul numéro 365

    MatrixXd Pred=EvaluateMeanVarGP(Xeval,vgp,ParamEval);
    MatrixXd Samples=DrawSamplesGP(Xeval,vgp,ParamEval,3,generator);
    ofstream ofile("results/predGP2.gnu");
    for(int i=0;i<Xeval.size();i++){
      ofile << Xeval(i) << " " << Pred(i,0) << " " << sqrt(Pred(i,1)) <<" " << Yeval(i) << " " << Samples.col(0)(i) << " " << Samples.col(1)(i) << " " << Samples.col(2)(i) << endl;
    }
    ofile.close();

    /*Evaluation du vgp taux de vide sur le DoE LHS. par polfit*/
    
    {
      VectorXd Xpred=mr_lhs[1][0]; //les x de prédiction du dts.
      double err_tv=0;
      double err_tp=0;
      for(const auto &v:mr_lhs){
        //taux de vide moyen sur le dataset validation
        double tv=tauxmoyen(v.second);
        double tp=tauxparoi(v.second);
        //paramètre correspondant
        VectorXd Theta=m_lhs[v.first];
        //prédiction moyenne du vgp.
        VectorXd Pred =EvaluateMeanGP(Xeval,vgp,RtoGP(Theta));
        //calcul du taux de vide moyen
        double tv2=tauxmoyen(Pred,Xpred);
        double tp2=Pred(Pred.size()-1);
        //calcul de l'erreur relative : 
        double err_rel_tv=abs(tv-tv2)/tv;
        double err_rel_tp=abs(tp-tp2)/tp;
        //ajout : 
        err_tv+=err_rel_tv;
        err_tp+=err_rel_tp;
      }
      err_tv/=mr_lhs.size();
      err_tp/=mr_lhs.size();
      cout << "erreur relative polfit taux moyen : " << err_tv*100 << " pct" << endl;
      cout << "erreur relative polfit tp : " << err_tp*100 << " pct" << endl;
    }
    //affichage des directions principales
    {
      cout << "coef 0 : "; principal_directions(data_gp0,20);
      cout << "coef 1 : "; principal_directions(data_gp1,20);
      cout << "coef 2 : "; principal_directions(data_gp2,20);
      cout << "coef 3 : "; principal_directions(data_gp3,20);
    }
    /*Comparaison des prédictions sur un cas de training, et un cas de validation.*/
    {
      int nrtraining=500;
      int nrval=500;
      VectorXd Xeval_training=mr[nrtraining][0];
      VectorXd Xeval_validation=mr_lhs[nrval][0];
      //récupération des paramètres associés à ces calculs
      VectorXd Theta_training=m[nrtraining];
      VectorXd Theta_validation=m_lhs[nrval];
      //prédictions du vgp
      MatrixXd Pred_training=EvaluateMeanVarGP(Xeval_training,vgp,RtoGP(Theta_training));
      MatrixXd Pred_validation=EvaluateMeanVarGP(Xeval_validation,vgp,RtoGP(Theta_validation));
      //récupération des vraies valeurs
      VectorXd True_training=mr[nrtraining][1]; //1 pour le tdv
      VectorXd True_validation=mr_lhs[nrval][1];
      MatrixXd Samples_training=DrawSamplesGP(Xeval_training,vgp,RtoGP(Theta_training),3,generator);
      MatrixXd Samples_validation=DrawSamplesGP(Xeval_validation,vgp,RtoGP(Theta_validation),3,generator);
      
      ofstream ofile("results/tdvtraining.gnu");
      for(int i=0;i<Xeval_training.size();i++){
       ofile << Xeval_training(i) << " " << True_training(i) << " " << Pred_training(i,0) << " " << sqrt(Pred_training(i,1)) << 
        " " << Samples_training(i,0) << " " << Samples_training(i,1) << " " << Samples_training(i,2) << endl;
      }
      ofile.close();
      ofile.open("results/tdvvali.gnu");
      for(int i=0;i<Xeval_validation.size();i++){
        ofile << Xeval_validation(i) << " " << True_validation(i) << " " << Pred_validation(i,0) << " " << sqrt(Pred_validation(i,1))  << 
        " " << Samples_validation(i,0) << " " << Samples_validation(i,1) << " " << Samples_validation(i,2) << endl;
      }
      ofile.close();      
    }
    
    

  }

  /*Construction du vecteur de PGs sur la prédiction du diamètre.*/

  {
    vector<GP> vgp(6); //régression par poly d'ordre 5, donc 6 gps.
    //construction des data.
    vector<DATA> data_gp0;
    vector<DATA> data_gp1;
    vector<DATA> data_gp2;
    vector<DATA> data_gp3;
    vector<DATA> data_gp4;
    vector<DATA> data_gp5;
    for (auto &v:mr){
      VectorXd X=m[v.first];
      VectorXd coefs=polfitD(v.second[2],v.second[0],5); //fit du profil de taux de vide. l'entier est l'ordre du polynome ! pas la dimension de coefs.
      DATA dat; dat.SetX(RtoGP(X)); dat.SetValue(coefs(0));
      data_gp0.push_back(dat);
      dat.SetValue(coefs(1));
      data_gp1.push_back(dat);
      dat.SetValue(coefs(2));
      data_gp2.push_back(dat);
      dat.SetValue(coefs(3));
      data_gp3.push_back(dat);
      dat.SetValue(coefs(4));
      data_gp4.push_back(dat);
      dat.SetValue(coefs(5));
      data_gp5.push_back(dat);
    }    
    //hpars GP. on essaye avec les mêmes bornes d'opti pour le moment.
    int nhpars_gp=7;
    MatrixXd Bounds_hpars_gp(2,nhpars_gp);
    Bounds_hpars_gp(0,0)=1E-5; Bounds_hpars_gp(1,0)=1E6; //variance
    Bounds_hpars_gp(0,2)=1E-9; Bounds_hpars_gp(1,2)=1E-2; //sigma obs
    list<int> l={1,3,4,5,6};
    for (int i:l){
      Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=15; //lcors.
    }
    VectorXd hpars_gp_guess(7);
    //valeurs guessed avec les optimisations précédentes.
    hpars_gp_guess(0)=1e2;
    hpars_gp_guess(1)=0.32;
    hpars_gp_guess(2)=1e-6;
    hpars_gp_guess(3)=1.95;
    hpars_gp_guess(4)=1.93;
    hpars_gp_guess(5)=0.01;
    hpars_gp_guess(6)=1.53;

    VectorXd hpars_gp0(7);
    hpars_gp0 << 4.89e-4,12.39,3.77e-8,10.19,10.78,15,15; //matern décalé

    
    GP gp0(Kernel_GP_Matern);
    gp0.SetData(data_gp0);
    gp0.SetGP(hpars_gp0);

    
    VectorXd hpars_gp1(7);
    //sqe hpars_gp1 << 0.01,0.0336,1.13e-5,0.024,0.0498,0.017,0.254; //attention la var est à la borne..
    hpars_gp1 << 2.47e-2,0.93,1e-9,1.094,1.12,4.61,14.24; //matern décalé

    GP gp1(Kernel_GP_Matern);
    gp1.SetData(data_gp1);
    gp1.SetGP(hpars_gp1);

  
    VectorXd hpars_gp2(7);
    //sqe hpars_gp2 << 19.06,0.04,0.002,0.07,0.07,0.04,0.7;
    hpars_gp2<<10.06,0.78,1e-2,1.14,1.08,8.62,15; //matern décalé
    GP gp2(Kernel_GP_Matern);
    gp2.SetData(data_gp2);
    gp2.SetGP(hpars_gp2);

    VectorXd hpars_gp3(7);
    //sqe hpars_gp3 << 3292,0.06,0.002,0.07,0.12,0.01,0.7;
    hpars_gp3 << 805.97,0.232,1e-9,0.369,0.378,1.32,5.78; //maten décalé
    GP gp3(Kernel_GP_Matern);
    gp3.SetData(data_gp3);
    gp3.SetGP(hpars_gp3);

    VectorXd hpars_gp4(7);
    //sqe hpars_gp4 << 144037,0.06,0.002,0.05,0.01,0.247,0.01;
    hpars_gp4 << 218687,0.26,1e-9,0.344,0.43,1.616,4.64; //matern décalé
    GP gp4(Kernel_GP_Matern);
    gp4.SetData(data_gp4);
    gp4.SetGP(hpars_gp4);

    VectorXd hpars_gp5(7);
    hpars_gp5 << 1e6,0.10,5e-3,0.14,0.17,0.375,0.86; //matern. A refaire : variance au max
    GP gp5(Kernel_GP_Matern);
    gp5.SetData(data_gp5);
    gp5.SetGP(hpars_gp5);


    vgp[0]=gp0;
    vgp[1]=gp1;
    vgp[2]=gp2;
    vgp[3]=gp3;
    vgp[4]=gp4;
    vgp[5]=gp5;

    //évaluation des performances du vgp pour les diamètres
    VectorXd Xeval=mr[365][0]; //me["X"];//;
    VectorXd Yeval=mr[365][2];//me["Alpha"];//mr[365][1];
    VectorXd ParamEval(5);
    ParamEval << 0.5,0.5,0.5,0.5,0.5;
    ParamEval=RtoGP(m[365]); //paramètres du calcul numéro 365

    MatrixXd Pred=EvaluateMeanVarGPD(Xeval,vgp,ParamEval);
    MatrixXd Samples=DrawSamplesGPD(Xeval,vgp,ParamEval,3,generator);
    ofstream ofile("results/diampredGP2.gnu");
    VectorXd diamcoefs=polfitD(Yeval,Xeval,5);
    //cout << "true coefs : " << diamcoefs.transpose() << endl;
    for(int i=0;i<Xeval.size();i++){
      ofile << Xeval(i) << " " << Pred(i,0) << " " << sqrt(Pred(i,1)) <<" " << Yeval(i) << " " << Samples.col(0)(i) << " " << Samples.col(1)(i) << " " << Samples.col(2)(i) << endl;
    }
    ofile.close();

        /*Evaluation du vgp diamètre sur le DoE LHS. par polfit*/
    
    {
      VectorXd Xpred=mr_lhs[1][0]; //les x de prédiction du dts.
      double err_diamparoi=0;
      double err_diamcoeur=0;
      for(const auto &v:mr_lhs){
        //taux de vide moyen sur le dataset validation
        double dc=diamcoeur(v.second);
        double dp=diamparoi(v.second);
        //paramètre correspondant
        VectorXd Theta=m_lhs[v.first];
        //prédiction moyenne du vgp.
        VectorXd Pred =EvaluateMeanGPD(Xeval,vgp,RtoGP(Theta));
        //calcul du taux de vide moyen
        double dc2=Pred(0);
        double dp2=Pred(Pred.size()-1);
        //calcul de l'erreur relative : 
        double err_rel_dc=abs(dc-dc2)/dc;
        double err_rel_dp=abs(dp-dp2)/dp;
        //ajout : 
        err_diamparoi+=err_rel_dp;
        err_diamcoeur+=err_rel_dc;
      }
      err_diamparoi/=mr_lhs.size();
      err_diamcoeur/=mr_lhs.size();
      cout << "erreur relative diam paroi : " << err_diamparoi*100 << " pct" << endl;
      cout << "erreur relative diam coeur : " << err_diamcoeur*100 << " pct" << endl;
    }
    //affichage des directions principales
    {
      cout << "coef 0 : "; principal_directions(data_gp0,20);
      cout << "coef 1 : "; principal_directions(data_gp1,20);
      cout << "coef 2 : "; principal_directions(data_gp2,20);
      cout << "coef 3 : "; principal_directions(data_gp3,20);
      cout << "coef 4 : "; principal_directions(data_gp4,20);
      cout << "coef 5 : "; principal_directions(data_gp5,20);
    }

    /*Comparaison des prédictions sur un cas de training, et un cas de validation.*/
    {
      int nrtraining=500;
      int nrval=500;
      VectorXd Xeval_training=mr[nrtraining][0];
      VectorXd Xeval_validation=mr_lhs[nrval][0];
      //récupération des paramètres associés à ces calculs
      VectorXd Theta_training=m[nrtraining];
      VectorXd Theta_validation=m_lhs[nrval];
      //prédictions du vgp
      MatrixXd Pred_training=EvaluateMeanVarGPD(Xeval_training,vgp,RtoGP(Theta_training));
      MatrixXd Pred_validation=EvaluateMeanVarGPD(Xeval_validation,vgp,RtoGP(Theta_validation));
      //récupération des vraies valeurs
      VectorXd True_training=mr[nrtraining][2]; //2 pour le diamètre
      VectorXd True_validation=mr_lhs[nrval][2];
      MatrixXd Samples_training=DrawSamplesGPD(Xeval_training,vgp,RtoGP(Theta_training),3,generator);
      MatrixXd Samples_validation=DrawSamplesGPD(Xeval_validation,vgp,RtoGP(Theta_validation),3,generator);
      ofstream ofile("results/diamtraining.gnu");
      for(int i=0;i<Xeval_training.size();i++){
        ofile << Xeval_training(i) << " " << True_training(i) << " " << Pred_training(i,0) << " " << sqrt(Pred_training(i,1)) << 
        " " << Samples_training(i,0) << " " << Samples_training(i,1) << " " << Samples_training(i,2) << endl;
      }
      ofile.close();
      ofile.open("results/diamvali.gnu");
      for(int i=0;i<Xeval_validation.size();i++){
        ofile << Xeval_validation(i) << " " << True_validation(i) << " " << Pred_validation(i,0) << " " << sqrt(Pred_validation(i,1))  << 
        " " << Samples_validation(i,0) << " " << Samples_validation(i,1) << " " << Samples_validation(i,2) << endl;
      }
      ofile.close();      
    }
  }

  exit(0);
  

  /*test du p.g : construction sur un dataset réduit, et évaluation sur le complémentaire.*/
  /*
  {
    vector<DATA> data_reduit_gp;
    vector<DATA> data_verif_gp;
    for (auto &v:mr){
      VectorXd X=m[v.first];
      double tvmoyen=tauxmoyen(v.second);
      DATA dat; dat.SetX(RtoGP(X)); dat.SetValue(tvmoyen);
      if(distU(generator)>=0.9){
        data_verif_gp.push_back((dat));
      }
      else{data_reduit_gp.push_back(dat);}
    }
    //hpars GP.
    int nhpars_gp=7;
    MatrixXd Bounds_hpars_gp(2,nhpars_gp);
    Bounds_hpars_gp(0,0)=1E-3; Bounds_hpars_gp(1,0)=1E-1; //variance
    Bounds_hpars_gp(0,2)=1E-3; Bounds_hpars_gp(1,2)=2E-3; //sigma obs
    list<int> l={1,3,4,5,6};
    for (int i:l){
      Bounds_hpars_gp(0,i)=1E-2;Bounds_hpars_gp(1,i)=1; //lcors.
    }
    VectorXd hpars_gp_guess(7);
    for (int i=0;i<nhpars_gp;i++){
      hpars_gp_guess(i)=0.5*(Bounds_hpars_gp(1,i)+Bounds_hpars_gp(0,i));
    }
    
    GP gp(Kernel_GP);
    gp.SetData(data_reduit_gp);
    gp.SetGP(hpars_gp_guess);
    cout << "optimisation du GP réduit..." << endl;

    auto begin=chrono::steady_clock::now();
    gp.OptimizeGP(myoptfunc_gp,&Bounds_hpars_gp,&hpars_gp_guess,nhpars_gp);
    cout  << "par (Guess) : " << hpars_gp_guess.transpose() << endl;
	  hpars_gp_guess = gp.GetPar();
	  cout  << "par (Optim) (sigmaedm,lcorphi,sigmaobs,lcor2,lcor3,lcor4,lcor5) : " << hpars_gp_guess.transpose() << endl;
    auto end=chrono::steady_clock::now();
    cout << "time : " << chrono::duration_cast<chrono::seconds>(end-begin).count() << " s" << endl;

    //évaluation du gp sur le dataset de validation
    double err=0;
    for(DATA d:data_verif_gp){
      err+=abs(d.Value()-gp.EvalMean(d.GetX()))/d.Value();
    }
    err/=data_verif_gp.size();
    cout << "erreur relative moyenne : " << err*100 << " % sur " << data_verif_gp.size() << " data." << endl; 

    int ndisc=50;
    for (int indice=0;indice<5;indice++){
      string filename="results/tdvmoy"+to_string(indice)+".gnu";
      file.open(filename);
      for (int i=0;i<ndisc;i++){
        double var=(double (i))/ndisc;
        double tdv=tdv_moyen_given_vari(gp,indice,var,generator);
        file << var << " " << tdv << endl;
      }
      file.close();
    }
  }
  exit(0);
  */
};