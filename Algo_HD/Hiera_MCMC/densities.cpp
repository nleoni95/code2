//Création d'une classe pour la densité, et une classe pour le DoE.
#include "densities.h"
#include <ctime>
#include <random>

using namespace std;
using namespace Eigen;




/* Fonctions de la classe DoE*/
DoE::DoE()
{

};

DoE::DoE(VectorXd lb, VectorXd ub, int n): m_lb(lb),m_ub(ub),m_dimension(lb.size())
{
  //n correspond au nombre de points par dimension
  //initialisation en grid régulier ayant le même nombre de points par dimension.
  int npts=pow(n,m_dimension);
  VectorXd theta_courant(m_dimension);
  VectorXd ind_courant(m_dimension);
  double Vol=1;
  for(int i=0;i<m_dimension;i++){Vol*=(m_ub(i)-m_lb(i));}
  if (Vol<=0){cout << "warning : volume négatif. Are you sure ?" << endl;}
  for(int i=0;i<npts;i++){
    ind_courant=indices(i,n,m_dimension);
    for (int j=0;j<m_dimension;j++){
      theta_courant(j)=m_lb(j)+(ind_courant(j)+0.5)*(m_ub(j)-m_lb(j))/double(n);
    }
    m_grid.push_back(theta_courant);
    m_weights.push_back(Vol/double(npts));
  }
};

DoE::DoE(VectorXd lb, VectorXd ub, int ntotal,std::default_random_engine &generator): m_lb(lb),m_ub(ub),m_dimension(lb.size())
{
  std::uniform_real_distribution<double> distU(0,1);
  //Construction en LHS uniforme.
  // n correspond au nombre de points dans le grid.
  double Vol=1;
  for(int i=0;i<m_dimension;i++){Vol*=(ub(i)-lb(i));}
  // division de chaque dimension en npoints : on génère m_dimension permutations de {0,npoints-1}.
  std::vector<VectorXd> perm(m_dimension);
  for (int i=0;i<m_dimension;i++){
    perm[i]=Randpert(ntotal);
  }
  // calcul des coordonnées de chaque point par un LHS.
  VectorXd theta_courant(m_dimension);
  for(int i=0;i<ntotal;i++){
    for (int j=0;j<m_dimension;j++){
      theta_courant(j)=lb(j)+(ub(j)-lb(j))*(perm[j](i)+distU(generator))/double(ntotal);
    }
    m_grid.push_back(theta_courant);
    m_weights.push_back(Vol/double(ntotal));   
  }
};

void DoE::Fill(double (*my_model)(VectorXd const &,VectorXd const &),vector<DATA> *obs)
{
  //déjà on set le modèle et les observations
  SetModel(my_model);
  SetObs(*obs);
  clock_t c_start = std::clock();
  cout << "Building DoE with " << m_grid.size() << " points and " << obs->size() << " observations. Total number of points : " << obs->size()*m_grid.size() << endl;
  for (int i=0;i<m_grid.size();i++)
  {
    VectorXd X(obs->size());
    for (int j=0;j<obs->size();j++)
    {
      X(j)=my_model((*obs)[j].GetX(),m_grid[i]);
    }
    m_model_evals.push_back(X);
  }

  clock_t c_end = std::clock();
  double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  double time_exec_doe=time_elapsed_ms / 1000.0;
  std::cout << "Temps pour construction du DoE : " << time_exec_doe << " s\n";
} //remplit le DoE avec un modèle et observations données.

void DoE::Fill_Predictions(vector<VectorXd> &vectorx)
{
  m_vectorx=vectorx;
  //On évalue le modèle sur chaque point dont on fera la prédiction.
  clock_t c_start = std::clock();
  cout << "Evaluate model on " << m_grid.size() << " parameter points and " << vectorx.size() << " sample points. Total number of points : " << vectorx.size()*m_grid.size() << endl;
  for (int i=0;i<vectorx.size();i++){
    VectorXd Model(m_grid.size());
    for (int j=0;j<m_grid.size();j++){
      Model(j)=m_my_model(m_vectorx[i],m_grid[j]);
    }
    m_model_evals_predictions.push_back(Model);
  }
  clock_t c_end = std::clock();
  double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  double time_exec_doe=time_elapsed_ms / 1000.0;
  std::cout << "Temps pour évaluation du modèle : " << time_exec_doe << " s\n";
} //remplit le DoE avec un modèle et observations données.
  
VectorXd DoE::Randpert(int const n){
  VectorXd result(n);
  std::default_random_engine generator;
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

VectorXd DoE::indices(int const s, int const n, int const d)
{
  //renvoie le multi-indice correspondant à l'indice courant s dans un tableau de dimension d et de taille n dans chaque direction.
  VectorXd multiindice(d);
  int indloc;
  int remainder=s;
  for(int pp=d-1;pp>-1;pp--){
    indloc=(int) remainder % n; //On commence par le coefficient le plus à droite.
    multiindice(pp)=indloc;
    remainder=(remainder-indloc)/n;
  }
  return multiindice;
};

int DoE::grosindice(VectorXd const &v, int const n)
{
  //fonction réciproque de indices
  // n : taille du tableau dans une dimension
  //on compte bien de la façon suivante : 000 001 010 011 etc.
  int p=v.size(); //dimension du tableau
  int ans=0;
  for (int i=0;i<p;i++){
    ans+=v(p-1-i)*pow(n,i);
  }
  return ans;
};
  
/*Fonctions de la classe Density*/
Density::Density()
{


};

Density::Density(Density &d) : m_Kernel(d.m_Kernel),m_logpriorhpars(d.m_logpriorhpars),m_logpriorpars(d.m_logpriorpars),m_my_model(d.m_my_model),m_priormean(d.m_priormean)
{
  m_npts=d.m_npts;
  m_Grid=d.m_Grid;
  m_lb_hpars=d.m_lb_hpars;
  m_ub_hpars=d.m_ub_hpars;
  m_obs=d.m_obs;
  m_dim_hpars=d.m_dim_hpars;
  m_dim_pars=d.m_dim_pars;
  m_values=d.m_values;
  m_model_evals=d.m_model_evals;
  m_model_evals_predictions=d.m_model_evals_predictions;
  m_vectorx=d.m_vectorx;
  m_norm_cst=d.m_norm_cst;
  m_filtre=d.m_filtre;
};

Density::Density(DoE g) : m_Grid(g),m_npts(g.GetGrid().size()),m_dim_pars(g.GetDimension()),m_model_evals_predictions(g.GetModelPredEvals()),m_model_evals(g.GetModelEvals()),m_vectorx(g.GetVectorX()),m_obs(g.GetObs())
{


};

vector<double> Density::RandHpars(default_random_engine &generator) const {
//renvoie une valeur d'hyperparamètes aléatoire dans les bornes (pour un first guess d'optimisation)
std::uniform_real_distribution<double> distU(0,1);
vector<double> x(m_dim_hpars);
  for (int j=0;j<m_dim_hpars;j++){
    if(m_lb_hpars[j]>0 && m_lb_hpars[j]>0 ){
      x[j]=exp(log(m_lb_hpars[j])+(log(m_ub_hpars[j])-log(m_lb_hpars[j]))*distU(generator));
    }
    else {
      x[j]=m_lb_hpars[j]+(m_ub_hpars[j]-m_lb_hpars[j])*distU(generator);
    }
  }
  //si les bornes sont positives, on fait moyenne log. Sinon moyenne arithmétique
return x;
}


void Density::Build(Eigen::VectorXd const &hpars){
  //inversion de la matrice de covariance
  int nd=m_obs.size();
  MatrixXd G=Gamma(&m_obs,hpars);
  VectorXd obs(nd);
  for(unsigned i=0; i<nd; i++) obs(i) = m_obs[i].Value(); // copie des valeurs observées dans un VectorXd
  LDLT<MatrixXd> ldlt(G);
  for (int i=0;i<m_npts;i++){
    m_values.push_back(m_logpriorpars(GetDoE().GetGrid()[i])+
      loglikelihood_theta_fast(i,hpars,ldlt));
  }
}

void Density::WritePost(const char* file_name)
{
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<m_npts;i++){
    for (int j=0;j<m_Grid.GetDimension();j++){
      fprintf(out,"%e ",m_Grid.GetGrid()[i](j));
    }
    fprintf(out,"%e\n",m_values[i]);
  }
  fclose(out);
};

double Density::Entropy() const
{
  double ent=0;
  for(int i=0;i<m_npts;i++){ent-=m_Grid.GetWeights()[i]*m_values[i]*log(m_values[i]);}
  return ent;
};
int Density::indmax() const 
{
  return distance(m_values.begin(),max_element(m_values.begin(),m_values.end()));
}

Eigen::VectorXd Density::MAP() const
{
  return m_Grid.GetGrid()[indmax()];
};

Eigen::VectorXd Density::Mean() const
{
  VectorXd mean=VectorXd::Zero(m_Grid.GetDimension());
  for(int i=0;i<m_npts;i++){mean+=m_Grid.GetWeights()[i]*m_values[i]*m_Grid.GetGrid()[i];}
  return mean;
};

Eigen::MatrixXd Density::Cov() const
{
  MatrixXd cov=MatrixXd::Zero(m_Grid.GetDimension(),m_Grid.GetDimension());
  VectorXd mean=Mean();
  for(int i=0;i<m_npts;i++){cov+=(m_Grid.GetGrid()[i]-mean)*(m_Grid.GetGrid()[i]-mean).transpose()*m_Grid.GetWeights()[i]*m_values[i];}
  return cov;
};

double Density::KLDiv(Density &d) const
{
  //calcule la KLDiv de cette densité à la densité d. D'abord check voir si il n'y a pas de zéro.
  if(m_Grid.GetGrid()!=d.GetDoE().GetGrid() | m_Grid.GetWeights()!=d.GetDoE().GetWeights())
  {
    cout << "pas définies sur le même grid !" << endl;
    return 0;
  }
  for(int i=0;i<m_npts;i++){
    if(m_values[i]==0 | d.GetValues()[i]==0)
    {
      cout << "une densité vaut 0 !" << endl;
      return 0;
    }
  }   
  double kldiv=0;
  for (int i=0;i<m_npts;i++)
  {
    kldiv+=m_Grid.GetWeights()[i]*m_values[i]*log(m_values[i]/d.GetValues()[i]);
  }
  return kldiv;
}

void Density::vectorweight1()
{
  double weight=0;
  for(int i=0;i<m_npts;i++)
  {
    weight+=m_Grid.GetWeights()[i]*m_values[i];
  }
  for(int i=0;i<m_npts;i++){m_values[i]/=weight;}
};

void Density::vectorlog()
{
  for(int i=0;i<m_npts;i++){m_values[i]=log(m_values[i]);}
};

void Density::vectorexp()
{
  //on passe à l'exponentielle, mais d'abord on a retiré le plus grand élément. Ainsi on évite les overflow.
  //si jamais l'écart est tellement grand que la probabilité vaut 0, pas grave.
  double maximum=*max_element(m_values.begin(),m_values.end());
  for(int i=0;i<m_npts;i++){m_values[i]=exp(m_values[i]-maximum);}
};


Eigen::MatrixXd Density::Gamma(void const *data, Eigen::VectorXd const &hpar) const {
  // Renvoie la matrice de corrélation avec  bruit
  vector<DATA>* data2 = (vector<DATA>*) data; // cast du null pointer en type désiré
  int nd=data2->size();
  Eigen::MatrixXd A(nd,nd);
  for(int i=0; i<nd; i++){
    for(int j=i; j<nd; j++){
      A(i,j) = m_Kernel((*data2)[i].GetX(),(*data2)[j].GetX(), hpar);
      if(i!=j){
	A(j,i) = A(i,j);
      }else{
	A(i,j) += hpar(1)*hpar(1);					//Noise correlation
      }
    }
  }
  return A;
}

double Density::loglikelihood_fast(VectorXd const &obs, VectorXd const &Alpha, Eigen::VectorXd const &hpar, LDLT<MatrixXd> const &ldlt)const{
  //renvoie la log-vraisemblance des hyperparamètres hpar, étant donné les données data.
  int nd=obs.size();
  return -0.5*obs.dot(Alpha)-0.5*(ldlt.vectorD().array().log()).sum() -0.5*nd*log(2*3.1415);
}

double Density::loglikelihood_theta_fast(int d, Eigen::VectorXd const &hpar, LDLT<MatrixXd> const &ldlt)const{
  //renvoie log p(y|pars,hpars), pars étant le d-ième point du DoEen type désiré
  int nd=m_obs.size();
  VectorXd obs_theta(nd);
  for (int i=0;i<nd;i++){obs_theta(i)=m_obs[i].Value()-(m_model_evals)[d](i)-m_priormean(m_obs[i].GetX(),hpar);}
  VectorXd Alpha=ldlt.solve(obs_theta);
  return loglikelihood_fast(obs_theta,Alpha,hpar,ldlt);
}

double Density::loglikelihood(void *data, Eigen::VectorXd const &hpar)const{
  //renvoie la log-vraisemblance des hyperparamètres hpar, étant donné les données data.
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  MatrixXd G=Gamma(data2,hpar);
  VectorXd obs(nd);
  for(unsigned i=0; i<nd; i++) obs(i) = (*data2)[i].Value(); // copie des valeurs observées dans un VectorXd
  LDLT<MatrixXd> ldlt(G);
  VectorXd Alpha=ldlt.solve(obs);
  return loglikelihood_fast(obs,Alpha,hpar,ldlt);
}

double Density::loglikelihood_theta(int d, Eigen::VectorXd const &hpar)const{
  //renvoie la log-vraisemblance des hyperparamètres hpar, étant donné les données data et les paramètres du modèle theta.en type désiré
  int nd=m_obs.size();
  std::vector<DATA> data3;
  for(unsigned ie=0; ie<nd; ie++){
    DATA dat; dat.SetX(m_obs[ie].GetX()); dat.SetValue(m_obs[ie].Value()-(m_model_evals)[d](ie)-m_priormean(m_obs[ie].GetX(),hpar));
    data3.push_back(dat);
  }
  return loglikelihood(&data3,hpar);
}

void Density::ComputeFiltre(double rapport)
{
  //on cherche le maximum du vecteur m_values*m_weights et on vérifie pour chaque theta si le rapport (proba*poids) par rapport à ce max est inférieur à l'argument de cette méthode.
  //sert à filtrer les valeurs de theta qui n'auront pas de poids dans la prédiction, pour éviter de les calculer.
  //en argument : le rapport des probabilités pour filtrer.
  double weight(0);
  vector<double> produit(m_values.size());
  vector<bool> filtre(m_values.size());
  for (int i=0;i<produit.size();i++){produit[i]=m_Grid.GetWeights()[i]*m_values[i];}
  double max=*max_element(produit.begin(),produit.end());
  for (int i=0;i<filtre.size();i++){
    if(produit[i]==0){
      filtre[i]=false;
    }
    else if (max/produit[i]>= rapport){
      filtre[i]=false;
    }
    else
    {
      filtre[i]=true;
      weight+=produit[i];
    }
  }
  m_filtre=filtre;
}

void Density::FiltreDensite()
{
  for(int i=0;i<m_values.size();i++)
  {
    if(!m_filtre[i])
    {
      m_values[i]=0;
    }
  }
}

void Density::WriteMapToFile(map<VectorXd,VectorXd,map_compare> &map, std::string filename) const 
{
  FILE* out=fopen(&filename[0],"w"); //pour convertir std::string en char*
  for (const auto &Pair : map)
  {
    VectorXd t1=Pair.first;
    VectorXd t2=Pair.second;
    for (int i=0;i<t1.size();i++){
      fprintf(out,"%e ",t1(i));
    }
     for (int i=0;i<t2.size();i++){
      fprintf(out,"%e ",t2(i));
    }
    fprintf(out,"\n");
  }
  fclose(out);
}

void Density::WriteMapToFile(map<VectorXd,double,map_compare> &map, std::string filename) const 
{
  FILE* out=fopen(&filename[0],"w"); //pour convertir std::string en char*
  for (const auto &Pair : map)
  {
    VectorXd t1=Pair.first;
    double t2=Pair.second;
    for (int i=0;i<t1.size();i++){
      fprintf(out,"%e ",t1(i));
    }
    fprintf(out,"%e\n ",t2);
  }
  fclose(out);
}

map<VectorXd,double,map_compare> Density::Marg1D(int h1) const
{
  //ecriture d'une marginale 1D selon une map pour pouvoir faire des courbes.
  if (h1<0 || h1>= m_dim_pars){cout << "erreur de dimension !" << endl;}
  map<VectorXd,double,map_compare> pars;
  map<VectorXd,double,map_compare> weights;
  for (int i=0;i<m_npts;i++)
  {
    VectorXd thetacourant=m_Grid.GetGrid()[i];
    VectorXd thetacourantreduit(1);
    thetacourantreduit(0)=thetacourant(h1); //on regarde selon la projection sur l'indice h1.
    map<VectorXd,double,map_compare>::iterator it=pars.find(thetacourantreduit);
    if (it==pars.end()){pars[thetacourantreduit]=0;}
    //apparemment les valeurs de la map sont intialisées à 0 la première fois qu'on les appelle. A vérifier si ça plante.
    //cout << hpars[thetacourantreduit].transpose()<< endl;
    //cout << (m_hpars_opti[i]*m_values[i]*m_Grid.GetWeights()[i]).transpose() << endl;
    pars[thetacourantreduit]+=m_values[i]*m_Grid.GetWeights()[i];
    weights[thetacourantreduit]+=m_Grid.GetWeights()[i];
  }
  for (const auto &Pair : pars)
  {
    if (!weights[Pair.first]==0){pars[Pair.first]/=weights[Pair.first];}
  }
  return pars;
}

map<VectorXd,double,map_compare> Density::Marg2D(int h1, int h2) const
{
  //ecriture d'une marginale 1D selon une map pour pouvoir faire des courbes.
  if (h1<0 || h1>= m_dim_pars){cout << "erreur de dimension !" << endl;}
   if (h2<0 || h2>= m_dim_pars){cout << "erreur de dimension !" << endl;}
  map<VectorXd,double,map_compare> pars;
  map<VectorXd,double,map_compare> weights;
  for (int i=0;i<m_npts;i++)
  {
    VectorXd thetacourant=m_Grid.GetGrid()[i];
    VectorXd thetacourantreduit(2);
    thetacourantreduit(0)=thetacourant(h1); //on regarde selon la projection sur l'indice h1.
    thetacourantreduit(1)=thetacourant(h2); //on regarde selon la projection sur l'indice h1.
    map<VectorXd,double,map_compare>::iterator it=pars.find(thetacourantreduit);
    if (it==pars.end()){pars[thetacourantreduit]=0;}
    //apparemment les valeurs de la map sont intialisées à 0 la première fois qu'on les appelle. A vérifier si ça plante.
    //cout << hpars[thetacourantreduit].transpose()<< endl;
    //cout << (m_hpars_opti[i]*m_values[i]*m_Grid.GetWeights()[i]).transpose() << endl;
    pars[thetacourantreduit]+=m_values[i]*m_Grid.GetWeights()[i];
    weights[thetacourantreduit]+=m_Grid.GetWeights()[i];
  }
  for (const auto &Pair : pars)
  {
    if (!weights[Pair.first]==0){pars[Pair.first]/=weights[Pair.first];}
  }
  return pars;
}

void Density::WriteMarginals(const char* folder_name) const
{
  std::string foldname(folder_name);
  map<VectorXd,double,map_compare> map1=Marg1D(0);
  WriteMapToFile(map1,foldname+"1.gnu");
  map<VectorXd,double,map_compare> map2=Marg1D(1);
  WriteMapToFile(map2,foldname+"2.gnu");
  map<VectorXd,double,map_compare> map3=Marg1D(2);
  WriteMapToFile(map3,foldname+"3.gnu");
  map<VectorXd,double,map_compare> map12=Marg2D(1,0);
  WriteMapToFile(map12,foldname+"21.gnu");
  map<VectorXd,double,map_compare> map23=Marg2D(2,1);
  WriteMapToFile(map23,foldname+"32.gnu");
  map<VectorXd,double,map_compare> map31=Marg2D(2,0);
  WriteMapToFile(map31,foldname+"31.gnu");
};

MatrixXd Density::VarParamUncertainty() const {
  //Calcul de Var(f_theta) aux points du vectorx. Puisque cette variance est indépendante entre les points, on renvoie une matrice diagonale.
  //first : récupérer le vecteur d'évaluations du modèle aux points d'intérêt.
  int xsize=m_vectorx.size();
  //un premier calcul pour savoir combien de thetas ont une probabilité non nulle
  int tsize(0);
  for (int i=0;i<m_npts;i++){if(m_filtre[i]){++tsize;}}
  //on fait les évaluations du modèle
  vector<VectorXd> model_evals(xsize);
  VectorXd weights(tsize);
  int c(0);
  for (int i=0;i<m_npts;i++){if(m_filtre[i]){weights(c)=m_values[i]*m_Grid.GetWeights()[i];++c;}}
  for (int i=0;i<xsize;i++)
  {
    VectorXd X(tsize);
    c=0;
    for (int j=0;j<m_npts;j++){
      if(m_filtre[j]){
        X(c)=(m_model_evals_predictions)[i](j);
        ++c;
      }
    }
    model_evals[i]=X;
  }
  //ok j'ai mon vecteur d'évaluations du modèle. Maintenant il faut calculer sa variance.
  MatrixXd COV=MatrixXd::Zero(xsize,xsize);
  for (int i=0;i<xsize;i++){
    double var(0);
    double esp(0);
    for (int j=0;j<tsize;j++){
      esp+=weights(j)*model_evals[i](j);
      var+=weights(j)*pow(model_evals[i](j),2);
    }
    COV(i,i)=var-pow(esp,2);
  }
  return COV;
}

VectorXd Density::DrawMVN (VectorXd &Mean, MatrixXd &COV, default_random_engine &generator) {
  std::normal_distribution<double> distN(0,1);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> DEC(COV);
	Eigen::VectorXd D = DEC.eigenvalues();
  for(unsigned i=0; i<D.rows(); i++) D(i) = sqrt(fabs(D(i)));
  Eigen::VectorXd Sample(Mean.size());
  Eigen::VectorXd Eta(COV.cols());
	for(unsigned i=0; i<COV.cols(); i++) Eta(i) = distN(generator)*D(i);
	Sample = DEC.eigenvectors()*Eta + Mean;
  return Sample;
}

DensityKOH::DensityKOH(DoE g) : Density(g){
  
}

DensityKOH::DensityKOH(Density &d, int c) : Density(d){
  m_norm_cst=c;
}

void DensityKOH::Build()
{
  VectorXd hparskoh=HparsKOH();
  Density::Build(hparskoh);
  vectorexp();
  vectorweight1();
  ComputeFiltre(100);
  FiltreDensite();
  vectorweight1();
}

double DensityKOH::optfunc(const std::vector<double> &x, std::vector<double> &grad, void *data){
  DensityKOH* D = (DensityKOH*) data; //pointer to densitykoh
  vector<DATA>* obs; //j'ai fait ça car je n'arrivais pas à obtenir simplement un pointeur vers les obs de D.
  vector<DATA> observations=(D->GetObs());
  obs=&observations;
  /*Fonction à optimiser, Kennedy O'Hagan. On cherche à estimer l'intégrale moyennée sur un grid uniforme (avec priors uniformes) */
  Eigen::VectorXd hpar(D->m_dim_hpars);
  int npts=D->GetNpts();
  for(int p=0; p<D->m_dim_hpars; p++) {hpar(p) =x[p];} // copie des hyperparamètres dans un VectorXd
  MatrixXd G=D->Gamma(obs,hpar); //les valeurs de data2 ne sont pas correctes car non retranchées de ftheta. Cependant on n'utilise que les X pour calculer G.
  LDLT<MatrixXd> ldlt(G);
  double avg=0;
  for (int i=0;i<npts;i++){
    double ok=exp(D->loglikelihood_theta_fast(i,hpar,ldlt)-D->GetNormCst());
    //if (ok>1){cout << "logvs supérieure au max opti : "<< D->loglikelihood_theta_fast(i,hpar,ldlt) << endl;
    //cout << "avec pars : " << D->GetDoE().GetGrid()[i].transpose() << " et hpars " << hpar.transpose() << endl;
    //}
    avg+=ok;
  }
  avg=avg*exp(D->m_logpriorhpars(hpar));
  return avg;
}

VectorXd DensityKOH::HparsKOH() {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distU(0,1);
  /*paramètres de l'optimisation*/
  VectorXd hpars_max_koh(m_dim_hpars);
  double time_exec_koh;
  int maxeval=5000;
  double ftol=1e-3;
  cout << "Début de l'optimisation KOH..." << endl;
  /*Pointer to member*/
  {
    clock_t c_start = std::clock();
    //1ère étape : optimisation globale par algo génétique
    {
      int pop=2000; // population size
      //cout << "Début de l'optimisation globale... Pop size : " << pop << endl;
      /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
      std::vector<double> x(m_dim_hpars);
      x=RandHpars(generator); // initial guess
      nlopt::opt opt(nlopt::GN_ISRES, m_dim_hpars);    /* algorithm and dimensionality */
      opt.set_max_objective(DensityKOH::optfunc, this);

      opt.set_lower_bounds(m_lb_hpars);
      opt.set_upper_bounds(m_ub_hpars);
      opt.set_maxeval(maxeval);
      opt.set_population(pop);
      opt.set_ftol_rel(ftol);
      double msup; /* the maximum objective value, upon return */

      int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 

      for(int i=0;i<m_dim_hpars;i++){hpars_max_koh(i)=x[i];}
      cout << "Message d'arrêt : " << fin <<". max trouvé : " << hpars_max_koh.transpose() << ", valeur du critère : " << msup <<endl;
    }
    //2ème étape : optimisation locale par sbplx en partant du point trouvé précédemment.
    {
      //cout << "Début de l'optimisation locale..." << endl;
      double msuploc;
      /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
      std::vector<double> x(m_dim_hpars);
      for (int j=0;j<m_dim_hpars;j++){x[j]=hpars_max_koh(j);}
      nlopt::opt opt(nlopt::LN_SBPLX, m_dim_hpars);    /* algorithm and dimensionality */
      opt.set_max_objective(DensityKOH::optfunc, this);
      opt.set_lower_bounds(m_lb_hpars);
      opt.set_upper_bounds(m_ub_hpars);
      opt.set_maxeval(maxeval);
      opt.set_ftol_rel(ftol);
      double msup; /* the maximum objective value, upon return */
      int fin=opt.optimize(x, msup);
      for(int i=0;i<m_dim_hpars;i++){hpars_max_koh(i)=x[i];}
      msuploc=msup;
      cout << "Message d'arrêt : " << fin  << ". Max : " << hpars_max_koh.transpose() << ", valeur du critère : " << msuploc << endl;
      int niter_optimisations=500; // On recommence l'opti tant qu'on n'a pas stagné 100 fois.
      int nopt=0;
      while (nopt<niter_optimisations){
	nopt++;
	x=RandHpars(generator);
	int fin= opt.optimize(x, msup);
	if(msup>msuploc){
	  nopt=0;
	  msuploc=msup;
	  for(int k=0;k<m_dim_hpars;k++){
	    hpars_max_koh(k)=x[k];
	  }
	  //cout << "Message d'arrêt : " << fin  << ". Nouveau max : " << hpars_max_koh.transpose() << ", valeur du critère : " << msuploc << endl;
	}
      }
      double max_vrais_koh=msuploc;
      cout << "hyperparametres KOH : (edm, exp, lcor) : " << hpars_max_koh.transpose() << " a la vraisemblance : " << max_vrais_koh << endl;
    }
    clock_t c_end = std::clock();
    double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    time_exec_koh=time_elapsed_ms / 1000.0;
    std::cout << "Temps pour opti KOH : " << time_exec_koh << " s\n";
  }
  m_hpars=hpars_max_koh;
  return hpars_max_koh;
}


VectorXd DensityKOH::FPredCondTheta(int t) const{
  //renvoie f(xpred,theta)
  VectorXd Thetacourant=m_Grid.GetGrid()[t]; //valeur de theta utilisée dans la fonction
  VectorXd Hparscourant=m_hpars; //valeur des hpars
  VectorXd Fpred(m_vectorx.size());
  for(int i=0;i<Fpred.size();i++)
  {
    Fpred(i)=(m_model_evals_predictions)[i](t);
  }
  return Fpred;
}

VectorXd DensityKOH::ZMeanCondTheta(int t) const{
  //renvoie postmean de z given theta
  //revient à postmean de z pour KOH
  VectorXd Thetacourant=m_Grid.GetGrid()[t]; //valeur de theta utilisée dans la fonction
  VectorXd Hparscourant=m_hpars; //valeur des hpars
  VectorXd Zmean(m_vectorx.size());
  VectorXd y(m_obs.size());
  for (int i=0;i<y.size();i++){
    y(i)=m_obs[i].Value()-((m_model_evals)[t](i)+m_priormean(m_obs[i].GetX(),Hparscourant));
  }
  MatrixXd Kstar(m_vectorx.size(),m_obs.size());
  for (int i=0;i<m_vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar(i,j)=m_Kernel(m_vectorx[i],m_obs[j].GetX(),Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  VectorXd predmean=Kstar*ldlt.solve(y);
  for(int i=0;i<Zmean.size();i++)
  {
    Zmean(i)=predmean(i)+m_priormean(m_vectorx[i],Hparscourant);
  }
  return Zmean;
}

MatrixXd DensityKOH::VarZCondTheta(int t) const{
  //renvoie la variance de z a posteriori given theta
  VectorXd Hparscourant=m_hpars; //valeur des hpars
  MatrixXd Kstar1(m_vectorx.size(),m_obs.size());
  MatrixXd Kstar2(m_obs.size(),m_vectorx.size());
  for (int i=0;i<m_vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar1(i,j)=m_Kernel(m_vectorx[i],m_obs[j].GetX(),Hparscourant);
      Kstar2(j,i)=m_Kernel(m_obs[j].GetX(),m_vectorx[i],Hparscourant);
    }
  }
  MatrixXd Kprior(m_vectorx.size(),m_vectorx.size());
  for (int i=0;i<m_vectorx.size();i++){
    for (int j=0;j<m_vectorx.size();j++){
      Kprior(i,j)=m_Kernel(m_vectorx[i],m_vectorx[j],Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  MatrixXd varred=Kstar1*ldlt.solve(Kstar2);
  return Kprior-varred;
}

VectorXd DensityKOH::EspF() const{
  //renvoie E_{theta}[f]
  VectorXd pred=VectorXd::Zero(m_vectorx.size());
  for(int t=0;t<m_npts;t++)
  {
    if(m_filtre[t]){
      VectorXd pred_theta=FPredCondTheta(t);
      pred+=pred_theta*m_Grid.GetWeights()[t]*m_values[t];
    }
  }
  return pred;
}

VectorXd DensityKOH::EspZ() const{
  //renvoie E_{theta}[z]
  VectorXd pred=VectorXd::Zero(m_vectorx.size());
  for(int t=0;t<m_npts;t++)
  {
    if(m_filtre[t]){
      VectorXd pred_theta=ZMeanCondTheta(t);
      pred+=pred_theta*m_Grid.GetWeights()[t]*m_values[t];
    }
  }
  return pred;
}

MatrixXd DensityKOH::EspVarZ() const{
  //renvoie E_{theta}[Var z]
  MatrixXd pred=MatrixXd::Zero(m_vectorx.size(),m_vectorx.size());
  for(int t=0;t<m_npts;t++)
  {
    if(m_filtre[t]){
      MatrixXd pred_theta=VarZCondTheta(t);
      pred+=pred_theta*m_Grid.GetWeights()[t]*m_values[t];
    }
  }
  return pred;
}

MatrixXd DensityKOH::VarF() const{
  //renvoie Var_{theta}[f]
  //revient à Var_{theta}[f] pour KOH
  //renvoie une matrice carrée (m_vectorx.size())
  int xsize=m_vectorx.size();
  MatrixXd COV=MatrixXd::Zero(xsize,xsize);
  //on récupère le vecteur d'évaluations du modèle.
  //un premier calcul pour savoir combien de thetas ont une probabilité non nulle
  int tsize(0); for (int i=0;i<m_npts;i++){if(m_filtre[i]){++tsize;}}
  //on fait les évaluations du modèle
  int c(0);
  VectorXd weights(tsize); for (int i=0;i<m_npts;i++){if(m_filtre[i]){weights(c)=m_values[i]*m_Grid.GetWeights()[i];++c;}}
  MatrixXd model_evals(xsize,tsize);
  c=0;
  for (int i=0;i<m_npts;i++){
    if(m_filtre[i]){
      model_evals.col(c)=FPredCondTheta(i);
      ++c;
    }
  }
  for (int i=0;i<xsize;i++){
    double var(0);
    double esp(0);
    for (int j=0;j<tsize;j++){
      esp+=weights(j)*model_evals(i,j);
      var+=weights(j)*pow(model_evals(i,j),2);
    }
  COV(i,i)=var-pow(esp,2);
  }
  return COV;
}

VectorXd DensityKOH::DrawSample(default_random_engine &generator) const{
  //tirage d'un theta aléatoire
  std::uniform_real_distribution<double> distU(0,1);
  double u=distU(generator);
  double sum(0);
  int indcour(0);
  for (int i=0;i<m_npts;i++){
    sum+=m_Grid.GetWeights()[i]*m_values[i];
    if(sum>u){
      indcour=i;
      break;
    }
  }
  //tire un échantillon de la prédiction de f+z sans edm.
  VectorXd avg=FPredCondTheta(indcour)+ZMeanCondTheta(indcour); //renvoie les prédictions moyennes
  MatrixXd COV=VarZCondTheta(0); //calcul de la matrice de covariance de prédiction.
  //peu importe de mettre 0 car la variance de z ne dépend pas de theta ici.
  VectorXd Sample=DrawMVN(avg,COV,generator);
  return Sample;
}


void DensityKOH::WritePredictions(const char* file_name) const{
  //Ecriture de la prédiction moyenne et de l'incertitude associée, obtenue par méthode KOH. On fait 3 écart-type.
  VectorXd Mean=EspF()+EspZ();//prédiction moyenne f+z
  MatrixXd EDM=EspVarZ(); //variance de prédiction de z : E[c* -c*K-1c*] : "erreur de prédiction due à l'edm"
  MatrixXd Varf=VarF(); //variance de f_theta+z_theta : "erreur de prédiction due à l'incertitude paramétrique"
  //on écrit dans le fichier directement les prédictions (et non les écarts)
  VectorXd Contribf(Mean.size()),ContribEDM(Mean.size());
  for (int i=0;i<Mean.size();i++)
  {
    Contribf(i)=2*Varf.diagonal()(i)/(sqrt(Varf.diagonal()(i)+EDM.diagonal()(i)));
    ContribEDM(i)=2*EDM.diagonal()(i)/(sqrt(Varf.diagonal()(i)+EDM.diagonal()(i)));
    //Contribf(i)=3*sqrt(Varf.diagonal()(i)); //termes absolus
    //ContribEDM(i)=3*sqrt(EDM.diagonal()(i));
  }
  FILE* out=fopen(file_name,"w");
  fprintf(out, "#X, Pred Mean, stdf, stdedm\n");
  for (int i=0;i<Mean.size();i++)
  {
    fprintf(out, "%e %e %e %e\n",m_vectorx[i](0),Mean(i),Contribf(i),ContribEDM(i));
  }
  fclose(out);
}

void DensityKOH::WritePredictionsFZ(const char* file_name) const{
  //Ecriture de la prédiction moyenne et de l'incertitude associée, obtenue par méthode KOH. On fait 3 écart-type.
  VectorXd MeanF=EspF();
  VectorXd MeanZ=EspZ();
  MatrixXd Varf=VarF(); //variance de f_theta+z_theta : "erreur de prédiction due à l'incertitude paramétrique"
  //on écrit dans le fichier directement les prédictions (et non les écarts)
  VectorXd Contribf(MeanF.size());
  for (int i=0;i<MeanF.size();i++)
  {
    Contribf(i)=2*sqrt(Varf.diagonal()(i));
  }
  FILE* out=fopen(file_name,"w");
  fprintf(out, "#X, Pred Mean f, stdf, Pred Mean z\n");
  for (int i=0;i<MeanF.size();i++)
  {
    fprintf(out, "%e %e %e %e\n",m_vectorx[i](0),MeanF(i),Contribf(i),MeanZ(i));
  }
  fclose(out);
}
  
DensityOpt::DensityOpt(Density &d) : Density(d){
}

DensityOpt::DensityOpt(DoE g) : Density(g){
}

void DensityOpt::WriteMeanPreds(const char* file_name) const{
  //Ecriture de toutes les prédictions moyennes
  FILE* out=fopen(file_name,"w");
  vector<VectorXd> Preds;
  for(int t=0;t<m_npts;t++){
    if(m_filtre[t]){
      VectorXd Pred=FPredCondTheta(t)+ZMeanCondTheta(t);
      Preds.push_back(Pred);
    }
  }
  //phase de test : pondération par les probabilités.
  for (int i=0;i<m_vectorx.size();i++){
    fprintf(out,"%e ",m_vectorx[i](0));
    for(int j=0;j<Preds.size();j++){
      fprintf(out,"%e ",Preds[j](i));
    }
    fprintf(out,"\n");
  }
  fclose(out);
  {
    FILE* out=fopen("results/probasopt.gnu","w");
    for(int t=0;t<m_npts;t++){
      if(m_filtre[t]){
        VectorXd Thetacourant=m_Grid.GetGrid()[t];
        fprintf(out,"%e %e %e %e\n",Thetacourant(0),Thetacourant(1),Thetacourant(2),m_values[t]);
      }
    }
    fclose(out);
  }
}

void DensityKOH::WriteMeanPreds(const char* file_name) const{
  //Ecriture de toutes les prédictions moyennes
  FILE* out=fopen(file_name,"w");
  vector<VectorXd> Preds;
  for(int t=0;t<m_npts;t++){
    if(m_filtre[t]){
      VectorXd Pred=FPredCondTheta(t)+ZMeanCondTheta(t);
      Preds.push_back(Pred);
    }
  }

  for (int i=0;i<m_vectorx.size();i++){
    fprintf(out,"%e ",m_vectorx[i](0));
    for(int j=0;j<Preds.size();j++){
      fprintf(out,"%e ",Preds[j](i));
    }
    fprintf(out,"\n");
  }
  fclose(out);
    {
    FILE* out=fopen("results/probaskoh.gnu","w");
    for(int t=0;t<m_npts;t++){
      if(m_filtre[t]){
        VectorXd Thetacourant=m_Grid.GetGrid()[t];
        fprintf(out,"%e %e %e %e\n",Thetacourant(0),Thetacourant(1),Thetacourant(2),m_values[t]);
      }
    }
    fclose(out);
  }
}

void DensityOpt::WritePredictions(const char* file_name) const{
  //Ecriture de la prédiction moyenne et de l'incertitude associée, obtenue par méthode KOH. On fait 3 écart-type.
  VectorXd Mean=EspF()+EspZ();//prédiction moyenne f+z
  MatrixXd EDM=EspVarZ(); //variance de prédiction de z : E[c* -c*K-1c*] : "erreur de prédiction due à l'edm"
  MatrixXd Varf=VarFZ(); //variance de f_theta+z_theta : "erreur de prédiction due à l'incertitude paramétrique"
  //on écrit dans le fichier directement les écarts
  VectorXd Contribf(Mean.size()),ContribEDM(Mean.size());
  for (int i=0;i<Mean.size();i++)
  {
    Contribf(i)=2*Varf.diagonal()(i)/(sqrt(Varf.diagonal()(i)+EDM.diagonal()(i)));
    ContribEDM(i)=2*EDM.diagonal()(i)/(sqrt(Varf.diagonal()(i)+EDM.diagonal()(i)));
    //Contribf(i)=3*sqrt(Varf.diagonal()(i)); //termes absolus
    //ContribEDM(i)=3*sqrt(EDM.diagonal()(i));
  }
  FILE* out=fopen(file_name,"w");
  fprintf(out, "#X, Pred Mean, stdf, stdedm\n");
  for (int i=0;i<Mean.size();i++)
  {
    fprintf(out, "%e %e %e %e\n",m_vectorx[i](0),Mean(i),Contribf(i),ContribEDM(i));
  }
  fclose(out);
}

void DensityOpt::WritePredictionsFZ(const char* file_name) const{
  //Ecriture de la prédiction moyenne et de l'incertitude associée, obtenue par méthode KOH. On fait 3 écart-type.
  VectorXd MeanF=EspF();
  VectorXd MeanZ=EspZ();
  MatrixXd Varf=VarF(); //variance de f_theta+z_theta : "erreur de prédiction due à l'incertitude paramétrique"
  //on écrit dans le fichier directement les prédictions (et non les écarts)
  VectorXd Contribf(MeanF.size());
  for (int i=0;i<MeanF.size();i++)
  {
    Contribf(i)=2*sqrt(Varf.diagonal()(i));
  }
  FILE* out=fopen(file_name,"w");
  fprintf(out, "#X, Pred Mean f, stdf, Pred Mean z\n");
  for (int i=0;i<MeanF.size();i++)
  {
    fprintf(out, "%e %e %e %e\n",m_vectorx[i](0),MeanF(i),Contribf(i),MeanZ(i));
  }
  fclose(out);
}

double DensityOpt::optfunc(const std::vector<double> &x, std::vector<double> &grad, void *data){
  AugmentedDensityOpt* AD = (AugmentedDensityOpt*) data; //récupération de l'argument par pointeur
  DensityOpt *D=AD->D;
  std::vector<DATA> newobs=*(AD->newobs);
  Eigen::VectorXd hpars(D->m_dim_hpars);
  for(int p=0; p<D->m_dim_hpars; p++) {hpars(p) =x[p];} // copie des hyperparamètres dans un VectorXd
  //maintenant qu'on connaît les hyperparamètres, il faut soustraire la moyenne a priori aux données.
  for(int i=0;i<newobs.size();i++){
    newobs[i].SetValue(newobs[i].Value()-D->m_priormean(D->GetObs()[i].GetX(),hpars));
  }
  double d=D->loglikelihood(&newobs,hpars)+D->LogPriorHpars(hpars);
  return d;
}

void DensityOpt::Build()
{
  cout << "Construction de la densité Opti..." << endl;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distU(0,1);
  double time_exec_opti;
  int maxeval=5000;
  double ftol=1e-4;
  int nd=this->GetObs().size(); //nombre d'observations
  clock_t c_start = std::clock();
  //construction de la densité opt. On fait une boucle sur les theta.
  for (int i=0;i<m_npts;i++){
       	//Creation des data
	  std::vector<DATA> data2;
	  for(int ie=0; ie<nd; ie++){
	    DATA dat; 
      dat.SetX(GetObs()[ie].GetX());
      dat.SetValue((GetObs()[ie].Value())-(m_model_evals)[i](ie)); //observations moins évaluations du modèle
      data2.push_back(dat); // on construit les données y-f_t      
    }
    //on construit la structure AugmentedDensityOpt
    AugmentedDensityOpt AD(this,&data2);
    AugmentedDensityOpt* ADpt=&AD;
    VectorXd hpars_max_opti(m_dim_hpars);
    {
	  int pop=2000; // population size
	  //	cout << "Début de l'optimisation globale... Pop size : " << pop << endl;
	  std::vector<double> x(m_dim_hpars);
    
    if(i>0){
      for (int j=0;j<m_dim_hpars;j++){
        x[j]=m_hpars_opti[i-1](j); //warm_restart
      }
    }
    else{
      x=RandHpars(generator);
    } 
    // initial guess}
    nlopt::opt opt(nlopt::GN_ISRES, m_dim_hpars);    /* algorithm and dimensionality */
	  opt.set_max_objective(DensityOpt::optfunc, ADpt);
	  opt.set_lower_bounds(m_lb_hpars);
	  opt.set_upper_bounds(m_ub_hpars);
	  opt.set_maxeval(maxeval);
	  opt.set_population(pop);
	  opt.set_ftol_rel(ftol);		
	  double msup; /* the maximum objective value, upon return */
	  int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 
	  for(int i=0;i<m_dim_hpars;i++){hpars_max_opti(i)=x[i];}
	  //cout << "Message d'arrêt : " << fin <<". max trouvé : " << hpars_max_opti.transpose() << ", valeur du critère : " << msup <<endl;
	}
	//Etape 2 : opti locale
	{
	  //	cout << "Début de l'optimisation locale..." << endl;
	  double msuploc;
	  /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
	  std::vector<double> x(m_dim_hpars);
	  for (int j=0;j<m_dim_hpars;j++){x[j]=hpars_max_opti(j);}
	  nlopt::opt opt(nlopt::LN_SBPLX, m_dim_hpars);    /* algorithm and dimensionality */
	  opt.set_max_objective(DensityOpt::optfunc, ADpt);
	  opt.set_lower_bounds(m_lb_hpars);
	  opt.set_upper_bounds(m_ub_hpars);
	  opt.set_maxeval(maxeval);
	  opt.set_ftol_rel(ftol);
	  double msup; /* the maximum objective value, upon return */
	  int fin=opt.optimize(x, msup);
	  for(int i=0;i<m_dim_hpars;i++){hpars_max_opti(i)=x[i];}
	  msuploc=msup;
	  //cout << "Message d'arrêt : " << fin  << ". Max : " << hpars_max_opti.transpose() << ", valeur du critère : " << msuploc << endl;
	  int niter_optimisations=500; // On recommence l'opti tant qu'on n'a pas stagné 100 fois.
	  int nopt=0;
	  while (nopt<niter_optimisations){
	    nopt++;
	    x=RandHpars(generator);
	    int fin= opt.optimize(x, msup);
	    if(msup>msuploc){
	      nopt=0;
	      msuploc=msup;
	      for(int k=0;k<m_dim_hpars;k++){
		hpars_max_opti(k)=x[k];
	      }
	       //cout << "Message d'arrêt : " << fin  << ". Nouveau max : " << hpars_max_opti.transpose() << ", valeur du critère : " << msuploc << endl;
	    }
  	  }
	  double max_vrais_opti=msuploc;
	  //	cout << "hyperparametres KOH : (edm, exp, lcor) : " << hpars_max_koh.transpose() << " a la vraisemblance : " << max_vrais_koh << endl << endl;
  	  m_hpars_opti.push_back(hpars_max_opti);
     m_values.push_back(msuploc+m_logpriorpars((*this).GetDoE().GetGrid()[i])); //on renvoie la logpostérieure
  	}
	  data2.clear();
  }
  clock_t c_end = std::clock();
  time_exec_opti = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  std::cout << "Temps pour opti Opti : " << time_exec_opti / 1000.0 << " s\n";
  //on récupère la constante de normalisation. C'est logvrais+prior pars +prior hpars !
  m_norm_cst=*max_element(m_values.begin(),m_values.end());
  cout << " max opti :" << m_norm_cst << " et quelques valeurs : " << m_values[5] << ", " << m_values[100] << endl;
  vectorexp();
  vectorweight1();
  ComputeFiltre(100);
  FiltreDensite();
  vectorweight1();
}

VectorXd DensityOpt::FPredCondTheta(int t) const{
  //renvoie f(xpred,theta)
  VectorXd Thetacourant=m_Grid.GetGrid()[t]; //valeur de theta utilisée dans la fonction
  VectorXd Hparscourant=m_hpars_opti[t]; //valeur des hpars
  VectorXd Fpred(m_vectorx.size());
  for(int i=0;i<Fpred.size();i++)
  {
    Fpred(i)=(m_model_evals_predictions)[i](t);
  }
  return Fpred;
}

VectorXd DensityOpt::ZMeanCondTheta(int t) const{
  //renvoie postmean de z given theta
  //revient à postmean de z pour KOH
  VectorXd Thetacourant=m_Grid.GetGrid()[t]; //valeur de theta utilisée dans la fonction
  VectorXd Hparscourant=m_hpars_opti[t]; //valeur des hpars
  VectorXd Zmean(m_vectorx.size());
  VectorXd y(m_obs.size());
  for (int i=0;i<y.size();i++){
    y(i)=m_obs[i].Value()-((m_model_evals)[t](i)+m_priormean(m_obs[i].GetX(),Hparscourant));
  }
  MatrixXd Kstar(m_vectorx.size(),m_obs.size());
  for (int i=0;i<m_vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar(i,j)=m_Kernel(m_vectorx[i],m_obs[j].GetX(),Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  VectorXd predmean=Kstar*ldlt.solve(y);
  for(int i=0;i<Zmean.size();i++)
  {
    Zmean(i)=predmean(i)+m_priormean(m_vectorx[i],Hparscourant);
  }
  return Zmean;
}

MatrixXd DensityOpt::VarZCondTheta(int t) const{
  //renvoie la variance de z a posteriori given theta
  VectorXd Hparscourant=m_hpars_opti[t]; //valeur des hpars
  MatrixXd Kstar1(m_vectorx.size(),m_obs.size());
  MatrixXd Kstar2(m_obs.size(),m_vectorx.size());
  for (int i=0;i<m_vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar1(i,j)=m_Kernel(m_vectorx[i],m_obs[j].GetX(),Hparscourant);
      Kstar2(j,i)=m_Kernel(m_obs[j].GetX(),m_vectorx[i],Hparscourant);
    }
  }
  MatrixXd Kprior(m_vectorx.size(),m_vectorx.size());
  for (int i=0;i<m_vectorx.size();i++){
    for (int j=0;j<m_vectorx.size();j++){
      Kprior(i,j)=m_Kernel(m_vectorx[i],m_vectorx[j],Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  MatrixXd varred=Kstar1*ldlt.solve(Kstar2);
  return Kprior-varred;
}

VectorXd DensityOpt::EspF() const{
  //renvoie E_{theta}[f]
  VectorXd pred=VectorXd::Zero(m_vectorx.size());
  for(int t=0;t<m_npts;t++)
  {
    if(m_filtre[t]){
      VectorXd pred_theta=FPredCondTheta(t);
      pred+=pred_theta*m_Grid.GetWeights()[t]*m_values[t];
    }
  }
  return pred;
}

VectorXd DensityOpt::EspZ() const{
  //renvoie E_{theta}[z]
  VectorXd pred=VectorXd::Zero(m_vectorx.size());
  for(int t=0;t<m_npts;t++)
  {
    if(m_filtre[t]){
      VectorXd pred_theta=ZMeanCondTheta(t);
      pred+=pred_theta*m_Grid.GetWeights()[t]*m_values[t];
    }
  }
  return pred;
}

MatrixXd DensityOpt::EspVarZ() const{
  //renvoie E_{theta}[Var z]
  MatrixXd pred=MatrixXd::Zero(m_vectorx.size(),m_vectorx.size());
  for(int t=0;t<m_npts;t++)
  {
    if(m_filtre[t]){
      MatrixXd pred_theta=VarZCondTheta(t);
      pred+=pred_theta*m_Grid.GetWeights()[t]*m_values[t];
    }
  }
  return pred;
}

MatrixXd DensityOpt::VarF() const{
  //renvoie Var_{theta}[f]
  //renvoie une matrice carrée (m_vectorx.size())
  int xsize=m_vectorx.size();
  MatrixXd COV=MatrixXd::Zero(xsize,xsize);
  //on récupère le vecteur d'évaluations du modèle.
  //un premier calcul pour savoir combien de thetas ont une probabilité non nulle
  int tsize(0); for (int i=0;i<m_npts;i++){if(m_filtre[i]){++tsize;}}
  //on fait les évaluations du modèle
  int c(0);
  VectorXd weights(tsize); for (int i=0;i<m_npts;i++){if(m_filtre[i]){weights(c)=m_values[i]*m_Grid.GetWeights()[i];++c;}}
  MatrixXd model_evals(xsize,tsize);
  c=0;
  for (int i=0;i<m_npts;i++){
    if(m_filtre[i]){
      model_evals.col(c)=FPredCondTheta(i);
      ++c;
    }
  }
  for (int i=0;i<xsize;i++){
    double var(0);
    double esp(0);
    for (int j=0;j<tsize;j++){
      esp+=weights(j)*model_evals(i,j);
      var+=weights(j)*pow(model_evals(i,j),2);
    }
  COV(i,i)=var-pow(esp,2);
  }
  return COV;
}

MatrixXd DensityOpt::VarFZ() const{
  //renvoie Var_{theta}[f+z]
  //renvoie une matrice carrée (m_vectorx.size())
  //renvoie une matrice diagonale 
  int xsize=m_vectorx.size();
  MatrixXd COV=MatrixXd::Zero(xsize,xsize);
  //on récupère le vecteur d'évaluations du modèle.
  //un premier calcul pour savoir combien de thetas ont une probabilité non nulle
  int tsize(0); for (int i=0;i<m_npts;i++){if(m_filtre[i]){++tsize;}}
  //on fait les évaluations du modèle
  int c(0);
  VectorXd weights(tsize); for (int i=0;i<m_npts;i++){if(m_filtre[i]){weights(c)=m_values[i]*m_Grid.GetWeights()[i];++c;}}
  MatrixXd model_evals(xsize,tsize);
  c=0;
  for (int i=0;i<m_npts;i++){
    if(m_filtre[i]){
      model_evals.col(c)=FPredCondTheta(i)+ZMeanCondTheta(i);
      ++c;
    }
  }
  for (int i=0;i<xsize;i++){
    double var(0);
    double esp(0);
    for (int j=0;j<tsize;j++){
      esp+=weights(j)*model_evals(i,j);
      var+=weights(j)*pow(model_evals(i,j),2);
    }
    COV(i,i)=var-pow(esp,2);
  }
  return COV;
}



void DensityOpt::WriteHpars(const char* file_name) const
{
  int dim_hpars=m_lb_hpars.size();
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<m_npts;i++){
    for (int j=0;j<m_Grid.GetDimension();j++){
      fprintf(out,"%e ",m_Grid.GetGrid()[i](j));
    }
    for (int j=0;j<dim_hpars;j++){
      fprintf(out,"%e ",m_hpars_opti[i](j));
    }
    fprintf(out,"\n");
  }
  fclose(out);
};

VectorXd DensityOpt::DrawSample(default_random_engine &generator) const{
  std::uniform_real_distribution<double> distU(0,1);
  //choix aléatoire d'un theta, puis tirage de la loi normale
  double u=distU(generator);
  double sum(0);
  int indcour(0);
  for (int i=0;i<m_npts;i++){
    sum+=m_Grid.GetWeights()[i]*m_values[i];
    if(sum>u){
      indcour=i;
      break;
    }
  }
  VectorXd Predmean=FPredCondTheta(indcour)+ZMeanCondTheta(indcour);
  MatrixXd Cov(m_vectorx.size(),m_vectorx.size());
  Cov=VarZCondTheta(indcour);
  return DrawMVN(Predmean,Cov,generator);
}

map<VectorXd,VectorXd,map_compare> DensityOpt::Hpars1D(int h1) const
{
  //estimation de la densité à posteriori des hyperparamètres optimaux selon une map pour pouvoir faire des courbes.
  if (h1<0 || h1>= m_dim_pars){cout << "erreur de dimension !" << endl;}
  map<VectorXd,VectorXd,map_compare> hpars;
  map<VectorXd,double,map_compare> weights;
  for (int i=0;i<m_npts;i++)
  {
    VectorXd thetacourant=m_Grid.GetGrid()[i];
    VectorXd thetacourantreduit(1);
    thetacourantreduit(0)=thetacourant(h1); //on regarde selon la projection sur l'indice h1.
    map<VectorXd,VectorXd,map_compare>::iterator it=hpars.find(thetacourantreduit);
    if (it==hpars.end()){hpars[thetacourantreduit]=VectorXd::Zero(m_dim_hpars);}
    //apparemment les valeurs de la map sont intialisées à 0 la première fois qu'on les appelle. A vérifier si ça plante.
    //cout << hpars[thetacourantreduit].transpose()<< endl;
    //cout << (m_hpars_opti[i]*m_values[i]*m_Grid.GetWeights()[i]).transpose() << endl;
    hpars[thetacourantreduit]+=m_hpars_opti[i]*m_values[i]*m_Grid.GetWeights()[i];
    weights[thetacourantreduit]+=m_values[i]*m_Grid.GetWeights()[i];
  }
  for (const auto &Pair : hpars)
  {
    if (!weights[Pair.first]==0){hpars[Pair.first]/=weights[Pair.first];}
  }
  return hpars;
}

map<VectorXd,VectorXd,map_compare> DensityOpt::Hpars2D(int h1, int h2) const
{
  //estimation de la densité à posteriori des hyperparamètres optimaux selon une map pour pouvoir faire des courbes.
  if (h1<0 || h1>= m_dim_pars){cout << "erreur de dimension !" << endl;}
  if (h2<0 || h2>= m_dim_pars){cout << "erreur de dimension !" << endl;}
   map<VectorXd,VectorXd,map_compare> hpars;
  map<VectorXd,double,map_compare> weights;
  for (int i=0;i<m_npts;i++)
  {
    VectorXd thetacourant=m_Grid.GetGrid()[i];
    VectorXd thetacourantreduit(2);
    thetacourantreduit(0)=thetacourant(h1); //on regarde selon la projection sur l'indice h1.
    thetacourantreduit(1)=thetacourant(h2); //on regarde selon la projection sur l'indice h1.
    map<VectorXd,VectorXd,map_compare>::iterator it=hpars.find(thetacourantreduit);
    if (it==hpars.end()){hpars[thetacourantreduit]=VectorXd::Zero(m_dim_hpars);}
    //apparemment les valeurs de la map sont intialisées à 0 la première fois qu'on les appelle. A vérifier si ça plante.
    //cout << hpars[thetacourantreduit].transpose()<< endl;
    //cout << (m_hpars_opti[i]*m_values[i]*m_Grid.GetWeights()[i]).transpose() << endl;
    hpars[thetacourantreduit]+=m_hpars_opti[i]*m_values[i]*m_Grid.GetWeights()[i];
    weights[thetacourantreduit]+=m_values[i]*m_Grid.GetWeights()[i];
  }
  for (const auto &Pair : hpars)
  {
    if (!weights[Pair.first]==0){hpars[Pair.first]/=weights[Pair.first];}
  }
  return hpars;
}

void DensityOpt::WritePostHpars(const char* folder_name) const
{
  std::string foldname(folder_name);
  map<VectorXd,VectorXd,map_compare> map1=Hpars1D(0);
  WriteMapToFile(map1,foldname+"1.gnu");
  map<VectorXd,VectorXd,map_compare> map2=Hpars1D(1);
  WriteMapToFile(map2,foldname+"2.gnu");
  map<VectorXd,VectorXd,map_compare> map3=Hpars1D(2);
  WriteMapToFile(map3,foldname+"3.gnu");
  map<VectorXd,VectorXd,map_compare> map12=Hpars2D(1,0);
  WriteMapToFile(map12,foldname+"21.gnu");
  map<VectorXd,VectorXd,map_compare> map23=Hpars2D(2,1);
  WriteMapToFile(map23,foldname+"32.gnu");
  map<VectorXd,VectorXd,map_compare> map31=Hpars2D(2,0);
  WriteMapToFile(map31,foldname+"31.gnu");
};

DensityBayes::DensityBayes(Density &d) : Density(d){

}

void DensityBayes::Build()
{
  cout << "Début du calcul Bayes..." << endl;
  //construction de la densité.
  int seed_bayes=666;
  default_random_engine generator(seed_bayes);
  int nsim_bayes=500000; // nombre de tirages par valeur de paramètres
  VectorXd probs=VectorXd::Zero(m_npts);
  clock_t c_start = std::clock();
  //Tirage des hyperparamètres et évaluation des décompositions de Cholesky.
  for (int i=0;i<nsim_bayes;i++){
	  VectorXd hpars=m_sample_hpars(generator);
	  MatrixXd G=Gamma(&m_obs,hpars);
	  LDLT<MatrixXd> ldlt(G);
	  for (int j=0;j<m_npts;j++){
	    probs(j)+=exp(loglikelihood_theta_fast(j,hpars,ldlt));
	  }
  }
  for (int j=0;j<m_npts;j++){
	    probs(j)+=exp(m_logpriorpars(m_Grid.GetGrid()[j]));
      m_values.push_back(probs(j));
	}
  vectorweight1();
  clock_t c_end = std::clock();
  double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  std::cout << "Temps pour calcul Bayes : " << time_elapsed_ms / 1000.0 << " s\n";
}

DensityCV::DensityCV(Density &d) : DensityKOH(d,0)
{

};

VectorXd DensityCV::HparsCV() {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distU(0,1);
  /*paramètres de l'optimisation*/
  VectorXd hpars_max_cv(m_dim_hpars);
  double time_exec_cv;
  int maxeval=5000;
  double ftol=1e-3;
  cout << "Début de l'optimisation CV..." << endl;
  /*Pointer to member*/
  {
    clock_t c_start = std::clock();
    //1ère étape : optimisation globale par algo génétique
    {
      int pop=2000; // population size
      //cout << "Début de l'optimisation globale... Pop size : " << pop << endl;
      /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
      std::vector<double> x(m_dim_hpars);
   
      cout << m_dim_hpars << endl;
      x=RandHpars(generator); // initial guess
      nlopt::opt opt(nlopt::GN_ISRES, m_dim_hpars);    /* algorithm and dimensionality */

      opt.set_max_objective(DensityCV::optfunc, this);

      opt.set_lower_bounds(m_lb_hpars);
      opt.set_upper_bounds(m_ub_hpars);
      opt.set_maxeval(maxeval);
      opt.set_population(pop);
      opt.set_ftol_rel(ftol);
      double msup; /* the maximum objective value, upon return */

      int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 

      for(int i=0;i<m_dim_hpars;i++){hpars_max_cv(i)=x[i];}
      //cout << "Message d'arrêt : " << fin <<". max trouvé : " << hpars_max_koh.transpose() << ", valeur du critère : " << msup <<endl;
    }
    //2ème étape : optimisation locale par sbplx en partant du point trouvé précédemment.
    {
      //cout << "Début de l'optimisation locale..." << endl;
      double msuploc;
      /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
      std::vector<double> x(m_dim_hpars);
      for (int j=0;j<m_dim_hpars;j++){x[j]=hpars_max_cv(j);}
      nlopt::opt opt(nlopt::LN_SBPLX, m_dim_hpars);    /* algorithm and dimensionality */
      opt.set_max_objective(DensityCV::optfunc, this);
      opt.set_lower_bounds(m_lb_hpars);
      opt.set_upper_bounds(m_ub_hpars);
      opt.set_maxeval(maxeval);
      opt.set_ftol_rel(ftol);
      double msup; /* the maximum objective value, upon return */
      int fin=opt.optimize(x, msup);
      for(int i=0;i<m_dim_hpars;i++){hpars_max_cv(i)=x[i];}
      msuploc=msup;
      //cout << "Message d'arrêt : " << fin  << ". Max : " << hpars_max_koh.transpose() << ", valeur du critère : " << msuploc << endl;
      int niter=100; // On recommence l'opti tant qu'on n'a pas stagné 100 fois.
      int nopt=0;
      while (nopt<niter){
	nopt++;
	x=RandHpars(generator);
	int fin= opt.optimize(x, msup);
	if(msup>msuploc){
	  nopt=0;
	  msuploc=msup;
	  for(int k=0;k<m_dim_hpars;k++){
	    hpars_max_cv(k)=x[k];
	  }
	  //cout << "Message d'arrêt : " << fin  << ". Nouveau max : " << hpars_max_koh.transpose() << ", valeur du critère : " << msuploc << endl;
	}
      }
      double max_vrais_cv=msuploc;
      cout << "hyperparametres LOOCV : (edm, exp, lcor) : " << hpars_max_cv.transpose() << " au score : " << max_vrais_cv << endl;
    }
    clock_t c_end = std::clock();
    double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    time_exec_cv=time_elapsed_ms / 1000.0;
    std::cout << "Temps pour opti LOOCV : " << time_exec_cv << " s\n";
  }
  m_hpars=hpars_max_cv;
  return hpars_max_cv;
}

double DensityCV::optfunc(const std::vector<double> &x, std::vector<double> &grad, void *data){
  DensityCV* d = (DensityCV*) data; //pointer to densitykoh
  vector<DATA>* obs;
  vector<DATA> observations=(d->GetObs());
  obs=&observations;
  VectorXd obsvector(observations.size());
  for (int i=0;i<obsvector.size();i++){obsvector(i)=observations[i].Value();}
  /*Fonction à optimiser, Kennedy O'Hagan. On cherche à estimer l'intégrale moyennée sur un grid uniforme (avec priors uniformes) */
  Eigen::VectorXd hpar(d->m_dim_hpars);
  int npts=d->GetNpts();
  for(int p=0; p<d->m_dim_hpars; p++) {hpar(p) =x[p];} // copie des hyperparamètres dans un VectorXd
  MatrixXd G=d->Gamma(obs,hpar); //les valeurs de data2 ne sont pas correctes car non retranchées de ftheta. Cependant on n'utilise que les X pour calculer G.
  LDLT<MatrixXd> ldlt(G);
  VectorXd Alpha = ldlt.solve(obsvector);
  MatrixXd Kinv = ldlt.solve(MatrixXd::Identity(observations.size(),observations.size())); 
  double score =0;
  for (int i=0;i<observations.size();i++){
    score+=0.5*log(Kinv(i,i))-0.5*pow(Alpha(i),2)/Kinv(i,i);
  }
  return score+d->m_logpriorhpars(hpar); //avec priors
}

void DensityCV::Build()
{
  VectorXd hparscv=HparsCV();
  Density::Build(hparscv);
  vectorexp();
  vectorweight1();
  ComputeFiltre(100);
  FiltreDensite();
  vectorweight1();
}

DensitySimple::DensitySimple(Density &d, double c) : DensityKOH(d,c)
{
  m_Kernel=DensitySimple::KernelNull;
};

void DensitySimple::Build()
{
  VectorXd hparssimple=HparsSimple();
  Density::Build(hparssimple);
  vectorexp();
  vectorweight1();
  ComputeFiltre(100);
  FiltreDensite();
  vectorweight1();
}

VectorXd DensitySimple::HparsSimple() {
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distU(0,1);
  /*paramètres de l'optimisation*/
  VectorXd hpars_max_koh(m_dim_hpars);
  double time_exec_koh;
  int maxeval=5000;
  double ftol=1e-3;
  cout << "Début de l'optimisation densité sans edm..." << endl;
  /*Pointer to member*/
  {
    clock_t c_start = std::clock();
    //1ère étape : optimisation globale par algo génétique
    {
      int pop=2000; // population size
      //cout << "Début de l'optimisation globale... Pop size : " << pop << endl;
      /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
      std::vector<double> x(m_dim_hpars); //m_dim_hpars devrait valoir 1
      x=RandHpars(generator);
      nlopt::opt opt(nlopt::GN_ISRES, m_dim_hpars);    /* algorithm and dimensionality */

      opt.set_max_objective(DensityKOH::optfunc, this);

      opt.set_lower_bounds(m_lb_hpars);
      opt.set_upper_bounds(m_ub_hpars);
      opt.set_maxeval(maxeval);
      opt.set_population(pop);
      opt.set_ftol_rel(ftol);
      double msup; /* the maximum objective value, upon return */

      int fin=opt.optimize(x, msup); //messages d'arrêt :3=ftol_reached, 4=xtol reached, 5=maxeval_reached; 

      for(int i=0;i<m_dim_hpars;i++){hpars_max_koh(i)=x[i];}
      cout << "Message d'arrêt : " << fin <<". max trouvé : " << hpars_max_koh.transpose() << ", valeur du critère : " << msup <<endl;
    }
    //2ème étape : optimisation locale par sbplx en partant du point trouvé précédemment.
    {
      //cout << "Début de l'optimisation locale..." << endl;
      double msuploc;
      /*Optimisation sur les hyperparamètres avec la fonction KOH.*/
      std::vector<double> x(m_dim_hpars);
      for (int j=0;j<m_dim_hpars;j++){x[j]=hpars_max_koh(j);}
      nlopt::opt opt(nlopt::LN_SBPLX, m_dim_hpars);    /* algorithm and dimensionality */
      opt.set_max_objective(DensityKOH::optfunc, this);
      opt.set_lower_bounds(m_lb_hpars);
      opt.set_upper_bounds(m_ub_hpars);
      opt.set_maxeval(maxeval);
      opt.set_ftol_rel(ftol);
      double msup; /* the maximum objective value, upon return */
      int fin=opt.optimize(x, msup);
      for(int i=0;i<m_dim_hpars;i++){hpars_max_koh(i)=x[i];}
      msuploc=msup;
      cout << "Message d'arrêt : " << fin  << ". Max : " << hpars_max_koh.transpose() << ", valeur du critère : " << msuploc << endl;
      int niter_optimisations=500; // On recommence l'opti tant qu'on n'a pas stagné 100 fois.
      int nopt=0;
      while (nopt<niter_optimisations){
	nopt++;
	x=RandHpars(generator);
	int fin= opt.optimize(x, msup);
	if(msup>msuploc){
	  nopt=0;
	  msuploc=msup;
	  for(int k=0;k<m_dim_hpars;k++){
	    hpars_max_koh(k)=x[k];
	  }
	  //cout << "Message d'arrêt : " << fin  << ". Nouveau max : " << hpars_max_koh.transpose() << ", valeur du critère : " << msuploc << endl;
	}
      }
      double max_vrais_koh=msuploc;
      cout << "hyperparametres sans edm: (edm) : " << hpars_max_koh(1) << " a la vraisemblance : " << max_vrais_koh << endl;
    }
    clock_t c_end = std::clock();
    double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
    time_exec_koh=time_elapsed_ms / 1000.0;
    std::cout << "Temps pour opti sans edm : " << time_exec_koh << " s\n";
  }
  m_hpars=hpars_max_koh;
  return hpars_max_koh;
}

MCMC::MCMC(Density &d, int nchain) : Density(d)
{
  m_nchain=nchain;
  m_dim_mcmc=m_dim_hpars+m_dim_pars;
  m_naccept=0;
  m_oob=VectorXd::Zero(m_dim_mcmc);
}

double MCMC::loglikelihood_theta(void *data, Eigen::VectorXd const &hpar, VectorXd const &theta)const{
  //besoin de réécrire cette fonction pour être plus générale que celle de Density. On veut pouvoir évaluer le modèle à un theta et X quelconque.
  //renvoie la log-vraisemblance des hyperparamètres hpar, étant donné les données data et les paramètres du modèle theta.
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  std::vector<DATA> data3;
  for(unsigned ie=0; ie<nd; ie++){
    DATA dat; dat.SetX((*data2)[ie].GetX()); dat.SetValue((*data2)[ie].Value()-(m_my_model((*data2)[ie].GetX(),theta)+m_priormean((*data2)[ie].GetX(),hpar)));
    data3.push_back(dat);
  }
  return loglikelihood(&data3,hpar);
}

bool MCMC::in_bounds(Eigen::VectorXd &X)
{
  for (int i=0;i<m_Grid.GetDimension();i++){
    if (X(i)<m_Grid.GetParsLb()(i) || X(i)>m_Grid.GetParsUb()(i)){++m_oob(i); return false;}
  }
  for (int i=0;i<m_dim_hpars;i++){
    if (X(i+m_Grid.GetDimension())<m_lb_hpars[i] || X(i+m_Grid.GetDimension())>m_ub_hpars[i]) {++m_oob(i+m_Grid.GetDimension()); return false;}
  }
  return true;
}

void MCMC::Run(Eigen::VectorXd &Xinit, Eigen::MatrixXd &COV_init,default_random_engine &generator)
{
  std::normal_distribution<double> distN(0,1);
  std::uniform_real_distribution<double> distU(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  cout << "Running MCMC with " << m_nchain << " steps..." << endl;
  if(!COV_init.cols()==m_dim_mcmc){cout <<"erreur de dimension MCMC" << endl;}
  VectorXd Xcurrent=Xinit;
  VectorXd Xcandidate(m_dim_mcmc);
  double fcurrent=loglikelihood_theta(&m_obs,Xcurrent.tail(m_dim_hpars),Xcurrent.head(m_dim_pars))+m_logpriorhpars(Xcurrent.tail(m_dim_hpars))+m_logpriorpars(Xcurrent.head(m_dim_pars));
  double fcandidate(0);
  clock_t c_start = std::clock();
  for (int i=0;i<m_nchain;i++){
    VectorXd Step(m_dim_mcmc);
    for (int j=0;j<Step.size();j++){Step[j]=distN(generator);}
    Xcandidate=Xcurrent+sqrtCOV*Step;
    fcandidate=loglikelihood_theta(&m_obs,Xcandidate.tail(m_dim_hpars),Xcandidate.head(m_dim_pars))+m_logpriorhpars(Xcandidate.tail(m_dim_hpars))+m_logpriorpars(Xcandidate.head(m_dim_pars));
    if(in_bounds(Xcandidate)){
      if(fcandidate>fcurrent | fcandidate-fcurrent>log(distU(generator))){
        m_naccept++;
        Xcurrent=Xcandidate;
        fcurrent=fcandidate;
      }
    }
    m_all_samples.push_back(Xcurrent);
    m_all_values.push_back(fcurrent);
    //ici ajouter les valeurs de la chaîne si l'on souhaite les conserver
  }
  clock_t c_end = std::clock();
  double time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC;
  double time_mcmc=time_elapsed_ms / 1000.0;
  std::cout << "Temps pour MCMC : " << time_mcmc << " s\n";
  cout << "accept rate : " << 100*double(m_naccept)/double(m_nchain) << endl;
}

void MCMC::SelectSamples(int nsamples)
{
  //renvoie nsamples pris uniformément de la MCMC.
    for (int i=0;i<m_all_samples.size();i++){
    if (i>nsamples && i%(m_all_samples.size()/nsamples)==0){
      m_selected_samples.push_back(m_all_samples[i]);
      m_selected_values.push_back(m_all_values[i]);
    }
  }
}

VectorXd MCMC::MAP() const
{
  int indmax(0);
  double currentmax(m_selected_values[0]);
  for (int i=0;i<m_selected_samples.size();i++)
  {
    if(m_selected_values[i]>currentmax)
    {
      indmax=i;
      currentmax=m_selected_values[i];
    }
  }
  return m_selected_samples[indmax];
}

VectorXd MCMC::Mean() const
{
  VectorXd mean=VectorXd::Zero(m_dim_mcmc);
  for (int i=0;i<m_selected_samples.size();i++)
  {
    mean+=m_selected_samples[i];
  }
  mean/=double(m_selected_samples.size());
  return mean;
}

MatrixXd MCMC::Cov() const
{
  VectorXd mean=Mean();
  MatrixXd cov=MatrixXd::Zero(m_dim_mcmc,m_dim_mcmc);
  for (int i=0;i<m_selected_samples.size();i++)
  {
    VectorXd X=m_selected_samples[i]-mean;
    cov+=X*X.transpose();
  }
  cov/=double(m_selected_samples.size()-1);
  return cov;
}

VectorXd MCMC::FPredCondX(int t) const{
  //renvoie f(xpred,theta)
  VectorXd Thetacourant=m_selected_samples[t].head(m_dim_pars); //valeur de theta utilisée dans la fonction
  VectorXd Hparscourant=m_selected_samples[t].tail(m_dim_hpars); //valeur des hpars
  VectorXd Fpred(m_vectorx.size());
  for(int i=0;i<Fpred.size();i++)
  {
    Fpred(i)=m_my_model(m_vectorx[i],Thetacourant);
  }
  return Fpred;
}

VectorXd MCMC::ZMeanCondX(int t) const{
  //renvoie postmean de z given theta
  //revient à postmean de z pour KOH
  VectorXd Thetacourant=m_selected_samples[t].head(m_dim_pars); //valeur de theta utilisée dans la fonction
  VectorXd Hparscourant=m_selected_samples[t].tail(m_dim_hpars); //valeur des hpars
  VectorXd Zmean(m_vectorx.size());
  VectorXd y(m_obs.size());
  for (int i=0;i<y.size();i++){
    y(i)=m_obs[i].Value()-(m_my_model(m_obs[i].GetX(),Thetacourant)+m_priormean(m_obs[i].GetX(),Hparscourant));
  }
  MatrixXd Kstar(m_vectorx.size(),m_obs.size());
  for (int i=0;i<m_vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar(i,j)=m_Kernel(m_vectorx[i],m_obs[j].GetX(),Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  VectorXd predmean=Kstar*ldlt.solve(y);
  for(int i=0;i<Zmean.size();i++)
  {
    Zmean(i)=predmean(i)+m_priormean(m_vectorx[i],Hparscourant);
  }
  return Zmean;
}

MatrixXd MCMC::VarZCondX(int t) const{
  //renvoie la variance de z a posteriori given theta
  VectorXd Thetacourant=m_selected_samples[t].head(m_dim_pars); //valeur de theta utilisée dans la fonction
  VectorXd Hparscourant=m_selected_samples[t].tail(m_dim_hpars); //valeur des hpars
  MatrixXd Kstar1(m_vectorx.size(),m_obs.size());
  MatrixXd Kstar2(m_obs.size(),m_vectorx.size());
  for (int i=0;i<m_vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar1(i,j)=m_Kernel(m_vectorx[i],m_obs[j].GetX(),Hparscourant);
      Kstar2(j,i)=m_Kernel(m_obs[j].GetX(),m_vectorx[i],Hparscourant);
    }
  }
  MatrixXd Kprior(m_vectorx.size(),m_vectorx.size());
  for (int i=0;i<m_vectorx.size();i++){
    for (int j=0;j<m_vectorx.size();j++){
      Kprior(i,j)=m_Kernel(m_vectorx[i],m_vectorx[j],Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  MatrixXd varred=Kstar1*ldlt.solve(Kstar2);
  return Kprior-varred;
}

VectorXd MCMC::EspF() const{
  //renvoie E_{theta}[f]
  VectorXd pred=VectorXd::Zero(m_vectorx.size());
  for(int t=0;t<m_selected_samples.size();t++)
  {
      VectorXd pred_theta=FPredCondX(t);
      pred+=pred_theta;
  }
  pred/=(double) m_selected_samples.size();
  return pred;
}

VectorXd MCMC::EspZ() const{
  //renvoie E_{theta}[z]
  VectorXd pred=VectorXd::Zero(m_vectorx.size());
  for(int t=0;t<m_selected_samples.size();t++)
  {
    VectorXd pred_theta=ZMeanCondX(t);
    pred+=pred_theta;
  }
  pred/=(double) m_selected_samples.size();
  return pred;
}

MatrixXd MCMC::EspVarZ() const{
  //renvoie E_{theta}[Var z]
  MatrixXd pred=MatrixXd::Zero(m_vectorx.size(),m_vectorx.size());
  for(int t=0;t<m_selected_samples.size();t++)
  {
    MatrixXd pred_theta=VarZCondX(t);
    pred+=pred_theta;
  }
  pred/=(double) m_selected_samples.size();
  return pred;
}

MatrixXd MCMC::VarF() const{
  //renvoie Var_{theta}[f]
  //renvoie une matrice carrée (m_vectorx.size())
  int xsize=m_vectorx.size();
  MatrixXd COV=MatrixXd::Zero(xsize,xsize);
  //on récupère le vecteur d'évaluations du modèle.
  //un premier calcul pour savoir combien de thetas ont une probabilité non nulle
  int tsize(m_selected_samples.size());
  //on fait les évaluations du modèle
  MatrixXd model_evals(xsize,tsize);
  for (int i=0;i<tsize;i++){
    model_evals.col(i)=FPredCondX(i);
  }
  for (int i=0;i<xsize;i++){
    double var(0);
    double esp(0);
    for (int j=0;j<tsize;j++){
      esp+=model_evals(i,j);
      var+=pow(model_evals(i,j),2);
    }
    esp/=(double) tsize;
    var/=(double) tsize;
    COV(i,i)=var-pow(esp,2);
  }
  return COV;
}

MatrixXd MCMC::VarFZ() const{
  //renvoie Var_{theta}[f+z]
  //renvoie une matrice carrée (m_vectorx.size())
  //renvoie une matrice diagonale 
   int xsize=m_vectorx.size();
  MatrixXd COV=MatrixXd::Zero(xsize,xsize);
  //on récupère le vecteur d'évaluations du modèle.
  //un premier calcul pour savoir combien de thetas ont une probabilité non nulle
  int tsize(m_selected_samples.size());
  //on fait les évaluations du modèle
  MatrixXd model_evals(xsize,tsize);
  for (int i=0;i<tsize;i++){
    model_evals.col(i)=FPredCondX(i)+ZMeanCondX(i);
  }
  for (int i=0;i<xsize;i++){
    double var(0);
    double esp(0);
    for (int j=0;j<tsize;j++){
      esp+=model_evals(i,j);
      var+=pow(model_evals(i,j),2);
    }
    esp/=(double) tsize;
    var/=(double) tsize;
    COV(i,i)=var-pow(esp,2);
  }
  return COV;
}

void MCMC::WritePredictions(const char* file_name) const{
  //Ecriture de la prédiction moyenne et de l'incertitude associée, obtenue par méthode KOH. On fait 3 écart-type.
  VectorXd Mean=EspF()+EspZ();//prédiction moyenne
  MatrixXd EDM=EspVarZ(); //variance de prédiction de z : E[c* -c*K-1c*] : "erreur de prédiction due à l'edm"
  MatrixXd Varf=VarFZ(); //variance de f_theta+z_theta : "erreur de prédiction due à l'incertitude paramétrique"
  //on écrit dans le fichier directement les écarts
  VectorXd Contribf(Mean.size()),ContribEDM(Mean.size());
  for (int i=0;i<Mean.size();i++)
  {
    Contribf(i)=2*Varf.diagonal()(i)/(sqrt(Varf.diagonal()(i)+EDM.diagonal()(i)));
    ContribEDM(i)=2*EDM.diagonal()(i)/(sqrt(Varf.diagonal()(i)+EDM.diagonal()(i)));
  }
  FILE* out=fopen(file_name,"w");
  fprintf(out, "#X, Pred Mean, stdf, stdedm\n");
  for (int i=0;i<Mean.size();i++)
  {
    fprintf(out, "%e %e %e %e\n",m_vectorx[i](0),Mean(i),Contribf(i),ContribEDM(i));
  }
  fclose(out);
}

void MCMC::WritePredictionsFZ(const char* file_name) const{
  //Ecriture de la prédiction moyenne et de l'incertitude associée, obtenue par méthode KOH. On fait 3 écart-type.
  VectorXd MeanF=EspF();
  VectorXd MeanZ=EspZ();
  MatrixXd Varf=VarF(); //variance de f_theta+z_theta : "erreur de prédiction due à l'incertitude paramétrique"
  //on écrit dans le fichier directement les prédictions (et non les écarts)
  VectorXd Contribf(MeanF.size());
  for (int i=0;i<MeanF.size();i++)
  {
    Contribf(i)=2*sqrt(Varf.diagonal()(i));
  }
  FILE* out=fopen(file_name,"w");
  fprintf(out, "#X, Pred Mean f, stdf, Pred Mean z\n");
  for (int i=0;i<MeanF.size();i++)
  {
    fprintf(out, "%e %e %e %e\n",m_vectorx[i](0),MeanF(i),Contribf(i),MeanZ(i));
  }
  fclose(out);
}

void MCMC::WriteAllSamples(const char* file_name) const
{
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<m_all_samples.size();i++){
    for (int j=0;j<m_dim_mcmc;j++){
      fprintf(out,"%e ",m_all_samples[i](j));
    }
    fprintf(out,"%e\n",m_all_values[i]);
  }
  fclose(out);
};

void MCMC::WriteSelectedSamples(const char* file_name) const
{
  FILE* out=fopen(file_name,"w");
  for (int i=0;i<m_selected_samples.size();i++){
    for (int j=0;j<m_dim_mcmc;j++){
      fprintf(out,"%e ",m_selected_samples[i](j));
    }
    fprintf(out,"%e\n",m_selected_values[i]);
  }
  fclose(out);
};

void MCMC::PrintOOB() const
{
  cout << "oob :" << endl;
  cout << "t1: " << m_oob(0) << endl;
  cout << "t2: " << m_oob(1) << endl;
  cout << "t3: " << m_oob(2) << endl;
  cout << "edm: " << m_oob(3) << endl;
  cout << "exp: " << m_oob(4) << endl;
  cout << "lcor: "  << m_oob(5) << endl;
}



