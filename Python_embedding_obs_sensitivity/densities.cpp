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
};

Density::Density(DoE g) : m_Grid(g),m_npts(g.GetGrid().size()),m_dim_pars(g.GetDimension())
{


};

void Density::Build(Eigen::VectorXd const &hpars){
  //inversion de la matrice de covariance
  int nd=m_obs.size();
  MatrixXd G=Gamma(&m_obs,hpars);
  VectorXd obs(nd);
  for(unsigned i=0; i<nd; i++) obs(i) = m_obs[i].Value(); // copie des valeurs observées dans un VectorXd
  LDLT<MatrixXd> ldlt(G);
  for (int i=0;i<m_npts;i++){
    m_values.push_back(m_logpriorpars((*this).GetDoE().GetGrid()[i])+
      loglikelihood_theta_fast(&m_obs,hpars,ldlt,(*this).GetDoE().GetGrid()[i]));
  }
  this->vectorexp();
  this->vectorweight1();
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

Eigen::VectorXd Density::MAP() const
{
  return m_Grid.GetGrid()[indmax()];
};

Eigen::VectorXd Density::Mean() const
{
  VectorXd mean(m_Grid.GetDimension());
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
  for(int i=0;i<m_npts;i++){weight+=m_Grid.GetWeights()[i]*m_values[i];}
  for(int i=0;i<m_npts;i++){m_values[i]/=weight;}
};

void Density::vectorlog()
{
  for(int i=0;i<m_npts;i++){m_values[i]=log(m_values[i]);}
};

void Density::vectorexp()
{
  for(int i=0;i<m_npts;i++){m_values[i]=exp(m_values[i]);}
};

int Density::indmax() const
{
  int imax=0;
  double maxloc=m_values[imax];
  for(int i=0;i<m_npts;i++)
  {
    if(m_values[i]>maxloc){
      imax=i;
      maxloc=m_values[i];
    }
  }
  return imax;
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

double Density::loglikelihood_theta_fast(void *data, Eigen::VectorXd const &hpar, LDLT<MatrixXd> const &ldlt, VectorXd const &theta)const{
  //renvoie log p(y|pars,hpars)
  vector<DATA>* data2 = static_cast<vector<DATA>*>(data); // cast du null pointer en type désiré
  int nd=data2->size();
  VectorXd obs_theta(nd);
  for (int i=0;i<nd;i++){obs_theta(i)=(*data2)[i].Value()-(m_my_model((*data2)[i].GetX(),theta)+m_priormean((*data2)[i].GetX(),hpar));}
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

double Density::loglikelihood_theta(void *data, Eigen::VectorXd const &hpar, VectorXd const &theta)const{
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

DensityKOH::DensityKOH(Density &d) : Density(d){
}

void DensityKOH::Build()
{
  VectorXd hparskoh=HparsKOH();
  Density::Build(hparskoh);
  vectorexp();
  vectorweight1();
}

double DensityKOH::optfunc(const std::vector<double> &x, std::vector<double> &grad, void *data){
  DensityKOH* d = (DensityKOH*) data; //pointer to densitykoh
  vector<DATA>* obs;
  vector<DATA> observations=(d->GetObs());
  obs=&observations;
  /*Fonction à optimiser, Kennedy O'Hagan. On cherche à estimer l'intégrale moyennée sur un grid uniforme (avec priors uniformes) */
  Eigen::VectorXd hpar(d->m_dim_hpars);
  int npts=d->GetNpts();
  for(int p=0; p<d->m_dim_hpars; p++) {hpar(p) =x[p];} // copie des hyperparamètres dans un VectorXd
  MatrixXd G=d->Gamma(obs,hpar); //les valeurs de data2 ne sont pas correctes car non retranchées de ftheta. Cependant on n'utilise que les X pour calculer G.
  LDLT<MatrixXd> ldlt(G);
  double avg=0;
  for (int i=0;i<npts;i++){
    avg+=exp(d->loglikelihood_theta_fast(obs,hpar,ldlt,(d->GetDoE()).GetGrid()[i]));
  }
  avg=avg*exp(d->m_logpriorhpars(hpar));
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
   
      cout << m_dim_hpars << endl;
      for (int j=0;j<m_dim_hpars;j++){x[j]=exp(log(m_lb_hpars[j])+(log(m_ub_hpars[j])-log(m_lb_hpars[j]))*distU(generator));} // initial guess
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
      //cout << "Message d'arrêt : " << fin <<". max trouvé : " << hpars_max_koh.transpose() << ", valeur du critère : " << msup <<endl;
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
      //cout << "Message d'arrêt : " << fin  << ". Max : " << hpars_max_koh.transpose() << ", valeur du critère : " << msuploc << endl;
      int niter=100; // On recommence l'opti tant qu'on n'a pas stagné 100 fois.
      int nopt=0;
      while (nopt<niter){
	nopt++;
	for (int j=0;j<m_dim_hpars;j++){x[j]=exp(log(m_lb_hpars[j])+(log(m_ub_hpars[j])-log(m_lb_hpars[j]))*distU(generator));}
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

double DensityKOH::MeanPred(VectorXd &xstar) const{
  //renvoie la moyenne de prédiction du modèle en xstar. 
  //phase de calcul de la matrice de cov. et du vecteur associé.
  VectorXd Kstar(m_obs.size());
  for (int i=0;i<Kstar.size();i++){Kstar[i]=m_Kernel(xstar,m_obs[i].GetX(),m_hpars);}
  MatrixXd G=Gamma(&m_obs,m_hpars);
  LDLT<MatrixXd> ldlt(G);
  VectorXd K=(Kstar.transpose()*ldlt.solve(MatrixXd::Identity(m_obs.size(),m_obs.size()))).transpose();
  double pred(0);
  for (int i=0;i<m_npts;i++){
    VectorXd obsvalues(m_obs.size());
    for (int j=0;j<m_obs.size();j++){ obsvalues[j]=m_obs[j].Value()-(m_my_model(m_obs[j].GetX(),m_Grid.GetGrid()[i])+m_priormean(m_obs[j].GetX(),m_hpars));}
    double zmean=K.transpose()*obsvalues+m_priormean(xstar,m_hpars);
    pred+=(m_my_model(xstar,m_Grid.GetGrid()[i])+zmean)*m_Grid.GetWeights()[i]*m_values[i];
  }
  return pred;
}

VectorXd DensityKOH::MeanPred(vector<VectorXd> &vectorx) const{
  VectorXd pred(vectorx.size());
  for (int i =0;i<vectorx.size();i++){
    pred(i)=MeanPred(vectorx[i]);
  }
  return pred;
}

VectorXd DensityKOH::DrawSample(vector<VectorXd> &vectorx,default_random_engine &generator) const{
  //tire un échantillon de la prédiction de f+z.
  VectorXd avg=MeanPred(vectorx); //renvoie les prédictions moyennes
  MatrixXd COV=VarPred(vectorx); //calcul de la matrice de covariance de prédiction.
  VectorXd Xavg(avg.size()); for (int i=0;i<Xavg.size();i++){Xavg(i)=avg[i];}
  VectorXd Sample=DrawMVN(Xavg,COV,generator);
  return Sample;
}

double DensityKOH::VarPred(VectorXd &xstar) const{
  //renvoie la variance de prédiction prédiction du modèle en xstar. 
  //phase de calcul de la matrice de cov. et du vecteur associé.
  VectorXd Kstar1(m_obs.size());
  for (int i=0;i<Kstar1.size();i++){Kstar1[i]=m_Kernel(xstar,m_obs[i].GetX(),m_hpars);}
  VectorXd Kstar2(m_obs.size());
  for (int i=0;i<Kstar2.size();i++){Kstar2[i]=m_Kernel(m_obs[i].GetX(),xstar,m_hpars);}
  MatrixXd G=Gamma(&m_obs,m_hpars);
  LDLT<MatrixXd> ldlt(G);
  double varred=Kstar1.transpose()*ldlt.solve(Kstar2);
  return m_Kernel(xstar,xstar,m_hpars)-varred;
}

MatrixXd DensityKOH::VarPred(vector<VectorXd> &vectorx) const{
  //on fait tout ça pour traiter les covariances non stationnaires.
  MatrixXd Kstar1(vectorx.size(),m_obs.size());
  MatrixXd Kstar2(m_obs.size(),vectorx.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar1(i,j)=m_Kernel(vectorx[i],m_obs[j].GetX(),m_hpars);
      Kstar2(j,i)=m_Kernel(m_obs[j].GetX(),vectorx[i],m_hpars);
    }
  }
  MatrixXd Kprior(vectorx.size(),vectorx.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<vectorx.size();j++){
      Kprior(i,j)=m_Kernel(vectorx[i],vectorx[j],m_hpars);
    }
  }
  MatrixXd G=Gamma(&m_obs,m_hpars);
  LDLT<MatrixXd> ldlt(G);
  MatrixXd varred=Kstar1*ldlt.solve(Kstar2);
  return Kprior-varred;
}
  
DensityOpt::DensityOpt(Density &d) : Density(d){

}

DensityOpt::DensityOpt(DoE g) : Density(g){
}

double DensityOpt::optfunc(const std::vector<double> &x, std::vector<double> &grad, void *data){
  AugmentedDensityOpt* AD = (AugmentedDensityOpt*) data; //récupération de l'argument par pointeur
  DensityOpt *D=AD->D;
  std::vector<DATA> *newobs=AD->newobs;
  Eigen::VectorXd hpars(D->m_dim_hpars);
  for(int p=0; p<D->m_dim_hpars; p++) {hpars(p) =x[p];} // copie des hyperparamètres dans un VectorXd
  return D->loglikelihood(newobs,hpars)+D->LogPriorHpars(hpars);
}

void DensityOpt::Build()
{
  cout << "Construction de la densité Opti..." << endl;
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distU(0,1);
  double time_exec_opti;
  int maxeval=5000;
  double ftol=1e-3;
  int nd=this->GetObs().size(); //nombre d'observations
  clock_t c_start = std::clock();
  //construction de la densité opt. On fait une boucle sur les theta.
  for (int i=0;i<m_npts;i++){
       	//Creation des data
	  std::vector<DATA> data2;
	  for(int ie=0; ie<nd; ie++){
	    DATA dat; 
      dat.SetX(this->GetObs()[ie].GetX());
      dat.SetValue((this->GetObs()[ie].Value())-(this->MyModel(this->GetObs()[ie].GetX(),this->GetDoE().GetGrid()[i])));
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
	  for (int j=0;j<m_dim_hpars;j++){x[j]=exp(log(m_lb_hpars[j])+(log(m_ub_hpars[j])-log(m_lb_hpars[j]))*distU(generator));} // initial guess
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
	  //	cout << "Message d'arrêt : " << fin <<". max trouvé : " << hpars_max_opti.transpose() << ", valeur du critère : " << msup <<endl;
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
	  //	cout << "Message d'arrêt : " << fin  << ". Max : " << hpars_max_opti.transpose() << ", valeur du critère : " << msuploc << endl;
	  int niter=100; // On recommence l'opti tant qu'on n'a pas stagné 100 fois.
	  int nopt=0;
	  while (nopt<niter){
	    nopt++;
	    for (int j=0;j<m_dim_hpars;j++){x[j]=exp(log(m_lb_hpars[j])+(log(m_ub_hpars[j])-log(m_lb_hpars[j]))*distU(generator));}
	    int fin= opt.optimize(x, msup);
	    if(msup>msuploc){
	      nopt=0;
	      msuploc=msup;
	      for(int k=0;k<m_dim_hpars;k++){
		hpars_max_opti(k)=x[k];
	      }
	      //  cout << "Message d'arrêt : " << fin  << ". Nouveau max : " << hpars_max_opti.transpose() << ", valeur du critère : " << msuploc << endl;
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
  vectorexp();
  vectorweight1();
}

double DensityOpt::MeanPred(VectorXd &xstar) const{
  //renvoie la moyenne de prédiction du modèle en xstar. 
  //phase de calcul de la matrice de cov. et du vecteur associé.
  double pred(0);
  for(int t=0;t<m_npts;t++)
  {
    VectorXd Kstar(m_obs.size());
    for (int i=0;i<Kstar.size();i++){Kstar[i]=m_Kernel(xstar,m_obs[i].GetX(),m_hpars_opti[t]);}
    MatrixXd G=Gamma(&m_obs,m_hpars_opti[t]);
    LDLT<MatrixXd> ldlt(G);
    VectorXd K=(Kstar.transpose()*ldlt.solve(MatrixXd::Identity(m_obs.size(),m_obs.size()))).transpose();

    VectorXd obsvalues(m_obs.size());
    for (int j=0;j<m_obs.size();j++){ obsvalues[j]=m_obs[j].Value()-(m_my_model(m_obs[j].GetX(),m_Grid.GetGrid()[t])+m_priormean(m_obs[j].GetX(),m_hpars_opti[t]));}
    double zmean=K.transpose()*obsvalues+m_priormean(xstar,m_hpars_opti[t]);
    pred+=(m_my_model(xstar,m_Grid.GetGrid()[t])+zmean)*m_Grid.GetWeights()[t]*m_values[t];
    
  }
  return pred;
}

VectorXd DensityOpt::MeanPred(std::vector<Eigen::VectorXd> &vectorx) const{
  VectorXd pred=VectorXd::Zero(vectorx.size());
  for(int t=0;t<m_npts;t++)
  {
    VectorXd pred_theta=MeanPredCondTheta(vectorx,t);
    pred+=pred_theta*m_Grid.GetWeights()[t]*m_values[t];
  }
  return pred;
}

VectorXd DensityOpt::MeanPredCondTheta(std::vector<Eigen::VectorXd> &vectorx,int t) const{
  if(t>m_npts){cout << "erreur" << endl;}
  VectorXd Thetacourant=m_Grid.GetGrid()[t]; //valeur de theta utilisée dans la fonction
  VectorXd Hparscourant=m_hpars_opti[t]; //valeur des hpars
  VectorXd y(m_obs.size());
  for (int i=0;i<y.size();i++){
    y(i)=m_obs[i].Value()-(m_my_model(m_obs[i].GetX(),Thetacourant)+m_priormean(m_obs[i].GetX(),Hparscourant));
  }
  MatrixXd Kstar(vectorx.size(),m_obs.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar(i,j)=m_Kernel(vectorx[i],m_obs[j].GetX(),Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  VectorXd predmean=Kstar*ldlt.solve(y);
  VectorXd pmean(vectorx.size());
  for (int i=0;i<vectorx.size();i++){pmean(i)=m_priormean(vectorx[i],Hparscourant)+m_my_model(vectorx[i],Thetacourant);}
  return pmean+predmean;
}

double DensityOpt::VarPred(VectorXd &xstar) const{
  //renvoie la variance de prédiction prédiction du modèle en xstar. 
  //phase de calcul de la matrice de cov. et du vecteur associé.
  double varpred(0);
  vector<VectorXd> X(1);
  X[0]=xstar;
  for(int t=0;t<m_npts;t++){
    double var=VarPredCondTheta(X,t)(0,0);
    varpred+=var*m_Grid.GetWeights()[t]*m_values[t];
  }
  return varpred;
}

MatrixXd DensityOpt::VarPred(vector<VectorXd> &vectorx) const{
  //renvoie la variance de prédiction prédiction du modèle en xstar. 
  //phase de calcul de la matrice de cov. et du vecteur associé.
  MatrixXd varpred(vectorx.size(),vectorx.size());
  for(int t=0;t<m_npts;t++){
    MatrixXd var=VarPredCondTheta(vectorx,t);
    varpred+=var*m_Grid.GetWeights()[t]*m_values[t];
  }
  return varpred;
}

MatrixXd DensityOpt::VarPredCondTheta(std::vector<Eigen::VectorXd> &vectorx,int t) const{
  if(t>m_npts){cout << "erreur" << endl;}
  VectorXd Hparscourant=m_hpars_opti[t]; //valeur des hpars
  MatrixXd Kstar1(vectorx.size(),m_obs.size());
  MatrixXd Kstar2(m_obs.size(),vectorx.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar1(i,j)=m_Kernel(vectorx[i],m_obs[j].GetX(),Hparscourant);
      Kstar2(j,i)=m_Kernel(m_obs[j].GetX(),vectorx[i],Hparscourant);
    }
  }
  MatrixXd Kprior(vectorx.size(),vectorx.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<vectorx.size();j++){
      Kprior(i,j)=m_Kernel(vectorx[i],vectorx[j],Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  MatrixXd varred=Kstar1*ldlt.solve(Kstar2);
  return Kprior-varred;
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

VectorXd DensityOpt::DrawSample(vector<VectorXd> &vectorx,default_random_engine &generator) const{
  VectorXd ans=VectorXd::Zero(vectorx.size());
  std::uniform_real_distribution<double> distU(0,1);
  //tirage uniforme d'un des theta
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
  VectorXd Predmean=MeanPredCondTheta(vectorx,indcour);
  MatrixXd Cov=VarPredCondTheta(vectorx,indcour);
  return DrawMVN(Predmean,Cov,generator);
}

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
	    probs(j)+=exp(loglikelihood_theta_fast(&m_obs,hpars,ldlt,m_Grid.GetGrid()[j]));
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

DensityCV::DensityCV(Density &d) : DensityKOH(d)
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
      for (int j=0;j<m_dim_hpars;j++){x[j]=exp(log(m_lb_hpars[j])+(log(m_ub_hpars[j])-log(m_lb_hpars[j]))*distU(generator));} // initial guess
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
	for (int j=0;j<m_dim_hpars;j++){x[j]=exp(log(m_lb_hpars[j])+(log(m_ub_hpars[j])-log(m_lb_hpars[j]))*distU(generator));}
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
  return score; //pas de priors
}

void DensityCV::Build()
{
  VectorXd hparscv=HparsCV();
  Density::Build(hparscv);
  vectorexp();
  vectorweight1();
}

MCMC::MCMC(Density &d, int nchain) : Density(d)
{
  m_nchain=nchain;
  m_dim_mcmc=m_dim_hpars+m_dim_pars;
  m_naccept=0;
}

bool MCMC::in_bounds(Eigen::VectorXd &X) const
{
  for (int i=0;i<m_Grid.GetDimension();i++){
    if (X(i)<m_Grid.GetParsLb()(i) || X(i)>m_Grid.GetParsUb()(i)){return false;}
  }
  for (int i=0;i<m_dim_hpars;i++){
    if (X(i+m_Grid.GetDimension())<m_lb_hpars[i] || X(i+m_Grid.GetDimension())>m_ub_hpars[i]) {return false;}
  }
  return true;
}

void MCMC::Run(Eigen::VectorXd &Xinit, Eigen::MatrixXd &COV_init,default_random_engine &generator)
{
  std::normal_distribution<double> distN(0,1);
  std::uniform_real_distribution<double> distU(0,1);
  MatrixXd sqrtCOV=COV_init.llt().matrixL();
  cout << "Running MCMC with " << m_nchain << " steps..." << endl;
  VectorXd Xcurrent=Xinit;
  VectorXd Xcandidate(m_dim_mcmc);
  double fcurrent=loglikelihood_theta(&m_obs,Xcurrent.tail(m_dim_hpars),Xcurrent.head(m_dim_pars));
  double fcandidate(0);
  clock_t c_start = std::clock();

  for (int i=0;i<m_nchain;i++){
    VectorXd Step(m_dim_mcmc);
    for (int j=0;j<Step.size();j++){Step[j]=distN(generator);}
    Xcandidate=Xcurrent+sqrtCOV*Step;
    fcandidate=loglikelihood_theta(&m_obs,Xcandidate.tail(m_dim_hpars),Xcandidate.head(m_dim_pars));
    if(fcandidate>fcurrent && in_bounds(Xcandidate)){
      m_naccept++;
      Xcurrent=Xcandidate;
      fcurrent=fcandidate;
    }
    else if(fcandidate-fcurrent>log(distU(generator))  && in_bounds(Xcandidate)) {
      m_naccept++;
      Xcurrent=Xcandidate;
      fcurrent=fcandidate;
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

VectorXd MCMC::MeanPredCondX(vector<VectorXd> &vectorx,VectorXd const &Xcurrent) const
{
  //renvoie la prédiction moyenne, aux points indiqués par le vectorx, aux pars et hpars indiqués par Xcurrent.
  VectorXd Thetacourant=Xcurrent.head(m_dim_pars); //valeur de theta utilisée dans la fonction
  VectorXd Hparscourant=Xcurrent.tail(m_dim_hpars); //valeur des hpars
  VectorXd y(m_obs.size());
  for (int i=0;i<y.size();i++){
    y(i)=m_obs[i].Value()-(m_my_model(m_obs[i].GetX(),Thetacourant)+m_priormean(m_obs[i].GetX(),Hparscourant));
  }
  MatrixXd Kstar(vectorx.size(),m_obs.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar(i,j)=m_Kernel(vectorx[i],m_obs[j].GetX(),Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  VectorXd predmean=Kstar*ldlt.solve(y);
  VectorXd pmean(vectorx.size());
  for (int i=0;i<vectorx.size();i++){pmean(i)=m_priormean(vectorx[i],Hparscourant)+m_my_model(vectorx[i],Thetacourant);}
  return pmean+predmean;
}

MatrixXd MCMC::VarPredCondX(vector<VectorXd> &vectorx,VectorXd const &Xcurrent) const
{
  VectorXd Hparscourant=Xcurrent.tail(m_dim_hpars); //valeur des hpars
  MatrixXd Kstar1(vectorx.size(),m_obs.size());
  MatrixXd Kstar2(m_obs.size(),vectorx.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<m_obs.size();j++){
      Kstar1(i,j)=m_Kernel(vectorx[i],m_obs[j].GetX(),Hparscourant);
      Kstar2(j,i)=m_Kernel(m_obs[j].GetX(),vectorx[i],Hparscourant);
    }
  }
  MatrixXd Kprior(vectorx.size(),vectorx.size());
  for (int i=0;i<vectorx.size();i++){
    for (int j=0;j<vectorx.size();j++){
      Kprior(i,j)=m_Kernel(vectorx[i],vectorx[j],Hparscourant);
    }
  }
  MatrixXd G=Gamma(&m_obs,Hparscourant);
  LDLT<MatrixXd> ldlt(G);
  MatrixXd varred=Kstar1*ldlt.solve(Kstar2);
  return Kprior-varred;
}

VectorXd MCMC::MeanPred(std::vector<Eigen::VectorXd> &vectorx) const
{
  VectorXd global_pred=VectorXd::Zero(vectorx.size());
  for (const auto Xcurrent:m_selected_samples)
  {
    VectorXd mean=MeanPredCondX(vectorx,Xcurrent);
    global_pred+=mean;
  }
  return global_pred/double(m_selected_samples.size());
}

MatrixXd MCMC::VarPred(std::vector<Eigen::VectorXd> &vectorx) const
{
  MatrixXd global_cov=MatrixXd::Zero(vectorx.size(),vectorx.size());
  for (const auto Xcurrent:m_selected_samples)
  {
    MatrixXd cov=VarPredCondX(vectorx,Xcurrent);
    global_cov+=cov;
  }
  return global_cov/double(m_selected_samples.size());
}




