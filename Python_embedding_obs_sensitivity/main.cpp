// Dans ce fichier : on met en oeuvre différents algorithmes d'optimisation pour KOH. Voir lequel est le meilleur.
// On met en place une quadrature pour évaluer de manière précise l'intégrale KOH.
// On regarde maintenant la sensibilité aux observations.


#define PY_SSIZE_T_CLEAN
#include <iostream>
#include <Python.h>
#include <Eigen/Dense>
using namespace std;
PyObject *pFunc, *pValue, *pArgs;
PyObject *pParamsRavik; //liste comprenant les valeurs nominales des paramètres de Ravik.



double my_model(Eigen::VectorXd &x, Eigen::VectorXd &theta){
  //taille totale des paramètres dans le modèle : 10.
  //création d'une liste pour les arguments
  PyList_SetItem(pParamsRavik,0,PyFloat_FromDouble(theta(0))); //angle. VN 40deg
  PyList_SetItem(pParamsRavik,1,PyFloat_FromDouble(theta(1))); //coef multiplicateur. VN 18.9E6
  PyList_SetItem(pParamsRavik,3,PyFloat_FromDouble(theta(2))); //param de DTsup. VN 0.75
  pArgs=PyTuple_New(2);
  PyTuple_SetItem(pArgs,0,PyFloat_FromDouble(x(0)));
  PyTuple_SetItem(pArgs,1,PyList_AsTuple(pParamsRavik));
  cout << PyFloat_AsDouble(PyTuple_GetItem(PyTuple_GetItem(pArgs,1),9)) << endl;
  pValue = PyObject_CallObject(pFunc, pArgs); //pvalue est alors l'objet de retour.
  return PyFloat_AsDouble(pValue);
}

int main(int argc, char **argv){
  
  if(argc != 3){
    cout << "Usage :\n";
    cout << "\t Number of obs, seed_obs\n";
    exit(0);
  }
  const int cas=6;
  cout << "bonjour" << endl;
  int nd  = atoi(argv[1]);
  uint32_t seed_obs=atoi(argv[2]);//
  Py_Initialize(); 
  PyRun_SimpleString("import sys"); // déjà initialisé par Py_Initialize ?
  PyRun_SimpleString("import os");
  PyRun_SimpleString("sys.path.append(os.getcwd())");
  PyObject *pName, *pModule;
 //https://medium.com/datadriveninvestor/how-to-quickly-embed-python-in-your-c-application-23c19694813
  pName = PyUnicode_FromString((char*)"model"); //nom du fichier sans .py
  pModule = PyImport_Import(pName);
  //PyErr_Print(); pratique !!
  pFunc= PyObject_GetAttrString(pModule, (char*)"initialize_case");//nom de la fonction
  pArgs = PyTuple_Pack(1, PyLong_FromLong(cas));//premier argument : nombre d'arguments de la fonction python, ensuite les arguments.
  pValue = PyObject_CallObject(pFunc, pArgs); //pvalue est alors l'objet de retour. //appel à initialize_case

  pFunc = PyObject_GetAttrString(pModule, (char*)"exp_datab");//nom de la fonction
  pArgs = PyTuple_New(0); //tuple vide
  pValue = PyObject_CallObject(pFunc, pArgs); //pvalue est alors l'objet de retour.
  //récupération des observations. Attention au déréférencement du pointeur et au nombre de données.
  if(PyList_Check(pValue)!=1){cerr << "erreur : la fonction exp_datab n'a pas renvoyé une liste" << endl;}

  for (int i=0;i<PyList_Size(pValue);i++){
    //remplissage des observations
    double a=PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(pValue,i),0));
    double b=PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(pValue,i),1));
    cout << a << " ; " << b << endl;
  }


  //initialisation des paramètres de Ravik
  pParamsRavik=PyList_New(10);
  PyList_SetItem(pParamsRavik,0,PyFloat_FromDouble(40)); //angle de contact
  PyList_SetItem(pParamsRavik,1,PyFloat_FromDouble(18.9e-6)); //paramètres de corrélation Dd
  PyList_SetItem(pParamsRavik,2,PyFloat_FromDouble(0.27));
  PyList_SetItem(pParamsRavik,3,PyFloat_FromDouble(0.75));
  PyList_SetItem(pParamsRavik,4,PyFloat_FromDouble(-0.3));
  PyList_SetItem(pParamsRavik,5,PyFloat_FromDouble(-0.26));
  PyList_SetItem(pParamsRavik,6,PyFloat_FromDouble(6.1E-3)); //corrélation twait
  PyList_SetItem(pParamsRavik,7,PyFloat_FromDouble(0.6317));
  PyList_SetItem(pParamsRavik,8,PyFloat_FromDouble(0.1237)); //corrélation rappD
  PyList_SetItem(pParamsRavik,9,PyFloat_FromDouble(-0.373));

  pFunc = PyObject_GetAttrString(pModule, (char*)"run_model");//nom de la fonction
  Eigen::VectorXd X=Eigen::VectorXd::Zero(1);
  Eigen::VectorXd theta=Eigen::VectorXd::Zero(3);
  X(0)=8;
  theta(0)=41;
  theta(1)=18.9e-6;
  theta(2)=0.75;
  double resulto=my_model(X,theta);
  cout << "résultat final :" << resulto << endl;
  Py_Finalize();

  exit(0);
  
};
