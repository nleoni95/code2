#include <iostream>
#include <iomanip>
#include <fstream>
#include <list>
#include <Eigen/Dense>
#include <math.h>
#include "pcb++.h"
using namespace std;
using namespace Eigen;

void PCB::BasisProd(){/// Construct the spectral multiplication tensor of the basis.
	cout << "\tConstruct multiplication tensor\n";
	if(nord+1>20){ cout << "Nord is too large for BasisPr (max 20) " << endl; return;}
	double aleg[21][21][21];
	double aher[21][21][21];
	double eps = 1.e-8;
//First define the tensor for the 1-D case
	if(Ctype =='U' || Ctype=='M'){
		for(int i1=0; i1<=nord; i1++){
			for(int i2=0; i2<=nord; i2++){
				for(int i3=0; i3<=nord; i3++) aleg[i1][i2][i3] = 0;
			}
		}
		for(int iq=0; iq<nq1d; iq++){
			VectorXd px = pleg_at_x(xleg[iq]);
			for(int i1=0; i1<=nord; i1++){
				for(int i2=0; i2<=nord; i2++)
					for(int i3=0; i3<=nord; i3++) aleg[i1][i2][i3] += px[i1]*px[i2]*px[i3]*wleg[iq];
			}
		}
		for(int i1=0; i1<=nord; i1++){
			for(int i2=0; i2<=nord; i2++)
				for(int i3=0; i3<=nord; i3++) if(fabs(aleg[i1][i2][i3])<=eps) aleg[i1][i2][i3]=0.;
		}
	}
	if(Ctype =='N' || Ctype=='M'){
		for(int i1=0; i1<=nord; i1++){
			for(int i2=0; i2<=nord; i2++)
				for(int i3=0; i3<=nord; i3++) aher[i1][i2][i3] = 0;
		}
		for(int iq=0; iq<nq1d; iq++){
			VectorXd px =pher_at_x(xher[iq]);
			for(int i1=0; i1<=nord; i1++){
				for(int i2=0; i2<=nord; i2++)
					for(int i3=0; i3<=nord; i3++) aher[i1][i2][i3] += px[i1]*px[i2]*px[i3]*wher[iq];
			}
		}
		for(int i1=0; i1<=nord; i1++){
			for(int i2=0; i2<=nord; i2++)
				for(int i3=0; i3<=nord; i3++) if(fabs(aher[i1][i2][i3])<=eps) aher[i1][i2][i3]=0.;
		}
	}
//Second tensorize:
	Prod = new list<op_pc> [npol];
#pragma omp parallel for
	for(int i1=0; i1<npol; i1++){
		op_pc elem; elem.l=0;
		for(int i2=0; i2<npol; i2++){
			for(int i3=i2; i3<npol; i3++){
				double prod = 1.;
				for(int id = 0; id <ndim ; id++){
					if( (alp(i1,id)+alp(i2,id)+alp(i3,id))%2 == 1){
						prod = 0.; break;
					} else if(type[id]=='U'){
						prod *= aleg[alp(i1,id)][alp(i2,id)][alp(i3,id)];
					}else{
						prod *= aher[alp(i1,id)][alp(i2,id)][alp(i3,id)];
					}
					if(fabs(prod) < eps) break;
				}
				if(fabs(prod)>=eps){
					elem.i = i2; elem.j = i3; elem.c = prod; Prod[i1].push_back(elem);
					if(i2 != i3){ elem.i = i3; elem.j = i2;  Prod[i1].push_back(elem);}
				}
			}
		}
	}
};

void PCB::BasisTProd(){/// Construct the triple multiplication tensor of the basis.
	cout << "\tConstruct triple multiplication tensor\n";
	if(nord+1>20){ cout << "Nord is too large for BasisPr (max 20) " << endl; return;}
	double aleg[21][21][21][21];
	double aher[21][21][21][21];
	double eps = 1.e-8;
//First define the tensor for the 1-D case
	if(Ctype =='U' || Ctype=='M'){
		for(int i1=0; i1<=nord; i1++){
			for(int i2=0; i2<=nord; i2++){
				for(int i3=0; i3<=nord; i3++)
					for(int i4=0; i4<=nord; i4++) aleg[i1][i2][i3][i4] = 0;
			}
		}
		for(int iq=0; iq<nq1d; iq++){
			VectorXd px = pleg_at_x(xleg(iq));
			for(int i1=0; i1<=nord; i1++){
				for(int i2=0; i2<=nord; i2++){
					for(int i3=0; i3<=nord; i3++){
						double p123w = px[i1]*px[i2]*px[i3]*wleg[iq];
						for(int i4=0; i4<=nord; i4++) aleg[i1][i2][i3][i4] += px[i4]*p123w;
					}
				}
			}
		}
		for(int i1=0; i1<=nord; i1++){
			for(int i2=0; i2<=nord; i2++){
				for(int i3=0; i3<=nord; i3++)
					for(int i4=0; i4<=nord; i4++) if(fabs(aleg[i1][i2][i3][i4])<=eps) aleg[i1][i2][i3][i4]=0.;
			}
		}
	}
	if(Ctype =='N' || Ctype=='M'){
		for(int i1=0; i1<=nord; i1++){
			for(int i2=0; i2<=nord; i2++){
				for(int i3=0; i3<=nord; i3++)
					for(int i4=0; i4<=nord; i4++) aher[i1][i2][i3][i4] = 0;
			}
		}
		for(int iq=0; iq<nq1d; iq++){
			VectorXd px = pher_at_x(xher[iq]);
			for(int i1=0; i1<=nord; i1++){
				for(int i2=0; i2<=nord; i2++){
					for(int i3=0; i3<=nord; i3++){
						double p123w = px[i1]*px[i2]*px[i3]*wher[iq];
						for(int i4=0; i4<=nord; i4++) aher[i1][i2][i3][i4] += px[i4]*p123w;
					}
				}
			}
		}
		for(int i1=0; i1<=nord; i1++){
			for(int i2=0; i2<=nord; i2++){
				for(int i3=0; i3<=nord; i3++)
					for(int i4=0; i4<=nord; i4++) if(fabs(aher[i1][i2][i3][i4])<=eps) aher[i1][i2][i3][i4]=0.;
			}
		}
	}
//Second tensorize:
	Triple = new list<op_pc> [npol];
#pragma omp parallel for
	for(int i1=0; i1<npol; i1++){
		op_pc elem;
		for(int i2=0; i2<npol; i2++){
			for(int i3=i2; i3<npol; i3++){
				for(int i4=i3; i4<npol; i4++){
					double prod = 1.;
					for(int id = 0; id <ndim ; id++){
						if( (alp(i1,id)+alp(i2,id)+alp(i3,id)+alp(i4,id))%2 == 1){
							prod = 0.;
						} else if(type[id]=='U'){
							prod *= aleg[alp(i1,id)][alp(i2,id)][alp(i3,id)][alp(i4,id)];
						}else{
							prod *= aher[alp(i1,id)][alp(i2,id)][alp(i3,id)][alp(i4,id)];
						}
						if(fabs(prod) < eps) break;
					}
					if(fabs(prod)>=eps){
						elem.c = prod;
						if(i2 == i3){
							if(i3==i4){ elem.i = i2; elem.j = i3; elem.l = i4; Triple[i1].push_back(elem);
							}else{
								elem.i = i2; elem.j = i3; elem.l = i4; Triple[i1].push_back(elem);
								elem.i = i2; elem.j = i4; elem.l = i3; Triple[i1].push_back(elem);
								elem.i = i4; elem.j = i3; elem.l = i2; Triple[i1].push_back(elem);
							}
						}else{
							if(i3==i4){
								elem.i = i2; elem.j = i3; elem.l = i4; Triple[i1].push_back(elem);
								elem.i = i3; elem.j = i2; elem.l = i4; Triple[i1].push_back(elem);
								elem.i = i4; elem.j = i3; elem.l = i2; Triple[i1].push_back(elem);
							}else{
								elem.i = i2; elem.j = i3; elem.l = i4; Triple[i1].push_back(elem);
								elem.i = i2; elem.j = i4; elem.l = i3; Triple[i1].push_back(elem);
								elem.i = i3; elem.j = i2; elem.l = i4; Triple[i1].push_back(elem);
								elem.i = i3; elem.j = i4; elem.l = i2; Triple[i1].push_back(elem);
								elem.i = i4; elem.j = i2; elem.l = i3; Triple[i1].push_back(elem);
								elem.i = i4; elem.j = i3; elem.l = i2; Triple[i1].push_back(elem);
							}
						}
					}
				} //next i4
			}//next i3
		}//next i2
	}//next mode
};
