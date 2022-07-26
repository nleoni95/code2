Eigen::VectorXd halton ( int i, unsigned m );
Eigen::VectorXd halton_base ( int i, int m, int b[] );
int halton_inverse ( double r[], int m );
Eigen::MatrixXd halton_sequence ( int i1, int i2, int m );
int prime ( int n );
double r8_mod ( double x, double y );
Eigen::MatrixXd LHS(unsigned n, unsigned dim, std::default_random_engine &generator);

