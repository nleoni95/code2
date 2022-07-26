/* The class for the observations */
class DATA {
	friend class GP;
	friend class TPS;
	friend class EGO;
public:
	DATA(){};
	DATA(Eigen::VectorXd const &x, double const &f){ X=x; F=f;};
	DATA(DATA const &d){ X = d.X; F= d.F;};
	void operator = (const DATA d){ X = d.X; F= d.F;};
	Eigen::VectorXd GetX() const { return X; };
	double Value() const { return F; };
    void SetX(Eigen::VectorXd x) { X=x;};
    void SetValue(double f) { F=f;};
private:
	Eigen::VectorXd X;
	double F;
};
