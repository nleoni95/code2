/*=============================================================================

  Nicolas Leoni
  Copyright 2022
  
  Supplementary material of the article:
  Leoni, N., Congedo, P.M., Le Maître, O. and Rodio, M.G., Bayesian calibration with adaptive model discrepancy, International Journal for Uncertainty Quantification (2022)

  Implementation of the MIT Boiling model found in: 
  Kommajosyula, R. Development and assessment of a physics-based model for subcooled flow boiling with application to CFD (2022)

=============================================================================*/

#include <vector>

double MITB(double const DTsup, vector<double> const & par){
  //MIT Boiling model. DTsup is the wall superheat, par contains the three coefficients calibrated in the article.
  
  //normalization of coefficients according to the article
  vector<double> parnorm(3);
  parnorm[0] = 20 + 40 * par[0];
  parnorm[1] = 9.45e-6 + 18.9e-6 * par[1];
  parnorm[2] = 0.375 + 0.75 * par[2];
  
  //conditions specific to the Kennel 6 experimental case.
  //must be changed if one want to work with other cases. Physical properties were obtained using the Python library pyXsteam.
  double p = 4;
  double vel = 1.22;
  double DTsub = 55.5;
  double Dh = 1.956e-2;
  double Tsat = 1.436125e+02;
  double rhof = 9.667158e+02;
  double muf = 3.214146e-04;
  double rhog = 2.162668e+00;
  double cpf = 4.202403e+03;
  double kf = 6.721438e-01;
  double hfg = 2.133333e+06;
  double sigma = 5.009629e-02;
  double fric = 1.930011e-02;
  
  // MITB
  double Twall = Tsat + DTsup;
  double Tbulk = Tsat - DTsub;
  double angle = parnorm[0] * M_PI / 180;
  double Re = rhof * vel * Dh / muf;
  double Pr = muf * cpf / kf;
  double Jasub = rhof * cpf * DTsub / (rhog * hfg);
  double Jasup = rhof * cpf * DTsup / (rhog * hfg);
  double etaf = kf / (rhof * cpf);
  double NuD = ((fric / 8) * (Re - 1000) * Pr) / ( 1 + 12.7 * sqrt(fric / 8) * (pow(Pr, 2. / 3.) - 1));
  double hfc = NuD * kf / Dh;
  double Dd = parnorm[1] * pow(((rhof - rhog) / rhog), 0.27) * pow(Jasup, parnorm[2]) * pow(1 + Jasub, -0.3) * pow(vel, -0.26);
  double twait = 6.1e-3 * pow(Jasub, 0.6317) / DTsup;
  double chi = max(0., 0.05 * DTsub / DTsup);
  double c1 = 1.243 / sqrt(Pr);
  double c2 = 1.954 * chi;
  double c3 = -1 * min(abs(c2), 0.5 * c1);
  double K = (c1 + c3) * Jasup * sqrt(etaf);
  double tgrowth = pow(0.25 * Dd / K, 2);
  double freq = 1. / (twait + tgrowth);
  double N0 = freq * tgrowth * 3.1415 * pow(0.5 * Dd, 2);
  double pp=p * 1e5;
  double Tg = Tsat + DTsup + 273.15;
  double TTsat = Tsat + 273.15;
  double rhoplus = log10((rhof - rhog) / rhog);
  double frhoplus = -0.01064 + 0.48246 * rhoplus - 0.22712 * pow(rhoplus, 2) + 0.05468 * pow(rhoplus, 3);
  double Rc = (2 * sigma * (1 + (rhog / rhof))) / (pp * (exp(hfg * (Tg - TTsat) / (462 * Tg * TTsat)) - 1));
  double Npp = (4.72E5) * (1 - exp(-(pow(angle, 2)) / (8 * (pow(0.722, 2))))) * (exp(frhoplus * (2.5E-6) / Rc) - 1);
  double Nppb;
  if(N0 * Npp < exp(-1)){
    Nppb = Npp;
  }
  else if(N0 * Npp < exp(1)){
    Nppb = (0.2689 * N0 * Npp + 0.2690) / N0;
  }
  else{
    Nppb = (log(N0 * Npp) - log(log(N0 * Npp))) / N0;
  }
  double Ca = (muf * K) / (sigma * sqrt(tgrowth));
  double Ca0 = 2.16 * 1E-4 * (pow(DTsup, 1.216));
  double rappD = max(0.1237 * pow(Ca, -0.373) * sin(angle), 1.);
  double Dinception = rappD * Dd;
  double Dml = Dinception / 2.;
  double deltaml = 4E-6 * sqrt(Ca / Ca0);
  double phiml = rhof * hfg * freq * Nppb * (deltaml * (pow(Dml, 2)) * (3.1415 / 12.) * (2 - (pow(rappD, 2) + rappD)));
  double phiinception = 1.33 * 3.1415 * pow(Dinception / 2.,3) * rhog * hfg * freq * Nppb;
  double phie = phiml + phiinception;
  double Dlo = 1.2 * Dd;
  double Asl = (Dlo + Dd) / (2 * sqrt(Nppb));
  double tstar = (pow(kf, 2)) / ((pow(hfc, 2)) * 3.1415 * etaf); tstar = min(tstar, twait);
  double Ssl = min(1., Asl * Nppb * tstar * freq);
  double phisc = 2 * hfc * Ssl * (DTsup + DTsub);
  double phifc = (1 - Ssl) * hfc * (DTsup + DTsub);
  return phisc + phifc + phie;
}
