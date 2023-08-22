#include <iostream>
#include <vector>
#include <stdio.h>      
#include <math.h>       
#include <random>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
namespace py = pybind11;
/*-----------------------------------------------------------
Function for Tau-leap SSA (Gillespie algorithm), 
for the Prokaryotic autoregulator model 
 ------------------------------------------------------------- */
 std::array<int, 4>  SEIR(py::array_t<double> rates, 
   py::array_t<int> init_molecules, double_t tsart, double_t tend){
  // set fixed elements:
  auto c = rates.unchecked<1>();
  auto x0 = init_molecules.unchecked<1>();
  std::array<int, 4> x ={x0(0), x0(1), x0(2), x0(3)};
  double h0, h1,h2,h3,u1,u2,dt;
  double t = tsart;
  double N = 300.0;
  std::random_device rd;  
  std::mt19937 gen(rd()); 

  std::uniform_real_distribution<double> u(0.0, 1.0);
  //the propensity is encoded columnwise here
  std::vector<std::vector<int> > propensity{{-1, 1, 0, 0}, 
							                              {0, -1, 1, 0},
                                            {0, 0, -1, 1}};     

  while (t<tend)
  {
    
    h1=c(0)*x[0]*x[2]/N; h2=c(1)*x[1]; h3=c(2)*x[2];
    h0=h1 + h2 + h3;
 
    if ((h0<1e-10)||(x[1]>=10000))
      t=1e99;
    else{
      u1 = u(gen);
      dt = -log(u1) / h0;
      t += dt;
    }

 
    std::discrete_distribution<> d({h1,h2,h3});
    int index = d(gen);
    for (int j = 0; j < propensity[index].size(); j++)
                x[j] = x[j] + propensity[index][j];       
  }
  return x;
}
PYBIND11_PLUGIN(seirssa) {
    pybind11::module m("seirssa", "auto-compiled c++ extension");
    m.def("SEIR", &SEIR, py::return_value_policy::reference_internal);
    return m.ptr();
}