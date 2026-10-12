#include "qnn/layernorm_stability_reference.h"
#include <iostream>
#include <iomanip>
using namespace phonelm::qnn::lnvalidation;
int main(){bool ok=true;double worst=0;
 for(size_t d:{8u,16u,64u,128u}) for(const auto& c:cases(d)){
  auto a=doubleReference(c,d),b=doubleReference(c,d,true);
  double eq=std::max({error(a.y,b.y),error(a.dx,b.dx),error(a.dg,b.dg),error(a.db,b.db)});
  double numeric=finiteDifferenceError(c,d,1e-6,true);
  double numeric2=finiteDifferenceError(c,d,2e-6,true);
  auto f=reference(c.x,c.dy,c.gamma,c.beta,d,1e-5f,true);
  double fp=std::max({error(a.y,f.y),error(a.dx,f.dx),error(a.dg,f.dg),error(a.db,f.db)});
  bool pass=eq<1e-10 && numeric<5e-4 && numeric2<5e-4;
  ok&=pass;worst=std::max(worst,numeric);
  std::cout<<std::setprecision(12)<<"d="<<d<<" case="<<c.name<<" scaled_equivalence="<<eq
   <<" numeric_h1="<<numeric<<" numeric_h2="<<numeric2<<" fp32_max_abs="<<fp<<" pass="<<pass<<'\n';
 }
 std::cout<<"cpu_analytic_vs_numeric_max_abs_error="<<worst<<"\nstatus="<<(ok?"SUCCESS":"FAILED")<<'\n';return ok?0:1;
}
