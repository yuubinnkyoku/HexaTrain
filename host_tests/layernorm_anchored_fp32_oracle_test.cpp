#include "qnn/layernorm_redesign_reference.h"
#include <iostream>
#include <iomanip>
using namespace phonelm::qnn::lnvalidation;
int main(){bool all=true;double worst=0;size_t count=0;
 if(halfLattice(65505)!=65504 || halfLattice(-65505)!=-65504 || halfLattice(1e8)!=65504 || halfLattice(65504)!=65504)return 2;
 for(size_t d:{8u,16u,32u,64u,128u})for(const auto& c:redesignCases(d)){
  auto ref=doubleReference(c,d);auto fp=anchoredFp32Oracle(c,d);
  for(double v:ref.dx)if(!std::isfinite(v)||std::abs(v)>65504)return 3;
  double y=error(ref.y,fp.y),h=error(ref.xhat,fp.xhat),dx=error(ref.dx,fp.dx),dg=error(ref.dg,fp.dg),db=error(ref.db,fp.db);
  // Stronger ordinary forward bound applies to ALL oracle cases.
  bool pass=y<=.002&&h<=.002&&dx<=.02&&dg<=.02&&db<=.02;all&=pass;worst=std::max(worst,dx);++count;
  std::cout<<std::setprecision(12)<<"oracle d="<<d<<" name="<<c.name<<" y_abs="<<y<<" xhat_abs="<<h<<" dx_abs="<<dx<<" dg_abs="<<dg<<" db_abs="<<db<<" pass="<<pass<<'\n';
 }std::cout<<"oracle_cases="<<count<<"\noracle_dx_max_abs="<<worst<<"\nDSP_VERIFIED=false\nTRAINING_WIRED=false\nstatus="<<(all?"SUCCESS":"FAILED")<<'\n';return all?0:1;
}
