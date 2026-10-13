#include "one_layer_reference.h"
#include <iostream>
int main(int argc,char** argv){if(argc!=2)return 1;auto fs=one::read(argv[1]);unsigned checks=0;
 for(auto& z:fs){auto ref=one::reference(z),fp=one::reference(z,true);for(auto& p:ref){auto e=mmfixture::ln::errors(p.second,fp.at(p.first));if(e.nonfinite||e.absolute>.002){std::cerr<<z.index<<":"<<p.first;return 2;}++checks;}
  for(unsigned h=0;h<2;++h){auto pr="h"+std::to_string(h)+"_";auto p=one::d(z.v.at(pr+"p")),v=one::d(z.v.at(pr+"v")),dc=ref.at(pr+"dc");
   for(unsigned r=0;r<32;++r){double sum=0;for(unsigned c=0;c<32;++c){sum+=p[r*32+c];if(c>r&&p[r*32+c]!=0)return 3;}if(std::fabs(sum-1)>2e-7)return 4;}
   auto selected=one::split(ref.at("dc"),h);for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c)if(selected[r*32+c]!=ref.at("dc")[r*64+h*32+c])return 5;
   // Independent finite difference of <dContext, P@V> w.r.t V.
   for(unsigned position:{0u,41u,511u}){auto a=v,b=v;a[position]+=1e-5;b[position]-=1e-5;auto ca=one::mm(p,a,32,32,32),cb=one::mm(p,b,32,32,32);double numeric=0;for(unsigned j=0;j<1024;++j)numeric+=dc[j]*(ca[j]-cb[j])/2e-5;if(std::fabs(numeric-ref.at(pr+"dv")[position])>1e-6)return 6;++checks;}
   // Softmax analytic Jacobian versus independent perturbed logits.
   one::Vec logits(1024);for(unsigned j=0;j<1024;++j)logits[j]=std::log(std::max(1e-300,p[j]));auto dp=ref.at(pr+"dp");auto prob=one::softmax(logits);auto prod=one::mul(dp,prob),dot=one::sumRows(prod);for(unsigned position:{33u,65u,991u}){auto a=logits,b=logits;a[position]+=1e-5;b[position]-=1e-5;auto pa=one::softmax(a),pb=one::softmax(b);double numeric=0;for(unsigned j=0;j<1024;++j)numeric+=dp[j]*(pa[j]-pb[j])/2e-5;double analytic=prob[position]*(dp[position]-dot[position/32]);if(std::fabs(numeric-analytic)>1e-6)return 7;++checks;}
  }
 }std::cout<<"one_layer_reference_test=PASS cases="<<fs.size()<<" checks="<<checks<<" shape_transpose_gate_causal_fd=PASS\n";
}
