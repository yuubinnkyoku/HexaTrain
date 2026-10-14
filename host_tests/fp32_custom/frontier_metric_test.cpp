#define ONE_GRAPH_HOST_AUDIT
#include "one_layer_probe.cpp"
int main(){one::Vec ref={1,-2,3};
 if(metric("metricHost",0,"equal",ref,ref,{1,-2,3},.01)!=0)return 1;
 // Perfect cosine must not turn a magnitude error into an accuracy PASS.
 if(metric("metricHost",1,"scaled",ref,ref,{2,-4,6},.01)!=1)return 2;
 if(metric("metricHost",2,"opposite",ref,ref,{-1,2,-3},.01)!=1)return 3;
 if(metric("metricHost",3,"zero",{0,0,0},{0,0,0},{0,0,0},.01)!=0)return 4;
 std::puts("frontier_metric_host=PASS maxAbs_gate_independent_of_cosine=1");return 0;
}
