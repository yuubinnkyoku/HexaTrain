// Bounded standard-op controls. Host-created operands are explicit standalone
// controls, never a native Saved-P certification.
#define main unusedSavedPMain
#include "saved_p_backward_probe.cpp"
#undef main
int main(int argc,char** argv){
 if(argc!=7)return 1;
 try{
  auto cs=doubleqkfixture::load(argv[1]);auto gs=savedp::loadGradients(argv[4]);
  savedp::rawEvidence=std::fopen("saved-p-words.bin","wbx");
  unsigned header[]={0x42504301,1,2,6};savedp::writeRaw(header,4);
  initializeDoubleQk();unsigned cases=0;
  for(unsigned rank:{2u,4u})for(unsigned stage=0;stage<6;++stage){
   DiagnosticGraph g("primitive"+std::to_string(rank)+std::to_string(stage));
   auto matrix=[&](const char* name,Qnn_TensorType_t type,unsigned cols=32){return g.tensor(name,type,rank==2?std::vector<unsigned>{32,cols}:std::vector<unsigned>{1,1,32,cols});};
   unsigned a=matrix("a",QNN_TENSOR_TYPE_APP_WRITE),b=~0u;
   if(stage!=1)b=matrix("b",QNN_TENSOR_TYPE_APP_WRITE,stage==3?1:32);
   auto y=matrix("output",QNN_TENSOR_TYPE_APP_READ,(stage==1||stage==2)?1:32);
   if(stage==0||stage==4)g.node("multiply",QNN_OP_ELEMENT_WISE_MULTIPLY,"qti.aisw",{a,b},{y});
   if(stage==3)g.node("subtract",QNN_OP_ELEMENT_WISE_SUBTRACT,"qti.aisw",{a,b},{y});
   if(stage==5)g.node("dvalue",QNN_OP_MAT_MUL,"qti.aisw",{a,b},{y},{savedp::boolean(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,true),savedp::boolean(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,false)});
   if(stage==1||stage==2){
    auto input=a;
    if(stage==2){input=matrix("product",QNN_TENSOR_TYPE_NATIVE);g.node("product",QNN_OP_ELEMENT_WISE_MULTIPLY,"qti.aisw",{a,b},{input});}
    auto axis=g.constant("axis",{1},{from(rank-1)},QNN_DATATYPE_UINT_32);
    Qnn_Param_t axes=QNN_PARAM_INIT;axes.name=QNN_OP_REDUCE_SUM_PARAM_AXES;axes.paramType=QNN_PARAMTYPE_TENSOR;axes.tensorParam=g.tensors[axis];
    g.node("reduce",QNN_OP_REDUCE_SUM,"qti.aisw",{input},{y},{axes,savedp::boolean(QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS,true)});
   }
   g.finalize();
   for(unsigned ci=0;ci<24;++ci)for(unsigned h=0;h<2;++h){
    std::vector<float> p(1024),grad(1024);
    if(ci<16){unsigned ids[]={2,4,10,14,16,19,23,44};unsigned fi=ids[ci/2],gi=ci%2?(fi<44?9:25):0;p=cs[fi].hostP[h];grad=gs[gi][h][stage==5?1:0];}
    else for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c){unsigned i=r*32+c,s=ci-16;
     p[i]=s==0?(r==0&&c==3):s==1?(r==c):s==2?(c==(r*7+3)%32):s==3?(c==(r+h+1)%32):s==4?float((i%11)-5.f)/32.f:s==5?((c&1)?-.25f:.25f):s==6?((i&1)?1.f/1024:16.f):from(i%2?0x80000000:1);
     grad[i]=s==4?float(int(r)*3-int(c)*5+int(h)*7):s==5?((r&1)?-128.f:128.f):s==6?((i&1)?1024.f:1.f/1024):s==7?from(i%2?0x80000001:0):float((i*13+h*7)%31)-15.f;
    }
    std::vector<float> aa=p,bb=grad,prod(1024),dot(32,0),center(1024);
    for(unsigned i=0;i<1024;++i){volatile float v=p[i]*grad[i];prod[i]=v;}
    for(unsigned r=0;r<32;++r)for(unsigned c=0;c<32;++c)dot[r]=dot[r]+prod[r*32+c];
    for(unsigned i=0;i<1024;++i)center[i]=grad[i]-dot[i/32];
    if(stage==1)aa=prod;if(stage==3){aa=grad;bb=dot;}if(stage==4)bb=center;
    std::map<unsigned,std::vector<float>> in={{a,aa}};if(b!=~0u)in[b]=bb;
    auto out=g.run(in);
    unsigned key[]={rank,stage,ci,h,b==~0u?2u:3u};savedp::writeRaw(key,5);
    for(unsigned id:{a,b,y})if(id!=~0u){unsigned field[]={id==a?1u:id==b?2u:3u,g.size(id)};savedp::writeRaw(field,2);
     auto values=id==y?out.at(y):[&](){savedp::Words v;for(float f:in.at(id))v.push_back(word(f));return v;}();savedp::writeRaw(values.data(),values.size());}
    ++cases;
   }
  }
  std::fclose(savedp::rawEvidence);closeProbe();
  std::printf("summary,primitive_controls,cases=%u,finite=1,input_mutation=0,poison_remaining=0,canary=1,teardown=1,cpu_arithmetic_fallback=0\n",cases);return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 5;}
}
