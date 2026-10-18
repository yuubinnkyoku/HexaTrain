// Diagnostic only. Production operators, with no new DSP arithmetic.
#pragma once
#include "double_qk_initialize.h"
#include "double_qk_fixture.h"
#include <array>
namespace savedp {
enum class Mode { StandaloneSoftmax, StandaloneDValue, Forward, NativeSoftmax, NativeDValue, Dual };
inline const char* label(Mode m) {
  switch(m) {
    case Mode::StandaloneSoftmax:return "standalone_softmax";
    case Mode::StandaloneDValue:return "standalone_dvalue";
    case Mode::Forward:return "forward";
    case Mode::NativeSoftmax:return "native_softmax";
    case Mode::NativeDValue:return "native_dvalue";
    default:return "dual";
  }
}
inline Qnn_Param_t boolean(const char* name,bool value) {
  Qnn_Param_t p=QNN_PARAM_INIT;p.name=name;p.paramType=QNN_PARAMTYPE_SCALAR;
  p.scalarParam.dataType=QNN_DATATYPE_BOOL_8;p.scalarParam.bool8Value=value;return p;
}
struct Graph {
  DiagnosticGraph g;
  Mode mode;bool observed,tap,native,softmax,dvalue;
  std::array<unsigned,2> p{},dp{},dc{},product{},dot{},centered{},ds{},dv{};
  unsigned x=~0u,ga=~0u,be=~0u,dy=~0u;
  std::array<unsigned,3> w{};
  double finalizeUs=0;
  Graph(Mode m,bool o,bool t):g(std::string("savedp_")+label(m)+std::to_string(o)+std::to_string(t)),
    mode(m),observed(o),tap(t),native(m!=Mode::StandaloneSoftmax&&m!=Mode::StandaloneDValue),
    softmax(m==Mode::StandaloneSoftmax||m==Mode::NativeSoftmax||m==Mode::Dual),
    dvalue(m==Mode::StandaloneDValue||m==Mode::NativeDValue||m==Mode::Dual) {
    g.receiptEnabled=o;
    auto mid=t?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE;
    auto mat=[&](const std::string& n,Qnn_TensorType_t type,unsigned r=32,unsigned c=32) {
      return g.tensor(n.c_str(),type,{1,1,r,c});
    };
    auto rec=[&](const char* n,unsigned count) {
      return g.tensor(n,QNN_TENSOR_TYPE_APP_READ,{1,1,1,count},QNN_DATATYPE_INT_32);
    };
    if(native) {
      // Same forward arithmetic and common local-LN backward control as the protected graph.
      x=mat("x",QNN_TENSOR_TYPE_APP_WRITE,32,64);
      ga=mat("gamma",QNN_TENSOR_TYPE_APP_WRITE,1,64);
      be=mat("beta",QNN_TENSOR_TYPE_APP_WRITE,1,64);
      dy=mat("dy",QNN_TENSOR_TYPE_APP_WRITE,32,64);
      auto y=mat("y",mid,32,64),h=mat("h",mid,32,64),iv=mat("iv",mid,32,1);
      auto eps=g.constant("eps",{1,1,1,1},{1e-5f});
      std::vector<unsigned> lo={y,h,iv};if(o)lo.push_back(rec("ln_receipt",6313));
      g.node("ln1",o?"S2":"S2NoReceipt","HexaSelectiveLn1",{x,ga,be,eps},lo);
      auto dx=mat("dx",QNN_TENSOR_TYPE_APP_READ,32,64);
      auto dg=mat("dg",QNN_TENSOR_TYPE_APP_READ,1,64),db=mat("db",QNN_TENSOR_TYPE_APP_READ,1,64);
      g.node("local_backward","LnBackward","HexaFp32",{dy,ga,h,iv},{dx,dg,db,rec("ln_back_receipt",6248)});
      std::array<unsigned,3> q{};
      for(unsigned j=0;j<3;++j) {
        w[j]=mat("w"+std::to_string(j),QNN_TENSOR_TYPE_APP_WRITE,64,64);
        q[j]=mat("qkv"+std::to_string(j),mid,32,64);
      }
      std::vector<unsigned> qo={q[0],q[1],q[2]};if(o)qo.push_back(rec("mixed_receipt",20512));
      g.node("mixed",o?"MixedPrecisionQKV":"MixedPrecisionQKVNoReceipt","HexaMixedQkv",{y,w[0],w[1],w[2]},qo);
      std::array<unsigned,6> heads{};
      for(unsigned j=0;j<6;++j)heads[j]=mat("head"+std::to_string(j),j>=4?QNN_TENSOR_TYPE_APP_READ:mid);
      g.node("heads","QKVHeads","HexaFp32",{q[0],q[1],q[2]},
        {heads[0],heads[1],heads[2],heads[3],heads[4],heads[5],rec("head_receipt",12296)});
      std::array<unsigned,2> raw{},scaled{},masked{};
      for(unsigned j=0;j<2;++j) {
        raw[j]=mat("raw"+std::to_string(j),mid);
        scaled[j]=mat("scaled"+std::to_string(j),mid);
        masked[j]=mat("masked"+std::to_string(j),mid);
        p[j]=mat("p"+std::to_string(j),m==Mode::Forward?QNN_TENSOR_TYPE_APP_READ:mid);
      }
      std::vector<unsigned> ro={raw[0],raw[1]};if(o)ro.push_back(rec("qk_receipt",6176));
      g.node("double_qk",o?"ForwardRawQkDouble":"ForwardRawQkDoubleNoReceipt","HexaDoubleQk",
        {heads[0],heads[1],heads[2],heads[3]},ro);
      auto sc=g.constant("scale",{1,1,1,1},{1.f/std::sqrt(32.f)});
      std::vector<float> mask(1024,0.f);
      for(unsigned r=0;r<32;++r)for(unsigned c=r+1;c<32;++c)mask[r*32+c]=-10000.f;
      auto mc=g.constant("mask",{1,1,32,32},mask);
      g.node("scale","ForwardScoreScale","HexaFp32",{raw[0],raw[1],sc},{scaled[0],scaled[1],rec("scale_receipt",4105)});
      g.node("mask","ForwardMaskAdd","HexaFp32",{scaled[0],scaled[1],mc},{masked[0],masked[1],rec("mask_receipt",5128)});
      g.node("softmax","ForwardSoftmax","HexaFp32",{masked[0],masked[1]},{p[0],p[1],rec("p_receipt",8328)});
    } else for(unsigned j=0;j<2;++j)p[j]=mat("p"+std::to_string(j),QNN_TENSOR_TYPE_APP_WRITE);
    unsigned axis=~0u;
    if(softmax)axis=g.constant("last_axis",{1},{from(3)},QNN_DATATYPE_UINT_32);
    for(unsigned j=0;j<2;++j) {
      auto n="h"+std::to_string(j)+"_";
      if(dvalue) {
        dc[j]=mat(n+"dcontext",QNN_TENSOR_TYPE_APP_WRITE);
        dv[j]=mat(n+"dvalue",QNN_TENSOR_TYPE_APP_READ);
        g.node((n+"dvalue").c_str(),QNN_OP_MAT_MUL,"qti.aisw",{p[j],dc[j]},{dv[j]},
          {boolean(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0,true),boolean(QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1,false)});
      }
      if(softmax) {
        dp[j]=mat(n+"dp",QNN_TENSOR_TYPE_APP_WRITE);
        product[j]=mat(n+"softmax_product",mid);
        dot[j]=mat(n+"softmax_dot",mid,32,1);
        centered[j]=mat(n+"softmax_centered",mid);
        ds[j]=mat(n+"dscores",QNN_TENSOR_TYPE_APP_READ);
        g.node((n+"softmax_product").c_str(),QNN_OP_ELEMENT_WISE_MULTIPLY,"qti.aisw",{dp[j],p[j]},{product[j]});
        Qnn_Param_t axes=QNN_PARAM_INIT;axes.name=QNN_OP_REDUCE_SUM_PARAM_AXES;
        axes.paramType=QNN_PARAMTYPE_TENSOR;axes.tensorParam=g.tensors[axis];
        g.node((n+"softmax_dot").c_str(),QNN_OP_REDUCE_SUM,"qti.aisw",{product[j]},{dot[j]},
          {axes,boolean(QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS,true)});
        g.node((n+"softmax_centered").c_str(),QNN_OP_ELEMENT_WISE_SUBTRACT,"qti.aisw",{dp[j],dot[j]},{centered[j]});
        g.node((n+"dscores").c_str(),QNN_OP_ELEMENT_WISE_MULTIPLY,"qti.aisw",{p[j],centered[j]},{ds[j]});
      }
    }
    auto start=std::chrono::steady_clock::now();g.finalize();
    finalizeUs=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count();
  }
};
} // namespace savedp
