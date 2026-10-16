#pragma once
// Fixed FFN2 diagnostic graph only; existing Add/LN arithmetic is reused.
namespace orderedhvxstudy {
struct Graph {
    unsigned stage;bool hvx,observed,tap;std::string name;DiagnosticGraph g;
    unsigned a,b,mid,mr,res=0,x,ar=0,ga=0,be=0,dy=0,y=0,hat=0,iv=0,fr=0,dx=0,dg=0,db=0,br=0;
    unsigned lastM=0,lastA=0,lastF=0,lastB=0;double finalizeUs=0;
    Graph(std::string n,unsigned s,bool h,bool o,bool t=false):stage(s),hvx(h),observed(o),tap(t),name(n),g(n){
        g.receiptEnabled=observed;
        a=g.tensor("a",QNN_TENSOR_TYPE_APP_WRITE,{1,1,32,128});
        b=g.tensor("b",QNN_TENSOR_TYPE_APP_WRITE,{1,1,128,64});
        auto native=tap?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE;
        mid=g.tensor("mid",stage?native:QNN_TENSOR_TYPE_APP_READ,{1,1,32,64});x=mid;
        mr=g.tensor("mr",QNN_TENSOR_TYPE_APP_READ,{1,1,1,hvx?14352u:14344u},QNN_DATATYPE_INT_32);
        g.node("ffn",hvx?"FFN2Hvx":"FFN2","HexaFp32",{a,b},{mid,mr});
        if(stage){
            res=g.tensor("res",QNN_TENSOR_TYPE_APP_WRITE,{1,1,32,64});
            x=g.tensor("x",stage==1?QNN_TENSOR_TYPE_APP_READ:native,{1,1,32,64});
            ar=g.tensor("ar",QNN_TENSOR_TYPE_APP_READ,{1,1,1,8200},QNN_DATATYPE_INT_32);
            g.node("add","Add","HexaFp32",{mid,res},{x,ar});
        }
        if(stage==2){
            ga=g.tensor("ga",QNN_TENSOR_TYPE_APP_WRITE,{1,1,1,64});
            be=g.tensor("be",QNN_TENSOR_TYPE_APP_WRITE,{1,1,1,64});
            dy=g.tensor("dy",QNN_TENSOR_TYPE_APP_WRITE,{1,1,32,64});
            y=g.tensor("y",QNN_TENSOR_TYPE_APP_READ,{1,1,32,64});
            hat=g.tensor("hat",native,{1,1,32,64});iv=g.tensor("iv",native,{1,1,32,1});
            fr=g.tensor("fr",QNN_TENSOR_TYPE_APP_READ,{1,1,1,4265},QNN_DATATYPE_INT_32);
            auto eps=g.constant("eps",{1,1,1,1},{1e-5f});
            g.node("ln","LnForward","HexaFp32",{x,ga,be,eps},{y,hat,iv,fr});
            dx=g.tensor("dx",QNN_TENSOR_TYPE_APP_READ,{1,1,32,64});
            dg=g.tensor("dg",QNN_TENSOR_TYPE_APP_READ,{1,1,1,64});
            db=g.tensor("db",QNN_TENSOR_TYPE_APP_READ,{1,1,1,64});
            br=g.tensor("br",QNN_TENSOR_TYPE_APP_READ,{1,1,1,6248},QNN_DATATYPE_INT_32);
            g.node("back","LnBackward","HexaFp32",{dy,ga,hat,iv},{dx,dg,db,br});
        }
        auto started=std::chrono::steady_clock::now();g.finalize();
        finalizeUs=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
        std::printf("hvxgraph,%s,stage=%u,hvx=%u,observed=%u,tap=%u,finalize_wrapper_us=%.3f\n",name.c_str(),stage,hvx,observed,tap,finalizeUs);
    }
    template<class C> std::map<unsigned,one::FV> inputs(const C& c)const{
        std::map<unsigned,one::FV> in={{a,c.a},{b,c.b}};if(stage)in[res]=c.res;
        if(stage==2){in[ga]=c.z.v.at("gamma1");in[be]=c.z.v.at("beta1");in[dy]=c.z.v.at("dy");}return in;
    }
    std::map<std::string,one::FV> finals(const std::map<unsigned,std::vector<uint32_t>>& o)const{
        if(!stage)return {{"ffn2",decode(o.at(mid))}};
        if(stage==1)return {{"x",decode(o.at(x))}};
        return {{"y",decode(o.at(y))},{"dx",decode(o.at(dx))},{"dg",decode(o.at(dg))},{"db",decode(o.at(db))}};
    }
};
}
