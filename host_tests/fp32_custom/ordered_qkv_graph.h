#pragma once
namespace orderedqkvstudy {
struct Graph {
    unsigned stage;bool hvx,observed,tap;std::string name;DiagnosticGraph g;
    unsigned x,y,ga=0,be=0,dy=0,hat=0,iv=0,fr=0,dx=0,dg=0,db=0,br=0,qr=0,hr=0;
    std::array<unsigned,3>w,q;std::array<unsigned,6>heads{};
    unsigned lastQ=0,lastF=0,lastB=0,lastH=0;
    Graph(std::string n,unsigned s,bool h,bool o,bool t=false,bool defer=false):stage(s),hvx(h),observed(o),tap(t),name(n),g(n){
        g.receiptEnabled=o;auto native=t?QNN_TENSOR_TYPE_APP_READ:QNN_TENSOR_TYPE_NATIVE;
        x=g.tensor("x",QNN_TENSOR_TYPE_APP_WRITE,{1,1,32,64});y=x;
        if(stage){
            ga=g.tensor("ga",QNN_TENSOR_TYPE_APP_WRITE,{1,1,1,64});be=g.tensor("be",QNN_TENSOR_TYPE_APP_WRITE,{1,1,1,64});dy=g.tensor("dy",QNN_TENSOR_TYPE_APP_WRITE,{1,1,32,64});
            y=g.tensor("y",native,{1,1,32,64});hat=g.tensor("hat",native,{1,1,32,64});iv=g.tensor("iv",native,{1,1,32,1});
            fr=g.tensor("fr",QNN_TENSOR_TYPE_APP_READ,{1,1,1,4265},QNN_DATATYPE_INT_32);auto eps=g.constant("eps",{1,1,1,1},{1e-5f});
            g.node("ln1","LnForward","HexaFp32",{x,ga,be,eps},{y,hat,iv,fr});
            dx=g.tensor("dx",QNN_TENSOR_TYPE_APP_READ,{1,1,32,64});dg=g.tensor("dg",QNN_TENSOR_TYPE_APP_READ,{1,1,1,64});db=g.tensor("db",QNN_TENSOR_TYPE_APP_READ,{1,1,1,64});
            br=g.tensor("br",QNN_TENSOR_TYPE_APP_READ,{1,1,1,6248},QNN_DATATYPE_INT_32);
        }
        for(unsigned p=0;p<3;++p){w[p]=g.tensor(("w"+std::to_string(p)).c_str(),QNN_TENSOR_TYPE_APP_WRITE,{1,1,64,64});q[p]=g.tensor(("q"+std::to_string(p)).c_str(),stage==2?native:QNN_TENSOR_TYPE_APP_READ,{1,1,32,64});}
        qr=g.tensor("qr",QNN_TENSOR_TYPE_APP_READ,{1,1,1,h?20504u:20488u},QNN_DATATYPE_INT_32);
        g.node("projection",h?"ProjectionQKVHvx":"ProjectionQKV","HexaFp32",{y,w[0],w[1],w[2]},{q[0],q[1],q[2],qr});
        if(stage==2){for(unsigned p=0;p<6;++p)heads[p]=g.tensor(("head"+std::to_string(p)).c_str(),defer?native:QNN_TENSOR_TYPE_APP_READ,{1,1,32,32});
            hr=g.tensor("hr",QNN_TENSOR_TYPE_APP_READ,{1,1,1,12296},QNN_DATATYPE_INT_32);
            g.node("heads","QKVHeads","HexaFp32",{q[0],q[1],q[2]},{heads[0],heads[1],heads[2],heads[3],heads[4],heads[5],hr});}
        if(stage)g.node("local_backward","LnBackward","HexaFp32",{dy,ga,hat,iv},{dx,dg,db,br});
        auto started=std::chrono::steady_clock::now();if(!defer)g.finalize();
        auto elapsed=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-started).count();
        std::printf("qkvgraph,%s,stage=%u,hvx=%u,observed=%u,tap=%u,finalize_wrapper_us=%.3f\n",name.c_str(),s,h,o,t,elapsed);
    }
};
}
