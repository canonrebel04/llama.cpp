// fork: CPU sanity test for the ik_llama.cpp row-scaled types (row metadata before the blocks):
// mul_mat / get_rows / cpy vs per-row dequantization.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
int main(){
  ggml_type ts[]={GGML_TYPE_IQ4_KS,GGML_TYPE_IQ2_KS,GGML_TYPE_IQ4_KSS,GGML_TYPE_IQ5_KS,GGML_TYPE_IQ3_KS,GGML_TYPE_IQ2_KL};
  const int K=512,M=5,N=3; int bad=0;
  for(auto t:ts){
    ggml_init_params ip={256*1024*1024,nullptr,false}; auto ctx=ggml_init(ip);
    auto a=ggml_new_tensor_2d(ctx,t,K,M); auto b=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,N);
    std::mt19937 rng(1); std::normal_distribution<float> nd;
    std::vector<float> af(K*M); for(auto&x:af)x=nd(rng);
    for(int i=0;i<K*N;i++)((float*)b->data)[i]=nd(rng);
    ggml_quantize_chunk(t,af.data(),a->data,0,M,K,nullptr);
    // get_rows + cpy to f32 paths
    auto ids=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,2); ((int*)ids->data)[0]=3;((int*)ids->data)[1]=1;
    auto mm=ggml_mul_mat(ctx,a,b); auto gr=ggml_get_rows(ctx,a,ids); auto cp=ggml_cpy(ctx,a,ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,M));
    auto gf=ggml_new_graph(ctx); ggml_build_forward_expand(gf,mm);ggml_build_forward_expand(gf,gr);ggml_build_forward_expand(gf,cp);
    ggml_graph_compute_with_ctx(ctx,gf,4);
    std::vector<float> deq(K*M); auto tr=ggml_get_type_traits(t);
    for(int r=0;r<M;r++) tr->to_float((char*)a->data+r*a->nb[1],deq.data()+r*K,K);
    double e=0,n=0,eg=0,ec=0;
    for(int j=0;j<N;j++)for(int i=0;i<M;i++){double s=0;for(int k=0;k<K;k++)s+=deq[i*K+k]*((float*)b->data)[j*K+k];double g=((float*)mm->data)[j*M+i];e+=(s-g)*(s-g);n+=s*s;}
    for(int k=0;k<K;k++){eg+=fabs(((float*)gr->data)[k]-deq[3*K+k])+fabs(((float*)gr->data)[K+k]-deq[1*K+k]);}
    for(int k=0;k<K*M;k++)ec+=fabs(((float*)cp->data)[k]-deq[k]);
    double qe=0,qn=0; for(int k=0;k<K*M;k++){qe+=(deq[k]-af[k])*(deq[k]-af[k]);qn+=af[k]*af[k];}
    printf("%-8s nb1=%zu mm_rel=%.4g getrows=%g cpy=%g quant_rmse_rel=%.4f\n",ggml_type_name(t),a->nb[1],sqrt(e/n),eg,ec,sqrt(qe/qn));
    if(sqrt(e/n)>0.05||eg>0||ec>0)bad++;
    ggml_free(ctx);
  }
  return bad;
}
