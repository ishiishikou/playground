#include "llama.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
struct Q { std::string text; bool yes; };
struct R { std::string name; double first_ms=0, six_ms=0, hundred_ms=0, dps=0; int correct6=-1; };
static double ms(Clock::time_point t){ return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }

static std::vector<llama_token> tok(const llama_vocab *v,const std::string&s,bool special){
 int32_t n=llama_tokenize(v,s.c_str(),(int32_t)s.size(),nullptr,0,special,true); if(n>=0)return{};
 std::vector<llama_token>x((size_t)-n); n=llama_tokenize(v,s.c_str(),(int32_t)s.size(),x.data(),(int32_t)x.size(),special,true);
 if(n<0)throw std::runtime_error("tokenize failed"); x.resize((size_t)n); return x;
}
static std::string piece(const llama_vocab*v,llama_token t){ char b[256]; int32_t n=llama_token_to_piece(v,t,b,sizeof(b),0,true); return n<0?"<err>":std::string(b,(size_t)n); }
static llama_token one(const llama_vocab*v,const std::vector<std::string>&cs,std::string&sel){ for(auto&s:cs){auto x=tok(v,s,false);if(x.size()==1){sel=s;return x[0];}} throw std::runtime_error("no single-token label"); }
static llama_context* ctx_new(llama_model*m,int th){ auto p=llama_context_default_params();p.n_ctx=4096;p.n_batch=2048;p.n_ubatch=512;p.no_perf=false;auto*c=llama_init_from_model(m,p);if(!c)throw std::runtime_error("context init failed");llama_set_n_threads(c,th,th);return c; }
static void clear(llama_context*c){ llama_memory_clear(llama_get_memory(c),false); }
static void eval(llama_context*c,std::vector<llama_token>&x){auto b=llama_batch_get_one(x.data(),(int32_t)x.size());int rc=llama_decode(c,b);if(rc)throw std::runtime_error("decode failed rc="+std::to_string(rc));}
static std::vector<llama_token> cat(const std::vector<llama_token>&a,const std::vector<llama_token>&b){auto x=a;x.insert(x.end(),b.begin(),b.end());return x;}
static bool direct(llama_context*c,llama_token y,llama_token n){float*l=llama_get_logits_ith(c,-1);if(!l)throw std::runtime_error("missing logits");return l[y]>l[n];}
static llama_token sample1(llama_context*c){auto*s=llama_sampler_chain_init(llama_sampler_chain_default_params());llama_sampler_chain_add(s,llama_sampler_init_greedy());auto t=llama_sampler_sample(s,c,-1);llama_sampler_free(s);return t;}
static void gen(llama_context*c,const llama_vocab*v,int n){auto*s=llama_sampler_chain_init(llama_sampler_chain_default_params());llama_sampler_chain_add(s,llama_sampler_init_greedy());for(int i=0;i<n;i++){auto t=llama_sampler_sample(s,c,-1);if(llama_vocab_is_eog(v,t))break;if(i+1<n){std::vector<llama_token>x{t};eval(c,x);}}llama_sampler_free(s);}
static void emit(const R&r){std::cout<<std::fixed<<std::setprecision(3)<<"{\"type\":\"mode_result\",\"name\":\""<<r.name<<"\",\"first_ms\":"<<r.first_ms<<",\"six_ms\":"<<r.six_ms<<",\"hundred_ms\":"<<r.hundred_ms<<",\"decisions_per_sec_100\":"<<r.dps<<",\"correct_6\":"<<r.correct6<<"}\n"<<std::flush;}

int main(int argc,char**argv){
 std::string path;int th=4,nt=12;for(int i=1;i<argc;i++){if((!strcmp(argv[i],"-m")||!strcmp(argv[i],"--model"))&&i+1<argc)path=argv[++i];else if(!strcmp(argv[i],"--threads")&&i+1<argc)th=atoi(argv[++i]);else if(!strcmp(argv[i],"--normal-tokens")&&i+1<argc)nt=atoi(argv[++i]);}
 if(path.empty()){std::cerr<<"usage: llama-decision-bench -m model.gguf [--threads 4] [--normal-tokens 12]\n";return 2;}
 ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto t0=Clock::now();auto*model=llama_model_load_from_file(path.c_str(),mp);double load=ms(t0);if(!model)return 3;auto*v=llama_model_get_vocab(model);
 std::string yl,nl;auto yt=one(v,{" A","A"," 1","1"},yl);auto ntok=one(v,{" B","B"," 0","0"},nl);
 const std::string prefix="You are a deterministic binary decision engine. Use only the facts and policy below. A means YES and B means NO. Customer profile: age 45; home city Yokohama; membership Silver; payment current; marketing opt-in enabled; no cancellation request; smartphone purchased three months ago; one unresolved support ticket open for fourteen days. Policy: marketing campaign eligibility requires marketing opt-in and current payment status. Retention escalation is required for a cancellation request or an unresolved support ticket older than thirty days. Human review is required for an unresolved support ticket older than seven days. A recent-device-buyer purchased a device within six months. The senior offer requires age sixty-five or older. The Yokohama local event requires home city Yokohama. For binary output use exactly A for YES and B for NO, with no explanation.\n\n";
 const std::vector<Q>qs={{"Is this customer eligible for the marketing campaign?",true},{"Does this customer require retention escalation?",false},{"Does this customer require human review?",true},{"Is this customer a recent-device-buyer?",true},{"Is this customer eligible for the senior offer?",false},{"Is this customer eligible for the Yokohama local event?",true}};
 auto pfx=tok(v,prefix,true);std::vector<std::vector<llama_token>>suffix,full,jfull;for(auto&q:qs){auto b=tok(v,"Question: "+q.text+"\nAnswer:",false);auto j=tok(v,"Question: "+q.text+"\nReturn compact JSON with keys answer and reason; answer must be YES or NO and reason must be very short.\nJSON:",false);suffix.push_back(b);full.push_back(cat(pfx,b));jfull.push_back(cat(pfx,j));}
 std::cerr<<std::fixed<<std::setprecision(3)<<"model_load_ms="<<load<<" prefix_tokens="<<pfx.size()<<" threads="<<th<<" yes_token="<<yt<<" yes_piece="<<piece(v,yt)<<" no_token="<<ntok<<" no_piece="<<piece(v,ntok)<<"\n";
 std::cout<<std::fixed<<std::setprecision(3)<<"{\"type\":\"meta\",\"model_load_ms\":"<<load<<",\"threads\":"<<th<<",\"prefix_tokens\":"<<pfx.size()<<",\"normal_generation_tokens_max\":"<<nt<<"}\n"<<std::flush;
 {auto*c=ctx_new(model,th);auto x=full[0];eval(c,x);(void)direct(c,yt,ntok);llama_free(c);}
 // A
 {R r;r.name="A_normal_json_generation";auto*c=ctx_new(model,th);auto run=[&](int i){clear(c);auto x=jfull[(size_t)i];auto t=Clock::now();eval(c,x);gen(c,v,nt);return ms(t);};r.first_ms=run(0);auto t=Clock::now();for(int i=0;i<6;i++)run(i);r.six_ms=ms(t);t=Clock::now();for(int i=0;i<100;i++)run(i%6);r.hundred_ms=ms(t);r.dps=100000.0/r.hundred_ms;llama_free(c);emit(r);}
 // B
 {R r;r.name="B_one_token_generation";auto*c=ctx_new(model,th);auto run=[&](int i,bool score){clear(c);auto x=full[(size_t)i];auto t=Clock::now();eval(c,x);auto z=sample1(c);if(score){bool py=z==yt,pn=z==ntok;if((py&&qs[(size_t)i].yes)||(pn&&!qs[(size_t)i].yes))r.correct6++;}return ms(t);};r.correct6=0;r.first_ms=run(0,false);auto t=Clock::now();for(int i=0;i<6;i++)run(i,true);r.six_ms=ms(t);t=Clock::now();for(int i=0;i<100;i++)run(i%6,false);r.hundred_ms=ms(t);r.dps=100000.0/r.hundred_ms;llama_free(c);emit(r);}
 // C
 {R r;r.name="C_direct_logits_no_cache";auto*c=ctx_new(model,th);auto run=[&](int i,bool score){clear(c);auto x=full[(size_t)i];auto t=Clock::now();eval(c,x);bool p=direct(c,yt,ntok);if(score&&p==qs[(size_t)i].yes)r.correct6++;return ms(t);};r.correct6=0;r.first_ms=run(0,false);auto t=Clock::now();for(int i=0;i<6;i++)run(i,true);r.six_ms=ms(t);t=Clock::now();for(int i=0;i<100;i++)run(i%6,false);r.hundred_ms=ms(t);r.dps=100000.0/r.hundred_ms;llama_free(c);emit(r);}
 // D: snapshot the complete context after evaluating the shared prefix. This restores recurrent state as well as KV state.
 {R r;r.name="D_direct_logits_shared_prefix_snapshot";auto*c=ctx_new(model,th);clear(c);auto px=pfx;auto tp=Clock::now();eval(c,px);double pms=ms(tp);size_t sz=llama_state_get_size(c);std::vector<uint8_t>state(sz);size_t got=llama_state_get_data(c,state.data(),state.size());if(got==0||got>state.size())throw std::runtime_error("prefix state snapshot failed");state.resize(got);
  auto restore=[&](){size_t used=llama_state_set_data(c,state.data(),state.size());if(used==0)throw std::runtime_error("prefix state restore failed");};
  auto run=[&](int i,bool score){restore();auto x=suffix[(size_t)i];auto t=Clock::now();eval(c,x);bool p=direct(c,yt,ntok);if(score&&p==qs[(size_t)i].yes)r.correct6++;return ms(t);};r.correct6=0;r.first_ms=pms+run(0,false);auto t=Clock::now();for(int i=0;i<6;i++)run(i,true);r.six_ms=ms(t);t=Clock::now();for(int i=0;i<100;i++)run(i%6,false);r.hundred_ms=ms(t);r.dps=100000.0/r.hundred_ms;llama_free(c);emit(r);}
 llama_model_free(model);return 0;
}
