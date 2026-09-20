#include "llama.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using Clock=std::chrono::steady_clock;
struct Q{std::string text;bool yes;};
static double ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static std::vector<llama_token> tok(const llama_vocab*v,const std::string&s,bool special){int32_t n=llama_tokenize(v,s.c_str(),(int32_t)s.size(),nullptr,0,special,true);if(n>=0)return{};std::vector<llama_token>x((size_t)-n);n=llama_tokenize(v,s.c_str(),(int32_t)s.size(),x.data(),(int32_t)x.size(),special,true);if(n<0)throw std::runtime_error("tokenize");x.resize((size_t)n);return x;}
static std::vector<llama_token> cat(std::vector<llama_token>a,const std::vector<llama_token>&b){a.insert(a.end(),b.begin(),b.end());return a;}
static llama_token one(const llama_vocab*v,const std::vector<std::string>&cs){for(auto&s:cs){auto x=tok(v,s,false);if(x.size()==1)return x[0];}throw std::runtime_error("label token");}
static llama_context* ctx(llama_model*m,int th){auto p=llama_context_default_params();p.n_ctx=4096;p.n_batch=2048;p.n_ubatch=512;p.no_perf=false;auto*c=llama_init_from_model(m,p);if(!c)throw std::runtime_error("ctx");llama_set_n_threads(c,th,th);return c;}
static void clear(llama_context*c){llama_memory_clear(llama_get_memory(c),false);}
static void eval(llama_context*c,std::vector<llama_token>&x){auto b=llama_batch_get_one(x.data(),(int32_t)x.size());if(llama_decode(c,b))throw std::runtime_error("decode");}
static bool direct(llama_context*c,llama_token a,llama_token b){auto*l=llama_get_logits_ith(c,-1);return l[a]>l[b];}
static llama_sampler* ab(const llama_vocab*v){auto*s=llama_sampler_chain_init(llama_sampler_chain_default_params());llama_sampler_chain_add(s,llama_sampler_init_grammar(v,"root ::= \" A\" | \" B\"","root"));llama_sampler_chain_add(s,llama_sampler_init_greedy());return s;}
int main(int argc,char**argv){std::string path;int th=4;for(int i=1;i<argc;i++){if((!strcmp(argv[i],"-m")||!strcmp(argv[i],"--model"))&&i+1<argc)path=argv[++i];else if(!strcmp(argv[i],"--threads")&&i+1<argc)th=atoi(argv[++i]);}if(path.empty())return 2;ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*m=llama_model_load_from_file(path.c_str(),mp);if(!m)return 3;auto*v=llama_model_get_vocab(m);auto A=one(v,{" A","A"}),B=one(v,{" B","B"});
const std::string prefix="You are a deterministic binary decision engine. Use only the facts and policy below. A means YES and B means NO. Customer profile: age 45; home city Yokohama; membership Silver; payment current; marketing opt-in enabled; no cancellation request; smartphone purchased three months ago; one unresolved support ticket open for fourteen days. Policy: marketing campaign eligibility requires marketing opt-in and current payment status. Retention escalation is required for a cancellation request or an unresolved support ticket older than thirty days. Human review is required for an unresolved support ticket older than seven days. A recent-device-buyer purchased a device within six months. The senior offer requires age sixty-five or older. The Yokohama local event requires home city Yokohama. For binary output use exactly A for YES and B for NO, with no explanation.\n\n";
const std::vector<Q>qs={{"Is this customer eligible for the marketing campaign?",true},{"Does this customer require retention escalation?",false},{"Does this customer require human review?",true},{"Is this customer a recent-device-buyer?",true},{"Is this customer eligible for the senior offer?",false},{"Is this customer eligible for the Yokohama local event?",true}};auto p=tok(v,prefix,true);std::vector<std::vector<llama_token>>full;for(auto&q:qs)full.push_back(cat(p,tok(v,"Question: "+q.text+"\nAnswer:",false)));
{auto*c=ctx(m,th);auto x=full[0];eval(c,x);(void)direct(c,A,B);llama_free(c);} // identical warm-up
for(int mode=0;mode<2;mode++){auto*c=ctx(m,th);auto*s=mode==0?ab(v):nullptr;int correct=0;double first=0;auto six0=Clock::now();for(int i=0;i<6;i++){clear(c);if(s)llama_sampler_reset(s);auto x=full[i];auto t=Clock::now();eval(c,x);bool pred;if(s){auto z=llama_sampler_sample(s,c,-1);pred=z==A;}else pred=direct(c,A,B);double d=ms(t);if(i==0)first=d;if(pred==qs[i].yes)correct++;}double six=ms(six0);auto h0=Clock::now();for(int i=0;i<100;i++){clear(c);if(s)llama_sampler_reset(s);auto x=full[i%6];eval(c,x);if(s)(void)llama_sampler_sample(s,c,-1);else(void)direct(c,A,B);}double hundred=ms(h0);std::cout<<std::fixed<<std::setprecision(3)<<"{\"mode\":\""<<(mode==0?"B_constrained_one_token_generation":"C_direct_logits_no_cache")<<"\",\"correct_6\":"<<correct<<",\"first_ms\":"<<first<<",\"six_ms\":"<<six<<",\"hundred_ms\":"<<hundred<<",\"decisions_per_sec_100\":"<<(100000.0/hundred)<<"}\n";if(s)llama_sampler_free(s);llama_free(c);}llama_model_free(m);}
