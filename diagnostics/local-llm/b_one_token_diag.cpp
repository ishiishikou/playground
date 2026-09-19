#include "llama.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Q { std::string text; bool yes; };

static std::vector<llama_token> tok(const llama_vocab *v, const std::string &s, bool special) {
    int32_t n = llama_tokenize(v, s.c_str(), (int32_t) s.size(), nullptr, 0, special, true);
    if (n >= 0) return {};
    std::vector<llama_token> x((size_t) -n);
    n = llama_tokenize(v, s.c_str(), (int32_t) s.size(), x.data(), (int32_t) x.size(), special, true);
    if (n < 0) throw std::runtime_error("tokenize failed");
    x.resize((size_t) n);
    return x;
}

static std::string piece(const llama_vocab *v, llama_token t) {
    char b[512];
    int32_t n = llama_token_to_piece(v, t, b, sizeof(b), 0, true);
    return n < 0 ? std::string("<err>") : std::string(b, (size_t) n);
}

static std::string hex_bytes(const std::string &s) {
    std::ostringstream os;
    os << std::hex << std::setfill('0');
    for (unsigned char c : s) os << std::setw(2) << (unsigned int) c;
    return os.str();
}

static llama_token one(const llama_vocab *v, const std::vector<std::string> &candidates, std::string &selected) {
    for (const auto &s : candidates) {
        auto x = tok(v, s, false);
        if (x.size() == 1) {
            selected = s;
            return x[0];
        }
    }
    throw std::runtime_error("no single-token label");
}

static llama_context * ctx_new(llama_model *m, int threads) {
    auto p = llama_context_default_params();
    p.n_ctx = 4096;
    p.n_batch = 2048;
    p.n_ubatch = 512;
    p.no_perf = false;
    auto *c = llama_init_from_model(m, p);
    if (!c) throw std::runtime_error("context init failed");
    llama_set_n_threads(c, threads, threads);
    return c;
}

static void clear_ctx(llama_context *c) {
    llama_memory_clear(llama_get_memory(c), false);
}

static void eval(llama_context *c, std::vector<llama_token> &x) {
    auto b = llama_batch_get_one(x.data(), (int32_t) x.size());
    int rc = llama_decode(c, b);
    if (rc) throw std::runtime_error("decode failed rc=" + std::to_string(rc));
}

static llama_token sample1(llama_context *c) {
    auto *s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(s, llama_sampler_init_greedy());
    auto t = llama_sampler_sample(s, c, -1);
    llama_sampler_free(s);
    return t;
}

static std::string join_ids(const std::vector<llama_token> &xs) {
    std::ostringstream os;
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) os << ",";
        os << xs[i];
    }
    return os.str();
}

int main(int argc, char **argv) {
    std::string path;
    int threads = 4;
    int trace_tokens = 16;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--trace-tokens") && i + 1 < argc) trace_tokens = atoi(argv[++i]);
    }
    if (path.empty()) {
        std::cerr << "usage: llama-b-one-token-diag -m model.gguf [--threads 4] [--trace-tokens 16]\n";
        return 2;
    }

    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) return 3;
    auto *v = llama_model_get_vocab(model);

    std::string yl, nl;
    auto yt = one(v, {" A", "A", " 1", "1"}, yl);
    auto ntok = one(v, {" B", "B", " 0", "0"}, nl);

    const std::string prefix =
        "You are a deterministic binary decision engine. Use only the facts and policy below. "
        "A means YES and B means NO. Customer profile: age 45; home city Yokohama; membership Silver; "
        "payment current; marketing opt-in enabled; no cancellation request; smartphone purchased three months ago; "
        "one unresolved support ticket open for fourteen days. Policy: marketing campaign eligibility requires "
        "marketing opt-in and current payment status. Retention escalation is required for a cancellation request or "
        "an unresolved support ticket older than thirty days. Human review is required for an unresolved support "
        "ticket older than seven days. A recent-device-buyer purchased a device within six months. The senior offer "
        "requires age sixty-five or older. The Yokohama local event requires home city Yokohama. "
        "For binary output use exactly A for YES and B for NO, with no explanation.\n\n";

    const std::vector<Q> qs = {
        {"Is this customer eligible for the marketing campaign?", true},
        {"Does this customer require retention escalation?", false},
        {"Does this customer require human review?", true},
        {"Is this customer a recent-device-buyer?", true},
        {"Is this customer eligible for the senior offer?", false},
        {"Is this customer eligible for the Yokohama local event?", true},
    };

    std::cout << "{\"type\":\"meta\",\"yes_token\":" << yt
              << ",\"yes_piece_hex\":\"" << hex_bytes(piece(v, yt))
              << "\",\"no_token\":" << ntok
              << ",\"no_piece_hex\":\"" << hex_bytes(piece(v, ntok))
              << "\",\"threads\":" << threads
              << ",\"trace_tokens\":" << trace_tokens << "}\n";

    int exact_correct = 0;
    int traced_semantic_correct = 0;

    for (size_t i = 0; i < qs.size(); ++i) {
        auto *c = ctx_new(model, threads);
        clear_ctx(c);
        auto x = tok(v, prefix + "Question: " + qs[i].text + "\nAnswer:", true);
        eval(c, x);

        std::vector<llama_token> trace;
        std::vector<std::string> pieces;
        int first_ab_pos = -1;
        bool first_ab_yes = false;

        for (int j = 0; j < trace_tokens; ++j) {
            llama_token z = sample1(c);
            trace.push_back(z);
            pieces.push_back(piece(v, z));

            if (first_ab_pos < 0 && (z == yt || z == ntok)) {
                first_ab_pos = j;
                first_ab_yes = (z == yt);
            }

            if (llama_vocab_is_eog(v, z)) break;
            std::vector<llama_token> one_tok{z};
            eval(c, one_tok);
        }

        bool first_exact = !trace.empty() &&
            ((trace[0] == yt && qs[i].yes) || (trace[0] == ntok && !qs[i].yes));
        bool traced_correct = first_ab_pos >= 0 && first_ab_yes == qs[i].yes;
        if (first_exact) exact_correct++;
        if (traced_correct) traced_semantic_correct++;

        std::ostringstream ph;
        for (size_t j = 0; j < pieces.size(); ++j) {
            if (j) ph << ",";
            ph << hex_bytes(pieces[j]);
        }

        std::cout << "{\"type\":\"question\",\"index\":" << i
                  << ",\"expected\":\"" << (qs[i].yes ? "A" : "B")
                  << "\",\"first_token\":" << (trace.empty() ? -1 : trace[0])
                  << ",\"first_piece_hex\":\"" << (trace.empty() ? "" : hex_bytes(pieces[0]))
                  << "\",\"first_exact_correct\":" << (first_exact ? "true" : "false")
                  << ",\"first_ab_pos\":" << first_ab_pos
                  << ",\"first_ab_choice\":\"" << (first_ab_pos < 0 ? "NONE" : (first_ab_yes ? "A" : "B"))
                  << "\",\"traced_semantic_correct\":" << (traced_correct ? "true" : "false")
                  << ",\"trace_token_ids\":\"" << join_ids(trace)
                  << "\",\"trace_piece_hex\":\"" << ph.str() << "\"}\n";

        llama_free(c);
    }

    std::cout << "{\"type\":\"summary\",\"exact_first_token_correct\":" << exact_correct
              << ",\"traced_first_ab_correct\":" << traced_semantic_correct
              << ",\"questions\":" << qs.size() << "}\n";

    llama_model_free(model);
    return 0;
}
