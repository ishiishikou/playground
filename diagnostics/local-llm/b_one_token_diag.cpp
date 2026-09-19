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

static double elapsed_ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static std::vector<llama_token> tok(const llama_vocab *v, const std::string &s, bool special) {
    int32_t n = llama_tokenize(v, s.c_str(), (int32_t) s.size(), nullptr, 0, special, true);
    if (n >= 0) return {};
    std::vector<llama_token> x((size_t) -n);
    n = llama_tokenize(v, s.c_str(), (int32_t) s.size(), x.data(), (int32_t) x.size(), special, true);
    if (n < 0) throw std::runtime_error("tokenize failed");
    x.resize((size_t) n);
    return x;
}

static std::vector<llama_token> cat(const std::vector<llama_token> &a, const std::vector<llama_token> &b) {
    auto x = a;
    x.insert(x.end(), b.begin(), b.end());
    return x;
}

static std::string piece(const llama_vocab *v, llama_token t) {
    char b[256];
    int32_t n = llama_token_to_piece(v, t, b, sizeof(b), 0, true);
    return n < 0 ? std::string("<err>") : std::string(b, (size_t) n);
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

static llama_context *ctx_new(llama_model *m, int threads) {
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

static bool direct(llama_context *c, llama_token yes_tok, llama_token no_tok) {
    float *l = llama_get_logits_ith(c, -1);
    if (!l) throw std::runtime_error("missing logits");
    return l[yes_tok] > l[no_tok];
}

static llama_sampler *new_ab_sampler(const llama_vocab *v) {
    auto *s = llama_sampler_chain_init(llama_sampler_chain_default_params());
    auto *g = llama_sampler_init_grammar(v, "root ::= \" A\" | \" B\"", "root");
    if (!g) throw std::runtime_error("grammar init failed");
    llama_sampler_chain_add(s, g);
    llama_sampler_chain_add(s, llama_sampler_init_greedy());
    return s;
}

int main(int argc, char **argv) {
    std::string path;
    int threads = 4;
    for (int i = 1; i < argc; ++i) {
        if ((!strcmp(argv[i], "-m") || !strcmp(argv[i], "--model")) && i + 1 < argc) path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    }
    if (path.empty()) {
        std::cerr << "usage: llama-b-one-token-diag -m model.gguf [--threads 4]\n";
        return 2;
    }

    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    auto *model = llama_model_load_from_file(path.c_str(), mp);
    if (!model) return 3;
    auto *v = llama_model_get_vocab(model);

    std::string yes_label, no_label;
    auto yes_tok = one(v, {" A", "A", " 1", "1"}, yes_label);
    auto no_tok  = one(v, {" B", "B", " 0", "0"}, no_label);

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

    auto pfx = tok(v, prefix, true);
    std::vector<std::vector<llama_token>> full;
    for (const auto &q : qs) {
        auto suffix = tok(v, "Question: " + q.text + "\nAnswer:", false);
        full.push_back(cat(pfx, suffix));
    }

    std::cout << "{\"type\":\"meta\",\"threads\":" << threads
              << ",\"yes_token\":" << yes_tok
              << ",\"yes_piece\":\"" << piece(v, yes_tok)
              << "\",\"no_token\":" << no_tok
              << ",\"no_piece\":\"" << piece(v, no_tok)
              << "\",\"grammar\":\"A_or_B_single_token\"}\n" << std::flush;

    // Warm-up exactly the same model/prompt evaluation path.
    {
        auto *c = ctx_new(model, threads);
        auto x = full[0];
        eval(c, x);
        (void) direct(c, yes_tok, no_tok);
        llama_free(c);
    }

    // B-fixed: constrained one-token generation using llama.cpp sampler chain + GBNF grammar.
    {
        auto *c = ctx_new(model, threads);
        auto *sampler = new_ab_sampler(v);
        int correct = 0;
        double first_ms = 0.0;
        auto six_start = Clock::now();

        for (size_t i = 0; i < qs.size(); ++i) {
            clear_ctx(c);
            llama_sampler_reset(sampler);
            auto x = full[i];
            auto t = Clock::now();
            eval(c, x);
            llama_token z = llama_sampler_sample(sampler, c, -1);
            double one_ms = elapsed_ms(t);
            if (i == 0) first_ms = one_ms;

            bool predicted_yes = z == yes_tok;
            bool recognized = z == yes_tok || z == no_tok;
            bool ok = recognized && predicted_yes == qs[i].yes;
            if (ok) correct++;

            std::cout << "{\"type\":\"question\",\"mode\":\"B_constrained_one_token_generation\",\"index\":" << i
                      << ",\"expected\":\"" << (qs[i].yes ? "A" : "B")
                      << "\",\"token\":" << z
                      << ",\"piece\":\"" << piece(v, z)
                      << "\",\"correct\":" << (ok ? "true" : "false")
                      << ",\"latency_ms\":" << std::fixed << std::setprecision(3) << one_ms << "}\n" << std::flush;
        }

        double six_ms = elapsed_ms(six_start);
        std::cout << "{\"type\":\"summary\",\"mode\":\"B_constrained_one_token_generation\",\"correct_6\":" << correct
                  << ",\"first_ms\":" << std::fixed << std::setprecision(3) << first_ms
                  << ",\"six_ms\":" << six_ms << "}\n" << std::flush;
        llama_sampler_free(sampler);
        llama_free(c);
    }

    // C reference: direct comparison of the exact same A/B logits on the exact same prompts.
    {
        auto *c = ctx_new(model, threads);
        int correct = 0;
        double first_ms = 0.0;
        auto six_start = Clock::now();

        for (size_t i = 0; i < qs.size(); ++i) {
            clear_ctx(c);
            auto x = full[i];
            auto t = Clock::now();
            eval(c, x);
            bool predicted_yes = direct(c, yes_tok, no_tok);
            double one_ms = elapsed_ms(t);
            if (i == 0) first_ms = one_ms;
            bool ok = predicted_yes == qs[i].yes;
            if (ok) correct++;

            std::cout << "{\"type\":\"question\",\"mode\":\"C_direct_logits_no_cache\",\"index\":" << i
                      << ",\"expected\":\"" << (qs[i].yes ? "A" : "B")
                      << "\",\"choice\":\"" << (predicted_yes ? "A" : "B")
                      << "\",\"correct\":" << (ok ? "true" : "false")
                      << ",\"latency_ms\":" << std::fixed << std::setprecision(3) << one_ms << "}\n" << std::flush;
        }

        double six_ms = elapsed_ms(six_start);
        std::cout << "{\"type\":\"summary\",\"mode\":\"C_direct_logits_no_cache\",\"correct_6\":" << correct
                  << ",\"first_ms\":" << std::fixed << std::setprecision(3) << first_ms
                  << ",\"six_ms\":" << six_ms << "}\n" << std::flush;
        llama_free(c);
    }

    llama_model_free(model);
    return 0;
}
