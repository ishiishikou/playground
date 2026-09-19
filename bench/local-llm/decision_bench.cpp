#include "llama.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Q { std::string text; bool yes; };
struct R {
    std::string name;
    double first_ms = 0;
    double six_ms = 0;
    double hundred_ms = 0;
    double dps = 0;
    int correct6 = -1;
};

static double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::vector<llama_token> tok(const llama_vocab * v, const std::string & s, bool special) {
    int32_t n = llama_tokenize(v, s.c_str(), (int32_t) s.size(), nullptr, 0, special, true);
    if (n >= 0) return {};
    std::vector<llama_token> out((size_t) -n);
    n = llama_tokenize(v, s.c_str(), (int32_t) s.size(), out.data(), (int32_t) out.size(), special, true);
    if (n < 0) throw std::runtime_error("tokenize failed");
    out.resize((size_t) n);
    return out;
}

static std::string token_piece(const llama_vocab * v, llama_token t) {
    char b[256];
    int32_t n = llama_token_to_piece(v, t, b, sizeof(b), 0, true);
    if (n < 0) return "<piece-error>";
    return std::string(b, (size_t) n);
}

static llama_token one_token(const llama_vocab * v, const std::vector<std::string> & choices, std::string & selected) {
    for (const auto & s : choices) {
        auto x = tok(v, s, false);
        if (x.size() == 1) {
            selected = s;
            return x[0];
        }
    }
    throw std::runtime_error("no single-token label");
}

static llama_context * new_ctx(llama_model * m, int threads) {
    auto p = llama_context_default_params();
    p.n_ctx = 4096;
    p.n_batch = 2048;
    p.n_ubatch = 512;
    p.no_perf = false;
    auto * c = llama_init_from_model(m, p);
    if (!c) throw std::runtime_error("context init failed");
    llama_set_n_threads(c, threads, threads);
    return c;
}

static void clear_ctx(llama_context * c) {
    llama_memory_clear(llama_get_memory(c), false);
}

static void eval(llama_context * c, std::vector<llama_token> & x) {
    auto b = llama_batch_get_one(x.data(), (int32_t) x.size());
    int32_t rc = llama_decode(c, b);
    if (rc != 0) throw std::runtime_error("decode failed rc=" + std::to_string(rc));
}

static std::vector<llama_token> cat(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    std::vector<llama_token> x;
    x.reserve(a.size() + b.size());
    x.insert(x.end(), a.begin(), a.end());
    x.insert(x.end(), b.begin(), b.end());
    return x;
}

static bool direct(llama_context * c, llama_token yes, llama_token no) {
    float * l = llama_get_logits_ith(c, -1);
    if (!l) throw std::runtime_error("missing logits");
    return l[yes] > l[no];
}

static llama_token sample_one(llama_context * c) {
    auto sp = llama_sampler_chain_default_params();
    auto * s = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(s, llama_sampler_init_greedy());
    llama_token out = llama_sampler_sample(s, c, -1);
    llama_sampler_free(s);
    return out;
}

static void generate_n(llama_context * c, const llama_vocab * v, int n) {
    auto sp = llama_sampler_chain_default_params();
    auto * s = llama_sampler_chain_init(sp);
    llama_sampler_chain_add(s, llama_sampler_init_greedy());
    for (int i = 0; i < n; ++i) {
        llama_token t = llama_sampler_sample(s, c, -1);
        if (llama_vocab_is_eog(v, t)) break;
        if (i + 1 < n) {
            std::vector<llama_token> x{t};
            eval(c, x);
        }
    }
    llama_sampler_free(s);
}

int main(int argc, char ** argv) {
    std::string model_path;
    int threads = 4;
    int normal_tokens = 12;
    for (int i = 1; i < argc; ++i) {
        if ((!std::strcmp(argv[i], "-m") || !std::strcmp(argv[i], "--model")) && i + 1 < argc) model_path = argv[++i];
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--normal-tokens") && i + 1 < argc) normal_tokens = std::atoi(argv[++i]);
    }
    if (model_path.empty()) {
        std::cerr << "usage: llama-decision-bench -m model.gguf [--threads 4] [--normal-tokens 12]\n";
        return 2;
    }

    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;

    auto load0 = Clock::now();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    double load_ms = elapsed_ms(load0);
    if (!model) return 3;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string yes_label, no_label;
    llama_token yes_token = one_token(vocab, {" A", "A", " 1", "1"}, yes_label);
    llama_token no_token  = one_token(vocab, {" B", "B", " 0", "0"}, no_label);

    const std::string prefix =
        "You are a deterministic binary decision engine. Use only the facts and policy below. "
        "A means YES and B means NO. Customer profile: age 45; home city Yokohama; membership Silver; "
        "payment current; marketing opt-in enabled; no cancellation request; smartphone purchased three months ago; "
        "one unresolved support ticket open for fourteen days. "
        "Policy: marketing campaign eligibility requires marketing opt-in and current payment status. "
        "Retention escalation is required for a cancellation request or an unresolved support ticket older than thirty days. "
        "Human review is required for an unresolved support ticket older than seven days. "
        "A recent-device-buyer purchased a device within six months. "
        "The senior offer requires age sixty-five or older. "
        "The Yokohama local event requires home city Yokohama. "
        "For binary output use exactly A for YES and B for NO, with no explanation.\n\n";

    const std::vector<Q> qs = {
        {"Is this customer eligible for the marketing campaign?", true},
        {"Does this customer require retention escalation?", false},
        {"Does this customer require human review?", true},
        {"Is this customer a recent-device-buyer?", true},
        {"Is this customer eligible for the senior offer?", false},
        {"Is this customer eligible for the Yokohama local event?", true}
    };

    auto pfx = tok(vocab, prefix, true);
    std::vector<std::vector<llama_token>> bin_suffix, bin_full, json_full;
    for (const auto & q : qs) {
        std::string b = "Question: " + q.text + "\nAnswer:";
        std::string j = "Question: " + q.text +
            "\nReturn compact JSON with keys answer and reason; answer must be YES or NO and reason must be very short.\nJSON:";
        auto bt = tok(vocab, b, false);
        auto jt = tok(vocab, j, false);
        bin_suffix.push_back(bt);
        bin_full.push_back(cat(pfx, bt));
        json_full.push_back(cat(pfx, jt));
    }

    std::cerr << std::fixed << std::setprecision(3)
              << "model_load_ms=" << load_ms
              << " prefix_tokens=" << pfx.size()
              << " threads=" << threads
              << " yes_token=" << yes_token << " yes_piece=" << token_piece(vocab, yes_token)
              << " no_token=" << no_token << " no_piece=" << token_piece(vocab, no_token) << "\n";

    // Warm once so the mode comparison is not dominated by first-touch effects.
    {
        auto * c = new_ctx(model, threads);
        auto x = bin_full[0];
        eval(c, x);
        (void) direct(c, yes_token, no_token);
        llama_free(c);
    }

    std::vector<R> out;

    // A: normal short generation
    {
        R r; r.name = "A_normal_json_generation";
        auto * c = new_ctx(model, threads);
        auto run = [&](int qi) {
            clear_ctx(c);
            auto x = json_full[(size_t) qi];
            auto t0 = Clock::now();
            eval(c, x);
            generate_n(c, vocab, normal_tokens);
            return elapsed_ms(t0);
        };
        r.first_ms = run(0);
        auto t6 = Clock::now(); for (int i = 0; i < 6; ++i) run(i); r.six_ms = elapsed_ms(t6);
        auto t100 = Clock::now(); for (int i = 0; i < 100; ++i) run(i % 6); r.hundred_ms = elapsed_ms(t100);
        r.dps = 100000.0 / r.hundred_ms;
        llama_free(c);
        out.push_back(r);
    }

    // B: one generated token
    {
        R r; r.name = "B_one_token_generation";
        auto * c = new_ctx(model, threads);
        auto run = [&](int qi, bool score) {
            clear_ctx(c);
            auto x = bin_full[(size_t) qi];
            auto t0 = Clock::now();
            eval(c, x);
            llama_token t = sample_one(c);
            if (score) {
                bool pred_yes = t == yes_token;
                bool pred_no  = t == no_token;
                if ((pred_yes && qs[(size_t) qi].yes) || (pred_no && !qs[(size_t) qi].yes)) r.correct6++;
            }
            return elapsed_ms(t0);
        };
        r.correct6 = 0;
        r.first_ms = run(0, false);
        auto t6 = Clock::now(); for (int i = 0; i < 6; ++i) run(i, true); r.six_ms = elapsed_ms(t6);
        auto t100 = Clock::now(); for (int i = 0; i < 100; ++i) run(i % 6, false); r.hundred_ms = elapsed_ms(t100);
        r.dps = 100000.0 / r.hundred_ms;
        llama_free(c);
        out.push_back(r);
    }

    // C: compare A/B logits directly, re-evaluating the full prompt every time
    {
        R r; r.name = "C_direct_logits_no_cache";
        auto * c = new_ctx(model, threads);
        auto run = [&](int qi, bool score) {
            clear_ctx(c);
            auto x = bin_full[(size_t) qi];
            auto t0 = Clock::now();
            eval(c, x);
            bool pred = direct(c, yes_token, no_token);
            if (score && pred == qs[(size_t) qi].yes) r.correct6++;
            return elapsed_ms(t0);
        };
        r.correct6 = 0;
        r.first_ms = run(0, false);
        auto t6 = Clock::now(); for (int i = 0; i < 6; ++i) run(i, true); r.six_ms = elapsed_ms(t6);
        auto t100 = Clock::now(); for (int i = 0; i < 100; ++i) run(i % 6, false); r.hundred_ms = elapsed_ms(t100);
        r.dps = 100000.0 / r.hundred_ms;
        llama_free(c);
        out.push_back(r);
    }

    // D: evaluate shared prefix once; after each question, remove only its suffix from KV memory
    {
        R r; r.name = "D_direct_logits_shared_prefix_cache";
        auto * c = new_ctx(model, threads);
        clear_ctx(c);
        auto px = pfx;
        auto tp = Clock::now();
        eval(c, px);
        double prefix_ms = elapsed_ms(tp);
        llama_pos prefix_len = (llama_pos) pfx.size();

        auto run = [&](int qi, bool score) {
            auto sx = bin_suffix[(size_t) qi];
            auto t0 = Clock::now();
            eval(c, sx);
            bool pred = direct(c, yes_token, no_token);
            double ms = elapsed_ms(t0);
            if (score && pred == qs[(size_t) qi].yes) r.correct6++;
            if (!llama_memory_seq_rm(llama_get_memory(c), 0, prefix_len, -1)) {
                throw std::runtime_error("KV rollback failed");
            }
            return ms;
        };

        r.correct6 = 0;
        r.first_ms = prefix_ms + run(0, false);
        auto t6 = Clock::now(); for (int i = 0; i < 6; ++i) run(i, true); r.six_ms = elapsed_ms(t6);
        auto t100 = Clock::now(); for (int i = 0; i < 100; ++i) run(i % 6, false); r.hundred_ms = elapsed_ms(t100);
        r.dps = 100000.0 / r.hundred_ms;
        llama_free(c);
        out.push_back(r);
    }

    std::cout << "{\n";
    std::cout << "  \"model_load_ms\": " << std::fixed << std::setprecision(3) << load_ms << ",\n";
    std::cout << "  \"threads\": " << threads << ",\n";
    std::cout << "  \"prefix_tokens\": " << pfx.size() << ",\n";
    std::cout << "  \"normal_generation_tokens_max\": " << normal_tokens << ",\n";
    std::cout << "  \"results\": [\n";
    for (size_t i = 0; i < out.size(); ++i) {
        const auto & r = out[i];
        std::cout << "    {\"name\":\"" << r.name << "\","
                  << "\"first_ms\":" << r.first_ms << ","
                  << "\"six_ms\":" << r.six_ms << ","
                  << "\"hundred_ms\":" << r.hundred_ms << ","
                  << "\"decisions_per_sec_100\":" << r.dps << ","
                  << "\"correct_6\":" << r.correct6 << "}"
                  << (i + 1 == out.size() ? "" : ",") << "\n";
    }
    std::cout << "  ]\n}\n";

    llama_model_free(model);
    return 0;
}
