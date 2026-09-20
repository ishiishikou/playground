#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Result {
    std::string name;
    double first_ms = 0;
    double six_ms = 0;
    int correct6 = -1;
};

static double elapsed_ms(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static void show_info(int, char ** argv) {
    std::cerr << "Usage: " << argv[0]
              << " -hf Qwen/Qwen3-VL-4B-Instruct-GGUF:Q4_K_M"
              << " --image img0.png ... --image img5.png -ngl 0 -t 4\n";
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & s, bool add_special) {
    int32_t n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), nullptr, 0, add_special, true);
    if (n >= 0) {
        return {};
    }
    std::vector<llama_token> out((size_t) -n);
    n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(), out.data(), (int32_t) out.size(), add_special, true);
    if (n < 0) {
        throw std::runtime_error("tokenization failed");
    }
    out.resize((size_t) n);
    return out;
}

static llama_token one_token(const llama_vocab * vocab, const std::vector<std::string> & candidates, std::string & selected) {
    for (const auto & s : candidates) {
        auto x = tokenize(vocab, s, false);
        if (x.size() == 1) {
            selected = s;
            return x[0];
        }
    }
    throw std::runtime_error("no single-token A/B label");
}

static std::string token_piece(const llama_vocab * vocab, llama_token t) {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
    if (n < 0) {
        return "<err>";
    }
    return std::string(buf, (size_t) n);
}

static void clear_llama(llama_context * lctx) {
    llama_memory_clear(llama_get_memory(lctx), false);
}

static mtmd::input_chunks make_chunks(
        mtmd_context * mctx,
        const std::string & formatted,
        const mtmd_bitmap * bitmap,
        bool add_special) {
    const std::string marker = mtmd_default_marker();
    const size_t pos = formatted.find(marker);
    if (pos == std::string::npos) {
        if (bitmap != nullptr) {
            throw std::runtime_error("bitmap supplied but media marker missing");
        }
        mtmd_input_text text{formatted.data(), formatted.size(), add_special, true};
        mtmd_input_part part{&text, nullptr};
        const mtmd_input_part * parts[] = {&part};
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        int32_t rc = mtmd_tokenize_from_parts(mctx, chunks.ptr.get(), parts, 1, add_special);
        if (rc != 0) {
            throw std::runtime_error("mtmd_tokenize_from_parts(text) rc=" + std::to_string(rc));
        }
        return chunks;
    }

    if (bitmap == nullptr) {
        throw std::runtime_error("media marker present but bitmap missing");
    }
    if (formatted.find(marker, pos + marker.size()) != std::string::npos) {
        throw std::runtime_error("benchmark expects exactly one media marker");
    }

    std::string before = formatted.substr(0, pos);
    std::string after  = formatted.substr(pos + marker.size());
    mtmd_input_text t0{before.data(), before.size(), false, true};
    mtmd_input_text t1{after.data(), after.size(), false, true};
    mtmd_input_part p0{&t0, nullptr};
    mtmd_input_part p1{nullptr, bitmap};
    mtmd_input_part p2{&t1, nullptr};
    const mtmd_input_part * parts[] = {&p0, &p1, &p2};

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    int32_t rc = mtmd_tokenize_from_parts(mctx, chunks.ptr.get(), parts, 3, add_special);
    if (rc != 0) {
        throw std::runtime_error("mtmd_tokenize_from_parts(media) rc=" + std::to_string(rc));
    }
    return chunks;
}

static llama_pos eval_chunks(
        mtmd_context * mctx,
        llama_context * lctx,
        const mtmd_input_chunks * chunks,
        llama_pos n_past,
        int32_t n_batch) {
    const size_t n_chunks = mtmd_input_chunks_size(chunks);

    for (size_t i = 0; i < n_chunks; ++i) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks, i);
        const auto type = mtmd_input_chunk_get_type(chunk);
        const bool is_last = i + 1 == n_chunks;

        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            llama_pos new_n_past = n_past;
            int32_t rc = mtmd_helper_eval_chunk_single(
                mctx, lctx, chunk, n_past, 0, n_batch, is_last, &new_n_past);
            if (rc != 0) {
                throw std::runtime_error("text chunk eval failed rc=" + std::to_string(rc));
            }
            n_past = new_n_past;
        } else {
            mtmd::batch_ptr mbatch(mtmd_batch_init(mctx));
            int32_t rc = mtmd_batch_add_chunk(mbatch.get(), chunk);
            if (rc != 0) {
                throw std::runtime_error("mtmd_batch_add_chunk failed rc=" + std::to_string(rc));
            }
            rc = mtmd_batch_encode(mbatch.get());
            if (rc != 0) {
                throw std::runtime_error("mtmd_batch_encode failed rc=" + std::to_string(rc));
            }
            float * embd = mtmd_batch_get_output_embd(mbatch.get(), chunk);
            if (!embd) {
                throw std::runtime_error("missing media embedding");
            }
            llama_pos new_n_past = n_past;
            rc = mtmd_helper_decode_image_chunk(
                mctx, lctx, chunk, embd, n_past, 0, n_batch, &new_n_past, nullptr, nullptr);
            if (rc != 0) {
                throw std::runtime_error("image chunk decode failed rc=" + std::to_string(rc));
            }
            n_past = new_n_past;
        }
    }
    return n_past;
}

static bool direct_ab(llama_context * lctx, llama_token a, llama_token b) {
    float * logits = llama_get_logits_ith(lctx, -1);
    if (!logits) {
        throw std::runtime_error("missing final logits");
    }
    return logits[a] > logits[b];
}

static llama_token constrained_ab_sample(llama_context * lctx, const llama_vocab * vocab) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_grammar(vocab, "root ::= \" A\" | \" B\"", "root"));
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    llama_token t = llama_sampler_sample(chain, lctx, -1);
    llama_sampler_free(chain);
    return t;
}

static void greedy_generate(llama_context * lctx, const llama_vocab * vocab, int n_tokens) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    for (int i = 0; i < n_tokens; ++i) {
        llama_token t = llama_sampler_sample(chain, lctx, -1);
        if (llama_vocab_is_eog(vocab, t)) {
            break;
        }
        if (i + 1 < n_tokens) {
            llama_batch batch = llama_batch_get_one(&t, 1);
            if (llama_decode(lctx, batch) != 0) {
                llama_sampler_free(chain);
                throw std::runtime_error("generation decode failed");
            }
        }
    }
    llama_sampler_free(chain);
}

static void emit(const Result & r) {
    std::cout << std::fixed << std::setprecision(3)
              << "{\"type\":\"mode_result\",\"name\":\"" << r.name
              << "\",\"first_ms\":" << r.first_ms
              << ",\"six_ms\":" << r.six_ms
              << ",\"correct_6\":" << r.correct6 << "}\n" << std::flush;
}

int main(int argc, char ** argv) {
    common_init();
    ggml_backend_load_all();

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_info)) {
        return 2;
    }
    if (params.image.size() < 6) {
        std::cerr << "Need at least six --image inputs\n";
        return 2;
    }

    const auto load_start = Clock::now();
    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    llama_context * lctx = llama_init->context();
    if (!model || !lctx) {
        return 3;
    }
    const double model_load_ms = elapsed_ms(load_start);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu          = params.mmproj_use_gpu;
    mparams.device           = params.mmproj_device;
    mparams.print_timings    = true;
    mparams.n_threads        = params.cpuparams.n_threads;
    mparams.flash_attn_type  = params.flash_attn_type;
    mparams.warmup           = params.warmup;
    mparams.image_min_tokens = params.image_min_tokens;
    mparams.image_max_tokens = params.image_max_tokens;

    mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams));
    if (!mctx) {
        throw std::runtime_error("mtmd init failed");
    }

    mtmd_helper_init_opt init_opt = mtmd_helper_init_opt_default();
    std::vector<mtmd::bitmap> images;
    for (size_t i = 0; i < 6; ++i) {
        auto loaded = mtmd_helper_bitmap_init_from_file(mctx.get(), params.image[i].c_str(), false, init_opt);
        if (!loaded.bitmap) {
            throw std::runtime_error("image load failed: " + params.image[i]);
        }
        images.emplace_back(loaded.bitmap);
    }

    std::string a_label, b_label;
    llama_token a_tok = one_token(vocab, {" A", "A", " 1", "1"}, a_label);
    llama_token b_tok = one_token(vocab, {" B", "B", " 0", "0"}, b_label);

    const std::vector<bool> expected = {true, false, true, true, false, true};

    const std::string system_text =
        "You are a deterministic binary visual decision engine. "
        "Inspect only the current image. A means YES and B means NO. "
        "For binary output use exactly A for YES and B for NO, with no explanation.";

    const std::string marker = mtmd_default_marker();
    const std::string binary_user =
        marker + std::string("\nQuestion: Does the image contain a large solid red square near the center?\nAnswer:");
    const std::string json_user =
        marker + std::string("\nQuestion: Does the image contain a large solid red square near the center?\n"
                             "Return compact JSON with keys answer and reason; answer must be YES or NO and reason must be very short.\nJSON:");

    common_chat_templates_ptr tmpls = common_chat_templates_init(model, params.chat_template);
    std::vector<common_chat_msg> history;
    common_chat_msg sys_msg;
    sys_msg.role = "system";
    sys_msg.content = system_text;
    const std::string sys_fmt = common_chat_format_single(
        tmpls.get(), history, sys_msg, false, params.use_jinja);
    history.push_back(sys_msg);

    common_chat_msg bin_msg;
    bin_msg.role = "user";
    bin_msg.content = binary_user;
    const std::string bin_fmt = common_chat_format_single(
        tmpls.get(), history, bin_msg, true, params.use_jinja);

    common_chat_msg json_msg;
    json_msg.role = "user";
    json_msg.content = json_user;
    const std::string json_fmt = common_chat_format_single(
        tmpls.get(), history, json_msg, true, params.use_jinja);

    const std::string full_bin  = sys_fmt + bin_fmt;
    const std::string full_json = sys_fmt + json_fmt;

    const int32_t n_batch = params.n_batch > 0 ? params.n_batch : 2048;

    std::cerr << std::fixed << std::setprecision(3)
              << "model_load_ms=" << model_load_ms
              << " threads=" << params.cpuparams.n_threads
              << " a_token=" << a_tok << " a_piece=" << token_piece(vocab, a_tok)
              << " b_token=" << b_tok << " b_piece=" << token_piece(vocab, b_tok)
              << " system_chars=" << system_text.size() << "\n";

    std::cout << std::fixed << std::setprecision(3)
              << "{\"type\":\"meta\",\"model_load_ms\":" << model_load_ms
              << ",\"threads\":" << params.cpuparams.n_threads
              << ",\"system_chars\":" << system_text.size()
              << ",\"normal_generation_tokens_max\":12}\n" << std::flush;

    auto eval_full = [&](const std::string & formatted, int i) {
        clear_llama(lctx);
        auto chunks = make_chunks(mctx.get(), formatted, images[(size_t) i].ptr.get(), true);
        return eval_chunks(mctx.get(), lctx, chunks.ptr.get(), 0, n_batch);
    };

    // Warmup: one full multimodal direct-logit pass.
    eval_full(full_bin, 0);
    (void) direct_ab(lctx, a_tok, b_tok);

    // A: normal structured generation.
    {
        Result r;
        r.name = "A_normal_json_generation";
        auto run = [&](int i) {
            auto t = Clock::now();
            eval_full(full_json, i);
            greedy_generate(lctx, vocab, 12);
            return elapsed_ms(t);
        };
        r.first_ms = run(0);
        auto t = Clock::now();
        for (int i = 0; i < 6; ++i) {
            (void) run(i);
        }
        r.six_ms = elapsed_ms(t);
        emit(r);
    }

    // B: constrained very-short A/B generation.
    {
        Result r;
        r.name = "B_constrained_ab_generation";
        r.correct6 = 0;
        auto run = [&](int i, bool score) {
            auto t = Clock::now();
            eval_full(full_bin, i);
            llama_token z = constrained_ab_sample(lctx, vocab);
            if (score) {
                const bool pred = z == a_tok ? true : (z == b_tok ? false : !expected[(size_t) i]);
                if ((z == a_tok || z == b_tok) && pred == expected[(size_t) i]) {
                    r.correct6++;
                }
            }
            return elapsed_ms(t);
        };
        r.first_ms = run(0, false);
        auto t = Clock::now();
        for (int i = 0; i < 6; ++i) {
            (void) run(i, true);
        }
        r.six_ms = elapsed_ms(t);
        emit(r);
    }

    // C: direct A/B logits, full image prompt evaluated every time.
    {
        Result r;
        r.name = "C_direct_logits_no_cache";
        r.correct6 = 0;
        auto run = [&](int i, bool score) {
            auto t = Clock::now();
            eval_full(full_bin, i);
            const bool pred = direct_ab(lctx, a_tok, b_tok);
            if (score && pred == expected[(size_t) i]) {
                r.correct6++;
            }
            return elapsed_ms(t);
        };
        r.first_ms = run(0, false);
        auto t = Clock::now();
        for (int i = 0; i < 6; ++i) {
            (void) run(i, true);
        }
        r.six_ms = elapsed_ms(t);
        emit(r);
    }

    // D: direct logits + complete llama-state snapshot after the common text prefix.
    {
        Result r;
        r.name = "D_direct_logits_shared_prefix_snapshot";
        r.correct6 = 0;

        clear_llama(lctx);
        auto prefix_chunks = make_chunks(mctx.get(), sys_fmt, nullptr, true);
        auto prefix_start = Clock::now();
        llama_pos prefix_n_past = eval_chunks(mctx.get(), lctx, prefix_chunks.ptr.get(), 0, n_batch);
        const double prefix_ms = elapsed_ms(prefix_start);

        size_t state_size = llama_state_get_size(lctx);
        std::vector<uint8_t> state(state_size);
        size_t got = llama_state_get_data(lctx, state.data(), state.size());
        if (got == 0 || got > state.size()) {
            throw std::runtime_error("prefix state snapshot failed");
        }
        state.resize(got);

        auto restore = [&]() {
            size_t used = llama_state_set_data(lctx, state.data(), state.size());
            if (used == 0) {
                throw std::runtime_error("prefix state restore failed");
            }
        };

        auto run = [&](int i, bool score) {
            restore();
            auto chunks = make_chunks(mctx.get(), bin_fmt, images[(size_t) i].ptr.get(), false);
            auto t = Clock::now();
            (void) eval_chunks(mctx.get(), lctx, chunks.ptr.get(), prefix_n_past, n_batch);
            const bool pred = direct_ab(lctx, a_tok, b_tok);
            if (score && pred == expected[(size_t) i]) {
                r.correct6++;
            }
            return elapsed_ms(t);
        };

        r.first_ms = prefix_ms + run(0, false);
        auto t = Clock::now();
        for (int i = 0; i < 6; ++i) {
            (void) run(i, true);
        }
        r.six_ms = elapsed_ms(t);
        emit(r);
    }

    return 0;
}
