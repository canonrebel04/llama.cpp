#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"
#include "speculative.h"

// TODO: replace with #include "llama-ext.h" in the future
#include "../src/llama-arch.h"
#include "../src/llama-context.h"
#include "../src/llama-ext.h"
#include "../src/llama-memory-hybrid-idx.h"
#include "../src/llama-model-saver.h"
#include "test-llama-batch-legacy.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <random>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static bool arch_matches(const std::string & filter, llm_arch arch) {
    if (filter.empty()) {
        return true;
    }
    return std::regex_search(llama_arch_name(arch), std::regex(filter));
}

// normalized mean squared error = mse(a, b) / mse(a, 0)
static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    GGML_ASSERT(a.size() == b.size());
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;

    for (size_t i = 0; i < a.size(); i++) {
        float a_i = a[i];
        float b_i = b[i];

        mse_a_b += (a_i - b_i) * (a_i - b_i);
        mse_a_0 += a_i * a_i;
    }

    return mse_a_b / mse_a_0;
}

struct tensor_data_params {
    size_t seed;
    float  stdev;
};

// fills the values of one tensor, the name carries whatever the property needs
typedef void (*tensor_filler)(std::vector<float> & dst, const char * name, const tensor_data_params & params);

// a pool of normal draws, generated once and reused for every model
static std::vector<float> g_normal_pool;
static const size_t NORMAL_POOL_SIZE = 16 * 1024 * 1024;

// fill the pool once; each tensor copies a slice of it
static void init_normal_pool(const size_t seed, const float stdev) {
    g_normal_pool.resize(NORMAL_POOL_SIZE);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, stdev);
    for (float & v : g_normal_pool) {
        v = dist(gen);
    }
}

// a normal draw keyed on a name, so that tensors which share a draw ask for the same key
static void fill_normal(std::vector<float> & dst, const char * key, const tensor_data_params &) {
    const size_t n = dst.size();
    const size_t pool_size = g_normal_pool.size();
    const size_t start = std::hash<std::string>{}(key) % pool_size;

    // copy a cyclic slice of the pool, so a tensor larger than the pool still fills
    size_t i = 0;
    size_t pos = start;
    while (i < n) {
        const size_t chunk = std::min(pool_size - pos, n - i);
        std::memcpy(dst.data() + i, g_normal_pool.data() + pos, chunk * sizeof(float));
        i += chunk;
        pos = 0;
    }
}

// Mamba A must be negative (state decay)
static void fill_ssm_a(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = -fabsf(v);
    }
}

// RoPE factors divide the frequencies, a value close to 0 gives angles too large for F32 sin/cos to stay accurate
static void fill_rope_factors(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = 1.0f + fabsf(v);
    }
}

// FFN act scales divide the activation, a value close to 0 overflows the F16 staging of the Metal mul-mat kernel
static void fill_ffn_act(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    fill_normal(dst, name, params);
    for (float & v : dst) {
        v = copysignf(1.0f + fabsf(v), v);
    }
}

// a norm preserving residual gate holds alpha^2 + beta^2 == 1, and the conversion resolves both from one raw weight
// the alpha and the beta of a block share that weight, so the key drops the suffix and keeps the block
static void fill_resid_gate(std::vector<float> & dst, const char * name, const tensor_data_params & params) {
    const char * suffix = strstr(name, "_resid_");
    const bool is_beta = strstr(suffix, "_resid_beta") != nullptr;
    fill_normal(dst, std::string(name, suffix - name).c_str(), params);
    for (float & v : dst) {
        const float beta = 1.0f / (1.0f + expf(-v));
        v = is_beta ? beta : sqrtf(1.0f - beta*beta);
    }
}

// a random router bias can tie within the noise between the backends, a ramp keeps the top-k selection the same
static void fill_exp_probs_b(std::vector<float> & dst, const char *, const tensor_data_params &) {
    for (size_t i = 0; i < dst.size(); i++) {
        dst[i] = 2.0f*i;
    }
}

// a tensor whose values hold a property of their own, matched on a substring of its name, first match wins
static const struct { const char * match; tensor_filler fill; } TENSOR_FILLERS[] = {
    { "ssm_a",        fill_ssm_a        },
    { "_resid_",      fill_resid_gate   },
    { "exp_probs_b",  fill_exp_probs_b  },
    { "ffn.act",      fill_ffn_act      },
    { "rope_freqs",   fill_rope_factors },
    { "rope_factors", fill_rope_factors },
};

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    const tensor_data_params & params = *(const tensor_data_params *) userdata;

    tensor_filler fill = fill_normal;
    for (const auto & filler : TENSOR_FILLERS) {
        if (strstr(tensor->name, filler.match) != nullptr) {
            fill = filler.fill;
            break;
        }
    }

    std::vector<float> tmp(ggml_nelements(tensor));
    fill(tmp, tensor->name, params);

    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp16(tmp.size());
        for (size_t i = 0; i < tmp.size(); i++) {
            tmp16[i] = ggml_fp32_to_fp16(tmp[i]);
        }
        ggml_backend_tensor_set(tensor, tmp16.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("fatal error");
    }
}

static void usage(char ** argv) {
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("Options:\n");
    LOG("  -a, --arch <arch|regex>  Run only matching LLM architectures (default: all supported)\n");
    LOG("  -s, --seed <seed>        Set the random seed for tensor initialization and token generation\n");
    LOG("  -d, --stdev <stdev>      Set the standard deviation of the tensor initialization distribution (default: 0.1f)\n");
    LOG("  -o, --out <dir>          Save generated test models to <dir> instead of running backend tests\n");
    LOG("  -v <N>                   Set log verbosity level\n");
    LOG("  -b, --backend <backend>  Run only on the given backend device\n");
    LOG("  --kv-shift               Check K-shift against a decode at the shifted positions (CPU only)\n");
    LOG("  --test-phase-workspace   Check shared target/draft phase workspace ownership\n");
    LOG("  --test-live-context-workspace Check live context workspace resizing\n");
    LOG("  --test-speculative-limits Check retained speculative draft limits\n");
    LOG("  -h, --help               Show this help message\n\n");
    LOG("Examples:\n");
    LOG("  %s\n", argv[0]);
    LOG("  %s -a qwen35moe\n", argv[0]);
    LOG("  %s -a deepseek4 -o tests/test-models/\n", argv[0]);
    LOG("  %s -a cohere2moe -v 5\n", argv[0]);
}

static std::vector<llama_token> get_tokens(const uint32_t n_tokens, const uint32_t n_vocab, const size_t seed){
    std::mt19937 gen(seed);
    std::uniform_int_distribution<> dis(0, n_vocab - 1);
    std::vector<llama_token> ret;
    ret.reserve(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        ret.push_back(dis(gen));
    }
    return ret;
}

// MoE archs that are also tested with the experts in host memory
static bool host_experts_test(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_DEEPSEEK2:
            return true;
        default:
            return false;
    }
}

// writes arch-resolved GGUF keys into the fixture context, the test-side equivalent of llama_model_saver
struct kv_writer {
    struct gguf_context * gguf_ctx;
    const llm_arch        arch;

    kv_writer(enum llm_arch arch, struct gguf_context * gguf_ctx) : gguf_ctx(gguf_ctx), arch(arch) {}

    void add_kv(enum llm_kv key, uint32_t     value) { gguf_set_val_u32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, int32_t      value) { gguf_set_val_i32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, uint64_t     value) { gguf_set_val_u64 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, float        value) { gguf_set_val_f32 (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, bool         value) { gguf_set_val_bool(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }
    void add_kv(enum llm_kv key, const char * value) { gguf_set_val_str (gguf_ctx, llama_arch_kv_name(arch, key).c_str(), value); }

    template <typename T>
    void add_kv(enum llm_kv key, const std::vector<T> & value) {
        if (value.empty()) {
            return;
        }
        if (std::is_same<T, uint32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT32, value.data(), value.size());
        } else if (std::is_same<T, int32_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_INT32, value.data(), value.size());
        } else if (std::is_same<T, uint64_t>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_UINT64, value.data(), value.size());
        } else if (std::is_same<T, float>::value) {
            gguf_set_arr_data(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), GGUF_TYPE_FLOAT32, value.data(), value.size());
        } else {
            GGML_ABORT("fatal error");
        }
    }

    void add_kv(enum llm_kv key, const std::vector<std::string> & value) {
        std::vector<const char *> tmp(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            tmp[i] = value[i].c_str();
        }
        gguf_set_arr_str(gguf_ctx, llama_arch_kv_name(arch, key).c_str(), tmp.data(), tmp.size());
    }
};

static gguf_context_ptr get_gguf_ctx(const llm_arch arch, const bool moe, const bool mtp = false, const uint32_t n_layer_min = 0) {
    gguf_context_ptr ret(gguf_init_empty());
    kv_writer ms(arch, ret.get());
    const uint32_t n_ctx = 256;

    uint32_t n_vocab = 128;
    uint32_t n_embd  = 256;
    uint32_t n_head  = 2;
    uint32_t n_ff    = 384;
    uint32_t n_layer = 2;
    if (arch == LLM_ARCH_LLAMA4) {
        n_layer = 4; // hparams.n_no_rope_layer_step is hard-coded to 4
    } else if (arch == LLM_ARCH_MAION_CODER) {
        n_head = 4;
        n_layer = 4; // dense and MoE layers, each with local and full attention
    } else if (arch == LLM_ARCH_GEMMA4) {
        // TODO: expand coverage to both GQA and MQA (see https://github.com/ggml-org/llama.cpp/pull/31401#pullrequestreview-5710165130)
        n_head = 4;
        n_layer = 10; // last 5 layers reuse KV, both SWA and full attention
    } else if (arch == LLM_ARCH_GEMMA3N) {
        n_embd = 64;
        n_head = 1;
        n_ff   = 96;
        n_layer = 22; // hparams.n_layer_kv_from_start = 20 is hardcoded
    } else if (arch == LLM_ARCH_LFM3) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_DEEPSEEK4) {
        // head size 64 so that GPU flash attention kernels support the model
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 4;
    } else if (arch == LLM_ARCH_DEEPSEEK41) {
        // window-only, Engram, ratio-2 and ratio-1 sources, top-k reuse and a candidate-masked index layer
        n_embd  = 512;
        n_head  = 8;
        n_ff    = 1024;
        n_layer = 6;
    } else if (arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_LAGUNA) {
        n_embd = 160; // exercise per-head tensor split granularity with head size 80
    } else if (arch == LLM_ARCH_QWEN3 || arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_RICURSIVE) {
        n_head = 4;
    } else if (arch == LLM_ARCH_LIMITE) {
        n_head  = 4;
        n_layer = 4; // dense connections on the last two layers
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_XING4_0) {
        n_embd = 128;
        n_head = 1;
        n_ff   = 192;
        if (arch == LLM_ARCH_MISTRAL_LARGE4) {
            n_layer = 4; // cover NoPE, shared indexer and KV sharing layers
        }
    } else if (arch == LLM_ARCH_GLM5_NEXT) {
        // several heads so that -sm tensor can split them and the MLA flash attention has GQA > 1
        n_embd = 128;
        n_head = 16;
        n_ff   = 192;
    } else if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        n_layer = 3;
    } else if (arch == LLM_ARCH_METIS16) {
        n_layer = 4; // 2 passes over 2 physical layers, one attention and one Mamba-2
    } else if (arch == LLM_ARCH_CHAMELEON) {
        n_vocab = 10240;
    } else if (arch == LLM_ARCH_QWEN3TTS) {
        //n_vocab = 4096; // must be >= the hard-coded codec head size (3072)
        n_vocab = 3072; // TODO: should be 4096, but user code cannot get `n_vocab_out` yet [TAG_LLAMA_N_VOCAB_OUT]
    } else if (arch == LLM_ARCH_HRM_TEXT) {
        n_layer = 6; // 1 layer per stack x 2 h-cycles x (2 l-cycles + 1) cache slots
    } else if (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) {
        n_embd  = 128;
        n_head  = 4;
        n_ff    = 24;
        n_layer = 3;
    }

    GGML_ASSERT(!mtp || arch == LLM_ARCH_QWEN35);

    uint32_t n_head_kv = n_head;
    if (arch == LLM_ARCH_QWEN3) {
        n_head_kv = 1; // MQA coverage
    } else if (arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_AFMOE || arch == LLM_ARCH_MAION_CODER ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE || arch == LLM_ARCH_GEMMA4 || arch == LLM_ARCH_RICURSIVE ||
            arch == LLM_ARCH_ZAYA1) {
        n_head_kv = 2; // GQA coverage
    }
    if (arch == LLM_ARCH_BEAM) {
        n_head    = 4;
        n_head_kv = 2;
        n_layer   = 4;
    }
    n_layer = std::max(n_layer, n_layer_min);
    const uint32_t n_layer_all = n_layer + (mtp ? 1 : 0);
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llama_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            n_ctx);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer_all);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    if (mtp) {
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS, uint32_t(1));
    }

    if (arch == LLM_ARCH_K2_HORIZON) {
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_FAST, 128.0f);
        ms.add_kv(LLM_KV_ROPE_SCALING_YARN_BETA_SLOW,   4.0f);
    }

    if (arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE) {
        std::vector<uint32_t> n_ff_per_layer;
        n_ff_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_ff_per_layer.push_back(il <= 1 ? 0 : n_ff);
        }
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff_per_layer);
    } else {
        ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,   false);
    ms.add_kv(LLM_KV_LOGIT_SCALE,             1.0f);
    ms.add_kv(LLM_KV_TIME_MIX_EXTRA_DIM,      uint32_t(64));
    ms.add_kv(LLM_KV_TIME_DECAY_EXTRA_DIM,    uint32_t(128));
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(2));

    if (arch == LLM_ARCH_PLAMO2 || arch == LLM_ARCH_JAMBA || arch == LLM_ARCH_NEMOTRON_H || arch == LLM_ARCH_NEMOTRON_H_MOE ||
            arch == LLM_ARCH_GRANITE_HYBRID || arch == LLM_ARCH_LFM2 || arch == LLM_ARCH_LFM2MOE || arch == LLM_ARCH_LFM3 ||
            arch == LLM_ARCH_KIMI_LINEAR || arch == LLM_ARCH_BAILINGMOE3 || arch == LLM_ARCH_KIMI_K3 || arch == LLM_ARCH_KIMI_K31 ||
            arch == LLM_ARCH_GLM5_NEXT
            ) {
        GGML_ASSERT(n_layer >= 2);
        std::vector<uint32_t> n_head_per_layer;
        n_head_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_per_layer.push_back(il == 1 ? 0 : n_head);
        }
        // GLM5 next KDA heads come from the uniform head count, only head_count_kv is per layer.
        if (arch == LLM_ARCH_GLM5_NEXT) {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
            // the MLA cache holds one latent head
            for (uint32_t & n : n_head_per_layer) {
                n = std::min(n, uint32_t(1));
            }
        } else {
            ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head_per_layer);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_per_layer);
    } else if (arch == LLM_ARCH_METIS16) {
        // the layer types repeat in every pass
        std::vector<uint32_t> n_head_kv_per_layer;
        n_head_kv_per_layer.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            n_head_kv_per_layer.push_back(il % 2 == 0 ? n_head_kv : 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, n_head_kv_per_layer);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT, n_head);
        ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV, (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(1) : n_head_kv);
    }

    ms.add_kv(LLM_KV_ATTENTION_MAX_ALIBI_BIAS, 8.0f);
    if (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,   n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH, n_embd_head);
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,   n_embd_head/2);
    } else if (arch == LLM_ARCH_DEEPSEEK2
            || arch == LLM_ARCH_DEEPSEEK32
            || arch == LLM_ARCH_GLM_DSA
            || arch == LLM_ARCH_DOTS3NOTE
            || arch == LLM_ARCH_KIMI_LINEAR
            || arch == LLM_ARCH_BAILINGMOE3
            || arch == LLM_ARCH_KIMI_K3
            || arch == LLM_ARCH_KIMI_K31
            || arch == LLM_ARCH_GLM5_NEXT
            || arch == LLM_ARCH_HY_V4
            || arch == LLM_ARCH_MISTRAL4
            || arch == LLM_ARCH_MISTRAL_LARGE4
            || arch == LLM_ARCH_XING4_0) {
        // GLM5 next MLA is nope only, the cache row is the compressed latent alone.
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(512) : uint32_t(576));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH,     uint32_t(512));
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       arch == LLM_ARCH_GLM5_NEXT ? uint32_t(0) : uint32_t(64));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA,   uint32_t(192));
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, uint32_t(128));
        if (arch == LLM_ARCH_DOTS3NOTE) {
            // SWA layers reuse the same MLA geometry as the full layers in this fixture
            ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,       uint32_t(576));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,     uint32_t(512));
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_MLA_SWA,   uint32_t(192));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_MLA_SWA, uint32_t(128));
            ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,             10000.0f);
            // indexer on the full-attention layers (inverse of the swa pattern)
            std::vector<uint32_t> indexer_types;
            indexer_types.reserve(n_layer);
            for (uint32_t il = 0; il < n_layer; il++) {
                indexer_types.push_back(il % 2 ? 0 : 1);
            }
            ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
        }
    } else if (arch == LLM_ARCH_MINIMAX_M3) {
        // partial rotary: n_rot must not exceed the indexer key length (64)
        ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,       uint32_t(64));
    }
    if (arch == LLM_ARCH_SENSENOVA_U1) {
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SPATIAL, 10000.0f);
    }

    if (arch == LLM_ARCH_XING4_0) {
        // Manifold-Constrained Hyper-Connections (MHC) multi-residual-stream blocks
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,               uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(20));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,             1e-6f);
    }
    ms.add_kv(LLM_KV_ATTENTION_CLAMP_KQV,              1.0f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,      1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_EPS,          1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_GROUPNORM_GROUPS,       uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_Q_LORA_RANK,            (arch == LLM_ARCH_DEEPSEEK4 || arch == LLM_ARCH_DEEPSEEK41) ? uint32_t(64) : uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_KV_LORA_RANK,           uint32_t(512));
    ms.add_kv(LLM_KV_ATTENTION_RELATIVE_BUCKETS_COUNT, uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,         n_ctx/8);

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>{1, 1, 0, 0});
    } else if (arch == LLM_ARCH_GEMMA4) {
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,      n_embd/2);
        ms.add_kv(LLM_KV_ATTENTION_SHARED_KV_LAYERS,      uint32_t(5));
        ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,        n_embd_head);
        ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,      n_embd_head);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,              10000.0f);
        // SWA pattern: every 5th layer is full attention (matches E2B layer_types)
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back((il + 1) % 5 != 0);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
    } else if (arch == LLM_ARCH_COHERE2MOE || arch == LLM_ARCH_MIMO2 || arch == LLM_ARCH_STEP35 || arch == LLM_ARCH_SPARK2_5 ||
            arch == LLM_ARCH_MUSE_GLIMMER || arch == LLM_ARCH_MUSE_SPARK || arch == LLM_ARCH_GRANITE_SWA || arch == LLM_ARCH_DOTS3NOTE ||
            arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_MAPLE || arch == LLM_ARCH_FLINT || arch == LLM_ARCH_BEAM || arch == LLM_ARCH_RICURSIVE || arch == LLM_ARCH_DOMYN_EDGE) {
        std::vector<uint32_t> pattern;
        pattern.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            pattern.push_back(il % 2);
        }
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, pattern);
        if (arch == LLM_ARCH_FLINT || arch == LLM_ARCH_DOMYN_EDGE) {
            ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN, pattern);
        }
        if (arch == LLM_ARCH_DOMYN_EDGE) {
            // SWA (rope) layers use a smaller head dim than the full-attention (nope) layers
            ms.add_kv(LLM_KV_ATTENTION_KEY_LENGTH_SWA,   n_embd_head/2);
            ms.add_kv(LLM_KV_ATTENTION_VALUE_LENGTH_SWA, n_embd_head/2);
            ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT_SWA, n_embd_head/2);
        }
    } else if (arch == LLM_ARCH_LFM3) {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, std::vector<uint32_t>({0, 0, 1}));
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,               10000.0f);
    } else {
        ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    }

    if (arch == LLM_ARCH_MUSE_SPARK) {
        ms.add_kv(LLM_KV_POST_NORM_EPS,           1e-5f);
        ms.add_kv(LLM_KV_ROPE_FREQ_BASE_SWA,      10000.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING, 30.0f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,         n_embd/2);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_KOLIBRI1 || arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, false);
    }

    // MSA requires one indexer head per GQA (KV) head, unlike the DSA archs where the
    // indexer head count is independent of the main attention head count.
    if (arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,    uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,  1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_LOW_RANK, uint32_t(8));
        // without this the QSA layers fall back to dense and go uncovered
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS, std::vector<uint32_t>(n_layer, 4));

        // has_cell_ext() needs ple_n_heads here: the indexer cache serializes no ext without it
        const uint32_t ple_ngram_size      = 3;
        const uint32_t ple_heads_per_ngram = 2;
        const uint32_t ple_n_heads         = (ple_ngram_size - 1)*ple_heads_per_ngram;
        GGML_ASSERT(n_embd % ple_n_heads == 0);
        const uint32_t ple_head_dim = n_embd/ple_n_heads;

        std::vector<uint64_t> ple_head_offsets(ple_n_heads);
        std::vector<uint64_t> ple_head_vocab_sizes(ple_n_heads, n_vocab);
        for (uint32_t h = 0; h < ple_n_heads; h++) {
            ple_head_offsets[h] = uint64_t(h)*n_vocab;
        }

        // the PLE history lives in the recurrent cache, so it must sit on a linear attention layer
        ms.add_kv(LLM_KV_PLE_LAYERS,                  std::vector<uint32_t>({ 0 }));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,              ple_ngram_size);
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,         ple_heads_per_ngram);
        ms.add_kv(LLM_KV_PLE_CONV_KERNEL,             uint32_t(4));
        ms.add_kv(LLM_KV_PLE_EOS_TOKEN_ID,            uint32_t(0));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,  ple_head_dim);
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,       std::vector<uint64_t>({ 1, 3, 5 }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,            ple_head_offsets);
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,        ple_head_vocab_sizes);
    }

    // minimax-m3 keeps one indexer head per GQA head; the rest use a fixed 64 to match the fused
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   arch == LLM_ARCH_MINIMAX_M3 ? n_head : uint32_t(64));
    // qwen4exp ropes indexer keys with the main rotary width, so its head can't be < n_rot
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,
              arch == LLM_ARCH_QWEN4EXP ? n_embd_head : uint32_t(128));

    // note: using a realistic top-k here makes the results unstable and hard to match between CPU and GPU
    //       a large value makes things deterministic since all data is selected by the indexer
    //ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(8));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_TOP_K,        uint32_t(131072));

    ms.add_kv(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL,        uint32_t(4));
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_KPOOL_SELECT_TAIL, true);
    ms.add_kv(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, uint32_t(1));
    // mrope sections count rope pairs; Ling 3.0 VL files carry [t, h, w] sections
    // summing to n_rot / 2 (n_rot is 64 in this fixture)
    if (arch == LLM_ARCH_BAILINGMOE3) {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({8, 12, 12, 0}));
    } else {
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    }

    if (arch == LLM_ARCH_HY_V4) {
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,     uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,   1.0e-6f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_MAGNITUDE, 2.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,           10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,       1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,        true);
        // layer 0 must own an indexer, the odd layers share it
        std::vector<uint32_t> indexer_types;
        indexer_types.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            indexer_types.push_back(il % 2 ? 0 : 1);
        }
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES, indexer_types);
    }

    if (arch == LLM_ARCH_MISTRAL_LARGE4) {
        // layer 3 reads the KV cache of layer 2, both are NoPE
        ms.add_kv(LLM_KV_ATTENTION_ROPE_PATTERN,     std::vector<uint32_t>({1, 1, 0, 0}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_TYPES,    std::vector<uint32_t>({1, 0, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_MAP,    std::vector<uint32_t>({0, 1, 2, 2}));
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,    30.0f);
    }

    if (arch == LLM_ARCH_DEEPSEEK4) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 4, 128}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
    }

    if (arch == LLM_ARCH_DEEPSEEK41) {
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,          uint32_t(8));
        ms.add_kv(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,            uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_RATIOS,             std::vector<uint32_t>({0, 0, 2, 2, 1, 1}));
        ms.add_kv(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,     160000.0f);
        ms.add_kv(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,            std::vector<uint32_t>({2, 4}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS,       std::vector<uint32_t>({2, 4, 5}));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, int32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, uint32_t(4096));
        ms.add_kv(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS,  uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_EPSILON,              1.0e-6f);
        ms.add_kv(LLM_KV_HASH_LAYER_COUNT,                      uint32_t(0));
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,                      10.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,                  1.0f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,                   true);
        std::vector<int32_t> token_map(n_vocab);
        for (uint32_t i = 0; i < n_vocab; ++i) {
            token_map[i] = (int32_t) (i % 61);
        }
        ms.add_kv(LLM_KV_PLE_LAYERS,                   std::vector<uint32_t>({1, 3}));
        ms.add_kv(LLM_KV_PLE_NGRAM_SIZE,               uint32_t(3));
        ms.add_kv(LLM_KV_PLE_HEADS_PER_NGRAM,          uint32_t(2));
        ms.add_kv(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,   uint32_t(64));
        ms.add_kv(LLM_KV_PLE_LAYER_MULTIPLIERS,        std::vector<uint64_t>({ 76632096046245ull, 4839876093313ull, 35959672319349ull,
                                                                               67716810739261ull, 51510806800915ull, 30921347202721ull }));
        ms.add_kv(LLM_KV_PLE_HEAD_OFFSETS,             std::vector<uint64_t>({0, 101, 204, 311, 420, 533, 660, 791}));
        ms.add_kv(LLM_KV_PLE_HEAD_VOCAB_SIZES,         std::vector<uint64_t>({101, 103, 107, 109, 113, 127, 131, 137}));
        ms.add_kv(LLM_KV_ENGRAM_PAD_TOKEN_ID,          uint32_t(2));
        ms.add_kv(LLM_KV_ENGRAM_TOKEN_MAP,             token_map);
    }

    if (arch == LLM_ARCH_HRM_TEXT) {
        // 6 cache slots alias 2 physical blocks: 1 low-stack layer + 1 high-stack layer
        ms.add_kv(LLM_KV_HRM_LAYERS_PER_STACK, uint32_t(1));
        ms.add_kv(LLM_KV_HRM_H_CYCLES,         uint32_t(2));
        ms.add_kv(LLM_KV_HRM_L_CYCLES,         uint32_t(2));
    }

    if (arch == LLM_ARCH_MAPLE) {
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP, 7.0f);
    }

    if (arch == LLM_ARCH_RICURSIVE) {
        std::vector<float> swiglu_clamp;
        swiglu_clamp.reserve(n_layer);
        for (uint32_t il = 0; il < n_layer; il++) {
            // the random weight scales keep the FFN activations small, a small limit still clamps so the roundtrip checks the per-layer values
            swiglu_clamp.push_back(il % 3 == 1 ? 0.0f : 1e-3f);
        }
        ms.add_kv(LLM_KV_EMBEDDING_SCALE, 16.0f);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP,    swiglu_clamp);
    }

    if (arch == LLM_ARCH_METIS16) {
        ms.add_kv(LLM_KV_METIS_PASS_COUNT,               uint32_t(2));
        ms.add_kv(LLM_KV_HYPER_CONNECTION_COUNT,         uint32_t(4));
        ms.add_kv(LLM_KV_METIS_MEMORY_LENGTH,            uint32_t(64));
        ms.add_kv(LLM_KV_METIS_ROUTE_LENGTH,             uint32_t(32));
        ms.add_kv(LLM_KV_ATTENTION_PASS_LORA_RANK,       uint32_t(8));
        ms.add_kv(LLM_KV_METIS_CONTINUATION_LOGIT_LIMIT, 4.0f);
        ms.add_kv(LLM_KV_SSM_NORM_EPS,                   1e-5f);
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,                n_embd/2);
    }

    if (arch == LLM_ARCH_LIMITE) {
        ms.add_kv(LLM_KV_ATTENTION_SCALE,             0.1f);
        ms.add_kv(LLM_KV_ATTENTION_XSA_EPS,           1e-4f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SIGMOID_CAPPING, std::vector<float>({23.0f, 5.0f, 7.5f}));
        ms.add_kv(LLM_KV_MUDD_FEED_FORWARD_LENGTH,    uint32_t(8));
        ms.add_kv(LLM_KV_MUDD_TAP_COUNT,              uint32_t(2));
        ms.add_kv(LLM_KV_MUDD_TAP_INDICES,            std::vector<int32_t>({-1, -1, -1, -1, 0, 2, 1, 3}));
    }

    if (arch == LLM_ARCH_INKLING2 || arch == LLM_ARCH_INKLING2_LOOP) {
        const bool loop = arch == LLM_ARCH_INKLING2_LOOP;
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_LENGTH,   uint32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_EXTENT,   uint32_t(16));
        ms.add_kv(LLM_KV_ATTENTION_QK_CLIP,           0.25f);
        ms.add_kv(LLM_KV_ATTENTION_RELATIVE_CLIP,     0.1f);
        ms.add_kv(LLM_KV_ATTENTION_LOG_SCALE_FLOOR,   uint32_t(4));
        ms.add_kv(LLM_KV_ATTENTION_LOG_SCALE_ALPHA,   0.1f);
        ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(2));
        if (loop) {
            // prelude 0, core 1, coda 2; the loader adds slot 3 for the second pass over the core
            ms.add_kv(LLM_KV_LOOP_START,     uint32_t(1));
            ms.add_kv(LLM_KV_LOOP_END,       uint32_t(2));
            ms.add_kv(LLM_KV_LOOP_COUNT,     uint32_t(2));
            ms.add_kv(LLM_KV_LOOP_GATE_RANK, uint32_t(8));
            ms.add_kv(LLM_KV_LOOP_VARYING,   false);
        }
        ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(4));
        ms.add_kv(LLM_KV_VOCAB_SIZE,                  n_vocab + 2);
        ms.add_kv(LLM_KV_VOCAB_SIZE_UNPADDED,         n_vocab);
        ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        0.2f);
        ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 0.3f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,        8.0f);
        ms.add_kv(LLM_KV_FINAL_LOGIT_SOFTCAPPING,     30.0f);
        ms.add_kv(LLM_KV_EXPERT_COUNT,                uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,           uint32_t(2));
        std::vector<bool> pattern(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            pattern[i] = i % 2 == 0;
        }
        std::vector<uint8_t> values(pattern.begin(), pattern.end());
        gguf_set_arr_data(ret.get(), llama_arch_kv_name(arch, LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN).c_str(), GGUF_TYPE_BOOL, values.data(), values.size());
    }

    // dummy tokenizer: token ids are derived from fixed-size chunks and detokenized as hex ids
    {
        std::vector<std::string> tokenizer_list(n_vocab);
        std::vector<float>       tokenizer_scores(n_vocab, 0.0f);

        ms.add_kv(LLM_KV_TOKENIZER_MODEL,         "test");
        for (uint32_t i = 0; i < n_vocab; i++) {
            tokenizer_list[i] = "tok_" + std::to_string(i);
        }
        ms.add_kv(LLM_KV_TOKENIZER_LIST,   tokenizer_list);
        ms.add_kv(LLM_KV_TOKENIZER_SCORES, tokenizer_scores);
    }

    // ms.add_kv(LLM_KV_DENSE_2_FEAT_OUT,     n_embd);
    // ms.add_kv(LLM_KV_DENSE_3_FEAT_IN,      n_embd);

    if (moe) {
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
        ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, n_ff / 2);  // distinct from n_ff so a saver key-clobber surfaces on reload
        ms.add_kv(LLM_KV_EXPERT_LATENT_LENGTH,       n_ff);
        ms.add_kv(LLM_KV_INTERLEAVE_MOE_LAYER_STEP,  uint32_t(2));
        // with more experts than a ubatch uses, the copy of the used experts in host memory skips some of them
        ms.add_kv(LLM_KV_EXPERT_COUNT,               uint32_t(host_experts_test(arch) ? 64 : arch == LLM_ARCH_CLOVER_MOE ? 4 : 2));
        ms.add_kv(LLM_KV_EXPERT_USED_COUNT,          uint32_t(arch == LLM_ARCH_ZAYA1 ? 1 : 2)); // ZAYA routes top-1
        ms.add_kv(LLM_KV_EXPERT_SHARED_COUNT,        uint32_t(1));

        uint32_t gating_func = 2;
        switch (arch) {
            case LLM_ARCH_MELLUM:
                gating_func = 1;
                break;
            case LLM_ARCH_DEEPSEEK4:
            case LLM_ARCH_DEEPSEEK41:
                gating_func = 4;
                break;
            default:
                break;
        }

        ms.add_kv(LLM_KV_EXPERT_GATING_FUNC,         gating_func); // sigmoid : softmax : sqrtsoftplus
        ms.add_kv(LLM_KV_EXPERT_GROUP_SCALE,         1.0f);
        ms.add_kv(LLM_KV_EXPERTS_PER_GROUP,          uint32_t(1));
        if (arch == LLM_ARCH_FLINT) {
            ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM, true);
        }
        if (arch == LLM_ARCH_K2_HORIZON) {
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_COUNT,      uint32_t(2));
            ms.add_kv(LLM_KV_ATTENTION_VALUE_EXPERT_USED_COUNT, uint32_t(2));
        }
    }

    if (arch == LLM_ARCH_BEAM) {
        ms.add_kv(LLM_KV_EXPERT_COUNT,            uint32_t(8));
        ms.add_kv(LLM_KV_EXPERT_GROUP_COUNT,      uint32_t(4));
        ms.add_kv(LLM_KV_EXPERT_GROUP_USED_COUNT, uint32_t(2));
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_SCALE,    2.5f);
        ms.add_kv(LLM_KV_EXPERT_WEIGHTS_NORM,     true);
    }

    if (arch == LLM_ARCH_MAION_CODER) {
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE, n_embd / 2);
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, std::vector<uint32_t>{n_ff, n_ff / 2, n_ff, n_ff});
    }

    if (arch == LLM_ARCH_ZONOS3) {
        ms.add_kv(LLM_KV_MOE_LATENT_SIZE,            n_embd / 2);
        ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff);
    }

    ms.add_kv(LLM_KV_POSNET_EMBEDDING_LENGTH,   n_embd);
    ms.add_kv(LLM_KV_POSNET_BLOCK_COUNT,        n_layer);
    ms.add_kv(LLM_KV_CONVNEXT_EMBEDDING_LENGTH, n_embd);
    ms.add_kv(LLM_KV_CONVNEXT_BLOCK_COUNT,      n_layer);
    ms.add_kv(LLM_KV_XIELU_ALPHA_N,             1.0f);
    ms.add_kv(LLM_KV_XIELU_ALPHA_P,             1.0f);
    ms.add_kv(LLM_KV_XIELU_BETA,                1.0f);
    ms.add_kv(LLM_KV_XIELU_EPS,                 1.0e-7f);
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,            arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_CLOVER_MOE || arch == LLM_ARCH_ZONOS3 ? 256 : 2*n_embd);
    ms.add_kv(LLM_KV_ROUTER_HIDDEN_SIZE,        n_embd);
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,           uint32_t(arch == LLM_ARCH_ZAYA1 ? 2 : 4)); // ZAYA CCA convs have kernel 2
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,            uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,        n_head);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,           arch == LLM_ARCH_PLAMO2 ? 0 : uint32_t(2));
    ms.add_kv(LLM_KV_KDA_HEAD_DIM,              uint32_t(128));
    ms.add_kv(LLM_KV_KDA_SAFE_GATE,             true);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,      -5.0f);
    if (arch == LLM_ARCH_BAILINGMOE3) {
        std::vector<float> clamp_exp;
        std::vector<float> clamp_shexp;
        for (uint32_t il = 0; il < n_layer; il++) {
            clamp_exp  .push_back(il % 2 ? 4.0f : 0.0f);
            clamp_shexp.push_back(il % 2 ? 5.0f : 0.0f);
        }
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_EXP,   clamp_exp);
        ms.add_kv(LLM_KV_SWIGLU_CLAMP_SHEXP, clamp_shexp);
    }
    ms.add_kv(LLM_KV_WKV_HEAD_SIZE,               n_embd/n_head);
    ms.add_kv(LLM_KV_SHORTCONV_L_CACHE,           uint32_t(3));
    ms.add_kv(LLM_KV_RESIDUAL_SCALE,              3.5565588200778455f);
    // the inkling2_loop core boundaries must be multiples of the block size
    ms.add_kv(LLM_KV_ATTN_RES_BLOCK_SIZE,         uint32_t(arch == LLM_ARCH_INKLING2_LOOP ? 1 : 12));
    ms.add_kv(LLM_KV_ACTIVATION_SITU_BETA,        4.0f);
    ms.add_kv(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, 25.0f);
    ms.add_kv(LLM_KV_KDA_GATE_LOWER_BOUND,        -5.0f);

    for (uint32_t il = 0; il < n_layer; il++) {
        ggml_tensor t;
        memset(&t, 0, sizeof(ggml_tensor));
        t.type = GGML_TYPE_F16;
        ggml_format_name(&t, "conv%" PRIu32 "d.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv1.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "posnet.%" PRIu32 ".conv2.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
        ggml_format_name(&t, "convnext.%" PRIu32 ".dw.weight", il);
        gguf_add_tensor(ms.gguf_ctx, &t);
    }
    return ret;
}

static bool silent_model_load_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

static std::string get_tmp_gguf_path() {
    static size_t id = 0;
    return (std::filesystem::temp_directory_path() / ("test-llama-archs-" + std::to_string(++id) + ".gguf")).string();
}

static llama_model_ptr get_model(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr) {
    GGML_ASSERT((gguf_ctx == nullptr) != (file == nullptr));
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs_copy = devs;
    devs_copy.push_back(nullptr);
    model_params.devices = devs_copy.data();
    model_params.split_mode = split_mode;
    model_params.tensor_buft_overrides = tensor_buft_overrides;

    tensor_data_params tensor_params = { seed, stdev };
    llama_model_ptr model(gguf_ctx != nullptr ?
        llama_model_init_from_user(gguf_ctx, set_tensor_data, &tensor_params, model_params) :
        llama_model_load_from_file_ptr(file, model_params));
    if (!model) {
        throw std::runtime_error("failed to create llama model");
    }
    return model;
}

static std::pair<llama_model_ptr, llama_context_ptr> get_model_and_ctx(
        struct gguf_context * gguf_ctx, FILE * file, const size_t seed, const float stdev,
        const std::vector<ggml_backend_dev_t> & devs,
        const llama_split_mode split_mode = LLAMA_SPLIT_MODE_LAYER, bool encode = false,
        const llama_model_tensor_buft_override * tensor_buft_overrides = nullptr, const size_t moe_cache_size = 0) {
    llama_model_ptr model = get_model(gguf_ctx, file, seed, stdev, devs, split_mode, tensor_buft_overrides);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    if (!encode) {
        ctx_params.n_ubatch = 64;
    }
    if (moe_cache_size > 0) {
        // the MoE cache is only used for small ubatches
        ctx_params.moe_cache_size = moe_cache_size;
        ctx_params.n_ubatch = 2;
    }

    llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }
    return std::make_pair(std::move(model), std::move(lctx));
}

static bool phase_workspace_fail_alloc = false;
static bool phase_workspace_fail_once = false;
static int phase_workspace_alloc_failures = 0;
static ggml_backend_buffer_t (*phase_workspace_alloc_buffer)(ggml_backend_buffer_type_t, size_t) = nullptr;
static ggml_backend_t phase_workspace_sync_backends[2] = {};
static void (*phase_workspace_synchronize[2])(ggml_backend_t) = {};
static int phase_workspace_sync_calls[2] = {};

static ggml_backend_buffer_t phase_workspace_test_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (phase_workspace_fail_alloc) {
        phase_workspace_alloc_failures++;
        if (phase_workspace_fail_once) {
            phase_workspace_fail_alloc = false;
        }
        return nullptr;
    }
    return phase_workspace_alloc_buffer(buft, size);
}

static ggml_backend_t phase_workspace_get_buffer_backend(ggml_backend_sched_t sched) {
    ggml_backend_t result = nullptr;
    size_t result_size = 0;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        const size_t size = ggml_backend_sched_get_buffer_size(sched, backend);
        if (size > result_size) {
            result = backend;
            result_size = size;
        }
    }
    return result;
}

static void phase_workspace_count_synchronize(ggml_backend_t backend) {
    for (int i = 0; i < 2; ++i) {
        if (phase_workspace_sync_backends[i] == backend) {
            phase_workspace_sync_calls[i]++;
            if (phase_workspace_synchronize[i] != nullptr) {
                phase_workspace_synchronize[i](backend);
            }
            return;
        }
    }
}

struct mtp_finish_observation {
    int32_t n_embd;
    bool active = false;
    size_t n_batched_eh_proj = 0;
};

static bool observe_mtp_finish(ggml_tensor * tensor, bool ask, void * user_data) {
    auto * observation = static_cast<mtp_finish_observation *>(user_data);
    if (observation->active && ask && strcmp(tensor->name, "mtp_eh_proj-2") == 0 &&
            tensor->ne[0] == observation->n_embd && tensor->ne[1] == 2 && tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        observation->n_batched_eh_proj++;
    }
    return false;
}

static llama_context_ptr make_phase_workspace_context(
        llama_model * model, llama_context_type type, llama_context * other = nullptr, uint32_t n_seq_max = 1,
        ggml_backend_sched_eval_callback cb_eval = nullptr, void * cb_eval_user_data = nullptr);

static llama_model_ptr make_live_context_workspace_model(llm_arch arch, size_t seed, bool moe = false) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);
    return model;
}

static llama_context_params make_live_context_workspace_params() {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 64;
    ctx_params.n_ubatch = 64;
    ctx_params.n_seq_max = 1;
    ctx_params.n_outputs_max = 1;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx_params.live_context_workspace = true;
    return ctx_params;
}

static void test_live_context_workspace_reserve(size_t seed) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_LLAMA, seed);
    llama_context_params ctx_params = make_live_context_workspace_params();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    GGML_ASSERT(!ctx->get_cparams().phase_aware_workspace);

    ggml_backend_sched_t sched = ctx->get_sched();
    ggml_backend_t backend_cpu = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU) {
            backend_cpu = backend;
            break;
        }
    }
    GGML_ASSERT(backend_cpu != nullptr);

    const auto initial = ctx->make_sched_reserve_plan(1, 1);
    GGML_ASSERT(initial.live_kv);
    GGML_ASSERT(initial.n_kv_capacity == ctx_params.n_ctx);
    GGML_ASSERT(initial.n_kv == 256);
    const size_t size_256 = ggml_backend_sched_get_buffer_size(sched, backend_cpu);

    ctx->sched_reserve(1, 257);
    const size_t size_512 = ggml_backend_sched_get_buffer_size(sched, backend_cpu);
    GGML_ASSERT(size_512 > size_256);

    const auto half_bin = ctx->make_sched_reserve_plan(1, 256);
    GGML_ASSERT(half_bin.n_kv == 512);
    ctx->sched_reserve(1, 256);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_cpu) == size_512);

    ctx->sched_reserve(1, 513);
    const size_t size_1024 = ggml_backend_sched_get_buffer_size(sched, backend_cpu);
    GGML_ASSERT(size_1024 > size_512);

    const auto contraction = ctx->make_sched_reserve_plan(1, 256);
    GGML_ASSERT(contraction.n_kv == 256);
    ctx->sched_reserve(1, 256);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_cpu) == size_256);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const std::vector<llama_token> tokens = get_tokens(10 * ctx_params.n_batch, n_vocab, seed);
    int32_t pos = 0;
    for (int32_t chunk = 0; chunk < 9; ++chunk) {
        llama_batch batch = llama_batch_init(ctx_params.n_batch, 0, 1);
        for (int32_t i = 0; i < (int32_t) ctx_params.n_batch; ++i) {
            test_batch_add(batch, tokens[pos], pos, { 0 }, i + 1 == (int32_t) ctx_params.n_batch);
            ++pos;
        }
        GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
        llama_synchronize(ctx.get());
        llama_batch_free(batch);
    }

    const auto runtime_high = ctx->make_sched_reserve_plan(ctx_params.n_batch);
    GGML_ASSERT(runtime_high.n_tokens == ctx_params.n_batch);
    GGML_ASSERT(runtime_high.n_kv == 1024);

    GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(ctx.get()), 0, 0, 512));
    llama_memory_seq_add(llama_get_memory(ctx.get()), 0, 512, pos, -512);

    llama_batch shifted = llama_batch_init(1, 0, 1);
    test_batch_add(shifted, tokens[pos], pos - 512, { 0 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), shifted) == 0);
    llama_synchronize(ctx.get());
    llama_batch_free(shifted);

    // Logical positions are compacted, but live rows at the physical tail still require the high reserve.
    const auto physical_high = ctx->make_sched_reserve_plan(0);
    GGML_ASSERT(physical_high.n_tokens == ctx_params.n_batch);
    GGML_ASSERT(physical_high.n_kv == 1024);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_cpu) > size_256);

    llama_memory_clear(llama_get_memory(ctx.get()), false);
    llama_batch short_batch = llama_batch_init(1, 0, 1);
    test_batch_add(short_batch, tokens[0], 0, { 0 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), short_batch) == 0);
    llama_synchronize(ctx.get());
    llama_batch_free(short_batch);

    const auto runtime_contraction = ctx->make_sched_reserve_plan(0);
    GGML_ASSERT(runtime_contraction.n_tokens == ctx_params.n_batch);
    GGML_ASSERT(runtime_contraction.n_kv == 256);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_cpu) == size_256);
    GGML_ASSERT(llama_trim_transient_memory(ctx.get()) == 0);
}

static void test_live_context_workspace_iswa_reserve(size_t seed) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_GEMMA4, seed);
    llama_context_params ctx_params = make_live_context_workspace_params();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    GGML_ASSERT(!ctx->get_cparams().phase_aware_workspace);

    const auto initial = ctx->make_sched_reserve_plan(1, 1);
    GGML_ASSERT(initial.live_kv);
    GGML_ASSERT(initial.n_kv_capacity == ctx_params.n_ctx);
    GGML_ASSERT(initial.n_kv == 256);

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const std::vector<llama_token> tokens = get_tokens(9 * ctx_params.n_batch, n_vocab, seed);
    int32_t pos = 0;
    for (int32_t chunk = 0; chunk < 9; ++chunk) {
        llama_batch batch = llama_batch_init(ctx_params.n_batch, 0, 1);
        for (int32_t i = 0; i < (int32_t) ctx_params.n_batch; ++i) {
            test_batch_add(batch, tokens[pos], pos, { 0 }, i + 1 == (int32_t) ctx_params.n_batch);
            ++pos;
        }
        GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
        llama_synchronize(ctx.get());
        llama_batch_free(batch);
    }

    const auto runtime = ctx->make_sched_reserve_plan(ctx_params.n_batch);
    GGML_ASSERT(runtime.n_kv == 1024);

    llama_memory_clear(llama_get_memory(ctx.get()), false);
    llama_batch short_batch = llama_batch_init(1, 0, 1);
    test_batch_add(short_batch, tokens[0], 0, { 0 }, true);
    GGML_ASSERT(llama_decode(ctx.get(), short_batch) == 0);
    llama_synchronize(ctx.get());
    llama_batch_free(short_batch);

    const auto contraction = ctx->make_sched_reserve_plan(0);
    GGML_ASSERT(contraction.n_kv == 256);
}

static void test_live_context_workspace_indexer_reserve(size_t seed) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_QWEN4EXP, seed, true);
    llama_context_params ctx_params = make_live_context_workspace_params();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);

    const auto initial = ctx->make_sched_reserve_plan(1, 1);
    GGML_ASSERT(initial.live_kv);
    GGML_ASSERT(initial.n_kv == 256);

    auto assert_reserve = [&](uint32_t n_kv) {
        llama_memory_context_ptr reserve = ctx->get_memory()->init_reserve(n_kv);
        auto * hybrid = dynamic_cast<llama_memory_hybrid_idx_context *>(reserve.get());
        GGML_ASSERT(hybrid != nullptr);
        GGML_ASSERT(hybrid->get_attn()->get_n_kv() == n_kv);
        GGML_ASSERT(hybrid->get_idx() != nullptr);
        GGML_ASSERT(hybrid->get_idx()->get_n_kv() == n_kv);
    };
    assert_reserve(256);

    ctx->sched_reserve(ctx_params.n_batch, 257);
    const auto grown = ctx->make_sched_reserve_plan(ctx_params.n_batch, 257);
    GGML_ASSERT(grown.n_kv == 512);
    assert_reserve(512);
}

static void test_live_context_workspace_unsupported(size_t seed) {
    llama_model_ptr model = make_live_context_workspace_model(LLM_ARCH_MAMBA, seed);
    llama_context_params ctx_params = make_live_context_workspace_params();

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    GGML_ASSERT(!ctx->get_cparams().live_context_workspace);
}

static void test_phase_workspace_runtime_reserve(size_t seed) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);

    phase_workspace_fail_alloc = false;
    phase_workspace_fail_once = false;
    phase_workspace_alloc_failures = 0;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 32;
    ctx_params.n_batch = 32;
    ctx_params.n_ubatch = 32;
    ctx_params.n_seq_max = 1;
    ctx_params.n_outputs_max = 0;
    ctx_params.n_threads = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.no_perf = false;
    ctx_params.phase_aware_workspace = true;

    llama_context_ptr ctx(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    GGML_ASSERT(llama_n_batch(ctx.get()) == ctx_params.n_batch);
    GGML_ASSERT(ctx->get_cparams().n_outputs_max == ctx_params.n_batch);

    ggml_backend_sched_t sched = ctx->get_sched();
    ggml_backend_t backend_workspace = phase_workspace_get_buffer_backend(sched);
    GGML_ASSERT(backend_workspace != nullptr);
    ggml_backend_buffer_type_t workspace_buft = ggml_backend_sched_get_buffer_type(sched, backend_workspace);
    phase_workspace_alloc_buffer = workspace_buft->iface.alloc_buffer;
    workspace_buft->iface.alloc_buffer = phase_workspace_test_alloc_buffer;
    const size_t size_small = ggml_backend_sched_get_buffer_size(sched, backend_workspace);

    llama_batch batch = llama_batch_init(16, 0, 1);
    const std::vector<llama_token> tokens = get_tokens(16, llama_vocab_n_tokens(llama_model_get_vocab(model.get())), seed);
    for (int32_t i = 0; i < 16; ++i) {
        test_batch_add(batch, tokens[i], i, { 0 }, true);
    }

    phase_workspace_fail_alloc = true;
    GGML_ASSERT(llama_decode(ctx.get(), batch) == -2);
    GGML_ASSERT(phase_workspace_alloc_failures > 0);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_workspace) == 0);

    phase_workspace_fail_alloc = false;
    const int64_t retry_start_us = ggml_time_us();
    GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
    llama_synchronize(ctx.get());
    const int64_t retry_elapsed_us = ggml_time_us() - retry_start_us;
    const llama_perf_context_data perf = llama_perf_context(ctx.get());
    GGML_ASSERT(perf.n_p_eval == 16);
    GGML_ASSERT(perf.t_p_eval_ms >= 0.0);
    GGML_ASSERT(perf.t_p_eval_ms <= retry_elapsed_us*1e-3 + 0.001);
    GGML_ASSERT(llama_get_logits_ith(ctx.get(), 0) != nullptr);
    GGML_ASSERT(llama_get_logits_ith(ctx.get(), 15) != nullptr);
    const size_t size_grown = ggml_backend_sched_get_buffer_size(sched, backend_workspace);
    GGML_ASSERT(size_grown > size_small);

    ctx->sched_reserve(ctx_params.n_ubatch);
    const size_t size_prompt = ggml_backend_sched_get_buffer_size(sched, backend_workspace);
    GGML_ASSERT(size_prompt == size_grown);

    llama_batch_free(batch);
    ctx.reset();

    ctx_params.n_outputs_max = 4;
    ctx.reset(llama_init_from_model(model.get(), ctx_params));
    GGML_ASSERT(ctx);
    sched = ctx->get_sched();
    backend_workspace = phase_workspace_get_buffer_backend(sched);
    GGML_ASSERT(backend_workspace != nullptr);
    const size_t size_declared = ggml_backend_sched_get_buffer_size(sched, backend_workspace);

    int32_t pos = 0;
    for (int32_t width = 1; width <= 4; width *= 2) {
        llama_batch declared = llama_batch_init(width, 0, 1);
        for (int32_t i = 0; i < width; ++i) {
            test_batch_add(declared, tokens[i], pos++, { 0 }, true);
        }
        GGML_ASSERT(llama_decode(ctx.get(), declared) == 0);
        llama_synchronize(ctx.get());
        GGML_ASSERT(ggml_backend_sched_get_buffer_size(sched, backend_workspace) == size_declared);
        llama_batch_free(declared);
    }

    ctx.reset();
    workspace_buft->iface.alloc_buffer = phase_workspace_alloc_buffer;
    phase_workspace_alloc_buffer = nullptr;
}

static void test_phase_workspace_late_pipeline_fallback(size_t seed) {
    std::vector<ggml_backend_dev_t> devices;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (props.caps.async && props.caps.events) {
            devices.push_back(dev);
        }
    }
    if (devices.size() < 2) {
        printf("test_phase_workspace_late_pipeline_fallback: skipped, two async GPU devices required\n");
        return;
    }

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);
    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.n_gpu_layers = 99;
    ggml_backend_dev_t model_devices[] = { devices[0], devices[1], nullptr };
    model_params.devices = model_devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);

    llama_context_ptr ctx = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    GGML_ASSERT(ctx);
    GGML_ASSERT(ggml_backend_sched_get_n_copies(ctx->get_sched()) > 1);

    ggml_backend_buffer_type_t gpu_buft = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(ctx->get_sched()); ++i) {
        ggml_backend_t backend = ggml_backend_sched_get_backend(ctx->get_sched(), i);
        if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu_buft = ggml_backend_get_default_buffer_type(backend);
            break;
        }
    }
    GGML_ASSERT(gpu_buft != nullptr);

    phase_workspace_alloc_failures = 0;
    phase_workspace_alloc_buffer = gpu_buft->iface.alloc_buffer;
    phase_workspace_fail_once = true;
    phase_workspace_fail_alloc = true;
    gpu_buft->iface.alloc_buffer = phase_workspace_test_alloc_buffer;

    llama_batch batch = llama_batch_init(16, 0, 1);
    const std::vector<llama_token> tokens = get_tokens(16,
            llama_vocab_n_tokens(llama_model_get_vocab(model.get())), seed);
    for (int32_t i = 0; i < 16; ++i) {
        test_batch_add(batch, tokens[i], i, { 0 }, i == 15);
    }
    GGML_ASSERT(llama_decode(ctx.get(), batch) == 0);
    llama_synchronize(ctx.get());
    GGML_ASSERT(phase_workspace_alloc_failures == 1);
    GGML_ASSERT(ggml_backend_sched_get_n_copies(ctx->get_sched()) == 1);

    gpu_buft->iface.alloc_buffer = phase_workspace_alloc_buffer;
    phase_workspace_alloc_buffer = nullptr;
    phase_workspace_fail_alloc = false;
    phase_workspace_fail_once = false;
    llama_batch_free(batch);
}

// fork: mirror a legacy llama_batch into a common_batch for the common_speculative API
static common_batch to_common_batch(llama_context * ctx, const llama_batch & b) {
    common_batch res(ctx);
    for (int32_t i = 0; i < b.n_tokens; ++i) {
        const std::vector<llama_seq_id> seq_ids(b.seq_id[i], b.seq_id[i] + b.n_seq_id[i]);
        res.add(b.token[i], b.pos[i], seq_ids, b.logits[i] != 0);
    }
    return res;
}

static llama_batch make_mtp_batch(
        int32_t n_tokens, int32_t n_embd, int32_t pos, uint32_t n_vocab, size_t seed) {
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.token = (llama_token *) malloc(sizeof(llama_token) * n_tokens);
    GGML_ASSERT(batch.token != nullptr);

    const std::vector<llama_token> tokens = get_tokens(n_tokens, n_vocab, seed);
    for (int32_t i = 0; i < n_tokens; ++i) {
        test_batch_add(batch, tokens[i], pos + i, { 0 }, i == n_tokens - 1);
        for (int32_t j = 0; j < n_embd; ++j) {
            batch.embd[(size_t) i * n_embd + j] = (float) ((i + j) % 17) / 17.0f;
        }
    }
    return batch;
}

static llama_context_ptr make_phase_workspace_context(
        llama_model * model, llama_context_type type, llama_context * other, uint32_t n_seq_max,
        ggml_backend_sched_eval_callback cb_eval, void * cb_eval_user_data) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 32;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_seq_max = n_seq_max;
    params.n_outputs_max = type == LLAMA_CONTEXT_TYPE_MTP ? n_seq_max : 4;
    params.n_threads = 4;
    params.n_threads_batch = 4;
    params.ctx_type = type;
    params.ctx_other = other;
    params.phase_aware_workspace = true;
    params.cb_eval = cb_eval;
    params.cb_eval_user_data = cb_eval_user_data;
    return llama_context_ptr(llama_init_from_model(model, params));
}

static void test_speculative_limits(size_t seed) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false, true);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);

    const uint32_t n_seq = 2;
    llama_context_ptr target = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT, nullptr, n_seq);
    llama_context_ptr draft = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get(), n_seq);
    GGML_ASSERT(target && draft);

    const llama_tokens prompt_match = { 99, 1, 2, 3, 4, 1 };
    const llama_tokens prompt_none  = { 99, 1, 2, 3, 4, 5 };

    auto make_spec = [&](int32_t n_max, int32_t n_min, std::vector<common_speculative_type> types) {
        common_params_speculative params;
        params.types = std::move(types);
        params.draft.n_max = n_max;
        params.draft.n_min = n_min;
        params.draft.p_min = 0.0f;
        params.draft.ctx_tgt = target.get();
        params.draft.ctx_dft = draft.get();
        params.ngram_simple.size_n = 2;
        params.ngram_simple.size_m = 2;
        return common_speculative_ptr(common_speculative_init(params, n_seq));
    };

    {
        auto spec = make_spec(2, 0, { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, COMMON_SPECULATIVE_TYPE_DRAFT_MTP });
        llama_tokens result_zero;
        llama_tokens result_positive;
        common_speculative_get_draft_params(spec.get(), 0) = {
            true, 0, 0, 2, &prompt_match, &result_zero,
        };
        common_speculative_get_draft_params(spec.get(), 1) = {
            true, 1, 0, 6, &prompt_none, &result_positive,
        };

        common_speculative_draft(spec.get());

        GGML_ASSERT(result_zero.empty());
        GGML_ASSERT(result_positive.size() == 1);
        GGML_ASSERT(!common_speculative_get_draft_params(spec.get(), 0).drafting);
        GGML_ASSERT(!common_speculative_get_draft_params(spec.get(), 1).drafting);
        GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 0) == -1);
        GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 1) == 0);
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 1, 0, -1));
    }

    {
        auto spec = make_spec(0, 0, { COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, COMMON_SPECULATIVE_TYPE_DRAFT_MTP });
        llama_tokens result_ngram;
        llama_tokens result_none;
        common_speculative_get_draft_params(spec.get(), 0) = {
            true, -1, 0, 2, &prompt_match, &result_ngram,
        };
        common_speculative_get_draft_params(spec.get(), 1) = {
            true, -1, 0, 6, &prompt_none, &result_none,
        };

        common_speculative_draft(spec.get());

        GGML_ASSERT(result_ngram == llama_tokens({ 3, 4 }));
        GGML_ASSERT(result_none.empty());
        GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 0) == -1);
        GGML_ASSERT(llama_memory_seq_pos_max(llama_get_memory(draft.get()), 1) == -1);
    }

    {
        auto spec = make_spec(2, 0, { COMMON_SPECULATIVE_TYPE_DRAFT_MTP });
        llama_tokens result_short;
        llama_tokens result_long;
        common_speculative_get_draft_params(spec.get(), 0) = {
            true, 1, 0, 2, &prompt_none, &result_short,
        };
        common_speculative_get_draft_params(spec.get(), 1) = {
            true, -1, 0, 6, &prompt_none, &result_long,
        };

        common_speculative_draft(spec.get());

        GGML_ASSERT(result_short.size() == 1);
        GGML_ASSERT(result_long.size() == 2);
        GGML_ASSERT(!common_speculative_retain_draft_state(spec.get(), 1));
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 0, -1));
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 1, 0, -1));
    }

    {
        auto spec = make_spec(2, 2, { COMMON_SPECULATIVE_TYPE_DRAFT_MTP });
        llama_tokens result;
        common_speculative_get_draft_params(spec.get(), 0) = {
            true, 1, 0, 2, &prompt_none, &result,
        };

        common_speculative_draft(spec.get());

        GGML_ASSERT(result.empty());
        GGML_ASSERT(llama_memory_seq_rm(llama_get_memory(draft.get()), 0, 0, -1));
    }

    {
        mtp_finish_observation observation = { llama_model_n_embd_out(model.get()) };
        llama_context_ptr target_retained = make_phase_workspace_context(
                model.get(), LLAMA_CONTEXT_TYPE_DEFAULT, nullptr, n_seq);
        llama_context_ptr draft_retained = make_phase_workspace_context(
                model.get(), LLAMA_CONTEXT_TYPE_MTP, nullptr, n_seq, observe_mtp_finish, &observation);
        GGML_ASSERT(target_retained && draft_retained);
        GGML_ASSERT(llama_get_ctx_other(draft_retained.get()) == nullptr);

        common_params_speculative params;
        params.types = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        params.draft.n_max = 1;
        params.draft.n_min = 0;
        params.draft.p_min = 0.0f;
        params.draft.ctx_tgt = target_retained.get();
        params.draft.ctx_dft = draft_retained.get();
        common_speculative_ptr spec(common_speculative_init(params, n_seq));
        GGML_ASSERT(spec);

        llama_tokens proposals[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            common_speculative_get_draft_params(spec.get(), seq_id) = {
                true, 1, 0, 2 + seq_id, &prompt_none, &proposals[seq_id],
            };
        }
        common_speculative_draft(spec.get());
        GGML_ASSERT(proposals[0].size() == 1 && proposals[1].size() == 1);

        auto get_seq_state = [](llama_context * ctx, llama_seq_id seq_id) {
            const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);
            GGML_ASSERT(size > 0);
            std::vector<uint8_t> data(size);
            GGML_ASSERT(llama_state_seq_get_data_ext(
                    ctx, data.data(), data.size(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == size);
            return data;
        };
        auto get_mtp_state = [](common_speculative * spec, llama_seq_id seq_id) {
            std::vector<uint8_t> data;
            GGML_ASSERT(common_speculative_get_mtp_state(spec, seq_id, data));
            return data;
        };

        std::vector<uint8_t> proposal_state[2];
        std::vector<uint8_t> pending_initial[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            proposal_state[seq_id] = get_seq_state(draft_retained.get(), seq_id);
            pending_initial[seq_id] = get_mtp_state(spec.get(), seq_id);
        }

        llama_batch verify = llama_batch_init(4, 0, 1);
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            test_batch_add(verify, 2 + seq_id, 0, { seq_id }, true);
            test_batch_add(verify, proposals[seq_id][0], 1, { seq_id }, true);
        }
        GGML_ASSERT(llama_decode(target_retained.get(), verify) == 0);
        llama_synchronize(target_retained.get());

        auto * draft_mem = llama_get_memory(draft_retained.get());
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
        }
        GGML_ASSERT(common_speculative_process(spec.get(), to_common_batch(target_retained.get(), verify)));

        std::vector<uint8_t> canonical_accepted_state[2];
        std::vector<uint8_t> canonical_accepted_pending[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            canonical_accepted_state[seq_id] = get_seq_state(draft_retained.get(), seq_id);
            common_speculative_accept(spec.get(), seq_id, 1);
            canonical_accepted_pending[seq_id] = get_mtp_state(spec.get(), seq_id);
        }

        common_speculative_accept(spec.get(), 0, 0);
        GGML_ASSERT(llama_memory_seq_rm(draft_mem, 0, 1, -1));
        GGML_ASSERT(llama_memory_seq_rm(draft_mem, 1, 2, -1));

        std::vector<uint8_t> canonical_state[2];
        std::vector<uint8_t> canonical_pending[2];
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            canonical_state[seq_id] = get_seq_state(draft_retained.get(), seq_id);
            canonical_pending[seq_id] = get_mtp_state(spec.get(), seq_id);

            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(llama_state_seq_set_data_ext(
                    draft_retained.get(), proposal_state[seq_id].data(), proposal_state[seq_id].size(),
                    seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[seq_id].size());
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
            GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), seq_id));
        }

        GGML_ASSERT(common_speculative_process(spec.get(), to_common_batch(target_retained.get(), verify)));
        common_speculative_accept(spec.get(), 0, 0);
        common_speculative_accept(spec.get(), 1, 1);
        GGML_ASSERT(common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(common_speculative_has_deferred_accept(spec.get(), 1));
        GGML_ASSERT(common_speculative_finish_accept(spec.get(), {
            { 0, 0 },
            { 1, 1 },
        }));

        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), seq_id));
            GGML_ASSERT(get_seq_state(draft_retained.get(), seq_id) == canonical_state[seq_id]);
            GGML_ASSERT(get_mtp_state(spec.get(), seq_id) == canonical_pending[seq_id]);

            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(llama_state_seq_set_data_ext(
                    draft_retained.get(), proposal_state[seq_id].data(), proposal_state[seq_id].size(),
                    seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[seq_id].size());
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
            GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), seq_id));
        }

        GGML_ASSERT(common_speculative_process(spec.get(), to_common_batch(target_retained.get(), verify)));
        common_speculative_accept(spec.get(), 0, 1);
        common_speculative_accept(spec.get(), 1, 1);

        llama_synchronize(draft_retained.get());
        observation.active = true;
        GGML_ASSERT(common_speculative_finish_accept(spec.get(), {
            { 0, 1 },
            { 1, 1 },
        }));
        llama_synchronize(draft_retained.get());
        observation.active = false;
        GGML_ASSERT(observation.n_batched_eh_proj == 1);
        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), seq_id));
            GGML_ASSERT(get_seq_state(draft_retained.get(), seq_id) == canonical_accepted_state[seq_id]);
            GGML_ASSERT(get_mtp_state(spec.get(), seq_id) == canonical_accepted_pending[seq_id]);

            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(llama_state_seq_set_data_ext(
                    draft_retained.get(), proposal_state[seq_id].data(), proposal_state[seq_id].size(),
                    seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[seq_id].size());
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
            GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), seq_id));
        }

        GGML_ASSERT(common_speculative_process(spec.get(), to_common_batch(target_retained.get(), verify)));
        common_speculative_accept(spec.get(), 0, 1);
        common_speculative_accept(spec.get(), 1, 1);
        GGML_ASSERT(!common_speculative_finish_accept(spec.get(), {
            { 1, 2 },
            { 0, 1 },
        }));
        GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), 1));

        for (llama_seq_id seq_id = 0; seq_id < 2; ++seq_id) {
            GGML_ASSERT(llama_memory_seq_rm(draft_mem, seq_id, -1, -1));
            GGML_ASSERT(llama_state_seq_set_data_ext(
                    draft_retained.get(), proposal_state[seq_id].data(), proposal_state[seq_id].size(),
                    seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[seq_id].size());
            GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), seq_id, pending_initial[seq_id]));
            GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), seq_id));
        }

        GGML_ASSERT(common_speculative_process(spec.get(), to_common_batch(target_retained.get(), verify)));
        common_speculative_accept(spec.get(), 0, 1);
        common_speculative_accept(spec.get(), 1, 1);
        GGML_ASSERT(!common_speculative_finish_accept(spec.get(), {
            { 0, 2 },
        }));
        GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(common_speculative_has_deferred_accept(spec.get(), 1));
        GGML_ASSERT(common_speculative_finish_accept(spec.get(), {
            { 1, 0 },
        }));
        GGML_ASSERT(get_seq_state(draft_retained.get(), 1) == proposal_state[1]);

        GGML_ASSERT(llama_memory_seq_rm(draft_mem, 0, -1, -1));
        GGML_ASSERT(llama_state_seq_set_data_ext(
                draft_retained.get(), proposal_state[0].data(), proposal_state[0].size(),
                0, LLAMA_STATE_SEQ_FLAGS_NONE) == proposal_state[0].size());
        GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), 0, pending_initial[0]));
        GGML_ASSERT(common_speculative_retain_draft_state(spec.get(), 0));
        GGML_ASSERT(common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(common_speculative_set_mtp_state(spec.get(), 0, {}));
        GGML_ASSERT(!common_speculative_has_deferred_accept(spec.get(), 0));
        GGML_ASSERT(llama_memory_seq_rm(draft_mem, 0, -1, -1));

        llama_batch_free(verify);
    }
}

static void test_phase_workspace_mtp_lifecycle(size_t seed) {
    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false, true);

    llama_model_params model_params = llama_model_default_params();
    model_params.progress_callback = silent_model_load_progress;
    model_params.load_mtp = true;
    ggml_backend_dev_t devices[] = { nullptr };
    model_params.devices = devices;

    size_t tensor_seed = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tensor_seed, model_params));
    GGML_ASSERT(model);

    llama_context_ptr target = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    llama_context_ptr draft = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get());
    GGML_ASSERT(target && draft);
    GGML_ASSERT(!llama_contexts_share_workspace(target.get(), draft.get()));

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
    const int32_t n_embd = llama_model_n_embd_out(model.get());
    llama_batch target_batch = llama_batch_init(4, 0, 1);
    const std::vector<llama_token> tokens = get_tokens(4, n_vocab, seed);
    for (int32_t i = 0; i < 4; ++i) {
        test_batch_add(target_batch, tokens[i], i, { 0 }, true);
    }
    llama_batch draft_batch = make_mtp_batch(4, n_embd, 0, n_vocab, seed + 1);

    GGML_ASSERT(llama_decode(target.get(), target_batch) == 0);
    GGML_ASSERT(llama_decode(draft.get(), draft_batch) == 0);
    llama_synchronize(draft.get());
    llama_synchronize(target.get());
    llama_batch_free(draft_batch);
    llama_batch_free(target_batch);

    ggml_backend_t target_workspace = phase_workspace_get_buffer_backend(target->get_sched());
    ggml_backend_t draft_workspace = phase_workspace_get_buffer_backend(draft->get_sched());
    GGML_ASSERT(target_workspace && draft_workspace);
    ggml_backend_buffer_type_t workspace_buft = ggml_backend_sched_get_buffer_type(
            target->get_sched(), target_workspace);
    phase_workspace_alloc_failures = 0;
    phase_workspace_alloc_buffer = workspace_buft->iface.alloc_buffer;
    phase_workspace_fail_once = true;
    phase_workspace_fail_alloc = true;
    workspace_buft->iface.alloc_buffer = phase_workspace_test_alloc_buffer;
    GGML_ASSERT(llama_attach_shared_workspace(draft.get(), target.get()) == -1);
    GGML_ASSERT(phase_workspace_alloc_failures == 1);
    GGML_ASSERT(draft->get_sched() == nullptr);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace) == 0);
    workspace_buft->iface.alloc_buffer = phase_workspace_alloc_buffer;
    phase_workspace_alloc_buffer = nullptr;
    phase_workspace_fail_alloc = false;
    phase_workspace_fail_once = false;

    GGML_ASSERT(llama_attach_shared_workspace(draft.get(), target.get()) == 1);
    GGML_ASSERT(llama_contexts_share_workspace(target.get(), draft.get()));
    GGML_ASSERT(draft->make_sched_reserve_plan(4).n_tokens == 4);

    const size_t decode_size = ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace);
    GGML_ASSERT(decode_size > 0);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(draft->get_sched(), draft_workspace) == decode_size);
    draft->sched_reserve(4);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(draft->get_sched(), draft_workspace) == decode_size);

    phase_workspace_sync_backends[0] = target_workspace;
    phase_workspace_sync_backends[1] = draft_workspace;
    phase_workspace_synchronize[0] = target_workspace->iface.synchronize;
    phase_workspace_synchronize[1] = draft_workspace->iface.synchronize;
    phase_workspace_sync_calls[0] = 0;
    phase_workspace_sync_calls[1] = 0;
    target_workspace->iface.synchronize = phase_workspace_count_synchronize;
    draft_workspace->iface.synchronize = phase_workspace_count_synchronize;

    llama_batch target_catchup = llama_batch_init(1, 0, 1);
    test_batch_add(target_catchup, tokens[0], 4, { 0 }, true);
    llama_batch draft_catchup = make_mtp_batch(1, n_embd, 4, n_vocab, seed + 2);
    llama_batch target_reacquire = llama_batch_init(1, 0, 1);
    test_batch_add(target_reacquire, tokens[1], 5, { 0 }, true);
    GGML_ASSERT(llama_decode(target.get(), target_catchup) == 0);
    GGML_ASSERT(llama_decode(draft.get(), draft_catchup) == 0);
    GGML_ASSERT(phase_workspace_sync_calls[0] > 0);
    const int draft_syncs_before_reacquire = phase_workspace_sync_calls[1];
    GGML_ASSERT(llama_decode(target.get(), target_reacquire) == 0);
    GGML_ASSERT(phase_workspace_sync_calls[1] == draft_syncs_before_reacquire + 1);
    GGML_ASSERT(llama_get_logits_ith(target.get(), 0) != nullptr);
    target_workspace->iface.synchronize = phase_workspace_synchronize[0];
    draft_workspace->iface.synchronize = phase_workspace_synchronize[1];
    memset(phase_workspace_sync_backends, 0, sizeof(phase_workspace_sync_backends));
    memset(phase_workspace_synchronize, 0, sizeof(phase_workspace_synchronize));
    memset(phase_workspace_sync_calls, 0, sizeof(phase_workspace_sync_calls));
    llama_batch_free(target_reacquire);
    llama_batch_free(draft_catchup);
    llama_batch_free(target_catchup);

    llama_batch draft_failure = make_mtp_batch(1, n_embd, 5, n_vocab, seed + 3);
    GGML_ASSERT(llama_decode(draft.get(), draft_failure) == 0);
    phase_workspace_alloc_failures = 0;
    phase_workspace_alloc_buffer = workspace_buft->iface.alloc_buffer;
    phase_workspace_fail_once = true;
    phase_workspace_fail_alloc = true;
    workspace_buft->iface.alloc_buffer = phase_workspace_test_alloc_buffer;
    bool reserve_failed = false;
    try {
        target->sched_reserve(32);
    } catch (const std::exception &) {
        reserve_failed = true;
    }
    GGML_ASSERT(reserve_failed);
    GGML_ASSERT(phase_workspace_alloc_failures == 1);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace) == 0);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(draft->get_sched(), draft_workspace) == 0);
    workspace_buft->iface.alloc_buffer = phase_workspace_alloc_buffer;
    phase_workspace_alloc_buffer = nullptr;
    phase_workspace_fail_alloc = false;
    phase_workspace_fail_once = false;
    llama_batch_free(draft_failure);

    target->sched_reserve(32);
    const size_t prompt_size = ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace);
    GGML_ASSERT(prompt_size > decode_size);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(draft->get_sched(), draft_workspace) == prompt_size);

    target->sched_reserve(0);
    draft->sched_reserve(0);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace) == decode_size);
    target->sched_reserve(32);
    GGML_ASSERT(ggml_backend_sched_get_buffer_size(target->get_sched(), target_workspace) == prompt_size);

    draft.reset();
    draft = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get());
    GGML_ASSERT(draft);
    GGML_ASSERT(!llama_contexts_share_workspace(target.get(), draft.get()));
    GGML_ASSERT(llama_attach_shared_workspace(draft.get(), target.get()) == 1);

    llama_batch teardown_batch = make_mtp_batch(1, n_embd, 0, n_vocab, seed + 4);
    GGML_ASSERT(llama_decode(draft.get(), teardown_batch) == 0);
    target.reset();
    llama_batch_free(teardown_batch);
    draft->sched_reserve(0);

    target = make_phase_workspace_context(model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    GGML_ASSERT(target);
    GGML_ASSERT(llama_attach_shared_workspace(draft.get(), target.get()) == 1);
    target.reset();

    llama_batch retry_batch = make_mtp_batch(1, n_embd, 4, n_vocab, seed + 2);
    GGML_ASSERT(llama_decode(draft.get(), retry_batch) == 0);
    llama_synchronize(draft.get());
    llama_batch_free(retry_batch);
    draft.reset();
}

static void test_phase_workspace_mismatched_placement(size_t seed) {
    ggml_backend_dev_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = dev;
            break;
        }
    }
    if (gpu == nullptr) {
        printf("test_phase_workspace_mismatched_placement: skipped, GPU device required\n");
        return;
    }

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_QWEN35, false, true);
    llama_model_params target_params = llama_model_default_params();
    target_params.progress_callback = silent_model_load_progress;
    target_params.n_gpu_layers = 99;
    target_params.load_mtp = true;
    ggml_backend_dev_t target_devices[] = { gpu, nullptr };
    target_params.devices = target_devices;

    llama_model_params draft_params = llama_model_default_params();
    draft_params.progress_callback = silent_model_load_progress;
    draft_params.load_mtp = true;
    ggml_backend_dev_t draft_devices[] = { nullptr };
    draft_params.devices = draft_devices;

    size_t target_seed = seed;
    size_t draft_seed = seed;
    llama_model_ptr target_model(llama_model_init_from_user(
            gguf_ctx.get(), set_tensor_data, &target_seed, target_params));
    llama_model_ptr draft_model(llama_model_init_from_user(
            gguf_ctx.get(), set_tensor_data, &draft_seed, draft_params));
    GGML_ASSERT(target_model && draft_model);

    llama_context_ptr target = make_phase_workspace_context(target_model.get(), LLAMA_CONTEXT_TYPE_DEFAULT);
    llama_context_ptr draft = make_phase_workspace_context(
            draft_model.get(), LLAMA_CONTEXT_TYPE_MTP, target.get());
    GGML_ASSERT(target && draft);
    GGML_ASSERT(!llama_contexts_share_workspace(target.get(), draft.get()));

    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(target_model.get()));
    const int32_t n_embd = llama_model_n_embd_out(draft_model.get());
    llama_batch target_batch = llama_batch_init(1, 0, 1);
    test_batch_add(target_batch, get_tokens(1, n_vocab, seed)[0], 0, { 0 }, true);
    llama_batch draft_batch = make_mtp_batch(1, n_embd, 0, n_vocab, seed + 1);
    GGML_ASSERT(llama_decode(target.get(), target_batch) == 0);
    GGML_ASSERT(llama_decode(draft.get(), draft_batch) == 0);
    llama_synchronize(draft.get());
    llama_synchronize(target.get());
    llama_batch_free(draft_batch);
    llama_batch_free(target_batch);

    const int32_t status = llama_attach_shared_workspace(draft.get(), target.get());
    GGML_ASSERT(status >= 0);
    GGML_ASSERT(llama_contexts_share_workspace(target.get(), draft.get()) == (status == 1));
}

static std::vector<float> get_logits(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, bool encode = false) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_ctx    = llama_n_ctx(lctx);
    const uint32_t n_tokens = tokens.size();
    common_batch batch(lctx);
    GGML_ASSERT(n_tokens <= n_ctx);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        batch.add(tokens[pos], pos, 0, true);
    }
    if (encode) {
        if (llama_process(lctx, LLAMA_PROCESS_TYPE_ENCODE, batch.get())) {
            throw std::runtime_error("failed to encode batch");
        }
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> ret;
    ret.reserve(n_tokens*n_vocab);
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        for (uint32_t j = 0; j < n_vocab; j++) {
            ret.push_back(logits_ith[j]);
        }
    }
    return ret;
}

// decode two sequences in one batch, compare each with a decode of it alone
// logits_a: logits of tokens decoded alone with the same device config
static bool test_parallel_seqs(llama_model * model, const std::vector<llama_token> & tokens, const std::vector<float> & logits_a, bool encode) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_tokens = tokens.size();

    // same per-sequence n_ctx as the reference context
    const auto make_cparams = [&](uint32_t n_seq_max) {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx           = n_seq_max*llama_model_n_ctx_train(model);
        cparams.n_threads       = 4;
        cparams.n_threads_batch = 4;
        cparams.n_seq_max       = n_seq_max;
        if (!encode) {
            cparams.n_ubatch = 64;
        }
        return cparams;
    };

    llama_context_ptr ctx_par(llama_init_from_model(model, make_cparams(2)));
    llama_context_ptr ctx_b(llama_init_from_model(model, make_cparams(1)));
    if (!ctx_par || !ctx_b) {
        return false;
    }

    std::vector<llama_token> tokens_b(n_tokens);
    for (uint32_t i = 0; i < n_tokens; i++) {
        tokens_b[i] = tokens[(i + 1) % n_tokens];
    }

    llama_batch batch_par = llama_batch_init(2*n_tokens, 0, 1);
    llama_batch batch_b   = llama_batch_init(n_tokens, 0, 1);
    for (uint32_t pos = 0; pos < n_tokens; pos++) {
        test_batch_add(batch_par, tokens[pos],   pos, {0}, true);
        test_batch_add(batch_par, tokens_b[pos], pos, {1}, true);
        test_batch_add(batch_b,   tokens_b[pos], pos, {0}, true);
    }
    bool ok = true;
    if (encode) {
        ok = llama_encode(ctx_par.get(), batch_par) == 0 && llama_encode(ctx_b.get(), batch_b) == 0;
    }
    ok = ok && llama_decode(ctx_par.get(), batch_par) == 0 && llama_decode(ctx_b.get(), batch_b) == 0;
    llama_batch_free(batch_par);
    llama_batch_free(batch_b);
    if (!ok) {
        return false;
    }

    std::vector<float> logits_par_a;
    std::vector<float> logits_par_b;
    std::vector<float> logits_b;
    for (uint32_t i = 0; i < n_tokens; i++) {
        const float * l_par_a = llama_get_logits_ith(ctx_par.get(), 2*i);
        const float * l_par_b = llama_get_logits_ith(ctx_par.get(), 2*i + 1);
        const float * l_b     = llama_get_logits_ith(ctx_b.get(), i);
        if (l_par_a == nullptr || l_par_b == nullptr || l_b == nullptr) {
            return false;
        }
        logits_par_a.insert(logits_par_a.end(), l_par_a, l_par_a + n_vocab);
        logits_par_b.insert(logits_par_b.end(), l_par_b, l_par_b + n_vocab);
        logits_b.insert(logits_b.end(), l_b, l_b + n_vocab);
    }

    // ubatch splits differ from the reference, so use the NMSE limit of the CPU comparison; NaN fails
    return nmse(logits_a, logits_par_a) <= 1e-4 && nmse(logits_b, logits_par_b) <= 1e-4;
}

// entries [n/4, n/2) are embd rows, decoded either as token/embd/token chunks or as one mixed batch
// returns the llama_process() error code
static int32_t get_logits_mixed(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens, const std::vector<float> & embd, bool mixed,
        std::vector<float> & ret) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_embd   = llama_model_n_embd_inp(model);
    const uint32_t n_tokens = tokens.size();
    const uint32_t i_embd_0 = n_tokens/4;
    const uint32_t i_embd_1 = n_tokens/2;

    const std::vector<uint32_t> bounds = mixed
        ? std::vector<uint32_t>{0, n_tokens}
        : std::vector<uint32_t>{0, i_embd_0, i_embd_1, n_tokens};

    llama_memory_clear(llama_get_memory(lctx), true);
    llama_batch_ext_ptr batch(llama_batch_ext_init(lctx));

    ret.clear();
    ret.reserve(n_tokens*n_vocab);
    for (size_t c = 0; c + 1 < bounds.size(); c++) {
        llama_batch_ext_clear(batch.get());
        for (uint32_t i = bounds[c]; i < bounds[c + 1]; i++) {
            const bool is_embd = i >= i_embd_0 && i < i_embd_1;
            const int32_t idx = is_embd
                ? llama_batch_ext_add_embd(batch.get(), 0, { embd.data() + (size_t) (i - i_embd_0)*n_embd, 1, n_embd })
                : llama_batch_ext_add_token(batch.get(), 0, tokens[i]);
            GGML_ASSERT(idx >= 0);
            const llama_pos pos[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, 0 };
            llama_batch_ext_set_pos(batch.get(), idx, pos);
            llama_batch_ext_set_output_logits(batch.get(), idx, true);
        }
        const int32_t err = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (err != 0) {
            return err;
        }
        for (uint32_t i = 0; i < bounds[c + 1] - bounds[c]; i++) {
            const float * logits_ith = llama_get_logits_ith(lctx, i);
            ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
        }
    }
    return 0;
}

static bool check_causal_attn_toggle(
        llama_model * model, llama_context * lctx, const std::vector<llama_token> & tokens) {
    const uint32_t n_vocab  = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_past   = tokens.size();
    const uint32_t n_ubatch = llama_n_ubatch(lctx);

    GGML_ASSERT(n_past + n_ubatch/2 + n_ubatch <= llama_n_ctx(lctx));

    llama_set_causal_attn(lctx, false);

    common_batch batch(lctx);

    bool ok = true;
    uint32_t pos = n_past;
    for (const uint32_t n_tokens : { n_ubatch/2, n_ubatch }) {
        batch.clear();
        for (uint32_t i = 0; i < n_tokens; i++) {
            batch.add(tokens[i], pos++, 0, true);
        }

        const int32_t rc = llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());
        if (rc != 0) {
            LOG_ERR("%s: n_tokens=%u: llama_process returned %d\n", __func__, n_tokens, rc);
            ok = false;
            break;
        }

        const float * logits = llama_get_logits_ith(lctx, n_tokens - 1);
        if (logits == nullptr) {
            LOG_ERR("%s: n_tokens=%u: no logits\n", __func__, n_tokens);
            ok = false;
            break;
        }
        for (uint32_t j = 0; j < n_vocab; j++) {
            if (std::isnan(logits[j])) {
                LOG_ERR("%s: n_tokens=%u: nan logit\n", __func__, n_tokens);
                ok = false;
                break;
            }
        }
    }

    return ok;
}

// speculative decoding: each step decodes a token and 0 to n_draft_max drafts, then rejects the drafts
// the logits of the token must match a decode without drafts
static double check_draft_rollback(llama_model * model, const std::vector<llama_token> & tokens, const std::vector<float> & logits_ref) {
    const uint32_t n_vocab     = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const uint32_t n_tokens    = tokens.size();
    const uint32_t n_draft_max = 2;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx           = 0;
    ctx_params.n_threads       = 4;
    ctx_params.n_threads_batch = 4;
    ctx_params.n_ubatch        = 64;
    ctx_params.n_rs_seq        = n_draft_max;

    llama_context_ptr lctx(llama_init_from_model(model, ctx_params));
    if (!lctx) {
        throw std::runtime_error("failed to create llama context");
    }

    common_batch batch(lctx.get());
    for (uint32_t pos = 0; pos < n_tokens/2; pos++) {
        batch.add(tokens[pos], pos, 0, false);
    }
    if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }

    std::vector<float> logits;
    std::vector<float> ref;
    for (uint32_t pos = n_tokens/2; pos < n_tokens; pos++) {
        batch.clear();
        batch.add(tokens[pos], pos, 0, true);
        const uint32_t n_draft = pos % (n_draft_max + 1);
        for (uint32_t i = 1; i <= n_draft; i++) {
            batch.add((tokens[pos] + i) % n_vocab, pos + i, 0, true);
        }
        if (llama_process(lctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
            throw std::runtime_error("failed to decode batch");
        }

        const float * logits_ith = llama_get_logits_ith(lctx.get(), 0);
        logits.insert(logits.end(), logits_ith, logits_ith + n_vocab);
        ref.insert(ref.end(), logits_ref.begin() + (size_t) pos*n_vocab, logits_ref.begin() + (size_t) (pos + 1)*n_vocab);

        if (!llama_memory_seq_rm(llama_get_memory(lctx.get()), 0, pos + 1, -1)) {
            throw std::runtime_error("failed to reject the drafts");
        }
    }

    return nmse(ref, logits);
}

static bool moe_mandatory(const llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_FLINT:
        case LLM_ARCH_ZAYA1:
        case LLM_ARCH_LLAMA4:
        case LLM_ARCH_MAION_CODER:
        case LLM_ARCH_COHERE2MOE:
        case LLM_ARCH_GROK:
        case LLM_ARCH_QWEN2MOE:
        case LLM_ARCH_QWEN3MOE:
        case LLM_ARCH_QWEN3NEXT:
        case LLM_ARCH_QWEN3VLMOE:
        case LLM_ARCH_QWEN35MOE:
        case LLM_ARCH_CLOVER_MOE:
        case LLM_ARCH_QWEN4EXP:
        case LLM_ARCH_PHIMOE:
        case LLM_ARCH_DBRX:
        case LLM_ARCH_OLMOE:
        case LLM_ARCH_ARCTIC:
        case LLM_ARCH_DEEPSEEK:
        case LLM_ARCH_DEEPSEEK2:
        case LLM_ARCH_DEEPSEEK32:
        case LLM_ARCH_DOTS3NOTE:
        case LLM_ARCH_DEEPSEEK4:
        case LLM_ARCH_DEEPSEEK41:
        case LLM_ARCH_XING4_0:
        case LLM_ARCH_GLM4_MOE:
        case LLM_ARCH_GLM_DSA:
        case LLM_ARCH_EXAONE_MOE:
        case LLM_ARCH_BAILINGMOE:
        case LLM_ARCH_BAILINGMOE2:
        case LLM_ARCH_BAILINGMOE3:
        case LLM_ARCH_DOTS1:
        case LLM_ARCH_AFMOE:
        case LLM_ARCH_BEAM:
        case LLM_ARCH_ERNIE4_5:
        case LLM_ARCH_ERNIE4_5_MOE:
        case LLM_ARCH_HUNYUAN_MOE:
        case LLM_ARCH_HY_V3:
        case LLM_ARCH_HY_V4:
        case LLM_ARCH_OPENAI_MOE:
        case LLM_ARCH_LFM2MOE:
        case LLM_ARCH_SMALLTHINKER:
        case LLM_ARCH_LLADA_MOE:
        case LLM_ARCH_GROVEMOE:
        case LLM_ARCH_MINIMAX_01:
        case LLM_ARCH_MINIMAX_M2:
        case LLM_ARCH_MINIMAX_M3:
        case LLM_ARCH_RND1:
        case LLM_ARCH_PADDLEOCR:
        case LLM_ARCH_MIMO2:
        case LLM_ARCH_KIMI_LINEAR:
        case LLM_ARCH_KIMI_K3:
        case LLM_ARCH_KIMI_K31:
        case LLM_ARCH_GLM5_NEXT:
        case LLM_ARCH_STEP35:
        case LLM_ARCH_MISTRAL4:
        case LLM_ARCH_MISTRAL_LARGE4:
        case LLM_ARCH_MELLUM:
        case LLM_ARCH_LAGUNA:
        case LLM_ARCH_MAPLE:
        case LLM_ARCH_MUSE_SPARK:
        case LLM_ARCH_KOLIBRI1:
        case LLM_ARCH_METIS16:
        case LLM_ARCH_ZONOS3:
        case LLM_ARCH_INKLING2:
        case LLM_ARCH_INKLING2_LOOP:
            return true;
        default:
            return false;
    }
}

static bool moe_implemented(const llm_arch arch) {
    if (moe_mandatory(arch)) {
        return true;
    }
    switch (arch) {
        case LLM_ARCH_LLAMA:
        case LLM_ARCH_REFACT:
        case LLM_ARCH_MINICPM:
        case LLM_ARCH_GRANITE:
        case LLM_ARCH_GRANITE_MOE:
        case LLM_ARCH_MISTRAL3:
        case LLM_ARCH_LLAMA_EMBED:
        case LLM_ARCH_K2_HORIZON:
            return true;
        default:
            return false;
    }
}

static bool arch_supported(const llm_arch arch) {
    if (arch == LLM_ARCH_CLIP || arch == LLM_ARCH_GPTJ || arch == LLM_ARCH_UNKNOWN) {
        return false; // These models don't have usable implementations.
    }
    if (arch == LLM_ARCH_CHAMELEON) {
        return false; // Only half-implemented and to be removed in the future.
    }
    if (arch == LLM_ARCH_WAVTOKENIZER_DEC) {
        return false; // FIXME CUDA backend crashes.
    }
    if (arch == LLM_ARCH_GRANITE_SWITCH) {
        return false; // FIXME adapter fixture
    }
    if (arch == LLM_ARCH_LLAMA_EMBED || arch == LLM_ARCH_GEMMA_EMBEDDING || arch == LLM_ARCH_GEMMA_EMBEDDING2 || arch == LLM_ARCH_T5ENCODER) {
        return false; // FIXME Embedding (?) models produce inconsistent results.
    }
    if (arch == LLM_ARCH_RWKV6 || arch == LLM_ARCH_RWKV6QWEN2 || arch == LLM_ARCH_RWKV7 || arch == LLM_ARCH_ARWKV7) {
        return false; // FIXME RWKV models hang indefinitely.
    }
    if (arch == LLM_ARCH_BERT || arch == LLM_ARCH_MODERN_BERT || arch == LLM_ARCH_NOMIC_BERT || arch == LLM_ARCH_NOMIC_BERT_MOE ||
            arch == LLM_ARCH_NEO_BERT || arch == LLM_ARCH_JINA_BERT_V2 || arch == LLM_ARCH_JINA_BERT_V3 || arch == LLM_ARCH_EUROBERT) {
        return false; // TODO vocab
    }
    if (arch == LLM_ARCH_KOKORO || arch == LLM_ARCH_TINY_VOCA) {
        return false; // TODO encoder without memory, a sequence must fit in one ubatch
    }
    if (arch == LLM_ARCH_PLM) {
        return false; // TODO tensor shapes
    }
    if (arch == LLM_ARCH_CLEF || arch == LLM_ARCH_VLLM_DECISION2) {
        return false; // TODO decision head tensors
    }
    if (arch == LLM_ARCH_DEEPSEEK2OCR) {
        return false;
    }
    if (arch == LLM_ARCH_FUSE3 || arch == LLM_ARCH_BAILING_HYBRID) {
        return false; // FIXME fork archs: the random-model fixture does not write their required hparams (e.g. fuse3.expert_feed_forward_length, bailing-hybrid MLA key lengths)
    }
    // FIXME: these hit scheduler/view-backed-output issues with WebGPU on CI.
#ifdef GGML_USE_WEBGPU
    if (arch == LLM_ARCH_DEEPSEEK32 || arch == LLM_ARCH_GLM_DSA || arch == LLM_ARCH_MINIMAX_M3 || arch == LLM_ARCH_DOTS3NOTE || arch == LLM_ARCH_QWEN4EXP ||
            arch == LLM_ARCH_HY_V4) {
        return false;
    }
#endif // GGML_USE_WEBGPU

    // FIXME: jamba produces incorrect output (~0.55 NMSE vs CPU) on the HIP
    // backend on RDNA3.5 (gfx1151); the SSM kernels need investigation.
#ifdef GGML_USE_HIP
    if (arch == LLM_ARCH_JAMBA) {
        return false;
    }
#endif // GGML_USE_HIP

    return true;
}

static int save_models(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const std::string & dir) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    bool ok = true;
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            if (!llama_arch_supports_save(arch) || !arch_supported(arch)) {
                LOG_INF("%s: %s model (%s) is unsupported, skipping\n", __func__, llama_arch_name(arch), moe ? "MoE" : "dense");
                continue;
            }
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            llama_model_ptr model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            const std::string path = dir + "/" + llama_arch_name(arch) + (moe ? "-moe.gguf" : "-dense.gguf");
            LOG_INF("%s: '%s %s' -> %s ...", __func__, llama_arch_name(arch), moe ? "MoE" : "dense", path.c_str());
            llama_model_save_to_file(model.get(), path.c_str());
            const std::uintmax_t size = std::filesystem::file_size(path);
            LOG_CNT(" (%.2f MB)\n", (double) size / (1024.0 * 1024.0));

            const std::string path_q4_0 = dir + "/" + llama_arch_name(arch) + (moe ? "-moe-q4_0.gguf" : "-dense-q4_0.gguf");
            LOG_INF("%s: Quantizing %s model (%s) to %s...\n", __func__, llama_arch_name(arch), moe ? "MoE" : "dense", path_q4_0.c_str());
            llama_model_quantize_params qparams = llama_model_quantize_default_params();
            qparams.ftype   = LLAMA_FTYPE_MOSTLY_Q4_0;
            qparams.nthread = 1;
            if (llama_model_quantize(path.c_str(), path_q4_0.c_str(), &qparams) != 0) {
                LOG_ERR("%s: failed to quantize %s\n", __func__, path.c_str());
                std::filesystem::remove(path_q4_0);
                ok = false;
            }
        }
    }
    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return ok ? 0 : 1;
}

// the CPU and the GPU of an Apple SoC carry the same description, the name tells them apart (CPU, MTL0)
static std::string get_device_label(ggml_backend_dev_t dev) {
    const char * desc = ggml_backend_dev_description(dev);
    const char * name = ggml_backend_dev_name(dev);

    if (strcmp(desc, name) == 0) {
        return desc;
    }

    return std::string(desc) + " (" + name + ")";
}

static int test_backends(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity, const char * target_backend) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    const std::vector<llama_token> tokens = get_tokens(128, 128, seed);

    struct device_config {
        std::vector<ggml_backend_dev_t> devs;
        std::string                     label;
        llama_split_mode                split_mode;
        bool                            host_experts; // keep the experts in host memory, see host_experts_test
        size_t                          moe_cache_size;

        device_config(std::vector<ggml_backend_dev_t> devs, std::string name, llama_split_mode split_mode, bool host_experts = false, size_t moe_cache_size = 0)
            : devs(std::move(devs)), label(std::move(name)), split_mode(split_mode), host_experts(host_experts), moe_cache_size(moe_cache_size) {}
    };

    const llama_model_tensor_buft_override host_experts_overrides[] = {
        { LLM_FFN_EXPS_REGEX, ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    std::vector<device_config> dev_configs;
    size_t max_device_label_length = 4;
    {
        std::vector<ggml_backend_dev_t> devices_meta;
        {
            const size_t device_count = ggml_backend_dev_count();
            for (size_t i = 0; i < device_count; i++) {
                ggml_backend_dev_t dev = ggml_backend_dev_get(i);
                if (target_backend != nullptr && strcmp(target_backend, ggml_backend_dev_name(dev)) != 0) {
                    continue;
                }
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{dev}, get_device_label(dev), LLAMA_SPLIT_MODE_LAYER);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());

                // cpu-based devices cannot be used in tensor split mode
                if (ggml_backend_dev_buffer_type(dev) != ggml_backend_cpu_buffer_type()) {
                    devices_meta.push_back(dev);
                }
            }
        }

        if (target_backend == nullptr) {
            dev_configs.emplace_back(devices_meta, "Meta", LLAMA_SPLIT_MODE_TENSOR);
        }

        // the ops that use the experts are offloaded to the first device and the scheduler copies the used experts
        if (!devices_meta.empty()) {
            dev_configs.emplace_back(devices_meta, "Host experts", LLAMA_SPLIT_MODE_LAYER, true);
            max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
        }

        // the ops that use the host experts run on a GPU and read the experts from a cache
        // the cache has only a few slots (4 for 288 KiB experts), so the experts are evicted and uploaded again
        if (!devices_meta.empty()) {
            const enum ggml_backend_dev_type type = ggml_backend_dev_type(devices_meta[0]);
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                dev_configs.emplace_back(std::vector<ggml_backend_dev_t>{devices_meta[0]}, "MoE cache", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                // each GPU caches the layers assigned to it
                if (devices_meta.size() > 1) {
                    dev_configs.emplace_back(devices_meta, "MoE cache, layer split", LLAMA_SPLIT_MODE_LAYER, true, 1536*1024);
                    max_device_label_length = std::max(max_device_label_length, dev_configs.back().label.length());
                }
            }
        }
    }

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }

    const std::string template_header  = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|%15s|%9s|%15s|%9s|\n";
    const std::string template_row_cfg = std::string("|%" + std::to_string(max_arch_name_length) + "s|%") + std::to_string(max_device_label_length) + "s|%6s|";
    const std::string template_row_res = "%15s %10s|%20s|%15s %10s|%20s|\n";

    bool all_ok = true;
    size_t n_tests = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_header.c_str(), "Model arch.", "Device", "Config", "NMSE vs. CPU", "Roundtrip", "Mixed batch", "Parallel");
    LOG("|");
    for (size_t i = 0; i < max_arch_name_length; i++) {
        LOG("-");
    }
    LOG("|");
    for (size_t i = 0; i < max_device_label_length; i++) {
        LOG("-");
    }
    LOG("|------|---------------|---------|---------------|---------|\n");
    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN) {
            continue;
        }
        if (!arch_matches(arch_filter, arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }

        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const std::string config_name = moe ? "MoE" : "Dense";
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }
            std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_cpu;
            std::vector<float> logits_cpu;
            for (device_config & dc : dev_configs) {
                if (dc.host_experts && (!moe || !host_experts_test(arch))) {
                    continue;
                }
                const llama_model_tensor_buft_override * overrides = dc.host_experts ? host_experts_overrides : nullptr;

                // print test config first; should anything fail during model loading or inference, at least we know which test case caused it
                LOG(template_row_cfg.c_str(), llama_arch_name(arch), dc.label.c_str(), config_name.c_str());
                fflush(stdout);

                std::pair<llama_model_ptr, llama_context_ptr> model_and_ctx_dev;
                std::vector<float> logits_dev;
                std::string status_nmse      = "\033[1;33mSKIP\033[0m";
                std::string status_roundtrip = "\033[1;33mSKIP\033[0m";
                std::string status_parallel  = "\033[1;33mSKIP\033[0m";
                std::string status_mixed     = "\033[1;33mSKIP\033[0m";
                char nmse_str[12] = {0};
                char mixed_str[12] = {0};

                bool skip = !arch_supported(arch) || (dc.split_mode == LLAMA_SPLIT_MODE_TENSOR && dc.devs.empty());
                bool test_executed = false;
                bool test_ok = true;
                if (!skip) {
                    if (logits_cpu.empty()) {
                        model_and_ctx_cpu = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, {}, LLAMA_SPLIT_MODE_LAYER, encode);
                        logits_cpu = get_logits(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, encode);
                    }
                    if (dc.split_mode != LLAMA_SPLIT_MODE_TENSOR || llama_arch_supports_sm_tensor(arch)) {
                        test_executed = true;
                        model_and_ctx_dev = get_model_and_ctx(gguf_ctx.get(), nullptr, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                        logits_dev = get_logits(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens, encode);
                        const double nmse_val = nmse(logits_cpu, logits_dev);
                        snprintf(nmse_str, sizeof(nmse_str), "(%.2e)", nmse_val);
                        status_nmse = "\033[1;32mOK\033[0m";
                        if (nmse_val > 1e-4) {
                            test_ok = false;
                            status_nmse = "\033[1;31mFAIL\033[0m";
                        }

                        // FIXME: T5 kq_b does not broadcast over KV streams, so context init with n_seq_max > 1 aborts
                        if (arch != LLM_ARCH_T5) {
                            status_parallel = "\033[1;32mOK\033[0m";
                            if (!test_parallel_seqs(model_and_ctx_dev.first.get(), tokens, logits_dev, encode)) {
                                test_ok = false;
                                status_parallel = "\033[1;31mFAIL\033[0m";
                            }
                        }
                        // chunked decode matches a single batch only with causal attention over a memory
                        llama_context * lctx_dev = model_and_ctx_dev.second.get();
                        if (!encode && llama_get_memory(lctx_dev) != nullptr) {
                            std::vector<float> embd_mixed((size_t) llama_model_n_embd_inp(model_and_ctx_dev.first.get())*tokens.size()/4);
                            std::mt19937 gen(seed);
                            std::normal_distribution<float> dis(0.0f, stdev);
                            for (float & v : embd_mixed) {
                                v = dis(gen);
                            }
                            std::vector<float> logits_mixed;
                            std::vector<float> logits_chunks;
                            if (llama_arch_supports_mixed_batch(arch)) {
                                if (get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, false, logits_chunks) != 0 ||
                                    get_logits_mixed(model_and_ctx_dev.first.get(), lctx_dev, tokens, embd_mixed, true,  logits_mixed)  != 0) {
                                    throw std::runtime_error("failed to decode mixed batch");
                                }
                                const double nmse_mixed = nmse(logits_chunks, logits_mixed);
                                snprintf(mixed_str, sizeof(mixed_str), "(%.2e)", nmse_mixed);
                                status_mixed = "\033[1;32mOK\033[0m";
                                if (nmse_mixed > 1e-4) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            } else {
                                // must be rejected as an invalid batch, mute the expected error log
                                ud.verbosity = LOG_LEVEL_OUTPUT;
                                const int32_t err = get_logits_mixed(model_and_ctx_cpu.first.get(), model_and_ctx_cpu.second.get(), tokens, embd_mixed, true, logits_mixed);
                                ud.verbosity = verbosity;
                                if (err != -1) {
                                    test_ok = false;
                                    status_mixed = "\033[1;31mFAIL\033[0m";
                                }
                            }
                        }

                        // runs after the mixed batch check, as it leaves the context with non-causal attention
                        if (!encode && !check_causal_attn_toggle(model_and_ctx_dev.first.get(), model_and_ctx_dev.second.get(), tokens)) {
                            if (test_ok) {
                                status_nmse = "\033[1;31mFAIL\033[0m (toggle)";
                            }
                            test_ok = false;
                        }

                        // the k-pool layout is updated in place when drafts are rejected
                        // TODO: the meta backend aborts with n_rs_seq > 0 (view of permuted tensor not implemented)
                        if ((arch == LLM_ARCH_QWEN4EXP || arch == LLM_ARCH_GLM5_NEXT) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                            const double nmse_rollback = check_draft_rollback(model_and_ctx_dev.first.get(), tokens, logits_dev);
                            if (!(nmse_rollback <= 1e-4)) {
                                if (test_ok) {
                                    status_nmse = "\033[1;31mFAIL\033[0m (rollback)";
                                }
                                test_ok = false;
                            }
                        }
                    }

                    if (llama_arch_supports_save(arch) && dc.split_mode != LLAMA_SPLIT_MODE_TENSOR) {
                        test_executed = true;
                        GGML_ASSERT(model_and_ctx_dev.first && model_and_ctx_dev.second);
                        // FIXME: when adding a tensor to a gguf_context a copy is made, this changes the pointer which the meta backend
                        //     in turn uses to map the tensors to their simple equivalents - this is fundamentally incompatible
                        const std::string path = get_tmp_gguf_path();
                        llama_model_save_to_file(model_and_ctx_dev.first.get(), path.c_str());
                        FILE * file = fopen(path.c_str(), "rb");
                        GGML_ASSERT(file != nullptr);

                        {
                            auto model_and_ctx_roundtrip = get_model_and_ctx(nullptr, file, seed, stdev, dc.devs, dc.split_mode, encode, overrides, dc.moe_cache_size);
                            const std::vector<float> logits_roundtrip = get_logits(
                                model_and_ctx_roundtrip.first.get(), model_and_ctx_roundtrip.second.get(), tokens, encode);
                            status_roundtrip = "\033[1;32mOK\033[0m";
                            GGML_ASSERT(logits_roundtrip.size() == logits_dev.size());
                            for (size_t i = 0; i < logits_roundtrip.size(); i++) {
                                if (logits_roundtrip[i] != logits_dev[i]) {
                                    test_ok = false;
                                    status_roundtrip = "\033[1;31mFAIL\033[0m";
                                    break;
                                }
                            }
                        }

                        fclose(file);
                        std::filesystem::remove(path);
                    }
                }

                if (test_executed) {
                    n_tests++;
                    if (!test_ok) {
                        n_failed++;
                        all_ok = false;
                    }
                }

                // log the results for this test case
                LOG(template_row_res.c_str(), status_nmse.c_str(), nmse_str, status_roundtrip.c_str(),
                    status_mixed.c_str(), mixed_str, status_parallel.c_str());
            }
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return all_ok ? 0 : 1;
}

struct kv_shift_move {
    llama_pos p0;
    llama_pos p1;
    llama_pos delta;
};

static void decode_range(llama_context * lctx, const llama_token * tokens, uint32_t n_tokens, llama_pos pos0, llama_seq_id seq_id, bool output) {
    common_batch batch(lctx);
    for (uint32_t i = 0; i < n_tokens; i++) {
        batch.add(tokens[i], pos0 + i, seq_id, output || i == n_tokens - 1);
    }
    if (llama_process(lctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        throw std::runtime_error("failed to decode batch");
    }
}

// the raw attention scores of each layer are linear in K, unlike the logits they do not depend on how sharp the attention is
struct kq_capture {
    bool active = false;
    std::vector<std::pair<std::string, std::vector<float>>> data;
};

// the scores are compared cell by cell, so the cells of the probe must match: prefix and probe fit in the SWA window,
// otherwise the probe can reuse a masked SWA cell in one run and not in the other
static bool kq_capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    kq_capture * cap = (kq_capture *) user_data;
    const char * name = ggml_get_name(t);
    const bool want = cap->active && t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
        (strncmp(name, "kq-", 3) == 0 || strncmp(name, "indexer_kq-", 11) == 0 || strncmp(name, "indexer_score-", 14) == 0);
    if (ask) {
        return want;
    }
    if (want) {
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
        auto it = std::find_if(cap->data.begin(), cap->data.end(), [&](const auto & e) { return e.first == name; });
        if (it == cap->data.end()) {
            cap->data.emplace_back(name, std::move(buf));
        } else {
            it->second = std::move(buf);
        }
    }
    return true;
}

struct kq_diff {
    double      worst = 0.0;
    std::string first_bad; // first tensor above the threshold, later layers can inherit its error through Q
    std::string bad;       // all tensors above the threshold, with their nmse
};

static kq_diff compare_kq(const std::vector<std::pair<std::string, std::vector<float>>> & ref,
                          const std::vector<std::pair<std::string, std::vector<float>>> & cur, double thold) {
    if (ref.size() != cur.size()) {
        throw std::runtime_error("kq tensor count mismatch");
    }
    kq_diff res;
    for (size_t i = 0; i < ref.size(); i++) {
        if (ref[i].first != cur[i].first || ref[i].second.size() != cur[i].second.size()) {
            throw std::runtime_error("kq tensor mismatch: " + ref[i].first);
        }
        double mse_a_b = 0.0;
        double mse_a_0 = 0.0;
        for (size_t j = 0; j < ref[i].second.size(); j++) {
            const double a = ref[i].second[j];
            const double b = cur[i].second[j];
            // masked rows may hold -inf in both
            if (!std::isfinite(a) && a == b) {
                continue;
            }
            mse_a_b += (a - b)*(a - b);
            mse_a_0 += a*a;
        }
        const double val = mse_a_b == 0.0 ? 0.0 : mse_a_0 == 0.0 ? INFINITY : mse_a_b/mse_a_0;
        res.worst = std::max(res.worst, std::isnan(val) ? INFINITY : val);
        if (!(val <= thold)) {
            if (res.first_bad.empty()) {
                res.first_bad = ref[i].first;
            }
            char buf[64];
            snprintf(buf, sizeof(buf), " %s=%.1e", ref[i].first.c_str(), val);
            res.bad += buf;
        }
    }
    return res;
}

// decodes the prefix at pos_prefix, applies each step of moves with its own K-shift and returns the logits of the probe at pos_probe
static std::vector<float> get_logits_kv_shift(
        llama_model * model, llama_context * lctx, kq_capture & cap, const std::vector<llama_token> & prefix, const std::vector<llama_token> & probe,
        llama_pos pos_prefix, const std::vector<std::vector<kv_shift_move>> & steps, llama_pos pos_probe) {
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    llama_memory_t mem = llama_get_memory(lctx);

    cap.data.clear();
    llama_memory_clear(mem, true);
    decode_range(lctx, prefix.data(), prefix.size(), pos_prefix, 0, false);

    for (size_t i = 0; i < steps.size(); i++) {
        for (const kv_shift_move & mv : steps[i]) {
            llama_memory_seq_add(mem, 0, mv.p0, mv.p1, mv.delta);
        }
        // there is no public API to apply a pending shift, a decode of a throwaway token in seq 1 does it
        if (i + 1 < steps.size()) {
            decode_range(lctx, probe.data(), 1, 0, 1, false);
            llama_memory_seq_rm(mem, 1, -1, -1);
        }
    }

    cap.active = true;
    decode_range(lctx, probe.data(), probe.size(), pos_probe, 0, true);
    cap.active = false;

    std::vector<float> ret;
    ret.reserve(probe.size()*n_vocab);
    for (uint32_t i = 0; i < probe.size(); i++) {
        const float * logits_ith = llama_get_logits_ith(lctx, i);
        ret.insert(ret.end(), logits_ith, logits_ith + n_vocab);
    }
    return ret;
}

// a K-shift is correct if decoding at [d, d+n) and shifting by -d gives the same probe logits as decoding at [0, n)
static int test_kv_shift(const std::string & arch_filter, const size_t seed, const float stdev, const int verbosity) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } log_old;

        int verbosity;

        user_data_t(int verbosity) : verbosity(verbosity) {
            llama_log_get(&log_old.callback, &log_old.user_data);
        }
    };
    user_data_t ud(verbosity);

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        int verbosity = common_log_get_verbosity(level);
        if (verbosity <= ud->verbosity) {
            ud->log_old.callback(level, text, ud->log_old.user_data);
        }
    }, &ud);

    // n_prefix + n_probe stays below the fixture SWA window (n_ctx/8 = 32)
    const uint32_t  n_prefix = 20;
    const uint32_t  n_probe  = 8;
    const llama_pos d        = 64;
    const llama_pos gap      = 4;

    const std::vector<llama_token> tokens = get_tokens(n_prefix + n_probe, 128, seed);
    const std::vector<llama_token> prefix(tokens.begin(), tokens.begin() + n_prefix);
    const std::vector<llama_token> probe (tokens.begin() + n_prefix, tokens.end());

    const std::vector<std::vector<kv_shift_move>> steps_shift  = { { { d, d + (llama_pos) n_prefix, -d } } };
    const std::vector<std::vector<kv_shift_move>> steps_shift2 = { { { d, d + (llama_pos) n_prefix, -d/2 } }, { { d/2, d/2 + (llama_pos) n_prefix, -d/2 } } };
    // control: the second half of the prefix ends up gap positions too far, which tells if the probe is sensitive to K positions at all
    const std::vector<std::vector<kv_shift_move>> steps_ctrl   = { {
        { d,                           d + (llama_pos) n_prefix/2, -d       },
        { d + (llama_pos) n_prefix/2,  d + (llama_pos) n_prefix,   -d + gap },
    } };

    size_t max_arch_name_length = 0;
    for (const llm_arch & arch : llama_arch_all()) {
        max_arch_name_length = std::max(max_arch_name_length, strlen(llama_arch_name(arch)));
    }
    const std::string template_row = "|%" + std::to_string(max_arch_name_length) + "s|%6s|%10s|%10s|%10s|%10s|%s\n";

    size_t n_tests  = 0;
    size_t n_failed = 0;
    common_log_flush(common_log_main());
    LOG(template_row.c_str(), "Model arch.", "Config", "Shift kq", "Shift2 kq", "Logits", "Control kq", "Status");

    for (const llm_arch & arch : llama_arch_all()) {
        if (arch == LLM_ARCH_UNKNOWN || !arch_matches(arch_filter, arch) || !arch_supported(arch)) {
            continue;
        }
        if (arch == LLM_ARCH_EAGLE3 || arch == LLM_ARCH_DFLASH || arch == LLM_ARCH_GEMMA4_ASSISTANT) {
            continue;
        }
        const bool encode = arch == LLM_ARCH_T5 || arch == LLM_ARCH_DREAM || arch == LLM_ARCH_LLADA || arch == LLM_ARCH_LLADA_MOE || arch == LLM_ARCH_RND1;
        if (encode) {
            continue;
        }
        for (bool moe : {false, true}) {
            if (moe && !moe_implemented(arch)) {
                continue;
            }
            if (!moe && moe_mandatory(arch)) {
                continue;
            }
            const char * config_name = moe ? "MoE" : "Dense";

            // the attention scale depends on the absolute position, so the hidden states of the prefix do too
            const double thold = arch == LLM_ARCH_MISTRAL_LARGE4 ? 1e-4 : 1e-6;

            // at least 4 layers, so that "every 4th layer" patterns (NoPE, full attention) are covered
            // exaone4 is SWA only with 64 layers, smallthinker always ropes with 4
            const uint32_t n_layer_min = arch == LLM_ARCH_EXAONE4 ? 64 : arch == LLM_ARCH_SMALLTHINKER ? 8 : 4;
            gguf_context_ptr gguf_ctx = get_gguf_ctx(arch, moe, false, n_layer_min);
            // different bases, so that a shift using the wrong one is caught
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE).c_str(),     20000.0f);
            gguf_set_val_f32(gguf_ctx.get(), llama_arch_kv_name(arch, LLM_KV_ROPE_FREQ_BASE_SWA).c_str(), 10000.0f);
            if (arch == LLM_ARCH_BAILINGMOE3) {
                GGML_ASSERT(gguf_remove_key(gguf_ctx.get(), "bailingmoe3.kda.safe_gate") >= 0);
            }

            llama_model_ptr model;
            try {
                model = get_model(gguf_ctx.get(), nullptr, seed, stdev, {});
            } catch (const std::exception & err) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;31mERROR\033[0m (model load)");
                n_tests++;
                n_failed++;
                continue;
            }
            // absolute position embeddings are not translation invariant, the comparison does not hold
            if (llama_model_rope_type(model.get()) == LLAMA_ROPE_TYPE_NONE) {
                LOG(template_row.c_str(), llama_arch_name(arch), config_name, "", "", "", "", "\033[1;33mSKIP\033[0m (no rope)");
                continue;
            }

            LOG("|%*s|%6s|", (int) max_arch_name_length, llama_arch_name(arch), config_name);
            fflush(stdout);

            llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = 0;
            ctx_params.n_threads       = 4;
            ctx_params.n_threads_batch = 4;
            ctx_params.n_seq_max       = 2;
            ctx_params.kv_unified      = true;
            ctx_params.type_k          = GGML_TYPE_F32;
            ctx_params.type_v          = GGML_TYPE_F32;
            ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            // differs from the SWA default, so that a shift using the wrong scale is caught
            ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
            ctx_params.rope_freq_scale   = 0.5f;
            kq_capture cap;
            ctx_params.cb_eval           = kq_capture_cb;
            ctx_params.cb_eval_user_data = &cap;

            ud.verbosity = LOG_LEVEL_OUTPUT;
            llama_context_ptr lctx(llama_init_from_model(model.get(), ctx_params));
            ud.verbosity = verbosity;
            if (!lctx) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (context)\n", "", "", "", "");
                continue;
            }
            llama_memory_t mem = llama_get_memory(lctx.get());
            if (mem == nullptr || !llama_memory_can_shift(mem)) {
                LOG("%10s|%10s|%10s|%10s|\033[1;33mSKIP\033[0m (no shift)\n", "", "", "", "");
                continue;
            }

            std::string status;
            char shift_str[16]  = "";
            char shift2_str[16] = "";
            char logits_str[16] = "";
            char ctrl_str[16]   = "";
            n_tests++;
            try {
                const std::vector<float> logits_ref = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, 0, {}, n_prefix);
                const auto kq_ref = cap.data;
                if (verbosity >= LOG_LEVEL_INFO) {
                    for (const auto & e : kq_ref) {
                        LOG("[%s: %zu] ", e.first.c_str(), e.second.size());
                    }
                }
                const std::vector<float> logits_shift = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift, n_prefix);
                const kq_diff diff_shift = compare_kq(kq_ref, cap.data, thold);
                get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_shift2, n_prefix);
                const kq_diff diff_shift2 = compare_kq(kq_ref, cap.data, thold);
                const std::vector<float> logits_ctrl = get_logits_kv_shift(model.get(), lctx.get(), cap, prefix, probe, d, steps_ctrl, n_prefix + gap);
                const kq_diff diff_ctrl = compare_kq(kq_ref, cap.data, thold);

                const double nmse_logits = nmse(logits_ref, logits_shift);
                snprintf(shift_str,  sizeof(shift_str),  "%.2e", diff_shift.worst);
                snprintf(shift2_str, sizeof(shift2_str), "%.2e", diff_shift2.worst);
                snprintf(logits_str, sizeof(logits_str), "%.2e", nmse_logits);
                snprintf(ctrl_str,   sizeof(ctrl_str),   "%.2e", diff_ctrl.worst);

                if (kq_ref.empty()) {
                    status = "\033[1;33mSKIP\033[0m (no kq captured)";
                } else if (!diff_shift.first_bad.empty() || !diff_shift2.first_bad.empty()) {
                    n_failed++;
                    status = "\033[1;31mFAIL\033[0m" + (diff_shift.bad.empty() ? " x2:" + diff_shift2.bad : diff_shift.bad);
                } else if (!(diff_ctrl.worst > 10*thold)) {
                    status = "\033[1;33mOK\033[0m (low sensitivity)";
                } else {
                    status = "\033[1;32mOK\033[0m";
                }
            } catch (const std::exception & err) {
                n_failed++;
                status = std::string("\033[1;31mERROR\033[0m (") + err.what() + ")";
            }
            LOG("%10s|%10s|%10s|%10s|%s\n", shift_str, shift2_str, logits_str, ctrl_str, status.c_str());
        }
    }

    if (n_tests == 0) {
        LOG("Summary: no tests executed\n");
    } else if (n_failed == 0) {
        LOG("Summary: all %zu test(s) passed\n", n_tests);
    } else {
        LOG("Summary: %zu test(s) executed, %zu failed\n", n_tests, n_failed);
    }

    llama_log_set(ud.log_old.callback, ud.log_old.user_data);
    return n_failed == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    // init the logger at max verbosity. filter with a custom callback respecting the user-configure verbosity
    common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);
    common_init();
    llama_backend_init();

    std::random_device rd;

    std::string arch_filter;
    size_t seed = rd();
    float stdev = 0.1f;
    std::string out;
    bool test_phase_workspace = false;
    bool test_live_context_workspace = false;
    bool run_speculative_limits = false;
    const char * target_backend = nullptr;
    bool kv_shift = false;

    int verbosity = LOG_LEVEL_ERROR;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--test-phase-workspace") == 0) {
            test_phase_workspace = true;
            continue;
        }
        if (strcmp(argv[i], "--test-live-context-workspace") == 0) {
            test_live_context_workspace = true;
            continue;
        }
        if (strcmp(argv[i], "--test-speculative-limits") == 0) {
            run_speculative_limits = true;
            continue;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv);
            return 0;
        } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--arch") == 0) {
            if (i + 1 < argc) {
                const std::string arch_name = argv[++i];
                if (llama_arch_from_string(arch_name) != LLM_ARCH_UNKNOWN) {
                    // exact architecture name
                    arch_filter = "^" + arch_name + "$";
                } else {
                    try {
                        std::regex re(arch_name);
                        arch_filter = arch_name;
                    } catch (const std::regex_error & err) {
                        LOG_ERR("%s: invalid architecture regex: %s (%s)\n", __func__, arch_name.c_str(), err.what());
                        return 1;
                    }
                }
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) {
                seed = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--stdev") == 0) {
            if (i + 1 < argc) {
                stdev = std::stof(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0) {
            if (i + 1 < argc) {
                verbosity = std::stoull(argv[++i]);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out = argv[++i];
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--backend") == 0) {
            if (i + 1 < argc) {
                const char * backend_name = argv[++i];
                ggml_backend_dev_t dev = ggml_backend_dev_by_name(backend_name);
                if (dev == nullptr) {
                    LOG_ERR("%s: unknown backend device: %s\n", __func__, backend_name);
                    return 1;
                }
                target_backend = ggml_backend_dev_name(dev);
            } else {
                usage(argv);
                return 1;
            }
        } else if (strcmp(argv[i], "--kv-shift") == 0) {
            kv_shift = true;
        } else {
            LOG_ERR("%s: unknown argument: %s\n", __func__, argv[i]);
            usage(argv);
            return 1;
        }
    }
    if (test_phase_workspace || test_live_context_workspace || run_speculative_limits) {
        common_log_set_verbosity_thold(verbosity);
    }
    if (stdev <= 0.0f) {
        LOG_ERR("%s: stdev must be > 0\n", __func__);
        return 1;
    }
    LOG_INF("%s: using seed %zu, stdev %f\n", __func__, seed, stdev);

    init_normal_pool(seed, stdev);

    try {
        if (test_phase_workspace) {
            test_phase_workspace_runtime_reserve(seed);
            test_phase_workspace_mtp_lifecycle(seed);
            test_phase_workspace_mismatched_placement(seed);
            test_phase_workspace_late_pipeline_fallback(seed);
            return 0;
        }
        if (test_live_context_workspace) {
            test_live_context_workspace_reserve(seed);
            test_live_context_workspace_iswa_reserve(seed);
            test_live_context_workspace_indexer_reserve(seed);
            test_live_context_workspace_unsupported(seed);
            return 0;
        }
        if (run_speculative_limits) {
            test_speculative_limits(seed);
            return 0;
        }
        if (!out.empty()) {
            return save_models(arch_filter, seed, stdev, verbosity, out);
        }
        if (kv_shift) {
            return test_kv_shift(arch_filter, seed, stdev, verbosity);
        }
        return test_backends(arch_filter, seed, stdev, verbosity, target_backend);
    } catch (const std::exception & err) {
        fprintf(stderr, "encountered runtime error: %s\n", err.what());
        return -1;
    }
}
