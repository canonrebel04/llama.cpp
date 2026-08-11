// Fuse3 model header for llama.cpp
// Place in src/models/ alongside the other model headers

#pragma once

#include "models.h"
#include "../llama-memory-hybrid-iswa.h"
#include "../llama-memory-hybrid.h"

// Fuse3 layer tensors
struct llama_model_fuse3_layer {
    ggml_tensor * fuse3_router = nullptr;
    ggml_tensor * fuse3_expert_scale = nullptr;
    ggml_tensor * fuse3_expert_gate = nullptr;
    ggml_tensor * fuse3_expert_up   = nullptr;
    ggml_tensor * fuse3_expert_down = nullptr;
    int n_experts = 0;
};

struct llama_model_fuse3 : public llama_model_lfm2 {
    llama_model_fuse3(const struct llama_model_params & params) : llama_model_lfm2(params) {}
    std::vector<llama_model_fuse3_layer> fuse3_layers;
    std::vector<int> fuse3_expert_counts;

    // Extra hparams
    float fuse3_swiglu_limit = 10.0f;

    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;

    template <bool iswa>
    struct graph : llm_graph_context {
        graph(const llama_model & model, const llm_graph_params & params);
    };
};
