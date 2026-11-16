#include "llama-context.h"
#include "llama-batch.h"

#include "llama-impl.h"
#include "llama-memory-hybrid-idx.h"
#include "llama-model.h"

bool llama_context::can_decode_sampled_host() const {
    const auto * hybrid = dynamic_cast<const llama_memory_hybrid_idx *>(memory.get());
    return model.arch == LLM_ARCH_QWEN4EXP && n_seq_max() == 1 && !shared_workspace_peer() && hybrid &&
        hybrid->llama_memory_hybrid::can_decode_sampled() &&
        (!hybrid->get_mem_idx() || hybrid->get_mem_idx()->can_decode_sampled());
}

int32_t llama_context::decode_sampled_host(
        const llama_sampled_decode_item * items,
        int32_t n_items,
        const std::vector<ggml_tensor *> & sources,
        llama_token * previous) {
    synchronize();

    if (!sampled_batch) {
        sampled_batch = std::make_unique<llama_batch_ext>(this);
    }
    sampled_batch->clear();
    for (int32_t i = 0; i < n_items; ++i) {
        ggml_backend_tensor_get(sources[i], &previous[i], 0, sizeof(llama_token));
        llama_batch_ext::token token;
        token.id = previous[i];
        token.pos[0] = items[i].pos;
        token.seq_ids.insert(items[i].seq_id);
        token.output = true;
        sampled_batch->tokens.push_back(std::move(token));
    }

    // PLE hashes and KV token history need the real tokens before the next graph is queued.
    const int32_t ret = decode(*sampled_batch);
    if (ret == 0) {
        LLAMA_LOG_DEBUG("%s: decode overlap with host token inputs, n_seqs = %d\n", __func__, n_items);
    }
    return ret == 0 ? 0 : (ret < 0 ? ret : -3);
}
