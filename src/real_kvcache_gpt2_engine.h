#pragma once

/**
 * Real GPT-2 Engine with KV Cache Integration
 * 
 * This integrates KV caching with the actual picoGPT GPT-2 engine
 * for real NPU inference with authentic text generation
 */

#include "../../picogpt/src/gpt2_engine.h"
#include "../../picogpt/src/gpt2_types.h"
#include "../include/kv_cache.h"
#include <memory>
#include <chrono>

namespace picogpt {

/**
 * GPT-2 Engine with KV Cache for Accelerated Generation
 * 
 * Two variants:
 *   - Non-fused: Separate KV cache operations + FlashAttention-2  
 *   - Fused: Single NPU kernel combining cache + attention
 */
class KVCacheGPT2Engine : public GPT2Engine {
private:
    // KV cache infrastructure
    std::unique_ptr<atsentia::models::gpt2::KVCache> kv_cache_;
    atsentia::models::gpt2::KVCacheConfig cache_config_;
    
    // Performance tracking
    mutable std::chrono::microseconds total_prefill_time_{0};
    mutable std::chrono::microseconds total_generation_time_{0};
    mutable uint32_t prefill_tokens_ = 0;
    mutable uint32_t generated_tokens_ = 0;
    mutable bool is_fused_variant_ = false;
    
public:
    KVCacheGPT2Engine(const GPT2Config& config, 
                      std::unique_ptr<GPT2Weights> weights,
                      std::unique_ptr<Tokenizer> tokenizer,
                      bool fused = false);
    
    // Enhanced generation with KV cache
    std::string generate(const std::string& prompt, 
                        int max_tokens = 50,
                        float temperature = 1.0f) override;
    
    // KV cache specific methods
    void clear_cache();
    void print_cache_statistics() const;
    
    // Performance analysis
    void print_performance_comparison() const;
    
private:
    // Dual-phase forward pass
    std::vector<float> forward_prefill(const std::vector<int32_t>& input_ids);
    std::vector<float> forward_cached(int32_t new_token, uint32_t position);
    
    // KV cache operations
    void append_to_cache(const std::vector<float>& keys, 
                        const std::vector<float>& values, 
                        uint32_t layer_idx);
    
    // Enhanced attention with cache
    npu::Tensor multi_head_attention_with_cache(const npu::Tensor& x, 
                                               int block_idx, 
                                               uint32_t position);
    
    // Implementation methods
    std::vector<float> forward_cached_nonfused(int32_t new_token, uint32_t position);
    std::vector<float> forward_cached_fused(int32_t new_token, uint32_t position);
    
    // Token sampling
    int32_t sample_token(const std::vector<float>& logits, float temperature);
};

/**
 * Factory functions for creating engine variants
 */
std::unique_ptr<KVCacheGPT2Engine> create_nonfused_kvcache_engine(
    const GPT2Config& config,
    std::unique_ptr<GPT2Weights> weights,
    std::unique_ptr<Tokenizer> tokenizer);

std::unique_ptr<KVCacheGPT2Engine> create_fused_kvcache_engine(
    const GPT2Config& config,
    std::unique_ptr<GPT2Weights> weights, 
    std::unique_ptr<Tokenizer> tokenizer);

} // namespace picogpt