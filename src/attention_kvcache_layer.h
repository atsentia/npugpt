#pragma once

/**
 * Standalone Attention Layer with KV Cache Support
 * 
 * Implements a single attention layer with integrated KV caching
 * for efficient autoregressive generation on Qualcomm NPU
 */

#include "../include/kv_cache.h"
#include "../include/standalone_qnn_static_graph.h"
#include <vector>
#include <memory>
#include <cmath>
#include <iostream>

namespace atsentia {
namespace models {
namespace gpt2 {

/**
 * Configuration for attention layer with KV cache
 */
struct AttentionKVCacheConfig {
    uint32_t num_heads = 12;
    uint32_t head_dim = 64;
    uint32_t max_seq_len = 2048;
    uint32_t layer_idx = 0;
    
    // KV cache specific
    bool enable_kv_cache = true;
    uint32_t cache_block_size = 64;
    
    // NPU optimization
    bool use_npu = true;
    bool fuse_qkv_projection = true;
    bool fuse_attention_output = true;
    
    // FlashAttention-2 parameters
    bool use_flash_attention = true;
    uint32_t block_size_q = 64;
    uint32_t block_size_k = 64;
    
    AttentionKVCacheConfig() = default;
    
    AttentionKVCacheConfig(uint32_t heads, uint32_t dim, uint32_t max_len, uint32_t layer)
        : num_heads(heads), head_dim(dim), max_seq_len(max_len), layer_idx(layer) {}
    
    uint32_t hidden_size() const { return num_heads * head_dim; }
    
    void print() const {
        std::cout << "Attention Layer Config (Layer " << layer_idx << "):" << std::endl;
        std::cout << "  Heads: " << num_heads << ", Head dim: " << head_dim << std::endl;
        std::cout << "  Max seq len: " << max_seq_len << std::endl;
        std::cout << "  KV cache: " << (enable_kv_cache ? "enabled" : "disabled") << std::endl;
        std::cout << "  NPU: " << (use_npu ? "enabled" : "disabled") << std::endl;
        std::cout << "  FlashAttention: " << (use_flash_attention ? "enabled" : "disabled") << std::endl;
    }
};

/**
 * Standalone Attention Layer with KV Cache
 * Can be used independently or as part of a larger model
 */
class AttentionKVCacheLayer {
private:
    AttentionKVCacheConfig config_;
    std::unique_ptr<LayerKVCache> kv_cache_;
    std::unique_ptr<qualcomm_npu::QNNStaticGraph> npu_graph_;
    
    // Weight storage
    std::vector<float> qkv_weight_;  // [hidden_size, 3 * hidden_size]
    std::vector<float> qkv_bias_;    // [3 * hidden_size]
    std::vector<float> out_weight_;  // [hidden_size, hidden_size]
    std::vector<float> out_bias_;    // [hidden_size]
    
    // Performance metrics
    struct Metrics {
        uint32_t forward_calls = 0;
        uint32_t cache_hits = 0;
        uint32_t cache_misses = 0;
        std::chrono::microseconds total_time{0};
        std::chrono::microseconds qkv_time{0};
        std::chrono::microseconds attention_time{0};
        std::chrono::microseconds output_time{0};
        
        void print() const {
            std::cout << "  Forward calls: " << forward_calls << std::endl;
            std::cout << "  Cache hit rate: " << std::fixed << std::setprecision(1)
                      << (100.0 * cache_hits / (std::max)(1u, cache_hits + cache_misses)) << "%" << std::endl;
            std::cout << "  Avg total time: " << (forward_calls > 0 ? total_time.count() / forward_calls : 0) << " μs" << std::endl;
            std::cout << "  Breakdown - QKV: " << (forward_calls > 0 ? qkv_time.count() / forward_calls : 0) << " μs"
                      << ", Attention: " << (forward_calls > 0 ? attention_time.count() / forward_calls : 0) << " μs"
                      << ", Output: " << (forward_calls > 0 ? output_time.count() / forward_calls : 0) << " μs" << std::endl;
        }
    } metrics_;

public:
    explicit AttentionKVCacheLayer(const AttentionKVCacheConfig& config)
        : config_(config) {
        
        // Initialize KV cache if enabled
        if (config_.enable_kv_cache) {
            kv_cache_ = std::make_unique<LayerKVCache>(
                config_.num_heads,
                config_.head_dim,
                config_.max_seq_len,
                config_.cache_block_size
            );
        }
        
        // Initialize weights (would be loaded from model in practice)
        initialize_weights();
        
        // Initialize NPU graph if enabled
        if (config_.use_npu) {
            initialize_npu_graph();
        }
    }
    
    /**
     * Forward pass with optional KV caching
     * 
     * @param input Hidden states [seq_len, hidden_size]
     * @param position Current position in sequence (for caching)
     * @param use_cache Whether to use KV cache for this forward pass
     * @return Attention output [seq_len, hidden_size]
     */
    std::vector<float> forward(const std::vector<float>& input,
                              uint32_t position = 0,
                              bool use_cache = true) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        uint32_t seq_len = input.size() / config_.hidden_size();
        uint32_t hidden_size = config_.hidden_size();
        
        // Step 1: QKV projection
        auto qkv_start = std::chrono::high_resolution_clock::now();
        auto [queries, keys, values] = compute_qkv(input, seq_len);
        auto qkv_end = std::chrono::high_resolution_clock::now();
        metrics_.qkv_time += std::chrono::duration_cast<std::chrono::microseconds>(qkv_end - qkv_start);
        
        // Step 2: KV caching logic
        auto attn_start = std::chrono::high_resolution_clock::now();
        std::vector<float> attention_output;
        
        if (use_cache && config_.enable_kv_cache && kv_cache_) {
            if (seq_len == 1 && position > 0) {
                // Generation phase - use cached K,V
                attention_output = forward_with_cache(queries, keys, values, position);
                metrics_.cache_hits++;
            } else {
                // Prefill phase - populate cache
                attention_output = forward_prefill(queries, keys, values);
                metrics_.cache_misses++;
            }
        } else {
            // No caching - standard attention
            attention_output = compute_attention(queries, keys, values, seq_len);
        }
        auto attn_end = std::chrono::high_resolution_clock::now();
        metrics_.attention_time += std::chrono::duration_cast<std::chrono::microseconds>(attn_end - attn_start);
        
        // Step 3: Output projection
        auto out_start = std::chrono::high_resolution_clock::now();
        auto output = compute_output_projection(attention_output, seq_len);
        auto out_end = std::chrono::high_resolution_clock::now();
        metrics_.output_time += std::chrono::duration_cast<std::chrono::microseconds>(out_end - out_start);
        
        // Update metrics
        auto end_time = std::chrono::high_resolution_clock::now();
        metrics_.total_time += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        metrics_.forward_calls++;
        
        return output;
    }
    
    /**
     * Clear KV cache for new sequence
     */
    void clear_cache() {
        if (kv_cache_) {
            kv_cache_->clear();
        }
        metrics_ = Metrics{};
    }
    
    /**
     * Get current cache sequence length
     */
    uint32_t get_cache_seq_len() const {
        return kv_cache_ ? kv_cache_->get_seq_len() : 0;
    }
    
    /**
     * Print layer statistics
     */
    void print_stats() const {
        std::cout << "\nAttention Layer " << config_.layer_idx << " Statistics:" << std::endl;
        metrics_.print();
        if (kv_cache_) {
            kv_cache_->print_stats();
        }
    }
    
    /**
     * Get/set weights for external loading
     */
    void set_weights(const std::vector<float>& qkv_w, const std::vector<float>& qkv_b,
                     const std::vector<float>& out_w, const std::vector<float>& out_b) {
        qkv_weight_ = qkv_w;
        qkv_bias_ = qkv_b;
        out_weight_ = out_w;
        out_bias_ = out_b;
    }

private:
    void initialize_weights() {
        uint32_t hidden_size = config_.hidden_size();
        
        // Initialize with small random values (in practice, load from model)
        qkv_weight_.resize(hidden_size * 3 * hidden_size, 0.02f);
        qkv_bias_.resize(3 * hidden_size, 0.0f);
        out_weight_.resize(hidden_size * hidden_size, 0.02f);
        out_bias_.resize(hidden_size, 0.0f);
    }
    
    void initialize_npu_graph() {
        // Initialize NPU graph for fused operations
        // In real implementation, this would create QNN graph
        npu_graph_ = std::make_unique<qualcomm_npu::QNNStaticGraph>();
    }
    
    std::tuple<std::vector<float>, std::vector<float>, std::vector<float>>
    compute_qkv(const std::vector<float>& input, uint32_t seq_len) {
        uint32_t hidden_size = config_.hidden_size();
        std::vector<float> qkv(seq_len * 3 * hidden_size);
        
        // QKV projection: input @ qkv_weight + qkv_bias
        for (uint32_t s = 0; s < seq_len; ++s) {
            for (uint32_t i = 0; i < 3 * hidden_size; ++i) {
                float sum = qkv_bias_[i];
                for (uint32_t j = 0; j < hidden_size; ++j) {
                    sum += input[s * hidden_size + j] * qkv_weight_[j * 3 * hidden_size + i];
                }
                qkv[s * 3 * hidden_size + i] = sum;
            }
        }
        
        // Split into Q, K, V
        std::vector<float> queries(seq_len * hidden_size);
        std::vector<float> keys(seq_len * hidden_size);
        std::vector<float> values(seq_len * hidden_size);
        
        for (uint32_t s = 0; s < seq_len; ++s) {
            for (uint32_t i = 0; i < hidden_size; ++i) {
                queries[s * hidden_size + i] = qkv[s * 3 * hidden_size + i];
                keys[s * hidden_size + i] = qkv[s * 3 * hidden_size + hidden_size + i];
                values[s * hidden_size + i] = qkv[s * 3 * hidden_size + 2 * hidden_size + i];
            }
        }
        
        return {queries, keys, values};
    }
    
    std::vector<float> forward_prefill(const std::vector<float>& queries,
                                      const std::vector<float>& keys,
                                      const std::vector<float>& values) {
        uint32_t seq_len = queries.size() / config_.hidden_size();
        
        // Cache all K,V pairs
        for (uint32_t pos = 0; pos < seq_len; ++pos) {
            const float* k_ptr = keys.data() + pos * config_.hidden_size();
            const float* v_ptr = values.data() + pos * config_.hidden_size();
            kv_cache_->append_kv(k_ptr, v_ptr);
        }
        
        // Compute attention with all cached values
        return compute_attention(queries, keys, values, seq_len);
    }
    
    std::vector<float> forward_with_cache(const std::vector<float>& queries,
                                         const std::vector<float>& keys,
                                         const std::vector<float>& values,
                                         uint32_t position) {
        // Append new K,V to cache
        kv_cache_->append_kv(keys.data(), values.data());
        
        // Get all cached K,V
        uint32_t cache_len = kv_cache_->get_seq_len();
        std::vector<float> all_keys(cache_len * config_.hidden_size());
        std::vector<float> all_values(cache_len * config_.hidden_size());
        
        // Convert from block layout to standard layout
        for (uint32_t h = 0; h < config_.num_heads; ++h) {
            const float* cached_k = kv_cache_->get_key_head(h);
            const float* cached_v = kv_cache_->get_value_head(h);
            
            for (uint32_t pos = 0; pos < cache_len; ++pos) {
                uint32_t block_idx = pos / config_.cache_block_size;
                uint32_t block_offset = pos % config_.cache_block_size;
                
                size_t src_offset = block_idx * config_.cache_block_size * config_.head_dim +
                                   block_offset * config_.head_dim;
                
                for (uint32_t d = 0; d < config_.head_dim; ++d) {
                    all_keys[pos * config_.hidden_size() + h * config_.head_dim + d] = cached_k[src_offset + d];
                    all_values[pos * config_.hidden_size() + h * config_.head_dim + d] = cached_v[src_offset + d];
                }
            }
        }
        
        // Compute attention with cached K,V
        return compute_attention(queries, all_keys, all_values, 1, cache_len);
    }
    
    std::vector<float> compute_attention(const std::vector<float>& queries,
                                        const std::vector<float>& keys,
                                        const std::vector<float>& values,
                                        uint32_t q_len,
                                        uint32_t kv_len = 0) {
        if (kv_len == 0) kv_len = q_len;
        
        uint32_t hidden_size = config_.hidden_size();
        uint32_t num_heads = config_.num_heads;
        uint32_t head_dim = config_.head_dim;
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        
        std::vector<float> output(q_len * hidden_size, 0.0f);
        
        // Process each attention head
        for (uint32_t h = 0; h < num_heads; ++h) {
            // Compute attention scores for this head
            std::vector<float> scores(q_len * kv_len);
            
            for (uint32_t i = 0; i < q_len; ++i) {
                for (uint32_t j = 0; j < kv_len; ++j) {
                    float score = 0.0f;
                    for (uint32_t d = 0; d < head_dim; ++d) {
                        score += queries[i * hidden_size + h * head_dim + d] *
                                keys[j * hidden_size + h * head_dim + d];
                    }
                    scores[i * kv_len + j] = score * scale;
                }
            }
            
            // Apply causal mask if needed
            for (uint32_t i = 0; i < q_len; ++i) {
                for (uint32_t j = i + 1; j < kv_len; ++j) {
                    scores[i * kv_len + j] = -INFINITY;
                }
            }
            
            // Softmax
            for (uint32_t i = 0; i < q_len; ++i) {
                float max_score = -INFINITY;
                for (uint32_t j = 0; j <= i; ++j) {
                    max_score = (std::max)(max_score, scores[i * kv_len + j]);
                }
                
                float sum_exp = 0.0f;
                for (uint32_t j = 0; j <= i; ++j) {
                    scores[i * kv_len + j] = std::exp(scores[i * kv_len + j] - max_score);
                    sum_exp += scores[i * kv_len + j];
                }
                
                for (uint32_t j = 0; j <= i; ++j) {
                    scores[i * kv_len + j] /= sum_exp;
                }
            }
            
            // Apply attention to values
            for (uint32_t i = 0; i < q_len; ++i) {
                for (uint32_t d = 0; d < head_dim; ++d) {
                    float sum = 0.0f;
                    for (uint32_t j = 0; j <= i; ++j) {
                        sum += scores[i * kv_len + j] * values[j * hidden_size + h * head_dim + d];
                    }
                    output[i * hidden_size + h * head_dim + d] = sum;
                }
            }
        }
        
        return output;
    }
    
    std::vector<float> compute_output_projection(const std::vector<float>& attention_output,
                                                uint32_t seq_len) {
        uint32_t hidden_size = config_.hidden_size();
        std::vector<float> output(seq_len * hidden_size);
        
        // Output projection: attention_output @ out_weight + out_bias
        for (uint32_t s = 0; s < seq_len; ++s) {
            for (uint32_t i = 0; i < hidden_size; ++i) {
                float sum = out_bias_[i];
                for (uint32_t j = 0; j < hidden_size; ++j) {
                    sum += attention_output[s * hidden_size + j] * out_weight_[j * hidden_size + i];
                }
                output[s * hidden_size + i] = sum;
            }
        }
        
        return output;
    }
};

} // namespace gpt2
} // namespace models
} // namespace atsentia