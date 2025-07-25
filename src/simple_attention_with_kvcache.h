#pragma once

/**
 * Simple Attention Layer with KV Cache - Minimal Working Implementation
 * 
 * This is a simplified attention implementation that:
 * 1. Uses working dependencies from parent directory
 * 2. Implements basic KV cache functionality  
 * 3. Compiles successfully with current setup
 * 4. Provides foundation for real NPU integration
 */

#include <vector>
#include <memory>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace npugpt {

/**
 * Basic configuration for the attention layer
 */
struct AttentionConfig {
    size_t n_heads = 12;
    size_t head_dim = 64;
    size_t hidden_size = 768;  // n_heads * head_dim
    size_t max_seq_len = 1024;
    
    AttentionConfig() = default;
    AttentionConfig(size_t heads, size_t h_dim, size_t max_len = 1024) 
        : n_heads(heads), head_dim(h_dim), hidden_size(heads * h_dim), max_seq_len(max_len) {}
};

/**
 * Simple KV Cache for storing key and value tensors
 */
class SimpleKVCache {
private:
    AttentionConfig config_;
    
    // Cache storage: [seq_pos][head][head_dim]
    std::vector<std::vector<std::vector<float>>> cached_keys_;
    std::vector<std::vector<std::vector<float>>> cached_values_;
    
    size_t current_seq_len_ = 0;
    mutable size_t cache_hits_ = 0;
    mutable size_t cache_misses_ = 0;
    
public:
    explicit SimpleKVCache(const AttentionConfig& config) 
        : config_(config) {
        reset();
    }
    
    void reset() {
        current_seq_len_ = 0;
        cache_hits_ = 0;
        cache_misses_ = 0;
        
        // Initialize cache storage
        cached_keys_.clear();
        cached_values_.clear();
        cached_keys_.resize(config_.max_seq_len);
        cached_values_.resize(config_.max_seq_len);
        
        for (size_t i = 0; i < config_.max_seq_len; ++i) {
            cached_keys_[i].resize(config_.n_heads);
            cached_values_[i].resize(config_.n_heads);
            for (size_t h = 0; h < config_.n_heads; ++h) {
                cached_keys_[i][h].resize(config_.head_dim, 0.0f);
                cached_values_[i][h].resize(config_.head_dim, 0.0f);
            }
        }
    }
    
    // Add new key-value pair to cache
    void append_kv(const std::vector<std::vector<float>>& keys,
                   const std::vector<std::vector<float>>& values) {
        if (current_seq_len_ >= config_.max_seq_len) {
            throw std::runtime_error("KV cache overflow");
        }
        
        for (size_t h = 0; h < config_.n_heads; ++h) {
            cached_keys_[current_seq_len_][h] = keys[h];
            cached_values_[current_seq_len_][h] = values[h];
        }
        current_seq_len_++;
    }
    
    // Get cached keys for all positions up to current sequence length
    std::vector<std::vector<std::vector<float>>> get_cached_keys() const {
        std::vector<std::vector<std::vector<float>>> result;
        for (size_t i = 0; i < current_seq_len_; ++i) {
            result.push_back(cached_keys_[i]);
        }
        cache_hits_++;
        return result;
    }
    
    // Get cached values for all positions up to current sequence length  
    std::vector<std::vector<std::vector<float>>> get_cached_values() const {
        std::vector<std::vector<std::vector<float>>> result;
        for (size_t i = 0; i < current_seq_len_; ++i) {
            result.push_back(cached_values_[i]);
        }
        cache_hits_++;
        return result;
    }
    
    size_t get_seq_len() const { return current_seq_len_; }
    size_t get_cache_hits() const { return cache_hits_; }
    size_t get_cache_misses() const { return cache_misses_; }
    
    // Memory usage estimation
    size_t get_memory_usage_mb() const {
        size_t total_elements = current_seq_len_ * config_.n_heads * config_.head_dim * 2; // keys + values
        return (total_elements * sizeof(float)) / (1024 * 1024);
    }
};

/**
 * Simple Attention Layer with KV Cache Support
 */
class SimpleAttentionWithKVCache {
private:
    AttentionConfig config_;
    std::unique_ptr<SimpleKVCache> kv_cache_;
    
    // Performance tracking
    mutable size_t forward_calls_ = 0;
    mutable double total_forward_time_ms_ = 0.0;
    
public:
    explicit SimpleAttentionWithKVCache(const AttentionConfig& config)
        : config_(config), kv_cache_(std::make_unique<SimpleKVCache>(config)) {}
    
    // Forward pass with KV caching
    std::vector<float> forward(const std::vector<float>& input_hidden_states,
                              bool use_cache = true) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Simulate query, key, value projection
        // In real implementation, this would be actual matrix multiplication
        auto qkv = compute_qkv_projections(input_hidden_states);
        auto& queries = qkv.queries;
        auto& keys = qkv.keys;
        auto& values = qkv.values;
        
        std::vector<float> output;
        
        if (use_cache) {
            // Add current key-value to cache
            kv_cache_->append_kv(keys, values);
            
            // Get all cached keys and values
            auto cached_keys = kv_cache_->get_cached_keys();
            auto cached_values = kv_cache_->get_cached_values();
            
            // Compute attention with cached KV
            output = compute_attention_with_cache(queries, cached_keys, cached_values);
        } else {
            // Standard attention without cache
            output = compute_attention_standard(queries, keys, values);
        }
        
        // Track performance
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
        total_forward_time_ms_ += duration.count();
        forward_calls_++;
        
        return output;
    }
    
    void reset_cache() {
        kv_cache_->reset();
    }
    
    // Performance metrics
    double get_avg_forward_time_ms() const {
        return forward_calls_ > 0 ? total_forward_time_ms_ / forward_calls_ : 0.0;
    }
    
    size_t get_cache_hits() const { return kv_cache_->get_cache_hits(); }
    size_t get_memory_usage_mb() const { return kv_cache_->get_memory_usage_mb(); }
    size_t get_seq_len() const { return kv_cache_->get_seq_len(); }
    
private:
    struct QKVProjections {
        std::vector<std::vector<float>> queries;
        std::vector<std::vector<float>> keys;
        std::vector<std::vector<float>> values;
    };
    
    // Simulate QKV projections (in real implementation, this would use NPU MatMul)
    QKVProjections compute_qkv_projections(const std::vector<float>& input) {
        QKVProjections qkv;
        qkv.queries.resize(config_.n_heads);
        qkv.keys.resize(config_.n_heads);
        qkv.values.resize(config_.n_heads);
        
        for (size_t h = 0; h < config_.n_heads; ++h) {
            qkv.queries[h].resize(config_.head_dim);
            qkv.keys[h].resize(config_.head_dim);
            qkv.values[h].resize(config_.head_dim);
            
            // Simplified projection simulation
            for (size_t d = 0; d < config_.head_dim; ++d) {
                float base_val = (h * config_.head_dim + d < input.size()) ? 
                    input[h * config_.head_dim + d] : 0.1f;
                qkv.queries[h][d] = base_val + 0.1f;
                qkv.keys[h][d] = base_val + 0.2f;
                qkv.values[h][d] = base_val + 0.3f;
            }
        }
        
        return qkv;
    }
    
    // Simplified attention computation with cache
    std::vector<float> compute_attention_with_cache(
        const std::vector<std::vector<float>>& queries,
        const std::vector<std::vector<std::vector<float>>>& cached_keys,
        const std::vector<std::vector<std::vector<float>>>& cached_values) {
        
        std::vector<float> output(config_.hidden_size, 0.0f);
        
        // Simplified attention: for each head, compute attention over cached sequence
        for (size_t h = 0; h < config_.n_heads; ++h) {
            std::vector<float> head_output(config_.head_dim, 0.0f);
            
            // Compute attention weights for this head over cached sequence
            std::vector<float> attention_weights(cached_keys.size(), 0.0f);
            float sum_weights = 0.0f;
            
            for (size_t t = 0; t < cached_keys.size(); ++t) {
                // Simplified dot-product attention
                float score = 0.0f;
                for (size_t d = 0; d < config_.head_dim; ++d) {
                    score += queries[h][d] * cached_keys[t][h][d];
                }
                attention_weights[t] = std::exp(score / std::sqrt(config_.head_dim));
                sum_weights += attention_weights[t];
            }
            
            // Normalize attention weights
            if (sum_weights > 0.0f) {
                for (auto& weight : attention_weights) {
                    weight /= sum_weights;
                }
            }
            
            // Apply attention weights to cached values
            for (size_t t = 0; t < cached_values.size(); ++t) {
                for (size_t d = 0; d < config_.head_dim; ++d) {
                    head_output[d] += attention_weights[t] * cached_values[t][h][d];
                }
            }
            
            // Copy head output to final output
            for (size_t d = 0; d < config_.head_dim; ++d) {
                output[h * config_.head_dim + d] = head_output[d];
            }
        }
        
        return output;
    }
    
    // Standard attention without cache (for comparison)
    std::vector<float> compute_attention_standard(
        const std::vector<std::vector<float>>& queries,
        const std::vector<std::vector<float>>& keys,
        const std::vector<std::vector<float>>& values) {
        
        std::vector<float> output(config_.hidden_size, 0.0f);
        
        // Simplified single-token attention
        for (size_t h = 0; h < config_.n_heads; ++h) {
            for (size_t d = 0; d < config_.head_dim; ++d) {
                output[h * config_.head_dim + d] = values[h][d];
            }
        }
        
        return output;
    }
};

} // namespace npugpt