/**
 * FlashAttention-2 with KV Cache NPU GPT-2 Engine Implementation (Non-Fused)
 * 
 * Combines memory-efficient O(N) attention with KV caching for 
 * accelerated autoregressive generation on Qualcomm Snapdragon X Elite NPU
 */

#include "flashattention2_gpt2_engine.h"
#include "../include/kv_cache.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <thread>
#include <cstdlib>
#include <filesystem>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace atsentia {
namespace models {
namespace gpt2 {

// ============================================================================
// NPUFlashAttention2KVCache Implementation - Non-Fused Version
// ============================================================================

class NPUFlashAttention2KVCache : public NPUFlashAttention2 {
private:
    std::unique_ptr<KVCache> kv_cache_;
    KVCacheConfig kv_config_;
    
    // Performance metrics specific to KV cache
    struct KVCacheMetrics {
        uint32_t cache_appends = 0;
        uint32_t cache_reads = 0;
        uint32_t tokens_generated = 0;
        std::chrono::microseconds total_cache_time{0};
        std::chrono::microseconds total_generation_time{0};
        
        double avg_cache_time_us() const {
            if (cache_appends == 0) return 0.0;
            return static_cast<double>(total_cache_time.count()) / cache_appends;
        }
        
        double speedup_vs_no_cache() const {
            // Theoretical speedup based on avoiding recomputation
            if (tokens_generated <= 1) return 1.0;
            return static_cast<double>(tokens_generated * tokens_generated) / 
                   (2.0 * tokens_generated - 1.0);
        }
    } kv_metrics_;

public:
    NPUFlashAttention2KVCache(const FlashAttentionConfig& flash_config,
                              const KVCacheConfig& kv_config)
        : NPUFlashAttention2(flash_config), kv_config_(kv_config) {
        
        std::cout << "[FlashAttention2KVCache] Initializing with KV caching support" << std::endl;
        std::cout << "  KV cache enabled for " << kv_config.n_layers << " layers" << std::endl;
        std::cout << "  Max sequence length: " << kv_config.max_seq_len << std::endl;
        std::cout << "  Cache block size: " << kv_config.cache_block_size << std::endl;
        
        // Initialize KV cache
        kv_cache_ = std::make_unique<KVCache>(kv_config);
    }
    
    // Override forward to use KV cache
    std::vector<float> forward_with_cache(
        const std::vector<float>& queries,
        const std::vector<float>& keys,
        const std::vector<float>& values,
        uint32_t layer_idx,
        uint32_t position) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Get layer cache
        auto* layer_cache = kv_cache_->get_layer_cache(layer_idx);
        
        // For single token generation (position > 0), only process new token
        bool is_prefill = (position == 0) || (layer_cache->get_seq_len() == 0);
        
        if (is_prefill) {
            // Prefill phase: process entire sequence
            return forward_prefill(queries, keys, values, layer_idx);
        } else {
            // Generation phase: use cached K,V
            return forward_cached(queries, keys, values, layer_idx, position);
        }
    }
    
    // Clear cache for new sequence
    void clear_cache() {
        kv_cache_->clear_all();
        kv_metrics_ = KVCacheMetrics{};
    }
    
    // Get cache statistics
    void print_cache_stats() const {
        std::cout << "\n📊 KV Cache Performance:" << std::endl;
        std::cout << "  Cache appends: " << kv_metrics_.cache_appends << std::endl;
        std::cout << "  Cache reads: " << kv_metrics_.cache_reads << std::endl;
        std::cout << "  Tokens generated: " << kv_metrics_.tokens_generated << std::endl;
        std::cout << "  Avg cache time: " << std::fixed << std::setprecision(2)
                  << kv_metrics_.avg_cache_time_us() << " μs" << std::endl;
        std::cout << "  Theoretical speedup: " << std::fixed << std::setprecision(1)
                  << kv_metrics_.speedup_vs_no_cache() << "x" << std::endl;
        
        kv_cache_->print_performance_stats();
    }

private:
    // Prefill phase: process entire sequence and populate cache
    std::vector<float> forward_prefill(
        const std::vector<float>& queries,
        const std::vector<float>& keys,
        const std::vector<float>& values,
        uint32_t layer_idx) {
        
        auto cache_start = std::chrono::high_resolution_clock::now();
        
        // Get layer cache
        auto* layer_cache = kv_cache_->get_layer_cache(layer_idx);
        
        // Process all tokens and cache K,V
        uint32_t seq_len = queries.size() / (config_.num_heads * config_.head_dim);
        
        for (uint32_t pos = 0; pos < seq_len; ++pos) {
            // Extract K,V for current position
            const float* k_ptr = keys.data() + pos * config_.num_heads * config_.head_dim;
            const float* v_ptr = values.data() + pos * config_.num_heads * config_.head_dim;
            
            // Append to cache
            layer_cache->append_kv(k_ptr, v_ptr);
        }
        
        auto cache_end = std::chrono::high_resolution_clock::now();
        kv_metrics_.total_cache_time += 
            std::chrono::duration_cast<std::chrono::microseconds>(cache_end - cache_start);
        kv_metrics_.cache_appends += seq_len;
        
        // Run standard FlashAttention-2 for prefill
        return NPUFlashAttention2::forward(queries, keys, values);
    }
    
    // Generation phase: use cached K,V for efficient attention
    std::vector<float> forward_cached(
        const std::vector<float>& queries,
        const std::vector<float>& keys,
        const std::vector<float>& values,
        uint32_t layer_idx,
        uint32_t position) {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Get layer cache
        auto* layer_cache = kv_cache_->get_layer_cache(layer_idx);
        
        // Append new K,V to cache
        auto cache_start = std::chrono::high_resolution_clock::now();
        layer_cache->append_kv(keys.data(), values.data());
        auto cache_end = std::chrono::high_resolution_clock::now();
        
        kv_metrics_.total_cache_time += 
            std::chrono::duration_cast<std::chrono::microseconds>(cache_end - cache_start);
        kv_metrics_.cache_appends++;
        
        // Prepare tensors for FlashAttention-2 with cached K,V
        uint32_t cached_seq_len = layer_cache->get_seq_len();
        std::vector<float> all_keys(cached_seq_len * config_.num_heads * config_.head_dim);
        std::vector<float> all_values(cached_seq_len * config_.num_heads * config_.head_dim);
        
        // Copy cached K,V to contiguous buffers
        // This is non-fused version, so we do separate copy
        for (uint32_t h = 0; h < config_.num_heads; ++h) {
            const float* cached_k = layer_cache->get_key_head(h);
            const float* cached_v = layer_cache->get_value_head(h);
            
            // Convert from block layout to standard layout
            for (uint32_t pos = 0; pos < cached_seq_len; ++pos) {
                uint32_t block_idx = pos / kv_config_.cache_block_size;
                uint32_t block_offset = pos % kv_config_.cache_block_size;
                
                size_t src_offset = block_idx * kv_config_.cache_block_size * config_.head_dim +
                                   block_offset * config_.head_dim;
                size_t dst_offset = (pos * config_.num_heads + h) * config_.head_dim;
                
                std::memcpy(&all_keys[dst_offset], 
                           cached_k + src_offset, 
                           config_.head_dim * sizeof(float));
                
                std::memcpy(&all_values[dst_offset],
                           cached_v + src_offset,
                           config_.head_dim * sizeof(float));
            }
        }
        
        kv_metrics_.cache_reads++;
        
        // Run FlashAttention-2 with full K,V sequence
        auto output = NPUFlashAttention2::forward(queries, all_keys, all_values);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        kv_metrics_.total_generation_time += 
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        kv_metrics_.tokens_generated++;
        
        return output;
    }
};

// ============================================================================
// FlashAttention2KVCacheGPT2Engine - Complete Engine with KV Cache
// ============================================================================

class FlashAttention2KVCacheGPT2Engine : public NPUGpt2Engine {
private:
    std::vector<std::unique_ptr<NPUFlashAttention2KVCache>> attention_layers_;
    KVCacheConfig kv_config_;
    FlashAttentionConfig flash_config_;
    
    // Generation-specific state
    uint32_t current_position_ = 0;
    bool is_generating_ = false;

public:
    FlashAttention2KVCacheGPT2Engine() {
        std::cout << "🚀 Initializing FlashAttention-2 + KV Cache GPT-2 Engine (Non-Fused)" << std::endl;
    }
    
    void initialize(std::unique_ptr<GPT2Weights> weights) override {
        weights_ = std::move(weights);
        config_ = weights_->config;
        
        // Configure FlashAttention-2
        flash_config_ = FlashAttentionConfig(
            config_.max_position_embeddings,
            config_.hidden_size / config_.num_heads,
            config_.num_heads
        );
        
        // Configure KV cache
        kv_config_ = KVCacheConfig(
            config_.num_layers,
            config_.num_heads,
            config_.hidden_size / config_.num_heads,
            config_.max_position_embeddings
        );
        
        // Initialize attention layers with KV cache
        attention_layers_.clear();
        for (uint32_t i = 0; i < config_.num_layers; ++i) {
            attention_layers_.emplace_back(
                std::make_unique<NPUFlashAttention2KVCache>(flash_config_, kv_config_)
            );
        }
        
        std::cout << "✅ FlashAttention-2 + KV Cache initialization complete" << std::endl;
        std::cout << "  " << config_.num_layers << " layers with KV caching" << std::endl;
        std::cout << "  Max sequence length: " << config_.max_position_embeddings << std::endl;
    }
    
    std::vector<int> generate(const std::vector<int>& input_ids, 
                             uint32_t max_new_tokens,
                             float temperature = 1.0f,
                             uint32_t top_k = 0) override {
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Clear KV cache for new generation
        for (auto& layer : attention_layers_) {
            layer->clear_cache();
        }
        current_position_ = 0;
        is_generating_ = true;
        
        std::vector<int> output_ids = input_ids;
        
        // Prefill phase - process entire input
        std::cout << "📝 Prefill phase: " << input_ids.size() << " tokens" << std::endl;
        auto hidden_states = forward_batch(input_ids);
        current_position_ = input_ids.size();
        
        // Generation phase - one token at a time with KV cache
        std::cout << "🔄 Generation phase: " << max_new_tokens << " tokens" << std::endl;
        
        for (uint32_t i = 0; i < max_new_tokens; ++i) {
            // Forward pass for single token
            std::vector<int> next_token_id = {output_ids.back()};
            hidden_states = forward_single_token(next_token_id, current_position_);
            
            // Get logits for last position
            auto logits = compute_logits(hidden_states);
            
            // Sample next token
            int next_token = sample_token(logits, temperature, top_k);
            output_ids.push_back(next_token);
            current_position_++;
            
            // Early stopping on EOS
            if (next_token == 50256) break; // GPT-2 EOS token
        }
        
        is_generating_ = false;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Print generation statistics
        uint32_t tokens_generated = output_ids.size() - input_ids.size();
        std::cout << "\n⚡ Generation complete:" << std::endl;
        std::cout << "  Tokens generated: " << tokens_generated << std::endl;
        std::cout << "  Total time: " << duration.count() << " ms" << std::endl;
        std::cout << "  Speed: " << std::fixed << std::setprecision(1) 
                  << (duration.count() > 0 ? 1000.0 * tokens_generated / duration.count() : 0)
                  << " tokens/sec" << std::endl;
        
        // Print cache statistics
        attention_layers_[0]->print_cache_stats();
        
        return output_ids;
    }

private:
    // Forward pass for prefill (full sequence)
    std::vector<float> forward_batch(const std::vector<int>& input_ids) {
        uint32_t seq_len = input_ids.size();
        uint32_t hidden_size = config_.hidden_size;
        
        // Token embeddings
        std::vector<float> hidden_states = embed_tokens(input_ids);
        
        // Add positional embeddings
        add_positional_embeddings(hidden_states, 0, seq_len);
        
        // Process through transformer layers
        for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
            hidden_states = forward_transformer_layer(hidden_states, layer_idx, seq_len, 0);
        }
        
        // Final layer norm
        apply_layer_norm(hidden_states, weights_->ln_f_weight, weights_->ln_f_bias);
        
        return hidden_states;
    }
    
    // Forward pass for single token (with KV cache)
    std::vector<float> forward_single_token(const std::vector<int>& token_id, uint32_t position) {
        uint32_t hidden_size = config_.hidden_size;
        
        // Token embedding for single token
        std::vector<float> hidden_states = embed_tokens(token_id);
        
        // Add positional embedding for current position
        add_positional_embeddings(hidden_states, position, 1);
        
        // Process through transformer layers with KV cache
        for (uint32_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
            hidden_states = forward_transformer_layer(hidden_states, layer_idx, 1, position);
        }
        
        // Final layer norm
        apply_layer_norm(hidden_states, weights_->ln_f_weight, weights_->ln_f_bias);
        
        return hidden_states;
    }
    
    // Transformer layer with FlashAttention-2 + KV cache
    std::vector<float> forward_transformer_layer(
        const std::vector<float>& input,
        uint32_t layer_idx,
        uint32_t seq_len,
        uint32_t position) {
        
        const auto& layer = weights_->layers[layer_idx];
        uint32_t hidden_size = config_.hidden_size;
        
        // Pre-attention layer norm
        std::vector<float> normalized = input;
        apply_layer_norm(normalized, layer.ln_1_weight, layer.ln_1_bias);
        
        // QKV projection
        auto qkv = compute_qkv(normalized, layer);
        
        // Split into Q, K, V
        std::vector<float> queries(seq_len * hidden_size);
        std::vector<float> keys(seq_len * hidden_size);
        std::vector<float> values(seq_len * hidden_size);
        
        split_qkv(qkv, queries, keys, values, seq_len);
        
        // FlashAttention-2 with KV cache
        auto attention_output = attention_layers_[layer_idx]->forward_with_cache(
            queries, keys, values, layer_idx, position
        );
        
        // Output projection
        attention_output = linear_forward(attention_output, layer.c_proj_weight, layer.c_proj_bias);
        
        // Residual connection
        for (size_t i = 0; i < attention_output.size(); ++i) {
            attention_output[i] += input[i];
        }
        
        // FFN with residual
        auto ffn_output = forward_ffn(attention_output, layer);
        
        return ffn_output;
    }
    
    // Helper functions (implement as needed)
    std::vector<float> embed_tokens(const std::vector<int>& token_ids);
    void add_positional_embeddings(std::vector<float>& hidden_states, uint32_t start_pos, uint32_t length);
    void apply_layer_norm(std::vector<float>& input, const std::vector<float>& weight, const std::vector<float>& bias);
    std::vector<float> compute_qkv(const std::vector<float>& input, const TransformerLayer& layer);
    void split_qkv(const std::vector<float>& qkv, std::vector<float>& q, std::vector<float>& k, std::vector<float>& v, uint32_t seq_len);
    std::vector<float> linear_forward(const std::vector<float>& input, const std::vector<float>& weight, const std::vector<float>& bias);
    std::vector<float> forward_ffn(const std::vector<float>& input, const TransformerLayer& layer);
    std::vector<float> compute_logits(const std::vector<float>& hidden_states);
    int sample_token(const std::vector<float>& logits, float temperature, uint32_t top_k);
};

// ============================================================================
// Factory Function
// ============================================================================

std::unique_ptr<InferenceEngine<GPT2Weights>> create_flashattention2_kvcache_engine() {
    return std::make_unique<FlashAttention2KVCacheGPT2Engine>();
}

} // namespace gpt2
} // namespace models
} // namespace atsentia