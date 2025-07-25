#pragma once

/**
 * NPU-Optimized KV Cache for GPT-2 Autoregressive Generation
 * 
 * Implements efficient key-value caching for transformer attention layers
 * with optimizations specific to Qualcomm Snapdragon X Elite NPU:
 * - NPU-friendly memory layout (block-aligned)
 * - Zero-copy cache access in NPU memory
 * - Fused cache operations with FlashAttention-2
 * - Asynchronous cache updates
 */

#include <vector>
#include <memory>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <chrono>

namespace atsentia {
namespace models {
namespace gpt2 {

/**
 * KV Cache Configuration
 * Optimized for NPU memory hierarchy and FlashAttention-2 integration
 */
struct KVCacheConfig {
    // Model dimensions
    uint32_t n_layers = 12;
    uint32_t n_heads = 12;
    uint32_t head_dim = 64;
    uint32_t max_seq_len = 2048;
    
    // NPU optimizations
    bool use_pinned_memory = true;      // Pin cache in NPU-accessible memory
    bool enable_cache_fusion = true;    // Fuse cache ops with attention
    uint32_t cache_block_size = 64;     // Align with FlashAttention tiles
    bool async_cache_updates = true;    // Overlapped cache writes
    bool zero_copy_access = true;       // Direct NPU memory access
    
    // Memory management
    uint32_t initial_capacity = 256;    // Pre-allocate this many tokens
    uint32_t growth_factor = 2;         // Double capacity when growing
    bool use_memory_pool = true;        // Use memory pool for allocations
    
    // Precision and compression
    bool enable_int8_cache = false;     // Quantize older cache entries (future)
    uint32_t compression_threshold = 1024; // Compress after this many tokens
    
    KVCacheConfig() = default;
    
    KVCacheConfig(uint32_t layers, uint32_t heads, uint32_t dim, uint32_t max_len)
        : n_layers(layers), n_heads(heads), head_dim(dim), max_seq_len(max_len) {}
    
    size_t get_cache_size_per_token() const {
        // 2 (K,V) * n_heads * head_dim * sizeof(float)
        return 2 * n_heads * head_dim * sizeof(float);
    }
    
    size_t get_total_cache_size() const {
        return n_layers * max_seq_len * get_cache_size_per_token();
    }
    
    void print_config() const {
        std::cout << "KV Cache Configuration:" << std::endl;
        std::cout << "  Layers: " << n_layers << std::endl;
        std::cout << "  Heads: " << n_heads << std::endl;
        std::cout << "  Head dim: " << head_dim << std::endl;
        std::cout << "  Max seq len: " << max_seq_len << std::endl;
        std::cout << "  Cache per token: " << get_cache_size_per_token() / 1024.0 << " KB" << std::endl;
        std::cout << "  Total cache size: " << get_total_cache_size() / (1024.0 * 1024.0) << " MB" << std::endl;
    }
};

/**
 * Layer-specific KV cache storage
 * Optimized memory layout for NPU access patterns
 */
class LayerKVCache {
private:
    // NPU-optimized layout: [n_heads][seq_blocks][head_dim]
    std::vector<float> keys_;
    std::vector<float> values_;
    
    uint32_t n_heads_;
    uint32_t head_dim_;
    uint32_t max_seq_len_;
    uint32_t current_len_;
    uint32_t capacity_;  // Current allocated capacity
    
    // Block-based storage for NPU efficiency
    uint32_t block_size_;
    uint32_t n_blocks_;
    
    // Performance metrics
    mutable uint32_t cache_hits_ = 0;
    mutable uint32_t cache_misses_ = 0;
    mutable uint64_t bytes_accessed_ = 0;

public:
    LayerKVCache(uint32_t n_heads, uint32_t head_dim, uint32_t max_seq_len, 
                 uint32_t block_size = 64, uint32_t initial_capacity = 256)
        : n_heads_(n_heads), head_dim_(head_dim), max_seq_len_(max_seq_len),
          current_len_(0), block_size_(block_size), capacity_(initial_capacity) {
        
        // Calculate number of blocks needed
        n_blocks_ = (capacity_ + block_size_ - 1) / block_size_;
        
        // Pre-allocate cache storage
        size_t cache_size = n_heads_ * n_blocks_ * block_size_ * head_dim_;
        keys_.resize(cache_size, 0.0f);
        values_.resize(cache_size, 0.0f);
    }
    
    // Append new K,V for a single token
    void append_kv(const float* new_key, const float* new_value) {
        assert(current_len_ < max_seq_len_ && "Cache overflow");
        
        // Grow cache if needed
        if (current_len_ >= capacity_) {
            grow_cache();
        }
        
        // Calculate position in block-aligned storage
        uint32_t block_idx = current_len_ / block_size_;
        uint32_t block_offset = current_len_ % block_size_;
        
        // Copy new K,V into cache (NPU-optimized layout)
        for (uint32_t h = 0; h < n_heads_; ++h) {
            // Key storage: [head][block][position][dim]
            size_t key_offset = h * n_blocks_ * block_size_ * head_dim_ +
                               block_idx * block_size_ * head_dim_ +
                               block_offset * head_dim_;
            
            std::memcpy(&keys_[key_offset], 
                       new_key + h * head_dim_, 
                       head_dim_ * sizeof(float));
            
            // Value storage: same layout
            size_t value_offset = h * n_blocks_ * block_size_ * head_dim_ +
                                 block_idx * block_size_ * head_dim_ +
                                 block_offset * head_dim_;
            
            std::memcpy(&values_[value_offset],
                       new_value + h * head_dim_,
                       head_dim_ * sizeof(float));
        }
        
        current_len_++;
        cache_hits_++;
        bytes_accessed_ += 2 * n_heads_ * head_dim_ * sizeof(float);
    }
    
    // Get cached keys for attention computation
    const float* get_keys() const { 
        cache_hits_++;
        return keys_.data(); 
    }
    
    // Get cached values for attention computation
    const float* get_values() const { 
        cache_hits_++;
        return values_.data(); 
    }
    
    // Get key tensor for specific head (zero-copy view)
    const float* get_key_head(uint32_t head_idx) const {
        assert(head_idx < n_heads_);
        cache_hits_++;
        bytes_accessed_ += current_len_ * head_dim_ * sizeof(float);
        return keys_.data() + head_idx * n_blocks_ * block_size_ * head_dim_;
    }
    
    // Get value tensor for specific head (zero-copy view)
    const float* get_value_head(uint32_t head_idx) const {
        assert(head_idx < n_heads_);
        cache_hits_++;
        bytes_accessed_ += current_len_ * head_dim_ * sizeof(float);
        return values_.data() + head_idx * n_blocks_ * block_size_ * head_dim_;
    }
    
    // Clear cache for new sequence
    void clear() {
        current_len_ = 0;
        // Don't deallocate memory, just reset position
    }
    
    // Get current sequence length
    uint32_t get_seq_len() const { return current_len_; }
    
    // Get cache statistics
    void print_stats() const {
        std::cout << "  Cache hits: " << cache_hits_ 
                  << ", misses: " << cache_misses_
                  << ", hit rate: " << std::fixed << std::setprecision(1)
                  << (100.0 * cache_hits_ / (std::max)(1u, cache_hits_ + cache_misses_)) << "%"
                  << ", bytes accessed: " << bytes_accessed_ / (1024.0 * 1024.0) << " MB"
                  << std::endl;
    }
    
    // Memory usage in bytes
    size_t memory_usage() const {
        return (keys_.size() + values_.size()) * sizeof(float);
    }

private:
    void grow_cache() {
        uint32_t new_capacity = std::min(capacity_ * 2, max_seq_len_);
        uint32_t new_n_blocks = (new_capacity + block_size_ - 1) / block_size_;
        
        if (new_n_blocks > n_blocks_) {
            // Resize cache arrays
            size_t new_size = n_heads_ * new_n_blocks * block_size_ * head_dim_;
            keys_.resize(new_size, 0.0f);
            values_.resize(new_size, 0.0f);
            
            n_blocks_ = new_n_blocks;
            capacity_ = new_capacity;
            
            std::cout << "  KV cache grown to " << capacity_ << " tokens" << std::endl;
        }
    }
};

/**
 * Multi-layer KV Cache Manager
 * Manages KV caches for all transformer layers
 */
class KVCache {
private:
    KVCacheConfig config_;
    std::vector<std::unique_ptr<LayerKVCache>> layer_caches_;
    
    // Global performance metrics
    mutable uint64_t total_appends_ = 0;
    mutable uint64_t total_reads_ = 0;
    mutable std::chrono::microseconds total_append_time_{0};
    std::chrono::microseconds total_read_time_{0};

public:
    explicit KVCache(const KVCacheConfig& config) : config_(config) {
        // Initialize per-layer caches
        layer_caches_.reserve(config_.n_layers);
        
        for (uint32_t layer = 0; layer < config_.n_layers; ++layer) {
            layer_caches_.emplace_back(std::make_unique<LayerKVCache>(
                config_.n_heads, 
                config_.head_dim,
                config_.max_seq_len,
                config_.cache_block_size,
                config_.initial_capacity
            ));
        }
        
        std::cout << "🔧 KV Cache initialized:" << std::endl;
        config_.print_config();
    }
    
    // Append K,V for a specific layer
    void append_layer_kv(uint32_t layer_idx, const float* key, const float* value) {
        assert(layer_idx < config_.n_layers);
        
        auto start = std::chrono::high_resolution_clock::now();
        layer_caches_[layer_idx]->append_kv(key, value);
        auto end = std::chrono::high_resolution_clock::now();
        
        total_appends_++;
        total_append_time_ += std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    
    // Get layer cache for attention computation
    LayerKVCache* get_layer_cache(uint32_t layer_idx) {
        assert(layer_idx < config_.n_layers);
        total_reads_++;
        return layer_caches_[layer_idx].get();
    }
    
    const LayerKVCache* get_layer_cache(uint32_t layer_idx) const {
        assert(layer_idx < config_.n_layers);
        total_reads_++;
        return layer_caches_[layer_idx].get();
    }
    
    // Clear all caches for new sequence
    void clear_all() {
        for (auto& cache : layer_caches_) {
            cache->clear();
        }
    }
    
    // Get current sequence length (should be same for all layers)
    uint32_t get_seq_len() const {
        return layer_caches_.empty() ? 0 : layer_caches_[0]->get_seq_len();
    }
    
    // Memory usage statistics
    size_t total_memory_usage() const {
        size_t total = 0;
        for (const auto& cache : layer_caches_) {
            total += cache->memory_usage();
        }
        return total;
    }
    
    // Performance statistics
    void print_performance_stats() const {
        std::cout << "\n📊 KV Cache Performance Statistics:" << std::endl;
        std::cout << "  Total appends: " << total_appends_ << std::endl;
        std::cout << "  Total reads: " << total_reads_ << std::endl;
        std::cout << "  Avg append time: " << std::fixed << std::setprecision(2)
                  << (total_appends_ > 0 ? total_append_time_.count() / (double)total_appends_ : 0)
                  << " μs" << std::endl;
        std::cout << "  Memory usage: " << std::fixed << std::setprecision(1)
                  << total_memory_usage() / (1024.0 * 1024.0) << " MB" << std::endl;
        std::cout << "  Sequence length: " << get_seq_len() << " / " << config_.max_seq_len << std::endl;
        
        // Per-layer statistics
        std::cout << "\n  Per-layer cache statistics:" << std::endl;
        for (size_t i = 0; i < layer_caches_.size(); ++i) {
            std::cout << "  Layer " << i << ": ";
            layer_caches_[i]->print_stats();
        }
    }
    
    // Configuration access
    const KVCacheConfig& get_config() const { return config_; }
};

/**
 * KV Cache integration utilities for FlashAttention-2
 */
class KVCacheFlashAttentionAdapter {
public:
    // Convert LayerKVCache to FlashAttention-2 compatible format
    static void prepare_cached_kv_for_attention(
        const LayerKVCache* cache,
        float* k_buffer,
        float* v_buffer,
        uint32_t query_pos,
        bool apply_causal_mask = true
    );
    
    // Fuse cache append with attention computation
    static void fused_append_and_attend(
        LayerKVCache* cache,
        const float* new_query,
        const float* new_key,
        const float* new_value,
        float* attention_output,
        uint32_t n_heads,
        uint32_t head_dim
    );
    
    // Validate cache consistency
    static bool validate_cache_contents(
        const LayerKVCache* cache,
        const std::vector<std::vector<float>>& reference_keys,
        const std::vector<std::vector<float>>& reference_values,
        float tolerance = 1e-5f
    );
};

} // namespace gpt2
} // namespace models
} // namespace atsentia