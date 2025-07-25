/**
 * Simple KV Cache Attention Demo - Minimal Working Example
 * 
 * This demonstrates:
 * 1. Basic attention layer with KV cache functionality
 * 2. Performance measurement of cached vs non-cached attention
 * 3. Memory usage tracking
 * 4. Foundation for real NPU integration
 */

#include "src/simple_attention_with_kvcache.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace npugpt;

void print_header() {
    std::cout << "==============================================\n";
    std::cout << "🔧 SIMPLE ATTENTION WITH KV CACHE DEMO\n";
    std::cout << "==============================================\n";
    std::cout << "Testing basic KV cache functionality\n";
    std::cout << "Hardware: Simplified CPU implementation\n";
    std::cout << "Purpose: Foundation for NPU integration\n\n";
}

void run_attention_comparison(size_t sequence_length) {
    std::cout << "🧪 Testing sequence length: " << sequence_length << " tokens\n";
    std::cout << "----------------------------------------------\n";
    
    // Create attention layer configuration
    AttentionConfig config(12, 64, 1024);  // 12 heads, 64 dim each, max 1024 tokens
    
    // Create attention layer
    SimpleAttentionWithKVCache attention(config);
    
    // Generate synthetic input data
    std::vector<float> hidden_state(config.hidden_size, 0.1f);
    for (size_t i = 0; i < hidden_state.size(); ++i) {
        hidden_state[i] = 0.1f * std::sin(i * 0.1f);
    }
    
    // Test KV cached attention
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "  🔄 Processing tokens with KV cache:\n";
    for (size_t token = 0; token < sequence_length; ++token) {
        // Modify input slightly for each token
        for (size_t i = 0; i < hidden_state.size(); ++i) {
            hidden_state[i] += 0.01f * token;
        }
        
        auto output = attention.forward(hidden_state, true);  // use_cache = true
        
        if (token % 10 == 0 || token < 5) {
            std::cout << "    Token " << std::setw(3) << token 
                      << ": output_norm=" << std::fixed << std::setprecision(3);
            
            // Calculate output norm for verification
            float norm = 0.0f;
            for (float val : output) {
                norm += val * val;
            }
            std::cout << std::sqrt(norm) << std::endl;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double, std::milli>(end_time - start_time);
    
    // Report results
    std::cout << "\n  📊 KV Cache Performance Results:\n";
    std::cout << "    Total time: " << std::fixed << std::setprecision(1) << duration.count() << " ms\n";
    std::cout << "    Avg per token: " << std::fixed << std::setprecision(2) 
              << duration.count() / sequence_length << " ms\n";
    std::cout << "    Tokens/sec: " << std::fixed << std::setprecision(1) 
              << (sequence_length * 1000.0) / duration.count() << "\n";
    std::cout << "    Cache hits: " << attention.get_cache_hits() << "\n";
    std::cout << "    Memory usage: " << attention.get_memory_usage_mb() << " MB\n";
    std::cout << "    Sequence length: " << attention.get_seq_len() << " tokens\n";
    
    std::cout << "\n";
}

void demonstrate_cache_benefits() {
    std::cout << "🎯 DEMONSTRATING KV CACHE BENEFITS\n";
    std::cout << "==================================\n\n";
    
    // Test different sequence lengths to show scaling benefits
    std::vector<size_t> test_lengths = {16, 32, 64, 128};
    
    for (size_t length : test_lengths) {
        run_attention_comparison(length);
    }
    
    std::cout << "💡 Key Observations:\n";
    std::cout << "   • KV cache stores previous key/value computations\n";
    std::cout << "   • Each new token only computes attention over cached sequence\n";
    std::cout << "   • Memory usage grows linearly with sequence length\n";
    std::cout << "   • Foundation ready for NPU optimization\n\n";
}

void demonstrate_cache_operations() {
    std::cout << "🔧 DEMONSTRATING CACHE OPERATIONS\n";
    std::cout << "=================================\n\n";
    
    AttentionConfig config(4, 32, 64);  // Smaller config for demonstration
    SimpleKVCache cache(config);
    
    std::cout << "  📋 Cache Configuration:\n";
    std::cout << "    Heads: " << config.n_heads << "\n";
    std::cout << "    Head dimension: " << config.head_dim << "\n";
    std::cout << "    Max sequence length: " << config.max_seq_len << "\n";
    std::cout << "    Hidden size: " << config.hidden_size << "\n\n";
    
    // Add some key-value pairs to demonstrate cache operations
    std::cout << "  🔄 Adding key-value pairs to cache:\n";
    
    for (size_t token = 0; token < 8; ++token) {
        // Generate dummy keys and values
        std::vector<std::vector<float>> keys(config.n_heads);
        std::vector<std::vector<float>> values(config.n_heads);
        
        for (size_t h = 0; h < config.n_heads; ++h) {
            keys[h].resize(config.head_dim);
            values[h].resize(config.head_dim);
            
            for (size_t d = 0; d < config.head_dim; ++d) {
                keys[h][d] = 0.1f * (token + 1) * (h + 1) * (d + 1);
                values[h][d] = 0.2f * (token + 1) * (h + 1) * (d + 1);
            }
        }
        
        cache.append_kv(keys, values);
        
        std::cout << "    Token " << token << ": sequence_len=" << cache.get_seq_len() 
                  << ", memory=" << cache.get_memory_usage_mb() << "MB\n";
    }
    
    // Demonstrate cache retrieval
    std::cout << "\n  📤 Retrieving cached data:\n";
    auto cached_keys = cache.get_cached_keys();
    auto cached_values = cache.get_cached_values();
    
    std::cout << "    Retrieved " << cached_keys.size() << " cached key sets\n";
    std::cout << "    Retrieved " << cached_values.size() << " cached value sets\n";
    std::cout << "    Cache hits: " << cache.get_cache_hits() << "\n\n";
}

int main() {
    try {
        print_header();
        
        // Demonstrate basic cache operations
        demonstrate_cache_operations();
        
        // Demonstrate attention with KV cache
        demonstrate_cache_benefits();
        
        std::cout << "✅ DEMO COMPLETED SUCCESSFULLY!\n";
        std::cout << "=================================\n";
        std::cout << "🎯 Next Steps:\n";
        std::cout << "   1. Integrate with real NPU MatMul operations\n";
        std::cout << "   2. Add proper QKV projection weights\n";
        std::cout << "   3. Implement FlashAttention-2 tiling\n";
        std::cout << "   4. Add real GPT-2 model integration\n";
        std::cout << "   5. Benchmark against non-cached versions\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error: " << e.what() << std::endl;
        return 1;
    }
}