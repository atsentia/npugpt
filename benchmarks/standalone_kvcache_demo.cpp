/**
 * Standalone KV Cache Demonstration
 * Shows the performance benefits of KV caching for autoregressive generation
 */

#include "../include/kv_cache.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <random>

using namespace atsentia::models::gpt2;

// Simulate attention computation without KV cache
double compute_attention_no_cache(int seq_len, int num_heads, int head_dim) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate O(n²) attention computation
    int operations = seq_len * seq_len * num_heads * head_dim;
    double dummy = 0.0;
    for (int i = 0; i < operations / 1000; ++i) {
        dummy += std::sin(i * 0.001);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // ms
}

// Simulate attention computation with KV cache
double compute_attention_with_cache(int current_pos, int num_heads, int head_dim) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Only compute attention for new token - O(n) instead of O(n²)
    int operations = current_pos * num_heads * head_dim;
    double dummy = 0.0;
    for (int i = 0; i < operations / 1000; ++i) {
        dummy += std::sin(i * 0.001);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // ms
}

int main() {
    std::cout << "🚀 KV CACHE PERFORMANCE DEMONSTRATION" << std::endl;
    std::cout << "=====================================" << std::endl;
    std::cout << "Comparing autoregressive generation with and without KV cache\n" << std::endl;
    
    // GPT-2 configuration
    const int num_layers = 12;
    const int num_heads = 12;
    const int head_dim = 64;
    const int max_seq_len = 512;
    
    // Initialize KV cache
    KVCacheConfig config(num_layers, num_heads, head_dim, max_seq_len);
    std::cout << "📊 Model Configuration:" << std::endl;
    config.print_config();
    std::cout << std::endl;
    
    KVCache kv_cache(config);
    
    // Simulate generation of different sequence lengths
    std::vector<int> test_lengths = {32, 64, 128, 256, 512};
    
    std::cout << "🔄 Simulating autoregressive generation...\n" << std::endl;
    std::cout << std::setw(10) << "Seq Len" 
              << std::setw(20) << "Without KV Cache" 
              << std::setw(20) << "With KV Cache"
              << std::setw(15) << "Speedup"
              << std::setw(20) << "Memory Saved" << std::endl;
    std::cout << std::string(85, '-') << std::endl;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int target_len : test_lengths) {
        double time_no_cache = 0.0;
        double time_with_cache = 0.0;
        
        // Clear cache for new sequence
        kv_cache.clear_all();
        
        // Simulate token-by-token generation
        for (int pos = 1; pos <= target_len; ++pos) {
            // Without KV cache - recompute all previous tokens
            time_no_cache += compute_attention_no_cache(pos, num_heads, head_dim);
            
            // With KV cache - only compute for new token
            if (pos == 1) {
                // First token - no cache benefit
                time_with_cache += compute_attention_no_cache(1, num_heads, head_dim);
            } else {
                // Subsequent tokens - use cache
                time_with_cache += compute_attention_with_cache(pos, num_heads, head_dim);
            }
            
            // Simulate cache append (all layers)
            for (int layer = 0; layer < num_layers; ++layer) {
                std::vector<float> dummy_key(num_heads * head_dim);
                std::vector<float> dummy_value(num_heads * head_dim);
                
                // Fill with random data
                for (int i = 0; i < num_heads * head_dim; ++i) {
                    dummy_key[i] = dis(gen);
                    dummy_value[i] = dis(gen);
                }
                
                kv_cache.append_layer_kv(layer, dummy_key.data(), dummy_value.data());
            }
        }
        
        // Calculate metrics
        double speedup = time_no_cache / time_with_cache;
        
        // Memory usage comparison
        size_t memory_no_cache = target_len * target_len * num_heads * head_dim * sizeof(float);
        size_t memory_with_cache = kv_cache.total_memory_usage();
        double memory_saved_pct = 100.0 * (1.0 - (double)memory_with_cache / memory_no_cache);
        
        std::cout << std::setw(10) << target_len
                  << std::setw(20) << std::fixed << std::setprecision(2) << time_no_cache << " ms"
                  << std::setw(20) << std::fixed << std::setprecision(2) << time_with_cache << " ms"
                  << std::setw(15) << std::fixed << std::setprecision(1) << speedup << "x"
                  << std::setw(18) << std::fixed << std::setprecision(1) << memory_saved_pct << "%" << std::endl;
    }
    
    std::cout << std::string(85, '-') << std::endl;
    
    // Show cache statistics
    std::cout << "\n📈 KV Cache Statistics:" << std::endl;
    kv_cache.print_performance_stats();
    
    // Theoretical analysis
    std::cout << "\n📊 Theoretical Analysis:" << std::endl;
    std::cout << "  Without KV Cache: O(n²) time complexity per sequence" << std::endl;
    std::cout << "  With KV Cache: O(n) time complexity per sequence" << std::endl;
    std::cout << "  For 512 tokens: " << (512 * 512) << " vs " << (512 * (512 + 1) / 2) 
              << " attention computations" << std::endl;
    std::cout << "  Theoretical speedup: " << std::fixed << std::setprecision(1) 
              << (512.0 * 512.0) / (512.0 * 513.0 / 2.0) << "x" << std::endl;
    
    // Real-world impact
    std::cout << "\n🌟 Real-World Impact:" << std::endl;
    std::cout << "  • ChatGPT-style applications: Essential for responsive interaction" << std::endl;
    std::cout << "  • Code completion: Enables real-time suggestions" << std::endl;
    std::cout << "  • Mobile deployment: Critical for battery life" << std::endl;
    std::cout << "  • Edge devices: Makes on-device AI feasible" << std::endl;
    
    std::cout << "\n✅ KV Cache demonstration complete!" << std::endl;
    
    return 0;
}