#pragma once

/**
 * Atsentia AI Accelerator - NPU GPT-2 Engine Implementation
 * 
 * Real NPU hardware execution with comprehensive timing and validation
 * Two variants: Non-fused (individual operations) and Fused (optimized graphs)
 */

// Skip problematic core framework includes for standalone build
// #include "../include/gpt2_engine.h"
#include "../include/gpt2_types.h"
// Forward declaration for BPE tokenizer interface
class BPETokenizer {
public:
    virtual ~BPETokenizer() = default;
    virtual std::vector<int32_t> encode(const std::string& text) = 0;
    virtual std::string decode(const std::vector<int32_t>& tokens) = 0;
    virtual int32_t vocab_size() const = 0;
    
    // Additional methods for loading
    virtual bool load_from_files(const std::string& encoder_path, const std::string& vocab_path) = 0;
    virtual bool initialize_simple_mode() = 0;
};
#include "../include/standalone_qnn_static_graph.h"
#include "npu_callback_profiling.h"
#include <memory>
#include <chrono>
#include <string>
#include <vector>
#include <map>

namespace atsentia {
namespace models {
namespace gpt2 {

/**
 * GPT-2 Configuration for standalone NPU implementation
 */
struct GPT2Config {
    // Model architecture
    uint32_t vocab_size = 50257;
    uint32_t n_positions = 1024;
    uint32_t n_ctx = 1024;
    uint32_t n_embd = 768;
    uint32_t n_head = 12;
    uint32_t n_layer = 12;
    
    // Model variant ("124M", "355M", "774M", "1558M")
    std::string model_size = "124M";
    
    // Paths
    std::string weights_path;
    std::string tokenizer_path;
    
    // Performance settings
    uint32_t max_sequence_length = 512;
    bool use_kv_cache = true;
    bool enable_flashattention2 = true;
    
    // Validation
    bool is_valid() const;
    void apply_defaults_for_size(const std::string& size);
};

/**
 * GPT-2 Model Weights Structure (standalone)
 */
struct GPT2Weights {
    // Embedding layers
    std::vector<float> token_embeddings;    // [vocab_size, n_embd]
    std::vector<float> position_embeddings; // [n_positions, n_embd]
    
    // Transformer layers (12 layers for 124M)
    struct TransformerLayer {
        // Attention
        std::vector<float> attention_norm_weight;  // [n_embd]
        std::vector<float> attention_norm_bias;    // [n_embd]
        std::vector<float> attention_qkv_weight;   // [n_embd, 3*n_embd]
        std::vector<float> attention_qkv_bias;     // [3*n_embd]
        std::vector<float> attention_proj_weight;  // [n_embd, n_embd]
        std::vector<float> attention_proj_bias;    // [n_embd]
        
        // Feed-forward
        std::vector<float> ffn_norm_weight;        // [n_embd]
        std::vector<float> ffn_norm_bias;          // [n_embd]
        std::vector<float> ffn_fc_weight;          // [n_embd, 4*n_embd]
        std::vector<float> ffn_fc_bias;            // [4*n_embd]
        std::vector<float> ffn_proj_weight;        // [4*n_embd, n_embd]
        std::vector<float> ffn_proj_bias;          // [n_embd]
    };
    
    std::vector<TransformerLayer> layers;
    
    // Final layer norm
    std::vector<float> final_norm_weight;      // [n_embd]
    std::vector<float> final_norm_bias;        // [n_embd]
    
    // Language modeling head (often shared with token embeddings)
    std::vector<float> lm_head_weight;         // [n_embd, vocab_size]
    
    // Metadata
    std::string model_size;
    size_t total_parameters = 0;
    
    void calculate_total_parameters();
    bool is_valid() const;
};

/**
 * GPT-2 Loader for weights and tokenizer (standalone)
 */
class GPT2Loader {
public:
    static std::unique_ptr<GPT2Weights> load_from_path(const std::string& weights_path);
    static std::unique_ptr<BPETokenizer> load_tokenizer(const std::string& model_size);
    
private:
    // Helper methods for weight loading
    static std::unique_ptr<GPT2Weights> create_placeholder_weights();
    
    // Helper methods for tokenizer loading
    static std::unique_ptr<BPETokenizer> create_simple_tokenizer();
};

/**
 * NPU-specific timing and validation logging
 */
class NPUPerformanceLogger {
public:
    struct OperationTiming {
        std::string operation_name;
        std::chrono::microseconds preparation_time{0};
        std::chrono::microseconds npu_execution_time{0};
        std::chrono::microseconds data_transfer_time{0};
        std::chrono::microseconds total_time{0};
        size_t input_bytes = 0;
        size_t output_bytes = 0;
        bool npu_execution_confirmed = false;
        std::string validation_status;
    };
    
    void start_operation(const std::string& op_name);
    void mark_preparation_complete(const std::string& op_name);
    void mark_npu_execution_complete(const std::string& op_name);
    void mark_data_transfer_complete(const std::string& op_name);
    void end_operation(const std::string& op_name, bool npu_confirmed, const std::string& validation);
    
    void set_data_sizes(const std::string& op_name, size_t input_bytes, size_t output_bytes);
    
    const std::vector<OperationTiming>& get_timings() const { return timings_; }
    OperationTiming get_total_timing() const;
    
    void print_performance_report() const;
    void save_performance_log(const std::string& filename) const;
    
private:
    std::vector<OperationTiming> timings_;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> start_times_;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> preparation_times_;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> execution_times_;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> transfer_times_;
};

/**
 * NPU GPT-2 Engine - Non-Fused Implementation
 * 
 * Uses individual NPU operations for each transformer component.
 * Provides detailed timing and validation for each operation.
 */
class NPUGpt2Engine {
public:
    NPUGpt2Engine();
    ~NPUGpt2Engine();
    
    // Initialization with NPU validation
    bool initialize(const GPT2Config& config);
    bool load_weights(const std::string& weights_path);
    
    // Core inference with timing
    std::string generate(const std::string& prompt, 
                        uint32_t max_tokens = 50,
                        float temperature = 1.0f);
    
    // NPU validation and monitoring
    bool validate_npu_execution() const;
    const NPUPerformanceLogger& get_performance_logger() const { return performance_logger_; }
    const NPUCallbackProfiler& get_callback_profiler() const { return callback_profiler_; }
    
    // Timing configuration
    void enable_callback_timing(bool enable) { use_callback_timing_ = enable; }
    void print_comprehensive_timing_report() const;
    
    // Individual transformer components (for testing and validation)
    std::vector<float> npu_embedding_lookup(const std::vector<uint32_t>& token_ids);
    std::vector<float> npu_layer_norm(const std::vector<float>& input, 
                                     const std::vector<float>& weight,
                                     const std::vector<float>& bias);
    std::vector<float> npu_multihead_attention(const std::vector<float>& input,
                                              const std::vector<float>& qkv_weight,
                                              const std::vector<float>& qkv_bias,
                                              const std::vector<float>& proj_weight,
                                              const std::vector<float>& proj_bias,
                                              uint32_t num_heads);
    std::vector<float> npu_feed_forward(const std::vector<float>& input,
                                       const std::vector<float>& fc_weight,
                                       const std::vector<float>& fc_bias,
                                       const std::vector<float>& proj_weight,
                                       const std::vector<float>& proj_bias);
    
    // Configuration
    void enable_detailed_logging(bool enable) { detailed_logging_ = enable; }
    void set_validation_tolerance(float tolerance) { validation_tolerance_ = tolerance; }
    
private:
    // NPU infrastructure
    std::unique_ptr<atsentia::qualcomm_npu::QNNStaticGraph> npu_context_;
    GPT2Config config_;
    std::unique_ptr<GPT2Weights> weights_;
    std::unique_ptr<BPETokenizer> tokenizer_;
    
    // Performance monitoring
    mutable NPUPerformanceLogger performance_logger_;
    mutable NPUCallbackProfiler callback_profiler_;  // Accurate NPU timing
    bool detailed_logging_ = true;
    float validation_tolerance_ = 1e-4f;
    bool use_callback_timing_ = true;
    
    // NPU operation helpers
    bool initialize_npu_context();
    bool validate_npu_operation(const std::string& op_name, 
                               const std::vector<float>& npu_result,
                               const std::vector<float>& cpu_reference) const;
    
    // Transformer layer implementation
    std::vector<float> forward_transformer_layer(const std::vector<float>& input, 
                                                uint32_t layer_idx);
    
    // Utility functions
    std::vector<float> cpu_reference_layer_norm(const std::vector<float>& input,
                                               const std::vector<float>& weight,
                                               const std::vector<float>& bias) const;
    std::vector<float> cpu_reference_multihead_attention(const std::vector<float>& input,
                                                        const std::vector<float>& qkv_weight,
                                                        const std::vector<float>& qkv_bias,
                                                        const std::vector<float>& proj_weight,
                                                        const std::vector<float>& proj_bias,
                                                        uint32_t num_heads) const;
    std::vector<float> cpu_reference_feed_forward(const std::vector<float>& input,
                                                 const std::vector<float>& fc_weight,
                                                 const std::vector<float>& fc_bias,
                                                 const std::vector<float>& proj_weight,
                                                 const std::vector<float>& proj_bias) const;
    
    void log_operation_start(const std::string& op_name) const;
    void log_operation_end(const std::string& op_name, bool success) const;
};

/**
 * NPU GPT-2 Engine - Fused Implementation
 * 
 * Uses fused NPU operations for optimal performance.
 * Combines multiple operations into single NPU graphs.
 */
class FusedNPUGpt2Engine {
public:
    FusedNPUGpt2Engine();
    ~FusedNPUGpt2Engine();
    
    // Initialization
    bool initialize(const GPT2Config& config);
    bool load_weights(const std::string& weights_path);
    
    // Core inference with optimized NPU graphs
    std::string generate(const std::string& prompt, 
                        uint32_t max_tokens = 50,
                        float temperature = 1.0f);
    
    // Fused operations for maximum NPU efficiency
    std::vector<float> npu_fused_attention_block(const std::vector<float>& input,
                                                uint32_t layer_idx);
    std::vector<float> npu_fused_ffn_block(const std::vector<float>& input,
                                          uint32_t layer_idx);
    std::vector<float> npu_fused_transformer_layer(const std::vector<float>& input,
                                                  uint32_t layer_idx);
    
    // Performance comparison
    const NPUPerformanceLogger& get_performance_logger() const { return performance_logger_; }
    void compare_with_non_fused(const NPUGpt2Engine& non_fused_engine) const;
    
private:
    // Fused NPU graphs
    std::unique_ptr<atsentia::qualcomm_npu::QNNStaticGraph> attention_fused_graph_;
    std::unique_ptr<atsentia::qualcomm_npu::QNNStaticGraph> ffn_fused_graph_;
    std::unique_ptr<atsentia::qualcomm_npu::QNNStaticGraph> layer_fused_graph_;
    
    GPT2Config config_;
    std::unique_ptr<GPT2Weights> weights_;
    std::unique_ptr<BPETokenizer> tokenizer_;
    mutable NPUPerformanceLogger performance_logger_;
    
    // Fused graph creation
    bool create_fused_attention_graph();
    bool create_fused_ffn_graph();
    bool create_fused_layer_graph();
    
    // NPU validation for fused operations
    bool validate_fused_operation(const std::string& op_name,
                                 const std::vector<float>& fused_result,
                                 const std::vector<float>& reference_result) const;
};

/**
 * NPU GPT-2 Benchmark and Comparison Utility
 */
class NPUGpt2Benchmark {
public:
    struct ComparisonResult {
        std::string test_name;
        double non_fused_time_ms = 0.0;
        double fused_time_ms = 0.0;
        double speedup_ratio = 0.0;
        uint32_t npu_operations_count = 0;
        bool all_npu_validated = false;
        std::string performance_summary;
    };
    
    // Benchmark individual engines
    static ComparisonResult benchmark_non_fused_engine(NPUGpt2Engine& engine,
                                                      const std::string& test_prompt);
    static ComparisonResult benchmark_fused_engine(FusedNPUGpt2Engine& engine,
                                                  const std::string& test_prompt);
    
    // Head-to-head comparison
    static std::vector<ComparisonResult> compare_engines(NPUGpt2Engine& non_fused,
                                                        FusedNPUGpt2Engine& fused,
                                                        const std::vector<std::string>& test_prompts);
    
    // Comprehensive NPU validation
    static bool validate_100_percent_npu_execution(const NPUGpt2Engine& engine);
    static bool validate_100_percent_npu_execution(const FusedNPUGpt2Engine& engine);
    
    // Reporting
    static void print_comparison_report(const std::vector<ComparisonResult>& results);
    static void save_benchmark_results(const std::vector<ComparisonResult>& results,
                                      const std::string& filename);
};

} // namespace gpt2
} // namespace models
} // namespace atsentia