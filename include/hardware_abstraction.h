#pragma once

/**
 * Atsentia AI Accelerator - Hardware Abstraction Layer
 * 
 * Abstract interface for different hardware acceleration platforms.
 * Current implementations: Qualcomm NPU
 * Future: NVIDIA GPU, Intel NPU, Apple Silicon, etc.
 */

#include <string>
#include <vector>
#include <memory>
#include <map>

namespace atsentia {

// Forward declarations
struct AccelerationConfig;
struct PerformanceMetrics;

/**
 * Abstract base class for hardware accelerators
 * 
 * Each hardware platform (Qualcomm NPU, NVIDIA GPU, etc.) implements
 * this interface to provide hardware-specific acceleration.
 */
class HardwareAccelerator {
public:
    virtual ~HardwareAccelerator() = default;

    // Hardware identification
    virtual std::string get_name() const = 0;
    virtual std::string get_version() const = 0;
    virtual std::string get_device_info() const = 0;
    
    // Capabilities
    virtual bool supports_model(const std::string& model_name) const = 0;
    virtual bool supports_sequence_length(uint32_t length) const = 0;
    virtual bool supports_flashattention2() const = 0;
    virtual bool supports_quantization() const = 0;
    
    // Initialization and configuration
    virtual bool initialize(const AccelerationConfig& config) = 0;
    virtual void configure(const AccelerationConfig& config) = 0;
    virtual void shutdown() = 0;
    
    // Model loading and management
    virtual bool load_model(const std::string& model_path,
                           const std::string& model_name) = 0;
    virtual void unload_model(const std::string& model_name) = 0;
    virtual std::vector<std::string> get_loaded_models() const = 0;
    
    // Core inference operations
    virtual std::vector<float> forward_pass(const std::vector<float>& input_tokens,
                                           const std::string& model_name,
                                           bool is_warmup_phase = false) = 0;
    
    virtual std::vector<float> generate_next_token(const std::vector<float>& context,
                                                  const std::string& model_name) = 0;
    
    // Memory and resource management
    virtual size_t get_memory_usage() const = 0;
    virtual size_t get_available_memory() const = 0;
    virtual void free_unused_memory() = 0;
    
    // Performance monitoring
    virtual PerformanceMetrics get_hardware_metrics() const = 0;
    virtual void reset_metrics() = 0;
    
    // Hardware-specific optimizations
    virtual void optimize_for_latency() = 0;
    virtual void optimize_for_throughput() = 0;
    virtual void enable_profiling(bool enable) = 0;

protected:
    // Helper methods for derived classes
    virtual void log_debug(const std::string& message) const;
    virtual void log_error(const std::string& message) const;
};

/**
 * Hardware factory for creating platform-specific accelerators
 */
class HardwareFactory {
public:
    // Hardware types
    enum class HardwareType {
        AUTO_DETECT,
        QUALCOMM_NPU,
        NVIDIA_GPU,
        INTEL_NPU,
        APPLE_SILICON,
        CPU_FALLBACK
    };
    
    // Factory methods
    static std::unique_ptr<HardwareAccelerator> create_accelerator(HardwareType type);
    static std::unique_ptr<HardwareAccelerator> create_auto_detected();
    static std::unique_ptr<HardwareAccelerator> create_from_name(const std::string& name);
    
    // Detection and enumeration
    static std::vector<HardwareType> detect_available_hardware();
    static std::string hardware_type_to_string(HardwareType type);
    static HardwareType string_to_hardware_type(const std::string& name);
    
    // Capability queries
    static bool is_hardware_available(HardwareType type);
    static std::vector<std::string> get_supported_models(HardwareType type);

private:
    static void register_hardware_implementations();
};

/**
 * Hardware-specific tensor operations interface
 * 
 * Provides low-level tensor operations that can be optimized
 * for specific hardware platforms.
 */
class TensorOperations {
public:
    virtual ~TensorOperations() = default;
    
    // Basic operations
    virtual std::vector<float> matrix_multiply(const std::vector<float>& a,
                                             const std::vector<float>& b,
                                             uint32_t rows_a, uint32_t cols_a,
                                             uint32_t cols_b) = 0;
    
    virtual std::vector<float> add_vectors(const std::vector<float>& a,
                                         const std::vector<float>& b) = 0;
    
    virtual std::vector<float> layer_norm(const std::vector<float>& input,
                                        const std::vector<float>& gamma,
                                        const std::vector<float>& beta,
                                        float epsilon = 1e-5f) = 0;
    
    virtual std::vector<float> softmax(const std::vector<float>& input,
                                     uint32_t dim_size) = 0;
    
    virtual std::vector<float> gelu_activation(const std::vector<float>& input) = 0;
    
    // Advanced operations
    virtual std::vector<float> attention_operation(const std::vector<float>& queries,
                                                  const std::vector<float>& keys,
                                                  const std::vector<float>& values,
                                                  bool use_flashattention2 = true) = 0;
    
    virtual std::vector<float> fused_linear_layer(const std::vector<float>& input,
                                                 const std::vector<float>& weights,
                                                 const std::vector<float>& bias) = 0;
    
    // Memory management
    virtual void* allocate_tensor_memory(size_t size_bytes) = 0;
    virtual void free_tensor_memory(void* ptr) = 0;
    virtual void optimize_memory_layout() = 0;
};

} // namespace atsentia