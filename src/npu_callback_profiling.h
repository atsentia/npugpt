#pragma once

/**
 * Atsentia AI Accelerator - NPU Callback-Based Timing
 * 
 * Uses QNN SDK graph execution callbacks for accurate NPU timing.
 * This provides hardware-level timing directly from the NPU driver.
 */

#include "QNN/QnnInterface.h"
#include "QNN/QnnTypes.h"
#include "QNN/QnnGraph.h"
#include "QNN/QnnProfile.h"
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <functional>

namespace atsentia {
namespace models {
namespace gpt2 {

/**
 * NPU execution event types from QNN callbacks
 */
enum class NPUExecutionEvent {
    GRAPH_PREPARE_START,
    GRAPH_PREPARE_END,
    GRAPH_EXECUTE_START,
    GRAPH_EXECUTE_END,
    NODE_EXECUTE_START,
    NODE_EXECUTE_END,
    DATA_TRANSFER_START,
    DATA_TRANSFER_END
};

/**
 * NPU timing data from callbacks
 */
struct NPUCallbackTiming {
    std::string operation_name;
    uint64_t npu_timestamp_start = 0;      // NPU hardware timestamp
    uint64_t npu_timestamp_end = 0;        // NPU hardware timestamp
    uint64_t npu_execution_cycles = 0;     // Actual NPU cycles used
    double npu_frequency_mhz = 0.0;        // NPU frequency during execution
    
    // Derived accurate timing
    double actual_execution_time_us = 0.0;
    double actual_gflops = 0.0;
    double actual_memory_bw_gbps = 0.0;
    double actual_npu_utilization = 0.0;
    
    // Validation
    bool callback_timing_valid = false;
    bool hardware_timestamp_valid = false;
    std::string timing_source = "unknown";
};

/**
 * QNN Callback-based profiler for accurate NPU timing
 */
class NPUCallbackProfiler {
public:
    NPUCallbackProfiler();
    ~NPUCallbackProfiler();
    
    // Initialize callback profiling with QNN interface
    bool initialize(QnnInterface_t& qnn_interface, Qnn_ContextHandle_t context);
    
    // Register callbacks for graph execution
    bool register_execution_callbacks(Qnn_GraphHandle_t graph_handle);
    
    // Start timing for an operation
    void start_operation_timing(const std::string& operation_name);
    
    // End timing for an operation
    void end_operation_timing(const std::string& operation_name, 
                             uint64_t operations_count = 0,
                             size_t data_bytes = 0);
    
    // Get accurate timing results
    const std::vector<NPUCallbackTiming>& get_callback_timings() const { return timings_; }
    NPUCallbackTiming get_operation_timing(const std::string& operation_name) const;
    
    // Performance reporting with accurate data
    void print_callback_performance_report() const;
    void save_callback_timing_data(const std::string& filename) const;
    
    // Validation
    bool validate_callback_accuracy() const;
    double get_timing_accuracy_confidence() const;
    
private:
    // QNN handles
    QnnInterface_t* qnn_interface_ = nullptr;
    Qnn_ContextHandle_t context_handle_ = nullptr;
    Qnn_GraphHandle_t graph_handle_ = nullptr;
    
    // Callback data
    std::vector<NPUCallbackTiming> timings_;
    std::map<std::string, size_t> active_operations_;
    
    // Callback functions (static members for C compatibility)
    static void graph_execution_callback(Qnn_GraphHandle_t graphHandle,
                                       void* userData,
                                       Qnn_ErrorHandle_t error,
                                       void* profileData);
    
    static void node_execution_callback(Qnn_GraphHandle_t graphHandle,
                                      const char* nodeName,
                                      void* userData,
                                      uint64_t startTime,
                                      uint64_t endTime,
                                      uint64_t cycles);
    
    // Internal callback handlers
    void handle_graph_execution_callback(Qnn_GraphHandle_t graphHandle,
                                       Qnn_ErrorHandle_t error,
                                       void* profileData);
    
    void handle_node_execution_callback(const char* nodeName,
                                      uint64_t startTime,
                                      uint64_t endTime,
                                      uint64_t cycles);
    
    // Timing calculation from NPU hardware timestamps
    double calculate_execution_time_us(uint64_t start_timestamp, uint64_t end_timestamp) const;
    double calculate_gflops_from_cycles(uint64_t cycles, uint64_t operations) const;
    double calculate_memory_bandwidth(size_t bytes, double time_us) const;
    
    // NPU hardware info
    double get_npu_timestamp_frequency() const;  // NPU timestamp frequency
    bool extract_npu_frequency(void* profile_data, double& frequency_mhz) const;
    
    // Instance pointer for static callbacks
    static NPUCallbackProfiler* instance_;
};

/**
 * Enhanced NPU graph with callback timing
 */
class TimedNPUGraph {
public:
    TimedNPUGraph(const std::string& graph_name);
    ~TimedNPUGraph();
    
    // Initialize with callback profiling
    bool initialize_with_profiling(QnnInterface_t& qnn_interface, Qnn_ContextHandle_t context);
    
    // Graph operations with automatic timing
    bool add_timed_operation(const std::string& op_name, 
                           std::function<bool()> operation_func,
                           uint64_t expected_operations = 0,
                           size_t expected_data_bytes = 0);
    
    // Execute graph with callback timing
    bool execute_with_timing(const std::string& execution_name);
    
    // Get timing results
    const NPUCallbackProfiler& get_profiler() const { return profiler_; }
    
    // Timing utilities
    void print_execution_summary() const;
    bool validate_execution_timing() const;
    
private:
    std::string graph_name_;
    NPUCallbackProfiler profiler_;
    QnnInterface_t* qnn_interface_ = nullptr;
    Qnn_ContextHandle_t context_handle_ = nullptr;
    Qnn_GraphHandle_t graph_handle_ = nullptr;
    
    bool profiling_initialized_ = false;
};

/**
 * RAII wrapper for timed NPU operations
 */
class TimedNPUOperation {
public:
    TimedNPUOperation(NPUCallbackProfiler& profiler, 
                     const std::string& operation_name,
                     uint64_t expected_operations = 0,
                     size_t expected_data_bytes = 0);
    ~TimedNPUOperation();
    
    // Update operation metrics during execution
    void update_operation_count(uint64_t operations);
    void update_data_size(size_t bytes);
    void mark_success(bool success = true);
    
private:
    NPUCallbackProfiler& profiler_;
    std::string operation_name_;
    uint64_t operation_count_;
    size_t data_bytes_;
    bool success_ = false;
};

/**
 * Timing validation and analysis utilities
 */
class NPUTimingValidator {
public:
    struct TimingAnalysis {
        bool timing_realistic = false;
        bool hardware_timestamps_valid = false;
        bool performance_within_limits = false;
        double confidence_score = 0.0;
        std::string analysis_summary;
        std::vector<std::string> recommendations;
    };
    
    // Validate callback timing accuracy
    static TimingAnalysis analyze_callback_timing(const std::vector<NPUCallbackTiming>& timings);
    
    // Compare callback timing vs system timing
    static void compare_timing_methods(const NPUCallbackTiming& callback_timing,
                                     std::chrono::microseconds system_timing);
    
    // Detect timing anomalies
    static std::vector<std::string> detect_timing_issues(const NPUCallbackTiming& timing);
    
    // Performance bounds checking
    static bool is_performance_realistic(double gflops, double utilization, double memory_bw);
    
private:
    // Qualcomm X Elite NPU limits for validation
    static constexpr double MAX_REALISTIC_GFLOPS = 15.0;    // Conservative estimate
    static constexpr double MAX_REALISTIC_UTILIZATION = 85.0; // 85% max realistic
    static constexpr double MAX_MEMORY_BW = 68.0;           // Peak bandwidth
    static constexpr double MIN_EXECUTION_TIME_US = 50.0;   // Minimum realistic time
};

} // namespace gpt2
} // namespace models
} // namespace atsentia