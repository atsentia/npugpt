#pragma once

/**
 * Atsentia AI Accelerator - QNN Context Implementation
 * Basic NPU context and tensor management for Qualcomm QNN SDK
 */

#include "qualcomm_npu_accelerator.h"
#include <vector>
#include <string>
#include <memory>
#include <map>
#include <cstdint>

namespace atsentia {
namespace qualcomm_npu {

/**
 * QNN Tensor wrapper - basic NPU tensor operations
 */
class QNNTensor {
public:
    uint32_t id = 0;
    std::vector<uint32_t> shape;
    std::vector<float> data;
    bool is_graph_input = false;
    bool is_graph_output = false;

    QNNTensor() = default;
    QNNTensor(std::vector<uint32_t> s);

    // Helper methods
    size_t size() const;
    void resize_data();
};

/**
 * Basic QNN Context - manages NPU hardware connection
 * This is the foundational layer for all NPU operations
 */
class QNNContext {
public:
    QNNContext();
    ~QNNContext();
    QNNContext(const QNNContext&) = delete;
    QNNContext& operator=(const QNNContext&) = delete;

    // Initialization and validation
    bool initialize();
    bool is_initialized() const;
    std::string get_last_error() const;

private:
    friend class QNNModel;
    struct ContextImpl;
    std::unique_ptr<ContextImpl> pImpl;
};

/**
 * Basic QNN Model - computation graph management
 * Implements core transformer operations on NPU
 */
class QNNModel {
public:
    QNNModel(QNNContext& context);
    ~QNNModel();

    // Core operations for transformer models
    QNNTensor MatMul(const QNNTensor& a, const QNNTensor& b);
    QNNTensor Add(const QNNTensor& a, const QNNTensor& b);
    QNNTensor Softmax(const QNNTensor& a);
    QNNTensor GELU(const QNNTensor& a);
    QNNTensor SiLU(const QNNTensor& a);
    QNNTensor LayerNorm(const QNNTensor& a, const QNNTensor& weight, 
                       const QNNTensor& bias, float epsilon = 1e-5f);

    // Graph compilation and execution
    void compile();
    std::map<uint32_t, QNNTensor> execute(const std::map<uint32_t, QNNTensor>& inputs);

    // Performance monitoring
    struct BasicPerformanceMetrics {
        uint32_t operations_executed = 0;
        std::chrono::microseconds total_execution_time{0};
        std::chrono::microseconds compilation_time{0};
        
        double avg_ms_per_operation() const {
            if (operations_executed == 0) return 0.0;
            return (total_execution_time.count() / 1000.0) / operations_executed;
        }
        
        void print_summary() const;
    };
    
    const BasicPerformanceMetrics& get_metrics() const;
    void reset_metrics();

private:
    struct ModelImpl;
    std::unique_ptr<ModelImpl> pImpl; 
};

} // namespace qualcomm_npu
} // namespace atsentia