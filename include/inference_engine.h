#pragma once

// Minimal inference engine interface for standalone npugpt build
namespace atsentia {

template<typename ConfigT>
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual bool initialize(const ConfigT& config) = 0;
    virtual bool is_initialized() const = 0;
};

} // namespace atsentia