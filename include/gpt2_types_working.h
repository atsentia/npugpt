#pragma once

/**
 * Atsentia AI Accelerator - GPT-2 Data Types and Structures
 * 
 * Note: This file provides forward declarations only to avoid conflicts
 * with the main gpt2_engine.h definitions. The actual structures are
 * defined in gpt2_engine.h.
 */

#include <vector>
#include <string>
#include <memory>

namespace atsentia {
namespace models {
namespace gpt2 {

// Forward declarations - actual definitions are in gpt2_engine.h
struct GPT2Config;
struct GPT2Weights;
class GPT2Loader;
class BPETokenizer;

} // namespace gpt2
} // namespace models
} // namespace atsentia