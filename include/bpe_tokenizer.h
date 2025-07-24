#pragma once

/**
 * Atsentia AI Accelerator - BPE Tokenizer for GPT-2
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace atsentia {
namespace models {
namespace gpt2 {

/**
 * Byte-Pair Encoding Tokenizer for GPT-2
 */
class BPETokenizer {
public:
    BPETokenizer();
    ~BPETokenizer();
    
    // Load tokenizer data
    bool load_from_path(const std::string& tokenizer_path);
    
    // Tokenization
    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& tokens) const;
    std::string decode(uint32_t token) const;
    
    // Vocabulary
    uint32_t get_vocab_size() const { return vocab_size_; }
    
    // Special tokens
    uint32_t get_eos_token() const { return eos_token_; }
    uint32_t get_bos_token() const { return bos_token_; }
    uint32_t get_pad_token() const { return pad_token_; }
    
private:
    std::unordered_map<std::string, uint32_t> token_to_id_;
    std::unordered_map<uint32_t, std::string> id_to_token_;
    std::unordered_map<std::string, std::string> byte_encoder_;
    std::unordered_map<std::string, std::string> byte_decoder_;
    
    uint32_t vocab_size_ = 50257;
    uint32_t eos_token_ = 50256;
    uint32_t bos_token_ = 50256;
    uint32_t pad_token_ = 50256;
    
    // BPE implementation
    std::vector<std::string> bpe_encode(const std::string& token) const;
    std::string bytes_to_unicode() const;
};

} // namespace gpt2
} // namespace models
} // namespace atsentia