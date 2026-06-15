#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <cstdint>

namespace py = pybind11;

static inline uint64_t pack_pair(uint32_t a, uint32_t b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

struct PairInfo {
    int rank;
    uint32_t new_id;
};

class FastBPEEncoder {
public:
    std::unordered_map<std::string, uint32_t> vocab;
    std::unordered_map<uint64_t, PairInfo> pair_info;

    FastBPEEncoder(
        const std::vector<std::pair<int, py::bytes>>& vocab_items,
        const std::vector<std::pair<py::bytes, py::bytes>>& merges
    ) {
        vocab.reserve(vocab_items.size() * 2);

        for (const auto& item : vocab_items) {
            int id = item.first;
            std::string token = item.second;
            vocab[token] = static_cast<uint32_t>(id);
        }

        pair_info.reserve(merges.size() * 2);

        for (int rank = 0; rank < static_cast<int>(merges.size()); ++rank) {
            std::string left = merges[rank].first;
            std::string right = merges[rank].second;
            std::string merged = left + right;

            auto it_l = vocab.find(left);
            auto it_r = vocab.find(right);
            auto it_m = vocab.find(merged);

            if (it_l == vocab.end() || it_r == vocab.end() || it_m == vocab.end()) {
                continue;
            }

            uint32_t left_id = it_l->second;
            uint32_t right_id = it_r->second;
            uint32_t merged_id = it_m->second;

            uint64_t key = pack_pair(left_id, right_id);

            pair_info[key] = PairInfo{
                rank,
                merged_id
            };
        }
    }

    std::vector<uint32_t> encode_piece(py::bytes piece_bytes) const {
        std::string piece = piece_bytes;

        std::vector<uint32_t> tokens;
        tokens.reserve(piece.size());

        for (unsigned char c : piece) {
            tokens.push_back(static_cast<uint32_t>(c));
        }

        if (tokens.size() <= 1) {
            return tokens;
        }

        while (true) {
            int best_rank = std::numeric_limits<int>::max();
            uint64_t best_pair = 0;
            uint32_t best_new_id = 0;
            bool found = false;

            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                uint64_t key = pack_pair(tokens[i], tokens[i + 1]);
                auto it = pair_info.find(key);

                if (it != pair_info.end()) {
                    int rank = it->second.rank;

                    if (rank < best_rank) {
                        best_rank = rank;
                        best_pair = key;
                        best_new_id = it->second.new_id;
                        found = true;
                    }
                }
            }

            if (!found) {
                break;
            }

            uint32_t left = static_cast<uint32_t>(best_pair >> 32);
            uint32_t right = static_cast<uint32_t>(best_pair & 0xffffffffu);

            std::vector<uint32_t> new_tokens;
            new_tokens.reserve(tokens.size());

            size_t i = 0;
            while (i < tokens.size()) {
                if (
                    i + 1 < tokens.size() &&
                    tokens[i] == left &&
                    tokens[i + 1] == right
                ) {
                    new_tokens.push_back(best_new_id);
                    i += 2;
                } else {
                    new_tokens.push_back(tokens[i]);
                    i += 1;
                }
            }

            tokens.swap(new_tokens);

            if (tokens.size() <= 1) {
                break;
            }
        }

        return tokens;
    }
};

PYBIND11_MODULE(bpe_fast, m) {
    py::class_<FastBPEEncoder>(m, "FastBPEEncoder")
        .def(
            py::init<
                const std::vector<std::pair<int, py::bytes>>&,
                const std::vector<std::pair<py::bytes, py::bytes>>&
            >()
        )
        .def("encode_piece", &FastBPEEncoder::encode_piece);
}
