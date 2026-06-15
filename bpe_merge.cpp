#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <algorithm>
#include <iostream>

namespace py = pybind11;

using TokenId = uint32_t;
using WordId = uint32_t;
using Count = uint64_t;

static inline uint64_t pack_pair(TokenId a, TokenId b) {
    return (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
}

static inline TokenId first_token(uint64_t p) {
    return static_cast<TokenId>(p >> 32);
}

static inline TokenId second_token(uint64_t p) {
    return static_cast<TokenId>(p & 0xffffffffu);
}

struct PairHash {
    std::size_t operator()(const uint64_t& x) const {
        return std::hash<uint64_t>()(x);
    }
};

std::vector<std::pair<TokenId, TokenId>> bpe_train_core(
    const std::vector<std::string>& words,
    const std::vector<Count>& freqs,
    int num_merges,
    int initial_vocab_size,
    bool verbose
) {
    const size_t num_words = words.size();

    std::vector<std::vector<TokenId>> word_tokens;
    word_tokens.reserve(num_words);

    std::unordered_map<uint64_t, Count, PairHash> pair_counts;
    std::unordered_map<uint64_t, std::unordered_set<WordId>, PairHash> pair_to_words;

    pair_counts.reserve(1 << 20);
    pair_to_words.reserve(1 << 20);

    // 初始化：每个 byte 是一个 token，id = 0..255
    for (WordId word_id = 0; word_id < num_words; ++word_id) {
        const std::string& w = words[word_id];

        std::vector<TokenId> tokens;
        tokens.reserve(w.size());

        for (unsigned char c : w) {
            tokens.push_back(static_cast<TokenId>(c));
        }

        word_tokens.push_back(std::move(tokens));

        const auto& toks = word_tokens.back();
        Count freq = freqs[word_id];

        if (toks.size() >= 2) {
            for (size_t i = 0; i + 1 < toks.size(); ++i) {
                uint64_t p = pack_pair(toks[i], toks[i + 1]);
                pair_counts[p] += freq;
                pair_to_words[p].insert(word_id);
            }
        }
    }

    std::vector<std::pair<TokenId, TokenId>> merges;
    merges.reserve(num_merges);

    TokenId next_token_id = static_cast<TokenId>(initial_vocab_size);

    for (int merge_step = 0; merge_step < num_merges; ++merge_step) {
        if (pair_counts.empty()) {
            break;
        }

        // 找当前最高频 pair
        // tie-break：count 大优先；count 相同则 token id pair 字典序大优先
        uint64_t best_pair = 0;
        Count best_count = 0;
        bool has_best = false;

        for (const auto& kv : pair_counts) {
            uint64_t p = kv.first;
            Count c = kv.second;

            if (!has_best) {
                best_pair = p;
                best_count = c;
                has_best = true;
                continue;
            }

            TokenId a1 = first_token(p);
            TokenId b1 = second_token(p);
            TokenId a2 = first_token(best_pair);
            TokenId b2 = second_token(best_pair);

            if (
                c > best_count ||
                (c == best_count && (a1 > a2 || (a1 == a2 && b1 > b2)))
            ) {
                best_pair = p;
                best_count = c;
            }
        }

        if (!has_best || best_count == 0) {
            break;
        }

        TokenId left = first_token(best_pair);
        TokenId right = second_token(best_pair);
        TokenId new_token = next_token_id++;

        merges.push_back({left, right});

        std::vector<WordId> affected_words;

        auto it_words = pair_to_words.find(best_pair);
        if (it_words != pair_to_words.end()) {
            affected_words.reserve(it_words->second.size());
            for (WordId wid : it_words->second) {
                affected_words.push_back(wid);
            }
            pair_to_words.erase(it_words);
        }

        // 从 pair_counts 中移除 best_pair
        pair_counts.erase(best_pair);

        for (WordId word_id : affected_words) {
            auto& old_tokens = word_tokens[word_id];
            Count freq = freqs[word_id];

            if (old_tokens.size() < 2) {
                continue;
            }

            // 1. 删除这个 word 的所有旧 pair 贡献
            for (size_t i = 0; i + 1 < old_tokens.size(); ++i) {
                uint64_t old_pair = pack_pair(old_tokens[i], old_tokens[i + 1]);

                auto it_count = pair_counts.find(old_pair);
                if (it_count != pair_counts.end()) {
                    if (it_count->second <= freq) {
                        pair_counts.erase(it_count);
                        pair_to_words.erase(old_pair);
                    } else {
                        it_count->second -= freq;

                        auto it_set = pair_to_words.find(old_pair);
                        if (it_set != pair_to_words.end()) {
                            it_set->second.erase(word_id);
                            if (it_set->second.empty()) {
                                pair_to_words.erase(it_set);
                            }
                        }
                    }
                }
            }

            // 2. 合并 best_pair
            std::vector<TokenId> new_tokens;
            new_tokens.reserve(old_tokens.size());

            size_t i = 0;
            while (i < old_tokens.size()) {
                if (
                    i + 1 < old_tokens.size() &&
                    old_tokens[i] == left &&
                    old_tokens[i + 1] == right
                ) {
                    new_tokens.push_back(new_token);
                    i += 2;
                } else {
                    new_tokens.push_back(old_tokens[i]);
                    i += 1;
                }
            }

            old_tokens.swap(new_tokens);

            // 3. 加回新 pair 贡献
            if (old_tokens.size() >= 2) {
                for (size_t j = 0; j + 1 < old_tokens.size(); ++j) {
                    uint64_t new_pair = pack_pair(old_tokens[j], old_tokens[j + 1]);
                    pair_counts[new_pair] += freq;
                    pair_to_words[new_pair].insert(word_id);
                }
            }
        }

        if (verbose && merge_step % 100 == 0) {
            std::cout
                << "merge "
                << merge_step
                << " / "
                << num_merges
                << ", best_count="
                << best_count
                << ", num_pairs="
                << pair_counts.size()
                << std::endl;
        }
    }

    return merges;
}

PYBIND11_MODULE(bpe_merge, m) {
    m.def(
        "bpe_train_core",
        &bpe_train_core,
        py::arg("words"),
        py::arg("freqs"),
        py::arg("num_merges"),
        py::arg("initial_vocab_size") = 256,
        py::arg("verbose") = false
    );
}
