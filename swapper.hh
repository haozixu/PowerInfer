#pragma once

#include <unistd.h>
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// checks if a tensor is really swap enabled
int is_swap_enabled_tensor(const struct ggml_tensor *);

// make sure a swappable tensor is ready in memory
void validate_swap_enabled_tensor(struct ggml_tensor *);

// free (evict) swappable tensor
void free_swap_enabled_tensor(struct ggml_tensor *);

// collect predictor output
void collect_predictor_output(const struct ggml_tensor *, void *swapper);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>

#include <string>
#include <vector>
#include <mutex>
#include <future>
#include <utility>
#include <unordered_map>
#include <algorithm>

#include <fcntl.h>

// hard-coded for now
static inline bool is_swap_target_tensor(const char *name) {
    // assert(name != nullptr);
    return strstr(name, "ffn_gate.weight") ||
        strstr(name, "ffn_up.weight") ||
        strstr(name, "ffn_down.weight") ||
        strstr(name, "ffn_down_t.weight");
}

static inline bool is_swap_target_tensor(const ggml_tensor *x) {
    return is_swap_target_tensor(ggml_get_name(x));
}

static inline size_t round_up(size_t x, size_t alignment) {
    return ((x + alignment - 1) / alignment) * alignment;
}

// check if two intervals intersect
static inline bool intersects(size_t l0, size_t r0, size_t l1, size_t r1) {
    GGML_ASSERT(l0 < r0 && l1 < r1);

    return l1 < r0 && r1 > l0;
}

struct TensorInfo {
    int index; // auto-recorderd access index
    int layer_idx;
    size_t file_offset;
    size_t padded_len;
};

using Range = std::pair<size_t, size_t>;

struct Swapper {
    // static constexpr int TENSOR_ALIGN = 64;
    static constexpr uint64_t SWAPPER_MAGIC = 0xdeadbeefbadc0deULL;
    static constexpr uintptr_t POISON_ADDR = 0xffffffff00000000UL;
    static constexpr size_t MY_PAGE_SIZE = 4096;

    uint64_t magic;
    int file_fd; // descriptor of the weights file, only support single file
    bool defer_load;
    bool is_sparse_mode;

    size_t buf_size; // total size of swappable area
    uint8_t *buffer;

    std::vector<ggml_tensor *> access_order;
    std::unordered_map<ggml_tensor *, TensorInfo> info;
    
    // mutable fields
    std::mutex lock;
    std::unordered_map<ggml_tensor *, Range> valid_ranges; // tensors in memory
    std::unordered_map<ggml_tensor *, std::future<void *>> pending_reqs;

    int next_load_idx;
    size_t next_buf_pos;

    // sparse mode bookkeeping
    std::unordered_map<int, std::vector<int>> active_indices; // layer idx => active sparse indices made by predictor 

    Swapper() : magic{SWAPPER_MAGIC}, file_fd{-1}, defer_load{false}, is_sparse_mode{false},
        buf_size{0}, buffer{nullptr},
        next_load_idx{-1}, next_buf_pos(0) {}
    
    void init(size_t size, int fd, bool sparse_mode = false) {
        buf_size = size;
        file_fd = dup(fd); // duplicate incoming fd here for longer life-time
        // file_fd = open("/dev/block/dm-23", O_RDONLY | O_DIRECT);
        GGML_ASSERT(file_fd != -1);

        defer_load = getenv("LLAMA_SWAP_DEFER_LOAD") != nullptr; // if env var is non-NULL, defer async weight load
        is_sparse_mode = sparse_mode;

        int ret = posix_memalign((void **) &buffer, MY_PAGE_SIZE, size);
        GGML_ASSERT(!ret);

        fprintf(stderr, "swapper: buffer size: %.2f MiB, addr: %p\n", size / 1048576.0, buffer);
        if (defer_load) {
            fprintf(stderr, "swapper: async weight load deferred\n");
        }
    }

    ~Swapper() {
        // TODO
    }

    // hope this won't cause segfault
    bool check_magic() const { return magic == SWAPPER_MAGIC; }

    bool is_valid_buffer_addr(void *vp) const {
        auto p = (uint8_t *) vp;
        return buffer <= p && p < (buffer + buf_size);
    }

    void add_tensor_ordered(ggml_tensor *x, size_t offset_in_file, int layer_idx = -1) {
        int idx = access_order.size();
        access_order.push_back(x);

        TensorInfo ti;
        ti.index = idx;
        ti.layer_idx = layer_idx;
        ti.file_offset = offset_in_file;
        ti.padded_len = ggml_nbytes_pad(x);
        info[x] = ti;

        GGML_ASSERT(ti.padded_len < buf_size);
        
        // special mark
        // NOTE: we must use a non-NULL poison address here, otherwise cause bugs
        // other code logic will incorrectly handle a tensor whose data is null
        x->data = (void *) POISON_ADDR;
        x->extra = this;
    }

    void load_initial_tensors() {
        int i, n = access_order.size();
        size_t pos = 0;

        for (i = 0; i < n; ++i) {
            auto tensor = access_order[i];
            auto &ti = info[tensor];
            size_t end = pos + ti.padded_len;
            if (end > buf_size) {
                break;
            }

            // buffer can hold tensor[i]
            valid_ranges[tensor] = std::make_pair(pos, end);
            issue_load(tensor, buffer + pos);
            pos = round_up(end, MY_PAGE_SIZE);
        }

        GGML_ASSERT(i > 0);
        fprintf(stderr, "swapper: buffer can hold %d tensors initially\n", i);
        next_load_idx = i;
    }

    void validate(ggml_tensor *x) {
        if (is_valid_buffer_addr(x->data))
            return;

        std::future<void *> fut;
        {
            std::lock_guard<std::mutex> guard{lock};

            auto it = pending_reqs.find(x);
            GGML_ASSERT(it != pending_reqs.end());
            fut = std::move(it->second);
            pending_reqs.erase(it);
        }

        void *p = fut.get();
        GGML_ASSERT(p);
        x->data = p;
        // fprintf(stderr, "validate: tensor %s (index %d) ready at %p\n",
        //     ggml_get_name(x), info[x].index, p);
    }

    void issue_load(ggml_tensor *x, uint8_t *dst) {
        auto len = ggml_nbytes(x);
        auto off = info[x].file_offset;
        int layer_idx = info[x].layer_idx;
        int fd = file_fd;

        // TODO: direct file read
        auto launch_policy = defer_load ? std::launch::deferred : std::launch::async;
        pending_reqs[x] = std::async(launch_policy, [fd, dst, len, off, this, x, layer_idx]() -> void * {
            bool partial_read = false;
            std::vector<int> indices;
            if (this->is_sparse_mode) {
                // check if predictor output is ready, otherwise we fallback to dense mode
                std::lock_guard<std::mutex> guard{this->lock};

                auto it = this->active_indices.find(layer_idx);
                if (it != this->active_indices.end()) {
                    partial_read = true;
                    indices = it->second; // copy indices
                }
            }

            if (!partial_read) {
                ssize_t ret = pread(fd, dst, len, off);
                if (ret < 0) {
                    fprintf(stderr, "load weight failed: %s\n", strerror(errno));
                    return nullptr;
                }
            } else {
                // TODO: ensure weight memory layout
                size_t row_size = 4096 * ggml_type_size(x->type)/ggml_blck_size(x->type);

                for (int i : indices) {
                    GGML_ASSERT(0 <= i && i < x->ne[1]);
                    GGML_ASSERT((i + 1) * row_size <= len);

                    ssize_t ret = pread(fd, dst + i * row_size, row_size, off + i * row_size);
                    if (ret < 0) {
                        fprintf(stderr, "load weight block %d failed: %s\n", i, strerror(errno));
                        return nullptr;
                    }
                }
            }

            return dst;
        });
    }

    // NOTE: we assumes that there's no repeat access to weight tensors
    void free(ggml_tensor *x) {
        std::lock_guard<std::mutex> guard{lock};

        int n = access_order.size();
        if (next_load_idx >= n) {
            // no need to swap any tensor, keep this tensor valid
            return;
        }

        GGML_ASSERT(is_valid_buffer_addr(x->data));
        x->data = (void *) POISON_ADDR;

        GGML_ASSERT(valid_ranges.count(x));
        valid_ranges.erase(x);

        auto next = access_order[next_load_idx];
        auto start = next_buf_pos;
        auto end = start + info[next].padded_len;

        bool has_conflict = false;
        for (const auto &[_, range] : valid_ranges) {
            if (intersects(start, end, range.first, range.second)) {
                has_conflict = true;
                break;
            }
        }
        if (has_conflict) {
            return;
        }

        valid_ranges[next] = std::make_pair(start, end);
        next_load_idx = (next_load_idx + 1) % n;
        next_buf_pos = round_up(end, MY_PAGE_SIZE);
        if (buf_size < next_buf_pos + info[access_order[next_load_idx]].padded_len)
            next_buf_pos = 0;

        issue_load(next, buffer + start);
    }

    void add_predictor_output(const ggml_tensor *pred_out) {
        int nr_neurons = pred_out->ne[0];
        GGML_ASSERT(pred_out->type == GGML_TYPE_F32);

        // predictor output during prefill phase, skip
        if (pred_out->ne[1] > 1)
            return;

        auto name = std::string(ggml_get_name(pred_out));
        auto pos = name.find("-");
        GGML_ASSERT(pos != std::string::npos);
        int layer_idx = std::stoi(name.substr(pos + 1));
        
        std::vector<int> indices;
        float *data = (float *) pred_out->data;
        
        if (true) {
            // fixed sparsity ratio
            const float SPARSITY_RATIO = 0.15;
            int nr_active_neurons = int(nr_neurons * SPARSITY_RATIO);

            // only keep indices that correspond to top-k activation values
            for (int i = 0; i < nr_neurons; ++i)
                indices.push_back(i);
            std::sort(indices.begin(), indices.end(), [data, nr_neurons](int i, int j) {
                GGML_ASSERT(0 <= i && i < nr_neurons && 0 <= j && j < nr_neurons);
                return data[i] > data[j];
            });
            indices.resize(nr_active_neurons);
            std::sort(indices.begin(), indices.end());
        } else {
            for (int i = 0; i < nr_neurons; ++i)
                if (data[i] > 0.0)
                    indices.push_back(i);
        }

        {
            std::lock_guard<std::mutex> guard{lock};
            active_indices[layer_idx] = std::move(indices);
        }
    }
};

#endif
