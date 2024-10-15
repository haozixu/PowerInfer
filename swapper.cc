#include "swapper.hh"

extern "C" {

int is_swap_enabled_tensor(const ggml_tensor *x) {
    if (!x->extra) {
        return 0;
    }

    auto swapper = reinterpret_cast<const Swapper *>(x->extra);
    return swapper->check_magic();
}

void validate_swap_enabled_tensor(ggml_tensor *x) {
    auto swapper = reinterpret_cast<Swapper *>(x->extra);

    swapper->validate(x);
}

void free_swap_enabled_tensor(ggml_tensor *x) {
    auto swapper = reinterpret_cast<Swapper *>(x->extra);

    swapper->free(x);
}

void collect_predictor_output(const struct ggml_tensor *pred_out, void *swapper) {
    reinterpret_cast<Swapper *>(swapper)->add_predictor_output(pred_out);
}

}
