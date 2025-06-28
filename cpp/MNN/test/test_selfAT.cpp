#include "MSelfAT.hpp"
#include "test_common.hpp"
#include <iomanip>

using namespace mnn;
using namespace mtb;

#define FP_T float

void test_SelfAT(){
    START_TEST();
    {
        MSelfAT<FP_T> att(768, 16, 768/16);
        auto k_w = load_data<FP_T>("build/test/temp/k_proj_weight.bin", 768*768);
        auto v_w = load_data<FP_T>("build/test/temp/v_proj_weight.bin", 768*768);
        auto q_w = load_data<FP_T>("build/test/temp/q_proj_weight.bin", 768*768);
        auto o_w = load_data<FP_T>("build/test/temp/out_proj_weight.bin", 768*768);
        auto o_b = load_data<FP_T>("build/test/temp/out_proj_bias.bin", 768);
        auto it = load_data<FP_T>("build/test/temp/input.bin", 1*5*768);
        auto ot = load_data<FP_T>("build/test/temp/output.bin", 1*5*768);
        
        att.fill_k(k_w.data(), k_w.size());
        att.fill_v(v_w.data(), v_w.size());
        att.fill_q(q_w.data(), q_w.size());
        att.fill_out(o_w.data(), o_b.data(), o_w.size(), o_b.size());
        auto input_t = Tensor<FP_T>({1, 5, 768}, it);
        auto output_t = att.forward(input_t).copy();
  
        compare_data(output_t.data().get(), 
                    ot.data(), ot.size(), 1e-3);
    }  
    PASSLOG();
}

int main(int argc, char** argv) {
    test_SelfAT();
    return 0;
}