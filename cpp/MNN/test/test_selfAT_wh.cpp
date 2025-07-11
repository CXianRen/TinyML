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
     
        auto it_s = load_data<FP_T>("build/test/temp/input_single.bin", 1*1*768);
        auto ot_s = load_data<FP_T>("build/test/temp/output_single.bin", 1*1*768);
        auto k_h = load_data<FP_T>("build/test/temp/k_h.bin", 1*16*5*48);
        auto v_h = load_data<FP_T>("build/test/temp/v_h.bin", 1*16*5*48);
        
        att.fill_k(k_w.data(), k_w.size());
        att.fill_v(v_w.data(), v_w.size());
        att.fill_q(q_w.data(), q_w.size());
        att.fill_out(o_w.data(), o_b.data(), o_w.size(), o_b.size());

        auto it_s_t = Tensor<FP_T>({1, 1, 768}, it_s);

        auto k_h_t = Tensor<FP_T>({1, 16, 5, 48}, k_h);
        auto v_h_t = Tensor<FP_T>({1, 16, 5, 48}, v_h);

        auto ot_s_t = att.forward(it_s_t, &k_h_t, &v_h_t).copy();
  
        compare_data(ot_s_t.data().get(), 
                    ot_s.data(), ot_s.size(), 1e-3);
    }  
    PASSLOG();
}

int main(int argc, char** argv) {
    test_SelfAT();
    return 0;
}