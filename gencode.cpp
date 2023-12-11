#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)                 \
  try {                                                      \
    __VA_ARGS__                                              \
  } catch (const std::exception& e) {                        \
    std::cerr << "Error: " << e.what() << std::endl;         \
    return AOTI_RUNTIME_FAILURE;                             \
  } catch (...) {                                            \
    std::cerr << "Unknown exception occurred." << std::endl; \
    return AOTI_RUNTIME_FAILURE;                             \
  }                                                          \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() : prev_mode(aoti_torch_grad_mode_is_enabled()) {
    aoti_torch_grad_mode_set_enabled(false);
  }
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  bool prev_mode;
};

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0" << std::endl;
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, is_cpu, cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream = reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles,
        output_handles,
        stream,
        proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_inactive_constant_buffer(*input_map);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto constant_array = std::make_shared<std::vector<AtenTensorHandle>>();
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          ""
      );

      if (input_map) {
        for (auto const& kv : *input_map) {
          constant_map->emplace(kv.first, kv.second);
        }
      } else {
        model->load_constants(/*is_cpu*/true);
      }

      *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
  })
}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType)nullptr,
        nullptr);
  })
}


AOTIRuntimeError AOTInductorModelDelete(
    AOTInductorModelHandle model_handle
) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      delete model;
  })
}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      for (auto const& kv : *input_map) {
        constant_map->emplace(kv.first, kv.second);
      }
      model->update_constants_map(std::move(constant_map));
  })
}

#define CACHE_TORCH_DTYPE(typename) static auto cached_torch_dtype_##typename = aoti_torch_dtype_##typename()

  static auto cached_torch_device_type_cpu = aoti_torch_device_type_cpu();
  static auto cached_torch_device_type_cuda = aoti_torch_device_type_cuda();
} // extern "C"

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/native/BinaryOps.h>
#include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
#include <torch/csrc/inductor/inductor_ops.h>
#define reinterpret_tensor torch::inductor::_reinterpret_tensor
#define alloc_from_pool torch::inductor::_alloc_from_pool
#include <c10/util/generic_math.h>

[[maybe_unused]] static int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_embedding_mean_mul_pow_rsqrt_0(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                    auto tmp10 = 1 / std::sqrt(tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_1(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_embedding_mean_mul_pow_rsqrt_2(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = static_cast<float>(1e-05);
                    auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                    auto tmp12 = 1 / std::sqrt(tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp16 = tmp14 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_3(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_embedding_mean_mul_pow_rsqrt_4(const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp9 = tmp8 * tmp8;
                        tmp_acc0_vec = tmp_acc0_vec + tmp9;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = in_ptr0[static_cast<long>(x0)];
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp9 = out_ptr0[static_cast<long>(x0)];
                    auto tmp17 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                    auto tmp2 = tmp0 < 0;
                    auto tmp3 = tmp2 ? tmp1 : tmp0;
                    TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                    auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp10 = static_cast<float>(768.0);
                    auto tmp11 = tmp9 / tmp10;
                    auto tmp12 = static_cast<float>(1e-05);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = 1 / std::sqrt(tmp13);
                    auto tmp15 = at::vec::Vectorized<float>(tmp14);
                    auto tmp16 = tmp8 * tmp15;
                    auto tmp18 = tmp16 * tmp17;
                    tmp18.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_5(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_embedding_mean_mul_pow_rsqrt_6(float* in_out_ptr0,
                       const long* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = in_ptr0[static_cast<long>(x0)];
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = decltype(tmp0)(tmp0 + 32000);
                        auto tmp2 = tmp0 < 0;
                        auto tmp3 = tmp2 ? tmp1 : tmp0;
                        TORCH_CHECK((0 <= tmp3) & (tmp3 < 32000L), "index out of bounds: 0 <= tmp3 < 32000L")
                        auto tmp4 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*tmp3)));
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp8 = tmp6 + tmp7;
                        auto tmp10 = tmp8 + tmp9;
                        auto tmp11 = tmp10 * tmp10;
                        tmp10.store(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        tmp_acc0_vec = tmp_acc0_vec + tmp11;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = 1 / std::sqrt(tmp5);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp0 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_7(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_8(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp2 * tmp9;
                    auto tmp12 = tmp10 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_9(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_10(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                    auto tmp10 = 1 / std::sqrt(tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_11(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_12(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = static_cast<float>(1e-05);
                    auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                    auto tmp12 = 1 / std::sqrt(tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp16 = tmp14 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_13(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_14(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = 1 / std::sqrt(tmp5);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp0 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_15(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_16(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp2 * tmp9;
                    auto tmp12 = tmp10 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_17(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_18(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                    auto tmp10 = 1 / std::sqrt(tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_19(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_20(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = static_cast<float>(1e-05);
                    auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                    auto tmp12 = 1 / std::sqrt(tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp16 = tmp14 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_21(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_22(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = 1 / std::sqrt(tmp5);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp0 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_23(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_24(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp2 * tmp9;
                    auto tmp12 = tmp10 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_25(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_26(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                    auto tmp10 = 1 / std::sqrt(tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_27(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_28(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = static_cast<float>(1e-05);
                    auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                    auto tmp12 = 1 / std::sqrt(tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp16 = tmp14 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_29(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_30(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = 1 / std::sqrt(tmp5);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp0 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_31(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_32(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp2 * tmp9;
                    auto tmp12 = tmp10 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_33(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_34(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                    auto tmp10 = 1 / std::sqrt(tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_35(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_36(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = static_cast<float>(1e-05);
                    auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                    auto tmp12 = 1 / std::sqrt(tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp16 = tmp14 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_37(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_38(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = 1 / std::sqrt(tmp5);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp0 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_39(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_40(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = out_ptr0[static_cast<long>(x0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp2 * tmp9;
                    auto tmp12 = tmp10 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_41(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_42(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp5 = tmp4 * tmp4;
                        tmp_acc0_vec = tmp_acc0_vec + tmp5;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = out_ptr0[static_cast<long>(x0)];
                    auto tmp13 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = static_cast<float>(768.0);
                    auto tmp7 = tmp5 / tmp6;
                    auto tmp8 = static_cast<float>(1e-05);
                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                    auto tmp10 = 1 / std::sqrt(tmp9);
                    auto tmp11 = at::vec::Vectorized<float>(tmp10);
                    auto tmp12 = tmp4 * tmp11;
                    auto tmp14 = tmp12 * tmp13;
                    tmp14.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_43(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_44(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp4 = tmp2 + tmp3;
                        auto tmp6 = tmp4 + tmp5;
                        auto tmp7 = tmp6 * tmp6;
                        tmp_acc0_vec = tmp_acc0_vec + tmp7;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp7 = out_ptr0[static_cast<long>(x0)];
                    auto tmp15 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp6 = tmp4 + tmp5;
                    auto tmp8 = static_cast<float>(768.0);
                    auto tmp9 = tmp7 / tmp8;
                    auto tmp10 = static_cast<float>(1e-05);
                    auto tmp11 = decltype(tmp9)(tmp9 + tmp10);
                    auto tmp12 = 1 / std::sqrt(tmp11);
                    auto tmp13 = at::vec::Vectorized<float>(tmp12);
                    auto tmp14 = tmp6 * tmp13;
                    auto tmp16 = tmp14 * tmp15;
                    tmp16.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_stack_45(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(12L); x1+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x2=static_cast<long>(0L); x2<static_cast<long>(32L); x2+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x3=static_cast<long>(0L); x3<static_cast<long>(2L); x3+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x3);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp7 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                                auto tmp9 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp10 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp8)(tmp8 - tmp11);
                                return tmp12;
                            }
                            ;
                            auto tmp13 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp14 = tmp0 >= tmp3;
                            auto tmp15 = static_cast<long>(2);
                            auto tmp16 = tmp0 < tmp15;
                            auto tmp17 = [&]
                            {
                                auto tmp18 = in_ptr0[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp19 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                                auto tmp21 = in_ptr0[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp22 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                                auto tmp24 = decltype(tmp20)(tmp20 + tmp23);
                                return tmp24;
                            }
                            ;
                            auto tmp25 = tmp14 ? tmp17() : static_cast<decltype(tmp17())>(0.0);
                            auto tmp26 = tmp4 ? tmp13 : tmp25;
                            auto tmp27 = [&]
                            {
                                auto tmp28 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp29 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                                auto tmp31 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp32 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp33 = decltype(tmp31)(tmp31 * tmp32);
                                auto tmp34 = decltype(tmp30)(tmp30 - tmp33);
                                return tmp34;
                            }
                            ;
                            auto tmp35 = tmp4 ? tmp27() : static_cast<decltype(tmp27())>(0.0);
                            auto tmp36 = [&]
                            {
                                auto tmp37 = in_ptr3[static_cast<long>((2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp38 = in_ptr2[static_cast<long>(x2 + (32L*x0))];
                                auto tmp39 = decltype(tmp37)(tmp37 * tmp38);
                                auto tmp40 = in_ptr3[static_cast<long>(1L + (2L*x2) + (64L*x1) + (768L*x0))];
                                auto tmp41 = in_ptr1[static_cast<long>(x2 + (32L*x0))];
                                auto tmp42 = decltype(tmp40)(tmp40 * tmp41);
                                auto tmp43 = decltype(tmp39)(tmp39 + tmp42);
                                return tmp43;
                            }
                            ;
                            auto tmp44 = tmp14 ? tmp36() : static_cast<decltype(tmp36())>(0.0);
                            auto tmp45 = tmp4 ? tmp35 : tmp44;
                            out_ptr0[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp26;
                            out_ptr1[static_cast<long>(x3 + (2L*x2) + (64L*x1) + (768L*x0))] = tmp45;
                        }
                    }
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_mean_mul_pow_rsqrt_46(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       const float* in_ptr3,
                       const float* in_ptr4,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x0));
                auto tmp5 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                auto tmp7 = at::vec::Vectorized<float>::loadu(in_ptr3 + static_cast<long>(x0));
                auto tmp2 = tmp0 + tmp1;
                auto tmp4 = tmp2 + tmp3;
                auto tmp6 = tmp4 + tmp5;
                auto tmp8 = tmp6 + tmp7;
                tmp8.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = tmp0 * tmp0;
                        tmp_acc0_vec = tmp_acc0_vec + tmp1;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x1 + (768L*x0)));
                    auto tmp1 = out_ptr0[static_cast<long>(x0)];
                    auto tmp9 = at::vec::Vectorized<float>::loadu(in_ptr4 + static_cast<long>(x1));
                    auto tmp2 = static_cast<float>(768.0);
                    auto tmp3 = tmp1 / tmp2;
                    auto tmp4 = static_cast<float>(1e-05);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = 1 / std::sqrt(tmp5);
                    auto tmp7 = at::vec::Vectorized<float>(tmp6);
                    auto tmp8 = tmp0 * tmp7;
                    auto tmp10 = tmp8 * tmp9;
                    tmp10.store(out_ptr1 + static_cast<long>(x1 + (768L*x0)));
                }
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_mul_silu_47(float* in_out_ptr0,
                       const float* in_ptr0,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(2048L*ks0); x0+=static_cast<long>(16L))
            {
                auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
                auto tmp3 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x0));
                auto tmp1 = decltype(tmp0)(1)/(decltype(tmp0)(1) + tmp0.neg().exp());
                auto tmp2 = tmp0 * tmp1;
                auto tmp4 = tmp2 * tmp3;
                tmp4.store(in_out_ptr0 + static_cast<long>(x0));
            }
        }
    }
}

#include "/tmp/torchinductor_bertrand/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void cpp_fused_add_index_mean_mul_pow_rsqrt_48(const float* in_ptr0,
                       const float* in_ptr1,
                       const float* in_ptr2,
                       float* out_ptr0,
                       float* out_ptr1,
                       const long ks0)
{
    #pragma omp parallel num_threads(96)
    {
        {
            #pragma omp for 
            for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
            {
                {
                    #pragma omp declare reduction(+:at::vec::Vectorized<float>:omp_out = omp_out + omp_in) initializer(omp_priv={at::vec::Vectorized<float>(0)})
                    float tmp_acc0 = 0;
                    at::vec::Vectorized<float> tmp_acc0_vec = at::vec::Vectorized<float>(0);
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(768L); x1+=static_cast<long>(16L))
                    {
                        auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>(x1 + (768L*x0)));
                        auto tmp2 = tmp0 + tmp1;
                        auto tmp3 = tmp2 * tmp2;
                        tmp_acc0_vec = tmp_acc0_vec + tmp3;
                    }
                    tmp_acc0 = tmp_acc0 + at::vec::vec_reduce_all<float>([](at::vec::Vectorized<float>& x, at::vec::Vectorized<float>& y) { return x + y; }, tmp_acc0_vec);
                    out_ptr0[static_cast<long>(x0)] = static_cast<float>(tmp_acc0);
                }
            }
        }
        #pragma omp single
        {
            {
                for(long x0=static_cast<long>(0L); x0<static_cast<long>(768L); x0+=static_cast<long>(16L))
                {
                    auto tmp0 = at::vec::Vectorized<float>::loadu(in_ptr0 + static_cast<long>((-768L) + x0 + (768L*ks0)));
                    auto tmp1 = at::vec::Vectorized<float>::loadu(in_ptr1 + static_cast<long>((-768L) + x0 + (768L*ks0)));
                    auto tmp3 = out_ptr0[static_cast<long>((-1L) + ks0)];
                    auto tmp11 = at::vec::Vectorized<float>::loadu(in_ptr2 + static_cast<long>(x0));
                    auto tmp2 = tmp0 + tmp1;
                    auto tmp4 = static_cast<float>(768.0);
                    auto tmp5 = tmp3 / tmp4;
                    auto tmp6 = static_cast<float>(1e-05);
                    auto tmp7 = decltype(tmp5)(tmp5 + tmp6);
                    auto tmp8 = 1 / std::sqrt(tmp7);
                    auto tmp9 = at::vec::Vectorized<float>(tmp8);
                    auto tmp10 = tmp2 * tmp9;
                    auto tmp12 = tmp10 * tmp11;
                    tmp12.store(out_ptr1 + static_cast<long>(x0));
                }
            }
        }
    }
}
namespace torch {
namespace aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
};
}  // namespace

AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<AtenTensorHandle>> constants_array,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(1, 1, 112, cubin_dir) {
    inputs_info_[0].name = "arg113_1";
    constants_info_[0].name = "L__self___layers_0_attention_norm_weight";
    constants_info_[0].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 3072;
    constants_info_[0].shape = {768};
    constants_info_[0].stride = {1};
    constants_info_[1].name = "L__self___layers_0_ffn_norm_weight";
    constants_info_[1].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[1].offset = 0;
    constants_info_[1].data_size = 3072;
    constants_info_[1].shape = {768};
    constants_info_[1].stride = {1};
    constants_info_[2].name = "L__self___layers_1_attention_norm_weight";
    constants_info_[2].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[2].offset = 0;
    constants_info_[2].data_size = 3072;
    constants_info_[2].shape = {768};
    constants_info_[2].stride = {1};
    constants_info_[3].name = "L__self___layers_1_ffn_norm_weight";
    constants_info_[3].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[3].offset = 0;
    constants_info_[3].data_size = 3072;
    constants_info_[3].shape = {768};
    constants_info_[3].stride = {1};
    constants_info_[4].name = "L__self___layers_2_attention_norm_weight";
    constants_info_[4].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[4].offset = 0;
    constants_info_[4].data_size = 3072;
    constants_info_[4].shape = {768};
    constants_info_[4].stride = {1};
    constants_info_[5].name = "L__self___layers_2_ffn_norm_weight";
    constants_info_[5].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[5].offset = 0;
    constants_info_[5].data_size = 3072;
    constants_info_[5].shape = {768};
    constants_info_[5].stride = {1};
    constants_info_[6].name = "L__self___layers_3_attention_norm_weight";
    constants_info_[6].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[6].offset = 0;
    constants_info_[6].data_size = 3072;
    constants_info_[6].shape = {768};
    constants_info_[6].stride = {1};
    constants_info_[7].name = "L__self___layers_3_ffn_norm_weight";
    constants_info_[7].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[7].offset = 0;
    constants_info_[7].data_size = 3072;
    constants_info_[7].shape = {768};
    constants_info_[7].stride = {1};
    constants_info_[8].name = "L__self___layers_4_attention_norm_weight";
    constants_info_[8].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[8].offset = 0;
    constants_info_[8].data_size = 3072;
    constants_info_[8].shape = {768};
    constants_info_[8].stride = {1};
    constants_info_[9].name = "L__self___layers_4_ffn_norm_weight";
    constants_info_[9].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[9].offset = 0;
    constants_info_[9].data_size = 3072;
    constants_info_[9].shape = {768};
    constants_info_[9].stride = {1};
    constants_info_[10].name = "L__self___layers_5_attention_norm_weight";
    constants_info_[10].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[10].offset = 0;
    constants_info_[10].data_size = 3072;
    constants_info_[10].shape = {768};
    constants_info_[10].stride = {1};
    constants_info_[11].name = "L__self___layers_5_ffn_norm_weight";
    constants_info_[11].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[11].offset = 0;
    constants_info_[11].data_size = 3072;
    constants_info_[11].shape = {768};
    constants_info_[11].stride = {1};
    constants_info_[12].name = "L__self___layers_6_attention_norm_weight";
    constants_info_[12].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[12].offset = 0;
    constants_info_[12].data_size = 3072;
    constants_info_[12].shape = {768};
    constants_info_[12].stride = {1};
    constants_info_[13].name = "L__self___layers_6_ffn_norm_weight";
    constants_info_[13].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[13].offset = 0;
    constants_info_[13].data_size = 3072;
    constants_info_[13].shape = {768};
    constants_info_[13].stride = {1};
    constants_info_[14].name = "L__self___layers_7_attention_norm_weight";
    constants_info_[14].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[14].offset = 0;
    constants_info_[14].data_size = 3072;
    constants_info_[14].shape = {768};
    constants_info_[14].stride = {1};
    constants_info_[15].name = "L__self___layers_7_ffn_norm_weight";
    constants_info_[15].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[15].offset = 0;
    constants_info_[15].data_size = 3072;
    constants_info_[15].shape = {768};
    constants_info_[15].stride = {1};
    constants_info_[16].name = "L__self___layers_8_attention_norm_weight";
    constants_info_[16].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[16].offset = 0;
    constants_info_[16].data_size = 3072;
    constants_info_[16].shape = {768};
    constants_info_[16].stride = {1};
    constants_info_[17].name = "L__self___layers_8_ffn_norm_weight";
    constants_info_[17].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[17].offset = 0;
    constants_info_[17].data_size = 3072;
    constants_info_[17].shape = {768};
    constants_info_[17].stride = {1};
    constants_info_[18].name = "L__self___layers_9_attention_norm_weight";
    constants_info_[18].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[18].offset = 0;
    constants_info_[18].data_size = 3072;
    constants_info_[18].shape = {768};
    constants_info_[18].stride = {1};
    constants_info_[19].name = "L__self___layers_9_ffn_norm_weight";
    constants_info_[19].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[19].offset = 0;
    constants_info_[19].data_size = 3072;
    constants_info_[19].shape = {768};
    constants_info_[19].stride = {1};
    constants_info_[20].name = "L__self___layers_10_attention_norm_weight";
    constants_info_[20].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[20].offset = 0;
    constants_info_[20].data_size = 3072;
    constants_info_[20].shape = {768};
    constants_info_[20].stride = {1};
    constants_info_[21].name = "L__self___layers_10_ffn_norm_weight";
    constants_info_[21].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[21].offset = 0;
    constants_info_[21].data_size = 3072;
    constants_info_[21].shape = {768};
    constants_info_[21].stride = {1};
    constants_info_[22].name = "L__self___layers_11_attention_norm_weight";
    constants_info_[22].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[22].offset = 0;
    constants_info_[22].data_size = 3072;
    constants_info_[22].shape = {768};
    constants_info_[22].stride = {1};
    constants_info_[23].name = "L__self___layers_11_ffn_norm_weight";
    constants_info_[23].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[23].offset = 0;
    constants_info_[23].data_size = 3072;
    constants_info_[23].shape = {768};
    constants_info_[23].stride = {1};
    constants_info_[24].name = "L__self___norm_weight";
    constants_info_[24].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[24].offset = 0;
    constants_info_[24].data_size = 3072;
    constants_info_[24].shape = {768};
    constants_info_[24].stride = {1};
    constants_info_[25].name = "L__self___tok_embeddings_weight";
    constants_info_[25].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[25].offset = 0;
    constants_info_[25].data_size = 98304000;
    constants_info_[25].shape = {32000, 768};
    constants_info_[25].stride = {768, 1};
    constants_info_[26].name = "L__self___layers_0_attention_wq_weight";
    constants_info_[26].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[26].offset = 0;
    constants_info_[26].data_size = 2359296;
    constants_info_[26].shape = {768, 768};
    constants_info_[26].stride = {768, 1};
    constants_info_[27].name = "L__self___layers_0_attention_wk_weight";
    constants_info_[27].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[27].offset = 0;
    constants_info_[27].data_size = 2359296;
    constants_info_[27].shape = {768, 768};
    constants_info_[27].stride = {768, 1};
    constants_info_[28].name = "L__self___layers_0_attention_wv_weight";
    constants_info_[28].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[28].offset = 0;
    constants_info_[28].data_size = 2359296;
    constants_info_[28].shape = {768, 768};
    constants_info_[28].stride = {768, 1};
    constants_info_[29].name = "L__self___layers_0_attention_wo_weight";
    constants_info_[29].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[29].offset = 0;
    constants_info_[29].data_size = 2359296;
    constants_info_[29].shape = {768, 768};
    constants_info_[29].stride = {768, 1};
    constants_info_[30].name = "L__self___layers_0_feed_forward_w1_weight";
    constants_info_[30].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[30].offset = 0;
    constants_info_[30].data_size = 6291456;
    constants_info_[30].shape = {2048, 768};
    constants_info_[30].stride = {768, 1};
    constants_info_[31].name = "L__self___layers_0_feed_forward_w3_weight";
    constants_info_[31].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[31].offset = 0;
    constants_info_[31].data_size = 6291456;
    constants_info_[31].shape = {2048, 768};
    constants_info_[31].stride = {768, 1};
    constants_info_[32].name = "L__self___layers_0_feed_forward_w2_weight";
    constants_info_[32].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[32].offset = 0;
    constants_info_[32].data_size = 6291456;
    constants_info_[32].shape = {768, 2048};
    constants_info_[32].stride = {2048, 1};
    constants_info_[33].name = "L__self___layers_1_attention_wq_weight";
    constants_info_[33].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[33].offset = 0;
    constants_info_[33].data_size = 2359296;
    constants_info_[33].shape = {768, 768};
    constants_info_[33].stride = {768, 1};
    constants_info_[34].name = "L__self___layers_1_attention_wk_weight";
    constants_info_[34].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[34].offset = 0;
    constants_info_[34].data_size = 2359296;
    constants_info_[34].shape = {768, 768};
    constants_info_[34].stride = {768, 1};
    constants_info_[35].name = "L__self___layers_1_attention_wv_weight";
    constants_info_[35].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[35].offset = 0;
    constants_info_[35].data_size = 2359296;
    constants_info_[35].shape = {768, 768};
    constants_info_[35].stride = {768, 1};
    constants_info_[36].name = "L__self___layers_1_attention_wo_weight";
    constants_info_[36].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[36].offset = 0;
    constants_info_[36].data_size = 2359296;
    constants_info_[36].shape = {768, 768};
    constants_info_[36].stride = {768, 1};
    constants_info_[37].name = "L__self___layers_1_feed_forward_w1_weight";
    constants_info_[37].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[37].offset = 0;
    constants_info_[37].data_size = 6291456;
    constants_info_[37].shape = {2048, 768};
    constants_info_[37].stride = {768, 1};
    constants_info_[38].name = "L__self___layers_1_feed_forward_w3_weight";
    constants_info_[38].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[38].offset = 0;
    constants_info_[38].data_size = 6291456;
    constants_info_[38].shape = {2048, 768};
    constants_info_[38].stride = {768, 1};
    constants_info_[39].name = "L__self___layers_1_feed_forward_w2_weight";
    constants_info_[39].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[39].offset = 0;
    constants_info_[39].data_size = 6291456;
    constants_info_[39].shape = {768, 2048};
    constants_info_[39].stride = {2048, 1};
    constants_info_[40].name = "L__self___layers_2_attention_wq_weight";
    constants_info_[40].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[40].offset = 0;
    constants_info_[40].data_size = 2359296;
    constants_info_[40].shape = {768, 768};
    constants_info_[40].stride = {768, 1};
    constants_info_[41].name = "L__self___layers_2_attention_wk_weight";
    constants_info_[41].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[41].offset = 0;
    constants_info_[41].data_size = 2359296;
    constants_info_[41].shape = {768, 768};
    constants_info_[41].stride = {768, 1};
    constants_info_[42].name = "L__self___layers_2_attention_wv_weight";
    constants_info_[42].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[42].offset = 0;
    constants_info_[42].data_size = 2359296;
    constants_info_[42].shape = {768, 768};
    constants_info_[42].stride = {768, 1};
    constants_info_[43].name = "L__self___layers_2_attention_wo_weight";
    constants_info_[43].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[43].offset = 0;
    constants_info_[43].data_size = 2359296;
    constants_info_[43].shape = {768, 768};
    constants_info_[43].stride = {768, 1};
    constants_info_[44].name = "L__self___layers_2_feed_forward_w1_weight";
    constants_info_[44].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[44].offset = 0;
    constants_info_[44].data_size = 6291456;
    constants_info_[44].shape = {2048, 768};
    constants_info_[44].stride = {768, 1};
    constants_info_[45].name = "L__self___layers_2_feed_forward_w3_weight";
    constants_info_[45].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[45].offset = 0;
    constants_info_[45].data_size = 6291456;
    constants_info_[45].shape = {2048, 768};
    constants_info_[45].stride = {768, 1};
    constants_info_[46].name = "L__self___layers_2_feed_forward_w2_weight";
    constants_info_[46].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[46].offset = 0;
    constants_info_[46].data_size = 6291456;
    constants_info_[46].shape = {768, 2048};
    constants_info_[46].stride = {2048, 1};
    constants_info_[47].name = "L__self___layers_3_attention_wq_weight";
    constants_info_[47].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[47].offset = 0;
    constants_info_[47].data_size = 2359296;
    constants_info_[47].shape = {768, 768};
    constants_info_[47].stride = {768, 1};
    constants_info_[48].name = "L__self___layers_3_attention_wk_weight";
    constants_info_[48].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[48].offset = 0;
    constants_info_[48].data_size = 2359296;
    constants_info_[48].shape = {768, 768};
    constants_info_[48].stride = {768, 1};
    constants_info_[49].name = "L__self___layers_3_attention_wv_weight";
    constants_info_[49].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[49].offset = 0;
    constants_info_[49].data_size = 2359296;
    constants_info_[49].shape = {768, 768};
    constants_info_[49].stride = {768, 1};
    constants_info_[50].name = "L__self___layers_3_attention_wo_weight";
    constants_info_[50].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[50].offset = 0;
    constants_info_[50].data_size = 2359296;
    constants_info_[50].shape = {768, 768};
    constants_info_[50].stride = {768, 1};
    constants_info_[51].name = "L__self___layers_3_feed_forward_w1_weight";
    constants_info_[51].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[51].offset = 0;
    constants_info_[51].data_size = 6291456;
    constants_info_[51].shape = {2048, 768};
    constants_info_[51].stride = {768, 1};
    constants_info_[52].name = "L__self___layers_3_feed_forward_w3_weight";
    constants_info_[52].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[52].offset = 0;
    constants_info_[52].data_size = 6291456;
    constants_info_[52].shape = {2048, 768};
    constants_info_[52].stride = {768, 1};
    constants_info_[53].name = "L__self___layers_3_feed_forward_w2_weight";
    constants_info_[53].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[53].offset = 0;
    constants_info_[53].data_size = 6291456;
    constants_info_[53].shape = {768, 2048};
    constants_info_[53].stride = {2048, 1};
    constants_info_[54].name = "L__self___layers_4_attention_wq_weight";
    constants_info_[54].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[54].offset = 0;
    constants_info_[54].data_size = 2359296;
    constants_info_[54].shape = {768, 768};
    constants_info_[54].stride = {768, 1};
    constants_info_[55].name = "L__self___layers_4_attention_wk_weight";
    constants_info_[55].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[55].offset = 0;
    constants_info_[55].data_size = 2359296;
    constants_info_[55].shape = {768, 768};
    constants_info_[55].stride = {768, 1};
    constants_info_[56].name = "L__self___layers_4_attention_wv_weight";
    constants_info_[56].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[56].offset = 0;
    constants_info_[56].data_size = 2359296;
    constants_info_[56].shape = {768, 768};
    constants_info_[56].stride = {768, 1};
    constants_info_[57].name = "L__self___layers_4_attention_wo_weight";
    constants_info_[57].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[57].offset = 0;
    constants_info_[57].data_size = 2359296;
    constants_info_[57].shape = {768, 768};
    constants_info_[57].stride = {768, 1};
    constants_info_[58].name = "L__self___layers_4_feed_forward_w1_weight";
    constants_info_[58].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[58].offset = 0;
    constants_info_[58].data_size = 6291456;
    constants_info_[58].shape = {2048, 768};
    constants_info_[58].stride = {768, 1};
    constants_info_[59].name = "L__self___layers_4_feed_forward_w3_weight";
    constants_info_[59].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[59].offset = 0;
    constants_info_[59].data_size = 6291456;
    constants_info_[59].shape = {2048, 768};
    constants_info_[59].stride = {768, 1};
    constants_info_[60].name = "L__self___layers_4_feed_forward_w2_weight";
    constants_info_[60].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[60].offset = 0;
    constants_info_[60].data_size = 6291456;
    constants_info_[60].shape = {768, 2048};
    constants_info_[60].stride = {2048, 1};
    constants_info_[61].name = "L__self___layers_5_attention_wq_weight";
    constants_info_[61].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[61].offset = 0;
    constants_info_[61].data_size = 2359296;
    constants_info_[61].shape = {768, 768};
    constants_info_[61].stride = {768, 1};
    constants_info_[62].name = "L__self___layers_5_attention_wk_weight";
    constants_info_[62].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[62].offset = 0;
    constants_info_[62].data_size = 2359296;
    constants_info_[62].shape = {768, 768};
    constants_info_[62].stride = {768, 1};
    constants_info_[63].name = "L__self___layers_5_attention_wv_weight";
    constants_info_[63].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[63].offset = 0;
    constants_info_[63].data_size = 2359296;
    constants_info_[63].shape = {768, 768};
    constants_info_[63].stride = {768, 1};
    constants_info_[64].name = "L__self___layers_5_attention_wo_weight";
    constants_info_[64].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[64].offset = 0;
    constants_info_[64].data_size = 2359296;
    constants_info_[64].shape = {768, 768};
    constants_info_[64].stride = {768, 1};
    constants_info_[65].name = "L__self___layers_5_feed_forward_w1_weight";
    constants_info_[65].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[65].offset = 0;
    constants_info_[65].data_size = 6291456;
    constants_info_[65].shape = {2048, 768};
    constants_info_[65].stride = {768, 1};
    constants_info_[66].name = "L__self___layers_5_feed_forward_w3_weight";
    constants_info_[66].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[66].offset = 0;
    constants_info_[66].data_size = 6291456;
    constants_info_[66].shape = {2048, 768};
    constants_info_[66].stride = {768, 1};
    constants_info_[67].name = "L__self___layers_5_feed_forward_w2_weight";
    constants_info_[67].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[67].offset = 0;
    constants_info_[67].data_size = 6291456;
    constants_info_[67].shape = {768, 2048};
    constants_info_[67].stride = {2048, 1};
    constants_info_[68].name = "L__self___layers_6_attention_wq_weight";
    constants_info_[68].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[68].offset = 0;
    constants_info_[68].data_size = 2359296;
    constants_info_[68].shape = {768, 768};
    constants_info_[68].stride = {768, 1};
    constants_info_[69].name = "L__self___layers_6_attention_wk_weight";
    constants_info_[69].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[69].offset = 0;
    constants_info_[69].data_size = 2359296;
    constants_info_[69].shape = {768, 768};
    constants_info_[69].stride = {768, 1};
    constants_info_[70].name = "L__self___layers_6_attention_wv_weight";
    constants_info_[70].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[70].offset = 0;
    constants_info_[70].data_size = 2359296;
    constants_info_[70].shape = {768, 768};
    constants_info_[70].stride = {768, 1};
    constants_info_[71].name = "L__self___layers_6_attention_wo_weight";
    constants_info_[71].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[71].offset = 0;
    constants_info_[71].data_size = 2359296;
    constants_info_[71].shape = {768, 768};
    constants_info_[71].stride = {768, 1};
    constants_info_[72].name = "L__self___layers_6_feed_forward_w1_weight";
    constants_info_[72].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[72].offset = 0;
    constants_info_[72].data_size = 6291456;
    constants_info_[72].shape = {2048, 768};
    constants_info_[72].stride = {768, 1};
    constants_info_[73].name = "L__self___layers_6_feed_forward_w3_weight";
    constants_info_[73].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[73].offset = 0;
    constants_info_[73].data_size = 6291456;
    constants_info_[73].shape = {2048, 768};
    constants_info_[73].stride = {768, 1};
    constants_info_[74].name = "L__self___layers_6_feed_forward_w2_weight";
    constants_info_[74].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[74].offset = 0;
    constants_info_[74].data_size = 6291456;
    constants_info_[74].shape = {768, 2048};
    constants_info_[74].stride = {2048, 1};
    constants_info_[75].name = "L__self___layers_7_attention_wq_weight";
    constants_info_[75].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[75].offset = 0;
    constants_info_[75].data_size = 2359296;
    constants_info_[75].shape = {768, 768};
    constants_info_[75].stride = {768, 1};
    constants_info_[76].name = "L__self___layers_7_attention_wk_weight";
    constants_info_[76].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[76].offset = 0;
    constants_info_[76].data_size = 2359296;
    constants_info_[76].shape = {768, 768};
    constants_info_[76].stride = {768, 1};
    constants_info_[77].name = "L__self___layers_7_attention_wv_weight";
    constants_info_[77].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[77].offset = 0;
    constants_info_[77].data_size = 2359296;
    constants_info_[77].shape = {768, 768};
    constants_info_[77].stride = {768, 1};
    constants_info_[78].name = "L__self___layers_7_attention_wo_weight";
    constants_info_[78].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[78].offset = 0;
    constants_info_[78].data_size = 2359296;
    constants_info_[78].shape = {768, 768};
    constants_info_[78].stride = {768, 1};
    constants_info_[79].name = "L__self___layers_7_feed_forward_w1_weight";
    constants_info_[79].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[79].offset = 0;
    constants_info_[79].data_size = 6291456;
    constants_info_[79].shape = {2048, 768};
    constants_info_[79].stride = {768, 1};
    constants_info_[80].name = "L__self___layers_7_feed_forward_w3_weight";
    constants_info_[80].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[80].offset = 0;
    constants_info_[80].data_size = 6291456;
    constants_info_[80].shape = {2048, 768};
    constants_info_[80].stride = {768, 1};
    constants_info_[81].name = "L__self___layers_7_feed_forward_w2_weight";
    constants_info_[81].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[81].offset = 0;
    constants_info_[81].data_size = 6291456;
    constants_info_[81].shape = {768, 2048};
    constants_info_[81].stride = {2048, 1};
    constants_info_[82].name = "L__self___layers_8_attention_wq_weight";
    constants_info_[82].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[82].offset = 0;
    constants_info_[82].data_size = 2359296;
    constants_info_[82].shape = {768, 768};
    constants_info_[82].stride = {768, 1};
    constants_info_[83].name = "L__self___layers_8_attention_wk_weight";
    constants_info_[83].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[83].offset = 0;
    constants_info_[83].data_size = 2359296;
    constants_info_[83].shape = {768, 768};
    constants_info_[83].stride = {768, 1};
    constants_info_[84].name = "L__self___layers_8_attention_wv_weight";
    constants_info_[84].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[84].offset = 0;
    constants_info_[84].data_size = 2359296;
    constants_info_[84].shape = {768, 768};
    constants_info_[84].stride = {768, 1};
    constants_info_[85].name = "L__self___layers_8_attention_wo_weight";
    constants_info_[85].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[85].offset = 0;
    constants_info_[85].data_size = 2359296;
    constants_info_[85].shape = {768, 768};
    constants_info_[85].stride = {768, 1};
    constants_info_[86].name = "L__self___layers_8_feed_forward_w1_weight";
    constants_info_[86].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[86].offset = 0;
    constants_info_[86].data_size = 6291456;
    constants_info_[86].shape = {2048, 768};
    constants_info_[86].stride = {768, 1};
    constants_info_[87].name = "L__self___layers_8_feed_forward_w3_weight";
    constants_info_[87].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[87].offset = 0;
    constants_info_[87].data_size = 6291456;
    constants_info_[87].shape = {2048, 768};
    constants_info_[87].stride = {768, 1};
    constants_info_[88].name = "L__self___layers_8_feed_forward_w2_weight";
    constants_info_[88].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[88].offset = 0;
    constants_info_[88].data_size = 6291456;
    constants_info_[88].shape = {768, 2048};
    constants_info_[88].stride = {2048, 1};
    constants_info_[89].name = "L__self___layers_9_attention_wq_weight";
    constants_info_[89].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[89].offset = 0;
    constants_info_[89].data_size = 2359296;
    constants_info_[89].shape = {768, 768};
    constants_info_[89].stride = {768, 1};
    constants_info_[90].name = "L__self___layers_9_attention_wk_weight";
    constants_info_[90].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[90].offset = 0;
    constants_info_[90].data_size = 2359296;
    constants_info_[90].shape = {768, 768};
    constants_info_[90].stride = {768, 1};
    constants_info_[91].name = "L__self___layers_9_attention_wv_weight";
    constants_info_[91].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[91].offset = 0;
    constants_info_[91].data_size = 2359296;
    constants_info_[91].shape = {768, 768};
    constants_info_[91].stride = {768, 1};
    constants_info_[92].name = "L__self___layers_9_attention_wo_weight";
    constants_info_[92].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[92].offset = 0;
    constants_info_[92].data_size = 2359296;
    constants_info_[92].shape = {768, 768};
    constants_info_[92].stride = {768, 1};
    constants_info_[93].name = "L__self___layers_9_feed_forward_w1_weight";
    constants_info_[93].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[93].offset = 0;
    constants_info_[93].data_size = 6291456;
    constants_info_[93].shape = {2048, 768};
    constants_info_[93].stride = {768, 1};
    constants_info_[94].name = "L__self___layers_9_feed_forward_w3_weight";
    constants_info_[94].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[94].offset = 0;
    constants_info_[94].data_size = 6291456;
    constants_info_[94].shape = {2048, 768};
    constants_info_[94].stride = {768, 1};
    constants_info_[95].name = "L__self___layers_9_feed_forward_w2_weight";
    constants_info_[95].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[95].offset = 0;
    constants_info_[95].data_size = 6291456;
    constants_info_[95].shape = {768, 2048};
    constants_info_[95].stride = {2048, 1};
    constants_info_[96].name = "L__self___layers_10_attention_wq_weight";
    constants_info_[96].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[96].offset = 0;
    constants_info_[96].data_size = 2359296;
    constants_info_[96].shape = {768, 768};
    constants_info_[96].stride = {768, 1};
    constants_info_[97].name = "L__self___layers_10_attention_wk_weight";
    constants_info_[97].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[97].offset = 0;
    constants_info_[97].data_size = 2359296;
    constants_info_[97].shape = {768, 768};
    constants_info_[97].stride = {768, 1};
    constants_info_[98].name = "L__self___layers_10_attention_wv_weight";
    constants_info_[98].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[98].offset = 0;
    constants_info_[98].data_size = 2359296;
    constants_info_[98].shape = {768, 768};
    constants_info_[98].stride = {768, 1};
    constants_info_[99].name = "L__self___layers_10_attention_wo_weight";
    constants_info_[99].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[99].offset = 0;
    constants_info_[99].data_size = 2359296;
    constants_info_[99].shape = {768, 768};
    constants_info_[99].stride = {768, 1};
    constants_info_[100].name = "L__self___layers_10_feed_forward_w1_weight";
    constants_info_[100].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[100].offset = 0;
    constants_info_[100].data_size = 6291456;
    constants_info_[100].shape = {2048, 768};
    constants_info_[100].stride = {768, 1};
    constants_info_[101].name = "L__self___layers_10_feed_forward_w3_weight";
    constants_info_[101].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[101].offset = 0;
    constants_info_[101].data_size = 6291456;
    constants_info_[101].shape = {2048, 768};
    constants_info_[101].stride = {768, 1};
    constants_info_[102].name = "L__self___layers_10_feed_forward_w2_weight";
    constants_info_[102].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[102].offset = 0;
    constants_info_[102].data_size = 6291456;
    constants_info_[102].shape = {768, 2048};
    constants_info_[102].stride = {2048, 1};
    constants_info_[103].name = "L__self___layers_11_attention_wq_weight";
    constants_info_[103].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[103].offset = 0;
    constants_info_[103].data_size = 2359296;
    constants_info_[103].shape = {768, 768};
    constants_info_[103].stride = {768, 1};
    constants_info_[104].name = "L__self___layers_11_attention_wk_weight";
    constants_info_[104].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[104].offset = 0;
    constants_info_[104].data_size = 2359296;
    constants_info_[104].shape = {768, 768};
    constants_info_[104].stride = {768, 1};
    constants_info_[105].name = "L__self___layers_11_attention_wv_weight";
    constants_info_[105].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[105].offset = 0;
    constants_info_[105].data_size = 2359296;
    constants_info_[105].shape = {768, 768};
    constants_info_[105].stride = {768, 1};
    constants_info_[106].name = "L__self___layers_11_attention_wo_weight";
    constants_info_[106].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[106].offset = 0;
    constants_info_[106].data_size = 2359296;
    constants_info_[106].shape = {768, 768};
    constants_info_[106].stride = {768, 1};
    constants_info_[107].name = "L__self___layers_11_feed_forward_w1_weight";
    constants_info_[107].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[107].offset = 0;
    constants_info_[107].data_size = 6291456;
    constants_info_[107].shape = {2048, 768};
    constants_info_[107].stride = {768, 1};
    constants_info_[108].name = "L__self___layers_11_feed_forward_w3_weight";
    constants_info_[108].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[108].offset = 0;
    constants_info_[108].data_size = 6291456;
    constants_info_[108].shape = {2048, 768};
    constants_info_[108].stride = {768, 1};
    constants_info_[109].name = "L__self___layers_11_feed_forward_w2_weight";
    constants_info_[109].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[109].offset = 0;
    constants_info_[109].data_size = 6291456;
    constants_info_[109].shape = {768, 2048};
    constants_info_[109].stride = {2048, 1};
    constants_info_[110].name = "L__self___freqs_cos";
    constants_info_[110].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[110].offset = 0;
    constants_info_[110].data_size = 131072;
    constants_info_[110].shape = {1024, 32};
    constants_info_[110].stride = {32, 1};
    constants_info_[111].name = "L__self___freqs_sin";
    constants_info_[111].dtype = static_cast<int32_t>(at::kFloat);
    constants_info_[111].offset = 0;
    constants_info_[111].data_size = 131072;
    constants_info_[111].shape = {1024, 32};
    constants_info_[111].stride = {32, 1};
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = "[1, {\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": null, \"context\": null, \"children_spec\": []}]}, {\"type\": \"builtins.dict\", \"context\": \"[]\", \"children_spec\": []}]}]";
    out_spec_ = "[1, {\"type\": null, \"context\": null, \"children_spec\": []}]";
    outputs_info_[0].name = "output0";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {

    auto inputs = alloc_tensors_by_stealing_from_handles(input_handles, num_inputs());
    auto arg113_1 = std::move(inputs[0]);
    auto L__self___layers_0_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(0));
    auto L__self___layers_0_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(1));
    auto L__self___layers_1_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(2));
    auto L__self___layers_1_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(3));
    auto L__self___layers_2_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(4));
    auto L__self___layers_2_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(5));
    auto L__self___layers_3_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(6));
    auto L__self___layers_3_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(7));
    auto L__self___layers_4_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(8));
    auto L__self___layers_4_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(9));
    auto L__self___layers_5_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(10));
    auto L__self___layers_5_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(11));
    auto L__self___layers_6_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(12));
    auto L__self___layers_6_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(13));
    auto L__self___layers_7_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(14));
    auto L__self___layers_7_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(15));
    auto L__self___layers_8_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(16));
    auto L__self___layers_8_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(17));
    auto L__self___layers_9_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(18));
    auto L__self___layers_9_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(19));
    auto L__self___layers_10_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(20));
    auto L__self___layers_10_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(21));
    auto L__self___layers_11_attention_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(22));
    auto L__self___layers_11_ffn_norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(23));
    auto L__self___norm_weight = *tensor_handle_to_tensor_pointer(constants_->at(24));
    auto L__self___tok_embeddings_weight = *tensor_handle_to_tensor_pointer(constants_->at(25));
    auto L__self___layers_0_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(26));
    auto L__self___layers_0_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(27));
    auto L__self___layers_0_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(28));
    auto L__self___layers_0_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(29));
    auto L__self___layers_0_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(30));
    auto L__self___layers_0_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(31));
    auto L__self___layers_0_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(32));
    auto L__self___layers_1_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(33));
    auto L__self___layers_1_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(34));
    auto L__self___layers_1_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(35));
    auto L__self___layers_1_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(36));
    auto L__self___layers_1_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(37));
    auto L__self___layers_1_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(38));
    auto L__self___layers_1_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(39));
    auto L__self___layers_2_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(40));
    auto L__self___layers_2_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(41));
    auto L__self___layers_2_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(42));
    auto L__self___layers_2_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(43));
    auto L__self___layers_2_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(44));
    auto L__self___layers_2_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(45));
    auto L__self___layers_2_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(46));
    auto L__self___layers_3_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(47));
    auto L__self___layers_3_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(48));
    auto L__self___layers_3_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(49));
    auto L__self___layers_3_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(50));
    auto L__self___layers_3_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(51));
    auto L__self___layers_3_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(52));
    auto L__self___layers_3_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(53));
    auto L__self___layers_4_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(54));
    auto L__self___layers_4_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(55));
    auto L__self___layers_4_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(56));
    auto L__self___layers_4_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(57));
    auto L__self___layers_4_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(58));
    auto L__self___layers_4_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(59));
    auto L__self___layers_4_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(60));
    auto L__self___layers_5_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(61));
    auto L__self___layers_5_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(62));
    auto L__self___layers_5_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(63));
    auto L__self___layers_5_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(64));
    auto L__self___layers_5_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(65));
    auto L__self___layers_5_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(66));
    auto L__self___layers_5_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(67));
    auto L__self___layers_6_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(68));
    auto L__self___layers_6_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(69));
    auto L__self___layers_6_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(70));
    auto L__self___layers_6_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(71));
    auto L__self___layers_6_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(72));
    auto L__self___layers_6_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(73));
    auto L__self___layers_6_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(74));
    auto L__self___layers_7_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(75));
    auto L__self___layers_7_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(76));
    auto L__self___layers_7_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(77));
    auto L__self___layers_7_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(78));
    auto L__self___layers_7_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(79));
    auto L__self___layers_7_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(80));
    auto L__self___layers_7_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(81));
    auto L__self___layers_8_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(82));
    auto L__self___layers_8_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(83));
    auto L__self___layers_8_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(84));
    auto L__self___layers_8_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(85));
    auto L__self___layers_8_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(86));
    auto L__self___layers_8_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(87));
    auto L__self___layers_8_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(88));
    auto L__self___layers_9_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(89));
    auto L__self___layers_9_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(90));
    auto L__self___layers_9_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(91));
    auto L__self___layers_9_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(92));
    auto L__self___layers_9_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(93));
    auto L__self___layers_9_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(94));
    auto L__self___layers_9_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(95));
    auto L__self___layers_10_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(96));
    auto L__self___layers_10_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(97));
    auto L__self___layers_10_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(98));
    auto L__self___layers_10_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(99));
    auto L__self___layers_10_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(100));
    auto L__self___layers_10_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(101));
    auto L__self___layers_10_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(102));
    auto L__self___layers_11_attention_wq_weight = *tensor_handle_to_tensor_pointer(constants_->at(103));
    auto L__self___layers_11_attention_wk_weight = *tensor_handle_to_tensor_pointer(constants_->at(104));
    auto L__self___layers_11_attention_wv_weight = *tensor_handle_to_tensor_pointer(constants_->at(105));
    auto L__self___layers_11_attention_wo_weight = *tensor_handle_to_tensor_pointer(constants_->at(106));
    auto L__self___layers_11_feed_forward_w1_weight = *tensor_handle_to_tensor_pointer(constants_->at(107));
    auto L__self___layers_11_feed_forward_w3_weight = *tensor_handle_to_tensor_pointer(constants_->at(108));
    auto L__self___layers_11_feed_forward_w2_weight = *tensor_handle_to_tensor_pointer(constants_->at(109));
    auto L__self___freqs_cos = *tensor_handle_to_tensor_pointer(constants_->at(110));
    auto L__self___freqs_sin = *tensor_handle_to_tensor_pointer(constants_->at(111));
    auto arg113_1_size = arg113_1.sizes();
    auto s0 = arg113_1_size[1];
    inputs.clear();
    auto& kernels = *dynamic_cast<AOTInductorModelKernels*>(this->kernels_.get());
    auto buf0 = at::empty_strided({1L, s0, 1L}, {s0, 1L, s0}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    auto buf1 = at::empty_strided({1L, s0, 768L}, {768L*s0, 768L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_0((long*)(arg113_1.data_ptr()), (float*)(L__self___tok_embeddings_weight.data_ptr()), (float*)(L__self___layers_0_attention_norm_weight.data_ptr()), (float*)(buf0.data_ptr()), (float*)(buf1.data_ptr()), s0);
    auto buf2 = at::empty_strided({s0, 768L}, {768L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    // Source Nodes: [xq], Original ATen: [aten.mm]
    at::mm_out(buf2, reinterpret_tensor(buf1, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_0_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf3 = at::empty_strided({s0, 768L}, {768L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    // Source Nodes: [xk], Original ATen: [aten.mm]
    at::mm_out(buf3, reinterpret_tensor(buf1, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_0_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf4 = at::empty_strided({s0, 768L}, {768L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    // Source Nodes: [xv], Original ATen: [aten.mm]
    at::mm_out(buf4, reinterpret_tensor(buf1, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_0_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf5 = reinterpret_tensor(buf1, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf6 = at::empty_strided({1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    cpp_fused_stack_1((float*)(buf2.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf3.data_ptr()), (float*)(buf5.data_ptr()), (float*)(buf6.data_ptr()), s0);
    // Source Nodes: [output_1], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf7 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf5, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf6, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf4, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf8 = std::get<0>(buf7);

    auto buf15 = reinterpret_tensor(buf6, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_3], Original ATen: [aten.mm]
    at::mm_out(buf15, reinterpret_tensor(buf8, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_0_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf16 = buf0; ;  // reuse
    auto buf17 = reinterpret_tensor(buf8, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_2((long*)(arg113_1.data_ptr()), (float*)(L__self___tok_embeddings_weight.data_ptr()), (float*)(buf15.data_ptr()), (float*)(L__self___layers_0_ffn_norm_weight.data_ptr()), (float*)(buf16.data_ptr()), (float*)(buf17.data_ptr()), s0);
    auto buf18 = at::empty_strided({s0, 2048L}, {2048L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    // Source Nodes: [l__self___layers_0_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf18, reinterpret_tensor(buf17, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_0_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf19 = at::empty_strided({s0, 2048L}, {2048L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    // Source Nodes: [l__self___layers_0_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf19, reinterpret_tensor(buf17, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_0_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf20 = reinterpret_tensor(buf18, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_3((float*)(buf20.data_ptr()), (float*)(buf19.data_ptr()), s0);
    auto buf21 = reinterpret_tensor(buf17, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_0_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf21, reinterpret_tensor(buf20, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_0_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf22 = buf16; ;  // reuse
    auto buf23 = reinterpret_tensor(buf5, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_4((long*)(arg113_1.data_ptr()), (float*)(L__self___tok_embeddings_weight.data_ptr()), (float*)(buf15.data_ptr()), (float*)(buf21.data_ptr()), (float*)(L__self___layers_1_attention_norm_weight.data_ptr()), (float*)(buf22.data_ptr()), (float*)(buf23.data_ptr()), s0);
    auto buf24 = buf4; ;  // reuse
    // Source Nodes: [xq_4], Original ATen: [aten.mm]
    at::mm_out(buf24, reinterpret_tensor(buf23, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_1_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf25 = buf3; ;  // reuse
    // Source Nodes: [xk_5], Original ATen: [aten.mm]
    at::mm_out(buf25, reinterpret_tensor(buf23, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_1_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf26 = buf2; ;  // reuse
    // Source Nodes: [xv_4], Original ATen: [aten.mm]
    at::mm_out(buf26, reinterpret_tensor(buf23, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_1_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf27 = reinterpret_tensor(buf23, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf28 = at::empty_strided({1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    cpp_fused_stack_5((float*)(buf24.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf25.data_ptr()), (float*)(buf27.data_ptr()), (float*)(buf28.data_ptr()), s0);
    // Source Nodes: [output_7], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf29 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf27, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf28, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf26, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf30 = std::get<0>(buf29);

    auto buf37 = reinterpret_tensor(buf28, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_9], Original ATen: [aten.mm]
    at::mm_out(buf37, reinterpret_tensor(buf30, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_1_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf38 = reinterpret_tensor(buf15, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    auto buf39 = buf22; ;  // reuse
    auto buf40 = reinterpret_tensor(buf30, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_embedding_mean_mul_pow_rsqrt_6((float*)(buf38.data_ptr()), (long*)(arg113_1.data_ptr()), (float*)(L__self___tok_embeddings_weight.data_ptr()), (float*)(buf21.data_ptr()), (float*)(buf37.data_ptr()), (float*)(L__self___layers_1_ffn_norm_weight.data_ptr()), (float*)(buf39.data_ptr()), (float*)(buf40.data_ptr()), s0);

    auto buf41 = reinterpret_tensor(buf20, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_1_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf41, reinterpret_tensor(buf40, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_1_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf42 = buf19; ;  // reuse
    // Source Nodes: [l__self___layers_1_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf42, reinterpret_tensor(buf40, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_1_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf43 = reinterpret_tensor(buf41, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_7((float*)(buf43.data_ptr()), (float*)(buf42.data_ptr()), s0);
    auto buf44 = reinterpret_tensor(buf40, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_1_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf44, reinterpret_tensor(buf43, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_1_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf45 = buf39; ;  // reuse
    auto buf46 = reinterpret_tensor(buf37, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_8((float*)(buf38.data_ptr()), (float*)(buf44.data_ptr()), (float*)(L__self___layers_2_attention_norm_weight.data_ptr()), (float*)(buf45.data_ptr()), (float*)(buf46.data_ptr()), s0);
    auto buf47 = buf21; ;  // reuse
    // Source Nodes: [xq_8], Original ATen: [aten.mm]
    at::mm_out(buf47, reinterpret_tensor(buf46, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_2_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf48 = reinterpret_tensor(buf27, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [xk_10], Original ATen: [aten.mm]
    at::mm_out(buf48, reinterpret_tensor(buf46, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_2_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf49 = buf26; ;  // reuse
    // Source Nodes: [xv_8], Original ATen: [aten.mm]
    at::mm_out(buf49, reinterpret_tensor(buf46, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_2_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf50 = reinterpret_tensor(buf46, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf51 = reinterpret_tensor(buf25, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    cpp_fused_stack_9((float*)(buf47.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf48.data_ptr()), (float*)(buf50.data_ptr()), (float*)(buf51.data_ptr()), s0);
    // Source Nodes: [output_13], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf52 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf50, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf51, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf49, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf53 = std::get<0>(buf52);

    auto buf60 = reinterpret_tensor(buf51, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_15], Original ATen: [aten.mm]
    at::mm_out(buf60, reinterpret_tensor(buf53, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_2_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf61 = buf45; ;  // reuse
    auto buf62 = reinterpret_tensor(buf53, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_10((float*)(buf38.data_ptr()), (float*)(buf44.data_ptr()), (float*)(buf60.data_ptr()), (float*)(L__self___layers_2_ffn_norm_weight.data_ptr()), (float*)(buf61.data_ptr()), (float*)(buf62.data_ptr()), s0);
    auto buf63 = reinterpret_tensor(buf43, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_2_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf63, reinterpret_tensor(buf62, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_2_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf64 = buf42; ;  // reuse
    // Source Nodes: [l__self___layers_2_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf64, reinterpret_tensor(buf62, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_2_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf65 = reinterpret_tensor(buf63, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_11((float*)(buf65.data_ptr()), (float*)(buf64.data_ptr()), s0);
    auto buf66 = reinterpret_tensor(buf62, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_2_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf66, reinterpret_tensor(buf65, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_2_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf67 = buf61; ;  // reuse
    auto buf68 = reinterpret_tensor(buf50, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_12((float*)(buf38.data_ptr()), (float*)(buf44.data_ptr()), (float*)(buf60.data_ptr()), (float*)(buf66.data_ptr()), (float*)(L__self___layers_3_attention_norm_weight.data_ptr()), (float*)(buf67.data_ptr()), (float*)(buf68.data_ptr()), s0);
    auto buf69 = buf49; ;  // reuse
    // Source Nodes: [xq_12], Original ATen: [aten.mm]
    at::mm_out(buf69, reinterpret_tensor(buf68, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_3_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf70 = buf48; ;  // reuse
    // Source Nodes: [xk_15], Original ATen: [aten.mm]
    at::mm_out(buf70, reinterpret_tensor(buf68, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_3_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf71 = buf47; ;  // reuse
    // Source Nodes: [xv_12], Original ATen: [aten.mm]
    at::mm_out(buf71, reinterpret_tensor(buf68, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_3_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf72 = reinterpret_tensor(buf68, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf73 = reinterpret_tensor(buf24, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    cpp_fused_stack_13((float*)(buf69.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf70.data_ptr()), (float*)(buf72.data_ptr()), (float*)(buf73.data_ptr()), s0);


    // Source Nodes: [output_19], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf74 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf72, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf73, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf71, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf75 = std::get<0>(buf74);

    auto buf82 = reinterpret_tensor(buf73, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_21], Original ATen: [aten.mm]
    at::mm_out(buf82, reinterpret_tensor(buf75, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_3_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf83 = buf38; ;  // reuse
    auto buf84 = buf67; ;  // reuse
    auto buf85 = reinterpret_tensor(buf75, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_14((float*)(buf83.data_ptr()), (float*)(buf44.data_ptr()), (float*)(buf60.data_ptr()), (float*)(buf66.data_ptr()), (float*)(buf82.data_ptr()), (float*)(L__self___layers_3_ffn_norm_weight.data_ptr()), (float*)(buf84.data_ptr()), (float*)(buf85.data_ptr()), s0);
    auto buf86 = reinterpret_tensor(buf65, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_3_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf86, reinterpret_tensor(buf85, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_3_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf87 = buf64; ;  // reuse
    // Source Nodes: [l__self___layers_3_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf87, reinterpret_tensor(buf85, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_3_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf88 = reinterpret_tensor(buf86, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_15((float*)(buf88.data_ptr()), (float*)(buf87.data_ptr()), s0);
    auto buf89 = reinterpret_tensor(buf85, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_3_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf89, reinterpret_tensor(buf88, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_3_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf90 = buf84; ;  // reuse
    auto buf91 = reinterpret_tensor(buf82, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_16((float*)(buf83.data_ptr()), (float*)(buf89.data_ptr()), (float*)(L__self___layers_4_attention_norm_weight.data_ptr()), (float*)(buf90.data_ptr()), (float*)(buf91.data_ptr()), s0);
    auto buf92 = buf66; ;  // reuse
    // Source Nodes: [xq_16], Original ATen: [aten.mm]
    at::mm_out(buf92, reinterpret_tensor(buf91, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_4_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf93 = buf60; ;  // reuse
    // Source Nodes: [xk_20], Original ATen: [aten.mm]
    at::mm_out(buf93, reinterpret_tensor(buf91, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_4_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf94 = buf44; ;  // reuse
    // Source Nodes: [xv_16], Original ATen: [aten.mm]
    at::mm_out(buf94, reinterpret_tensor(buf91, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_4_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf95 = reinterpret_tensor(buf91, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf96 = buf72; ;  // reuse
    cpp_fused_stack_17((float*)(buf92.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf93.data_ptr()), (float*)(buf95.data_ptr()), (float*)(buf96.data_ptr()), s0);
    // Source Nodes: [output_25], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf97 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf95, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf96, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf94, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf98 = std::get<0>(buf97);

    auto buf105 = reinterpret_tensor(buf96, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_27], Original ATen: [aten.mm]
    at::mm_out(buf105, reinterpret_tensor(buf98, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_4_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf106 = buf90; ;  // reuse
    auto buf107 = reinterpret_tensor(buf98, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_18((float*)(buf83.data_ptr()), (float*)(buf89.data_ptr()), (float*)(buf105.data_ptr()), (float*)(L__self___layers_4_ffn_norm_weight.data_ptr()), (float*)(buf106.data_ptr()), (float*)(buf107.data_ptr()), s0);
    auto buf108 = reinterpret_tensor(buf88, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_4_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf108, reinterpret_tensor(buf107, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_4_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf109 = buf87; ;  // reuse
    // Source Nodes: [l__self___layers_4_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf109, reinterpret_tensor(buf107, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_4_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf110 = reinterpret_tensor(buf108, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_19((float*)(buf110.data_ptr()), (float*)(buf109.data_ptr()), s0);
    auto buf111 = reinterpret_tensor(buf107, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_4_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf111, reinterpret_tensor(buf110, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_4_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf112 = buf106; ;  // reuse
    auto buf113 = reinterpret_tensor(buf95, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_20((float*)(buf83.data_ptr()), (float*)(buf89.data_ptr()), (float*)(buf105.data_ptr()), (float*)(buf111.data_ptr()), (float*)(L__self___layers_5_attention_norm_weight.data_ptr()), (float*)(buf112.data_ptr()), (float*)(buf113.data_ptr()), s0);
    auto buf114 = buf94; ;  // reuse
    // Source Nodes: [xq_20], Original ATen: [aten.mm]
    at::mm_out(buf114, reinterpret_tensor(buf113, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_5_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf115 = buf93; ;  // reuse
    // Source Nodes: [xk_25], Original ATen: [aten.mm]
    at::mm_out(buf115, reinterpret_tensor(buf113, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_5_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf116 = buf92; ;  // reuse
    // Source Nodes: [xv_20], Original ATen: [aten.mm]
    at::mm_out(buf116, reinterpret_tensor(buf113, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_5_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf117 = reinterpret_tensor(buf113, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf118 = reinterpret_tensor(buf71, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    cpp_fused_stack_21((float*)(buf114.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf115.data_ptr()), (float*)(buf117.data_ptr()), (float*)(buf118.data_ptr()), s0);


    // Source Nodes: [output_31], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf119 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf117, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf118, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf116, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf120 = std::get<0>(buf119);

    auto buf127 = reinterpret_tensor(buf118, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_33], Original ATen: [aten.mm]
    at::mm_out(buf127, reinterpret_tensor(buf120, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_5_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf128 = reinterpret_tensor(buf105, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    auto buf129 = buf112; ;  // reuse
    auto buf130 = reinterpret_tensor(buf120, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_22((float*)(buf128.data_ptr()), (float*)(buf83.data_ptr()), (float*)(buf89.data_ptr()), (float*)(buf111.data_ptr()), (float*)(buf127.data_ptr()), (float*)(L__self___layers_5_ffn_norm_weight.data_ptr()), (float*)(buf129.data_ptr()), (float*)(buf130.data_ptr()), s0);
    auto buf131 = reinterpret_tensor(buf110, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_5_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf131, reinterpret_tensor(buf130, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_5_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf132 = buf109; ;  // reuse
    // Source Nodes: [l__self___layers_5_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf132, reinterpret_tensor(buf130, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_5_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf133 = reinterpret_tensor(buf131, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_23((float*)(buf133.data_ptr()), (float*)(buf132.data_ptr()), s0);
    auto buf134 = reinterpret_tensor(buf130, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_5_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf134, reinterpret_tensor(buf133, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_5_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf135 = buf129; ;  // reuse
    auto buf136 = reinterpret_tensor(buf89, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_24((float*)(buf128.data_ptr()), (float*)(buf134.data_ptr()), (float*)(L__self___layers_6_attention_norm_weight.data_ptr()), (float*)(buf135.data_ptr()), (float*)(buf136.data_ptr()), s0);
    auto buf137 = reinterpret_tensor(buf83, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [xq_24], Original ATen: [aten.mm]
    at::mm_out(buf137, reinterpret_tensor(buf136, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_6_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf138 = buf127; ;  // reuse
    // Source Nodes: [xk_30], Original ATen: [aten.mm]
    at::mm_out(buf138, reinterpret_tensor(buf136, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_6_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf139 = buf111; ;  // reuse
    // Source Nodes: [xv_24], Original ATen: [aten.mm]
    at::mm_out(buf139, reinterpret_tensor(buf136, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_6_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf140 = reinterpret_tensor(buf136, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf141 = buf117; ;  // reuse
    cpp_fused_stack_25((float*)(buf137.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf138.data_ptr()), (float*)(buf140.data_ptr()), (float*)(buf141.data_ptr()), s0);
    // Source Nodes: [output_37], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf142 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf140, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf141, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf139, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf143 = std::get<0>(buf142);

    auto buf150 = reinterpret_tensor(buf141, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_39], Original ATen: [aten.mm]
    at::mm_out(buf150, reinterpret_tensor(buf143, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_6_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf151 = buf135; ;  // reuse
    auto buf152 = reinterpret_tensor(buf143, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_26((float*)(buf128.data_ptr()), (float*)(buf134.data_ptr()), (float*)(buf150.data_ptr()), (float*)(L__self___layers_6_ffn_norm_weight.data_ptr()), (float*)(buf151.data_ptr()), (float*)(buf152.data_ptr()), s0);
    auto buf153 = reinterpret_tensor(buf133, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_6_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf153, reinterpret_tensor(buf152, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_6_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf154 = buf132; ;  // reuse
    // Source Nodes: [l__self___layers_6_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf154, reinterpret_tensor(buf152, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_6_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf155 = reinterpret_tensor(buf153, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_27((float*)(buf155.data_ptr()), (float*)(buf154.data_ptr()), s0);
    auto buf156 = reinterpret_tensor(buf152, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_6_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf156, reinterpret_tensor(buf155, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_6_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf157 = buf151; ;  // reuse
    auto buf158 = reinterpret_tensor(buf140, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_28((float*)(buf128.data_ptr()), (float*)(buf134.data_ptr()), (float*)(buf150.data_ptr()), (float*)(buf156.data_ptr()), (float*)(L__self___layers_7_attention_norm_weight.data_ptr()), (float*)(buf157.data_ptr()), (float*)(buf158.data_ptr()), s0);
    auto buf159 = buf139; ;  // reuse
    // Source Nodes: [xq_28], Original ATen: [aten.mm]
    at::mm_out(buf159, reinterpret_tensor(buf158, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_7_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf160 = buf138; ;  // reuse
    // Source Nodes: [xk_35], Original ATen: [aten.mm]
    at::mm_out(buf160, reinterpret_tensor(buf158, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_7_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf161 = buf137; ;  // reuse
    // Source Nodes: [xv_28], Original ATen: [aten.mm]
    at::mm_out(buf161, reinterpret_tensor(buf158, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_7_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf162 = reinterpret_tensor(buf158, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf163 = reinterpret_tensor(buf116, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    cpp_fused_stack_29((float*)(buf159.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf160.data_ptr()), (float*)(buf162.data_ptr()), (float*)(buf163.data_ptr()), s0);


    // Source Nodes: [output_43], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf164 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf162, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf163, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf161, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf165 = std::get<0>(buf164);

    auto buf172 = reinterpret_tensor(buf163, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_45], Original ATen: [aten.mm]
    at::mm_out(buf172, reinterpret_tensor(buf165, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_7_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf173 = buf128; ;  // reuse
    auto buf174 = buf157; ;  // reuse
    auto buf175 = reinterpret_tensor(buf165, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_30((float*)(buf173.data_ptr()), (float*)(buf134.data_ptr()), (float*)(buf150.data_ptr()), (float*)(buf156.data_ptr()), (float*)(buf172.data_ptr()), (float*)(L__self___layers_7_ffn_norm_weight.data_ptr()), (float*)(buf174.data_ptr()), (float*)(buf175.data_ptr()), s0);
    auto buf176 = reinterpret_tensor(buf155, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_7_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf176, reinterpret_tensor(buf175, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_7_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf177 = buf154; ;  // reuse
    // Source Nodes: [l__self___layers_7_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf177, reinterpret_tensor(buf175, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_7_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf178 = reinterpret_tensor(buf176, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_31((float*)(buf178.data_ptr()), (float*)(buf177.data_ptr()), s0);
    auto buf179 = reinterpret_tensor(buf175, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_7_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf179, reinterpret_tensor(buf178, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_7_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf180 = buf174; ;  // reuse
    auto buf181 = reinterpret_tensor(buf172, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_32((float*)(buf173.data_ptr()), (float*)(buf179.data_ptr()), (float*)(L__self___layers_8_attention_norm_weight.data_ptr()), (float*)(buf180.data_ptr()), (float*)(buf181.data_ptr()), s0);
    auto buf182 = buf156; ;  // reuse
    // Source Nodes: [xq_32], Original ATen: [aten.mm]
    at::mm_out(buf182, reinterpret_tensor(buf181, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_8_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf183 = buf150; ;  // reuse
    // Source Nodes: [xk_40], Original ATen: [aten.mm]
    at::mm_out(buf183, reinterpret_tensor(buf181, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_8_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf184 = buf134; ;  // reuse
    // Source Nodes: [xv_32], Original ATen: [aten.mm]
    at::mm_out(buf184, reinterpret_tensor(buf181, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_8_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf185 = reinterpret_tensor(buf181, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf186 = buf162; ;  // reuse
    cpp_fused_stack_33((float*)(buf182.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf183.data_ptr()), (float*)(buf185.data_ptr()), (float*)(buf186.data_ptr()), s0);
    // Source Nodes: [output_49], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf187 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf185, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf186, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf184, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf188 = std::get<0>(buf187);

    auto buf195 = reinterpret_tensor(buf186, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_51], Original ATen: [aten.mm]
    at::mm_out(buf195, reinterpret_tensor(buf188, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_8_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf196 = buf180; ;  // reuse
    auto buf197 = reinterpret_tensor(buf188, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_34((float*)(buf173.data_ptr()), (float*)(buf179.data_ptr()), (float*)(buf195.data_ptr()), (float*)(L__self___layers_8_ffn_norm_weight.data_ptr()), (float*)(buf196.data_ptr()), (float*)(buf197.data_ptr()), s0);
    auto buf198 = reinterpret_tensor(buf178, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_8_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf198, reinterpret_tensor(buf197, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_8_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf199 = buf177; ;  // reuse
    // Source Nodes: [l__self___layers_8_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf199, reinterpret_tensor(buf197, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_8_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf200 = reinterpret_tensor(buf198, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_35((float*)(buf200.data_ptr()), (float*)(buf199.data_ptr()), s0);
    auto buf201 = reinterpret_tensor(buf197, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_8_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf201, reinterpret_tensor(buf200, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_8_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf202 = buf196; ;  // reuse
    auto buf203 = reinterpret_tensor(buf185, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_36((float*)(buf173.data_ptr()), (float*)(buf179.data_ptr()), (float*)(buf195.data_ptr()), (float*)(buf201.data_ptr()), (float*)(L__self___layers_9_attention_norm_weight.data_ptr()), (float*)(buf202.data_ptr()), (float*)(buf203.data_ptr()), s0);
    auto buf204 = buf184; ;  // reuse
    // Source Nodes: [xq_36], Original ATen: [aten.mm]
    at::mm_out(buf204, reinterpret_tensor(buf203, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_9_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf205 = buf183; ;  // reuse
    // Source Nodes: [xk_45], Original ATen: [aten.mm]
    at::mm_out(buf205, reinterpret_tensor(buf203, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_9_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf206 = buf182; ;  // reuse
    // Source Nodes: [xv_36], Original ATen: [aten.mm]
    at::mm_out(buf206, reinterpret_tensor(buf203, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_9_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf207 = reinterpret_tensor(buf203, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf208 = reinterpret_tensor(buf161, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    cpp_fused_stack_37((float*)(buf204.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf205.data_ptr()), (float*)(buf207.data_ptr()), (float*)(buf208.data_ptr()), s0);


    // Source Nodes: [output_55], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf209 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf207, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf208, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf206, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf210 = std::get<0>(buf209);

    auto buf217 = reinterpret_tensor(buf208, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_57], Original ATen: [aten.mm]
    at::mm_out(buf217, reinterpret_tensor(buf210, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_9_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf218 = buf173; ;  // reuse
    auto buf219 = buf202; ;  // reuse
    auto buf220 = reinterpret_tensor(buf210, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_38((float*)(buf218.data_ptr()), (float*)(buf179.data_ptr()), (float*)(buf195.data_ptr()), (float*)(buf201.data_ptr()), (float*)(buf217.data_ptr()), (float*)(L__self___layers_9_ffn_norm_weight.data_ptr()), (float*)(buf219.data_ptr()), (float*)(buf220.data_ptr()), s0);
    auto buf221 = reinterpret_tensor(buf200, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_9_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf221, reinterpret_tensor(buf220, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_9_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf222 = buf199; ;  // reuse
    // Source Nodes: [l__self___layers_9_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf222, reinterpret_tensor(buf220, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_9_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf223 = reinterpret_tensor(buf221, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_39((float*)(buf223.data_ptr()), (float*)(buf222.data_ptr()), s0);
    auto buf224 = reinterpret_tensor(buf220, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_9_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf224, reinterpret_tensor(buf223, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_9_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf225 = buf219; ;  // reuse
    auto buf226 = reinterpret_tensor(buf217, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_40((float*)(buf218.data_ptr()), (float*)(buf224.data_ptr()), (float*)(L__self___layers_10_attention_norm_weight.data_ptr()), (float*)(buf225.data_ptr()), (float*)(buf226.data_ptr()), s0);
    auto buf227 = buf201; ;  // reuse
    // Source Nodes: [xq_40], Original ATen: [aten.mm]
    at::mm_out(buf227, reinterpret_tensor(buf226, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_10_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf228 = buf195; ;  // reuse
    // Source Nodes: [xk_50], Original ATen: [aten.mm]
    at::mm_out(buf228, reinterpret_tensor(buf226, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_10_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf229 = buf179; ;  // reuse
    // Source Nodes: [xv_40], Original ATen: [aten.mm]
    at::mm_out(buf229, reinterpret_tensor(buf226, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_10_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf230 = reinterpret_tensor(buf226, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf231 = buf207; ;  // reuse
    cpp_fused_stack_41((float*)(buf227.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf228.data_ptr()), (float*)(buf230.data_ptr()), (float*)(buf231.data_ptr()), s0);
    // Source Nodes: [output_61], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf232 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf230, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf231, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf229, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);
    auto buf233 = std::get<0>(buf232);

    auto buf240 = reinterpret_tensor(buf231, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_63], Original ATen: [aten.mm]
    at::mm_out(buf240, reinterpret_tensor(buf233, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_10_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf241 = buf225; ;  // reuse
    auto buf242 = reinterpret_tensor(buf233, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_42((float*)(buf218.data_ptr()), (float*)(buf224.data_ptr()), (float*)(buf240.data_ptr()), (float*)(L__self___layers_10_ffn_norm_weight.data_ptr()), (float*)(buf241.data_ptr()), (float*)(buf242.data_ptr()), s0);
    auto buf243 = reinterpret_tensor(buf223, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_10_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf243, reinterpret_tensor(buf242, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_10_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf244 = buf222; ;  // reuse
    // Source Nodes: [l__self___layers_10_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf244, reinterpret_tensor(buf242, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_10_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf245 = reinterpret_tensor(buf243, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_43((float*)(buf245.data_ptr()), (float*)(buf244.data_ptr()), s0);
    auto buf246 = reinterpret_tensor(buf242, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_10_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf246, reinterpret_tensor(buf245, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_10_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));
    auto buf247 = buf241; ;  // reuse
    auto buf248 = reinterpret_tensor(buf230, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_44((float*)(buf218.data_ptr()), (float*)(buf224.data_ptr()), (float*)(buf240.data_ptr()), (float*)(buf246.data_ptr()), (float*)(L__self___layers_11_attention_norm_weight.data_ptr()), (float*)(buf247.data_ptr()), (float*)(buf248.data_ptr()), s0);
    auto buf249 = buf229; ;  // reuse
    // Source Nodes: [xq_44], Original ATen: [aten.mm]
    at::mm_out(buf249, reinterpret_tensor(buf248, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_11_attention_wq_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf250 = buf228; ;  // reuse
    // Source Nodes: [xk_55], Original ATen: [aten.mm]
    at::mm_out(buf250, reinterpret_tensor(buf248, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_11_attention_wk_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf251 = buf227; ;  // reuse
    // Source Nodes: [xv_44], Original ATen: [aten.mm]
    at::mm_out(buf251, reinterpret_tensor(buf248, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_11_attention_wv_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf252 = reinterpret_tensor(buf248, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    auto buf253 = reinterpret_tensor(buf206, {1L, s0, 12L, 32L, 2L}, {768L*s0, 768L, 64L, 2L, 1L}, 0L);   // reuse
    cpp_fused_stack_45((float*)(buf249.data_ptr()), (float*)(L__self___freqs_cos.data_ptr()), (float*)(L__self___freqs_sin.data_ptr()), (float*)(buf250.data_ptr()), (float*)(buf252.data_ptr()), (float*)(buf253.data_ptr()), s0);


    // Source Nodes: [output_67], Original ATen: [aten._scaled_dot_product_flash_attention]
    auto buf254 = at::_scaled_dot_product_flash_attention(reinterpret_tensor(buf252, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf253, {1L, 12L, s0, 64L}, {0L, 64L, 768L, 1L}, 0L), reinterpret_tensor(buf251, {1L, 12L, s0, 64L}, {768L*s0, 64L, 768L, 1L}, 0L), 0.0, true, false, c10::nullopt);


    auto buf255 = std::get<0>(buf254);

    auto buf262 = reinterpret_tensor(buf253, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [output_69], Original ATen: [aten.mm]
    at::mm_out(buf262, reinterpret_tensor(buf255, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_11_attention_wo_weight, {768L, 768L}, {1L, 768L}, 0L));
    auto buf263 = buf218; ;  // reuse
    auto buf264 = buf247; ;  // reuse
    auto buf265 = reinterpret_tensor(buf255, {1L, s0, 768L}, {768L*s0, 768L, 1L}, 0L);   // reuse
    cpp_fused_add_mean_mul_pow_rsqrt_46((float*)(buf263.data_ptr()), (float*)(buf224.data_ptr()), (float*)(buf240.data_ptr()), (float*)(buf246.data_ptr()), (float*)(buf262.data_ptr()), (float*)(L__self___layers_11_ffn_norm_weight.data_ptr()), (float*)(buf264.data_ptr()), (float*)(buf265.data_ptr()), s0);




    auto buf266 = reinterpret_tensor(buf245, {s0, 2048L}, {2048L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_11_feed_forward_w1], Original ATen: [aten.mm]
    at::mm_out(buf266, reinterpret_tensor(buf265, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_11_feed_forward_w1_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf267 = buf244; ;  // reuse
    // Source Nodes: [l__self___layers_11_feed_forward_w3], Original ATen: [aten.mm]
    at::mm_out(buf267, reinterpret_tensor(buf265, {s0, 768L}, {768L, 1L}, 0L), reinterpret_tensor(L__self___layers_11_feed_forward_w3_weight, {768L, 2048L}, {1L, 768L}, 0L));
    auto buf268 = reinterpret_tensor(buf266, {1L, s0, 2048L}, {2048L*s0, 2048L, 1L}, 0L);   // reuse
    cpp_fused_mul_silu_47((float*)(buf268.data_ptr()), (float*)(buf267.data_ptr()), s0);

    auto buf269 = reinterpret_tensor(buf265, {s0, 768L}, {768L, 1L}, 0L);   // reuse
    // Source Nodes: [l__self___layers_11_feed_forward_w2], Original ATen: [aten.mm]
    at::mm_out(buf269, reinterpret_tensor(buf268, {s0, 2048L}, {2048L, 1L}, 0L), reinterpret_tensor(L__self___layers_11_feed_forward_w2_weight, {2048L, 768L}, {1L, 2048L}, 0L));

    auto buf270 = buf264; ;  // reuse
    auto buf271 = at::empty_strided({1L, 1L, 768L}, {768L, 768L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    cpp_fused_add_index_mean_mul_pow_rsqrt_48((float*)(buf263.data_ptr()), (float*)(buf269.data_ptr()), (float*)(L__self___norm_weight.data_ptr()), (float*)(buf270.data_ptr()), (float*)(buf271.data_ptr()), s0);



    auto buf272 = at::empty_strided({1L, 32000L}, {32000L, 1L}, at::TensorOptions(at::kCPU).dtype(at::kFloat));
    // Source Nodes: [logits], Original ATen: [aten.mm]
    at::mm_out(buf272, reinterpret_tensor(buf271, {1L, 768L}, {0L, 1L}, 0L), reinterpret_tensor(L__self___tok_embeddings_weight, {768L, 32000L}, {1L, 768L}, 0L));
    output_handles[0] = reinterpret_cast<AtenTensorHandle>(new at::Tensor(reinterpret_tensor(buf272, {1L, 1L, 32000L}, {32000L, 32000L, 1L}, 0L)));
} // AOTInductorModel::run_impl
} // namespace aot_inductor
} // namespace torch
