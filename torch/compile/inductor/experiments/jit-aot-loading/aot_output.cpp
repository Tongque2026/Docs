
#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/thread_local.h>

#include <iostream>
#include <sstream>
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
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
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
        num_models, std::string(device_str), cubin_dir_opt);
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

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(container_handle,
          constant_map_handle,
          /*use_inactive*/ true,
          /*validate_full_update*/ true);
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
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
    AOTInductorConstantMapHandle constant_map_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto constant_array = std::make_shared<std::vector<torch::aot_inductor::ConstantHandle>>();
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          "cpu", // device_str is hardcoded, as AOTInductorModelCreate is only use for CPU models
          ""
      );

      if (input_map) {
        for (auto const& kv : *input_map) {
          constant_map->emplace(kv.first, kv.second);
        }
      } else {
        model->load_constants();
      }

      *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
    })}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

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
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

} // extern "C"
// NOTE: Like interface.cpp, this file will be copied into AOTInductor
// generated output. This file is intended to keep implementation
// details separate from the implementation of the AOTI public
// interface. Note also that #includes should go into interface.cpp
// for simplicity of maintenance.

namespace torch {
namespace aot_inductor {
template <typename T>
void convert_output_to_handle(
    const ArrayRefTensor<T>& output,
    AtenTensorHandle& handle) {
  handle = output.expensiveCopyToTensor();
}

template <typename... Ts, std::size_t... Is>
void convert_outputs_to_handles_helper(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles,
    std::index_sequence<Is...>) {
  (convert_output_to_handle(std::get<Is>(outputs), output_handles[Is]), ...);
}
template <typename... Ts>
void convert_outputs_to_handles(
    const std::tuple<ArrayRefTensor<Ts>...>& outputs,
    AtenTensorHandle* output_handles) {
  convert_outputs_to_handles_helper(
      outputs, output_handles, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void convert_handle_to_arrayref_tensor(
    AtenTensorHandle handle,
    ArrayRefTensor<T>& input) {
  void* data_ptr;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_data_ptr(handle, &data_ptr));
  int64_t dim;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dim(handle, &dim));
  int64_t numel;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_numel(handle, &numel));
  int64_t* sizes;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes(handle, &sizes));
  int64_t* strides;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides(handle, &strides));
  int32_t dtype;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(handle, &dtype));
  int32_t device_type;
  AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(handle, &device_type));
  int32_t device_index;
  AOTI_TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_device_index(handle, &device_index));

  input = ArrayRefTensor<T>(
      MiniArrayRef<T>(reinterpret_cast<T*>(data_ptr), numel),
      MiniArrayRef<const int64_t>(sizes, dim),
      MiniArrayRef<const int64_t>(strides, dim),
      device_type,
      device_index);
}

template <typename... Ts, std::size_t... Is>
void convert_handles_to_inputs_helper(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs,
    std::index_sequence<Is...>) {
  (convert_handle_to_arrayref_tensor(input_handles[Is], std::get<Is>(inputs)),
   ...);
}

template <typename... Ts>
void convert_handles_to_inputs(
    AtenTensorHandle* input_handles,
    std::tuple<ArrayRefTensor<Ts>...>& inputs) {
  convert_handles_to_inputs_helper(
      input_handles, inputs, std::make_index_sequence<sizeof...(Ts)>());
}

template <typename T>
void assert_numel(const ArrayRefTensor<T>& tensor, uint64_t numel) {
  if (tensor.numel() != numel) {
    std::stringstream err;
    err << "incorrect numel for input tensor. expected " << numel << ", got " << tensor.numel();
    throw std::runtime_error(err.str());
  }
}
} // namespace aot_inductor
} // namespace torch
#include <torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h>

#include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
#include <torch/csrc/inductor/aoti_runtime/thread_local.h>
#include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>

#include <torch/csrc/inductor/aoti_runtime/model.h>
typedef at::Half half;
typedef at::BFloat16 bfloat16;
#include <c10/util/generic_math.h>

[[maybe_unused]] static int64_t align(int64_t nbytes) {
  return (nbytes + 64 - 1) & -64;
}

#include "/work/learning/gpt6-1sol/torch/compile/inductor/experiments/jit-aot-loading/cache-9wi6ogyn/vu/cvuvp4i7roujum4xemrfwnb3t4c5t3r3mihr4b7iegh6tcqvdg43.h"
extern "C"  void cpp_fused_add_0(const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(int64_t x0=static_cast<int64_t>(0L); x0<static_cast<int64_t>(2L); x0+=static_cast<int64_t>(1L))
        {
            #pragma GCC ivdep
            for(int64_t x1=static_cast<int64_t>(0L); x1<static_cast<int64_t>(3L); x1+=static_cast<int64_t>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<int64_t>(x1 + (3L*x0))];
                auto tmp1 = in_ptr1[static_cast<int64_t>(x1)];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                out_ptr0[static_cast<int64_t>(x1 + (3L*x0))] = tmp2;
            }
        }
    }
}
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(cpu);
CACHE_TORCH_LAYOUT(strided);
namespace torch {
namespace aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
};
}  // namespace

AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(1, 1, 1, device_str, cubin_dir) {
    inputs_info_[0].name = "arg1_1";
    constants_info_[0].name = "weight";
    constants_info_[0].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 64;
    constants_info_[0].from_folded = false;
    constants_info_[0].shape = {3};
    constants_info_[0].stride = {1};
    constants_info_[0].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[0].original_fqn = "weight";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = "[1, {\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": null, \"context\": null, \"children_spec\": []}]}, {\"type\": \"builtins.dict\", \"context\": \"[]\", \"children_spec\": []}]}]";
    out_spec_ = "[1, {\"type\": \"builtins.tuple\", \"context\": \"null\", \"children_spec\": [{\"type\": null, \"context\": null, \"children_spec\": []}]}]";
    outputs_info_[0].name = "output0";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {

    if (!initialization) {
        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                  << "aot_inductor.use_runtime_constant_folding=False\n";
    }
    return {};
}

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {}

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

    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 1);
    auto arg1_1 = std::move(inputs[0]);
    auto weight = constants_->at(0);
    inputs.clear();
    auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());
    static constexpr int64_t int_array_0[] = {2L, 3L};
    static constexpr int64_t int_array_1[] = {3L, 1L};
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cpu, this->device_idx_, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    auto* var_0 = get_data_ptr_wrapper(arg1_1);
    auto* var_1 = get_data_ptr_wrapper(weight);
    auto* var_2 = get_data_ptr_wrapper(buf0);
    cpp_fused_add_0((const float*)(var_0), (const float*)(var_1), (float*)(var_2));
    arg1_1.reset();
    if constexpr (std::is_same_v<std::decay_t<decltype(buf0)>,RAIIAtenTensorHandle> || std::is_same_v<std::decay_t<decltype(buf0)>,AtenTensorHandle> || std::is_same_v<std::decay_t<decltype(buf0)>,ConstantHandle>) {
        output_handles[0] = buf0.release();
    } else {
        thread_local ThreadLocalCachedOutputTensor<std::decay_t<decltype(buf0)>> cached_output_0(buf0);
        cached_output_0.copy_data_from(buf0);
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_new_uninitialized_tensor(&output_handles[0]));
        AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors(cached_output_0.tensor(), output_handles[0]));
    }
} // AOTInductorModel::run_impl
} // namespace aot_inductor
} // namespace torch
