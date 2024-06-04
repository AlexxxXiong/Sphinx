# `ArenaPlanner`

---

```c++
/// 请注意，将来可能会添加新的错误状态值，以便表示更细粒度的内部状态，因此，应用程序应该不依赖状态值是枚举的成员。
typedef enum TfLiteStatus {
  /// Success
  kTfLiteOk = 0,

  /// 通常指的是运行时的错误（即解释器）
  kTfLiteError = 1,

  /// 通常指的是TfLiteDelegate本身的错误。
  kTfLiteDelegateError = 2,

  /// 一般来说，指的是申请委托人时的错误，因为运行时和委托之间不兼容，例如，返回此错误当尝试将TF Lite委托应用于已经已有的模型图时不可变。
  kTfLiteApplicationError = 3,

  /// 通常是指找不到序列化委托数据。参见 tflite::delegates::Serialization。
  kTfLiteDelegateDataNotFound = 4,

  /// Generally referring to data-writing issues in delegate serialization.
  /// See tflite::delegates::Serialization.
  kTfLiteDelegateDataWriteError = 5,

  /// 通常指的是委托序列化中的数据读取问题。参见 tflite::delegates::Serialization。
  kTfLiteDelegateDataReadError = 6,

  /// 通常指的是TF Lite模型有不能操作的问题在运行时解决。当特定的操作不是时，可能会发生这种情况使用TF Lite框架注册或构建。
  kTfLiteUnresolvedOps = 7,

  /// 通常是指用户取消的调用。
  /// See `interpreter::Cancel`.
  // TODO(b/194915839): Implement `interpreter::Cancel`.
  // TODO(b/250636993): Cancellation triggered by `SetCancellationFunction`
  // should also return this status code.
  kTfLiteCancelled = 8,
} TfLiteStatus;
==============================================================================*/
#ifndef TENSORFLOW_LITE_MEMORY_PLANNER_H_
#define TENSORFLOW_LITE_MEMORY_PLANNER_H_

#include <vector>

#include "tensorflow/lite/core/c/common.h"

namespace tflite {

// MemoryPlanner负责规划和执行一些TF Lite中必要的内存相关操作。
class MemoryPlanner {
 public:
  virtual ~MemoryPlanner() {}

  // 计划必要的内存分配。这是MemoryPlanner的预处理步骤，当图形结构已知时调用，但张量的实际大小不是。  Success
  virtual TfLiteStatus PlanAllocations() = 0;

  // 分配必要的内存来执行区间内的所有节点 [first_node，last_node]。  Success
  virtual TfLiteStatus ExecuteAllocations(int first_node, int last_node) = 0;

  // 使之前的分配无效。当张量大小发生变化时，这被称为。
  // 所有计划的分配都保留，但在调用 ExecuteAllocations() 之前无法使用。
  virtual TfLiteStatus ResetAllocations() = 0;

  // 在给定节点执行后使分配无效。
  virtual TfLiteStatus ResetAllocationsAfter(int node) = 0;

  // 注意：以下两种方法修改非持久arena上所有张量（输入、输出、中间体）的数据指针。
  // 如果用户手动设置了其中任何一个的指针，则需要再次设置。

  // 这释放了分配给非持久张量的内存。
	// 它没有清除分配计划，但在调用 AcquireNonPersistentMemory() 之前，无法使用内存。
	// 在此方法之后调用Reset/PlanAllocations是安全的，在张量大小发生变化的情况下，无需调用ReleaseTemporaryAllocations。
  virtual TfLiteStatus ReleaseNonPersistentMemory() = 0;

  // 分配必要的内存来包含非持久张量。
  virtual TfLiteStatus AcquireNonPersistentMemory() = 0;

  // 如果非持久内存可用，则返回true。
  virtual bool HasNonPersistentMemory() = 0;

  // 将内存规划信息转储到指定的操作节点执行计划（即“execution_plan”）上，以便调试。
  virtual void DumpDebugInfo(const std::vector<int>& execution_plan) const = 0;

  // 返回分配信息的映射。它仅用于调试。
  virtual void GetAllocInfo(size_t *arena_size,
                            size_t *arena_persist_size) const = 0;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MEMORY_PLANNER_H_

```



```c++
/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_ARENA_PLANNER_H_
#define TENSORFLOW_LITE_ARENA_PLANNER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/graph_info.h"
#include "tensorflow/lite/memory_planner.h"
#include "tensorflow/lite/simple_memory_arena.h"
#include "tensorflow/lite/util.h"

namespace tflite {

constexpr const int kDefaultArenaAlignment = 64;

// A memory planner that makes all the allocations using arenas.
//
// Before a model is executed by the interpreter, this class determines when
// each tensor needs to be allocated and deallocated, and preallocates all the
// necessary memory (the PlanAllocations phase). It then assigns portions of
// this memory buffer to each tensor (the ExecuteAllocations phase). Tensors may
// share some of the buffer if a tensor B is to be allocated after another
// tensor A has been deallocated.
//
// If dynamic tensors are used the planning steps can be repeated during model
// execution. Since dynamic tensors don't have sizes until after the
// corresponding operation is executed, this class supports incremental
// planning.
class ArenaPlanner : public MemoryPlanner {
 public:
  // Ownership of 'context' is not taken and it must remain util the
  // ArenaPlanner is destroyed. The inputs to the graph will not share
  // memory with any other tensor, effectively preserving them until the end
  // of inference.
  ArenaPlanner(TfLiteContext* context, std::unique_ptr<GraphInfo> graph_info,
               bool preserve_all_tensors, int tensor_alignment,
               int subgraph_index = 0);
  ~ArenaPlanner() override;
  ArenaPlanner(const ArenaPlanner&) = delete;
  ArenaPlanner& operator=(const ArenaPlanner&) = delete;

  TfLiteStatus ResetAllocations() override;
  TfLiteStatus ResetAllocationsAfter(int node) override;
  TfLiteStatus PlanAllocations() override;
  TfLiteStatus ExecuteAllocations(int first_node, int last_node) override;
  TfLiteStatus ReleaseNonPersistentMemory() override;
  TfLiteStatus AcquireNonPersistentMemory() override;
  bool HasNonPersistentMemory() override;
  void DumpDebugInfo(const std::vector<int>& execution_plan) const override;
  void GetAllocInfo(size_t* arena_size,
                    size_t* arena_persist_size) const override;

  // Returns the base arena location for a given allocation type.
  std::intptr_t BasePointer(TfLiteAllocationType type);

 private:
  // Check whether the input tensor's memory may be shared the output tensor.
  // tensor_changed: true if the output tensor modifies the tensor data. For
  // example, `Reshape` doesn't modify data but Add does.
  bool InputTensorCanBeShared(const TfLiteTensor& input,
                              const TfLiteTensor& output, int input_id,
                              int output_id, bool tensor_changed);

  // Identify tensors which can share memory with another.
  void IdentifyInPlaceTensors();

  // Make sure all the arenas have reserved enough memory to store all their
  // tensors.
  TfLiteStatus Commit(bool* arena_reallocated);

  // Sorts tensors_to_allocate` using by the following ordering:
  // - Tensors that have lifespan through the whole model inference time go
  // first;
  // - Other tensors (e.g. intermediate and temporary ones) are sorted from
  // largest to smallest. For equal sized tensors, the tensor which is used
  // first goes first.
  void CreateTensorAllocationVector(std::vector<int32_t>* tensors_to_allocate);

  // Returns vector containing the indices of all tensors allocated between
  // `first_node` and `last_node`.
  std::vector<int32_t> GetTensorsToAllocate(int first_node, int last_node);

  // Traverse the allocation queue and reserve space in the appropriate arena
  // for all tensors affected by ops in the interval [first_node, last_node].
  TfLiteStatus CalculateAllocations(int first_node, int last_node,
                                    std::vector<int32_t>* tensors_allocated);

  // Assign absolute memory location to a tensor, based on its relative
  // position inside the corresponding arena buffer.
  TfLiteStatus ResolveTensorAllocation(int32_t tensor_index,
                                       TfLiteTensor* tensors);

  // Register an allocation for all internal (temporary) tensors of
  // 'node_index'.
  TfLiteStatus CalculateAllocationOfInternalTensors(int node_index);

  // Register a deallocation for all internal (temporary) tensors of
  // 'node_index'.
  TfLiteStatus CalculateDeallocationOfInternalTensors(int node_index);

  // Return the index of the tensor owing `tensor_index's` buffer.
  int FindSharedTensor(int tensor_index);

  TfLiteContext* context_;
  std::unique_ptr<GraphInfo> graph_info_;

  // Stores allocation data for all tensors.
  std::vector<ArenaAllocWithUsageInterval> allocs_;

  // Map of Tensors allocated by each node.
  // NOLINTNEXTLINE - absl::flat_hash_set increases binary size by 106kB.
  std::vector<std::unordered_set<int32_t>> nodes_to_tensors_;

  // First node, that uses the tensor. It needs to be allocated before
  // execution of the node's operation.
  std::vector<int32_t> alloc_node_;

  // Last node, that uses the tensor. It can be deallocated after execution of
  // the node's operation.
  std::vector<int32_t> dealloc_node_;

  // Raw memory buffer that is allocated for all temporary and graph outputs
  // that are declared kTfLiteArenaRw.
  SimpleMemoryArena arena_;
  // True when the arena_ has allocated memory (Commit was called).
  bool has_nonpersistent_memory_;

  // Raw memory buffer that is allocated for persistent tensors that are
  // declared as kTfLiteArenaRwPersistent.
  SimpleMemoryArena persistent_arena_;

  // If true, then no overlapping of memory areas is done, meaning intermediate
  // tensors and temporary tensors can be queried after running.
  // (modulo running delegates)
  bool preserve_all_tensors_;

  // Number of bytes that tensor buffers should be aligned to.
  int tensor_alignment_;

  // Index of the last node whose tensors were allocated.
  int last_active_node_;

  // Holds index of original tensor if the tensor is sharing underlined
  // data with another tensor.
  // NOLINTNEXTLINE - absl::flat_hash_map increases binary size by 106kB.
  std::unordered_map<int32_t, int32_t> actual_tensor_id_;

  // Store number of references to each tensor.
  std::vector<int> refcounts_;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_ARENA_PLANNER_H_

```

