/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_C_PJRT_C_API_EXECUTE_CHAIN_EXTENSION_H_
#define XLA_PJRT_C_PJRT_C_API_EXECUTE_CHAIN_EXTENSION_H_

#include <stdbool.h>
#include <stddef.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PJRT_API_EXECUTE_CHAIN_EXTENSION_VERSION 0

typedef enum PJRT_ExecuteChain_InputKind {
  PJRT_ExecuteChain_InputKind_Buffer = 0,
  PJRT_ExecuteChain_InputKind_Output = 1,
} PJRT_ExecuteChain_InputKind;

struct PJRT_ExecuteChain_Input {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_ExecuteChain_InputKind kind;
  PJRT_Buffer* buffer;
  size_t output_step;
  size_t output_index;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteChain_Input, output_index);

struct PJRT_ExecuteChain_Step {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_LoadedExecutable* executable;
  // Flattened [num_devices][num_args].
  PJRT_ExecuteChain_Input* argument_refs;
  size_t num_args;
  // Same shape as PJRT_LoadedExecutable_Execute_Args.output_lists:
  // [num_devices][num_outputs]. Only entries marked in returned_outputs are
  // populated.
  PJRT_Buffer** const* output_lists;
  const bool* returned_outputs;
  size_t num_outputs;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteChain_Step, num_outputs);

struct PJRT_LoadedExecutable_ExecuteChain_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;
  PJRT_ExecuteOptions* options;
  PJRT_ExecuteChain_Step* steps;
  size_t num_steps;
  size_t num_devices;
  // Optional [num_devices] output, filled with completion events for the final
  // chain step.
  PJRT_Event** device_complete_events;
};
PJRT_DEFINE_STRUCT_TRAITS(PJRT_LoadedExecutable_ExecuteChain_Args,
                          device_complete_events);

typedef PJRT_Error* PJRT_LoadedExecutable_ExecuteChain(
    PJRT_LoadedExecutable_ExecuteChain_Args* args);

typedef struct PJRT_ExecuteChain_Extension {
  PJRT_Extension_Base base;
  PJRT_LoadedExecutable_ExecuteChain* execute_chain;
} PJRT_ExecuteChain_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_ExecuteChain_Extension, execute_chain);

#ifdef __cplusplus
}
#endif

#endif  // XLA_PJRT_C_PJRT_C_API_EXECUTE_CHAIN_EXTENSION_H_
