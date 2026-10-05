/******************************************************************************
 *
 * Copyright (C) 2026 Ittiam Systems Pvt Ltd, Bangalore
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 ******************************************************************************/

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

// clang-format off
#include "ihevc_typedefs.h"
#include "ihevc_defs.h"
extern "C" {
#include "ihevcd_itrans_recon_dc.h"
#include "ihevcd_function_selector.h"
#include "iv.h"
}
// clang-format on

#include "BenchmarkCommon.h"
#include "TestCommon.h"

namespace {

using HbdITransReconDcLumaFn = ihevcd_hbd_itrans_recon_dc_luma_ft*;
using HbdITransReconDcChromaFn = ihevcd_hbd_itrans_recon_dc_chroma_ft*;

struct ItransReconDcBenchConfig {
  int trans_size;
  WORD16 coeff_value;
  IV_ARCH_T arch;
  int bit_depth = 10;
};

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
const func_selector_t dec_test_ssse3 = []() {
  func_selector_t ret = {};
  ihevcd_init_function_ptr_ssse3(&ret);
  return ret;
}();

const func_selector_t dec_test_sse42 = []() {
  func_selector_t ret = {};
  ihevcd_init_function_ptr_sse42(&ret);
  return ret;
}();
#elif defined(__aarch64__)
const func_selector_t dec_test_arm64 = []() {
  func_selector_t ret = {};
#ifdef DARWIN
  ihevcd_init_function_ptr_generic(&ret);
#else
  ihevcd_init_function_ptr_av8(&ret);
#endif
  return ret;
}();
#elif defined(__arm__)
const func_selector_t dec_test_arm32 = []() {
  func_selector_t ret = {};
#ifdef DARWIN
  ihevcd_init_function_ptr_generic(&ret);
#else
  ihevcd_init_function_ptr_a9q(&ret);
#endif
  return ret;
}();
#endif

const func_selector_t dec_ref = []() {
  func_selector_t ret = {};
  ihevcd_init_function_ptr_generic(&ret);
  return ret;
}();

const func_selector_t* GetDecoderFuncPtr(IV_ARCH_T arch) {
  switch (arch) {
    case ARCH_NA:
      return &dec_ref;
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || \
    defined(_M_IX86)
    case ARCH_X86_SSSE3:
      return &dec_test_ssse3;
    case ARCH_X86_SSE42:
      return &dec_test_sse42;
#elif defined(__aarch64__)
    case ARCH_ARMV8_GENERIC:
      return &dec_test_arm64;
#elif defined(__arm__)
    case ARCH_ARM_A9Q:
    case ARCH_ARM_A7:
    case ARCH_ARM_A5:
    case ARCH_ARM_A15:
    case ARCH_ARM_NEONINTR:
      return &dec_test_arm32;
#endif
    default:
      return nullptr;
  }
}

HbdITransReconDcLumaFn GetHbdITransReconDcLumaFn(IV_ARCH_T arch) {
  const func_selector_t* selector = GetDecoderFuncPtr(arch);
  if (selector && selector->ihevcd_hbd_itrans_recon_dc_luma_fptr) {
    return selector->ihevcd_hbd_itrans_recon_dc_luma_fptr;
  }
  if (arch == ARCH_NA) {
    return ihevcd_hbd_itrans_recon_dc_luma;
  }
  return nullptr;
}

HbdITransReconDcChromaFn GetHbdITransReconDcChromaFn(IV_ARCH_T arch) {
  const func_selector_t* selector = GetDecoderFuncPtr(arch);
  if (selector && selector->ihevcd_hbd_itrans_recon_dc_chroma_fptr) {
    return selector->ihevcd_hbd_itrans_recon_dc_chroma_fptr;
  }
  if (arch == ARCH_NA) {
    return ihevcd_hbd_itrans_recon_dc_chroma;
  }
  return nullptr;
}

void BM_HbdITransReconDcLuma(benchmark::State& state,
                             ItransReconDcBenchConfig config) {
  const int trans_size = config.trans_size;
  const WORD16 coeff_value = config.coeff_value;
  const IV_ARCH_T arch = config.arch;
  const WORD32 bit_depth = config.bit_depth;
  const WORD32 log2_trans_size =
      (trans_size == 4) ? 2 : (trans_size == 8) ? 3 : (trans_size == 16) ? 4 : 5;

  HbdITransReconDcLumaFn fn = GetHbdITransReconDcLumaFn(arch);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  const int pad = 16;
  const WORD32 pred_strd = trans_size;
  const WORD32 dst_strd = trans_size;

  std::vector<UWORD16> pu2_pred(pred_strd * trans_size + pad);
  std::vector<UWORD16> pu2_dst(dst_strd * trans_size + pad);

  FillRandom(pu2_pred, static_cast<UWORD16>(0),
             static_cast<UWORD16>((1 << bit_depth) - 1), 1337);
  std::fill(pu2_dst.begin(), pu2_dst.end(), 0xAAAA);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    HbdITransReconDcLumaFn ref_fn = GetHbdITransReconDcLumaFn(ARCH_NA);
    if (ref_fn) {
      std::vector<UWORD16> ref_dst(dst_strd * trans_size + pad, 0xAAAA);
      ref_fn(pu2_pred.data(), ref_dst.data(), pred_strd, dst_strd,
             log2_trans_size, coeff_value, bit_depth);
      fn(pu2_pred.data(), pu2_dst.data(), pred_strd, dst_strd,
         log2_trans_size, coeff_value, bit_depth);
      if (!VerifyOutput2D(ref_dst.data(), pu2_dst.data(), trans_size,
                          trans_size, dst_strd)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(pu2_pred.data(), pu2_dst.data(), pred_strd, dst_strd,
       log2_trans_size, coeff_value, bit_depth);
    benchmark::DoNotOptimize(pu2_dst.data());
  }
}

void BM_HbdITransReconDcChroma(benchmark::State& state,
                               ItransReconDcBenchConfig config) {
  const int trans_size = config.trans_size;
  const WORD16 coeff_value = config.coeff_value;
  const IV_ARCH_T arch = config.arch;
  const WORD32 bit_depth = config.bit_depth;
  const WORD32 log2_trans_size =
      (trans_size == 4) ? 2 : (trans_size == 8) ? 3 : (trans_size == 16) ? 4 : 5;

  HbdITransReconDcChromaFn fn = GetHbdITransReconDcChromaFn(arch);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  const int pad = 16;
  const WORD32 pred_strd = 2 * trans_size;
  const WORD32 dst_strd = 2 * trans_size;

  std::vector<UWORD16> pu2_pred(pred_strd * trans_size + pad);
  std::vector<UWORD16> pu2_dst(dst_strd * trans_size + pad);

  FillRandom(pu2_pred, static_cast<UWORD16>(0),
             static_cast<UWORD16>((1 << bit_depth) - 1), 1337);
  std::fill(pu2_dst.begin(), pu2_dst.end(), 0xAAAA);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    HbdITransReconDcChromaFn ref_fn = GetHbdITransReconDcChromaFn(ARCH_NA);
    if (ref_fn) {
      std::vector<UWORD16> ref_dst(dst_strd * trans_size + pad, 0xAAAA);
      ref_fn(pu2_pred.data(), ref_dst.data(), pred_strd, dst_strd,
             log2_trans_size, coeff_value, bit_depth);
      fn(pu2_pred.data(), pu2_dst.data(), pred_strd, dst_strd,
         log2_trans_size, coeff_value, bit_depth);
      if (!VerifyOutput2D(ref_dst.data(), pu2_dst.data(), 2 * trans_size,
                          trans_size, dst_strd)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(pu2_pred.data(), pu2_dst.data(), pred_strd, dst_strd,
       log2_trans_size, coeff_value, bit_depth);
    benchmark::DoNotOptimize(pu2_dst.data());
  }
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();
  const int bit_depth = 10;

  for (int size : {4, 8, 16, 32}) {
    for (WORD16 coeff : {512, -512}) {
      for (auto arch : arches) {
        if (arch != ARCH_NA && !GetHbdITransReconDcLumaFn(arch)) {
          continue;
        }
        std::string arch_name = GetArchName(arch);
        std::string name =
            "BM_HbdITransReconDcLuma/" + std::to_string(size) + "x" +
            std::to_string(size) + "/coeff_" + std::to_string(coeff) + "/" +
            std::to_string(bit_depth) + "bit/" + arch_name;

        ItransReconDcBenchConfig cfg{size, coeff, arch, bit_depth};
        benchmark::RegisterBenchmark(
            name.c_str(),
            [cfg](benchmark::State& st) { BM_HbdITransReconDcLuma(st, cfg); })
            ->Unit(benchmark::kNanosecond);
      }
    }
  }

  for (int size : {4, 8, 16, 32}) {
    for (WORD16 coeff : {512, -512}) {
      for (auto arch : arches) {
        if (arch != ARCH_NA && !GetHbdITransReconDcChromaFn(arch)) {
          continue;
        }
        std::string arch_name = GetArchName(arch);
        std::string name =
            "BM_HbdITransReconDcChroma/" + std::to_string(size) + "x" +
            std::to_string(size) + "/coeff_" + std::to_string(coeff) + "/" +
            std::to_string(bit_depth) + "bit/" + arch_name;

        ItransReconDcBenchConfig cfg{size, coeff, arch, bit_depth};
        benchmark::RegisterBenchmark(
            name.c_str(),
            [cfg](benchmark::State& st) { BM_HbdITransReconDcChroma(st, cfg); })
            ->Unit(benchmark::kNanosecond);
      }
    }
  }
}
