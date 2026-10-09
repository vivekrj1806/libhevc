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
#include <string>
#include <utility>
#include <vector>

#include "TestCommon.h"
#include "ihevc_intra_pred_utils.h"

namespace {

void BM_HbdLumaIntraPred(benchmark::State& state, IntraPredConfig config) {
  const int nt = config.nt;
  const int mode = config.mode;
  const int mode_or_flag =
      (mode == 10 || mode == 26) ? config.disable_boundary_filter : mode;
  const IV_ARCH_T arch = config.arch;
  const UWORD8 bit_depth = static_cast<UWORD8>(config.bit_depth);

  HbdLumaIntraPredFn fn = GetHbdLumaIntraPredFn(arch, mode);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  IntraPredLumaBuffers<UWORD16> buf = CreateIntraPredLumaBuffers<UWORD16>(
      nt, config.dst_strd_mul, config.bit_depth);

  // Correctness verification against C reference prior to measurement
  if (arch != ARCH_NA) {
    HbdLumaIntraPredFn ref_fn = GetHbdLumaIntraPredFn(ARCH_NA, mode);
    if (ref_fn) {
      IntraPredLumaBuffers<UWORD16> ref_buf = buf;
      IntraPredLumaBuffers<UWORD16> tst_buf = buf;
      ref_fn(ref_buf.ref.data(), ref_buf.src_strd, ref_buf.dst.data(),
             ref_buf.dst_strd, nt, mode_or_flag, bit_depth);
      fn(tst_buf.ref.data(), tst_buf.src_strd, tst_buf.dst.data(),
         tst_buf.dst_strd, nt, mode_or_flag, bit_depth);
      if (!VerifyOutput2D(ref_buf.dst.data(), tst_buf.dst.data(), nt, nt,
                          ref_buf.dst_strd, tst_buf.dst_strd)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(buf.ref.data(), buf.src_strd, buf.dst.data(), buf.dst_strd, nt,
       mode_or_flag, bit_depth);
    benchmark::DoNotOptimize(buf.dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * nt * nt);
  state.SetBytesProcessed(state.iterations() * nt * nt * sizeof(UWORD16));
}

void BM_HbdLumaRefSubstitution(benchmark::State& state,
                               IntraPredConfig config) {
  const int nt = config.nt;
  const IV_ARCH_T arch = config.arch;
  const UWORD8 bit_depth = static_cast<UWORD8>(config.bit_depth);

  HbdLumaRefSubstitutionFn fn = GetHbdLumaRefSubstitutionFn(arch);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  const int src_strd = 64;
  const int total_samples = 4 * nt + 1;
  const UWORD16 max_val = static_cast<UWORD16>((1 << bit_depth) - 1);
  const WORD32 nbr_flags =
      (nt <= 8) ? 0x10180 : ((nt == 16) ? 0x11144 : 0x13333);

  std::vector<UWORD16> top_left(16);
  std::vector<UWORD16> top(2 * nt + 16);
  std::vector<UWORD16> left(2 * nt * src_strd + 16);
  std::vector<UWORD16> dst(total_samples + 16, 0xCD);
  FillRandom(top_left, static_cast<UWORD16>(0), max_val, 111);
  FillRandom(top, static_cast<UWORD16>(0), max_val, 222);
  FillRandom(left, static_cast<UWORD16>(0), max_val, 333);

  if (arch != ARCH_NA) {
    HbdLumaRefSubstitutionFn ref_fn = GetHbdLumaRefSubstitutionFn(ARCH_NA);
    if (ref_fn) {
      std::vector<UWORD16> ref_dst(total_samples + 16, 0xCD);
      std::vector<UWORD16> tst_dst(total_samples + 16, 0xCD);
      ref_fn(top_left.data(), top.data(), left.data(), src_strd, nt, nbr_flags,
             ref_dst.data(), 1, bit_depth);
      fn(top_left.data(), top.data(), left.data(), src_strd, nt, nbr_flags,
         tst_dst.data(), 1, bit_depth);
      if (!VerifyOutput2D(ref_dst.data(), tst_dst.data(), total_samples, 1,
                          total_samples, total_samples)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(top_left.data(), top.data(), left.data(), src_strd, nt, nbr_flags,
       dst.data(), 1, bit_depth);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * total_samples);
  state.SetBytesProcessed(state.iterations() * total_samples * sizeof(UWORD16));
}

void BM_HbdLumaRefFiltering(benchmark::State& state, IntraPredConfig config) {
  const int nt = config.nt;
  const int mode = config.mode;
  const int smoothing_flags = config.disable_boundary_filter;
  const IV_ARCH_T arch = config.arch;
  const UWORD8 bit_depth = static_cast<UWORD8>(config.bit_depth);

  HbdLumaRefFilteringFn fn = GetHbdLumaRefFilteringFn(arch);
  if (!fn) {
    state.SkipWithError("Target function pointer is null");
    return;
  }

  const int total_samples = 4 * nt + 1;
  const UWORD16 max_val = static_cast<UWORD16>((1 << bit_depth) - 1);
  std::vector<UWORD16> src(total_samples + 16);
  std::vector<UWORD16> dst(total_samples + 16, 0xCD);
  FillRandom(src, static_cast<UWORD16>(0), max_val, 12345);
  if (smoothing_flags == 1 && nt == 32) {
    src[0] = 64;
    src[nt] = 128;
    src[2 * nt] = 192;
    src[3 * nt] = 192;
    src[4 * nt] = 192;
  }

  if (arch != ARCH_NA) {
    HbdLumaRefFilteringFn ref_fn = GetHbdLumaRefFilteringFn(ARCH_NA);
    if (ref_fn) {
      std::vector<UWORD16> ref_dst(total_samples + 16, 0xCD);
      std::vector<UWORD16> tst_dst(total_samples + 16, 0xCD);
      ref_fn(src.data(), nt, ref_dst.data(), mode, smoothing_flags, bit_depth);
      fn(src.data(), nt, tst_dst.data(), mode, smoothing_flags, bit_depth);
      if (!VerifyOutput2D(ref_dst.data(), tst_dst.data(), total_samples, 1,
                          total_samples, total_samples)) {
        state.SkipWithError("Output mismatch between SIMD and C reference");
        return;
      }
    }
  }

  for (auto _ : state) {
    fn(src.data(), nt, dst.data(), mode, smoothing_flags, bit_depth);
    benchmark::DoNotOptimize(dst.data());
    benchmark::ClobberMemory();
  }

  state.SetItemsProcessed(state.iterations() * total_samples);
  state.SetBytesProcessed(state.iterations() * total_samples * sizeof(UWORD16));
}

}  // namespace

void RegisterAllBenchmarks() {
  const auto& arches = GetBenchmarkArchitectures();

  for (int bit_depth : {8, 10}) {
    // Register HBD Luma Reference Substitution and Filtering benchmarks
    for (int nt : GetIntraPredBenchmarkSizes()) {
      for (auto arch : arches) {
        if (arch == ARCH_NA || GetHbdLumaRefSubstitutionFn(arch)) {
          std::string arch_name = GetArchName(arch);
          std::string name = "BM_HbdLumaIntraPred/ref_substitution/" +
                             std::to_string(nt) + "x" + std::to_string(nt) +
                             "/" + std::to_string(bit_depth) + "bit/" +
                             arch_name;
          IntraPredConfig cfg{nt, 0, 1, 0, arch, bit_depth};
          benchmark::RegisterBenchmark(name.c_str(),
                                       [cfg](benchmark::State& st) {
                                         BM_HbdLumaRefSubstitution(st, cfg);
                                       })
              ->Unit(benchmark::kNanosecond);
        }
        if (arch == ARCH_NA || GetHbdLumaRefFilteringFn(arch)) {
          std::string arch_name = GetArchName(arch);
          std::string name = "BM_HbdLumaIntraPred/ref_filtering/" +
                             std::to_string(nt) + "x" + std::to_string(nt) +
                             "/" + std::to_string(bit_depth) + "bit/" +
                             arch_name;
          IntraPredConfig cfg{nt, /*mode=*/2, 1, /*smoothing_flags=*/0, arch,
                              bit_depth};
          benchmark::RegisterBenchmark(name.c_str(),
                                       [cfg](benchmark::State& st) {
                                         BM_HbdLumaRefFiltering(st, cfg);
                                       })
              ->Unit(benchmark::kNanosecond);

          if (nt == 32) {
            std::string strong_name =
                "BM_HbdLumaIntraPred/ref_filtering_strong/32x32/" +
                std::to_string(bit_depth) + "bit/" + arch_name;
            IntraPredConfig strong_cfg{nt, /*mode=*/2, 1, /*smoothing_flags=*/1,
                                       arch, bit_depth};
            benchmark::RegisterBenchmark(
                strong_name.c_str(),
                [strong_cfg](benchmark::State& st) {
                  BM_HbdLumaRefFiltering(st, strong_cfg);
                })
                ->Unit(benchmark::kNanosecond);
          }
        }
      }
    }

    // Register HBD Luma Intra Prediction benchmarks
    for (const auto& mode_info : GetIntraPredBenchmarkModes()) {
      for (int nt : GetIntraPredBenchmarkSizes()) {
        for (auto arch : arches) {
          if (arch != ARCH_NA && !GetHbdLumaIntraPredFn(arch, mode_info.mode)) {
            continue;
          }
          std::string arch_name = GetArchName(arch);
          std::string name =
              "BM_HbdLumaIntraPred/" + std::string(mode_info.name) + "/" +
              std::to_string(nt) + "x" + std::to_string(nt) + "/" +
              std::to_string(bit_depth) + "bit/" + arch_name;
          IntraPredConfig cfg{nt,
                              mode_info.mode,
                              /*dst_strd_mul=*/1,
                              /*disable_boundary_filter=*/0,
                              arch,
                              bit_depth};
          benchmark::RegisterBenchmark(name.c_str(), [cfg](
                                                         benchmark::State& st) {
            BM_HbdLumaIntraPred(st, cfg);
          })->Unit(benchmark::kNanosecond);
        }
      }
    }
  }
}
