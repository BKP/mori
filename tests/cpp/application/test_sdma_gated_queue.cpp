// Copyright © Advanced Micro Devices, Inc. All rights reserved.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <vector>

#include "mori/application/transport/sdma/anvil.hpp"

namespace {

int Node(int hipDev) { return static_cast<int>(anvil::AnvilLib::nodeForHipDevice(hipDev)); }

int DeviceCount() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess ? n : 0;
}

struct DeviceBuffer {
  void* p = nullptr;
  DeviceBuffer(int dev, size_t bytes) {
    if (hipSetDevice(dev) != hipSuccess || hipMalloc(&p, bytes) != hipSuccess) p = nullptr;
  }
  ~DeviceBuffer() {
    if (p != nullptr) (void)hipFree(p);
  }
};

// Writes `packets` COPY_LINEAR packets splitting src -> dst, rings the doorbell
// from the host, and waits for the engine to consume them.
bool HostCopy(const anvil::SdmaGatedQueueHandle& h, const void* src, void* dst, size_t bytes,
              int packets) {
  const size_t chunk = bytes / packets;
  const uint64_t start = *h.wptr;
  uint64_t at = start;
  for (int k = 0; k < packets; k++) {
    const uintptr_t s = reinterpret_cast<uintptr_t>(src) + k * chunk;
    const uintptr_t d = reinterpret_cast<uintptr_t>(dst) + k * chunk;
    SDMA_PKT_COPY_LINEAR p{};
    p.HEADER_UNION.op = SDMA_OP_COPY;
    p.HEADER_UNION.sub_op = SDMA_SUBOP_COPY_LINEAR;
    p.COUNT_UNION.count = static_cast<uint32_t>(chunk - 1);
    p.SRC_ADDR_LO_UNION.src_addr_31_0 = static_cast<uint32_t>(s);
    p.SRC_ADDR_HI_UNION.src_addr_63_32 = static_cast<uint32_t>(s >> 32);
    p.DST_ADDR_LO_UNION.dst_addr_31_0 = static_cast<uint32_t>(d);
    p.DST_ADDR_HI_UNION.dst_addr_63_32 = static_cast<uint32_t>(d >> 32);
    uint32_t unit[8] = {};  // one 32-byte unit; the zero tail dwords are NOPs
    static_assert(sizeof(p) <= sizeof(unit), "COPY_LINEAR fits one unit");
    std::memcpy(unit, &p, sizeof(p));
    auto* slot = reinterpret_cast<volatile uint32_t*>(reinterpret_cast<char*>(h.ring) +
                                                      (at & (h.ringBytes - 1)));
    for (int i = 0; i < 8; i++) slot[i] = unit[i];
    at += sizeof(unit);
  }
  __atomic_thread_fence(__ATOMIC_SEQ_CST);
  *h.hdpFlush = 1u;
  (void)*h.hdpFlush;  // the read-back completes only after the flush has landed
  __atomic_store_n(h.wptr, at, __ATOMIC_RELEASE);
  __atomic_store_n(h.doorbell, at, __ATOMIC_RELEASE);

  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (__atomic_load_n(h.rptr, __ATOMIC_ACQUIRE) < at) {
    if (std::chrono::steady_clock::now() >= deadline) return false;
  }
  return true;
}

std::vector<uint32_t> Pattern(size_t bytes, uint32_t seed) {
  std::vector<uint32_t> v(bytes / 4);
  for (size_t i = 0; i < v.size(); i++) v[i] = static_cast<uint32_t>((i + seed) * 2654435761u);
  return v;
}

TEST(SdmaGatedQueue, CreateIdle) {
  if (DeviceCount() < 1) GTEST_SKIP() << "no GPU";
  auto q = anvil::anvil.createGatedQueue(Node(0), Node(0));
  ASSERT_NE(q, nullptr);
  const anvil::SdmaGatedQueueHandle h = q->hostHandle();
  EXPECT_NE(h.ring, nullptr);
  EXPECT_EQ(h.ringBytes, anvil::SDMA_QUEUE_SIZE);
  ASSERT_NE(h.rptr, nullptr);
  ASSERT_NE(h.wptr, nullptr);
  EXPECT_NE(h.doorbell, nullptr);
  EXPECT_NE(h.hdpFlush, nullptr);
  EXPECT_EQ(*h.rptr, *h.wptr);
}

TEST(SdmaGatedQueue, RejectsBadArguments) {
  if (DeviceCount() < 1) GTEST_SKIP() << "no GPU";
  const int n = Node(0);
  EXPECT_EQ(anvil::anvil.createGatedQueue(n, n, 2048), nullptr);
  EXPECT_EQ(anvil::anvil.createGatedQueue(n, n, 12288), nullptr);
  EXPECT_NE(anvil::anvil.createGatedQueue(n, n, 4096), nullptr);
}

TEST(SdmaGatedQueue, HostDrivenCopy) {
  if (DeviceCount() < 1) GTEST_SKIP() << "no GPU";
  ASSERT_EQ(hipSetDevice(0), hipSuccess);
  auto q = anvil::anvil.createGatedQueue(Node(0), Node(0));
  ASSERT_NE(q, nullptr);
  const anvil::SdmaGatedQueueHandle h = q->hostHandle();

  constexpr size_t kBytes = 1 << 20;
  const std::vector<uint32_t> want = Pattern(kBytes, 0);
  std::vector<uint32_t> got(want.size(), 0);
  DeviceBuffer src(0, kBytes), dst(0, kBytes);
  ASSERT_NE(src.p, nullptr);
  ASSERT_NE(dst.p, nullptr);
  ASSERT_EQ(hipMemcpy(src.p, want.data(), kBytes, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemset(dst.p, 0, kBytes), hipSuccess);
  ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

  ASSERT_TRUE(HostCopy(h, src.p, dst.p, kBytes, 1)) << "engine did not consume the packet";
  ASSERT_EQ(hipMemcpy(got.data(), dst.p, kBytes, hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(std::memcmp(got.data(), want.data(), kBytes), 0);
}

}  // namespace
