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
#include <set>
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
  EXPECT_EQ(anvil::anvil.createGatedQueue(n, n, 0, 2048), nullptr);
  EXPECT_EQ(anvil::anvil.createGatedQueue(n, n, 0, 12288), nullptr);
  EXPECT_EQ(anvil::anvil.createGatedQueue(n, n, -1), nullptr);
  EXPECT_NE(anvil::anvil.createGatedQueue(n, n, 0, 4096), nullptr);
}

TEST(SdmaGatedQueue, EngineOffsetMovesTheQueue) {
  if (DeviceCount() < 2) GTEST_SKIP() << "needs two GPUs";
  auto q0 = anvil::anvil.createGatedQueue(Node(0), Node(1), 0);
  ASSERT_NE(q0, nullptr);
  auto q1 = anvil::anvil.createGatedQueue(Node(0), Node(1), 1);
  ASSERT_NE(q1, nullptr);
  EXPECT_NE(q1->hostHandle().engineId, q0->hostHandle().engineId);
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

// Host-rung queues on every engine of the link's class, odd ids included (device
// doorbells avoid those on MI308X). Peer link 0 -> 1 if there are two GPUs.
TEST(SdmaGatedQueue, HostDrivenCopyOnEveryEngine) {
  if (DeviceCount() < 1) GTEST_SKIP() << "no GPU";
  const int peer = DeviceCount() >= 2 ? 1 : 0;
  if (peer != 0) {
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    const hipError_t e = hipDeviceEnablePeerAccess(peer, 0);
    (void)hipGetLastError();
    if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled)
      GTEST_SKIP() << "no peer access 0 -> " << peer;
  }

  constexpr size_t kBytes = 1 << 20;
  constexpr int kPackets = 4;
  DeviceBuffer src(0, kBytes), dst(peer, kBytes);
  ASSERT_NE(src.p, nullptr);
  ASSERT_NE(dst.p, nullptr);
  ASSERT_EQ(hipSetDevice(0), hipSuccess);

  std::set<uint32_t> engines;
  int odd = 0;
  for (int offset = 0; offset < 32; offset++) {
    auto q = anvil::anvil.createGatedQueue(Node(0), Node(peer), offset);
    ASSERT_NE(q, nullptr) << "offset " << offset;
    const anvil::SdmaGatedQueueHandle h = q->hostHandle();
    if (!engines.insert(h.engineId).second) break;  // wrapped around the class
    odd += h.engineId & 1u;

    const std::vector<uint32_t> want = Pattern(kBytes, h.engineId);
    std::vector<uint32_t> got(want.size(), 0);
    ASSERT_EQ(hipMemcpy(src.p, want.data(), kBytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemset(dst.p, 0, kBytes), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    ASSERT_TRUE(HostCopy(h, src.p, dst.p, kBytes, kPackets)) << "engine " << h.engineId << " hung";
    ASSERT_EQ(hipMemcpy(got.data(), dst.p, kBytes, hipMemcpyDeviceToHost), hipSuccess);
    EXPECT_EQ(std::memcmp(got.data(), want.data(), kBytes), 0) << "engine " << h.engineId;
  }
  RecordProperty("engines", static_cast<int>(engines.size()));
  RecordProperty("odd_engines", odd);
  EXPECT_GT(odd, 0) << "no odd engine reached";
}

TEST(SdmaGatedQueue, ConnectFailsInsteadOfExiting) {
  if (DeviceCount() < 2) GTEST_SKIP() << "needs two GPUs";
  const int n = Node(0), p = Node(1);
  std::vector<std::unique_ptr<anvil::SdmaGatedQueue>> held;
  while (held.size() < 64) {
    auto q = anvil::anvil.createGatedQueue(n, p);
    if (q == nullptr) break;
    held.push_back(std::move(q));
  }
  ASSERT_FALSE(held.empty());
  if (anvil::anvil.connect(n, p, 1))
    GTEST_SKIP() << "connect() used another engine (MI308X balances peer links)";
  EXPECT_EQ(anvil::anvil.getSdmaQueue(n, p, 0), nullptr);
  held.pop_back();
  EXPECT_TRUE(anvil::anvil.connect(n, p, 1));
  EXPECT_NE(anvil::anvil.getSdmaQueue(n, p, 0), nullptr);
}

TEST(SdmaDoorbell, ClassifiesOddEngineDoorbells) {
  using anvil::ClassifyOddEngineDoorbell;
  using anvil::OddEngineDoorbell;
  EXPECT_EQ(anvil::DoorbellIndex(reinterpret_cast<void*>(0x7f00000008a0)), 276u);
  EXPECT_EQ(ClassifyOddEngineDoorbell(2, 276, 3, 286), OddEngineDoorbell::kMisrouted);
  EXPECT_EQ(ClassifyOddEngineDoorbell(2, 277, 3, 287), OddEngineDoorbell::kMisrouted);
  EXPECT_EQ(ClassifyOddEngineDoorbell(4, 296, 3, 286), OddEngineDoorbell::kMisrouted);
  EXPECT_EQ(ClassifyOddEngineDoorbell(2, 276, 3, 288), OddEngineDoorbell::kRouted);
  EXPECT_EQ(ClassifyOddEngineDoorbell(2, 276, 3, 289), OddEngineDoorbell::kRouted);
  EXPECT_EQ(ClassifyOddEngineDoorbell(2, 276, 3, 290), OddEngineDoorbell::kFixed);
  EXPECT_EQ(ClassifyOddEngineDoorbell(2, 276, 3, 291), OddEngineDoorbell::kFixed);
  EXPECT_EQ(ClassifyOddEngineDoorbell(14, 506, 15, 2), OddEngineDoorbell::kMisrouted);
  EXPECT_EQ(ClassifyOddEngineDoorbell(14, 506, 15, 6), OddEngineDoorbell::kFixed);
  EXPECT_EQ(ClassifyOddEngineDoorbell(2, 276, 3, 300), OddEngineDoorbell::kUnknown);
}

__global__ void DevicePut(anvil::SdmaQueueDeviceHandle* h, void* dst, void* src, size_t bytes) {
  anvil::SdmaQueueDeviceHandle handle = *h;
  anvil::put(handle, dst, src, bytes);
}

// One channel from GPU 0 to every peer; a kernel copies through each. Odd engines appear
// only if deviceQueuesUseOddEngines().
TEST(SdmaDeviceQueue, FollowsTheDoorbellVerdict) {
  const int count = DeviceCount();
  if (count < 2) GTEST_SKIP() << "needs two GPUs";
  const int n = Node(0);
  const bool odd = anvil::anvil.deviceQueuesUseOddEngines(n);
  RecordProperty("odd_engines_allowed", odd ? 1 : 0);

  constexpr size_t kBytes = 1 << 20;
  DeviceBuffer src(0, kBytes);
  ASSERT_NE(src.p, nullptr);
  int oddUsed = 0;
  for (int peer = 1; peer < count; peer++) {
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    const hipError_t e = hipDeviceEnablePeerAccess(peer, 0);
    (void)hipGetLastError();
    ASSERT_TRUE(e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled) << "peer " << peer;
    ASSERT_TRUE(anvil::anvil.connect(n, Node(peer), 1)) << "peer " << peer;
    anvil::SdmaQueue* q = anvil::anvil.getSdmaQueue(n, Node(peer), 0);
    ASSERT_NE(q, nullptr);
    const uint32_t engine = q->engineId();
    if (!odd) EXPECT_EQ(engine & 1u, 0u) << "peer " << peer;
    oddUsed += engine & 1u;

    DeviceBuffer dst(peer, kBytes);
    ASSERT_NE(dst.p, nullptr);
    ASSERT_EQ(hipMemset(dst.p, 0, kBytes), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    const std::vector<uint32_t> want = Pattern(kBytes, engine);
    ASSERT_EQ(hipMemcpy(src.p, want.data(), kBytes, hipMemcpyHostToDevice), hipSuccess);
    DevicePut<<<1, 1>>>(q->deviceHandle(), dst.p, src.p, kBytes);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // A misrouted doorbell leaves the copy until the engine reads its write pointer on its
    // own, hundreds of ms later; a delivered one lands in well under a millisecond.
    const auto start = std::chrono::steady_clock::now();
    std::vector<uint32_t> got(want.size(), 0);
    bool same = false;
    while (!same && std::chrono::steady_clock::now() < start + std::chrono::seconds(5)) {
      ASSERT_EQ(hipMemcpy(got.data(), dst.p, kBytes, hipMemcpyDeviceToHost), hipSuccess);
      same = std::memcmp(got.data(), want.data(), kBytes) == 0;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    EXPECT_TRUE(same) << "engine " << engine << " (peer " << peer << ") ran no device-rung copy";
    EXPECT_LT(ms, 50.0) << "engine " << engine << " (peer " << peer
                        << "): the doorbell did not reach the engine";
  }
  RecordProperty("odd_engines_used", oddUsed);
}

}  // namespace
