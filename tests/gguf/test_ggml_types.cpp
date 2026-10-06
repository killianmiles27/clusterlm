#include <doctest/doctest.h>

#include <cmath>
#include <limits>

#include "clusterlm/objects/ggml_types.hpp"
#include "clusterlm/objects/tensor_codec.hpp"

using namespace clusterlm;
using namespace clusterlm::objects;

namespace {
struct Expect {
  std::uint32_t id;
  const char* name;
  std::uint32_t elems, bytes;
};
// Verified against the pinned llama.cpp (6753a033f058fbf778d282556ed9b16c78de7c71): ggml.c type_traits for names
// and block sizes, sizeof(block_*) from ggml-common.h for bytes per block.
constexpr Expect kPinned[] = {
    {0, "f32", 1, 4},      {1, "f16", 1, 2},      {2, "q4_0", 32, 18},    {3, "q4_1", 32, 20},    {6, "q5_0", 32, 22},
    {7, "q5_1", 32, 24},   {8, "q8_0", 32, 34},   {9, "q8_1", 32, 36},    {10, "q2_k", 256, 84},  {11, "q3_k", 256, 110},
    {12, "q4_k", 256, 144}, {13, "q5_k", 256, 176}, {14, "q6_k", 256, 210}, {15, "q8_k", 256, 292}, {16, "iq2_xxs", 256, 66},
    {17, "iq2_xs", 256, 74}, {18, "iq3_xxs", 256, 98}, {19, "iq1_s", 256, 50}, {20, "iq4_nl", 32, 18}, {21, "iq3_s", 256, 110},
    {22, "iq2_s", 256, 82}, {23, "iq4_xs", 256, 136}, {24, "i8", 1, 1},   {25, "i16", 1, 2},     {26, "i32", 1, 4},
    {27, "i64", 1, 8},     {28, "f64", 1, 8},     {29, "iq1_m", 256, 56}, {30, "bf16", 1, 2},     {34, "tq1_0", 256, 54},
    {35, "tq2_0", 256, 66}, {39, "mxfp4", 32, 17}, {40, "nvfp4", 64, 36}, {41, "q1_0", 128, 18},  {42, "q2_0", 64, 18},
};
}  // namespace

TEST_CASE("ggml type table matches the pinned llama.cpp values") {
  for (const Expect& e : kPinned) {
    INFO(e.name);
    const GgmlTypeInfo* t = ggml_type_info(e.id);
    REQUIRE(t != nullptr);
    CHECK(t->name == e.name);
    CHECK(t->block_elems == e.elems);
    CHECK(t->block_bytes == e.bytes);
    CHECK(static_cast<std::uint32_t>(t->type) == e.id);
    CHECK(t->quantized == (e.elems > 1));
    CHECK(ggml_type_by_name(e.name) == t);
  }
  CHECK(all_ggml_types().size() == std::size(kPinned));
  // The sizes called out in the task, checked explicitly.
  CHECK(ggml_type_info(GgmlType::kIQ3_S).block_bytes == 110);
  CHECK(ggml_type_info(GgmlType::kIQ2_XS).block_bytes == 74);
  CHECK(ggml_type_info(GgmlType::kQ8_0).block_bytes == 34);
}

TEST_CASE("unknown and removed ggml type ids are errors, never guessed") {
  for (std::uint32_t id : {4u, 5u, 31u, 32u, 33u, 36u, 37u, 38u, 43u, 44u, 1000u, 0xFFFFFFFFu}) CHECK(ggml_type_info(id) == nullptr);
  CHECK(ggml_type_by_name("q9_9") == nullptr);
  CHECK(ggml_type_by_name("") == nullptr);
  CHECK(ggml_type_by_name("IQ3_S") == ggml_type_by_name("iq3_s"));
  CHECK(ggml_type_by_name("Q4_K")->type == GgmlType::kQ4_K);
}

TEST_CASE("row and tensor byte sizes are exact and overflow-safe") {
  CHECK(*ggml_row_bytes(GgmlType::kIQ3_S, 256) == 110);
  CHECK(*ggml_row_bytes(GgmlType::kIQ3_S, 2560) == 1100);
  CHECK(*ggml_row_bytes(GgmlType::kF32, 7) == 28);
  CHECK_FALSE(ggml_row_bytes(GgmlType::kIQ3_S, 255).is_ok());
  CHECK_FALSE(ggml_row_bytes(GgmlType::kQ8_0, 33).is_ok());
  const std::uint64_t d3[] = {2560, 640, 512};
  CHECK(*ggml_tensor_bytes(GgmlType::kQ8_0, d3) == 2560 / 32 * 34ull * 640 * 512);
  const std::uint64_t zero[] = {256, 0};
  CHECK_FALSE(ggml_tensor_bytes(GgmlType::kF32, zero).is_ok());
  const std::uint64_t huge[] = {std::numeric_limits<std::uint64_t>::max() / 2, 1000, 1000};
  const auto r = ggml_tensor_bytes(GgmlType::kF32, huge);
  CHECK_FALSE(r.is_ok());
  const std::uint64_t huge_row[] = {std::numeric_limits<std::uint64_t>::max() / 2};  // x 4 bytes/element overflows
  CHECK(ggml_tensor_bytes(GgmlType::kF32, huge_row).status().code() == ErrorCode::kOutOfRange);
  CHECK_FALSE(ggml_tensor_bytes(GgmlType::kF32, std::span<const std::uint64_t>{}).is_ok());
}

TEST_CASE("binary16 conversion round-trips and rounds to nearest even") {
  for (float f : {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 65504.0f, 6.103515625e-05f /*min normal*/, 5.960464477539063e-08f /*min subnormal*/, 3.14159f, -123.456f, 1e-4f}) {
    const float back = half_to_float(float_to_half(f));
    if (std::fabs(f) < 65520.0f) CHECK(std::fabs(back - f) <= std::fabs(f) * 1.0f / 1024.0f + 6e-8f);
  }
  CHECK(float_to_half(1.0f) == 0x3C00);
  CHECK(float_to_half(-2.0f) == 0xC000);
  CHECK(float_to_half(65504.0f) == 0x7BFF);
  CHECK(float_to_half(1e10f) == 0x7C00);
  CHECK(half_to_float(0x7C00) == std::numeric_limits<float>::infinity());
  CHECK(std::isnan(half_to_float(0x7E00)));
  CHECK(half_to_float(0x0001) == doctest::Approx(5.960464477539063e-08f));
  // 1 + 2^-11 is a tie between 1.0 and 1 + 2^-10: ties-to-even picks 1.0; 1 + 3*2^-11 picks 1 + 2^-9... (0x3C02).
  CHECK(float_to_half(1.0f + 1.0f / 2048.0f) == 0x3C00);
  CHECK(float_to_half(1.0f + 3.0f / 2048.0f) == 0x3C02);
  for (std::uint32_t h = 0; h < 0x7C00; h += 7) CHECK(float_to_half(half_to_float(static_cast<std::uint16_t>(h))) == h);
}

TEST_CASE("ggml q8_0 codec: sizes, decode matches the definition, error bound") {
  CHECK(tensor_bytes("q8_0", 64) == 68);
  CHECK(tensor_bytes("q8_0", 33) == 0);
  std::vector<float> v(96);
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = std::sin(static_cast<float>(i) * 0.37f) * (1.0f + static_cast<float>(i / 32));
  Bytes enc;
  REQUIRE(encode_q8_0_ggml(v, enc).is_ok());
  CHECK(enc.size() == 3 * 34);
  std::vector<float> dec(v.size());
  REQUIRE(decode_tensor("q8_0", enc, dec).is_ok());
  for (std::size_t b = 0; b < 3; ++b) {
    float amax = 0;
    for (std::size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[b * 32 + i]));
    for (std::size_t i = 0; i < 32; ++i) CHECK(std::fabs(dec[b * 32 + i] - v[b * 32 + i]) <= amax / 127.0f * 0.51f + amax * 1e-3f);
  }
  CHECK(decode_tensor("q8_0", ByteSpan(enc).subspan(0, 67), dec).code() == ErrorCode::kDataLoss);
  CHECK_FALSE(encode_q8_0_ggml(std::span<const float>(v).subspan(0, 33), enc).is_ok());
}
