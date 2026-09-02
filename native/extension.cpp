#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

#include "mlx/mlx.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {

mx::Dtype dtype_from_name(const std::string& name) {
  if (name == "uint32") return mx::uint32;
  if (name == "bfloat16") return mx::bfloat16;
  throw std::invalid_argument("unsupported owned-pool dtype: " + name);
}

mx::array owned_zeros(const std::vector<int>& shape, const std::string& dtype_name) {
  const mx::Dtype dtype = dtype_from_name(dtype_name);
  const mx::Shape mlx_shape(shape.begin(), shape.end());
  size_t elements = 1;
  for (int dimension : shape) {
    if (dimension <= 0) throw std::invalid_argument("pool dimensions must be positive");
    elements *= static_cast<size_t>(dimension);
  }
  const size_t nbytes = elements * static_cast<size_t>(mx::size_of(dtype));
  auto buffer = mx::allocator::malloc(nbytes);
  std::memset(buffer.raw_ptr(), 0, nbytes);
  return mx::array(buffer, mlx_shape, dtype);
}

uintptr_t data_pointer(const mx::array& input) {
  mx::array array = input;
  array.eval();
  return reinterpret_cast<uintptr_t>(array.data<uint8_t>());
}

long copy_record_into(
    const std::vector<mx::array>& pools,
    int slot,
    const std::vector<long>& segment_bytes,
    const nb::bytes& record) {
  if (slot < 0) throw std::invalid_argument("slot must be non-negative");
  if (pools.size() != segment_bytes.size()) {
    throw std::invalid_argument("pool and segment counts differ");
  }
  size_t expected = 0;
  for (long bytes : segment_bytes) {
    if (bytes <= 0) throw std::invalid_argument("segment bytes must be positive");
    expected += static_cast<size_t>(bytes);
  }
  if (record.size() != expected) {
    throw std::invalid_argument(
        "staged expert size differs from packed record size");
  }
  const auto* source = static_cast<const uint8_t*>(record.data());
  size_t source_offset = 0;
  for (size_t index = 0; index < pools.size(); ++index) {
    mx::array pool = pools[index];
    pool.eval();
    const size_t bytes = static_cast<size_t>(segment_bytes[index]);
    if (bytes * static_cast<size_t>(slot + 1) > pool.nbytes()) {
      throw std::invalid_argument("slot exceeds owned pool bounds");
    }
    std::memcpy(
        pool.data<uint8_t>() + static_cast<size_t>(slot) * bytes,
        source + source_offset,
        bytes);
    source_offset += bytes;
  }
  return static_cast<long>(expected);
}

class PackReader {
 public:
  PackReader(const std::string& path, bool nocache) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed for " + path + ": " + std::strerror(errno));
    if (nocache && ::fcntl(fd_, 48, 1) < 0) {
      const std::string message = "F_NOCACHE failed for " + path + ": " + std::strerror(errno);
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error(message);
    }
  }

  PackReader(const PackReader&) = delete;
  PackReader& operator=(const PackReader&) = delete;

  ~PackReader() {
    if (fd_ >= 0) ::close(fd_);
  }

  long read_into(
      const std::vector<mx::array>& pools,
      int slot,
      long file_offset,
      const std::vector<long>& segment_bytes) {
    if (slot < 0) throw std::invalid_argument("slot must be non-negative");
    if (pools.size() != segment_bytes.size()) {
      throw std::invalid_argument("pool and segment counts differ");
    }
    std::vector<struct iovec> vectors;
    vectors.reserve(pools.size());
    long expected = 0;
    for (size_t index = 0; index < pools.size(); ++index) {
      mx::array pool = pools[index];
      pool.eval();
      const long bytes = segment_bytes[index];
      if (bytes <= 0 || static_cast<size_t>(bytes) * static_cast<size_t>(slot + 1) > pool.nbytes()) {
        throw std::invalid_argument("slot exceeds owned pool bounds");
      }
      vectors.push_back({
          pool.data<uint8_t>() + static_cast<size_t>(slot) * static_cast<size_t>(bytes),
          static_cast<size_t>(bytes),
      });
      expected += bytes;
    }
    ssize_t count;
    do {
      count = ::preadv(fd_, vectors.data(), static_cast<int>(vectors.size()), file_offset);
    } while (count < 0 && errno == EINTR);
    if (count < 0) {
      throw std::runtime_error("preadv failed for " + path_ + ": " + std::strerror(errno));
    }
    if (count != expected) {
      throw std::runtime_error(
          "short preadv for " + path_ + ": expected " + std::to_string(expected) +
          ", got " + std::to_string(count));
    }
    return static_cast<long>(count);
  }

 private:
  std::string path_;
  int fd_ = -1;
};

}  // namespace

NB_MODULE(_vference_native, module) {
  nb::module_::import_("mlx.core");
  module.doc() = "Native stable-slot storage primitives for vference.";
  module.def("owned_zeros", &owned_zeros, "shape"_a, "dtype"_a);
  module.def("data_pointer", &data_pointer, "array"_a);
  module.def(
      "copy_record_into",
      &copy_record_into,
      "pools"_a,
      "slot"_a,
      "segment_bytes"_a,
      "record"_a);
  nb::class_<PackReader>(module, "PackReader")
      .def(nb::init<const std::string&, bool>(), "path"_a, "nocache"_a = false)
      .def(
          "read_into",
          &PackReader::read_into,
          "pools"_a,
          "slot"_a,
          "file_offset"_a,
          "segment_bytes"_a,
          nb::call_guard<nb::gil_scoped_release>());
}
