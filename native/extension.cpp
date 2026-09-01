#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>
#include <zlib.h>

#include "mlx/mlx.h"

namespace nb = nanobind;
namespace mx = mlx::core;
using namespace nb::literals;

namespace {

std::mutex owned_mutex;
std::vector<mx::allocator::Buffer> owned_buffers;

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
  {
    std::lock_guard<std::mutex> lock(owned_mutex);
    owned_buffers.push_back(buffer);
  }
  return mx::array(buffer, mlx_shape, dtype, [](mx::allocator::Buffer) {});
}

uintptr_t data_pointer(const mx::array& input) {
  mx::array array = input;
  array.eval();
  return reinterpret_cast<uintptr_t>(array.data<uint8_t>());
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
      const std::vector<long>& segment_bytes,
      long expected_crc32) {
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
    if (expected_crc32 >= 0) {
      uLong checksum = ::crc32(0L, Z_NULL, 0);
      for (const auto& vector : vectors) {
        checksum = ::crc32(
            checksum,
            static_cast<const Bytef*>(vector.iov_base),
            static_cast<uInt>(vector.iov_len));
      }
      if (checksum != static_cast<uLong>(expected_crc32)) {
        throw std::runtime_error(
            "expert CRC32 mismatch for " + path_ + " at offset " +
            std::to_string(file_offset));
      }
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
  nb::class_<PackReader>(module, "PackReader")
      .def(nb::init<const std::string&, bool>(), "path"_a, "nocache"_a = false)
      .def(
          "read_into",
          &PackReader::read_into,
          "pools"_a,
          "slot"_a,
          "file_offset"_a,
          "segment_bytes"_a,
          "expected_crc32"_a = -1);
}
