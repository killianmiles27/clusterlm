#pragma once
// ProvisionedObject / ObjectResolver: how an execution domain reaches the bytes it owns.
//
// Every pointer an ObjectResolver hands out is process-local and valid only while the owning store (Father's
// canonical model reader or a Node's lease store) keeps the object alive. No remote address ever pretends to
// be a pointer.
#include <cstdint>
#include <span>
#include <string_view>

#include "clusterlm/common/status.hpp"
#include "clusterlm/objects/manifest.hpp"

namespace clusterlm::objects {

struct ProvisionedObject {
  const ManifestObject* entry = nullptr;   // manifest description (owned by the resolver's manifest)
  std::span<const std::uint8_t> bytes;     // validated object bytes in process-local memory
  AllocationTarget target = AllocationTarget::kCpuResident;
};

class ObjectResolver {
 public:
  virtual ~ObjectResolver() = default;
  // Returns kNotFound if the object is not provisioned to this domain. Backends must treat that as a fatal
  // plan error — never as a cue to fetch weights from elsewhere during execution.
  virtual Result<ProvisionedObject> resolve(std::string_view name) const = 0;
};

}  // namespace clusterlm::objects
