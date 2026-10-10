#pragma once
// Bounded-fairness queue: one round-robin ring of clients per priority class, FIFO inside each client. Pure data
// structure (no locks, no clock): the scheduler calls it under its lock. Tie-breaks use the monotonic ticket sequence
// number only — never a timestamp or a string compare — so equal enqueue times cannot reorder anything.
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "clusterlm/scheduler/backend.hpp"
#include "clusterlm/scheduler/types.hpp"

namespace clusterlm::scheduler {

class FairQueue {
 public:
  struct Entry { TicketId id = 0; std::string client; Priority prio = Priority::kApi; };
  using DemotedFn = std::function<bool(const std::string& client)>;

  void push(const Entry& e);
  bool erase(TicketId id);
  bool contains(TicketId id) const { return index_.count(id) != 0; }
  // The ticket dispatch must consider next. A demoted client is skipped while any other client in the class is not.
  std::optional<TicketId> head(const DemotedFn& demoted) const;
  // Advance the class's round-robin cursor past `client` (call BEFORE erase(): the ring key disappears with the client's last ticket).
  void served(const std::string& client, Priority prio);
  std::size_t depth() const { return index_.size(); }
  std::size_t depth(Priority p) const;
  std::size_t client_depth(const std::string& client) const;
  // Complete dispatch order if nothing else changed (class by class, round robin), for queue positions.
  std::vector<TicketId> order(const DemotedFn& demoted) const;

 private:
  struct Client { std::set<TicketId> ids; TicketId ring_key = 0; };
  struct Ring {
    std::map<std::string, Client> clients;
    std::map<TicketId, std::string> ring;   // ring_key -> client, the order of the round robin
    TicketId cursor = 0;                    // ring key of the client served last
  };
  static std::optional<TicketId> head_of(const Ring& r, const DemotedFn& demoted);
  Ring rings_[kPriorityCount];
  std::map<TicketId, Entry> index_;
};

}  // namespace clusterlm::scheduler
