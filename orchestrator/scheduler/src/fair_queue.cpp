#include "clusterlm/scheduler/fair_queue.hpp"

namespace clusterlm::scheduler {

void FairQueue::push(const Entry& e) {
  if (index_.count(e.id)) return;
  index_[e.id] = e;
  Ring& r = rings_[static_cast<std::size_t>(e.prio)];
  Client& c = r.clients[e.client];
  if (c.ids.empty()) {
    c.ring_key = e.id;
    r.ring[e.id] = e.client;
  }
  c.ids.insert(e.id);
}

bool FairQueue::erase(TicketId id) {
  auto it = index_.find(id);
  if (it == index_.end()) return false;
  Ring& r = rings_[static_cast<std::size_t>(it->second.prio)];
  auto cit = r.clients.find(it->second.client);
  cit->second.ids.erase(id);
  if (cit->second.ids.empty()) {
    r.ring.erase(cit->second.ring_key);
    r.clients.erase(cit);
  }
  index_.erase(it);
  return true;
}

std::optional<TicketId> FairQueue::head_of(const Ring& r, const DemotedFn& demoted) {
  if (r.ring.empty()) return std::nullopt;
  // Candidates in round-robin order: keys after the cursor, then wrapping to the start.
  std::vector<const std::string*> order;
  for (auto it = r.ring.upper_bound(r.cursor); it != r.ring.end(); ++it) order.push_back(&it->second);
  for (auto it = r.ring.begin(); it != r.ring.end() && it->first <= r.cursor; ++it) order.push_back(&it->second);
  const std::string* pick = order.front();
  if (demoted) {
    for (const std::string* c : order)
      if (!demoted(*c)) { pick = c; break; }
  }
  return *r.clients.at(*pick).ids.begin();
}

std::optional<TicketId> FairQueue::head(const DemotedFn& demoted) const {
  for (const Ring& r : rings_)
    if (auto h = head_of(r, demoted)) return h;
  return std::nullopt;
}

void FairQueue::served(const std::string& client, Priority prio) {
  Ring& r = rings_[static_cast<std::size_t>(prio)];
  auto it = r.clients.find(client);
  if (it != r.clients.end()) r.cursor = it->second.ring_key;  // call before erase(): the key dies with the client
}

std::size_t FairQueue::depth(Priority p) const {
  std::size_t n = 0;
  for (const auto& kv : rings_[static_cast<std::size_t>(p)].clients) n += kv.second.ids.size();
  return n;
}

std::size_t FairQueue::client_depth(const std::string& client) const {
  std::size_t n = 0;
  for (const Ring& r : rings_) {
    auto it = r.clients.find(client);
    if (it != r.clients.end()) n += it->second.ids.size();
  }
  return n;
}

std::vector<TicketId> FairQueue::order(const DemotedFn& demoted) const {
  std::vector<TicketId> out;
  FairQueue copy = *this;
  while (auto h = copy.head(demoted)) {
    const Entry e = copy.index_.at(*h);
    out.push_back(*h);
    copy.served(e.client, e.prio);
    copy.erase(*h);
  }
  return out;
}

}  // namespace clusterlm::scheduler
