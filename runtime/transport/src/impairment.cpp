#include "clusterlm/transport/impairment.hpp"

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <optional>
#include <random>
#include <thread>

namespace clusterlm::transport {
namespace {

using Clock = std::chrono::steady_clock;

Clock::duration to_duration(double seconds) {
  if (!(seconds > 0)) return Clock::duration::zero();
  return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
}
Clock::duration ms_to_duration(double ms) { return to_duration(ms / 1000.0); }

// Synthetic development parameters (see impairment.hpp): not measurements of any real network.
const NetworkConditions kPresets[] = {
    {"unlimited", 0, 0, 0, 1},
    {"gige-simulated", 110e6, 0.15, 0.05, 1},
    {"gige-degraded", 72e6, 0.4, 0.3, 1},
    {"gige-high-latency", 110e6, 2.0, 0.5, 1},
    {"slow-link", 20e6, 3.0, 2.0, 1},
};

constexpr std::size_t kMaxQueuedFrames = 64;
constexpr std::size_t kMaxQueuedBytes = 32u * 1024 * 1024;

void corrupt_frame(Frame& f) {
  if (f.payload.empty()) f.correlation ^= 1;
  else f.payload[f.payload.size() / 2] ^= 0xFF;
}

class ImpairedConnection final : public Connection {
 public:
  ImpairedConnection(std::unique_ptr<Connection> inner, NetworkConditions cond, std::shared_ptr<SimulatedLink> link,
                     std::shared_ptr<FaultInjector> faults)
      : inner_(std::move(inner)),
        cond_(std::move(cond)),
        link_(std::move(link)),
        faults_(std::move(faults)),
        rng_(cond_.seed) {
    sender_ = std::thread([this] { sender_loop(); });
  }

  ~ImpairedConnection() override {
    close();
    if (sender_.joinable()) sender_.join();
  }

  Status send(const Frame& f) override {
    FaultInjector::Decision d;
    if (faults_) d = faults_->on_frame(FaultDirection::kSend, f);
    if (d.close_before) {
      close();
      return make_error(ErrorCode::kUnavailable, "fault injected: connection closed before send");
    }
    Pending p;
    p.frame = f;
    if (d.corrupt) corrupt_frame(p.frame);
    p.close_after = d.close_after;
    p.stall = d.stall;
    p.bytes = kFrameHeaderSize + f.payload.size();

    std::unique_lock<std::mutex> lk(mu_);
    // Backpressure. An empty queue always admits one frame so a frame larger than the byte bound cannot deadlock.
    space_cv_.wait(lk, [&] {
      return closed_.load() || queue_.empty() ||
             (queue_.size() < kMaxQueuedFrames && queued_bytes_ + p.bytes <= kMaxQueuedBytes);
    });
    if (closed_.load()) return make_error(ErrorCode::kUnavailable, "connection closed");

    // The reservation, jitter draw and ordering floor are taken under one lock so delivery order equals send order.
    const Clock::time_point tx_done = link_->reserve(p.bytes);
    Clock::duration extra = to_duration(cond_.latency_ms / 1000.0) + ms_to_duration(d.delay_ms);
    if (cond_.jitter_ms > 0) {
      std::uniform_real_distribution<double> dist(0.0, cond_.jitter_ms);
      extra += ms_to_duration(dist(rng_));
    }
    p.deliver_at = std::max(last_delivery_, tx_done + extra);
    last_delivery_ = p.deliver_at;
    queued_bytes_ += p.bytes;
    queue_.push_back(std::move(p));
    lk.unlock();
    queue_cv_.notify_one();
    return Status::ok();
  }

  Result<Frame> receive(std::chrono::milliseconds timeout) override {
    const Clock::time_point deadline = Clock::now() + timeout;
    if (closed_.load()) return make_error(ErrorCode::kUnavailable, "connection closed");

    if (!held_) {
      if (faults_ && !faults_->wait_unstalled(FaultDirection::kReceive, deadline, closed_))
        return stall_failure();
      auto r = inner_->receive(std::max(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()),
                                        std::chrono::milliseconds(0)));
      if (!r.is_ok()) return r.status();
      Frame f = std::move(r).value();
      if (faults_) {
        const FaultInjector::Decision d = faults_->on_frame(FaultDirection::kReceive, f);
        if (d.close_before) {
          close();
          return make_error(ErrorCode::kUnavailable, "fault injected: connection closed before receive");
        }
        if (d.corrupt) corrupt_frame(f);
        if (d.delay_ms > 0) interruptible_sleep(ms_to_duration(d.delay_ms));
        close_after_receive_ = d.close_after;
      }
      held_ = std::move(f);
    }
    // A stall armed by the held frame keeps it back until cleared; it is kept across a deadline so the next
    // receive() resumes rather than loses it.
    if (faults_ && !faults_->wait_unstalled(FaultDirection::kReceive, deadline, closed_)) return stall_failure();
    Frame out = std::move(*held_);
    held_.reset();
    if (close_after_receive_) {
      close_after_receive_ = false;
      close();
    }
    return out;
  }

  void close() override {
    {
      std::lock_guard<std::mutex> lk(mu_);
      closed_.store(true);
    }
    queue_cv_.notify_all();
    space_cv_.notify_all();
    inner_->close();
  }

  PeerIdentity peer() const override { return inner_->peer(); }
  ConnectionStats stats() const override { return inner_->stats(); }
  void set_max_payload(std::uint32_t m) override { inner_->set_max_payload(m); }

 private:
  struct Pending {
    Frame frame;
    Clock::time_point deliver_at;
    std::size_t bytes = 0;
    bool close_after = false;
    bool stall = false;
  };

  Status stall_failure() const {
    if (closed_.load()) return make_error(ErrorCode::kUnavailable, "connection closed");
    return make_error(ErrorCode::kDeadlineExceeded, "receive stalled by fault injection");
  }

  void interruptible_sleep(Clock::duration d) {
    const Clock::time_point end = Clock::now() + d;
    while (!closed_.load() && Clock::now() < end)
      std::this_thread::sleep_for(std::min<Clock::duration>(end - Clock::now(), std::chrono::milliseconds(5)));
  }

  void sender_loop() {
    std::unique_lock<std::mutex> lk(mu_);
    for (;;) {
      queue_cv_.wait(lk, [&] { return closed_.load() || !queue_.empty(); });
      if (closed_.load()) return;
      // The element stays queued until delivered so the bound reflects frames genuinely in flight.
      const Clock::time_point due = queue_.front().deliver_at;
      if (queue_cv_.wait_until(lk, due, [&] { return closed_.load(); })) return;
      Pending p = std::move(queue_.front());
      lk.unlock();

      Status s = Status::ok();
      if (p.stall && faults_ && !faults_->wait_unstalled(FaultDirection::kSend, Clock::time_point::max(), closed_))
        s = make_error(ErrorCode::kUnavailable, "connection closed");
      if (s.is_ok()) s = inner_->send(p.frame);
      if (s.is_ok() && p.close_after) {
        close();
        return;
      }
      if (!s.is_ok()) {
        close();
        return;
      }

      lk.lock();
      queue_.pop_front();
      queued_bytes_ -= p.bytes;
      space_cv_.notify_all();
    }
  }

  std::unique_ptr<Connection> inner_;
  NetworkConditions cond_;
  std::shared_ptr<SimulatedLink> link_;
  std::shared_ptr<FaultInjector> faults_;

  std::mutex mu_;
  std::condition_variable queue_cv_;  // sender waits for work / due time
  std::condition_variable space_cv_;  // producers wait for queue space
  std::deque<Pending> queue_;
  std::size_t queued_bytes_ = 0;
  Clock::time_point last_delivery_{};
  std::mt19937_64 rng_;
  std::atomic<bool> closed_{false};
  std::thread sender_;

  // Reader-thread state (receive() is single-reader).
  std::optional<Frame> held_;
  bool close_after_receive_ = false;
};

}  // namespace

Result<NetworkConditions> network_preset(std::string_view name) {
  for (const auto& p : kPresets)
    if (p.name == name) return p;
  return make_error(ErrorCode::kNotFound, "unknown network preset '" + std::string(name) + "'");
}

std::vector<std::string> network_preset_names() {
  std::vector<std::string> names;
  for (const auto& p : kPresets) names.push_back(p.name);
  return names;
}

Clock::time_point SimulatedLink::reserve(std::size_t bytes) {
  std::lock_guard<std::mutex> lk(mu_);
  const Clock::time_point now = Clock::now();
  if (bandwidth_ <= 0) return now;
  busy_until_ = std::max(busy_until_, now) + to_duration(static_cast<double>(bytes) / bandwidth_);
  return busy_until_;
}

std::unique_ptr<Connection> impair(std::unique_ptr<Connection> inner, NetworkConditions conditions,
                                   std::shared_ptr<SimulatedLink> link, std::shared_ptr<FaultInjector> faults) {
  if (!link) link = std::make_shared<SimulatedLink>(conditions.bandwidth_bytes_per_s);
  return std::make_unique<ImpairedConnection>(std::move(inner), std::move(conditions), std::move(link),
                                              std::move(faults));
}

}  // namespace clusterlm::transport
