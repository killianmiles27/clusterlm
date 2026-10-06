#pragma once
// ThreadTeam: a fixed set of worker threads that execute one batch of tasks at a time. The calling thread is
// worker 0, so a team of size N uses N-1 extra threads. Tasks are claimed from an atomic counter, which
// gives separate tasks (experts) to separate threads without static partitioning.
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace clusterlm::bench {

class ThreadTeam {
 public:
  explicit ThreadTeam(unsigned size) : size_(size == 0 ? 1 : size) {
    for (unsigned w = 1; w < size_; ++w) threads_.emplace_back([this, w] { loop(w); });
  }
  ~ThreadTeam() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& t : threads_) t.join();
  }
  ThreadTeam(const ThreadTeam&) = delete;
  ThreadTeam& operator=(const ThreadTeam&) = delete;

  unsigned size() const { return size_; }

  // Runs fn(task, worker) for task in [0, tasks) and returns when every task has completed.
  void run(std::size_t tasks, const std::function<void(std::size_t, unsigned)>& fn) {
    if (size_ == 1 || tasks <= 1) {
      for (std::size_t t = 0; t < tasks; ++t) fn(t, 0);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      fn_ = &fn;
      tasks_ = tasks;
      next_.store(0);
      active_ = size_ - 1;
      ++generation_;
    }
    cv_.notify_all();
    drain(0);
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [this] { return active_ == 0; });
    fn_ = nullptr;
  }

 private:
  void drain(unsigned worker) {
    while (true) {
      const std::size_t t = next_.fetch_add(1);
      if (t >= tasks_) return;
      (*fn_)(t, worker);
    }
  }
  void loop(unsigned worker) {
    std::uint64_t seen = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) return;
        seen = generation_;
      }
      drain(worker);
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (--active_ == 0) done_cv_.notify_one();
      }
    }
  }

  unsigned size_;
  std::vector<std::thread> threads_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(std::size_t, unsigned)>* fn_ = nullptr;
  std::size_t tasks_ = 0;
  std::atomic<std::size_t> next_{0};
  unsigned active_ = 0;
  std::uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace clusterlm::bench
