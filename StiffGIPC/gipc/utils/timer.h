#pragma once
#include <stack>
#include <memory>
#include <list>
#include <set>
#include <unordered_map>
#include <iomanip>
#include <string>
#include <chrono>
#include <gipc/utils/json.h>
#include <iostream>

namespace gipc
{
class GlobalTimer;
class Timer;
}  // namespace gipc

namespace gipc::details
{
class ScopedTimer
{
  public:
    using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
    using Duration = std::chrono::duration<double>;

  private:
    friend class gipc::GlobalTimer;
    friend class gipc::Timer;
    std::string             name;
    std::string             full_name;
    TimePoint               start;
    TimePoint               end;
    Duration                duration;
    std::list<ScopedTimer*> children;
    ScopedTimer*            parent = nullptr;
    size_t                  depth  = 0;

    ScopedTimer(std::string_view name)
        : name(name)
        , duration(0)
    {
    }

    void   tick();
    void   tock();
    double elapsed() const;
    void   traverse(Json& j);
    void   setup_full_name();

  public:
    ~ScopedTimer() = default;
};
}  // namespace gipc::details

namespace gipc
{
class Timer
{
  public:
    Timer(std::string_view blockName, bool force_on = false);
    ~Timer();

    double elapsed() const;
    static void disable_all() { m_global_on = false; }
    static void enable_all() { m_global_on = true; }
  private:
    details::ScopedTimer* m_timer = nullptr;
    bool                  m_force_on;
    static bool           m_global_on;
};

class GlobalTimer
{
    using STimer = details::ScopedTimer;
    template <typename T>
    using U = std::unique_ptr<T>;

    std::stack<STimer*> m_timer_stack;
    STimer*             m_root;
    friend class ScopedTimer;

    std::list<U<STimer>> m_timers;

    friend class Timer;
    STimer& push_timer(std::string_view);
    STimer& pop_timer();

    thread_local static GlobalTimer default_instance;
    static GlobalTimer*             m_current;

    void _print_timings(std::ostream& o, const STimer* timer, int depth);

    size_t max_full_name_length() const;
    size_t max_depth() const;

    struct MergeResult
    {

        std::string name;
        std::string parent_full_name;
        std::string parent_name;

        double                  duration = 0.0;
        size_t                  count    = 0;
        std::list<MergeResult*> children;

        size_t depth = 0;
    };

    std::unordered_map<std::string, U<MergeResult>> m_merge_timers;
    MergeResult*                                    m_merge_root = nullptr;

    void merge_timers();
    void _print_merged_timings(std::ostream&      o,
                               const MergeResult* timer,
                               size_t             max_name_length,
                               size_t             max_depth);
    void _traverse_merge_timers(Json& j, const MergeResult* timer);

  public:
    GlobalTimer(std::string_view name = "GlobalTimer");
    ~GlobalTimer();

    void                set_as_current();
    static GlobalTimer* current();
    Json                report_as_json();
    Json                report_merged_as_json();

    void print_timings(std::ostream& o = std::cout);

    void print_merged_timings(std::ostream& o = std::cout);

    void clear();
};
}  // namespace gipc
