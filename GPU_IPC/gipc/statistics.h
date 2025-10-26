#pragma once
#include <gipc/utils/json.h>
namespace gipc
{
class Statistics
{
  private:
    Json   m_json;
    size_t m_frame = 0;
    Statistics();

  public:
    auto&              json() { return m_json; }
    static Statistics& instance()
    {
        thread_local static Statistics instance;
        return instance;
    }
    auto& at_frame(int i) { return m_json["frames"][i]; }
    auto  frame(int i) { m_frame = i; }
    auto  frame() { return m_frame; }
    auto& at_current_frame() { return m_json["frames"][m_frame]; }
    void write_to_file(const std::string& filename);
};
}  // namespace gipc
