#pragma once
#include <spdlog/spdlog.h>

#define GIPC_INFO(...)                                                         \
    {                                                                          \
        spdlog::info(__VA_ARGS__);                                             \
    }
#define GIPC_ERROR(...)                                                        \
    {                                                                          \
        spdlog::error(__VA_ARGS__);                                            \
    }
#define GIPC_ERROR_WITH_LOCATION(fmt, ...)                                                   \
    {                                                                                        \
        spdlog::error(fmt ". In {}, {}({})", __VA_ARGS__, __FUNCTION__, __FILE__, __LINE__); \
    }
#define GIPC_WARN(...)                                                         \
    {                                                                          \
        spdlog::warn(__VA_ARGS__);                                             \
    }
#define GIPC_ASSERT(expr, fmt, ...)                                            \
    {                                                                          \
        if(!(expr))                                                            \
        {                                                                      \
            spdlog::error("Assertion failed: {}." fmt, #expr, __VA_ARGS__);    \
            std::abort();                                                      \
        }                                                                      \
    }
#define GIPC_ASSERT_WITH_LOCATION(expr, fmt, ...)                              \
    {                                                                          \
        if(!(expr))                                                            \
        {                                                                      \
            spdlog::error("Assertion failed: {}." fmt ". In {}, {}({})",       \
                          #expr,                                               \
                          __VA_ARGS__,                                         \
                          __FUNCTION__,                                        \
                          __FILE__,                                            \
                          __LINE__);                                           \
            std::abort();                                                      \
        }                                                                      \
    }