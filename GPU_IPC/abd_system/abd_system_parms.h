#pragma once
#include <gipc/type_define.h>
namespace gipc
{
class ABDSystemParms
{
  public:
    Vector12 init_q_v     = Vector12::Zero();
    Vector3  gravity      = Vector3{0, -9.8, 0};  // m/s^2
    Float    dt           = 0.01;                 // s
    Float    mass_density = 1e3;                  // kg/m^3
    Float    kappa        = 1e8;
    Float    motor_speed  = 31.4;  // rad/s
    Float    motor_strength = 10; // how strong the motor is, related to the body mass
};
}  // namespace gipc