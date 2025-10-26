#pragma once
#include <Eigen/Core>
#include <gipc/type_define.h>
namespace gipc
{
class ContactGradient
{
  public:
    I32     point_id;
    Vector3 gradient;
};

class ContactHessian
{
  public:
    Vector2i  point_id;
    Matrix3x3 hessian;
};
}  // namespace gipc
