#ifndef _MONKE_RAY_H__
#define _MONKE_RAY_H__
#include "Core.h"
#include "Vector3f.h"

namespace monke
{

class Ray
{
public:
    __both__ Ray() = default;
    __both__ Ray(const Vector3f& position, const Vector3f& direction);
    
    __both__ const Vector3f& GetPosition() const;
    __both__ const Vector3f& GetDirection() const;
    __both__ const Vector3f GetPointOnRay(float t) const;
    __both__ inline void SetPosition(const Vector3f& position) { m_Position = position; }
    __both__ inline void SetDirection(const Vector3f& direction) { m_Direction = direction; }
private:
    Vector3f m_Position, m_Direction;
};


/* -------------- */
Ray::Ray(const Vector3f& position, const Vector3f& direction)
: m_Position(position), m_Direction(direction)
{
    
}

const Vector3f Ray::GetPointOnRay(float t) const
{
    return m_Position + m_Direction * t;
}

const Vector3f& Ray::GetPosition() const
{
    return m_Position;
}

const Vector3f& Ray::GetDirection() const
{
    return m_Direction;
}
/* -------------- */

} // end of monke

#endif