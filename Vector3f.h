#ifndef _MONKE_VECTOR_H_
#define _MONKE_VECTOR_H_
#include "Core.h"

namespace monke
{

class Vector3f
{
public:
    __both__ Vector3f() { m_xyz[0] = m_xyz[1] = m_xyz[2] = 0.0f; }
    __both__ Vector3f(float x, float y, float z) { m_xyz[0] = x; m_xyz[1] = y; m_xyz[2] = z; }
    __both__ Vector3f(float xyz) { m_xyz[0] = m_xyz[1] = m_xyz[2] = xyz; }
    __both__ ~Vector3f() { }
    __both__ Vector3f(const Vector3f& copy) 
    {  
        m_xyz[0] = copy[0];
        m_xyz[1] = copy[1];
        m_xyz[2] = copy[2];
    }

    // binary +
    __both__ const Vector3f operator+(const Vector3f& vec) const
    {
        return Vector3f(m_xyz[0] + vec.m_xyz[0], m_xyz[1] + vec.m_xyz[1], m_xyz[2] + vec.m_xyz[2]);
    }
    // binary -
    __both__ const Vector3f operator-(const Vector3f& vec) const
    {
        return Vector3f(m_xyz[0] - vec.m_xyz[0], m_xyz[1] - vec.m_xyz[1], m_xyz[2] - vec.m_xyz[2]);
    }

    // unary -
    friend __both__ const Vector3f operator-(const Vector3f& vec)
    {
        return Vector3f(-vec.m_xyz[0], -vec.m_xyz[1], -vec.m_xyz[2]);
    }

    // scalar * vector
    friend __both__ const Vector3f operator*(float a, const Vector3f &vec)
    {
        return vec * a;
    }
    // vector * scalar
    __both__ const Vector3f operator*(float a) const
    {
        return Vector3f(m_xyz[0] * a, m_xyz[1] * a, m_xyz[2] * a);
    }
    // dot product
    __both__ float operator*(const Vector3f& vec) const
    {
        return m_xyz[0] * vec.m_xyz[0] + m_xyz[1] * vec.m_xyz[1] + m_xyz[2] * vec.m_xyz[2];
    }
    // cross product
    __both__ const Vector3f operator^(const Vector3f& vec) const
    {
        return Vector3f(m_xyz[1] * vec.m_xyz[2] - m_xyz[2] * vec.m_xyz[1],
                        m_xyz[2] * vec.m_xyz[0] - m_xyz[0] * vec.m_xyz[2],
                        m_xyz[0] * vec.m_xyz[1] - m_xyz[1] * vec.m_xyz[0]);
    }
    // component-wise multiplication
    __both__ const Vector3f operator|(const Vector3f& vec) const 
    {
        return Vector3f(m_xyz[0] * vec.m_xyz[0], m_xyz[1] * vec.m_xyz[1], m_xyz[2] * vec.m_xyz[2]);
    }

    __both__ float& operator[](unsigned int index)
    {   
	    return m_xyz[index];
    }

    __both__ const float& operator[](unsigned int index) const
    {
        return m_xyz[index];
    }

    __both__ Vector3f& operator=(const Vector3f& rhs)
    {
        m_xyz[0] = rhs[0];
        m_xyz[1] = rhs[1];
        m_xyz[2] = rhs[2];
        return *this;
    }

    __both__ const Vector3f operator/(float a) const
    {
        return Vector3f(m_xyz[0] / a, m_xyz[1] / a, m_xyz[2] / a);
    }

    __both__ float Length() const
    {
        return std::sqrt( m_xyz[0] * m_xyz[0] + m_xyz[1] * m_xyz[1] + m_xyz[2] * m_xyz[2] );
    }

    __both__ const Vector3f UnitVector() const
    {
        return *this / Length();
    }

    __both__ void Normalize()
    {
        float len = this->Length();
        m_xyz[0] /= len;
        m_xyz[1] /= len;
        m_xyz[2] /= len;
    }

private:
    float m_xyz[3];
};


} // end of monke

#endif