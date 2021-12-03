#ifndef _MONKE_CAMERA_H_
#define _MONKE_CAMERA_H_
#include "Core.h"
#include "Vector3f.h"

namespace monke
{

struct CameraSetting
{
    Vector3f position{0.0f, 0.0f, -2.0f};
    Vector3f viewDirection{0.0f, 0.0f, 1.0f};
    Vector3f upDirection{0.0f, 1.0f, 0.0f};
    Vector3f rightDirection{1.0f, 0.0f, 0.0f};
    float FOV = 90.0f; 
    float aspectRatio = 1.0f;
};


class Camera
{
public:
    __both__ Camera();
    __both__ ~Camera() { }
    __both__ Camera(const CameraSetting& camset);

    __both__ void SetupCamera(const CameraSetting& camset);
    __both__ inline CameraSetting GetCameraSetting() const { return m_Setting; } 
    __both__ inline const Vector3f GetPosition() const { return m_Setting.position; }
    __both__ void SetViewDirection(const Vector3f& viewDir);
    __both__ inline float GetTanHFOV() const { return m_tanHFOV2; }
    __both__ inline float GetTanVFOV() const { return m_tanVFOV2; }

    __both__ const Vector3f View(float x, float y) const;
private:
    CameraSetting m_Setting;
    float m_tanHFOV2, m_tanVFOV2;
};

/* ------ */
Camera::Camera()
: m_tanHFOV2(0.0f), m_tanVFOV2(0.0f)
{
    CameraSetting camset;
    SetupCamera(camset);
}

Camera::Camera(const CameraSetting& camset)
 : m_Setting(camset), m_tanHFOV2(0.0f), m_tanVFOV2(0.0f)
{
    SetupCamera(camset);
}

void Camera::SetViewDirection(const Vector3f& viewDir)
{
    m_Setting.viewDirection = viewDir;
    SetupCamera(m_Setting);
}

void Camera::SetupCamera(const CameraSetting& camset)
{
    m_Setting = camset;
    const float PYEE = 3.14159265f;
    m_tanHFOV2 = std::tan(m_Setting.FOV * 0.5f * PYEE / 180.0f);
    m_tanVFOV2 = m_tanHFOV2 / m_Setting.aspectRatio;

    m_Setting.viewDirection.Normalize();

    m_Setting.upDirection = m_Setting.upDirection - (m_Setting.viewDirection * m_Setting.upDirection) * m_Setting.viewDirection;
    m_Setting.upDirection.Normalize();

    m_Setting.rightDirection = m_Setting.viewDirection ^ m_Setting.upDirection;
    m_Setting.rightDirection.Normalize();
}

const Vector3f Camera::View(float x, float y) const
{
    float xx = (2.0f * x - 1.0f) * m_tanHFOV2;
    float yy = (2.0f * y - 1.0f) * m_tanVFOV2;

    Vector3f X = xx * m_Setting.rightDirection + yy * m_Setting.upDirection;
    return (X + m_Setting.viewDirection).UnitVector();
}
/* ------ */


} // namespace vzl

#endif