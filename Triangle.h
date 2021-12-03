#ifndef _MONKE_TRIANGLE_H_
#define _MONKE_TRIANGLE_H_
#include "Core.h"
#include "Ray.h"
#include "Vector3f.h"

namespace monke
{

class Triangle 
{
public:
    Triangle() = default;
    __both__ Triangle(const Vector3f& p0, const Vector3f& p1, const Vector3f& p2) : m_Vertices{ p0, p1, p2 } 
    { 
        CalculateNormal();
        m_Color = {0.8};
    }

    __both__ const Vector3f& operator[](unsigned int index) const
    {
        return m_Vertices[index];
    }

    __both__ ~Triangle() { }
    __both__ inline const Vector3f GetNormal() const { return m_Normal; }
    __both__  void TestPrint(int aaa) const;
    __both__ float Intersection(const Ray& theRay) const;
    __both__ void PrintInfo() const ;

    __both__ void CalculateNormal() 
    {
        // p0, p1, p2
        Vector3f e1 = m_Vertices[1] - m_Vertices[0];
        Vector3f e2 = m_Vertices[2] - m_Vertices[0];
        m_Normal = e1 ^ e2;
        m_Normal.Normalize();
    }

    Vector3f m_Vertices[3];
    Vector3f m_Color;
private:
    Vector3f m_Normal;
};

__both__ void Triangle::TestPrint(int aaa) const
{
    printf("Hello from triangle!\n");
}

__both__ void Triangle::PrintInfo() const
{
    printf("Vertices: \n");
    for(int i = 0; i < 3; i++)
    {
        printf("Vertex #%d\n", i);
        printf("%f, %f, %f\n\n", m_Vertices[i][0], m_Vertices[i][1], m_Vertices[i][2]);
    }
    printf("\n");
}

__both__ float Triangle::Intersection(const Ray& theRay) const
{
    Vector3f e1 = m_Vertices[1] - m_Vertices[0];
    Vector3f e2 = m_Vertices[2] - m_Vertices[0];
    Vector3f d = theRay.GetDirection();
    Vector3f R0 = theRay.GetPosition();
    Vector3f P0 = m_Vertices[0];

    float denom = d * m_Normal;

    if(denom == 0.0f)
    {
        return -1.0f; 
    }

    float numer = -(R0 - P0) * m_Normal;
    float t = numer / denom;

    if(t < 0.0f)
    {
        return -1.0f;
    }
    
    Vector3f p = theRay.GetPosition() + t * theRay.GetDirection();
    Vector3f pp0 = p - m_Vertices[0];
    Vector3f e2e1 = e2 ^ e1;
    float e2e1Mag = e2e1.Length();
    float u = ((e2 ^ pp0) * e2e1) / (e2e1Mag * e2e1Mag);

    Vector3f e1e2 = e1 ^ e2;
    float e1e2Mag = e1e2.Length();
    float v = ((e1 ^ pp0) * e1e2) / (e2e1Mag * e2e1Mag);

    if(u < 0.0f || u > 1.0f)
    {
        return -1.0f;
    }
    if(v < 0.0f || v > 1.0f)
    {
        return -1.0f;
    }
    float uv = u + v;
    if(uv < 0.0f || uv > 1.0f)
    {
        return -1.0f;
    }

    return t;
}

} // end of monke


#endif