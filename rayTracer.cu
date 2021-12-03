/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */


#include "cuda.h"
#include "common/book.h"
#include "common/cpu_bitmap.h"
#include "Vector3f.h"
#include "Triangle.h"
#include "Camera.h"
#include "Ray.h"
#include <vector>
#include <string>
#include <cmath>
#include "Mesh.h"

#define DIM 512

#define rnd( x ) (x * rand() / RAND_MAX)
#define INF     2e10f

namespace monke
{

struct PointLight
{
    Vector3f position;
    Vector3f color;
};

} // end of monke


__global__ void monkeKernel( monke::Camera* camera, monke::Triangle* triangleList, int triangleListSize, 
monke::PointLight* lightList, int lightListSize, unsigned char *ptr ) 
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;
    
    //
    // camera view to get a ray
    float xx = (float) x / (float) DIM;
    float yy = (float) y / (float) DIM;
    monke::Vector3f position = camera->GetPosition();
    monke::Vector3f direction = camera->View(xx, yy);
    monke::Ray theRay(position, direction);
    
    float r = 0.1f, g = 0.1f, b = 0.1f;
    float t = INF;
    float tHit;
    int closestTriIdx = -1;
    //printf("%d\n", lightListSize);
    for(int i = 0; i < triangleListSize; i++)
    {
        //t = 
        //triangleList[0].PrintInfo();
        tHit = triangleList[i].Intersection(theRay);
        if(tHit <= 0.0f) continue;// no hit
        if(tHit < t)
        {
            closestTriIdx = i;
            t = tHit; 
        }
    }

    if(closestTriIdx >= 0)
    {
        // hit
        monke::Vector3f lightPos;
        monke::Vector3f lightColor;
        monke::Vector3f finalColor(0.0f);
        for(int i = 0; i < lightListSize; i++)
        {
            lightPos = lightList[i].position;
            lightColor = lightList[i].color;

            monke::Vector3f pos = theRay.GetPointOnRay(t);
            monke::Vector3f lightDir = lightPos - pos;
            lightDir.Normalize();
            monke::Vector3f normal = triangleList[closestTriIdx].GetNormal();
            float m = fmaxf(0.0, lightDir * normal);
            monke::Vector3f theColor = m * (triangleList[closestTriIdx].m_Color | lightColor);
            finalColor = theColor + finalColor;
        }

        r = finalColor[0];
        g = finalColor[1];
        b = finalColor[2];
    }

    ptr[offset*4 + 0] = (int)(r * 255);
    ptr[offset*4 + 1] = (int)(g * 255);
    ptr[offset*4 + 2] = (int)(b * 255);
    ptr[offset*4 + 3] = 255;
}

int main(void)
{
    monke::Mesh model;
    model.LoadModel("data/models/Monke.obj");

    monke::Vector3f cameraLookAt { 0.0f, 0.0f, 0.0f };
    monke::Vector3f cameraPosition { 0.0f, 1.2f, -2.0f }; 
    monke::CameraSetting camset;
    camset.position = cameraPosition;
    camset.viewDirection = cameraLookAt - cameraPosition;

    monke::Camera camera(camset);
    monke::Camera* dev_camera;

    HANDLE_ERROR( cudaMalloc( (void**)&dev_camera, sizeof(monke::Camera) ) );
    HANDLE_ERROR( cudaMemcpy(dev_camera, &camera, sizeof(monke::Camera), cudaMemcpyHostToDevice) );

    // lights
    const int lightListSize = 2;
    monke::PointLight* lightList, *dev_lightList;
    lightList = (monke::PointLight*) malloc( sizeof(monke::PointLight) * lightListSize );
    
    lightList[0].position = monke::Vector3f(1.0f, 1.0f, -2.0f);
    lightList[0].color = monke::Vector3f(0.5f, 0.0f, 0.0f);


    lightList[1].position = monke::Vector3f(-1.0f, 1.0f, -2.0f);
    lightList[1].color = monke::Vector3f(0.0f, 0.5f, 0.0f);
    
    HANDLE_ERROR( cudaMalloc( (void**)&dev_lightList, sizeof(monke::PointLight) * lightListSize ) );
    HANDLE_ERROR( cudaMemcpy(dev_lightList, lightList, sizeof(monke::PointLight) * lightListSize , cudaMemcpyHostToDevice) );
    
    //
    CPUBitmap bitmap( DIM, DIM );
    unsigned char   *dev_bitmap;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_bitmap, bitmap.image_size() ) );
    //

    dim3 grids(DIM/16,DIM/16);
    dim3 threads(16,16);
    
    monkeKernel<<<grids, threads>>>( dev_camera, model.GetCudaTriangleList(), 
                   model.GetCudaTriangleSize(), dev_lightList, lightListSize, dev_bitmap );
    
    cudaDeviceSynchronize();

    HANDLE_ERROR( cudaMemcpy( bitmap.get_ptr(), dev_bitmap,
                              bitmap.image_size(),
                              cudaMemcpyDeviceToHost ) );

    
    bitmap.display_and_exit();
    // free 
    HANDLE_ERROR( cudaFree( dev_bitmap ) );
    HANDLE_ERROR( cudaFree( dev_camera ) );
    HANDLE_ERROR( cudaFree( dev_lightList ) );
    free(lightList);
    

    return 0;
}




