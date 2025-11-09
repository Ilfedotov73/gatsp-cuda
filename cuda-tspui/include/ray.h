#ifndef RAY_H
#define RAY_H

#include "vec2.h"

class ray
{
private:
    vec2 orig;
    vec2 dir;
public:
    __device__ ray() {}
    __device__ ray(const vec2& origin, const vec2& directrion) : orig(origin), dir(directrion) {}

    __device__ const vec2& origin() const { return orig; }
    __device__ const vec2& direction() const { return dir; }
    __device__ vec2 at(double t) const { return orig + t*dir; }
};

#endif