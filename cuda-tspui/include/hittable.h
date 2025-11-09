#ifndef HITTABLE_H
#define HITTABLE_H

#include "vec2.h"
#include "ray.h"

struct hit_record
{
public:
    vec2 p;
    double t;
};

class hittable
{
public:
    __device__ virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const = 0;
};

#endif