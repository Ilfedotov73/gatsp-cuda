#ifndef CITY_H
#define CITY_H

#include "../../cuda-tspui/include/vec2.h"

class city
{
private:
    vec2 p;
public:
    int id;
    __host__ __device__ city() {}
    __host__ __device__ city(vec2 p, int id) : p(p), id(id) {}

    __host__ __device__ vec2 get_point() { return p; }
};

using gen = city;

#endif