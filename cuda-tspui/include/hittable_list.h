#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"

class hittable_list : public hittable
{
public:
    hittable** objects;
    int obj_count;

    __device__ hittable_list() {}
    __device__ hittable_list(hittable** objects, int obj_count) : objects(objects), obj_count(obj_count) {}
    __device__ bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override
    {
        for (int i = 0; i < obj_count; ++i) {
            if (objects[i]->hit(r, t_min, t_max, rec)) { return true; }
        }
        return false;
    }
};

#endif