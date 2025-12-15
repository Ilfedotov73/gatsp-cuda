#ifndef POINT_H
#define POINT_H

#include "vec2.h"
#include "hittable.h"

class circle : public hittable
{
private:
    vec2 center;
    double radius;
public:
    __device__ circle() {}
    __device__ circle(const vec2& center, double radius) : center(center), radius(radius) {} 
    __device__ bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const override
    {
        vec2 oc = center - r.origin();
        double a = r.direction().length_squared();
        double h = dot(r.direction(), oc);
        double discriminant = h*h - a*(oc.length_squared() - radius*radius);

        if (discriminant < 0) { return false; }
        double sqrtd = std::sqrt(discriminant);
        double root = (h - sqrtd) / a;
        if (!(t_min < root && root < t_max)) {
            root = (h + sqrtd) / a;
            if (!(t_min < root && root < t_max)) { return false; }
        }

        rec.p = r.at(root);
        rec.t = root;
    }
};

#endif