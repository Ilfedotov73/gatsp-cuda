#ifndef VEC2_H
#define VEC2_H

class vec2
{
public:
    double e[2];
    __host__ __device__ vec2() : e{ .0, .0 } {}
    __host__ __device__ vec2(double e0, double e1) : e{ e0, e1 } {}
    
    __host__ __device__ double x() const { return e[0]; }
    __host__ __device__ double y() const { return e[1]; }

    __host__ __device__ vec2 operator-() const { return vec2(-e[0], -e[1]); }
    __host__ __device__ double operator[](int i) const { return e[i]; }
    __host__ __device__ double& operator[](int i) { return e[i]; }

    __host__ __device__ vec2& operator+=(const vec2& v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        return *this;
    }
    __host__ __device__ vec2& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        return *this;
    }
    __host__ __device__ vec2& operator*=(const vec2& v)
    {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        return *this;
    }
    __host__ __device__ vec2& operator/=(double t) { return *this *= 1/t; }
    __host__ __device__ double length_squared() const { return e[0]*e[0] + e[1]*e[1]; }
    __host__ __device__ double length() const { return std::sqrt(length_squared()); }
};

__host__ __device__ inline vec2 operator+(const vec2& u, const vec2& v)
{
    return vec2(
        u.e[0] + v.e[0],
        u.e[1] + v.e[1]
    );
}
__host__ __device__ inline vec2 operator-(const vec2& u, const vec2& v)
{
    return vec2(
        u.e[0] - v.e[0],
        u.e[1] - v.e[1]
    );
}
__host__ __device__ inline vec2 operator*(const vec2& u, const vec2& v)
{
    return vec2(
      u.e[0] * v.e[0],
      u.e[1] * v.e[1]  
    );
}
__host__ __device__ inline vec2 operator*(double t, const vec2& v)
{
    return vec2(
      t*v.e[0],
      t*v.e[1]  
    );
}
__host__ __device__ inline vec2 operator*(const vec2& v, double t) { return t*v; }
__host__ __device__ inline vec2 operator/(const vec2& v, double t) { return (1/t)*v; }

__host__ __device__ inline double dot(const vec2& u, const vec2& v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1];
}

__host__ __device__ inline double distance(const vec2& u, const vec2& v)
{
    vec2 dist(u[0]-v[0], u[1]-v[1]);
    return dist.length();
}

#endif