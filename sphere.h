#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere : public hitable {
 public:
  __device__ sphere(){};
  __device__ sphere(vec3 cen, float r, material *m) :
                    center(cen), radius(r), mat_ptr(m) {};
  __device__ virtual bool hit(const ray& r, float t_min, float t_max,
                   hit_record& rec) const;
  vec3 center;
  float radius;
  material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max,
                 hit_record& rec) const {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = 2.0f * dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - 4.0f * a * c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(discriminant)) / (2.0f * a);
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
    temp = (-b + sqrt(discriminant)) / (2.0f * a);
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }
  return false;
}

#endif