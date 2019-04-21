#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"

class sphere : public hitable {
 public:
  sphere(){};
  sphere(vec3 cen, float r, material* m) : center(cen), radius(r), mat_ptr(m){};
  bool bounding_box(float t0, float t1, aabb& box) const;
  virtual bool hit(const ray& r, float t_min, float t_max,
                   hit_record& rec) const;
  vec3 center;
  float radius;
  material* mat_ptr;
};

bool sphere::hit(const ray& r, float t_min, float t_max,
                 hit_record& rec) const {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = 2.0 * dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - 4 * a * c;
  if (discriminant > 0) {
    float temp = (-b - sqrt(discriminant)) / (2.0 * a);
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
    temp = (-b + sqrt(discriminant)) / (2.0 * a);
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      return true;
    }
  }
  return false;
}

bool sphere::bounding_box(float t0, float t1, aabb& box) const {
  box = aabb(center - vec3(radius, radius, radius), 
             center - vec3(radius, radius, radius));
  return true;
}

#endif