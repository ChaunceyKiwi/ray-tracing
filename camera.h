#ifndef CAMERA_H
#define CAMERA_H

#include "common.h"

class camera {
 private:
  point3 origin;
  point3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;

 public:
  camera(double vfov,  // vertical field-of-view in degress
         double aspect_ratio) {
    auto theta = degrees_to_radians(vfov);
    auto focal_length = 1.0;
    auto viewport_height = 2.0 * tan(theta / 2) * focal_length;
    auto viewport_width = aspect_ratio * viewport_height;

    origin = point3(0, 0, 0);
    horizontal = vec3(viewport_width, 0, 0);
    vertical = vec3(0, viewport_height, 0);
    lower_left_corner =
        origin - horizontal / 2 - vertical / 2 - vec3(0, 0, focal_length);
  }

  ray get_ray(double u, double v) {
    return ray(origin,
               lower_left_corner + u * horizontal + v * vertical - origin);
  }
};

#endif