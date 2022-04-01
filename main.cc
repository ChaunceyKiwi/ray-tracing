#include <iostream>
#include "camera.h"
#include "color.h"
#include "common.h"
#include "hittable_list.h"
#include "sphere.h"
using namespace std;

color ray_color(const ray& r, const hittable& world, int depth) {
  // If we've exceeded the ray bounce limit, no more light is gathered.
  if (depth <= 0) {
    return color(0, 0, 0);
  }

  hit_record rec;
  if (world.hit(r, 0.0001, infinity, rec)) {
    point3 target = rec.p + rec.normal + random_in_unit_sphere();
    return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);
  }
  vec3 unit_direction = unit_vector(r.direction());
  auto t = 0.5 * (unit_direction.y() + 1.0);
  return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main() {
  // Image
  const auto aspect_ratio = 16.0 / 9.0;
  const int image_width = 400;
  const int image_height = (int)(image_width / aspect_ratio);
  const int samples_per_pixel = 100;
  const int max_depth = 50;

  // World
  hittable_list world;
  world.add(make_shared<sphere>(point3(0, 0, -1), 0.5));
  world.add(make_shared<sphere>(point3(0, -100.5, -1), 100));

  // Camera
  camera cam;

  // Render
  cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

  // Print from top-left to bottom-right
  for (int j = image_height - 1; j >= 0; j--) {
    cerr << "\rScanlines remaining: " << j << ' ' << flush;
    for (int i = 0; i < image_width; i++) {
      color pixel_color(0, 0, 0);
      for (int s = 0; s < samples_per_pixel; s++) {
        auto u = double(i + random_double()) / (image_width - 1);
        auto v = double(j + random_double()) / (image_height - 1);
        ray r = cam.get_ray(u, v);
        pixel_color += ray_color(r, world, max_depth);
      }
      write_color(cout, pixel_color, samples_per_pixel);
    }
  }

  cerr << "\nDone.\n";
}
