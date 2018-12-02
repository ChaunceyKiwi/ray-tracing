#include <iostream>
#include <time.h>
#include "vec3.h"
#include "ray.h"

using namespace std;

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples 
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func,
                const char *const file, int const line) {
  if (result) {
    cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at "
         << file << " : " << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

__device__ vec3 color(const ray& r) {
  vec3 unit_direction = unit_vector(r.direction());

  // y is in [-1, 1], thus t is in [0, 1]
  float t = 0.5f * (unit_direction.y() + 1.0f);

  return (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left_corner,
                        vec3 horizontal, vec3 vertical, vec3 origin) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) {
    return;
  }
  int pixel_index = j * max_x + i;
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);
  ray r(origin, lower_left_corner + u * horizontal + v * vertical);
  fb[pixel_index] = color(r);
}

int main() {
  int nx = 200;
  int ny = 200;
  int tx = 8;
  int ty = 8;

  std::cerr << "Rendering a " << nx << "*" << ny << " image";
  std::cerr << "in " << tx << "*" << ty << " blocks." << endl;

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vec3);

  // allocate FB
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  clock_t start, stop;
  start = clock();
  // render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  render<<<blocks, threads>>>(fb, nx, ny,
                              vec3(-2.0, -1.0, -1.0),
                              vec3( 4.0,  0.0,  0.0),
                              vec3( 0.0,  2.0,  0.0),
                              vec3( 0.0,  0.0,  0.0));
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * nx + i;
      int ir = int(255.99 * fb[pixel_index + 0].r());
      int ig = int(255.99 * fb[pixel_index + 0].g());
      int ib = int(255.99 * fb[pixel_index + 0].b());
      cout << ir << " " << ig << " " << ib << endl;
    }
  }
  checkCudaErrors(cudaFree(fb));
}