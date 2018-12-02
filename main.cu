#include <iostream>
#include <time.h>
#include "vec3.h"
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

__global__ void render(vec3 *fb, int max_x, int max_y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) {
    return;
  }
  int pixel_index = j * max_x + i;
  fb[pixel_index] = vec3(float(i) / max_x, float(j) / max_y, 0.2f);
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
  render<<<blocks, threads>>>(fb, nx, ny);
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