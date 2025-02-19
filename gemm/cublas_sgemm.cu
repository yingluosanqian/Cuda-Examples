#include <cublas_v2.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


int main(int argc, char **argv) {
  int m = 5120, n = 5120, k = 4096;

  if (argc >= 2) {
    sscanf(argv[1], "%d", &m);
  }
  if (argc >= 3) {
    sscanf(argv[2], "%d", &n);
  }
  if (argc >= 4) {
    sscanf(argv[3], "%d", &k);
  }

  char validation_char = 'N';
  if (argc >= 5) {
    sscanf(argv[4], "%c", &validation_char);
  }
  bool validation = validation_char == 'Y' ? true : false;

  using TA = float;
  using TB = float;
  using TC = float;

  thrust::host_vector<TA> h_A(m * k);
  thrust::host_vector<TB> h_B(n * k);
  thrust::host_vector<TC> h_C(m * n);
  thrust::host_vector<TC> h_validation(m * n);

  if (validation) {
    for (int i = 0; i < m * k; i++) {
      h_A[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    }
    for (int i = 0; i < n * k; i++) {
      h_B[i] = static_cast<TA>(2 * (rand() / double(RAND_MAX)) - 1);
    }
    for (int i = 0; i < m * n; i++) {
      h_C[i] = h_validation[i] = 0;
    }
  }

  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.0;
  float beta = 0.0;
  auto result = cublasSgemm(handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    m, n, k,
    &alpha,
    d_A.data().get(), k,
    d_B.data().get(), k,
    &beta,
    d_C.data().get(), n);

  thrust::host_vector<TC> cute_result = d_C;

  if (validation) {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        float C = 0;
        for (int x = 0; x < k; x++) {
          C += h_A[i * k + x] * h_B[j * k + x];
        }
        h_validation[i + j * m] = C;
      }
    }
    bool same = true;
    for (int i = 0; i < m * n; i++) {
      if (fabs(cute_result[i] - h_validation[i]) > 1e-3) {
        printf("i: %d, result: %f, ref: %f\n", i, cute_result[i],
               h_validation[i]);
        same = false;
        break;
      }
    }
    assert(same && "Computation result error.");
    printf("Pass.\n");
  }

  return 0;
}