#include <stdint.h>

#include "nw_align.h"

int32_t nw_align_c_kernel(
    const uint8_t* arr1,
    int l1,
    const uint8_t* arr2,
    int l2,
    const int32_t* matrix,
    int matrix_cols,
    const int32_t* char_to_idx,
    int32_t gap_o,
    int32_t gap_e,
    int32_t min_inf,
    int32_t* M,
    int32_t* X,
    int32_t* Y) {
  int cols = l2 + 1;

  M[0 * cols + 0] = 0;
  for (int i = 1; i <= l1; i++) {
    X[i * cols + 0] = gap_o + (i - 1) * gap_e;
  }
  for (int j = 1; j <= l2; j++) {
    Y[0 * cols + j] = gap_o + (j - 1) * gap_e;
  }

  for (int i = 1; i <= l1; i++) {
    for (int j = 1; j <= l2; j++) {
      int32_t idx1 = char_to_idx[arr1[i - 1]];
      int32_t idx2 = char_to_idx[arr2[j - 1]];
      int32_t match_score = matrix[idx1 * matrix_cols + idx2];

      int32_t m_prev = M[(i - 1) * cols + (j - 1)];
      int32_t x_prev = X[(i - 1) * cols + (j - 1)];
      int32_t y_prev = Y[(i - 1) * cols + (j - 1)];
      M[i * cols + j] = match_score + max2(m_prev, max2(x_prev, y_prev));

      int32_t m_left = gap_o + M[(i - 1) * cols + j];
      int32_t x_left = gap_e + X[(i - 1) * cols + j];
      X[i * cols + j] = (m_left > x_left) ? m_left : x_left;

      int32_t m_up = gap_o + M[i * cols + (j - 1)];
      int32_t y_up = gap_e + Y[i * cols + (j - 1)];
      Y[i * cols + j] = (m_up > y_up) ? m_up : y_up;
    }
  }

  int32_t best_score = max2(M[l1 * cols + l2], max2(X[l1 * cols + l2], Y[l1 * cols + l2]));
  return best_score;
}
