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
    int32_t* Y,
    int cols) {
  M[0 * cols + 0] = 0;
  X[0 * cols + 0] = min_inf;
  Y[0 * cols + 0] = min_inf;

  for (int i = 1; i <= l1; i++) {
    X[i * cols + 0] = gap_o + (i - 1) * gap_e;
    M[i * cols + 0] = min_inf;
    Y[i * cols + 0] = min_inf;
  }
  for (int j = 1; j <= l2; j++) {
    Y[0 * cols + j] = gap_o + (j - 1) * gap_e;
    M[0 * cols + j] = min_inf;
    X[0 * cols + j] = min_inf;
  }

  for (int i = 1; i <= l1; i++) {
    int32_t idx1 = char_to_idx[arr1[i - 1]];
    for (int j = 1; j <= l2; j++) {
      int32_t idx2 = char_to_idx[arr2[j - 1]];
      int32_t match_score = matrix[idx1 * matrix_cols + idx2];

      int32_t m_pre = M[(i - 1) * cols + (j - 1)];
      int32_t x_pre = X[(i - 1) * cols + (j - 1)];
      int32_t y_pre = Y[(i - 1) * cols + (j - 1)];
      int32_t score_m = max2(m_pre, max2(x_pre, y_pre)) + match_score;
      M[i * cols + j] = score_m;

      int32_t m_up = M[(i - 1) * cols + j];
      int32_t x_up = X[(i - 1) * cols + j];
      X[i * cols + j] = max2(m_up + gap_o, x_up + gap_e);

      int32_t m_left = M[i * cols + (j - 1)];
      int32_t y_left = Y[i * cols + (j - 1)];
      Y[i * cols + j] = max2(m_left + gap_o, y_left + gap_e);
    }
  }

  return max2(M[l1 * cols + l2], max2(X[l1 * cols + l2], Y[l1 * cols + l2]));
}
