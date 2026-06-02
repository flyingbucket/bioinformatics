#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nw_align.h"

Result* nw_align_c_full(
    int num_records,
    const uint8_t* full_seq,
    const int32_t* seq_len,
    const int32_t* seq_offset,
    const char* full_id,
    const int32_t* matrix,
    int matrix_cols,
    const int32_t* char_to_idx,
    int32_t gap_o,
    int32_t gap_e,
    int32_t min_inf) {
  Result* res = (Result*)malloc(sizeof(Result) * (num_records - 1) * num_records / 2);
  if (res == NULL) return NULL;
  int32_t* M = (int32_t*)malloc(sizeof(int32_t) * MAX_PROTEIN_LEN * MAX_PROTEIN_LEN);
  int32_t* X = (int32_t*)malloc(sizeof(int32_t) * MAX_PROTEIN_LEN * MAX_PROTEIN_LEN);
  int32_t* Y = (int32_t*)malloc(sizeof(int32_t) * MAX_PROTEIN_LEN * MAX_PROTEIN_LEN);

  if (M == NULL || X == NULL || Y == NULL) goto cleanup;
  int curr_line = 0;

  for (int i = 0; i < num_records; i++) {
    const uint8_t* aa1 = full_seq + seq_offset[i];
    const char* id1 = full_id + i * ID_LEN;
    int l1 = seq_len[i];
    for (int j = i + 1; j < num_records; j++) {
      const uint8_t* aa2 = full_seq + seq_offset[j];
      const char* id2 = full_id + j * ID_LEN;

      memcpy(res[curr_line].id1, id1, ID_LEN);
      memcpy(res[curr_line].id2, id2, ID_LEN);
      res[curr_line].id1[ID_LEN] = '\0';
      res[curr_line].id2[ID_LEN] = '\0';
      int l2 = seq_len[j];

      size_t matrix_bytes = sizeof(int32_t) * MAX_PROTEIN_LEN * MAX_PROTEIN_LEN;
      memset(M, 0, matrix_bytes);
      memset(X, 0, matrix_bytes);
      memset(Y, 0, matrix_bytes);

      int32_t score = nw_align_c_kernel(
          aa1, l1, aa2, l2, matrix, matrix_cols, char_to_idx, gap_o, gap_e, min_inf, M, X, Y);
      res[curr_line].score = score;
      uint8_t* out_al1 = res[curr_line].al1;
      uint8_t* out_al2 = res[curr_line].al2;
      int32_t aligned_len = nw_backtrace_c(
          aa1, l1, aa2, l2, M, X, Y, score, matrix, matrix_cols, char_to_idx, gap_o, gap_e, out_al1,
          out_al2);
      int matches = 0;
      for (int k = 0; k < aligned_len; k++)
        matches += (out_al1[k] == out_al2[k]) && (out_al1[k] != GAP_ASCII);

      res[curr_line].identity_len1 = (double)matches / l1;
      res[curr_line].identity_len2 = (double)matches / l2;
      curr_line += 1;
    }
  }

cleanup:
  if (M != NULL) free(M);
  if (X != NULL) free(X);
  if (Y != NULL) free(Y);

  return res;
}

// bactrace status
#define STATE_M 0
#define STATE_X 1
#define STATE_Y 2

int32_t nw_backtrace_c(
    const uint8_t* aa1,
    int l1,
    const uint8_t* aa2,
    int l2,
    const int32_t* M,
    const int32_t* X,
    const int32_t* Y,
    int32_t best_score,
    const int32_t* matrix,
    int matrix_cols,
    const int32_t* char_to_idx,
    int32_t gap_o,
    int32_t gap_e,
    uint8_t* out_al1,
    uint8_t* out_al2) {
  int i = l1;
  int j = l2;
  int cols = l2 + 1;
  int max_len = l1 + l2;

  uint8_t tmp1[512];
  uint8_t tmp2[512];

  int idx = max_len;
  int curr_state;

  if (best_score == M[i * cols + j]) {
    curr_state = STATE_M;
  } else if (best_score == X[i * cols + j]) {
    curr_state = STATE_X;
  } else {
    curr_state = STATE_Y;
  }

  while (i > 0 || j > 0) {
    idx--;

    if (i > 0 && j > 0) {
      if (curr_state == STATE_M) {
        uint8_t u1 = aa1[i - 1];
        uint8_t u2 = aa2[j - 1];
        tmp1[idx] = u1;
        tmp2[idx] = u2;

        int32_t match_score = matrix[char_to_idx[u1] * matrix_cols + char_to_idx[u2]];

        if (M[i * cols + j] == match_score + M[(i - 1) * cols + (j - 1)]) {
          curr_state = STATE_M;
        } else if (M[i * cols + j] == match_score + X[(i - 1) * cols + (j - 1)]) {
          curr_state = STATE_X;
        } else {
          curr_state = STATE_Y;
        }
        i--;
        j--;
      } else if (curr_state == STATE_X) {
        tmp1[idx] = aa1[i - 1];
        tmp2[idx] = GAP_ASCII;

        if (X[i * cols + j] == gap_o + M[(i - 1) * cols + j]) {
          curr_state = STATE_M;
        } else if (X[i * cols + j] == gap_e + X[(i - 1) * cols + j]) {
          curr_state = STATE_X;
        } else {
          printf("Error: Backtrace failed in X");
        }
        i--;
      } else if (curr_state == STATE_Y) {
        tmp1[idx] = GAP_ASCII;
        tmp2[idx] = aa2[j - 1];

        if (Y[i * cols + j] == gap_o + M[i * cols + (j - 1)]) {
          curr_state = STATE_M;
        } else if (Y[i * cols + j] == gap_e + Y[i * cols + (j - 1)]) {
          curr_state = STATE_Y;
        } else {
          printf("Error: Backtrace failed in Y");
        }
        j--;
      }
    } else if (i > 0) {
      tmp1[idx] = aa1[i - 1];
      tmp2[idx] = GAP_ASCII;
      i--;
    } else if (j > 0) {
      tmp1[idx] = GAP_ASCII;
      tmp2[idx] = aa2[j - 1];
      j--;
    }
  }

  int32_t aligned_len = max_len - idx;

  memcpy(out_al1, &tmp1[idx], aligned_len);
  memcpy(out_al2, &tmp2[idx], aligned_len);

  out_al1[aligned_len] = '\0';
  out_al2[aligned_len] = '\0';
  return aligned_len;
}
