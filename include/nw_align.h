#include <stdint.h>

#define MAX_PROTEIN_LEN 320
#define ID_LEN 6
#define GAP_ASCII 45

typedef struct {
  char id1[7];
  char id2[7];
  int32_t score;
  uint8_t al1[MAX_PROTEIN_LEN];
  uint8_t al2[MAX_PROTEIN_LEN];
  double identity_len1;
  double identity_len2;
} Result;

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
    int cols);

static inline int32_t max2(int32_t a, int32_t b) { return (a > b) ? a : b; }

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
    uint8_t* out_al2);

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
    int32_t min_inf);

Result* nw_align_c_full_omp(
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
    int32_t min_inf);
