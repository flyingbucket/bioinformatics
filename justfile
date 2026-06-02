default:
    @just --list

compile:
  mkdir lib
  gcc -O3 -march=native -fPIC -c nw_align_all.c -o lib/nw_align_all.o
  gcc -O3 -march=native -fPIC -c nw_align_kernel.c -o lib/nw_align_kernel.o
  gcc -shared lib/nw_align_all.o lib/nw_align_kernel.o -o lib/libnw_align.so
