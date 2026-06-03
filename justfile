CC := "clang"
CFLAGS := "-O3 -march=native -fPIC -Iinclude"
OMPFLAGS := "-fopenmp=libomp"
LDFLAGS := "-shared"
OMPLIBS := "-lomp"
OUTDIR := "lib"

default:
    @just --list

compile:
    mkdir -p {{OUTDIR}}
    {{CC}} {{CFLAGS}} {{OMPFLAGS}} -c src/c/nw_align_all.c -o {{OUTDIR}}/nw_align_all.o
    {{CC}} {{CFLAGS}} {{OMPFLAGS}} -c src/c/nw_align_kernel.c -o {{OUTDIR}}/nw_align_kernel.o
    {{CC}} {{LDFLAGS}} {{OMPFLAGS}} {{OUTDIR}}/nw_align_all.o {{OUTDIR}}/nw_align_kernel.o {{OMPLIBS}} -o {{OUTDIR}}/libnw_align.so

clean:
    rm -rf {{OUTDIR}}/*.o {{OUTDIR}}/libnw_align.so
