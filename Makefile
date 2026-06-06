CC       := clang
CFLAGS   := -O3 -march=native -fPIC -Iinclude -fopenmp=libomp
LDFLAGS  := -shared -fopenmp=libomp
LIBS     := -lomp
OUTDIR   := lib

TARGET   := $(OUTDIR)/libnw_align.so
OBJS     := $(OUTDIR)/nw_align_all.o $(OUTDIR)/nw_align_kernel.o

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(OUTDIR)
	$(CC) $(LDFLAGS) $(OBJS) $(LIBS) -o $@

$(OUTDIR)/%.o: src/c/%.c
	@mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OUTDIR)
