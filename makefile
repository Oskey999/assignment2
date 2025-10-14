# ====== Project Configuration ======
TARGET = ass2
SRC = ass2.c

# ====== Compiler and Flags ======
CC = cc                     # MPI compiler wrapper
CFLAGS = -O3 -fopenmp -Wall    # Optimization, OpenMP, warnings
LDFLAGS = -fopenmp             # Linker flags

# ====== Build Rules ======
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# ====== Run with MPI ======
# Usage example:
# make run NP=4 ARGS="256 256 3 3 1 1"
run: $(TARGET)
	@echo "Running: mpirun -np $(NP) ./$(TARGET) $(ARGS)"
	mpirun -np $(NP) ./$(TARGET) $(ARGS)

# ====== Cleanup ======
clean:
	rm -f $(TARGET) *.o