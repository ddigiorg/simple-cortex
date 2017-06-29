CXX = g++
INCLUDE = -I./source/
CFLAGS = -c -g #-Wall
LDLIBS = -lOpenCL -lGL -lsfml-system -lsfml-window -lsfml-graphics 

EXECUTE = execute.exe

all: $(EXECUTE)

OBJS_E = test.o region.o compute-system.o compute-program.o
$(EXECUTE): $(OBJS_E)
	$(CXX) $(LDLIBS) $(OBJS_E) -o $(EXECUTE)

# ==========
# Demo
# ==========
PATH_D = ./demos/

test.o: $(PATH_D)test/test.cpp
	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_D)test/test.cpp

# ==========
# App
# ==========
PATH_A = ./source/app/

region.o: $(PATH_A)region.cpp
	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_A)region.cpp

# ==========
# Compute
# ==========
PATH_C = ./source/compute/

compute-system.o: $(PATH_C)compute-system.cpp
	$(CXX) $(CFLAGS) $(PATH_C)compute-system.cpp

compute-program.o: $(PATH_C)compute-program.cpp
	$(CXX) $(CFLAGS) $(PATH_C)compute-program.cpp

# ==========
# Cleanup
# ==========
.PHONY : clean
clean:
	rm -rf *o $(EXECUTE)
