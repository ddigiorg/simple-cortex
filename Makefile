CXX = g++
INCLUDE = -I./source/
CFLAGS = -c -g #-Wall
LDLIBS = -lOpenCL -lGL -lsfml-system -lsfml-window -lsfml-graphics 

EXECUTE = execute.exe

all: $(EXECUTE)

OBJS_E = ball.o pattern.o area.o compute-system.o compute-program.o
$(EXECUTE): $(OBJS_E)
	$(CXX) $(LDLIBS) $(OBJS_E) -o $(EXECUTE)

# ==========
# Demo
# ==========
PATH_D = ./demos/

#ball.o: $(PATH_D)ball-1.0/ball.cpp
#	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_D)ball-1.0/ball.cpp

ball.o: $(PATH_D)ball-2.0/ball.cpp
	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_D)ball-2.0/ball.cpp

# ===========
# Application
# ===========
PATH_A = ./source/cortex/

pattern.o: $(PATH_A)pattern.cpp
	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_A)pattern.cpp

area.o: $(PATH_A)area.cpp
	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_A)area.cpp

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
