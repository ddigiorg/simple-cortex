CXX = g++
INCLUDE = -I./source/
CFLAGS = -c -g #-Wall
LDLIBS = -lOpenCL -lGL -lsfml-system -lsfml-window -lsfml-graphics -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs 

EXECUTE = execute.exe

all: $(EXECUTE)

OBJS_E = ball.o stimulae.o forest.o area.o compute-system.o compute-program.o input-image.o
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

#occlude.o: $(PATH_D)occlude/occlude.cpp
#	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_D)occlude/occlude.cpp

# ===========
# Cortex
# ===========
PATH_A = ./source/cortex/

stimulae.o: $(PATH_A)stimulae.cpp
	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_A)stimulae.cpp

forest.o: $(PATH_A)forest.cpp
	$(CXX) $(INCLUDE) $(CFLAGS) $(PATH_A)forest.cpp

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
# Utils
# ==========
PATH_U = ./source/utils/

input-image.o: $(PATH_U)input-image.cpp
	$(CXX) $(CFLAGS) $(PATH_U)input-image.cpp


# ==========
# Cleanup
# ==========
.PHONY : clean
clean:
	rm -rf *o $(EXECUTE)
