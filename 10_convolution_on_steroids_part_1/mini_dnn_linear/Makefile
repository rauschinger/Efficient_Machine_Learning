BUILD_DIR ?= ./build
CXXFLAGS ?=
LDFLAGS ?=
RPATHS ?=
LIBXSMM_DIR ?= libxsmm
OPTIONS = -O2 -std=c++20 -pedantic -Wall -Wextra -DTORCH_API_INCLUDE_EXTENSION_H -I.

PYTORCH_INCLUDE = $(shell python -c 'from torch.utils.cpp_extension import include_paths; [print(p) for p in include_paths()]')
PYTORCH_LINK = $(shell python -c 'from torch.utils.cpp_extension import library_paths; [print(p) for p in library_paths()]')

CXXFLAGS += $(foreach inc,$(PYTORCH_INCLUDE),-isystem$(inc)) -fopenmp
LDFLAGS += $(foreach lin,$(PYTORCH_LINK),-L$(lin)) -lc10 -ltorch -ltorch_cpu ${LIBXSMM_DIR}/lib/libxsmm.a -ldl
RPATHS += $(foreach lin,$(PYTORCH_LINK),-Wl,-rpath,$(lin))

$(info $$CXXFLAGS is [${CXXFLAGS}])
$(info $$LDFLAGS is [${LDFLAGS}])

matmul: src/backend/MatmulAtenBlocked.cpp src/backend/MatmulAtenBlocked.test.cpp src/backend/MatmulLibxsmm.cpp src/backend/MatmulLibxsmm.test.cpp src/backend/MatmulReluAten.cpp src/backend/MatmulReluAten.test.cpp
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/MatmulAtenBlocked.cpp -o ${BUILD_DIR}/backend/MatmulAtenBlocked.o
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/MatmulAtenBlocked.test.cpp -o ${BUILD_DIR}/tests/MatmulAtenBlocked.test.o

		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/MatmulLibxsmm.cpp -I${LIBXSMM_DIR}/include -o ${BUILD_DIR}/backend/MatmulLibxsmm.o
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/MatmulLibxsmm.test.cpp -o ${BUILD_DIR}/tests/MatmulLibxsmm.test.o

		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/MatmulReluAten.cpp -I${LIBXSMM_DIR}/include -o ${BUILD_DIR}/backend/MatmulReluAten.o
		$(CXX) ${OPTIONS} ${CXXFLAGS} -c src/backend/MatmulReluAten.test.cpp -I${LIBXSMM_DIR}/include -o ${BUILD_DIR}/tests/MatmulReluAten.test.o

		$(CXX) ${OPTIONS} ${CXXFLAGS} src/performance_matmul.cpp  ${BUILD_DIR}/backend/Matmul*.o -o ${BUILD_DIR}/performance_matmul ${RPATHS} ${LDFLAGS}

all: matmul
		$(CXX) ${OPTIONS} ${CXXFLAGS} src/test.cpp ${BUILD_DIR}/backend/*.o ${BUILD_DIR}/tests/*.o -o ${BUILD_DIR}/test ${RPATHS} ${LDFLAGS}

$(shell mkdir -p build/backend)
$(shell mkdir -p build/tests)