# Native build of the real firmware engine for parity testing.
#   make            -> builds host_harness
#   make check      -> builds + runs parity vs golden vectors
CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++14 -Wall -Wextra

SRC  := ../rlm_esp32/src
GOLD := ../../tools/out/golden_vectors.txt

host_harness: host_harness.cpp $(SRC)/rlm_engine.cpp $(SRC)/tokenizer.cpp \
              $(SRC)/rlm_engine.h $(SRC)/rlm_config.h $(SRC)/rlm_weights.h
	$(CXX) $(CXXFLAGS) -I$(SRC) host_harness.cpp $(SRC)/rlm_engine.cpp \
	       $(SRC)/tokenizer.cpp -lm -o $@

check: host_harness
	./host_harness $(GOLD) --bench 20

clean:
	rm -f host_harness

.PHONY: check clean
