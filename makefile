CC = g++
CFLAGS1 = -std=c++11

make: main.cc
	${CC} ${CFLAGS1} -o main main.cc

clean:
	rm main *.ppm