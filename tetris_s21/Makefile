CC=gcc -std=c11
CFLAGS = -Wall -Wextra -Werror
INSTALL_FILES = main.c gui/cli/*.c brick_game.a -lncurses
INSTALL_TEST = brick_game/tetris/*.c gui/cli/*.c  brick_game.a -lncurses
FLAGSCH = -lcheck
TARGET = brick_game


ifeq ($(shell uname),Linux)
	FLAGSCH += -lm -lsubunit
endif
	
all: clean install

install: backend.a
	$(CC) $(CFLAGS) $(INSTALL_FILES) -o game
	./game

uninstall:
	rm -rf game

clean:
	rm -rf *.o *.a *.gcno *.gcda *.info report *.dSYM game test dist *.gz *.dvi *.aux *.log

test: clean backend.a
	$(CC) $(INSTALL_TEST) --coverage tests/test.c -o test $(FLAGSCH)
	./test

backend.a: backend.o
	ar rcs $(TARGET).a *.o
	ranlib $(TARGET).a

backend.o: 
	$(CC) -c brick_game/tetris/*.c 

dist: dist_dir
	tar cf dist/tetris-1.0.tar

dvi:
	rm -rf doxygen
	doxygen

dist_dir:
	mkdir -p dist

gcov_report: test
	lcov -t "result" -o res.info -c -d .
	genhtml -o report res.info
	rm -rf *gcda *gcno res.info test

clang_check:
	cp ../materials/linters/.clang-format .
	clang-format -n gui/*.c
	clang-format -n brick_game/tetris/*.c
	clang-format -n main.c
	clang-format -n tests/*.c
	clang-format -n *.h
	rm -rf .clang-format

clang:
	cp ../materials/linters/.clang-format .
	clang-format -i gui/*.c
	clang-format -i brick_game/tetris/*.c
	clang-format -i main.c
	clang-format -i tests/*.c
	clang-format -i *.h
	rm -rf .clang-format	

