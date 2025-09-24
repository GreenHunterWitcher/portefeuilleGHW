#ifndef GAME_H
#define GAME_H

#define FIELD_WIDTH 12
#define FIELD_HEIGHT 22
#define FIGURE_SIZE 5

#define TET_GAMEOVER 0
#define TET_PLAYING 1

#define TET_TICKS_START 32

#include <locale.h>
#include <ncurses.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <wchar.h>

typedef enum {
  Start,
  Pause,
  Terminate,
  Left,
  Right,
  Up,
  Down,
  Action
} UserAction_t;

typedef struct {
  int** field;
  int** next;
  int score;
  int high_score;
  int level;
  int speed;
  int pause;
  int playing_status;
} GameInfo_t;

typedef enum {
  START,
  SPAWN,
  MOVING,
  SHIFTING,
  ATTACHING,
  GAMEOVER,
} State;

typedef struct Figure {
  int x;
  int y;
  int** figure;
} Figure;

typedef struct Time {
  int ticks;
  int ticks_left;
} Time;

int get_random_int(int min, int max);
void generate_next_figure(GameInfo_t* game);
void create_next_figure(GameInfo_t* game);
void free_next_figure(GameInfo_t* game);
void create_field(GameInfo_t* game);
void free_field(GameInfo_t* game);
GameInfo_t* create_game();
void free_game(GameInfo_t* game);
void move_figure_down(Figure* currentFigure);
void move_figure_up(Figure* currentFigure);
void move_figure_left(Figure* currentFigure);
void move_figure_right(Figure* currentFigure);
int colision(Figure* currentFigure, GameInfo_t* game);
void plant_figure(Figure* currentFigure, GameInfo_t* game);
int line_full(int i, GameInfo_t* game);
void drope_line(int i, GameInfo_t* game);
int erase_lines(GameInfo_t* game);
Figure* create_figure();
void free_figure(Figure* figure);
void drop_new_figure(Figure* currentFigure, GameInfo_t* game);
Figure* rot_figure(Figure* currentFigure);
void user_input(UserAction_t action, Figure* currentFigure, GameInfo_t* game,
                Time* time, State* state);
void set_hiscore(GameInfo_t* game);
void level_monitoring(GameInfo_t* game);
void avtomat(GameInfo_t* game, Figure* currentFigure, Time* time,
             UserAction_t action, State* state);

void pause_game();
void start_game();
void print_game(GameInfo_t* game, Figure* currentFigure);
void print_info(GameInfo_t* game);
void print_pole(int left_x, int right_x, int up_y, int down_y);
void curs();

#endif
