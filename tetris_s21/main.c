
#include "game.h"

int main() {
  srand(time(NULL));
  int ch = ' ';
  GameInfo_t* tetg = NULL;
  Figure* currentFigure;
  UserAction_t player;
  player = Pause;
  Time time;
  time.ticks = 30;
  time.ticks_left = TET_TICKS_START;
  State state;
  state = START;
  initscr();
  curs();
  tetg = create_game();
  currentFigure = create_figure();
  while (tetg->playing_status != TET_GAMEOVER && ch != 'q') {
    ch = getch();
    switch (ch) {
      case KEY_LEFT:
        player = Left;
        break;
      case KEY_RIGHT:
        player = Right;
        break;
      case KEY_DOWN:
        player = Down;
        break;
      case ' ':
        player = Action;
        break;
      case 'p':
        player = Pause;
        break;
      default:
        player = Up;
        break;
    }
    avtomat(tetg, currentFigure, &time, player, &state);
    print_game(tetg, currentFigure);
    print_info(tetg);
    refresh();
    delay_output(20);
  }
  set_hiscore(tetg);
  free_figure(currentFigure);
  free_game(tetg);
  endwin();
  return 0;
}
void curs() {
  curs_set(0);            // Hide cursor
  cbreak();               // Disable line buffering
  noecho();               // Do not display input characters
  keypad(stdscr, TRUE);   // Enable special keys
  nodelay(stdscr, TRUE);  // Non-blocking input
}

// void curs() {
//   curs_set(0);
//   cbreak();
//   noecho();
//   keypad(stdscr, TRUE);
//   nodelay(stdscr, TRUE);
//   scrollok(stdscr, TRUE);

//   start_color();

//   init_pair(0, COLOR_BLACK, COLOR_BLACK);
//   init_pair(1, COLOR_GREEN, COLOR_RED);
//   init_pair(2, COLOR_WHITE, COLOR_WHITE);
//   init_pair(3, COLOR_YELLOW, COLOR_YELLOW);
//   init_pair(4, COLOR_MAGENTA, COLOR_MAGENTA);
//   init_pair(5, COLOR_GREEN, COLOR_GREEN);
//   init_pair(6, COLOR_BLUE, COLOR_BLUE);
//   init_pair(7, COLOR_CYAN, COLOR_CYAN);
// }

