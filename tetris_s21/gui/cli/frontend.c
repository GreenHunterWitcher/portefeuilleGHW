#include "../../game.h"

// void print_game(GameInfo_t* game, Figure* currentFigure) {
//   int sym = 0;
//   for (int i = 0; i < FIELD_HEIGHT; i++) {
//     for (int j = 0; j < FIELD_WIDTH; j++) {
//       sym = 0;
//       if (game->field[i][j] != 0) {
//         sym = game->field[i][j];
//       } else {
//         int x = j - currentFigure->x;
//         int y = i - currentFigure->y;
//         if (x >= 0 && x < FIGURE_SIZE && y >= 0 && y < FIGURE_SIZE) {
//           if (currentFigure->figure[y][x] != 0) {
//             sym = currentFigure->figure[y][x];
//           }
//         }
//       }
//       if (sym != 0 && sym!=99 && sym!=98) {
//         mvaddch(i, j * 2, '0');  // Use '#' to represent the blocks of the
//         figure
//       } else {
//         mvaddch(i, j * 2, ' ');  // Empty cell
//       }
//     }
//   }

//   // Draw borders
// //   mvaddch(0, 0, '+');
// //   mvaddch(0, FIELD_WIDTH * 2 - 1, '+');
// //   mvaddch(FIELD_HEIGHT - 1, 0, '+');
// //   mvaddch(FIELD_HEIGHT - 1, FIELD_WIDTH * 2 - 1, '+');

// //   for (int i = 1; i < FIELD_WIDTH * 2 - 1; i++) {
// //     mvaddch(0, i, '-');
// //     mvaddch(FIELD_HEIGHT - 1, i, '-');
// //   }

// //   for (int i = 1; i < FIELD_HEIGHT - 1; i++) {
// //     mvaddch(i, 0, ')');
// //     mvaddch(i, FIELD_WIDTH * 2 - 1, '(');
// //   }
//     print_pole(0, FIELD_WIDTH, 0, FIELD_HEIGHT);
// }
void print_game(GameInfo_t* game, Figure* currentFigure) {
  int sym = 0;
  for (int i = 0; i < FIELD_HEIGHT; i++) {
    for (int j = 0; j < FIELD_WIDTH; j++) {
      sym = 0;
      if (game->field[i][j] != 0) {
        sym = game->field[i][j];
      } else {
        int x = j - currentFigure->x;
        int y = i - currentFigure->y;
        if (x >= 0 && x <= 5 && y >= 0 && y < 5) {
          if (currentFigure->figure[y][x] != 0) {
            sym = currentFigure->figure[y][x];
          }
        }
      }
      if (sym == 99) {
        mvaddch(i, j * 2, ACS_VLINE);
      } else if (sym == 98) {
        mvaddch(i, (j * 2) - 1, ACS_HLINE);
        mvaddch(i, (j * 2) + 1, ACS_HLINE);
        mvaddch(i, j * 2, ACS_HLINE);
      } else if (sym != 0) {
        mvaddch(i, j * 2, '0');
      } else {
        mvaddstr(i, j * 2, "  ");
      }
    }
  }
  mvaddch(0, 0, '+');
  mvaddch(0, 22, '+');
  mvaddch(21, 0, '+');
  mvaddch(21, 22, '+');
}

void print_pole(int left_x, int right_x, int up_y, int down_y) {
  mvaddch(up_y, left_x, '+');     // Top left corner
  mvaddch(up_y, right_x, '+');    // Top right corner
  mvaddch(down_y, left_x, '+');   // Bottom left corner
  mvaddch(down_y, right_x, '+');  // Bottom right corner

  // Draw horizontal lines
  for (int i = left_x + 1; i < right_x; i++) {
    mvaddch(up_y, i, ACS_HLINE);
    mvaddch(down_y, i, ACS_HLINE);
    // mvaddch(up_y, i, '-');
    // mvaddch(down_y, i, '-');
  }

  // Draw vertical lines
  for (int i = up_y + 1; i < down_y; i++) {
    mvaddch(i, left_x, ACS_VLINE);
    mvaddch(i, right_x, ACS_VLINE);
    // mvaddch(i, left_x, '|');
    // mvaddch(i, right_x, '|');
  }
}

void print_next_fig(GameInfo_t* game) {
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      if (game->next[i][j] != 0) {
        mvaddch(i + 11, (j + 13) * 2,
                '0');  // Use '#' to represent the blocks of the figure
      } else {
        mvaddch(i + 11, (j + 13) * 2, ' ');  // Empty space
      }
    }
  }
}

void print_info(GameInfo_t* game) {
  print_pole(24, 38, 1, 3);
  print_pole(24, 38, 4, 6);
  print_pole(24, 38, 7, 9);
  print_pole(24, 38, 10, 15);

  mvprintw(1, 25, " Score ");
  mvprintw(2, 26, "%d", game->score);

  mvprintw(4, 25, " High score ");
  mvprintw(5, 26, "%d", game->high_score);

  mvprintw(7, 25, " Level ");
  mvprintw(8, 26, "%d", game->level);

  if (game->next != NULL) {
    mvprintw(10, 25, " Next figure ");
    print_next_fig(game);
  }
  mvprintw(16, 25, "Action  Space");
  mvprintw(17, 25, "Pause   p");
  mvprintw(18, 25, "Exit    q");
}

void pause_game() {
  while (getch() != 'p') {
    mvaddstr(10, 4, "Pause");
    refresh();
  }
}

void start_game() {
  while (getch() != '\n') {
    mvaddstr(10, 4, "To start the game, press Enter");
    refresh();
  }
}

