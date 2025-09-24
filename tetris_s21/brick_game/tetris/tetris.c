#include "../../game.h"

int get_random_int(int min, int max) { return rand() % (max - min + 1) + min; }

void generate_next_figure(GameInfo_t *game) {
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      game->next[i][j] = 0;
    }
  }

  int type = get_random_int(1, 7);
  switch (type) {
    case 1:
      for (int i = 1; i < 5; i++) {
        game->next[2][i] = type;
      }
      break;

    case 2:
      game->next[1][1] = type;
      for (int i = 1; i < 4; i++) {
        game->next[2][i] = type;
      }
      break;

    case 3:
      game->next[1][3] = type;
      for (int i = 1; i < 4; i++) {
        game->next[2][i] = type;
      }
      break;

    case 4:
      for (int i = 1; i < 3; i++) {
        game->next[1][i] = type;
        game->next[2][i] = type;
      }
      break;

    case 5:
      for (int i = 1; i < 3; i++) {
        game->next[1][i + 1] = type;
        game->next[2][i] = type;
      }
      break;

    case 6:
      game->next[1][2] = type;
      for (int i = 1; i < 4; i++) {
        game->next[2][i] = type;
      }
      break;

    case 7:
      for (int i = 1; i < 3; i++) {
        game->next[1][i] = type;
        game->next[2][i + 1] = type;
      }
      break;
  }
}

void create_next_figure(GameInfo_t *game) {
  game->next = (int **)calloc(sizeof(int *), FIGURE_SIZE);
  for (int i = 0; i < FIGURE_SIZE; i++) {
    game->next[i] = (int *)calloc(sizeof(int), FIGURE_SIZE);
  }
  generate_next_figure(game);
}

void free_next_figure(GameInfo_t *game) {
  if (game->next) {
    for (int i = 0; i < 5; i++) {
      if (game->next[i]) {
        free(game->next[i]);
      }
    }
    free(game->next);
  }
}

void create_field(GameInfo_t *game) {
  game->field = (int **)calloc(sizeof(int *), FIELD_HEIGHT);
  for (int k = 0; k < FIELD_HEIGHT; k++) {
    game->field[k] = (int *)calloc(sizeof(int), FIELD_WIDTH);
  }
  for (int i = 0; i < FIELD_HEIGHT; i++) {
    for (int j = 0; j < FIELD_WIDTH; j++) {
      if (j == 0 || j == 11) {
        game->field[i][j] = 99;
      } else if (i == 0 || i == 21) {
        game->field[i][j] = 98;
      } else {
        game->field[i][j] = 0;
      }
    }
  }
}

void free_field(GameInfo_t *game) {
  if (game->field) {
    for (int i = 0; i < FIELD_HEIGHT; i++) {
      if (game->field[i]) {
        free(game->field[i]);
      }
    }
    free(game->field);
  }
}

void set_hiscore(GameInfo_t *game) {
  FILE *high_score = fopen("progress.txt", "w");
  if (high_score != NULL) {
    if (game->score > game->high_score)
      fprintf(high_score, "%d", game->score);
    else
      fprintf(high_score, "%d", game->high_score);
    fclose(high_score);
  }
}

GameInfo_t *create_game() {
  GameInfo_t *game = (GameInfo_t *)calloc(sizeof(GameInfo_t), 1);
  create_field(game);
  create_next_figure(game);
  game->score = 0;
  game->level = 1;
  game->speed = 1;
  game->pause = 0;
  game->playing_status = TET_PLAYING;
  FILE *file = fopen("progress.txt", "r");
  fscanf(file, "%d", &(game->high_score));
  fclose(file);
  return game;
}

void free_game(GameInfo_t *game) {
  free_field(game);
  free_next_figure(game);
  free(game);
}

void move_figure_down(Figure *currentFigure) { currentFigure->y++; }

void move_figure_up(Figure *currentFigure) { currentFigure->y--; }

void move_figure_left(Figure *currentFigure) { currentFigure->x--; }

void move_figure_right(Figure *currentFigure) { currentFigure->x++; }

int colision(Figure *currentFigure, GameInfo_t *game) {
  int result = 0;
  for (int i = 0; i < FIGURE_SIZE && !result; i++) {
    for (int j = 0; j < FIGURE_SIZE && !result; j++) {
      if (currentFigure->figure[i][j] != 0) {
        int x = currentFigure->x + j;
        int y = currentFigure->y + i;
        if (x < 0 || x >= 11 || y < 0 || y >= 21) {
          result = 1;
        } else if (game->field[y][x] != 0) {
          result = 1;
        }
      }
    }
  }
  return result;
}

void plant_figure(Figure *currentFigure, GameInfo_t *game) {
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      if (currentFigure->figure[i][j] != 0) {
        int x = currentFigure->x + j;
        int y = currentFigure->y + i;
        game->field[y][x] = currentFigure->figure[i][j];
      }
    }
  }
}

int line_full(int i, GameInfo_t *game) {
  int result = 1;
  for (int j = 1; j < 11 && result; j++) {
    if (game->field[i][j] == 0) {
      result = 0;
    }
  }
  return result;
}

void drope_line(int i, GameInfo_t *game) {
  if (i == 1) {
    for (int j = 1; j < 11; j++) {
      game->field[i][j] = 0;
    }
  } else {
    for (int k = i; k > 1; k--) {
      for (int j = 1; j < 11; j++) {
        game->field[k][j] = game->field[k - 1][j];
      }
    }
  }
}

int erase_lines(GameInfo_t *game) {
  int count = 0;
  for (int i = 20; i >= 1; i--) {
    while (line_full(i, game)) {
      drope_line(i, game);
      count++;
    }
  }
  if (count == 1) {
    count = 100;
  } else if (count == 2) {
    count = 300;
  } else if (count == 3) {
    count = 700;
  } else if (count == 4) {
    count = 1500;
  }
  return count;
}

Figure *create_figure() {
  Figure *figure = (Figure *)calloc(sizeof(Figure), 1);
  figure->figure = (int **)calloc(sizeof(int *), FIGURE_SIZE);
  for (int k = 0; k < FIGURE_SIZE; k++) {
    figure->figure[k] = (int *)calloc(sizeof(int), FIGURE_SIZE);
  }
  return figure;
}

void free_figure(Figure *fig) {
  if (fig && fig->figure) {
    for (int i = 0; i < FIGURE_SIZE; i++) {
      if (fig->figure[i]) {
        free(fig->figure[i]);
      }
    }
    free(fig->figure);
    free(fig);
  }
}

void drop_new_figure(Figure *currentFigure, GameInfo_t *game) {
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      currentFigure->figure[i][j] = game->next[i][j];
    }
  }
  currentFigure->x = 3;
  currentFigure->y = 0;
}

Figure *rot_figure(Figure *currentFigure) {
  Figure *rot_figure = create_figure();
  rot_figure->x = currentFigure->x;
  rot_figure->y = currentFigure->y;
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      rot_figure->figure[j][4 - i] = currentFigure->figure[i][j];
    }
  }
  return rot_figure;
}

void user_input(UserAction_t action, Figure *currentFigure, GameInfo_t *game,
                Time *time, State *state) {
  switch (action) {
    case Right:
      move_figure_right(currentFigure);
      if (colision(currentFigure, game)) {
        move_figure_left(currentFigure);
      }
      break;
    case Left:
      move_figure_left(currentFigure);
      if (colision(currentFigure, game)) {
        move_figure_right(currentFigure);
      }
      break;
    case Down:
      while (!colision(currentFigure, game)) {
        move_figure_down(currentFigure);
      }
      if (colision(currentFigure, game)) {
        move_figure_up(currentFigure);
      }
      break;
    case Action:
      Figure *rotFigure = rot_figure(currentFigure);
      if (colision(rotFigure, game)) {
        free_figure(rotFigure);
      } else {
        for (int i = 0; i < FIGURE_SIZE; i++) {
          for (int j = 0; j < FIGURE_SIZE; j++) {
            currentFigure->figure[i][j] = rotFigure->figure[i][j];
          }
        }
        free_figure(rotFigure);
      }
      break;
    case Pause:
      pause_game();
      break;
    default:
      time->ticks = TET_TICKS_START - game->level * 3;
      if (time->ticks_left <= 0) {
        time->ticks_left = time->ticks;
        *state = SHIFTING;
      }
      time->ticks_left--;
      break;
  }
}

void level_monitoring(GameInfo_t *game) {
  game->level = (game->score / 600) + 1 < 10 ? (game->score / 600) + 1 : 10;
}

void avtomat(GameInfo_t *game, Figure *currentFigure, Time *time,
             UserAction_t action, State *state) {
  switch (*state) {
    case START:
      start_game();
      clear();
      *state = SPAWN;
      break;
    case SPAWN:
      drop_new_figure(currentFigure, game);
      create_next_figure(game);
      if (colision(currentFigure, game)) {
        *state = GAMEOVER;
      } else {
        *state = MOVING;
      }
      break;
    case MOVING:
      user_input(action, currentFigure, game, time, state);
      level_monitoring(game);
      if (colision(currentFigure, game)) {
        move_figure_up(currentFigure);
        *state = ATTACHING;
      }
      break;
    case SHIFTING:
      if (colision(currentFigure, game)) {
        *state = ATTACHING;
      } else {
        move_figure_down(currentFigure);
        *state = MOVING;
      }
      break;
    case ATTACHING:
      plant_figure(currentFigure, game);
      game->score += erase_lines(game);
      if (game->score > game->high_score) {
        game->high_score = game->score;
      }
      *state = SPAWN;
      break;
    case GAMEOVER:
      game->playing_status = TET_GAMEOVER;
      break;
    default:
      break;
  }
}
