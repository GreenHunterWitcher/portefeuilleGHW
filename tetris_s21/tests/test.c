#include <check.h>

#include "../game.h"

START_TEST(test_createGame) {
  GameInfo_t* game = create_game();
  ck_assert_ptr_nonnull(game);
  ck_assert_int_eq(game->score, 0);
  ck_assert_int_eq(game->level, 1);
  ck_assert_int_eq(game->playing_status, TET_PLAYING);
  free_game(game);
}
END_TEST

START_TEST(test_freeGame) {
  GameInfo_t* game = create_game();
  free_game(game);
}
END_TEST

START_TEST(test_generateNextFigure) {
  GameInfo_t* game = create_game();
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  generate_next_figure(game);
  ck_assert_int_ne(game->next[2][2], 0);
  free_game(game);
}
END_TEST

START_TEST(test_set_hiscore_max) {
  int old_score;
  FILE* file = fopen("progress.txt", "r");
  ck_assert_ptr_nonnull(file);
  fscanf(file, "%d", &(old_score));
  GameInfo_t* game = create_game();
  ck_assert_int_eq(game->high_score, old_score);
  game->score = 2147483647;
  set_hiscore(game);
  fclose(file);

  FILE* file_hscore = fopen("progress.txt", "r");
  fscanf(file_hscore, "%d", &(game->high_score));
  fclose(file_hscore);

  ck_assert_int_eq(game->high_score, 2147483647);

  FILE* file_score = fopen("progress.txt", "w");
  fprintf(file_score, "%d", old_score);
  fclose(file_score);

  free_game(game);
}
END_TEST

START_TEST(test_set_hiscore_min) {
  int old_score;
  FILE* file = fopen("progress.txt", "r");
  ck_assert_ptr_nonnull(file);
  fscanf(file, "%d", &(old_score));
  GameInfo_t* game = create_game();
  ck_assert_int_eq(game->high_score, old_score);
  game->high_score = 2147483647;
  set_hiscore(game);
  fclose(file);

  FILE* file_hscore = fopen("progress.txt", "r");
  fscanf(file_hscore, "%d", &(game->high_score));
  fclose(file_hscore);

  ck_assert_int_eq(game->high_score, 2147483647);

  FILE* file_score = fopen("progress.txt", "w");
  fprintf(file_score, "%d", old_score);
  fclose(file_score);

  free_game(game);
}
END_TEST

START_TEST(test_moveFigureDown) {
  Figure* figure = create_figure();
  figure->x = 3;
  figure->y = 0;
  move_figure_down(figure);
  ck_assert_int_eq(figure->y, 1);
  free_figure(figure);
}
END_TEST

START_TEST(test_moveFigureUp) {
  Figure* figure = create_figure();
  figure->x = 3;
  figure->y = 1;
  move_figure_up(figure);
  ck_assert_int_eq(figure->y, 0);
  free_figure(figure);
}
END_TEST

START_TEST(test_moveFigureLeft) {
  Figure* figure = create_figure();
  figure->x = 3;
  figure->y = 1;
  move_figure_left(figure);
  ck_assert_int_eq(figure->x, 2);
  free_figure(figure);
}
END_TEST

START_TEST(test_moveFigureRight) {
  Figure* figure = create_figure();
  figure->x = 3;
  figure->y = 1;
  move_figure_right(figure);
  ck_assert_int_eq(figure->x, 4);
  free_figure(figure);
}
END_TEST

START_TEST(test_colision_figure) {
  Figure* figure = create_figure();
  GameInfo_t* game = create_game();
  figure->x = 1;
  figure->y = 1;
  figure->figure[1][1] = 1;
  game->field[2][2] = 1;
  ck_assert_int_eq(colision(figure, game), 1);

  figure->figure[1][1] = 0;
  figure->figure[2][1] = 1;
  game->field[2][2] = 1;
  ck_assert_int_eq(colision(figure, game), 0);
  free_game(game);
  free_figure(figure);
}
END_TEST

START_TEST(test_colision_pole) {
  Figure* figure = create_figure();
  GameInfo_t* game = create_game();
  figure->x = 10;
  figure->y = 1;
  figure->figure[1][1] = 1;
  ck_assert_int_eq(colision(figure, game), 1);

  figure->x = 9;
  figure->y = 1;
  figure->figure[1][1] = 1;
  ck_assert_int_eq(colision(figure, game), 0);

  free_game(game);
  free_figure(figure);
}
END_TEST

START_TEST(test_plantFigure) {
  Figure* figure = create_figure();
  GameInfo_t* game = create_game();
  figure->x = 1;
  figure->y = 1;
  figure->figure[1][1] = 1;
  figure->figure[2][1] = 1;
  plant_figure(figure, game);
  ck_assert_int_eq(game->field[2][2], 1);
  ck_assert_int_eq(game->field[1][2], 0);
  ck_assert_int_eq(game->field[3][2], 1);
  ck_assert_int_eq(game->field[3][3], 0);

  free_game(game);
  free_figure(figure);
}
END_TEST

START_TEST(test_lineFull) {
  GameInfo_t* game = create_game();
  for (int j = 1; j < 11; j++) {
    game->field[1][j] = 1;
  }
  ck_assert_int_eq(line_full(1, game), 1);
  ck_assert_int_eq(line_full(2, game), 0);
  free_game(game);
}
END_TEST

START_TEST(test_dropeLine) {
  GameInfo_t* game = create_game();
  for (int j = 1; j < 11; j++) {
    game->field[1][j] = 1;
  }
  drope_line(1, game);
  for (int j = 1; j < 11; j++) {
    ck_assert_int_eq(game->field[1][j], 0);
  }

  for (int j = 1; j < 11; j++) {
    game->field[3][j] = 1;
  }
  drope_line(3, game);
  for (int j = 1; j < 11; j++) {
    ck_assert_int_eq(game->field[3][j], 0);
  }

  free_game(game);
}
END_TEST

START_TEST(test_eraseLines) {
  GameInfo_t* game = create_game();
  for (int j = 1; j < 11; j++) {
    game->field[1][j] = 1;
  }
  ck_assert_int_eq(erase_lines(game), 100);

  for (int i = 2; i < 4; i++) {
    for (int j = 1; j < 11; j++) {
      game->field[i][j] = 1;
    }
  }
  ck_assert_int_eq(erase_lines(game), 300);

  for (int i = 4; i < 7; i++) {
    for (int j = 1; j < 11; j++) {
      game->field[i][j] = 1;
    }
  }
  ck_assert_int_eq(erase_lines(game), 700);

  for (int i = 7; i < 11; i++) {
    for (int j = 1; j < 11; j++) {
      game->field[i][j] = 1;
    }
  }
  ck_assert_int_eq(erase_lines(game), 1500);

  ck_assert_int_eq(erase_lines(game), 0);

  free_game(game);
}
END_TEST

START_TEST(test_dropNewFigure) {
  GameInfo_t* game = create_game();
  Figure* figure = create_figure();
  drop_new_figure(figure, game);

  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      ck_assert_int_eq(figure->figure[i][j], game->next[i][j]);
    }
  }
  ck_assert_int_eq(figure->x, 3);
  ck_assert_int_eq(figure->y, 0);

  free_figure(figure);
  free_game(game);
}
END_TEST

START_TEST(test_rotFigure) {
  Figure* figure = create_figure();
  figure->figure[1][1] = 1;
  for (int i = 1; i < 4; i++) {
    figure->figure[2][i] = 1;
  }
  Figure* rotFigure = rot_figure(figure);
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      ck_assert_int_eq(figure->figure[i][j], rotFigure->figure[j][4 - i]);
    }
  }
  free_figure(rotFigure);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      figure->figure[i][j] = 0;
    }
  }
  for (int i = 1; i < 5; i++) {
    figure->figure[2][i] = 1;
  }
  rotFigure = rot_figure(figure);
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      ck_assert_int_eq(figure->figure[i][j], rotFigure->figure[j][4 - i]);
    }
  }
  free_figure(rotFigure);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      figure->figure[i][j] = 0;
    }
  }
  figure->figure[1][3] = 1;
  for (int i = 1; i < 4; i++) {
    figure->figure[2][i] = 1;
  }
  rotFigure = rot_figure(figure);
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      ck_assert_int_eq(figure->figure[i][j], rotFigure->figure[j][4 - i]);
    }
  }
  free_figure(rotFigure);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      figure->figure[i][j] = 0;
    }
  }
  for (int i = 1; i < 3; i++) {
    figure->figure[1][i] = 1;
    figure->figure[2][i] = 1;
  }
  rotFigure = rot_figure(figure);
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      ck_assert_int_eq(figure->figure[i][j], rotFigure->figure[j][4 - i]);
    }
  }
  free_figure(rotFigure);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      figure->figure[i][j] = 0;
    }
  }
  for (int i = 1; i < 3; i++) {
    figure->figure[1][i + 1] = 1;
    figure->figure[2][i] = 1;
  }
  rotFigure = rot_figure(figure);
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      ck_assert_int_eq(figure->figure[i][j], rotFigure->figure[j][4 - i]);
    }
  }
  free_figure(rotFigure);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      figure->figure[i][j] = 0;
    }
  }
  figure->figure[1][2] = 1;
  for (int i = 1; i < 4; i++) {
    figure->figure[2][i] = 1;
  }
  rotFigure = rot_figure(figure);
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      ck_assert_int_eq(figure->figure[i][j], rotFigure->figure[j][4 - i]);
    }
  }
  free_figure(rotFigure);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      figure->figure[i][j] = 0;
    }
  }
  for (int i = 1; i < 3; i++) {
    figure->figure[1][i] = 1;
    figure->figure[2][i + 1] = 1;
  }
  rotFigure = rot_figure(figure);
  for (int i = 0; i < FIGURE_SIZE; i++) {
    for (int j = 0; j < FIGURE_SIZE; j++) {
      ck_assert_int_eq(figure->figure[i][j], rotFigure->figure[j][4 - i]);
    }
  }

  free_figure(rotFigure);
  free_figure(figure);
}
END_TEST

START_TEST(test_userInput) {
  GameInfo_t* game = create_game();
  Figure* figure = create_figure();
  UserAction_t player;
  Time time;
  time.ticks = 0;
  time.ticks_left = 0;
  State state;
  state = START;

  figure->figure[1][1] = 1;
  player = Right;
  user_input(player, figure, game, &time, &state);
  ck_assert_int_eq(figure->x, 1);

  figure->x = 3;
  figure->y = 0;
  player = Left;
  user_input(player, figure, game, &time, &state);
  ck_assert_int_eq(figure->x, 2);

  figure->x = 0;
  figure->y = 0;
  player = Down;
  user_input(player, figure, game, &time, &state);
  ck_assert_int_eq(figure->y, 19);

  figure->x = 3;
  figure->y = 0;
  player = Action;
  user_input(player, figure, game, &time, &state);

  figure->x = 10;
  figure->y = 0;
  player = Action;
  user_input(player, figure, game, &time, &state);

  figure->x = 0;
  figure->y = 0;
  player = Up;
  user_input(player, figure, game, &time, &state);
  ck_assert_int_eq(state, SHIFTING);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      figure->figure[i][j] = 0;
    }
  }
  figure->x = 0;
  figure->y = 0;

  player = Right;
  figure->x = 10;
  figure->figure[1][4] = 1;
  user_input(player, figure, game, &time, &state);
  ck_assert_int_eq(figure->x, 10);

  player = Left;
  figure->x = 0;
  figure->figure[1][1] = 1;
  user_input(player, figure, game, &time, &state);
  ck_assert_int_eq(figure->x, 0);

  free_game(game);
  free_figure(figure);
}
END_TEST

START_TEST(test_level_monitoring) {
  GameInfo_t* game = create_game();

  for (int i = 1; i < 10; i++) {
    game->score = 600 * i;
    level_monitoring(game);
    ck_assert_int_eq(game->level, i + 1);
  }

  game->score = 5000000;
  level_monitoring(game);
  ck_assert_int_eq(game->level, 10);

  free_game(game);
}
END_TEST

START_TEST(test_avtomat) {
  Figure* figure = create_figure();
  GameInfo_t* game = create_game();
  Time time;
  time.ticks = 0;
  time.ticks_left = 0;
  UserAction_t player;
  player = Down;
  State state;

  state = SPAWN;
  avtomat(game, figure, &time, player, &state);
  ck_assert_int_eq(state, MOVING);

  state = MOVING;
  game->score = 600;
  avtomat(game, figure, &time, player, &state);
  ck_assert_int_eq(state, MOVING);
  ck_assert_int_eq(game->level, 2);

  state = SHIFTING;
  avtomat(game, figure, &time, player, &state);
  ck_assert_int_eq(state, MOVING);
  ck_assert_int_eq(figure->x, 3);

  state = ATTACHING;
  for (int i = 2; i < 4; i++) {
    for (int j = 1; j < 11; j++) {
      game->field[i][j] = 1;
    }
  }
  avtomat(game, figure, &time, player, &state);
  ck_assert_int_eq(game->score, 900);
  ck_assert_int_eq(state, SPAWN);

  state = ATTACHING;
  game->score = 5000000;
  for (int i = 2; i < 4; i++) {
    for (int j = 1; j < 11; j++) {
      game->field[i][j] = 1;
    }
  }
  avtomat(game, figure, &time, player, &state);
  ck_assert_int_eq(game->score, 5000300);
  ck_assert_int_eq(state, SPAWN);

  state = GAMEOVER;
  avtomat(game, figure, &time, player, &state);
  ck_assert_int_eq(game->playing_status, TET_GAMEOVER);

  state = 6;
  avtomat(game, figure, &time, player, &state);

  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      figure->figure[i][j] = 0;
    }
  }

  state = SPAWN;
  figure->x = 3;
  figure->y = 0;
  for (int i = 1; i < 3; i++) {
    for (int j = 1; j < 11; j++) {
      game->field[i][j] = 1;
    }
  }
  avtomat(game, figure, &time, player, &state);
  ck_assert_int_eq(state, GAMEOVER);

  state = MOVING;
  figure->x = 3;
  figure->y = 1;
  figure->figure[1][1] = 1;
  player = Up;
  for (int j = 1; j < 11; j++) {
    game->field[2][j] = 1;
  }
  avtomat(game, figure, &time, player, &state);
  ck_assert_int_eq(state, ATTACHING);
  ck_assert_int_eq(figure->y, 0);

  state = SHIFTING;
  figure->x = 3;
  figure->y = 0;
  figure->figure[1][1] = 1;
  avtomat(game, figure, &time, player, &state);
  ck_assert_int_eq(state, ATTACHING);

  free_figure(figure);
  free_game(game);
}
END_TEST

Suite* tetris_tests_suite(void) {
  Suite* s;
  TCase* tc_core;

  s = suite_create("Tetris Core Tests");

  tc_core = tcase_create("Core");

  tcase_add_test(tc_core, test_createGame);
  tcase_add_test(tc_core, test_freeGame);
  tcase_add_test(tc_core, test_generateNextFigure);
  tcase_add_test(tc_core, test_set_hiscore_max);
  tcase_add_test(tc_core, test_set_hiscore_min);
  tcase_add_test(tc_core, test_moveFigureDown);
  tcase_add_test(tc_core, test_moveFigureUp);
  tcase_add_test(tc_core, test_moveFigureLeft);
  tcase_add_test(tc_core, test_moveFigureRight);
  tcase_add_test(tc_core, test_colision_figure);
  tcase_add_test(tc_core, test_colision_pole);
  tcase_add_test(tc_core, test_plantFigure);
  tcase_add_test(tc_core, test_lineFull);
  tcase_add_test(tc_core, test_dropeLine);
  tcase_add_test(tc_core, test_eraseLines);
  tcase_add_test(tc_core, test_dropNewFigure);
  tcase_add_test(tc_core, test_rotFigure);
  tcase_add_test(tc_core, test_userInput);
  tcase_add_test(tc_core, test_level_monitoring);
  tcase_add_test(tc_core, test_avtomat);

  suite_add_tcase(s, tc_core);

  return s;
}

int main() {
  srand(time(NULL));
  int failed = 0;
  Suite* result[] = {tetris_tests_suite(), NULL};

  for (int i = 0; result[i] != NULL; i++) {
    SRunner* runner = srunner_create(result[i]);
    srunner_run_all(runner, CK_NORMAL);
    failed += srunner_ntests_failed(runner);
    srunner_free(runner);
  }

  return (failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
