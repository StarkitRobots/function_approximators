set(SOURCES
  adaptative_tree.cpp
  constant_approximator.cpp
  fa_tree.cpp
  fake_split.cpp
  forest_approximator.cpp
  forest_trainer.cpp
  function_approximator.cpp
  function_approximator_factory.cpp
  linear_approximator.cpp
  linear_split.cpp
  optimizer_trainer.cpp
  optimizer_trainer_factory.cpp
  orthogonal_split.cpp
  point_split.cpp
  pwc_forest.cpp
  pwc_forest_trainer.cpp
  pwl_forest.cpp
  pwl_forest_trainer.cpp
  split.cpp
  split_factory.cpp
  trainer.cpp
  trainer_factory.cpp
)

if (RHOBAN_FA_USES_DNN)
  set(SOURCES ${SOURCES}
    dnn_approximator.cpp
    dnn_approximator_trainer.cpp
    )
endif(RHOBAN_FA_USES_DNN)

if (RHOBAN_FA_USES_GP)
  set (SOURCES ${SOURCES}
    gp.cpp
    gp_trainer.cpp
    gp_forest.cpp
    gp_forest_trainer.cpp
    )
endif(RHOBAN_FA_USES_GP)
