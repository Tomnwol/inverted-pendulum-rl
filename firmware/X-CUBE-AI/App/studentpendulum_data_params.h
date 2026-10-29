/**
  ******************************************************************************
  * @file    studentpendulum_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-10-29T23:44:01+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef STUDENTPENDULUM_DATA_PARAMS_H
#define STUDENTPENDULUM_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_STUDENTPENDULUM_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_studentpendulum_data_weights_params[1]))
*/

#define AI_STUDENTPENDULUM_DATA_CONFIG               (NULL)


#define AI_STUDENTPENDULUM_DATA_ACTIVATIONS_SIZES \
  { 128, }
#define AI_STUDENTPENDULUM_DATA_ACTIVATIONS_SIZE     (128)
#define AI_STUDENTPENDULUM_DATA_ACTIVATIONS_COUNT    (1)
#define AI_STUDENTPENDULUM_DATA_ACTIVATION_1_SIZE    (128)



#define AI_STUDENTPENDULUM_DATA_WEIGHTS_SIZES \
  { 1412, }
#define AI_STUDENTPENDULUM_DATA_WEIGHTS_SIZE         (1412)
#define AI_STUDENTPENDULUM_DATA_WEIGHTS_COUNT        (1)
#define AI_STUDENTPENDULUM_DATA_WEIGHT_1_SIZE        (1412)



#define AI_STUDENTPENDULUM_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_studentpendulum_activations_table[1])

extern ai_handle g_studentpendulum_activations_table[1 + 2];



#define AI_STUDENTPENDULUM_DATA_WEIGHTS_TABLE_GET() \
  (&g_studentpendulum_weights_table[1])

extern ai_handle g_studentpendulum_weights_table[1 + 2];


#endif    /* STUDENTPENDULUM_DATA_PARAMS_H */
