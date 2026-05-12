#pragma once

#ifdef LLAMA2
    #define N_ROWS 11008
    #define N_COLS 4096
#elif defined(QWEN2)
    #define N_ROWS 18944
    #define N_COLS 3584
#elif defined(MISTRAL)
    #define N_ROWS 14336
    #define N_COLS 4096
#endif