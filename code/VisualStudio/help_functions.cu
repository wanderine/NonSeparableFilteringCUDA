#ifndef HELP_FUNCTIONS_CU_
#define HELP_FUNCTIONS_CU_


__device__ int Get_2D_Index(int x, int y, int DATA_W)
{
    return x + y * DATA_W;
}

__device__ int Get_3D_Index(int x, int y, int z, int DATA_W, int DATA_H)
{
    return x + y * DATA_W + z * DATA_W * DATA_H;
}

__device__ int Get_4D_Index(int x, int y, int z, int t, int DATA_W, int DATA_H, int DATA_D)
{
    return x + y * DATA_W + z * DATA_W * DATA_H + t * DATA_W * DATA_H * DATA_D;
}






#endif
