/*
	Non-separable 2D, 3D and 4D Filtering with CUDA
	Copyright (C) <2013>  Anders Eklund, andek034@gmail.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
