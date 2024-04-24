#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

#define C_SQ		(1.0 / 3.0) /* square of speed of sound */
#define W0			(4.0 / 9.0)  /* weighting factor */
#define W1			(W0 / 4)
#define W2			(W1 / 4)
#define IN_C_SQ	    1/C_SQ
#define A	        1.f/(2.f * C_SQ * C_SQ)
#define C_SQ_2		(1.0 / ((C_SQ * C_SQ) + (C_SQ * C_SQ)))
#define C_SQ_R_2	(3.0 / 2.0)
#define INDEX(ii,jj,nx,ny,speed) (((nx)*(ny)*(speed))+ ((ii)*(nx)+(jj)))

#ifndef BLOCK_I
#define BLOCK_I 16
#endif
#ifndef BLOCK_J
#define BLOCK_J 16
#endif

typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

kernel void accelerate_flow(global t_speed* cells,
                            global int* obstacles,
                            int nx, int ny,
                            float density, float accel)
{
  /* compute weighting factors */
  float w1 = density * accel / 9.0;
  float w2 = density * accel / 36.0;

  /* modify the 2nd row of the grid */
  int jj = ny - 2;

  /* get column index */
  int ii = get_global_id(0);

  /* if the cell is not occupied and
  ** we don't send a negative density */
  if (!obstacles[ii + jj* nx]
      && (cells[ii + jj* nx].speeds[3] - w1) > 0.f
      && (cells[ii + jj* nx].speeds[6] - w2) > 0.f
      && (cells[ii + jj* nx].speeds[7] - w2) > 0.f)
  {
    /* increase 'east-side' densities */
    cells[ii + jj* nx].speeds[1] += w1;
    cells[ii + jj* nx].speeds[5] += w2;
    cells[ii + jj* nx].speeds[8] += w2;
    /* decrease 'west-side' densities */
    cells[ii + jj* nx].speeds[3] -= w1;
    cells[ii + jj* nx].speeds[6] -= w2;
    cells[ii + jj* nx].speeds[7] -= w2;
  }
}

/*kernel void propagate(global t_speed* cells, global t_speed* tmp_cells, global int* obstacles, int nx, int ny, float omega, global float* tt_vels){
    *//* get column and row indices *//*
    int ii = get_global_id(0);
    int jj = get_global_id(1);

    float local_tot_u[128];

    int jj_local = get_local_id(0);
    int ii_local = get_local_id(1);
    int nx_local = get_local_size(0);
    int ny_local = get_local_size(1);
    int local_index = ii_local*nx_local + jj_local;
    int local_size = nx_local * ny_local;

    int idx = ii + jj * nx;
    *//* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) *//*
    int y_n = ((ii + 1) & (ny-1));
    int y_s = ((ii == 0) ? (ii + ny - 1) : (ii - 1));
    int x_e = ((jj + 1) & (nx-1));
    int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);

    float tot_u = 0.f;
    *//* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid *//*
    if(local_index == 0 && ii == 0 && jj == 0)printf("%f\n", cells[0].speeds[0]);

    float speeds[9];
    speeds[0] = cells[ii + jj*nx].speeds[0];
    speeds[1] = cells[x_w + jj*nx].speeds[1];
    speeds[3] = cells[x_e + jj*nx].speeds[3];
    speeds[5] = cells[x_w + y_s*nx].speeds[5];
    speeds[2] = cells[ii + y_s*nx].speeds[2];
    speeds[6] = cells[x_e + y_s*nx].speeds[6];
    speeds[8] = cells[x_w + y_n*nx].speeds[8];
    speeds[4] = cells[ii + y_n*nx].speeds[4];
    speeds[7] = cells[x_e + y_n*nx].speeds[7];
    int obstacle = !((int)obstacles[jj*nx + ii]);
    *//* compute local density total *//*
    float local_density = 0.0;

    #pragma unroll
    for (int i = 0; i < 9; i++) {
        local_density += speeds[i];
    }

    float a = speeds[8] - speeds[6];
    float b = speeds[5] - speeds[7];
    float c = speeds[1] - speeds[3];
    float d = speeds[2] - speeds[4];
    *//* compute x velocity component *//*
    float u_x = b;
    float u_y = b;
    u_x = u_x + a + c;
    u_y = u_y + d - a;

    u_x = native_divide(u_x,local_density);
    u_y = native_divide(u_y, local_density);
    float w_local = W0 * local_density;
    *//* velocity squared *//*

    *//* directional velocity components *//*


    //Compiler can you please vectorize this ty xoxo

    //before: 9, now:
    float u_x_y = u_x * u_y * 2.0 * C_SQ_2;
    float u2[NSPEEDS];
    u2[1] = u_x * u_x;
    u2[2] = u_y * u_y;
    float u_sq = (u2[1] + u2[2]);
    float u_sq_recip = u_sq *C_SQ_R_2 - 1.0;

    u2[0] = 0;
    u2[1] = u2[1] * C_SQ_2 - u_sq_recip;
    u2[2] *= C_SQ_2;
    u2[3] = u2[1];
    u2[4] = u2[2] - u_sq_recip;
    u2[5] = u2[1] + u2[2] + u_x_y;
    u2[6] = u2[1] + u2[2] - u_x_y;
    u2[7] = u2[5];
    u2[8] = u2[6];
    u2[2] -= u_sq_recip;
    tot_u = native_sqrt(u_sq)*obstacle;
    u_x  = native_divide(u_x, (float)C_SQ);
    u_y = native_divide(u_y, (float)C_SQ);
    float u[NSPEEDS];
    u[0] = -u_sq_recip;
    u[1] = u_x + u2[1];
    u[2] = u_y + u2[2];
    u[3] = -u_x + u2[3];
    u[4] = -u_y + u2[4];
    u[5] = u_x + u_y + u2[5];
    u[6] = -u_x + u_y + u2[6];
    u[7] = -u_x - u_y + u2[7];
    u[8] = u_x - u_y + u2[8];

    u[0] = w_local * (u[0]) - speeds[0];
    u[0] = (speeds[0] + omega*u[0])*obstacle;
    float w1_local = W1 * local_density;
    #pragma unroll
    for (int i = 1; i < 5; i++) {
        float a = w1_local * (u[i]) - speeds[i];
        u[i] = speeds[i] + omega*a;
    }
    float w2_local = W2 * local_density;
    #pragma unroll
    for (int i = 5; i < 9; i++) {
        float a = w2_local * (u[i]) - speeds[i];
        u[i] = speeds[i] + omega*a;
    }
    if(!obstacle){
        u[1] = speeds[3];
        u[2] = speeds[4];
        u[3] = speeds[1];
        u[4] = speeds[2];
        u[5] = speeds[7];
        u[6] = speeds[8];
        u[7] = speeds[5];
        u[8] = speeds[6];
    }
    //printf("%f\n", tot_u);
    //__local float local_tot_u[LOCAL_SIZE];

    *//*local_tot_u[local_index] = tot_u;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = get_local_size(0) / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            local_tot_u[local_index] += local_tot_u[local_index + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);  // Synchronize at each step of the reduction
    }

    // The first thread in each workgroup writes the local sum to the global memory
    if (local_index == 0) {
        tt_vels[get_group_id(0)] = local_tot_u[0];
    }*//*
    local_tot_u[local_index] = tot_u;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i=0; i < 9 ; i++){
        tmp_cells[ii + jj*nx].speeds[i] = u[i];
    }

#pragma unroll
    for(int offset = local_size/2; offset > 0; offset = offset / 2){
        if(local_index < offset){
            float other = local_tot_u[local_index + offset];
            float mine =local_tot_u[local_index];
            local_tot_u[local_index] = mine + other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_index == 0){
        //printf("%f\n", tot_u);
        tt_vels[get_group_id(0)] = local_tot_u[0];
    }

}*/

kernel void propagate(global t_speed* cells, global t_speed* tmp_cells, global int* obstacles, int nx, int ny, float omega,  global float* tot_vel)
{
    local float scratch[64*2];

    int ii = get_global_id(0);
    int jj = get_global_id(1);

    int jj_local = get_local_id(0);
    int ii_local = get_local_id(1);

    int nx_local = get_local_size(0);
    int ny_local = get_local_size(1);
    float tot_u = 0;
    int local_index = ii_local*nx_local + jj_local;
    int local_size = nx_local * ny_local;
    int y_n = ((ii + 1) & (ny-1));
    int y_s = ((ii == 0) ? (ii + ny - 1) : (ii - 1));
    //wait until flow acceleration has been computed

    /*Propagate step*/
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int x_e = ((jj + 1) & (nx-1));
    int x_w = (jj == 0) ? (jj + nx - 1) : (jj - 1);
    /* propagate densities to neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    if(local_index == 0 && ii == 0 && jj == 0)printf("%f\n", cells[0].speeds[0]);
    float speeds[9];
    speeds[0] = cells[ii + jj*nx].speeds[0];
    speeds[1] = cells[x_w + jj*nx].speeds[1];
    speeds[3] = cells[x_e + jj*nx].speeds[3];
    speeds[5] = cells[x_w + y_s*nx].speeds[5];
    speeds[2] = cells[ii + y_s*nx].speeds[2];
    speeds[6] = cells[x_e + y_s*nx].speeds[6];
    speeds[8] = cells[x_w + y_n*nx].speeds[8];
    speeds[4] = cells[ii + y_n*nx].speeds[4];
    speeds[7] = cells[x_e + y_n*nx].speeds[7];
    int obstacle = !((int)obstacles[jj*nx + ii]);
    /* compute local density total */
    float local_density = 0.0;

#pragma unroll
    for (int i = 0; i < 9; i++) {
        local_density += speeds[i];
    }

    float a = speeds[8] - speeds[6];
    float b = speeds[5] - speeds[7];
    float c = speeds[1] - speeds[3];
    float d = speeds[2] - speeds[4];
    /* compute x velocity component */
    float u_x = b;
    float u_y = b;
    u_x = u_x + a + c;
    u_y = u_y + d - a;

    u_x = native_divide(u_x,local_density);
    u_y = native_divide(u_y, local_density);
    float w_local = W0 * local_density;
    /* velocity squared */

    /* directional velocity components */


    //Compiler can you please vectorize this ty xoxo

    //before: 9, now:
    float u_x_y = u_x * u_y * 2.0 * C_SQ_2;
    float u2[NSPEEDS];
    u2[1] = u_x * u_x;
    u2[2] = u_y * u_y;
    float u_sq = (u2[1] + u2[2]);
    float u_sq_recip = u_sq *C_SQ_R_2 - 1.0;

    u2[0] = 0;
    u2[1] = u2[1] * C_SQ_2 - u_sq_recip;
    u2[2] *= C_SQ_2;
    u2[3] = u2[1];
    u2[4] = u2[2] - u_sq_recip;
    u2[5] = u2[1] + u2[2] + u_x_y;
    u2[6] = u2[1] + u2[2] - u_x_y;
    u2[7] = u2[5];
    u2[8] = u2[6];
    u2[2] -= u_sq_recip;
    tot_u = native_sqrt(u_sq)*obstacle;
    u_x  = native_divide(u_x, (float)C_SQ);
    u_y = native_divide(u_y, (float)C_SQ);
    float u[NSPEEDS];
    u[0] = -u_sq_recip;
    u[1] = u_x + u2[1];
    u[2] = u_y + u2[2];
    u[3] = -u_x + u2[3];
    u[4] = -u_y + u2[4];
    u[5] = u_x + u_y + u2[5];
    u[6] = -u_x + u_y + u2[6];
    u[7] = -u_x - u_y + u2[7];
    u[8] = u_x - u_y + u2[8];

    u[0] = w_local * (u[0]) - speeds[0];
    u[0] = (speeds[0] + omega*u[0])*obstacle;
    float w1_local = W1 * local_density;
#pragma unroll
    for (int i = 1; i < 5; i++) {
        float a = w1_local * (u[i]) - speeds[i];
        u[i] = speeds[i] + omega*a;
    }
    float w2_local = W2 * local_density;
#pragma unroll
    for (int i = 5; i < 9; i++) {
        float a = w2_local * (u[i]) - speeds[i];
        u[i] = speeds[i] + omega*a;
    }
    if(!obstacle){
        u[1] = speeds[3];
        u[2] = speeds[4];
        u[3] = speeds[1];
        u[4] = speeds[2];
        u[5] = speeds[7];
        u[6] = speeds[8];
        u[7] = speeds[5];
        u[8] = speeds[6];
    }


    scratch[local_index] = tot_u;
    barrier(CLK_LOCAL_MEM_FENCE);
#pragma unroll
    for(int i=0; i < 9 ; i++){
        tmp_cells[ii + jj*nx].speeds[i] = u[i];
    }
    for(int offset = local_size/2; offset > 0; offset = offset / 2){
        if(local_index < offset){
            float other = scratch[local_index + offset];
            float mine = scratch[local_index];
            scratch[local_index] = mine + other;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_index == 0){
        //printf("%f\n", tot_u);
        tot_vel[(get_group_id(0) + get_group_id(1)*get_num_groups(0))] = scratch[0];
    }


}
