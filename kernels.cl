#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define NSPEEDS         9

#define C_SQ		(1.0 / 3.0) /* square of speed of sound */
#define W0			(4.0 / 9.0)  /* weighting factor */
#define W1			(W0 / 4)
#define W2			(W1 / 4)
#define IN_C_SQ	    1/C_SQ
#define A	        1.f/(2.f * C_SQ * C_SQ)
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

kernel void propagate(global t_speed* cells, global t_speed* tmp_cells, global int* obstacles, int nx, int ny, float omega, global float* tt_vels){
    /* get column and row indices */
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
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = (jj + 1) % ny;
    int x_e = (ii + 1) % nx;
    int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
    int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);

    float tot_u = 0.f;
    /* propagate densities from neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    const float tmpC0 = cells[idx].speeds[0]; /* central cell, no movement */
    const float tmpC1 = cells[x_w + jj * nx].speeds[1]; /* east */
    const float tmpC2 = cells[ii + y_s * nx].speeds[2]; /* north */
    const float tmpC3 = cells[x_e + jj * nx].speeds[3]; /* west */
    const float tmpC4 = cells[ii + y_n * nx].speeds[4]; /* south */
    const float tmpC5 = cells[x_w + y_s * nx].speeds[5]; /* north-east */
    const float tmpC6 = cells[x_e + y_s * nx].speeds[6]; /* north-west */
    const float tmpC7 = cells[x_e + y_n * nx].speeds[7]; /* south-west */
    const float tmpC8 = cells[x_w + y_n * nx].speeds[8]; /* south-east */

    /* don't consider occupied cells */
    if (obstacles[idx])
    {

        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_cells[idx].speeds[1] = tmpC3;
        tmp_cells[idx].speeds[2] = tmpC4;
        tmp_cells[idx].speeds[3] = tmpC1;
        tmp_cells[idx].speeds[4] = tmpC2;
        tmp_cells[idx].speeds[5] = tmpC7;
        tmp_cells[idx].speeds[6] = tmpC8;
        tmp_cells[idx].speeds[7] = tmpC5;
        tmp_cells[idx].speeds[8] = tmpC6;
    }
    if (!obstacles[idx]){

        const float local_density = tmpC0 + tmpC1 + tmpC2 + tmpC3 + tmpC4
                                    + tmpC5 + tmpC6 + tmpC7 + tmpC8;

        /* compute x velocity component */
        const float inv_localDensity = 1/local_density;
        const float u_x = (tmpC1
                           + tmpC5
                           + tmpC8
                           - (tmpC3
                              + tmpC6
                              + tmpC7))
                          * inv_localDensity;
        /* compute y velocity component */
        const float u_y = (tmpC2
                           + tmpC5
                           + tmpC6
                           - (tmpC4
                              + tmpC7
                              + tmpC8))
                          * inv_localDensity;

        /* velocity squared */
        const float u_sq = u_x * u_x + u_y * u_y;

        const float t = -u_sq/(2.f * C_SQ);
        //                const float a = 1.f/(2.f * c_sq * c_sq);
        //                const float in_c_sq = 1/c_sq;
        /* zero velocity density: weight w0 */
        const float d_equ0 = W0 * local_density
                             * (1.f +t);
        /* axis speeds: weight w1 */
        const float tempW = W1 * local_density;

        const float d_equ1 = tempW * (1.f + u_x * IN_C_SQ
                                      + (u_x * u_x) * A
                                      + t);

        const float d_equ2 = tempW * (1.f + u_y * IN_C_SQ
                                      + (u_y * u_y) * A
                                      + t);

        const float d_equ3 = tempW * (1.f - u_x *IN_C_SQ
                                      + (u_x * u_x) * A
                                      + (t));

        const float d_equ4 = tempW * (1.f - u_y *IN_C_SQ
                                      + (u_y * u_y) * A
                                      + (t));
        /* diagonal speeds: weight w2 */
        const float tempW2 = W2 * local_density;
        const float u_xy = u_x + u_y;
        const float d_equ5 = tempW2 * (1.f + u_xy *IN_C_SQ
                                       + (u_xy * u_xy) * A
                                       + (t));
        const float u__xy = -u_x + u_y;
        const float d_equ6 = tempW2 * (1.f + u__xy *IN_C_SQ
                                       + (u__xy * u__xy) * A
                                       + (t));
        const float u__x_y = -u_x - u_y;
        const float d_equ7 = tempW2 * (1.f + u__x_y *IN_C_SQ
                                       + (u__x_y * u__x_y) * A
                                       + (t));
        const float u_x_y = u_x - u_y;
        const float d_equ8 = tempW2 * (1.f + u_x_y *IN_C_SQ
                                       + (u_x_y * u_x_y) * A
                                       + (t));
        /* relaxation step */

        tmp_cells[idx].speeds[0] = tmpC0 + omega * (d_equ0 - tmpC0);
        tmp_cells[idx].speeds[1] = tmpC1 + omega * (d_equ1- tmpC1);
        tmp_cells[idx].speeds[2] = tmpC2 + omega * (d_equ2 - tmpC2);
        tmp_cells[idx].speeds[3] = tmpC3 + omega * (d_equ3 - tmpC3);
        tmp_cells[idx].speeds[4] = tmpC4 + omega * (d_equ4 - tmpC4);
        tmp_cells[idx].speeds[5] = tmpC5 + omega * (d_equ5 - tmpC5);
        tmp_cells[idx].speeds[6] = tmpC6 + omega * (d_equ6 - tmpC6);
        tmp_cells[idx].speeds[7] = tmpC7 + omega * (d_equ7 - tmpC7);
        tmp_cells[idx].speeds[8] = tmpC8 + omega * (d_equ8 - tmpC8);

        tot_u += native_sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        //++*tot_cells;
    }

    //__local float local_tot_u[LOCAL_SIZE];

    /*local_tot_u[local_index] = tot_u;
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
    }*/
    local_tot_u[local_index] = tot_u;
    barrier(CLK_LOCAL_MEM_FENCE);
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
        tot_vel[(get_num_groups(0)*get_num_groups(1)) + (get_group_id(0) + get_group_id(1)*get_num_groups(0))] = local_tot_u[0];
    }

}
