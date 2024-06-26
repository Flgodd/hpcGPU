
#define NSPEEDS         9

#define C_SQ		(1.0 / 3.0) /* square of speed of sound */
#define W0			(4.0 / 9.0)  /* weighting factor */
#define W1			(W0 / 4)
#define W2			(W1 / 4)
#define IN_C_SQ	    1/C_SQ
#define A	        1.f/(2.f * C_SQ * C_SQ)
#ifndef BLOCK_I
#define BLOCK_I 16
#endif
#ifndef BLOCK_J
#define BLOCK_J 16
#endif

typedef struct
{
    float *speeds[NSPEEDS];
} t_speed;
kernel void accelerate_flow(
        global float* speeds0,
        global float* speeds1,
        global float* speeds2,
        global float* speeds3,
        global float* speeds4,
        global float* speeds5,
        global float* speeds6,
        global float* speeds7,
        global float* speeds8,
        global int* obstacles,
        int nx, int ny,
        float density,
        float accel
) {
    /* compute weighting factors */
    float w1 = density * accel / 9.0;
    float w2 = density * accel / 36.0;

    /* modify the 2nd row of the grid */
    int jj = ny - 2;

    /* get column index */
    int ii = get_global_id(0);

    /* if the cell is not occupied and
    ** we don't send a negative density */
    if (!obstacles[ii + jj * nx] &&
        (speeds3[ii + jj * nx] - w1) > 0.f &&
        (speeds6[ii + jj * nx] - w2) > 0.f &&
        (speeds7[ii + jj * nx] - w2) > 0.f) {
        /* increase 'east-side' densities */
        speeds1[ii + jj * nx] += w1;
        speeds5[ii + jj * nx] += w2;
        speeds8[ii + jj * nx] += w2;

        /* decrease 'west-side' densities */
        speeds3[ii + jj * nx] -= w1;
        speeds6[ii + jj * nx] -= w2;
        speeds7[ii + jj * nx] -= w2;
    }
}

kernel void combineReCol(
        global float* speeds0,
        global float* speeds1,
        global float* speeds2,
        global float* speeds3,
        global float* speeds4,
        global float* speeds5,
        global float* speeds6,
        global float* speeds7,
        global float* speeds8,
        global float* tmp_speeds0,
        global float* tmp_speeds1,
        global float* tmp_speeds2,
        global float* tmp_speeds3,
        global float* tmp_speeds4,
        global float* tmp_speeds5,
        global float* tmp_speeds6,
        global float* tmp_speeds7,
        global float* tmp_speeds8,
        global int* obstacles,
        int nx, int ny,
        float omega,
        global float* tot_vel,
        int tt
)
{
    local float local_tot_u[16*16];

    int ii = get_global_id(0);
    int jj = get_global_id(1);
    int idx = ii + jj * nx;

    float tot_u = 0;
    int y_n = (jj + 1) % ny;
    int x_e = (ii + 1) % nx;
    int y_s = (jj == 0) ? (jj + ny - 1) : (jj - 1);
    int x_w = (ii == 0) ? (ii + nx - 1) : (ii - 1);
    /* propagate densities to neighbouring cells, following
    ** appropriate directions of travel and writing into
    ** scratch space grid */
    //if(local_index == 0 && ii == 0 && jj == 0)printf("%f\n", cells[0]);
    const float tmpC0 = speeds0[idx];
    const float tmpC1 = speeds1[x_w + jj * nx];
    const float tmpC2 = speeds2[ii + y_s * nx];
    const float tmpC3 = speeds3[x_e + jj * nx];
    const float tmpC4 = speeds4[ii + y_n * nx];
    const float tmpC5 = speeds5[x_w + y_s * nx];
    const float tmpC6 = speeds6[x_e + y_s * nx];
    const float tmpC7 = speeds7[x_e + y_n * nx];
    const float tmpC8 = speeds8[x_w + y_n * nx]; /* south-east */

    /* don't consider occupied cells */
    if (obstacles[idx])
    {
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        tmp_speeds1[idx] = tmpC3;
        tmp_speeds2[idx] = tmpC4;
        tmp_speeds3[idx] = tmpC1;
        tmp_speeds4[idx] = tmpC2;
        tmp_speeds5[idx] = tmpC7;
        tmp_speeds6[idx] = tmpC8;
        tmp_speeds7[idx] = tmpC5;
        tmp_speeds8[idx] = tmpC6;
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

        tmp_speeds0[idx] = tmpC0 + omega * (d_equ0 - tmpC0);
        tmp_speeds1[idx] = tmpC1 + omega * (d_equ1- tmpC1);
        tmp_speeds2[idx] = tmpC2 + omega * (d_equ2 - tmpC2);
        tmp_speeds3[idx] = tmpC3 + omega * (d_equ3 - tmpC3);
        tmp_speeds4[idx] = tmpC4 + omega * (d_equ4 - tmpC4);
        tmp_speeds5[idx] = tmpC5 + omega * (d_equ5 - tmpC5);
        tmp_speeds6[idx] = tmpC6 + omega * (d_equ6 - tmpC6);
        tmp_speeds7[idx] = tmpC7 + omega * (d_equ7 - tmpC7);
        tmp_speeds8[idx] = tmpC8 + omega * (d_equ8 - tmpC8);

        tot_u += native_sqrt((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        //++*tot_cells;
    }

    int ii_l = get_local_id(0);
    int jj_l = get_local_id(1);

    int nx_l = get_local_size(0);
    int ny_l = get_local_size(1);

    int local_index = jj_l*nx_l + ii_l;
    int local_size = nx_l * ny_l;

    local_tot_u[local_index] = tot_u;
    barrier(CLK_LOCAL_MEM_FENCE);


    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
        if (local_index < stride) {
            local_tot_u[local_index] += local_tot_u[local_index + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if(local_index == 0){

        tot_vel[tt*(get_num_groups(1)*get_num_groups(0)) + (get_group_id(0) + get_group_id(1)*get_num_groups(0))] = local_tot_u[0];
    }


}



