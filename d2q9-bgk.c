/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<string.h>
#include <mm_malloc.h>
#ifdef __unix__
#include<sys/time.h>
#include<sys/resource.h>
#else
#include <time.h>
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define OCLFILE         "kernels.cl"

/* struct to hold the parameter values */
typedef struct
{
    int    nx;            /* no. of cells in x-direction */
    int    ny;            /* no. of cells in y-direction */
    int    maxIters;      /* no. of iterations */
    int    reynolds_dim;  /* dimension for Reynolds number */
    float density;       /* density per link */
    float accel;         /* density redistribution */
    float omega;         /* relaxation parameter */
} t_param;

/* struct to hold OpenCL objects */
typedef struct
{
    cl_device_id      device;
    cl_context        context;
    cl_command_queue  queue;

    cl_program program;
    cl_kernel  accelerate_flow;
    cl_kernel  propagate;
    cl_kernel  combineReCol;
    cl_kernel  rebound;
    cl_kernel  av_velocity;


    cl_mem cells;
    cl_mem tmp_cells;
    cl_mem obstacles;
    cl_mem total_vel;
    cl_mem cspeeds0;
    cl_mem cspeeds1;
    cl_mem cspeeds2;
    cl_mem cspeeds3;
    cl_mem cspeeds4;
    cl_mem cspeeds5;
    cl_mem cspeeds6;
    cl_mem cspeeds7;
    cl_mem cspeeds8;
    cl_mem tspeeds0;
    cl_mem tspeeds1;
    cl_mem tspeeds2;
    cl_mem tspeeds3;
    cl_mem tspeeds4;
    cl_mem tspeeds5;
    cl_mem tspeeds6;
    cl_mem tspeeds7;
    cl_mem tspeeds8;
    int workGroups;
    size_t workGroupSize;

} t_ocl;

typedef struct
{
    float *speeds[NSPEEDS];
} t_speed;
/*
** function prototypes
*/
/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl* ocl,  float* gl_obs_u, float** tt_vels);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_ocl ocl, int tt);
int accelerate_flow(const t_param params, t_ocl ocl);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, t_ocl ocl);
int combineReCol(const t_param params, t_ocl ocl, int tt);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl);

/* utility functions */
void checkError(cl_int err, const char *op, const int line);
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

cl_device_id selectOpenCLDevice();

char* options = " -cl-mad-enable -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math";
/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
    char*    paramfile = NULL;    /* name of the input parameter file */
    char*    obstaclefile = NULL; /* name of a the input obstacle file */
    t_param  params;              /* struct to hold parameter values */
    t_ocl    ocl;                 /* struct to hold OpenCL objects */
    t_speed* cells = NULL;    /* grid containing fluid densities */
    t_speed* tmp_cells = NULL;    /* scratch space */
    int*     obstacles = NULL;    /* grid indicating which cells are blocked */
    float* av_vels = NULL;     /* a record of the av. velocity computed for each timestep */
    float* tt_vels = NULL;
    float gl_obs_u;
    cl_int err;
#ifdef __unix__
    struct timeval timstr;        /* structure to hold elapsed time */
	struct rusage ru;             /* structure to hold CPU time--system and user */

	double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
	double usrtim;                /* floating point number to record elapsed user CPU time */
	double systim;                /* floating point number to record elapsed system CPU time */
#endif
    /* parse the command line */
    if (argc != 3)
    {
        usage(argv[0]);
    }
    else
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    /* initialise our data structures and load values from file */
    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &ocl, &gl_obs_u, &tt_vels);

    // Write cells to OpenCL buffer

        err = clEnqueueWriteBuffer(
                ocl.queue,
                ocl.cspeeds0,
                CL_FALSE,
                0,
                sizeof(float) * params.nx * params.ny,
                cells->speeds[0],
                0,
                NULL,
                NULL
        );
        checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(
            ocl.queue,
            ocl.cspeeds1,
            CL_FALSE,
            0,
            sizeof(float) * params.nx * params.ny,
            cells->speeds[1],
            0,
            NULL,
            NULL
    );
    checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(
            ocl.queue,
            ocl.cspeeds2,
            CL_FALSE,
            0,
            sizeof(float) * params.nx * params.ny,
            cells->speeds[2],
            0,
            NULL,
            NULL
    );
    checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(
            ocl.queue,
            ocl.cspeeds3,
            CL_FALSE,
            0,
            sizeof(float) * params.nx * params.ny,
            cells->speeds[3],
            0,
            NULL,
            NULL
    );
    checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(
            ocl.queue,
            ocl.cspeeds4,
            CL_FALSE,
            0,
            sizeof(float) * params.nx * params.ny,
            cells->speeds[4],
            0,
            NULL,
            NULL
    );
    checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(
            ocl.queue,
            ocl.cspeeds5,
            CL_FALSE,
            0,
            sizeof(float) * params.nx * params.ny,
            cells->speeds[5],
            0,
            NULL,
            NULL
    );
    checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(
            ocl.queue,
            ocl.cspeeds6,
            CL_FALSE,
            0,
            sizeof(float) * params.nx * params.ny,
            cells->speeds[6],
            0,
            NULL,
            NULL
    );
    checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(
            ocl.queue,
            ocl.cspeeds7,
            CL_FALSE,
            0,
            sizeof(float) * params.nx * params.ny,
            cells->speeds[7],
            0,
            NULL,
            NULL
    );
    checkError(err, "writing cells data", __LINE__);
    err = clEnqueueWriteBuffer(
            ocl.queue,
            ocl.cspeeds8,
            CL_FALSE,
            0,
            sizeof(float) * params.nx * params.ny,
            cells->speeds[8],
            0,
            NULL,
            NULL
    );
    checkError(err, "writing cells data", __LINE__);


    // Write obstacles to OpenCL buffer00
    err = clEnqueueWriteBuffer(
            ocl.queue, ocl.obstacles, CL_TRUE, 0,
            sizeof(cl_int) * params.nx * params.ny, obstacles, 0, NULL, NULL);
    checkError(err, "writing obstacles data", __LINE__);


    err = clSetKernelArg(ocl.combineReCol, 18, sizeof(cl_mem), &ocl.obstacles);
    checkError(err, "setting collision arg 2", __LINE__);
    err = clSetKernelArg(ocl.combineReCol, 19, sizeof(cl_int), &params.nx);
    checkError(err, "setting collision arg 3", __LINE__);
    err = clSetKernelArg(ocl.combineReCol, 20, sizeof(cl_int), &params.ny);
    checkError(err, "setting collision arg 4", __LINE__);
    err = clSetKernelArg(ocl.combineReCol, 21, sizeof(cl_float), &params.omega);
    checkError(err, "setting collision arg 5", __LINE__);
    err = clSetKernelArg(ocl.combineReCol, 22, sizeof(cl_mem), &ocl.total_vel);
    checkError(err, "setting collision arg 6", __LINE__);

    err = clSetKernelArg(ocl.accelerate_flow, 9, sizeof(cl_mem), &ocl.obstacles);
    checkError(err, "setting accelerate_flow arg 1", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 10, sizeof(cl_int), &params.nx);
    checkError(err, "setting accelerate_flow arg 2", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 11, sizeof(cl_int), &params.ny);
    checkError(err, "setting accelerate_flow arg 3", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 12, sizeof(cl_float), &params.density);
    checkError(err, "setting accelerate_flow arg 4", __LINE__);
    err = clSetKernelArg(ocl.accelerate_flow, 13, sizeof(cl_float), &params.accel);
    checkError(err, "setting accelerate_flow arg 5", __LINE__);

    /* iterate for maxIters timesteps */
#ifdef __unix__
    gettimeofday(&timstr, NULL);
	tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
#else
    clock_t start = clock();
#endif
    clock_t b = clock();
    for (int tt = 0; tt < params.maxIters; tt++)
    {
        timestep(params, ocl, tt);

        cl_mem temp0 = ocl.cspeeds0;
        cl_mem temp1 = ocl.cspeeds1;
        cl_mem temp2 = ocl.cspeeds2;
        cl_mem temp3 = ocl.cspeeds3;
        cl_mem temp4 = ocl.cspeeds4;
        cl_mem temp5 = ocl.cspeeds5;
        cl_mem temp6 = ocl.cspeeds6;
        cl_mem temp7 = ocl.cspeeds7;
        cl_mem temp8 = ocl.cspeeds8;

        ocl.cspeeds0 = ocl.tspeeds0;
        ocl.cspeeds1 = ocl.tspeeds1;
        ocl.cspeeds2 = ocl.tspeeds2;
        ocl.cspeeds3 = ocl.tspeeds3;
        ocl.cspeeds4 = ocl.tspeeds4;
        ocl.cspeeds5 = ocl.tspeeds5;
        ocl.cspeeds6 = ocl.tspeeds6;
        ocl.cspeeds7 = ocl.tspeeds7;
        ocl.cspeeds8 = ocl.tspeeds8;

        ocl.tspeeds0 = temp0;
        ocl.tspeeds1 = temp1;
        ocl.tspeeds2 = temp2;
        ocl.tspeeds3 = temp3;
        ocl.tspeeds4 = temp4;
        ocl.tspeeds5 = temp5;
        ocl.tspeeds6 = temp6;
        ocl.tspeeds7 = temp7;
        ocl.tspeeds8 = temp8;



        //err = clFinish(ocl.queue);
#ifdef DEBUG
        printf("==timestep: %d==\n", tt);
		printf("av velocity: %.12E\n", av_vels[tt]);
		printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }
    clock_t e = clock();
    float t = (float)(e - b) / CLOCKS_PER_SEC;
    printf("time: %f\n", t);
#ifdef  __unix__
    gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  double time = toc - tic;
#else
    clock_t end = clock();
    float time = (float)(end - start) / CLOCKS_PER_SEC;
#endif
    /* write final values and free memory */
    err = clEnqueueReadBuffer(
            ocl.queue, ocl.cspeeds0, CL_TRUE, 0,
            sizeof(float) * params.nx * params.ny, cells->speeds[0], 0, NULL, NULL);
    checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(
            ocl.queue, ocl.cspeeds1, CL_TRUE, 0,
            sizeof(float) * params.nx * params.ny, cells->speeds[1], 0, NULL, NULL);
    checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(
            ocl.queue, ocl.cspeeds2, CL_TRUE, 0,
            sizeof(float) * params.nx * params.ny, cells->speeds[2], 0, NULL, NULL);
    checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(
            ocl.queue, ocl.cspeeds3, CL_TRUE, 0,
            sizeof(float) * params.nx * params.ny, cells->speeds[3], 0, NULL, NULL);
    checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(
            ocl.queue, ocl.cspeeds4, CL_TRUE, 0,
            sizeof(float) * params.nx * params.ny, cells->speeds[4], 0, NULL, NULL);
    checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(
            ocl.queue, ocl.cspeeds5, CL_TRUE, 0,
            sizeof(float) * params.nx * params.ny, cells->speeds[5], 0, NULL, NULL);
    checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(
            ocl.queue, ocl.cspeeds6, CL_TRUE, 0,
            sizeof(float) * params.nx * params.ny, cells->speeds[6], 0, NULL, NULL);
    checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(
            ocl.queue, ocl.cspeeds7, CL_TRUE, 0,
            sizeof(float) * params.nx * params.ny, cells->speeds[7], 0, NULL, NULL);
    checkError(err, "reading cells data", __LINE__);
    err = clEnqueueReadBuffer(
            ocl.queue, ocl.cspeeds8, CL_TRUE, 0,
            sizeof(float) * params.nx * params.ny, cells->speeds[8], 0, NULL, NULL);
    checkError(err, "reading cells data", __LINE__);

    err = clEnqueueReadBuffer(
            ocl.queue, ocl.total_vel, CL_TRUE, 0,
            sizeof(cl_float)*params.maxIters*(ocl.workGroups), tt_vels, 0, NULL, NULL);
    checkError(err, "reading total_vel data", __LINE__);

    for (int i = 0; i < params.maxIters; i++) {
        float tv = 0;
        for(int j = 0; j<ocl.workGroups; j++){
            tv += tt_vels[i*ocl.workGroups + j];
        }
        av_vels[i] = tv/gl_obs_u;
    }

    free(tt_vels);
    //av_velocity(params, cells, obstacles);

    checkError(err, "reading tmp_cells data", __LINE__);
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, ocl));
#ifdef __unix__
    printf("Elapsed time:\t\t\t%.6f (s)\n", time);
  printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
  printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
#else
    printf("Elapsed time:\t\t\t%.6f (s)\n", time);

#endif
    write_values(params, cells, obstacles, av_vels);
    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels, ocl);

    return EXIT_SUCCESS;
}

int timestep(const t_param params, t_ocl ocl, int tt)
{
    cl_int err;

    accelerate_flow(params,  ocl);
    combineReCol(params,  ocl, tt);


    return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_ocl ocl)
{
    cl_int err;

//    err = clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cells);
//    checkError(err, "setting accelerate_flow arg 0", __LINE__);
    clSetKernelArg(ocl.accelerate_flow, 0, sizeof(cl_mem), &ocl.cspeeds0);
    clSetKernelArg(ocl.accelerate_flow, 1, sizeof(cl_mem), &ocl.cspeeds1);
    clSetKernelArg(ocl.accelerate_flow, 2, sizeof(cl_mem), &ocl.cspeeds2);
    clSetKernelArg(ocl.accelerate_flow, 3, sizeof(cl_mem), &ocl.cspeeds3);
    clSetKernelArg(ocl.accelerate_flow, 4, sizeof(cl_mem), &ocl.cspeeds4);
    clSetKernelArg(ocl.accelerate_flow, 5, sizeof(cl_mem), &ocl.cspeeds5);
    clSetKernelArg(ocl.accelerate_flow, 6, sizeof(cl_mem), &ocl.cspeeds6);
    clSetKernelArg(ocl.accelerate_flow, 7, sizeof(cl_mem), &ocl.cspeeds7);
    clSetKernelArg(ocl.accelerate_flow, 8, sizeof(cl_mem), &ocl.cspeeds8);


    // Enqueue kernel
    size_t global[1] = {params.nx};
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.accelerate_flow,
                                 1, NULL, global, NULL, 0, NULL, NULL);
    checkError(err, "enqueueing accelerate_flow kernel", __LINE__);


    return EXIT_SUCCESS;
}



int combineReCol(const t_param params, t_ocl ocl , int tt)
{
    cl_int err;

    // Set kernel arguments
//    err = clSetKernelArg(ocl.combineReCol, 0, sizeof(cl_mem), &ocl.cells);
//    checkError(err, "setting collision arg 0", __LINE__);
//    err = clSetKernelArg(ocl.combineReCol, 1, sizeof(cl_mem), &ocl.tmp_cells);
//    checkError(err, "setting collision arg 1", __LINE__);
    clSetKernelArg(ocl.combineReCol, 0, sizeof(cl_mem), &ocl.cspeeds0);
    clSetKernelArg(ocl.combineReCol, 1, sizeof(cl_mem), &ocl.cspeeds1);
    clSetKernelArg(ocl.combineReCol, 2, sizeof(cl_mem), &ocl.cspeeds2);
    clSetKernelArg(ocl.combineReCol, 3, sizeof(cl_mem), &ocl.cspeeds3);
    clSetKernelArg(ocl.combineReCol, 4, sizeof(cl_mem), &ocl.cspeeds4);
    clSetKernelArg(ocl.combineReCol, 5, sizeof(cl_mem), &ocl.cspeeds5);
    clSetKernelArg(ocl.combineReCol, 6, sizeof(cl_mem), &ocl.cspeeds6);
    clSetKernelArg(ocl.combineReCol, 7, sizeof(cl_mem), &ocl.cspeeds7);
    clSetKernelArg(ocl.combineReCol, 8, sizeof(cl_mem), &ocl.cspeeds8);

    clSetKernelArg(ocl.combineReCol, 9, sizeof(cl_mem), &ocl.tspeeds0);
    clSetKernelArg(ocl.combineReCol, 10, sizeof(cl_mem), &ocl.tspeeds1);
    clSetKernelArg(ocl.combineReCol, 11, sizeof(cl_mem), &ocl.tspeeds2);
    clSetKernelArg(ocl.combineReCol, 12, sizeof(cl_mem), &ocl.tspeeds3);
    clSetKernelArg(ocl.combineReCol, 13, sizeof(cl_mem), &ocl.tspeeds4);
    clSetKernelArg(ocl.combineReCol, 14, sizeof(cl_mem), &ocl.tspeeds5);
    clSetKernelArg(ocl.combineReCol, 15, sizeof(cl_mem), &ocl.tspeeds6);
    clSetKernelArg(ocl.combineReCol, 16, sizeof(cl_mem), &ocl.tspeeds7);
    clSetKernelArg(ocl.combineReCol, 17, sizeof(cl_mem), &ocl.tspeeds8);
    err = clSetKernelArg(ocl.combineReCol, 23, sizeof(cl_int), &tt);
    checkError(err, "setting collision arg 7", __LINE__);
    // Enqueue kernel
    size_t global[2] = { params.nx, params.ny };
    size_t local[2] = { 16, 16};
    err = clEnqueueNDRangeKernel(ocl.queue, ocl.combineReCol,
                                 2, NULL, global, local, 0, NULL, NULL);

    checkError(err, "enqueueing collision kernel", __LINE__);

    err = clFinish(ocl.queue);
    checkError(err, "waiting for collision kernel", __LINE__);

    return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
    int    tot_cells = 0;  /* no. of cells used in calculation */
    float tot_u;          /* accumulated magnitudes of velocity for each cell */

    /* initialise */
    tot_u = 0.f;

    /* loop over all non-blocked cells */
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* ignore occupied cells */
            if (!obstacles[ii + jj*params.nx])
            {
                /* local density total */
                float local_density = 0.f;

                const int idx = ii + jj*params.nx;

                local_density = cells->speeds[0][idx] + cells->speeds[1][idx] + cells->speeds[2][idx] + cells->speeds[3][idx]
                                + cells->speeds[4][idx] + cells->speeds[5][idx] + cells->speeds[6][idx] + cells->speeds[7][idx] + cells->speeds[8][idx];

                /* x-component of velocity */
                const float u_x = (cells->speeds[1][idx]
                                   + cells->speeds[5][idx]
                                   + cells->speeds[8][idx]
                                   - (cells->speeds[3][idx]
                                      + cells->speeds[6][idx]
                                      + cells->speeds[7][idx]))
                                  / local_density;
                /* compute y velocity component */
                const float u_y = (cells->speeds[2][idx]
                                   + cells->speeds[5][idx]
                                   + cells->speeds[6][idx]
                                   - (cells->speeds[4][idx]
                                      + cells->speeds[7][idx]
                                      + cells->speeds[8][idx]))
                                  / local_density;
                /* accumulate the norm of x- and y- velocity components */
                tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                /* increase counter of inspected cells */
                ++tot_cells;
            }
        }
    }

    return tot_u / (float)tot_cells;
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, t_ocl *ocl, float *gl_obs_u, float** tt_vels)
{
    char   message[1024];  /* message buffer */
    FILE*   fp;            /* file pointer */
    int    xx, yy;         /* generic array indices */
    int    blocked;        /* indicates whether a cell is blocked by an obstacle */
    int    retval;         /* to hold return value for checking */
    char*  ocl_src;        /* OpenCL kernel source */
    long   ocl_size;       /* size of OpenCL kernel source */

    /* open the parameter file */
    fp = fopen(paramfile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input parameter file: %s", paramfile);
        die(message, __LINE__, __FILE__);
    }

    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));

    if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->ny));

    if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->maxIters));

    if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

    if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

    double a;

    retval = fscanf(fp, "%lf\n", &(a));
    params->density = (float) a;

    if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

    retval = fscanf(fp, "%lf\n", &(a));
    params->accel = (float) a;

    if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

    retval = fscanf(fp, "%lf\n", &(a));
    params->omega = (float) a;

    if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

    /* and close up the file */
    fclose(fp);

    /*
    ** Allocate memory.
    **
    ** Remember C is pass-by-value, so we need to
    ** pass pointers into the initialise function.
    **
    ** NB we are allocating a 1D array, so that the
    ** memory will be contiguous.  We still want to
    ** index this memory as if it were a (row major
    ** ordered) 2D array, however.  We will perform
    ** some arithmetic using the row and column
    ** coordinates, inside the square brackets, when
    ** we want to access elements of this array.
    **
    ** Note also that we are using a structure to
    ** hold an array of 'speeds'.  We will allocate
    ** a 1D array of these structs.
    */

    /* main grid */
    *cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed)*((params->ny) * params->nx), 64);
    (*cells_ptr)->speeds[0] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*cells_ptr)->speeds[1] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*cells_ptr)->speeds[2] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*cells_ptr)->speeds[3] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*cells_ptr)->speeds[4] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*cells_ptr)->speeds[5] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*cells_ptr)->speeds[6] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*cells_ptr)->speeds[7] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*cells_ptr)->speeds[8] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);

    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* 'helper' grid, used as scratch space */
    //*tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx)*NSPEEDS);
    *tmp_cells_ptr = (t_speed*)_mm_malloc(sizeof(t_speed) * ((params->ny) * params->nx), 64);
    (*tmp_cells_ptr)->speeds[0] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*tmp_cells_ptr)->speeds[1] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*tmp_cells_ptr)->speeds[2] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*tmp_cells_ptr)->speeds[3] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*tmp_cells_ptr)->speeds[4] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*tmp_cells_ptr)->speeds[5] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*tmp_cells_ptr)->speeds[6] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*tmp_cells_ptr)->speeds[7] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);
    (*tmp_cells_ptr)->speeds[8] = _mm_malloc(sizeof(float)*((params->ny) * params->nx),64);

    if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    /* initialise densities */
    float w0 = params->density * 4.0 / 9.0;
    float w1 = params->density      / 9.0;
    float w2 = params->density      / 36.0;

    //total_cells = params->ny * params->nx;
    for (int jj = 0; jj < params->ny; jj++)
    {
        for (int ii = 0; ii < params->nx; ii++)
        {
            /* centre */
            int idx = ii + jj*params->nx;
            // centre
            (*cells_ptr)->speeds[0][idx] = w0;
            // axis directions //
            (*cells_ptr)->speeds[1][idx] = w1;
            (*cells_ptr)->speeds[2][idx] = w1;
            (*cells_ptr)->speeds[3][idx] = w1;
            (*cells_ptr)->speeds[4][idx] = w1;
            // diagonals
            (*cells_ptr)->speeds[5][idx] = w2;
            (*cells_ptr)->speeds[6][idx] = w2;
            (*cells_ptr)->speeds[7][idx] = w2;
            (*cells_ptr)->speeds[8][idx] = w2;
        }
    }

    /* first set all cells in obstacle array to zero */
    for (int jj = 0; jj < params->ny; jj++)
    {
        for (int ii = 0; ii < params->nx; ii++)
        {
            (*obstacles_ptr)[ii + jj*params->nx] = 0;
        }
    }

    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
        /* some checks */
        if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

        if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

        if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

        if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

        /* assign to array */
        (*obstacles_ptr)[yy * params->nx + xx] = blocked;

    }
    float temp = 0;
    for (int jj = 0; jj < params->ny; jj++)
    {
        for (int ii = 0; ii < params->nx; ii++)
        {
            temp += (!(*obstacles_ptr)[ii + jj*params->nx] ? 1 : 0);
        }
    }
    *gl_obs_u = temp;
    /* and close the file */
    fclose(fp);

    /*
    ** allocate space to hold a record of the avarage velocities computed
    ** at each timestep
    */
    *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);


    cl_int err;

    ocl->device = selectOpenCLDevice();

    // Create OpenCL context
    ocl->context = clCreateContext(NULL, 1, &ocl->device, NULL, NULL, &err);
    checkError(err, "creating context", __LINE__);

    fp = fopen(OCLFILE, "r");
    if (fp == NULL)
    {
        sprintf(message, "could not open OpenCL kernel file: %s", OCLFILE);
        die(message, __LINE__, __FILE__);
    }

    // Create OpenCL command queue
    ocl->queue = clCreateCommandQueue(ocl->context, ocl->device, 0, &err);
    checkError(err, "creating command queue", __LINE__);

    // Load OpenCL kernel source
    fseek(fp, 0, SEEK_END);
    ocl_size = ftell(fp) + 1;
    ocl_src = (char*)malloc(ocl_size);
    memset(ocl_src, 0, ocl_size);
    fseek(fp, 0, SEEK_SET);
    fread(ocl_src, 1, ocl_size, fp);
    fclose(fp);

    // Create OpenCL program
    ocl->program = clCreateProgramWithSource(
            ocl->context, 1, (const char**)&ocl_src, NULL, &err);
    free(ocl_src);
    checkError(err, "creating program", __LINE__);

    // Build OpenCL program
    err = clBuildProgram(ocl->program, 1, &ocl->device, options, NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE)
    {
        size_t sz;
        clGetProgramBuildInfo(
                ocl->program, ocl->device,
                CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);
        char *buildlog = malloc(sz);
        clGetProgramBuildInfo(
                ocl->program, ocl->device,
                CL_PROGRAM_BUILD_LOG, sz, buildlog, NULL);
        fprintf(stderr, "\nOpenCL build log:\n\n%s\n", buildlog);
        free(buildlog);
    }
    checkError(err, "building program", __LINE__);

    // Create OpenCL kernels
    ocl->accelerate_flow = clCreateKernel(ocl->program, "accelerate_flow", &err);
    checkError(err, "creating accelerate_flow kernel", __LINE__);
    ocl->combineReCol = clCreateKernel(ocl->program, "combineReCol", &err);
    checkError(err, "creating collision kernel", __LINE__);

    // Allocate OpenCL buffers
    ocl->cspeeds0 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(float) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
    ocl->cspeeds1 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(float) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
    ocl->cspeeds2 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(float) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
    ocl->cspeeds3 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(float) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
    ocl->cspeeds4 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(float) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
    ocl->cspeeds5 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(float) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
    ocl->cspeeds6 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(float) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
    ocl->cspeeds7 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(float) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
    ocl->cspeeds8 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(float) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating cells buffer", __LINE__);
    ocl->tspeeds0 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(t_speed) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);
    ocl->tspeeds1 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(t_speed) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);
    ocl->tspeeds2 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(t_speed) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);
    ocl->tspeeds3 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(t_speed) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);
    ocl->tspeeds4 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(t_speed) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);
    ocl->tspeeds5 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(t_speed) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);
    ocl->tspeeds6 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(t_speed) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);
    ocl->tspeeds7 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(t_speed) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);
    ocl->tspeeds8 = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(t_speed) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);

    ocl->obstacles = clCreateBuffer(
            ocl->context, CL_MEM_READ_ONLY,
            sizeof(int) * params->nx * params->ny, NULL, &err);
    checkError(err, "creating tmp_cells buffer", __LINE__);

    ocl->workGroupSize = 16*16;
    ocl->workGroups = (params->nx*params->ny) / ocl->workGroupSize;

    printf("workgroup size: %d \n", (int)ocl->workGroupSize);
    printf("workgroup count: %d \n", (int)(ocl->workGroups));

    ocl->total_vel = clCreateBuffer(
            ocl->context, CL_MEM_READ_WRITE,
            sizeof(cl_float)*params->maxIters*(ocl->workGroups), NULL, &err);
    checkError(err, "creating vel buffer", __LINE__);

    *tt_vels = (float*)malloc(sizeof(float)*params->maxIters*(ocl->workGroups));

    return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr, t_ocl ocl)
{
    /*
    ** free up allocated memory
    */
    free(*cells_ptr);
    *cells_ptr = NULL;

    free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;

    free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    free(*av_vels_ptr);
    *av_vels_ptr = NULL;

    clReleaseMemObject(ocl.cspeeds0);
    clReleaseMemObject(ocl.cspeeds1);
    clReleaseMemObject(ocl.cspeeds2);
    clReleaseMemObject(ocl.cspeeds3);
    clReleaseMemObject(ocl.cspeeds4);
    clReleaseMemObject(ocl.cspeeds5);
    clReleaseMemObject(ocl.cspeeds6);
    clReleaseMemObject(ocl.cspeeds7);
    clReleaseMemObject(ocl.cspeeds8);

    clReleaseMemObject(ocl.tspeeds0);
    clReleaseMemObject(ocl.tspeeds1);
    clReleaseMemObject(ocl.tspeeds2);
    clReleaseMemObject(ocl.tspeeds3);
    clReleaseMemObject(ocl.tspeeds4);
    clReleaseMemObject(ocl.tspeeds5);
    clReleaseMemObject(ocl.tspeeds6);
    clReleaseMemObject(ocl.tspeeds7);
    clReleaseMemObject(ocl.tspeeds8);


    clReleaseMemObject(ocl.obstacles);
    clReleaseKernel(ocl.combineReCol);

    clReleaseProgram(ocl.program);
    clReleaseCommandQueue(ocl.queue);
    clReleaseContext(ocl.context);

    return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, t_ocl ocl)
{
    const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);
    return  av_velocity(params, cells, obstacles)* params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
    float total = 0.f;  /* accumulator */

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            const int idx = ii + jj*params.nx;
            total += cells->speeds[0][idx] + cells->speeds[1][idx] + cells->speeds[2][idx]+ cells->speeds[3][idx]
                     + cells->speeds[4][idx] + cells->speeds[5][idx] + cells->speeds[6][idx] + cells->speeds[7][idx] +cells->speeds[8][idx];
        }
    }

    return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
    FILE* fp;                     /* file pointer */
    const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
    float local_density;         /* per grid cell sum of densities */
    float pressure;              /* fluid pressure in grid cell */
    float u_x;                   /* x-component of velocity in grid cell */
    float u_y;                   /* y-component of velocity in grid cell */
    float u;                     /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(FINALSTATEFILE, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* an occupied cell */
            if (obstacles[ii + jj*params.nx])
            {
                u_x = u_y = u = 0.f;
                pressure = params.density * c_sq;
            }
                /* no obstacle */
            else
            {
                const int idx = ii + jj*params.nx;
                local_density = 0.f;

                const float local_density = cells->speeds[0][idx] + cells->speeds[1][idx]+ cells->speeds[2][idx] + cells->speeds[3][idx]
                                            + cells->speeds[4][idx] + cells->speeds[5][idx]+ cells->speeds[6][idx] +cells->speeds[7][idx] + cells->speeds[8][idx];

                /* compute x velocity component */
                u_x = (cells->speeds[1][idx]
                       + cells->speeds[5][idx]
                       + cells->speeds[8][idx]
                       - (cells->speeds[3][idx]
                          + cells->speeds[6][idx]
                          + cells->speeds[7][idx]))
                      / local_density;
                /* compute y velocity component */
                u_y = (cells->speeds[2][idx]
                       + cells->speeds[5][idx]
                       + cells->speeds[6][idx]
                       - (cells->speeds[4][idx]
                          + cells->speeds[7][idx]
                          + cells->speeds[8][idx]))
                      / local_density;
                /* compute norm of velocity */
                u = sqrtf((u_x * u_x) + (u_y * u_y));
                /* compute pressure */
                pressure = local_density * c_sq;
            }

            /* write to file */
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
        }
    }

    fclose(fp);

    fp = fopen(AVVELSFILE, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int ii = 0; ii < params.maxIters; ii++)
    {
        fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);

    return EXIT_SUCCESS;
}

void checkError(cl_int err, const char *op, const int line)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "OpenCL error during '%s' on line %d: %d\n", op, line, err);
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
}

void die(const char* message, const int line, const char* file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    getchar();
    exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}

#define MAX_DEVICES 32
#define MAX_DEVICE_NAME 1024

cl_device_id selectOpenCLDevice()
{
    cl_int err;
    cl_uint num_platforms = 0;
    cl_uint total_devices = 0;
    cl_platform_id platforms[8];
    cl_device_id devices[MAX_DEVICES];
    char name[MAX_DEVICE_NAME];

    // Get list of platforms
    err = clGetPlatformIDs(8, platforms, &num_platforms);
    checkError(err, "getting platforms", __LINE__);

    // Get list of devices
    for (cl_uint p = 0; p < num_platforms; p++)
    {
        cl_uint num_devices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL,
                             MAX_DEVICES-total_devices, devices+total_devices,
                             &num_devices);
        checkError(err, "getting device name", __LINE__);
        total_devices += num_devices;
    }

    // Print list of devices
    printf("\nAvailable OpenCL devices:\n");
    for (cl_uint d = 0; d < total_devices; d++)
    {
        clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_DEVICE_NAME, name, NULL);
        printf("%2d: %s\n", d, name);
    }
    printf("\n");

    // Use first device unless OCL_DEVICE environment variable used
    cl_uint device_index = 0;
    char *dev_env = getenv("OCL_DEVICE");
    if (dev_env)
    {
        char *end;
        device_index = strtol(dev_env, &end, 10);
        if (strlen(end))
            die("invalid OCL_DEVICE variable", __LINE__, __FILE__);
    }

    if (device_index >= total_devices)
    {
        fprintf(stderr, "device index set to %d but only %d devices available\n",
                device_index, total_devices);
        exit(1);
    }

    // Print OpenCL device name
    clGetDeviceInfo(devices[device_index], CL_DEVICE_NAME,
                    MAX_DEVICE_NAME, name, NULL);
    printf("Selected OpenCL device:\n-> %s (index=%d)\n\n", name, device_index);

    return devices[device_index];
}
