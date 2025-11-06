#include "atconstants.h"
// #include "atelem.c"
// #include "atlalib.c"
#include "atphyslib.c"
#include "driftkick.c"		/* fastdrift and bndthinkick */
// #include "quadfringe.c"		/* QuadFringePassP, QuadFringePassN */
#include <stdbool.h>

void BndMPoleSymplectic4Pass(double *r, double le, double irho, double *A, double *B,
        int max_order, int num_int_steps,
        double entrance_angle, double exit_angle,
        int FringeBendEntrance, int FringeBendExit,
        double fint1, double fint2, double gap)
        // int FringeQuadEntrance, int FringeQuadExit,
        // double *fringeIntM0,  /* I0m/K1, I1m/K1, I2m/K1, I3m/K1, Lambda2m/K1 */
        // double *fringeIntP0)  /* I0p/K1, I1p/K1, I2p/K1, I3p/K1, Lambda2p/K1 */
{
    double SL = le/num_int_steps;
    double L1 = SL*DRIFT1;
    double L2 = SL*DRIFT2;
    double K1 = SL*KICK1;
    double K2 = SL*KICK2;
    // bool useLinFrEleEntrance = (fringeIntM0 != NULL && fringeIntP0 != NULL  && FringeQuadEntrance==2);
    // bool useLinFrEleExit = (fringeIntM0 != NULL && fringeIntP0 != NULL  && FringeQuadExit==2);

    int m;
    double p_norm, NormL1, NormL2;

    p_norm = 1.0/(1.0+r[4]);
    NormL1 = L1*p_norm;
    NormL2 = L2*p_norm;

    /* edge focus */
    edge_fringe_entrance(r, irho, entrance_angle, fint1, gap, FringeBendEntrance);
    /* quadrupole gradient fringe entrance*/
    // if (FringeQuadEntrance && B[1]!=0) {
    //     if (useLinFrEleEntrance) /*Linear fringe fields from elegant*/
    //         linearQuadFringeElegantEntrance(r, B[1], fringeIntM0, fringeIntP0);
    //     else
    //         QuadFringePassP(r, B[1]);
    // }
    /* integrator */
    for (m=0; m < num_int_steps; m++) { /* Loop over slices */
        fastdrift(r, NormL1);
        bndthinkick(r, A, B, K1, irho, max_order);
        fastdrift(r, NormL2);
        bndthinkick(r, A, B, K2, irho, max_order);
        fastdrift(r, NormL2);
        bndthinkick(r, A, B, K1, irho, max_order);
        fastdrift(r, NormL1);
    }
    /* quadrupole gradient fringe */
    // if (FringeQuadExit && B[1]!=0) {
    //     if (useLinFrEleExit) /*Linear fringe fields from elegant*/
    //         linearQuadFringeElegantExit(r, B[1], fringeIntM0, fringeIntP0);
    //     else
    //         QuadFringePassN(r, B[1]);
    // }
    /* edge focus */
    edge_fringe_exit(r, irho, exit_angle, fint2, gap, FringeBendExit);
}
