// #include "atlalib.c"
#include "diff_bend_fringe.c"
#include "diff_bnd_kick.c"
#include "diff_drift.c"
// #include "quadfringe.c"		/* QuadFringePassP, QuadFringePassN */


void BndMPoleSymplectic4RadPass(double *r, double le, double irho, double *A, double *B,
        int max_order, int num_int_steps,
        double entrance_angle, double exit_angle,
        int FringeBendEntrance, int FringeBendExit,
        double fint1, double fint2, double gap,
        double gamma, 
        double *bdiff)
{
    double SL = le/num_int_steps;
    double L1 = SL*DRIFT1;
    double L2 = SL*DRIFT2;
    double K1 = SL*KICK1;
    double K2 = SL*KICK2;

    double rad_const = RAD_CONST*pow(gamma, 3);
    double diff_const = DIF_CONST*pow(gamma, 5);

    int m;
    /* edge focus */
    diff_bend_fringe(r, irho, entrance_angle, fint1, gap, FringeBendEntrance, 1.0, bdiff);
    /* quadrupole gradient fringe entrance*/   // default skip 
    // /if (FringeQuadEntrance && B[1]!=0) {
    //     if (useLinFrEleEntrance) /*Linear fringe fields from elegant*/
    //         linearQuadFringeElegantEntrance(r6, B[1], fringeIntM0, fringeIntP0);
    //     else
    //         QuadFringePassP(r6, B[1]);
    // }
    /* integrator */
    for (m=0; m < num_int_steps; m++) { /* Loop over slices */
        diff_drift(r, L1, bdiff);
        diff_bnd_kick(r, A, B, max_order, K1, irho, rad_const, diff_const, bdiff);
        diff_drift(r, L2, bdiff);
        diff_bnd_kick(r, A, B, max_order, K2, irho, rad_const, diff_const, bdiff);
        diff_drift(r, L2, bdiff);
        diff_bnd_kick(r, A, B, max_order, K1, irho, rad_const, diff_const, bdiff);
        diff_drift(r, L1, bdiff);
    }
    /* quadrupole gradient fringe */
    // if (FringeQuadExit && B[1]!=0) {
    //     if (useLinFrEleExit) /*Linear fringe fields from elegant*/
    //         linearQuadFringeElegantExit(r6, B[1], fringeIntM0, fringeIntP0);
    //     else
    //         QuadFringePassN(r6, B[1]);
    // }
    /* edge focus */
    diff_bend_fringe(r, irho, exit_angle, fint2, gap, FringeBendExit, -1.0, bdiff);
}
