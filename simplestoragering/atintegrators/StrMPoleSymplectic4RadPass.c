#include "diff_str_kick.c"
#include "diff_drift.c"

void StrMPoleSymplectic4RadPass(double *r, double le, double *A, double *B,
        int max_order, int num_int_steps,
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

    /* integrator */
    for (int m=0; m < num_int_steps; m++) { /* Loop over slices */
            diff_drift(r,L1, bdiff);
            diff_str_kick(r, A, B, max_order, K1, rad_const, diff_const, bdiff);
            diff_drift(r,L2, bdiff);
            diff_str_kick(r, A, B, max_order, K2, rad_const, diff_const, bdiff);
            diff_drift(r,L2, bdiff);
            diff_str_kick(r, A, B, max_order, K1, rad_const, diff_const, bdiff);
            diff_drift(r,L1, bdiff);
    }
}
