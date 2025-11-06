#include "atconstants.h"
#include "driftkick.c"  	/* fastdrift.c, strthinkick.c */

void StrMPoleSymplectic4Pass(double *r, double le, double *A, double *B,
        int max_order, int num_int_steps)
{
    double SL = le/num_int_steps;
    double L1 = SL*DRIFT1;
    double L2 = SL*DRIFT2;
    double K1 = SL*KICK1;
    double K2 = SL*KICK2;
    double B0 = B[0];
    double A0 = A[0];

    int m;
    double p_norm, NormL1, NormL2;
    /* Check for change of reference momentum */
    p_norm = 1.0/(1.0+r[4]);
    NormL1 = L1*p_norm;
    NormL2 = L2*p_norm;
    /* integrator */
    for (m=0; m < num_int_steps; m++) { /* Loop over slices */
        fastdrift(r, NormL1);
        strthinkick(r, A, B, K1, max_order);
        fastdrift(r, NormL2);
        strthinkick(r, A, B, K2, max_order);
        fastdrift(r, NormL2);
        strthinkick(r, A, B, K1, max_order);
        fastdrift(r, NormL1);
    }
}
