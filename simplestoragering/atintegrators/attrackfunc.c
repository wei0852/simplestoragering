#include <math.h>
#include "driftkickrad.c"

void trackRFCavity(double *r_in, double le, double nv, double freq, double h, double lag, double philag,
                  int nturn, double T0)
/* le - physical length
   nv - peak voltage (V) normalized to the design enegy (eV)
   r is a 6-by-N matrix of initial conditions reshaped into
   1-d array of 6*N elements
*/
{
    int c;

    /* If nv is 0 and length is 0, then skip this whole loop (good for passive rf cavities
        anyway if there is a cavity length, we have to loop through the particles
    */

    if (le == 0) {
        if (nv != 0) {
            r_in[4] += -nv*sin(TWOPI*freq*((r_in[5]-lag)/C0 - (h/freq-T0)*nturn) - philag);
        }
    }
    else {
        double halflength = le/2;
        /* Propagate through a drift equal to half cavity length */
        drift6(r_in, halflength);
        /* Longitudinal momentum kick */
        if(nv!=0.0) r_in[4] += -nv*sin(TWOPI*freq*((r_in[5]-lag)/C0 - (h/freq-T0)*nturn) - philag);
        /* Propagate through a drift equal to half cavity length */
        drift6(r_in, halflength);
    }
}
