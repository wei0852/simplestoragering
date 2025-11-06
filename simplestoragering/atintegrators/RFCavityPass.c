/* 
 *  RFCavityPass.c
 *  Accelerator Toolbox 
 *  22/09/2015
 *  Nicola Carmignani
 */

#include "atconstants.h"
#include "attrackfunc.c"


void RFCavityPass(double *r_in, double le, double nv, double freq, double h, double lag, double philag,
                  int nturn, double T0)
/* le - physical length
   nv - peak voltage (V) normalized to the design enegy (eV)
   r is a 6-by-N matrix of initial conditions reshaped into
   1-d array of 6*N elements
*/
{
    trackRFCavity(r_in, le, nv, freq, h, lag, philag, nturn, T0);
}
