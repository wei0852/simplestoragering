#define SQR(X) ((X)*(X))

static double B2perp(double bx, double by, double irho,
                            double x, double xpr, double y, double ypr)
/* Calculates sqr(|e x B|) , where e is a unit vector in the direction of velocity   */

{
    double v_norm2 = 1.0/(SQR(1.0+x*irho)+ SQR(xpr) + SQR(ypr));

	/* components of the  velocity vector:
	   ex = xpr;
	   ey = ypr;
	   ez = (1+x*irho);
	*/

	return (SQR(by*(1+x*irho)) + SQR(bx*(1+x*irho)) + SQR(bx*ypr - by*xpr))*v_norm2 ;
}

static void diff_bnd_kick(double *r6, double *A, double *B, int max_order,
                          double L, double irho, double rad_const, double diff_const, double *bdiff) {
  /* clang-format off */
/*****************************************************************************
The design magnetic field Byo that provides this curvature By0 = irho * E0 /(c*e)
MUST NOT be included in the dipole term PolynomB(1)(MATLAB notation)(B[0] C notation)
of the By field expansion
HOWEVER!!! to calculate the effect of classical radiation the full field must be
used in the square of the |v x B|.
When calling B2perp(Bx, By, ...), use the By = ReSum + irho, where ReSum is the
normalized vertical field - sum of the polynomial terms in PolynomB.

The kick is given by

           e L      L delta      L x
theta  = - --- B  + -------  -  -----  ,
     x     p    y     rho           2
            0                    rho

         e L
theta  = --- B
     y    p   x
           0

                          max_order
                            ----
                            \                       n
	   (B + iB  )/ B rho  =  >   (iA  + B ) (x + iy)
         y    x             /       n    n
  	                        ----
                            n=0

  ******************************************************************************/
  /* clang-format on */
  int i;
  double ImSum = A[max_order];
  double ReSum = B[max_order];
  double x, xpr, y, ypr, p_norm, dp_0, B2P, factor;
  double ReSumTemp;

  /* recursively calculate the local transverse magnetic field */
  for (i = max_order - 1; i >= 0; i--) {
    ReSumTemp = ReSum * r6[0] - ImSum * r6[2] + B[i];
    ImSum = ImSum * r6[0] + ReSum * r6[2] + A[i];
    ReSum = ReSumTemp;
  }
  /* calculate angles from momenta */
  p_norm = 1.0 / (1.0+r6[4]);
  x = r6[0];
  xpr = r6[1] * p_norm;
  y = r6[2];
  ypr = r6[3] * p_norm;

  B2P = B2perp(ImSum, ReSum+irho, irho, x, xpr, y, ypr);
  factor = (1.0 + x*irho + (SQR(xpr) + SQR(ypr)) / 2.0) / SQR(p_norm) * L;

  // if (bdiff) {
  //   thinkickM(r6, A, B, max_order, L, irho, bdiff);
  //   thinkickB(r6, ReSum, ImSum, diff_const, B2P, factor, bdiff);
  // }

  dp_0 = r6[4]; /* save a copy of the initial value of dp/p */

  r6[4] -= rad_const * B2P * factor;

  /* recalculate momenta from angles after losing energy */
  p_norm = 1.0 / (1.0 + r6[4]);
  r6[1] = xpr / p_norm;
  r6[3] = ypr / p_norm;

  r6[1] -= L * (ReSum - (dp_0 - r6[0] * irho) * irho);
  r6[3] += L * ImSum;
  r6[5] += L * irho * r6[0]; /* pathlength */
}