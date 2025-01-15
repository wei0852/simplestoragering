// #include "atconstants.h"
// #include "atelem.c"
// #include "atlalib.c"
#include "atphyslib.c"
#include "driftkick.c"  /* strthinkick.c */

#include <math.h>
#include <stdbool.h>
/* Straight dipole w/ multipole using Symplectic Integration and rotation at
 * dipole faces.
 * Created by Xiaobiao Huang, 7/31/2018 */
#define SQR(X) ((X)*(X))
#define DRIFT1    0.6756035959798286638
#define DRIFT2   -0.1756035959798286639
#define KICK1     1.351207191959657328
#define KICK2    -1.702414383919314656


void E1rotation(double *r,double X0ref, double E1)
/* At Entrance Edge:
 * move particles to the field edge and convert coordinates to x, dx/dz, y,
 * dy/dz, then convert to x, px, y, py as integration is done with px, py */
{
    double x0,dxdz0, dydz0, psi;
    double fac;

    dxdz0 = r[1]/sqrt(SQR(1+r[4])-SQR(r[1])-SQR(r[3]));
    dydz0 = r[3]/sqrt(SQR(1+r[4])-SQR(r[1])-SQR(r[3]));
    x0 = r[0];

    psi = atan(dxdz0);
    r[0] = r[0]*cos(psi)/cos(E1+psi)+X0ref;
    r[1] = tan(E1+psi);
    r[3] = dydz0/(cos(E1)-dxdz0*sin(E1));
    r[2] += x0*sin(E1)*r[3];
    r[5] += x0*tan(E1)/(1-dxdz0*tan(E1))*sqrt(1+SQR(dxdz0)+SQR(dydz0));
    /* convert to px, py */
    fac = sqrt(1+SQR(r[1])+SQR(r[3]));
    r[1] = r[1]*(1+r[4])/fac;
    r[3] = r[3]*(1+r[4])/fac;
}

void E2rotation(double *r,double X0ref, double E2)
/* At Exit Edge:
 * move particles to arc edge and convert coordinates to x, px, y, py */
{
    double x0;
    double dxdz0, dydz0, psi, fac;

    dxdz0 = r[1]/sqrt(SQR(1+r[4])-SQR(r[1])-SQR(r[3]));
    dydz0 = r[3]/sqrt(SQR(1+r[4])-SQR(r[1])-SQR(r[3]));
    x0 = r[0];

    psi = atan(dxdz0);
    fac = sqrt(1+SQR(dxdz0)+SQR(dydz0));
    r[0] = (r[0]-X0ref)*cos(psi)/cos(E2+psi);
    r[1] = tan(E2+psi);
    r[3] = dydz0/(cos(E2)-dxdz0*sin(E2));
    r[2] += r[3]*(x0-X0ref)*sin(E2);
    r[5] += (x0-X0ref)*tan(E2)/(1-dxdz0*tan(E2))*fac;
    /* convert to px, py */
    fac = sqrt(1+SQR(r[1])+SQR(r[3]));
    r[1] = r[1]*(1+r[4])/fac;
    r[3] = r[3]*(1+r[4])/fac;
}

void edgey(double* r, double inv_rho, double edge_angle)
/* Edge focusing in dipoles with hard-edge field for vertical only */
{
    double psi = inv_rho*tan(edge_angle);
    /*r[1]+=r[0]*psi;*/
    r[3]-=r[2]*psi;
}

void edgey_fringe(double* r, double inv_rho, double edge_angle, double fint, double gap)
/* Edge focusing in dipoles with fringe field, for vertical only */
{
    /*double fx = inv_rho*tan(edge_angle);*/
    double psi_bar = edge_angle-inv_rho*gap*fint*(1+sin(edge_angle)*sin(edge_angle))/cos(edge_angle)/(1+r[4]);
    double fy = inv_rho*tan(psi_bar);
    /*r[1]+=r[0]*fx;*/
    r[3]-=r[2]*fy;
}

void ladrift6(double* r, double L)
/* large angle drift, X. Huang, 7/31/2018
 * Input parameter L is the physical length
 * 1/(1+delta) normalization is done internally
 * Hamiltonian H = (1+\delta)-sqrt{(1+\delta)^2-p_x^2-p_y^2}, change sign for
 * $\Delta z$ in AT */
{
    double p_norm = 1./sqrt(SQR(1+r[4])-SQR(r[1])-SQR(r[3]));
    double NormL = L*p_norm;
    r[0]+= NormL*r[1];
    r[2]+= NormL*r[3];
    r[5]+= L*(p_norm*(1+r[4])-1.);
}


void BndStrMPoleSymplectic4Pass(double *r, double le, double irho, double *A, double *B,
        int max_order, int num_int_steps,
        double entrance_angle, double exit_angle,
        double fint1, double fint2, double gap)
{
    double SL = le/num_int_steps;
    double L1 = SL*DRIFT1;
    double L2 = SL*DRIFT2;
    double K1 = SL*KICK1;
    double K2 = SL*KICK2;
    bool useFringe1 = (fint1 != 0) && (gap != 0);
    bool useFringe2 = (fint2 != 0) && (gap != 0);
    double B0 = B[0];
    double A0 = A[0];

    B[0] += irho;
    int m;
	/* edge focus */
	if (useFringe1)
	    edgey_fringe(r, irho, entrance_angle,fint1,gap);
	else
	    edgey(r, irho, entrance_angle);
    /* Rotate and translate to straight Cartesian coordinate */
    E1rotation(r, 0.0, entrance_angle);
    /* integrator */
    for (m=0; m < num_int_steps; m++) { /* Loop over slices */
		ladrift6(r,L1);
	    strthinkick(r, A, B, K1, max_order);
		ladrift6(r,L2);
	    strthinkick(r, A, B, K2, max_order);
		ladrift6(r,L2);
		strthinkick(r, A, B, K1, max_order);
		ladrift6(r,L1);
	}
    /* Rotate and translate back to curvilinear coordinate */
    E2rotation(r, 0.0, exit_angle);
    /* edge focus */
	if (useFringe2)
	    edgey_fringe(r, irho, exit_angle,fint2,gap);
	else    /* edge focus */
	    edgey(r, irho, exit_angle);
    B[0] = B0;
    A[0] = A0;
}
