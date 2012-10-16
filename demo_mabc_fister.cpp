//============================================================================
// Name        : Firefly.cpp
// Authors     : Dr. Iztok Fister
// Version     : v1.0
// Created on  : Jan 30, 2012
//============================================================================

/* Demo of the Memetic Artificial Bee Colony algorithm for Large-Scale Global Optimization coded using C/C++ programming language */

/* Reference Papers*/

/* ˇ
 I. Fister, I.Jr. Fister, J. Brest and V. Zumer. Memetic Artificial Bee Colony Algorithm for Large-Scale Global Optimization. In IEEE Congress on Evolutionary Computation –
CEC 2012.

 I.Jr. Fister, I. Fister and J. Brest. A Hybrid Artificial Bee Colony Algorithm for Graph 3-
Coloring. In Swarm and Evolutionary Computation, Lecture Notes in Computer Science,
7269, Springer Berlin / Heidelberg, 66–74 (2012).

 */

/*Contact:
Iztok Fister (iztok.fister@uni-mb.si)
*/

#include "Header.h"
#include "stat.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <sys/time.h>
#include <cstdio>
#include <unistd.h>
#include <cfloat>

#define sqr(a)	((a)*(a))
#define MAX_FUNC	20
#define OUTPUT_FILE_NAME	"//home//iztok//Documents//Articles//CEC\'2012//demo"
#define OUTPUT_FILE_NAME2	"//home//iztok//Documents//Articles//CEC\'2012//M_ABC"
#define OUTPUT_DIR_NAME		"//home//iztok//Documents//Articles//CEC\'2012"
#define NP 2000 /* The number of colony size (employed bees+onlooker bees)*/
#define FOOD_NUMBER NP/2 /*The number of food sources equals the half of the colony size*/
int FoodNumber = FOOD_NUMBER;
//#define limit 100  /*A food source which could not be improved through "limit" trials is abandoned by its employed bee*/
int limit = 100;
//#define maxCycle 2500 /*The number of cycles for foraging {a stopping criteria}*/

#define DUMP 		1
#define TAO			0.1
#define LSA			1
#define LSA_RATE	0.001
#define INIT_STEP	0.020

#define INIT_CR		0.005
#define MIN_CR		0.001
#define MAX_CR		0.020

#define INIT_F		1.000
#define MIN_F		0.800
#define MAX_F		1.000

#define MIN_TRIES	1
#define MAX_TRIES	5	//20

#define MEAS_XI 	1
#define MEAS_v 		2
#define MEAS_PSI	3
#define MEAS_CHI	4
#define MEAS_PHI	5

/* Problem specific variables*/
#define dim 1000 /*The number of parameters of the problem to be optimized*/
#define runtime 25  /*Algorithm can be run many times in order to see its robustness*/

int lb = -100.0;
int ub = 100.0;

double Foods[FOOD_NUMBER][dim]; /*Foods is the population of food sources. Each row of Foods matrix is a vector holding D parameters to be optimized. The number of rows of Foods matrix equals to the FoodNumber*/
double f[FOOD_NUMBER];  /*f is a vector holding objective function values associated with food sources */
double fitness[FOOD_NUMBER]; /*fitness is a vector holding fitness (quality) values associated with food sources*/
double trial[FOOD_NUMBER]; /*trial is a vector holding trial numbers through which solutions can not be improved*/
double prob[FOOD_NUMBER]; /*prob is a vector holding probabilities of food sources (solutions) to be chosen*/
double solution [dim]; /*New solution (neighbour) produced by v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij}) j is a randomly chosen parameter and k is a randomlu chosen solution different from i*/
double ObjValSol; /*Objective function value of new solution*/
double FitnessSol; /*Fitness value of new solution*/
int neighbour, param2change; /*param2change corrresponds to j, neighbour corresponds to k in equation v_{ij}=x_{ij}+\phi_{ij}*(x_{kj}-x_{ij})*/
double GlobalMin; /*Optimum solution obtained by ABC algorithm*/
double GlobalPos; /*Optimum solution location obtained by ABC algorithm*/
double MaxFitDif; /*maximum abs(f_best-f_avg)*/
double GlobalParams[dim]; /*Parameters of the optimum solution*/
double GlobalMins[runtime]; /*GlobalMins holds the GlobalMin of each run in multiple runs*/
double r; /*a random number in the range [0,1)*/

/* parameters */
int max_fes = 3*pow(10.0,6);
int timesOfRun = 25;
int u = 0;			// generation counter
int measure = MEAS_CHI;		//PSI;
int appendix = 0;	// defaults file name appendix
double rate = LSA_RATE;
double cr0 = INIT_CR;	// crossover rate

int rwde_call;		// RWDE local search statistics
int rwde_eval;
int rwde_imp;
double step0 = INIT_STEP;

double ls_avg;		// average value of LS exploration/exploitation rate (used in LocalImproveBees())
double ls_std;		// standard deviation of LS exploration/exploitation rate (used in LocalImproveBees())
int ls_call;		// number of local search calls

int nma_call;		// NMA local search statistics
int nma_improve;
int nma_reflect;
int nma_expand;
int nma_inside;
int nma_outside;
int nma_shrink;

int scout_num;		// number of generated scouts

double* f_step;		// RWDE step size
double* step;		// RWDE step size
double* CR;
double* F;
int* imp1;
int* imp2;
int* simp1;
int* simp2;
int* ls_all;		// local search distribution
int* ls_imp;		// local search improvement distribution
int* rel;
int* tt;

void RWDE(int i);
void NMA(int index);
double f_avg(double& std, double& min_t, double& max_t, double& median_t);

Benchmarks* fp=NULL;

Benchmarks* generateFuncObj(int funcID){
	Benchmarks *fp;
	// run each of specified function in "configure.ini"
	if (funcID==1){
		fp = new F1();
		ub = 100.0;
        lb = -100.0;
	}else if (funcID==2){
		fp = new F2();
        lb = -5.0;
        ub = 5.0;
	}else if (funcID==3){
		fp = new F3();
        lb = -32.0;
        ub = 32.0;
	}else if (funcID==4){
		fp = new F4();
        ub = 100;
        lb = -100;
	}else if (funcID==5){
		fp = new F5();
        ub = 5.0;
        lb = -5.0;
	}else if (funcID==6){
		fp = new F6();
        ub = 32.0;
        lb = -32.0;
	}else if (funcID==7){
		fp = new F7();
        lb = -100.0;
        ub = 100.0;
	}else if (funcID==8){
		fp = new F8();
        ub = 100.0;
        lb = -100.0;
	}else if (funcID==9){
		fp = new F9();
        ub = 100.0;
        lb = -100.0;
	}else if (funcID==10){
		fp = new F10();
        ub = 5.0;
        lb = -5.0;
	}else if (funcID==11){
		fp = new F11();
        lb = -32.0;
        ub = 32.0;
	}else if (funcID==12){
		fp = new F12();
        ub = 100.0;
        lb = -100.0;
	}else if (funcID==13){
		fp = new F13();
        ub = 100.0;
        lb = -100.0;
	}else if (funcID==14){
		fp = new F14();
        ub = 100.0;
        lb = -100.0;
	}else if (funcID==15){
		fp = new F15();
        ub = 5.0;
        lb = -5.0;
	}else if (funcID==16){
		fp = new F16();
        lb = -32.0;
        ub = 32.0;
	}else if (funcID==17){
		fp = new F17();
        ub = 100.0;
        lb = -100.0;
	}else if (funcID==18){
		fp = new F18();
        ub = 100.0;
        lb = -100.0;
	}else if (funcID==19){
		fp = new F19();
        ub = 100.0;
        lb = -100.0;
	}else if (funcID==20){
		fp = new F20();
        ub = 100.0;
        lb = -100.0;
	}else{
		cerr<<"Fail to locate Specified Function Index"<<endl;
		exit(-1);
	}
	return fp;
}

double CalculateFitness(double fun)
{
	return fun;
}

/*The best food source is memorized*/
void MemorizeBestSource()
{
   int i,j;
    
	for(i=0;i<FoodNumber;i++)
	{
		if (f[i]<GlobalMin)
		{
			GlobalMin=f[i];
			GlobalPos=i;
			for(j=0;j<dim;j++)
				GlobalParams[j]=Foods[i][j];
//			cout << "Best f(x) at " << u << " found " << GlobalMin << "." << endl;
        }
	}
 }

/*Variables are initialized in the range [lb,ub]. If each parameter has different range, use arrays lb[j], ub[j] instead of lb and ub */
/* Counters of food sources are also initialized in this function*/
double init(int index)
{
	int j;

	for (j=0;j<dim;j++)
	{
		r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
        Foods[index][j]=r*(ub-lb)+lb;	//[index][j]
		solution[j]=Foods[index][j];		//[index][j];
	}

	f[index] = fp->compute(solution);
	fitness[index] = CalculateFitness(f[index]);
	trial[index]=0;
	tt[index]=u;
	u++;

	return fitness[index];
}

void alloc()
{
	tt = new int[FoodNumber];
	imp1 = new int[FoodNumber];
	imp2 = new int[FoodNumber];
	simp1 = new int[FoodNumber];
	simp2 = new int[FoodNumber];
	rel = new int[FoodNumber];
	ls_imp = new int[FoodNumber];
	ls_all = new int[FoodNumber];
	CR = new double[FoodNumber];
	F = new double[FoodNumber];
	step = new double[FoodNumber];
	f_step = new double[FoodNumber];
}
/*All food sources are initialized */
void initial()
{
	int i;

	for (i=0;i<FoodNumber;i++)
	{
		step[i] = step0*(ub-lb)/200.0;		// update step
		f_step[i] = 0;
		CR[i] = cr0;		// fixed crossover rate
		F[i] = INIT_F;			// fixed crossover rate
		tt[i] = 0;
		imp1[i] = 0;
		imp2[i] = 0;
		simp1[i] = 0;
		simp2[i] = 0;
		ls_imp[i] = 0;
		ls_all[i] = 0;
		rel[i] = 0;
	}

	rwde_call = 0;		// local search statistics
	rwde_eval = 0;
	rwde_imp = 0;
	ls_avg = 0;
	ls_std = 0;
	ls_call = 0;
	scout_num = 0;
	nma_call = 0;
	nma_improve =0;
	nma_reflect = 0;
	nma_expand = 0;
	nma_inside = 0;
	nma_outside = 0;
	nma_shrink = 0;
	u = 0;

	for(i=0;i<FoodNumber;i++)
	{
		init(i);
	}
	GlobalMin=f[0];
	GlobalPos=0;
	MaxFitDif=0;
    for(i=0;i<dim;i++)
    	GlobalParams[i]=Foods[0][i];

	time_t curtime;
	struct tm *loctime;

	/* Get the current time.  */
	curtime = time (NULL);

	/* Convert it to local time representation.  */
	loctime = localtime (&curtime);

    cout << "Start of run at " << asctime(loctime) << endl;
}

void finish(double elapsed)
{
	time_t curtime;
	struct tm *loctime;

	/* Get the current time.  */
	curtime = time (NULL);

	/* Convert it to local time representation.  */
	loctime = localtime (&curtime);

	cout << "End of run at " << asctime (loctime) << endl;
	double intpart, fract = modf(elapsed, &intpart);
	double sec = ((int) intpart)%60+fract;
	intpart = intpart/60;
	int min = ((int) intpart)%60;
	int hour = intpart/60;

	cout << "Elapsed= " << elapsed << " second(s), i.e. " << hour << ":";
	cout << min << ":" << sec << "." << endl;

	delete tt;
	delete imp1;
	delete imp2;
	delete simp1;
	delete simp2;
	delete rel;
	delete ls_imp;
	delete ls_all;
	delete CR;
	delete F;
	delete step;
	delete f_step;
}

void SelectVec2(int candidate, int* r1, int* r2)
{
	/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
	r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
	*r1=(int)(r*FoodNumber);

	/*Randomly selected solution must be different from the solution i*/
	while(*r1==candidate)
	{
		r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
		*r1=(int)(r*FoodNumber);
	}

	/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
	r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
	*r2=(int)(r*FoodNumber);

	/*Randomly selected solution must be different from the solution i*/
	while(*r2==candidate || *r2==*r1)
	{
		r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
		*r2=(int)(r*FoodNumber);
	}

}

void SelectVec3(int candidate, int* r1, int* r2, int* r3)
{
	/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
	r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
	*r1=(int)(r*FoodNumber);

	/*Randomly selected solution must be different from the solution i*/
	while(*r1==candidate)
	{
		r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
		*r1=(int)(r*FoodNumber);
	}

	/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
	r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
	*r2=(int)(r*FoodNumber);

	/*Randomly selected solution must be different from the solution i*/
	while(*r2==candidate || *r2==*r1)
	{
		r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
		*r2=(int)(r*FoodNumber);
	}

	/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
	r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
	*r3=(int)(r*FoodNumber);

	/*Randomly selected solution must be different from the solution i*/
	while(*r2==candidate || *r3==*r1 || *r3==*r2)
	{
		r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
		*r3=(int)(r*FoodNumber);
	}

}

double SendEmployedBees()
{
	int i,j;
	int r1, r2, r3;

    /*Employed Bee Phase*/
	for (i=0;i<FoodNumber;i++)
    {
#ifdef SA_BEES
		r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)));
		if(r < TAO)		// generate a new CR
			CR[i] = (MAX_CR-MIN_CR)*((double)rand() / ((double)(RAND_MAX)+(double)(1)))+MIN_CR;
		r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)));
		if(r < TAO)		// generate a new CR
			F[i] = (MAX_F-MIN_F)*((double)rand() / ((double)(RAND_MAX)+(double)(1)))+MIN_F;
#endif
		for(j=0;j<dim;j++)	// copy current solution
			solution[j]=Foods[i][j];		//[i][j];

		r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)) );
        param2change=(int)(r*FoodNumber);

		for (j=0;j<dim;j++)
        {
			r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)));
			if(r < CR[i] || j == param2change)
			{
				/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
				SelectVec3(i, &r1, &r2, &r3);
				double delta = (Foods[r2][param2change]-Foods[r3][param2change]);
#ifdef DEB
				r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
				delta=(ub-lb)*(1-pow(r, 1-u/(double)max_fes))*delta;
#endif
				solution[param2change]=Foods[r1][param2change]+F[i]*delta;

				/*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
				if (solution[param2change]<lb)
					solution[param2change]=lb;
				if (solution[param2change]>ub)
					solution[param2change]=ub;
			}
			param2change = (param2change+1)%dim;
	    }

        ObjValSol = fp->compute(solution);
        FitnessSol=CalculateFitness(ObjValSol);
        simp1[i]++;
    	u++;

        /*a greedy selection is applied between the current solution i and its mutant*/
        if (FitnessSol<fitness[i])
        {
        /*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
        	trial[i]=0;
        	for(j=0;j<dim;j++)
        		Foods[i][j]=solution[j];
        	f[i]=ObjValSol;
        	fitness[i]=FitnessSol;
        	imp1[i]++;
        }
        else
        {   /*if the solution i can not be improved, increase its trial counter*/
            trial[i]=trial[i]+1;
        }
	}
    /*end of employed bee phase*/
   	return FitnessSol;
}

/* A food source is chosen with the probability which is proportional to its quality*/
/*Different schemes can be used to calculate the probability values*/
/*For example prob(i)=fitness(i)/sum(fitness)*/
/*or in a way used in the method below prob(i)=a*fitness(i)/max(fitness)+b*/
/*probability values are calculated by using fitness values and normalized by dividing maximum fitness value*/
int CalculateProbabilities()
{
     int i, best = 0;
     double maxfit, minfit;
     double sumfit = 0.0;
     minfit=maxfit=fitness[0];

     for (i=1;i<FoodNumber;i++)
     {
      	if (fitness[i]<minfit)
     	{
         	minfit=fitness[i];
         	best = i;
     	}
     	if (fitness[i]>maxfit)
        	maxfit=fitness[i];
     	sumfit += fitness[i];
     }

     for (i=0;i<FoodNumber;i++)
     {
      	prob[i]=(0.9*((fitness[i]-minfit)/(maxfit-minfit)))+0.1;
     }
     return best;
}

double SendOnlookerBees(int best)
{
	int i,j,t;
	int r1, r2;
	i=0;
	t=0;
	/*onlooker Bee Phase*/
	while(t<FoodNumber)
    {
		r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
        if(r<prob[i]) /*choose a food source depending on its probability to be chosen*/
        {
        	t++;

#ifdef SA_BEES
        	r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)));
        	if(r < TAO)		// generate a new CR
    			CR[i] = (MAX_CR-MIN_CR)*((double)rand() / ((double)(RAND_MAX)+(double)(1)))+MIN_CR;
        	r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)));
        	if(r < TAO)		// generate a new F
    			F[i] = (MAX_F-MIN_F)*((double)rand() / ((double)(RAND_MAX)+(double)(1)))+MIN_F;
#endif
        	for(j=0;j<dim;j++)	// copy current solution
        		solution[j]=Foods[i][j];		//[i][j];

        	/*The parameter to be changed is determined randomly*/
        	r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)) );
        	param2change=(int)(r*FoodNumber);

        	for (j=0;((double)rand() / ((double)(RAND_MAX)+(double)(1)) < CR[i]) && (j<dim);j++)
        	{
    			r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)));
    			if(r < CR[i] || j == param2change)
    			{
    				/*A randomly chosen solution is used in producing a mutant solution of the solution i*/
    				SelectVec2(i, &r1, &r2);
    				// RandToBest/1/Bin
    				double delta = (Foods[best][param2change]-Foods[i][param2change])+(Foods[r1][param2change]-Foods[r2][param2change]);
#ifdef DEB
    				r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
    				delta=(ub-lb)*(1-pow(r, 1-u/(double)max_fes))*delta;
#endif
    				solution[param2change]=Foods[i][param2change]+F[i]*delta;

    				/*if generated parameter value is out of boundaries, it is shifted onto the boundaries*/
    				if (solution[param2change]<lb)
    					solution[param2change]=lb;
    				if (solution[param2change]>ub)
    					solution[param2change]=ub;
    			}
    			param2change = (param2change+1)%dim;
        	}

            ObjValSol = fp->compute(solution);
            FitnessSol=CalculateFitness(ObjValSol);
            simp2[i]++;
        	u++;

        	/*a greedy selection is applied between the current solution i and its mutant*/
        	if (FitnessSol<fitness[i])
        	{
        		/*If the mutant solution is better than the current solution i, replace the solution with the mutant and reset the trial counter of solution i*/
        		trial[i]=0;
        		for(j=0;j<dim;j++)
        			Foods[i][j]=solution[j];
        		f[i]=ObjValSol;
        		fitness[i]=FitnessSol;
            	imp2[i]++;
        	}
        	else
        	{   /*if the solution i can not be improved, increase its trial counter*/
        		trial[i]=trial[i]+1;
        	}
        } /*if */
        i++;
        if (i==FoodNumber-1)
        	i=0;
	}/*while*/

	return FitnessSol;
    /*end of onlooker bee phase     */
}

void LocalImproveBees(int best)
{
//	int i;
	double stdev, min_t, max_t, median_t;
	double avg = f_avg(stdev, min_t, max_t, median_t);
	double LAMBDA;

	/* prepare data for measue CHI */
	if(abs(min_t-avg) > MaxFitDif)
		MaxFitDif = abs(min_t-avg);

	/* calculate different measures*/
	double XI = min(abs((min_t-avg)/min_t), 1.0);		// EA, highly multi-modal, non scalable
	double v = min(1.0, stdev/abs(avg));				// SIA, DE, flexible, non scalable
	double PSI = 1-abs((avg-min_t)/(max_t-min_t));	// EA, Plateaus, flat landscape, very sensitive
	double CHI = abs(min_t-avg)/MaxFitDif;				// SIA, DE, flexible, very DE and SIA tailored
	double PHI = stdev/abs(min_t-max_t);				// SIA, DE, flexible, very sensitive

	/* select type of measure */
	switch(measure)
	{
	case MEAS_XI:
		LAMBDA = XI;
		break;
	case MEAS_v:
		LAMBDA = v;
		break;
	case MEAS_PHI:
		LAMBDA = PHI;
		break;
	case MEAS_CHI:
		LAMBDA = CHI;
		break;
	default: 	// PSI
		LAMBDA = PSI;
		break;
	}

	/* Donald Knuth's "The Art of Computer Programming, Volume 2: Seminumerical Algorithms", section 4.2.2 */
	if((ls_call-1) == 0)
	{
		ls_avg = LAMBDA;
		ls_std = 0.0;
	}
	else
	{
		double M_nm1 = ls_avg;
		double S_nm1 = (ls_call-2)*sqr(ls_std);
		ls_avg = M_nm1+(LAMBDA-M_nm1)/(double) ls_call;
		S_nm1 = S_nm1+(LAMBDA-M_nm1)*(LAMBDA-ls_avg);
		ls_std = sqrt(S_nm1/(double) (ls_call-1));
	}

	r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
	if(r < rate)
	{
		/* increment local search counter */
		ls_call++;

		/* distribution of the local search exploration/exploitation */
		double p = (ls_std == 0.0)?LAMBDA:exp(-(LAMBDA-ls_avg)/(2.0*sqr(ls_std)));	// SA
		/* reset statistical fields */
		ls_avg = LAMBDA;
		ls_std = 0.0;
		ls_call = 1;
		/* end of reset */
		r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );

		if(p < r)		// 'stochastic' if p=LAMBDA is enabled else 'SA'
			RWDE(best);		// exploitation
		else
			NMA(best);			// exploration
	}
}

void RWDE(int i)
{
	int j, t = 0;
	int t_max = MAX_TRIES;	//(MAX_TRIES-MIN_TRIES)*((double)u/(double)max_fes)+MIN_TRIES;
	double l = step[i];	//F[i];
	double s, z[dim];

	rwde_call++;
	for(j=0;j<dim;j++)
		solution[j]=Foods[i][j];
	while(t < t_max)
	{
		s = 0;
		for(j=0;j<dim;j++)				// generate a unit-length random vector
		{
			// computing of a new vector
			r = ((double)rand() / ((double)(RAND_MAX)+(double)(1)) );
			z[j] = (r-0.5)*2;
			s += (z[j]*z[j]);
		}

		for(j=0;j<dim;j++)				// generate a new decision vector
		{
			solution[j] = solution[j]+l*z[j]/sqrt(s);	 	//step				// generate new vector

			// adjusting of the new vector
			if(solution[j] > ub)			//Y_max[i])
			{
				solution[j] = ub;
			}
			if(solution[j] < lb)			//Y_min[i])
			{
				solution[j] = lb;
			}
		}

        ObjValSol = fp->compute(solution);
        FitnessSol=CalculateFitness(ObjValSol);
    	u++;

    	/*a greedy selection is applied between the current solution i and its mutant*/

     	if (FitnessSol<fitness[i])
     	{
     		if((fitness[i]-FitnessSol) > f_step[i])
     		{
     			f_step[i] = fitness[i]-FitnessSol;
     			step[i] = l;
     			cout << "Best lambda i= " << i << " step= " << step[i] << " f_step= " << f_step[i] << "." << endl;
     		}
    		trial[i]=0;
    		for(j=0;j<dim;j++)
    			Foods[i][j]=solution[j];
    		f[i]=ObjValSol;
    		fitness[i]=FitnessSol;
    		rwde_imp++;
    		ls_imp[i]++;
    	}
    	else
    	{
    		trial[i]=trial[i]+1;
    		l = l/2.0;
    	}

		t = t+1;
		rwde_eval++;
		ls_all[i]++;
	}
}

/* Nelder-Mead simplex Algorithm */
void NMA(int index)
{
	int i, j;
	int t = 0;
	int xnp1 = 0;		// largest point value
	int xn = 0;			// next smallest point value
	int x1 = 0;			// smallest point value
	double fr, fe;		// fitness values
	double fc, fcc;

	double centroid[dim];	// centriod
	double xr[dim]; 		// reflected vector
	double xe[dim]; 		// expanded vector
	double xc[dim]; 		// outside contraction vector
	double xcc[dim]; 		// inside contraction vector

	nma_call++;

	/* init of control parameters */
	double ALPHA = 1.0;
	double BETA = 1.0+2.0/(double) FoodNumber;
	double GAMMA = 0.75-1.0/(double)(2*FoodNumber);
	double DELTA = 1.0-1.0/(double) FoodNumber;

	/* main loop */
	while(t < MAX_TRIES)
	{
		/* calculate min and max point values */
		/* calculate second highest point value */
		x1 = xnp1 = xn = 0;
		for(i=1;i<FoodNumber;i++)
		{
			if(f[i] > f[xnp1])
				xnp1 = i;
			if(f[i] < f[x1])
				x1 = i;
			if(i != xnp1 && f[i] > f[xn])
				xn = i;
		}

		/* calculate centroid */
		for(j=0;j<dim;j++)
		{
			centroid[j]=0;
			for(i=0;i<FoodNumber;i++)
			{
				if(i != xnp1)
					centroid[j] += Foods[i][j];
			}
			centroid[j] /= (double) (FoodNumber-1);
		}

		for(j=0;j<dim;j++)		// reflection
			xr[j]=centroid[j]+ALPHA*(centroid[j]-Foods[xnp1][j]);

        ObjValSol = fp->compute(xr);
        fr=CalculateFitness(ObjValSol);
        u++;

        if(fr >= f[x1] && fr < f[xn])
        {
        	for(j=0;j<dim;j++)	// copy reflected vector
        		Foods[xnp1][j]=xr[j];
        	f[xnp1] = fr;
        	nma_reflect++;
        	nma_improve++;
        }
        else if(fr < f[x1])
        {
    		for(j=0;j<dim;j++)	// expansion
    			xe[j]=centroid[j]+BETA*(xr[j]-centroid[j]);

            ObjValSol = fp->compute(xe);
            fe=CalculateFitness(ObjValSol);
            u++;

            if(fe < fr)
            {
            	for(j=0;j<dim;j++)	// copy expanded vector
            		Foods[xnp1][j]=xe[j];
            	f[xnp1] = fe;
            	nma_expand++;
            	nma_improve++;
            }
            else
            {
            	for(j=0;j<dim;j++)	// copy reflected vector
            		Foods[xnp1][j]=xr[j];
            	f[xnp1] = fr;
            	nma_reflect++;
            	nma_improve++;
            }
        }
        else 		//if(fr >= f[xn])
        {
        	if(fr >= f[xn] && fr < f[xnp1])
        	{
        		for(j=0;j<dim;j++)	// outside contraction
        			xc[j]=centroid[j]+GAMMA*(xr[j]-centroid[j]);

                ObjValSol = fp->compute(xc);
                fc=CalculateFitness(ObjValSol);
                u++;

                if(fc <= fr)
                {
                	for(j=0;j<dim;j++)	// copy contracted vector
                		Foods[xnp1][j]=xc[j];
                	f[xnp1] = fc;
                	nma_outside++;
                	nma_improve++;
                }
                else	// shrink
                {
                	int maxtrialindex = (x1 == 0)?1:0;
                	for(i=1;i<FoodNumber;i++)
                	{
                		if ((i != x1) && (trial[i]>trial[maxtrialindex]))
							maxtrialindex=i;
               	    }
                	for(j=0;j<dim;j++)	// obtain a new neighbor vector
                    	Foods[maxtrialindex][j]=Foods[x1][j]+DELTA*(Foods[maxtrialindex][j]-Foods[x1][j]);
                	trial[maxtrialindex] = 0;
                	nma_shrink++;
                }
        	}
        	else 	// if(fr >= f[xnp1])
        	{
        		for(j=0;j<dim;j++)	// inside contraction
        			xcc[j]=centroid[j]-GAMMA*(centroid[j]-Foods[xnp1][j]);

                ObjValSol = fp->compute(xcc);
                fcc=CalculateFitness(ObjValSol);
                u++;

                if(fcc < f[xnp1])
                {
                	for(j=0;j<dim;j++)	// copy contracted vector
                		Foods[xnp1][j]=xcc[j];
                	f[xnp1] = fcc;
                	nma_inside++;
                	nma_improve++;
                }
                else	// shrink
                {
                	int maxtrialindex = (x1 == 0)?1:0;
                	for(i=1;i<FoodNumber;i++)
                	{
                		if ((i != x1) && (trial[i]>trial[maxtrialindex]))
							maxtrialindex=i;
               	    }
                	for(j=0;j<dim;j++)	// obtain a new neighbor vector
                    	Foods[maxtrialindex][j]=Foods[x1][j]+DELTA*(Foods[maxtrialindex][j]-Foods[x1][j]);
                	trial[maxtrialindex] = 0;
                	nma_shrink++;
                }
        	}
        }
        t++;
	}
}

/*determine the food sources whose trial counter exceeds the "limit" value. In Basic ABC, only one scout is allowed to occur in each cycle*/
void SendScoutBees()
{
	int maxtrialindex,i;
	r = (   (double)rand() / ((double)(RAND_MAX)+(double)(1)) );
	int n = (int)(r*FoodNumber);

	maxtrialindex=n;
	n = (maxtrialindex+1)%FoodNumber;
	for (i=0;i<FoodNumber;i++)
    {
         if (trial[n]>trial[maxtrialindex])	// && u > tt[n])
        	 maxtrialindex=n;
         n = (maxtrialindex+1)%FoodNumber;
    }
	if(trial[maxtrialindex]>=limit)
	{
		init(maxtrialindex);
		tt[maxtrialindex] = imp1[maxtrialindex];
		rel[maxtrialindex]++;
		scout_num++;
	}
}

/* average fitness */
double f_avg(double& std, double& min_t, double& max_t, double& median_t)
{
	double avg = 0;
	double median[dim];
	min_t = DBL_MAX;
	max_t = 0;

	for(int i=0;i<FoodNumber;i++)
	{
		if(f[i] < min_t)
			min_t = f[i];
		if(f[i] > max_t)
			max_t = f[i];
		avg += f[i];
		median[i] = f[i];
	}
	avg = avg/(double) FoodNumber;
	std = 0;
	for(int i=0;i<FoodNumber;i++)
	{
		std += sqr(f[i]-avg);
		for(int j=i+1;j<FoodNumber;j++)
		{
			if(median[i] < median[j])
			{
				double x = median[i];
				median[i] = median[j];
				median[j] = x;
			}
		}
	}
	std = sqrt(std/(double)(FoodNumber-1));
	median_t = median[FoodNumber/2];
	return avg;
}

/* average CR */
double CR_avg(double& std, double& min_t, double& max_t)
{
	double avg = 0;
	min_t = DBL_MAX;
	max_t = 0;

	for(int i=0;i<FoodNumber;i++)
	{
		if(CR[i] < min_t)
			min_t = CR[i];
		if(CR[i] > max_t)
			max_t = CR[i];
		avg += CR[i];
	}
	avg = avg/(double) FoodNumber;
	std = 0;
	for(int i=0;i<FoodNumber;i++)
	{
		std += sqr(CR[i]-avg);
	}
	std = sqrt(std/(double)(FoodNumber-1));
	return avg;
}

/* average CR */
double F_avg(double& std, double& min_t, double& max_t)
{
	double avg = 0;
	min_t = DBL_MAX;
	max_t = 0;

	for(int i=0;i<FoodNumber;i++)
	{
		if(F[i] < min_t)
			min_t = F[i];
		if(F[i] > max_t)
			max_t = F[i];
		avg += F[i];
	}
	avg = avg/(double) FoodNumber;
	std = 0;
	for(int i=0;i<FoodNumber;i++)
	{
		std += sqr(F[i]-avg);
	}
	std = sqrt(std/(double)(FoodNumber-1));
	return avg;
}

/* average trial */
double trial_avg(double& std, double& min_t, double& max_t)
{
	double avg = 0;
	min_t = DBL_MAX;
	max_t = 0;

	for(int i=0;i<FoodNumber;i++)
	{
		if(trial[i] < min_t)
			min_t = trial[i];
		if(trial[i] > max_t)
			max_t = trial[i];
		avg += trial[i];
	}
	avg = avg/(double) FoodNumber;
	std = 0;
	for(int i=0;i<FoodNumber;i++)
	{
		std += sqr(trial[i]-avg);
	}
	std = sqrt(std/(double)(FoodNumber-1));
	return avg;
}

void pop_stat(int fun, int level)
{
	double stdev, min_t, max_t, median_t;
	double avg = f_avg(stdev, min_t, max_t, median_t);

	cout << "Population statistics after " << u << " run(s): min= " << GlobalMin;
	cout << " median= " << median_t << " max= " << max_t << " mean= " << avg << " stdev= " << stdev << "." << endl;
	avg = trial_avg(stdev, min_t, max_t);
	cout << "Trial statistic(s) after " << scout_num << " scout(s): min= " << min_t << " max= " << max_t;
	cout << " avg= " << avg << " stdev= " << stdev << "." << endl;
	cout << "NMA local search statistic(s) after " << nma_call << " improve= " << nma_improve << " { " << nma_reflect;
	cout << ", " << nma_expand << ", " << nma_outside << ", " << nma_inside << " } shrink= " << nma_shrink << endl;
	cout << "RWDE local search statistic(s) after " << rwde_call << " eval= " << rwde_eval << " improve= " << rwde_imp << endl;
	cout << "Local search global ratio= " << (rwde_call+nma_call)/(double) max_fes << " Scout sent= " << scout_num << endl;
}

/* display syntax messages */
void help()
{
	cout << "Syntax:" << endl;
	cout << "  Cebelar [-h|-?] [-l] [-p] [-c] [-k] [-s] [-t]" << endl;
	cout << "    Parameters: -h|-? = command syntax" << endl;
	cout << " 				 -l = maximum random starts" << endl;
	cout << "				 -p = population size" << endl;
	cout << "				 -r = local search rate" << endl;
	cout << "				 -k = scout limit" << endl;
	cout << "				 -s = RWDE step size" << endl;
	cout << "				 -c = crossover rate (CR)" << endl;
	cout << "				 -mi = exploration/exploitation switch function" << endl;
	cout << "				 	[m1] = measure XI used" << endl;
	cout << "				 	[m2] = measure v used" << endl;
	cout << "				 	[m3] = measure PSI used (default)" << endl;
	cout << "				 	[m4] = measure CHI used" << endl;
	cout << "				 	[m5] = measure PHI used" << endl;
}

int main(int argc, char* argv[])
{
	/*  Test the basic benchmark function */	
	unsigned funToRun[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25};
	unsigned funNum = MAX_FUNC;
	bool test_flag = false;
	char fn[256], fn2[256];

	vector<double> runTimeVec;

    for(int i=1;i<argc;i++)
    {
    	if((strncmp(argv[i], "-h", 2) == 0) || (strncmp(argv[i], "-?", 2) == 0))
    	{
    		help();
    		return 0;
    	}
    	if(strncmp(argv[i], "-l", 2) == 0)
    	{
    		timesOfRun = atoi(&argv[i][2]);
    	}
    	else if(strncmp(argv[i], "-p", 2) == 0)
    	{
    		int k = atoi(&argv[i][2]);
    		if(k > NP || k < 2)
    		{
    			cout << "Error: array size overflow/underflow." << endl;
    			return -1;
    		}
    		FoodNumber = k/2;
    	}
    	else if(strncmp(argv[i], "-r", 2) == 0)		// LS rate
    	{
    		rate = atof(&argv[i][2]);
    	}
    	else if(strncmp(argv[i], "-k", 2) == 0)		// limit size
    	{
    		limit = atoi(&argv[i][2]);
    	}
    	else if(strncmp(argv[i], "-s", 2) == 0)		// RWDE step size
    	{
    		step0 = atof(&argv[i][2]);
    	}
    	else if(strncmp(argv[i], "-c", 2) == 0)		// crossover rate
    	{
    		cr0 = atof(&argv[i][2]);
    	}
    	else if(strncmp(argv[i], "-m", 2) == 0)		// measure
    	{
    		measure = atoi(&argv[i][2]);
    	}
    	else if(strncmp(argv[i], "-a", 2) == 0)		// appendix
    	{
    		appendix = atoi(&argv[i][2]);
    	}
    	else if(strncmp(argv[i], "-t", 2) == 0)		// test samples
    	{
    		test_flag = true;
    	}
    	else
    	{
    		cerr << "Fatal error: invalid parameter: " << argv[i] << endl;
    		return -1;
    	}
    }

    // allocating dynamic arrays
	stat st;
	time_t start, end;
	start = time(NULL);
	st.init(funNum, timesOfRun, 3);
	alloc();

	if(test_flag)
	{
		sprintf(fn2, "%s%d.txt", OUTPUT_FILE_NAME2, appendix);
		st.comp(OUTPUT_DIR_NAME, fn2);
		exit(0);
	}

	srand(1);		//time(NULL));

	for (unsigned i=0; i<funNum; i++)
	{
		fp = generateFuncObj(funToRun[i]);

        cout << "Optimize function: " << funToRun[i] << endl;
        cout << "Parameters: FEs= " << max_fes << " limit= " << limit << " pop_size= " << FoodNumber;
        cout << " LS_rate= " << rate << " X-over_rate= " << cr0 << " div.meas.= " << measure << endl;

        for (int j=0; j < timesOfRun; j++)
		{
        	int n = -1;
			cout << "Run: " << j+1 << endl;

			initial();
        	MemorizeBestSource();

			while(u <= max_fes)
		    {
				SendEmployedBees();
				int best = CalculateProbabilities();
#ifdef LSA
				LocalImproveBees(best);
#endif
				SendOnlookerBees(best);
				MemorizeBestSource();		// obtain a global best position
#ifdef LSA
				LocalImproveBees(GlobalPos);		//MEAS_CHI
#endif
				SendScoutBees();

				if(u/30000 > n)
				{
					n = u/30000;
					if(n == 4) {
						st.add(i, j, 0, GlobalMin);
						pop_stat(i, n);
					} else if(n == 20) {
						st.add(i, j, 1, GlobalMin);
						pop_stat(i, n);
					}
				}
			}
			st.add(i, j, 2, GlobalMin);

			cout << "Result of run " << j+1 << " is " << GlobalMin << "." << endl;
			cout << "NMA local search statistic(s) after " << nma_call << " improve= " << nma_improve << " { " << nma_reflect;
	    	cout << ", " << nma_expand << ", " << nma_outside << ", " << nma_inside << " } shrink= " << nma_shrink;
	    	cout << " ratio= " << nma_improve/(double) (nma_improve+nma_shrink) << endl;
	    	cout << "RWDE local search statistic(s) after " << rwde_call << " eval= " << rwde_eval << " improve= " << rwde_imp;
	    	cout << " ratio= " << rwde_imp/(double) rwde_eval << endl;
	    	cout << "Local search global ratio= " << (nma_improve+nma_shrink+rwde_eval)/(double) max_fes;
	    	cout << " Explore/Exploit ratio= " << (nma_improve+nma_shrink)/(double) rwde_eval << endl;
	    	cout << "Scout sent= " << scout_num << endl;
		}
        st.eval(i);
    	cout << "Results: min= " << st.get_min(i) << " max= " << st.get_max(i) << " avg= " << st.get_avg(i) << "." << endl;
		delete fp;
	}

	sprintf(fn, "%s%d.tex", OUTPUT_FILE_NAME, appendix);
	sprintf(fn2, "%s%d.txt", OUTPUT_FILE_NAME2, appendix);
	st.save(fn, fn2);
	st.comp(OUTPUT_DIR_NAME, fn2);
	end = time(NULL);
	finish(difftime(end, start));

	return 0;
}

