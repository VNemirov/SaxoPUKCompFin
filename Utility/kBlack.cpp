#include "kBlack.h"
#include "kSolver.h"
#include "kFd1d.h"

#include <string>
#include <iostream>

class kBlackObj : public kSolverObjective
{
public:

	//	constructor
	kBlackObj(
		double	expiry,
		double	strike,
		double	price,
		double	forward)
		: kSolverObjective(),
		myExpiry(expiry),
		myStrike(strike),
		myPrice(price),
		myForward(forward)
	{}

	//	value
	virtual double	value(double x)
	{
		double res = kBlack::call(myExpiry, myStrike, myForward, x) - myPrice;

		//	done
		return res;
	}

	//	deriv
	virtual double	deriv(double x)
	{
		double res = kBlack::vega(myExpiry, myStrike, myForward, x);

		//	done
		return res;
	}

	//	private parts
private:

	//	expiry
	double	myExpiry;
	double	myStrike;
	double	myPrice;
	double	myForward;

};

// implied vol
double
kBlack::implied(
	double	expiry,
	double	strike,
	double	price,
	double	forward)
{
	//	calc intrinsic
	double intrinc = max(forward - strike,0.0);
	if (price <= intrinc) return 0.0;

	//	objective
	kBlackObj obj(expiry, strike, price, forward);

	//	start guess
	double volatility = 0.1;
	int    numIter = 10;
	double epsilon = (price - intrinc) * kConstants::epsilon();

	//	solve
	kSolver::newtonRapson(obj, volatility, numIter, epsilon, nullptr);

	//	bound
	volatility = max(0.0, volatility);

	//	done
	return volatility;
}

//	fd runner
bool
kBlack::fdRunner(
	const double		s0,
	const double		r,
	const double		mu,
	const double		sigma,

	const double		expiry,
	const double		strike,
	const bool			dig,
	const int			pc,			//	put (-1) call (1)
	const int			ead,		//	european (0), american (1)
	const int			smooth,		//	smoothing
	const double		barrier,	//	barrier

	const double		theta,
	const int			wind,
	const double		numStd,
	const int			numT,
	const int			numS,
	const bool			update,
	const int			numPr,
	const bool			eec,		// early exercise curve
	double& res0,
	kVector<double>& s,
	kVector<double>& res,
	kVector<double>& eecv,
	string& error)
{
	//	helps
	int h, i, p, bi;

	//	construct s axis
	double t = max(0.0, expiry);
	double std = sigma * sqrt(t);
	
	int    nums = 2 * (max(0,numS) / 2 + 1);
	double dx = 2 * numStd * std / nums;

	//	early exercise curve matrix
	
	eecv.resize(numT);

	s.resize(nums);
	for (i = 0; i < nums; ++i)
	{
		s(i) = s0 * exp((i - nums / 2.0) * dx);
	}

	//	adder barrier to grid if option is down-and-out
	if (ead == 3)
	{
		for (i = 0; i < nums; i++) {
			if (s(i) > barrier && i > 0) {
				// determening wether s(i) or s(i-1) is closer to barrier
				if (abs(s(i) - barrier) < abs(s(i - 1) - barrier)) {
					bi = i;
					s(bi) = barrier;
					break;
				}
				else
				{
					bi = i-1;
					s(bi) = barrier;
					break;
				}
			}
			else if (s(i) > barrier && i == 0) {
				bi = i;
				s(bi) = barrier;
				break;
			}
		}
	}

	//	construct fd grid
	kFd1d<double> fd;
	fd.init(1, s, false);

	//	set terminal result
	double xl, xu;
	res.resize(nums);
	for (i = 0; i < nums; ++i)
	{
		if (smooth == 0 || i == 0 || i == nums - 1)
		{
			if (dig) res(i) = 0.5 * (kInlines::sign(s(i) - strike) + 1.0);
			else     res(i) = max(0.0, s(i) - strike);
		}
		else
		{
			xl = 0.5 * (s(i - 1) + s(i));
			xu = 0.5 * (s(i) + s(i + 1));
			if (dig) res(i) = kFiniteDifference::smoothDigital(xl, xu, strike);
			else	 res(i) = kFiniteDifference::smoothCall(xl, xu, strike);
		}

		if (pc < 0)
		{
			if (dig) res(i) = 1.0 - res(i);
			else	 res(i) -= (s(i) - strike);
		}
	}

	//	time steps
	int    numt = max(0, numT);
	double dt = t / max(1, numt);

	//	repeat
	int nump = max(1, numPr);
	for (p = 0; p < nump; ++p)
	{
		//	set parameters
		for (i = 0; i < nums; ++i)
		{
			fd.r()(i) = r;
			fd.mu()(i) = mu * s(i);
			fd.var()(i) = sigma * sigma * s(i) * s(i);
		}

		if (ead == 3) { //improved barrier
			fd.mu()(bi) = 0.0;
			fd.var()(bi) = 0.0;
		}
		
		//	roll
		fd.res()(0) = res;
		for (h = numt-1; h >= 0; h--)
		{
			fd.rollBwd(dt, update || h == (numt - 1), theta, wind, fd.res());
			if (ead == 1)
			{
				for (i = nums-1; i >= 0; i--) {
					if (res(i)> abs(fd.res()(0)(i)))
					{
						if (eec == 1) eecv(h) = s(i);
						fd.res()(0)(i) = res(i);
					}
					//fd.res()(0)(i) = max(res(i), fd.res()(0)(i));
				}
			}
			else if (ead >= 2) 
			{
				for (i = 0; i < nums; i++) {
					if (s(i) <= barrier) fd.res()(0)(i) = 0.0;
				}
			}
		}
	}

	//	set result
	if (eec == 0 || ead != 1) {
		res = fd.res()(0);
		res0 = fd.res()(0)(nums / 2);
	}
		
	
		
	//	done
	return true;
}


bool
kBlack::fdFwdRunner(
	const double		s0,
	const double		r,
	const double		mu,
	const double		sigma,

	const double		expiry,
	const double		strike,
	const bool			dig,
	const int			pc,			//	put (-1) call (1)
	const int			smooth,		//	smoothing

	const double		theta,
	const int			wind,
	const double		numStd,
	const int			numT,
	const int			numS,
	const bool			update,
	const int			numPr,
	const int			pSetting,

	double& res0,
	kVector<double>& s,
	kMatrix<double>& pMatrix,
	kMatrix<double>& res,
	string& error)
{
	//	helps
	int h, i, j, p;

	//	construct s axis
	double t = max(0.0, expiry);
	double std = sigma * sqrt(t);

	int    nums = 2 * (max(0, numS) / 2 + 1);
	double dx = 2 * numStd * std / nums;

	s.resize(nums);
	for (i = 0; i < nums; ++i)
	{
		s(i) = s0 * exp((i - nums / 2.0) * dx);
	}

	//	construct fd grid
	kFd1d<double> fd;
	fd.init(1, s, false);

	//	set inital values result
	kVector<double> initialP;
	initialP.resize(nums);
	for (i = 0; i < nums; ++i)
	{
		initialP(i) = s(i) == s0 ? 1.0 : 0.0;
	}

	//	time steps
	int    numt = max(0, numT);
	double dt = t / max(1, numt);

	pMatrix.resize(nums, numt);

	//	repeat
	int nump = max(1, numPr);
	for (p = 0; p < nump; ++p)
	{
		//	set parameters
		for (i = 0; i < nums; ++i)
		{
			fd.r()(i) = r;
			fd.mu()(i) = mu * s(i);
			fd.var()(i) = sigma * sigma * s(i) * s(i);
		} 

		fd.res()(0) = initialP;
		for (h = 0; h < numt; ++h)
		{
			for (int c = 0; c < nums; c++) {
				pMatrix(c, h) = fd.res()(0)(c);
			}
			fd.rollFwd(dt, update || h == 0, theta, wind, fd.res());	
		}

		//	roll
		

	}

	//	set result
	res.resize(nums, numT);

	for (int t = 0; t < numt; t++) {
		for (i = 0; i < nums; i++) {
			res(i,t) = 0.0;
			for (int c = 0; c < nums; c++) {
				res(i,t) += pMatrix(c,t) * max(s(c) - s(i), 0.0);
			}
		}
	}

	res0 = res(nums/2, numt-1);

	//	done
	return true;
}