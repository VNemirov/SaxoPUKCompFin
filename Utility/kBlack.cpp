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
	const int			ea,			//	european (0), american (1)
	const int			smooth,		//	smoothing
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
	kMatrix<double>& eecm,
	string& error)
{
	//	helps
	int h, i, p;

	//	construct s axis
	double t = max(0.0, expiry);
	double std = sigma * sqrt(t);
	
	int    nums = 2 * (max(0,numS) / 2 + 1);
	double dx = 2 * numStd * std / nums;

	//	early exercise curve matrix
	
	eecm.resize(nums, numT);

	s.resize(nums);
	for (i = 0; i < nums; ++i)
	{
		s(i) = s0 * exp((i - nums / 2.0) * dx);
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

		//	roll
		fd.res()(0) = res;
		for (h = numt - 1; h >= 0; --h)
		{
			fd.rollBwd(dt, update || h == (numt - 1), theta, wind, fd.res());
			if (ea > 0)
			{
				for (i = 0; i < nums; ++i) {
					fd.res()(0)(i) = max(res(i), fd.res()(0)(i));
					//	for i=0 fd.res()(0)(i) is greater than res(i), why?
					if (eec == 1) eecm(nums - 1 - i, h) = (fd.res()(0)(i) > res(i) ? 0 : 1);	
				}
			}
		}
	}

	//	set result
	if (eec == 0 || ea == 0) {
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
	const int			numK,
	const bool			update,
	const int			numPr,
	const int			pSetting,
	double& res0,
	kVector<double>& s,
	kVector<double>& k,
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
	k.resize(numK);

	double dk = strike / numK;

	k(0) = strike / 2;

	res.resize(numK, numT);


	for (i = 1; i < numK; i++) {
		k(i) = k(i-1) + dk;
	}


	for (int t = 0; t < numt; t++) {
		for (i = 0; i < numK; i++) {
			res(i,t) = 0.0;
			for (int c = 0; c < nums; c++) {
				res(i,t) += pMatrix(c,t) * max(s(c) - k(i), 0.0);
			}
		}
	}





	//for (i = 0; i < nums; i++) {
	//	res(i) = 0;
	//	for (j = i; j < nums; j++) {
	//		res(i) += fd.res()(0)(j) * max(s(j) - strike,0.0);
	//		//res(i) += (s(j)-s(i));
	//	}
	//}
	//res = fd.res()(0);

	res0 = res(numK/2 , (numt -1));



	//	done
	return true;
}