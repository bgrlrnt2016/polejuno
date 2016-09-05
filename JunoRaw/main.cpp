#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>    

using namespace cv;
using namespace std;

#define TREEL double
#define TESTNIVEAU(x) x<valSeuil

void DeriveHomography(double *x, double q[], double *y, double **dyda, int na)
{
	double xx = x[0], yy = x[1];

	y[0] = q[0] * xx + q[1] * yy + q[2] ;
    y[1] = q[3] * xx + q[4] * yy + q[5];
	dyda[0][0] = xx;
	dyda[1][0] = yy;
	dyda[2][0] = 1;
	dyda[3][0] = 0;
	dyda[4][0] = 0;
	dyda[5][0] = 0;
	dyda[0][1] = 0;
	dyda[1][1] = 0;
	dyda[2][1] = 0;
	dyda[3][1] = xx;
	dyda[4][1] = yy;
	dyda[5][1] = 1;

}





#define SWAP(a,b) {TREEL temp=(a);(a)=(b);(b)=temp;}

char gaussj(TREEL **a, long n, TREEL **b, long m)
{
	long *indxc, *indxr, *ipiv;
	long i, icol, irow, j, k, l, ll;
	TREEL big, dum, pivinv;

	indxc = new long[n];
	indxr = new long[n];
	ipiv = new long[n];
	for (j = 0; j<n; j++) ipiv[j] = 0;
	for (i = 0; i<n; i++) {
		big = 0.0;
		for (j = 0; j<n; j++)
			if (ipiv[j] != 1)
				for (k = 0; k<n; k++) {
					if (ipiv[k] == 0) {
						if (fabs(a[j][k]) >= big) {
							big = fabs(a[j][k]);
							irow = j;
							icol = k;
						}
					}
					else if (ipiv[k] > 1) return 1;
				}
		++(ipiv[icol]);
		if (irow != icol) {
			for (l = 0; l<n; l++) SWAP(a[irow][l], a[icol][l])
				for (l = 0; l<m; l++) SWAP(b[irow][l], b[icol][l])
		}
		indxr[i] = irow;
		indxc[i] = icol;
		if (a[icol][icol] == 0.0) return 2;
		pivinv = 1.0 / a[icol][icol];
		a[icol][icol] = 1.0;
		for (l = 0; l<n; l++) a[icol][l] *= pivinv;
		for (l = 0; l<m; l++) b[icol][l] *= pivinv;
		for (ll = 0; ll<n; ll++)
			if (ll != icol) {
				dum = a[ll][icol];
				a[ll][icol] = 0.0;
				for (l = 0; l<n; l++) a[ll][l] -= a[icol][l] * dum;
				for (l = 0; l<m; l++) b[ll][l] -= b[icol][l] * dum;
			}
	}
	for (l = n - 1; l >= 0; l--) {
		if (indxr[l] != indxc[l])
			for (k = 0; k<n; k++)
				SWAP(a[k][indxr[l]], a[k][indxc[l]]);
	}
	delete ipiv;
	delete indxr;
	delete indxc;
	return 0;
}

void covsrt(TREEL **covar, long ma, long *lista, long mfit)
{
	int i, j;
	TREEL swap;

	for (j = 0; j<ma - 1; j++)
		for (i = j + 1; i<ma; i++) covar[i][j] = 0.0;
	for (i = 0; i<mfit - 1; i++)
		for (j = i + 1; j<mfit; j++) {
			if (lista[j] > lista[i])
				covar[lista[j]][lista[i]] = covar[i][j];
			else
				covar[lista[i]][lista[j]] = covar[i][j];
		}
	swap = covar[0][0];
	for (j = 0; j<ma; j++) {
		covar[0][j] = covar[j][j];
		covar[j][j] = 0.0;
	}
	covar[lista[0]][lista[0]] = swap;
	for (j = 1; j<mfit; j++) covar[lista[j]][lista[j]] = covar[0][j];
	for (j = 1; j<ma; j++)
		for (i = 0; i <= j - 1; i++) covar[i][j] = covar[j][i];
}

void mrqcof(TREEL *x, TREEL *y, TREEL *sig, long ndata, TREEL *a,
	long ma, long *lista, long mfit, TREEL **alpha, TREEL *beta,
	TREEL *chisq, void(*funcs)(TREEL *x, TREEL a[], TREEL *y, TREEL **dyda, int na))
	/* ANSI: void (*funcs)(TREEL *x,TREEL a[],TREEL *y,TREEL **dyda,int na); */
{
	int k, j, i;
	TREEL ymod[2], wt[2], sig2i, dy[2], **dyda;

	dyda = new double*[ma + 1];
	for (j = 0; j<ma + 1; j++)
        dyda[j] = new double[2];
	for (j = 0; j<mfit; j++)
	{
		for (k = 0; k <= j; k++)
			alpha[j][k] = 0.0;
		beta[j] = 0.0;
	}
	*chisq = 0.0;
	for (i = 0; i<ndata; i++)
	{
		(*funcs)(&x[2 * i], a, ymod, dyda, ma);
		sig2i = (sig[i] * sig[i]);
		dy[0] = y[2*i] - ymod[0];
		dy[1] = y[2*i+1] - ymod[1];
		for (j = 0; j<mfit; j++)
		{
			wt[0] = dyda[lista[j]][0] * sig2i;
			wt[1] = dyda[lista[j]][1] * sig2i;
			for (k = 0; k <= j; k++)
				alpha[j][k] += wt[0]*dyda[lista[k]][0]+ wt[1]*dyda[lista[k]][1];
			beta[j] += dy[0]*wt[0]+dy[1]*wt[1];
		}
		(*chisq) += (dy[0]*dy[0]+dy[1]*dy[1])*sig2i;
	}
	for (j = 1; j<mfit; j++)
		for (k = 0; k <= j - 1; k++)
			alpha[k][j] = alpha[j][k];
	delete dyda;
}

char mrqmin(TREEL *x, TREEL *y, TREEL *sig, long ndata, TREEL *a,
	long ma, long *lista, long mfit, TREEL **covar, TREEL **alpha,
	TREEL *chisq, void(*funcs)(TREEL *x, TREEL a[], TREEL *y, TREEL **dyda, int na), TREEL *alamda)
{
	int k, kk, j, ihit;
	static TREEL *da, *atry, **oneda, *beta, ochisq;

	if (*alamda < 0.0) {
		oneda = new double*[mfit];
		for (int i = 0; i<mfit; i++)
			oneda[i] = new double[1];
		atry = new double[ma];
		da = new double[ma];
		beta = new double[ma];
		kk = mfit + 1;
		for (j = 0; j<ma; j++) {
			ihit = 0;
			for (k = 0; k<mfit; k++)
				if (lista[k] == j) ihit++;
			if (ihit == 0)
				lista[kk++] = j;
			else if (ihit > 1) return 2;
		}
		if (kk != ma + 1) return 3;
		*alamda = 0.001;
		mrqcof(x, y, sig, ndata, a,  ma, lista, mfit, alpha, beta, chisq, funcs);
		ochisq = (*chisq);
	}
	for (j = 0; j<mfit; j++) {
		for (k = 0; k<mfit; k++) covar[j][k] = alpha[j][k];
		covar[j][j] = alpha[j][j] * (1.0 + (*alamda));
		oneda[j][0] = beta[j];
	}
	int wwww = gaussj(covar, mfit, oneda, 1);
	if (wwww)
		return 4;
	for (j = 0; j<mfit; j++)
		da[j] = oneda[j][0];
	if (*alamda == 0.0) {
		covsrt(covar, ma, lista, mfit);
		delete beta;
		delete da;
		delete atry;
		for (int i = 0; i<mfit; i++)
			delete oneda[i];
		delete oneda[0];
		return 0;
	}
	for (j = 0; j<ma; j++) atry[j] = a[j];
	for (j = 0; j<mfit; j++)
		atry[lista[j]] = a[lista[j]] + da[j];
	mrqcof(x, y, sig, ndata, atry,  ma, lista, mfit, covar, da, chisq, funcs);
	if (*chisq < ochisq) {
		*alamda *= 0.1;
		ochisq = (*chisq);
		for (j = 0; j<mfit; j++) {
			for (k = 0; k<mfit; k++) alpha[j][k] = covar[j][k];
			beta[j] = da[j];
			a[lista[j]] = atry[lista[j]];
		}
	}
	else {
		*alamda *= 10.0;
		*chisq = ochisq;
	}
	return 0;
}




vector<double> AjusteHomography(vector<Point2d> x,vector <Point2d> y,vector<double> paramIni)
{
	// Bruit sur les données
	double	*sig = new double[y.size()];
	// Dérivée de f(x,y) par rapport aux paramètres
	double	*dyda;
	// Niveau rouge,vert et bleu des pixels dans les marches
	long	itst;
	long 	ma=6, mfit=6;
	double	alamda = -1, chisq, oldchisq;
	// Matrice de covariance et de de gain voir Numerical Recipes
	double	**covar=new double*[ma], **alpha= new double*[ma];
	int	i, j;
    for (int i = 0; i<y.size();i++)
        sig[i] = 0.001 * (fabs(y[i].x)+fabs(y[i].y));
	for (int i = 0; i < mfit; i++)
	{
		covar[i] = new double[mfit];
		alpha[i] = new double[mfit];
	}
	int nbPixelsFit = y.size();
	vector<double> q;
	q.resize(paramIni.size());
	q = paramIni;
		// Minimisation
	alamda = -1;
	vector<long> lista(6);
	for (int i = 0; i<mfit; i++)
		lista[i] = i;
	int statusMin = mrqmin((double*)x.data(), (double*)y.data(), sig, nbPixelsFit, q.data(),  ma, lista.data(), mfit, covar, alpha,
		&chisq, &DeriveHomography, &alamda);
	itst = 0;
	int nbIteration = 100;
	while (itst<nbIteration)
	{
		oldchisq = chisq;
		mrqmin((double*)x.data(), (double*)y.data(), sig, nbPixelsFit, q.data(),  ma, lista.data(), mfit, covar, alpha,
			&chisq, &DeriveHomography, &alamda);
		if (chisq>oldchisq)
			itst = 0;
		else if (fabs(oldchisq - chisq)<FLT_EPSILON)
			itst++;
	}
	return q;
}


double PolyVal(Mat q, double x, double y)
{
    double z=0;

    z = q.at<double>(5,0) + q.at<double>(4,0) * y + q.at<double>(3,0)*x+q.at<double>(2,0)*x*y+q.at<double>(1,0)*y*y+q.at<double>(0,0)*x*x;
//    z = q.at<double>(2,0)+q.at<double>(1,0)*y+q.at<double>(0,0)*x;
    return z;
}
Mat PolyFit( vector<double> z, vector<Point2f> pt ,int n)
{

Mat M = Mat_<double>(z.size(),6);
Mat I=Mat_<double>(z.size(),1);
for (int i=0;i<z.size();i++)
    {
        double x=pt[i].x,y= pt[i].y;
        M.at<double>(i, 0) = x*x;
        M.at<double>(i, 1) = y*y;
        M.at<double>(i, 2) = x*y;
        M.at<double>(i, 3) = x;
        M.at<double>(i, 4) = y;
        M.at<double>(i, 5) = 1;
       /* M.at<double>(i, 0) = x;
        M.at<double>(i, 1) = y;
        M.at<double>(i, 2) = 1;*/
        I.at<double>(i, 0) = z[i];
    }
SVD s(M);
Mat q;
s.backSubst(I,q);
return q;
}

class JunoCam {
public :
    double focalLength; //Focal length in meter
    double pixelSize; // Pixel size in meter
    double omega; // Angular speed rotation
    double Ts; // Grabber period
    double threshStdLevel;
    double threshStdGr;
    int nbChannel; // Channel used
    int nbFrameLet;
    bool rawFormat;
    vector <double> yOc; //y-axis optical center ordinate for channel 
    int xOc;//x-axis optical center  
    vector<int> gapChannelInf;
    vector<int> gapChannelSup;
    vector<double> dAngle;
    int nbZone;
    Mat x;
//http://www.unmannedspaceflight.com/index.php?s=06b9e2696feec057060998762f8be5d8&showtopic=2548&st=465&p=203948&#entry203948
    bool SetMat(Mat a);

    JunoCam() :nbFrameLet(82),focalLength(0.010997), pixelSize(7.4e-6), omega(2 * acos(-1.0) / 30), Ts(0.365)
    {
//        rawFormat = false; yOc.push_back(3);yOc.push_back(-128);yOc.push_back(-128-128-2);
//        rawFormat = false; yOc.push_back(0);yOc.push_back(-121);yOc.push_back(-252);
//        rawFormat = false; yOc.push_back(7);yOc.push_back(-127);yOc.push_back(-248);
        rawFormat = false; /*yOc.push_back(766-600);*/;yOc.push_back(611-600);yOc.push_back(456-600);yOc.push_back(291-600);;
//        rawFormat = false; yOc.push_back(134);yOc.push_back(2);yOc.push_back(-119.5);
        dAngle.push_back(0);dAngle.push_back(0);dAngle.push_back(0);
        threshStdGr=10;threshStdLevel=3;

    }
    Mat JunoCam::ExtractRaw(int begFrameLet,int lastFrameLet);
    double LookForSeam(Mat a,Mat b,int seamHeight,int nbZone,vector<Point2f> &h,vector<bool> &status);// h  homography matrix such b(*(x,y)=v)=a for seam. return value is error. a is upper image 
    Mat MatchChannel(vector<Mat> plandst);
    Point MatchTemplate(Mat a,Mat b,Point2f p);
};

Point JunoCam::MatchTemplate(Mat a, Mat b,Point2f p)
{
    Mat rab;    
    matchTemplate(a,b,rab,CV_TM_CCORR_NORMED);
    double minx,maxx;
    Point minLoc,maxLoc;
    minMaxLoc(rab,&minx,&maxx,&minLoc,&maxLoc);
    maxLoc.x=-maxLoc.x+p.x;
    maxLoc.y=-maxLoc.y+p.y;
    return maxLoc;
}

/*
    Mat af,afgx,afgy,bf,bfgx,bfgy,afg,bfg,rab;
    a.convertTo(af,CV_32F);
    Scharr(a,afgx,CV_32F,1,0);
    Scharr(a,afgy,CV_32F,0,1);
    afg= afgx.mul(afgx) + afgy.mul(afgy);
    double reponse;
    char debug=0;
    for (int i = 0; i < nbZone; i++)
    {
        errorMin=DBL_MAX;
        int windowWidth=a.cols/nbZone;
        Point2f ps1,ps2;
        b(Rect(i*b.cols/nbZone,0,b.cols/nbZone,seamHeight)).copyTo(bSeam);
        // Gradient match
        Scharr(bSeam,bfgx,CV_32F,1,0);
        Scharr(bSeam,bfgy,CV_32F,0,1);
        bfg = bfgx.mul(bfgx) + bfgy.mul(bfgy);
        ps1 = MatchTemplate(afgx, bfgx, Point2f(i*b.cols/nbZone,a.rows));
        meanStdDev(bf,meanGr,stdDevGr);
        bSeam.convertTo(bf,CV_32F);
        // Image match
        ps2= MatchTemplate(af, bf, Point2f(i*b.cols/nbZone,a.rows));

*/

double  JunoCam::LookForSeam(Mat a, Mat b,int seamHeight,    int nbZone, vector<Point2f> &dm,vector<bool> &status)
{
    double errorMin=DBL_MAX;
    dm.clear();
    status.clear();
    Mat aSeam(seamHeight,a.cols,CV_8UC1),bSeam(a.rows,a.cols,CV_8UC1),bSeam2;
    Mat c;
    Mat meanIm,stdDevIm;
    Mat meanGr,stdDevGr;
    
    Mat af,afgx,afgy,bf,bfgx,bfgy,afg,bfg,rab;
    a.convertTo(af,CV_32F);
    Scharr(a,afgy,CV_32F,0,1);
    Scharr(a,afgx,CV_32F,1,0);
    double reponse;
    char debug=0;
    int windowStep=a.cols/(nbZone+3);
    int windowWidth=2*windowStep;
    for (int i = 0; i < nbZone; i++)
    {
        errorMin=DBL_MAX;
        Point2f ps1,ps2,ps3;
        b(Rect(i*windowStep,0,windowWidth,seamHeight)).copyTo(bSeam);
        // Gradient match
        Scharr(bSeam,bfgy,CV_32F,0,1);
        Scharr(bSeam,bfgx,CV_32F,1,0);
        ps1 = MatchTemplate(afgx, bfgx, Point2f(i*windowStep,a.rows));
        ps3 = MatchTemplate(afgy, bfgy, Point2f(i*windowStep,a.rows));
        meanStdDev(bfgx,meanGr,stdDevGr);
        bSeam.convertTo(bf,CV_32F);
        // Image match
        ps2= MatchTemplate(af, bf, Point2f(i*windowStep,a.rows));

        meanStdDev(bf,meanIm,stdDevIm);
        if (debug == 1)
        {
            imshow("seamA",a);
            imshow("seamB",bSeam);
            cout << stdDevIm.at<double>(0, 0) << "\t" << stdDevGr.at<double>(0, 0) << "\n";

            waitKey(); 
        }
        if (stdDevIm.at<double>(0, 0)>threshStdLevel && norm(ps1-ps2)<3 && stdDevGr.at<double>(0, 0)>threshStdGr && i>0 && i<nbZone-1)
        // if (stdDevIm.at<double>(0, 0)>threshStdLevel && i>0 && i<nbZone-1)
            status.push_back(true);
        else
        {
            status.push_back(false);
            if (stdDevIm.at<double>(0, 0)>threshStdLevel)
                ps2=ps1;
            if (stdDevGr.at<double>(0, 0)>threshStdGr)
                ps1=ps2;
        }
        dm.push_back(Point2f((ps1+ps2)/2));
    }
    double dy=0;
    int nb=0;
    for (int i = 0; i<dm.size();i++)
        if (status[i])
        {
            dy += dm[i].y;
            nb++;
        }
    if (nb>0)
        dm[0].y=dy/nb;
    return errorMin;

}



Mat JunoCam::ExtractRaw(int begFrameLet,int lastFrameLet)
{
    int nbFrameLet=lastFrameLet-begFrameLet+1;
    Mat result(128*(lastFrameLet-begFrameLet+1),x.cols,CV_8UC(nbChannel));
    Mat mapX( result.size(), CV_32FC1 );
    Mat mapY( result.size(), CV_32FC1 );
    result.setTo(Vec3b(0,0,0));
    vector<Mat> planDst(nbChannel);
    vector<Mat> planSrc(nbChannel);
    split(result,planSrc);
    nbZone=20;
    int offsetFrameLet=begFrameLet;
    for (int k = 0; k <nbChannel; k++)
    {
        for (int i = 0; i < nbFrameLet; i++)
        {
            int offset=i+offsetFrameLet;
            if (offset>=nbFrameLet)
                offset-=nbFrameLet;
            Range r(i*nbChannel*128+k*128,i*nbChannel*128+(k+1)*128);
            Rect rDst(0,offset*(128),x.cols,128);
            if (rDst.y + rDst.height <= planSrc[k].rows && rDst.y  >= 0)
                x.rowRange(r).copyTo(planSrc[k](rDst));
        }
        vector<vector<Point2f>> h(nbFrameLet);
        vector<vector<bool>> status(nbFrameLet);
        for (int i = 0; i < nbFrameLet; i++)
        {
            double error;
            if (i!=nbFrameLet-1)
                error=LookForSeam(planSrc[k].rowRange(Range(i*128,(i+1)*128)),planSrc[k].rowRange(Range((i+1)*128,(i+2)*128)),8,nbZone,h[i],status[i]);
            else
                error=LookForSeam(planSrc[k].rowRange(Range(i*128,(i+1)*128)),planSrc[k].rowRange(Range(i*128,(i+1)*128)),16,nbZone,h[i],status[i]);

        }
        vector<Point2f> moyZone(nbZone);
        vector<Point2f> moyFrameLet(nbFrameLet);
        vector<int> nbSuccessZone(nbZone);
        vector<int> nbSuccessFramelet(nbFrameLet);
/*        for (int j=0;j<nbFrameLet;j++)
        {
            nbSuccessFramelet[j]=0;
            moyFrameLet[j] = Point2f(0,0);
            for (int i=0;i<nbZone; i++)
                if (status[j][i])
                {
                   moyFrameLet[j] += h[j][i];
                   nbSuccessFramelet[j]++;
                }
        }
        Point2f lastPt;
        for (int j=0;j<nbFrameLet-1;j++)
        {
            if (nbSuccessFramelet[j]>0)
            {
                moyFrameLet[j] /= nbSuccessFramelet[j];
                lastPt=moyFrameLet[j];
            }
            else if (j>=1) 
                moyFrameLet[j] = lastPt;

        }
        for (int j=nbFrameLet-2;j>=0;j--)
        {
        if (nbSuccessFramelet[j]==0)
                moyFrameLet[j] = (moyFrameLet[j]+lastPt)/2;
            else
                lastPt=moyFrameLet[j];

        }
        for (int i=0;i<nbZone; i++)
        {
            nbSuccessZone[i]=0;
            moyZone[i] = Point2f(0,0);
            for (int j=0;j<nbFrameLet-1;j++)
                if (status[j][i]&& fabs(h[j][i].y-moyFrameLet[j].y)<2)
                {
                    nbSuccessZone[i]++;
                    moyZone[i]+=h[j][i];

                }
        }
        for (int i=0;i<nbZone; i++)
        {
            if (nbSuccessZone[i]>0)
            {
                moyZone[i]/=nbSuccessZone[i];
                lastPt=moyZone[i];
            }
            else if (i>=1 && nbSuccessZone[i-1]>0)
                moyZone[i] =moyZone[i-1];

        }
        for (int i=nbZone-2;i>=0; i--)
        {
            if (nbSuccessZone[i]==0 &&nbSuccessZone[i+1]>0 )
            {
                moyZone[i] = (moyZone[i] )/2 ;
            }
            else
                lastPt=moyZone[i];
        }
        for (int i=0;i<nbFrameLet;i++)
        {
            
            for (int j=0;j<nbZone; j++)
                if (!status[i][j] && nbSuccessFramelet[i]>0)
                {
                    status[i][j]=1;    
                    h[i][j].x = moyZone[j].x;
                    h[i][j].y = moyFrameLet[i].y;

                }

        }*/
        vector<double> zx,zy;
        vector<Point2f> pt;
        
        for (int i=0;i<nbFrameLet-1;i++)
        {
            
            for (int j=1;j<nbZone-1; j++)
                if (status[i][j])
                {
                        pt.push_back(Point2f(j,i));
                        zx.push_back(h[i][j].x);
                        zy.push_back(h[i][j].y);

                }

        }

        Mat qx = PolyFit(zx,pt,2);
        Mat qy = PolyFit(zy,pt,2);

        ofstream fs;
        if(k==0) fs.open("motion0.txt",ios::app);
        if(k==1) fs.open("motion1.txt",ios::app);
        if(k==2) fs.open("motion2.txt",ios::app);
        if (fs.is_open())
        {
            for (int i=0;i<nbFrameLet;i++)
            {
                
 /*               for (int j = 0; j < h[0].size(); j++)
                {
                    fs.width(10);
                    fs.right;
                    fs.scientific;
                    fs <<setprecision(4) << h[i][j].x << "\t" << h[i][j].y << "\t" ;
                }
                fs << "\n";*/
                fs.width(10);
                for (int j = 0; j < h[0].size(); j++)
                {
                    fs.width(10);
                    fs.right;
                    fs.scientific;
                    fs <<setprecision(4)<< PolyVal(qx,j,i) << "\t" << PolyVal(qy,j,i)  << "\t";
                }
                fs << "\n";
            }
        }
        fs.flush();
        fs.close();

        ;
        double offsetY=0;
        int prevFrameLet=0;
        int larZone = planSrc[k].cols/(nbZone+3);
        vector<double> dyCum(mapX.cols);
        vector<double> dXCum(mapX.cols);
#ifndef TOTO
        int frameLetRow=-1;
        int indFrame=0,indFramePrev=0;
        int offsetFameLetY=0;
        for (int j = 0; j < dyCum.size(); j++)
        {
            dyCum[j]= 0;
        }
        for (int i=0;i<mapX.rows;i++)
        {
            float *ptrX = (float*)mapX.ptr(i);
            float *ptrY = (float*)mapY.ptr(i);
            frameLetRow++;
            if (frameLetRow/ 128!=indFrame)
            {
                double maxOffset=0;
                for (int j = 0; j < dyCum.size(); j++)
                {
                    int indZone=j/larZone;
                    dyCum[j] = PolyVal(qy,indZone+static_cast<double>(j-indZone*larZone)/larZone,indFrame);
                    if (dyCum[j]>maxOffset)
                        maxOffset=dyCum[j];
                }

                indFrame=frameLetRow/128;
            }
            for (int j=0;j<mapX.cols;j++,ptrX++,ptrY++)
            {   
                int indZone=j/larZone;
                double dx=PolyVal(qx,indZone+static_cast<double>(j-indZone*larZone)/larZone,indFrame-1)*(1-static_cast<double>(frameLetRow-indFrame*128)/128);
                double dy=dyCum[j]*(1-static_cast<double>(frameLetRow-indFrame*128)/128);
                *ptrX=j+dx;
                *ptrY=frameLetRow+dy;

            }


        }



#else
        for (int ind = 0; ind < nbFrameLet; ind++)
        {
            int begRow,endRow;
            double dy0;
            if (ind == 0)
            {
                begRow=0;
                dy0=0;
                for (int ii=0;ii<dyCum.size();ii++)
                    dyCum[ii]=0;
            }
            else
            {
                dy0=PolyVal(qy,0,ind);
                offsetY -= dy0;
                begRow=offsetY+ind*128;
            }
//            begRow=0;
            endRow=begRow+128;
            if (endRow>=mapX.rows)
                endRow=mapX.rows;
            for (int i = begRow; i <endRow; i++)
            {
                float *ptrX = (float*)mapX.ptr(i);
                float *ptrY = (float*)mapY.ptr(i);
                for (int j = 0; j < mapX.cols; j++, ptrX++, ptrY++)
                {
                    int indZone=j/larZone;
                    if (indZone>=nbZone)
                        indZone=nbZone-1;
                    double dx,dy;
                    dx = PolyVal(qx,indZone+static_cast<double>(j-indZone*larZone)/larZone,ind);
                    dy = PolyVal(qy,indZone+static_cast<double>(j-indZone*larZone)/larZone,ind);
                    *ptrX=static_cast<float>(j+dx*(1.0-static_cast<double>(i-begRow)/(endRow-begRow-1))); // no radial distortion +3.8325e-8*(j-mapX.cols/2)*(j-mapX.cols/2)
                   //*ptrX=static_cast<float>(j+h[ind][indZone].x*(1.0-static_cast<double>(i-endRow)/(endRow-begRow-1))); // no radial distortion +3.8325e-8*(j-mapX.cols/2)*(j-mapX.cols/2)
                    int offsetY=(ind)*128;
                    if (offsetY>mapX.rows)
                        offsetY -=mapX.rows;
                    if (offsetY<0)
                        offsetY +=mapX.rows;
//                    *ptrY=static_cast<float>(offsetY+i-(begRow+dy0-dy));
//                    if (i-begRow+dy<128)
//                        *ptrY=static_cast<float>(i-begRow+offsetY);
                     *ptrY=static_cast<float>(i-begRow+offsetY/*-dyCum[j]-dy*/);
//                    *ptrY=static_cast<float>(offsetY+i-(begRow+h[ind][0].y-h[ind][indZone].y));
 //                  *ptrY=static_cast<float>(offsetY+i-(begRow+h[ind][0].y-dy));

//                   *ptrY=static_cast<float>(offsetY+i-(begRow+dy0-dy));
                }
            

            }
            for (int j = 0; j < dyCum.size(); j++)
            {
                int indZone=j/larZone;
                dyCum[j]+= PolyVal(qy,indZone+static_cast<double>(j-indZone*larZone)/larZone,ind);;

            }


        }
#endif
        if (k==0)
            remap(1.2*planSrc[k],planDst[k],mapX,mapY, CV_INTER_CUBIC);
        else
            remap(planSrc[k],planDst[k],mapX,mapY, CV_INTER_CUBIC);
        imwrite(format("pSrc%d.png",k),planSrc[k]);
        imwrite(format("pDst%d.png",k),planDst[k]);
        if(k==0)imshow("planDst0",planDst[0]);
        if(k==1)imshow("planDst1",planDst[1]);
        if(k==2)imshow("planDst2",planDst[2]);
        waitKey(30);
//            imshow("org",planSrc[k]);
    }
    result = MatchChannel(planDst);
    imshow("org",result);
    waitKey();
    return result;
}


bool JunoCam::SetMat(Mat a)
{
    // how many channel 
    nbChannel=4;// B,V,R,CH4
    xOc=a.cols/2;
    int nbFrame=a.rows/nbChannel;
    if (nbFrame*nbChannel-a.rows==0 && nbFrame/128>=82)
    {
        nbChannel = 4; 
        nbFrameLet = nbFrame/nbChannel/128;
        if (nbChannel * 128 * nbFrameLet == a.rows)
        {
            rawFormat=true;
            x=a;
            return true;
        }
    }
    nbChannel=3;// B,V,R
    nbFrame=a.rows/nbChannel;
    if (nbFrame*nbChannel-a.rows==0 )
    {
        nbFrameLet = nbFrame/128;
        if (nbChannel * 128 * nbFrameLet == a.rows)
        {
            rawFormat=true;
            x=a;
            return true;
        }
        rawFormat=false;
        nbChannel=-1;
        nbFrameLet=-1;
        return false;

    }
    nbChannel=1;// ?
    nbFrame=a.rows/nbChannel;
    if (nbFrame*nbChannel-a.rows==0 && nbFrame>82)
    {
        nbFrameLet = nbFrame/nbChannel/128;
        if (nbChannel * 128 * nbFrameLet == a.rows)
        {
            x=a;
            rawFormat=true;
            return true;
        }
        rawFormat=false;
        nbChannel=-1;
        nbFrameLet=-1;
        return false;
    }
    rawFormat=false;
    return false;
}
Mat JunoCam::MatchChannel(vector<Mat> planDst)
{
        Ptr<Feature2D> b;
    vector<Mat> descImg(nbChannel);
    vector<vector<KeyPoint>> keyImg(nbChannel);
    Mat result;

    for (int i = 0; i < nbChannel; i++)
    {
        b = cv::xfeatures2d::SURF::create(8+i,4,3,true);
        vector<KeyPoint> k;
        Mat d;

        b->detectAndCompute(planDst[i],Mat(),keyImg[i],descImg[i]);
        }
    vector<Mat> h;
    vector<Mat> pTst(nbChannel);
    for (int i = 1; i < nbChannel; i++)
    {
        int prev,curr=1;;
        if (i==1)
            prev=0;
        else
            prev=2;
        vector<DMatch> matches;
        BFMatcher descriptorMatcher(b->defaultNorm(),true);
        descriptorMatcher.match(descImg[prev], descImg[curr], matches, Mat());
        // Keep best matches 
        // We sort distance between descriptor matches
        Mat index;
        int nbMatch=int(matches.size());
        Mat tab(nbMatch, 1, CV_32F);
        for (int ii = 0; ii<nbMatch; ii++)
        {
            tab.at<float>(ii, 0) = matches[ii].distance;
        }
        sortIdx(tab, index, SORT_EVERY_COLUMN + SORT_ASCENDING);
        nbMatch =(nbMatch*2)/4;
        Point2f cornerLeftUp,cornerRightBottom;
        cornerLeftUp=keyImg[curr][matches[index.at<int>(0,0)].trainIdx].pt;
        cornerRightBottom=keyImg[curr][matches[index.at<int>(0,0)].trainIdx].pt;
        for (int k = 1; k < max(nbMatch, 6); k++)
        {
            int j = index.at<int>(k,0);
            if (keyImg[curr][matches[j].trainIdx].pt.x<cornerLeftUp.x)
                cornerLeftUp.x=keyImg[curr][matches[j].trainIdx].pt.x;
            if (keyImg[curr][matches[j].trainIdx].pt.y<cornerLeftUp.y)
                cornerLeftUp.y=keyImg[curr][matches[j].trainIdx].pt.y;
            if (keyImg[curr][matches[j].trainIdx].pt.x>cornerRightBottom.x)
                cornerRightBottom.x=keyImg[curr][matches[j].trainIdx].pt.x;
            if (keyImg[curr][matches[j].trainIdx].pt.y>cornerRightBottom.y)
                cornerRightBottom.y=keyImg[curr][matches[j].trainIdx].pt.y;
        }
        vector<Point2d> src,dst;
        Mat mapX( planDst[0].size(), CV_32FC1 );
        Mat mapY( planDst[0].size(), CV_32FC1 );
        int width=cornerRightBottom.x-cornerLeftUp.x,height=cornerRightBottom.y-cornerLeftUp.y;
        vector<Rect> squareZone;
        int nbZoneX=2,nbZoneY=1;
        int nbZone=nbZoneX*nbZoneY;
        for (int nbSquare = 0; nbSquare < nbZone; nbSquare++)
        {
            src.clear();
            dst.clear();
            Rect zone(cornerLeftUp.x+width/nbZoneX*(nbSquare%nbZoneX),cornerLeftUp.y+height/nbZoneY*(nbSquare/nbZoneX),width/nbZoneX,height/nbZoneY);
            while (src.size() < nbMatch/nbZone)
            {
                src.clear();
                dst.clear();
                
                for (int k = 0; k < max(nbMatch,6); k++)
                {
                    int j = index.at<int>(k,0);
                    if (keyImg[prev][matches[j].queryIdx].pt.inside(zone))
                    {
                    src.push_back(keyImg[prev][matches[j].queryIdx].pt); 
                    dst.push_back(keyImg[curr][matches[j].trainIdx].pt);
                    }
                }
                zone.x -= zone.width*0.05;
                zone.y -= zone.height*0.05;
                if (zone.x<0)
                    zone.x=0;
                if (zone.x>=planDst[0].cols)
                    zone.x=planDst[0].cols-1;
                if (zone.y<0)
                    zone.y=0;
                if (zone.y>=planDst[0].rows)
                    zone.x=planDst[0].rows-1;
                zone.width*=1.1;
                zone.height*=1.1;
            }
            squareZone.push_back(zone);
            Mat hh;
            hh = Mat::zeros(3,3,CV_64FC1);
            hh.at<double>(0,0)=1;
            hh.at<double>(0,1)=0;
            hh.at<double>(1,1)=1;
            hh.at<double>(1,0)=0;
            if (src.size() > 6 && dst.size() > 6)
            {
                Mat o;
                hh=findHomography(src,dst,RANSAC,1,o,3000,0.998);
                vector<double> paramIni = {hh.at<double>(0,0),hh.at<double>(0,1),hh.at<double>(0,2),hh.at<double>(1,0),hh.at<double>(1,1),hh.at<double>(1,2)};
                vector<Point2d> srca,dsta;
                for (int ii=0;ii<o.rows;ii++)
                    if (o.at<uchar>(ii, 0) != 0)
                    {
                        srca.push_back(src[ii]);
                        dsta.push_back(dst[ii]);
                    }
                vector<double> hhh=AjusteHomography(srca,dsta,paramIni);
                hh= (Mat_<double>(3,3) << hhh[0],hhh[1],hhh[2],hhh[3],hhh[4],hhh[5],0,0,1);
            }
            if (hh.empty() )
            {
                hh = Mat::zeros(3,3,CV_64FC1);
                hh.at<double>(0,0)=1;
                hh.at<double>(0,1)=0;
                hh.at<double>(1,1)=1;
                hh.at<double>(1,0)=0;

            }
            cout << "Zone " << zone << "****Match :" << src.size()<<endl;
            cout<<hh<<endl;
            h.push_back(hh);
            for (int ii = 0; ii <zone.height; ii++)
            {
                if (ii+zone.y>=0 && ii+zone.y<mapX.rows)
                {
                    float *ptrX = (float*)mapX.ptr(ii+zone.y)+zone.x;
                    float *ptrY = (float*)mapY.ptr(ii+zone.y)+zone.x;
                    for (int j = 0; j < zone.width; j++, ptrX++, ptrY++)
                    {
                        if (j+zone.x>=0 && j+zone.x<mapX.cols)
                        {
                            Mat p=(Mat_<double>(3,1)<<j+zone.x,ii+zone.y,1);
                            Mat q=hh.inv()*p;
                            *ptrX = q.at<double>(0,0)/q.at<double>(2,0);
                            *ptrY = q.at<double>(1,0)/q.at<double>(2,0);
                        }
                    }
                }
            }
        }
        if (i==1)
            remap(planDst[0],pTst[0],mapX,mapY, CV_INTER_CUBIC);
        if (i==2)
            remap(planDst[2],pTst[2],mapX,mapY, CV_INTER_CUBIC);

    }
    pTst[1] = planDst[1];
    //warpPerspective(1.2*planDst[0], pTst[0], h[0], planDst[0].size());
    //warpPerspective(planDst[2], pTst[2], h[1], planDst[2].size());
    merge(pTst,result);
    return result;

}





int main (int argc,char **argv)
{
    if (argc != 2)
    {
        cout << "give folder name";
        exit(-1);
    }

    JunoCam juno;
    vector<String> filenames; 
    String folder(argv[1]); // again we are using the Opencv's embedded "String" class

    glob(folder, filenames); // new function that does the job ;-)
    for(size_t i = 0; i < filenames.size(); ++i)
    {
        int posExt=filenames[i].find_last_of(".");
        int posFile=filenames[i].find_last_of("\\")+1;
        String s(filenames[i].substr(posFile,posExt-posFile)+"_OCVRGB.png");
        Mat src = imread(filenames[i],cv::IMREAD_UNCHANGED);

        if(src.empty())
            cerr << "Image "<<i<<" : "<<filenames[i]<<" is empty" << endl;
        else
        {
            cout << "Image "<<i<<" : "<<filenames[i]<<" size is " << src.size()<< endl;
            if (src.channels()>1)
                cvtColor(src,src,COLOR_BGR2GRAY);
            Mat r;
            if (juno.SetMat(src))
            {
//            Mat r = ProcessRaw(src,0,81);
              r =   juno.ExtractRaw(0,23);
//                r = juno.ProcessRaw(0,81);
                imshow("Image",r);

            }
            else
                cout << "Unknown format - File " << filenames[i] << "\n";
            imwrite(s,r);
            waitKey();

        }
    }
}




