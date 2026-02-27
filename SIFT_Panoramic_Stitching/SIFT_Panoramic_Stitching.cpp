#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <algorithm>

#pragma comment(lib, "opencv_world4120")


using namespace cv;
using namespace std;

vector<Mat> LaplacianPyramid(const Mat& src) {
    vector<Mat> ret;
    Mat org, half, resized;
    org = src.clone();
    for (int i = 0; i < 5; i++) {
        pyrDown(org, half);
        pyrUp(half, resized, org.size());
        ret.push_back(org - resized);
        org = half;
    }
    ret.push_back(org);
    return ret;
}

Mat reconstruct(const vector<Mat>& pyr) {
    Mat resized, low;
    low = pyr.back();
    for (int i = pyr.size() - 1; i > 0; i--) {
        pyrUp(low, resized, pyr[i - 1].size());
        low = resized + pyr[i - 1];
    }
    return low;
}

vector<Mat> GaussianPyramid(const Mat& src) {
    vector<Mat> ret;
    Mat org = src.clone();
    ret.push_back(org);
    for (int i = 0; i < 5; i++) {
        Mat half;
        pyrDown(org, half);
        ret.push_back(half);
        org = half;
    }
    return ret;
}

Mat multiband_blend(const Mat& imgL, const Mat& imgR, const Mat& maskL, const Mat& maskR) {
    Mat maskL_3ch, maskR_3ch;
    cvtColor(maskL, maskL_3ch, COLOR_GRAY2BGR);
    cvtColor(maskR, maskR_3ch, COLOR_GRAY2BGR);

    auto lap_L = LaplacianPyramid(imgL);
    auto lap_R = LaplacianPyramid(imgR);
    auto mask_L_gauss = GaussianPyramid(maskL_3ch);
    auto mask_R_gauss = GaussianPyramid(maskR_3ch);

    vector<Mat> blended_pyr;
    for (size_t i = 0; i < lap_L.size(); i++) {
        Mat L_masked, R_masked, blended;
        multiply(lap_L[i], mask_L_gauss[i], L_masked);
        multiply(lap_R[i], mask_R_gauss[i], R_masked);
        blended = L_masked + R_masked;
        blended_pyr.push_back(blended);
    }

    return reconstruct(blended_pyr);
}

int main() {
    Mat srcL = imread("left.jpg");
    Mat srcR = imread("right.jpg");

    if (srcL.empty() || srcR.empty()) {
        cout << "오류 발생!" << endl;
        return -1;
    }
    cout << "왼쪽 이미지 크기: " << srcL.size() << endl;
    cout << "오른쪽 이미지 크기: " << srcR.size() << endl;

    Ptr<SIFT> sift = SIFT::create();

    vector<KeyPoint> keypointsL, keypointsR;
    Mat descriptorL, descriptorR;

    sift->detectAndCompute(srcL, noArray(), keypointsL, descriptorL);
    sift->detectAndCompute(srcR, noArray(), keypointsR, descriptorR);

    cout << "왼쪽 키포인트 수: " << keypointsL.size() << endl;
    cout << "오른쪽 키포인트 수: " << keypointsR.size() << endl;

    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(descriptorL, descriptorR, matches);

    cout << "전체 매칭 수: " << matches.size() << endl;

    sort(matches.begin(), matches.end(), [](const DMatch& a, const DMatch& b) {
        return a.distance < b.distance;
        });

    vector<DMatch> goodMatches;
    int numGoodMatches = max(10, (int)(matches.size() * 0.15));
    for (int i = 0; i < numGoodMatches && i < (int)matches.size(); i++) {
        goodMatches.push_back(matches[i]);
    }

    cout << "Good 매칭 수: " << goodMatches.size() << endl;

    Mat matchImg;
    drawMatches(srcL, keypointsL, srcR, keypointsR, goodMatches, matchImg);
    imshow("Good Matches", matchImg);

    vector<Point2f> ptsL, ptsR;
    for (const auto& m : goodMatches) {
        ptsL.push_back(keypointsL[m.queryIdx].pt);
        ptsR.push_back(keypointsR[m.trainIdx].pt);
    }

    Mat H = findHomography(ptsR, ptsL, RANSAC, 5.0);

    cout << "Homography matrix:" << endl << H << endl;

    Size resultSize(srcL.cols * 2, srcL.rows);

    Mat srcL32F, srcR32F;
    srcL.convertTo(srcL32F, CV_32FC3, 1.0 / 255.0);
    srcR.convertTo(srcR32F, CV_32FC3, 1.0 / 255.0);

    Mat largeL = Mat::zeros(resultSize, CV_32FC3);
    srcL32F.copyTo(largeL(Rect(Point(0, 0), srcL32F.size())));

    Mat largeR;
    warpPerspective(srcR32F, largeR, H, resultSize);

    Mat largeL_display, largeR_display;
    largeL.convertTo(largeL_display, CV_8UC3, 255.0);
    largeR.convertTo(largeR_display, CV_8UC3, 255.0);
    imshow("largeL", largeL_display);
    imshow("largeR", largeR_display);

    Point2f centerL(srcL.cols / 2.0f, srcL.rows / 2.0f);

    vector<Point2f> centerR_src = { Point2f(srcR.cols / 2.0f, srcR.rows / 2.0f) };
    vector<Point2f> centerR_dst;
    perspectiveTransform(centerR_src, centerR_dst, H);
    Point2f centerR = centerR_dst[0];

    cout << "왼쪽 중점: " << centerL << endl;
    cout << "오른쪽 중점 (변환 후): " << centerR << endl;

    Mat distL_input = Mat::ones(resultSize, CV_8UC1) * 255;
    if (centerL.x >= 0 && centerL.x < resultSize.width &&
        centerL.y >= 0 && centerL.y < resultSize.height) {
        distL_input.at<uchar>((int)centerL.y, (int)centerL.x) = 0;
    }
    Mat distL;
    distanceTransform(distL_input, distL, DIST_L2, DIST_MASK_3);

    Mat distR_input = Mat::ones(resultSize, CV_8UC1) * 255;
    if (centerR.x >= 0 && centerR.x < resultSize.width &&
        centerR.y >= 0 && centerR.y < resultSize.height) {
        distR_input.at<uchar>((int)centerR.y, (int)centerR.x) = 0;
    }
    Mat distR;
    distanceTransform(distR_input, distR, DIST_L2, DIST_MASK_3);

    Mat maskL = Mat::zeros(resultSize, CV_32FC1);
    maskL.setTo(1, distL < distR);

    Mat maskR = Mat::zeros(resultSize, CV_32FC1);
    maskR.setTo(1, distR <= distL);

    
    Mat maskL_display, maskR_display;
    maskL.convertTo(maskL_display, CV_8UC1, 255.0);
    maskR.convertTo(maskR_display, CV_8UC1, 255.0);
    imshow("maskL", maskL_display);
    imshow("maskR", maskR_display);

    Mat result = multiband_blend(largeL, largeR, maskL, maskR);

    Mat result_display;
    result.convertTo(result_display, CV_8UC3, 255.0);
    result_display.setTo(0, result_display < 0);
    result_display.setTo(255, result_display > 255);

    imshow("Final Panorama", result_display);

    cout << "\n 종료! " << endl;
    waitKey(0);

    return 0;
}