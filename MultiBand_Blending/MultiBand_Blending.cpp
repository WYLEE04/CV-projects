#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#pragma comment (lib, "opencv_world4120")
using namespace cv;
std::vector<Mat> LaplacianPyramid(const Mat& src) {
	std::vector<Mat> ret;
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
Mat reconstruct(const std::vector<Mat>& pyr) {
	Mat resized, low;
	low = pyr.back();
	for (int i = pyr.size() - 1; i > 0; i--) {
		pyrUp(low, resized, pyr[i - 1].size());
		low = resized + pyr[i - 1];
	}
	return low;
}
std::vector<Mat> GaussianPyramid(const Mat& src) {
	std::vector<Mat> ret;
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
int main()
{
	Mat apple = imread("C:/Users/이우열/Downloads/burt_apple.png");
	Mat orange = imread("C:/Users/이우열/Downloads/burt_orange.png");
	Mat mask_apple = imread("C:/Users/이우열/Downloads/burt_mask.png");
	apple.convertTo(apple, CV_32F, 1 / 255.f);
	orange.convertTo(orange, CV_32F, 1 / 255.f);
	mask_apple.convertTo(mask_apple, CV_32F, 1 / 255.f);
	Mat mask_orange = Scalar(1, 1, 1) - mask_apple;
	auto lap_apple = LaplacianPyramid(apple);
	auto lap_orange = LaplacianPyramid(orange);
	auto mask_apple_gauss = GaussianPyramid(mask_apple);
	auto mask_orange_gauss = GaussianPyramid(mask_orange);
	std::vector<Mat> added;
	for (int i = 0; i < lap_apple.size(); i++) {
		Mat apple_masked, orange_masked, blended;
		multiply(lap_apple[i], mask_apple_gauss[i], apple_masked);
		multiply(lap_orange[i], mask_orange_gauss[i], orange_masked);
		blended = apple_masked + orange_masked;
		added.push_back(blended);
	}
	Mat result = reconstruct(added);
	imshow("Result", result);
	waitKey();
}