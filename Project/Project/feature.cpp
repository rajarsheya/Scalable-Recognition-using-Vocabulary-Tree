#include <opencv2/opencv.hpp>

class FeatureDetector {
private:
    cv::Ptr<cv::SIFT> sift;
    cv::Ptr<cv::ORB> orb;

public:
    FeatureDetector() {
        sift = cv::SIFT::create();
        orb = cv::ORB::create();
    }

    void detect(cv::Mat& img1, std::vector<cv::KeyPoint>& kp1, cv::Mat& des1, std::string method = "SIFT") {
        cv::Mat gray1;
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);

        if (method == "SIFT") {
            sift->detectAndCompute(gray1, cv::noArray(), kp1, des1);
        } else if (method == "ORB") {
            orb->detectAndCompute(gray1, cv::noArray(), kp1, des1);
        }
    }

    std::vector<std::tuple<cv::Point2f, cv::Point2f, float>> match(
            std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
            cv::Mat& des1, cv::Mat& des2) {
        std::vector<std::tuple<cv::Point2f, cv::Point2f, float>> result;

        for (int i = 0; i < des1.rows; i++) {
            cv::Mat distance;
            cv::reduce(cv::abs(des2 - des1.row(i)), distance, 1, cv::REDUCE_SUM);
            cv::Mat sortedDist;
            cv::sortIdx(distance, sortedDist, cv::SORT_ASCENDING);

            int smallestIdx = sortedDist.at<int>(0);
            int secondSmallestIdx = sortedDist.at<int>(1);
            float smallestDistance = distance.at<float>(smallestIdx);
            float secondSmallestDistance = distance.at<float>(secondSmallestIdx);
            float ratio = smallestDistance / secondSmallestDistance;

            if (ratio < 0.8) {
                cv::Point2f pt1 = kp1[i].pt;
                cv::Point2f pt2 = kp2[smallestIdx].pt;
                result.push_back(std::make_tuple(pt2, pt1, ratio));
            }
        }

        return result;
    }

    std::vector<std::tuple<cv::Point2f, cv::Point2f, float>> detectAndMatch(
            cv::Mat& img1, cv::Mat& img2, std::string method = "SIFT") {
        cv::Mat gray1, gray2;
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);

        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat des1, des2;

        if (method == "SIFT") {
            sift->detectAndCompute(gray1, cv::noArray(), kp1, des1);
            sift->detectAndCompute(gray2, cv::noArray(), kp2, des2);
        } else if (method == "ORB") {
            orb->detectAndCompute(gray1, cv::noArray(), kp1, des1);
            orb->detectAndCompute(gray2, cv::noArray(), kp2, des2);
        }

        return match(kp1, kp2, des1, des2);
    }

    void drawCircle(cv::Mat& image, std::vector<cv::KeyPoint>& kp) {
        int H = image.rows;
        int W = image.cols;

        for (int i = 0; i < kp.size(); i++) {
            cv::Point2f pt = kp[i].pt;
            cv::circle(image, cv::Point(pt.x, pt.y), static_cast<int>(kp[i].size), cv::Scalar(0, 255, 0), 1);
        }
    }

    std::vector<std::tuple<cv::Point2f, cv::Point2f, float>> filterMatchPoints(std::vector<cv::KeyPoint>& kp1, std::vector<cv::KeyPoint>& kp2,
        cv::Mat& des1, cv::Mat& des2) {
    
        std::vector<std::tuple<cv::Point2f, cv::Point2f, float>> result;

        for (int i = 0; i < des1.rows; i++) {
            float smallestDistance = std::numeric_limits<float>::max();
            float secondSmallestDistance = std::numeric_limits<float>::max();
            int smallestIdx = 0;

            for (int j = 0; j < des2.rows; j++) {
                float distance = cv::norm(des1.row(i), des2.row(j), cv::NORM_L2);

                if (distance < smallestDistance) {
                    secondSmallestDistance = smallestDistance;
                    smallestDistance = distance;
                    smallestIdx = j;
                }
            }

            float ratio = smallestDistance / secondSmallestDistance;

            if (ratio < 0.8) {
                cv::Point2f pt1 = kp1[i].pt;
                cv::Point2f pt2 = kp2[smallestIdx].pt;
                result.push_back(std::make_tuple(pt2, pt1, ratio));
            }
        }

        return result;
    }

    std::vector<std::tuple<cv::Point2f, cv::Point2f, float>> SIFTMatchPoints(cv::Mat& img1, cv::Mat& img2) {
        cv::Mat gray1, gray2;
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

        std::vector<cv::KeyPoint> kp1, kp2;
        cv::Mat des1, des2;

        sift->detectAndCompute(gray1, cv::noArray(), kp1, des1);
        sift->detectAndCompute(gray2, cv::noArray(), kp2, des2);

        return filterMatchPoints(kp1, kp2, des1, des2);
    }

    std::vector<cv::KeyPoint> SIFTMatchPointsSingle(cv::Mat& img1) {
        cv::Mat gray1;
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
        cv::Ptr<cv::SIFT> sift = cv::SIFT::create();

        std::vector<cv::KeyPoint> kp1;
        cv::Mat des1;

        sift->detectAndCompute(gray1, cv::noArray(), kp1, des1);

        return kp1;
    }

    cv::Mat drawCircle(cv::Mat& image, std::vector<cv::KeyPoint>& kp) {
        int H = image.rows;
        int W = image.cols;

        cv::Mat result = image.clone();

        for (int i = 0; i < kp.size(); i++) {
            cv::Point2f pt = kp[i].pt;
            cv::circle(result, cv::Point(static_cast<int>(pt.x), static_cast<int>(pt.y)),
                   static_cast<int>(kp[i].size), cv::Scalar(0, 255, 0), 1);
        }

        return result;
    }

