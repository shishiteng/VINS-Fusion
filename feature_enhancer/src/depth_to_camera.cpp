#include <ros/ros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
// #include <pcl/registration/gicp.h>
// #include <pcl/registration/ndt.h>
#include <pcl_conversions/pcl_conversions.h>

// #include <pcl/filters/filter.h>
// #include <pcl/filters/radius_outlier_removal.h>
// #include <pcl/filters/random_sample.h>
// #include <pcl/filters/statistical_outlier_removal.h>
// #include <pcl/filters/voxel_grid.h>

// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>

#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <vector>
#include <thread>

using namespace cv;
using namespace std;

typedef pcl::PointXYZRGB PointT;
// pcl::PointXYZRGB;

Mat cam_matrix_;
Mat dist_coeff_;
Eigen::Matrix4f d2c_;

ros::Publisher pub_projection_image_, pub_depth_image_, pub_points_;

void ProjectPoints2Image(pcl::PointCloud<PointT>::Ptr depth_points, Mat image, Eigen::Matrix4f d2c)
{
    // printf("depth points:%lu\n", depth_points->points.size());

    vector<Point2f> image_points;
    vector<Point3f> all_points;
    for (auto &pt : depth_points->points)
    {
        if (isnan(pt.x) || isnan(pt.y) || isnan(pt.z) ||
            isinf(pt.x) || isinf(pt.y) || isinf(pt.z) ||
            pt.z < 0.2 || pt.z > 6)
        {
            continue;
        }
        Eigen::Vector4f pt_l(pt.x, pt.y, pt.z, 1);
        Eigen::Vector4f pt_c = d2c * pt_l;
        // if (pt_c.z > 0.2 && pt_c.z < 6.0 && fabs(pt_c.x) < 1 && fabs(pt_c.y) < 1)
        all_points.push_back(Point3f(pt_c[0], pt_c[1], pt_c[2]));
    }

    if (all_points.size() <= 0)
    {
        ROS_WARN("no valid depth points!");
        return;
    }

    // printf("all points:%lu\n", all_points.size());

    double zero_data[3] = {0};
    Mat rvec(3, 1, cv::DataType<double>::type, zero_data);
    Mat tvec(3, 1, cv::DataType<double>::type, zero_data);
    projectPoints(all_points, rvec, tvec, cam_matrix_, dist_coeff_, image_points);

    Mat depth_img(image.size(), CV_16UC1);
    Mat draw_img(image.size(), CV_8UC3);
    cvtColor(image, draw_img, CV_GRAY2BGR);
    for (unsigned int i = 0; i < image_points.size(); i++)
    {
        Point2f pt = image_points[i];
        int x = (int)(pt.x + 0.5);
        int y = (int)(pt.y + 0.5);
        // printf("%d: %.3f %.3f\n", i, pt.x, pt.y);
        if (x >= 0 && x < image.cols && y >= 0 && y < image.rows)
        {
            depth_img.at<unsigned short>(y, x) = (unsigned short)(all_points[i].z * 1000.f + 0.5);
            circle(draw_img, pt, 1, Scalar(255, 255, 0), -1);
        }
    }

    // 发布debug信息
    depth_points->header.frame_id = "camera";
    pub_points_.publish(*depth_points);

    std_msgs::Header header;
    pcl_conversions::fromPCL(depth_points->header, header);
    cv_bridge::CvImage projected_image(header, "bgr8", draw_img);
    pub_projection_image_.publish(projected_image.toImageMsg());

    cv_bridge::CvImage depth_image(header, "mono16", depth_img);
    pub_depth_image_.publish(depth_image.toImageMsg());
}

void Callback(const sensor_msgs::ImageConstPtr &img_msg, const sensor_msgs::PointCloud2ConstPtr &depth_points_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
    {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);
    }

    Mat image = ptr->image;
    // Mat depth_image = depth_ptr->image;

    pcl::PointCloud<PointT>::Ptr depth_points(new pcl::PointCloud<PointT>);
    pcl::fromROSMsg(*depth_points_msg.get(), *depth_points);

    // transform points into camera frame
    pcl::PointCloud<PointT>::Ptr cam_points(new pcl::PointCloud<PointT>);
    pcl::transformPointCloud(*depth_points, *cam_points, d2c_);

    // project points to image
    ProjectPoints2Image(cam_points, image, d2c_);
}

int width_ = 640;
int height_ = 480;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_enhancer_rgbd");
    ros::NodeHandle nh;

    // 导入相机内参数: 100t8
    double intrinsics[9] = {529.5672208220217, 0, 323.3005704161258,
                            0, 529.0072659619808, 212.2195288154228,
                            0, 0, 1};
    double dist_coeff[4] = {0.06267162023991868, -0.13788087101000782, 0.0008989379554903053, -0.0005280427124752625};

    Eigen::Matrix4f c2d;
    c2d << 0.999621, 0.0272572, -0.0036097, -0.00504635,
        -0.0274624, 0.996195, -0.0827093, 0.0775292,
        0.00134154, 0.0827772, 0.996567, 0.0216895,
        0, 0, 0, 1;
    d2c_ = c2d.inverse();

    cam_matrix_ = Mat(3, 3, CV_64F, intrinsics);
    dist_coeff_ = Mat(4, 1, CV_64F, dist_coeff);
    cout << "camera intrinsics:\n"
         << cam_matrix_ << endl;
    cout << "distortion ceoffs:\n  "
         << dist_coeff_.t() << endl;
    cout << "depth to camera:\n  "
         << d2c_ << endl;

    pub_projection_image_ = nh.advertise<sensor_msgs::Image>("/projection_image", 1);
    pub_depth_image_ = nh.advertise<sensor_msgs::Image>("/depth_image", 1);
    pub_points_ = nh.advertise<sensor_msgs::PointCloud2>("/depth_points", 1);

    // realsense d435i
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/cam0/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> depth_sub(nh, "/camera/depth/color/points", 10);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(1), image_sub, depth_sub);
    sync.registerCallback(boost::bind(&Callback, _1, _2));

    ros::spin();
}
