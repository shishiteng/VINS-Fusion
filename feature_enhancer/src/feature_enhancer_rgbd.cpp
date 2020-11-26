#include <ros/ros.h>

#include <pcl_ros/point_cloud.h>
// #include <pcl/registration/gicp.h>
// #include <pcl/registration/ndt.h>
// #include <pcl_conversions/pcl_conversions.h>

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

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <vector>
#include <thread>

using namespace cv;
using namespace std;

Mat cam_matrix_;
Mat dist_coeff_;
double v_fov_, h_fov_;

ros::Publisher pub_debug_image_, pub_xfeature_;

float getDepth(Mat img, int x, int y)
{
    Mat depth_img = img;
    if (depth_img.type() != CV_32F)
        depth_img.convertTo(depth_img, CV_32F, 1.0);

    // 必须这么访问：cv::Mat.at<float>(row,col)
    float depth = depth_img.at<float>(y, x);
    // printf("%d %d  %.3f\n", x, y, depth);

    // valid range: 0.4m~4m
    if (depth > 4.0 || depth < 0.4 || isnan(depth) || isinf(depth))
        return -1.0;

    return depth;
}

void EnhanceFeatures(Mat image, Mat depth_img, const sensor_msgs::PointCloudConstPtr feature_points)
{
    Mat draw_img(image.size(), CV_8UC3);
    cvtColor(image, draw_img, CV_GRAY2BGR);

    sensor_msgs::PointCloudPtr xfeature_points(new sensor_msgs::PointCloud);
    xfeature_points->header = feature_points->header;
    xfeature_points->header.frame_id = "world";
    sensor_msgs::ChannelFloat32 id_of_point;
    sensor_msgs::ChannelFloat32 u_of_point;
    sensor_msgs::ChannelFloat32 v_of_point;
    sensor_msgs::ChannelFloat32 velocity_x_of_point;
    sensor_msgs::ChannelFloat32 velocity_y_of_point;

    for (unsigned int i = 0; i < feature_points->points.size(); i++)
    {
        float x = feature_points->points[i].x;
        float y = feature_points->points[i].y;
        float z = feature_points->points[i].z;
        float id = feature_points->channels[0].values[i];
        float p_u = feature_points->channels[1].values[i];
        float p_v = feature_points->channels[2].values[i];
        float velocity_x = feature_points->channels[3].values[i];
        float velocity_y = feature_points->channels[4].values[i];

        float depth = getDepth(depth_img, p_u, p_v);
        circle(draw_img, cv::Point2f(p_u, p_v), 1, Scalar(0, 255, 0), 1);

        // 更新点的深度
        if (depth > 1.0 && depth < 3.0)
        {
            z = depth;
            circle(draw_img, cv::Point2f(p_u, p_v), 1, Scalar(0, 0, 255), 1);
            char text[10];
            sprintf(text, "%.1f", depth);
            cv::putText(draw_img, text, cv::Point2f(p_u, p_v), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 255));
        }

        geometry_msgs::Point32 p;
        p.x = x;
        p.y = y;
        p.z = z;
        xfeature_points->points.push_back(p);
        id_of_point.values.push_back(id);
        u_of_point.values.push_back(p_u);
        v_of_point.values.push_back(p_v);
        velocity_x_of_point.values.push_back(velocity_x);
        velocity_y_of_point.values.push_back(velocity_y);
    }

    // 发布增强后的features
    xfeature_points->channels.push_back(id_of_point);
    xfeature_points->channels.push_back(u_of_point);
    xfeature_points->channels.push_back(v_of_point);
    xfeature_points->channels.push_back(velocity_x_of_point);
    xfeature_points->channels.push_back(velocity_y_of_point);
    pub_xfeature_.publish(xfeature_points);

    //发布debug图像
    std_msgs::Header header;
    cv_bridge::CvImage debug_img(header, "bgr8", draw_img);
    pub_debug_image_.publish(debug_img.toImageMsg());

    imshow("debug", draw_img);
    waitKey(3);
}

void Callback(const sensor_msgs::ImageConstPtr &img_msg,
              const sensor_msgs::ImageConstPtr &depth_img_msg,
              const sensor_msgs::PointCloudConstPtr &feature_msg)
{
    cv_bridge::CvImageConstPtr ptr, depth_ptr;
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
    depth_ptr = cv_bridge::toCvCopy(depth_img_msg);

    Mat image = ptr->image;
    Mat depth_image = depth_ptr->image;

    // pcl::PointCloud<pcl::PointXYZ>::Ptr depth_points(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::fromROSMsg(*points_msg.get(), *depth_points.get());

    //
    EnhanceFeatures(image, depth_image, feature_msg);
}

int width_ = 640;
int height_ = 480;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_enhancer");
    ros::NodeHandle nh;

    // 导入相机内参数
#if 0
    // realsense d435i
    double intrinsics[9] = {624.5519731856232, 0, 319.2056827801858,
                            0, 621.1944399796929, 241.12556633733737,
                            0, 0, 1};
    double dist_coeff[4] = {0.1880280371074495, -0.32585125085926375, 0.0005316708069306743, -0.00013569756249956323};
#else
    double intrinsics[9] = {548.5998426668384, 0, 326.7901516013681,
                            0, 552.0481016349838, 241.50581259335883,
                            0, 0, 1};
    double dist_coeff[4] = {0.068708182169908, -0.1251952650118514, -0.0018113507440439055, 0.0069944015513969};
#endif
    cam_matrix_ = Mat(3, 3, CV_64F, intrinsics);
    dist_coeff_ = Mat(4, 1, CV_64F, dist_coeff);
    double min_v = min(intrinsics[2], (double)width_ - intrinsics[2]);
    double min_h = min(intrinsics[5], (double)height_ - intrinsics[5]);
    v_fov_ = atan(min_v / intrinsics[0]) * 57.3 * 2;
    h_fov_ = atan(min_h / intrinsics[4]) * 57.3 * 2;
    cout << "camera intrinsics:\n"
         << cam_matrix_ << endl;
    cout << "distortion ceoffs:\n  "
         << dist_coeff_.t() << endl;
    cout << "v fov: " << v_fov_ << endl;
    cout << "h fov: " << h_fov_ << endl;

    pub_debug_image_ = nh.advertise<sensor_msgs::Image>("/debug_image", 1);
    pub_xfeature_ = nh.advertise<sensor_msgs::PointCloud>("/xfeature", 10);

#if 0
    // realsense d435i
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/color/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> depth_sub(nh, "/camera/depth/color/points", 10);
#else
    // asus xtion pro
    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/rgb/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image", 10);
#endif
    message_filters::Subscriber<sensor_msgs::PointCloud> features_sub(nh, "/feature_tracker/feature", 10);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), image_sub, depth_sub, features_sub);
    sync.registerCallback(boost::bind(&Callback, _1, _2, _3));

    ros::spin();
}
