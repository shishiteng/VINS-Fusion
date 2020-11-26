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
Eigen::Matrix4f camera_to_lidar_;
double v_fov_;

ros::Publisher pub_projection_image_, pub_debug_image_, pub_xfeature_, pub_xpoints_, pub_xpoints_cut_;

void EnhanceFeatures(pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_points,
                     const sensor_msgs::PointCloudConstPtr feature_points,
                     Mat image,
                     Eigen::Matrix4f c2l)
{
    Eigen::Matrix4f l2c = c2l.inverse();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cut_points(new pcl::PointCloud<pcl::PointXYZ>);
    cut_points->header = lidar_points->header;

    vector<Point2f> image_points;
    vector<Point3f> all_points;
    for (auto &pt : lidar_points->points)
    {
        if (pt.x <= 0)
            continue;

        Eigen::Vector4f pt_l(pt.x, pt.y, pt.z, 1);
        Eigen::Vector4f pt_c = l2c * pt_l;

        double theta = atan(pt_c[0] / (pt_c[2] == 0 ? 1e-5 : pt_c[2])) * 57.3;
        if (pt_c[2] > 0 && abs(theta) < v_fov_ / 2.f)
        {
            all_points.push_back(Point3f(pt_c[0], pt_c[1], pt_c[2]));

            pcl::PointXYZ p;
            p.z = pt_c[0];
            p.x = pt_c[1];
            p.y = pt_c[2];
            cut_points->points.push_back(p);
        }
    }
    cut_points->height = 1;
    cut_points->width = cut_points->points.size();
    sensor_msgs::PointCloud2 outputss;
    pcl::toROSMsg(*cut_points.get(), outputss);
    pub_xpoints_cut_.publish(outputss);

    double zero_data[3] = {0};
    Mat rvec(3, 1, cv::DataType<double>::type, zero_data);
    Mat tvec(3, 1, cv::DataType<double>::type, zero_data);
    projectPoints(all_points, rvec, tvec, cam_matrix_, dist_coeff_, image_points);
    Mat draw_img(image.size(), CV_8UC3);
    cvtColor(image, draw_img, CV_GRAY2BGR);

    // 建立索引与点深度的映射
    pcl::PointCloud<pcl::PointXYZI>::Ptr indesity_points(new pcl::PointCloud<pcl::PointXYZI>);
    map<int, vector<float>> map_points;
    for (unsigned int i = 0; i < image_points.size(); i++)
    {
        cv::Point2f pt = image_points[i];
        cv::Point3f pt2 = all_points[i];
        circle(draw_img, pt, 1, Scalar(255, 255, 0), -1);

        int index = (int)pt.y * image.cols + (int)pt.x;
        map<int, vector<float>>::iterator iter = map_points.find(index);
        vector<float> v;
        if (iter != map_points.end())
            v = iter->second;
        v.push_back(sqrt(pt2.x * pt2.x + pt2.y * pt2.y + pt2.z * pt2.z));
        map_points.insert(pair<int, vector<float>>(index, v));

        // 灰度点云
        pcl::PointXYZI p;
        p.z = pt2.x;
        p.x = pt2.y;
        p.y = pt2.z;
        p.intensity = image.ptr<uchar>((int)pt.x)[(int)pt.y];

        // 把p加入到点云中
        indesity_points->points.push_back(p);
    }
    indesity_points->header = lidar_points->header;
    indesity_points->header.frame_id = "world";
    indesity_points->height = 1;
    indesity_points->width = indesity_points->points.size();
    indesity_points->is_dense = false;
    sensor_msgs::PointCloud2 output;
    pcl::toROSMsg(*indesity_points.get(), output);
    pub_xpoints_.publish(output);

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

        float depth = 0.0;
        int n = 0;
        int w = 0; //窗口大小
        //ComputeMeanDepth;
        for (int i = int(p_u - w); i > 1 && i <= int(p_u + w) && i < image.cols; i++)
            for (int j = int(p_v - w); j > 1 && j <= int(p_v + w) && j < image.rows; j++)
            {
                int index = j * image.cols + i;
                map<int, vector<float>>::iterator iter = map_points.find(index);
                if (iter != map_points.end())
                {
                    vector<float> v = iter->second;
                    for (unsigned int m = 0; m < v.size(); m++)
                    {
                        depth += v[m];
                        n++;
                    }
                }
            }

        circle(draw_img, cv::Point2f(p_u, p_v), 1, Scalar(0, 255, 0), 3);

        // 更新点的深度
        if (depth > 1.0)
        {
            z = depth;
            circle(draw_img, cv::Point2f(p_u, p_v), 1, Scalar(0, 0, 255), 3);
            char text[10];
            sprintf(text, "%.1f", depth / n);
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
    cv_bridge::CvImage projected_image(header, "bgr8", draw_img);
    pub_projection_image_.publish(projected_image.toImageMsg());

    imshow("debug", draw_img);
    waitKey(3);
}

void Callback(const sensor_msgs::ImageConstPtr &img_msg,
              const sensor_msgs::PointCloudConstPtr &feature_msg,
              const sensor_msgs::PointCloud2ConstPtr &points_msg)
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

    pcl::PointCloud<pcl::PointXYZ>::Ptr lidar_points(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*points_msg.get(), *lidar_points.get());

    //
    EnhanceFeatures(lidar_points, feature_msg, image, camera_to_lidar_);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "feature_enhancer");
    ros::NodeHandle nh;

    // 导入相机内参数
    double intrinsics[9] = {529.5672208220217, 0, 323.3005704161258,
                            0, 529.0072659619808, 212.2195288154228,
                            0, 0, 1};
    double dist_coeff[4] = {0.06267162023991868, -0.13788087101000782, 0.0008989379554903053, -0.0005280427124752625};
    cam_matrix_ = Mat(3, 3, CV_64F, intrinsics);
    dist_coeff_ = Mat(4, 1, CV_64F, dist_coeff);
    v_fov_ = atan(intrinsics[2] / intrinsics[0]) * 57.3 * 2;
    cout << "camera intrinsics:\n"
         << cam_matrix_ << endl;
    cout << "distortion ceoffs:\n"
         << dist_coeff_.t() << endl;
    cout << "v fov:\n"
         << v_fov_ << endl;

    // 相机雷达外参
    camera_to_lidar_ << 0.00434982, -0.00334089, 0.999992, 0.0477512,
        -0.999916, -0.013374, 0.00430067, 0.0333444,
        0.0133595, -0.999909, -0.00339832, 0.0532358,
        0, 0, 0, 1;

    pub_debug_image_ = nh.advertise<sensor_msgs::Image>("/debug_image", 1);
    pub_projection_image_ = nh.advertise<sensor_msgs::Image>("/projection_image", 1); //lidar点云投影到图像上
    pub_xfeature_ = nh.advertise<sensor_msgs::PointCloud>("/xfeature", 10);
    pub_xpoints_ = nh.advertise<sensor_msgs::PointCloud2>("/xpoints", 10);
    pub_xpoints_cut_ = nh.advertise<sensor_msgs::PointCloud2>("/sync_scan_cloud_filtered", 10);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/cam0/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud> features_sub(nh, "/feature_tracker/feature", 10);
    message_filters::Subscriber<sensor_msgs::PointCloud2> points_sub(nh, "/rslidar_points", 10);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud, sensor_msgs::PointCloud2> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), image_sub, features_sub, points_sub);
    sync.registerCallback(boost::bind(&Callback, _1, _2, _3));

    ros::spin();
}
