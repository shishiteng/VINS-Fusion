#include <ros/ros.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/gicp.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_search.h>

#include <nav_msgs/Path.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <vector>
#include <thread>

using namespace std;

#define BUFFSIZE 128

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloud;

std::vector<PointCloud> point_cloud_buff_;
nav_msgs::Path path_;
double timeshift_ = 0;
double scale_ = 1.0;
Eigen::Vector3d euler_(0, 0, 0);
double top_bound_ = 100.0;
double down_bound_ = -100.0;
double depth_factor_ = 0.001;

PointCloud::Ptr map_data_;
pcl::octree::OctreePointCloudSearch<PointT>::Ptr map_octree_;

cv::Mat cam_matrix_;
cv::Mat dist_coeff_;
cv::Mat depth_cam_matrix_;
Eigen::Matrix4f d2c_;

ros::Publisher pub_points_, pub_octmap_, pub_path_;

Eigen::Matrix4d PosestampedToEigen(geometry_msgs::PoseStamped pose_stamped)
{
    Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
    Eigen::Quaterniond q(pose_stamped.pose.orientation.w,
                         pose_stamped.pose.orientation.x,
                         pose_stamped.pose.orientation.y,
                         pose_stamped.pose.orientation.z);
    Eigen::Vector3d t(pose_stamped.pose.position.x,
                      pose_stamped.pose.position.y,
                      pose_stamped.pose.position.z);

    m.block(0, 0, 3, 3) = q.normalized().toRotationMatrix();
    m.block(0, 3, 3, 1) = t;

    return m;
}

PointCloud image2Pointcloud(cv::Mat dep_img, cv::Mat rgb_img)
{
    double fx = depth_cam_matrix_.at<double>(0);
    double fy = depth_cam_matrix_.at<double>(4);
    double cx = depth_cam_matrix_.at<double>(2);
    double cy = depth_cam_matrix_.at<double>(5);

    ROS_INFO("depth:%dx%d type_%d", dep_img.cols, dep_img.rows, dep_img.type());

    PointCloud points_depth;
    vector<cv::Point2f> points_cam_2d;
    vector<cv::Point3f> points_cam_3d;
    for (int x = 0; x < dep_img.cols; x++)
    {
        for (int y = 0; y < dep_img.rows; y++)
        {
            double d = dep_img.at<unsigned short>(y, x) * depth_factor_;
            double wx = (x - cx) / fx * d;
            double wy = (y - cy) / fy * d;

            if (d < 0.2 || d > 6)
                continue;
            // printf("%d %d-->%lf %lf %lf\n", x, y, wx, wy, d);

            Eigen::Vector4f pt_d(wx, wy, d, 1);
            Eigen::Vector4f pt_c = d2c_ * pt_d;
            points_cam_3d.push_back(cv::Point3f(pt_c[0], pt_c[1], pt_c[2]));
        }
    }
    ROS_INFO("all points 3d: %d", points_cam_3d.size());

    //
    double zero_data[3] = {0};
    cv::Mat rvec(3, 1, cv::DataType<double>::type, zero_data);
    cv::Mat tvec(3, 1, cv::DataType<double>::type, zero_data);
    cv::projectPoints(points_cam_3d, rvec, tvec, cam_matrix_, dist_coeff_, points_cam_2d);

    for (unsigned int i = 0; i < points_cam_2d.size(); i++)
    {
        cv::Point2f pt2d = points_cam_2d[i];
        cv::Point3f pt3d = points_cam_3d[i];
        int x = (int)(pt2d.x + 0.5);
        int y = (int)(pt2d.y + 0.5);
        // printf("%d: %.3f %.3f\n", i, pt.x, pt.y);
        if (x >= 0 && x < dep_img.cols && y >= 0 && y < dep_img.rows)
        {
            unsigned char gray_value = rgb_img.at<unsigned char>(y, x);
            PointT p;
            p.x = pt3d.x;
            p.y = pt3d.y;
            p.z = pt3d.z;
            p.intensity = (float)gray_value;
            points_depth.points.push_back(p);
        }
    }

    points_depth.header.frame_id = "world";
    return points_depth;
}

void filtPoints(PointCloud in, PointCloud &out)
{
    double grid_res = 0.02;
    pcl::VoxelGrid<PointT> grid;
    grid.setLeafSize(grid_res, grid_res, grid_res);
    grid.setInputCloud(in.makeShared());
    grid.filter(out);
}

bool findPose(double timestamp, Eigen::Matrix4d &pose)
{
    // path_
    for (auto &posestamped : path_.poses)
    {
        double t = posestamped.header.stamp.toSec();
        if (fabs(timestamp - t) < 0.003)
        {
            pose = PosestampedToEigen(posestamped);
            Eigen::Matrix4d c2i;
            c2i << 0.99992325, 0.00930136, 0.00818451, -0.00591438,
                0.00917658, -0.99984306, 0.01515436, 0.00450056,
                0.00832418, -0.01507809, -0.99985167, -0.05169857,
                0, 0, 0, 1;
            pose = pose * c2i;
            return true;
        }
    }
    return false;
}

void insertToMap(PointCloud points)
{
    for (size_t ii = 0; ii < points.points.size(); ii++)
    {
        const PointT p = points.points[ii];
        if (p.z > 1.5)
        {
            continue;
        }
        double min_x, min_y, min_z, max_x, max_y, max_z;
        map_octree_->getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
        bool isInBox = (p.x >= min_x && p.x <= max_x) && (p.y >= min_y && p.y <= max_y) && (p.z >= min_z && p.z <= max_z);
        if (!isInBox || !map_octree_->isVoxelOccupiedAtPoint(p))
            map_octree_->addPointToCloud(p, map_data_);
    }
}

void ImageCallback(const sensor_msgs::Image::ConstPtr &img_msg, const sensor_msgs::Image::ConstPtr &depth_img_msg)
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

    cv::Mat image = ptr->image;
    cv::Mat depth_image = depth_ptr->image;

    double t = img_msg->header.stamp.toSec();

    PointCloud points;
    points = image2Pointcloud(depth_image, image);
    ROS_INFO("image to points: %lu", points.points.size());

    PointCloud points_filtered;
    filtPoints(points, points_filtered);
    ROS_INFO("filt points: %lu", points_filtered.points.size());
    // pub_points_.publish(points_filtered);

    Eigen::Matrix4d pose;
    if (findPose(t, pose))
    {
        PointCloud points_world;
        pcl::transformPointCloud(points_filtered, points_world, pose);

        insertToMap(points_world);

        pub_points_.publish(points_world);
        pub_octmap_.publish(*map_data_);
        pub_path_.publish(path_);
    }
}
void saveMap()
{
    ROS_INFO("save map: %lu", map_data_->points.size());
    pcl::io::savePCDFileASCII("map.pcd", *map_data_);
}

void readPath(char *file_name)
{
    printf("lode path from: %s \n", file_name);
    printf("path loading...\n");
    FILE *pFile = fopen(file_name, "r");
    if (pFile == NULL)
    {
        printf("open path failed \n");
        return;
    }
    double t, x, y, z, qw, qx, qy, qz;
    int i = 0;
    int count = 0;
    //index time_stamp Tx Ty Tz Qw Qx Qy Qz
    char buff[1024];
    while (!feof(pFile))
    {
        fgets(buff, sizeof(buff), pFile);
        sscanf(buff, "%d %lf %lf %lf %lf %lf %lf %lf %lf", &i, &t, &x, &y, &z, &qw, &qx, &qy, &qz);
        if (count > 0)
        {
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.frame_id = "world";
            pose_stamped.header.stamp = ros::Time(t);
            pose_stamped.pose.orientation.w = qw;
            pose_stamped.pose.orientation.x = qx;
            pose_stamped.pose.orientation.y = qy;
            pose_stamped.pose.orientation.z = qz;
            pose_stamped.pose.position.x = x;
            pose_stamped.pose.position.y = y;
            pose_stamped.pose.position.z = z;
            path_.poses.push_back(pose_stamped);
            // printf("%d %lf %lf %lf %lf\n", i, t, x, y, z);
        }
        // printf("%s\n", buff);
        memset(buff, 0, sizeof(buff));
        count++;
    }
    fclose(pFile);
    printf("lode path: %d \n", count);

    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = "world";
    pub_path_.publish(path_);
}

void CommandProcess()
{
    char buff[BUFFSIZE] = {0};
    while (1)
    {
        if (NULL != fgets(buff, BUFFSIZE, stdin))
        {
            char c;
            double value;
            sscanf(buff, "%c%lf", &c, &value);
            switch (c)
            {
            case 's':
                saveMap();
                break;
            default:
                break;
            }
        }

        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "mapping");
    ros::NodeHandle nh;

    if (argc != 2)
    {
        printf("mapping [path_file]\n");
        return -1;
    }

    // 导入相机内参数: 100t8
    double intrinsics[9] = {529.5672208220217, 0, 323.3005704161258,
                            0, 529.0072659619808, 212.2195288154228,
                            0, 0, 1};
    double dist_coeff[4] = {0.06267162023991868, -0.13788087101000782, 0.0008989379554903053, -0.0005280427124752625};
    double depth_intrinsics[9] = {462.06640625, 0.0, 344.25,
                                  0.0, 462.71875, 244.271484375,
                                  0.0, 0.0, 1.0};

    Eigen::Matrix4f c2d;
    c2d << 0.999621, 0.0272572, -0.0036097, -0.00504635,
        -0.0274624, 0.996195, -0.0827093, 0.0775292,
        0.00134154, 0.0827772, 0.996567, 0.0216895,
        0, 0, 0, 1;
    d2c_ = c2d.inverse();

    cam_matrix_ = cv::Mat(3, 3, CV_64F, intrinsics);
    dist_coeff_ = cv::Mat(4, 1, CV_64F, dist_coeff);
    depth_cam_matrix_ = cv::Mat(3, 3, CV_64F, depth_intrinsics);
    cout << "camera intrinsics:\n"
         << cam_matrix_ << endl;
    cout << "distortion ceoffs:\n  "
         << dist_coeff_.t() << endl;
    cout << "depth_camera intrinsics:\n"
         << depth_cam_matrix_ << endl;
    cout << "depth to camera:\n  "
         << d2c_ << endl;

    map_data_.reset(new PointCloud);
    map_data_->header.frame_id = "world";
    map_octree_.reset(new pcl::octree::OctreePointCloudSearch<PointT>(0.02)); //octree_resolution:0.05
    map_octree_->setInputCloud(map_data_);

    message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/cam0/image_raw", 10);
    message_filters::Subscriber<sensor_msgs::Image> depth_sub(nh, "/camera/depth/image_rect_raw", 10);
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), image_sub, depth_sub);
    sync.registerCallback(boost::bind(&ImageCallback, _1, _2));

    pub_path_ = nh.advertise<nav_msgs::Path>("/mapping/path", 1);
    pub_points_ = nh.advertise<sensor_msgs::PointCloud2>("/mapping/pointcloud", 1);
    pub_octmap_ = nh.advertise<sensor_msgs::PointCloud2>("/mapping/oct_map", 1);

    readPath(argv[1]);

    std::thread keyboard_command_process = std::thread(CommandProcess);

    ros::spin();
}
