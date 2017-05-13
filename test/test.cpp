

#include "../src/chull.h"

int main()
{
    Eigen::Matrix<double, 2, 2> points2d;
    chull::PointRn<double, 2> p1, p2;
    p1[0] = 3.0; p1[1] = 1.0;
    p2[0] = 2.0; p2[1] = 2.0;
    points2d.col(0) = p1;
    points2d.col(1) = p2;
    chull::PointRn<double, 2> n2 = chull::get_normal(points2d);
    std::cout << n2 << std::endl;
    Eigen::Matrix<double, 3, 3> points3d;
    chull::PointRn<double, 3> p3, p4, p5;
    p3[0] = 1.0; p3[1] = 0.0; p3[2] = 0.0;
    p4[0] = 0.0; p4[1] = 1.0; p4[2] = 0.0;
    p5[0] = 0.0; p5[1] = 0.0; p5[2] = 1.0;
    points3d.col(0) = p3;
    points3d.col(1) = p4;
    points3d.col(2) = p5;
    chull::PointRn<double, 3> n3 = chull::get_normal(points3d);
    std::cout << n3 << std::endl;

    Eigen::Matrix<double, 4, 4> points4d;
    Eigen::Matrix<double, 4, 3> points4d3;
    Eigen::Matrix<double, 4, 2> points4d2;
    chull::PointRn<double, 4> p41, p42, p43, p44, p45;
    p41[0] = 1.0; p41[1] = 0.0; p41[2] = 0.0; p41[3] = 0.0;
    p42[0] = 0.0; p42[1] = 1.5; p42[2] = 0.5; p42[3] = 0.0;
    p43[0] = 0.2; p43[1] = -1.0; p43[2] = 1.0; p43[3] = 0.0;
    p41[0] = 1.0; p41[1] = 1.0; p41[2] = 1.0; p41[3] = 0.0;
    p41[0] = 1.0; p41[1] = 0.0; p41[2] = 0.0; p41[3] = 1.0;
    points4d3.col(0) = p41;
    points4d3.col(1) = p42;
    points4d3.col(2) = p43;
    points4d2.col(0) = p42;
    points4d2.col(1) = p43;
    std::vector< chull::PointRn<double, 4> > ps = {p41, p42, p43, p44, p45};
    std::vector< int > indices = {1, 2, 0,4};
    chull::SubSpace<double> space(ps, indices, indices.size());
    std::cout << "Subspace:" << std::endl;
    std::cout << "dim: " << space.dim() << std::endl;
    std::cout << "basis : \n"<< space.basis() << std::endl;
    std::cout << "projection: \n"<< space.projection() << std::endl;
    std::cout << "distance sq. (space, p1): "<< space.distance(p41) << std::endl;


    Eigen::Matrix<double, 3, 30> points3d100 = Eigen::Matrix<double, 3, 30>::Random();
    chull::ConvexHull<double> hull(points3d100);
    hull.compute();

    return 0;
}
