#pragma once

#include <memory>
#include <vector>
#include <string>
#include <queue>
#include <set>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <functional>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Geometry>

namespace chull{

enum copy_type{shallow, deep};

template<class P>
constexpr P zero() { return P(0.0); }

template<class P>
constexpr P tolerance() { return P(1e-5); } // TODO: set externally?

using Eigen::Dynamic;
template<class P, int D=Dynamic>
using PointRn = Eigen::Matrix<P, D, 1>;

template<class P, int D=Dynamic>
using PlaneRn = Eigen::Hyperplane<P, D>;

template<class P, int D=Dynamic>
using VectorRn = PointRn<P,D>;

template<class IndexIterator>
inline std::vector<int> __make_index_vector(IndexIterator first, IndexIterator last, int N)
{
    if(first != last)
        return std::vector<int>(first, last);
    std::vector<int> v(N);
    for(int i = 0; i < N; i ++) v[i] = i;
    return v;
}

struct dim{
    const unsigned int value;
    template<class P, int D1, int D2>
    dim(const Eigen::Matrix<P, D1, D2>& m) : value((D1 > 0) ? D1 : m.rows()) { }
    template<class P, int D>
    dim(const Eigen::Hyperplane<P, D>& p) : value((D > 0) ? D : p.dim()) { }
};


template<class P>
struct support_function // TODO: optimize this by removing expensive vector copy operations!
{
    typedef typename Eigen::Matrix<P, Eigen::Dynamic, Eigen::Dynamic > MatrixType;

    typedef typename MatrixType::Index IndexType;

    support_function(const MatrixType& points) : verts(points), m_Indices(__make_index_vector(0,0, points.rows()))
    {
    }

    template<class VectorType, class IteratorType>
    support_function(const std::vector< VectorType > & points, IteratorType first=0, IteratorType last=0)
        : m_Indices(__make_index_vector(first, last, points.size()))
    {
        verts.resize(m_Indices.size(), points[0].rows());
        for(IndexType i = 0; i < m_Indices.size(); i++)
            verts.row(i) = points[m_Indices[i]];
    }

    P operator () (const PointRn<P>& v)
    {
        // h(P, v) = sup { a . v : a in P }
        Eigen::VectorXd dots = verts*v;
        return dots.maxCoeff();
    }

    IndexType index(const PointRn<P>& v, bool bMap=true)
    {
        Eigen::VectorXd dots = verts*v;
        P maxv = dots[0];
        IndexType maxi = 0;
        for(IndexType i=1; i < dots.rows(); i++)
        {
            if(dots[i] > maxv)
            {
                maxv = dots[i];
                maxi = i;
            }
        }

        return bMap ? m_Indices[maxi] : maxi;
    }

    PointRn<P> element(const PointRn<P>& v) { return PointRn<P>(verts.row(index(v, false))); }

    MatrixType verts;
    std::vector<int> m_Indices;
};

template<class P, int D=3> // Here D is the ambient space dimension
struct counter_clockwise
{
    counter_clockwise(const PointRn<P,D>& ref, const VectorRn<P,D>& n) : reference(as3D(ref)), normal(as3D(n)), tol(tolerance<P>())
    {
        static_assert(D == 2 || D == 3 || D == Dynamic, "Counter Clockwise is only defined for dimemsions 2 and 3.");
        if(reference.rows() > 3)
            throw std::runtime_error("Counter Clockwise is only defined for dimemsions 2 and 3.");
        normal.normalize();
    }
    // returns true if point p1's polar angle is less than p2's polar angle from
    // a given a reference point in the plane normal vector.
    bool less2(const PointRn<P,3>& a, const PointRn<P,3>& b) const
    {
        Eigen::Vector3d va, vb, vab;
        vab = a-b;
        va = a - reference;
        vb = b - reference;
        if(vab.squaredNorm() < tol)
            return false;
        return normal.dot(va.cross(vb)) > 0;
    }
    // Container must have the [] operator overloaded
    template<class IndexType, class Container>
    bool less(const IndexType& a, const IndexType& b, const Container& pointset) const
    {
        VectorRn<P,3> va, vb;
        va = as3D(pointset[a]) - reference;
        vb = as3D(pointset[b]) - reference;
        if(a == b)
            return false;
        else if(va.squaredNorm() < tol)
            return true;
        else if(vb.squaredNorm() < tol)
            return false;
        return normal.dot(va.cross(vb)) > 0;
    }

    template<class IndexType, class Container>
    std::function<bool(const IndexType& , const IndexType&) > lessfn(const Container& points) const
    {
        return [this, points](const IndexType& a, const IndexType& b) {
            return this->less(a, b, points);
        };
    }

    // returns true if a->b->c makes a left turn, false otherwise (colinear or right turn)
    bool operator()(const PointRn<P,D>& a, const PointRn<P,D>& b, const PointRn<P,D>& c) const
    {
        // VectorRn<P,3> e1 = {}
        // VectorRn<P,3> e2  = {}
        VectorRn<P,3> vb = as3D(b);
        VectorRn<P,3> va, vc;
        va = as3D(a) - vb;
        vc = as3D(c) - vb;
        // std::cout << "a = \n"<< a << std::endl<< std::endl;
        // std::cout << "b = \n"<< b << std::endl<< std::endl;
        // std::cout << "c = \n"<< c << std::endl<< std::endl;
        //
        // std::cout << "va = \n"<< va << std::endl<< std::endl;
        // std::cout << "vc = \n"<< vc << std::endl<< std::endl;
        // std::cout << "n = \n"<< normal << std::endl<< std::endl;
        // std::cout << "vc x va = \n"<< vc.cross(va) << std::endl<< std::endl;
        // std::cout << "n * (vc x va) = \n"<< normal.dot(vc.cross(va)) << std::endl<< std::endl;
        // std::cout << "e1 x e2 = \n"<< as3D<3>({-1.0, 0.0, 0.0}).cross(as3D<3>({0.0, -1.0, 0.0})) << std::endl<< std::endl;
        return normal.dot(vc.cross(va)) > 0;
    }

    template<int Do>
    VectorRn<P,3> as3D(const VectorRn<P,Do> v) const
    {
        if(v.rows() == 3)
            return v;
        else if (v.rows() == 2)
            return {v[0], v[1], P(0.0)};
        else
            return {v[0], v[1], v[2]};
    }

    PointRn<P,3>    reference;
    VectorRn<P,3>   normal;
    P tol;
};

template<class P, int D=Dynamic> // Here D is the ambient space dimension
class SubSpace // represents the span of a set of basis vectors.
{
    typedef typename Eigen::Matrix<P, Eigen::Dynamic, Eigen::Dynamic > MatrixType;
public:
    template<int D2> // ArrayType must have the accessor [] overloaded.
    SubSpace(const Eigen::Matrix<P, D, D2>& basis, P threshold = tolerance<P>())
    {
        from_basis(basis, threshold);
    }

    template<class ArrayType, int D2> // ArrayType must have the accessor [] overloaded.
    SubSpace(const std::vector< VectorRn<P,D2> >& points, const ArrayType& indices, unsigned int N, P threshold = tolerance<P>())
    {
        int d = points[indices[0]].rows();
        m_origin = points[indices[0]];
        MatrixType basis(d, int(N)-1);
        for(int i = 1; i < N; i++)
        {
            basis.col(i-1) = points[indices[i]] - m_origin;
        }
        from_basis(basis, threshold);
    }

    ~SubSpace() {}

    bool in(const PointRn<P, D>& p, P threshold = tolerance<P>()) { return fabs(distance(p, true)) < threshold; }

    template<int D2>
    PointRn<P, D> project(const PointRn<P, D2>& point) const
    {
        PointRn<P, D> proj, x0;
        x0 = point - m_origin;
        proj = m_Projection*x0;
        return proj + m_origin;
    }

    template<int D2>
    P distance(const PointRn<P, D2>& point, bool squared=true) const
    {
        PointRn<P, D> proj = project(point);
        VectorRn<P,D> dv = point - proj;
        P d2 = dv.dot(dv);
        return squared ? d2 : sqrt(d2);
    }

    const size_t& dim() const { return m_dim; }

    const MatrixType& basis() const { return m_Basis; }

    const MatrixType& projection() const { return m_Projection; }

    const PointRn<P, D>& origin() const { return m_origin; }

private:

    void from_basis(const MatrixType& basis, P threshold = tolerance<P>())
    {
        Eigen::FullPivLU< MatrixType > lu(basis.rows(), basis.cols());
        lu.setThreshold(threshold);
        lu.compute(basis);
        m_Basis = lu.image(basis);
        if(m_Basis.isZero(threshold))
        {
            std::cerr << "Basis dimension is zero!" << std::endl;
            std::cerr << "input: \n" << basis << std::endl;
            std::cerr << "output: \n" << m_Basis << std::endl;
            throw std::runtime_error("Could not initialize subspace.");
        }
        m_dim = m_Basis.cols(); // number of basis vectors.
        // now solve for the projection map.
        // P = A(A^T A)^{−1} A^T. where A is the basis.
        MatrixType bt = m_Basis.transpose();
        MatrixType btbinv = (bt*m_Basis).inverse();
        m_Projection = m_Basis * btbinv * bt;
    }

private:
    size_t          m_dim;
    MatrixType      m_Basis;
    MatrixType      m_Projection;
    PointRn<P, D>   m_origin;
};

template<class P, int D=Eigen::Dynamic>
class HalfSpace
{
public:
    HalfSpace() {}
    ~HalfSpace() {}
private:
    PlaneRn<P, D> m_plane;
};

// may move some of these functions inside the half space class.
template<class P, int D, int N>
inline VectorRn<P,D> get_normal(const Eigen::Matrix<P, D, N>& points)
{
    VectorRn<P,D> ret;
    Eigen::Matrix<P, Eigen::Dynamic, Eigen::Dynamic> space(points.cols()-1, points.rows());

    if(points.cols() == 1)
    {
        ret = VectorRn<P,D>(points.col(0));
        ret.normalize();
        return ret;
    }
    for(int i = 1; i < points.cols(); i++) // TODO: I am not happy about this copy. can we remove?
        space.row(i-1) = points.col(i) - points.col(0);
    Eigen::FullPivLU< Eigen::Matrix<P, Eigen::Dynamic, Eigen::Dynamic > > lu(space);
    Eigen::Matrix<P, Eigen::Dynamic, Eigen::Dynamic> null_space = lu.kernel();
    // Check the dimension is one.

    if(null_space.cols() != 1)
    {
        std::cout << "points : " << std::endl;
        std::cout << points << std::endl;
        std::cout << "kernel : " << std::endl;
        std::cout << null_space << std::endl;
        throw(std::runtime_error("space is either under or over specified."));
    }
    ret = null_space;
    ret.normalize();
    return ret;
}

template< class P, int D, class MatType, class IndexType>
inline void slice(const std::vector< VectorRn<P,D> >& points, const std::vector<IndexType>& indices, MatType& mat)
{
    for(size_t i = 0; i < indices.size(); i++)
        mat.col(i) = points[indices[i]];
}

template< class P, int D, class IndexType, class MatType >
inline void slicen(const std::vector< VectorRn<P,D> >& points, IndexType indices, size_t n, MatType& mat)
{
    assert(mat.cols() == n);
    for(size_t i = 0; i < n; i++)
        mat.col(i) = points[indices[i]];
}


// face is assumed to be an array of indices of triangular face of a convex body.
// points may contain points inside or outside the body defined by faces.
// faces may include faces that contain vertices that are inside the body.
// TODO: replace this as a constructor for the HalfSpace above?
template<class P, int D, class IndexType>
inline VectorRn<P,D> getOutwardNormal(const std::vector< VectorRn<P,D> >& points, const VectorRn<P,D>& inside_point, const std::vector<IndexType>& face)
{
    // const std::vector<unsigned int>& face = faces[faceid];
    unsigned int d = dim(inside_point).value;
    assert(d <= face.size()); // TODO: throw error? 
    Eigen::Matrix<P, D, D> facet(d,d);
    slicen(points, face, d, facet);
    VectorRn<P,D> di = inside_point - points[face[0]];
    VectorRn<P,D> normal = get_normal(facet);
    P x = normal.dot(di);
    if(fabs(d) < tolerance<P>())
        throw(std::runtime_error("inner point is in the plane."));
    return (x > 0) ? -normal : normal;
}


template<class P, int D = Dynamic>
class ConvexHull
{
    const unsigned int invalid_index=-1;
public:
    template<class VectorType>
    ConvexHull(const std::vector< VectorType >& points)
    {
        m_points.resize(points.size());
        for(int i = 0; i < points.size(); i++){
            m_points[i] = points[i]; // copies the points.
            // std::cout << m_points[i] <<std::endl;
        }
        assert(points.size() > 0);
        m_dim = dim(m_points[0]).value;
        m_ravg = VectorRn<P,Eigen::Dynamic>::Zero(m_dim);
    }

    template<class MatrixType>
    ConvexHull(const MatrixType& points)
    {
        m_dim = dim(points).value;
        m_ravg = VectorRn<P,Eigen::Dynamic>::Zero(m_dim);
        m_points.reserve(points.cols());
        for(int i = 0; i < points.cols(); i++)
        {
            m_points.push_back(points.col(i));
        }
    }

    void compute()
    {
        std::vector<bool> inside(m_points.size(), false); // all points are outside.
        std::vector<unsigned int> outside(m_points.size(), invalid_index);
        try{
        if(m_points.size() < m_dim+1) // problem is not well posed.
            return;

        // step 1: create a tetrahedron from the first 4 points.

        initialize(); // makes the tetrahedron
        for(unsigned int i = 0; i < m_points.size(); i++) // O((dim+1)*N) since we have a tetrahedron
        {
            // step 2: initialize the outside and inside sets
            for(unsigned int f = 0; f < m_faces.size() && !inside[i]; f++)
            {
                if(m_deleted[f])
                    continue;
                if(outside[i] == invalid_index && is_above(i,f))
                {
                    outside[i] = f;
                    break;
                }
            }
            if(!inside[i] && outside[i] == invalid_index)
            {
                inside[i] = true;
            }
        }

        unsigned int faceid = 0;
        // write_pos_frame(inside);

        while(faceid < m_faces.size()) //
        {
            if(m_deleted[faceid]) // this facet is deleted so we can skip it.
            {
                faceid++;
                continue;
            }

            P dist = 0.0;
            unsigned int _id = invalid_index;
            for(unsigned int out = 0; out < outside.size(); out++) // O(N)
            {
                if(outside[out] == faceid)
                {
                    P sd = signed_distance(out, faceid);
                    assert(sd > tolerance<P>());
                    if( sd > dist)
                    {
                        dist = sd;
                        _id = out;
                    }
                }
            }
            if(_id == invalid_index) // no point found.
            {
                faceid++;
                continue;
            }

            // step 3: Find the visible set
            std::vector< unsigned int > visible;
            build_visible_set(_id, faceid, visible);
            // step 4: Build the new faces
            std::vector< std::vector<unsigned int> > new_faces;
            build_horizon_set(visible, new_faces); // boundary of the visible set
            assert(visible[0] == faceid);
            for(unsigned int dd = 0; dd < visible.size(); dd++)
                {
                m_deleted[visible[dd]] = true;
                }

            for(unsigned int i = 0; i < new_faces.size(); i++)
                {
                new_faces[i].push_back(_id);
                std::sort(new_faces[i].begin(), new_faces[i].end());
                unsigned int fnew = m_faces.size();
                assert(m_normals.size() == m_faces.size());
                m_faces.push_back(new_faces[i]);
                m_normals.push_back(getOutwardNormal(m_points, m_ravg, m_faces[fnew]));
                m_deleted.push_back(false);
                build_adjacency_for_face(fnew);
                for(unsigned int out = 0; out < outside.size(); out++)
                    {
                    for(unsigned int v = 0; v < visible.size() && !inside[out]; v++)
                        {
                        if(outside[out] == visible[v])
                            {
                            if(is_above(out, fnew))
                                {
                                outside[out] = fnew;
                                }
                            break;
                            }
                        }
                    }
                }
            // update the inside set for fun.
            for(unsigned int out = 0; out < outside.size(); out++)
                {
                if(outside[out] != invalid_index && m_deleted[outside[out]])
                    {
                    outside[out] = invalid_index;
                    inside[out] = true;
                    }
                }
            inside[_id] = true;
            assert(m_deleted.size() == m_faces.size() && m_faces.size() == m_adjacency.size());
            faceid++;
            // write_pos_frame(inside);
            }
            #ifndef NDEBUG //TODO: remove in a bit. here for extra debug and its easy to turn off.
            for(size_t i = 0; i < m_faces.size(); i++)
            {
                if(m_deleted[i]) continue;
                for(size_t j = i+1; j < m_faces.size(); j++)
                {
                    if(m_deleted[j]) continue;
                    for(size_t k = 0; k < m_faces[j].size(); k++)
                    {
                        if(signed_distance(m_faces[j][k], i) > 0.1) //  Note this is a large tolerance but this sort of check is prone to numerical errors
                        {
                            std::cout << "ERROR!!! point " << m_faces[j][k] << ": [\n" << m_points[m_faces[j][k]] << "]" << std::endl
                                      << "         is above face " << i << ": [" << m_faces[i][0] << ", " << m_faces[i][1] << ", . . . ]" << std::endl
                                      << "         from the face " << j << ": [" << m_faces[j][0] << ", " << m_faces[j][1] << ", . . . ]" << std::endl
                                      << "         distance is " << signed_distance(m_faces[j][k], i) << ", " <<   signed_distance(m_faces[j][k], j) << std::endl
                                      << "         inside is " << std::boolalpha << inside[m_faces[j][k]] << std::endl;
                            throw std::runtime_error("ERROR in ConvexHull::compute() !");
                        }
                    }
                }
            }
            #endif
        remove_deleted_faces(); // actually remove the deleted faces.
        // build_edge_list();
        // sortFaces(m_points, m_faces, zero);
        }
        catch(std::runtime_error e){
            write_pos_frame(inside);
            throw(e);
        }
    }

private:
    void write_pos_frame(const std::vector<bool>& inside)
    {
        if(m_dim != 3) // todo, make it work with 2d as well
            return;

        std::ofstream file("convex_hull.pos", std::ios_base::out | std::ios_base::app);
        std::string inside_sphere  = "def In \"sphere 0.1 005F5F5F\"";
        std::string outside_sphere  = "def Out \"sphere 0.1 00FF5F5F\"";
        std::string avg_sphere  = "def avg \"sphere 0.2 00981C1D\"";
        std::stringstream ss, connections;
        std::set<unsigned int> verts;
        for(size_t f = 0; f < m_faces.size(); f++)
            {
            if(m_deleted[f]) continue;
            verts.insert(m_faces[f].begin(), m_faces[f].end());
            for(size_t k = 0; k < 3; k++)
                connections << "connection 0.05 005F5FFF "<< m_points[m_faces[f][k]][0] << " "<< m_points[m_faces[f][k]][1] << " "<< m_points[m_faces[f][k]][2] << " "
                                                          << m_points[m_faces[f][(k+1)%3]][0] << " "<< m_points[m_faces[f][(k+1)%3]][1] << " "<< m_points[m_faces[f][(k+1)%3]][2] << std::endl;
            }
        ss << "def hull \"poly3d " << verts.size() << " ";
        for(std::set<unsigned int>::iterator iter = verts.begin(); iter != verts.end(); iter++)
            for(int d = 0; d < m_dim; d++)
                ss << m_points[*iter][d] << " ";
        ss << "505984FF\"";
        std::string hull  = ss.str();

        // file<< "boxMatrix 10 0 0 0 10 0 0 0 10" << std::endl;
        file<< inside_sphere << std::endl;
        file<< outside_sphere << std::endl;
        file<< avg_sphere << std::endl;
        // file<< hull << std::endl;
        // file << "hull 0 0 0 1 0 0 0" << std::endl;
        file << connections.str();
        file << "avg ";
        for(int d = 0; d < m_dim; d++)
            file<< m_ravg[d] << " ";
        file << std::endl;
        for(size_t i = 0; i < m_points.size(); i++)
            {
            if(inside[i])
                file << "In ";
            else
                file << "Out ";
            for(int d = 0; d < m_dim; d++)
                file << m_points[i][d] << " ";
            file << std::endl;
            }
        file << "eof" << std::endl;
        }

    P signed_distance(const unsigned int& i, const unsigned int& faceid) const
    {
        VectorRn<P,D> dx = m_points[i] -  m_points[m_faces[faceid][0]];
        return dx.dot(m_normals[faceid]); // signed distance. either in the plane or outside.
    }

    bool is_above(const unsigned int& i, const unsigned int& faceid) const { return (signed_distance(i, faceid) > tolerance<P>()); }

    template<class ArrayType>
    bool is_coplanar(const ArrayType& indices, size_t n)
    {
        for(size_t i = 1; i < n; i++)
        {
            if(indices[0] == indices[i])
            {
                return true;
            }
        }
        SubSpace<P, D> space(m_points, indices, n-1);
        P dist = space.distance(m_points[indices[n-1]]);
        // std::cout << "is_coplanar: " << dist << " <= " << tolerance<P>() << std::boolalpha << (fabs(dist) <= tolerance<P>()) << std::endl;
        // TODO: Note I have seen that this will often times return false for nearly coplanar points!!
        //       How do we choose a good threshold. (an absolute threshold is not good)
        return dist <= tolerance<P>(); //fabs(d) <= zero;
    }

    void edges_from_face(const unsigned int& faceid, std::vector< std::vector<unsigned int> >& edges)
    {
        assert(faceid < m_faces.size());
        unsigned int N = m_faces[faceid].size();
        assert(N == 3);
        assert(!m_deleted[faceid]);
        for(unsigned int i = 0; i < m_faces[faceid].size(); i++)
        {
            std::vector<unsigned int> edge;
            unsigned int e1 = m_faces[faceid][i], e2 = m_faces[faceid][(i+1) % N];
            assert(e1 < m_points.size() && e2 < m_points.size());
            edge.push_back(std::min(e1, e2));
            edge.push_back(std::max(e1, e2));
            edges.push_back(edge);
        }
    }

    unsigned int farthest_point_point(const unsigned int& a, bool greater=false)
        {
        unsigned int ndx = a;
        P maxdsq = 0.0;
        for(unsigned int p = greater ? a+1 : 0; p < m_points.size(); p++)
            {
            VectorRn<P,D> dr = m_points[p] - m_points[a];
            P distsq = dr.dot(dr);
            if(distsq > maxdsq)
                {
                ndx = p;
                maxdsq = distsq;
                }
            }
        return ndx;
        }
/*
    unsigned int farthest_point_line(const unsigned int& a, const unsigned int& b)
        {
        unsigned int ndx = a;
        P maxdsq = 0.0, denom = 0.0;
        const VectorRn<P,D>& x1 = m_points[a],x2 = m_points[b];
        VectorRn<P,D> x3;
        x3 = x2-x1;
        denom = dot(x3,x3);
        if(denom <= zero)
            return a;
        for(unsigned int p = 0; p < m_points.size(); p++)
            {
            if( p == a || p == b)
                continue;
            const VectorRn<P,D>& x0 = m_points[p];
            VectorRn<P,D> cr = cross(x3,x1-x0);
            P numer = dot(cr,cr), distsq;
            distsq = numer/denom;
            if(distsq > maxdsq)
                {
                ndx = p;
                maxdsq = distsq;
                }
            }
        return ndx;
        }

    unsigned int farthest_point_plane(const unsigned int& a, const unsigned int& b, const unsigned int& c)
    {
        unsigned int ndx = a;
        P maxd = 0.0, denom = 0.0;
        const VectorRn<P,D>& x1 = m_points[a],x2 = m_points[b], x3 = m_points[c];
        VectorRn<P,D> n;
        n = cross(x2-x1, x3-x1);
        denom = dot(n,n);
        if(denom <= zero)
            return a;
        normalize_inplace(n);
        for(unsigned int p = 0; p < m_points.size(); p++)
            {
            if(p == a || p == b || p == c)
                continue;
            const VectorRn<P,D>& x0 = m_points[p];
            P dist = fabs(dot(n,x0-x1));
            if(dist > maxd)
                {
                ndx = p;
                maxd = dist;
                }
            }
        return ndx;
    }
*/
    template<class ArrayType>
    unsigned int farthest_point_subspace(const ArrayType& indices, size_t n)
    {
        unsigned int ndx = indices[0];
        P maxd = 0.0;
        SubSpace<P, D> space(m_points, indices, n);
        for(unsigned int p = 0; p < m_points.size(); p++)
        {
            P dist = space.distance(m_points[p]);
            if(dist > maxd)
            {
                ndx = p;
                maxd = dist;
            }
        }
        return ndx;
    }

    void initialize()
    {
        const unsigned int Nsym = m_dim+1; // number of points in the simplex.
        std::vector< unsigned int > ik(Nsym);
        for(size_t d = 0; d < Nsym; d++) ik[d] = invalid_index;
        m_faces.clear(); m_faces.reserve(100000);
        m_deleted.clear(); m_deleted.reserve(100000);
        m_adjacency.clear(); m_adjacency.reserve(100000);

        if(m_points.size() < Nsym) // TODO: the problem is basically done. but need to set up the data structures and return. not common in our use case so put it off until later.
        {
            throw(std::runtime_error("Could not initialize ConvexHull: need n+1 points to take the convex hull in nD"));
        }

        ik[0] = 0; // always use the first point.
        bool coplanar = true;
        while( coplanar )
        {
            ik[1] = farthest_point_point(ik[0], true); // will only search for points with a higher index than ik[0].

            for(size_t d = 1; d < Nsym-1; d++)
            {
                if(ik[d] == ik[0])
                    break;
                ik[d+1] = farthest_point_subspace(ik, d+1); // will only search for points with a higher index than ik[0].
            }

            // ik[2] = farthest_point_line(ik[0], ik[1]);
            // ik[3] = farthest_point_plane(ik[0], ik[1], ik[2]);
            if(!is_coplanar(ik, Nsym))
            {
                coplanar = false;
            }
            else
            {
                ik[0]++;

                for(size_t d = 1; d < Nsym; d++) ik[d] = invalid_index;
                if( ik[0] >= m_points.size() ) // tried all of the points and this will not.
                {
                    ik[0] = invalid_index; // exit loop and throw an error.
                    coplanar = false;
                }
            }
        }
        if(std::find(ik.begin(), ik.end(), invalid_index) != ik.end())
        {
            std::cerr << std::endl << std::endl<< "*************************" << std::endl;
            for(size_t i = 0; i < m_points.size(); i++)
            {
                std::cerr << "point " << i << ": [ \n" << m_points[i] << "]" << std::endl;
            }
            throw(std::runtime_error("Could not initialize ConvexHull: found only nearly coplanar points"));
        }
        m_ravg = VectorRn<P,D>::Zero(m_dim);
        for(size_t i = 0; i < Nsym; i++)
        {
            m_ravg += m_points[ik[i]];
        }
        m_ravg /= P(Nsym);

        std::vector<unsigned int> face(m_dim);
        assert(m_dim == 3); // I think in any dimension, we must enumerate the C(d+1, d) combinations for each face.
                            // This shouldn't be hard but I will do it later.
                            // TODO: is there anywhere else in the code where I assume 3d?
        // face 0
        face[0] = ik[0]; face[1] = ik[1]; face[2] = ik[2];
        std::sort(face.begin(), face.end());
        m_faces.push_back(face);
        // face 1
        face[0] = ik[0]; face[1] = ik[1]; face[2] = ik[3];
        std::sort(face.begin(), face.end());
        m_faces.push_back(face);
        // face 2
        face[0] = ik[0]; face[1] = ik[2]; face[2] = ik[3];
        std::sort(face.begin(), face.end());
        m_faces.push_back(face);
        // face 3
        face[0] = ik[1]; face[1] = ik[2]; face[2] = ik[3];
        std::sort(face.begin(), face.end());
        m_faces.push_back(face);
        m_deleted.resize(4, false); // we have 4 facets at this point.
        build_adjacency_for_face(0);
        build_adjacency_for_face(1);
        build_adjacency_for_face(2);
        build_adjacency_for_face(3);
        // std::cout << "initializing normals" << std::endl;
        for(int f = 0; f < m_faces.size(); f++){
            VectorRn<P, D> n = getOutwardNormal(m_points, m_ravg, m_faces[f]);
            // std::cout << "normal "<< f << ": \n" << n << std::endl;
            m_normals.push_back(n);
        }
        // std::cout << "done" << std::endl;
    }

    void build_adjacency_for_face(const unsigned int& f)
        {
        assert(m_dim == 3); // This assumes 3d!
        if(f >= m_faces.size())
            throw std::runtime_error("index out of range!");
        if(m_deleted[f]) return; // don't do anything there.

        m_adjacency.resize(m_faces.size());
        for(unsigned int g = 0; g < m_faces.size(); g++)
            {
            if(m_deleted[g] || g == f) continue;
            // note this is why we need the faces to be sorted here.
            std::vector<unsigned int> intersection(3, 0);
            assert(m_faces[f].size() == 3 && m_faces[g].size() == 3);
            std::vector<unsigned int>::iterator it = std::set_intersection(m_faces[f].begin(), m_faces[f].end(), m_faces[g].begin(), m_faces[g].end(), intersection.begin());
            intersection.resize(it-intersection.begin());
            if(intersection.size() == 2)
                {
                m_adjacency[f].insert(g);  // always insert both ways
                m_adjacency[g].insert(f);
                }
            }
        }

    void build_visible_set(const unsigned int& pointid, const unsigned int& faceid, std::vector< unsigned int >& visible)
        {
        // std::cout << "building visible set point: "<< pointid << " face: " << faceid << std::endl;
        assert(m_dim == 3); // This assumes 3d!
        visible.clear();
        visible.push_back(faceid);
        std::queue<unsigned int> worklist;
        std::vector<bool> found(m_deleted);
        worklist.push(faceid);
        found[faceid] = true;
        // std::cout << "point id: " << pointid << ", face id: " << faceid << std::endl;
        while(!worklist.empty())
            {
            unsigned int f = worklist.front();
            worklist.pop();
            // std::cout << "face " << f << ": " << "[ " << m_faces[f][0] << ", " << m_faces[f][1] << ", " << m_faces[f][2] << "] "<< std::endl;
            if(m_deleted[f]) continue;
            // std::cout << " m_adjacency.size = "<< m_adjacency.size() << " m_adjacency["<<f<<"].size = "<< m_adjacency[f].size() << std::endl;
            for(std::set<unsigned int>::iterator i = m_adjacency[f].begin(); i != m_adjacency[f].end(); i++)
                {
                // std::cout << "found: " << found[*i] << " - neighbor "<< *i << ": " << "[ " << m_faces[*i][0] << ", " << m_faces[*i][1] << ", " << m_faces[*i][2] << "] "<< std::endl;
                if(!found[*i]) // face was not found yet and the point is above the face.
                    {
                    found[*i] = true;
                    if( is_above(pointid, *i) )
                        {
                        assert(!m_deleted[*i]);
                        worklist.push(*i);
                        visible.push_back(*i);
                        }
                    }
                }
            }
        }

    void build_horizon_set(const std::vector< unsigned int >& visible, std::vector< std::vector<unsigned int> >& horizon)
        {
        assert(m_dim == 3); // This assumes 3d!
        std::vector< std::vector<unsigned int> > edges;
        for(unsigned int i = 0; i < visible.size(); i++)
            edges_from_face(visible[i], edges); // all visible edges.
        std::vector<bool> unique(edges.size(), true);
        for(unsigned int i = 0; i < edges.size(); i++)
            {
            for(unsigned int j = i+1; j < edges.size() && unique[i]; j++)
                {
                if( (edges[i][0] == edges[j][0] && edges[i][1] == edges[j][1]) ||
                    (edges[i][1] == edges[j][0] && edges[i][0] == edges[j][1]) )
                    {
                    unique[i] = false;
                    unique[j] = false;
                    }
                }
            if(unique[i])
                {
                horizon.push_back(edges[i]);
                }
            }
        }

    // void build_edge_list()
    //     {
    //     return;
    //     std::vector< std::vector<unsigned int> > edges;
    //     for(unsigned int i = 0; i < m_faces.size(); i++)
    //         edges_from_face(i, edges); // all edges.
    //
    //     for(unsigned int i = 0; i < edges.size(); i++)
    //         {
    //         bool unique = true;
    //         for(unsigned int j = i+1; j < edges.size(); j++)
    //             {
    //             if(edges[i][0] == edges[j][0] && edges[i][1] == edges[j][1])
    //                 {
    //                 unique = false;
    //                 }
    //             }
    //         if(unique)
    //             {
    //             m_edges.push_back(edges[i]);
    //             }
    //         }
    //     }

    void remove_deleted_faces()
        {
        std::vector< std::vector<unsigned int> >::iterator f;
        std::vector< bool >::iterator d;
        bool bContinue = true;
        while(bContinue)
            {
            bContinue = false;
            d = m_deleted.begin();
            f = m_faces.begin();
            for(; f != m_faces.end() && d != m_deleted.end(); f++, d++)
                {
                if(*d)
                    {
                    m_faces.erase(f);
                    m_deleted.erase(d);
                    bContinue = true;
                    break;
                    }
                }
            }
        m_adjacency.clear(); // the id's of the faces are all different so just clear the list.
        }

public:
    const std::vector< std::vector<unsigned int> >& getFaces() { return m_faces; }

    // const std::vector< std::vector<unsigned int> >& getEdges() { return m_edges; }

    const std::vector< VectorRn<P,D> >& getPoints() { return m_points; }

    void moveData(std::vector< std::vector<unsigned int> >& faces, std::vector< VectorRn<P,D> >& points)
        {
        // NOTE: *this is not valid after using this method!
        faces = std::move(m_faces);
        points = std::move(m_points);
        }

protected:
    size_t m_dim;
    VectorRn<P,D>                                   m_ravg;
    std::vector< VectorRn<P,D> >                    m_points;
    std::vector< std::vector<unsigned int> >        m_faces; // Always have d vertices in a face.
    std::vector< VectorRn<P,D> >                    m_normals;
    // std::vector< std::vector<unsigned int> >        m_edges; // Always have 2 vertices in an edge.
    std::vector< std::set<unsigned int> >       m_adjacency; // the face adjacency list.
    std::vector<bool>                           m_deleted;
};

template<class P>
class GrahamScan
{
public:
    template<class VectorType>
    GrahamScan(const std::vector< VectorType >& points)
    {
        assert(points.size() > 0);
        m_points.resize(points.size());
        for(int i = 0; i < points.size(); i++)
        {
            m_points[i] = points[i]; // copies the points.
        }
        m_dim = dim(m_points[0]).value;
    }

    template<class MatrixType>
    GrahamScan(const MatrixType& points)
    {
        m_dim = dim(points).value;
        m_points.reserve(points.cols());
        for(int i = 0; i < points.cols(); i++)
        {
            m_points.push_back(points.col(i));
        }
    }
    template<class OutputIterator, class IndexIterator=int>
    void compute(OutputIterator hull, IndexIterator first=0, IndexIterator last=0, VectorRn<P, 3> normal = {0,0,1}, P tol=tolerance<P>())
    {
        std::vector<int> Index(__make_index_vector(first, last, m_points.size()));
        size_t N = Index.size();
        SubSpace<P> plane(m_points, Index, 3, tol);
        if(plane.dim() != 2)
        {
            std::cerr << "output: \n" << plane.basis() << std::endl;
            throw std::runtime_error("Graham Scan is only valid on 2D subspaces");
        }
        VectorRn<P,3> e1 = plane.basis().col(0);
        VectorRn<P,3> e2 = plane.basis().col(1);
        e1.normalize();
        e2.normalize();
        support_function<P> supp(m_points, Index.begin(), Index.end());
        normal.normalize();

        counter_clockwise<P> ccw(supp.element(e1), normal);
        std::function<bool(const int&, const int&)> lessfn = ccw.template lessfn<int, std::vector< VectorRn<P,3> > >(m_points);

        std::vector<int> points(Index);
        std::sort(points.begin(), points.end(), lessfn);
        points.insert(points.begin(), points.back());

        int M = 1;
        for(int i = 2; i < N+1; i++)
        {
            while(!ccw(m_points[points[M-1]], m_points[points[M]], m_points[points[i]]) )
            {
                if( M > 1)      M--;
                else if(i >= N) break;
                else            i++;

            }
            M++;
            std::swap(points[M], points[i]);
        }
        std::copy(points.begin(), points.begin()+M, hull);
    }
protected:
    size_t m_dim;
    std::vector< VectorRn<P,3> >    m_points;
};

}
