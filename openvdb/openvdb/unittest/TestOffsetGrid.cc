// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0

/*
make && openvdb/openvdb/unittest/vdb_test --gtest_filter="TestOffsetGrid*" --gtest_break_on_failure

make && openvdb/openvdb/unittest/vdb_test --gtest_filter="TestOffsetGrid*benchmark*" --gtest_break_on_failure --gtest_repeat=5
*/

#include <openvdb/Exceptions.h>
#include <openvdb/openvdb.h>
#include <openvdb/Types.h>
#include <openvdb/util/Name.h>
#include <openvdb/math/Transform.h>
#include <openvdb/Grid.h>
#include <openvdb/tree/Tree.h>
#include <openvdb/util/CpuTimer.h>
#include <openvdb/tools/LevelSetSphere.h>
#include "gtest/gtest.h"
#include <iostream>
#include <memory> // for std::make_unique

#define ASSERT_DOUBLES_EXACTLY_EQUAL(expected, actual) \
    EXPECT_NEAR((expected), (actual), /*tolerance=*/0.0);

class TestOffsetGrid: public ::testing::Test
{
};

////////////////////////////////////////

namespace test_new_root {

// Define this to allow the root node to have a variable (Vs fixed) offset
//#define USE_VARIABLE_OFFSET    

// Define this to store tile values in the ValueAccessor
//#define CACHE_TILE_VALUES_IN_ACCESSOR

    template<typename ChildType>
    struct Root {
        using ChildT = ChildType;
        using CoordT = typename ChildT::CoordT;
        using ValueT = typename ChildT::ValueT;
        using LeafT  = typename ChildT::LeafT;
        static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL;
        struct Tile {
            ChildT* child;
            ValueT  value;
            bool    state;
            Tile(ChildT* c = nullptr) : child(c) {}
            Tile(const ValueT& v, bool s) : child(nullptr), value(v), state(s) {}
        };
        std::map<CoordT, Tile> mTable;
        ValueT mBackground;
#ifdef USE_VARIABLE_OFFSET
        const CoordT mOrigin;// immutable by design!
        // offset is rounded down so its divisible by the size of grand-child nodes, e.g. 16*8=128 
        Root(const ValueT& background, const CoordT &origin = CoordT(-(ChildT::DIM >> 1)) ) 
            : mBackground(background), mOrigin((origin >> ChildT::ChildT::TOTAL) << ChildT::ChildT::TOTAL)
        {
        }
#else
        static const CoordT mOrigin;
        Root(const ValueT& background) : mBackground(background){}
#endif
        inline const CoordT& origin() const {return mOrigin;}
        inline CoordT coordToKey(const CoordT& ijk) const { return (ijk - mOrigin) & ~ChildT::MASK; }
        void setValue(const CoordT &ijk, const ValueT& value) {
            ChildT* child = nullptr;
            const CoordT key = this->coordToKey(ijk);
            auto iter = mTable.find(key);
            if (iter == mTable.end()) {
                child = new ChildT(mBackground);// NB!
                mTable[key] = Tile(child);
                child->mOrigin = key + mOrigin;// NB!
            } else if (iter->second.child != nullptr) {
                child = iter->second.child;
            } else {
                child = new ChildT(iter->second.value, iter->second.state);// NB!
                iter->second.child = child;
                child->mOrigin = key + mOrigin;// NB!
            }
            assert(child);
            child->setValue(ijk, value);
        }
        const ValueT& getValue(const CoordT &ijk) const {
            auto iter = mTable.find(this->coordToKey(ijk));
            if (iter == mTable.end()) {
                return mBackground;
            } else if (iter->second.child) {
                return iter->second.child->getValue(ijk);
            } else {
                return iter->second.value;
            }
        }
        template <typename AccT>
        const ValueT& getValueAndCache(const CoordT &ijk, AccT &acc) const {
            const CoordT key = this->coordToKey(ijk);
            auto iter = mTable.find(key);
            if (iter == mTable.end()) {
                return mBackground;
            } else if (auto child = iter->second.child) {
                acc.insert(key, child);
                return child->getValueAndCache(ijk, acc);
            } else {
#ifdef CACHE_TILE_VALUES_IN_ACCESSOR
                acc.insert(key, ChildT::LEVEL, iter->second.value);// NB!
#endif
                return iter->second.value;
            }
        }
    };// Root

#ifndef USE_VARIABLE_OFFSET
    template<typename ChildT>
    const typename ChildT::CoordT Root<ChildT>::mOrigin(-(ChildT::DIM >> 1));
#endif

    template <typename ChildType>
    struct Internal {
        using ChildT = ChildType;
        static constexpr uint32_t LOG2DIM = ChildT::LOG2DIM + 1;
        static constexpr uint32_t TOTAL = LOG2DIM + ChildT::TOTAL; //dimension in index space
        static constexpr uint32_t DIM = 1u << TOTAL;
        static constexpr uint32_t SIZE = 1u << (3 * LOG2DIM); //number of tile values (or child pointers)
        static constexpr int32_t  MASK = DIM - 1;
        static constexpr uint32_t LEVEL = 1 + ChildT::LEVEL;
        using MaskT  = openvdb::util::NodeMask<LOG2DIM>;
        using CoordT = typename ChildT::CoordT;
        using ValueT = typename ChildT::ValueT;
        using LeafT  = typename ChildT::LeafT;

        static uint32_t CoordToOffset(const CoordT &ijk) {
            return (((ijk[0] & MASK) >> ChildT::TOTAL) << (2 * LOG2DIM)) +
                   (((ijk[1] & MASK) >> ChildT::TOTAL) << (LOG2DIM)) +
                    ((ijk[2] & MASK) >> ChildT::TOTAL);
        }
        static CoordT OffsetToLocalCoord(uint32_t n) {
            assert(n < SIZE);
            const uint32_t m = n & ((1 << 2 * LOG2DIM) - 1);
            return CoordT(n >> 2 * LOG2DIM, m >> LOG2DIM, m & ((1 << LOG2DIM) - 1));
        }
        void localToGlobalCoord(CoordT& ijk) const {
            ijk <<= ChildT::TOTAL;
            ijk += mOrigin;
        }
        CoordT offsetToGlobalCoord(uint32_t n) const {
            CoordT ijk = OffsetToLocalCoord(n);
            this->localToGlobalCoord(ijk);
            return ijk;
        }
        struct Tile {
            Tile(ChildT* c = nullptr) : child(c) {}
            union { ChildT* child; ValueT  value; };
        };
        CoordT mOrigin;
        MaskT  mValueMask;
        MaskT  mChildMask;
        Tile   mTable[SIZE];
        //const ChildT mChild;
        Internal(const ValueT& value) : mOrigin(), mValueMask(), mChildMask()//, mChild(value)
        {
            for (uint32_t i = 0; i < SIZE; ++i) mTable[i].value = value;
        }
        // NB!
        Internal(const ValueT& value, bool state) : mOrigin(), mValueMask(state), mChildMask()//, mChild(value, state)
        {
            for (uint32_t i = 0; i < SIZE; ++i) mTable[i].value = value;
        }
        Internal(const CoordT& ijk, const ValueT& value, bool state) : mOrigin(ijk & ~MASK), mValueMask(state)//, mChildMask(), mChild(value, state)
        {
            for (uint32_t i = 0; i < SIZE; ++i) mTable[i].value = value;
        }
        void setValue(const CoordT& ijk, const ValueT& value) {
            const uint32_t n = CoordToOffset(ijk);
            ChildT* child = nullptr;
            if (mChildMask.isOn(n)) {
                child = mTable[n].child;
            } else {
                child = new ChildT(ijk, mTable[n].value, mValueMask.isOn(n));
                mTable[n].child = child;
                mChildMask.setOn(n);
            }
            child->setValue(ijk, value);
        }
        const ValueT getFirstValue() const { return mTable[0].value;}
        const ChildT* getChild(const CoordT& ijk) const {
            const uint32_t n = CoordToOffset(ijk);
            return mChildMask.isOn(n) ? mTable[n].child : nullptr;
        }
        const ValueT& getValue(const CoordT& ijk) const {
            const uint32_t n = CoordToOffset(ijk);
            return mChildMask.isOn(n) ? mTable[n].child->getValue(ijk) : mTable[n].value;
        }
        template <typename AccT>
        const ValueT& getValueAndCache(const CoordT &ijk, AccT &acc) const {
            const uint32_t n = CoordToOffset(ijk);
            if (mChildMask.isOn(n)) {
                ChildT* child = mTable[n].child;
                acc.insert(ijk & ~MASK, child);// NB!
                return child->getValueAndCache(ijk, acc);
            }
#ifdef CACHE_TILE_VALUES_IN_ACCESSOR
            acc.insert(ijk & ~ChildT::MASK, ChildT::LEVEL, mTable[n].value);// NB!
#endif
            return mTable[n].value;
        }
    };// Internal

    template <typename ValueType>
    struct Leaf {
        static constexpr uint32_t LOG2DIM = 3;
        static constexpr uint32_t TOTAL = LOG2DIM; // needed by parent nodes
        static constexpr uint32_t DIM = 1u << TOTAL;
        static constexpr uint32_t SIZE = 1u << 3 * LOG2DIM; // total number of voxels represented by this node
        static constexpr int32_t  MASK = DIM - 1; // mask for bit operations
        static constexpr uint32_t LEVEL = 0;
        using CoordT = openvdb::Coord;
        using ValueT = ValueType;
        using MaskT  = openvdb::util::NodeMask<LOG2DIM>;
        using LeafT  = Leaf<ValueT>; 
        CoordT mOrigin;
        MaskT  mValueMask;
        ValueT mValues[SIZE];
        Leaf(const ValueT& value) {
            ValueT*  target = mValues;
            uint32_t n = SIZE;
            while (n--) *target++ = value;
        }
        Leaf(const ValueT& value, bool state) : mValueMask(state) {
            ValueT*  target = mValues;
            uint32_t n = SIZE;
            while (n--) *target++ = value;
        }
        Leaf(const CoordT& ijk, const ValueT& value, bool state) : mOrigin(ijk & ~MASK),mValueMask(state) {
            ValueT*  target = mValues;
            uint32_t n = SIZE;
            while (n--) *target++ = value;
        }
        static uint32_t CoordToOffset(const CoordT& ijk) {
            return ((ijk[0] & MASK) << (2 * LOG2DIM)) + 
                   ((ijk[1] & MASK) << LOG2DIM) + 
                    (ijk[2] & MASK);
        }
        const ValueT getFirstValue() const { return mValues[0];}
        const ValueT& getValue(const CoordT& ijk) const {
            return mValues[CoordToOffset(ijk)];
        }
        void setValue(const CoordT& ijk, const ValueT& value) {
            const uint32_t n = CoordToOffset(ijk);
            mValueMask.setOn(n);
            mValues[n] = value;
        }
        template <typename AccT>
        const ValueT& getValueAndCache(const CoordT &ijk, AccT&) const {
            return mValues[CoordToOffset(ijk)];
        }
    };// Leaf

    template <typename RootT>
    struct Accessor {
        using CoordT = typename RootT::CoordT;
        using ValueT = typename RootT::ValueT;
        using Node2 = typename RootT::ChildT; // upper internal node
        using Node1 = typename Node2::ChildT; // lower internal node
        using LeafT = typename Node1::ChildT; // Leaf node
    #ifdef CACHE_TILE_VALUES_IN_ACCESSOR
        mutable ValueT mValue[3];
    #endif
        mutable CoordT mKeys[3];
        mutable const void* mNode[4];
        template<typename NodeT>
        bool isCached(const CoordT& ijk) const {
            if (NodeT::LEVEL + 1 == 3) {// resolved at compile time
                return ((const RootT*)mNode[3])->coordToKey(ijk) == mKeys[2]; 
            } else {
                return (ijk[0] & ~NodeT::MASK) == mKeys[NodeT::LEVEL][0] && 
                       (ijk[1] & ~NodeT::MASK) == mKeys[NodeT::LEVEL][1] && 
                       (ijk[2] & ~NodeT::MASK) == mKeys[NodeT::LEVEL][2];
            }
        }
        template<typename NodeT>
        void insert(const CoordT& key, const NodeT* node) const {
            mKeys[NodeT::LEVEL] = key;
            mNode[NodeT::LEVEL] = node;
        }
#ifdef CACHE_TILE_VALUES_IN_ACCESSOR
        void insert(const CoordT& key, int n, const ValueT &v) const {
            mKeys[n] = key;
            mNode[n] = nullptr;
            mValue[n] = v;
        }
#endif
        Accessor(const RootT& root) : mKeys{CoordT::max(), CoordT::max(), CoordT::max()}, mNode{nullptr, nullptr, nullptr, &root} 
        {}
        const ValueT& getValue(const CoordT& ijk) const {
            if (this->isCached<LeafT>(ijk)) {
#ifdef CACHE_TILE_VALUES_IN_ACCESSOR
                const LeafT *leaf = static_cast<const LeafT*>(mNode[0]);
                return leaf ? leaf->getValue(ijk) : mValue[0];
#else
                return ((const LeafT*)mNode[0])->getValue(ijk);
#endif
            } else if (this->isCached<Node1>(ijk)) {
#ifdef CACHE_TILE_VALUES_IN_ACCESSOR
                const Node1 *node = static_cast<const Node1*>(mNode[1]);
                return node ? node->getValueAndCache(ijk, *this) : mValue[1];
#else
                return ((const Node1*)mNode[1])->getValueAndCache(ijk, *this);
#endif
            } else if (this->isCached<Node2>(ijk)) {
#ifdef CACHE_TILE_VALUES_IN_ACCESSOR
                const Node2 *node = static_cast<const Node2*>(mNode[2]);
                return node ? node->getValueAndCache(ijk, *this) : mValue[2];
#else
                return ((const Node2*)mNode[2])->getValueAndCache(ijk, *this);
#endif
            }
            return ((const RootT*)mNode[3])->getValueAndCache(ijk, *this);
        }
    };// Accessor

}// test_new_root namespace



////////////////////////////////////////

#ifdef USE_VARIABLE_OFFSET
TEST_F(TestOffsetGrid, NewRoot0)
{
    using LeafT = test_new_root::Leaf<float>;
    using Node1 = test_new_root::Internal<LeafT>;
    using Node2 = test_new_root::Internal<Node1>;
    using RootT = test_new_root::Root<Node2>;
    using AccT  = test_new_root::Accessor<RootT>;
    RootT root(0.0f, openvdb::Coord(2));
    EXPECT_EQ(openvdb::Coord(0), root.origin());
    openvdb::Coord p(1), q(-1);
    EXPECT_EQ(0.0f, root.getValue(p));
    
    root.setValue(p,  1.0f);
    root.setValue(q, -1.0f);
    EXPECT_EQ( 2u, root.mTable.size());
    EXPECT_EQ( 0.0f, root.getValue(openvdb::Coord(0)));
    EXPECT_EQ( 1.0f, root.getValue(p));
    EXPECT_EQ(-1.0f, root.getValue(q));

    AccT acc(root);
    EXPECT_EQ( 0.0f, acc.getValue(openvdb::Coord(0)));
    EXPECT_EQ( 1.0f, acc.getValue(p));
    EXPECT_EQ(-1.0f, acc.getValue(q));

    auto it = root.mTable.begin();
    EXPECT_NE(it, root.mTable.end());
    const Node2 *node2 = it->second.child;
    EXPECT_TRUE(node2);
    //std::cerr << it->second.child->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(-4096), node2->mOrigin);

    const Node1 *node1q = node2->getChild(q);
    EXPECT_TRUE(node1q);
    //std::cerr << node1q->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(-128), node1q->mOrigin);

    const LeafT *node0q = node1q->getChild(q);
    EXPECT_TRUE(node0q);
    //std::cerr << node0q->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(-8), node0q->mOrigin);

    ++it;
    EXPECT_NE(it, root.mTable.end());
    node2 = it->second.child;
    EXPECT_TRUE(node2);

    const Node1 *node1p = node2->getChild(p);
    EXPECT_TRUE(node1p);
    //std::cerr << node1p->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(0), node1p->mOrigin);

    const LeafT *node0p = node1p->getChild(p);
    EXPECT_TRUE(node0p);
    //std::cerr << node0p->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(0), node0p->mOrigin);

}// NewRoot0
#endif

TEST_F(TestOffsetGrid, NewRoot1)
{
    using LeafT = test_new_root::Leaf<float>;
    using Node1 = test_new_root::Internal<LeafT>;
    using Node2 = test_new_root::Internal<Node1>;
    using RootT = test_new_root::Root<Node2>;
    using AccT  = test_new_root::Accessor<RootT>;
    RootT root(0.0f);
    EXPECT_EQ(openvdb::Coord(-2048), root.origin());
    openvdb::Coord p(1), q(-1);
    EXPECT_EQ(0.0f, root.getValue(p));
    root.setValue(p,  1.0f);
    root.setValue(q, -1.0f);
    EXPECT_EQ( 1u, root.mTable.size());
    EXPECT_EQ( 0.0f, root.getValue(openvdb::Coord(0)));
    EXPECT_EQ( 1.0f, root.getValue(p));
    EXPECT_EQ(-1.0f, root.getValue(q));

    AccT acc(root);
    EXPECT_EQ( 0.0f, acc.getValue(openvdb::Coord(0)));
    EXPECT_EQ( 1.0f, acc.getValue(p));
    EXPECT_EQ(-1.0f, acc.getValue(q));

    auto it = root.mTable.begin();
    EXPECT_NE(it, root.mTable.end());
    const Node2 *node2 = it->second.child;
    EXPECT_TRUE(node2);
    //std::cerr << it->second.child->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(-2048), node2->mOrigin);

    const Node1 *node1p = node2->getChild(p);
    EXPECT_TRUE(node1p);
    //std::cerr << node1p->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(0), node1p->mOrigin);

    const Node1 *node1q = node2->getChild(q);
    EXPECT_TRUE(node1q);
    //std::cerr << node1q->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(-128), node1q->mOrigin);

    const LeafT *node0p = node1p->getChild(p);
    EXPECT_TRUE(node0p);
    //std::cerr << node0p->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(0), node0p->mOrigin);

    const LeafT *node0q = node1q->getChild(q);
    EXPECT_TRUE(node0q);
    //std::cerr << node0q->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(-8), node0q->mOrigin);
}// NewRoot1

#ifdef USE_VARIABLE_OFFSET
TEST_F(TestOffsetGrid, NewRoot1b)
{
    using LeafT = test_new_root::Leaf<float>;
    using Node1 = test_new_root::Internal<LeafT>;
    using Node2 = test_new_root::Internal<Node1>;
    using RootT = test_new_root::Root<Node2>;
    using AccT  = test_new_root::Accessor<RootT>;
    RootT root(0.0f, openvdb::Coord(-128));
    EXPECT_EQ(openvdb::Coord(-128), root.origin());
    openvdb::Coord p(1), q(-1);
    EXPECT_EQ(0.0f, root.getValue(p));
    root.setValue(p,  1.0f);
    root.setValue(q, -1.0f);
    EXPECT_EQ( 1u, root.mTable.size());
    EXPECT_EQ( 0.0f, root.getValue(openvdb::Coord(0)));
    EXPECT_EQ( 1.0f, root.getValue(p));
    EXPECT_EQ(-1.0f, root.getValue(q));

    AccT acc(root);
    EXPECT_EQ( 0.0f, acc.getValue(openvdb::Coord(0)));
    EXPECT_EQ( 1.0f, acc.getValue(p));
    EXPECT_EQ(-1.0f, acc.getValue(q));

    auto it = root.mTable.begin();
    EXPECT_NE(it, root.mTable.end());
    const Node2 *node2 = it->second.child;
    EXPECT_TRUE(node2);
    //std::cerr << it->second.child->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(-128), node2->mOrigin);

    const Node1 *node1p = node2->getChild(p);
    EXPECT_TRUE(node1p);
    //std::cerr << node1p->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(0), node1p->mOrigin);

    const Node1 *node1q = node2->getChild(q);
    EXPECT_TRUE(node1q);
    //std::cerr << node1q->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(-128), node1q->mOrigin);

    const LeafT *node0p = node1p->getChild(p);
    EXPECT_TRUE(node0p);
    //std::cerr << node0p->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(0), node0p->mOrigin);

    const LeafT *node0q = node1q->getChild(q);
    EXPECT_TRUE(node0q);
    //std::cerr << node0q->mOrigin << std::endl;
    EXPECT_EQ(openvdb::Coord(-8), node0q->mOrigin);
}// NewRoot1b
#endif

TEST_F(TestOffsetGrid, NewRoot2)
{
    using LeafT = test_new_root::Leaf<float>;
    using Node1 = test_new_root::Internal<LeafT>;
    using Node2 = test_new_root::Internal<Node1>;
    using RootT = test_new_root::Root<Node2>;
    using AccT  = test_new_root::Accessor<RootT>;
    RootT root(0.0f);
    EXPECT_EQ(openvdb::Coord(-2048), root.origin());
    const size_t voxelCount = 512;
    const int min = -2048, max = 2048 - 1;
    std::vector<openvdb::Coord> voxels;
    voxels.emplace_back(min);
    voxels.emplace_back(max);
    std::srand(98765);
    auto op = [&](){return rand() % (max - min) + min;};
    while (voxels.size() <  voxelCount) {
        const openvdb::Coord ijk(op(), op(), op());
        if (voxels.end() == std::find(voxels.begin(), voxels.end(), ijk)) {
            voxels.push_back(ijk);
        }
    }
    EXPECT_EQ(voxelCount, voxels.size());

    for (size_t i = 0; i < voxelCount; ++i) {
        root.setValue(voxels[i], float(i));
    }

    EXPECT_EQ(1u, root.mTable.size());

    for (size_t i = 0; i < voxelCount; ++i) {
        EXPECT_EQ(root.getValue(voxels[i]), float(i));
    }

    AccT acc(root);
    for (size_t i = 0; i < voxelCount; ++i) {
        EXPECT_EQ(acc.getValue(voxels[i]), float(i));
    }
}// NewRoot2

TEST_F(TestOffsetGrid, NewRoot3)
{
    using LeafT = test_new_root::Leaf<float>;
    using Node1 = test_new_root::Internal<LeafT>;
    using Node2 = test_new_root::Internal<Node1>;
    using RootT = test_new_root::Root<Node2>;
    using AccT  = test_new_root::Accessor<RootT>;
    RootT root(0.0f);
    EXPECT_EQ(openvdb::Coord(-2048), root.origin());
    const size_t voxelCount = 512;
    const int min = -4096, max = 4096 - 1;
    std::vector<openvdb::Coord> voxels;
    voxels.emplace_back(min);
    voxels.emplace_back(max);
    std::srand(98765);
    auto op = [&](){return rand() % (max - min) + min;};
    while (voxels.size() <  voxelCount) {
        const openvdb::Coord ijk(op(), op(), op());
        if (voxels.end() == std::find(voxels.begin(), voxels.end(), ijk)) {
            voxels.push_back(ijk);
        }
    }
    EXPECT_EQ(voxelCount, voxels.size());

    for (size_t i = 0; i < voxelCount; ++i) {
        root.setValue(voxels[i], float(i));
    }

    EXPECT_LE(1u, root.mTable.size());

    for (size_t i = 0; i < voxelCount; ++i) {
        EXPECT_EQ(root.getValue(voxels[i]), float(i));
    }

    AccT acc(root);
    for (size_t i = 0; i < voxelCount; ++i) {
        EXPECT_EQ(acc.getValue(voxels[i]), float(i));
    }
}// NewRoot3

// Benchmark the new tree with an offset
TEST_F(TestOffsetGrid, NewRoot_benchmark_with_offset)
{
#ifdef USE_VARIABLE_OFFSET
    std::cout << "Using variable offset\n";
#else
    std::cout << "Using fixed offset\n";
#endif

#ifdef CACHE_TILE_VALUES_IN_ACCESSOR
    std::cout << "Enabled caching of tile values in the ValueAccessor\n"; 
#else
    std::cout << "Disabled Caching of tile values in the ValueAccessor\n";
#endif
    openvdb::util::CpuTimer timer;
    using LeafT = test_new_root::Leaf<float>;
    using Node1 = test_new_root::Internal<LeafT>;
    using Node2 = test_new_root::Internal<Node1>;
    using RootT = test_new_root::Root<Node2>;
    using AccT  = test_new_root::Accessor<RootT>;
    RootT root(0.0f);
    AccT acc(root);
    EXPECT_EQ(openvdb::Coord(-2048), root.origin());

    float v;
    const double voxelSize = 1.0, halfWidth = 3.0;
    const float radius = 100.0f;
    const float delta = voxelSize*halfWidth;
    const openvdb::Vec3f center(0);
    const openvdb::Vec3d origin(0); 
    auto grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center,voxelSize, halfWidth);
    EXPECT_TRUE(grid);
    const auto bbox = grid->evalActiveVoxelBoundingBox();
    auto oldAcc = grid->getAccessor();
    for (auto it = bbox.begin(); it; ++it) {
        const openvdb::Coord p = *it;
        if (oldAcc.probeValue(p, v)) root.setValue(p, v);
    }

    auto &tree = grid->tree();
    timer.start("openvdb::tree::getValue"); 
    for (auto it = bbox.begin(); it; ++it) {
        EXPECT_LE(openvdb::math::Abs(tree.getValue(*it)), delta);
    }
    timer.stop();

    timer.start("offset::root::getValue"); 
    for (auto it = bbox.begin(); it; ++it) {
        EXPECT_LE(openvdb::math::Abs(root.getValue(*it)), delta);
    }
    timer.stop();

    timer.start("openvdb::ValueAccessor::getValue"); 
    for (auto it = bbox.begin(); it; ++it) {
        EXPECT_LE(openvdb::math::Abs(oldAcc.getValue(*it)), delta);
    }
    timer.stop();

    timer.start("offset::ValueAccessor::getValue"); 
    for (auto it = bbox.begin(); it; ++it) {
        EXPECT_LE(openvdb::math::Abs(acc.getValue(*it)), delta);
    }
    timer.stop();
    
}// NewRoot_benchmark_with_offset

#ifdef USE_VARIABLE_OFFSET
// Benchmark the new tree configured with no offset
TEST_F(TestOffsetGrid, NewRoot_bench_no_offset)
{
    openvdb::util::CpuTimer timer;
    using LeafT = test_new_root::Leaf<float>;
    using Node1 = test_new_root::Internal<LeafT>;
    using Node2 = test_new_root::Internal<Node1>;
    using RootT = test_new_root::Root<Node2>;
    using AccT  = test_new_root::Accessor<RootT>;
    RootT root(0.0f, openvdb::Coord(0));
    AccT acc(root);
    EXPECT_EQ(openvdb::Coord(0), root.origin());

    float v;
    const double voxelSize = 1.0, halfWidth = 3.0;
    const float radius = 100.0f;
    const float delta = voxelSize*halfWidth;
    const openvdb::Vec3f center(0);
    const openvdb::Vec3d origin(0); 
    auto grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, center,voxelSize, halfWidth);
    EXPECT_TRUE(grid);
    const auto bbox = grid->evalActiveVoxelBoundingBox();
    auto oldAcc = grid->getAccessor();
    for (auto it = bbox.begin(); it; ++it) {
        const openvdb::Coord p = *it;
        if (oldAcc.probeValue(p, v)) root.setValue(p, v);
    }

    auto &tree = grid->tree();
    timer.start("openvdb::tree::getValue"); 
    for (auto it = bbox.begin(); it; ++it) {
        EXPECT_LE(openvdb::math::Abs(tree.getValue(*it)), delta);
    }
    timer.stop();

    timer.start("offset::root::getValue"); 
    for (auto it = bbox.begin(); it; ++it) {
        EXPECT_LE(openvdb::math::Abs(root.getValue(*it)), delta);
    }
    timer.stop();

    timer.start("openvdb::ValueAccessor::getValue"); 
    for (auto it = bbox.begin(); it; ++it) {
        EXPECT_LE(openvdb::math::Abs(oldAcc.getValue(*it)), delta);
    }
    timer.stop();

    timer.start("offset::ValueAccessor::getValue"); 
    for (auto it = bbox.begin(); it; ++it) {
        EXPECT_LE(openvdb::math::Abs(acc.getValue(*it)), delta);
    }
    timer.stop();
}// NewRoot_bench_no_offset
#endif
