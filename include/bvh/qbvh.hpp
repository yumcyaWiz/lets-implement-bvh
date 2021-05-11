#ifndef _QBVH_H
#define _QBVH_H
#include <immintrin.h>

#include <numeric>
#include <stack>
#include <vector>

#include "core/triangle.hpp"

class QBVH {
 private:
  std::vector<Triangle> primitives;  // Primitive(三角形)の配列
  std::vector<int> primIndices;  // primitivesへのインデックスの配列.
                                 // 分割時にはこれがソートされる

  // ノードを表す構造体
  // NOTE: 128ByteにAlignmentすることでキャッシュ効率を良くする
  struct alignas(128) BVHNode {
    // バウンディングボックス(子ノード4つ分)
    float xmin[4];
    float xmax[4];
    float ymin[4];
    float ymax[4];
    float zmin[4];
    float zmax[4];
    int child
        [4];  // 子ノードへのインデックス(先頭1bitで葉ノードかどうかを表し、残り4bitにPrimitive数,
              // 残り27bitにprimIndicesへのオフセット)
    int axisTop;    // ノードの分割軸
    int axisLeft;   // 左ノードの分割軸
    int axisRight;  // 右ノードの分割軸
  };

  // 葉ノードの情報を表す構造体
  struct LeafInfo {
    int primIndicesOffset;  // primIndicesへのオフセット
    int nPrimitives;        // Primitiveの数
  };

  // BVHの統計情報を表す構造体
  struct BVHStatistics {
    int nNodes{0};          // ノード総数
    int nInternalNodes{0};  // 中間ノードの数
    int nLeafNodes{0};      // 葉ノードの数
  };

  std::vector<BVHNode> nodes;    // ノード配列(深さ優先順)
  std::vector<LeafInfo> leaves;  // 葉の情報の配列
  BVHStatistics stats;           // BVHの統計情報

  // 指定したPrimitiveを含むAABBを計算
  static AABB computeAABB(int primStart, int primEnd,
                          const std::vector<AABB>& bboxes,
                          const std::vector<int>& primIndices) {
    AABB ret;
    for (int i = primStart; i < primEnd; ++i) {
      const int primIdx = primIndices[i];
      ret = mergeAABB(ret, bboxes[primIdx]);
    }
    return ret;
  }

  // 指定したPrimitiveの中心を含むAABBを計算
  static AABB computeCentroidAABB(int primStart, int primEnd,
                                  const std::vector<AABB>& bboxes,
                                  const std::vector<int>& primIndices) {
    AABB ret;
    for (int i = primStart; i < primEnd; ++i) {
      const int primIdx = primIndices[i];
      ret = mergeAABB(ret, bboxes[primIdx].center());
    }
    return ret;
  }

  // AABBを等数分割する
  void splitAABB(int primStart, int primEnd, const std::vector<AABB>& bboxes,
                 std::vector<int>& primIndices, int& splitAxis, int& splitIdx) {
    // 分割用に各Primitiveの中心点を含むAABBを計算
    // NOTE: bboxをそのまま使ってしまうとsplitが失敗することが多い
    const AABB splitAABB =
        computeCentroidAABB(primStart, primEnd, bboxes, primIndices);

    // AABBの分割
    splitAxis = splitAABB.longestAxis();
    splitIdx = primStart + (primEnd - primStart) / 2;
    std::nth_element(primIndices.begin() + primStart,
                     primIndices.begin() + splitIdx,
                     primIndices.begin() + primEnd, [&](int idx1, int idx2) {
                       return bboxes[idx1].center()[splitAxis] <
                              bboxes[idx2].center()[splitAxis];
                     });
  }

  // 葉ノードの情報のパッキングを行う
  static int encodeLeaf(int nPrims, int primStart) {
    int enc = 0;
    // 先頭1bitに葉ノードフラグを書き込む
    enc |= (1 << 31);
    // 2bit目-5bit目にPrimitiveの数を書き込む
    enc |= ((nPrims & 0xf) << 27);
    // 残りの27bitにprimIndicesへのオフセットを書き込む
    enc |= (primStart & 0x07ffffff);
    return enc;
  }

  // 葉ノードの情報のアンパッキングを行う
  static void decodeLeaf(int child, bool& isLeaf, int& nPrims,
                         int& primIndicesOffset) {
    isLeaf = (child & 0x80000000) >> 31;
    nPrims = (child & 0x78000000) >> 27;
    primIndicesOffset = (child & 0x07ffffff);
  }

  // 再帰的にBVHのノードを構築していく
  void buildBVHNode(int primStart, int primEnd, const std::vector<AABB>& bboxes,
                    std::vector<int>& primIndices) {
    // AABBの分割
    int splitAxisTop, splitIdxTop;
    splitAABB(primStart, primEnd, bboxes, primIndices, splitAxisTop,
              splitIdxTop);

    // 左AABBの分割
    int splitAxisLeft, splitIdxLeft;
    splitAABB(primStart, splitIdxTop, bboxes, primIndices, splitAxisLeft,
              splitIdxLeft);

    // 右AABBの分割
    int splitAxisRight, splitIdxRight;
    splitAABB(splitIdxTop, primEnd, bboxes, primIndices, splitAxisRight,
              splitIdxRight);

    // 各子ノードのAABBの計算
    const AABB bbox0 =
        computeAABB(primStart, splitIdxLeft, bboxes, primIndices);
    const AABB bbox1 =
        computeAABB(splitIdxLeft, splitIdxTop, bboxes, primIndices);
    const AABB bbox2 =
        computeAABB(splitIdxTop, splitIdxRight, bboxes, primIndices);
    const AABB bbox3 = computeAABB(splitIdxRight, primEnd, bboxes, primIndices);

    // ノードの作成
    const AABB childboxes[4] = {bbox0, bbox1, bbox2, bbox3};
    BVHNode node;
    for (int i = 0; i < 4; ++i) {
      node.xmin[i] = childboxes[i].bounds[0][0];
      node.xmax[i] = childboxes[i].bounds[1][0];
      node.ymin[i] = childboxes[i].bounds[0][1];
      node.ymax[i] = childboxes[i].bounds[1][1];
      node.zmin[i] = childboxes[i].bounds[0][2];
      node.zmax[i] = childboxes[i].bounds[1][2];
    }
    node.axisTop = splitAxisTop;
    node.axisLeft = splitAxisLeft;
    node.axisRight = splitAxisRight;

    const int nPrimsChild[4] = {
        splitIdxLeft - primStart, splitIdxTop - splitIdxLeft,
        splitIdxRight - splitIdxTop, primEnd - splitIdxRight};
    const int primStartChild[4] = {primStart, splitIdxLeft, splitIdxTop,
                                   splitIdxRight};
    // 各子ノードについて葉ノードになるか判定
    for (int i = 0; i < 4; ++i) {
      if (nPrimsChild[i] <= 4) {
        // 葉ノードを作る
        node.child[i] = encodeLeaf(nPrimsChild[i], primStartChild[i]);
        nodes.push_back(node);
        stats.nLeafNodes++;
        return;
      }
    }

    // 中間ノードを配列に追加する. その際に自分の位置を覚えておく
    const int parentOffset = nodes.size();
    nodes.push_back(node);
    stats.nInternalNodes++;

    // 子ノード1へのオフセットを計算し, 親ノードにセットする
    const int child1Offset = nodes.size();
    nodes[parentOffset].child[0] = child1Offset;

    // 子ノード1の部分木を配列に追加していく
    buildBVHNode(primStart, splitIdxLeft, bboxes, primIndices);

    // 子ノード2へのオフセットを計算し, 親ノードにセットする
    const int child2Offset = nodes.size();
    nodes[parentOffset].child[1] = child2Offset;

    // 子ノード2の部分木を配列に追加していく
    buildBVHNode(splitIdxLeft, splitIdxTop, bboxes, primIndices);

    // 子ノード3へのオフセットを計算し, 親ノードにセットする
    const int child3Offset = nodes.size();
    nodes[parentOffset].child[2] = child3Offset;

    // 子ノード3の部分木を配列に追加していく
    buildBVHNode(splitIdxTop, splitIdxRight, bboxes, primIndices);

    // 子ノード4へのオフセットを計算し, 親ノードにセットする
    const int child4Offset = nodes.size();
    nodes[parentOffset].child[3] = child4Offset;

    // 子ノード4の部分木を配列に追加していく
    buildBVHNode(splitIdxRight, primEnd, bboxes, primIndices);
  }

  // SIMDでray-box intersectionを行う
  static int intersectAABB(const __m128 orig[3], const __m128 dirInv[3],
                           const int dirInvSign[3], const __m128 raytmin,
                           const __m128 raytmax, const __m128 bounds[2][3]) {
    // SIMD version of https://dl.acm.org/doi/abs/10.1145/1198555.1198748
    __m128 tmin =
        _mm_mul_ps(_mm_sub_ps(bounds[dirInvSign[0]][0], orig[0]), dirInv[0]);
    __m128 tmax = _mm_mul_ps(_mm_sub_ps(bounds[1 - dirInvSign[0]][0], orig[0]),
                             dirInv[0]);

    tmin = _mm_min_ps(
        tmin,
        _mm_mul_ps(_mm_sub_ps(bounds[dirInvSign[1]][1], orig[1]), dirInv[1]));
    tmax = _mm_max_ps(
        tmax, _mm_mul_ps(_mm_sub_ps(bounds[1 - dirInvSign[1]][1], orig[1]),
                         dirInv[1]));

    tmin = _mm_min_ps(
        tmin,
        _mm_mul_ps(_mm_sub_ps(bounds[dirInvSign[2]][2], orig[2]), dirInv[2]));
    tmax = _mm_max_ps(
        tmax, _mm_mul_ps(_mm_sub_ps(bounds[1 - dirInvSign[2]][2], orig[2]),
                         dirInv[2]));

    const __m128 comp1 = _mm_cmp_ps(tmax, tmin, _CMP_GT_OQ);
    const __m128 comp2 = _mm_and_ps(_mm_cmp_ps(tmin, raytmax, _CMP_LT_OQ),
                                    _mm_cmp_ps(tmax, raytmin, _CMP_GT_OQ));
    return _mm_movemask_ps(_mm_and_ps(comp1, comp2));
  }

  // 再帰的にBVHのtraverseを行う
  bool intersectNode(int nodeIdx, const Ray& ray, const Vec3& dirInv,
                     const int dirInvSign[3], IntersectInfo& info) const {
    return false;
  }

 public:
  QBVH(const Polygon& polygon) {
    // PolygonからTriangleを抜き出して追加していく
    for (unsigned int f = 0; f < polygon.nFaces(); ++f) {
      primitives.emplace_back(&polygon, f);
    }
  }

  // BVHを構築する
  void buildBVH() {
    // 各Primitiveのバウンディングボックスを事前計算
    std::vector<AABB> bboxes;
    for (const auto& prim : primitives) {
      bboxes.push_back(prim.calcAABB());
    }

    // 各Primitiveへのインデックスを表す配列を作成
    primIndices.resize(primitives.size());
    std::iota(primIndices.begin(), primIndices.end(), 0);

    // BVHの構築をルートノードから開始
    buildBVHNode(0, primitives.size(), bboxes, primIndices);

    // 総ノード数を計算
    stats.nNodes = stats.nInternalNodes + stats.nLeafNodes;
  }

  // ノード数を返す
  int nNodes() const { return stats.nNodes; }
  // 中間ノード数を返す
  int nInternalNodes() const { return stats.nInternalNodes; }
  // 葉ノード数を返す
  int nLeafNodes() const { return stats.nLeafNodes; }

  // 全体のバウンディングボックスを返す
  AABB rootAABB() const {
    if (nodes.size() > 0) {
      const BVHNode& root = nodes[0];
      const AABB bbox1 = AABB(Vec3(root.xmin[0], root.ymin[0], root.zmin[0]),
                              Vec3(root.xmax[0], root.ymax[0], root.zmax[0]));
      const AABB bbox2 = AABB(Vec3(root.xmin[1], root.ymin[1], root.zmin[1]),
                              Vec3(root.xmax[1], root.ymax[1], root.zmax[1]));
      const AABB bbox3 = AABB(Vec3(root.xmin[2], root.ymin[2], root.zmin[2]),
                              Vec3(root.xmax[2], root.ymax[2], root.zmax[2]));
      const AABB bbox4 = AABB(Vec3(root.xmin[3], root.ymin[3], root.zmin[3]),
                              Vec3(root.xmax[3], root.ymax[3], root.zmax[3]));
      return mergeAABB(bbox1, mergeAABB(bbox2, mergeAABB(bbox3, bbox4)));
    } else {
      return AABB();
    }
  }

  // traverseをする
  bool intersect(const Ray& ray, IntersectInfo& info) const {
    // レイの方向の逆数と符号を事前計算しておく
    const Vec3 dirInv = 1.0f / ray.direction;
    int dirInvSign[3];
    for (int i = 0; i < 3; ++i) {
      dirInvSign[i] = dirInv[i] > 0 ? 0 : 1;
    }
    return intersectNode(0, ray, dirInv, dirInvSign, info);
  }
};

#endif