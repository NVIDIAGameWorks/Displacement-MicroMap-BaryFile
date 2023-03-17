/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <bary/bary_core.h>
#include <cassert>
#include <cstring>
#include <cstdlib>

namespace bary
{
static_assert(sizeof(bool) == 1, "bool must be 1 byte in size");

static_assert(sizeof(bary::MeshHistogramEntry) == sizeof(bary::HistogramEntry),
              "bary::HistogramEntry mismatches bary::MeshHistogramEntry");
static_assert(sizeof(bary::MeshHistogramEntry::subdivLevel) == sizeof(bary::HistogramEntry::subdivLevel),
              "bary::HistogramEntry mismatches bary::MeshHistogramEntry");
static_assert(sizeof(bary::MeshHistogramEntry::count) == sizeof(bary::HistogramEntry::count),
              "bary::HistogramEntry mismatches bary::MeshHistogramEntry");
static_assert(sizeof(bary::MeshHistogramEntry::blockFormat) == sizeof(bary::HistogramEntry::blockFormat),
              "bary::HistogramEntry mismatches bary::MeshHistogramEntry");

static_assert(offsetof(bary::MeshHistogramEntry, subdivLevel) == offsetof(bary::HistogramEntry, subdivLevel),
              "bary::HistogramEntry mismatches bary::MeshHistogramEntry");
static_assert(offsetof(bary::MeshHistogramEntry, count) == offsetof(bary::HistogramEntry, count),
              "bary::HistogramEntry mismatches bary::MeshHistogramEntry");
static_assert(offsetof(bary::MeshHistogramEntry, blockFormat) == offsetof(bary::HistogramEntry, blockFormat),
              "bary::HistogramEntry mismatches bary::MeshHistogramEntry");
//////////////////////////////////////////////

struct BaryWUV_uint32
{
    uint32_t w;
    uint32_t u;
    uint32_t v;
};

struct BaryUV_uint32
{
    uint32_t u;
    uint32_t v;
};

struct BaryTriangle
{
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

// Interleave even bits from x with odd bits from y
static inline uint32_t bird_interleaveBits(uint32_t x, uint32_t y)
{
    x = (x | (x << 8)) & 0x00ff00ff;
    x = (x | (x << 4)) & 0x0f0f0f0f;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;

    y = (y | (y << 8)) & 0x00ff00ff;
    y = (y | (y << 4)) & 0x0f0f0f0f;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}

// Calculate exclusive prefix or (log(n) XOR's and SHF's)
static inline uint32_t bird_prefixEor(uint32_t x)
{
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;

    return x;
}

// Compute 2 16-bit prefix XORs in a 32-bit register
static inline uint32_t bird_prefixEor2(uint32_t x)
{
    x ^= (x >> 1) & 0x7fff7fff;
    x ^= (x >> 2) & 0x3fff3fff;
    x ^= (x >> 4) & 0x0fff0fff;
    x ^= (x >> 8) & 0x00ff00ff;

    return x;
}

// Interleave 16 even bits from x with 16 odd bits from y
static inline uint32_t bird_interleaveBits2(uint32_t x, uint32_t y)
{
    x = (x & 0xffff) | (y << 16);
    x = ((x >> 8) & 0x0000ff00) | ((x << 8) & 0x00ff0000) | (x & 0xff0000ff);
    x = ((x >> 4) & 0x00f000f0) | ((x << 4) & 0x0f000f00) | (x & 0xf00ff00f);
    x = ((x >> 2) & 0x0c0c0c0c) | ((x << 2) & 0x30303030) | (x & 0xc3c3c3c3);
    x = ((x >> 1) & 0x22222222) | ((x << 1) & 0x44444444) | (x & 0x99999999);

    return x;
}

// Compute index of a single triplet of compression coefficients from triangle's barycentric coordinates
// Assumes u, v and w have only 16 valid bits in the lsbs (good for subdivision depths up to 64K segments per edge)
// Triplets are ordered along the bird curve
static inline uint32_t bird_getTripletIndex(uint32_t u, uint32_t v, uint32_t w, uint32_t level)
{
    const uint32_t coordMask = ((1U << level) - 1);

    uint32_t b0 = ~(u ^ w) & coordMask;
    uint32_t t  = (u ^ v) & b0;  //  (equiv: (~u & v & ~w) | (u & ~v & w))
    uint32_t c  = (((u & v & w) | (~u & ~v & ~w)) & coordMask) << 16;
    uint32_t f  = bird_prefixEor2(t | c) ^ u;
    uint32_t b1 = (f & ~b0) | t;  // equiv: (~u & v & ~w) | (u & ~v & w) | (f0 & u & ~w) | (f0 & ~u & w))

    uint32_t dist = bird_interleaveBits2(b0, b1);  // 13 instructions

    // Adjust computed distance accounting for "skipped" triangles on the bird curve

    f >>= 16;
    b0 <<= 1;
    return (dist + (b0 & ~f) - (b0 & f)) >> 3;
}

static inline uint32_t bird_getVertexLevel(BaryWUV_uint32 coord, uint32_t subdivLevel)
{
    uint32_t maxCoord = 1 << subdivLevel;

    if(coord.w == maxCoord || coord.u == maxCoord || coord.v == maxCoord)
    {
        return 0;
    }

    uint32_t shift    = 0;
    uint32_t minCoord = coord.w | coord.u | coord.v;
    for(shift = 0; shift < subdivLevel; shift++)
    {
        if(minCoord & (1 << shift))
        {
            break;
        }
    }

    return subdivLevel - shift;
}

static inline uint32_t bird_getVertexLevelCoordIndex(BaryWUV_uint32 coord, uint32_t subdivLevel)
{
    if(subdivLevel == 0)
    {
        if(coord.w)
        {
            return 0;
        }
        else if(coord.u)
        {
            return 1;
        }
        else
        {
            return 2;
        }
    }

    // we need to descend appropriately until subdivLevel is reached
    BaryWUV_uint32 quadref = {coord.w & ~1, coord.u & ~1, coord.v & ~1};
    BaryWUV_uint32 rest    = {coord.w & 1, coord.u & 1, coord.v & 1};
    // edge 0 = AC split
    // edge 1 = CB split
    // edge 2 = BA split
    uint32_t edge  = rest.u == 0 ? 0 : ((rest.v == 1) ? 1 : 2);
    uint32_t index = bird_getTripletIndex(quadref.u, quadref.v, quadref.w, subdivLevel) * 3;
    index += edge;

    return index;
}

static uint32_t bird_getMicroVertexIndex(uint32_t u, uint32_t v, uint32_t subdivLevel)
{
    BaryWUV_uint32 coord = {(1u << subdivLevel) - u - v, u, v};

    // find out on which subdiv level our vertex sits on
    uint32_t level = bird_getVertexLevel(coord, subdivLevel);
    // adjust coord into level
    BaryWUV_uint32 base = {coord.w >> (subdivLevel - level), coord.u >> (subdivLevel - level), coord.v >> (subdivLevel - level)};
    // get index relative within level
    uint32_t index = bird_getVertexLevelCoordIndex(base, level);
    if(level)
    {
        // append previous levels' vertices
        index += baryValueFrequencyGetCount(ValueFrequency::ePerVertex, level - 1);
    }
    return index;
}

BARY_API void BARY_CALL baryBirdLayoutGetVertexLevelInfo(uint32_t u, uint32_t v, uint32_t subdivLevel, uint32_t* outLevel, uint32_t* outLevelCoordIndex)
{
    BaryWUV_uint32 coord = {(1u << subdivLevel) - u - v, u, v};

    // find out on which subdiv level our vertex sits on
    uint32_t level = bird_getVertexLevel(coord, subdivLevel);
    // adjust coord into level
    BaryWUV_uint32 base = {coord.w >> (subdivLevel - level), coord.u >> (subdivLevel - level), coord.v >> (subdivLevel - level)};
    // get index relative within level
    uint32_t index = bird_getVertexLevelCoordIndex(base, level);

    *outLevel           = level;
    *outLevelCoordIndex = index;
}

static uint32_t bird_getMicroTriangleIndex(uint32_t u, uint32_t v, uint32_t isUpperTriangle, uint32_t subdivLevel)
{
    // uvw.vw map to uv here
    uint32_t iu, iv, iw;

    iu = u;
    iv = v;
    iw = ~(iu + iv);
    if(isUpperTriangle)
        --iw;

    uint32_t b0 = ~(iu ^ iw);
    uint32_t t  = (iu ^ iv) & b0;
    uint32_t f  = bird_prefixEor(t);
    uint32_t b1 = ((f ^ iu) & ~b0) | t;

    return bird_interleaveBits(b0, b1);
}

// mapping from uv-coordinate of the barycentric micromesh grid to the value storage index
static uint32_t umajor_getMicroVertexIndex(uint32_t u, uint32_t v, uint32_t subdivLevel)
{
    uint32_t vtxPerEdge = (1 << subdivLevel) + 1;
    uint32_t x          = v;
    uint32_t y          = u;
    uint32_t trinum     = (y * (y + 1)) / 2;
    return y * (vtxPerEdge + 1) - trinum + x;
}

static uint32_t umajor_getMicroTriangleIndex(uint32_t u, uint32_t v, uint32_t isUpperTriangle, uint32_t subdivLevel)
{
    uint32_t triPerEdge = (1 << subdivLevel) * 2;
    uint32_t x          = v;
    uint32_t y          = u;
    uint32_t trinum     = y * y;
    return y * (triPerEdge)-trinum + x * 2 + (isUpperTriangle ? 1 : 0);
}

BARY_API uint32_t BARY_CALL baryValueLayoutGetIndex(ValueLayout order, ValueFrequency frequency, uint32_t u, uint32_t v, uint32_t isUpperTriangle, uint32_t subdivLevel)
{
    switch(frequency)
    {
    case ValueFrequency::ePerVertex:
        switch(order)
        {
        case ValueLayout::eTriangleUmajor:
            return umajor_getMicroVertexIndex(u, v, subdivLevel);
        case ValueLayout::eTriangleBirdCurve:
            return bird_getMicroVertexIndex(u, v, subdivLevel);
        default:
            return ~0;
        }
    case ValueFrequency::ePerTriangle:
        switch(order)
        {
        case ValueLayout::eTriangleUmajor:
            return umajor_getMicroTriangleIndex(u, v, isUpperTriangle, subdivLevel);
        case ValueLayout::eTriangleBirdCurve:
            return bird_getMicroTriangleIndex(u, v, isUpperTriangle, subdivLevel);
        default:
            return ~0;
        }
    default:
        return ~0;
    }
}

static inline BaryUV_uint32 baryUVsnapEdgeDecimation(BaryUV_uint32 coord, uint32_t subdivLevel, uint32_t edgeDecimationFlag)
{
    uint32_t baryMax = 1 << subdivLevel;
    uint32_t coord_w = baryMax - coord.u - coord.v;

    if(edgeDecimationFlag & 1 && coord.v == 0)
    {
        if(coord_w < baryMax / 2)
            return {(coord.u + 1) & ~1, 0};
        else
            return {(coord.u) & ~1, 0};
    }
    if(edgeDecimationFlag & 2 && coord_w == 0)
    {
        if(coord.u < baryMax / 2)
            return {(coord.u) & ~1, (coord.v + 1) & ~1};
        else
            return {(coord.u + 1) & ~1, (coord.v) & ~1};
    }
    if(edgeDecimationFlag & 4 && coord.u == 0)
    {
        if(coord.v < baryMax / 2)
            return {0, (coord.v) & ~1};
        else
            return {0, (coord.v + 1) & ~1};
    }
    return coord;
}

static inline BaryTriangle processTriangle(ValueLayout order, BaryUV_uint32 a, BaryUV_uint32 b, BaryUV_uint32 c, uint32_t subdivLevel, uint32_t edgeFlag)
{
    if(edgeFlag)
    {
        a = baryUVsnapEdgeDecimation(a, subdivLevel, edgeFlag);
        b = baryUVsnapEdgeDecimation(b, subdivLevel, edgeFlag);
        c = baryUVsnapEdgeDecimation(c, subdivLevel, edgeFlag);
    }

    BaryTriangle indices;
    indices.x = baryValueLayoutGetIndex(order, ValueFrequency::ePerVertex, a.u, a.v, 0, subdivLevel);
    indices.y = baryValueLayoutGetIndex(order, ValueFrequency::ePerVertex, b.u, b.v, 0, subdivLevel);
    indices.z = baryValueLayoutGetIndex(order, ValueFrequency::ePerVertex, c.u, c.v, 0, subdivLevel);

    return indices;
}

BARY_API Result BARY_CALL baryValueLayoutGetUVMesh(ValueLayout    order,
                                                   uint32_t       subdivLevel,
                                                   uint32_t       edgeFlag,
                                                   uint32_t       uvCount,
                                                   BaryUV_uint16* pUVs,
                                                   uint32_t       triangleCount,
                                                   uint32_t*      pTriangleIndices)
{
    uint32_t numVertices  = baryValueFrequencyGetCount(ValueFrequency::ePerVertex, subdivLevel);
    uint32_t numTriangles = baryValueFrequencyGetCount(ValueFrequency::ePerTriangle, subdivLevel);

    if(uvCount != numVertices || triangleCount != numTriangles)
        return Result::eErrorCount;

    uint32_t numSegmentsPerEdge = 1 << subdivLevel;
    uint32_t numVtxPerEdge      = numSegmentsPerEdge + 1;

    for(uint32_t u = 0; u < numVtxPerEdge; u++)
    {
        for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
        {
            uint32_t       idx       = baryValueLayoutGetIndex(order, ValueFrequency::ePerVertex, u, v, 0, subdivLevel);
            BaryUV_uint16* vertexUVs = pUVs + idx;
            vertexUVs->u             = uint16_t(u);
            vertexUVs->v             = uint16_t(v);
        }
    }

    for(uint32_t u = 0; u < numSegmentsPerEdge; u++)
    {
        for(uint32_t v = 0; v < numSegmentsPerEdge - u; v++)
        {
            {
                uint32_t     idx = baryValueLayoutGetIndex(order, ValueFrequency::ePerTriangle, u, v, 0, subdivLevel);
                BaryTriangle tri = processTriangle(order, {u, v}, {u + 1u, v}, {u, v + 1u}, subdivLevel, edgeFlag);
                uint32_t*    triIndices = pTriangleIndices + idx * 3;
                triIndices[0]           = tri.x;
                triIndices[1]           = tri.y;
                triIndices[2]           = tri.z;
            }
            if(v != numSegmentsPerEdge - u - 1)
            {
                uint32_t idx = baryValueLayoutGetIndex(order, ValueFrequency::ePerTriangle, u, v, 1, subdivLevel);
                // warning the order here was tuned for bird-curve, horizontal edge first, in theory need a different way of doing this
                BaryTriangle tri = processTriangle(order, {u + 1u, v + 1u}, {u, v + 1u}, {u + 1u, v}, subdivLevel, edgeFlag);
                uint32_t* triIndices = pTriangleIndices + idx * 3;
                triIndices[0]        = tri.x;
                triIndices[1]        = tri.y;
                triIndices[2]        = tri.z;
            }
        }
    }

    return Result::eSuccess;
}

//////////////////////////////////////////////////////////////////////////

BARY_API uint32_t BARY_CALL baryBlockFormatDispC1GetSubdivLevel(BlockFormatDispC1 blockFormat)
{
    switch(blockFormat)
    {
    case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
        return 3;
    case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
        return 4;
    case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
        return 5;
    default:
        return ~0;
    }
}

BARY_API uint32_t BARY_CALL baryBlockFormatDispC1GetMaxSubdivLevel()
{
    return 5;
}

BARY_API uint32_t BARY_CALL baryBlockFormatDispC1GetByteSize(BlockFormatDispC1 blockFormat)
{
    switch(blockFormat)
    {
    case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
        return 64;
    case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
        return 128;
    case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
        return 128;
    default:
        return ~0;
    }
}

BARY_API uint32_t BARY_CALL baryBlockFormatDispC1GetBlockCount(BlockFormatDispC1 blockFormat, uint32_t baseSubdivLevel)
{
    uint32_t blockSubdivLevel = baryBlockFormatDispC1GetSubdivLevel(blockFormat);
    if(blockSubdivLevel == ~uint32_t(0))
        return 0;


    uint32_t splitSubdiv = baseSubdivLevel > blockSubdivLevel ? baseSubdivLevel - blockSubdivLevel : 0;
    return 1 << (splitSubdiv * 2);
}

static inline void setVertices(BlockTriangle* tri, BaryUV_uint32 w, BaryUV_uint32 u, BaryUV_uint32 v)
{
    tri->vertices[0].u = uint16_t(w.u);
    tri->vertices[0].v = uint16_t(w.v);
    tri->vertices[1].u = uint16_t(u.u);
    tri->vertices[1].v = uint16_t(u.v);
    tri->vertices[2].u = uint16_t(v.u);
    tri->vertices[2].v = uint16_t(v.v);
    tri->signBits      = ((u.u > w.u) ? 1 : 0) | ((v.v > w.v) ? 2 : 0);
}

BARY_API void BARY_CALL baryBlockTriangleSplitDispC1(const BlockTriangle* inTri, BlockTriangle* outTris, uint32_t outStride)
{
    /* 
    // **********************
    // *         C(v)       *
    // *        / \         *
    // *       / V \        *
    // *      vw _ uv       *
    // *     / \ M / \      *
    // *    / W \ / U \     *
    // * A(w) __ uw __ B(u) *
    // **********************
    */

    const uint32_t triW = 0 * outStride;
    const uint32_t triM = 1 * outStride;
    const uint32_t triU = 2 * outStride;
    const uint32_t triV = 3 * outStride;

    // flip state
    outTris[triW].flipped = inTri->flipped;
    outTris[triM].flipped = inTri->flipped ^ 1;
    outTris[triU].flipped = inTri->flipped;
    outTris[triV].flipped = inTri->flipped ^ 1;

    BaryUV_uint32 w = {inTri->vertices[0].u, inTri->vertices[0].v};
    BaryUV_uint32 u = {inTri->vertices[1].u, inTri->vertices[1].v};
    BaryUV_uint32 v = {inTri->vertices[2].u, inTri->vertices[2].v};

    BaryUV_uint32 uw = {(u.u + w.u) / 2, (u.v + w.v) / 2};
    BaryUV_uint32 uv = {(u.u + v.u) / 2, (u.v + v.v) / 2};
    BaryUV_uint32 vw = {(v.u + w.u) / 2, (v.v + w.v) / 2};

    setVertices(&outTris[triW], w, uw, vw);
    setVertices(&outTris[triM], vw, uv, uw);
    setVertices(&outTris[triU], uw, u, uv);
    setVertices(&outTris[triV], uv, vw, v);

    const uint32_t baseEdge0      = (inTri->baseEdgeIndices >> 0) & 3;
    const uint32_t baseEdge1      = (inTri->baseEdgeIndices >> 2) & 3;
    const uint32_t baseEdge2      = (inTri->baseEdgeIndices >> 4) & 3;
    const uint32_t baseEdgeUnused = 3;

    outTris[triW].baseEdgeIndices = (baseEdge0 << 0) | (baseEdgeUnused << 2) | (baseEdge2 << 4);
    outTris[triM].baseEdgeIndices = (baseEdgeUnused << 0) | (baseEdgeUnused << 2) | (baseEdgeUnused << 4);
    outTris[triU].baseEdgeIndices = (baseEdge0 << 0) | (baseEdge1 << 2) | (baseEdgeUnused << 4);
    outTris[triV].baseEdgeIndices = (baseEdgeUnused << 0) | (baseEdge2 << 2) | (baseEdge1 << 4);
}

BARY_API Result BARY_CALL baryBlockFormatDispC1GetBlockTriangles(BlockFormatDispC1 blockFormat,
                                                                 uint32_t          baseSubdivLevel,
                                                                 uint32_t          blockTrisCount,
                                                                 BlockTriangle*    blockTris)
{
    uint32_t blockSubdivLevel = baryBlockFormatDispC1GetSubdivLevel(blockFormat);
    uint32_t blockByteSize    = baryBlockFormatDispC1GetByteSize(blockFormat);

    if(blockSubdivLevel == ~uint32_t(0))
    {
        return Result::eErrorBlockFormat;
    }

    if(blockTrisCount != baryBlockFormatDispC1GetBlockCount(blockFormat, baseSubdivLevel))
    {
        return Result::eErrorCount;
    }

    uint32_t selfSubdiv  = baseSubdivLevel < blockSubdivLevel ? blockSubdivLevel : baseSubdivLevel;
    uint32_t splitSubdiv = selfSubdiv - blockSubdivLevel;

    uint16_t splitMaxCoord       = (1 << selfSubdiv);
    blockTris[0].baseEdgeIndices = (0 << 2) | (1 << 2) | (2 << 4);
    blockTris[0].flipped         = 0;
    setVertices(&blockTris[0], {0, 0}, {splitMaxCoord, 0}, {0, splitMaxCoord});

    uint32_t stride = blockTrisCount;
    for(uint32_t level = 0; level < splitSubdiv; level++)
    {
        uint32_t strideNext = stride / 4;
        uint32_t levelCount = baryValueFrequencyGetCount(ValueFrequency::ePerTriangle, level);
        for(uint32_t i = 0; i < levelCount; i++)
        {
            // load previous split state
            BlockTriangle tri = blockTris[i * stride];
            // save new split states
            baryBlockTriangleSplitDispC1(&tri, blockTris + (i * stride), strideNext);
        }

        stride = strideNext;
    }

    for(uint32_t i = 0; i < blockTrisCount; i++)
    {
        blockTris[i].blockByteOffset = i * blockByteSize;
    }

    return Result::eSuccess;
}

//////////////////////////////////////////////////////////////////////////

BARY_API const char* BARY_CALL baryResultGetName(Result result)
{
#define HANDLE_TYPE(enum)                                                                                              \
    case Result::enum:                                                                                                 \
        return "" #enum "";

    switch(result)
    {
        HANDLE_TYPE(eSuccess)
        HANDLE_TYPE(eErrorUnknown)
        HANDLE_TYPE(eErrorFileSize)
        HANDLE_TYPE(eErrorRange)
        HANDLE_TYPE(eErrorIndex)
        HANDLE_TYPE(eErrorOffset)
        HANDLE_TYPE(eErrorCount)
        HANDLE_TYPE(eErrorSize)
        HANDLE_TYPE(eErrorAlignment)
        HANDLE_TYPE(eErrorVersionFormat)
        HANDLE_TYPE(eErrorVersion)
        HANDLE_TYPE(eErrorFormat)
        HANDLE_TYPE(eErrorBlockFormat)
        HANDLE_TYPE(eErrorValue)
        HANDLE_TYPE(eErrorFlag)
        HANDLE_TYPE(eErrorOffsetOrder)
        HANDLE_TYPE(eErrorIndexOrder)
        HANDLE_TYPE(eErrorOffsetAlignment)
        HANDLE_TYPE(eErrorMissingProperty)
        HANDLE_TYPE(eErrorDuplicateProperty)
        HANDLE_TYPE(eErrorInvalidPropertyCombination)
        HANDLE_TYPE(eErrorPropertyMismatch)
        HANDLE_TYPE(eErrorSupercompression)
        HANDLE_TYPE(eErrorIO)
        HANDLE_TYPE(eErrorOutOfMemory)
    default:
        return "unknown";
    }
#undef HANDLE_TYPE
}

BARY_API uint32_t BARY_CALL baryStandardPropertyGetElementSize(StandardPropertyType type)
{
    switch(type)
    {
    case StandardPropertyType::eValues:
        return sizeof(ValuesInfo);
    case StandardPropertyType::eGroups:
        return sizeof(Group);
    case StandardPropertyType::eTriangles:
        return sizeof(Triangle);
    case StandardPropertyType::eHistogramEntries:
        return sizeof(HistogramEntry);
    case StandardPropertyType::eGroupHistograms:
        return sizeof(GroupHistogramRange);
    case StandardPropertyType::eMeshGroups:
        return sizeof(MeshGroup);
    case StandardPropertyType::eMeshHistogramEntries:
        return sizeof(MeshHistogramEntry);
    case StandardPropertyType::eMeshGroupHistograms:
        return sizeof(MeshGroupHistogramRange);
    case StandardPropertyType::eMeshDisplacementDirectionBounds:
        return sizeof(MeshDisplacementDirectionBoundsInfo);
    case StandardPropertyType::eMeshDisplacementDirections:
        return sizeof(MeshDisplacementDirectionsInfo);
    case StandardPropertyType::eMeshPositions:
        return sizeof(MeshPositionsInfo);
    case StandardPropertyType::eMeshTriangleIndices:
        return sizeof(MeshTriangleIndicesInfo);
    case StandardPropertyType::eMeshTriangleMappings:
        return sizeof(MeshTriangleMappingsInfo);
    case StandardPropertyType::eMeshTriangleFlags:
        return sizeof(MeshTriangleFlagsInfo);
    case StandardPropertyType::eTriangleMinMaxs:
        return sizeof(TriangleMinMaxsInfo);
    default:
        return 0;
    }
}

BARY_API uint32_t BARY_CALL baryStandardPropertyGetInfoSize(StandardPropertyType type)
{
    switch(type)
    {
    case StandardPropertyType::eValues:
        return sizeof(ValuesInfo);
    case StandardPropertyType::eMeshDisplacementDirectionBounds:
        return sizeof(MeshDisplacementDirectionBoundsInfo);
    case StandardPropertyType::eMeshDisplacementDirections:
        return sizeof(MeshDisplacementDirectionsInfo);
    case StandardPropertyType::eMeshPositions:
        return sizeof(MeshPositionsInfo);
    case StandardPropertyType::eMeshTriangleIndices:
        return sizeof(MeshTriangleIndicesInfo);
    case StandardPropertyType::eMeshTriangleMappings:
        return sizeof(MeshTriangleMappingsInfo);
    case StandardPropertyType::eMeshTriangleFlags:
        return sizeof(MeshTriangleFlagsInfo);
    case StandardPropertyType::eTriangleMinMaxs:
        return sizeof(TriangleMinMaxsInfo);
    default:
        return 0;
    }
}

BARY_API const char* BARY_CALL baryStandardPropertyGetName(StandardPropertyType type)
{
#define HANDLE_TYPE(enum)                                                                                              \
    case StandardPropertyType::enum:                                                                                   \
        return "" #enum "";

    switch(type)
    {
        HANDLE_TYPE(eUnknown)
        HANDLE_TYPE(eValues)
        HANDLE_TYPE(eGroups)
        HANDLE_TYPE(eTriangles)
        HANDLE_TYPE(eTriangleMinMaxs)
        HANDLE_TYPE(eTriangleUncompressedMips)
        HANDLE_TYPE(eHistogramEntries)
        HANDLE_TYPE(eGroupHistograms)
        HANDLE_TYPE(eGroupUncompressedMips)
        HANDLE_TYPE(eMeshGroups)
        HANDLE_TYPE(eMeshHistogramEntries)
        HANDLE_TYPE(eMeshGroupHistograms)
        HANDLE_TYPE(eMeshDisplacementDirections)
        HANDLE_TYPE(eMeshDisplacementDirectionBounds)
        HANDLE_TYPE(eMeshTriangleMappings)
        HANDLE_TYPE(eMeshTriangleFlags)
        HANDLE_TYPE(eMeshPositions)
        HANDLE_TYPE(eMeshTriangleIndices)
        HANDLE_TYPE(eUncompressedMips)
    default:
        return "unknown";
    }
#undef HANDLE_TYPE
}

BARY_API PropertyIdentifier BARY_CALL baryStandardPropertyGetIdentifier(StandardPropertyType type)
{
#define HANDLE_TYPE(enum)                                                                                              \
    case StandardPropertyType::enum:                                                                                   \
        return baryMakeStandardPropertyIdentifierT<StandardPropertyType::enum>();
    switch(type)
    {
        HANDLE_TYPE(eValues)
        HANDLE_TYPE(eGroups)
        HANDLE_TYPE(eTriangles)
        HANDLE_TYPE(eTriangleMinMaxs)
        HANDLE_TYPE(eTriangleUncompressedMips)
        HANDLE_TYPE(eHistogramEntries)
        HANDLE_TYPE(eGroupHistograms)
        HANDLE_TYPE(eGroupUncompressedMips)
        HANDLE_TYPE(eMeshGroups)
        HANDLE_TYPE(eMeshHistogramEntries)
        HANDLE_TYPE(eMeshGroupHistograms)
        HANDLE_TYPE(eMeshDisplacementDirections)
        HANDLE_TYPE(eMeshDisplacementDirectionBounds)
        HANDLE_TYPE(eMeshTriangleMappings)
        HANDLE_TYPE(eMeshTriangleFlags)
        HANDLE_TYPE(eMeshPositions)
        HANDLE_TYPE(eMeshTriangleIndices)
        HANDLE_TYPE(eUncompressedMips)

    case StandardPropertyType::eUnknown:
    default:
        return {{0, 0, 0, 0}};
    }
#undef HANDLE_TYPE
}

BARY_API StandardPropertyType BARY_CALL baryPropertyGetStandardType(PropertyIdentifier identifier)
{
    struct Compare
    {
        StandardPropertyType type;
        PropertyIdentifier   identifier;
    };

#define HANDLE_TYPE(enum)                                                                                              \
    {StandardPropertyType::enum, baryMakeStandardPropertyIdentifierT<StandardPropertyType::enum>()}

    Compare compares[] = {
        HANDLE_TYPE(eValues),
        HANDLE_TYPE(eGroups),
        HANDLE_TYPE(eTriangles),
        HANDLE_TYPE(eTriangleMinMaxs),
        HANDLE_TYPE(eTriangleUncompressedMips),
        HANDLE_TYPE(eHistogramEntries),
        HANDLE_TYPE(eGroupHistograms),
        HANDLE_TYPE(eGroupUncompressedMips),
        HANDLE_TYPE(eMeshGroups),
        HANDLE_TYPE(eMeshHistogramEntries),
        HANDLE_TYPE(eMeshGroupHistograms),
        HANDLE_TYPE(eMeshDisplacementDirections),
        HANDLE_TYPE(eMeshDisplacementDirectionBounds),
        HANDLE_TYPE(eMeshTriangleMappings),
        HANDLE_TYPE(eMeshTriangleFlags),
        HANDLE_TYPE(eMeshPositions),
        HANDLE_TYPE(eMeshTriangleIndices),
        HANDLE_TYPE(eUncompressedMips),
    };

#undef HANDLE_TYPE

    for(size_t i = 0; i < sizeof(compares) / sizeof(compares[0]); i++)
    {
        if(baryPropertyIsEqual(identifier, compares[i].identifier))
        {
            return compares[i].type;
        }
    }

    return StandardPropertyType::eUnknown;
}

BARY_API const char* BARY_CALL baryFormatGetName(Format format)
{
    switch(format)
    {
    case Format::eUndefined:
        return "eUndefined";
    case Format::eR8_unorm:
        return "eR8_unorm";
    case Format::eR8_snorm:
        return "eR8_snorm";
    case Format::eR8_uint:
        return "eR8_uint";
    case Format::eR8_sint:
        return "eR8_sint";
    case Format::eRG8_unorm:
        return "eRG8_unorm";
    case Format::eRG8_snorm:
        return "eRG8_snorm";
    case Format::eRG8_uint:
        return "eRG8_uint";
    case Format::eRG8_sint:
        return "eRG8_sint";
    case Format::eRGB8_unorm:
        return "eRGB8_unorm";
    case Format::eRGB8_snorm:
        return "eRGB8_snorm";
    case Format::eRGB8_uint:
        return "eRGB8_uint";
    case Format::eRGB8_sint:
        return "eRGB8_sint";
    case Format::eRGBA8_unorm:
        return "eRGBA8_unorm";
    case Format::eRGBA8_snorm:
        return "eRGBA8_snorm";
    case Format::eRGBA8_uint:
        return "eRGBA8_uint";
    case Format::eRGBA8_sint:
        return "eRGBA8_sint";
    case Format::eR16_unorm:
        return "eR16_unorm";
    case Format::eR16_snorm:
        return "eR16_snorm";
    case Format::eR16_uint:
        return "eR16_uint";
    case Format::eR16_sint:
        return "eR16_sint";
    case Format::eR16_sfloat:
        return "eR16_sfloat";
    case Format::eRG16_unorm:
        return "eRG16_unorm";
    case Format::eRG16_snorm:
        return "eRG16_snorm";
    case Format::eRG16_uint:
        return "eRG16_uint";
    case Format::eRG16_sint:
        return "eRG16_sint";
    case Format::eRG16_sfloat:
        return "eRG16_sfloat";
    case Format::eRGB16_unorm:
        return "eRGB16_unorm";
    case Format::eRGB16_snorm:
        return "eRGB16_snorm";
    case Format::eRGB16_uint:
        return "eRGB16_uint";
    case Format::eRGB16_sint:
        return "eRGB16_sint";
    case Format::eRGB16_sfloat:
        return "eRGB16_sfloat";
    case Format::eRGBA16_unorm:
        return "eRGBA16_unorm";
    case Format::eRGBA16_snorm:
        return "eRGBA16_snorm";
    case Format::eRGBA16_uint:
        return "eRGBA16_uint";
    case Format::eRGBA16_sint:
        return "eRGBA16_sint";
    case Format::eRGBA16_sfloat:
        return "eRGBA16_sfloat";
    case Format::eR32_uint:
        return "eR32_uint";
    case Format::eR32_sint:
        return "eR32_sint";
    case Format::eR32_sfloat:
        return "eR32_sfloat";
    case Format::eRG32_uint:
        return "eRG32_uint";
    case Format::eRG32_sint:
        return "eRG32_sint";
    case Format::eRG32_sfloat:
        return "eRG32_sfloat";
    case Format::eRGB32_uint:
        return "eRGB32_uint";
    case Format::eRGB32_sint:
        return "eRGB32_sint";
    case Format::eRGB32_sfloat:
        return "eRGB32_sfloat";
    case Format::eRGBA32_uint:
        return "eRGBA32_uint";
    case Format::eRGBA32_sint:
        return "eRGBA32_sint";
    case Format::eRGBA32_sfloat:
        return "eRGBA32_sfloat";
    case Format::eOpaC1_rx_uint_block:
        return "eOpaC1_rx_uint_block";
    case Format::eDispC1_r11_unorm_block:
        return "eDispC1_r11_unorm_block";
    case Format::eR11_unorm_pack16:
        return "eR11_unorm_pack16";
    case Format::eR11_unorm_packed_align32:
        return "eR11_unorm_packed_align32";
    default:
        return "unknown";
    }
}

BARY_API const char* BARY_CALL baryValueFrequencyGetName(ValueFrequency freq)
{
    switch(freq)
    {
    case ValueFrequency::eUndefined:
        return "eUndefined";
    case ValueFrequency::ePerTriangle:
        return "ePerTriangle";
    case ValueFrequency::ePerVertex:
        return "ePerTriangle";
    default:
        return "unknown";
    }
}

BARY_API const char* BARY_CALL baryValueLayoutGetName(ValueLayout layout)
{
    switch(layout)
    {
    case ValueLayout::eUndefined:
        return "eUndefined";
    case ValueLayout::eTriangleBirdCurve:
        return "eTriangleBirdCurve";
    case ValueLayout::eTriangleUmajor:
        return "eTriangleUmajor";
    default:
        return "unknown";
    }
}


//////////////////////////////////////////////////////////////////////////
// validation

static inline Result returnPropError(Result result, StandardPropertyType* errorPropertyType, StandardPropertyType propType)
{
    if(errorPropertyType)
    {
        *errorPropertyType = propType;
    }
    return result;
}

// all properties that constitute the final file should be provided
// only standard properties can be validated here
BARY_API Result BARY_CALL baryValidateStandardProperties(uint32_t                   propertyCount,
                                                         const PropertyStorageInfo* propertyStorageInfos,
                                                         uint64_t                   validationFlags,
                                                         StandardPropertyType*      errorPropertyType)
{
    const ValuesInfo*         valueInfo            = nullptr;
    uint64_t                  valueDataSize        = 0;
    const Group*              groups               = nullptr;
    uint64_t                  groupsSize           = 0;
    const Triangle*           triangles            = nullptr;
    uint64_t                  trianglesSize        = 0;
    const HistogramEntry*     histoEntries         = nullptr;
    uint64_t                  histoEntriesSize     = 0;
    const MeshHistogramEntry* meshHistoEntries     = nullptr;
    uint64_t                  meshHistoEntriesSize = 0;

    uint64_t propertyUsage = 0;

    // find generic ones that have
    for(uint32_t p = 0; p < propertyCount; p++)
    {
        uint64_t    propSize     = propertyStorageInfos[p].dataSize;
        uint64_t    propInfoSize = propertyStorageInfos[p].infoSize;
        const void* propData     = propertyStorageInfos[p].data;
        const void* propDataInfo = propertyStorageInfos[p].info;

        StandardPropertyType propType = baryPropertyGetStandardType(propertyStorageInfos[p].identifier);
        if(propType == StandardPropertyType::eUnknown)
            continue;

        // developer might give us the whole data block in one already
        // or info/payload separate
        propDataInfo = propDataInfo ? propDataInfo : propData;

        uint64_t bitFlag = uint64_t(1) << uint32_t(propType);

        if(propertyUsage & bitFlag)
        {
            return returnPropError(Result::eErrorDuplicateProperty, errorPropertyType, propType);
        }

        propertyUsage |= bitFlag;

        uint64_t expectedInfoSize  = 0;
        uint64_t expectedAlignment = 4;

        switch(propType)
        {
        case StandardPropertyType::eValues:
            valueInfo        = reinterpret_cast<const ValuesInfo*>(propDataInfo);
            expectedInfoSize = sizeof(ValuesInfo);
            break;
        case StandardPropertyType::eGroups:
            groups     = reinterpret_cast<const Group*>(propData);
            groupsSize = propSize;
            break;
        case StandardPropertyType::eTriangles:
            triangles     = reinterpret_cast<const Triangle*>(propData);
            trianglesSize = propSize;
            break;
        case StandardPropertyType::eTriangleMinMaxs:
            expectedInfoSize = sizeof(TriangleMinMaxsInfo);
            break;
        case StandardPropertyType::eTriangleUncompressedMips:
            break;
        case StandardPropertyType::eUncompressedMips:
            expectedInfoSize = sizeof(UncompressedMipsInfo);
            break;
        case StandardPropertyType::eGroupUncompressedMips:
            break;
        case StandardPropertyType::eHistogramEntries:
            histoEntries     = reinterpret_cast<const HistogramEntry*>(propData);
            histoEntriesSize = propSize;
            break;
        case StandardPropertyType::eGroupHistograms:
            break;
        case StandardPropertyType::eMeshGroups:
            break;
        case StandardPropertyType::eMeshHistogramEntries:
            meshHistoEntries     = reinterpret_cast<const MeshHistogramEntry*>(propData);
            meshHistoEntriesSize = propSize;
            break;
        case StandardPropertyType::eMeshGroupHistograms:
            break;
        case StandardPropertyType::eMeshDisplacementDirections:
            expectedInfoSize = sizeof(MeshDisplacementDirectionsInfo);
            break;
        case StandardPropertyType::eMeshDisplacementDirectionBounds:
            expectedInfoSize = sizeof(MeshDisplacementDirectionBoundsInfo);
            break;
        case StandardPropertyType::eMeshTriangleMappings:
            expectedInfoSize = sizeof(MeshTriangleMappingsInfo);
            break;
        case StandardPropertyType::eMeshTriangleFlags:
            expectedInfoSize = sizeof(MeshTriangleFlagsInfo);
            break;
        case StandardPropertyType::eMeshPositions:
            expectedInfoSize = sizeof(MeshPositionsInfo);
            break;
        case StandardPropertyType::eMeshTriangleIndices:
            expectedInfoSize = sizeof(MeshTriangleIndicesInfo);
            break;
        }

        if(!baryPropertyStorageHasValidSize(propertyStorageInfos[p], expectedInfoSize))
        {
            return returnPropError(Result::eErrorSize, errorPropertyType, propType);
        }

        switch(propType)
        {
        case StandardPropertyType::eValues:
            if(propInfoSize)
            {
                valueDataSize = propSize;
            }
            else
            {
                valueDataSize = propSize - baryPayloadGetOffset(sizeof(ValuesInfo), valueInfo->valueByteAlignment);
            }
            expectedAlignment = valueInfo->valueByteAlignment;
            break;
        case StandardPropertyType::eTriangleMinMaxs:
            expectedAlignment = reinterpret_cast<const TriangleMinMaxsInfo*>(propDataInfo)->elementByteAlignment;
            break;
        case StandardPropertyType::eUncompressedMips:
            break;
        case StandardPropertyType::eMeshDisplacementDirections:
            expectedAlignment = reinterpret_cast<const MeshDisplacementDirectionsInfo*>(propDataInfo)->elementByteAlignment;
            break;
        case StandardPropertyType::eMeshDisplacementDirectionBounds:
            expectedAlignment = reinterpret_cast<const MeshDisplacementDirectionBoundsInfo*>(propDataInfo)->elementByteAlignment;
            break;
        case StandardPropertyType::eMeshTriangleMappings:
            expectedAlignment = reinterpret_cast<const MeshTriangleMappingsInfo*>(propDataInfo)->elementByteAlignment;
            break;
        case StandardPropertyType::eMeshTriangleFlags:
            expectedAlignment = reinterpret_cast<const MeshTriangleFlagsInfo*>(propDataInfo)->elementByteAlignment;
            break;
        case StandardPropertyType::eMeshPositions:
            expectedAlignment = reinterpret_cast<const MeshPositionsInfo*>(propDataInfo)->elementByteAlignment;
            break;
        case StandardPropertyType::eMeshTriangleIndices:
            expectedAlignment = reinterpret_cast<const MeshTriangleIndicesInfo*>(propDataInfo)->elementByteAlignment;
            break;
        }


        if(!baryPropertyStorageHasValidPadding(propertyStorageInfos[p], expectedInfoSize, expectedAlignment))
        {
            return returnPropError(Result::eErrorSize, errorPropertyType, propType);
        }
    }

    if(!valueInfo)
    {
        return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eValues);
    }

    if(!groups)
    {
        return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eGroups);
    }

    if(!triangles)
    {
        return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eTriangles);
    }

    if(valueInfo->valueByteSize == 0)
    {
        return returnPropError(Result::eErrorSize, errorPropertyType, StandardPropertyType::eValues);
    }

    if(valueInfo->valueFormat == Format::eDispC1_r11_unorm_block)
    {
        if(valueInfo->valueByteAlignment != 128)
        {
            return returnPropError(Result::eErrorAlignment, errorPropertyType, StandardPropertyType::eValues);
        }
    }
    else if(valueInfo->valueByteAlignment < 4)
    {
        return returnPropError(Result::eErrorAlignment, errorPropertyType, StandardPropertyType::eValues);
    }

    if(uint64_t(valueInfo->valueByteSize) * valueInfo->valueCount != valueDataSize)
    {
        return returnPropError(Result::eErrorCount, errorPropertyType, StandardPropertyType::eValues);
    }

    uint64_t valueCount            = valueInfo->valueCount;
    uint64_t groupCount            = groupsSize / sizeof(Group);
    uint64_t triangleCount         = trianglesSize / sizeof(Triangle);
    uint64_t histoEntriesCount     = histoEntriesSize / sizeof(HistogramEntry);
    uint64_t meshHistoEntriesCount = meshHistoEntriesSize / sizeof(MeshHistogramEntry);

    if(groupsSize % sizeof(Group) != 0)
    {
        return returnPropError(Result::eErrorSize, errorPropertyType, StandardPropertyType::eGroups);
    }

    if(trianglesSize % sizeof(Triangle) != 0)
    {
        return returnPropError(Result::eErrorSize, errorPropertyType, StandardPropertyType::eTriangles);
    }

    if(histoEntriesSize % sizeof(HistogramEntry) != 0)
    {
        return returnPropError(Result::eErrorSize, errorPropertyType, StandardPropertyType::eHistogramEntries);
    }

    // go over groups
    for(uint64_t g = 0; g < groupCount; g++)
    {
        const Group* currGroup = groups + g;
        if(g > 0)
        {
            const Group* prevGroup = groups + (g - 1);
            if(currGroup->triangleFirst < (prevGroup->triangleFirst + prevGroup->triangleCount))
                return returnPropError(Result::eErrorOffsetOrder, errorPropertyType, StandardPropertyType::eGroups);
            if(currGroup->valueFirst < (prevGroup->valueFirst + prevGroup->valueCount))
                return returnPropError(Result::eErrorOffsetOrder, errorPropertyType, StandardPropertyType::eGroups);
        }
        // Each group's triangles and values must be in range.
        if(currGroup->triangleFirst > triangleCount || currGroup->triangleCount > (triangleCount - currGroup->triangleFirst))
            return returnPropError(Result::eErrorRange, errorPropertyType, StandardPropertyType::eGroups);
        if(currGroup->valueFirst > valueCount || currGroup->valueCount > (valueCount - currGroup->valueFirst))
            return returnPropError(Result::eErrorRange, errorPropertyType, StandardPropertyType::eGroups);
    }


    ValueFrequency valueFrequency = valueInfo->valueFrequency;
    Format         valueFormat    = valueInfo->valueFormat;

    for(uint64_t g = 0; g < groupCount && (validationFlags & eValidationFlagArrayContents); g++)
    {
        const Group*    group          = groups + g;
        const Triangle* groupTriangles = triangles + (group->triangleFirst);

        uint32_t nextStart = ~0;

        for(uint32_t t = 0; t < group->triangleCount; t++)
        {
            const Triangle* tri = groupTriangles + t;

            uint32_t dataRange;

            // if the compression format is unknown allow possibility to skip detailed dataRange computation
            // use 0 means we only test that things are ascending and at least the start offset is within
            // given bounds
            // Also useful for sub-triangles using a separate extensions
            if(validationFlags & eValidationFlagTriangleValueRange)
            {
                if(valueFormat == Format::eDispC1_r11_unorm_block)
                {
                    uint32_t splitSubdiv = 0;
                    uint32_t byteSize    = 0;

                    switch(tri->blockFormatDispC1)
                    {
                    case BlockFormatDispC1::eR11_unorm_lvl3_pack512:
                        splitSubdiv = 3;
                        byteSize    = 512 / 8;
                        break;
                    case BlockFormatDispC1::eR11_unorm_lvl4_pack1024:
                        splitSubdiv = 4;
                        byteSize    = 1024 / 8;
                        break;
                    case BlockFormatDispC1::eR11_unorm_lvl5_pack1024:
                        splitSubdiv = 5;
                        byteSize    = 1024 / 8;
                        break;
                    default:
                        return returnPropError(Result::eErrorBlockFormat, errorPropertyType, StandardPropertyType::eTriangles);
                    }

                    if(tri->valuesOffset % byteSize)
                    {
                        return returnPropError(Result::eErrorOffsetAlignment, errorPropertyType, StandardPropertyType::eTriangles);
                    }

                    dataRange = baryValueFrequencyGetCount(ValueFrequency::ePerTriangle,
                                                           tri->subdivLevel > splitSubdiv ? tri->subdivLevel - splitSubdiv : 0)
                                * byteSize;
                }
                else if(valueFormat == Format::eOpaC1_rx_uint_block)
                {
                    uint32_t count = 0;
                    switch(tri->blockFormatOpaC1)
                    {
                    case BlockFormatOpaC1::eR1_uint_x8:
                        count = 8;
                        break;
                    case BlockFormatOpaC1::eR2_uint_x4:
                        count = 4;
                        break;
                    default:
                        return returnPropError(Result::eErrorBlockFormat, errorPropertyType, StandardPropertyType::eTriangles);
                    }

                    dataRange = (baryValueFrequencyGetCount(valueFrequency, tri->subdivLevel) + count - 1) / count;
                }
                else
                {
                    dataRange = baryValueFrequencyGetCount(valueFrequency, tri->subdivLevel);
                }
            }
            else
            {
                dataRange = 0;
            }

            if((tri->valuesOffset + dataRange) > (group->valueCount))
                return returnPropError(Result::eErrorRange, errorPropertyType, StandardPropertyType::eTriangles);
            if(nextStart != ~uint32_t(0) && tri->valuesOffset < nextStart)
                return returnPropError(Result::eErrorOffsetOrder, errorPropertyType, StandardPropertyType::eTriangles);

            nextStart = tri->valuesOffset + dataRange;
        }
    }

    if(propertyUsage
       & ((1ull << uint32_t(StandardPropertyType::eUncompressedMips)) | (1ull << uint32_t(StandardPropertyType::eTriangleUncompressedMips))
          | (1ull << uint32_t(StandardPropertyType::eGroupUncompressedMips))))
    {
        if(!(propertyUsage & (1ull << uint32_t(StandardPropertyType::eUncompressedMips))))
        {
            return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eUncompressedMips);
        }
        if(!(propertyUsage & (1ull << uint32_t(StandardPropertyType::eTriangleUncompressedMips))))
        {
            return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eTriangleUncompressedMips);
        }
        if(!(propertyUsage & (1ull << uint32_t(StandardPropertyType::eGroupUncompressedMips))))
        {
            return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eGroupUncompressedMips);
        }
    }

    for(uint32_t p = 0; p < propertyCount; p++)
    {
        uint64_t    propSize     = propertyStorageInfos[p].dataSize;
        uint64_t    propInfoSize = propertyStorageInfos[p].infoSize;
        const void* propData     = propertyStorageInfos[p].data;
        const void* propDataInfo = propertyStorageInfos[p].info;

        StandardPropertyType propType = baryPropertyGetStandardType(propertyStorageInfos[p].identifier);
        if(propType == StandardPropertyType::eUnknown)
            continue;

        propDataInfo = propDataInfo ? propDataInfo : propData;

        switch(propType)
        {
        case StandardPropertyType::eGroupHistograms: {
            const GroupHistogramRange* histoRanges = reinterpret_cast<const GroupHistogramRange*>(propData);

            if(!histoEntries)
            {
                return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eHistogramEntries);
            }
            if(propSize % sizeof(GroupHistogramRange))
            {
                return returnPropError(Result::eErrorSize, errorPropertyType, propType);
            }
            uint64_t propCount = (propSize / sizeof(GroupHistogramRange));
            if(propCount != groupCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
            for(uint64_t i = 0; i < propCount; i++)
            {
                if(!((histoRanges[i].entryFirst < histoEntriesCount)
                     && (histoRanges[i].entryCount <= histoEntriesCount - histoRanges[i].entryFirst)))
                {
                    return returnPropError(Result::eErrorRange, errorPropertyType, propType);
                }
            }
        }
        break;
        case StandardPropertyType::eMeshGroupHistograms: {
            const MeshGroupHistogramRange* histoRanges = reinterpret_cast<const MeshGroupHistogramRange*>(propData);

            if(!meshHistoEntries)
            {
                return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eMeshHistogramEntries);
            }
            if(propSize % sizeof(MeshGroupHistogramRange))
            {
                return returnPropError(Result::eErrorSize, errorPropertyType, propType);
            }
            uint64_t propCount = (propSize / sizeof(MeshGroupHistogramRange));
            for(uint64_t i = 0; i < propCount; i++)
            {
                if(!((histoRanges[i].entryFirst < meshHistoEntriesCount)
                     && (histoRanges[i].entryCount <= meshHistoEntriesCount - histoRanges[i].entryFirst)))
                {
                    return returnPropError(Result::eErrorRange, errorPropertyType, propType);
                }
            }
        }
        break;
        case StandardPropertyType::eMeshDisplacementDirections: {
            const MeshDisplacementDirectionsInfo* meshInfo = reinterpret_cast<const MeshDisplacementDirectionsInfo*>(propDataInfo);

            switch(meshInfo->elementFormat)
            {
            case Format::eRGB16_sfloat:
                if(meshInfo->elementByteSize != 6)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 4)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            case Format::eRGBA16_sfloat:
                if(meshInfo->elementByteSize != 8)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 8)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            case Format::eRGB32_sfloat:
                if(meshInfo->elementByteSize != 12)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 4)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            case Format::eRGBA32_sfloat:
                if(meshInfo->elementByteSize != 16)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 16)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            default:
                return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
            }

            uint64_t count =
                ((propSize - (!propInfoSize ? baryPayloadGetOffset(sizeof(MeshDisplacementDirectionsInfo), meshInfo->elementByteAlignment) : 0))
                 / meshInfo->elementByteSize);
            if(count != meshInfo->elementCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
        }
        break;
        case StandardPropertyType::eMeshDisplacementDirectionBounds: {
            const MeshDisplacementDirectionBoundsInfo* meshInfo =
                reinterpret_cast<const MeshDisplacementDirectionBoundsInfo*>(propDataInfo);

            switch(meshInfo->elementFormat)
            {
            case Format::eRG16_sfloat:
                if(meshInfo->elementByteSize != 4)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 4)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            case Format::eRG32_sfloat:
                if(meshInfo->elementByteSize != 8)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 8)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            default:
                return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
            }

            uint64_t count =
                ((propSize - (!propInfoSize ? baryPayloadGetOffset(sizeof(MeshDisplacementDirectionBoundsInfo), meshInfo->elementByteAlignment) : 0))
                 / meshInfo->elementByteSize);
            if(count != meshInfo->elementCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
        }
        break;
        case StandardPropertyType::eMeshPositions: {
            const MeshPositionsInfo* meshInfo = reinterpret_cast<const MeshPositionsInfo*>(propDataInfo);

            switch(meshInfo->elementFormat)
            {
            case Format::eRGB16_sfloat:
                if(meshInfo->elementByteSize != 6)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 4)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            case Format::eRGBA16_sfloat:
                if(meshInfo->elementByteSize != 8)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 8)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            case Format::eRGB32_sfloat:
                if(meshInfo->elementByteSize != 12)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 4)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            case Format::eRGBA32_sfloat:
                if(meshInfo->elementByteSize != 16)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 16)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            default:
                return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
            }

            uint64_t count =
                ((propSize - (!propInfoSize ? baryPayloadGetOffset(sizeof(MeshPositionsInfo), meshInfo->elementByteAlignment) : 0))
                 / meshInfo->elementByteSize);
            if(count != meshInfo->elementCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
        }
        break;
        case StandardPropertyType::eMeshTriangleIndices: {
            const MeshTriangleIndicesInfo* meshInfo = reinterpret_cast<const MeshTriangleIndicesInfo*>(propDataInfo);

            switch(meshInfo->elementFormat)
            {
            case Format::eR32_uint:
                if(meshInfo->elementByteSize != 4)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 4)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            case Format::eR16_uint:
                if(meshInfo->elementByteSize != 2)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 4)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            default:
                return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
            }

            uint64_t count =
                ((propSize - (!propInfoSize ? baryPayloadGetOffset(sizeof(MeshTriangleIndicesInfo), meshInfo->elementByteAlignment) : 0))
                 / meshInfo->elementByteSize);
            if(count != meshInfo->elementCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
        }
        break;
        case StandardPropertyType::eMeshTriangleFlags: {
            const MeshTriangleFlagsInfo* meshInfo = reinterpret_cast<const MeshTriangleFlagsInfo*>(propDataInfo);

            if(meshInfo->elementFormat != Format::eR8_uint)
            {
                return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
            }
            if(meshInfo->elementByteSize != 1)
            {
                return returnPropError(Result::eErrorSize, errorPropertyType, propType);
            }
            if(meshInfo->elementByteAlignment != 4)
            {
                return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
            }

            uint64_t count =
                ((propSize - (!propInfoSize ? baryPayloadGetOffset(sizeof(MeshTriangleFlagsInfo), meshInfo->elementByteAlignment) : 0))
                 / meshInfo->elementByteSize);
            if(count != meshInfo->elementCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
        }
        break;
        case StandardPropertyType::eMeshTriangleMappings: {
            const MeshTriangleMappingsInfo* meshInfo = reinterpret_cast<const MeshTriangleMappingsInfo*>(propDataInfo);

            switch(meshInfo->elementFormat)
            {
            case Format::eR32_uint:
            case Format::eR32_sint:

                if(meshInfo->elementByteSize != 4)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 4)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            case Format::eR16_uint:
            case Format::eR16_sint:
                if(meshInfo->elementByteSize != 2)
                    return returnPropError(Result::eErrorSize, errorPropertyType, propType);
                if(meshInfo->elementByteAlignment != 4)
                    return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
                break;
            default:
                return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
            }

            uint64_t count =
                ((propSize - (!propInfoSize ? baryPayloadGetOffset(sizeof(MeshTriangleMappingsInfo), meshInfo->elementByteAlignment) : 0))
                 / meshInfo->elementByteSize);
            if(count != meshInfo->elementCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
        }
        break;
        case StandardPropertyType::eTriangleMinMaxs: {
            const TriangleMinMaxsInfo* meshInfo = reinterpret_cast<const TriangleMinMaxsInfo*>(propDataInfo);

            if(valueFormat == Format::eDispC1_r11_unorm_block || valueFormat == Format::eR11_unorm_packed_align32)
            {
                if(meshInfo->elementFormat != Format::eR11_unorm_pack16)
                {
                    return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
                }
            }
            else if(valueFormat == Format::eOpaC1_rx_uint_block)
            {
                if(meshInfo->elementFormat != Format::eR8_uint)
                {
                    return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
                }
            }
            else if(meshInfo->elementFormat != valueFormat)
            {
                return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
            }

            if(meshInfo->elementByteSize == 0)
            {
                return returnPropError(Result::eErrorSize, errorPropertyType, propType);
            }
            if(meshInfo->elementByteAlignment < 4)
            {
                return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
            }

            uint64_t count =
                ((propSize - (!propInfoSize ? baryPayloadGetOffset(sizeof(TriangleMinMaxsInfo), meshInfo->elementByteAlignment) : 0))
                 / meshInfo->elementByteSize);
            if(count != meshInfo->elementCount || count != triangleCount * 2)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
        }
        break;
        case StandardPropertyType::eTriangleUncompressedMips: {
            if(propSize % sizeof(TriangleUncompressedMip))
            {
                return returnPropError(Result::eErrorSize, errorPropertyType, propType);
            }
            if((propSize / sizeof(TriangleUncompressedMip)) != triangleCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
            if(valueFormat != Format::eDispC1_r11_unorm_block)
            {
                return returnPropError(Result::eErrorInvalidPropertyCombination, errorPropertyType, propType);
            }
        }
        break;
        case StandardPropertyType::eGroupUncompressedMips: {
            if(propSize % sizeof(GroupUncompressedMip))
            {
                return returnPropError(Result::eErrorSize, errorPropertyType, propType);
            }
            if((propSize / sizeof(GroupUncompressedMip)) != groupCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
            if(valueFormat != Format::eDispC1_r11_unorm_block)
            {
                return returnPropError(Result::eErrorInvalidPropertyCombination, errorPropertyType, propType);
            }
        }
        break;
        case StandardPropertyType::eUncompressedMips: {
            const UncompressedMipsInfo* mipInfo = reinterpret_cast<const UncompressedMipsInfo*>(propDataInfo);

            if(valueFormat != Format::eDispC1_r11_unorm_block)
            {
                return returnPropError(Result::eErrorInvalidPropertyCombination, errorPropertyType, propType);
            }
            if(valueFormat == Format::eDispC1_r11_unorm_block && mipInfo->elementFormat != Format::eR11_unorm_packed_align32)
            {
                return returnPropError(Result::eErrorFormat, errorPropertyType, propType);
            }
            if(mipInfo->elementByteSize == 0)
            {
                return returnPropError(Result::eErrorSize, errorPropertyType, propType);
            }
            if(mipInfo->elementByteAlignment < 4)
            {
                return returnPropError(Result::eErrorAlignment, errorPropertyType, propType);
            }
            uint64_t count =
                ((propSize - (!propInfoSize ? baryPayloadGetOffset(sizeof(TriangleMinMaxsInfo), mipInfo->elementByteAlignment) : 0))
                 / mipInfo->elementByteSize);
            if(count != mipInfo->elementCount)
            {
                return returnPropError(Result::eErrorCount, errorPropertyType, propType);
            }
        }
        break;
        }
    }

    return Result::eSuccess;
}

static inline uint64_t alignedByteOffset(uint64_t byteSize, uint64_t alignment)
{
    return ((byteSize + alignment - 1) & ~(alignment - 1));
}

BARY_API VersionIdentifier BARY_CALL baryGetCurrentVersionIdentifier()
{
    //       0       1    2    3    4    5    6    7    8    9    10   11      12    13    14      15
    return {{'\xAB', 'B', 'A', 'R', 'Y', ' ', '0', '0', '1', '0', '0', '\xBB', '\r', '\n', '\x1A', '\n'}};
}

static VersionIdentifier zeroedVersionNumber(const VersionIdentifier& otherId)
{
    VersionIdentifier versionId = otherId;

    versionId.data[6]  = '0';
    versionId.data[7]  = '0';
    versionId.data[8]  = '0';
    versionId.data[9]  = '0';
    versionId.data[10] = '0';

    return versionId;
}

static uint32_t getVersionNumber(const VersionIdentifier& versionId)
{
    VersionIdentifier temp = versionId;
    temp.data[11]          = 0;
    return atoi(&temp.data[6]);
}

BARY_API Result baryVersionIdentifierGetVersion(const VersionIdentifier* identifier, uint32_t* pVersion)
{
    VersionIdentifier defaultVersion   = baryGetCurrentVersionIdentifier();
    VersionIdentifier defaultZeroed    = zeroedVersionNumber(defaultVersion);
    VersionIdentifier identifierZeroed = zeroedVersionNumber(*identifier);

    if(memcmp(&identifierZeroed, &defaultZeroed, sizeof(defaultZeroed)))
    {
        return Result::eErrorVersionFormat;
    }

    *pVersion = getVersionNumber(*identifier);

    return Result::eSuccess;
}

BARY_API Result BARY_CALL baryDataGetVersion(uint64_t fileSize, const void* fileData, uint32_t* pVersion)
{
    if(fileSize < sizeof(Header))
    {
        return Result::eErrorFileSize;
    }

    const Header* header = reinterpret_cast<const Header*>(fileData);
    return baryVersionIdentifierGetVersion(&header->version, pVersion);
}

BARY_API Result BARY_CALL baryDataIsValid(uint64_t fileSize, const void* fileData)
{
    if(fileSize < sizeof(Header))
    {
        return Result::eErrorFileSize;
    }
    const Header*  header    = reinterpret_cast<const Header*>(fileData);

    VersionIdentifier defaultVersion = baryGetCurrentVersionIdentifier();
    VersionIdentifier defaultZeroed  = zeroedVersionNumber(defaultVersion);
    VersionIdentifier headerZeroed   = zeroedVersionNumber(header->version);

    if(memcmp(&headerZeroed, &defaultZeroed, sizeof(defaultZeroed)))
    {
        return Result::eErrorVersionFormat;
    }

    if(memcmp(&header->version, &defaultVersion, sizeof(defaultVersion)))
    {
        return Result::eErrorVersion;
    }

    if(fileSize < header->totalByteSize)
    {
        return Result::eErrorFileSize;
    }

    if(header->propertyInfoRange.byteOffset != sizeof(Header))
    {
        return Result::eErrorOffset;
    }

    if(!baryDataIsRangeValid(fileSize, header->propertyInfoRange))
    {
        return Result::eErrorRange;
    }

    uint64_t            propertyCount = header->propertyInfoRange.byteLength / sizeof(PropertyInfo);
    const PropertyInfo* properties = baryDataGetByteRangeDataT<PropertyInfo>(fileSize, fileData, header->propertyInfoRange);

    uint64_t dataOffset = alignedByteOffset(sizeof(Header), 4) + (sizeof(PropertyInfo) * propertyCount);
    for(uint64_t i = 0; i < propertyCount; i++)
    {
        const PropertyInfo* propInfo = properties + i;

        dataOffset = alignedByteOffset(dataOffset, 4);
        if(propInfo->range.byteOffset != dataOffset)
        {
            return Result::eErrorOffset;
        }

        if(!baryDataIsRangeValid(fileSize, propInfo->range))
        {
            return Result::eErrorRange;
        }

        // next offset
        dataOffset += propInfo->range.byteLength;

        if (!baryDataIsRangeValid(fileSize, propInfo->supercompressionGlobalData))
        {
            return Result::eErrorRange;
        }

        if(propInfo->supercompressionScheme != SupercompressionScheme::eNone)
        {
            if(!propInfo->uncompressedByteLength)
            {
                return Result::eErrorSize;
            }
            if(propInfo->supercompressionGlobalData.byteLength)
            {
                dataOffset = alignedByteOffset(dataOffset, 8);
                if(propInfo->supercompressionGlobalData.byteOffset != dataOffset)
                {
                    return Result::eErrorOffset;
                }

                // next offset
                dataOffset += propInfo->supercompressionGlobalData.byteLength;
            }
        }
    }

    if(dataOffset != header->totalByteSize)
    {
        return Result::eErrorSize;
    }

    return Result::eSuccess;
}

//////////////////////////////////////////////////////////////////////////
// storage

static_assert(sizeof(Header) % 4 == 0, "Header size must be multiple of 4 bytes");

BARY_API uint64_t BARY_CALL baryStorageComputeSize(uint32_t propertyCount, const PropertyStorageInfo* propertyStorageInfos)
{
    uint64_t byteSize = 0;
    byteSize          = sizeof(Header) + sizeof(PropertyInfo) * propertyCount;
    for(uint32_t i = 0; i < propertyCount; i++)
    {
        const PropertyStorageInfo* propStore = propertyStorageInfos + i;

        uint64_t byteLength = propStore->dataSize + propStore->infoSize + propStore->infoPaddingSize;
        byteSize            = alignedByteOffset(byteSize, 4) + byteLength;

        if(propStore->supercompressionScheme != SupercompressionScheme::eNone && propStore->supercompressionGlobalDataSize)
        {
            byteSize = alignedByteOffset(byteSize, 8) + propStore->supercompressionGlobalDataSize;
        }
    }

    return byteSize;
}

BARY_API uint64_t BARY_CALL baryStorageComputePreambleSize(uint32_t propertyCount)
{
    uint64_t byteSize = 0;
    byteSize          = sizeof(Header) + sizeof(PropertyInfo) * propertyCount;

    return byteSize;
}

BARY_API Result BARY_CALL baryStorageOutputPreamble(uint32_t                   propertyCount,
                                                    const PropertyStorageInfo* propertyStorageInfos,
                                                    uint64_t                   outputSize,
                                                    uint64_t                   outputPreambleSize,
                                                    void*                      outputPreambleData)
{
    if(outputSize != baryStorageComputeSize(propertyCount, propertyStorageInfos))
    {
        return Result::eErrorFileSize;
    }

    uint8_t* outputBytes = reinterpret_cast<uint8_t*>(outputPreambleData);

    {
        Header* header                       = reinterpret_cast<Header*>(outputBytes);
        header->totalByteSize                = outputSize;
        header->version                      = baryGetCurrentVersionIdentifier();
        header->propertyInfoRange.byteOffset = sizeof(Header);
        header->propertyInfoRange.byteLength = sizeof(PropertyInfo) * propertyCount;
    }

    {
        uint64_t propsOffset = sizeof(Header) + (sizeof(PropertyInfo) * propertyCount);

        PropertyInfo* propInfosOut = reinterpret_cast<PropertyInfo*>(outputBytes + sizeof(Header));

        for(uint32_t i = 0; i < propertyCount; i++)
        {
            PropertyInfo*              propInfo  = propInfosOut + i;
            const PropertyStorageInfo* propStore = propertyStorageInfos + i;

            uint64_t byteLength = propStore->dataSize + propStore->infoSize + propStore->infoPaddingSize;

            if(!byteLength)
            {
                return Result::eErrorSize;
            }

            propsOffset = alignedByteOffset(propsOffset, 4);

            propInfo->identifier             = propStore->identifier;
            propInfo->range.byteLength       = byteLength;
            propInfo->range.byteOffset       = propsOffset;
            propInfo->supercompressionScheme = propStore->supercompressionScheme;

            propsOffset += byteLength;

            if(propertyStorageInfos[i].supercompressionScheme != SupercompressionScheme::eNone)
            {
                propInfo->uncompressedByteLength                = propStore->uncompressedSize;
                propInfo->supercompressionGlobalData.byteLength = propStore->supercompressionGlobalDataSize;

                if(propInfo->supercompressionGlobalData.byteLength)
                {
                    propsOffset                                     = alignedByteOffset(propsOffset, 8);
                    propInfo->supercompressionGlobalData.byteOffset = propsOffset;
                    propsOffset = propInfo->supercompressionGlobalData.byteOffset + propInfo->supercompressionGlobalData.byteLength;
                }
                else
                {
                    propInfo->supercompressionGlobalData.byteOffset = 0;
                }
            }
            else
            {
                propInfo->uncompressedByteLength     = 0;
                propInfo->supercompressionGlobalData = {0, 0};
            }

            assert(baryDataIsRangeValid(outputSize, propInfo->range));
        }
    }

    return Result::eSuccess;
}

BARY_API Result BARY_CALL baryStorageOutputAll(uint32_t                   propertyCount,
                                               const PropertyStorageInfo* propertyStorageInfos,
                                               uint64_t                   outputSize,
                                               void*                      outputData)
{
    Result result = baryStorageOutputPreamble(propertyCount, propertyStorageInfos, outputSize,
                                              baryStorageComputePreambleSize(propertyCount), outputData);
    if(result != Result::eSuccess)
    {
        return result;
    }

    uint8_t*      outputBytes  = reinterpret_cast<uint8_t*>(outputData);
    PropertyInfo* propInfosOut = reinterpret_cast<PropertyInfo*>(outputBytes + sizeof(Header));

    for(uint32_t i = 0; i < propertyCount; i++)
    {
        const PropertyInfo*        propInfo  = propInfosOut + i;
        const PropertyStorageInfo* propStore = propertyStorageInfos + i;

        assert(propInfo->range.byteLength == (propStore->dataSize + propStore->infoSize + propStore->infoPaddingSize));

        if(propStore->info && propStore->infoSize)
        {
            memcpy(outputBytes + propInfo->range.byteOffset, propStore->info, propStore->infoSize);
            memcpy(outputBytes + propInfo->range.byteOffset + propStore->infoSize + propStore->infoPaddingSize,
                   propStore->data, propStore->dataSize);
        }
        else
        {
            memcpy(outputBytes + propInfo->range.byteOffset, propStore->data, propStore->dataSize);
        }
    }

    return Result::eSuccess;
}

BARY_API Result BARY_CALL baryStorageOutputSaver(uint32_t                   propertyCount,
                                                 const PropertyStorageInfo* propertyStorageInfos,
                                                 uint64_t                   preambleSize,
                                                 const void*                preamble,
                                                 PFN_outputSaver            fnSaver,
                                                 void*                      userData)
{
    Result result;

    if(baryStorageComputePreambleSize(propertyCount) != preambleSize)
    {
        return Result::eErrorFileSize;
    }

    const PropertyInfo* propInfos =
        reinterpret_cast<const PropertyInfo*>(reinterpret_cast<const uint8_t*>(preamble) + sizeof(Header));

    uint8_t  paddingBytes[128] = {0};
    uint64_t fileOffset        = 0;

    result = fnSaver(0, nullptr, fileOffset, preambleSize, preamble, false, userData);
    fileOffset += preambleSize;
    if(result != Result::eSuccess)
    {
        return result;
    }

    for(uint32_t i = 0; i < propertyCount; i++)
    {
        const bary::PropertyInfo*        propInfo  = propInfos + i;
        const bary::PropertyStorageInfo* propStore = propertyStorageInfos + i;

#ifdef _DEBUG
        bary::StandardPropertyType propType = baryPropertyGetStandardType(propStore->identifier);
#endif

        assert(propInfo->range.byteLength == (propStore->dataSize + propStore->infoSize + propStore->infoPaddingSize));

        // padding
        uint64_t padding = propInfo->range.byteOffset - fileOffset;
        if(padding)
        {
            assert(padding < sizeof(paddingBytes));
            result = fnSaver(0, nullptr, fileOffset, padding, paddingBytes, false, userData);
            fileOffset += padding;
            if(result != Result::eSuccess)
            {
                return result;
            }
        }


        assert(fileOffset == propInfo->range.byteOffset);

        if(propStore->info && propStore->infoSize)
        {
            result = fnSaver(i, propStore, fileOffset, propStore->infoSize, propStore->info, true, userData);
            fileOffset += propStore->infoSize;
            if(result != Result::eSuccess)
            {
                return result;
            }


            if(propStore->infoPaddingSize)
            {
                assert(propStore->infoPaddingSize < sizeof(paddingBytes));
                result = fnSaver(0, nullptr, fileOffset, propStore->infoPaddingSize, paddingBytes, false, userData);
                fileOffset += propStore->infoPaddingSize;

                if(result != Result::eSuccess)
                {
                    return result;
                }
            }

            result = fnSaver(i, propStore, fileOffset, propStore->dataSize, propStore->data, false, userData);
            fileOffset += propStore->dataSize;
            if(result != Result::eSuccess)
            {
                return result;
            }
        }
        else
        {
            result = fnSaver(i, propStore, fileOffset, propStore->dataSize, propStore->data, false, userData);
            fileOffset += propStore->dataSize;
            if(result != Result::eSuccess)
            {
                return result;
            }
        }
    }

    return Result::eSuccess;
}

//////////////////////////////////////////////////////////////////////////
// retrieval

// can return nullptr if no properties
BARY_API const PropertyInfo* BARY_CALL baryDataGetAllPropertyInfos(uint64_t fileSize, const void* fileData, uint64_t* count)
{
    const Header*       header        = reinterpret_cast<const Header*>(fileData);
    uint64_t            propertyCount = header->propertyInfoRange.byteLength / sizeof(PropertyInfo);
    const PropertyInfo* properties = baryDataGetByteRangeDataT<PropertyInfo>(fileSize, fileData, header->propertyInfoRange);

    *count = propertyCount;

    return properties;
}

BARY_API const PropertyInfo* BARY_CALL baryDataGetPropertyInfo(uint64_t fileSize, const void* fileData, PropertyIdentifier identifier)
{
    assert(baryDataIsValid(fileSize, fileData) == Result::eSuccess);

    const Header*       header        = reinterpret_cast<const Header*>(fileData);
    uint64_t            propertyCount = header->propertyInfoRange.byteLength / sizeof(PropertyInfo);
    const PropertyInfo* properties = baryDataGetByteRangeDataT<PropertyInfo>(fileSize, fileData, header->propertyInfoRange);

    for(uint64_t i = 0; i < propertyCount; i++)
    {
        if(baryPropertyIsEqual(identifier, properties[i].identifier))
        {
            return &properties[i];
        }
    }

    return nullptr;
}

BARY_API const void* BARY_CALL baryDataGetPropertyData(uint64_t fileSize, const void* fileData, PropertyIdentifier identifier, uint64_t* pLength)
{
    const PropertyInfo* info = baryDataGetPropertyInfo(fileSize, fileData, identifier);
    if(info && baryDataIsRangeValid(fileSize, info->range))
    {
        if(pLength)
        {
            *pLength = info->range.byteLength;
        }
        return reinterpret_cast<const uint8_t*>(fileData) + info->range.byteOffset;
    }
    else
    {
        if(pLength)
        {
            *pLength = 0;
        }
        return nullptr;
    }
}

BARY_API Result BARY_CALL baryDataHasMandatoryStandardProperties(uint64_t fileSize, const void* fileData)
{
    return (baryDataGetPropertyInfo(fileSize, fileData, baryMakeStandardPropertyIdentifierT<StandardPropertyType::eValues>())
            && baryDataGetPropertyInfo(fileSize, fileData, baryMakeStandardPropertyIdentifierT<StandardPropertyType::eTriangles>())
            && baryDataGetPropertyInfo(fileSize, fileData, baryMakeStandardPropertyIdentifierT<StandardPropertyType::eGroups>())) ?
               Result::eSuccess :
               Result::eErrorMissingProperty;
}

BARY_API bool BARY_CALL baryDataHasAnySuperCompression(uint64_t fileSize, const void* fileData)
{
    const Header*       header        = reinterpret_cast<const Header*>(fileData);
    uint64_t            propertyCount = header->propertyInfoRange.byteLength / sizeof(PropertyInfo);
    const PropertyInfo* properties = baryDataGetByteRangeDataT<PropertyInfo>(fileSize, fileData, header->propertyInfoRange);

    for(uint64_t i = 0; i < propertyCount; i++)
    {
        if(properties[i].supercompressionScheme != SupercompressionScheme::eNone)
            return true;
    }

    return false;
}

//////////////////////////////////////////////////////////////////////////

BARY_API void BARY_CALL baryBasicViewGetMinMaxSubdivLevels(const BasicView* basic, uint32_t* min, uint32_t* max)
{
    uint32_t minSubdiv = ~0;
    uint32_t maxSubdiv = 0;
    for(uint32_t i = 0; i < basic->groupsCount; i++)
    {
        minSubdiv = basic->groups[i].minSubdivLevel < minSubdiv ? basic->groups[i].minSubdivLevel : minSubdiv;
        maxSubdiv = basic->groups[i].maxSubdivLevel > maxSubdiv ? basic->groups[i].maxSubdivLevel : maxSubdiv;
    }

    if(min)
    {
        *min = minSubdiv;
    }
    if(max)
    {
        *max = maxSubdiv;
    }
}

BARY_API Result BARY_CALL baryContentIsValid(ValueSemanticType valueSemantic, const ContentView* content, StandardPropertyType* errorPropertyType)
{
    const BasicView* basic = &content->basic;
    const MeshView*  mesh  = &content->mesh;
    /* const MiscView* misc  = &content->misc; // Currently unused for cross-validation here */

    // mandatory
    if(!basic->groupsCount || !basic->groups)
    {
        return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eGroups);
    }
    if(!basic->trianglesCount || !basic->triangles)
    {
        return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eTriangles);
    }
    if(!basic->valuesInfo || !basic->values)
    {
        return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eValues);
    }

    if(!basic->valuesInfo->valueByteSize)
    {
        return returnPropError(Result::eErrorSize, errorPropertyType, StandardPropertyType::eValues);
    }
    if(!basic->valuesInfo->valueCount)
    {
        return returnPropError(Result::eErrorCount, errorPropertyType, StandardPropertyType::eValues);
    }

    bool isCompressed = false;

    switch(basic->valuesInfo->valueFormat)
    {
    case Format::eOpaC1_rx_uint_block:
    case Format::eDispC1_r11_unorm_block:
        isCompressed = true;
        break;
    }

    if(isCompressed)
    {
        if(!basic->histogramEntries || !basic->histogramEntriesCount)
        {
            return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eHistogramEntries);
        }
        if(!basic->groupHistogramRanges || !basic->groupHistogramRangesCount)
        {
            return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eGroupHistograms);
        }
    }

    if(valueSemantic == ValueSemanticType::eDisplacement)
    {
        switch(basic->valuesInfo->valueFormat)
        {
        case Format::eR8_unorm:
        case Format::eR16_unorm:
        case Format::eR32_sfloat:
        case Format::eR11_unorm_pack16:
        case Format::eR11_unorm_packed_align32:
        case Format::eDispC1_r11_unorm_block:
            break;
        default:
            return returnPropError(Result::eErrorFormat, errorPropertyType, StandardPropertyType::eValues);
        }

        if(mesh->meshDisplacementDirectionBoundsInfo && mesh->meshDisplacementDirectionBounds)
        {
            for(uint32_t g = 0; g < basic->groupsCount; g++)
            {
                if(basic->groups[g].floatBias.r != 0 || basic->groups[g].floatScale.r != 1.0f)
                {
                    return returnPropError(Result::eErrorValue, errorPropertyType, StandardPropertyType::eGroups);
                }
            }
        }
    }

    return Result::eSuccess;
}

BARY_API uint32_t BARY_CALL baryContentComputePropertyCount(const ContentView* content)
{
    const BasicView* basic = &content->basic;
    const MeshView*  mesh  = &content->mesh;
    const MiscView*  misc  = &content->misc;

    uint32_t count = 0;
    count += basic->groups && basic->groupsCount ? 1 : 0;
    count += basic->valuesInfo && basic->values ? 1 : 0;
    count += basic->triangles && basic->trianglesCount ? 1 : 0;
    count += basic->triangleMinMaxs && basic->triangleMinMaxsInfo ? 1 : 0;

    count += basic->histogramEntries && basic->histogramEntriesCount ? 1 : 0;
    count += basic->groupHistogramRanges && basic->groupHistogramRangesCount ? 1 : 0;

    count += mesh->meshDisplacementDirectionBounds && mesh->meshDisplacementDirectionBoundsInfo ? 1 : 0;
    count += mesh->meshDisplacementDirections && mesh->meshDisplacementDirectionsInfo ? 1 : 0;
    count += mesh->meshTriangleFlags && mesh->meshTriangleFlagsInfo ? 1 : 0;
    count += mesh->meshTriangleMappings && mesh->meshTriangleMappingsInfo ? 1 : 0;
    count += mesh->meshGroups && mesh->meshGroupsCount ? 1 : 0;
    count += mesh->meshHistogramEntries && mesh->meshHistogramEntriesCount ? 1 : 0;
    count += mesh->meshGroupHistogramRanges && mesh->meshGroupHistogramRangesCount ? 1 : 0;
    count += mesh->meshPositions && mesh->meshPositionsInfo ? 1 : 0;
    count += mesh->meshTriangleIndices && mesh->meshTriangleIndicesInfo ? 1 : 0;

    count += misc->groupUncompressedMips && misc->groupUncompressedMipsCount ? 1 : 0;
    count += misc->triangleUncompressedMips && misc->triangleUncompressedMipsCount ? 1 : 0;
    count += misc->uncompressedMips && misc->uncompressedMipsInfo ? 1 : 0;

    return count;
}

template <typename Tinfo, typename Tdata>
static void setupElementProp(PropertyStorageInfo*& prop, const Tinfo* info, const Tdata* data)
{
    if(info && data)
    {
        *prop                 = PropertyStorageInfo();
        prop->identifier      = baryMakePropertyIdentifierT<Tinfo>();
        prop->data            = data;
        prop->info            = info;
        prop->dataSize        = info->elementByteSize * info->elementCount;
        prop->infoSize        = sizeof(Tinfo);
        prop->infoPaddingSize = baryPayloadGetPadding(sizeof(Tinfo), info->elementByteAlignment);
        prop++;
    }
}

template <typename Tdata>
static void setupCountProp(PropertyStorageInfo*& prop, size_t count, const Tdata* data)
{
    if(count && data)
    {
        *prop            = PropertyStorageInfo();
        prop->identifier = baryMakePropertyIdentifierT<Tdata>();
        prop->data       = data;
        prop->dataSize   = sizeof(Tdata) * count;
        prop++;
    }
}

BARY_API Result BARY_CALL baryContentSetupProperties(const ContentView* content, uint32_t propertyCount, PropertyStorageInfo* propertyStorageInfos)
{
    if(baryContentComputePropertyCount(content) != propertyCount)
    {
        return Result::eErrorCount;
    }

    const BasicView*     basic = &content->basic;
    const MeshView*      mesh  = &content->mesh;
    const MiscView*      misc  = &content->misc;
    PropertyStorageInfo* prop  = propertyStorageInfos;

    setupCountProp(prop, basic->groupsCount, basic->groups);
    setupCountProp(prop, basic->trianglesCount, basic->triangles);

    if(basic->valuesInfo && basic->values)
    {
        *prop                 = PropertyStorageInfo();
        prop->identifier      = baryMakePropertyIdentifierT<ValuesInfo>();
        prop->data            = basic->values;
        prop->dataSize        = basic->valuesInfo->valueByteSize * basic->valuesInfo->valueCount;
        prop->info            = basic->valuesInfo;
        prop->infoSize        = sizeof(ValuesInfo);
        prop->infoPaddingSize = baryPayloadGetPadding(sizeof(ValuesInfo), basic->valuesInfo->valueByteAlignment);
        prop++;
    }

    setupElementProp(prop, basic->triangleMinMaxsInfo, basic->triangleMinMaxs);
    setupCountProp(prop, basic->histogramEntriesCount, basic->histogramEntries);
    setupCountProp(prop, basic->groupHistogramRangesCount, basic->groupHistogramRanges);

    {
        setupElementProp(prop, mesh->meshDisplacementDirectionBoundsInfo, mesh->meshDisplacementDirectionBounds);
        setupElementProp(prop, mesh->meshDisplacementDirectionsInfo, mesh->meshDisplacementDirections);
        setupElementProp(prop, mesh->meshTriangleFlagsInfo, mesh->meshTriangleFlags);
        setupElementProp(prop, mesh->meshTriangleMappingsInfo, mesh->meshTriangleMappings);
        setupElementProp(prop, mesh->meshTriangleIndicesInfo, mesh->meshTriangleIndices);
        setupElementProp(prop, mesh->meshPositionsInfo, mesh->meshPositions);


        setupCountProp(prop, mesh->meshGroupsCount, mesh->meshGroups);
        setupCountProp(prop, mesh->meshHistogramEntriesCount, mesh->meshHistogramEntries);
        setupCountProp(prop, mesh->meshGroupHistogramRangesCount, mesh->meshGroupHistogramRanges);
    }

    {
        setupElementProp(prop, misc->uncompressedMipsInfo, misc->uncompressedMips);
        setupCountProp(prop, misc->groupUncompressedMipsCount, misc->groupUncompressedMips);
        setupCountProp(prop, misc->triangleUncompressedMipsCount, misc->triangleUncompressedMips);
    }


    assert((prop - propertyStorageInfos) == size_t(propertyCount));

    return Result::eSuccess;
}

template <typename Tdata>
static void getCountProp(const Tdata*& datas, uint32_t& datasCount, uint64_t fileSize, const void* fileData)
{
    uint64_t length = 0;

    datas      = baryDataGetPropertyDataT<Tdata>(fileSize, fileData, baryMakePropertyIdentifierT<Tdata>(), &length);
    datasCount = uint32_t(length / sizeof(Tdata));
}

template <typename Tinfo, typename Tdata>
static Result getElementProp(const Tinfo*&         varInfo,
                             const Tdata*&         varData,
                             StandardPropertyType  stype,
                             StandardPropertyType* errorPropertyType,
                             uint64_t              fileSize,
                             const void*           fileData)
{
    uint64_t length = 0;

    varInfo = baryDataGetPropertyDataT<Tinfo>(fileSize, fileData, baryMakePropertyIdentifierT<Tinfo>(), &length);
    if(!varInfo)
    {
        varData = nullptr;
        return Result::eSuccess;
    }

    varData = reinterpret_cast<const uint8_t*>(baryPayloadGetPointer(sizeof(Tinfo), varInfo->elementByteAlignment, varInfo));

    if(varInfo->elementCount * varInfo->elementByteSize + baryPayloadGetOffset(sizeof(Tinfo), varInfo->elementByteAlignment) > length)
    {
        return returnPropError(Result::eErrorSize, errorPropertyType, stype);
    }

    return Result::eSuccess;
}

BARY_API Result BARY_CALL baryDataGetContent(uint64_t fileSize, const void* fileData, ContentView* content, StandardPropertyType* errorPropertyType)
{
    if(baryDataHasAnySuperCompression(fileSize, fileData))
    {
        return Result::eErrorSupercompression;
    }

    BasicView* basic = &content->basic;
    MeshView*  mesh  = &content->mesh;
    MiscView*  misc  = &content->misc;
    Result     result;


    // mandatory
    getCountProp(basic->groups, basic->groupsCount, fileSize, fileData);
    getCountProp(basic->triangles, basic->trianglesCount, fileSize, fileData);

    {
        uint64_t length;
        basic->valuesInfo =
            baryDataGetPropertyDataT<ValuesInfo>(fileSize, fileData, baryMakePropertyIdentifierT<ValuesInfo>(), &length);
        if(!basic->valuesInfo)
        {
            // eValues is a mandatory property, so its Info must exist;
            // reaching this line indicates that the file wasn't validated
            // using baryContentIsValid().
            return returnPropError(Result::eErrorMissingProperty, errorPropertyType, StandardPropertyType::eValues);
        }
        basic->values = reinterpret_cast<const uint8_t*>(
            baryPayloadGetPointer(sizeof(ValuesInfo), basic->valuesInfo->valueByteAlignment, basic->valuesInfo));
        if((basic->valuesInfo->valueCount * basic->valuesInfo->valueByteSize
            + baryPayloadGetOffset(sizeof(ValuesInfo), basic->valuesInfo->valueByteAlignment))
           > length)
        {
            return returnPropError(Result::eErrorSize, errorPropertyType, StandardPropertyType::eValues);
        }
    }

    result = getElementProp(basic->triangleMinMaxsInfo, basic->triangleMinMaxs, StandardPropertyType::eTriangleMinMaxs,
                            errorPropertyType, fileSize, fileData);
    if(result != Result::eSuccess)
        return result;

    // compressed
    getCountProp(basic->histogramEntries, basic->histogramEntriesCount, fileSize, fileData);
    getCountProp(basic->groupHistogramRanges, basic->groupHistogramRangesCount, fileSize, fileData);


    // optionals
    {
        result = getElementProp(mesh->meshDisplacementDirectionBoundsInfo, mesh->meshDisplacementDirectionBounds,
                                StandardPropertyType::eMeshDisplacementDirectionBounds, errorPropertyType, fileSize, fileData);
        if(result != Result::eSuccess)
            return result;

        result = getElementProp(mesh->meshDisplacementDirectionBoundsInfo, mesh->meshDisplacementDirectionBounds,
                                StandardPropertyType::eMeshDisplacementDirectionBounds, errorPropertyType, fileSize, fileData);
        if(result != Result::eSuccess)
            return result;

        result = getElementProp(mesh->meshDisplacementDirectionsInfo, mesh->meshDisplacementDirections,
                                StandardPropertyType::eMeshDisplacementDirections, errorPropertyType, fileSize, fileData);
        if(result != Result::eSuccess)
            return result;

        result = getElementProp(mesh->meshTriangleFlagsInfo, mesh->meshTriangleFlags,
                                StandardPropertyType::eMeshTriangleFlags, errorPropertyType, fileSize, fileData);
        if(result != Result::eSuccess)
            return result;

        result = getElementProp(mesh->meshTriangleMappingsInfo, mesh->meshTriangleMappings,
                                StandardPropertyType::eMeshTriangleMappings, errorPropertyType, fileSize, fileData);
        if(result != Result::eSuccess)
            return result;

        result = getElementProp(mesh->meshTriangleIndicesInfo, mesh->meshTriangleIndices,
                                StandardPropertyType::eMeshTriangleIndices, errorPropertyType, fileSize, fileData);
        if(result != Result::eSuccess)
            return result;

        result = getElementProp(mesh->meshPositionsInfo, mesh->meshPositions, StandardPropertyType::eMeshPositions,
                                errorPropertyType, fileSize, fileData);
        if(result != Result::eSuccess)
            return result;

        getCountProp(mesh->meshGroups, mesh->meshGroupsCount, fileSize, fileData);
        getCountProp(mesh->meshHistogramEntries, mesh->meshHistogramEntriesCount, fileSize, fileData);
        getCountProp(mesh->meshGroupHistogramRanges, mesh->meshGroupHistogramRangesCount, fileSize, fileData);
    }

    {
        result = getElementProp(misc->uncompressedMipsInfo, misc->uncompressedMips,
                                StandardPropertyType::eUncompressedMips, errorPropertyType, fileSize, fileData);
        if(result != Result::eSuccess)
            return result;

        getCountProp(misc->groupUncompressedMips, misc->groupUncompressedMipsCount, fileSize, fileData);
        getCountProp(misc->triangleUncompressedMips, misc->triangleUncompressedMipsCount, fileSize, fileData);
    }

    return Result::eSuccess;
}

BARY_API uint32_t BARY_CALL baryHistogramGetBlockCount(uint32_t entriesCount, const bary::HistogramEntry* entries, bary::Format fmt)
{
    if(fmt != bary::Format::eDispC1_r11_unorm_block)
        return 0;

    uint32_t blocks = 0;
    for(uint32_t i = 0; i < entriesCount; i++)
    {
        blocks +=
            bary::baryBlockFormatDispC1GetBlockCount(entries[i].blockFormatDispC1, entries[i].subdivLevel) * entries[i].count;
    }

    return blocks;
}

BARY_API uint32_t BARY_CALL baryMeshHistogramGetBlockCount(uint32_t entriesCount, const bary::MeshHistogramEntry* entries, bary::Format fmt)
{
    if(fmt != bary::Format::eDispC1_r11_unorm_block)
        return 0;

    uint32_t blocks = 0;
    for(uint32_t i = 0; i < entriesCount; i++)
    {
        blocks +=
            bary::baryBlockFormatDispC1GetBlockCount(entries[i].blockFormatDispC1, entries[i].subdivLevel) * entries[i].count;
    }

    return blocks;
}

}  // namespace bary
