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
#pragma once


#include <cassert>
#include <cstring>
#include <bary/bary_core.h>
#include <vector>
#include <string>

// baryutils is designed to not depend on micromesh sdk
// so that it can be easily used in apps that just want to load/save or
// render existing data.

namespace baryutils
{
//////////////////////////////////////////////////////////////////////////

inline uint32_t baryDisplacementFormatGetNumBits(bary::Format fmt)
{
    switch(fmt)
    {
    case bary::Format::eR8_unorm:
        return 8;
    case bary::Format::eR16_unorm:
    case bary::Format::eR11_unorm_pack16:
        return 16;
    case bary::Format::eR32_sfloat:
        return 32;
    default:
        return 0;
    }
}

//////////////////////////////////////////////////////////////////////////

struct BaryBasicInfo
{
    bary::ValuesInfo          valuesInfo;
    bary::TriangleMinMaxsInfo triangleMinMaxsInfo;
};

struct BaryMeshInfo
{
    bary::MeshDisplacementDirectionsInfo      meshDisplacementDirectionsInfo;
    bary::MeshDisplacementDirectionBoundsInfo meshDisplacementDirectionBoundsInfo;
    bary::MeshPositionsInfo                   meshPositionsInfo;
    bary::MeshTriangleMappingsInfo            meshTriangleMappingsInfo;
    bary::MeshTriangleFlagsInfo               meshTriangleFlagsInfo;
    bary::MeshTriangleIndicesInfo             meshTriangleIndicesInfo;
};

struct BaryMiscInfo
{
    bary::UncompressedMipsInfo uncompressedMipsInfo;
};

struct BaryContentInfo
{
    BaryBasicInfo basic;
    BaryMeshInfo  mesh;
    BaryMiscInfo  misc;
};

//////////////////////////////////////////////////////////////////////////
struct BaryFileOpenOptions;

struct BaryBasicData
{
    uint32_t minSubdivLevel = 0;
    uint32_t maxSubdivLevel = 0;

    // mandatory for all
    std::vector<bary::Group>    groups;
    bary::ValuesInfo            valuesInfo{};
    std::vector<uint8_t>        values;
    std::vector<bary::Triangle> triangles;

    // mandatory for compressed
    std::vector<bary::HistogramEntry>      histogramEntries;
    std::vector<bary::GroupHistogramRange> groupHistogramRanges;

    // optional, but can improve displacement rasterization performance
    bary::TriangleMinMaxsInfo triangleMinMaxsInfo{};
    std::vector<uint8_t>      triangleMinMaxs;

    BaryBasicData() {}
    BaryBasicData(const bary::BasicView& basic) { setData(basic); }

    void            setData(const bary::BasicView& basic);
    bary::BasicView getView() const;
    void            updateMinMaxSubdivLevels();

    template <class T>
    T* getValues()
    {
        return reinterpret_cast<T*>(values.data());
    }

    template <class T>
    const T* getValues() const
    {
        return reinterpret_cast<const T*>(values.data());
    }

    bary::Result save(const char*                 filename,
                      const bary::MeshView*       mesh       = nullptr,
                      const bary::MiscView*       misc       = nullptr,
                      bary::StandardPropertyType* pErrorProp = nullptr) const;
    bary::Result save(const std::string&          filename,
                      const bary::MeshView*       mesh       = nullptr,
                      const bary::MiscView*       misc       = nullptr,
                      bary::StandardPropertyType* pErrorProp = nullptr) const
    {
        return save(filename.c_str(), mesh, misc, pErrorProp);
    }

    bary::Result load(size_t                      fileSize,
                      const void*                 fileData,
                      bary::ValueSemanticType     vtype      = bary::ValueSemanticType::eGeneric,
                      bary::StandardPropertyType* pErrorProp = nullptr);
    bary::Result load(const char*                 filename,
                      bary::ValueSemanticType     vtype       = bary::ValueSemanticType::eGeneric,
                      const BaryFileOpenOptions*  fileOptions = nullptr,
                      bary::StandardPropertyType* pErrorProp  = nullptr);
    bary::Result load(const std::string&          filename,
                      bary::ValueSemanticType     vtype       = bary::ValueSemanticType::eGeneric,
                      const BaryFileOpenOptions*  fileOptions = nullptr,
                      bary::StandardPropertyType* pErrorProp  = nullptr)
    {
        return load(filename.c_str(), vtype, fileOptions, pErrorProp);
    }
};

struct BaryMeshData
{
    BaryMeshData() {}
    BaryMeshData(const bary::MeshView& view) { setData(view); }

    // optional mesh properties
    // (these may be stored inside a 3d mesh file format, or not exist at all)

    std::vector<bary::MeshGroup> meshGroups;

    std::vector<bary::MeshHistogramEntry>      meshHistogramEntries;
    std::vector<bary::MeshGroupHistogramRange> meshGroupHistogramRanges;

    bary::MeshDisplacementDirectionsInfo      meshDisplacementDirectionsInfo{};
    std::vector<uint8_t>                      meshDisplacementDirections;
    bary::MeshDisplacementDirectionBoundsInfo meshDisplacementDirectionBoundsInfo{};
    std::vector<uint8_t>                      meshDisplacementDirectionBounds;

    bary::MeshTriangleMappingsInfo meshTriangleMappingsInfo{};
    std::vector<uint8_t>           meshTriangleMappings;
    bary::MeshTriangleFlagsInfo    meshTriangleFlagsInfo{};
    std::vector<uint8_t>           meshTriangleFlags;

    // uncommon, meant for debugging
    bary::MeshPositionsInfo       meshPositionsInfo{};
    std::vector<uint8_t>          meshPositions;
    bary::MeshTriangleIndicesInfo meshTriangleIndicesInfo{};
    std::vector<uint8_t>          meshTriangleIndices;

    void           setData(const bary::MeshView& view);
    bary::MeshView getView() const;
};

struct BaryMiscData
{
    BaryMiscData() {}
    BaryMiscData(const bary::MiscView& view) { setData(view); }

    // optional
    std::vector<bary::GroupUncompressedMip>    groupUncompressedMips;
    std::vector<bary::TriangleUncompressedMip> triangleUncompressedMips;
    bary::UncompressedMipsInfo                 uncompressedMipsInfo{};
    std::vector<uint8_t>                       uncompressedMips;

    void           setData(const bary::MiscView& view);
    bary::MiscView getView() const;
};

struct BaryContentData
{
    BaryContentData() {}
    BaryContentData(const bary::ContentView& view) { setData(view); }

    BaryBasicData basic;
    BaryMeshData  mesh;
    BaryMiscData  misc;

    void setData(const bary::ContentView& view)
    {
        basic.setData(view.basic);
        mesh.setData(view.mesh);
        misc.setData(view.misc);
    }

    bary::ContentView getView() const
    {
        bary::ContentView view;
        view.basic = basic.getView();
        view.mesh  = mesh.getView();
        view.misc  = misc.getView();
        return view;
    }

    bary::Result save(const char* filename, bary::StandardPropertyType* pErrorProp = nullptr) const;
    bary::Result save(const std::string& filename, bary::StandardPropertyType* pErrorProp = nullptr) const
    {
        return save(filename.c_str(), pErrorProp);
    }

    bary::Result load(size_t                      fileSize,
                      const void*                 fileData,
                      bary::ValueSemanticType     vtype      = bary::ValueSemanticType::eGeneric,
                      bary::StandardPropertyType* pErrorProp = nullptr);
    bary::Result load(const char*                 filename,
                      bary::ValueSemanticType     vtype       = bary::ValueSemanticType::eGeneric,
                      const BaryFileOpenOptions*  fileOptions = nullptr,
                      bary::StandardPropertyType* pErrorProp  = nullptr);
    bary::Result load(const std::string&          filename,
                      bary::ValueSemanticType     vtype       = bary::ValueSemanticType::eGeneric,
                      const BaryFileOpenOptions*  fileOptions = nullptr,
                      bary::StandardPropertyType* pErrorProp  = nullptr)
    {
        return load(filename.c_str(), vtype, fileOptions, pErrorProp);
    }
};

//////////////////////////////////////////////////////////////////////////

struct BaryStats
{
    static uint32_t       getHistoBin(uint32_t count);
    static const uint32_t MAX_HISTO_BINS = 32;

    uint32_t          minSubdivLevel = ~0u;
    uint32_t          maxSubdivLevel = 0;
    size_t            dataByteSize   = 0;
    bary::ValueLayout valueOrder     = bary::ValueLayout::eUndefined;
    bary::Format      valueFormat    = bary::Format::eUndefined;

    uint32_t microTriangles = 0;
    uint32_t microVertices  = 0;
    uint32_t mapTriangles   = 0;
    uint32_t blocks         = 0;
    // how many blocks of a certain format exist
    uint32_t blocksPerFormat[uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl5_pack1024) + 1] = {0};
    // histogram over how many blocks triangles have
    // histogram in power-of-2 bins
    uint32_t blocksPerTriangleHisto[MAX_HISTO_BINS] = {0};

    BaryStats() {}
    BaryStats(const bary::BasicView& basic) { append(basic); }

    // return true on inconsistent valueOrder
    bool append(const bary::BasicView& basic);
};

//////////////////////////////////////////////////////////////////////////

struct BaryMeshViewWithInfo : bary::MeshView
{
    BaryMeshInfo rw;

    void setupPointers()
    {
        meshDisplacementDirectionsInfo      = &rw.meshDisplacementDirectionsInfo;
        meshDisplacementDirectionBoundsInfo = &rw.meshDisplacementDirectionBoundsInfo;
        meshPositionsInfo                   = &rw.meshPositionsInfo;
        meshTriangleFlagsInfo               = &rw.meshTriangleFlagsInfo;
        meshTriangleMappingsInfo            = &rw.meshTriangleMappingsInfo;
        meshTriangleIndicesInfo             = &rw.meshTriangleIndicesInfo;
    }

    BaryMeshViewWithInfo() { setupPointers(); }
    BaryMeshViewWithInfo(const BaryMeshViewWithInfo& other) { *this = other; }
    BaryMeshViewWithInfo& operator=(const BaryMeshViewWithInfo& other)
    {
        // Avoid producing a warning about calling memcpy on a type without
        // trivial copy-assignment: in this case, we want to copy the bytes
        // of the class verbatim (for which the use of memcpy here is valid),
        // and then set up the new pointers.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
        memcpy(this, &other, sizeof(BaryMeshViewWithInfo));
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
        setupPointers();
        return *this;
    }
};

//////////////////////////////////////////////////////////////////////////

class BarySaver
{
  public:
    // pointers within `content` must remain valid until saving completed
    bary::Result initContent(const bary::ContentView* content, bary::StandardPropertyType* pErrorProp = nullptr);

    // pointers within views must remain valid until saving completed
    bary::Result initContent(const bary::BasicView*      basic,
                             const bary::MeshView*       mesh       = nullptr,
                             const bary::MiscView*       misc       = nullptr,
                             bary::StandardPropertyType* pErrorProp = nullptr);


    // pointers within `content` must remain valid until saving completed
    // creates a single output file where the content of "initContent" and "appendedContent"
    // are linearly stored in the order of appending.
    // - only works if info structs match except for "count" variable.
    // - uses internal copy for infos to account for aggregation of "elementCount" or "valueCount"
    // - internal temp allocation to account for shifts in
    //   the various group "<attribute>First" variables. Shift is based on the
    //   prefix sum of the maxima (first + count) in previous files.
    bary::Result appendContent(const bary::ContentView* content, bary::StandardPropertyType* pErrorProp = nullptr);

    // pointers within views must remain valid until saving completed
    bary::Result appendContent(const bary::BasicView*      basic,
                               const bary::MeshView*       mesh       = nullptr,
                               const bary::MiscView*       misc       = nullptr,
                               bary::StandardPropertyType* pErrorProp = nullptr);


    // do not use standard properties covered by above functions here
    // this must cover all content, including appended and must be called
    // after appending was completed (if used at all)
    // pointers within `sinfo` must remain valid until saving completed
    void addCustomProperties(bary::PropertyStorageInfo sinfo);

    bary::Result save(const char* filename) const;
    bary::Result save(const std::string& filename) const { return save(filename.c_str()); }

    uint64_t     computeFileSize() const;
    bary::Result save(uint64_t fileSize, void* fileData) const;

  private:
    struct SaverContext
    {
        const BarySaver* saver      = nullptr;
        uint64_t         fileSize   = 0;
        uint8_t*         fileData   = nullptr;
        void*            fileHandle = nullptr;

        bary::Result save(uint64_t offset, uint64_t size, const void* data);
    };

    static bary::Result saverCallback(uint32_t                         propertyIdx,
                                      const bary::PropertyStorageInfo* propertyStorageInfo,
                                      uint64_t                         offset,
                                      uint64_t                         size,
                                      const void*                      data,
                                      bool                             isInfo,
                                      void*                            userData);

    static bary::Result fillPropertyStorageInfos(std::vector<bary::PropertyStorageInfo>& props,
                                                 const bary::ContentView*                content,
                                                 bary::StandardPropertyType*             pErrorProp);

    bary::Result                           m_result = bary::Result::eErrorUnknown;
    std::vector<bary::PropertyStorageInfo> m_props;
    // only used for appended
    std::vector<std::vector<bary::PropertyStorageInfo>> m_propsList;
    BaryContentInfo                                     m_aggregatedInfo{};
};

//////////////////////////////////////////////////////////////////////////

class BaryFileHandle;

struct BaryMemoryApi
{
    void* (*alloc)(size_t size, void* userData);
    void (*free)(void* ptr, void* userData);
    void* userData = nullptr;
};

struct BaryFileApi
{
    bary::Result (*read)(const BaryMemoryApi* memoryApi, const BaryFileApi* fileApi, const char* path, size_t* size, void** data) = nullptr;
    void (*release)(const BaryMemoryApi* memoryApi, const BaryFileApi* fileApi, void* data) = nullptr;
    void* userData                                                                          = nullptr;
};

struct BaryFileOpenOptions
{
    BaryMemoryApi memoryApi;
    BaryFileApi   fileApi;
};


// Class to open a file and provide quick access to standard
// content properties.
// The lifetime of all pointers depend on either the file handle
// or the provided fileData in `open`
class BaryFile
{
  private:
    BaryFileHandle* m_handle = nullptr;

    bary::Result setupContent(bary::StandardPropertyType* outErrorType);

  public:
    bary::ContentView m_content;

    uint64_t m_fileSize = 0;
    // either points inside `m_handle` or provided `fileData`
    const void* m_fileData = nullptr;

    // the lifetime of the provided raw pointer is externally managed by developer and must be valid as long as BaryFile is open
    bary::Result open(size_t fileSize, const void* fileData, bary::StandardPropertyType* outErrorType = nullptr)
    {
        m_fileSize = fileSize;
        m_fileData = fileData;
        return setupContent(outErrorType);
    }
    bary::Result open(const char* name, const BaryFileOpenOptions* options = nullptr, bary::StandardPropertyType* outErrorType = nullptr);
    bary::Result open(const std::string& name, const BaryFileOpenOptions* options = nullptr, bary::StandardPropertyType* outErrorType = nullptr)
    {
        return open(name.c_str(), options, outErrorType);
    }
    void close();

    bary::Result validate(bary::ValueSemanticType vtype, bary::StandardPropertyType* outErrorType = nullptr) const
    {
        return bary::baryContentIsValid(vtype, &m_content, outErrorType);
    }

    template <class T>
    const T* getValues() const
    {
        return reinterpret_cast<const T*>(m_content.basic.values);
    }

    bool isCompressed() const
    {
        return m_content.basic.valuesInfo
               && (m_content.basic.valuesInfo->valueFormat == bary::Format::eDispC1_r11_unorm_block
                   || m_content.basic.valuesInfo->valueFormat == bary::Format::eOpaC1_rx_uint_block);
    }

    bool hasProperty(bary::StandardPropertyType prop) const;

    const bary::BasicView&   getBasic() const { return m_content.basic; }
    const bary::MeshView&    getMesh() const { return m_content.mesh; }
    const bary::MiscView&    getMisc() const { return m_content.misc; }
    const bary::ContentView& getContent() const { return m_content; }

    void fillBasicData(BaryBasicData& data) const { data.setData(m_content.basic); }
    void fillMeshData(BaryMeshData& data) const { data.setData(m_content.mesh); }
    void fillMiscData(BaryMiscData& data) const { data.setData(m_content.misc); }
    void fillContentData(BaryContentData& data) const { data.setData(m_content); }

    ~BaryFile() { close(); }
};

//////////////////////////////////////////////////////////////////////////

struct BaryWUV_uint16
{
    uint16_t w;
    uint16_t u;
    uint16_t v;
};

using BaryUV_uint16 = bary::BaryUV_uint16;

inline BaryWUV_uint16 makeWUV(uint32_t w, uint32_t u, uint32_t v)
{
    return {uint16_t(w), uint16_t(u), uint16_t(v)};
}

inline BaryUV_uint16 makeUV(uint32_t w, uint32_t u, uint32_t v)
{
    return {uint16_t(u), uint16_t(v)};
}

inline BaryWUV_uint16 makeWUV(bary::BaryUV_uint16 uv, uint32_t subdivLevel)
{
    return {uint16_t((1u << subdivLevel) - uv.u - uv.v), uv.u, uv.v};
}

// This class is useful as lookup table
// of the storage indices for the different
// subdivision levels and a provided value layout.
// It also provides a minimal mesh representation that
// these layouts create.

class BaryLevelsMap
{
  public:
    // 1<<MAX_LEVEL must fit into BaryCoordIndex / 16 bit
    static const uint32_t MAX_LEVEL = 15;

    // This function is used to generate watertight meshes. It computes a new
    // BaryCoord according to edge decimation that is the result of multiple triangles of
    // different subdivision levels being joined.
    // The input BaryCoord along an edge may be snapped to another position along that edge,
    // if the decimate flags indicate that no vertex exists on the original position.
    // Set 1st, 2nd etc. bit in decimateEdgeBits to encode which edges have a neighbor with
    // half resolution.
    static BaryWUV_uint16 joinVertex(BaryWUV_uint16 bary, uint32_t decimateEdgeBits, uint32_t subdivLevel);

    // auxiliary utility
    typedef uint64_t     BaryCoordHash;
    static BaryCoordHash getHash(BaryWUV_uint16 bary)
    {
        return uint64_t(bary.w) | (uint64_t(bary.u) << 16) | (uint64_t(bary.v) << 32);
    }

    struct Triangle
    {
        uint32_t a;
        uint32_t b;
        uint32_t c;
    };

    struct Level
    {
        uint32_t          subdivLevel{};
        bary::ValueLayout layout;

        // data is stored in provided value layout
        std::vector<BaryWUV_uint16> coordinates;  // barycentric coordinates of micro-vertices
        std::vector<Triangle>       triangles;    // index topology of micro-triangles

        void getFloatCoord(size_t idx, float* vec) const
        {
            BaryWUV_uint16 coord = coordinates[idx];
            float          mul   = 1.0f / float(1 << subdivLevel);
            vec[0]               = float(coord.w) * mul;
            vec[1]               = float(coord.u) * mul;
            vec[2]               = float(coord.v) * mul;
        }

        uint32_t getCoordIndex(BaryWUV_uint16 coord) const
        {
            return baryValueLayoutGetIndex(layout, bary::ValueFrequency::ePerVertex, coord.u, coord.v, 0, subdivLevel);
        }

        uint32_t getBaryMax() const { return 1 << subdivLevel; }

        // returns indices vector for triangles taking the joining information
        // into account to collapse triangles at the edges for watertightness (avoid T-junctions).
        // decimateEdgeBits: set bit for those edges that gets half of the original segments
        // useDegenerated: keeps the collapsed triangles using degenerated triangle indices
        std::vector<Triangle> buildTrianglesWithCollapsedEdges(uint32_t decimateEdgeBits, bool useDegenerated = false) const;
    };

    const Level& getLevel(uint32_t subdivLevel) const
    {
        assert(subdivLevel < uint32_t(m_levels.size()));
        return m_levels[subdivLevel];
    }

    bool hasLevel(uint32_t subdivLevel) const { return subdivLevel < uint32_t(m_levels.size()); }

    uint32_t getNumLevels() const { return uint32_t(m_levels.size()); }

    bary::ValueLayout getLayout() const { return m_layout; }

    // using layout == bary::ValueLayout::eUndefined
    // will result in zero sized levels array

    void initialize(bary::ValueLayout layout, uint32_t maxSubdivLevel);

    BaryLevelsMap() {}
    BaryLevelsMap(bary::ValueLayout layout, uint32_t maxSubdivLevel) { initialize(layout, maxSubdivLevel); }

  private:
    std::vector<Level> m_levels;
    bary::ValueLayout  m_layout = bary::ValueLayout::eUndefined;
};

//////////////////////////////////////////////////////////////////////////

// pre-compute the compression-dependent splitting of
// triangles into block-triangles

class BarySplitTable
{
  public:
    struct Entry
    {
        std::vector<bary::BlockTriangle> tris;

        uint32_t getCount() const { return static_cast<uint32_t>(tris.size()); }

        void init(bary::BlockFormatDispC1 format, uint32_t baseSubdiv);
    };

    const Entry& get(bary::BlockFormatDispC1 format, uint32_t level) const { return m_splits[getIndex(format, level)]; }

    // returns true on error
    // for now only bary::Format::eDispC1_r11_unorm_block supported
    bool init(bary::Format format, uint32_t maxLevel);

  private:
    static inline uint32_t getFormatIdx(bary::BlockFormatDispC1 format) { return uint32_t(format) - 1; }
    inline uint32_t        getIndex(bary::BlockFormatDispC1 format, uint32_t level) const
    {
        assert(m_format == bary::Format::eDispC1_r11_unorm_block);
        return level * m_numFormats + getFormatIdx(format);
    }

    std::vector<Entry> m_splits;
    bary::Format       m_format{};
    uint32_t           m_numFormats{};
};

}  // namespace baryutils
