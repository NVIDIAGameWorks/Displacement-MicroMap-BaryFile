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

#include <cstddef>
#include <cstdint>

namespace bary
{
// bary
// ====
//
// bary is a container for data stored on a barycentric grid that
// is the result of uniform subdivision.
// There are three main properties stored in a bary file:
//     - Triangles
//     - Values
//     - Groups
//
// An input `Triangle` is subdivided evenly and as result creates
// `2 ^ (sudvision level * 2)` output triangles. The bary file will store
// either per-vertex or per-triangle data (`ValueFrequency`) for those outputs
// on a spatial curve (`ValueOrder`) across the subdivided triangle.
// Each `Triangle` stores the subdivision level as well as an offset where
// the `Values` for this triangle start.
// There is typically a 1:1 mapping between a 3d mesh triangle and the
// bary file triangle, but it's also possible a custom mapping buffer is
// used (a bit like a UV coordinate).
//
// The bary container can contain multiple independent `Groups` of triangle
// and values pairings. This is useful to store barycentric data for multiple
// 3d meshes in one file.
//
// Be aware, as barycentric data is relative to the winding of mesh triangles, there is a
// strong coupling between the 3D mesh and its barycentric data.
// For regular textures UV-coordinates are responsible for the coupling and allow
// to work independent of topological changes, but for barycentric data that is not
// the case. Barycentric data should be considered "for final deployment" foremost
// and not necessarily as portable data container while assets are being changed.
//

//////////////////////////////////////////////

enum class Result : uint32_t
{
    eSuccess = 0,
    eErrorUnknown,
    // provided file size is mismatching or too small
    eErrorFileSize,
    // ?Offset/?Length or ?First/?Count pair doesn't fit
    eErrorRange,
    // index doesn't fit
    eErrorIndex,
    // ?Offset or ?First has unexpected value
    // typically affects values with strict offset/first requirements
    eErrorOffset,
    // ?Size is unexpected
    eErrorSize,
    // ?Length/?Count is unexpected
    eErrorCount,
    // ?Alignment is unexpected
    eErrorAlignment,
    // version identifier format is invalid
    eErrorVersionFormat,
    // version does not match this header
    eErrorVersion,
    // unexpected format usage
    eErrorFormat,
    // unexpected block format usage
    eErrorBlockFormat,
    // the value of a provided parameter is invalid
    eErrorValue,
    // the value of a provided bit flag is invalid
    eErrorFlag,
    // the ordering of offsets (?Offset / ?First) is wrong
    eErrorOffsetOrder,
    // the ordering of an index is wrong
    eErrorIndexOrder,
    // the alignment of an offset is wrong
    eErrorOffsetAlignment,
    // a mandatory property is missing
    eErrorMissingProperty,
    // a property existed more than once
    eErrorDuplicateProperty,
    // a property cannot be used due to another property's state
    eErrorInvalidPropertyCombination,
    // a property cannot be used due to incompatibility (e.g. format mismatch)
    eErrorPropertyMismatch,
    // the supercompression state is unexpected
    eErrorSupercompression,

    // these are for external tools
    // as the core lib doesn't allocate memory,
    // nor do file operations
    // IO operation failed
    eErrorIO,
    // Allocation failed
    eErrorOutOfMemory,
};

// defines the storage layout, which affects the ordering of values within a subdivided primitive
enum class ValueLayout : uint32_t
{
    eUndefined,

    // eTriangleUmajor
    // is a simple ordering of rows
    // parallel to the WV edge starting from W
    // towards U (aka u-major).
    //
    // Vertex ordering
    // *******************************
    // *              V              *
    // *              4              *
    // *            /  \             *
    // *           3 __  8           *
    // *         /  \  /  \          *
    // *        2 __  7 __ 11        *
    // *      /  \  /  \  /  \       *
    // *     1 __  6 __ 10 __ 13     *
    // *   /  \  /  \  /  \  /  \    *
    // *  0 __ 5 __   9 __ 12 __ 14  *
    // * W                         U *
    // *******************************
    //
    // Triangle ordering
    // ******************************
    // *              V             *
    // *              .             *
    // *            / 6\            *
    // *           . __  .          *
    // *         / 4\5 /11\         *
    // *        . __  . __  .       *
    // *      / 2\3 / 9\10/14\      *
    // *     . __  . __ . __  .     *
    // *   / 0\1 / 7\8 /12\13/15\   *
    // *  . __  . __  . __  . __ .  *
    // * W                        U *
    // ******************************
    // 
    // values stored left to right, bottom to top
    // triangles stored in same ordering
    //

    eTriangleUmajor,

    // eTriangleBirdCurve is a
    // special hierarchical space filling curve
    //
    //  each subdiv level adds the new vertices that are the result
    //  of splitting triangles along their edges.
    //
    // Vertex ordering
    // *****************************************************************
    // *              V              `` subdiv level 0                 *
    // *              2``             ` subdiv level 1 (splits: 0,1,2) *
    // *            /  \           rest subdiv level 2 (splits: 0,5,3  *
    // *          12 __ 13                                      5,1,4  *
    // *         /  \  /  \                                     3,4,2) *
    // *        3`__ 14   4`                                           *
    // *      /  \  /  \  /  \                                         *
    // *     6 __  7 __  9 __ 10                                       *
    // *   /  \  /  \  /  \  /  \                                      *
    // *  0``__ 8 __ 5` __ 11 __ 1``                                   *
    // * W                        U                                    *
    // *****************************************************************
    // 
    //  example triangle order for subdiv level 2
    //
    // ******************************
    // * Triangle ordering          *
    // *             V              *
    // *              .             *
    // *            /15\            *
    // *           . __  .          *
    // *         /14\13/12\         *
    // *        . __  . __  .       *
    // *      / 3\4 / 5\ 6/11\      *
    // *     . __  . __ . __  .     *
    // *   / 0\1 / 2\7 / 8\ 9/10\   *
    // *  . __  . __  . __  . __ .  *
    // * W                        U *
    // ******************************

    eTriangleBirdCurve,
};

enum class ValueFrequency : uint32_t
{
    eUndefined,
    ePerVertex,
    ePerTriangle,
};

enum class ValueSemanticType : uint32_t
{
    // any value format
    eGeneric,

    // For scalar displacement the following value formats are acceptable:
    //  eR8_unorm
    //  eDispC1_r11_unorm_block (is compressed)
    //  eR11_unorm_pack16
    //  eR11_unorm_packed_align32
    //  eR16_unorm
    //  eR32_sfloat
    eDisplacement,
};

// values
enum class Format : uint32_t
{
    // enum values match VK_FORMAT

    eUndefined     = 0,
    eR8_unorm      = 9,
    eR8_snorm      = 10,
    eR8_uint       = 13,
    eR8_sint       = 14,
    eRG8_unorm     = 16,
    eRG8_snorm     = 17,
    eRG8_uint      = 20,
    eRG8_sint      = 21,
    eRGB8_unorm    = 23,
    eRGB8_snorm    = 24,
    eRGB8_uint     = 27,
    eRGB8_sint     = 28,
    eRGBA8_unorm   = 37,
    eRGBA8_snorm   = 38,
    eRGBA8_uint    = 41,
    eRGBA8_sint    = 42,
    eR16_unorm     = 70,
    eR16_snorm     = 71,
    eR16_uint      = 74,
    eR16_sint      = 75,
    eR16_sfloat    = 76,
    eRG16_unorm    = 77,
    eRG16_snorm    = 78,
    eRG16_uint     = 81,
    eRG16_sint     = 82,
    eRG16_sfloat   = 83,
    eRGB16_unorm   = 84,
    eRGB16_snorm   = 85,
    eRGB16_uint    = 88,
    eRGB16_sint    = 89,
    eRGB16_sfloat  = 90,
    eRGBA16_unorm  = 91,
    eRGBA16_snorm  = 92,
    eRGBA16_uint   = 95,
    eRGBA16_sint   = 96,
    eRGBA16_sfloat = 97,
    eR32_uint      = 98,
    eR32_sint      = 99,
    eR32_sfloat    = 100,
    eRG32_uint     = 101,
    eRG32_sint     = 102,
    eRG32_sfloat   = 103,
    eRGB32_uint    = 104,
    eRGB32_sint    = 105,
    eRGB32_sfloat  = 106,
    eRGBA32_uint   = 107,
    eRGBA32_sint   = 108,
    eRGBA32_sfloat = 109,
    eR64_uint      = 110,
    eR64_sint      = 111,
    eR64_sfloat    = 112,
    eRG64_uint     = 113,
    eRG64_sint     = 114,
    eRG64_sfloat   = 115,
    eRGB64_uint    = 116,
    eRGB64_sint    = 117,
    eRGB64_sfloat  = 118,
    eRGBA64_uint   = 119,
    eRGBA64_sint   = 120,
    eRGBA64_sfloat = 121,


    // opacity encoding (based on VK NV extension reservation 397)

    // block-compressed opacity format
    // for uncompressed 1 or 2 bit data stored in 8-bit
    // valueByteSize = 1
    eOpaC1_rx_uint_block = 1000396000,

    // displacement encoding  (based on VK NV extension reservation 398)

    // block-compressed displacement format
    // for compressed data stored in blocks of 512- or 1024-bit
    // valueByteSize = 1
    // valueByteAlignment = 128
    // minmax as eR11_unorm_pack16
    eDispC1_r11_unorm_block = 1000397000,

    // for uncompressed data 1 x 11 bit stored in 16-bit
    eR11_unorm_pack16 = 1000397001,

    // variable packed format
    // for uncompressed data 1 x 11 bit densely packed sequence of 32bit values.
    // Each triangle starts at a 32-bit boundary.
    // valueByteSize = 1
    // minmax as eR11_unorm_pack16
    eR11_unorm_packed_align32 = 1000397002,
};

enum class BlockFormatOpaC1 : uint16_t
{
    eInvalid    = 0,
    eR1_uint_x8 = 1,
    eR2_uint_x4 = 2,
};

enum class BlockFormatDispC1 : uint16_t
{
    eInvalid                 = 0,
    eR11_unorm_lvl3_pack512  = 1,
    eR11_unorm_lvl4_pack1024 = 2,
    eR11_unorm_lvl5_pack1024 = 3,
};

struct ValueFloatVector
{
    float r;
    float g;
    float b;
    float a;
};

//////////////////////////////////////////////////////////////////////////
// Bary File Memory Layout
//
// - Preamble
//      - `Header`
//      - array of `PropertyInfo` (pointed to by `Header::propertyInfoRange`)
// - data for all properties (pointed to by `PropertyInfo::range`)
//   - data might be interleaved with `PropertyInfo::supercompressionGlobalData`
//     after each property
//
// All byte alignments must be power of 2 and at least 4 bytes

struct VersionIdentifier
{
    char data[16];
};

// use baryGetCurrentVersionIdentifier()
// '\xAB', 'B', 'A', 'R', 'Y', ' ', '0', '0', '1', '0', '0', '\xBB', '\r', '\n', '\x1A', '\n'

struct ByteRange
{
    // unless mentioned otherwise must be 4 byte aligned
    uint64_t byteOffset;
    uint64_t byteLength;
};

struct Header
{
    VersionIdentifier version;
    // size includes sizeof(header) and all subsequent properties
    uint64_t totalByteSize;
    // stores PropertyInfo[]
    // all PropertyInfo byte ranges must come after header and stay within
    // totalByteSize, their ranges must be non-overlapping and
    // ordered with ascending ByteRange::byteOffset
    ByteRange propertyInfoRange;
};

struct PropertyIdentifier
{
    uint32_t uuid4[4];
};


// Properties can be super-compressed.
// At the time of writing this feature is not in use yet, but anticipated for future use.
// If a property contains a leading preamble / "Info" struct, that struct will always
// exist uncompressed (including padding), only the variable length payload will be
// compressed.
enum class SupercompressionScheme : uint32_t
{
    eNone = 0,
};

struct PropertyInfo
{
    // UUID for a property, every new property or every time the
    // content definition of a property changes, its identifier must as well.
    PropertyIdentifier identifier;
    // byte range must be after the header, and within
    // byteSize of header.
    ByteRange range;

    SupercompressionScheme supercompressionScheme;
    // if `supercompressionScheme` is != SupercompressionScheme::eNone
    // then 'range.byteLength` is the supercompressed size and
    // this value provides the length after decompression.
    uint64_t uncompressedByteLength;
    // if exists, global data must be 8 byte aligned
    // and come after the above primary `range` for this
    // property
    ByteRange supercompressionGlobalData;
};

//////////////////////////////////////////////////////////////////////////
// Utilities for saving and validation

// utility structure not stored in bary file,
// used for saving & validation
struct PropertyStorageInfo
{
    // which property is stored
    PropertyIdentifier identifier;

    // total size of the property will be the sum of dataSize, infoSize and infoPaddingSize

    // data of the property
    uint64_t    dataSize = 0;
    const void* data     = nullptr;

    // optional, for convenience the leading "info" struct of a property
    // can be provided separately
    // `data` pointer is then assumed to be the payload after the info struct
    // `dataSize` is then the size of the payload alone.
    uint64_t infoSize = 0;
    // if padding between `info` and `data` section is required.
    uint64_t    infoPaddingSize = 0;
    const void* info            = nullptr;

    SupercompressionScheme supercompressionScheme         = SupercompressionScheme::eNone;
    uint64_t               uncompressedSize               = 0;
    uint64_t               supercompressionGlobalDataSize = 0;
    const void*            supercompressionGlobalData     = nullptr;

    PropertyStorageInfo() { identifier.uuid4[0] = identifier.uuid4[1] = identifier.uuid4[2] = identifier.uuid4[3] = 0; }
};


// Standard properties use the current uuids and definitions in this header.
//
// Custom or older properties can still be stored in a file.
// However, then such properties are defined separately
// (e.g. bary_deprecated.h) and are no longer a standard type
//
// utility enum not stored in bary file
// used for saving & validation
enum class StandardPropertyType : uint32_t
{
    // non-standard properties can be stored as well
    // those are strictly identified by the UUID4 PropertyIdentifier
    eUnknown,

    // Mandatory properties:

    // stores ValueInfo + values
    eValues,
    // stores Group[]
    eGroups,
    // stores Triangle[]
    eTriangles,

    // Optional properties:
    // some can be considered mandatory depending on the usage
    // of a file. See `bary_displacement.h` for such an example.

    // stores TriangleMinMaxsInfo + data
    // These contain lower and upper bounds on the data per triangle;
    // including or computing them can improve performance of rasterization
    // implementations (e.g. by improving occlusion culling bounds).
    eTriangleMinMaxs,

    // stores TriangleUncompressedMip[]
    // info for uncompressed lower resolution mip-level for triangles
    // (can be sparse)
    // Including or computing these can improve performance of rasterization
    // implementations by simplifying decoding of distance data at low LOD.
    eTriangleUncompressedMips,
    // stores TriangleUncompressedMipDataInfo + data
    // uncompressed mip data referenced by TriangleUncompressedMip
    eUncompressedMips,
    // stores GroupUncompressedMip[]
    // must match Group[] count
    eGroupUncompressedMips,

    // stores HistogramEntry[]
    eHistogramEntries,
    // stores GroupHistogramRange[] per group
    eGroupHistograms,

    // Mesh properties:
    // may, or may not be stored within a bary file
    // Depending on the application and 3d mesh file
    // that this bary file is used with, these properties may be stored
    // in the 3d mesh file itself or referenced as being part of the bary file.
    // It is important to note that a strong coupling between 3d mesh
    // and its barycentric data exist.

    // stores MeshGroup[] per mesh group
    eMeshGroups,

    // stores MeshHistogramEntry[]
    eMeshHistogramEntries,
    // stores MeshGroupHistogramRange[] per mesh group
    // meshTriangleMappings are applied to account
    // for bary::Triangle being instanced from
    // multiple mesh triangles.
    eMeshGroupHistograms,

    // stores {MeshDirectionsInfo + data}
    // used for displacement, 3d mesh per-vertex
    eMeshDisplacementDirections,

    // stores MeshDirectionBoundsInfo + data
    // used for displacement, 3d mesh per-vertex
    eMeshDisplacementDirectionBounds,

    // stores MeshTriangleMappingsInfo + data
    // index to map a 3d mesh triangle to a bary file triangle
    eMeshTriangleMappings,

    // stores MeshTriangleFlagsInfo + data
    // special per mesh triangle flags:
    //  - currently 1 bit per edge, if set means the neighboring triangle has
    //    one subdivision level less. Edge order for the triangle (v0,v1,v2)
    //    is (v0,v1) (v1,v2) (v2,v0)
    eMeshTriangleFlags,

    // stores MeshPositionsInfo + data
    // bary files typically don't store this, but useful for debugging
    // or asserting the 3d mesh matches expectations / triangle winding...
    eMeshPositions,

    // stores MeshTrianglesInfo + data
    // bary files typically don't store this, but useful for debugging
    // or asserting the 3d mesh matches expectations / triangle winding...
    eMeshTriangleIndices,
};

template <StandardPropertyType>
PropertyIdentifier baryMakeStandardPropertyIdentifierT();

template <class T>
PropertyIdentifier baryMakePropertyIdentifierT();

#define BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(t, e, u0, u1, u2, u3)                                                   \
    template <>                                                                                                        \
    inline PropertyIdentifier baryMakeStandardPropertyIdentifierT<e>()                                                 \
    {                                                                                                                  \
        return {u0, u1, u2, u3};                                                                                       \
    }                                                                                                                  \
    template <>                                                                                                        \
    inline PropertyIdentifier baryMakePropertyIdentifierT<t>()                                                         \
    {                                                                                                                  \
        return {u0, u1, u2, u3};                                                                                       \
    }

#define BARY_MAKE_PROPERTY_IDENTIFIER(t, u0, u1, u2, u3)                                                               \
    template <>                                                                                                        \
    inline PropertyIdentifier baryMakePropertyIdentifierT<t>()                                                         \
    {                                                                                                                  \
        return {u0, u1, u2, u3};                                                                                       \
    }

/*
    lua code to generate numbers from uuid strings
    for uuid in test:gmatch("([%w%-]+)") do
        uuid = uuid:gsub("%-","")
        print("0x"..uuid:sub(1,8)..",0x"..uuid:sub(9,16)..",0x"..uuid:sub(17,24)..",0x"..uuid:sub(25,32))
    end
*/

//////////////////////////////////////////////////////////////////////////
// Bary File Property Data

struct ValuesInfo
{
    Format         valueFormat;
    // spatial layout of values across the subdivided triangle
    ValueLayout    valueLayout;
    // per-vertex or per-triangle
    ValueFrequency valueFrequency;
    // how many values there are in total (or bytes for compressed / special packed formats)
    uint32_t       valueCount;
    // compressed or special packed formats must use
    // valueByteSize 1
    uint32_t valueByteSize;
    // valueByteAlignment must be at least 4 bytes, higher alignment only
    // if it is power of two and either matching valueByteSize, or if special formats
    // demand for it. (e.g. eRG32_sfloat is 8 byte aligned, but eRGB32_sfloat 4 byte)
    uint32_t valueByteAlignment;
    // followed by padding (if required) then values data
};
// stores ValueInfo + data
// property size = sizeof(ValueInfo) + (padding) + valueCount * valueByteSize
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(ValuesInfo, StandardPropertyType::eValues, 0xb44daa04, 0xc9e044d5, 0x9a944de0, 0xcfd8fe35)

struct Group
{
    // first/count must be linear ascending and non-overlapping

    uint32_t triangleFirst;
    uint32_t triangleCount;
    uint32_t valueFirst;
    uint32_t valueCount;

    uint32_t minSubdivLevel;
    uint32_t maxSubdivLevel;

    // for UNORM,SNORM,FLOAT values these
    // represent the final value range
    // (value * scale) + bias
    ValueFloatVector floatBias;
    ValueFloatVector floatScale;
};
// stores Group[]
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(Group, StandardPropertyType::eGroups, 0x39ee40d0, 0x9dc44517, 0x8e5ab15d, 0xb09c74bc)

struct Triangle
{
    // valuesOffset must be ascending from t to t+1
    // and are relative to the group that this triangle belongs to
    // for uncompressed: serves as indexOffset (valueFormat agnostic)
    // for compressed / special packed: serves as byteOffset (given valueByteSize is 1)
    uint32_t valuesOffset;
    uint16_t subdivLevel;
    union
    {
        uint16_t          blockFormat;
        BlockFormatDispC1 blockFormatDispC1;
        BlockFormatOpaC1  blockFormatOpaC1;
    };
};
// stores Triangle[]
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(Triangle, StandardPropertyType::eTriangles, 0x00458e68, 0xee59426c, 0xb3bf1b7f, 0x749deb8e)

struct HistogramEntry
{
    // The histogram provides detailed usages wihtin compressed files
    // to other processing steps and avoids iterating all triangles
    // manually.
    //
    // Each entry stores how many bary triangles are used
    // with a unique pairing of block format
    // and subdivision level.
    uint32_t count;
    uint32_t subdivLevel;
    union
    {
        // intentional 32-bit usage here
        uint32_t          blockFormat;
        BlockFormatDispC1 blockFormatDispC1;
        BlockFormatOpaC1  blockFormatOpaC1;
    };
};
// stores HistogramEntry[]
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(HistogramEntry, StandardPropertyType::eHistogramEntries, 0x6e756bc1, 0x839a438a, 0xad4a745e, 0x556c3851)

struct GroupHistogramRange
{
    // into StandardPropertyType::eHistogramEntries
    // which histogram entries are valid for this group
    uint32_t entryFirst;
    uint32_t entryCount;
};

// stores HistogramRange[]
//
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(GroupHistogramRange, StandardPropertyType::eGroupHistograms, 0x400d972f, 0x76dd4337, 0x9900c6e9, 0xc9c46ffe)

struct MeshGroup
{
    uint32_t triangleFirst;
    uint32_t triangleCount;
    uint32_t vertexFirst;
    uint32_t vertexCount;
};

BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(MeshGroup, StandardPropertyType::eMeshGroups, 0x94beebda, 0x7ec347f0, 0xa5f1de0e, 0xcfbb851b)

struct MeshHistogramEntry
{
    // guaranteed to match HistogramEntry

    // The histogram provides detailed usages wihtin compressed files
    // to other processing steps and avoids iterating all triangles
    // manually.
    //
    // Each entry stores how many mesh triangles are used
    // with a unique pairing of block format
    // and subdivision level.
    uint32_t count;
    uint32_t subdivLevel;
    union
    {
        // intentional 32-bit usage here
        uint32_t          blockFormat;
        BlockFormatDispC1 blockFormatDispC1;
        BlockFormatOpaC1  blockFormatOpaC1;
    };
};
// stores MeshHistogramEntry[]
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(MeshHistogramEntry, StandardPropertyType::eMeshHistogramEntries, 0x68ce84e0, 0x7f4448f3, 0xbf58ba6e, 0x08d8cfd0)

struct MeshGroupHistogramRange
{
    // into StandardPropertyType::eMeshHistogramEntries
    // which histogram entries are valid for this mesh group
    // mesh groups only need to exist if TriangleMappings exist
    // so that multiple mesh triangles may map to the same bary triangle.
    uint32_t entryFirst;
    uint32_t entryCount;
};

BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(MeshGroupHistogramRange, StandardPropertyType::eMeshGroupHistograms, 0x02a37898, 0x74564056, 0x93b1bed7, 0x7f05386e)

struct MeshDisplacementDirectionsInfo
{
    // eRGB32_sfloat, eRGBA32_sfloat, eRGB16_sfloat or eRGBA16_sfloat
    // per-vertex displacement directions (linearly interpolated, without normalization)
    // if omitted the mesh vertex normals are to be used
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment; // note: use 4 bytes for eRGB16_sfloat
    // followed by padding (if required) then values data
};

// stores MeshDirectionsInfo + data
// property size = sizeof(MeshDirectionsInfo) + count * elementByteSize
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(MeshDisplacementDirectionsInfo,
                                       StandardPropertyType::eMeshDisplacementDirections,
                                       0xf262d687,
                                       0xb9284aeb,
                                       0xa706803c,
                                       0xcbedae52)

struct MeshDisplacementDirectionBoundsInfo
{
    // eRG32_sfloat or eRG16_sfloat
    // per-vertex {bias,scale}
    // to maximize the quantization of displacement values, a per-vertex
    // {bias,scale} allows to define the shell for the unsigned normalized displacements.
    //
    // displaced_position = interpolated(vertex_position + vertex_displacement_direction * bounds_bias) +
    //                      interpolated(vertex_displacement_direction * bounds_scale) * displacement_value;
    //
    // `interpolated` stands for the linear barycentric interpolation of those resulting vertex values of the triangle
    //
    // If direction bounds are used, Group::floatBias must be 0 and Group::floatScale 1.0

    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};
// stores MeshDirectionBoundsInfo + data
// property size = sizeof(MeshDirectionBoundsInfo) + count * elementByteSize
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(MeshDisplacementDirectionBoundsInfo,
                                       StandardPropertyType::eMeshDisplacementDirectionBounds,
                                       0x25bf3c65,
                                       0x29234ae1,
                                       0x95efe43c,
                                       0xeb87066c)

struct MeshTriangleMappingsInfo
{
    // eR32_uint or eR16_uint
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};
// stores MeshTriangleMappingsInfo + data
// property size = sizeof(MeshTriangleMappingsInfo) + count * elementByteSize
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(MeshTriangleMappingsInfo, StandardPropertyType::eMeshTriangleMappings, 0x9cdc3ad0, 0x92bb4a8a, 0xb4658759, 0x42d41d54)

struct MeshTriangleFlagsInfo
{
    // eR8_uint
    // special per mesh triangle flags:
    //  - currently 1 bit per edge, if set means the neighboring triangle of that edge
    //    has one subdivision level less.
    //    Edge order for the triangle (v0,v1,v2) is (v0,v1) (v1,v2) (v2,v0)
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};
// stores MeshTriangleFlagsInfo + data
// property size = sizeof(MeshTriangleFlagsInfo) + count * elementByteSize
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(MeshTriangleFlagsInfo, StandardPropertyType::eMeshTriangleFlags, 0x90f9eed3, 0x4ec34974, 0x970c755c, 0xaf5b53a3)

struct MeshPositionsInfo
{
    // eRGB32_sfloat, eRGBA32_sfloat, eRGBA16_sfloat or eRGB16_sfloat (alpha ignored)
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;  // note: use 4 bytes for eRGB16_sfloat
    // followed by padding (if required) then values data
};
// stores MeshPositionsInfo + data
// property size = sizeof(MeshPositionsInfo) + count * elementByteSize
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(MeshPositionsInfo, StandardPropertyType::eMeshPositions, 0xac071cfe, 0xc01d430d, 0x936ff822, 0xa2d6b48e)

struct MeshTriangleIndicesInfo
{
    // eR32_uint or eR16_uint
    // 3 indices per triangle
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};
// stores MeshTriangleIndicesInfo + data
// property size = sizeof(MeshTriangleIndicesInfo) + count * elementByteSize
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(MeshTriangleIndicesInfo, StandardPropertyType::eMeshTriangleIndices, 0x48f106db, 0x1daf410f, 0x8cf1c35c, 0x69559309)

struct TriangleMinMaxsInfo
{
    // {min, max} value pairs per triangle
    // format must always be uncompressed
    Format elementFormat;
    // count must be 2 x triangle count
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};
// stores TriangleMinMaxsInfo + data
// property size = sizeof(TriangleMinMaxsInfo) + count * elementByteSize
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(TriangleMinMaxsInfo, StandardPropertyType::eTriangleMinMaxs, 0x23010706, 0x56744eb7, 0x8c0d6ced, 0x5138d2f9)

struct TriangleUncompressedMip
{
    // The element offset of this triangle's first value in the
    // UncompressedMipsInfo values array, relative to the start of the group.
    // This is equivalent to a byte offset of elementByteSize * mipOffset bytes.
    // Can be ~0 if this triangle doesn't need/have a special mip block;
    // otherwise, must be ascending from triangle t to t+1.
    uint32_t mipOffset;
    uint32_t subdivLevel;
};

// stores TriangleUncompressedMip[]
// count must match Triangle[]
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(TriangleUncompressedMip, StandardPropertyType::eTriangleUncompressedMips, 0xd8b5df1c, 0xcc2b41ef, 0x84fbb4df, 0xbff10a14)

struct GroupUncompressedMip
{
    // The element offset of this group's first value in the
    // UncompressedMipsInfo values array. This is equivalent to a byte offset
    // of elementByteSize * mipFirst bytes.
    uint32_t mipFirst;
    // The number of UncompressedMipsInfo values in this group. This group
    // spans elements mipFirst to (but not including) mipFirst + mipCount.
    uint32_t mipCount;
};

// stores GroupUncompressedMip[]
// count must match Group[]
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(GroupUncompressedMip, StandardPropertyType::eGroupUncompressedMips, 0x57094946, 0xc47b4603, 0x8cb8b6e8, 0x6221cd27)

struct UncompressedMipsInfo
{
    // if valueFormat == eDispC1_r11_unorm_block then format must be eR11_unorm_packed_align32
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};

// stores TriangleUncompressedMipDataInfo + data
// property size = sizeof(TriangleUncompressedMipDataInfo) + count * elementByteSize
BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER(UncompressedMipsInfo, StandardPropertyType::eUncompressedMips, 0x585a66e6, 0xcb334423, 0xbe821357, 0xc471f552)

#undef BARY_MAKE_STANDARD_PROPERTY_IDENTIFIER

//////////////////////////////////////////////
//
// Following types are used in utility functions
// and not stored in files itself.
//
//////////////////////////////////////////////

struct BaryUV_uint16
{
    uint16_t u;
    uint16_t v;
};

// BlockFormat compression splits a base triangle (bary::Triangle)
// into block triangles.
// Each of these occupies a region of the original base triangle
// values. Each block format can cover a different number of
// values depending on its implicit subdivision level.
// BlockTriangle provides the details about the region and
// orientation.

struct BlockTriangle
{
    // three UV coordinates of this block triangle
    // relative to original base triangle (bary::Triangle)
    BaryUV_uint16 vertices[3];
    // flipped winding 0/1
    uint8_t flipped;
    // u and v sign relative to first vertex
    // bit 0: set if verticesUV[1].u > verticesUV[0].u
    // bit 1: set if verticesUV[2].v > verticesUV[0].v
    uint8_t signBits;
    // 3 x 2 bits that specify which local edge (0,1,2) maps to what
    // base edge (0,1,2) The value 3 means the local edge does not lie
    // on any base edge
    uint8_t baseEdgeIndices;
    uint8_t _reserved;
    // where this block's data starts relative to
    // original base triangle valuesOffset (which is in bytes for compressed data)
    uint32_t blockByteOffset;
};

//////////////////////////////////////////////

// utility views on standard property bary content

// BasicView is for properties that are most typical
// stored in bary files
struct BasicView
{
    // mandatory for all
    const Group*      groups         = nullptr;
    uint32_t          groupsCount    = 0;
    const ValuesInfo* valuesInfo     = nullptr;
    const uint8_t*    values         = nullptr;
    const Triangle*   triangles      = nullptr;
    uint32_t          trianglesCount = 0;
    // mandatory for compressed
    const HistogramEntry*      histogramEntries          = nullptr;
    uint32_t                   histogramEntriesCount     = 0;
    const GroupHistogramRange* groupHistogramRanges      = nullptr;
    uint32_t                   groupHistogramRangesCount = 0;
    // mandatory for displacement otherwise optional
    const TriangleMinMaxsInfo* triangleMinMaxsInfo = nullptr;
    const uint8_t*             triangleMinMaxs     = nullptr;
};

// MeshProps is for properties that may be stored in the 3d model file
// rather than in the bary file
struct MeshView
{
    // optional mesh properties
    const MeshGroup* meshGroups      = nullptr;
    uint32_t         meshGroupsCount = 0;

    const MeshHistogramEntry*      meshHistogramEntries          = nullptr;
    uint32_t                       meshHistogramEntriesCount     = 0;
    const MeshGroupHistogramRange* meshGroupHistogramRanges      = nullptr;
    uint32_t                       meshGroupHistogramRangesCount = 0;

    const MeshDisplacementDirectionsInfo*      meshDisplacementDirectionsInfo      = nullptr;
    const uint8_t*                             meshDisplacementDirections          = nullptr;
    const MeshDisplacementDirectionBoundsInfo* meshDisplacementDirectionBoundsInfo = nullptr;
    const uint8_t*                             meshDisplacementDirectionBounds     = nullptr;

    const MeshTriangleMappingsInfo* meshTriangleMappingsInfo = nullptr;
    const uint8_t*                  meshTriangleMappings     = nullptr;
    const MeshTriangleFlagsInfo*    meshTriangleFlagsInfo    = nullptr;
    const uint8_t*                  meshTriangleFlags        = nullptr;

    // uncommon, meant for debugging
    const MeshPositionsInfo*       meshPositionsInfo       = nullptr;
    const uint8_t*                 meshPositions           = nullptr;
    const MeshTriangleIndicesInfo* meshTriangleIndicesInfo = nullptr;
    const uint8_t*                 meshTriangleIndices     = nullptr;
};

// MiscPropsView is for properties not part of typical files
struct MiscView
{
    // optional mip properties
    const GroupUncompressedMip*    groupUncompressedMips         = nullptr;
    uint32_t                       groupUncompressedMipsCount    = 0;
    const TriangleUncompressedMip* triangleUncompressedMips      = nullptr;
    uint32_t                       triangleUncompressedMipsCount = 0;
    const UncompressedMipsInfo*    uncompressedMipsInfo          = nullptr;
    const uint8_t*                 uncompressedMips              = nullptr;
};

struct ContentView
{
    BasicView basic;
    MeshView  mesh;
    MiscView  misc;
};

}  // namespace bary