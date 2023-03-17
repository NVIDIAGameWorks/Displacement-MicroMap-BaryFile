// 
// SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// 

// This is an ImHex (imhex.werwolv.net) hex editor pattern file for
// analyzing .bary files. These have the 16-byte magic number
//          AB 42 41 52 59 20 30 30 31 30 30 BB 0D 0A 1A 0A
//   (text)     B  A  R  Y     0  0  1  0  0
// To use it inside ImHex, go to File > Load Pattern... and then select
// this file. Alternatively, one can install this file inside ImHex's
// patterns/ folder. I recommend creating an ImHex project file using
// File > Save Project...; that makes it faster to reload .bary files
// when they change.
//
// This has two main limitations at the moment: 
//
// - Only the first array_limit values in an array are displayed. This
// because ImHex halts execution prematurely if an array has too many
// elements, and .bary files can be relatively large.
//
// - There is no security guarantee or validation.
//
// Please also note that the documentation here is informative, not
// normative, and is copied from the bary_core headers. Please see
// bary_core's headers for the latest documentation.

#include <cstdint.pat>
#include <std/io.pat>
#include <std/mem.pat>
#include <std/string.pat>

#define ARRAY_LIMIT 0x1000
#pragma array_limit 0x1000

enum ValueLayout : uint32_t
{
    eUndefined,
    eTriangleUmajor,
    eTriangleBirdCurve,
};

enum ValueFrequency : uint32_t
{
    eUndefined,
    ePerVertex,
    ePerTriangle,
};

enum ValueSemanticType : uint32_t
{
    eGeneric,
    eDisplacement,
};

enum Format : uint32_t
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

enum BlockFormatOpaC1 : uint16_t
{
    eInvalid    = 0,
    eR1_uint_x8 = 1,
    eR2_uint_x4 = 2,
};

enum BlockFormatDispC1 : uint16_t
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

struct ByteRange
{
    // Unless mentioned otherwise, must be 4-byte aligned
    uint64_t byteOffset;
    uint64_t byteLength;
};

struct Header
{
    char versionIdentifier[16];
    // size includes sizeof(header) and all subsequent properties
    uint64_t totalByteSize;
    // stores PropertyInfo[]
    // all PropertyInfo byte ranges must come after header and stay within
    // totalByteSize; their ranges must be non-overlapping and
    // ordered with ascending ByteRange::byteOffset
    ByteRange propertyInfoRange;
};

// Represents a bary property UUID.
// Note that this is not the same as StandardPropertyType - although we use
// the same names, we include the GUIDs here directly so they're readable in
// the Pattern Data box. Additionally, anyone can generate new GUIDs to add
// custom properties to a .bary file.
// The straightforward approach of `enum UUID : u128` doesn't quite seem to
// work - when reading, the top 64 bits are filled with 0s - so we rely upon
// how the lower 64 bits are unique for all known bary UUIDs.
using UUIDHi = u64;
enum UUIDLo : u64
{
    // stores ValueInfo + values
    eValues = 0xc9e044d5b44daa04,
    // stores Group[]
    eGroups = 0x9dc4451739ee40d0,
    // stores Triangle[]
    eTriangles = 0xee59426c00458e68,

    // stores TriangleMinMaxsInfo + data
    eTriangleMinMaxs = 0x56744eb723010706,

    // stores TriangleUncompressedMip[]
    // info for uncompressed lower resolution mip-level for triangles
    // (can be sparse)
    // useful to accelerate decoding of block format compression
    eTriangleUncompressedMips = 0xcc2b41efd8b5df1c,
    // stores UncompressedMipsInfo + data
    // uncompressed mip data referenced by TriangleUncompressedMip
    eUncompressedMips = 0xcb334423585a66e6,
    // stores GroupUncompressedMip[]
    // must match Group[] count
    eGroupUncompressedMips = 0xc47b460357094946,

    // stores HistogramEntry[]
    eHistogramEntries = 0x839a438a6e756bc1,
    // stores GroupHistogramRange[] per group
    eGroupHistograms = 0x76dd4337400d972f,

    // Mesh properties:
    // may, or may not be stored within a bary file
    // Depending on the application and 3d mesh file
    // that this bary file is used with, these properties may be stored
    // in the 3d mesh file itself or referenced as being part of the bary file.
    // It is important to note that a strong coupling between 3d mesh
    // and its barycentric data exist.

    // stores MeshGroup[] per mesh group
    eMeshGroups = 0x7ec347f094beebda,

    // stores MeshHistogramEntry[]
    eMeshHistogramEntries = 0x7f4448f368ce84e0,
    // stores MeshGroupHistogramRange[] per mesh group
    // meshTriangleMappings are applied to account
    // for bary::Triangle being instanced from
    // multiple mesh triangles.
    eMeshGroupHistograms = 0x7456405602a37898,

    // stores {MeshDirectionsInfo + data}
    // used for displacement, 3d mesh per-vertex
    eMeshDisplacementDirections = 0xb9284aebf262d687,

    // stores MeshDirectionBoundsInfo + data
    // used for displacement, 3d mesh per-vertex
    eMeshDisplacementDirectionBounds = 0x29234ae125bf3c65,

    // stores MeshTriangleMappingsInfo + data
    // index to map a 3d mesh triangle to a bary file triangle
    eMeshTriangleMappings = 0x92bb4a8a9cdc3ad0,

    // stores MeshTriangleFlagsInfo + data
    // special per mesh triangle flags:
    //  - currently 1 bit per edge, if set means the neighboring triangle has
    //    one subdivision level less. Edge order for the triangle (v0,v1,v2)
    //    is (v0,v1) (v1,v2) (v2,v0)
    eMeshTriangleFlags = 0x4ec3497490f9eed3,

    // stores MeshPositionsInfo + data
    // bary files typically don't store this, but useful for debugging
    // or asserting the 3d mesh matches expectations / triangle winding...
    eMeshPositions = 0xc01d430dac071cfe,

    // stores MeshTrianglesInfo + data
    // bary files typically don't store this, but useful for debugging
    // or asserting the 3d mesh matches expectations / triangle winding...
    eMeshTriangleIndices = 0x1daf410f48f106db,
};

enum SupercompressionScheme : u32
{
    eNone = 0,
};

// Clamps an array length to the ImHex maximum.
fn clampArray(u128 len){
    if (len > ARRAY_LIMIT)
        return ARRAY_LIMIT;
    else
        return len;
};

//////////////////////////////////////////////////////////////////////////
// Format visualization structures

// The pattern language doesn't currently have half floats,
// so at the moment we display half floats as their raw data.
// Similarly, we display UNORM values as integers.

struct RG8_u
{
    u8 r;
    u8 g;
};

struct RG8_s
{
    s8 r;
    s8 g;
};

struct RGB8_u
{
    u8 r;
    u8 g;
    u8 b;
};

struct RGB8_s
{
    s8 r;
    s8 g;
    s8 b;
};

struct RGBA8_u
{
    u8 r;
    u8 g;
    u8 b;
    u8 a;
};

struct RGBA8_s
{
    s8 r;
    s8 g;
    s8 b;
    s8 a;
};

struct RG16_u
{
    u16 r;
    u16 g;
};

struct RG16_s
{
    s16 r;
    s16 g;
};

struct RGB16_u
{
    u16 r;
    u16 g;
    u16 b;
};

struct RGB16_s
{
    s16 r;
    s16 g;
    s16 b;
};

struct RGBA16_u
{
    u16 r;
    u16 g;
    u16 b;
    u16 a;
};

struct RGBA16_s
{
    s16 r;
    s16 g;
    s16 b;
    s16 a;
};

struct R16_sfloat
{
    u16 r_u16;
};

struct RG16_sfloat
{
    u16 r_u16;
    u16 g_u16;
};

struct RGB16_sfloat
{
    u16 r_u16;
    u16 g_u16;
    u16 b_u16;
};

struct RGBA16_sfloat
{
    u16 r_u16;
    u16 g_u16;
    u16 b_u16;
    u16 a_u16;
};

struct RG32_u
{
    u32 r;
    u32 g;
};
struct RG32_s
{
    s32 r;
    s32 g;
};

struct RGB32_u
{
    u32 r;
    u32 g;
    u32 b;
};

struct RGB32_s
{
    s32 r;
    s32 g;
    s32 b;
};

struct RGBA32_u
{
    u32 r;
    u32 g;
    u32 b;
    u32 a;
};

struct RGBA32_s
{
    s32 r;
    s32 g;
    s32 b;
    s32 a;
};

struct RG32_sfloat
{
    float r;
    float g;
};

struct RGB32_sfloat
{
    float r;
    float g;
    float b;
};

struct RGBA32_sfloat
{
    float r;
    float g;
    float b;
    float a;
};

struct RG64_u
{
    u64 r;
    u64 g;
};

struct RG64_s
{
    s64 r;
    s64 g;
};

struct RGB64_u
{
    u32 r;
    u32 g;
    u32 b;
};

struct RGB64_s
{
    s64 r;
    s64 g;
    s64 b;
};

struct RGBA64_u
{
    u64 r;
    u64 g;
    u64 b;
    u64 a;
};

struct RGBA64_s
{
    s64 r;
    s64 g;
    s64 b;
    s64 a;
};

struct RG64_sfloat
{
    double r;
    double g;
};

struct RGB64_sfloat
{
    double r;
    double g;
    double b;
};

struct RGBA64_sfloat
{
    double r;
    double g;
    double b;
    double a;
};

bitfield R11_unorm_pack16
{
    padding: 5;
    r : 11;
};

//////////////////////////////////////////////////////////////////////////
// In a couple of places, we'd like to visualize data that can take
// a variety of different formats. A function that defines several
// arrays doesn't quite seem to work, so here's another solution
// (though it's not very elegant!). We define a large macro,
// DATA_ARRAY, that includes the if/else statement over the different
// formats. Macros end at newlines -- they don't support the line
// separator \ -- so DATA_ARRAY is one line. The preprocessor doesn't
// have function macros as of this writing, so we store parameters in
// global variables. The following CommonHeader struct makes this a
// bit easier: when .bary standard properties have an Info object,
// its first 16 bytes always follow the layout of this struct:
struct CommonInfo
{
    Format elementFormat;
    u32    elementCount;
    u32    elementByteSize;
    u32    elementByteAlignment;
} [[hidden]];
// So we can write a function that reads from this struct and sets
// up the necessary parameters for DATA_ARRAY.

Format g_format;
u128 g_startByte;
u128 g_byteLength;
fn dataArrayPrepare(Format format, u128 byteOffset, u128 elementByteAlignment, u128 structSize, u128 byteLength)
{
    g_format     = format;
    g_startByte  = byteOffset + std::mem::align_to(elementByteAlignment, structSize);
    g_byteLength = byteLength;
};

fn dataArrayPrepareC(CommonInfo commonInfo, u128 byteOffset, u128 structSize)
{
    dataArrayPrepare(commonInfo.elementFormat,        //
                     byteOffset,                      //
                     commonInfo.elementByteAlignment, //
                     structSize,                      //
                     commonInfo.elementByteSize * commonInfo.elementCount);
};

#define DATA_ARRAY if(g_format == Format::eR8_unorm || g_format == Format::eR8_uint){u8 data[clampArray(g_byteLength)] @ g_startByte;} else if(g_format == Format::eR8_snorm || g_format == Format::eR8_sint){s8 data[clampArray(g_byteLength)] @ g_startByte;} else if(g_format == Format::eRG8_unorm || g_format == Format::eRG8_uint){RG8_u data[clampArray(g_byteLength)/2] @ g_startByte;} else if(g_format == Format::eRG8_snorm || g_format == Format::eRG8_sint){RG8_s data[clampArray(g_byteLength)/2] @ g_startByte;} else if(g_format == Format::eRGB8_unorm || g_format == Format::eRGB8_uint){RGB8_u data[clampArray(g_byteLength)/3] @ g_startByte;} else if(g_format == Format::eRGB8_snorm || g_format == Format::eRGB8_sint){RGB8_s data[clampArray(g_byteLength)/3] @ g_startByte;} else if(g_format == Format::eRGBA8_unorm || g_format == Format::eRGBA8_uint){RGBA8_u data[clampArray(g_byteLength)/4] @ g_startByte;} else if(g_format == Format::eRGBA8_snorm || g_format == Format::eRGBA8_sint){RGBA8_s data[clampArray(g_byteLength)/4] @ g_startByte;} else if(g_format == Format::eR16_unorm || g_format == Format::eR16_uint){u16 data[clampArray(g_byteLength)/2] @ g_startByte;} else if(g_format == Format::eR16_snorm || g_format == Format::eR16_sint){s16 data[clampArray(g_byteLength)/2] @ g_startByte;} else if(g_format == Format::eR16_sfloat){R16_sfloat data[clampArray(g_byteLength)/2] @ g_startByte;} else if(g_format == Format::eRG16_unorm || g_format == Format::eRG16_uint){RG16_u data[clampArray(g_byteLength)/4] @ g_startByte;} else if(g_format == Format::eRG16_snorm || g_format == Format::eRG16_sint){RG16_s data[clampArray(g_byteLength)/4] @ g_startByte;} else if(g_format == Format::eRG16_sfloat){RG16_sfloat data[clampArray(g_byteLength)/4] @ g_startByte;} else if(g_format == Format::eRGB16_unorm || g_format == Format::eRGB16_uint){RGB16_u data[clampArray(g_byteLength)/6] @ g_startByte;} else if(g_format == Format::eRGB16_snorm || g_format == Format::eRGB16_sint){RGB16_s data[clampArray(g_byteLength)/6] @ g_startByte;} else if(g_format == Format::eRGB16_sfloat){RGB16_sfloat data[clampArray(g_byteLength)/6] @ g_startByte;} else if(g_format == Format::eRGBA16_unorm || g_format == Format::eRGBA16_uint){RGBA16_u data[clampArray(g_byteLength)/8] @ g_startByte;} else if(g_format == Format::eRGBA16_snorm || g_format == Format::eRGBA16_sint){RGBA16_s data[clampArray(g_byteLength)/8] @ g_startByte;} else if(g_format == Format::eRGBA16_sfloat){RGBA16_sfloat data[clampArray(g_byteLength)/8] @ g_startByte;} else if(g_format == Format::eR32_uint){u32 data[clampArray(g_byteLength)/4] @ g_startByte;} else if(g_format == Format::eR32_sint){s32 data[clampArray(g_byteLength)/4] @ g_startByte;} else if(g_format == Format::eR32_sfloat){float data[clampArray(g_byteLength)/4] @ g_startByte;} else if(g_format == Format::eRG32_uint){RG32_u data[clampArray(g_byteLength)/8] @ g_startByte;} else if(g_format == Format::eRG32_sint){RG32_s data[clampArray(g_byteLength)/8] @ g_startByte;} else if(g_format == Format::eRG32_sfloat){RG32_sfloat data[clampArray(g_byteLength)/8] @ g_startByte;} else if(g_format == Format::eRGB32_uint){RGB32_u data[clampArray(g_byteLength)/12] @ g_startByte;} else if(g_format == Format::eRGB32_sint){RGB32_s data[clampArray(g_byteLength)/12] @ g_startByte;} else if(g_format == Format::eRGB32_sfloat){RGB32_sfloat data[clampArray(g_byteLength)/12] @ g_startByte;} else if(g_format == Format::eRGBA32_uint){RGBA32_u data[clampArray(g_byteLength)/16] @ g_startByte;} else if(g_format == Format::eRGBA32_sint){RGBA32_s data[clampArray(g_byteLength)/16] @ g_startByte;} else if(g_format == Format::eRGBA32_sfloat){RGBA32_sfloat data[clampArray(g_byteLength)/16] @ g_startByte;} else if(g_format == Format::eR64_uint){u64 data[clampArray(g_byteLength)/8] @ g_startByte;} else if(g_format == Format::eR64_sint){s64 data[clampArray(g_byteLength)/8] @ g_startByte;} else if(g_format == Format::eR64_sfloat){double data[clampArray(g_byteLength)/8] @ g_startByte;} else if(g_format == Format::eRG64_uint){RG64_u data[clampArray(g_byteLength)/16] @ g_startByte;} else if(g_format == Format::eRG64_sint){RG64_s data[clampArray(g_byteLength)/16] @ g_startByte;} else if(g_format == Format::eRG64_sfloat){RG64_sfloat data[clampArray(g_byteLength)/16] @ g_startByte;} else if(g_format == Format::eRGB64_uint){RGB64_u data[clampArray(g_byteLength)/24] @ g_startByte;} else if(g_format == Format::eRGB64_sint){RGB64_s data[clampArray(g_byteLength)/24] @ g_startByte;} else if(g_format == Format::eRGB64_sfloat){RGB64_sfloat data[clampArray(g_byteLength)/24] @ g_startByte;} else if(g_format == Format::eRGBA64_uint){RGBA64_u data[clampArray(g_byteLength)/32] @ g_startByte;} else if(g_format == Format::eRGBA64_sint){RGBA64_s data[clampArray(g_byteLength)/32] @ g_startByte;} else if(g_format == Format::eRGBA64_sfloat){RGBA64_sfloat data[clampArray(g_byteLength)/32] @ g_startByte;} else if(g_format == Format::eR11_unorm_pack16){R11_unorm_pack16 data[clampArray(g_byteLength)/2] @ g_startByte;} else {u8 rawData[clampArray(g_byteLength)] @ g_startByte;}

struct ValuesInfo
{
    Format valueFormat;
    // spatial layout of values across the subdivided triangle
    ValueLayout valueLayout;
    // per-vertex or per-triangle
    ValueFrequency valueFrequency;
    // how many values there are in total (or bytes for compressed / special packed formats)
    uint32_t valueCount;
    // compressed or special packed formats must use
    // valueByteSize 1
    uint32_t valueByteSize;
    // valueByteAlignment must be at least 4 bytes, higher alignment only
    // if it is power of two and either matching valueByteSize, or if special formats
    // demand for it. (e.g. eRG32_sfloat is 8 byte aligned, but eRGB32_sfloat 4 byte)
    uint32_t valueByteAlignment;
    // followed by padding (if required) then values data
};
ValuesInfo g_sizeValuesInfo;

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
Group g_sizeGroup; // We use sizeof(g_size...) because sizeof(Group) isn't currently allowed in the pattern syntax.

union BlockFormat
{
    uint16_t          blockFormat;
    BlockFormatDispC1 blockFormatDispC1;
    BlockFormatOpaC1  blockFormatOpaC1;
};

struct Triangle
{
    // valuesOffset must be ascending from t to t+1
    // and are relative to the group that this triangle belongs to
    // for uncompressed: serves as indexOffset (valueFormat agnostic)
    // for compressed / special packed: serves as byteOffset (given valueByteSize is 1)
    uint32_t    valuesOffset;
    uint16_t    subdivLevel;
    BlockFormat blockFormat;
};
Triangle g_sizeTriangle;

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
TriangleMinMaxsInfo g_sizeTriangleMinMaxsInfo;

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
TriangleUncompressedMip g_sizeTriangleUncompressedMip;

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
GroupUncompressedMip g_sizeGroupUncompressedMip;

struct UncompressedMipsInfo
{
    // if valueFormat == eDispC1_r11_unorm_block then format must be eR11_unorm_packed_align32
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};
UncompressedMipsInfo g_sizeUncompressedMipsInfo;

struct HistogramEntry
{
    // The histogram provides detailed usages wihtin compressed files
    // to other processing steps and avoids iterating all triangles
    // manually.
    //
    // Each entry stores how many bary triangles are used
    // with a unique pairing of block format
    // and subdivision level.
    uint32_t    count;
    uint32_t    subdivLevel;
    BlockFormat blockFormat;
    padding[2];
};
HistogramEntry g_sizeHistogramEntry;

struct GroupHistogramRange
{
    // into StandardPropertyType::eHistogramEntries
    // which histogram entries are valid for this group
    uint32_t entryFirst;
    uint32_t entryCount;
};
GroupHistogramRange g_sizeGroupHistogramRange;

struct MeshDisplacementDirectionsInfo
{
    // eR32G32B32_sfloat or eR16G16B16A16_sfloat
    // per-vertex displacement directions (linearly interpolated, without normalization)
    // if omitted the mesh vertex normals are to be used
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};
MeshDisplacementDirectionsInfo g_sizeMeshDisplacementDirectionsInfo;

struct MeshDisplacementDirectionBoundsInfo
{
    // eR32G32_sfloat or eR16G16_sfloat
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
MeshDisplacementDirectionBoundsInfo g_sizeMeshDisplacementDirectionBoundsInfo;

struct MeshGroup
{
    uint32_t triangleFirst;
    uint32_t triangleCount;
    uint32_t vertexFirst;
    uint32_t vertexCount;
};
MeshGroup g_sizeMeshGroup;

struct MeshTriangleMappingsInfo
{
    // eR32_uint or eR16_uint
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};
MeshGroup g_sizeMeshTriangleMappingsInfo;

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
MeshTriangleFlagsInfo g_sizeMeshTriangleFlagsInfo;

struct MeshPositionsInfo
{
    // eR32G32B32_sfloat or eR16G16B16A16_sfloat
    Format   elementFormat;
    uint32_t elementCount;
    uint32_t elementByteSize;
    uint32_t elementByteAlignment;
    // followed by padding (if required) then values data
};
MeshPositionsInfo g_sizeMeshPositionsInfo;

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
MeshTriangleIndicesInfo g_sizeMeshTriangleIndicesInfo;

struct PropertyInfo
{
    // UUID for a property, every new property or every time the
    // content definition of a property changes, its identifier must as well.
    UUIDLo identifier;
    UUIDHi identifierHi;
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

    padding[4];
    
    CommonInfo commonInfo @ range.byteOffset;
    
    if(identifier == UUIDLo::eValues){
      ValuesInfo valuesInfo @ range.byteOffset;
      // Look at the Triangle properties to interpret the data!
      dataArrayPrepare(valuesInfo.valueFormat,
               range.byteOffset,
               valuesInfo.valueByteAlignment,
               sizeof(g_sizeValuesInfo),
               valuesInfo.valueCount * valuesInfo.valueByteSize);
      DATA_ARRAY
    }
    else if(identifier == UUIDLo::eGroups){
      Group groups[clampArray(range.byteLength/sizeof(g_sizeGroup))] @ range.byteOffset;
    }
    else if(identifier == UUIDLo::eTriangles){
      Triangle triangles[clampArray(range.byteLength/sizeof(g_sizeTriangle))] @ range.byteOffset;
    }
    else if(identifier == UUIDLo::eTriangleMinMaxs){
      TriangleMinMaxsInfo triangleMinMaxsInfo @ range.byteOffset;
      dataArrayPrepareC(commonInfo, range.byteOffset, sizeof(g_sizeTriangleMinMaxsInfo));
      DATA_ARRAY
    }
    else if(identifier == UUIDLo::eTriangleUncompressedMips){
      TriangleUncompressedMip triangleUncompressedMips[clampArray(range.byteLength/sizeof(g_sizeTriangleUncompressedMip))] @ range.byteOffset;
    }
    else if(identifier == UUIDLo::eGroupUncompressedMips){
      GroupUncompressedMip groupUncompressedMips[clampArray(range.byteLength/sizeof(g_sizeGroupUncompressedMip))] @ range.byteOffset;
    }
    else if(identifier == UUIDLo::eUncompressedMips){
      UncompressedMipsInfo uncompressedMipsInfo @ range.byteOffset;
      dataArrayPrepareC(commonInfo, range.byteOffset, sizeof(g_sizeUncompressedMipsInfo));
      DATA_ARRAY
    }
    else if(identifier == UUIDLo::eMeshGroups){
      MeshGroup meshGroups[clampArray(range.byteLength/sizeof(g_sizeMeshGroup))] @ range.byteOffset;
    }
    else if(identifier == UUIDLo::eHistogramEntries){
      HistogramEntry histogramEntries[clampArray(range.byteLength/sizeof(g_sizeHistogramEntry))] @ range.byteOffset;
    }
    else if(identifier == UUIDLo::eGroupHistograms){
      GroupHistogramRange groupHistogramRanges[clampArray(range.byteLength/sizeof(g_sizeGroupHistogramRange))] @ range.byteOffset;
    }
    else if(identifier == UUIDLo::eMeshDisplacementDirections){
      MeshDisplacementDirectionsInfo meshDisplacementDirectionsInfo @ range.byteOffset;
      dataArrayPrepareC(commonInfo, range.byteOffset, sizeof(g_sizeMeshDisplacementDirectionsInfo));
      DATA_ARRAY
    }
    else if(identifier == UUIDLo::eMeshDisplacementDirectionBounds){
      MeshDisplacementDirectionBoundsInfo meshDisplacementDirectionBoundsInfo @ range.byteOffset;
      dataArrayPrepareC(commonInfo, range.byteOffset, sizeof(g_sizeMeshDisplacementDirectionBoundsInfo));
      DATA_ARRAY
    }
    else if(identifier == UUIDLo::eMeshTriangleMappings){
      MeshTriangleMappingsInfo meshTriangleMappingsInfo @ range.byteOffset;
      dataArrayPrepareC(commonInfo, range.byteOffset, sizeof(g_sizeMeshTriangleMappingsInfo));
      DATA_ARRAY
    }
    else if(identifier == UUIDLo::eMeshTriangleFlags){
      MeshTriangleFlagsInfo meshTriangleFlagsInfo @ range.byteOffset;
      dataArrayPrepareC(commonInfo, range.byteOffset, sizeof(g_sizeMeshTriangleFlagsInfo));
      DATA_ARRAY
    }
    else if(identifier == UUIDLo::eMeshPositions){
      MeshPositionsInfo meshPositionsInfo @ range.byteOffset;
      dataArrayPrepareC(commonInfo, range.byteOffset, sizeof(g_sizeMeshPositionsInfo));
      DATA_ARRAY
    }
    else if(identifier == UUIDLo::eMeshTriangleIndices){
      MeshTriangleIndicesInfo meshTriangleIndicesInfo @ range.byteOffset;
      dataArrayPrepareC(commonInfo, range.byteOffset, sizeof(g_sizeMeshTriangleIndicesInfo));
      DATA_ARRAY
    }
};

//////////////////////////////////////////////////////////////////////////
// File structure

Header header @ 0x00;

if(std::string::substr(header.versionIdentifier, 0, 16) != "\xAB\x42\x41RY 00100\xBB\x0D\x0A\x1A\x0A")
{
    std::error(std::format("This isn't a .bary file, or it had an unknown version number!
(All released.bary files should have version number 00100, but this one had version number{}.)",
                           std::string::substr(header.versionIdentifier, 6, 5)));
}

PropertyInfo g_forPropertyInfoSize;
PropertyInfo properties[header.propertyInfoRange.byteLength/sizeof(g_forPropertyInfoSize)] @ header.propertyInfoRange.byteOffset;
