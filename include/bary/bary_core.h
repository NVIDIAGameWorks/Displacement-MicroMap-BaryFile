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

#include "bary_types.h"
#include "bary_api.h"

namespace bary
{
//////////////////////////////////////////////

BARY_API VersionIdentifier BARY_CALL baryGetCurrentVersionIdentifier();

BARY_API Result baryVersionIdentifierGetVersion(const VersionIdentifier* identifier, uint32_t* pVersion);

//////////////////////////////////////////////

// get min/max across all groups, pointers are optional
BARY_API void BARY_CALL baryBasicViewGetMinMaxSubdivLevels(const BasicView* basic, uint32_t* min, uint32_t* max);

BARY_API uint32_t BARY_CALL baryContentComputePropertyCount(const ContentView* content);

// only tests non-generic semantic related validation, not full validation
BARY_API Result BARY_CALL baryContentIsValid(ValueSemanticType valueSemantic, const ContentView* content, StandardPropertyType* errorPropertyType);

BARY_API Result BARY_CALL baryContentSetupProperties(const ContentView* content, uint32_t propertyCount, PropertyStorageInfo* propertyStorageInfos);


// content structs point to section of fileData
// mesh and misc are optional
// only valid for non-supercompressed data
BARY_API Result BARY_CALL baryDataGetContent(uint64_t fileSize, const void* fileData, ContentView* content, StandardPropertyType* errorPropertyType);

//////////////////////////////////////////////

BARY_API const char* BARY_CALL baryResultGetName(Result result);

//////////////////////////////////////////////

BARY_API const char* BARY_CALL baryFormatGetName(Format format);

//////////////////////////////////////////////

BARY_API PropertyIdentifier BARY_CALL baryStandardPropertyGetIdentifier(StandardPropertyType type);

BARY_API const char* BARY_CALL baryStandardPropertyGetName(StandardPropertyType type);

// returns zero if unknown
BARY_API uint32_t BARY_CALL baryStandardPropertyGetElementSize(StandardPropertyType type);

// returns zero if unknown or no info
BARY_API uint32_t BARY_CALL baryStandardPropertyGetInfoSize(StandardPropertyType type);

//////////////////////////////////////////////

inline bool baryPropertyIsEqual(PropertyIdentifier a, PropertyIdentifier b)
{
    return a.uuid4[0] == b.uuid4[0] && a.uuid4[1] == b.uuid4[1] && a.uuid4[2] == b.uuid4[2] && a.uuid4[3] == b.uuid4[3];
}

BARY_API StandardPropertyType BARY_CALL baryPropertyGetStandardType(PropertyIdentifier identifier);

// will also return 0 if not a standard property
// derives group/triangle count etc. from standard properties
inline uint64_t baryPropertyGetStandardElementCount(PropertyIdentifier identifier, ByteRange range)
{
    uint64_t elementSize = baryStandardPropertyGetElementSize(baryPropertyGetStandardType(identifier));
    return elementSize ? range.byteLength / elementSize : 0;
}


//////////////////////////////////////////////////////////////////////////

// return padding size for a property that leads with an info struct
// and is followed by payload with a certain alignment requirement
inline uint64_t baryPayloadGetPadding(uint64_t infoStructSize, uint32_t payloadByteAlignment)
{
    if(payloadByteAlignment <= 1)
        return 0;
    uint64_t rest = infoStructSize % payloadByteAlignment;
    return rest ? uint64_t(payloadByteAlignment) - rest : 0;
}

// return (info size + padding size) for a property that leads with an info struct
// and is followed by payload with a certain alignment requirement
inline uint64_t baryPayloadGetOffset(uint64_t infoStructSize, uint32_t payloadByteAlignment)
{
    if(payloadByteAlignment <= 1)
        return infoStructSize;
    uint64_t rest = infoStructSize % payloadByteAlignment;
    return infoStructSize + (rest ? uint64_t(payloadByteAlignment) - rest : 0);
}

// if a property leads with an info struct and then comes payload
// return the payload start pointer location.
// can account for special padding between info and payload.
inline const void* baryPayloadGetPointer(uint64_t infoStructSize, uint32_t payloadByteAlignment, const void* ptr)
{
    uint64_t payloadOffset = baryPayloadGetOffset(infoStructSize, payloadByteAlignment);

    return reinterpret_cast<const uint8_t*>(ptr) + payloadOffset;
}

// if a property leads with an info struct and then comes payload
// return the payload start pointer location.
// can account for special padding between info and payload.
inline void* baryPayloadGetPointer(uint64_t infoStructSize, uint32_t payloadByteAlignment, void* ptr)
{
    uint64_t payloadOffset = baryPayloadGetOffset(infoStructSize, payloadByteAlignment);

    return reinterpret_cast<uint8_t*>(ptr) + payloadOffset;
}

//////////////////////////////////////////////

inline bool baryPropertyStorageHasValidSize(const PropertyStorageInfo& prop, uint64_t expectedInfoSize)
{
    if(!prop.infoSize)
    {
        return prop.dataSize >= (expectedInfoSize);
    }
    else
    {
        return (prop.infoSize == expectedInfoSize);
    }
}

inline bool baryPropertyStorageHasValidPadding(const PropertyStorageInfo& prop, uint64_t expectedInfoSize, uint64_t paddingAlignment)
{
    uint64_t padding = baryPayloadGetPadding(expectedInfoSize, uint32_t(paddingAlignment));

    if(!prop.infoSize)
    {
        return prop.dataSize >= (expectedInfoSize + padding);
    }
    else
    {
        return (prop.infoSize == expectedInfoSize && prop.infoPaddingSize == padding);
    }
}

//////////////////////////////////////////////

inline bool baryUVisValid(BaryUV_uint16 coord, uint32_t subdiv)
{
    uint32_t max = 1u << subdiv;
    return (coord.u <= max && coord.v <= max && uint32_t(coord.u + coord.v) <= max);
}


BARY_API const char* BARY_CALL baryValueLayoutGetName(ValueLayout layout);

// returns ~0 if invalid combination
// isUpperTriangle means the triangle in a quad with lower left corner at u,v (see below)
//
// x___x
// |0\1|  number inside triangle reflects isUpperTriangle value for per_triangle
// uv__x
//
BARY_API uint32_t BARY_CALL
baryValueLayoutGetIndex(ValueLayout order, ValueFrequency frequency, uint32_t u, uint32_t v, uint32_t isUpperTriangle, uint32_t subdivLevel);

// generates a UV mesh for the provided ValueOrder
// triangles and vertices ordered accordingly
// pUVs == 2 x uint16_t x uvCount
// pTriangleIndices == 3 x uint32_t x triangleCount
// the use of edgeDecimateFlag can cause degenerate
// triangles and unreferenced vertices
BARY_API Result BARY_CALL baryValueLayoutGetUVMesh(ValueLayout    order,
                                                   uint32_t       subdivLevel,
                                                   uint32_t       edgeDecimateFlag,
                                                   uint32_t       uvCount,
                                                   BaryUV_uint16* pUVs,
                                                   uint32_t       triangleCount,
                                                   uint32_t*      pTriangleIndices);

// for a micro-vertex stored in bird-order retrieve details about its subdivLevel and index within the level
BARY_API void BARY_CALL baryBirdLayoutGetVertexLevelInfo(uint32_t u, uint32_t v, uint32_t subdivLevel, uint32_t* outLevel, uint32_t* outLevelCoordIndex);


BARY_API const char* BARY_CALL baryValueFrequencyGetName(ValueFrequency frequency);

// Computes how many values are stored for a certain frequency and subdivLevel.
// If the ValueFrequency is unknown or if integer overflow would occur, returns 0.
inline uint32_t baryValueFrequencyGetCount(ValueFrequency frequency, uint32_t subdivLevel)
{
    switch(frequency)
    {
    case ValueFrequency::ePerVertex: {
        if(subdivLevel > 15)
            return 0;
        uint32_t numVertsPerEdge = (1u << subdivLevel) + 1;
        return (numVertsPerEdge * (numVertsPerEdge + 1)) / 2;
    }
    case ValueFrequency::ePerTriangle:
        if(subdivLevel > 15)
            return 0;
        return 1 << (subdivLevel * 2);
    default:
        return 0;
    }
}


//////////////////////////////////////////////////////////////////////////
// compression related

inline BaryUV_uint16 baryBlockTriangleLocalToBaseUV(const BlockTriangle* info, BaryUV_uint16 locaUV)
{
    int32_t anchor[2] = {info->vertices[0].u, info->vertices[0].v};
    int32_t signs[2] = {info->vertices[1].u > info->vertices[0].u ? 1 : -1, info->vertices[2].v > info->vertices[0].v ? 1 : -1};
    int32_t local[2] = {locaUV.u, locaUV.v};
    local[0] *= signs[0];
    local[1] *= signs[1];

    BaryUV_uint16 baseUV;
    baseUV.u = uint16_t(anchor[0] + local[0] + (signs[0] != signs[1] ? -local[1] : 0));
    baseUV.v = uint16_t(anchor[1] + local[1]);
    return baseUV;
}

// may return invalid / out of bounds coords, use baryUVisValid
inline BaryUV_uint16 baryBlockTriangleBaseToLocalUV(const BlockTriangle* info, BaryUV_uint16 baseUV)
{
    int32_t base[2]   = {baseUV.u, baseUV.v};
    int32_t anchor[2] = {info->vertices[0].u, info->vertices[0].v};
    int32_t signs[2] = {info->vertices[1].u > info->vertices[0].u ? 1 : -1, info->vertices[2].v > info->vertices[0].v ? 1 : -1};
    int32_t local[2] = {};
    local[1]         = base[1] - anchor[1];
    local[0]         = base[0] - anchor[0] - (signs[0] != signs[1] ? -local[1] : 0);

    local[0] *= signs[0];
    local[1] *= signs[1];

    BaryUV_uint16 locaUV;
    locaUV.u = uint16_t(local[0]);
    locaUV.v = uint16_t(local[1]);
    return locaUV;
}

inline uint32_t baryBlockTriangleBaseToLocalFlags(const BlockTriangle* subSplit, uint32_t baseFlags)
{
    uint32_t baseTopo        = 0;
    uint32_t baseEdgeIndices = subSplit->baseEdgeIndices;
    // For each edge of the subprimitive...
    for(uint32_t e = 0; e < 3; e++)
    {
        // Look at the subprim info to determine which base triangle edge this
        // refers to. 3 indicates no base triangle edge.
        const uint32_t baseEdge = (baseEdgeIndices >> (e * 2)) & 3;
        if(baseEdge != 3)
        {
            const uint32_t baseEdgeFlag = (baseFlags >> baseEdge) & 1;
            baseTopo |= baseEdgeFlag << e;
        }
    }
    return baseTopo;
}

BARY_API uint32_t BARY_CALL baryBlockFormatDispC1GetMaxSubdivLevel();

// returns ~0 if invalid
BARY_API uint32_t BARY_CALL baryBlockFormatDispC1GetSubdivLevel(BlockFormatDispC1 blockFormat);
// returns ~0 if invalid
BARY_API uint32_t BARY_CALL baryBlockFormatDispC1GetByteSize(BlockFormatDispC1 blockFormat);

// returns 0 if invalid
BARY_API uint32_t BARY_CALL baryBlockFormatDispC1GetBlockCount(BlockFormatDispC1 blockFormat, uint32_t baseSubdivLevel);

BARY_API Result BARY_CALL baryBlockFormatDispC1GetBlockTriangles(BlockFormatDispC1 blockFormat,
                                                                 uint32_t          baseSubdivLevel,
                                                                 uint32_t          blockTrisCount,
                                                                 BlockTriangle*    blockTris);

BARY_API void BARY_CALL baryBlockTriangleSplitDispC1(const BlockTriangle* in, BlockTriangle* out, uint32_t outStride);


BARY_API uint32_t BARY_CALL baryHistogramGetBlockCount(uint32_t entriesCount, const bary::HistogramEntry* entries, bary::Format fmt);
BARY_API uint32_t BARY_CALL baryMeshHistogramGetBlockCount(uint32_t entriesCount, const bary::MeshHistogramEntry* entries, bary::Format fmt);

//////////////////////////////////////////////////////////////////////////
// validation

enum ValidationFlagBit : uint64_t
{
    // If set, then deeper validation is performed on content arrays
    eValidationFlagArrayContents = 1ull,

    // If set, then accurate testing of Triangle::valuesByteOffset will be done.
    // Might want to skip if the compression blockformat is not known to the standard validation,
    // or if the compression blocks are used in a more complex way than simple uniform
    // splitting.
    eValidationFlagTriangleValueRange = 2ull,
};

// all properties that constitute the final file should be provided
// only standard properties can be validated here
// errorPropertyType that caused an issue can be returned in pointer
BARY_API Result BARY_CALL baryValidateStandardProperties(uint32_t                   propertyCount,
                                                         const PropertyStorageInfo* propertyStorageInfos,
                                                         uint64_t                   validationFlags,
                                                         StandardPropertyType*      errorPropertyType);


//////////////////////////////////////////////////////////////////////////
// storage
//
// The typical ordering of operations would be:
//  1. `baryValidateStandardProperties(...);` - heavy full validation on standard property types
//  2. `fileSize = baryStorageComputeSize(...);`
//  3. `baryStorageOutputAll(..., fileSize, filePointer);`
// or
//  3. `baryStorageOutputHeaderAndPropertyInfos(..., fileSize, filePointer)`
//  4. iterate over properties:
//     `range = baryDataGetPropertyByteRange(fileSize, filePointer, property identifier);`
//     `dst   = baryDataGetByteRangeDataT(fileSize, filePointer, *range);`
//     write property data to dst

// computes size of header and prop infos
BARY_API uint64_t BARY_CALL baryStorageComputePreambleSize(uint32_t propertyCount);

// computes total outputSize required to store the provided properties along with all information
// that makes a bary file (header + PropertyInfo(s))
BARY_API uint64_t BARY_CALL baryStorageComputeSize(uint32_t propertyCount, const PropertyStorageInfo* propertyStorageInfos);


// all in one serialization to target outputData
// outputSize must match baryStorageComputeSize
BARY_API Result BARY_CALL baryStorageOutputAll(uint32_t                   propertyCount,
                                               const PropertyStorageInfo* propertyStorageInfos,
                                               uint64_t                   outputSize,
                                               void*                      outputData);

// splits serialization, leaves writing per-property data into appropriate byte range inside outputData
// to user. Use baryDataGetPropertyByteRange to get the range into outputData after this function was run
// and then write the property data there.
//
// outputSize is expected to match baryStorageComputeSize
// outputPreambleSize must match baryStorageComputePreambleSize
BARY_API Result BARY_CALL baryStorageOutputPreamble(uint32_t                   propertyCount,
                                                    const PropertyStorageInfo* propertyStorageInfos,
                                                    uint64_t                   outputSize,
                                                    uint64_t                   outputPreambleSize,
                                                    void*                      outputPreambleData);

// Callback to a function that saves `size` bytes, contained in `data`,
// to the bytes starting at `offset` in the output (e.g. a file or in-memory buffer).
// `offset` will be called in strictly ascending order without gaps, so
// an implementation using `fwrite()` wouldn't need to call `setg()`.
// `propertyStorageInfo != nullptr` if and only if a property is being written.
typedef Result (*PFN_outputSaver)(uint32_t                   propertyIdx,
                                  const PropertyStorageInfo* propertyStorageInfo,
                                  uint64_t                   offset,
                                  uint64_t                   size,
                                  const void*                data,
                                  bool                       isInfo,
                                  void*                      userData);

// output via saver callback
BARY_API Result BARY_CALL baryStorageOutputSaver(uint32_t                   propertyCount,
                                                 const PropertyStorageInfo* propertyStorageInfos,
                                                 uint64_t                   preambleSize,
                                                 const void*                preamble,
                                                 PFN_outputSaver            fnSaver,
                                                 void*                      userData);

//////////////////////////////////////////////////////////////////////////
// retrieval
//
// typical order of operations:
//   1. `baryDataIsValid(fileSize, filePointer);` - checks the header and if all property infos are valid
//  (2. `baryDataHasMandatoryStandardProperties(fileSize, filePointer);` - checks if the mandatory standard properties exist in file with current uuids
//   3. iterate properties and access their data
//      use `baryDataGetByteRangeDataT` and `baryDataGetByteRangeDataT`
//      or  `baryDataGetPropertyData` or `baryDataGetPropertyDataT`
//      to make the appropriate property identifier:
//          for StandardPropertyTypes use `baryMakeStandardPropertyIdentifierT<StandardPropertyType enum>()`
//          anything else needs extra header/implementation of `baryMakePropertyIdentifierT<class T>()` or
//          other means of storage

BARY_API Result BARY_CALL baryDataIsValid(uint64_t fileSize, const void* fileData);

BARY_API Result BARY_CALL baryDataGetVersion(uint64_t fileSize, const void* fileData, uint32_t* pVersion);

inline bool baryDataIsRangeValid(uint64_t fileSize, ByteRange range)
{
    // in practice all byteOffsets must be >= aligned(sizeof(Header), 4)
    // however we don't test this given the header size could change as well
    return (range.byteOffset < fileSize) && (range.byteLength <= fileSize - range.byteOffset);
}

BARY_API Result BARY_CALL baryDataHasMandatoryStandardProperties(uint64_t fileSize, const void* fileData);

BARY_API bool BARY_CALL baryDataHasAnySuperCompression(uint64_t fileSize, const void* fileData);

// can return nullptr if identifier not found in file
// returned pointer is within provided fileData memory
BARY_API const PropertyInfo* BARY_CALL baryDataGetPropertyInfo(uint64_t fileSize, const void* fileData, PropertyIdentifier identifier);

// can return nullptr if identifier not found in file
// returned pointer is within provided fileData memory
BARY_API const void* BARY_CALL baryDataGetPropertyData(uint64_t fileSize, const void* fileData, PropertyIdentifier identifier, uint64_t* pLength);

// can return nullptr if no properties
BARY_API const PropertyInfo* BARY_CALL baryDataGetAllPropertyInfos(uint64_t fileSize, const void* fileData, uint64_t* count);

// returned pointer is within provided fileData memory
template <class T>
const T* baryDataGetByteRangeDataT(uint64_t fileSize, const void* fileData, ByteRange range)
{
    if(baryDataIsRangeValid(fileSize, range))
    {
        return reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(fileData) + range.byteOffset);
    }

    return nullptr;
}

template <class T>
T* baryDataGetByteRangeDataT(uint64_t fileSize, void* fileData, ByteRange range)
{
    if(baryDataIsRangeValid(fileSize, range))
    {
        return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(fileData) + range.byteOffset);
    }

    return nullptr;
}

// can return nullptr if identifier not found in file
// returned pointer is within provided fileData memory
template <class T>
inline const T* baryDataGetPropertyDataT(uint64_t fileSize, const void* fileData, PropertyIdentifier identifier, uint64_t* pLength)
{
    uint64_t length = 0;
    const T* result = reinterpret_cast<const T*>(baryDataGetPropertyData(fileSize, fileData, identifier, &length));
    // Always set pLength; length will be 0 if result is nullptr
    if(pLength)
        *pLength = length;
    // Make sure the property data really is long enough to contain a T
    if(length < sizeof(T))
        return nullptr;
    return result;
}


}  // namespace bary