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

#include <cassert>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cmath>

#include <baryutils/baryutils.h>

namespace baryutils
{
//////////////////////////////////////////////////////////////////////////

uint32_t BaryStats::getHistoBin(uint32_t num)
{
    uint32_t usedbits = uint32_t(std::log(double(num)) / std::log(2.0)) + 1;
    uint32_t bin      = num > (1u << (usedbits - 1)) ? usedbits + 1 : usedbits;
    assert((bin - 1) < MAX_HISTO_BINS);
    return bin - 1;
}

bool BaryStats::append(const bary::BasicView& basic)
{
    bool inconsistent = false;

    uint32_t minLevel;
    uint32_t maxLevel;
    bary::baryBasicViewGetMinMaxSubdivLevels(&basic, &minLevel, &maxLevel);

    minSubdivLevel = std::min(minSubdivLevel, minLevel);
    maxSubdivLevel = std::max(maxSubdivLevel, maxLevel);
    dataByteSize += size_t(basic.valuesInfo->valueCount) * size_t(basic.valuesInfo->valueByteSize);
    if(valueOrder == bary::ValueLayout::eUndefined)
    {
        valueOrder = basic.valuesInfo->valueLayout;
    }
    if(valueOrder != basic.valuesInfo->valueLayout)
    {
        inconsistent = true;
    }

    if(valueFormat == bary::Format::eUndefined)
    {
        valueFormat = basic.valuesInfo->valueFormat;
    }
    if(valueFormat != basic.valuesInfo->valueFormat)
    {
        inconsistent = true;
    }

    bool isCompressedDisplacement = basic.valuesInfo->valueFormat == bary::Format::eDispC1_r11_unorm_block;

    mapTriangles += basic.trianglesCount;
    for(uint32_t i = 0; i < basic.trianglesCount; i++)
    {
        const bary::Triangle& tri = basic.triangles[i];
        microTriangles += bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerTriangle, tri.subdivLevel);
        microVertices += bary::baryValueFrequencyGetCount(bary::ValueFrequency::ePerVertex, tri.subdivLevel);
        if(isCompressedDisplacement)
        {
            uint32_t numBlocks = bary::baryBlockFormatDispC1GetBlockCount(tri.blockFormatDispC1, tri.subdivLevel);
            blocks += numBlocks;
            blocksPerFormat[tri.blockFormat] += numBlocks;
            blocksPerTriangleHisto[getHistoBin(numBlocks)] += 1;
        }
    }
    return inconsistent;
}

//////////////////////////////////////////////////////////////////////////
template <typename Tinfo, typename Tdata>
static void setElementProp(std::vector<Tdata>& vec, Tinfo& info, const Tdata* viewData, const Tinfo* viewInfo)
{
    if(viewInfo && viewData)
    {
        info = *viewInfo;
        vec.resize(viewInfo->elementCount * viewInfo->elementByteSize);
        memcpy(vec.data(), viewData, vec.size());
    }
}

template <typename Tdata>
static void setCountProp(std::vector<Tdata>& vec, const Tdata* viewData, size_t viewCount)
{
    if(viewCount && viewData)
    {
        vec.resize(viewCount);
        memcpy(vec.data(), viewData, sizeof(Tdata) * viewCount);
    }
}

void BaryMeshData::setData(const bary::MeshView& view)
{
    setCountProp(meshGroups, view.meshGroups, view.meshGroupsCount);
    setCountProp(meshHistogramEntries, view.meshHistogramEntries, view.meshHistogramEntriesCount);
    setCountProp(meshGroupHistogramRanges, view.meshGroupHistogramRanges, view.meshGroupHistogramRangesCount);

    setElementProp(meshDisplacementDirectionBounds, meshDisplacementDirectionBoundsInfo,
                   view.meshDisplacementDirectionBounds, view.meshDisplacementDirectionBoundsInfo);
    setElementProp(meshDisplacementDirections, meshDisplacementDirectionsInfo, view.meshDisplacementDirections,
                   view.meshDisplacementDirectionsInfo);
    setElementProp(meshPositions, meshPositionsInfo, view.meshPositions, view.meshPositionsInfo);

    setElementProp(meshTriangleIndices, meshTriangleIndicesInfo, view.meshTriangleIndices, view.meshTriangleIndicesInfo);
    setElementProp(meshTriangleMappings, meshTriangleMappingsInfo, view.meshTriangleMappings, view.meshTriangleMappingsInfo);
    setElementProp(meshTriangleFlags, meshTriangleFlagsInfo, view.meshTriangleFlags, view.meshTriangleFlagsInfo);
}

bary::MeshView BaryMeshData::getView() const
{
    bary::MeshView view;

    view.meshGroups      = meshGroups.data();
    view.meshGroupsCount = uint32_t(meshGroups.size());

    view.meshHistogramEntries      = meshHistogramEntries.data();
    view.meshHistogramEntriesCount = uint32_t(meshHistogramEntries.size());

    view.meshGroupHistogramRanges      = meshGroupHistogramRanges.data();
    view.meshGroupHistogramRangesCount = uint32_t(meshGroupHistogramRanges.size());

    view.meshDisplacementDirectionBoundsInfo = &meshDisplacementDirectionBoundsInfo;
    view.meshDisplacementDirectionBounds     = meshDisplacementDirectionBounds.data();

    view.meshDisplacementDirectionsInfo = &meshDisplacementDirectionsInfo;
    view.meshDisplacementDirections     = meshDisplacementDirections.data();

    view.meshPositionsInfo = &meshPositionsInfo;
    view.meshPositions     = meshPositions.data();

    view.meshTriangleIndicesInfo = &meshTriangleIndicesInfo;
    view.meshTriangleIndices     = meshTriangleIndices.data();

    view.meshTriangleMappingsInfo = &meshTriangleMappingsInfo;
    view.meshTriangleMappings     = meshTriangleMappings.data();

    view.meshTriangleFlagsInfo = &meshTriangleFlagsInfo;
    view.meshTriangleFlags     = meshTriangleFlags.data();

    return view;
}

//////////////////////////////////////////////////////////////////////////

void BaryMiscData::setData(const bary::MiscView& view)
{
    setCountProp(groupUncompressedMips, view.groupUncompressedMips, view.groupUncompressedMipsCount);
    setCountProp(triangleUncompressedMips, view.triangleUncompressedMips, view.triangleUncompressedMipsCount);
    setElementProp(uncompressedMips, uncompressedMipsInfo, view.uncompressedMips, view.uncompressedMipsInfo);
}

bary::MiscView BaryMiscData::getView() const
{
    bary::MiscView view;

    view.groupUncompressedMips      = groupUncompressedMips.data();
    view.groupUncompressedMipsCount = uint32_t(groupUncompressedMips.size());

    view.triangleUncompressedMips      = triangleUncompressedMips.data();
    view.triangleUncompressedMipsCount = uint32_t(triangleUncompressedMips.size());

    view.uncompressedMipsInfo = &uncompressedMipsInfo;
    view.uncompressedMips     = uncompressedMips.data();

    return view;
}

//////////////////////////////////////////////////////////////////////////

void BaryBasicData::setData(const bary::BasicView& view)
{
    setCountProp(groups, view.groups, view.groupsCount);
    if(view.valuesInfo && view.values)
    {
        valuesInfo = *view.valuesInfo;
        values.resize(view.valuesInfo->valueCount * view.valuesInfo->valueByteSize);
        memcpy(values.data(), view.values, values.size());
    }
    setCountProp(triangles, view.triangles, view.trianglesCount);

    setElementProp(triangleMinMaxs, triangleMinMaxsInfo, view.triangleMinMaxs, view.triangleMinMaxsInfo);

    setCountProp(histogramEntries, view.histogramEntries, view.histogramEntriesCount);
    setCountProp(groupHistogramRanges, view.groupHistogramRanges, view.groupHistogramRangesCount);

    updateMinMaxSubdivLevels();
}

bary::BasicView BaryBasicData::getView() const
{
    bary::BasicView view;

    view.groupsCount    = uint32_t(groups.size());
    view.groups         = groups.data();
    view.valuesInfo     = &valuesInfo;
    view.values         = values.data();
    view.trianglesCount = uint32_t(triangles.size());
    view.triangles      = triangles.data();

    view.triangleMinMaxsInfo = &triangleMinMaxsInfo;
    view.triangleMinMaxs     = triangleMinMaxs.data();

    view.histogramEntriesCount     = uint32_t(histogramEntries.size());
    view.histogramEntries          = histogramEntries.data();
    view.groupHistogramRangesCount = uint32_t(groupHistogramRanges.size());
    view.groupHistogramRanges      = groupHistogramRanges.data();

    return view;
}

void BaryBasicData::updateMinMaxSubdivLevels()
{
    minSubdivLevel = ~0;
    maxSubdivLevel = 0;
    for(const auto& grp : groups)
    {
        maxSubdivLevel = std::max(maxSubdivLevel, grp.maxSubdivLevel);
        minSubdivLevel = std::min(minSubdivLevel, grp.minSubdivLevel);
    }
}

bary::Result BaryBasicData::save(const char* filename, const bary::MeshView* mesh, const bary::MiscView* misc, bary::StandardPropertyType* pErrorProp) const
{
    BarySaver       saver;
    bary::BasicView basic  = getView();
    bary::Result    result = saver.initContent(&basic, mesh, misc, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    return saver.save(filename);
}

bary::Result BaryBasicData::load(size_t fileSize, const void* fileData, bary::ValueSemanticType vtype, bary::StandardPropertyType* pErrorProp)
{
    BaryFile     bfile;
    bary::Result result = bfile.open(fileSize, fileData, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    result = bfile.validate(vtype, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    bfile.fillBasicData(*this);
    return result;
}

bary::Result BaryBasicData::load(const char*                 filename,
                                 bary::ValueSemanticType     vtype,
                                 const BaryFileOpenOptions*  fileOptions,
                                 bary::StandardPropertyType* pErrorProp)
{
    BaryFile     bfile;
    bary::Result result = bfile.open(filename, fileOptions, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    result = bfile.validate(vtype, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    bfile.fillBasicData(*this);
    return result;
}

//////////////////////////////////////////////////////////////////////////

bary::Result BaryContentData::save(const char* filename, bary::StandardPropertyType* pErrorProp) const
{
    BarySaver         saver;
    bary::ContentView view   = getView();
    bary::Result      result = saver.initContent(&view, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    return saver.save(filename);
}

bary::Result BaryContentData::load(size_t fileSize, const void* fileData, bary::ValueSemanticType vtype, bary::StandardPropertyType* pErrorProp)
{
    BaryFile     bfile;
    bary::Result result = bfile.open(fileSize, fileData, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    result = bfile.validate(vtype, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    bfile.fillContentData(*this);
    return result;
}

bary::Result BaryContentData::load(const char*                 filename,
                                   bary::ValueSemanticType     vtype,
                                   const BaryFileOpenOptions*  fileOptions,
                                   bary::StandardPropertyType* pErrorProp)
{
    BaryFile     bfile;
    bary::Result result = bfile.open(filename, fileOptions, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    result = bfile.validate(vtype, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }
    bfile.fillContentData(*this);
    return result;
}

//////////////////////////////////////////////////////////////////////////

bary::Result BarySaver::initContent(const bary::BasicView*      basic,
                                    const bary::MeshView*       mesh /*= nullptr*/,
                                    const bary::MiscView*       misc /*= nullptr*/,
                                    bary::StandardPropertyType* pErrorProp)
{
    *this = BarySaver();

    bary::ContentView content;
    content.basic = basic ? *basic : bary::BasicView();
    content.mesh  = mesh ? *mesh : bary::MeshView();
    content.misc  = misc ? *misc : bary::MiscView();

    return initContent(&content, pErrorProp);
}

static bary::Result returnSetError(bary::Result result, bary::StandardPropertyType stdProp, bary::StandardPropertyType* pErrorProp)
{
    if(pErrorProp)
    {
        *pErrorProp = stdProp;
    }

    return result;
}

bary::Result BarySaver::fillPropertyStorageInfos(std::vector<bary::PropertyStorageInfo>& props,
                                                 const bary::ContentView*                content,
                                                 bary::StandardPropertyType*             pErrorProp)
{
    uint32_t propCount = bary::baryContentComputePropertyCount(content);
    props.resize(propCount);

    // preparation
    bary::Result result = bary::baryContentSetupProperties(content, propCount, props.data());
    if(result != bary::Result::eSuccess)
    {
        return returnSetError(result, bary::StandardPropertyType::eUnknown, pErrorProp);
    }

#ifdef _DEBUG
    uint64_t validationFlags = bary::ValidationFlagBit::eValidationFlagArrayContents
                               | bary::ValidationFlagBit::eValidationFlagTriangleValueRange;
#else
    // by default skip deep evaluation
    uint64_t validationFlags = 0;
#endif

    return bary::baryValidateStandardProperties(propCount, props.data(), validationFlags, pErrorProp);
}

bary::Result BarySaver::initContent(const bary::ContentView* content, bary::StandardPropertyType* pErrorProp)
{
    *this = BarySaver();

    m_result = fillPropertyStorageInfos(m_props, content, pErrorProp);

    if(m_result != bary::Result::eSuccess)
    {
        return m_result;
    }

    if(content->basic.valuesInfo)
        m_aggregatedInfo.basic.valuesInfo = *content->basic.valuesInfo;
    if(content->basic.triangleMinMaxsInfo)
        m_aggregatedInfo.basic.triangleMinMaxsInfo = *content->basic.triangleMinMaxsInfo;
    if(content->mesh.meshDisplacementDirectionBoundsInfo)
        m_aggregatedInfo.mesh.meshDisplacementDirectionBoundsInfo = *content->mesh.meshDisplacementDirectionBoundsInfo;
    if(content->mesh.meshDisplacementDirectionsInfo)
        m_aggregatedInfo.mesh.meshDisplacementDirectionsInfo = *content->mesh.meshDisplacementDirectionsInfo;
    if(content->mesh.meshPositionsInfo)
        m_aggregatedInfo.mesh.meshPositionsInfo = *content->mesh.meshPositionsInfo;
    if(content->mesh.meshTriangleFlagsInfo)
        m_aggregatedInfo.mesh.meshTriangleFlagsInfo = *content->mesh.meshTriangleFlagsInfo;
    if(content->mesh.meshTriangleIndicesInfo)
        m_aggregatedInfo.mesh.meshTriangleIndicesInfo = *content->mesh.meshTriangleIndicesInfo;
    if(content->mesh.meshTriangleMappingsInfo)
        m_aggregatedInfo.mesh.meshTriangleMappingsInfo = *content->mesh.meshTriangleMappingsInfo;
    if(content->misc.uncompressedMipsInfo)
        m_aggregatedInfo.misc.uncompressedMipsInfo = *content->misc.uncompressedMipsInfo;

    return m_result;
}

bary::Result BarySaver::appendContent(const bary::BasicView*      basic,
                                      const bary::MeshView*       mesh,
                                      const bary::MiscView*       misc,
                                      bary::StandardPropertyType* pErrorProp)
{
    bary::ContentView content;
    content.basic = basic ? *basic : bary::BasicView();
    content.mesh  = mesh ? *mesh : bary::MeshView();
    content.misc  = misc ? *misc : bary::MiscView();

    return appendContent(&content, pErrorProp);
}

template <typename T>
static bool appendElementInfo(T& refInfo, const T* inInfo)
{
    if(!inInfo)
        return false;

    T aData = *inInfo;
    T bData = refInfo;

    aData.elementCount = 0;
    bData.elementCount = 0;

    if(memcmp(&aData, &bData, sizeof(T)) != 0)
    {
        return true;
    }

    refInfo.elementCount += inInfo->elementCount;

    return false;
}

static bool appendValuesInfo(bary::ValuesInfo& refInfo, const bary::ValuesInfo* inInfo)
{
    if(!inInfo)
        return false;

    bary::ValuesInfo aData = *inInfo;
    bary::ValuesInfo bData = refInfo;

    aData.valueCount = 0;
    bData.valueCount = 0;

    if(memcmp(&aData, &bData, sizeof(bary::ValuesInfo)) != 0)
    {
        return true;
    }

    refInfo.valueCount += inInfo->valueCount;

    return false;
}

bary::Result BarySaver::appendContent(const bary::ContentView* content, bary::StandardPropertyType* pErrorProp)
{
    if(m_props.empty())
    {
        assert(0 && "must call initContent first");
        return returnSetError(bary::Result::eErrorMissingProperty, bary::StandardPropertyType::eUnknown, pErrorProp);
    }
    if(m_propsList.empty())
    {
        // first after init must push back props from init.
        m_propsList.push_back(m_props);
    }

    std::vector<bary::PropertyStorageInfo> props;
    bary::Result                           result = fillPropertyStorageInfos(props, content, pErrorProp);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }

    if(props.size() != m_props.size())
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch, bary::StandardPropertyType::eUnknown, pErrorProp);
    }

    // check compatibility of infos
    if(appendValuesInfo(m_aggregatedInfo.basic.valuesInfo, content->basic.valuesInfo))
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch, bary::StandardPropertyType::eValues, pErrorProp);
    }
    if(appendElementInfo(m_aggregatedInfo.basic.triangleMinMaxsInfo, content->basic.triangleMinMaxsInfo))
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch, bary::StandardPropertyType::eTriangleMinMaxs, pErrorProp);
    }
    if(appendElementInfo(m_aggregatedInfo.mesh.meshDisplacementDirectionBoundsInfo, content->mesh.meshDisplacementDirectionBoundsInfo))
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch,
                              bary::StandardPropertyType::eMeshDisplacementDirectionBounds, pErrorProp);
    }
    if(appendElementInfo(m_aggregatedInfo.mesh.meshDisplacementDirectionsInfo, content->mesh.meshDisplacementDirectionsInfo))
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch, bary::StandardPropertyType::eMeshDisplacementDirections, pErrorProp);
    }
    if(appendElementInfo(m_aggregatedInfo.mesh.meshPositionsInfo, content->mesh.meshPositionsInfo))
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch, bary::StandardPropertyType::eMeshPositions, pErrorProp);
    }
    if(appendElementInfo(m_aggregatedInfo.mesh.meshTriangleFlagsInfo, content->mesh.meshTriangleFlagsInfo))
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch, bary::StandardPropertyType::eMeshTriangleFlags, pErrorProp);
    }
    if(appendElementInfo(m_aggregatedInfo.mesh.meshTriangleIndicesInfo, content->mesh.meshTriangleIndicesInfo))
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch, bary::StandardPropertyType::eMeshTriangleIndices, pErrorProp);
    }
    if(appendElementInfo(m_aggregatedInfo.mesh.meshTriangleMappingsInfo, content->mesh.meshTriangleMappingsInfo))
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch, bary::StandardPropertyType::eMeshTriangleMappings, pErrorProp);
    }
    if(appendElementInfo(m_aggregatedInfo.misc.uncompressedMipsInfo, content->misc.uncompressedMipsInfo))
    {
        return returnSetError(bary::Result::eErrorPropertyMismatch, bary::StandardPropertyType::eUncompressedMips, pErrorProp);
    }

    uint32_t propertyCount = uint32_t(m_props.size());

    // append m_props sizes
    for(uint32_t i = 0; i < propertyCount; i++)
    {
        bary::StandardPropertyType stdProp = bary::baryPropertyGetStandardType(props[i].identifier);

        if(!bary::baryPropertyIsEqual(m_props[i].identifier, props[i].identifier))
        {
            return returnSetError(bary::Result::eErrorPropertyMismatch, stdProp, pErrorProp);
        }

        // always append size of payload
        m_props[i].dataSize += props[i].dataSize;

        if(m_propsList.size() == 1)
        {
            // reassign info pointers to the local ones that got modified

            switch(stdProp)
            {
            case bary::StandardPropertyType::eValues:
                m_props[i].info = &m_aggregatedInfo.basic.valuesInfo;
                break;
            case bary::StandardPropertyType::eTriangleMinMaxs:
                m_props[i].info = &m_aggregatedInfo.basic.triangleMinMaxsInfo;
                break;
            case bary::StandardPropertyType::eMeshDisplacementDirectionBounds:
                m_props[i].info = &m_aggregatedInfo.mesh.meshDisplacementDirectionBoundsInfo;
                break;
            case bary::StandardPropertyType::eMeshDisplacementDirections:
                m_props[i].info = &m_aggregatedInfo.mesh.meshDisplacementDirectionsInfo;
                break;
            case bary::StandardPropertyType::eMeshPositions:
                m_props[i].info = &m_aggregatedInfo.mesh.meshPositionsInfo;
                break;
            case bary::StandardPropertyType::eMeshTriangleFlags:
                m_props[i].info = &m_aggregatedInfo.mesh.meshTriangleFlagsInfo;
                break;
            case bary::StandardPropertyType::eMeshTriangleIndices:
                m_props[i].info = &m_aggregatedInfo.mesh.meshTriangleIndicesInfo;
                break;
            case bary::StandardPropertyType::eMeshTriangleMappings:
                m_props[i].info = &m_aggregatedInfo.mesh.meshTriangleMappingsInfo;
                break;
            case bary::StandardPropertyType::eGroupUncompressedMips:
                m_props[i].info = &m_aggregatedInfo.misc.uncompressedMipsInfo;
                break;
            }
        }
    }

    m_propsList.push_back(props);

    return result;
}

void BarySaver::addCustomProperties(bary::PropertyStorageInfo sinfo)
{
    assert(!m_props.empty() && "must call initContent first");
    m_props.push_back(sinfo);
}


bary::Result BarySaver::SaverContext::save(uint64_t offset, uint64_t size, const void* data)
{
    if(fileHandle)
    {
        FILE* file = reinterpret_cast<FILE*>(fileHandle);
        return fwrite(data, size, 1, file) == 1 ? bary::Result::eSuccess : bary::Result::eErrorIO;
    }
    else if(fileData)
    {
        if(offset + size > fileSize)
        {
            return bary::Result::eErrorFileSize;
        }
        memcpy(fileData + offset, data, size);
        return bary::Result::eSuccess;
    }

    return bary::Result::eErrorUnknown;
}


bary::Result BarySaver::saverCallback(uint32_t                         propertyIdx,
                                      const bary::PropertyStorageInfo* propStore,
                                      uint64_t                         offset,
                                      uint64_t                         size,
                                      const void*                      data,
                                      bool                             isInfo,
                                      void*                            userData)
{
    BarySaver::SaverContext*   ctx = reinterpret_cast<BarySaver::SaverContext*>(userData);
    bary::StandardPropertyType stdProp =
        propStore ? bary::baryPropertyGetStandardType(propStore->identifier) : bary::StandardPropertyType::eUnknown;

    if(stdProp == bary::StandardPropertyType::eUnknown || isInfo || ctx->saver->m_propsList.size() <= 1)
    {
        return ctx->save(offset, size, data);
    }
    else
    {
        // we intend to save multiple bary datas after another
        // this here is only triggered for data not for info, info was pre-altered already
        // through the use of BarySaver::m_aggregatedInfo

        // everything with a "first" needs to be altered
        bool needsPatching = stdProp == bary::StandardPropertyType::eGroups || stdProp == bary::StandardPropertyType::eGroupHistograms
                             || stdProp == bary::StandardPropertyType::eGroupUncompressedMips
                             || stdProp == bary::StandardPropertyType::eMeshGroups
                             || stdProp == bary::StandardPropertyType::eMeshGroupHistograms;

        uint64_t appendedSize = 0;

        if(!needsPatching)
        {
            // simple use case just iterate lists and append the data
            // of this property linearly

            for(const auto& it : ctx->saver->m_propsList)
            {
                const bary::PropertyStorageInfo& appendStore = it.at(propertyIdx);
                bary::Result                     result = ctx->save(offset, appendStore.dataSize, appendStore.data);
                offset += appendStore.dataSize;
                appendedSize += appendStore.dataSize;
                if(result != bary::Result::eSuccess)
                {
                    return result;
                }
            }

        }
        else
        {
            std::vector<uint8_t> tempBytes;

            uint32_t appendedTriangles      = 0;
            uint32_t appendedHistograms     = 0;
            uint32_t appendedValues         = 0;
            uint32_t appendedMeshHistograms = 0;
            uint32_t appendedMips           = 0;
            uint32_t appendedMeshVertices   = 0;
            uint32_t appendedMeshTriangles  = 0;

            for(const auto& it : ctx->saver->m_propsList)
            {
                uint32_t localTriangles      = 0;
                uint32_t localHistograms     = 0;
                uint32_t localValues         = 0;
                uint32_t localMeshHistograms = 0;
                uint32_t localMips           = 0;
                uint32_t localMeshVertices   = 0;
                uint32_t localMeshTriangles  = 0;

                const bary::PropertyStorageInfo& appendStore = it.at(propertyIdx);
                tempBytes.resize(ctx->fileHandle ? appendStore.dataSize : 0);

                // serializing to memory, we write directly, otherwise tempBytes
                uint8_t* tempData = ctx->fileHandle ? tempBytes.data() : (ctx->fileData + offset);

                switch(stdProp)
                {
                case bary::StandardPropertyType::eGroups: {
                    auto     inGroups  = reinterpret_cast<const bary::Group*>(appendStore.data);
                    auto     outGroups = reinterpret_cast<bary::Group*>(tempData);
                    uint64_t count     = appendStore.dataSize / sizeof(bary::Group);
                    for(uint64_t i = 0; i < count; i++)
                    {
                        outGroups[i] = inGroups[i];
                        outGroups[i].triangleFirst += appendedTriangles;
                        outGroups[i].valueFirst += appendedValues;

                        localValues    = std::max(localValues, inGroups[i].valueFirst + inGroups[i].valueCount);
                        localTriangles = std::max(localTriangles, inGroups[i].triangleFirst + inGroups[i].triangleCount);
                    }

                    appendedValues += localValues;
                    appendedTriangles += localTriangles;
                }
                break;
                case bary::StandardPropertyType::eGroupHistograms: {
                    auto     inGroups  = reinterpret_cast<const bary::GroupHistogramRange*>(appendStore.data);
                    auto     outGroups = reinterpret_cast<bary::GroupHistogramRange*>(tempData);
                    uint64_t count     = appendStore.dataSize / sizeof(bary::GroupHistogramRange);
                    for(uint64_t i = 0; i < count; i++)
                    {
                        outGroups[i] = inGroups[i];
                        outGroups[i].entryFirst += appendedHistograms;

                        localHistograms = std::max(localHistograms, inGroups[i].entryFirst + inGroups[i].entryCount);
                    }

                    appendedHistograms += localHistograms;
                }
                break;
                case bary::StandardPropertyType::eGroupUncompressedMips: {
                    auto     inGroups  = reinterpret_cast<const bary::GroupUncompressedMip*>(appendStore.data);
                    auto     outGroups = reinterpret_cast<bary::GroupUncompressedMip*>(tempData);
                    uint64_t count     = appendStore.dataSize / sizeof(bary::GroupUncompressedMip);
                    for(uint64_t i = 0; i < count; i++)
                    {
                        outGroups[i] = inGroups[i];
                        outGroups[i].mipFirst += appendedMips;

                        localMips = std::max(localMips, inGroups[i].mipFirst + inGroups[i].mipCount);
                    }

                    appendedMips += localMips;
                }
                break;
                case bary::StandardPropertyType::eMeshGroups: {
                    auto     inGroups  = reinterpret_cast<const bary::MeshGroup*>(appendStore.data);
                    auto     outGroups = reinterpret_cast<bary::MeshGroup*>(tempData);
                    uint64_t count     = appendStore.dataSize / sizeof(bary::MeshGroup);
                    for(uint64_t i = 0; i < count; i++)
                    {
                        outGroups[i] = inGroups[i];
                        outGroups[i].vertexFirst += appendedMeshVertices;
                        outGroups[i].triangleFirst += appendedMeshTriangles;

                        localMeshVertices = std::max(localMeshVertices, inGroups[i].vertexFirst + inGroups[i].vertexCount);
                        localMeshTriangles = std::max(localMeshTriangles, inGroups[i].triangleFirst + inGroups[i].triangleCount);
                    }

                    appendedMeshTriangles += localMeshTriangles;
                    appendedMeshVertices += localMeshVertices;
                }
                break;
                case bary::StandardPropertyType::eMeshGroupHistograms: {
                    auto     inGroups  = reinterpret_cast<const bary::MeshGroupHistogramRange*>(appendStore.data);
                    auto     outGroups = reinterpret_cast<bary::MeshGroupHistogramRange*>(tempData);
                    uint64_t count     = appendStore.dataSize / sizeof(bary::MeshGroupHistogramRange);
                    for(uint64_t i = 0; i < count; i++)
                    {
                        outGroups[i] = inGroups[i];
                        outGroups[i].entryFirst += appendedMeshHistograms;

                        localMeshHistograms = std::max(localMeshHistograms, inGroups[i].entryFirst + inGroups[i].entryCount);
                    }

                    appendedMeshHistograms += localMeshHistograms;
                }
                break;
                }

                if(tempBytes.size())
                {
                    bary::Result result = ctx->save(offset, tempBytes.size(), tempBytes.data());

                    if(result != bary::Result::eSuccess)
                    {
                        return result;
                    }
                }

                offset += appendStore.dataSize;
                appendedSize += appendStore.dataSize;
            }
        }

        // the sum of all appends must match
        // provided range at the end
        assert(appendedSize == size);
    }

    return bary::Result::eSuccess;
}

bary::Result BarySaver::save(const char* filename) const
{
    if(m_result != bary::Result::eSuccess)
    {
        return m_result;
    }

    uint32_t propertyCount = uint32_t(m_props.size());

    uint64_t             fileSize     = computeFileSize();
    uint64_t             preambleSize = bary::baryStorageComputePreambleSize(propertyCount);
    std::vector<uint8_t> preambleBytes(preambleSize);

    bary::Result result =
        bary::baryStorageOutputPreamble(propertyCount, m_props.data(), fileSize, preambleSize, preambleBytes.data());
    if(result != bary::Result::eSuccess)
    {
        return result;
    }

#ifdef _WIN32
    FILE* file = nullptr;
    if(fopen_s(&file, filename, "wb"))
#else
    FILE*    file            = fopen(filename, "wb");
    if(file == nullptr)
#endif
    {
        return bary::Result::eErrorIO;
    }

    SaverContext ctx;
    ctx.fileHandle = file;
    ctx.saver      = this;

    result = baryStorageOutputSaver(propertyCount, m_props.data(), preambleSize, preambleBytes.data(), saverCallback, &ctx);

    fclose(file);

    return result;
}

bary::Result BarySaver::save(uint64_t fileSize, void* fileData) const
{
    if(m_result != bary::Result::eSuccess)
    {
        return m_result;
    }

    if(fileSize != computeFileSize())
    {
        return bary::Result::eErrorFileSize;
    }

    uint32_t propertyCount = uint32_t(m_props.size());

    uint64_t             preambleSize = bary::baryStorageComputePreambleSize(propertyCount);
    std::vector<uint8_t> preambleBytes(preambleSize);

    bary::Result result =
        bary::baryStorageOutputPreamble(propertyCount, m_props.data(), fileSize, preambleSize, preambleBytes.data());
    if(result != bary::Result::eSuccess)
    {
        return result;
    }

    SaverContext ctx;
    ctx.fileSize = fileSize;
    ctx.fileData = reinterpret_cast<uint8_t*>(fileData);
    ctx.saver    = this;

    result = baryStorageOutputSaver(propertyCount, m_props.data(), preambleSize, preambleBytes.data(), saverCallback, &ctx);

    return result;
}

uint64_t BarySaver::computeFileSize() const
{
    return bary::baryStorageComputeSize(uint32_t(m_props.size()), m_props.data());
}

//////////////////////////////////////////////////////////////////////////

namespace
{
static void* baryDefaultAlloc(size_t size, void* userData)
{
    (void)userData;
    return malloc(size);
}

static void baryDefaultFree(void* ptr, void* userData)
{
    (void)userData;
    free(ptr);
}

#if defined(WIN32) && (defined(__amd64__) || defined(__x86_64__) || defined(_M_X64) || defined(__AMD64__))
#define xftell(f) _ftelli64(f)
#define xfseek(f, pos, encoded) _fseeki64(f, pos, encoded)
#else
#define xftell(f) ftell(f)
#define xfseek(f, pos, encoded) fseek(f, (long)pos, encoded)
#endif

static bary::Result baryDefaultFileRead(const BaryMemoryApi* memoryApi, const BaryFileApi* fileApi, const char* filename, size_t* size, void** data)
{
    (void)fileApi;
    void* (*memory_alloc)(size_t, void*) = memoryApi->alloc ? memoryApi->alloc : &baryDefaultAlloc;
    void (*memory_free)(void*, void*)    = memoryApi->free ? memoryApi->free : &baryDefaultFree;

#ifdef _WIN32
    FILE* file = nullptr;
    if(fopen_s(&file, filename, "rb"))
#else
    FILE* file = fopen(filename, "rb");
    if(file == nullptr)
#endif
    {
        return bary::Result::eErrorIO;
    }

    size_t fileSize = size ? *size : 0;

    if(fileSize == 0)
    {
        // load the full file to memory
        xfseek(file, 0, SEEK_END);
        int64_t length = (int64_t)xftell(file);

        if(length < 0)
        {
            fclose(file);
            return bary::Result::eErrorIO;
        }

        xfseek(file, 0, SEEK_SET);
        fileSize = (size_t)length;
    }

    char* fileData = (char*)memory_alloc(fileSize, memoryApi->userData);
    if(!fileData)
    {
        fclose(file);
        return bary::Result::eErrorOutOfMemory;
    }

    size_t read_size = fread(fileData, 1, fileSize, file);

    fclose(file);

    if(read_size != fileSize)
    {
        memory_free(fileData, memoryApi->userData);
        return bary::Result::eErrorIO;
    }

    if(size)
    {
        *size = fileSize;
    }
    if(data)
    {
        *data = fileData;
    }

    return bary::Result::eSuccess;
}

static void baryDefaultFileRelease(const BaryMemoryApi* memoryApi, const BaryFileApi* fileApi, void* data)
{
    (void)fileApi;
    void (*memfree)(void*, void*) = memoryApi->free ? memoryApi->free : &baryDefaultFree;
    memfree(data, memoryApi->userData);
}

}  // namespace

class BaryFileHandle
{
  public:
    bary::Result open(const char* filename, const BaryFileOpenOptions* options = nullptr)
    {
        if(options)
        {
            assert(!options->fileApi.read || (options->fileApi.read && options->fileApi.release));
            assert(!options->memoryApi.alloc || (options->memoryApi.alloc && options->memoryApi.free));
        }

        m_fileApi.read       = options && options->fileApi.read ? options->fileApi.read : &baryDefaultFileRead;
        m_fileApi.release    = options && options->fileApi.release ? options->fileApi.release : &baryDefaultFileRelease;
        m_fileApi.userData   = options ? options->fileApi.userData : nullptr;
        m_memoryApi.alloc    = options && options->memoryApi.alloc ? options->memoryApi.alloc : &baryDefaultAlloc;
        m_memoryApi.free     = options && options->memoryApi.free ? options->memoryApi.free : &baryDefaultFree;
        m_memoryApi.userData = options ? options->memoryApi.userData : nullptr;

        return m_fileApi.read(&m_memoryApi, &m_fileApi, filename, &m_fileSize, &m_fileData);
    }

    void close()
    {
        if(m_fileApi.release)
        {
            m_fileApi.release(&m_memoryApi, &m_fileApi, (void*)m_fileData);
        }
    }

    size_t        m_fileSize = 0;
    void*         m_fileData = nullptr;
    BaryFileApi   m_fileApi;
    BaryMemoryApi m_memoryApi;
};

//////////////////////////////////////////////////////////////////////////

bary::Result BaryFile::open(const char* name, const BaryFileOpenOptions* options, bary::StandardPropertyType* outErrorType)
{
    assert(!m_handle && "BaryFile handle already in-use / opened");
    m_handle            = new BaryFileHandle();
    bary::Result result = m_handle->open(name, options);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }

    m_fileData = m_handle->m_fileData;
    m_fileSize = m_handle->m_fileSize;

    return setupContent(outErrorType);
}

void BaryFile::close()
{
    if(m_handle)
    {
        m_handle->close();

        delete m_handle;
        m_handle = nullptr;
    }

    m_content  = bary::ContentView();
    m_fileSize = 0;
    m_fileData = nullptr;
}

bary::Result BaryFile::setupContent(bary::StandardPropertyType* outErrorType)
{
    bary::Result result = bary::baryDataIsValid(m_fileSize, m_fileData);
    if(result != bary::Result::eSuccess)
    {
        return result;
    }

    return bary::baryDataGetContent(m_fileSize, m_fileData, &m_content, outErrorType);
}

bool BaryFile::hasProperty(bary::StandardPropertyType prop) const
{
    if(!m_fileSize || !m_fileData)
    {
        return false;
    }
    bary::PropertyIdentifier id = bary::baryStandardPropertyGetIdentifier(prop);
    return baryDataGetPropertyInfo(m_fileSize, m_fileData, id) != nullptr;
}

//////////////////////////////////////////////////////////////////////////

namespace
{
inline BaryWUV_uint16 bc(uint32_t a, uint32_t b, uint32_t c)
{
    return {uint16_t(a), uint16_t(b), uint16_t(c)};
}

struct uvCoord
{
    uint32_t u;
    uint32_t v;
};

BaryLevelsMap::Triangle processTriangle(BaryLevelsMap::Level& level, uvCoord uvA, uvCoord uvB, uvCoord uvC)
{
    BaryLevelsMap::Triangle tri;
    tri.a = level.getCoordIndex(bc((1 << level.subdivLevel) - (uvA.u + uvA.v), uvA.u, uvA.v));
    tri.b = level.getCoordIndex(bc((1 << level.subdivLevel) - (uvB.u + uvB.v), uvB.u, uvB.v));
    tri.c = level.getCoordIndex(bc((1 << level.subdivLevel) - (uvC.u + uvC.v), uvC.u, uvC.v));
    return tri;
}

}  // namespace

BaryWUV_uint16 BaryLevelsMap::joinVertex(BaryWUV_uint16 coord, uint32_t decimateEdgeBits, uint32_t subdivLevel)
{
    uint32_t baryMax = 1 << subdivLevel;

    if(decimateEdgeBits & 1 && coord.v == 0)
    {
        if(coord.w < baryMax / 2)
            return bc((coord.w) & ~1, (coord.u + 1) & ~1, 0);
        else
            return bc((coord.w + 1) & ~1, (coord.u) & ~1, 0);
    }
    if(decimateEdgeBits & 2 && coord.w == 0)
    {
        if(coord.u < baryMax / 2)
            return bc(0, (coord.u) & ~1, (coord.v + 1) & ~1);
        else
            return bc(0, (coord.u + 1) & ~1, (coord.v) & ~1);
    }
    if(decimateEdgeBits & 4 && coord.u == 0)
    {
        if(coord.v < baryMax / 2)
            return bc((coord.w + 1) & ~1, 0, (coord.v) & ~1);
        else
            return bc((coord.w) & ~1, 0, (coord.v + 1) & ~1);
    }
    return coord;
}

std::vector<BaryLevelsMap::Triangle> BaryLevelsMap::Level::buildTrianglesWithCollapsedEdges(uint32_t decimateEdgeBits,
                                                                                            bool useDegenerated) const
{
    if(subdivLevel == 0 || decimateEdgeBits == 0)
    {
        return triangles;
    }

    std::vector<BaryLevelsMap::Triangle> joinIndices;
    joinIndices.reserve(triangles.size());

    for(const BaryLevelsMap::Triangle& triangle : triangles)
    {
        BaryWUV_uint16 baryA = joinVertex(coordinates[triangle.a], decimateEdgeBits, subdivLevel);
        BaryWUV_uint16 baryB = joinVertex(coordinates[triangle.b], decimateEdgeBits, subdivLevel);
        BaryWUV_uint16 baryC = joinVertex(coordinates[triangle.c], decimateEdgeBits, subdivLevel);

        BaryLevelsMap::Triangle tri;
        tri.a = getCoordIndex(baryA);
        tri.b = getCoordIndex(baryB);
        tri.c = getCoordIndex(baryC);

        if(useDegenerated || (tri.a != tri.b && tri.b != tri.c && tri.c != tri.a))
        {
            joinIndices.push_back(tri);
        }
    }

    return joinIndices;
}

void BaryLevelsMap::initialize(bary::ValueLayout layout, uint32_t maxLevel)
{
    assert(maxLevel <= MAX_LEVEL);

    m_layout = layout;
    m_levels.clear();

    if(layout == bary::ValueLayout::eUndefined)
    {
        return;
    }

    m_levels.resize(maxLevel + 1);

    for(uint32_t lvl = 0; lvl < maxLevel + 1; lvl++)
    {
        BaryLevelsMap::Level& level = m_levels[lvl];

        uint32_t numSegmentsPerEdge = (1 << lvl);
        uint32_t numVtxPerEdge      = numSegmentsPerEdge + 1;
        uint32_t numVertices        = (numVtxPerEdge * (numVtxPerEdge + 1)) / 2;

        level.coordinates.resize(numVertices);
        level.triangles.resize(numSegmentsPerEdge * numSegmentsPerEdge);
        level.layout      = m_layout;
        level.subdivLevel = lvl;

        for(uint32_t u = 0; u < numVtxPerEdge; u++)
        {
            for(uint32_t v = 0; v < numVtxPerEdge - u; v++)
            {
                BaryWUV_uint16 coord = bc(0, u, v);
                coord.w              = numSegmentsPerEdge - (coord.u + coord.v);
                uint32_t idx = bary::baryValueLayoutGetIndex(layout, bary::ValueFrequency::ePerVertex, coord.u, coord.v, 0, lvl);
                level.coordinates[idx] = coord;
            }
        }

        for(uint32_t u = 0; u < numSegmentsPerEdge; u++)
        {
            for(uint32_t v = 0; v < numSegmentsPerEdge - u; v++)
            {
                BaryWUV_uint16 coord = bc(0, u, v);
                coord.w              = numSegmentsPerEdge - (coord.u + coord.v);
                {
                    uint32_t idx =
                        bary::baryValueLayoutGetIndex(layout, bary::ValueFrequency::ePerTriangle, coord.u, coord.v, 0, lvl);
                    Triangle tri = processTriangle(level, {coord.u, coord.v}, {coord.u + 1u, coord.v}, {coord.u, coord.v + 1u});

                    level.triangles[idx] = tri;
                }
                if(v != numSegmentsPerEdge - u - 1)
                {
                    uint32_t idx =
                        bary::baryValueLayoutGetIndex(layout, bary::ValueFrequency::ePerTriangle, coord.u, coord.v, 1, lvl);
                    // warning the order here was tuned for bird-curve, horizontal edge first, in theory need a different way of doing this
                    Triangle tri = processTriangle(level, {coord.u + 1u, coord.v + 1u}, {coord.u, coord.v + 1u},
                                                   {coord.u + 1u, coord.v});

                    level.triangles[idx] = tri;
                }
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////

void BarySplitTable::Entry::init(bary::BlockFormatDispC1 format, uint32_t baseSubdiv)
{
    uint32_t count = bary::baryBlockFormatDispC1GetBlockCount(format, baseSubdiv);

    tris.resize(count);
    bary::baryBlockFormatDispC1GetBlockTriangles(format, baseSubdiv, count, tris.data());
}

bool BarySplitTable::init(bary::Format format, uint32_t maxLevel)
{
    if(format != bary::Format::eDispC1_r11_unorm_block)
    {
        return true;
    }
    m_numFormats = 3;
    m_format     = format;

    const uint32_t numLevels = maxLevel + 1;
    m_splits.resize(numLevels * m_numFormats);
    for(uint32_t f = 0; f < m_numFormats; f++)
    {
        bary::BlockFormatDispC1 format = bary::BlockFormatDispC1(uint32_t(bary::BlockFormatDispC1::eR11_unorm_lvl3_pack512) + f);

        for(uint32_t level = 0; level < numLevels; level++)
        {
            m_splits[getIndex(format, level)].init(format, level);
        }
    }

    return false;
}

//////////////////////////////////////////////////////////////////////////

}  // namespace baryutils
