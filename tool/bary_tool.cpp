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

#include <cstdio>
#include <memory>

#include <baryutils/baryutils.h>

#include "filemapping.hpp"


static bary::Result baryutils_read(const baryutils::BaryMemoryApi* memory_options,
                                   const baryutils::BaryFileApi*   file_options,
                                   const char*                     path,
                                   size_t*                         size,
                                   void**                          data)
{
    FileMappingList* mappings = (FileMappingList*)file_options->userData;
    if(mappings->open(path, size, data))
    {
        return bary::Result::eSuccess;
    }

    return bary::Result::eErrorIO;
}

static void baryutils_release(const baryutils::BaryMemoryApi* memory_options, const baryutils::BaryFileApi* file_options, void* data)
{
    FileMappingList* mappings = (FileMappingList*)file_options->userData;
    mappings->close(data);
}

static bool checkAndPrintError(const char*                operation,
                               bary::Result               result,
                               bary::StandardPropertyType errorProp = bary::StandardPropertyType::eUnknown)
{
    if(result != bary::Result::eSuccess)
    {
        printf("error in '%s': %s (prop: %s)\n", operation, bary::baryResultGetName(result),
               errorProp == bary::StandardPropertyType::eUnknown ? "-" : bary::baryStandardPropertyGetName(errorProp));
        return false;
    }

    return true;
}


bary::Result loadConverted(baryutils::BaryContentData& baryData, const baryutils::BaryFileOpenOptions& options, const char* filename, uint32_t versionNumber)
{
    // might react on versionNumber here
    return bary::Result::eErrorVersion;
}

void printHelp()
{
    bary::VersionIdentifier id = bary::baryGetCurrentVersionIdentifier();
    uint32_t                currentVersion;
    bary::baryVersionIdentifierGetVersion(&id, &currentVersion);

    printf("bary_tool <filename> ...commands...\n");
    printf("supported file versions:\n");
    printf("------------------------\n");
    printf("%d (current)\n", currentVersion);
    printf("commands:\n");
    printf("---------\n");
    printf("-appendonsave <filename>: appends another file when save is triggered (only standard properties)\n");
    printf("-save <filename>: saves file (only standard properties)\n");
}

bool validateContent(const bary::ContentView* content, const char* what)
{
    bary::Result result;
    bary::StandardPropertyType errorProp = bary::StandardPropertyType::eUnknown;

    printf("%s validation\n", what);

    uint32_t                               propStorageCount = bary::baryContentComputePropertyCount(content);
    std::vector<bary::PropertyStorageInfo> propStorages(propStorageCount);
    result = bary::baryContentSetupProperties(content, propStorageCount, propStorages.data());
    if(!checkAndPrintError("validation", result))
    {
        return false;
    }

    for(uint32_t i = 0; i < propStorageCount; i++)
    {
        bary::PropertyIdentifier   identifier = propStorages[i].identifier;
        bary::StandardPropertyType prop       = bary::baryPropertyGetStandardType(identifier);
        printf("  property %s\n", bary::baryStandardPropertyGetName(prop));
    }

    result = bary::baryValidateStandardProperties(propStorageCount, propStorages.data(), ~uint64_t(0), &errorProp);
    if(!checkAndPrintError("validation", result, errorProp))
    {
        return false;
    }
    printf("  all passed\n\n");

    return true;
}

extern "C" int main(int argc, const char** argv)
{
    if(argc < 2)
    {
        printHelp();
        return 0;
    }

    const char* filename = argv[1];

    printf("input file: %s\n", filename);

    uint32_t versionNumber = 0;

    {
        FILE* file;
#ifdef WIN32
        if(fopen_s(&file, filename, "rb"))
        {
#else
        if((file = fopen(filename, "rb")) == nullptr)
        {
#endif
            printf("could not open file\n");
            return -1;
        }

        bary::VersionIdentifier id;
        if(!fread(&id, sizeof(id), 1, file))
        {
            printf("could not read version identifier\n");
            return -1;
        }
        fclose(file);

        if(bary::baryVersionIdentifierGetVersion(&id, &versionNumber) != bary::Result::eSuccess)
        {
            printf("could not read version number\n");
            return -1;
        }

        printf("Version:  %d\n\n", versionNumber);
    }

    FileMappingList mappingList;

    std::vector<const char*> appendFilenames;
    std::vector<std::unique_ptr<baryutils::BaryFile>> appendFiles;

    baryutils::BaryFile        bfile;
    bary::ContentView          content;
    bary::BasicView            basicView;
    bary::MeshView             meshView;
    bary::MiscView             miscView;
    baryutils::BaryContentData baryConvert;

    bary::StandardPropertyType     errorProp   = bary::StandardPropertyType::eUnknown;
    baryutils::BaryFileOpenOptions openOptions = {0};
    openOptions.fileApi.userData               = &mappingList;
    openOptions.fileApi.read                   = baryutils_read;
    openOptions.fileApi.release                = baryutils_release;

    bary::Result result = bfile.open(filename, &openOptions, &errorProp);
    checkAndPrintError("open file", result, errorProp);

    if(result == bary::Result::eErrorVersion)
    {
        bfile.close();

        printf("attempting to convert older version\n");
        result = loadConverted(baryConvert, openOptions, filename, versionNumber);
        if(!checkAndPrintError("convert file", result))
        {
            return -1;
        }

        content.basic = baryConvert.basic.getView();
        content.mesh  = baryConvert.mesh.getView();
        content.misc  = baryConvert.misc.getView();
    }
    else if(result == bary::Result::eSuccess)
    {
        content.basic = bfile.getBasic();
        content.mesh  = bfile.getMesh();
        content.misc  = bfile.getMisc();

        // list all properties
        uint64_t propCount;
        printf("File properties\n");
        const bary::PropertyInfo* propInfos = bary::baryDataGetAllPropertyInfos(bfile.m_fileSize, bfile.m_fileData, &propCount);
        for(uint16_t i = 0; i < propCount; i++)
        {
            bary::PropertyIdentifier identifier = propInfos[i].identifier;
            printf("  property %d: ", i);
            bary::StandardPropertyType prop = bary::baryPropertyGetStandardType(identifier);
            if(prop == bary::StandardPropertyType::eUnknown)
            {
                printf("unknown identifier {0x%x, 0x%x, 0x%x, 0x%x}\n", identifier.uuid4[0], identifier.uuid4[1],
                       identifier.uuid4[2], identifier.uuid4[3]);
            }
            else
            {
                printf("%s\n", bary::baryStandardPropertyGetName(prop));
            }
        }
        printf("\n");
    }
    else
    {
        return -1;
    }

    if (!validateContent(&content, "input"))
    {
        return -1;
    }

    printf("Globals\n");
    printf("  triangles         %10d\n", content.basic.trianglesCount);
    printf("  groups            %10d\n", content.basic.groupsCount);
    printf("\n");

    auto valuesInfo = content.basic.valuesInfo;
    printf("ValueInfo\n");
    printf("  valueCount        %10d\n", valuesInfo->valueCount);
    printf("  valueByteSize     %10d\n", valuesInfo->valueByteSize);
    printf("  valueByteAlign    %10d\n", valuesInfo->valueByteAlignment);
    printf("  valueFormat       %s\n", bary::baryFormatGetName(valuesInfo->valueFormat));
    printf("  valueLayout       %s\n", bary::baryValueLayoutGetName(valuesInfo->valueLayout));
    printf("  valueFrequency    %s\n", bary::baryValueFrequencyGetName(valuesInfo->valueFrequency));
    printf("\n");

    // iterate groups
    for(uint32_t g = 0; g < content.basic.groupsCount; g++)
    {
        const bary::Group* group = content.basic.groups + g;

        printf("Group %d:\n", g);
        printf("  triangleCount     %10d\n", group->triangleCount);
        printf("  triangleFirst     %10d\n", group->triangleFirst);
        printf("  valueCount        %10d\n", group->valueCount);
        printf("  valueFirst        %10d\n", group->valueFirst);
        printf("  minSubdivLevel    %10d\n", group->minSubdivLevel);
        printf("  maxSubdivLevel    %10d\n", group->maxSubdivLevel);
        printf("  bias  {%f, %f, %f, %f}\n", group->floatBias.r, group->floatBias.g, group->floatBias.b,
               group->floatBias.a);
        printf("  scale {%f, %f, %f, %f}\n", group->floatScale.r, group->floatScale.g, group->floatScale.b,
               group->floatScale.a);
        printf("\n");

        // hack
        std::vector<uint32_t> histoLevel(group->maxSubdivLevel + 1, 0);
        const uint32_t        histoBlockEntries = 4;
        std::vector<uint32_t> histoBlock(histoBlockEntries, 0);

        for(uint32_t i = 0; i < group->triangleCount; i++)
        {
            const bary::Triangle* tri = content.basic.triangles + (group->triangleFirst + i);
            histoLevel[tri->subdivLevel]++;
            histoBlock[tri->blockFormat % histoBlockEntries]++;
        }

        printf("  computed primitive subdivlevel histogram:\n");
        for(uint32_t i = 0; i < group->maxSubdivLevel + 1; i++)
        {
            printf("    subdiv %2d: %9d\n", i, histoLevel[i]);
        }

        printf("  computed primitive blockformat histogram:\n");
        for(uint32_t i = 1; i < histoBlockEntries; i++)
        {
            printf("    blockformat %2d: %9d\n", i, histoBlock[i]);
        }

        if(content.basic.histogramEntries && content.basic.groupHistogramRanges)
        {
            printf("  file property block format histogram:\n");
            for(uint32_t i = 0; i < content.basic.groupHistogramRanges[g].entryCount; i++)
            {
                const bary::HistogramEntry* entry =
                    content.basic.histogramEntries + (content.basic.groupHistogramRanges[g].entryFirst + i);
                printf("    subdiv %2d blockformat %2d: %9d\n", entry->subdivLevel, entry->blockFormat, entry->count);
            }
        }

        printf("\n");
    }
    printf("\n");

    for(int i = 2; i < argc; i++)
    {
        if(strcmp(argv[i], "-appendonsave") == 0 && i + 1 < argc)
        {
            i++;
            const char* appendname = argv[i];

            std::unique_ptr<baryutils::BaryFile> bfileAppend = std::make_unique<baryutils::BaryFile>();
            
            printf("appendonsave file: %s\n", appendname);
            bary::Result result     = bfileAppend->open(appendname, &openOptions, &errorProp);
            if (!checkAndPrintError("open appendonsave file", result, errorProp))
            {
                return -1;
            }

            bary::ContentView appendContent = bfileAppend->getContent();
            if(!validateContent(&appendContent, "appendonsave"))
            {
                return -1;
            }

            appendFiles.push_back(std::move(bfileAppend));
            appendFilenames.push_back(appendname);
        }
        else if(strcmp(argv[i], "-save") == 0 && i + 1 < argc)
        {
            i++;
            const char* savename = argv[i];

            bary::StandardPropertyType errorProp = bary::StandardPropertyType::eUnknown;
            baryutils::BarySaver       saver;
            printf("save file: %s\n", savename);
            result = saver.initContent(&content, &errorProp);
            if(!checkAndPrintError("save init", result, errorProp))
            {
                return -1;
            }

            if (appendFiles.size())
            {
                for (size_t a = 0; a < appendFiles.size(); a++)
                {
                    printf("save append %s\n", appendFilenames[a]);

                    bary::ContentView appendContent = appendFiles[a]->getContent();
                    result = saver.appendContent(&appendContent, &errorProp);
                    if(!checkAndPrintError("save append", result, errorProp))
                    {
                        return -1;
                    }
                }
            }

            result = saver.save(savename);
            if(!checkAndPrintError("save file", result))
            {
                return -1;
            }
            printf("successfully saved\n");
        }
    }

    return 0;
}
