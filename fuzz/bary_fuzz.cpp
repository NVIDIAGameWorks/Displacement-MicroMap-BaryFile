/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Implements a libFuzzer target for bary files. This provides an automated
// way we can check bary_core is safe when trying to open untrusted .bary files.
// For more information about how to run a libFuzzer target, please see
// https://llvm.org/docs/LibFuzzer.html or
// https://github.com/google/fuzzing/blob/master/tutorial/libFuzzerTutorial.md.

#include <cstdlib>
#include <limits>
#include <vector>
#include "bary/bary_core.h"

// LLVM libFuzzer entrypoint.
// libFuzzer generates main() for us automatically, since we compile this using
// -fsanitize=fuzzer.
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* Data, size_t Size)
{
    // This fuzzer interprets the entire test input as if it was a .bary
    // file, and tests if opening and saving it would cause unexpected
    // program execution.
    // For now, we always allow libFuzzer to add inputs to the corpus
    // by returning 0.

    if(bary::Result::eSuccess != bary::baryDataIsValid(Size, Data))
    {
        return 0;
    }

    bary::ContentView          content{};
    bary::StandardPropertyType errorProperty{};
    if(bary::Result::eSuccess != bary::baryDataGetContent(Size, Data, &content, &errorProperty))
    {
        return 0;
    }

    if(bary::Result::eSuccess != bary::baryContentIsValid(bary::ValueSemanticType::eDisplacement, &content, &errorProperty))
    {
        return 0;
    }

    // Get all the properties, and try to access their first and last elements.
    // If a property somehow has a range that is not valid, halt: it must have
    // somehow slipped past the bary validator.
    uint64_t                  numProperties = 0;
    const bary::PropertyInfo* properties    = bary::baryDataGetAllPropertyInfos(Size, Data, &numProperties);
    for(uint64_t i = 0; i < numProperties; i++)
    {
        if(!bary::baryDataIsRangeValid(Size, properties[i].range))
        {
            abort();
        }
        if(!bary::baryDataIsRangeValid(Size, properties[i].supercompressionGlobalData))
        {
            abort();
        }
    }

    // Now, let's emulate an app that has loaded the .bary file into internal
    // structures, and is now trying to save the .bary file back out.
    if(numProperties > std::numeric_limits<uint32_t>::max())
    {
        return 0;
    }

    std::vector<bary::PropertyStorageInfo> propertyStorageInfos;
    try
    {
        for(uint64_t i = 0; i < numProperties; i++)
        {
            bary::PropertyStorageInfo psi{};
            psi.identifier                     = properties[i].identifier;
            psi.dataSize                       = properties[i].range.byteLength;
            psi.data                           = Data + properties[i].range.byteOffset;
            psi.supercompressionScheme         = properties[i].supercompressionScheme;
            psi.uncompressedSize               = properties[i].uncompressedByteLength;
            psi.supercompressionGlobalDataSize = properties[i].supercompressionGlobalData.byteLength;
            psi.supercompressionGlobalData     = Data + properties[i].supercompressionGlobalData.byteOffset;
            propertyStorageInfos.push_back(psi);
        }
    }
    catch(const std::bad_alloc& /* unused */)
    {
        return 0;
    }

    if(bary::Result::eSuccess
       != bary::baryValidateStandardProperties(static_cast<uint32_t>(numProperties), propertyStorageInfos.data(),
                                               bary::ValidationFlagBit::eValidationFlagArrayContents
                                                   | bary::ValidationFlagBit::eValidationFlagTriangleValueRange,
                                               &errorProperty))
    {
        return 0;
    }

    const uint64_t outputSize = bary::baryStorageComputeSize(static_cast<uint32_t>(numProperties), propertyStorageInfos.data());
    std::vector<uint8_t> outputData;
    try
    {
        outputData.resize(outputSize);
    }
    catch(const std::bad_alloc& /* unused */)
    {
        return 0;
    }

    if(bary::Result::eSuccess
       != bary::baryStorageOutputAll(static_cast<uint32_t>(numProperties), propertyStorageInfos.data(), outputSize,
                                     outputData.data()))
    {
        return 0;
    }

    return 0;
}
