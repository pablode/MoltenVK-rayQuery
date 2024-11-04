/*
 * MVKCmdAccelerationStructure.mm
 *
 * Copyright (c) 2015-2023 The Brenwill Workshop Ltd. (http://www.brenwill.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "MVKCmdAccelerationStructure.h"
#include "MVKCmdDebug.h"
#include "MVKCommandBuffer.h"
#include "MVKCommandPool.h"
#include "MVKAccelerationStructure.h"

#include <Metal/Metal.h>

#pragma mark -
#pragma mark MVKCmdBuildAccelerationStructure

VkResult MVKCmdBuildAccelerationStructure::setContent(MVKCommandBuffer*                                       cmdBuff,
                                                      uint32_t                                                infoCount,
                                                      const VkAccelerationStructureBuildGeometryInfoKHR*      pInfos,
                                                      const VkAccelerationStructureBuildRangeInfoKHR* const*  ppBuildRangeInfos) {
    _buildInfos.reserve(infoCount);
    for (uint32_t i = 0; i < infoCount; i++)
    {
        uint32_t geometryCount = pInfos[i].geometryCount;

        MVKAccelerationStructureBuildInfo& info = _buildInfos.emplace_back();
        info.info = pInfos[i];

        info.geometries.resize(geometryCount);
        info.ranges.resize(geometryCount);

        memcpy(info.geometries.data(), pInfos[i].pGeometries, geometryCount * sizeof(VkAccelerationStructureBuildGeometryInfoKHR));
        for (uint32_t j = 0; j < geometryCount; j++)
        {
            memcpy(&info.ranges[i], ppBuildRangeInfos[i], sizeof(VkAccelerationStructureBuildRangeInfoKHR));
        }

        info.info.geometryCount = geometryCount;
        info.info.pGeometries = info.geometries.data();
        info.info.ppGeometries = nullptr;
    }

    return VK_SUCCESS;
}

void MVKCmdBuildAccelerationStructure::encode(MVKCommandEncoder* cmdEncoder) {
    id<MTLAccelerationStructureCommandEncoder> accStructEncoder = cmdEncoder->getMTLAccelerationStructureEncoder(kMVKCommandUseBuildAccelerationStructure);
    
        MVKDevice* mvkDevice = cmdEncoder->getDevice();
    for (MVKAccelerationStructureBuildInfo& entry : _buildInfos)
    {
        VkAccelerationStructureBuildGeometryInfoKHR& buildInfo = entry.info;

        id<MTLAccelerationStructure> srcAccStruct = nullptr;
	if (buildInfo.srcAccelerationStructure != VK_NULL_HANDLE)
	{
		MVKAccelerationStructure* mvkSrcAccStruct = (MVKAccelerationStructure*)buildInfo.srcAccelerationStructure;
		srcAccStruct = mvkSrcAccStruct->getMTLAccelerationStructure();
	}

	MTLAccelerationStructureDescriptor* descriptor = nullptr;
        MVKAccelerationStructure* mvkDstAccStruct = (MVKAccelerationStructure*)buildInfo.dstAccelerationStructure;
        if (buildInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)
        {
            descriptor = mvkDstAccStruct->populateMTLDescriptor(
                mvkDevice,
                buildInfo,
                entry.ranges.data(),
                nullptr
            );

            id<MTLDevice> mtlDev = cmdEncoder->getPhysicalDevice()->getMTLDevice();
            MTLAccelerationStructureSizes sizes = [mtlDev accelerationStructureSizesWithDescriptor:descriptor];

            VkAccelerationStructureBuildSizesInfoKHR buildSizes = {};
            buildSizes.accelerationStructureSize = sizes.accelerationStructureSize;
            buildSizes.buildScratchSize = sizes.buildScratchBufferSize;
            buildSizes.updateScratchSize = sizes.refitScratchBufferSize;
	    mvkDstAccStruct->setBuildSizes(buildSizes);
	}

        id<MTLAccelerationStructure> dstAccStruct = mvkDstAccStruct->getMTLAccelerationStructure();
        id<MTLHeap> dstAccStructHeap = mvkDstAccStruct->getMTLHeap();
        
        // Should we throw an error here?
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/vkCmdBuildAccelerationStructuresKHR.html#VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03667
        if(buildInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR && !mvkDstAccStruct->getAllowUpdate())
            continue;
        
        MVKBuffer* mvkBuffer = mvkDevice->getBufferAtAddress(buildInfo.scratchData.deviceAddress);

        // TODO: throw error if mvkBuffer is null?
        
        id<MTLBuffer> scratchBuffer = mvkBuffer->getMTLBuffer();
        NSInteger scratchBufferOffset = mvkBuffer->getMTLBufferOffset();
        assert(scratchBuffer);
        assert(mvkBuffer->getMTLBufferGPUAddress());

        if (buildInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)
        {
            [accStructEncoder buildAccelerationStructure:dstAccStruct
                                                descriptor:descriptor
                                                scratchBuffer:scratchBuffer
                                                scratchBufferOffset:scratchBufferOffset];
        }
        else if (buildInfo.mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR)
        {
            descriptor = [MTLAccelerationStructureDescriptor new];
            
            if (mvkIsAnyFlagEnabled(buildInfo.flags, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR))
                descriptor.usage += MTLAccelerationStructureUsagePreferFastBuild;
            
            [accStructEncoder refitAccelerationStructure:srcAccStruct
                                              descriptor:descriptor
                                             destination:dstAccStruct
                                           scratchBuffer:scratchBuffer
                                     scratchBufferOffset:scratchBufferOffset];
        }
    }
}

#pragma mark -
#pragma mark MVKCmdCopyAccelerationStructure

VkResult MVKCmdCopyAccelerationStructure::setContent(MVKCommandBuffer*                  cmdBuff,
                                                     VkAccelerationStructureKHR         srcAccelerationStructure,
                                                     VkAccelerationStructureKHR         dstAccelerationStructure,
                                                     VkCopyAccelerationStructureModeKHR copyMode) {
    
    MVKAccelerationStructure* mvkSrcAccStruct = (MVKAccelerationStructure*)srcAccelerationStructure;
    MVKAccelerationStructure* mvkDstAccStruct = (MVKAccelerationStructure*)dstAccelerationStructure;
    
    _srcAccelerationStructure = mvkSrcAccStruct->getMTLAccelerationStructure();
    _dstAccelerationStructure = mvkDstAccStruct->getMTLAccelerationStructure();
    _copyMode = copyMode;
    return VK_SUCCESS;
}

void MVKCmdCopyAccelerationStructure::encode(MVKCommandEncoder* cmdEncoder) {
    id<MTLAccelerationStructureCommandEncoder> accStructEncoder = cmdEncoder->getMTLAccelerationStructureEncoder(kMVKCommandUseCopyAccelerationStructure);
    if(_copyMode == VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR)
    {
        [accStructEncoder
         copyAndCompactAccelerationStructure:_srcAccelerationStructure
         toAccelerationStructure:_dstAccelerationStructure];
        
        return;
    }
    
    [accStructEncoder
         copyAccelerationStructure:_srcAccelerationStructure
         toAccelerationStructure:_dstAccelerationStructure];
}

#pragma mark -
#pragma mark MVKCmdCopyAccelerationStructureToMemory

VkResult MVKCmdCopyAccelerationStructureToMemory::setContent(MVKCommandBuffer*                  cmdBuff,
                                                             VkAccelerationStructureKHR         srcAccelerationStructure,
                                                             uint64_t                           dstAddress,
                                                             VkCopyAccelerationStructureModeKHR copyMode) {
    _dstAddress = dstAddress;
    _copyMode = copyMode;
    
    MVKAccelerationStructure* mvkSrcAccStruct = (MVKAccelerationStructure*)srcAccelerationStructure;
    _srcAccelerationStructure = mvkSrcAccStruct->getMTLAccelerationStructure();
    

    _dstBuffer = _mvkDevice->getBufferAtAddress(_dstAddress);
    return VK_SUCCESS;
}
                                        
void MVKCmdCopyAccelerationStructureToMemory::encode(MVKCommandEncoder* cmdEncoder) {
    id<MTLBlitCommandEncoder> blitEncoder = cmdEncoder->getMTLBlitEncoder(kMVKCommandUseCopyAccelerationStructureToMemory);
    _mvkDevice = cmdEncoder->getDevice();
    
    [blitEncoder copyFromBuffer:_srcAccelerationStructureBuffer sourceOffset:0 toBuffer:_dstBuffer->getMTLBuffer() destinationOffset:0 size:_copySize];
}

#pragma mark -
#pragma mark MVKCmdCopyMemoryToAccelerationStructure

VkResult MVKCmdCopyMemoryToAccelerationStructure::setContent(MVKCommandBuffer* cmdBuff,
                                                             uint64_t srcAddress,
                                                             VkAccelerationStructureKHR dstAccelerationStructure,
                                                             VkCopyAccelerationStructureModeKHR copyMode) {
    _srcAddress = srcAddress;
    _copyMode = copyMode;
    
    _srcBuffer = _mvkDevice->getBufferAtAddress(_srcAddress);
    
    MVKAccelerationStructure* mvkDstAccStruct = (MVKAccelerationStructure*)dstAccelerationStructure;
    _dstAccelerationStructure = mvkDstAccStruct->getMTLAccelerationStructure();
    _dstAccelerationStructureBuffer = mvkDstAccStruct->getMTLBuffer();
    return VK_SUCCESS;
}

void MVKCmdCopyMemoryToAccelerationStructure::encode(MVKCommandEncoder* cmdEncoder) {
    id<MTLBlitCommandEncoder> blitEncoder = cmdEncoder->getMTLBlitEncoder(kMVKCommandUseCopyAccelerationStructureToMemory);
    _mvkDevice = cmdEncoder->getDevice();
    
    [blitEncoder copyFromBuffer:_srcBuffer->getMTLBuffer() sourceOffset:0 toBuffer:_dstAccelerationStructureBuffer destinationOffset:0 size:_copySize];
}

#pragma mark -
#pragma mark MVKCmdWriteAccelerationStructuresProperties

VkResult MVKCmdWriteAccelerationStructuresProperties::setContent(MVKCommandBuffer* cmdBuff,
                    uint32_t accelerationStructureCount,
                    const VkAccelerationStructureKHR* pAccelerationStructures,
                    VkQueryType queryType,
                    VkQueryPool queryPool,
                    uint32_t firstQuery) {
    
    _accelerationStructureCount = accelerationStructureCount;
    _pAccelerationStructures = (const MVKAccelerationStructure*)pAccelerationStructures;
    _queryType = queryType;
    _queryPool = queryPool;
    _firstQuery = firstQuery;
    return VK_SUCCESS;
}

void MVKCmdWriteAccelerationStructuresProperties::encode(MVKCommandEncoder* cmdEncoder) {
    
    for(int i = 0; i < _accelerationStructureCount; i++)
    {
        if(!_pAccelerationStructures[i].getBuildStatus()) {
            return;
        }
        
        // actually finish up the meat of the code here
    }
    
    switch(_queryType)
    {
        case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SIZE_KHR:
            break;
        case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_BOTTOM_LEVEL_POINTERS_KHR:
            break;
        case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR:
            break;
        case VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR:
            break;
        default:
            break;
    }
}
