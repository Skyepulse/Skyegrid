#include "../../../includes/Rendering/Pipelines/pipelines.hpp"
#include "../../../includes/constants.hpp"

//================================//
void CreateRenderPipelineDebug(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper)
{
    // Code in ../../../src/Shaders/debugShader.wgsl
    std::string shaderCode;
    if (getShaderCodeFromFile("Shaders/debug.wgsl", shaderCode) < 0)
    {
        throw std::runtime_error(
            "[PIPELINES] Failed to load debug shader code from path: " +
            (getExecutableDirectory() / "Shaders/debug.wgsl").string()
        );
    }

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shaderCode.c_str();

    wgpu::ShaderModuleDescriptor shaderModuleDesc{};
    shaderModuleDesc.nextInChain = &wgsl;

    pipelineWrapper.shaderModule = wgpuBundle.GetDevice().CreateShaderModule(&shaderModuleDesc);
    if (!pipelineWrapper.shaderModule)
    {
        throw std::runtime_error("[PIPELINES] Failed to create debug shader module.");
    }

    wgpu::ColorTargetState colorTargetState{};
    colorTargetState.format = wgpuBundle.GetSwapchainFormat();

    wgpu::FragmentState fragmentState{};
    fragmentState.module = pipelineWrapper.shaderModule;
    fragmentState.entryPoint = "f";
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    wgpu::VertexState vertexState{};
    vertexState.module = pipelineWrapper.shaderModule;
    vertexState.entryPoint = "v";

    wgpu::RenderPipelineDescriptor pipelineDesc{};
    pipelineDesc.vertex = vertexState;
    pipelineDesc.fragment = &fragmentState;

    pipelineWrapper.pipeline = wgpuBundle.GetDevice().CreateRenderPipeline(&pipelineDesc);
    if (!pipelineWrapper.pipeline)
    {
        throw std::runtime_error("[PIPELINES] Failed to create debug render pipeline.");
    }

    pipelineWrapper.init = 1;
    pipelineWrapper.AssertConsistent();
}

//================================//
void CreateComputeVoxelPipeline(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper)
{
    pipelineWrapper.isCompute = true;

    // WE SHOULD ENFORCE voxelResolution % 8 == 0
    assert(MAXIMUM_VOXEL_RESOLUTION % 8 == 0);
    assert(MAXIMUM_VOXEL_RESOLUTION % 64 == 0);

    // Number of textures: 1 (the output voxel texture)
    pipelineWrapper.textureSizes.resize(1);
    
    // Number of Uniforms: 1 (voxel parameters)
    pipelineWrapper.uniformSizes.resize(1);
    pipelineWrapper.uniformSizes[0] = AlignUp(COMPUTE_VOXEL_UNIFORM_SIZE, 256); // Uniform buffer size must be 256-byte aligned

    // SHADER 
    std::string shaderCode;
    if (getShaderCodeFromFile("Shaders/computeVoxel.wgsl", shaderCode) < 0)
    {
        throw std::runtime_error(
            "[PIPELINES] Failed to load compute voxel shader code from path: " +
            (getExecutableDirectory() / "Shaders/computeVoxel.wgsl").string()
        );
    }

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shaderCode.c_str();

    wgpu::ShaderModuleDescriptor shaderModuleDesc{};
    shaderModuleDesc.nextInChain = &wgsl;

    pipelineWrapper.shaderModule = wgpuBundle.GetDevice().CreateShaderModule(&shaderModuleDesc);

    // Textures
    pipelineWrapper.associatedTextures.resize(1);
    pipelineWrapper.associatedTextureViews.resize(1);
    pipelineWrapper.associatedUniforms.resize(1);

    // Uniforms
    wgpu::BufferDescriptor uniformBufferDesc{};
    uniformBufferDesc.size = pipelineWrapper.uniformSizes[0];
    uniformBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniformBufferDesc.mappedAtCreation = false;
    pipelineWrapper.associatedUniforms[0] = wgpuBundle.GetDevice().CreateBuffer(&uniformBufferDesc);

    // Bind Group Layout
    wgpu::BindGroupLayoutEntry entries[6]{};

    // output texture
    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Compute;
    entries[0].storageTexture.access = wgpu::StorageTextureAccess::WriteOnly;
    entries[0].storageTexture.format = wgpu::TextureFormat::RGBA8Unorm;
    entries[0].storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;

    // voxel parameters uniform
    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Compute;
    entries[1].buffer.type = wgpu::BufferBindingType::Uniform;
    entries[1].buffer.minBindingSize = pipelineWrapper.uniformSizes[0];
    entries[1].buffer.hasDynamicOffset = false;

    // BrickGrid
    entries[2].binding = 2;
    entries[2].visibility = wgpu::ShaderStage::Compute;
    entries[2].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    // BrickPool
    entries[3].binding = 3;
    entries[3].visibility = wgpu::ShaderStage::Compute;
    entries[3].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    // ColorPool
    entries[4].binding = 4;
    entries[4].visibility = wgpu::ShaderStage::Compute;
    entries[4].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    // Feedback Buffer
    entries[5].binding = 5;
    entries[5].visibility = wgpu::ShaderStage::Compute;
    entries[5].buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{};
    bindGroupLayoutDesc.entryCount = 6;
    bindGroupLayoutDesc.entries = entries;
    pipelineWrapper.bindGroupLayout = wgpuBundle.GetDevice().CreateBindGroupLayout(&bindGroupLayoutDesc);

    // Pipeline Layout
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &pipelineWrapper.bindGroupLayout;
    pipelineWrapper.pipelineLayout = wgpuBundle.GetDevice().CreatePipelineLayout(&pipelineLayoutDesc);

    // Compute Pipeline
    wgpu::ComputePipelineDescriptor computePipelineDesc{};
    computePipelineDesc.layout = pipelineWrapper.pipelineLayout;
    computePipelineDesc.compute.module = pipelineWrapper.shaderModule;
    computePipelineDesc.compute.entryPoint = "c";
    pipelineWrapper.computePipeline = wgpuBundle.GetDevice().CreateComputePipeline(&computePipelineDesc);

    pipelineWrapper.init = 1;
    pipelineWrapper.AssertConsistent();
}

//================================//
void CreateComputeUploadVoxelPipeline(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper)
{
    pipelineWrapper.isCompute = true;

    // SHADER 
    std::string shaderCode;
    if (getShaderCodeFromFile("Shaders/computeUploadVoxel.wgsl", shaderCode) < 0)
    {
        throw std::runtime_error(
            "[PIPELINES] Failed to load compute upload voxel shader code from path: " +
            (getExecutableDirectory() / "Shaders/computeUploadVoxel.wgsl").string()
        );
    }

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shaderCode.c_str();

    wgpu::ShaderModuleDescriptor shaderModuleDesc{};
    shaderModuleDesc.nextInChain = &wgsl;

    pipelineWrapper.shaderModule = wgpuBundle.GetDevice().CreateShaderModule(&shaderModuleDesc);

    // Bind Group Layout
    wgpu::BindGroupLayoutEntry entries[4]{};

    // Read upload buffer
    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Compute;
    entries[0].buffer.type = wgpu::BufferBindingType::ReadOnlyStorage;

    // BrickPool, colorPool, BrickGrid
    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Compute;
    entries[1].buffer.type = wgpu::BufferBindingType::Storage;

    entries[2].binding = 2;
    entries[2].visibility = wgpu::ShaderStage::Compute;
    entries[2].buffer.type = wgpu::BufferBindingType::Storage;

    entries[3].binding = 3;
    entries[3].visibility = wgpu::ShaderStage::Compute;
    entries[3].buffer.type = wgpu::BufferBindingType::Storage;

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{};
    bindGroupLayoutDesc.entryCount = 4;
    bindGroupLayoutDesc.entries = entries;
    pipelineWrapper.bindGroupLayout = wgpuBundle.GetDevice().CreateBindGroupLayout(&bindGroupLayoutDesc);

    // Pipeline Layout
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &pipelineWrapper.bindGroupLayout;
    pipelineWrapper.pipelineLayout = wgpuBundle.GetDevice().CreatePipelineLayout(&pipelineLayoutDesc);

    // Compute Pipeline
    wgpu::ComputePipelineDescriptor computePipelineDesc{};
    computePipelineDesc.layout = pipelineWrapper.pipelineLayout;
    computePipelineDesc.compute.module = pipelineWrapper.shaderModule;
    computePipelineDesc.compute.entryPoint = "c";
    pipelineWrapper.computePipeline = wgpuBundle.GetDevice().CreateComputePipeline(&computePipelineDesc);

    pipelineWrapper.init = 1;
    pipelineWrapper.AssertConsistent();
}

//================================//
void CreateBlitVoxelPipeline(WgpuBundle& wgpuBundle, RenderPipelineWrapper& pipelineWrapper)
{
    // Code in ../../../src/Shaders/blit.wgsl
    std::string shaderCode;
    if (getShaderCodeFromFile("Shaders/blit.wgsl", shaderCode) < 0)
    {
        throw std::runtime_error(
            "[PIPELINES] Failed to load blit shader code from path: " +
            (getExecutableDirectory() / "Shaders/blit.wgsl").string()
        );
    }

    wgpu::ShaderSourceWGSL wgsl{};
    wgsl.code = shaderCode.c_str();

    wgpu::ShaderModuleDescriptor shaderModuleDesc{};
    shaderModuleDesc.nextInChain = &wgsl;

    pipelineWrapper.shaderModule = wgpuBundle.GetDevice().CreateShaderModule(&shaderModuleDesc);
    if (!pipelineWrapper.shaderModule)
    {
        throw std::runtime_error("[PIPELINES] Failed to create blit shader module.");
    }

    // Bind Group Layout
    wgpu::BindGroupLayoutEntry entries[2]{};

        // blit texture
    entries[0].binding = 0;
    entries[0].visibility = wgpu::ShaderStage::Fragment;
    entries[0].texture.sampleType = wgpu::TextureSampleType::Float;
    entries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    entries[0].texture.multisampled = false;

        // blit sampler
    entries[1].binding = 1;
    entries[1].visibility = wgpu::ShaderStage::Fragment;
    entries[1].sampler.type = wgpu::SamplerBindingType::Filtering;

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{};
    bindGroupLayoutDesc.entryCount = 2;
    bindGroupLayoutDesc.entries = entries;
    pipelineWrapper.bindGroupLayout = wgpuBundle.GetDevice().CreateBindGroupLayout(&bindGroupLayoutDesc);

    // Pipeline Layout
    wgpu::PipelineLayoutDescriptor pipelineLayoutDesc{};
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &pipelineWrapper.bindGroupLayout;
    pipelineWrapper.pipelineLayout = wgpuBundle.GetDevice().CreatePipelineLayout(&pipelineLayoutDesc);

    // Pipeline
    wgpu::RenderPipelineDescriptor rpDesc{};
    rpDesc.layout = pipelineWrapper.pipelineLayout;

    rpDesc.vertex.module = pipelineWrapper.shaderModule;
    rpDesc.vertex.entryPoint = "v";

    wgpu::FragmentState fragmentState{};
    fragmentState.module = pipelineWrapper.shaderModule;
    fragmentState.entryPoint = "f";

    wgpu::ColorTargetState colorTargetState{};
    colorTargetState.format = wgpuBundle.GetSwapchainFormat();
    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTargetState;

    rpDesc.fragment = &fragmentState;
    rpDesc.vertex.bufferCount = 0;
    rpDesc.primitive.topology = wgpu::PrimitiveTopology::TriangleList;
    pipelineWrapper.pipeline = wgpuBundle.GetDevice().CreateRenderPipeline(&rpDesc);

    // Sampler
    wgpu::SamplerDescriptor samplerDesc{};
    samplerDesc.minFilter = wgpu::FilterMode::Nearest;
    samplerDesc.magFilter = wgpu::FilterMode::Nearest;
    samplerDesc.addressModeU = wgpu::AddressMode::ClampToEdge;
    samplerDesc.addressModeV = wgpu::AddressMode::ClampToEdge;
    pipelineWrapper.associatedSamplers.resize(1);
    pipelineWrapper.associatedSamplers[0] = wgpuBundle.GetDevice().CreateSampler(&samplerDesc);

    pipelineWrapper.init = 1;
    pipelineWrapper.AssertConsistent();
}
