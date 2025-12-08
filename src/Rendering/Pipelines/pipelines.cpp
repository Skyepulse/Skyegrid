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
    InitComputeVoxelPipelineResources(pipelineWrapper, MAXIMUM_VOXEL_RESOLUTION, COMPUTE_VOXEL_UNIFORM_SIZE);

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
    pipelineWrapper.associatedTextures.resize(2);
    pipelineWrapper.associatedTextureViews.resize(2);
    pipelineWrapper.associatedUniforms.resize(1);

    wgpu::TextureDescriptor textureDescriptor{};
    wgpu::TextureViewDescriptor viewDescriptor{};

    textureDescriptor.dimension = wgpu::TextureDimension::e2D;
    textureDescriptor.size = { MAXIMUM_WINDOW_WIDTH, MAXIMUM_WINDOW_HEIGHT, 1 };
    textureDescriptor.sampleCount = 1;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.format = wgpu::TextureFormat::RGBA8Unorm;
    textureDescriptor.usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::CopySrc | wgpu::TextureUsage::TextureBinding;
    pipelineWrapper.associatedTextures[0] = wgpuBundle.GetDevice().CreateTexture(&textureDescriptor);

    viewDescriptor.dimension = wgpu::TextureViewDimension::e2D;
    viewDescriptor.format = textureDescriptor.format;
    pipelineWrapper.associatedTextureViews[0] = pipelineWrapper.associatedTextures[0].CreateView(&viewDescriptor);

    textureDescriptor.dimension = wgpu::TextureDimension::e3D;
    textureDescriptor.size = { MAXIMUM_VOXEL_RESOLUTION / 4, MAXIMUM_VOXEL_RESOLUTION / 4, MAXIMUM_VOXEL_RESOLUTION / 8 };
    textureDescriptor.sampleCount = 1;
    textureDescriptor.mipLevelCount = 1;
    textureDescriptor.format = wgpu::TextureFormat::RGBA32Uint;
    textureDescriptor.usage = wgpu::TextureUsage::StorageBinding | wgpu::TextureUsage::CopyDst;
    pipelineWrapper.associatedTextures[1] = wgpuBundle.GetDevice().CreateTexture(&textureDescriptor);

    viewDescriptor.dimension = wgpu::TextureViewDimension::e3D;
    viewDescriptor.format = textureDescriptor.format;
    pipelineWrapper.associatedTextureViews[1] = pipelineWrapper.associatedTextures[1].CreateView(&viewDescriptor);

    // Uniforms
    wgpu::BufferDescriptor uniformBufferDesc{};
    uniformBufferDesc.size = pipelineWrapper.uniformSizes[0];
    uniformBufferDesc.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    uniformBufferDesc.mappedAtCreation = false;
    pipelineWrapper.associatedUniforms[0] = wgpuBundle.GetDevice().CreateBuffer(&uniformBufferDesc);

    // Bind Group Layout
    wgpu::BindGroupLayoutEntry textureBindingEntries[3]{};

        // output texture
    textureBindingEntries[0].binding = 0;
    textureBindingEntries[0].visibility = wgpu::ShaderStage::Compute;
    textureBindingEntries[0].storageTexture.access = wgpu::StorageTextureAccess::WriteOnly;
    textureBindingEntries[0].storageTexture.format = wgpu::TextureFormat::RGBA8Unorm;
    textureBindingEntries[0].storageTexture.viewDimension = wgpu::TextureViewDimension::e2D;

        // voxel storage texture
    textureBindingEntries[1].binding = 1;
    textureBindingEntries[1].visibility = wgpu::ShaderStage::Compute;
    textureBindingEntries[1].storageTexture.access = wgpu::StorageTextureAccess::ReadOnly;
    textureBindingEntries[1].storageTexture.format = wgpu::TextureFormat::RGBA32Uint;
    textureBindingEntries[1].storageTexture.viewDimension = wgpu::TextureViewDimension::e3D;

        // voxel parameters uniform
    textureBindingEntries[2].binding = 2;
    textureBindingEntries[2].visibility = wgpu::ShaderStage::Compute;
    textureBindingEntries[2].buffer.type = wgpu::BufferBindingType::Uniform;
    textureBindingEntries[2].buffer.minBindingSize = pipelineWrapper.uniformSizes[0];
    textureBindingEntries[2].buffer.hasDynamicOffset = false;

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{};
    bindGroupLayoutDesc.entryCount = 3;
    bindGroupLayoutDesc.entries = textureBindingEntries;
    pipelineWrapper.bindGroupLayout = wgpuBundle.GetDevice().CreateBindGroupLayout(&bindGroupLayoutDesc);

    // Bind Group
    std::vector<wgpu::BindGroupEntry> bindGroupEntries(3);

    bindGroupEntries[0].binding = 0;
    bindGroupEntries[0].textureView = pipelineWrapper.associatedTextureViews[0];

    bindGroupEntries[1].binding = 1;
    bindGroupEntries[1].textureView = pipelineWrapper.associatedTextureViews[1];

    bindGroupEntries[2].binding = 2;
    bindGroupEntries[2].buffer = pipelineWrapper.associatedUniforms[0];
    bindGroupEntries[2].offset = 0;
    bindGroupEntries[2].size = pipelineWrapper.uniformSizes[0];

    wgpu::BindGroupDescriptor bindGroupDesc{};
    bindGroupDesc.layout = pipelineWrapper.bindGroupLayout;
    bindGroupDesc.entryCount = static_cast<uint32_t>(bindGroupEntries.size());
    bindGroupDesc.entries = bindGroupEntries.data();
    pipelineWrapper.bindGroup = wgpuBundle.GetDevice().CreateBindGroup(&bindGroupDesc);

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
void InitComputeVoxelPipelineResources(RenderPipelineWrapper& pipelineWrapper, size_t voxelResolution, size_t voxelParamSize)
{
    // WE SHOULD ENFORCE voxelResolution % 8 == 0
    assert(voxelResolution % 8 == 0);

    // We initialize here the buffers sizes, in order to have the limits when querying
    // the device capabilities for buffer sizes.

    // Number of textures: 2 (the output voxel texture, the texel voxel storage texture)
    pipelineWrapper.textureSizes.resize(2);
    pipelineWrapper.textureSizes[0] = MAXIMUM_WINDOW_HEIGHT * MAXIMUM_WINDOW_WIDTH * 4; // Output voxel texture, RGBA8
    size_t texelCount = (voxelResolution * voxelResolution * voxelResolution) / (4 * 4 * 8);
    pipelineWrapper.textureSizes[1] = texelCount * 16; // Texel is size of uvec4 = 16 bytes
    
    // Number of Uniforms: 1 (voxel parameters)
    pipelineWrapper.uniformSizes.resize(1);
    pipelineWrapper.uniformSizes[0] = AlignUp(voxelParamSize, 256); // Uniform buffer size must be 256-byte aligned

    // Number of other buffers: 0
    pipelineWrapper.bufferSizes.resize(0);
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
    wgpu::BindGroupLayoutEntry textureBindingEntries[2]{};

        // blit texture
    textureBindingEntries[0].binding = 0;
    textureBindingEntries[0].visibility = wgpu::ShaderStage::Fragment;
    textureBindingEntries[0].texture.sampleType = wgpu::TextureSampleType::Float;
    textureBindingEntries[0].texture.viewDimension = wgpu::TextureViewDimension::e2D;
    textureBindingEntries[0].texture.multisampled = false;

        // blit sampler
    textureBindingEntries[1].binding = 1;
    textureBindingEntries[1].visibility = wgpu::ShaderStage::Fragment;
    textureBindingEntries[1].sampler.type = wgpu::SamplerBindingType::Filtering;

    wgpu::BindGroupLayoutDescriptor bindGroupLayoutDesc{};
    bindGroupLayoutDesc.entryCount = 2;
    bindGroupLayoutDesc.entries = textureBindingEntries;
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
