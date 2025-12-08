struct VSOut 
{
    @builtin(position) position : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

@vertex
fn v(@builtin(vertex_index) idx : u32) -> VSOut 
{
    // Fullscreen triangle (no vertex buffer)
    var positions = array<vec2<f32>, 3>(
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0),
    );

    var uvs = array<vec2<f32>, 3>(
        vec2(0.0, 0.0),
        vec2(2.0, 0.0),
        vec2(0.0, 2.0),
    );

    var out : VSOut;
    out.position = vec4<f32>(positions[idx], 0.0, 1.0);
    out.uv = uvs[idx];
    return out;
}

@group(0) @binding(0)
var blitTexture : texture_2d<f32>;
@group(0) @binding(1)
var blitSampler : sampler;

@fragment
fn f(in : VSOut) -> @location(0) vec4<f32> 
{
    return textureSample(blitTexture, blitSampler, in.uv);
}
