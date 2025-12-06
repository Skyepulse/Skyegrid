@vertex
fn v(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> 
{
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>(0.5, -0.5)
    );
    return vec4<f32>(pos[i], 0.0, 1.0);
}

@fragment 
fn f() -> @location(0) vec4<f32> 
{
    return vec4<f32>(0.0, 0.5, 0.5, 1.0);
}