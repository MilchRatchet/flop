// Reduction kernel to summarize stats in a histogram

struct PushConstants
{
    uint2 extent;
    uint input;
    uint output;
};
[[vk::push_constant]]
PushConstants constants;

[[vk::binding(1)]]
RWTexture2D<float4> rwtextures[];

[[vk::binding(2)]]
RWByteAddressBuffer rwbuffers[];

// Construct an LDS histogram with 64 entries
#define BUCKET_COUNT 64

[numthreads(8, 8, 1)]
void CSMain(uint3 id : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
    uint linear_gtid = gtid.x * 8 + gtid.y;
    if (linear_gtid < BUCKET_COUNT)
    {
        rwbuffers[constants.output].Store(linear_gtid * 4, 0);
    }
}
