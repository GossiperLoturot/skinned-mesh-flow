#ifndef SAMPLE_BUFFER_HLSL
#define SAMPLE_BUFFER_HLSL

StructuredBuffer<float4> EigenBuffer;

void SampleBuffer_float(float id, out float4 value)
{
    value = EigenBuffer[id];
}

#endif // SAMPLE_BUFFER_HLSL