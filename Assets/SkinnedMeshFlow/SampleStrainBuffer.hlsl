#ifndef SAMPLE_STRAIN_BUFFER_HLSL
#define SAMPLE_STRAIN_BUFFER_HLSL

StructuredBuffer<float2x3> StrainBuffer;

void SampleStrainBuffer_float(float id, out float3 compression, out float3 tension)
{
    compression = StrainBuffer[id][0];
    tension = StrainBuffer[id][1];
}

#endif // SAMPLE_STRAIN_BUFFER_HLSL