#ifndef SAMPLE_STRAIN_BUFFER_HLSL
#define SAMPLE_STRAIN_BUFFER_HLSL

StructuredBuffer<float3> StrainBuffer;

void SampleStrainBuffer_float(float id, out float3 strain)
{
    strain = StrainBuffer[id];
}

#endif // SAMPLE_STRAIN_BUFFER_HLSL