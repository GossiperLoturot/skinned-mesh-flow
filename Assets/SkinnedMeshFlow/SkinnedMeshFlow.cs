using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using UnityEngine.Rendering.Universal;
using UnityEngine.VFX;
using System.Collections.Generic;

public class SkinnedMeshFlow : MonoBehaviour
{
    public ComputeShader shader;
    public SkinnedMeshRenderer skinnedMeshRenderer;
    public VisualEffect visudalEffect;

    private SkinnedMeshFlowPass _pass;

    void OnEnable()
    {
        _pass = new SkinnedMeshFlowPass(shader, skinnedMeshRenderer, visudalEffect);

        RenderPipelineManager.beginCameraRendering += OnBeginCamera;
    }

    void OnDisable()
    {
        RenderPipelineManager.beginCameraRendering -= OnBeginCamera;

        _pass.Dispose();
    }

    private void OnBeginCamera(ScriptableRenderContext ctx, Camera cam)
    {
        cam.GetUniversalAdditionalCameraData().scriptableRenderer.EnqueuePass(_pass);
    }
}

class PassData
{
    public ComputeShader shader;

    public int vertexCount;

    public BufferHandle srcVertexBuffer;
    public BufferHandle dstVertexBuffer;
    public BufferHandle adjacentBuffer;
    public BufferHandle addressBuffer;
    public BufferHandle strainBuffer;

    public GraphicsBuffer srcVertexBufferResource;
    public GraphicsBuffer dstVertexBufferResource;
}

class SkinnedMeshFlowPass : ScriptableRenderPass
{
    private ComputeShader _shader;
    private SkinnedMeshRenderer _skinnedMeshRenderer;
    private VisualEffect _visudalEffect;

    private int _vertexCount;

    private GraphicsBuffer _adjacentBuffer;
    private GraphicsBuffer _addressBuffer;
    private GraphicsBuffer _strainBuffer;

    private MaterialPropertyBlock _propertyBlock;

    public SkinnedMeshFlowPass(ComputeShader shader, SkinnedMeshRenderer skinnedMeshRenderer, VisualEffect visudalEffect)
    {
        if (shader == null) throw new System.ArgumentNullException(nameof(shader));
        if (skinnedMeshRenderer == null) throw new System.ArgumentNullException(nameof(skinnedMeshRenderer));
        if (visudalEffect == null) throw new System.ArgumentNullException(nameof(visudalEffect));

        _shader = shader;
        _skinnedMeshRenderer = skinnedMeshRenderer;
        _visudalEffect = visudalEffect;

        _skinnedMeshRenderer.sharedMesh.vertexBufferTarget |= GraphicsBuffer.Target.Structured;
        _skinnedMeshRenderer.vertexBufferTarget |= GraphicsBuffer.Target.Structured;

        _vertexCount = _skinnedMeshRenderer.sharedMesh.vertexCount;

        // build ajdacent list
        var adjacentLists = new List<(int, int)>[_vertexCount];
        for (int i = 0; i < _vertexCount; i++) adjacentLists[i] = new List<(int, int)>();
        var indices = new List<int>();
        for (int i = 0; i < _skinnedMeshRenderer.sharedMesh.subMeshCount; i++)
        {
            _skinnedMeshRenderer.sharedMesh.GetTriangles(indices, i);
            for (int j = 0; j < indices.Count; j += 3)
            {
                var idx0 = indices[j + 0];
                var idx1 = indices[j + 1];
                var idx2 = indices[j + 2];
                adjacentLists[idx0].Add((idx1, idx2));
                adjacentLists[idx1].Add((idx2, idx0));
                adjacentLists[idx2].Add((idx0, idx1));
            }
            indices.Clear();
        }
        // build buffer data
        var adjacents = new List<int>();
        var neighbors = new List<int>();
        for (int i = 0; i < _vertexCount; i++)
        {
            var startIdx = adjacents.Count;
            foreach (var (idx0, idx1) in adjacentLists[i])
            {
                adjacents.Add(idx0);
                adjacents.Add(idx1);
            }
            var endIdx = adjacents.Count;
            neighbors.Add(startIdx / 2);
            neighbors.Add(endIdx / 2);
        }

        var adjacentCount = adjacents.Count;
        _adjacentBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, adjacentCount, 4 * 2);
        _adjacentBuffer.SetData(adjacents);

        _addressBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4 * 2);
        _addressBuffer.SetData(neighbors);

        _strainBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4 * 3);

        _propertyBlock = new MaterialPropertyBlock();
    }

    public override void RecordRenderGraph(RenderGraph renderGraph, ContextContainer frameData)
    {
        var srcVertexBuffer = _skinnedMeshRenderer.sharedMesh.GetVertexBuffer(0);
        var dstVertexBuffer = _skinnedMeshRenderer.GetVertexBuffer();

        if (srcVertexBuffer == null || dstVertexBuffer == null) return;
        Debug.Log("SkinnedMeshFlowPass: RecordRenderGraph");

        using (var builder = renderGraph.AddComputePass<PassData>("Skinned Mesh Flow Pass", out var passData))
        {
            passData.shader = _shader;

            passData.vertexCount = _vertexCount;

            passData.srcVertexBuffer = renderGraph.ImportBuffer(srcVertexBuffer);
            passData.dstVertexBuffer = renderGraph.ImportBuffer(dstVertexBuffer);
            passData.adjacentBuffer = renderGraph.ImportBuffer(_adjacentBuffer);
            passData.addressBuffer = renderGraph.ImportBuffer(_addressBuffer);
            passData.strainBuffer = renderGraph.ImportBuffer(_strainBuffer);
            passData.srcVertexBufferResource = srcVertexBuffer;
            passData.dstVertexBufferResource = dstVertexBuffer;

            builder.UseBuffer(passData.srcVertexBuffer, AccessFlags.Read);
            builder.UseBuffer(passData.dstVertexBuffer, AccessFlags.Read);
            builder.UseBuffer(passData.adjacentBuffer, AccessFlags.Read);
            builder.UseBuffer(passData.addressBuffer, AccessFlags.Read);
            builder.UseBuffer(passData.strainBuffer, AccessFlags.Write);

            builder.SetRenderFunc(static (PassData passData, ComputeGraphContext ctx) =>
            {
                // compute flow
                var kernelId = passData.shader.FindKernel("CSMain");
                var groupX = Mathf.CeilToInt(passData.vertexCount / 256.0f);
                ctx.cmd.SetComputeIntParam(passData.shader, "VertexCount", passData.vertexCount);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "SrcVertexBuffer", passData.srcVertexBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "DstVertexBuffer", passData.dstVertexBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "AdjacentBuffer", passData.adjacentBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "AddressBuffer", passData.addressBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "StrainBuffer", passData.strainBuffer);
                ctx.cmd.DispatchCompute(passData.shader, kernelId, groupX, 1, 1);

                passData.srcVertexBufferResource.Dispose();
                passData.dstVertexBufferResource.Dispose();
            });
        }

        _skinnedMeshRenderer.GetPropertyBlock(_propertyBlock);
        _propertyBlock.SetBuffer("StrainBuffer", _strainBuffer);
        _skinnedMeshRenderer.SetPropertyBlock(_propertyBlock);

        // DEBUG
        _visudalEffect.SetGraphicsBuffer("StrainBuffer", _strainBuffer);
    }

    public void Dispose()
    {
        _adjacentBuffer.Dispose();
        _addressBuffer.Dispose();
        _strainBuffer.Dispose();
    }
}