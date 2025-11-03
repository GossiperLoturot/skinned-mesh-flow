using UnityEngine;
using UnityEngine.VFX;
using UnityEngine.Rendering;
using UnityEngine.Rendering.RenderGraphModule;
using System.Collections.Generic;

[VFXType(VFXTypeAttribute.Usage.GraphicsBuffer)]
struct StandardVertex
{
    public Vector3 position;
    public Vector3 normal;
    public Vector4 tangent;
}

public class SkinnedMeshFlow : MonoBehaviour
{
    public ComputeShader shader;
    public SkinnedMeshRenderer skinnedMeshRenderer;
    public VisualEffect visualEffect;

    private RenderGraph _renderGraph;
    private SkinnedMeshFlowPass _pass;

    void OnEnable()
    {
        _renderGraph = new RenderGraph("Skinned Mesh Flow Graph");
        _pass = new SkinnedMeshFlowPass(shader, skinnedMeshRenderer, visualEffect);

        RenderPipelineManager.beginCameraRendering += OnBeginCamera;
    }

    void OnDisable()
    {
        RenderPipelineManager.beginCameraRendering -= OnBeginCamera;

        _pass.Dispose();
    }

    private void OnBeginCamera(ScriptableRenderContext ctx, Camera cam)
    {
        var cmd = CommandBufferPool.Get("Skinned Mesh Flow Command");
        var param = new RenderGraphParameters
        {
            scriptableRenderContext = ctx,
            commandBuffer = cmd,
            currentFrameIndex = Time.frameCount,
        };

        try
        {
            _renderGraph.BeginRecording(param);
            _pass.RecordRenderGraph(_renderGraph);
            _renderGraph.EndRecordingAndExecute();
        }
        catch (System.Exception e)
        {
            _renderGraph.Cleanup();
            throw e;
        }

        ctx.ExecuteCommandBuffer(cmd);
        CommandBufferPool.Release(cmd);
        ctx.Submit();
    }
}

class PassData
{
    public ComputeShader shader;

    public int vertexCount;
    public Matrix4x4 adjustMatrix;

    public BufferHandle srcVertexBuffer;
    public BufferHandle dstVertexBuffer;
    public BufferHandle adjacentBuffer;
    public BufferHandle addressBuffer;
    public BufferHandle eigenBuffer;
}

class SkinnedMeshFlowPass : System.IDisposable
{
    private ComputeShader _shader;
    private SkinnedMeshRenderer _skinnedMeshRenderer;
    private VisualEffect _visualEffect;

    private int _vertexCount;

    private GraphicsBuffer _adjacentBuffer;
    private GraphicsBuffer _addressBuffer;
    private GraphicsBuffer _eigenBuffer;

    public SkinnedMeshFlowPass(ComputeShader shader, SkinnedMeshRenderer skinnedMeshRenderer, VisualEffect visualEffect)
    {
        _shader = shader;
        _skinnedMeshRenderer = skinnedMeshRenderer;
        _visualEffect = visualEffect;

        _skinnedMeshRenderer.sharedMesh.vertexBufferTarget |= GraphicsBuffer.Target.Structured;
        _skinnedMeshRenderer.vertexBufferTarget |= GraphicsBuffer.Target.Structured;

        _vertexCount = _skinnedMeshRenderer.sharedMesh.vertexCount;

        // build edge
        var edges = new HashSet<(int, int)>();
        var indices = new List<int>();
        for (int i = 0; i < _skinnedMeshRenderer.sharedMesh.subMeshCount; i++)
        {
            _skinnedMeshRenderer.sharedMesh.GetTriangles(indices, i);
            for (int j = 0; j < indices.Count; j += 3)
            {
                edges.Add(CreateEdge(indices[j + 0], indices[j + 1]));
                edges.Add(CreateEdge(indices[j + 1], indices[j + 2]));
                edges.Add(CreateEdge(indices[j + 2], indices[j + 0]));
            }
            indices.Clear();
        }
        // build ajdacent list
        var adjacentLists = new List<int>[_vertexCount];
        for (int i = 0; i < _vertexCount; i++) adjacentLists[i] = new List<int>();
        foreach (var (idx0, idx1) in edges)
        {
            adjacentLists[idx0].Add(idx1);
            adjacentLists[idx1].Add(idx0);
        }
        // build buffer data
        var adjacents = new List<int>();
        var neighbors = new List<int>();
        for (int i = 0; i < _vertexCount; i++)
        {
            var startIdx = adjacents.Count;
            adjacents.AddRange(adjacentLists[i]);
            var endIdx = adjacents.Count;

            neighbors.Add(startIdx);
            neighbors.Add(endIdx);
        }

        var adjacentCount = adjacents.Count;
        _adjacentBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, adjacentCount, 4);
        _adjacentBuffer.SetData(adjacents);

        _addressBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4 * 2);
        _addressBuffer.SetData(neighbors);

        _eigenBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4 * 4);
    }

    public void RecordRenderGraph(RenderGraph renderGraph)
    {
        var srcVertexBuffer = _skinnedMeshRenderer.sharedMesh.GetVertexBuffer(0);
        var dstVertexBuffer = _skinnedMeshRenderer.GetVertexBuffer();

        if (dstVertexBuffer == null) return;

        var adjustMatrix = _skinnedMeshRenderer.worldToLocalMatrix * _skinnedMeshRenderer.rootBone.localToWorldMatrix;

        using (var builder = renderGraph.AddComputePass<PassData>("Skinned Mesh Flow Pass", out var passData))
        {
            passData.shader = _shader;

            passData.vertexCount = _vertexCount;
            passData.adjustMatrix = adjustMatrix;

            passData.srcVertexBuffer = renderGraph.ImportBuffer(srcVertexBuffer);
            passData.dstVertexBuffer = renderGraph.ImportBuffer(dstVertexBuffer);
            passData.adjacentBuffer = renderGraph.ImportBuffer(_adjacentBuffer);
            passData.addressBuffer = renderGraph.ImportBuffer(_addressBuffer);
            passData.eigenBuffer = renderGraph.ImportBuffer(_eigenBuffer);

            builder.SetRenderFunc(static (PassData passData, ComputeGraphContext ctx) =>
            {
                // compute flow
                var kernelId = passData.shader.FindKernel("CSMain");
                var groupX = Mathf.CeilToInt(passData.vertexCount / 256.0f);
                ctx.cmd.SetComputeIntParam(passData.shader, "VertexCount", passData.vertexCount);
                ctx.cmd.SetComputeMatrixParam(passData.shader, "AdjustMatrix", passData.adjustMatrix);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "SrcVertexBuffer", passData.srcVertexBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "DstVertexBuffer", passData.dstVertexBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "AdjacentBuffer", passData.adjacentBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "AddressBuffer", passData.addressBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "EigenBuffer", passData.eigenBuffer);
                ctx.cmd.DispatchCompute(passData.shader, kernelId, groupX, 1, 1);
            });
        }

        _visualEffect.SetInt("VertexCount", _vertexCount);
        _visualEffect.SetSkinnedMeshRenderer("SkinnedMeshRenderer", _skinnedMeshRenderer);
        _visualEffect.SetGraphicsBuffer("EigenBuffer", _eigenBuffer);
        _visualEffect.Play();
    }

    public void Dispose()
    {
        _adjacentBuffer.Dispose();
        _addressBuffer.Dispose();
        _eigenBuffer.Dispose();
    }

    private (int, int) CreateEdge(int index0, int index1)
    {
        return index0 < index1 ? (index0, index1) : (index1, index0);
    }
}