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
    public int edgeCount;
    public Matrix4x4 adjustMatrix;

    public BufferHandle srcVertexBuffer;
    public BufferHandle dstVertexBuffer;
    public BufferHandle edgeBuffer;
    public BufferHandle degreeBuffer;

    public BufferHandle fixedBuffer;
    public BufferHandle floatBuffer;
}

class SkinnedMeshFlowPass : System.IDisposable
{
    private ComputeShader _shader;
    private SkinnedMeshRenderer _skinnedMeshRenderer;
    private VisualEffect _visualEffect;

    private int _vertexCount;
    private int _edgeCount;

    private GraphicsBuffer _edgeBuffer;
    private GraphicsBuffer _degreeBuffer;

    private GraphicsBuffer _fixedBuffer;
    private GraphicsBuffer _floatBuffer;

    public SkinnedMeshFlowPass(ComputeShader shader, SkinnedMeshRenderer skinnedMeshRenderer, VisualEffect visualEffect)
    {
        _shader = shader;
        _skinnedMeshRenderer = skinnedMeshRenderer;
        _visualEffect = visualEffect;

        _skinnedMeshRenderer.sharedMesh.vertexBufferTarget |= GraphicsBuffer.Target.Structured;
        _skinnedMeshRenderer.vertexBufferTarget |= GraphicsBuffer.Target.Structured;

        _vertexCount = _skinnedMeshRenderer.sharedMesh.vertexCount;

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

        var edgeData = new List<int>();
        var degreeData = new int[_vertexCount];
        foreach (var (index0, index1) in edges)
        {
            edgeData.Add(index0);
            edgeData.Add(index1);
            degreeData[index0] ++;
            degreeData[index1] ++;
        }

        _edgeCount = edges.Count;
        _edgeBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _edgeCount, 8);
        _edgeBuffer.SetData(edgeData);

        _degreeBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4);
        _degreeBuffer.SetData(degreeData);

        _fixedBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4);
        _floatBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4);
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
            passData.edgeCount = _edgeCount;
            passData.adjustMatrix = adjustMatrix;

            passData.srcVertexBuffer = renderGraph.ImportBuffer(srcVertexBuffer);
            passData.dstVertexBuffer = renderGraph.ImportBuffer(dstVertexBuffer);
            passData.edgeBuffer = renderGraph.ImportBuffer(_edgeBuffer);
            passData.degreeBuffer = renderGraph.ImportBuffer(_degreeBuffer);

            passData.fixedBuffer = renderGraph.ImportBuffer(_fixedBuffer);
            passData.floatBuffer = renderGraph.ImportBuffer(_floatBuffer);

            builder.SetRenderFunc(static (PassData passData, ComputeGraphContext ctx) =>
            {
                int kernelId;
                int groupX;

                ctx.cmd.SetComputeIntParam(passData.shader, "VertexCount", passData.vertexCount);
                ctx.cmd.SetComputeIntParam(passData.shader, "EdgeCount", passData.edgeCount);
                ctx.cmd.SetComputeMatrixParam(passData.shader, "AdjustMatrix", passData.adjustMatrix);

                // initialize
                kernelId = passData.shader.FindKernel("CSInit");
                groupX = passData.vertexCount / 1024 + 1;
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "FixedBuffer", passData.fixedBuffer);
                ctx.cmd.DispatchCompute(passData.shader, kernelId, groupX, 1, 1);

                // compute flow
                kernelId = passData.shader.FindKernel("CSMain");
                groupX = passData.edgeCount / 1024 + 1;
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "SrcVertexBuffer", passData.srcVertexBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "DstVertexBuffer", passData.dstVertexBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "EdgeBuffer", passData.edgeBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "DegreeBuffer", passData.degreeBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "FixedBuffer", passData.fixedBuffer);
                ctx.cmd.DispatchCompute(passData.shader, kernelId, groupX, 1, 1);

                // post
                kernelId = passData.shader.FindKernel("CSPost");
                groupX = passData.vertexCount / 1024 + 1;
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "FixedBuffer", passData.fixedBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "FloatBuffer", passData.floatBuffer);
                ctx.cmd.DispatchCompute(passData.shader, kernelId, groupX, 1, 1);
            });
        }

        _visualEffect.SetInt("VertexCount", _vertexCount);
        _visualEffect.SetSkinnedMeshRenderer("SkinnedMeshRenderer", _skinnedMeshRenderer);
        _visualEffect.SetGraphicsBuffer("ElasticityBuffer", _floatBuffer);
        _visualEffect.Play();
    }

    public void Dispose()
    {
        _edgeBuffer.Dispose();
        _degreeBuffer.Dispose();

        _fixedBuffer.Dispose();
        _floatBuffer.Dispose();
    }

    private (int, int) CreateEdge(int index0, int index1)
    {
        return index0 < index1 ? (index0, index1) : (index1, index0);
    }
}