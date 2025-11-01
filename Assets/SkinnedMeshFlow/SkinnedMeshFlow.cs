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
    public Matrix4x4 matrix;
    public int vertexCount;
    public int edgeCount;
    public BufferHandle originalVertexBuffer;
    public BufferHandle vertexBuffer;
    public BufferHandle vertexAddBuffer;
    public BufferHandle edgeBuffer;
    public BufferHandle countBuffer;
    public BufferHandle output0Buffer;
    public BufferHandle output1Buffer;
}

class SkinnedMeshFlowPass : System.IDisposable
{
    private ComputeShader _shader;
    private SkinnedMeshRenderer _skinnedMeshRenderer;
    private VisualEffect _visualEffect;

    private int _vertexCount;
    private int _edgeCount;
    private GraphicsBuffer _edgeBuffer;
    private GraphicsBuffer _countBuffer;
    private GraphicsBuffer _output0Buffer;
    private GraphicsBuffer _output1Buffer;

    public SkinnedMeshFlowPass(ComputeShader shader, SkinnedMeshRenderer skinnedMeshRenderer, VisualEffect visualEffect)
    {
        _shader = shader;
        _skinnedMeshRenderer = skinnedMeshRenderer;
        _visualEffect = visualEffect;

        _skinnedMeshRenderer.sharedMesh.vertexBufferTarget |= GraphicsBuffer.Target.Structured;
        _skinnedMeshRenderer.vertexBufferTarget |= GraphicsBuffer.Target.Structured;

        _vertexCount = _skinnedMeshRenderer.sharedMesh.vertexCount;

        var edgeSet = new HashSet<(int, int)>();
        var indices = new List<int>();
        var counts = new int[_vertexCount];
        for (int i = 0; i < _skinnedMeshRenderer.sharedMesh.subMeshCount; i++)
        {
            _skinnedMeshRenderer.sharedMesh.GetTriangles(indices, i);
            for (int j = 0; j < indices.Count; j += 3)
            {
                edgeSet.Add((indices[j + 0], indices[j + 1]));
                edgeSet.Add((indices[j + 1], indices[j + 2]));
                edgeSet.Add((indices[j + 2], indices[j + 0]));
            }
            indices.Clear();
        }
        foreach (var (index0, index1) in edgeSet)
        {
            indices.Add(index0);
            indices.Add(index1);
            counts[index0]++;
            counts[index1]++;
        }

        _edgeCount = indices.Count / 2;
        _edgeBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _edgeCount, 8);
        _edgeBuffer.SetData(indices);

        _countBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4);
        _countBuffer.SetData(counts);

        _output0Buffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4);
        _output1Buffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, _vertexCount, 4);
    }

    public void RecordRenderGraph(RenderGraph renderGraph)
    {
        var originalVertexBuffer = _skinnedMeshRenderer.sharedMesh.GetVertexBuffer(0);
        var vertexBuffer = _skinnedMeshRenderer.GetVertexBuffer();

        var matrix = _skinnedMeshRenderer.worldToLocalMatrix * _skinnedMeshRenderer.rootBone.localToWorldMatrix;

        using (var builder = renderGraph.AddComputePass<PassData>("Skinned Mesh Flow Pass", out var passData))
        {
            passData.shader = _shader;
            passData.matrix = matrix;
            passData.vertexCount = _vertexCount;
            passData.edgeCount = _edgeCount;
            passData.originalVertexBuffer = renderGraph.ImportBuffer(originalVertexBuffer);
            passData.vertexBuffer = renderGraph.ImportBuffer(vertexBuffer);
            passData.edgeBuffer = renderGraph.ImportBuffer(_edgeBuffer);
            passData.countBuffer = renderGraph.ImportBuffer(_countBuffer);
            passData.output0Buffer = renderGraph.ImportBuffer(_output0Buffer);
            passData.output1Buffer = renderGraph.ImportBuffer(_output1Buffer);

            builder.SetRenderFunc(static (PassData passData, ComputeGraphContext ctx) =>
            {
                int kernelId;
                int groupX;

                // fill output buffer with zero
                kernelId = passData.shader.FindKernel("CSInit");
                groupX = passData.vertexCount / 1024 + 1;
                ctx.cmd.SetComputeIntParam(passData.shader, "VertexCount", passData.vertexCount);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "Output0Buffer", passData.output0Buffer);
                ctx.cmd.DispatchCompute(passData.shader, kernelId, groupX, 1, 1);

                // compute flow
                kernelId = passData.shader.FindKernel("CSMain");
                groupX = passData.edgeCount / 1024 + 1;
                ctx.cmd.SetComputeMatrixParam(passData.shader, "Matrix", passData.matrix);
                ctx.cmd.SetComputeIntParam(passData.shader, "VertexCount", passData.vertexCount);
                ctx.cmd.SetComputeIntParam(passData.shader, "EdgeCount", passData.edgeCount);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "OriginalVertexBuffer", passData.originalVertexBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "VertexBuffer", passData.vertexBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "EdgeBuffer", passData.edgeBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "CountBuffer", passData.countBuffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "Output0Buffer", passData.output0Buffer);
                ctx.cmd.DispatchCompute(passData.shader, kernelId, groupX, 1, 1);

                // post
                kernelId = passData.shader.FindKernel("CSPost");
                groupX = passData.vertexCount / 1024 + 1;
                ctx.cmd.SetComputeIntParam(passData.shader, "VertexCount", passData.vertexCount);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "Output0Buffer", passData.output0Buffer);
                ctx.cmd.SetComputeBufferParam(passData.shader, kernelId, "Output1Buffer", passData.output1Buffer);
                ctx.cmd.DispatchCompute(passData.shader, kernelId, groupX, 1, 1);
            });
        }

        _visualEffect.SetInt("VertexCount", _vertexCount);
        _visualEffect.SetSkinnedMeshRenderer("SkinnedMeshRenderer", _skinnedMeshRenderer);
        _visualEffect.SetGraphicsBuffer("PositionBuffer", _output1Buffer);
        _visualEffect.Play();
    }

    public void Dispose()
    {
        _edgeBuffer.Dispose();
        _countBuffer.Dispose();
        _output0Buffer.Dispose();
        _output1Buffer.Dispose();
    }
}