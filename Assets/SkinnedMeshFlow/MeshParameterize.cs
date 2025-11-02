using System.Collections.Generic;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Solvers;
using MathNet.Numerics.LinearAlgebra.Single;
using MathNet.Numerics.LinearAlgebra.Single.Solvers;
using UnityEngine;

public class MeshParamertize : MonoBehaviour
{
    public Mesh mesh;

    void Start()
    {
        var mesh = GetComponent<MeshFilter>().mesh;

        var vertices = mesh.vertices;
        var indices = mesh.triangles;
        var normals = mesh.normals;
        var tangents = mesh.tangents;

        var tangents_ = new Vector3[tangents.Length];
        for (var i = 0; i < vertices.Length; i++)
        {
            tangents_[i] = new Vector3(tangents[i].x, tangents[i].y, tangents[i].z);
        }

        var binormal = new Vector3[tangents.Length];
        for (var i = 0; i < vertices.Length; i++)
        {
            binormal[i] = Vector3.Cross(normals[i], tangents_[i]) * tangents[i].w;
        }

        var edges = new SortedSet<(int, int)>();
        for (var i = 0; i < indices.Length; i += 3)
        {
            int v0 = indices[i + 0];
            int v1 = indices[i + 1];
            int v2 = indices[i + 2];

            edges.Add((Mathf.Min(v0, v1), Mathf.Max(v0, v1)));
            edges.Add((Mathf.Min(v1, v2), Mathf.Max(v1, v2)));
            edges.Add((Mathf.Min(v2, v0), Mathf.Max(v2, v0)));
        }

        var seams = new SortedSet<(int, int)>();
        for (var i = 0; i < vertices.Length; i++)
        {
            for (var j = i + 1; j < vertices.Length; j++)
            {
                if (Vector3.Distance(vertices[i], vertices[j]) < 1e-4f)
                {
                    seams.Add((i, j));
                }
            }
        }

        var uCoords = Parametrize(vertices, tangents_, edges, seams, 0.0f);
        var vCoords = Parametrize(vertices, binormal, edges, seams, 0.0f);

        var uvs = new Vector2[vertices.Length];
        for (var i = 0; i < vertices.Length; i++)
        {
            uvs[i] = new Vector2(uCoords[i], vCoords[i]);
        }
        mesh.uv = uvs;
    }

    Vector<float> Parametrize(Vector3[] vertices, Vector3[] tangents, SortedSet<(int, int)> edges, SortedSet<(int, int)> seams, float lambda)
    {
        var n = vertices.Length;
        var matA = SparseMatrix.Create(n, n, 0);
        var vecB = Vector<float>.Build.Dense(n, 0);

        // laplacian term
        foreach (var (i, j) in edges)
        {
            var vi = vertices[i];
            var vj = vertices[j];

            var ti = tangents[i];
            var tj = tangents[j];

            var t = ti + tj;
            if (t.sqrMagnitude > 1e-6f) t = t.normalized;
            else t = ti;

            var proj = Vector3.Dot(vj - vi, t);
            var w = 1.0f / Mathf.Max(1e-6f, Vector3.Distance(vi, vj));

            matA[i, i] += w;
            matA[j, j] += w;
            matA[i, j] -= w;
            matA[j, i] -= w;

            vecB[i] += w * proj;
            vecB[j] -= w * proj;
        }

        // seam soft constraint
        foreach (var (i, j) in seams)
        {
            matA[i, i] += lambda;
            matA[j, j] += lambda;
            matA[i, j] -= lambda;
            matA[j, i] -= lambda;
        }

        // pin first vertex 
        var pinI = 0;
        for (var j = 0; j < n; j++) matA[pinI, j] = 0;
        matA[pinI, pinI] = 1;
        vecB[pinI] = 0;

        // Solve linear system
        var iterationCountStopCriterion = new IterationCountStopCriterion<float>(1000);
        var residualStopCriterion = new ResidualStopCriterion<float>(1e-10);
        var monitor = new Iterator<float>(iterationCountStopCriterion, residualStopCriterion);
        var solver = new GpBiCg();
        var vecX = matA.SolveIterative(vecB, solver, monitor);
        return vecX;
    }
}
