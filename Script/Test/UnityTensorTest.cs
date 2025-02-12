using System;
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityTensor.Network;
using UnityTensor.Utility;
using UT.Core;
using UT.Basic;

namespace UT.Test
{
    
    public static class UnityTensorTest
    {

        private class Test : Attribute
        {
                
        }

        private static string CheckValueSimilar(float f1, float f2)
        {
            if (float.IsNaN(f1)) return "NaN";
            if (float.IsNaN(f2)) return "NaN";
            return Mathf.Abs(f1 - f2) > 0.00001f ? $"Not similar: {f1} {f2} {Mathf.Abs(f1 - f2)}" : "";
        }

        private static bool CheckValueSimilar(IReadOnlyList<float> f1, params float[] f2)
        {
            if (f1.Count != f2.Length)
            {
                Debug.LogError("Length don't match");
                return false;
            }
            for (var i = 0; i < f1.Count; i++)
            {
                var error = CheckValueSimilar(f1[i], f2[i]);
                if (string.IsNullOrEmpty(error)) continue;
                Debug.LogError(error);
                return false;
            }
            return true;
        }

        private static void CheckValueSimilar(Tensor t1, MessagePacker packer)
        {
            var data = packer.UnpackArray(t1.Shape.FlattenSize);
            if (!CheckValueSimilar(t1.GetData<float>(), data))
            {
                t1.Print<float>();
                Debug.Log(string.Join(',', data));
            }
        }

#if UNITY_EDITOR
        [MenuItem("UnityTensor/TestAll")]
#endif
        public static void TestAll()
        {
            var p = new object[] { };
            foreach (var m in typeof(UnityTensorTest).GetMethods())
            {
                var isTest = false;
                foreach (var o in m.GetCustomAttributes(false))
                {
                    if (o is Test)
                    {
                        isTest = true;
                        break;
                    }
                }
                if (isTest) m.Invoke(null, p);
            }
        }

        [Test]
        public static void Sum()
        {
            var packer = new MessagePacker(new MemoryStream(Resources.Load<TextAsset>("UT/Test/Sum").bytes));
            
            var a = new Tensor(new TensorShape(2, 3, 4, 5), true);
            var b = a.Sum(1, 3);
            var t = new Tensor(new TensorShape(2, 1, 4, 1));
            var g = Op.MseLoss(b, t);
            using var graph = TensorGraph.BuildGraph(true);
            
            a.SetData(packer.UnpackArray(a.Shape.FlattenSize));
            t.SetData(packer.UnpackArray(t.Shape.FlattenSize));
            graph.Forward();
            graph.Backward();
            CheckValueSimilar(b, packer);
            CheckValueSimilar(a.Gradient, packer);
            
            Debug.Log("Sum test passed");
        }

        [Test]
        public static void MseLoss()
        {
            var packer = new MessagePacker(new MemoryStream(Resources.Load<TextAsset>("UT/Test/MseLoss").bytes));

            var a = new Tensor(new TensorShape(2, 4, 1), true);
            var at = a.Transpose(1, 2);
            var b = new Tensor(new TensorShape(2, 5, 4), true);
            var c = at + b;
            var t = new Tensor(new TensorShape(2, 5, 4));
            var g = Op.MseLoss(c, t);
            using var graph = TensorGraph.BuildGraph(true);
            
            a.SetData(packer.UnpackArray(a.Shape.FlattenSize));
            b.SetData(packer.UnpackArray(b.Shape.FlattenSize));
            t.SetData(packer.UnpackArray(t.Shape.FlattenSize));
            
            graph.Forward();
            graph.Backward();
            CheckValueSimilar(a.Gradient, packer);
            CheckValueSimilar(b.Gradient, packer);
            
            Debug.Log("MseLoss test passed");
        }

        [Test]
        public static void Adam()
        {
            var packer = new MessagePacker(new MemoryStream(Resources.Load<TextAsset>("UT/Test/Adam").bytes));
            
            var a = new Tensor(new TensorShape(16, 3, 2), true);
            var at = a.Transpose(1, 2);
            var b = new Tensor(new TensorShape(2, 3), true);
            var c = at + b;
            var t = new Tensor(new TensorShape(16, 2, 3));
            var g = Op.MseLoss(c, t);
            using var graph = TensorGraph.BuildGraph(true);
            
            a.SetData(packer.UnpackArray(a.Shape.FlattenSize));
            b.SetData(packer.UnpackArray(b.Shape.FlattenSize));
            t.SetData(packer.UnpackArray(t.Shape.FlattenSize));
            
            using var adam = new AdamOptimizer(new[] { a ,b }, 1.0f, beta1: 0.8f, beta2: 0.9f, eps: 0.001f);
            graph.Forward();
            graph.Backward();
            adam.Step();
            CheckValueSimilar(a, packer);
            
            Debug.Log("Adam test passed");
        }

        [Test]
        public static void MatMul()
        {
            var packer = new MessagePacker(new MemoryStream(Resources.Load<TextAsset>("UT/Test/MatMul").bytes));
            var a = new Tensor(new TensorShape(2, 3, 4, 5), true);
            var b = new Tensor(new TensorShape(2, 3, 5, 6), true);
            var c = Op.DummyMatMul(a, b);
            var t = new Tensor(new TensorShape(2, 3, 4, 6));
            var g = Op.MseLoss(c, t);
            using var graph = TensorGraph.BuildGraph(true);
            a.SetData(packer.UnpackArray(a.Shape.FlattenSize));
            b.SetData(packer.UnpackArray(b.Shape.FlattenSize));
            t.SetData(packer.UnpackArray(t.Shape.FlattenSize));
            graph.Forward();
            graph.Backward();
            CheckValueSimilar(c, packer);
            CheckValueSimilar(a.Gradient, packer);
            
            Debug.Log("MatMul test passed");
        }

        [Test]
        public static void Mul()
        {
            var packer = new MessagePacker(new MemoryStream(Resources.Load<TextAsset>("UT/Test/Mul").bytes));
            var a = new Tensor(new TensorShape(16, 10), true);
            var b = new Tensor(new TensorShape(10), true);
            var c = a * b;
            var t = new Tensor(new TensorShape(16, 10));
            var g = Op.MseLoss(c, t);
            using var graph = TensorGraph.BuildGraph(true);
            a.SetData(packer.UnpackArray(a.Shape.FlattenSize));
            b.SetData(packer.UnpackArray(b.Shape.FlattenSize));
            t.SetData(packer.UnpackArray(t.Shape.FlattenSize));
            graph.Forward();
            graph.Backward();
            CheckValueSimilar(c, packer);
            CheckValueSimilar(a.Gradient, packer);
            
            Debug.Log("Mul test passed");
        }

        [Test]
        public static void Linear()
        {
            var packer = new MessagePacker(new MemoryStream(Resources.Load<TextAsset>("UT/Test/Linear").bytes));
            var a = new Tensor(new TensorShape(3, 5, 8), true);
            using var linear = new Linear(8, 6);
            var b = linear.Forward(a);
            var c = Op.ReLU(b);
            var t = new Tensor(new TensorShape(3, 5, 6));
            var g = Op.MseLoss(c, t);
            using var graph = TensorGraph.BuildGraph(true);
            a.SetData(packer.UnpackArray(a.Shape.FlattenSize));
            linear.Weight.SetData(packer.UnpackArray(linear.Weight.Shape.FlattenSize));
            linear.Bias.SetData(packer.UnpackArray(linear.Bias.Shape.FlattenSize));
            t.SetData(packer.UnpackArray(t.Shape.FlattenSize));
            using var adam = new AdamOptimizer(new[] { linear.Weight, linear.Bias }, 1.0f, beta1: 0.8f, beta2: 0.9f, eps: 0.001f);
            graph.ClearGradient();
            graph.Forward();
            CheckValueSimilar(b, packer);
            CheckValueSimilar(c, packer);
            graph.Backward();
            CheckValueSimilar(linear.Bias.Gradient, packer);
            adam.Step();
            CheckValueSimilar(linear.Bias, packer);
            CheckValueSimilar(linear.Weight, packer);
            Debug.Log("Linear test passed");
        }

        [Test]
        public static void Max()
        {
            var packer = new MessagePacker(new MemoryStream(Resources.Load<TextAsset>("UT/Test/Max").bytes));
            var a = new Tensor(new TensorShape(2, 3, 4, 5), true);
            var pair = a.Max(1);
            var array = new int[pair.Value.count];
            var t = new Tensor(new TensorShape(2, 1, 4, 5));
            var g = Op.MseLoss(pair.Key, t);
            using var graph = TensorGraph.BuildGraph(true);
            a.SetData(packer.UnpackArray(a.Shape.FlattenSize));
            t.SetData(packer.UnpackArray(t.Shape.FlattenSize));
            graph.ClearGradient();
            graph.Forward();
            pair.Value.GetData(array);
            CheckValueSimilar(pair.Key, packer);
            graph.Backward();
            CheckValueSimilar(a.Gradient, packer);
            Debug.Log("Max test passed");
        }

        [Test]
        public static void Contiguous()
        {
            var packer = new MessagePacker(new MemoryStream(Resources.Load<TextAsset>("UT/Test/Contiguous").bytes));
            var a = new Tensor(new TensorShape(2, 1, 5), true);
            var b = a.Broadcast(new TensorShape(2, 3, 5)).Transpose(1, 2);
            var c = b.Contiguous();
            var t = new Tensor(new TensorShape(2, 5, 3));
            var g = Op.MseLoss(c, t);
            using var graph = TensorGraph.BuildGraph(true);
            a.SetData(packer.UnpackArray(a.Shape.FlattenSize));
            t.SetData(packer.UnpackArray(t.Shape.FlattenSize));
            graph.ClearGradient();
            graph.Forward();
            CheckValueSimilar(c, packer);
            graph.Backward();
            CheckValueSimilar(a.Gradient, packer);
            Debug.Log("Contiguous test passed");
        }
        
    }

}