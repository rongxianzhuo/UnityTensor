using System;
using System.Collections.Generic;
using UnityEngine;

namespace UT.Core
{
    public static class Op
    {

        public static Tensor Contiguous(Tensor a)
        {
            if (!a.Stride.IsBroadcast && a.Stride.IsContinuous)
            {
                return a;
            }
            return Copy(a);
        }

        public static Tensor Reshape(Tensor a, TensorShape shape)
        {
            if (a.Shape == shape) return a;
            if (a.Shape.FlattenSize != shape.FlattenSize)
            {
                throw new ArgumentException("The input tensor should have the same shape.");
            }
            if (a.Stride.IsBroadcast || !a.Stride.IsContinuous)
            {
                a = Contiguous(a);
            }
            var result = new Tensor(shape, a, shape.CalculateContinuousStride(), a.RequiresGrad);
            var operate = new Operate("Reshape", result, new []{a});
            if (a.RequiresGrad)
            {
                operate.AddBackward(a, grad => Reshape(grad, a.Shape));
            }
            return result.Operate(operate);
        }

        public static Tensor Add(Tensor a, Tensor b, Tensor c=null)
        {
            var broadcastShape = TensorShape.CalculateBroadcastShape(a.Shape, b.Shape);
            c ??= new Tensor(broadcastShape, a.RequiresGrad || b.RequiresGrad);
            if (a.Shape != c.Shape) a = a.Broadcast(c.Shape);
            if (b.Shape != c.Shape) b = b.Broadcast(c.Shape);
            var operate = new Operate("Add", "Forward");
            operate.SetInt("dim", c.Shape.Length);
            operate.SetTensor("a", a);
            operate.SetTensor("b", b);
            if (a.RequiresGrad) operate.AddBackward(a, grad => grad);
            if (b.RequiresGrad) operate.AddBackward(b, grad => grad);
            return c.Operate(operate, "c");
        }

        public static Operate AddInPlace(Tensor a, Tensor b)
        {
            if (b.Shape != a.Shape) b = b.Broadcast(a.Shape);
            var operate = new Operate("AddInPlace", "Forward");
            operate.SetInt("dim", a.Shape.Length);
            operate.SetTensor("b", b);
            operate.SetWriteTensor("a", a);
            return operate;
        }

        public static Tensor DummyMatMul(Tensor a, Tensor b, Tensor c=null)
        {
            var outputShape = new List<int>();
            if (a.Shape.Length >= b.Shape.Length)
            {
                outputShape.AddRange(a.Shape.Items);
                outputShape[^1] = b.Shape[^1];
                outputShape[^2] = b.Shape[^2];
                if (b.Shape != new TensorShape(outputShape.ToArray()))
                {
                    b = b.Broadcast(new TensorShape(outputShape.ToArray()));
                }
            }
            else
            {
                outputShape.AddRange(b.Shape.Items);
                outputShape[^1] = a.Shape[^1];
                outputShape[^2] = a.Shape[^2];
                if (a.Shape != new TensorShape(outputShape.ToArray()))
                {
                    a = a.Broadcast(new TensorShape(outputShape.ToArray()));
                }
            }
            outputShape[^1] = b.Shape[^1];
            outputShape[^2] = a.Shape[^2];
            c ??= new Tensor(new TensorShape(outputShape.ToArray()), a.RequiresGrad || b.RequiresGrad);
            var operate = new Operate("MatMul", "Forward");
            operate.SetInt("dim", c.Shape.Length);
            operate.SetInt("p", a.Shape[^1]);
            operate.SetTensor("a", a);
            operate.SetTensor("b", b);
            if (a.RequiresGrad)
            {
                operate.AddBackward(a, grad =>
                {
                    var t = Transpose(b, -1, -2);
                    var g = new Tensor(a.Shape);
                    DummyMatMul(grad, t, g);
                    return g;
                });
            }
            if (b.RequiresGrad)
            {
                operate.AddBackward(b, grad =>
                {
                    var t = Transpose(a, -1, -2);
                    var g = new Tensor(b.Shape);
                    DummyMatMul(t, grad, g);
                    return g;
                });
            }
            return c.Operate(operate, "c");
        }

        public static Tensor Transpose(Tensor tensor, int dim1, int dim2, Tensor result=null)
        {
            result ??= new Tensor(tensor.Shape.Transpose(dim1, dim2), tensor, tensor.Stride.Transpose(dim1, dim2), tensor.RequiresGrad);
            var operate = new Operate("Transpose", result, new []{tensor});
            if (tensor.RequiresGrad)
            {
                operate.AddBackward(tensor, grad => grad.Transpose(dim1, dim2));
            }
            return result.Operate(operate);
        }
        
        public static Tensor Squeeze(Tensor a, int[] dims, Tensor result=null)
        {
            var shapeList = new List<int>();
            shapeList.AddRange(a.Shape.Items);
            var strideList = new List<int>();
            strideList.AddRange(a.Stride.Items);
            var dimsList = new List<int>();
            dimsList.AddRange(dims);
            for (var i = shapeList.Count - 1; i >= 0; i--)
            {
                if (!dimsList.Contains(i) || shapeList[i] != 1) continue;
                shapeList.RemoveAt(i);
                strideList.RemoveAt(i);
            }

            result ??= new Tensor(new TensorShape(shapeList.ToArray())
                , a
                , new TensorStride(strideList.ToArray()),
                a.RequiresGrad);
            var operate = new Operate("Squeeze", result, new[] { a });
            if (a.RequiresGrad)
            {
                operate.AddBackward(a, grad => grad);
            }
            return result.Operate(operate);
        }
        
        public static Tensor Unsqueeze(Tensor a, int dim, Tensor result=null)
        {
            var shapeList = new List<int>();
            shapeList.AddRange(a.Shape.Items);
            shapeList.Insert(dim, 1);
            var strideList = new List<int>();
            strideList.AddRange(a.Stride.Items);
            var stride = 1;
            for (var i = dim; i < strideList.Count; i++)
            {
                stride = Mathf.Max(stride, strideList[i]);
            }
            strideList.Insert(dim, stride);
            result ??= new Tensor(new TensorShape(shapeList.ToArray())
                , a
                , new TensorStride(strideList.ToArray()),
                a.RequiresGrad);
            var operate = new Operate("Unsqueeze", result, new []{a});
            if (a.RequiresGrad)
            {
                operate.AddBackward(a, grad => grad);
            }
            return result.Operate(operate);
        }

        public static Tensor Mul(Tensor a, Tensor b, Tensor c=null)
        {
            var shape = TensorShape.CalculateBroadcastShape(a.Shape, b.Shape);
            c ??= new Tensor(shape, a.RequiresGrad || b.RequiresGrad);
            
            if (a.Shape != shape) a = a.Broadcast(shape);

            if (b.Shape != shape) b = b.Broadcast(shape);
            
            var operate = new Operate("Mul", "Forward");
            operate.SetInt("dim", shape.Length);
            operate.SetTensor("a", a);
            operate.SetTensor("b", b);

            if (a.RequiresGrad)
            {
                operate.AddBackward(a, grad => Mul(b, grad));
            }
            if (b.RequiresGrad)
            {
                operate.AddBackward(b, grad => Mul(a, grad));
            }
            
            return c.Operate(operate, "c");
        }
        
        public static Tensor Copy(Tensor a, Tensor b=null)
        {
            b ??= new Tensor(a.Shape, a.RequiresGrad);
            var operate = new Operate("Copy", "CSMain");
            operate.SetInt("dim", b.Shape.Length);
            operate.SetTensor("a", a);
            if (a.RequiresGrad) operate.AddBackward(a, grad => grad);
            return b.Operate(operate, "b");
        }
        
        public static Operate Clear(Tensor tensor, float value)
        {
            var operate = new Operate("Clear", "Forward");
            operate.SetFloat("f", value);
            operate.SetWriteTensor("a", tensor);
            return operate;
        }
        
        public static Operate Lerp(Tensor a, Tensor b, Tensor t)
        {
            var operate = new Operate("Lerp", "Forward");
            operate.SetTensor("t", t);
            operate.SetTensor("b", b);
            operate.SetWriteTensor("a", a);
            return operate;
        }

        public static Operate ClipNorm(Tensor tensor, float maxNorm)
        {
            return new Operate("ClipNorm", "CSMain")
                .SetInt("size", tensor.Shape.FlattenSize)
                .SetFloat("max_norm", maxNorm)
                .SetWriteTensor("buffer", tensor);
        }

        public static Tensor ReLU(Tensor tensor, Tensor result=null)
        {
            result ??= new Tensor(tensor.Shape, tensor.RequiresGrad);
            var operate = new Operate("ReLU", "Forward");
            operate.SetInt("dim", tensor.Shape.Length);
            operate.SetTensor("input", tensor);
            if (tensor.RequiresGrad)
            {
                operate.AddBackward(tensor, grad =>
                {
                    var r = new Tensor(tensor.Shape);
                    var o = new Operate("ReLU", "Backward");
                    o.SetInt("dim", tensor.Shape.Length);
                    o.SetTensor("r_output", result);
                    o.SetTensor("output_gradient", grad);
                    return r.Operate(o, "input_gradient");
                });
            }
            return result.Operate(operate, "rw_output");
        }

        public static Tensor Sum(Tensor tensor, int[] dims, Tensor result=null)
        {
            if (dims == null || dims.Length == 0)
            {
                dims = new int[tensor.Shape.Length];
                for (var i = 0; i < dims.Length; i++)
                {
                    dims[i] = i;
                }
            }

            var totalOffsetLength = 1;
            foreach (var d in dims) totalOffsetLength *= tensor.Shape[d];
            var position = new int[dims.Length];
            var totalOffset = new List<int>(totalOffsetLength);
            while (position[0] < tensor.Shape[dims[0]])
            {
                var index = 0;
                for (var i = 0; i < position.Length; i++)
                {
                    index += tensor.Stride[dims[i]] * position[i];
                }

                totalOffset.Add(index);
                position[^1]++;
                for (var i = position.Length - 1; i >= 0; i--)
                {
                    if (position[i] < tensor.Shape[dims[i]]) break;
                    if (i == 0) break;
                    position[i] = 0;
                    position[i - 1]++;
                }
            }

            var resultShape = new int[tensor.Shape.Length];
            Array.Copy(tensor.Shape.Items, resultShape, resultShape.Length);
            foreach (var d in dims)
            {
                resultShape[d] = 1;
            }

            var outputShape = new TensorShape(resultShape);
            var totalOffsetBuffer = new ComputeBuffer(totalOffset.Count, sizeof(int));
            totalOffsetBuffer.SetData(totalOffset);

            var operate = new Operate("Sum", "Forward");
            operate.SetInt("dim", tensor.Shape.Length);
            operate.SetInt("total_offset_length", totalOffsetLength);
            operate.SetBuffer("total_offset", totalOffsetBuffer);
            operate.SetWriteTensor("a", tensor);
            result ??= new Tensor(outputShape, tensor.RequiresGrad);
            if (tensor.RequiresGrad)
            {
                operate.AddBackward(tensor, grad => grad.Broadcast(tensor.Shape));
            }
            result.Operate(operate, "result");
            result.BindDisposable(totalOffsetBuffer);
            return result;
        }

        public static KeyValuePair<Tensor, ComputeBuffer> Max(Tensor tensor, int dim, Tensor result=null)
        {
            var maxDimStride = tensor.Stride[dim];
            var resultShape = new int[tensor.Shape.Length];
            Array.Copy(tensor.Shape.Items, resultShape, resultShape.Length);
            resultShape[dim] = 1;
            var outputShape = new TensorShape(resultShape);
            var maxIndexBuffer = new ComputeBuffer(outputShape.FlattenSize, sizeof(int));
            var operate = new Operate("Max", "Forward");
            operate.SetInt("dim", tensor.Shape.Length);
            operate.SetInt("max_dim_size", tensor.Shape[dim]);
            operate.SetInt("max_dim_stride", maxDimStride);
            operate.SetBuffer("result_max_index", maxIndexBuffer);
            operate.SetWriteTensor("a", tensor);
            result ??= new Tensor(outputShape, tensor.RequiresGrad);
            if (tensor.RequiresGrad)
            {
                operate.AddBackward(tensor, grad =>
                {
                    var g = new Tensor(tensor.Shape);
                    var o = new Operate("MaxBackward", "Forward")
                        .SetInt("dim", tensor.Shape.Length)
                        .SetInt("max_dim_stride", maxDimStride)
                        .SetInt("max_dim_size", tensor.Shape[dim])
                        .SetBuffer("result_max_index", maxIndexBuffer)
                        .SetTensor("input_gradient", grad.Broadcast(tensor.Shape));
                    return g.Operate(o, "output_gradient");
                });
            }
            result.Operate(operate, "result");
            result.BindDisposable(maxIndexBuffer);
            return new KeyValuePair<Tensor, ComputeBuffer>(result, maxIndexBuffer);
        }

        public static Tensor Broadcast(Tensor tensor, TensorShape newShape, Tensor result=null)
        {
            if (tensor.Shape == newShape) return result;
            var expendLeftSize = newShape.Length - tensor.Shape.Length;
            var extendShape = tensor.Shape.LeftPad(0, newShape.Length);
            var newStride = tensor.Stride.Extend(newShape.Length);
            for (var i = 0; i < extendShape.Length; i++)
            {
                if (extendShape[i] != newShape[i]) newStride[i] = 0;
            }
            
            result ??= new Tensor(newShape, tensor, newStride, tensor.RequiresGrad);
            var operate = new Operate("Broadcast", result, new []{tensor});
            if (tensor.RequiresGrad)
            {
                var sumDims = new List<int>();
                for (var i = 0; i < newStride.Length; i++)
                {
                    if (newStride[i] == 0) sumDims.Add(i);
                }
                operate.AddBackward(tensor, grad =>
                {
                    var r =  grad.Sum(sumDims.ToArray());
                    if (expendLeftSize > 0)
                    {
                        var squeezeDims = new List<int>();
                        for (var i = 0; i < expendLeftSize; i++)
                        {
                            squeezeDims.Add(i);
                        }
                        var rr = r.Squeeze(squeezeDims.ToArray());
                        rr.BindDisposable(r);
                        r = rr;
                    }
                    return r;
                });
            }
            return result.Operate(operate);
        }

        public static Tensor MseLoss(Tensor a, Tensor t)
        {
            var result = new Tensor(new TensorShape(1));
            var o = new Operate("MseLossForward", result, new []{a, t});
            if (a.RequiresGrad)
            {
                var gradient = new Tensor(a.Shape);
                o.AddBackward(a, _ =>
                {
                    var operate = new Operate("MseLoss", "Forward");
                    operate.SetInt("n", a.Shape.FlattenSize);
                    operate.SetInt("dim", a.Shape.Length);
                    operate.SetTensor("a", a);
                    operate.SetTensor("target", t);
                    return gradient.Operate(operate, "gradient");
                });
            }
            return result.Operate(o);
        }

        public static Tensor SmoothL1Loss(Tensor a, Tensor t, float beta = 1.0f, float scale = 1.0f)
        {
            var result = new Tensor(new TensorShape(1));
            var o = new Operate("SmoothL1LossForward", result, new []{a, t});
            if (a.RequiresGrad)
            {
                var gradient = new Tensor(a.Shape);
                o.AddBackward(a, _ =>
                {
                    var operate = new Operate("SmoothL1Loss", "CSMain")
                        .SetInt("n", a.Shape.FlattenSize)
                        .SetInt("dim", a.Shape.Length)
                        .SetFloat("beta", beta)
                        .SetFloat("scale", scale)
                        .SetTensor("output", a)
                        .SetTensor("target", t);
                    return gradient.Operate(operate, "gradient");
                });
            }
            return result.Operate(o);
        }
        
    }
}