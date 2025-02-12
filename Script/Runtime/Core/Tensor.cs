using System;
using UnityEngine;
using UT.Basic;

namespace UT.Core
{
    
    public partial class Tensor : DisposableObject
    {
        
        private static int _nextId;

        public readonly bool RequiresGrad;
        public readonly TensorShape Shape;
        public readonly TensorStride Stride;
        public readonly int Id = _nextId++;
        
        private readonly ComputeBuffer _buffer;
        private readonly TensorStride _continuousStride;

        private Array _internalBufferArray;
        
        public Tensor Gradient { get; private set;}

        public Tensor(TensorShape shape, bool requiresGrad=false) : this(shape, null, null, requiresGrad)
        {
        }

        public Tensor(TensorShape shape, Tensor origin, TensorStride stride, bool requiresGrad)
        {
            Shape = shape;
            Stride = stride == null ? Shape.CalculateContinuousStride() : stride;
            _continuousStride = shape.CalculateContinuousStride();
            RequiresGrad = requiresGrad;
            _buffer = origin == null ? BindDisposable(new ComputeBuffer(Shape.FlattenSize, 4)) : origin._buffer;
        }

        private T[] GetInternalBufferArray<T>()
        {
            _internalBufferArray ??= new T[_buffer.count];
            return (T[])_internalBufferArray;
        }

        public void AddGradient(Tensor grad)
        {
            if (Gradient == null) Gradient = grad;
            else Gradient += grad;
        }

        public void SetData<T>(T[] data)
        {
            if (Stride.IsBroadcast)
            {
                throw new NotSupportedException("Stride is broadcast.");
            }
            if (!Stride.IsContinuous)
            {
                var array = GetInternalBufferArray<T>();
                for (var i = 0; i < array.Length; i++)
                {
                    array[_continuousStride.CalculateNewIndex(i, Stride)] = data[i];
                }
                data = array;
            }
            _buffer.SetData(data);
        }

        public T[] GetData<T>()
        {
            var result = new T[Shape.FlattenSize];
            GetData(result);
            return result;
        }

        public void GetData<T>(T[] array)
        {
            var internalArray = GetInternalBufferArray<T>();
            _buffer.GetData(internalArray);
            for (var i = 0; i < array.Length; i++)
            {
                array[i] = internalArray[_continuousStride.CalculateNewIndex(i, Stride)];
            }
        }

        internal void SetToShader(ComputeShader cs, int kernel, string tensorName)
        {
            cs.SetBuffer(kernel, $"{tensorName}_stride", CreateStrideBuffer(Stride));
            cs.SetBuffer(kernel, tensorName, _buffer);
        }

        private ComputeBuffer CreateStrideBuffer(TensorStride stride)
        {
            var buffer = BindDisposable(new ComputeBuffer(stride.Length, sizeof(int)));
            buffer.SetData(stride.Items);
            return buffer;
        }

        public override string ToString()
        {
            return Id.ToString();
        }
    }
}