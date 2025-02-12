using System.Collections.Generic;
using UnityEngine;

namespace UT.Core
{
    public partial class Tensor
    {

        public Tensor Operate(Operate op)
        {
            return Operate(op, string.Empty);
        }

        public Tensor Operate(Operate op, string propertyName)
        {
            if (!string.IsNullOrEmpty(propertyName)) op.SetWriteTensor(propertyName, this);
            TensorGraph.RecordTensor(op);
            return this;
        }

        public Tensor Broadcast(TensorShape newShape)
        {
            return Op.Broadcast(this, newShape);
        }

        public Tensor Transpose(int dim1, int dim2)
        {
            return Op.Transpose(this, dim1, dim2);
        }

        public Tensor Sum(params int[] dims)
        {
            return Op.Sum(this, dims);
        }

        public Tensor Reshape(params int[] shape)
        {
            return Op.Reshape(this, new TensorShape(shape));
        }

        public Tensor Contiguous()
        {
            return Op.Contiguous(this);
        }

        public Tensor Copy()
        {
            return Op.Copy(this);
        }

        public KeyValuePair<Tensor, ComputeBuffer> Max(int dim)
        {
            return Op.Max(this, dim);
        }

        public Tensor Squeeze(params int[] dims)
        {
            return Op.Squeeze(this, dims);
        }

        public static Tensor operator +(Tensor a, Tensor b)
        {
            return Op.Add(a, b);
        }

        public static Tensor operator *(Tensor a, Tensor b)
        {
            return Op.Mul(a, b);
        }
        
    }
}