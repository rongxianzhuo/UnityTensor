using System;
using System.Collections.Generic;
using UnityEngine;
using UT.Basic;

namespace UT.Core
{
    public class TensorShape : IntArray
    {
        
        public readonly int FlattenSize;

        public TensorShape(params int[] items) : base(items)
        {
            FlattenSize = 1;
            foreach (var i in items) FlattenSize *= i;
        }

        public TensorShape Transpose(int dim1, int dim2)
        {
            if (dim1 < 0) dim1 += Items.Length;
            if (dim2 < 0) dim2 += Items.Length;
            var items = new int[Items.Length];
            Array.Copy(Items, items, items.Length);
            (items[dim1], items[dim2]) = (items[dim2], items[dim1]);
            return new TensorShape(items);
        }
        
        public TensorStride CalculateContinuousStride()
        {
            var stride = new int[Items.Length];
            var j = 1;
            for (var i = Items.Length - 1; i >= 0; i--)
            {
                stride[i] = j;
                j *= Items[i];
            }
            return new TensorStride(stride);
        }

        public static TensorShape CalculateBroadcastShape(TensorShape shape1, TensorShape shape2)
        {
            var extendShape = new List<int>();
            if (shape1.Length > shape2.Length) (shape2, shape1) = (shape1, shape2);
            extendShape.AddRange(shape2.Items);
            var d = shape2.Length - shape1.Length;
            for (var i = 0; i < shape1.Length; i++)
            {
                if (shape1[i] == shape2[i + d]) continue;
                if (shape1[i] != 1 && shape2[i + d] != 1) throw new Exception("cannot calculate broadcast shape");
                extendShape[i + d] = Mathf.Max(shape1[i], shape2[i + d]);
            }
            return new TensorShape(extendShape.ToArray());
        }
    }
}