using System;
using UT.Basic;

namespace UT.Core
{
    public class TensorStride : IntArray
    {

        public bool IsBroadcast
        {
            get
            {
                foreach (var i in Items) if (i == 0) return true;
                return false;
            }
        }

        public bool IsContinuous
        {
            get
            {
                var last = int.MaxValue;
                foreach (var i in Items)
                {
                    if (i == 0) return false;
                    if (i > last) return false;
                    last = i;
                }
                return true;
            }
        }
        
        public TensorStride(int[] items) : base(items)
        {
        }

        public TensorStride Transpose(int dim1, int dim2)
        {
            if (dim1 < 0) dim1 += Items.Length;
            if (dim2 < 0) dim2 += Items.Length;
            var items = new int[Items.Length];
            Array.Copy(Items, items, items.Length);
            (items[dim1], items[dim2]) = (items[dim2], items[dim1]);
            return new TensorStride(items);
        }

        public TensorStride Extend(int extendSize)
        {
            return new TensorStride(LeftPad(Items[0], extendSize));
        }
        
        public int CalculateNewIndex(int index, TensorStride newStride)
        {
            var newId = 0;
            for (var i = 0; i < Items.Length; i++)
            {
                var stride = Items[i];
                var d = index / stride;
                index -= d * stride;
                newId += d * newStride[i];
            }
            return newId;
        }
        
    }
}