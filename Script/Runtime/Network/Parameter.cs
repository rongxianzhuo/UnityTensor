using UT.Core;

namespace UnityTensor.Network
{
    public class Parameter : Tensor
    {
        public Parameter(params int[] shape) : base(new TensorShape(shape), true)
        {
        }
    }
}