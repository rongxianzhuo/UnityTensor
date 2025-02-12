using UnityEngine;
using UT.Core;

namespace UnityTensor.Network
{
    public class Linear : Module
    {

        public readonly Parameter Weight;

        public readonly Parameter Bias;

        public Linear(int inFeatures, int outFeatures)
        {
            var min = -1f / Mathf.Sqrt(inFeatures);
            var max = -min;
            Weight = new Parameter(outFeatures, inFeatures);
            var weightArray = new float[Weight.Shape.FlattenSize];
            for (var i = 0; i < weightArray.Length; i++)
            {
                weightArray[i] = Random.Range(min, max);
            }
            Weight.SetData(weightArray);
            Bias = new Parameter(outFeatures);
            var biasArray = new float[Bias.Shape.FlattenSize];
            for (var i = 0; i < biasArray.Length; i++)
            {
                biasArray[i] = Random.Range(min, max);
            }
            Bias.SetData(biasArray);
        }

        public override Tensor Forward(params Tensor[] input)
        {
            var tWeight = Weight.Transpose(0, 1);
            return Op.DummyMatMul(input[0], tWeight) + Bias;
        }
    }
}