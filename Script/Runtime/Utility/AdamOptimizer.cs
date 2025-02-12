using System;
using System.Collections.Generic;
using UT.Core;

namespace UnityTensor.Utility
{
    public class AdamOptimizer : IDisposable
    {

        private readonly int _tId = Operate.PropertyId("t");
        private readonly Dictionary<Tensor, Operate> _stepOperations = new Dictionary<Tensor, Operate>();
        private readonly HashSet<Tensor> _tempTensors = new HashSet<Tensor>();

        private float _t;

        public AdamOptimizer(IEnumerable<Tensor> parameters
            , float learningRate=1e-3f
            , float weightDecay=0f
            , float beta1=0.9f
            , float beta2=0.999f
            , float eps=1e-8f)
        {
            foreach (var p in parameters)
            {
                AddParameter(p, beta1, beta2, eps, learningRate, weightDecay);
            }
        }

        private void AddParameter(Tensor tensor, float beta1, float beta2, float eps, float learningRate, float weightDecay)
        {
            var m = new Tensor(tensor.Shape);
            Op.Clear(m, 0f).Dispatch();
            var v = new Tensor(tensor.Shape);
            Op.Clear(v, 0f).Dispatch();
            _tempTensors.Add(m);
            _tempTensors.Add(v);
            var op = new Operate("Adam", "CSMain")
                .SetInt("dim", tensor.Shape.Length)
                .SetFloat("beta1", beta1)
                .SetFloat("beta2", beta2)
                .SetFloat("eps", eps)
                .SetFloat("weight_decay", weightDecay)
                .SetFloat("learning_rate", learningRate)
                .SetTensor("g", tensor.Gradient)
                .SetTensor("m", m)
                .SetTensor("v", v)
                .SetWriteTensor("theta", tensor);
            _stepOperations.Add(tensor, op);
        }

        public void Step()
        {
            _t++;
            foreach (var pair in _stepOperations)
            {
                pair.Value.SetFloat(_tId, _t);
                pair.Value.Dispatch();
            }
        }

        public void Dispose()
        {
            foreach (var t in _tempTensors)
            {
                t?.Dispose();
            }
            _tempTensors.Clear();
            _stepOperations.Clear();
        }
    }
}