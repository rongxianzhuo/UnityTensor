using System;
using System.Collections.Generic;
using UnityTensor.Network;

namespace UT.Core
{
    public class TensorGraph : IDisposable
    {

        private static TensorGraph _recordingGraph = new TensorGraph();

        private readonly List<Operate> _forwardOperations = new List<Operate>();

        private readonly List<Operate> _backwardOperations = new List<Operate>();
        
        private readonly List<Operate> _clearGradientOps = new List<Operate>();

        private bool _recordingBackward;

        public static void RecordTensor(Operate operation)
        {
            _recordingGraph.PushTensor(operation);
        }

        public static TensorGraph BuildGraph(bool backward=false)
        {
            if (backward) _recordingGraph.BuildBackwardOps();
            var r = _recordingGraph;
            _recordingGraph = new TensorGraph();
            return r;
        }

        private TensorGraph()
        {
            
        }

        public void ClearGradient()
        {
            foreach (var op in _clearGradientOps)
            {
                op.Dispatch();
            }
        }

        private void BuildBackwardOps()
        {
            _recordingBackward = true;
            for (var i = _forwardOperations.Count - 1; i >= 0; i--)
            {
                var op = _forwardOperations[i];
                foreach (var pair in op.BackwardFunctions)
                {
                    var grad = pair.Value(op.Output.Gradient);
                    _clearGradientOps.Add(Op.Clear(grad, 0f));
                    pair.Key.AddGradient(grad);
                }
            }
        }

        private void PushTensor(Operate operation)
        {
            if (_recordingBackward) _backwardOperations.Add(operation);
            else _forwardOperations.Add(operation);
        }

        public void Forward()
        {
            foreach (var o in _forwardOperations)
            {
                o.Dispatch();
            }
        }

        public void Backward()
        {
            foreach (var o in _backwardOperations)
            {
                o.Dispatch();
            }
        }

        public void Dispose()
        {
            foreach (var o in _forwardOperations)
            {
                foreach (var t in o.Tensors)
                {
                    if (t is Parameter) continue;
                    t.Dispose();
                }
            }
            foreach (var o in _backwardOperations)
            {
                foreach (var t in o.Tensors)
                {
                    if (t is Parameter) continue;
                    t.Dispose();
                }
            }
        }
    }
}