using System;
using System.Collections.Generic;
using UnityEngine;
using UnityTensor.Network;
using UnityTensor.Utility;
using UT.Core;
using Random = System.Random;

namespace UnityTensor.DQN
{
    public class DqnAgent : IDisposable
    {

        private class Dqn : IDisposable
        {

            public readonly Module QNet;
            public readonly Tensor RuntimeInput;
            public readonly Tensor RuntimeOutput;
            
            public readonly Tensor TrainInput;
            public readonly Tensor TrainOutput;
            public readonly Tensor TrainTarget;
            public readonly TensorGraph RuntimeGraph;
            public readonly TensorGraph TrainGraph;

            public Dqn(int stateSize, int actionSize, int batchSize, Module qNetwork, bool backward)
            {
                QNet = qNetwork;
                RuntimeInput = new Tensor(new TensorShape(1, stateSize));
                RuntimeOutput = qNetwork.Forward(RuntimeInput);
                RuntimeGraph = TensorGraph.BuildGraph();
                
                TrainInput = new Tensor(new TensorShape(batchSize, stateSize));
                TrainOutput = qNetwork.Forward(TrainInput);
                if (backward)
                {
                    TrainTarget = new Tensor(TrainOutput.Shape);
                    Op.SmoothL1Loss(TrainOutput, TrainTarget, scale:actionSize);
                }
                TrainGraph = TensorGraph.BuildGraph(backward);
            }

            public void Dispose()
            {
                RuntimeGraph.Dispose();
                TrainGraph.Dispose();
            }
        }

        private readonly int _totalTrainSteps;
        private readonly Dqn _dqn;
        private readonly Dqn _targetDqn;
        private readonly AdamOptimizer _optimizer;
        private readonly Tensor _rewardTensor;
        private readonly Tensor _doneTensor;
        private readonly Tensor _actionTensor;
        private readonly ReplayMemory _memory;
        private readonly int _learningStarts = 100;
        private readonly int _trainFreq = 4;
        private readonly float[] _actionArray;
        private readonly Random _rand = new Random(1);
        private readonly TensorGraph _syncTargetQNetGraph;
        private readonly TensorGraph _targetQValueGraph;
        private readonly List<Operate> _clipGradNormOps = new List<Operate>();

        private uint _trainStep;
        
        public bool IsTrainCompleted => _trainStep >=  _totalTrainSteps;

        public DqnAgent(int stateSize
            , int actionSize
            , int batchSize
            , int replayBufferSize
            , int totalTrainSteps
            , Module qNetwork
            , Module targetQNetwork)
        {
            _totalTrainSteps = totalTrainSteps;
            _actionArray = new float[actionSize];
            _memory = new ReplayMemory(batchSize, replayBufferSize, stateSize, actionSize, _rand);

            _rewardTensor = new Tensor(new TensorShape(batchSize, actionSize));
            _doneTensor = new Tensor(new TensorShape(batchSize, actionSize));
            _actionTensor = new Tensor(new TensorShape(batchSize, actionSize));

            _dqn = new Dqn(stateSize, actionSize, batchSize, qNetwork, true);
            _targetDqn = new Dqn(stateSize, actionSize, batchSize, targetQNetwork, false);
            
            _optimizer = new AdamOptimizer(qNetwork.Parameters.Values, learningRate:0.0001f);
            
            Op.Add(_targetDqn.TrainOutput.Max(1).Key * _doneTensor, _rewardTensor, _dqn.TrainTarget);
            var lerp = Op.Lerp(_dqn.TrainTarget, _dqn.TrainOutput, _actionTensor);
            TensorGraph.RecordTensor(lerp);
            _targetQValueGraph = TensorGraph.BuildGraph();
            
            foreach (var node in _dqn.QNet.Parameters)
            {
                _clipGradNormOps.Add(Op.ClipNorm(node.Value.Gradient, 10f));
            }

            _syncTargetQNetGraph = targetQNetwork.CopyParameter(qNetwork);
        }

        public void TrainStep(IEnv trainEnv
            , float futureRewardDiscount = 0.99f
            , uint targetUpdateInterval = 10000)
        {
            if (IsTrainCompleted) return;
            _memory.Push(out var state, out var action, out var reward, out var nextState, out var done);
            trainEnv.GetState(state);
            _dqn.RuntimeInput.SetData(state);
            _dqn.RuntimeGraph.Forward();
            var a = SelectAction(_dqn.RuntimeOutput, false);
            var d = trainEnv.Step(a, out var r);
            if (d) trainEnv.Reset();
            else trainEnv.GetState(nextState);
            for (var j = 0; j < reward.Length; j++)
            {
                reward[j] = r;
                done[j] = d ? 0f : futureRewardDiscount;
                action[j] = j == a ? 0 : 1;
            }
            _trainStep++;
            if (_trainStep < _learningStarts) return;
            if (_trainStep % _trainFreq != 0) return;
                
            _memory.SampleBatch(out var sampleState
                , out var sampleAction
                , out var sampleReward
                , out var sampleNextState
                , out var sampleDone);
            
            _dqn.TrainInput.SetData(sampleState);
            _dqn.TrainGraph.ClearGradient();
            _dqn.TrainGraph.Forward();
            
            _actionTensor.SetData(sampleAction);
            _rewardTensor.SetData(sampleReward);
            _doneTensor.SetData(sampleDone);
            _targetDqn.TrainInput.SetData(sampleNextState);
            _targetDqn.TrainGraph.Forward();
            _targetQValueGraph.Forward();
            _dqn.TrainGraph.Backward();
            foreach (var o in _clipGradNormOps) o.Dispatch();
            _optimizer.Step();

            if (_trainStep % 10000 == 0) Debug.Log($"Train Step: {_trainStep}");
            if (_trainStep % targetUpdateInterval != 0) return;
            _syncTargetQNetGraph.Forward();
        }

        public int TakeAction(float[] state)
        {
            _targetDqn.RuntimeInput.SetData(state);
            _targetDqn.RuntimeGraph.Forward();
            return SelectAction(_targetDqn.RuntimeOutput, true);
        }

        public void Dispose()
        {
            _actionTensor.Dispose();
            _rewardTensor.Dispose();
            _doneTensor.Dispose();
            _dqn.Dispose();
            _targetDqn.Dispose();
            _optimizer.Dispose();
            _targetQValueGraph.Dispose();
            _syncTargetQNetGraph.Dispose();
        }

        private int SelectAction(Tensor actionOutput, bool deterministic)
        {
            float explorationRate = -1;
            if (!deterministic)
            {
                if (_trainStep < _learningStarts)
                {
                    explorationRate = 1;
                }
                else
                {
                    var trainProgress = (float)_trainStep / _totalTrainSteps;
                    if (trainProgress > 0.1f) explorationRate = 0.05f;
                    else explorationRate = 1.0f - trainProgress * (1.0f - 0.05f) / 0.1f;
                }
            }

            if (explorationRate >= 0 && _rand.NextDouble() < explorationRate)
            {
                return _rand.Next(0, actionOutput.Shape.FlattenSize);
            }
            return Max();
            int Max()
            {
                actionOutput.GetData(_actionArray);
                var index = 0;
                var maxValue = _actionArray[0];
                for (var i = 1; i < _actionArray.Length; i++)
                {
                    if (_actionArray[i] < maxValue) continue;
                    index = i;
                    maxValue = _actionArray[i];
                }
                return index;
            }
        }
    }
}