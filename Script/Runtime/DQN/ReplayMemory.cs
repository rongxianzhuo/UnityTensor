using System;
using System.Collections.Generic;

namespace UnityTensor.DQN
{
    public class ReplayMemory
    {
        public readonly int Capacity;
        public readonly int StateSize;
        public readonly int ActionSize;

        private readonly Random _rand;
        private readonly int _batchSize;
        
        private readonly List<float[]> _stateReplay;
        private readonly List<float[]> _actionReplay;
        private readonly List<float[]> _rewardReplay;
        private readonly List<float[]> _nextStateReplay;
        private readonly List<float[]> _doneReplay;
        
        private readonly float[] _sampleState;
        private readonly float[] _sampleAction;
        private readonly float[] _sampleReward;
        private readonly float[] _sampleNextState;
        private readonly float[] _sampleDone;

        private int _nextIndex;
        private bool _full;

        public int Size => _full ? Capacity : _nextIndex;

        public ReplayMemory(int batchSize, int capacity, int stateSize, int actionSize, Random rand)
        {
            _rand = rand;
            _batchSize = batchSize;
            Capacity = capacity;
            StateSize = stateSize;
            ActionSize = actionSize;
            _sampleState = new float[batchSize * stateSize];
            _sampleAction = new float[batchSize * actionSize];
            _sampleReward = new float[batchSize * actionSize];
            _sampleNextState = new float[batchSize * stateSize];
            _sampleDone = new float[batchSize * actionSize];
            _stateReplay = new List<float[]>(capacity);
            _actionReplay = new List<float[]>(capacity);
            _rewardReplay = new List<float[]>(capacity);
            _nextStateReplay = new List<float[]>(capacity);
            _doneReplay = new List<float[]>(capacity);
            for (var i = 0; i < capacity; i++)
            {
                _stateReplay.Add(new float[StateSize]);
                _actionReplay.Add(new float[ActionSize]);
                _rewardReplay.Add(new float[ActionSize]);
                _nextStateReplay.Add(new float[StateSize]);
                _doneReplay.Add(new float[ActionSize]);
            }
        }

        public void Push(out float[] state, out float[] action, out float[] reward, out float[] nextState, out float[] done)
        {
            state = _stateReplay[_nextIndex];
            action = _actionReplay[_nextIndex];
            reward = _rewardReplay[_nextIndex];
            nextState = _nextStateReplay[_nextIndex];
            done = _doneReplay[_nextIndex];
            _nextIndex++;
            _full = _full || _nextIndex >= Capacity;
            _nextIndex = _nextIndex % Capacity;
        }

        public void SampleBatch(out float[] state
            , out float[] action
            , out float[] reward
            , out float[] nextState
            , out float[] done)
        {
            state = _sampleState;
            action = _sampleAction;
            reward = _sampleReward;
            nextState = _sampleNextState;
            done = _sampleDone;

            for (var i = 0; i < _batchSize; i++)
            {
                var j = _rand.Next(0, Size);
                Array.Copy(_stateReplay[j], 0, state, i * StateSize, StateSize);
                Array.Copy(_actionReplay[j], 0, action, i * ActionSize, ActionSize);
                Array.Copy(_rewardReplay[j], 0, reward, i * ActionSize, ActionSize);
                Array.Copy(_nextStateReplay[j], 0, nextState, i * StateSize, StateSize);
                Array.Copy(_doneReplay[j], 0, done, i * ActionSize, ActionSize);
            }
        }
    }
}