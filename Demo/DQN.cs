using UnityEngine;
using UnityTensor.DQN;
using UnityTensor.Network;
using UT.Core;
using Random = System.Random;

public class DQN : MonoBehaviour
{

        public class Env : IEnv
        {
            
            private const float PlayerStep = 0.02f;
            private const float BallStep = 0.01f;
            private const float BallRadius = 0.025f;
            private const float BorderRadius = 1.0f;
            
            private static readonly float SqrtP5 = Mathf.Sqrt((BorderRadius - BallRadius) * 0.5f);

            private readonly Random _random = new Random(1);
            private int _tick;

            public Vector2 Player { get; private set; }

            public Vector2 Ball { get; private set; }
            
            private Vector2 RandomBornPosition =>
                new Vector2((float)(_random.NextDouble() * 2 - 1) * SqrtP5, (float)(_random.NextDouble() * 2 - 1) * SqrtP5);

            public Env()
            {
                Reset();
            }
            
            public void Reset()
            {
                _tick = 0;
                Player = RandomBornPosition;
                Ball = -Player;
            }

            public void GetState(float[] state)
            {
                state[0] = Player.x;
                state[1] = Player.y;
                state[2] = Ball.x;
                state[3] = Ball.y;
            }

            public bool Step(int action, out float reward)
            {
                var done = false;
                _tick++;
                Player += action switch
                {
                    0 => Vector2.up * PlayerStep,
                    1 => Vector2.right * PlayerStep,
                    2 => Vector2.down * PlayerStep,
                    3 => Vector2.left * PlayerStep,
                    _ => Vector2.zero
                };
                Ball += (Player - Ball).normalized * BallStep;
                reward = Vector2.Distance(Player, Ball) * 0.01f;
                if (_tick >= 1000)
                {
                    done = true;
                }
                else if (Vector2.Distance(Player, Ball) < BallRadius * 2)
                {
                    reward -= 1;
                    done = true;
                }
                else if (Player.magnitude > BorderRadius - BallRadius)
                {
                    reward -= 1;
                    done = true;
                }

                return done;
            }

            public void Render()
            {
                Gizmos.color = Color.red;
                Gizmos.DrawWireSphere(default, 1f);
                Gizmos.color = Color.green;
                Gizmos.DrawSphere(Player, BallRadius);
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(Ball, BallRadius);
            }
            
        }

        private class QNetwork : Module
        {

            private readonly Linear _fc1;
            private readonly Linear _fc2;
            private readonly Linear _fc3;

            public QNetwork()
            {
                _fc1 = new Linear(4, 64);
                _fc2 = new Linear(64, 64);
                _fc3 = new Linear(64, 4);
            }

            public override Tensor Forward(params Tensor[] tensors)
            {
                var x = tensors[0];
                x = Op.ReLU(_fc1.Forward(x));
                x = Op.ReLU(_fc2.Forward(x));
                return _fc3.Forward(x);
            }
        }

        private Env _previewEnv;
        private Env _trainEnv;
        private DqnAgent _dqn;
        private Module _qNetwork;
        private Module _targetQNetwork;
        private float[] _tempState;

        private void Awake()
        {
            _tempState = new float[4];
            _previewEnv = new Env();
            _trainEnv = new Env();
            _qNetwork = new QNetwork();
            _targetQNetwork = new QNetwork();
            _targetQNetwork.CopyParameter(_qNetwork).Forward();
            _dqn = new DqnAgent(4
                , 4
                , 32
                , 1000000
                , 500000
                , _qNetwork
                , _targetQNetwork);
        }

        private void Update()
        {
            if (!_dqn.IsTrainCompleted)
            {
                for (var i = 0; i < 64; i++)
                {
                    _dqn.TrainStep(_trainEnv);
                }
            }
            _previewEnv.GetState(_tempState);
            if (_previewEnv.Step(_dqn.TakeAction(_tempState), out _))
            {
                _previewEnv.Reset();
            }
        }

        private void OnDestroy()
        {
            _dqn.Dispose();
            _qNetwork.Dispose();
            _targetQNetwork.Dispose();
        }

        private void OnDrawGizmos()
        {
            _previewEnv?.Render();
        }
}
