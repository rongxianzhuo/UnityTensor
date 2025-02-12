namespace UnityTensor.DQN
{
    public interface IEnv
    {
        void Reset();
        void GetState(float[] state);
        bool Step(int action, out float reward);
    }
}