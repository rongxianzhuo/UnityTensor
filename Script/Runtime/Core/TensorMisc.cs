using UnityEngine;

namespace UT.Core
{
    public partial class Tensor
    {

        public void Print<T>()
        {
            Print(GetData<T>(), Shape);
        }

        public static void Print<T>(T[] array, TensorShape shape)
        {
            if (shape.Length > 1)
            {
                var builder = new System.Text.StringBuilder();
                var len = shape.FlattenSize / shape[0];
                for (var i = 0; i < shape[0]; i++)
                {
                    builder.Append(array[len * i]);
                    for (var j = 1; j < len; j++)
                    {
                        builder.Append(',');
                        builder.Append(array[len * i + j]);
                    }
                    if (i < shape[0] - 1) builder.Append('\n');
                }
                Debug.Log(builder);
            }
            else
            {
                Debug.Log(string.Join(',', array));
            }
        }
        
    }
}