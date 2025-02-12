using System;
using System.Collections.Generic;
using UnityEngine;
using Object = UnityEngine.Object;

namespace UT.Core
{
    public class Operate
    {

        public readonly string Name;
        
        private readonly ComputeShader _cs;
        private readonly int _kernel;
        private int _groupSizeX;

        public Tensor Output;
        
        private readonly List<Tensor> _allTensors = new List<Tensor>();
        
        private readonly Dictionary<Tensor, Func<Tensor, Tensor>> _backwardFunctions = 
            new Dictionary<Tensor, Func<Tensor, Tensor>>();

        public IReadOnlyDictionary<Tensor, Func<Tensor, Tensor>> BackwardFunctions => _backwardFunctions;

        public static int PropertyId(string propertyName) => Shader.PropertyToID(propertyName);
        
        public IEnumerable<Tensor> Tensors => _allTensors;

        public Operate(string name, Tensor output, Tensor[] inputs)
        {
            Output = output;
            _allTensors.AddRange(inputs);
            _allTensors.Add(output);
            Name = name;
        }

        public Operate(string name, string kernel)
        {
            _cs = Object.Instantiate(Resources.Load<ComputeShader>($"UT/Operate/{name}"));
            _kernel = _cs.FindKernel(kernel);
            Name = name;
        }

        public Operate AddBackward(Tensor input, Func<Tensor, Tensor> func)
        {
            _backwardFunctions.Add(input, func);
            return this;
        }

        public Operate SetInt(int id, int i)
        {
            _cs.SetInt(id, i);
            return this;
        }

        public Operate SetInt(string name, int i)
        {
            _cs.SetInt(name, i);
            return this;
        }

        public Operate SetFloat(int id, float f)
        {
            _cs.SetFloat(id, f);
            return this;
        }

        public Operate SetFloat(string name, float f)
        {
            _cs.SetFloat(name, f);
            return this;
        }

        public Operate SetBuffer(string name, ComputeBuffer buffer)
        {
            _cs.SetBuffer(_kernel, name, buffer);
            return this;
        }

        public Operate SetTensor(string name, Tensor tensor)
        {
            if (_allTensors.Contains(tensor)) throw new ArgumentException("Tensor already added");
            _allTensors.Add(tensor);
            tensor.SetToShader(_cs, _kernel, name);
            return this;
        }

        public Operate SetWriteTensor(string name, Tensor tensor)
        {
            Output = tensor;
            SetTensor(name, tensor);
            _cs.GetKernelThreadGroupSizes(_kernel, out var gx, out var gy, out var gz);
            _groupSizeX = tensor.Shape.FlattenSize / (int) gx;
            if (tensor.Shape.FlattenSize % gx != 0) _groupSizeX++;
            return this;
        }

        public Operate Dispatch()
        {
            if (_cs != null)
            {
                _cs.Dispatch(_kernel, _groupSizeX, 1, 1);
            }
            return this;
        }

        public override string ToString()
        {
            return Name;
        }
    }
}