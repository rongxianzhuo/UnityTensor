using System;
using System.Collections.Generic;
using System.Reflection;
using UT.Core;

namespace UnityTensor.Network
{
    public abstract class Module : IDisposable
    {

        private Dictionary<string, Parameter> _allParameters;

        public IReadOnlyDictionary<string, Parameter> Parameters
        {
            get
            {
                if (_allParameters != null) return _allParameters;
                _allParameters = new Dictionary<string, Parameter>();
                var pType = typeof(Parameter);
                var mType = typeof(Module);
                _allParameters.Clear();
                foreach (var fieldInfo in GetType().GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
                {
                    if (fieldInfo.FieldType == pType)
                    {
                        _allParameters[fieldInfo.Name] = (Parameter) fieldInfo.GetValue(this);
                    }
                    if (fieldInfo.FieldType.IsSubclassOf(mType))
                    {
                        foreach (var pair in ((Module)fieldInfo.GetValue(this)).Parameters)
                        {
                            _allParameters[$"{fieldInfo.Name}.{pair.Key}"] = pair.Value;
                        }
                    }
                }
                return _allParameters;
            }
        }

        public TensorGraph CopyParameter(Module other)
        {
            foreach (var pair in other.Parameters)
            {
                Op.Copy(pair.Value, Parameters[pair.Key]);
            }
            return TensorGraph.BuildGraph();
        }

        public virtual void Dispose()
        {
            var pType = typeof(Parameter);
            var mType = typeof(Module);
            foreach (var fieldInfo in GetType().GetFields(BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic))
            {
                if (fieldInfo.FieldType == pType)
                {
                    ((Parameter)fieldInfo.GetValue(this)).Dispose();
                }
                if (fieldInfo.FieldType.IsSubclassOf(mType))
                {
                    ((Module)fieldInfo.GetValue(this)).Dispose();
                }
            }
        }

        public abstract Tensor Forward(params Tensor[] tensors);

    }
}