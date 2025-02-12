using System;
using System.Collections.Generic;

namespace UT.Basic
{
    public class DisposableObject : IDisposable
    {
        
        private readonly List<IDisposable> _disposables = new List<IDisposable>();

        public T BindDisposable<T>(T disposable) where T : IDisposable
        {
            _disposables.Add(disposable);
            return disposable;
        }

        public virtual void Dispose()
        {
            foreach (var d in _disposables)
            {
                d.Dispose();
            }
            _disposables.Clear();
        }
    }
}