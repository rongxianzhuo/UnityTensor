using System;

namespace UT.Basic
{
    public class IntArray
    {

        public readonly int[] Items;

        public int Length => Items.Length;

        public int this[int i]
        {
            get => Items[i];
            set => Items[i] = value;
        }

        public IntArray(int[] items)
        {
            Items = new int[items.Length];
            Array.Copy(items, Items, items.Length);
        }

        public int[] LeftPad(int pad, int count)
        {
            var array = new int[count];
            var d = count - Items.Length;
            for (var j = 0; j < count; j++)
            {
                if (j < d) array[j] = pad;
                else array[j] = Items[j - d];
            }
            return array;
        }

        public override int GetHashCode()
        {
            var hash = 0;
            foreach (var i in Items) hash += i;
            return hash;
        }

        public override bool Equals(object obj)
        {
            if (obj is not IntArray s) return false;
            return Equals(s);
        }

        public bool Equals(IntArray other)
        {
            if (other == null) return false;
            if (Items.Length != other.Items.Length) return false;
            for (var i = 0; i < Items.Length; i++) if (Items[i] != other.Items[i]) return false;
            return true;
        }

        public static bool operator ==(IntArray a, IntArray b)
        {
            if (a is null)
            {
                return b is null;
            }
            return a.Equals(b);
        }

        public static bool operator !=(IntArray a, IntArray b)
        {
            if (a is null)
            {
                return b is not null;
            }
            return !a.Equals(b);
        }

        public override string ToString()
        {
            return $"[{string.Join(',', Items)}]";
        }
    }
}