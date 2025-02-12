using System;
using System.Collections.Generic;
using System.IO;

namespace UT.Basic
{
    public class MessagePacker
    {

        private readonly Stream _stream;
        private readonly byte[] _bytes = new byte[sizeof(float)];

        public MessagePacker(Stream stream)
        {
            _stream = stream;
        }

        private void ReadBytes()
        {
            for (var j = 0; j < _bytes.Length; j++)
            {
                _bytes[j] = (byte) _stream.ReadByte();
            }
        }

        private void WriteBytes()
        {
            for (var i = 0; i < _bytes.Length; i++)
            {
                _stream.WriteByte(_bytes[i]);
            }
        }

        public float Unpack()
        {
            ReadBytes();
            return BitConverter.ToSingle(_bytes);
        }

        public float[] UnpackArray(int length)
        {
            var array = new float[length];
            for (var i = 0; i < length; i++)
            {
                array[i] = Unpack();
            }

            return array;
        }

        public void Pack(float f)
        {
            if (!BitConverter.TryWriteBytes(_bytes, f))
            {
                throw new Exception("Unknown error");
            }
            WriteBytes();
        }

        public void Pack(IEnumerable<float> array)
        {
            foreach (var f in array)
            {
                Pack(f);
            }
        }

    }
}