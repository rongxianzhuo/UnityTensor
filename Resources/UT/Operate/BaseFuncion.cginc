

uint calculate_new_index(const uint dim
    , uint old_id
    , const StructuredBuffer<uint> old_stride
    , const StructuredBuffer<uint> new_stride)
{
    uint new_id = 0;
    for (uint i = 0; i < dim; i++)
    {
        const uint stride = old_stride[i];
        const uint d = old_id / stride;
        old_id -= d * stride;
        new_id += d * new_stride[i];
    }
    return new_id;
}


uint2 calculate_new_index(const uint dim
    , uint old_id
    , const StructuredBuffer<uint> old_stride
    , const StructuredBuffer<uint> new_stride1
    , const StructuredBuffer<uint> new_stride2)
{
    uint new_id1 = 0;
    uint new_id2 = 0;
    for (uint i = 0; i < dim; i++)
    {
        const uint stride = old_stride[i];
        const uint d = old_id / stride;
        old_id -= d * stride;
        new_id1 += d * new_stride1[i];
        new_id2 += d * new_stride2[i];
    }
    return uint2(new_id1, new_id2);
}

uint calculate_new_index(const uint dim
    , uint old_id
    , uint ignore_new_stride
    , const StructuredBuffer<uint> old_stride
    , const StructuredBuffer<uint> new_stride)
{
    uint new_id = 0;
    for (uint i = 0; i < dim; i++)
    {
        const uint stride = old_stride[i];
        const uint d = old_id / stride;
        old_id -= d * stride;
        if (ignore_new_stride != i) new_id += d * new_stride[i];
    }
    return new_id;
}