# pyopencl
import pyopencl as cl

platforms = cl.get_platforms()

print("---------------------------------Platform---------------------------------")
print("Number of platforms: {}".format(len(platforms)))
print("Platform names: {}".format([platform.name
                                   for platform in platforms]))
print("Platform versions: {}".format([platform.version
                                      for platform in platforms]))
print("Platform profiles: {}".format([platform.profile
                                      for platform in platforms]))
print("Platform vendor: {}".format([platform.vendor
                                    for platform in platforms]))
print("---------------------------------Device---------------------------------")
devices = [platform.get_devices() for platform in platforms]

print("Number of devices: {}".format([len(device)
                                      for device in devices]))

print("Device names: {}".format([[device.name
                                  for device in device]
                                 for device in devices]))

print("Device versions: {}".format([[device.version
                                     for device in device]
                                    for device in devices]))

print("Device OpenCL C versions: {}".format([[device.opencl_c_version
                                              for device in device]
                                             for device in devices]))

print("Device address bits: {}".format([[device.address_bits
                                         for device in device]
                                        for device in devices]))

print("---------------------------------Context---------------------------------")
# make me a simple program that calculates the square of each element of a vector

# create a context for nvidia gpu
context = cl.Context(devices[0])
context = cl.create_some_context()

# create a queue to the device
queue = cl.CommandQueue(context)

# create a program 2d matrix multiplication
program = cl.Program(context, """
__kernel void square(const int M ,const int N, __global float *a, __global float *b
                     global float *c)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int k;
    float sum = 0.0f;
    if (i < M && j < N)
    {
        for (k = 0; k < M; k++)
        {
            sum += a[i * M + k] * b[k * M + j];
        }
        c[i * M + j] = sum;
    }
}
""").build()

# create a buffer
a = np.random.rand(50000).astype(np.float32)
b = np.random.rand(50000).astype(np.float32)
c = np.empty_like(a)

# create a buffer
a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)

# create a buffer
c_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, c.nbytes)

# execute the program
program.square(queue, a.shape, None, np.int32(50000), np.int32(50000), a_buf, b_buf, c_buf)

# copy the result from the buffer to the host
cl.enqueue_copy(queue, c, c_buf)

# print the result
print(c)
