import ctypes as c


def check(result):
    if result:
        raise RuntimeError(f'CUDA driver returned {result}')


driver = c.CDLL('/usr/lib/wsl/lib/libcuda.so.1')
device = c.c_int()
context = c.c_void_p()
module = c.c_void_p()
kernel = c.c_void_p()
address = c.c_uint64()
major = c.c_int()
minor = c.c_int()
ptx = b'''.version 8.0
.target sm_80
.address_size 64
.visible .entry write42(.param .u64 dst) {
    .reg .u64 %ptr;
    ld.param.u64 %ptr, [dst];
    st.global.u32 [%ptr], 42;
    ret;
}'''

check(driver.cuInit(0))
check(driver.cuDeviceGet(c.byref(device), 0))
check(driver.cuDeviceGetAttribute(c.byref(major), 75, device))
check(driver.cuDeviceGetAttribute(c.byref(minor), 76, device))
check(driver.cuCtxCreate_v2(c.byref(context), 0, device))
try:
    check(driver.cuModuleLoadData(c.byref(module), c.c_char_p(ptx)))
    check(driver.cuModuleGetFunction(c.byref(kernel), module, c.c_char_p(b'write42')))
    check(driver.cuMemAlloc_v2(c.byref(address), c.c_size_t(4)))
    args = (c.c_void_p * 1)(c.cast(c.byref(address), c.c_void_p))
    check(driver.cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, None, args, None))
    check(driver.cuCtxSynchronize())
    result = c.c_uint32()
    check(driver.cuMemcpyDtoH_v2(c.byref(result), address, c.c_size_t(4)))
    assert result.value == 42, result.value
    print(f'CUDA driver kernel passed, capability {major.value}.{minor.value}, result {result.value}')
finally:
    check(driver.cuCtxDestroy_v2(context))
