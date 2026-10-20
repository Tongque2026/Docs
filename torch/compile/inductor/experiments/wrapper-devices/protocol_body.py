def generated(x, scalar):
    with device_api.guard(0):
        device_api.set_device(0)
        stream0 = get_raw_stream(0)
        kernel_a.run(x, scalar.item(), grid=grid(6), stream=stream0)
        kernel_b.run(x, scalar.item(), grid=grid(6), stream=stream0)
