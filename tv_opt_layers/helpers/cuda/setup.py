from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dpttr_cuda',
    ext_modules=[
        CUDAExtension('dpttr_cuda', [
            'dpttr_cuda.cpp',
            'dpttrf_cuda_kernel.cu',
            'dpttrs_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='tv1_op_utils_cuda',
    ext_modules=[
        CUDAExtension('tv1_op_utils_cuda', [
            'tv1_op_utils_cuda.cpp',
            'reduced_hessian_kernel.cu',
            'setup_cholesky_kernel.cu',
            'project_point_kernel.cu',
            'project_point_p_kernel.cu',
            'get_step_sizes_kernel.cu',
            'get_gradient_kernel.cu',
            'get_grd_kernel.cu',
            'check_inactive_kernel.cu',
            'quad_interp_kernel.cu',
            'dual_to_primal_kernel.cu',
            'dual_to_primal_cont_kernel.cu',
            'dual_to_primal_cont_p_kernel.cu',
            'assign_cont_update0_kernel.cu',
            'assign_cont_update1_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


setup(
    name='tv1_backward_cuda',
    ext_modules=[
        CUDAExtension('tv1_backward_cuda', [
            'tv1_backward_cuda.cpp',
            'check_s_active_kernel.cu',
            'setup_l_s_kernel.cu',
            'pad_ltl_diag_kernel.cu',
            'setup_sign_all_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
