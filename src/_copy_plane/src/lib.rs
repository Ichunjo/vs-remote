use std::ptr::copy_nonoverlapping;

use pyo3::{exceptions::PyValueError, prelude::*};

/// Copy uncompressed planar bytes to a strided frame buffer line by line.
///
/// # Arguments
///
/// - `dst_addr` (`usize`) - Base memory address of destination frame plane.
/// - `decompressed` (`&[u8]`) - Raw planar byte string.
/// - `width` (`usize`) - Plane width in pixels.
/// - `height` (`usize`) - Plane height in lines.
/// - `bps` (`usize`) - Number of bytes per sample (e.g. 1 for 8-bit, 2 for 16-bit, 4 for float).
/// - `stride` (`isize`) - Destination plane stride in bytes.
#[pyfunction]
#[pyo3(signature = (dst_addr, decompressed, width, height, bps, stride))]
fn copy_plane(
    py: Python<'_>,
    dst_addr: Option<usize>,
    decompressed: &[u8],
    width: usize,
    height: usize,
    bps: usize,
    stride: isize,
) -> PyResult<()> {
    let dst_addr = match dst_addr {
        Some(addr) if addr != 0 => addr,
        _ => return Err(PyValueError::new_err("dst_addr must be a non-null pointer")),
    };

    let row_size = width * bps;

    debug_assert!(
        decompressed.len() >= row_size * height,
        "decompressed buffer smaller than plane data"
    );

    py.detach(move || {
        let mut dst = dst_addr as *mut u8;
        let src = decompressed.as_ptr();
        if stride == row_size.cast_signed() {
            unsafe { copy_nonoverlapping(src, dst, row_size * height) };
        } else {
            let mut src_ptr = src;
            for _ in 0..height {
                unsafe {
                    copy_nonoverlapping(src_ptr, dst, row_size);
                    src_ptr = src_ptr.add(row_size);
                }
                dst = dst.wrapping_offset(stride);
            }
        }
    });

    Ok(())
}

#[pymodule]
mod _copy_plane {
    #[pymodule_export]
    use super::copy_plane;
}
