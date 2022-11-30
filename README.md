# npy-writer

This is a small Rust package for writing the NumPy file formats `.npy` and `.npz`. This package supports writing the following data types:

 * Single floating points or integers
 * Vectors of single floating points or integers
 * N-dimensional arrays of single floating points or integers (via the `ndarray` feature)
 * `npz` files mapping string keys to any of the data types above (via the `zip` feature)

# Usage

Example of writing a `.npy` file with integers:

```rust
use npy_writer::NumpyWriter;
use std::fs::File;

let mut f = File::create("out.npy").unwrap();
vec![3, 2, 1].write_npy(&mut f).unwrap();
```

With the feature `ndarray`, you can write N-dimensional arrays like so:

```rust
let mut arr = Array3::zeros((2, 3, 4));
arr.write_npy(&mut f).unwrap();
```
